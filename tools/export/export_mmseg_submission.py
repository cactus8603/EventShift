#!/usr/bin/env python
"""Export CoSEC test masks from an MMSegmentation semantic model."""

import argparse
import contextlib
import json
import os
import random
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from collections import OrderedDict
from mmengine.config import Config
from mmengine.utils import import_modules_from_strings
from mmseg.apis import inference_model, init_model
from mmseg.utils import register_all_modules
from tqdm import tqdm


def _eventshift_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "configs").is_dir() and (parent / "third_party").is_dir():
            return parent
    return Path(__file__).resolve().parents[1]


ROOT = _eventshift_root()


def configure_torch_reproducibility():
    if os.environ.get("EVENTSHIFT_DETERMINISTIC", "1") in {"0", "false", "False", "no"}:
        return
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--zip", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-desc", default="Export mmseg masks")
    parser.add_argument(
        "--scale-specs",
        default="s512:512:1200,s624:624:1200,s768:768:1400,s1024:1024:1600",
        help="Named TTA scale specs: name:min_size:max_size.",
    )
    parser.add_argument(
        "--scale-set",
        default="",
        help="Optional TTA scale names joined by '+', e.g. s512+s624+s768+s1024.",
    )
    parser.add_argument("--flip", action="store_true", help="Use horizontal flip inside each TTA scale.")
    return parser.parse_args()


def maybe_import_custom_modules(cfg):
    custom_imports = cfg.get("custom_imports", None)
    if custom_imports is not None:
        import_modules_from_strings(**custom_imports)


@contextlib.contextmanager
def legacy_torch_load_for_checkpoint():
    """Load trusted local training checkpoints created before PyTorch 2.6."""
    original_load = torch.load

    def compatible_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = compatible_load
    try:
        yield
    finally:
        torch.load = original_load


def iter_test_images(test_root, sequences=None):
    test_root = Path(test_root)
    keep_sequences = set(sequences) if sequences else None
    for seq_dir in sorted(path for path in test_root.iterdir() if path.is_dir()):
        if keep_sequences is not None and seq_dir.name not in keep_sequences:
            continue
        img_dir = seq_dir / "img_co_left"
        if not img_dir.is_dir():
            continue
        for img_path in sorted(img_dir.glob("*.png")):
            yield seq_dir.name, img_path


def split_csv(text):
    return [part.strip() for part in str(text).split(",") if part.strip()]


def parse_scale_specs(text):
    specs = OrderedDict()
    for item in split_csv(text):
        name, min_size, max_size = item.split(":")
        specs[name] = {"name": name, "min_size": int(min_size), "max_size": int(max_size)}
    return specs


def parse_scale_set(text):
    return [part.strip() for part in str(text).split("+") if part.strip()]


def resize_short_edge(image, min_size, max_size):
    height, width = image.shape[:2]
    short = min(height, width)
    long = max(height, width)
    scale = float(min_size) / float(short)
    if round(scale * long) > max_size:
        scale = float(max_size) / float(long)
    new_height = int(round(height * scale))
    new_width = int(round(width * scale))
    if (new_height, new_width) == (height, width):
        return image
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)


def pred_from_result(result, shape):
    if hasattr(result, "pred_sem_seg") and result.pred_sem_seg is not None:
        pred = result.pred_sem_seg.data.detach().cpu()
        if pred.ndim == 3:
            pred = pred[0]
    else:
        logits = result.seg_logits.data.detach().float().cpu()
        if tuple(logits.shape[-2:]) != tuple(shape):
            logits = F.interpolate(logits.unsqueeze(0), size=shape, mode="bilinear", align_corners=False)[0]
        pred = logits.argmax(dim=0)
    pred = pred.to(torch.uint8).numpy()
    if pred.shape != tuple(shape):
        pred = cv2.resize(pred, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.asarray(pred, dtype=np.uint8)


def prob_from_result(result, shape):
    logits = result.seg_logits.data.detach().float().cpu()
    if tuple(logits.shape[-2:]) != tuple(shape):
        logits = F.interpolate(logits.unsqueeze(0), size=shape, mode="bilinear", align_corners=False)[0]
    logits = logits - logits.max(dim=0, keepdim=True).values
    prob = torch.softmax(logits, dim=0)
    return prob


def pred_with_tta(model, image, shape, scale_names, scale_specs, use_flip):
    prob_sum = None
    count = 0
    for scale_name in scale_names:
        spec = scale_specs[scale_name]
        resized = resize_short_edge(image, spec["min_size"], spec["max_size"])
        result = inference_model(model, resized)
        prob = prob_from_result(result, shape)
        prob_sum = prob if prob_sum is None else prob_sum + prob
        count += 1
        if use_flip:
            flipped = np.ascontiguousarray(resized[:, ::-1])
            flip_result = inference_model(model, flipped)
            flip_prob = torch.flip(prob_from_result(flip_result, shape), dims=[2])
            prob_sum = prob_sum + flip_prob
            count += 1
    pred = (prob_sum / float(count)).argmax(dim=0).to(torch.uint8).numpy()
    if pred.shape != tuple(shape):
        pred = cv2.resize(pred, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.asarray(pred, dtype=np.uint8)


def zip_directory(src_dir, zip_path):
    src_dir = Path(src_dir)
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    max_path_len = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(src_dir.rglob("*")):
            if not path.is_file() or path.name == "manifest.json":
                continue
            rel_path = path.relative_to(src_dir)
            max_path_len = max(max_path_len, len(str(rel_path)))
            zf.write(path, rel_path)
            count += 1
    return {"entries": count, "max_path_len": max_path_len}


def main():
    args = parse_args()
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    configure_torch_reproducibility()

    out_dir = Path(args.out_dir)
    if out_dir.exists() and args.overwrite:
        import shutil

        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(args.config_file)
    maybe_import_custom_modules(cfg)
    with legacy_torch_load_for_checkpoint():
        model = init_model(cfg, args.checkpoint, device=args.device)
    scale_names = parse_scale_set(args.scale_set)
    scale_specs = parse_scale_specs(args.scale_specs)
    missing = [name for name in scale_names if name not in scale_specs]
    if missing:
        raise ValueError(f"Unknown scale names in --scale-set: {missing}. Available: {list(scale_specs)}")
    if scale_names:
        print(
            f"[export-mmseg] TTA enabled: scales={scale_names}, flip={args.flip}",
            flush=True,
        )

    images = list(iter_test_images(args.test_root, args.sequences))
    if args.limit is not None:
        images = images[: int(args.limit)]

    counts = {"total": 0, "skipped": 0, "sequences": {}}
    for seq_name, img_path in tqdm(images, desc=args.progress_desc):
        dst_dir = out_dir / seq_name / "segment_co"
        dst_path = dst_dir / img_path.name
        if args.skip_existing and dst_path.exists():
            counts["skipped"] += 1
            continue
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read image: {img_path}")
        if scale_names:
            pred = pred_with_tta(model, image, image.shape[:2], scale_names, scale_specs, args.flip)
        else:
            result = inference_model(model, str(img_path))
            pred = pred_from_result(result, image.shape[:2])
        dst_dir.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(dst_path), pred):
            raise RuntimeError(f"Could not write mask: {dst_path}")
        counts["total"] += 1
        counts["sequences"][seq_name] = counts["sequences"].get(seq_name, 0) + 1

    zip_info = zip_directory(out_dir, args.zip) if args.zip else None
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "backend": "mmseg",
        "config_file": str(Path(args.config_file).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "test_root": str(Path(args.test_root).resolve()),
        "out_dir": str(out_dir.resolve()),
        "zip": str(Path(args.zip).resolve()) if args.zip else None,
        "device": args.device,
        "sequences": args.sequences,
        "tta": {
            "scale_specs": args.scale_specs,
            "scale_set": args.scale_set,
            "flip": args.flip,
        },
        "counts": counts,
        "zip_info": zip_info,
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    if args.zip:
        print(f"Wrote zip: {args.zip}")
    print(f"Wrote masks to: {out_dir}")
    print(json.dumps({"counts": counts, "zip_info": zip_info}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
