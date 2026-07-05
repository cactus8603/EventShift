#!/usr/bin/env python
"""Export CoSEC test masks from a Mask2Former semantic segmentation checkpoint."""

import argparse
import json
import os
import random
import sys
import importlib.util
import zipfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
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


sys.path.insert(0, str(ROOT / "third_party" / "Mask2Former"))
if importlib.util.find_spec("detectron2") is None:
    sys.path.insert(0, str(ROOT / "third_party" / "detectron2"))

from detectron2.config import get_cfg  # noqa: E402
from detectron2.engine import DefaultPredictor  # noqa: E402
from detectron2.projects.deeplab import add_deeplab_config  # noqa: E402
from mask2former import SemanticSegmentorWithTTA, add_maskformer2_config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--test-root", default="data/test")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--zip", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Extra detectron2 config options after --.",
    )
    return parser.parse_args()


def setup_cfg(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    if args.opts:
        opts = args.opts[1:] if args.opts and args.opts[0] == "--" else args.opts
        cfg.merge_from_list(opts)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.DEVICE = args.device
    cfg.DATASETS.TEST = ()
    cfg.freeze()
    return cfg


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


def zip_directory(src_dir, zip_path):
    src_dir = Path(src_dir)
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    max_path_len = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(src_dir.rglob("*")):
            if not path.is_file():
                continue
            rel_path = path.relative_to(src_dir)
            max_path_len = max(max_path_len, len(str(rel_path)))
            zf.write(path, rel_path)
            count += 1
    return count, max_path_len


def write_manifest(out_dir, args, counts, zip_info=None):
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(ROOT),
        "config_file": str(Path(args.config_file).resolve()),
        "weights": str(Path(args.weights).resolve()),
        "test_root": str(Path(args.test_root).resolve()),
        "out_dir": str(Path(args.out_dir).resolve()),
        "zip": str(Path(args.zip).resolve()) if args.zip else None,
        "device": args.device,
        "sequences": args.sequences,
        "tta_enabled": bool(getattr(args, "tta_enabled", False)),
        "test_aug": getattr(args, "test_aug", None),
        "counts": counts,
        "zip_info": zip_info,
    }
    with (Path(out_dir) / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")


def main():
    args = parse_args()
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    configure_torch_reproducibility()

    out_dir = Path(args.out_dir)
    if out_dir.exists() and args.overwrite:
        import shutil

        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = setup_cfg(args)
    predictor = DefaultPredictor(cfg)
    tta_model = None
    if cfg.TEST.AUG.ENABLED:
        tta_model = SemanticSegmentorWithTTA(cfg, predictor.model)
        args.tta_enabled = True
        args.test_aug = {
            "min_sizes": list(cfg.TEST.AUG.MIN_SIZES),
            "max_size": int(cfg.TEST.AUG.MAX_SIZE),
            "flip": bool(cfg.TEST.AUG.FLIP),
            "min_size_test": int(cfg.INPUT.MIN_SIZE_TEST),
            "max_size_test": int(cfg.INPUT.MAX_SIZE_TEST),
        }
        print(f"[export] TTA enabled: {args.test_aug}", flush=True)
    else:
        args.tta_enabled = False
        args.test_aug = None

    images = list(iter_test_images(args.test_root, args.sequences))
    if args.limit is not None:
        images = images[: args.limit]
    counts = {"total": 0, "sequences": {}, "skipped": 0}

    for seq_name, img_path in tqdm(images, desc="Export masks"):
        dst_dir = out_dir / seq_name / "segment_co"
        dst_path = dst_dir / img_path.name
        if args.skip_existing and dst_path.exists():
            counts["skipped"] += 1
            continue

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read image: {img_path}")

        if tta_model is None:
            outputs = predictor(image)
        else:
            tta_image = image
            if cfg.INPUT.FORMAT == "RGB":
                tta_image = tta_image[:, :, ::-1]
            tta_image = torch.as_tensor(
                np.ascontiguousarray(tta_image.astype("float32").transpose(2, 0, 1))
            )
            with torch.no_grad():
                outputs = tta_model(
                    [
                        {
                            "file_name": str(img_path),
                            "image": tta_image,
                            "height": image.shape[0],
                            "width": image.shape[1],
                        }
                    ]
                )[0]
        sem_seg = outputs["sem_seg"]
        pred = sem_seg.argmax(dim=0).to(torch.uint8).cpu().numpy()
        if pred.shape[:2] != image.shape[:2]:
            pred = cv2.resize(
                pred,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        pred = np.asarray(pred, dtype=np.uint8)

        dst_dir.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(dst_path), pred):
            raise RuntimeError(f"Could not write mask: {dst_path}")
        counts["total"] += 1
        counts["sequences"][seq_name] = counts["sequences"].get(seq_name, 0) + 1

    zip_info = None
    if args.zip:
        zip_count, max_path_len = zip_directory(out_dir, args.zip)
        zip_info = {"entries": zip_count, "max_path_len": max_path_len}
        print(f"Wrote zip: {args.zip}")
        print(f"zip_entries: {zip_count}")
        print(f"max_zip_path_len: {max_path_len}")

    write_manifest(out_dir, args, counts, zip_info)
    print(f"Wrote masks to: {out_dir}")
    print(f"masks: {counts['total']}")
    print(f"sequences: {len(counts['sequences'])}")
    if counts["skipped"]:
        print(f"skipped_existing: {counts['skipped']}")


if __name__ == "__main__":
    main()
