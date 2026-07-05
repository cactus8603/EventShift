#!/usr/bin/env python
"""Export CoSEC test masks by routing among multiple inference scales."""

import argparse
import json
import os
import shutil
import sys
import importlib.util
import zipfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

def _eventshift_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "configs").is_dir() and (parent / "third_party").is_dir():
            return parent
    return Path(__file__).resolve().parents[1]


ROOT = _eventshift_root()
sys.path.insert(0, str(ROOT / "third_party" / "Mask2Former"))
if importlib.util.find_spec("detectron2") is None:
    sys.path.insert(0, str(ROOT / "third_party" / "detectron2"))

from detectron2.config import get_cfg  # noqa: E402
from detectron2.engine import DefaultPredictor  # noqa: E402
from detectron2.projects.deeplab import add_deeplab_config  # noqa: E402
from mask2former import add_maskformer2_config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument(
        "--scale-specs",
        default="s512:512:1200,s624:624:1200,s768:768:1400,s1024:1024:1600",
        help="Comma-separated name:min_size:max_size entries.",
    )
    parser.add_argument("--test-root", required=True)
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


def split_csv(text):
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_scale_specs(text):
    specs = []
    for item in split_csv(text):
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError(f"Bad scale spec '{item}', expected name:min:max")
        name, min_size, max_size = parts
        specs.append({"name": name, "min_size": int(min_size), "max_size": int(max_size)})
    return specs


def setup_cfg(args, min_size, max_size):
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
    cfg.TEST.AUG.ENABLED = False
    cfg.INPUT.MIN_SIZE_TEST = int(min_size)
    cfg.INPUT.MAX_SIZE_TEST = int(max_size)
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


def resize_scores(scores, shape):
    if tuple(scores.shape[-2:]) == tuple(shape):
        return scores
    return F.interpolate(
        scores.unsqueeze(0),
        size=shape,
        mode="bilinear",
        align_corners=False,
    )[0]


def normalize_scores(scores):
    prob = scores.float().clamp_min(1e-8)
    return prob / prob.sum(dim=0, keepdim=True).clamp_min(1e-8)


def pred_and_conf(outputs, image_shape):
    scores = outputs["sem_seg"].detach()
    prob = normalize_scores(resize_scores(scores, image_shape))
    top2 = torch.topk(prob, k=2, dim=0).values
    pred = prob.argmax(dim=0).to(torch.uint8).cpu().numpy()
    conf = top2[0].to(torch.float16).cpu().numpy()
    margin = (top2[0] - top2[1]).to(torch.float16).cpu().numpy()
    return pred, conf, margin


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


def update_best(best, image_idx, pred, score, scale_idx):
    if best[image_idx] is None:
        best[image_idx] = {
            "pred": pred.copy(),
            "score": score.copy(),
            "choice": np.full(pred.shape, scale_idx, dtype=np.uint8),
        }
        return

    take = score > best[image_idx]["score"]
    best[image_idx]["pred"][take] = pred[take]
    best[image_idx]["score"][take] = score[take]
    best[image_idx]["choice"][take] = scale_idx


def choice_histogram(choice, branch_names):
    values, counts = np.unique(choice, return_counts=True)
    return {branch_names[int(value)]: int(count) for value, count in zip(values, counts)}


def write_manifest(out_dir, args, scale_specs, counts, routing_stats, zip_info):
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
        "scale_specs": scale_specs,
        "routing": "choose_highest_confidence",
        "counts": counts,
        "routing_stats": routing_stats,
        "zip_info": zip_info,
    }
    with (Path(out_dir) / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")


def main():
    args = parse_args()
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    scale_specs = parse_scale_specs(args.scale_specs)

    out_dir = Path(args.out_dir)
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = list(iter_test_images(args.test_root, args.sequences))
    if args.limit is not None:
        images = images[: args.limit]
    if not images:
        raise ValueError("No test images matched the requested sequences.")

    loaded_images = []
    for seq_name, img_path in images:
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read image: {img_path}")
        loaded_images.append(
            {
                "seq_name": seq_name,
                "img_path": img_path,
                "image": image,
                "shape": image.shape[:2],
            }
        )

    best = [None for _ in loaded_images]
    branch_names = [spec["name"] for spec in scale_specs]
    for scale_idx, spec in enumerate(scale_specs):
        cfg = setup_cfg(args, spec["min_size"], spec["max_size"])
        predictor = DefaultPredictor(cfg)
        iterator = tqdm(loaded_images, desc=f"Scale {spec['name']}")
        for image_idx, item in enumerate(iterator):
            with torch.no_grad():
                outputs = predictor(item["image"])
            pred, conf, _margin = pred_and_conf(outputs, item["shape"])
            update_best(best, image_idx, pred, conf, scale_idx)

        del predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    counts = {"total": 0, "sequences": {}, "skipped": 0}
    routing_stats = {"by_sequence": {}, "global_choice_pixels": {}}
    global_choice_pixels = {}

    for item, result in tqdm(list(zip(loaded_images, best)), desc="Write routed masks"):
        seq_name = item["seq_name"]
        dst_dir = out_dir / seq_name / "segment_co"
        dst_path = dst_dir / item["img_path"].name
        if args.skip_existing and dst_path.exists():
            counts["skipped"] += 1
            continue

        dst_dir.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(dst_path), result["pred"]):
            raise RuntimeError(f"Could not write mask: {dst_path}")

        hist = choice_histogram(result["choice"], branch_names)
        seq_stats = routing_stats["by_sequence"].setdefault(seq_name, {})
        for scale_name, value in hist.items():
            seq_stats[scale_name] = seq_stats.get(scale_name, 0) + value
            global_choice_pixels[scale_name] = global_choice_pixels.get(scale_name, 0) + value

        counts["total"] += 1
        counts["sequences"][seq_name] = counts["sequences"].get(seq_name, 0) + 1

    routing_stats["global_choice_pixels"] = global_choice_pixels
    zip_info = zip_directory(out_dir, args.zip) if args.zip else None
    write_manifest(out_dir, args, scale_specs, counts, routing_stats, zip_info)

    if args.zip:
        print(f"Wrote zip: {args.zip}")
        print(json.dumps(zip_info, indent=2, sort_keys=True))
    print(f"Wrote masks to: {out_dir}")
    print(json.dumps({"counts": counts, "routing_stats": routing_stats}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
