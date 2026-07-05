#!/usr/bin/env python
"""Evaluate an anchor-preserving agreement ensemble over cached maps."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ensemble_feature_cache_common import CLASSES, SegmentationStats, load_label, write_csv, write_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", default="work_dirs/ensemble_feature_cache")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--anchor-model", required=True)
    parser.add_argument("--candidate-model", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--candidate-min-iou", type=float, default=50.0)
    parser.add_argument("--candidate-min-gt-pixels", type=int, default=10000)
    parser.add_argument(
        "--allow-class",
        action="append",
        default=None,
        help="Optional class name/id allowlist. If omitted, derive from candidate per-class IoU/support.",
    )
    return parser.parse_args()


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def cache_dir(cache_root, model):
    path = Path(cache_root) / model
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def map_paths(model_dir, dataset):
    paths = sorted((model_dir / "maps" / dataset).glob("*.npz"))
    if not paths:
        raise FileNotFoundError(model_dir / "maps" / dataset)
    return paths


def load_pred(path):
    return np.load(path)["pred"].astype(np.uint8, copy=False)


def json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def class_id(value):
    text = str(value)
    if text.isdigit():
        idx = int(text)
        if idx < 0 or idx >= len(CLASSES):
            raise ValueError(f"Class id out of range: {idx}")
        return idx
    if text not in CLASSES:
        raise ValueError(f"Unknown class: {text}")
    return CLASSES.index(text)


def derive_allowed_classes(model_dirs, dataset, min_iou, min_gt_pixels):
    allowed = set(range(len(CLASSES)))
    diagnostics = {}
    for model_name, model_dir in model_dirs:
        rows = read_csv(model_dir / f"{dataset}_per_class_iou.csv")
        model_allowed = set()
        model_rows = []
        for row in rows:
            idx = int(row["class_id"])
            iou_text = row.get("iou")
            gt_pixels = int(row.get("gt_pixels") or 0)
            iou = float(iou_text) if iou_text not in {"", None} else float("nan")
            keep = np.isfinite(iou) and iou >= min_iou and gt_pixels >= min_gt_pixels
            if keep:
                model_allowed.add(idx)
            model_rows.append(
                {
                    "class_id": idx,
                    "class_name": row.get("class_name", CLASSES[idx]),
                    "iou": None if not np.isfinite(iou) else iou,
                    "gt_pixels": gt_pixels,
                    "allowed": keep,
                }
            )
        diagnostics[model_name] = model_rows
        allowed &= model_allowed
    return allowed, diagnostics


def load_rows(model_dir, dataset):
    rows = read_csv(model_dir / f"{dataset}_per_image.csv")
    if not rows:
        raise FileNotFoundError(model_dir / f"{dataset}_per_image.csv")
    return rows


def validate_alignment(anchor_rows, candidate_rows_by_model):
    for model_name, rows in candidate_rows_by_model:
        if len(rows) != len(anchor_rows):
            raise ValueError(f"Row count mismatch for {model_name}: {len(rows)} vs {len(anchor_rows)}")
        for idx, (left, right) in enumerate(zip(anchor_rows, rows)):
            if left.get("image_id") != right.get("image_id"):
                raise ValueError(
                    f"Image order mismatch at {idx}: anchor={left.get('image_id')} "
                    f"{model_name}={right.get('image_id')}"
                )


def main():
    args = parse_args()
    cache_root = Path(args.cache_root)
    anchor_dir = cache_dir(cache_root, args.anchor_model)
    candidate_dirs = [(name, cache_dir(cache_root, name)) for name in args.candidate_model]
    if len(candidate_dirs) < 2:
        raise ValueError("Need at least two --candidate-model entries for agreement routing.")

    anchor_rows = load_rows(anchor_dir, args.dataset)
    candidate_rows = [(name, load_rows(path, args.dataset)) for name, path in candidate_dirs]
    validate_alignment(anchor_rows, candidate_rows)

    if args.allow_class:
        allowed_classes = {class_id(item) for item in args.allow_class}
        class_diagnostics = None
    else:
        allowed_classes, class_diagnostics = derive_allowed_classes(
            candidate_dirs,
            args.dataset,
            args.candidate_min_iou,
            args.candidate_min_gt_pixels,
        )

    anchor_maps = map_paths(anchor_dir, args.dataset)
    candidate_maps = [(name, map_paths(path, args.dataset)) for name, path in candidate_dirs]
    for name, paths in candidate_maps:
        if len(paths) != len(anchor_maps):
            raise ValueError(f"Map count mismatch for {name}: {len(paths)} vs {len(anchor_maps)}")

    meter = SegmentationStats()
    anchor_meter = SegmentationStats()
    replace_by_class = np.zeros(len(CLASSES), dtype=np.int64)
    changed_by_class = np.zeros(len(CLASSES), dtype=np.int64)
    total_pixels = 0
    changed_pixels = 0
    accepted_pixels = 0

    for index, row in enumerate(anchor_rows):
        label = load_label(row["label_path"])
        anchor = load_pred(anchor_maps[index])
        candidates = [load_pred(paths[index]) for _, paths in candidate_maps]
        if any(pred.shape != anchor.shape for pred in candidates):
            raise ValueError(f"Candidate shape mismatch at {row.get('image_id')}")
        if anchor.shape != label.shape:
            raise ValueError(f"Anchor/label shape mismatch at {row.get('image_id')}: {anchor.shape} vs {label.shape}")

        agree = np.ones(anchor.shape, dtype=bool)
        first = candidates[0]
        for pred in candidates[1:]:
            agree &= pred == first
        if allowed_classes:
            allowed_mask = np.isin(first, list(allowed_classes))
        else:
            allowed_mask = np.zeros(anchor.shape, dtype=bool)
        take = agree & allowed_mask & (first != anchor)

        ensemble = anchor.copy()
        ensemble[take] = first[take]

        meter.update(ensemble, label)
        anchor_meter.update(anchor, label)
        total_pixels += int(anchor.size)
        changed = ensemble != anchor
        changed_pixels += int(np.count_nonzero(changed))
        accepted_pixels += int(np.count_nonzero(take))
        for idx in range(len(CLASSES)):
            class_take = take & (first == idx)
            replace_by_class[idx] += int(np.count_nonzero(class_take))
            changed_by_class[idx] += int(np.count_nonzero(changed & (ensemble == idx)))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    class_rows = []
    for idx, name in enumerate(CLASSES):
        class_rows.append(
            {
                "class_id": idx,
                "class_name": name,
                "allowed": idx in allowed_classes,
                "accepted_pixels": int(replace_by_class[idx]),
                "changed_to_class_pixels": int(changed_by_class[idx]),
            }
        )
    write_csv(out_dir / "accepted_by_class.csv", class_rows)
    write_csv(out_dir / "ensemble_per_class_iou.csv", meter.class_rows())

    anchor_metrics = anchor_meter.metrics()
    ensemble_metrics = meter.metrics()
    summary = {
        "dataset": args.dataset,
        "anchor_model": args.anchor_model,
        "candidate_models": args.candidate_model,
        "candidate_min_iou": args.candidate_min_iou,
        "candidate_min_gt_pixels": args.candidate_min_gt_pixels,
        "allowed_classes": [CLASSES[idx] for idx in sorted(allowed_classes)],
        "anchor_metrics": anchor_metrics,
        "ensemble_metrics": ensemble_metrics,
        "delta_mIoU": ensemble_metrics["mIoU"] - anchor_metrics["mIoU"],
        "total_pixels": total_pixels,
        "accepted_pixels": accepted_pixels,
        "changed_pixels": changed_pixels,
        "changed_pixel_rate": changed_pixels / total_pixels if total_pixels else 0.0,
        "class_diagnostics": class_diagnostics,
    }
    summary = json_ready(summary)
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
