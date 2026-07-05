#!/usr/bin/env python
"""Diagnose single-class and greedy candidate-class routes on cached maps."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from ensemble_feature_cache_common import CLASSES, SegmentationStats, load_label, resize_if_needed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", default="work_dirs/ensemble_feature_cache")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--anchor-model", required=True)
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--route-json", required=True)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--single-only", action="store_true", help="Only evaluate independent single-class routes.")
    return parser.parse_args()


def read_rows(cache_root, model, dataset):
    with (cache_root / model / f"{dataset}_per_image.csv").open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def map_paths(cache_root, model, dataset):
    paths = sorted((cache_root / model / "maps" / dataset).glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No cached maps found: {cache_root / model / 'maps' / dataset}")
    return paths


def load_pred(path, shape):
    pred = np.load(path)["pred"].astype(np.uint8, copy=False)
    return resize_if_needed(pred, shape, cv2.INTER_NEAREST).astype(np.uint8, copy=False)


def main():
    args = parse_args()
    cache_root = Path(args.cache_root)
    route_payload = json.load(open(args.route_json, "r", encoding="utf-8"))
    routes = route_payload["routes"]
    candidate_class_ids = [int(route["class_id"]) for route in routes]

    rows = read_rows(cache_root, args.anchor_model, args.dataset)
    anchor_maps = map_paths(cache_root, args.anchor_model, args.dataset)
    candidate_maps = map_paths(cache_root, args.candidate_model, args.dataset)
    if len(anchor_maps) != len(candidate_maps) or len(anchor_maps) != len(rows):
        raise ValueError(
            f"Count mismatch: rows={len(rows)}, anchor={len(anchor_maps)}, candidate={len(candidate_maps)}"
        )

    labels = [load_label(row["label_path"]) for row in rows]
    anchors = [load_pred(path, label.shape) for path, label in zip(anchor_maps, labels)]
    candidates = [load_pred(path, label.shape) for path, label in zip(candidate_maps, labels)]

    def score(class_ids):
        meter = SegmentationStats()
        changed_pixels = 0
        total_pixels = 0
        for label, anchor, candidate in zip(labels, anchors, candidates):
            merged = anchor.copy()
            for class_id in class_ids:
                merged[candidate == class_id] = class_id
            changed_pixels += int((merged != anchor).sum())
            total_pixels += int(anchor.size)
            meter.update(merged, label)
        metrics = meter.metrics()
        return {
            "mIoU": metrics["mIoU"],
            "mAcc": metrics["mAcc"],
            "aAcc": metrics["aAcc"],
            "changed_pixels": changed_pixels,
            "changed_percent": 100.0 * changed_pixels / max(total_pixels, 1),
        }

    base = score([])
    singles = []
    for class_id in candidate_class_ids:
        result = score([class_id])
        singles.append(
            {
                "class_id": class_id,
                "class_name": CLASSES[class_id],
                **result,
                "delta_mIoU": result["mIoU"] - base["mIoU"],
            }
        )
    singles.sort(key=lambda item: item["delta_mIoU"], reverse=True)

    greedy = []
    if not args.single_only:
        selected = []
        remaining = list(candidate_class_ids)
        current = base
        while remaining:
            best = None
            for class_id in remaining:
                result = score(selected + [class_id])
                if best is None or result["mIoU"] > best["result"]["mIoU"]:
                    best = {"class_id": class_id, "result": result}
            if best["result"]["mIoU"] <= current["mIoU"]:
                break
            selected.append(best["class_id"])
            remaining.remove(best["class_id"])
            current = best["result"]
            greedy.append(
                {
                    "added_class_id": best["class_id"],
                    "added_class_name": CLASSES[best["class_id"]],
                    **current,
                    "delta_mIoU": current["mIoU"] - base["mIoU"],
                    "selected_classes": [CLASSES[class_id] for class_id in selected],
                }
            )

    payload = {
        "dataset": args.dataset,
        "anchor_model": args.anchor_model,
        "candidate_model": args.candidate_model,
        "route_json": str(Path(args.route_json).resolve()),
        "base": base,
        "singles": singles,
        "greedy": greedy,
    }
    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")

    print(f"base mIoU={base['mIoU']:.4f}")
    for row in singles:
        print(
            f"single {row['class_name']:14s} "
            f"mIoU={row['mIoU']:.4f} delta={row['delta_mIoU']:+.4f} "
            f"changed={row['changed_pixels']} ({row['changed_percent']:.4f}%)"
        )
    for row in greedy:
        print(
            f"greedy add {row['added_class_name']:14s} "
            f"mIoU={row['mIoU']:.4f} delta={row['delta_mIoU']:+.4f} "
            f"classes={row['selected_classes']}"
        )


if __name__ == "__main__":
    main()
