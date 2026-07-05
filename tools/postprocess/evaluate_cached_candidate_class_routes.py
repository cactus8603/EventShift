#!/usr/bin/env python
"""Evaluate candidate-class route JSONs on saved ensemble feature caches."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from ensemble_feature_cache_common import SegmentationStats, load_label, resize_if_needed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--anchor-model", required=True)
    parser.add_argument("--route-json", required=True)
    parser.add_argument("--cache-root", default="work_dirs/ensemble_feature_cache")
    parser.add_argument(
        "--candidate-model",
        action="append",
        default=[],
        help="Candidate mapping in the form route_name=cache_model_name. Repeatable.",
    )
    parser.add_argument("--out-json", default=None)
    return parser.parse_args()


def parse_mapping(items):
    parsed = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--candidate-model must be route_name=cache_model_name, got: {item}")
        key, value = item.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def read_rows(cache_root, model, dataset):
    path = cache_root / model / f"{dataset}_per_image.csv"
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def map_paths(cache_root, model, dataset):
    paths = sorted((cache_root / model / "maps" / dataset).glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No cached maps found for {model}:{dataset}")
    return paths


def load_pred(path, shape):
    pred = np.load(path)["pred"].astype(np.uint8, copy=False)
    return resize_if_needed(pred, shape, cv2.INTER_NEAREST).astype(np.uint8, copy=False)


def json_metrics(metrics):
    return {
        key: (value.tolist() if isinstance(value, np.ndarray) else value)
        for key, value in metrics.items()
    }


def main():
    args = parse_args()
    cache_root = Path(args.cache_root)
    candidates = parse_mapping(args.candidate_model)
    route_payload = json.load(open(args.route_json, "r", encoding="utf-8"))
    routes = route_payload["routes"]
    missing = sorted({route["candidate"] for route in routes} - set(candidates))
    if missing:
        raise ValueError(f"Missing --candidate-model entries for route names: {missing}")

    rows = read_rows(cache_root, args.anchor_model, args.dataset)
    anchor_maps = map_paths(cache_root, args.anchor_model, args.dataset)
    candidate_maps = {
        name: map_paths(cache_root, model, args.dataset)
        for name, model in candidates.items()
    }
    for name, paths in candidate_maps.items():
        if len(paths) != len(anchor_maps):
            raise ValueError(f"Map count mismatch: {name} has {len(paths)}, anchor has {len(anchor_maps)}")

    anchor_meter = SegmentationStats()
    merged_meter = SegmentationStats()
    changed_pixels = 0
    total_pixels = 0
    claimed_by_class = {route["class_name"]: 0 for route in routes}
    changed_by_class = {route["class_name"]: 0 for route in routes}

    for index, row in enumerate(rows):
        label = load_label(row["label_path"])
        anchor = load_pred(anchor_maps[index], label.shape)
        merged = anchor.copy()
        for route in routes:
            candidate = load_pred(candidate_maps[route["candidate"]][index], label.shape)
            class_id = int(route["class_id"])
            take = candidate == class_id
            class_name = route["class_name"]
            claimed_by_class[class_name] += int(take.sum())
            changed_by_class[class_name] += int((take & (anchor != class_id)).sum())
            merged[take] = class_id

        anchor_meter.update(anchor, label)
        merged_meter.update(merged, label)
        changed_pixels += int((merged != anchor).sum())
        total_pixels += int(anchor.size)

    anchor_metrics = anchor_meter.metrics()
    merged_metrics = merged_meter.metrics()
    print(f"dataset={args.dataset}")
    print(f"anchor_model={args.anchor_model}")
    print(f"anchor_mIoU={anchor_metrics['mIoU']:.4f}")
    print(f"merged_mIoU={merged_metrics['mIoU']:.4f}")
    print(f"delta_mIoU={merged_metrics['mIoU'] - anchor_metrics['mIoU']:+.4f}")
    print(f"changed_pixels={changed_pixels}")
    print(f"changed_percent={100.0 * changed_pixels / max(total_pixels, 1):.6f}")
    print("claimed_by_class=" + json.dumps(claimed_by_class, sort_keys=True))
    print("changed_by_class=" + json.dumps(changed_by_class, sort_keys=True))

    if args.out_json:
        payload = {
            "dataset": args.dataset,
            "anchor_model": args.anchor_model,
            "route_json": str(Path(args.route_json).resolve()),
            "candidate_models": candidates,
            "anchor": json_metrics(anchor_metrics),
            "merged": json_metrics(merged_metrics),
            "delta_mIoU": merged_metrics["mIoU"] - anchor_metrics["mIoU"],
            "changed_pixels": changed_pixels,
            "total_pixels": total_pixels,
            "changed_percent": 100.0 * changed_pixels / max(total_pixels, 1),
            "claimed_by_class": claimed_by_class,
            "changed_by_class": changed_by_class,
        }
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")


if __name__ == "__main__":
    main()
