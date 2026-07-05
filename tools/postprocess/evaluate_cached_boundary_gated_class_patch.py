#!/usr/bin/env python
"""Evaluate boundary-gated class patches on saved ensemble feature caches."""

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
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--class-id", action="append", type=int, required=True)
    parser.add_argument("--cache-root", default="work_dirs/ensemble_feature_cache")
    parser.add_argument("--gate-mode", choices=["none", "pixel", "component"], default="none")
    parser.add_argument("--boundary-radius", type=int, default=5)
    parser.add_argument(
        "--boundary-source",
        choices=["anchor", "candidate", "either"],
        default="anchor",
    )
    parser.add_argument("--component-min-boundary-rate", type=float, default=0.6)
    parser.add_argument("--component-max-area", type=int, default=0)
    parser.add_argument("--out-json", default=None)
    return parser.parse_args()


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


def semantic_boundary_band(pred, radius):
    edge = np.zeros(pred.shape, dtype=bool)
    edge[:, 1:] |= pred[:, 1:] != pred[:, :-1]
    edge[:, :-1] |= pred[:, 1:] != pred[:, :-1]
    edge[1:, :] |= pred[1:, :] != pred[:-1, :]
    edge[:-1, :] |= pred[1:, :] != pred[:-1, :]
    if radius <= 0:
        return edge
    kernel = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)
    return cv2.dilate(edge.astype(np.uint8), kernel, iterations=1) > 0


def boundary_for_source(anchor, candidate, radius, source):
    anchor_boundary = semantic_boundary_band(anchor, radius)
    if source == "anchor":
        return anchor_boundary
    candidate_boundary = semantic_boundary_band(candidate, radius)
    if source == "candidate":
        return candidate_boundary
    return anchor_boundary | candidate_boundary


def component_gate(raw_take, boundary, min_boundary_rate, max_area):
    if not np.any(raw_take):
        return raw_take, 0, 0
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        raw_take.astype(np.uint8),
        connectivity=8,
    )
    keep = np.zeros(raw_take.shape, dtype=bool)
    accepted = 0
    rejected = 0
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        comp = labels == label
        boundary_pixels = int(np.count_nonzero(comp & boundary))
        boundary_rate = boundary_pixels / max(1, area)
        take = boundary_rate >= min_boundary_rate or (max_area > 0 and area <= max_area)
        if take:
            keep[comp] = True
            accepted += 1
        else:
            rejected += 1
    return keep, accepted, rejected


def json_metrics(metrics):
    return {
        key: (value.tolist() if isinstance(value, np.ndarray) else value)
        for key, value in metrics.items()
    }


def main():
    args = parse_args()
    cache_root = Path(args.cache_root)
    class_ids = sorted(set(args.class_id))
    rows = read_rows(cache_root, args.anchor_model, args.dataset)
    anchor_maps = map_paths(cache_root, args.anchor_model, args.dataset)
    candidate_maps = map_paths(cache_root, args.candidate_model, args.dataset)
    if len(candidate_maps) != len(anchor_maps):
        raise ValueError(
            f"Map count mismatch: candidate has {len(candidate_maps)}, anchor has {len(anchor_maps)}"
        )

    anchor_meter = SegmentationStats()
    merged_meter = SegmentationStats()
    counts = {
        "files": 0,
        "pixels": 0,
        "raw_changed_pixels": 0,
        "kept_changed_pixels": 0,
        "raw_changed_pixels_by_class": {str(class_id): 0 for class_id in class_ids},
        "kept_changed_pixels_by_class": {str(class_id): 0 for class_id in class_ids},
        "candidate_class_pixels": {str(class_id): 0 for class_id in class_ids},
        "accepted_components": 0,
        "rejected_components": 0,
    }

    for index, row in enumerate(rows):
        label = load_label(row["label_path"])
        anchor = load_pred(anchor_maps[index], label.shape)
        candidate = load_pred(candidate_maps[index], label.shape)
        boundary = None
        if args.gate_mode != "none":
            boundary = boundary_for_source(
                anchor,
                candidate,
                radius=args.boundary_radius,
                source=args.boundary_source,
            )
        merged = anchor.copy()

        for class_id in class_ids:
            class_key = str(class_id)
            candidate_class = candidate == class_id
            raw_take = candidate_class & (anchor != class_id)
            counts["candidate_class_pixels"][class_key] += int(np.count_nonzero(candidate_class))
            raw_count = int(np.count_nonzero(raw_take))
            counts["raw_changed_pixels_by_class"][class_key] += raw_count
            counts["raw_changed_pixels"] += raw_count

            if args.gate_mode == "none":
                keep = raw_take
            elif args.gate_mode == "pixel":
                keep = raw_take & boundary
            else:
                keep, accepted, rejected = component_gate(
                    raw_take,
                    boundary,
                    min_boundary_rate=args.component_min_boundary_rate,
                    max_area=args.component_max_area,
                )
                counts["accepted_components"] += accepted
                counts["rejected_components"] += rejected

            kept_count = int(np.count_nonzero(keep))
            counts["kept_changed_pixels_by_class"][class_key] += kept_count
            counts["kept_changed_pixels"] += kept_count
            merged[keep] = class_id

        anchor_meter.update(anchor, label)
        merged_meter.update(merged, label)
        counts["files"] += 1
        counts["pixels"] += int(anchor.size)

    counts["raw_changed_pixel_rate"] = counts["raw_changed_pixels"] / max(1, counts["pixels"])
    counts["kept_changed_pixel_rate"] = counts["kept_changed_pixels"] / max(1, counts["pixels"])
    counts["kept_share_of_raw"] = counts["kept_changed_pixels"] / max(1, counts["raw_changed_pixels"])

    anchor_metrics = anchor_meter.metrics()
    merged_metrics = merged_meter.metrics()
    payload = {
        "dataset": args.dataset,
        "anchor_model": args.anchor_model,
        "candidate_model": args.candidate_model,
        "class_ids": class_ids,
        "gate_mode": args.gate_mode,
        "boundary_radius": args.boundary_radius,
        "boundary_source": args.boundary_source,
        "component_min_boundary_rate": args.component_min_boundary_rate,
        "component_max_area": args.component_max_area,
        "anchor": json_metrics(anchor_metrics),
        "merged": json_metrics(merged_metrics),
        "delta_mIoU": merged_metrics["mIoU"] - anchor_metrics["mIoU"],
        "counts": counts,
    }

    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.out_json:
        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")


if __name__ == "__main__":
    main()
