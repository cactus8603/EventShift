#!/usr/bin/env python
"""Derive class routes from cached per-class IoU tables."""

import argparse
import csv
import json
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", default="work_dirs/ensemble_feature_cache")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--anchor-model", required=True)
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--out-json", required=True)
    return parser.parse_args()


def parse_float(value):
    if value is None or value == "":
        return float("nan")
    return float(value)


def read_per_class(path):
    rows = {}
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            class_id = int(row["class_id"])
            rows[class_id] = {
                "class_id": class_id,
                "class_name": row["class_name"],
                "iou": parse_float(row.get("iou")),
                "gt_pixels": int(float(row.get("gt_pixels") or 0)),
                "pred_pixels": int(float(row.get("pred_pixels") or 0)),
            }
    return rows


def main():
    args = parse_args()
    cache_root = Path(args.cache_root)
    anchor = read_per_class(cache_root / args.anchor_model / f"{args.dataset}_per_class_iou.csv")
    candidate = read_per_class(cache_root / args.candidate_model / f"{args.dataset}_per_class_iou.csv")

    routes = []
    class_rows = []
    for class_id, anchor_row in sorted(anchor.items()):
        candidate_row = candidate[class_id]
        anchor_iou = anchor_row["iou"]
        candidate_iou = candidate_row["iou"]
        delta = candidate_iou - anchor_iou
        use_candidate = (
            not math.isnan(anchor_iou)
            and not math.isnan(candidate_iou)
            and delta > float(args.min_delta)
        )
        row = {
            "class_id": class_id,
            "class_name": anchor_row["class_name"],
            "anchor_iou": anchor_iou,
            "candidate_iou": candidate_iou,
            "delta_iou": delta,
            "gt_pixels": anchor_row["gt_pixels"],
            "use_candidate": use_candidate,
        }
        class_rows.append(row)
        if use_candidate:
            routes.append(
                {
                    "class_id": class_id,
                    "class_name": anchor_row["class_name"],
                    "candidate": args.candidate_name,
                    "anchor_iou": anchor_iou,
                    "candidate_iou": candidate_iou,
                    "delta_iou": delta,
                    "source_dataset": args.dataset,
                }
            )

    payload = {
        "dataset": args.dataset,
        "anchor_model": args.anchor_model,
        "candidate_model": args.candidate_model,
        "candidate_name": args.candidate_name,
        "min_delta": args.min_delta,
        "routes": routes,
        "class_rows": class_rows,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote routes: {out_json}")
    print(f"route_count={len(routes)}")
    for route in routes:
        print(
            f"{route['class_name']}: {route['anchor_iou']:.4f} -> "
            f"{route['candidate_iou']:.4f} ({route['delta_iou']:+.4f})"
        )


if __name__ == "__main__":
    main()
