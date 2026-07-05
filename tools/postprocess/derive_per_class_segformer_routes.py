#!/usr/bin/env python
"""Derive class-wise candidate routes from per-class validation CSVs."""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from cosec_finetune_splits import CLASSES


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-csv", required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Candidate mapping in the form name=/path/to/per_class_iou.csv. Repeatable.",
    )
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--metric", default="iou", choices=["iou", "acc"])
    parser.add_argument("--min-gap", type=float, default=0.0)
    parser.add_argument("--min-candidate-pred-pixels", type=int, default=0)
    parser.add_argument("--require-acc-not-worse", action="store_true")
    parser.add_argument("--allow-classes", nargs="*", default=None)
    parser.add_argument("--block-classes", nargs="*", default=[])
    parser.add_argument("--dataset-name", default="")
    return parser.parse_args()


def parse_candidate(items):
    parsed = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--candidate must be name=csv, got: {item}")
        name, path = item.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"Empty candidate name/path: {item}")
        if name in parsed:
            raise ValueError(f"Duplicate candidate name: {name}")
        parsed[name] = Path(path)
    if not parsed:
        raise ValueError("At least one --candidate is required.")
    return parsed


def parse_float(value):
    if value in ("", None):
        return None
    return float(value)


def load_rows(path):
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    by_class = {}
    for row in rows:
        name = row["class_name"]
        by_class[name] = row
    missing = [name for name in CLASSES if name not in by_class]
    if missing:
        raise ValueError(f"{path} is missing classes: {missing}")
    return by_class


def class_filter(allow_classes, block_classes):
    allow = set(allow_classes) if allow_classes else set(CLASSES)
    block = set(block_classes)
    unknown = (allow | block) - set(CLASSES)
    if unknown:
        raise ValueError(f"Unknown classes: {sorted(unknown)}")
    return allow - block


def main():
    args = parse_args()
    anchor_csv = Path(args.anchor_csv)
    candidates = parse_candidate(args.candidate)
    allowed = class_filter(args.allow_classes, args.block_classes)

    anchor = load_rows(anchor_csv)
    candidate_rows = {name: load_rows(path) for name, path in candidates.items()}

    routes = []
    rejected = []
    for class_id, class_name in enumerate(CLASSES):
        anchor_metric = parse_float(anchor[class_name].get(args.metric))
        anchor_acc = parse_float(anchor[class_name].get("acc"))
        if class_name not in allowed or anchor_metric is None:
            rejected.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "reason": "filtered_or_missing_anchor",
                    "anchor_metric": anchor_metric,
                }
            )
            continue

        best = None
        for candidate_name, rows in candidate_rows.items():
            row = rows[class_name]
            cand_metric = parse_float(row.get(args.metric))
            cand_acc = parse_float(row.get("acc"))
            cand_pred_pixels = int(float(row.get("pred_pixels") or 0))
            if cand_metric is None:
                continue
            if cand_pred_pixels < args.min_candidate_pred_pixels:
                continue
            if args.require_acc_not_worse and anchor_acc is not None and cand_acc is not None and cand_acc < anchor_acc:
                continue
            gain = cand_metric - anchor_metric
            item = {
                "class_id": class_id,
                "class_name": class_name,
                "candidate": candidate_name,
                "metric": args.metric,
                "anchor_metric": anchor_metric,
                "candidate_metric": cand_metric,
                "gain": gain,
                "anchor_acc": anchor_acc,
                "candidate_acc": cand_acc,
                "candidate_pred_pixels": cand_pred_pixels,
            }
            if best is None or item["gain"] > best["gain"]:
                best = item

        if best is not None and best["gain"] >= args.min_gap:
            routes.append(best)
        else:
            rejected.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "reason": "no_candidate_over_gap",
                    "anchor_metric": anchor_metric,
                    "best": best,
                }
            )

    routes.sort(key=lambda row: row["gain"], reverse=True)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_name": args.dataset_name,
        "anchor_csv": str(anchor_csv.resolve()),
        "candidates": {name: str(path.resolve()) for name, path in candidates.items()},
        "metric": args.metric,
        "min_gap": args.min_gap,
        "min_candidate_pred_pixels": args.min_candidate_pred_pixels,
        "require_acc_not_worse": args.require_acc_not_worse,
        "allow_classes": sorted(allowed),
        "block_classes": list(args.block_classes),
        "routes": routes,
        "rejected": rejected,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote routes: {out_json}")
    print("Selected routes:")
    for route in routes:
        print(
            f"  {route['class_name']:15s} -> {route['candidate']:28s} "
            f"{route['anchor_metric']:.3f} -> {route['candidate_metric']:.3f} "
            f"gain={route['gain']:.3f}"
        )


if __name__ == "__main__":
    main()
