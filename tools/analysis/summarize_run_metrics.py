#!/usr/bin/env python
"""Print compact summaries from Detectron2 metrics.json files."""

import argparse
import json
from pathlib import Path


DEFAULT_KEYS = (
    "iteration",
    "total_loss",
    "day_event_boundary_refiner/event_active_fraction",
    "day_event_boundary_refiner/allowed_loss_mask",
    "day_event_boundary_refiner/repair_positive_fraction",
    "day_event_boundary_refiner/repair_negative_fraction",
    "day_event_boundary_refiner/gate_mean",
    "day_event_boundary_refiner/gate_active_001",
    "day_event_boundary_refiner/effective_gate_mean",
    "day_event_boundary_refiner/effective_gate_max",
    "day_event_boundary_refiner/final_gate_scale",
    "day_event_boundary_refiner/gate_repair_positive_mean",
    "day_event_boundary_refiner/gate_repair_negative_mean",
    "day_event_boundary_refiner/effective_gate_repair_positive_mean",
    "day_event_boundary_refiner/effective_gate_repair_negative_mean",
    "day_event_boundary_refiner/alpha",
    "day_event_boundary_refiner/allowed_ce_candidate_delta",
    "day_event_boundary_refiner/allowed_ce_final_delta",
    "day_event_boundary_refiner/final_to_candidate_delta_ratio",
    "event_edge/f1",
    "event_edge/precision",
    "event_edge/recall",
)


def load_rows(path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def important_scores(record):
    return {
        key: value
        for key, value in sorted(record.items())
        if key.startswith("best_")
        or "mIoU" in key
        or key.endswith("/mIoU")
        or key.endswith("sem_seg/mIoU")
    }


def latest_score_record(rows):
    for record in reversed(rows):
        scores = important_scores(record)
        if scores:
            return record.get("iteration"), scores
    return None, {}


def best_scores(rows):
    best = {}
    for record in rows:
        for key, value in important_scores(record).items():
            if not isinstance(value, (int, float)):
                continue
            if key not in best or value > best[key]:
                best[key] = value
    return dict(sorted(best.items()))


def summarize(path, keys):
    rows = load_rows(path)
    print(f"== {path} ==")
    print(f"records: {len(rows)}")
    if not rows:
        return
    last = rows[-1]
    for key in keys:
        if key in last:
            print(f"{key}: {last[key]}")
    scores = important_scores(last)
    if scores:
        print("scores:")
        for key, value in scores.items():
            print(f"  {key}: {value}")
    latest_iter, latest_scores = latest_score_record(rows)
    if latest_scores and latest_scores != scores:
        print(f"latest score record: iteration {latest_iter}")
        for key, value in latest_scores.items():
            print(f"  {key}: {value}")
    all_best = best_scores(rows)
    if all_best:
        print("best scores seen:")
        for key, value in all_best.items():
            print(f"  {key}: {value}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", nargs="+", type=Path)
    parser.add_argument("--keys", nargs="*", default=list(DEFAULT_KEYS))
    args = parser.parse_args()
    for path in args.metrics:
        summarize(path, tuple(args.keys))


if __name__ == "__main__":
    main()
