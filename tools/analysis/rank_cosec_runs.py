#!/usr/bin/env python
"""Rank CoSEC training runs by best Day/Night validation mIoU."""

import argparse
import json
import math
from pathlib import Path


ANCHORS = {
    "day_checkpoint": 65.4352,
    "night_checkpoint": 50.4284,
    "day_tta": 66.7474,
    "night_tta": 52.4609,
}


def valid_number(value):
    return isinstance(value, (int, float)) and not math.isnan(float(value))


def read_metrics(path):
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def split_value(record, split):
    if split == "day":
        return record.get("cosec_day_val/sem_seg/mIoU", record.get("best_day_mIoU"))
    if split == "night":
        return record.get("cosec_night_val/sem_seg/mIoU", record.get("best_night_mIoU"))
    raise ValueError(split)


def validation_records(records):
    vals = []
    for record in records:
        day = split_value(record, "day")
        night = split_value(record, "night")
        if valid_number(day) or valid_number(night):
            vals.append(record)
    return vals


def best_record(vals, split):
    candidates = [record for record in vals if valid_number(split_value(record, split))]
    if not candidates:
        return None
    return max(candidates, key=lambda record: float(split_value(record, split)))


def fmt(value, digits=4):
    if not valid_number(value):
        return ""
    return f"{float(value):.{digits}f}"


def diff(value, anchor):
    if not valid_number(value):
        return ""
    return f"{float(value) - float(anchor):+.4f}"


def should_skip(path, include_aux):
    name = path.parent.name.lower()
    if include_aux:
        return False
    aux_markers = ("_smoke", "_sanity", "sanity_", "smoke_", "diagnostic", "diagnostics")
    return any(marker in name for marker in aux_markers)


def summarize_metrics_file(path):
    records = read_metrics(path)
    vals = validation_records(records)
    day_best = best_record(vals, "day")
    night_best = best_record(vals, "night")
    latest = vals[-1] if vals else {}

    return {
        "run": path.parent.name,
        "path": str(path),
        "records": len(records),
        "vals": len(vals),
        "latest_iter": latest.get("iteration", ""),
        "latest_day": split_value(latest, "day") if latest else None,
        "latest_night": split_value(latest, "night") if latest else None,
        "best_day": split_value(day_best, "day") if day_best else None,
        "best_day_iter": day_best.get("iteration", "") if day_best else "",
        "night_at_best_day": split_value(day_best, "night") if day_best else None,
        "best_night": split_value(night_best, "night") if night_best else None,
        "best_night_iter": night_best.get("iteration", "") if night_best else "",
        "day_at_best_night": split_value(night_best, "day") if night_best else None,
    }


def collect_runs(root, include_aux):
    rows = []
    for path in sorted(Path(root).glob("*/metrics.json")):
        if should_skip(path, include_aux):
            continue
        row = summarize_metrics_file(path)
        if row["vals"] > 0:
            rows.append(row)
    return rows


def print_table(rows, sort_key, top_k):
    sorted_rows = sorted(
        rows,
        key=lambda row: float(row.get(sort_key) or -999.0) if valid_number(row.get(sort_key)) else -999.0,
        reverse=True,
    )
    if top_k > 0:
        sorted_rows = sorted_rows[:top_k]

    print(
        "| Run | Vals | Best Day | dDay ckpt | dDay TTA | Iter | Night@BestDay | "
        "Best Night | dNight ckpt | dNight TTA | Iter | Day@BestNight | Latest Day | Latest Night |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in sorted_rows:
        print(
            f"| {row['run']} | {row['vals']} | {fmt(row['best_day'])} | "
            f"{diff(row['best_day'], ANCHORS['day_checkpoint'])} | {diff(row['best_day'], ANCHORS['day_tta'])} | "
            f"{row['best_day_iter']} | {fmt(row['night_at_best_day'])} | "
            f"{fmt(row['best_night'])} | {diff(row['best_night'], ANCHORS['night_checkpoint'])} | "
            f"{diff(row['best_night'], ANCHORS['night_tta'])} | {row['best_night_iter']} | "
            f"{fmt(row['day_at_best_night'])} | {fmt(row['latest_day'])} | {fmt(row['latest_night'])} |"
        )


def print_summary(rows):
    day_above_ckpt = sum(1 for row in rows if valid_number(row["best_day"]) and row["best_day"] >= ANCHORS["day_checkpoint"])
    day_above_tta = sum(1 for row in rows if valid_number(row["best_day"]) and row["best_day"] >= ANCHORS["day_tta"])
    night_above_ckpt = sum(
        1 for row in rows if valid_number(row["best_night"]) and row["best_night"] >= ANCHORS["night_checkpoint"]
    )
    night_above_tta = sum(
        1 for row in rows if valid_number(row["best_night"]) and row["best_night"] >= ANCHORS["night_tta"]
    )
    print(
        f"Scanned {len(rows)} runs with validation records. "
        f"Day >= checkpoint/TTA: {day_above_ckpt}/{day_above_tta}. "
        f"Night >= checkpoint/TTA: {night_above_ckpt}/{night_above_tta}."
    )
    print()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="work_dirs")
    parser.add_argument("--sort", choices=["best_day", "best_night", "latest_day", "latest_night"], default="best_day")
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--include-aux", action="store_true", help="Include smoke/sanity/diagnostic dirs.")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = collect_runs(args.root, args.include_aux)
    print_summary(rows)
    print_table(rows, args.sort, args.top_k)


if __name__ == "__main__":
    main()
