#!/usr/bin/env python
"""Summarize SSL exp9-style runs and recommend first-val actions."""

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

DEFAULT_RUNS = [
    "work_dirs/ssl-exp9b_currentbest_tta_segformer_agree_headonly_lr1e-7_bs8",
    "work_dirs/ssl-exp9d_currentbest_tta_segformer_agree_rare_boundary_headonly_lr5e-8_bs8",
    "work_dirs/ssl-exp9e_currentbest_tta_segformer_agree_gap_focus_headonly_lr5e-8_bs8",
    "work_dirs/ssl-exp9f_currentbest_tta_day_gap_focus_headonly_lr5e-8_bs8",
    "work_dirs/ssl-exp9g_currentbest_tta_night_gap_focus_headonly_lr5e-8_bs8",
]


def valid_number(value):
    return isinstance(value, (int, float)) and not math.isnan(float(value))


def read_metrics(path):
    metrics_path = Path(path) / "metrics.json"
    if not metrics_path.exists():
        return []
    records = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def validation_records(records):
    vals = []
    for record in records:
        day = record.get("cosec_day_val/sem_seg/mIoU", record.get("best_day_mIoU"))
        night = record.get("cosec_night_val/sem_seg/mIoU", record.get("best_night_mIoU"))
        if valid_number(day) or valid_number(night):
            vals.append(
                {
                    "iteration": record.get("iteration", ""),
                    "day": float(day) if valid_number(day) else None,
                    "night": float(night) if valid_number(night) else None,
                    "raw": record,
                }
            )
    return vals


def infer_primary(run_name):
    lower = run_name.lower()
    if "night_gap" in lower or "exp9g" in lower:
        return "night"
    if "day_gap" in lower or "exp9f" in lower:
        return "day"
    return "both"


def fmt(value, digits=4):
    if value is None or not valid_number(value):
        return ""
    return f"{float(value):.{digits}f}"


def diff(value, anchor):
    if value is None or not valid_number(value):
        return ""
    return f"{float(value) - float(anchor):+.4f}"


def best_value(vals, key):
    numbers = [item[key] for item in vals if valid_number(item.get(key))]
    return max(numbers) if numbers else None


def decision(primary, latest_day, latest_night, first_val_count):
    if first_val_count == 0:
        return "pending"

    checks = []
    if primary in {"day", "both"}:
        checks.append(("day", latest_day, ANCHORS["day_checkpoint"], ANCHORS["day_tta"]))
    if primary in {"night", "both"}:
        checks.append(("night", latest_night, ANCHORS["night_checkpoint"], ANCHORS["night_tta"]))

    if any(value is None for _, value, _, _ in checks):
        return "watch: missing split metric"

    if any(value >= tta_anchor for _, value, _, tta_anchor in checks):
        return "keep: beats practical TTA anchor"

    if all(value >= checkpoint_anchor for _, value, checkpoint_anchor, _ in checks):
        return "watch: above checkpoint anchor, below TTA"

    if any(value < checkpoint_anchor - 0.25 for _, value, checkpoint_anchor, _ in checks):
        return "stop: below checkpoint anchor by >0.25"

    return "watch: near checkpoint anchor"


def summarize_run(path):
    path = Path(path)
    records = read_metrics(path)
    vals = validation_records(records)
    primary = infer_primary(path.name)
    latest = vals[-1] if vals else {"iteration": "", "day": None, "night": None}
    best_day = best_value(vals, "day")
    best_night = best_value(vals, "night")
    action = decision(primary, latest["day"], latest["night"], len(vals))
    return {
        "run": path.name,
        "exists": path.exists(),
        "primary": primary,
        "val_count": len(vals),
        "latest_iter": latest["iteration"],
        "latest_day": latest["day"],
        "latest_night": latest["night"],
        "best_day": best_day,
        "best_night": best_night,
        "decision": action,
    }


def print_table(rows):
    print(
        "| Run | Primary | Vals | Latest Iter | Latest Day | dDay vs ckpt | dDay vs TTA | "
        "Latest Night | dNight vs ckpt | dNight vs TTA | Best Day | Best Night | Decision |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in rows:
        print(
            f"| {row['run']} | {row['primary']} | {row['val_count']} | {row['latest_iter']} | "
            f"{fmt(row['latest_day'])} | {diff(row['latest_day'], ANCHORS['day_checkpoint'])} | "
            f"{diff(row['latest_day'], ANCHORS['day_tta'])} | {fmt(row['latest_night'])} | "
            f"{diff(row['latest_night'], ANCHORS['night_checkpoint'])} | "
            f"{diff(row['latest_night'], ANCHORS['night_tta'])} | {fmt(row['best_day'])} | "
            f"{fmt(row['best_night'])} | {row['decision']} |"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="*", help="Run directories. Defaults to expected exp9b/d/e/f/g dirs.")
    parser.add_argument(
        "--include-existing-exp9",
        action="store_true",
        help="Append all existing work_dirs/ssl-exp9* directories.",
    )
    args = parser.parse_args()

    run_paths = list(args.runs or DEFAULT_RUNS)
    if args.include_existing_exp9:
        for path in sorted(Path("work_dirs").glob("ssl-exp9*")):
            path_str = str(path)
            if path_str not in run_paths:
                run_paths.append(path_str)

    rows = [summarize_run(path) for path in run_paths]
    print_table(rows)


if __name__ == "__main__":
    main()
