#!/usr/bin/env python
"""Compare exp9-style validation records against Day/Night anchors by class."""

import argparse
import json
import math
from pathlib import Path


CLASSES = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]

ANCHOR_PATHS = {
    "checkpoint": {
        "day": (
            "work_dirs/swinL_cosec_dayonly_from_day65_freeze_backbone_lr5e-7"
            "/inference/sem_seg_evaluation.pth"
        ),
        "night": "work_dirs/swinL_seqholdout305_from_latest_night/inference/sem_seg_evaluation.pth",
    },
    "tta": {
        "day": (
            "work_dirs/eval_dayonly65_43_rgb_tta5126247681024_flip_dayonly"
            "/inference_TTA/sem_seg_evaluation.pth"
        ),
        "night": (
            "work_dirs/eval_dayonly65_43_rgb_tta_512_624_768_1024_flip"
            "/inference_TTA/sem_seg_evaluation.pth"
        ),
    },
}

CHECKPOINT_ANCHOR_RUNS = {
    "day": "work_dirs/swinL_cosec_dayonly_from_day65_freeze_backbone_lr5e-7",
    "night": "work_dirs/swinL_seqholdout305_from_latest_night",
}

TARGETS = {
    "day": 75.0,
    "night": 65.0,
}

SPLIT_PREFIX = {
    "day": "cosec_day_val/sem_seg",
    "night": "cosec_night_val/sem_seg",
}


def valid_number(value):
    return isinstance(value, (int, float)) and not math.isnan(float(value))


def fmt(value, digits=4):
    if not valid_number(value):
        return ""
    return f"{float(value):.{digits}f}"


def fmt_signed(value, digits=4):
    if not valid_number(value):
        return ""
    return f"{float(value):+.{digits}f}"


def read_metrics(run_dir):
    metrics_path = Path(run_dir) / "metrics.json"
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


def split_miou(record, split):
    key = f"{SPLIT_PREFIX[split]}/mIoU"
    return record.get(key, record.get(f"best_{split}_mIoU"))


def validation_records(records, split):
    vals = []
    for record in records:
        miou = split_miou(record, split)
        if valid_number(miou):
            vals.append(record)
    return vals


def select_record(records, split, mode):
    vals = validation_records(records, split)
    if not vals:
        return None
    if mode == "latest":
        return vals[-1]
    if mode == "best":
        return max(vals, key=lambda record: float(split_miou(record, split)))
    raise ValueError(f"Unsupported record mode: {mode}")


def extract_split_metrics(record, split):
    prefix = SPLIT_PREFIX[split] + "/"
    metrics = {}
    for key, value in record.items():
        if key.startswith(prefix):
            metrics[key[len(prefix) :]] = value
    if not metrics and valid_number(record.get("mIoU")):
        metrics = dict(record)
    return metrics


def load_eval_pth(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Loading .pth anchors needs torch. Run with "
            "`conda run --no-capture-output -n mask2former python "
            "tools/analyze_ssl_exp9_class_effects.py ...`."
        ) from exc
    return torch.load(str(path), map_location="cpu")


def load_checkpoint_anchor_from_metrics(split):
    run_dir = CHECKPOINT_ANCHOR_RUNS[split]
    records = read_metrics(run_dir)
    record = select_record(records, split, "best")
    if record is None:
        raise FileNotFoundError(f"No checkpoint anchor metrics found for {split}: {run_dir}")
    return extract_split_metrics(record, split), f"{run_dir}/metrics.json best {split}"


def load_anchor(anchor_kind, split, override_path=None):
    if anchor_kind == "checkpoint" and override_path is None:
        return load_checkpoint_anchor_from_metrics(split)
    path = override_path or ANCHOR_PATHS[anchor_kind][split]
    return load_eval_pth(path), path


def class_rows(anchor, run_metrics, split):
    rows = []
    for cls in CLASSES:
        anchor_iou = anchor.get(f"IoU-{cls}", float("nan"))
        run_iou = run_metrics.get(f"IoU-{cls}", float("nan"))
        if not valid_number(anchor_iou) and not valid_number(run_iou):
            continue
        delta = float(run_iou) - float(anchor_iou) if valid_number(anchor_iou) and valid_number(run_iou) else float("nan")
        gap = max(0.0, TARGETS[split] - float(run_iou)) if valid_number(run_iou) else float("nan")
        rows.append(
            {
                "class": cls,
                "anchor": anchor_iou,
                "run": run_iou,
                "delta": delta,
                "gap": gap,
            }
        )
    return rows


def sort_rows(rows, sort_mode):
    if sort_mode == "delta":
        return sorted(rows, key=lambda row: row["delta"] if valid_number(row["delta"]) else -999.0)
    if sort_mode == "abs_delta":
        return sorted(rows, key=lambda row: abs(row["delta"]) if valid_number(row["delta"]) else -1.0, reverse=True)
    if sort_mode == "gap":
        return sorted(rows, key=lambda row: row["gap"] if valid_number(row["gap"]) else -1.0, reverse=True)
    raise ValueError(f"Unsupported sort: {sort_mode}")


def print_run_split(run_dir, split, anchor_kind, anchor_path, record, anchor, top_k, sort_mode):
    run_metrics = extract_split_metrics(record, split)
    run_miou = run_metrics.get("mIoU", split_miou(record, split))
    anchor_miou = anchor.get("mIoU")
    iteration = record.get("iteration", "")
    delta_miou = float(run_miou) - float(anchor_miou) if valid_number(run_miou) and valid_number(anchor_miou) else float("nan")
    target_gap = max(0.0, TARGETS[split] - float(run_miou)) if valid_number(run_miou) else float("nan")

    print()
    print(f"## {Path(run_dir).name} / {split} / iter {iteration}")
    print()
    print(f"- anchor: `{anchor_kind}` ({anchor_path})")
    print(f"- mIoU: {fmt(run_miou)} vs anchor {fmt(anchor_miou)} ({fmt_signed(delta_miou)})")
    print(f"- gap to target {TARGETS[split]:.1f}: {fmt(target_gap)}")
    print()

    rows = sort_rows(class_rows(anchor, run_metrics, split), sort_mode)
    if top_k > 0:
        if sort_mode == "delta":
            selected = rows[:top_k]
        else:
            selected = rows[:top_k]
    else:
        selected = rows

    print("| Class | Anchor IoU | Run IoU | Delta | Gap To Target |")
    print("|---|---:|---:|---:|---:|")
    for row in selected:
        print(
            f"| {row['class']} | {fmt(row['anchor'], 2)} | {fmt(row['run'], 2)} | "
            f"{fmt_signed(row['delta'], 2)} | {fmt(row['gap'], 2)} |"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="Run directories containing metrics.json.")
    parser.add_argument("--split", choices=["day", "night", "both"], default="both")
    parser.add_argument("--anchor", choices=["checkpoint", "tta"], default="tta")
    parser.add_argument("--day-anchor-path")
    parser.add_argument("--night-anchor-path")
    parser.add_argument("--record", choices=["latest", "best"], default="latest")
    parser.add_argument("--sort", choices=["abs_delta", "delta", "gap"], default="abs_delta")
    parser.add_argument("--top-k", type=int, default=12, help="Rows per run/split. Use 0 for all classes.")
    return parser.parse_args()


def main():
    args = parse_args()
    splits = ["day", "night"] if args.split == "both" else [args.split]
    anchor_overrides = {
        "day": args.day_anchor_path,
        "night": args.night_anchor_path,
    }

    anchors = {}
    for split in splits:
        anchors[split] = load_anchor(args.anchor, split, anchor_overrides[split])

    for run_dir in args.runs:
        records = read_metrics(run_dir)
        if not records:
            print()
            print(f"## {Path(run_dir).name}")
            print()
            print("No metrics.json records found.")
            continue
        for split in splits:
            record = select_record(records, split, args.record)
            if record is None:
                print()
                print(f"## {Path(run_dir).name} / {split}")
                print()
                print("No validation records found for this split.")
                continue
            anchor, anchor_path = anchors[split]
            print_run_split(run_dir, split, args.anchor, anchor_path, record, anchor, args.top_k, args.sort)


if __name__ == "__main__":
    main()
