#!/usr/bin/env python
"""Summarize current Day/Night anchor class-IoU gaps from eval pth files."""

import argparse
import math
from pathlib import Path

import torch


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

DEFAULT_DAY_BASE = (
    "work_dirs/eval_dayonly65_43_rgb_tta5126247681024_flip_dayonly"
    "/inference/sem_seg_evaluation.pth"
)
DEFAULT_DAY_TTA = (
    "work_dirs/eval_dayonly65_43_rgb_tta5126247681024_flip_dayonly"
    "/inference_TTA/sem_seg_evaluation.pth"
)
DEFAULT_NIGHT_BASE = (
    "work_dirs/eval_dayonly65_43_rgb_tta_512_624_768_1024_flip"
    "/inference/sem_seg_evaluation.pth"
)
DEFAULT_NIGHT_TTA = (
    "work_dirs/eval_dayonly65_43_rgb_tta_512_624_768_1024_flip"
    "/inference_TTA/sem_seg_evaluation.pth"
)


def load_eval(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(str(path), map_location="cpu")


def is_valid(value):
    return isinstance(value, (int, float)) and not math.isnan(float(value))


def fmt(value):
    if not is_valid(value):
        return ""
    return f"{float(value):.2f}"


def fmt_delta(base, tuned):
    if not is_valid(base) or not is_valid(tuned):
        return ""
    return f"{float(tuned) - float(base):+.2f}"


def present_count(metrics):
    return sum(1 for cls in CLASSES if is_valid(metrics.get(f"IoU-{cls}")))


def print_summary(day_base, day_tta, night_base, night_tta, day_target, night_target):
    rows = [
        ("Day", day_base, day_tta, day_target),
        ("Night", night_base, night_tta, night_target),
    ]
    print("| Split | Base mIoU | TTA mIoU | TTA Delta | Present Classes | Gap To Target | Needed Class-IoU Points |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for split, base, tuned, target in rows:
        count = present_count(tuned)
        miou = float(tuned["mIoU"])
        gap = max(0.0, float(target) - miou)
        needed = gap * count
        print(
            f"| {split} | {float(base['mIoU']):.4f} | {miou:.4f} | "
            f"{miou - float(base['mIoU']):+.4f} | {count} | {gap:.4f} | {needed:.2f} |"
        )


def print_class_table(day_base, day_tta, night_base, night_tta, day_target, night_target):
    print()
    print("| Class | Day TTA | Day Delta | Day Gap To 75 | Night TTA | Night Delta | Night Gap To 65 |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for cls in CLASSES:
        db = day_base.get(f"IoU-{cls}", float("nan"))
        dt = day_tta.get(f"IoU-{cls}", float("nan"))
        nb = night_base.get(f"IoU-{cls}", float("nan"))
        nt = night_tta.get(f"IoU-{cls}", float("nan"))
        day_gap = "" if not is_valid(dt) else f"{max(0.0, day_target - float(dt)):.2f}"
        night_gap = "" if not is_valid(nt) else f"{max(0.0, night_target - float(nt)):.2f}"
        print(
            f"| {cls} | {fmt(dt)} | {fmt_delta(db, dt)} | {day_gap} | "
            f"{fmt(nt)} | {fmt_delta(nb, nt)} | {night_gap} |"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--day-base", default=DEFAULT_DAY_BASE)
    parser.add_argument("--day-tta", default=DEFAULT_DAY_TTA)
    parser.add_argument("--night-base", default=DEFAULT_NIGHT_BASE)
    parser.add_argument("--night-tta", default=DEFAULT_NIGHT_TTA)
    parser.add_argument("--day-target", type=float, default=75.0)
    parser.add_argument("--night-target", type=float, default=65.0)
    args = parser.parse_args()

    day_base = load_eval(args.day_base)
    day_tta = load_eval(args.day_tta)
    night_base = load_eval(args.night_base)
    night_tta = load_eval(args.night_tta)
    print_summary(day_base, day_tta, night_base, night_tta, args.day_target, args.night_target)
    print_class_table(day_base, day_tta, night_base, night_tta, args.day_target, args.night_target)


if __name__ == "__main__":
    main()
