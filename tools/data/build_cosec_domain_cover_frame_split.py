#!/usr/bin/env python
"""Build a CoSEC frame split with strong per-domain class coverage."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

def _eventshift_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "configs").is_dir() and (parent / "third_party").is_dir():
            return parent
    return Path(__file__).resolve().parents[1]


ROOT = _eventshift_root()
sys.path.insert(0, str(ROOT / "tools"))

from cosec_finetune_splits import CLASSES, DEFAULT_COSEC_TRAIN_ROOT, SPLIT_DIR, cosec_domain  # noqa: E402


def iter_all_samples(root):
    root = Path(root)
    for seq_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if not seq_dir.name.startswith(("Day_", "Night_")):
            continue
        image_dir = seq_dir / "img_co_left"
        label_dir = seq_dir / "segment_co"
        if not image_dir.is_dir() or not label_dir.is_dir():
            continue
        for label_path in sorted(label_dir.glob("*.png")):
            image_path = image_dir / label_path.name
            if image_path.is_file():
                yield seq_dir.name, int(label_path.stem), image_path, label_path


def load_label_counts(label_path):
    label = np.asarray(Image.open(label_path))
    if label.ndim == 3:
        label = label[:, :, 0]
    valid = (label >= 0) & (label < len(CLASSES))
    return np.bincount(label[valid].reshape(-1), minlength=len(CLASSES)).astype(np.int64)


def load_samples(root):
    samples = []
    for seq_name, frame_id, _, label_path in iter_all_samples(root):
        counts = load_label_counts(label_path)
        samples.append(
            {
                "id": f"{seq_name}/{frame_id:06d}",
                "seq_name": seq_name,
                "domain": cosec_domain(seq_name),
                "counts": counts,
                "present": counts > 0,
            }
        )
    if not samples:
        raise RuntimeError(f"No CoSEC samples found under {root}")
    return samples


def class_weights(samples):
    image_counts = sum(
        (sample["present"].astype(np.int64) for sample in samples),
        np.zeros(len(CLASSES), dtype=np.int64),
    )
    weights = np.zeros(len(CLASSES), dtype=np.float64)
    present = image_counts > 0
    weights[present] = 1.0 / np.sqrt(image_counts[present])
    return image_counts, weights


def pick_coverage_seed(samples, target_count):
    image_counts, weights = class_weights(samples)
    available = image_counts > 0
    covered = np.zeros(len(CLASSES), dtype=bool)
    selected = []
    remaining = list(samples)

    while remaining and len(selected) < target_count and not np.all(covered[available]):
        best_index = None
        best_key = None
        for index, sample in enumerate(remaining):
            new_classes = sample["present"] & ~covered
            rare_gain = float((new_classes * weights).sum())
            if rare_gain <= 0:
                continue
            pixel_gain = int(sample["counts"][new_classes].sum())
            key = (rare_gain, pixel_gain, -index)
            if best_key is None or key > best_key:
                best_key = key
                best_index = index
        if best_index is None:
            break
        sample = remaining.pop(best_index)
        selected.append(sample)
        covered |= sample["present"]
    return selected, remaining, image_counts


def fill_to_target(selected, remaining, image_counts, target_count):
    if len(selected) >= target_count:
        return selected[:target_count]
    selected_present = sum(
        (sample["present"].astype(np.int64) for sample in selected),
        np.zeros(len(CLASSES), dtype=np.int64),
    )
    valid = image_counts > 0
    weights = np.zeros(len(CLASSES), dtype=np.float64)
    weights[valid] = 1.0 / np.sqrt(image_counts[valid])
    target_present = image_counts * (target_count / float(len(selected) + len(remaining)))

    while remaining and len(selected) < target_count:
        best_index = None
        best_key = None
        for index, sample in enumerate(remaining):
            present = sample["present"]
            deficits = np.maximum(target_present[present] - selected_present[present], 0.0)
            rare_balance_gain = float((deficits * weights[present]).sum())
            pixel_gain = int(sample["counts"][present].sum())
            key = (rare_balance_gain, pixel_gain, -index)
            if best_key is None or key > best_key:
                best_key = key
                best_index = index
        sample = remaining.pop(best_index)
        selected.append(sample)
        selected_present += sample["present"].astype(np.int64)
    return selected


def choose_domain_val(samples, val_fraction, min_val_frames):
    target_count = max(int(round(len(samples) * val_fraction)), int(min_val_frames))
    target_count = min(target_count, len(samples))
    selected, remaining, image_counts = pick_coverage_seed(samples, target_count)
    return fill_to_target(selected, remaining, image_counts, target_count)


def summarize(samples, reference_samples):
    pixel_counts = sum(
        (sample["counts"] for sample in samples),
        np.zeros(len(CLASSES), dtype=np.int64),
    )
    image_counts = sum(
        (sample["present"].astype(np.int64) for sample in samples),
        np.zeros(len(CLASSES), dtype=np.int64),
    )
    reference_image_counts = sum(
        (sample["present"].astype(np.int64) for sample in reference_samples),
        np.zeros(len(CLASSES), dtype=np.int64),
    )
    reference_present = reference_image_counts > 0
    missing_available = [
        CLASSES[index]
        for index, present in enumerate(reference_present)
        if present and pixel_counts[index] == 0
    ]
    unavailable = [
        CLASSES[index]
        for index, present in enumerate(reference_present)
        if not present
    ]
    return {
        "frames": len(samples),
        "sequences": sorted({sample["seq_name"] for sample in samples}),
        "present_class_count": int((pixel_counts > 0).sum()),
        "available_class_count": int(reference_present.sum()),
        "missing_available_classes": missing_available,
        "unavailable_classes": unavailable,
        "class_image_counts": {CLASSES[index]: int(image_counts[index]) for index in range(len(CLASSES))},
        "class_pixel_counts": {CLASSES[index]: int(pixel_counts[index]) for index in range(len(CLASSES))},
    }


def write_split(path, sample_ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample_id in sorted(sample_ids):
            handle.write(f"{sample_id}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_COSEC_TRAIN_ROOT))
    parser.add_argument("--prefix", default="domaincover20")
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--min-val-frames", type=int, default=1)
    parser.add_argument("--split-dir", default=str(SPLIT_DIR))
    parser.add_argument("--write-splits", action="store_true")
    args = parser.parse_args()

    samples = load_samples(args.root)
    by_domain = {
        domain: [sample for sample in samples if sample["domain"] == domain]
        for domain in ("day", "night")
    }
    selected = {}
    for domain, domain_samples in by_domain.items():
        selected[domain] = choose_domain_val(domain_samples, args.val_fraction, args.min_val_frames)

    val_ids = {sample["id"] for domain_samples in selected.values() for sample in domain_samples}
    all_ids = {sample["id"] for sample in samples}
    train_ids = all_ids - val_ids
    split_dir = Path(args.split_dir)
    written = []
    if args.write_splits:
        train_path = split_dir / f"train_{args.prefix}.txt"
        val_path = split_dir / f"val_{args.prefix}.txt"
        write_split(train_path, train_ids)
        write_split(val_path, val_ids)
        written.extend([str(train_path), str(val_path)])

    train_samples = [sample for sample in samples if sample["id"] in train_ids]
    val_samples = [sample for sample in samples if sample["id"] in val_ids]
    payload = {
        "root": str(Path(args.root).resolve()),
        "mode": "frame-level domain coverage",
        "prefix": args.prefix,
        "val_fraction": args.val_fraction,
        "total_frames": len(samples),
        "train_frames": len(train_ids),
        "val_frames": len(val_ids),
        "dataset_names": {
            "train": f"cosec_{args.prefix}_train",
            "val": f"cosec_{args.prefix}_val",
            "day_val": f"cosec_{args.prefix}_day_val",
            "night_val": f"cosec_{args.prefix}_night_val",
        },
        "train": summarize(train_samples, samples),
        "val": summarize(val_samples, samples),
        "domains": {
            domain: {
                "total_frames": len(domain_samples),
                "val": summarize(selected[domain], domain_samples),
            }
            for domain, domain_samples in by_domain.items()
        },
        "written_files": written,
    }
    payload["ok"] = all(
        not payload["domains"][domain]["val"]["missing_available_classes"]
        for domain in ("day", "night")
    )
    if args.write_splits:
        summary_path = split_dir / f"{args.prefix}_summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        payload["summary_path"] = str(summary_path)

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
