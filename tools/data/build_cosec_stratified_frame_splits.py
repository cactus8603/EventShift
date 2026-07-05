#!/usr/bin/env python
"""Build frame-level stratified CoSEC CV splits with class coverage audits."""

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
    for seq_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if not seq_dir.name.startswith(("Day_", "Night_")):
            continue
        image_dir = seq_dir / "img_co_left"
        label_dir = seq_dir / "segment_co"
        if not image_dir.is_dir() or not label_dir.is_dir():
            continue
        for label_path in sorted(label_dir.glob("*.png")):
            image_path = image_dir / label_path.name
            if not image_path.is_file():
                continue
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


def stratify(samples, folds):
    total_present = sum((sample["present"].astype(np.int64) for sample in samples), np.zeros(len(CLASSES), dtype=np.int64))
    valid_classes = np.where(total_present > 0)[0]
    class_weight = np.zeros(len(CLASSES), dtype=np.float64)
    class_weight[valid_classes] = 1.0 / np.sqrt(total_present[valid_classes])
    domain_totals = {
        "day": sum(1 for sample in samples if sample["domain"] == "day"),
        "night": sum(1 for sample in samples if sample["domain"] == "night"),
    }

    ordered = sorted(
        samples,
        key=lambda sample: (-float((sample["present"] * class_weight).sum()), sample["id"]),
    )
    fold_samples = [[] for _ in range(folds)]
    fold_present = np.zeros((folds, len(CLASSES)), dtype=np.int64)
    fold_domain = [{"day": 0, "night": 0} for _ in range(folds)]
    fold_size = np.zeros(folds, dtype=np.int64)
    target_present = total_present / float(folds)
    target_size = len(samples) / float(folds)
    target_domain = {domain: count / float(folds) for domain, count in domain_totals.items()}

    for sample in ordered:
        present = sample["present"]
        scores = []
        for fold_index in range(folds):
            deficits = np.maximum(target_present[present] - fold_present[fold_index, present], 0.0)
            rare_gain = float((deficits * class_weight[present]).sum())
            size_penalty = (fold_size[fold_index] + 1) / target_size
            domain_penalty = (fold_domain[fold_index][sample["domain"]] + 1) / target_domain[sample["domain"]]
            score = -rare_gain + 0.08 * size_penalty + 0.04 * domain_penalty
            scores.append((score, int(fold_size[fold_index]), fold_index))
        fold_index = min(scores)[2]
        fold_samples[fold_index].append(sample)
        fold_present[fold_index] += present.astype(np.int64)
        fold_domain[fold_index][sample["domain"]] += 1
        fold_size[fold_index] += 1
    return fold_samples


def summarize_fold(samples, global_present):
    pixel_counts = sum((sample["counts"] for sample in samples), np.zeros(len(CLASSES), dtype=np.int64))
    image_counts = sum((sample["present"].astype(np.int64) for sample in samples), np.zeros(len(CLASSES), dtype=np.int64))
    domain_counts = {
        "day": sum(1 for sample in samples if sample["domain"] == "day"),
        "night": sum(1 for sample in samples if sample["domain"] == "night"),
    }
    missing = [CLASSES[index] for index, present in enumerate(global_present) if present and pixel_counts[index] == 0]
    return {
        "frames": len(samples),
        "domain_counts": domain_counts,
        "sequences": sorted({sample["seq_name"] for sample in samples}),
        "missing_present_classes": missing,
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
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--split-dir", default=str(SPLIT_DIR))
    parser.add_argument("--write-splits", action="store_true")
    args = parser.parse_args()

    samples = load_samples(args.root)
    fold_samples = stratify(samples, args.folds)
    all_ids = {sample["id"] for sample in samples}
    global_present = sum((sample["present"].astype(np.int64) for sample in samples), np.zeros(len(CLASSES), dtype=np.int64)) > 0
    global_missing = [CLASSES[index] for index, present in enumerate(global_present) if not present]
    prefix = args.prefix or f"stratframe{args.folds}"
    split_dir = Path(args.split_dir)
    written = []
    summaries = []

    for fold_index, val_samples in enumerate(fold_samples):
        val_ids = {sample["id"] for sample in val_samples}
        train_ids = all_ids - val_ids
        fold_prefix = f"{prefix}_fold{fold_index}"
        if args.write_splits:
            train_path = split_dir / f"train_{fold_prefix}.txt"
            val_path = split_dir / f"val_{fold_prefix}.txt"
            write_split(train_path, train_ids)
            write_split(val_path, val_ids)
            written.extend([str(train_path), str(val_path)])
        summaries.append(
            {
                "fold": fold_index,
                "dataset_names": {
                    "train": f"cosec_{fold_prefix}_train",
                    "val": f"cosec_{fold_prefix}_val",
                    "day_train": f"cosec_{fold_prefix}_day_train",
                    "day_val": f"cosec_{fold_prefix}_day_val",
                    "night_train": f"cosec_{fold_prefix}_night_train",
                    "night_val": f"cosec_{fold_prefix}_night_val",
                },
                "train_frames": len(train_ids),
                "val": summarize_fold(val_samples, global_present),
            }
        )

    payload = {
        "root": str(Path(args.root).resolve()),
        "mode": "frame-level stratified",
        "folds": args.folds,
        "prefix": prefix,
        "total_frames": len(samples),
        "global_missing_classes": global_missing,
        "folds_summary": summaries,
        "written_files": written,
        "ok": all(not summary["val"]["missing_present_classes"] for summary in summaries),
    }

    if args.write_splits:
        summary_path = split_dir / f"{prefix}_summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        payload["summary_path"] = str(summary_path)

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
