#!/usr/bin/env python
"""Evaluate class-level merge rules between two saved prediction JSON files."""

import argparse
import json
import os
import sys
import importlib.util
from collections import OrderedDict
from pathlib import Path

import numpy as np

def _eventshift_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "configs").is_dir() and (parent / "third_party").is_dir():
            return parent
    return Path(__file__).resolve().parents[1]


ROOT = _eventshift_root()
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "third_party" / "Mask2Former"))
if importlib.util.find_spec("detectron2") is None:
    sys.path.insert(0, str(ROOT / "third_party" / "detectron2"))

from cosec_finetune_splits import CLASSES  # noqa: E402
from detectron2.data import DatasetCatalog  # noqa: E402
from train_mask2former_cosec import register_cosec  # noqa: E402

from diagnose_pair_transition_from_predictions import (  # noqa: E402
    decode_prediction,
    load_label,
    load_prediction_index,
    prediction_key,
    valid_label_mask,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-predictions", required=True)
    parser.add_argument("--new-predictions", required=True)
    parser.add_argument("--dataset", default="cosec_night_val")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--improved-classes",
        default="person,bicycle,traffic sign,motorcycle,pole,sky,sidewalk,vegetation,road",
    )
    parser.add_argument("--degraded-classes", default="building,wall,fence")
    return parser.parse_args()


def split_classes(text):
    return {CLASSES.index(item.strip()) for item in text.split(",") if item.strip()}


class ConfusionMeter:
    def __init__(self, num_classes=19, ignore_label=255):
        self.num_classes = int(num_classes)
        self.ignore_label = int(ignore_label)
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, pred, label):
        pred = np.asarray(pred, dtype=np.int64)
        label = np.asarray(label, dtype=np.int64)
        keep = (label != self.ignore_label) & (label >= 0) & (label < self.num_classes)
        keep &= (pred >= 0) & (pred < self.num_classes)
        indices = self.num_classes * label[keep] + pred[keep]
        self.matrix += np.bincount(indices, minlength=self.num_classes**2).reshape(
            self.num_classes,
            self.num_classes,
        )

    def metrics(self):
        hist = self.matrix.astype(np.float64)
        true_positive = np.diag(hist)
        pos_gt = hist.sum(axis=1)
        pos_pred = hist.sum(axis=0)
        union = pos_gt + pos_pred - true_positive
        iou = np.divide(true_positive, union, out=np.full_like(true_positive, np.nan), where=union > 0)
        acc = np.divide(true_positive, pos_gt, out=np.full_like(true_positive, np.nan), where=pos_gt > 0)
        total = hist.sum()
        return {
            "mIoU": float(100.0 * np.nanmean(iou)),
            "mAcc": float(100.0 * np.nanmean(acc)),
            "aAcc": float(100.0 * true_positive.sum() / total) if total > 0 else float("nan"),
            "class_iou": {
                CLASSES[idx]: (None if np.isnan(value) else float(100.0 * value))
                for idx, value in enumerate(iou)
            },
        }


def mask_in_classes(pred, classes):
    out = np.zeros(pred.shape, dtype=bool)
    for class_id in classes:
        out |= pred == class_id
    return out


def main():
    args = parse_args()
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    register_cosec()

    improved = split_classes(args.improved_classes)
    degraded = split_classes(args.degraded_classes)
    base_index = load_prediction_index(args.base_predictions)
    new_index = load_prediction_index(args.new_predictions)

    meters = OrderedDict((name, ConfusionMeter(num_classes=len(CLASSES))) for name in [
        "base",
        "new",
        "base_accept_new_pred_improved",
        "base_accept_base_pred_improved",
        "new_revert_base_pred_degraded",
        "new_revert_new_pred_degraded",
        "new_revert_either_pred_degraded",
        "new_revert_changed_base_pred_degraded",
        "new_revert_changed_new_pred_degraded",
    ])
    accepted_pixels = {name: 0 for name in meters}

    missing = []
    records = DatasetCatalog.get(args.dataset)
    for record in records:
        key = prediction_key(record["file_name"])
        base_rows = base_index.get(record["file_name"]) or base_index.get(key)
        new_rows = new_index.get(record["file_name"]) or new_index.get(key)
        if not base_rows or not new_rows:
            missing.append(record["file_name"])
            continue

        label = load_label(record)
        base_pred = decode_prediction(base_rows, label.shape)
        new_pred = decode_prediction(new_rows, label.shape)
        valid = valid_label_mask(label, base_pred, new_pred)
        changed = valid & (base_pred != new_pred)

        variants = OrderedDict()
        variants["base"] = base_pred
        variants["new"] = new_pred

        take = changed & mask_in_classes(new_pred, improved)
        merged = base_pred.copy()
        merged[take] = new_pred[take]
        variants["base_accept_new_pred_improved"] = merged
        accepted_pixels["base_accept_new_pred_improved"] += int(take.sum())

        take = changed & mask_in_classes(base_pred, improved)
        merged = base_pred.copy()
        merged[take] = new_pred[take]
        variants["base_accept_base_pred_improved"] = merged
        accepted_pixels["base_accept_base_pred_improved"] += int(take.sum())

        for name, revert_mask in [
            ("new_revert_base_pred_degraded", mask_in_classes(base_pred, degraded)),
            ("new_revert_new_pred_degraded", mask_in_classes(new_pred, degraded)),
            (
                "new_revert_either_pred_degraded",
                mask_in_classes(base_pred, degraded) | mask_in_classes(new_pred, degraded),
            ),
            ("new_revert_changed_base_pred_degraded", changed & mask_in_classes(base_pred, degraded)),
            ("new_revert_changed_new_pred_degraded", changed & mask_in_classes(new_pred, degraded)),
        ]:
            merged = new_pred.copy()
            merged[revert_mask] = base_pred[revert_mask]
            variants[name] = merged
            accepted_pixels[name] += int(revert_mask.sum())

        for name, pred in variants.items():
            meters[name].update(pred, label)

    results = OrderedDict()
    for name, meter in meters.items():
        results[name] = meter.metrics()
        results[name]["accepted_or_reverted_pixels"] = accepted_pixels[name]

    output = {
        "args": vars(args),
        "classes": list(CLASSES),
        "improved_classes": [CLASSES[idx] for idx in sorted(improved)],
        "degraded_classes": [CLASSES[idx] for idx in sorted(degraded)],
        "sample_count": len(records),
        "missing_prediction_count": len(missing),
        "missing_predictions": missing[:20],
        "results": results,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote class-rule evaluation: {out_path}")
    print(f"records={len(records)}, missing={len(missing)}")
    for name, values in sorted(results.items(), key=lambda item: item[1]["mIoU"], reverse=True):
        print(
            f"{name}: mIoU={values['mIoU']:.4f}, mAcc={values['mAcc']:.4f}, "
            f"aAcc={values['aAcc']:.4f}, pixels={values['accepted_or_reverted_pixels']}"
        )


if __name__ == "__main__":
    main()
