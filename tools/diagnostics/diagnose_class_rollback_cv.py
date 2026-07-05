#!/usr/bin/env python
"""Cross-validate class rollback rules between saved prediction JSON files."""

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
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--fixed-classes", default="building,wall,fence")
    parser.add_argument("--out", required=True)
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


def class_mask(pred, class_ids):
    mask = np.zeros(pred.shape, dtype=bool)
    for class_id in class_ids:
        mask |= pred == class_id
    return mask


def rollback_prediction(base_pred, new_pred, class_ids):
    if not class_ids:
        return new_pred
    rollback = (base_pred != new_pred) & class_mask(base_pred, class_ids)
    merged = new_pred.copy()
    merged[rollback] = base_pred[rollback]
    return merged


def evaluate(records, decoded, indices, mode, class_ids=None):
    meter = ConfusionMeter(num_classes=len(CLASSES))
    pixels = 0
    for idx in indices:
        item = decoded[idx]
        if mode == "base":
            pred = item["base"]
        elif mode == "new":
            pred = item["new"]
        elif mode == "rollback":
            pred = rollback_prediction(item["base"], item["new"], class_ids or set())
            pixels += int(((item["base"] != item["new"]) & class_mask(item["base"], class_ids or set())).sum())
        else:
            raise ValueError(mode)
        meter.update(pred, item["label"])
    return {**meter.metrics(), "rolled_back_pixels": int(pixels)}


def decode_records(records, base_index, new_index):
    decoded = []
    missing = []
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
        valid_label_mask(label, base_pred, new_pred)
        decoded.append({"record": record, "label": label, "base": base_pred, "new": new_pred})
    return decoded, missing


def select_classes(decoded, train_indices, min_delta):
    new_metrics = evaluate(None, decoded, train_indices, "new")
    rows = []
    selected = set()
    for class_id, class_name in enumerate(CLASSES):
        metrics = evaluate(None, decoded, train_indices, "rollback", {class_id})
        delta = metrics["mIoU"] - new_metrics["mIoU"]
        row = {
            "class": class_name,
            "class_id": class_id,
            "single_rollback_mIoU": metrics["mIoU"],
            "delta_vs_new": delta,
            "rolled_back_pixels": metrics["rolled_back_pixels"],
        }
        rows.append(row)
        if delta > min_delta:
            selected.add(class_id)
    rows.sort(key=lambda item: item["delta_vs_new"], reverse=True)
    return selected, rows, new_metrics


def main():
    args = parse_args()
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    register_cosec()
    base_index = load_prediction_index(args.base_predictions)
    new_index = load_prediction_index(args.new_predictions)
    records = DatasetCatalog.get(args.dataset)
    decoded, missing = decode_records(records, base_index, new_index)
    if missing:
        raise RuntimeError(f"Missing predictions: {len(missing)}; first={missing[:3]}")

    fixed_classes = split_classes(args.fixed_classes)
    all_indices = list(range(len(decoded)))
    overall = OrderedDict(
        [
            ("base", evaluate(records, decoded, all_indices, "base")),
            ("new", evaluate(records, decoded, all_indices, "new")),
            ("fixed_rollback", evaluate(records, decoded, all_indices, "rollback", fixed_classes)),
        ]
    )

    folds = []
    for fold in range(args.folds):
        test_indices = [idx for idx in all_indices if idx % args.folds == fold]
        train_indices = [idx for idx in all_indices if idx % args.folds != fold]
        selected, class_rows, train_new = select_classes(decoded, train_indices, args.min_delta)
        fold_result = OrderedDict(
            [
                ("fold", fold),
                ("train_count", len(train_indices)),
                ("test_count", len(test_indices)),
                ("selected_classes", [CLASSES[idx] for idx in sorted(selected)]),
                ("train_new_mIoU", train_new["mIoU"]),
                ("train_class_candidates", class_rows),
                ("test_base", evaluate(records, decoded, test_indices, "base")),
                ("test_new", evaluate(records, decoded, test_indices, "new")),
                ("test_selected_rollback", evaluate(records, decoded, test_indices, "rollback", selected)),
                ("test_fixed_rollback", evaluate(records, decoded, test_indices, "rollback", fixed_classes)),
            ]
        )
        folds.append(fold_result)

    def average_metric(key, metric):
        return float(np.mean([fold[key][metric] for fold in folds]))

    summary = {
        "avg_test_base_mIoU": average_metric("test_base", "mIoU"),
        "avg_test_new_mIoU": average_metric("test_new", "mIoU"),
        "avg_test_selected_rollback_mIoU": average_metric("test_selected_rollback", "mIoU"),
        "avg_test_fixed_rollback_mIoU": average_metric("test_fixed_rollback", "mIoU"),
        "selected_minus_new": average_metric("test_selected_rollback", "mIoU")
        - average_metric("test_new", "mIoU"),
        "fixed_minus_new": average_metric("test_fixed_rollback", "mIoU")
        - average_metric("test_new", "mIoU"),
    }

    output = {
        "args": vars(args),
        "sample_count": len(decoded),
        "classes": list(CLASSES),
        "fixed_classes": [CLASSES[idx] for idx in sorted(fixed_classes)],
        "overall": overall,
        "folds": folds,
        "summary": summary,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote class rollback CV: {out_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("overall:")
    for name, metrics in overall.items():
        print(
            f"  {name}: mIoU={metrics['mIoU']:.4f}, mAcc={metrics['mAcc']:.4f}, "
            f"aAcc={metrics['aAcc']:.4f}, rolled={metrics['rolled_back_pixels']}"
        )
    for fold in folds:
        print(
            f"fold {fold['fold']}: selected={fold['selected_classes']} "
            f"new={fold['test_new']['mIoU']:.4f} "
            f"selected={fold['test_selected_rollback']['mIoU']:.4f} "
            f"fixed={fold['test_fixed_rollback']['mIoU']:.4f}"
        )


if __name__ == "__main__":
    main()
