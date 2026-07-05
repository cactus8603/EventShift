#!/usr/bin/env python
"""Compare two Detectron2 semantic prediction JSON files."""

import argparse
import json
import sys
import importlib.util
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_util

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-json", required=True)
    parser.add_argument("--new-json", required=True)
    parser.add_argument("--dataset", default="dsec19_val")
    parser.add_argument("--boundary-radius", type=int, default=3)
    parser.add_argument("--ignore-label", type=int, default=255)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def load_label(path):
    label = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if label is None:
        raise RuntimeError(f"Could not read label: {path}")
    if label.ndim == 3:
        label = label[:, :, 0]
    return label.astype(np.int64)


def valid_mask(label, ignore_label, num_classes):
    return (label != ignore_label) & (label >= 0) & (label < num_classes)


def semantic_boundary(label, radius, valid):
    if radius <= 0:
        return np.zeros(label.shape, dtype=bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    low = label.astype(np.float32, copy=True)
    high = label.astype(np.float32, copy=True)
    low[~valid] = -1000.0
    high[~valid] = 1000.0
    local_max = cv2.dilate(low, kernel)
    local_min = cv2.erode(high, kernel)
    return valid & (local_max != local_min)


def prediction_boundary(pred, radius, valid):
    return semantic_boundary(pred, radius, valid)


def load_prediction_json(path):
    grouped = {}
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    for row in rows:
        grouped.setdefault(row["file_name"], []).append(row)
    return grouped


def decode_prediction(rows, shape, fill_value=255):
    pred = np.full(shape, fill_value, dtype=np.int64)
    for row in rows:
        rle = dict(row["segmentation"])
        if isinstance(rle.get("counts"), str):
            rle["counts"] = rle["counts"].encode("ascii")
        mask = mask_util.decode(rle).astype(bool)
        if mask.shape != shape:
            raise RuntimeError(f"Mask shape mismatch: got {mask.shape}, expected {shape}")
        pred[mask] = int(row["category_id"])
    return pred


class Confusion:
    def __init__(self, num_classes):
        self.num_classes = int(num_classes)
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, pred, label, valid):
        keep = valid & (pred >= 0) & (pred < self.num_classes)
        idx = self.num_classes * label[keep].astype(np.int64) + pred[keep].astype(np.int64)
        self.matrix += np.bincount(idx, minlength=self.num_classes * self.num_classes).reshape(
            self.num_classes,
            self.num_classes,
        )

    def metrics(self):
        hist = self.matrix.astype(np.float64)
        tp = np.diag(hist)
        pos_gt = hist.sum(axis=1)
        pos_pred = hist.sum(axis=0)
        union = pos_gt + pos_pred - tp
        iou = np.divide(tp, union, out=np.full_like(tp, np.nan), where=union > 0)
        acc = np.divide(tp, pos_gt, out=np.full_like(tp, np.nan), where=pos_gt > 0)
        total = hist.sum()
        return {
            "mIoU": float(100.0 * np.nanmean(iou)),
            "mAcc": float(100.0 * np.nanmean(acc)),
            "aAcc": float(100.0 * tp.sum() / total) if total else 0.0,
            "IoU": OrderedDict(
                (CLASSES[idx], None if np.isnan(value) else float(100.0 * value))
                for idx, value in enumerate(iou)
            ),
        }


def empty_counts():
    return {
        "pixels": 0,
        "base_wrong": 0,
        "new_wrong": 0,
        "base_correct": 0,
        "changed": 0,
        "repaired": 0,
        "damaged": 0,
        "changed_repaired": 0,
        "changed_damaged": 0,
    }


def add_counts(counts, region, base_pred, new_pred, label):
    base_wrong = region & (base_pred != label)
    new_wrong = region & (new_pred != label)
    base_correct = region & (base_pred == label)
    changed = region & (base_pred != new_pred)
    repaired = base_wrong & (new_pred == label)
    damaged = base_correct & new_wrong

    counts["pixels"] += int(region.sum())
    counts["base_wrong"] += int(base_wrong.sum())
    counts["new_wrong"] += int(new_wrong.sum())
    counts["base_correct"] += int(base_correct.sum())
    counts["changed"] += int(changed.sum())
    counts["repaired"] += int(repaired.sum())
    counts["damaged"] += int(damaged.sum())
    counts["changed_repaired"] += int((changed & repaired).sum())
    counts["changed_damaged"] += int((changed & damaged).sum())


def div(num, den):
    return float(num / den) if den else 0.0


def finalize_counts(counts, valid_pixels):
    out = dict(counts)
    out.update(
        {
            "coverage": div(counts["pixels"], valid_pixels),
            "base_error_rate": div(counts["base_wrong"], counts["pixels"]),
            "new_error_rate": div(counts["new_wrong"], counts["pixels"]),
            "changed_rate": div(counts["changed"], counts["pixels"]),
            "repair_rate_of_base_wrong": div(counts["repaired"], counts["base_wrong"]),
            "damage_rate_of_base_correct": div(counts["damaged"], counts["base_correct"]),
            "net_repaired": int(counts["repaired"] - counts["damaged"]),
            "changed_repair_precision": div(counts["changed_repaired"], counts["changed"]),
            "changed_damage_rate": div(counts["changed_damaged"], counts["changed"]),
        }
    )
    return out


def boundary_f1(pred_boundary, gt_boundary, valid):
    pred_boundary = pred_boundary & valid
    gt_boundary = gt_boundary & valid
    tp = int((pred_boundary & gt_boundary).sum())
    pred_count = int(pred_boundary.sum())
    gt_count = int(gt_boundary.sum())
    precision = div(tp, pred_count)
    recall = div(tp, gt_count)
    return {
        "precision": precision,
        "recall": recall,
        "f1": div(2.0 * precision * recall, precision + recall),
        "pred_pixels": pred_count,
        "gt_pixels": gt_count,
        "tp_pixels": tp,
    }


def main():
    args = parse_args()
    register_cosec()

    base_rows = load_prediction_json(args.base_json)
    new_rows = load_prediction_json(args.new_json)
    records = DatasetCatalog.get(args.dataset)
    num_classes = len(CLASSES)
    base_confusion = Confusion(num_classes)
    new_confusion = Confusion(num_classes)
    counts = OrderedDict((name, empty_counts()) for name in ["all", "gt_boundary"])
    boundary_scores = {
        "base": {"tp": 0, "pred": 0, "gt": 0},
        "new": {"tp": 0, "pred": 0, "gt": 0},
    }
    valid_total = 0
    missing = []

    for record in records:
        file_name = record["file_name"]
        if file_name not in base_rows or file_name not in new_rows:
            missing.append(file_name)
            continue
        label = load_label(record["sem_seg_file_name"])
        valid = valid_mask(label, args.ignore_label, num_classes)
        valid_total += int(valid.sum())
        base_pred = decode_prediction(base_rows[file_name], label.shape, fill_value=args.ignore_label)
        new_pred = decode_prediction(new_rows[file_name], label.shape, fill_value=args.ignore_label)
        gt_boundary = semantic_boundary(label, args.boundary_radius, valid)
        base_boundary = prediction_boundary(base_pred, args.boundary_radius, valid)
        new_boundary = prediction_boundary(new_pred, args.boundary_radius, valid)

        base_confusion.update(base_pred, label, valid)
        new_confusion.update(new_pred, label, valid)
        add_counts(counts["all"], valid, base_pred, new_pred, label)
        add_counts(counts["gt_boundary"], gt_boundary, base_pred, new_pred, label)

        for name, pred_boundary in (("base", base_boundary), ("new", new_boundary)):
            score = boundary_f1(pred_boundary, gt_boundary, valid)
            boundary_scores[name]["tp"] += score["tp_pixels"]
            boundary_scores[name]["pred"] += score["pred_pixels"]
            boundary_scores[name]["gt"] += score["gt_pixels"]

    def finalize_boundary(accum):
        precision = div(accum["tp"], accum["pred"])
        recall = div(accum["tp"], accum["gt"])
        return {
            "precision": precision,
            "recall": recall,
            "f1": div(2.0 * precision * recall, precision + recall),
            "pred_pixels": accum["pred"],
            "gt_pixels": accum["gt"],
            "tp_pixels": accum["tp"],
        }

    output = {
        "args": vars(args),
        "records": len(records),
        "evaluated_records": len(records) - len(missing),
        "missing_records": missing[:20],
        "metrics": {
            "base": base_confusion.metrics(),
            "new": new_confusion.metrics(),
        },
        "regions": OrderedDict((name, finalize_counts(value, valid_total)) for name, value in counts.items()),
        "boundary": {
            "radius": args.boundary_radius,
            "base": finalize_boundary(boundary_scores["base"]),
            "new": finalize_boundary(boundary_scores["new"]),
        },
    }
    output["metrics"]["delta_mIoU"] = (
        output["metrics"]["new"]["mIoU"] - output["metrics"]["base"]["mIoU"]
    )
    output["boundary"]["delta_f1"] = output["boundary"]["new"]["f1"] - output["boundary"]["base"]["f1"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
        f.write("\n")

    print(f"Wrote {out_path}")
    print(
        "mIoU "
        f"base={output['metrics']['base']['mIoU']:.4f} "
        f"new={output['metrics']['new']['mIoU']:.4f} "
        f"delta={output['metrics']['delta_mIoU']:+.4f}"
    )
    for region, values in output["regions"].items():
        print(
            f"{region}: changed={100.0 * values['changed_rate']:.4f}% "
            f"repair={100.0 * values['repair_rate_of_base_wrong']:.2f}% "
            f"damage={100.0 * values['damage_rate_of_base_correct']:.2f}% "
            f"net={values['net_repaired']}"
        )
    print(
        "boundary F1 "
        f"base={100.0 * output['boundary']['base']['f1']:.2f}% "
        f"new={100.0 * output['boundary']['new']['f1']:.2f}% "
        f"delta={100.0 * output['boundary']['delta_f1']:+.2f}%"
    )


if __name__ == "__main__":
    main()
