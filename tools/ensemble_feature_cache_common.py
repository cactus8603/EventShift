import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

from cosec_finetune_splits import CLASSES


IGNORE_LABEL = 255
CLASS_COUNT = len(CLASSES)


def safe_name(text):
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text))


def load_label(path):
    label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise RuntimeError(f"Could not read label: {path}")
    if label.ndim == 3:
        label = label[:, :, 0]
    return label.astype(np.int64, copy=False)


def resize_if_needed(array, shape, interpolation):
    if array.shape[:2] == tuple(shape):
        return array
    return cv2.resize(array, (int(shape[1]), int(shape[0])), interpolation=interpolation)


def valid_label_mask(label):
    return (label != IGNORE_LABEL) & (label >= 0) & (label < CLASS_COUNT)


def split_name_from_path(path):
    path = Path(path)
    parts = path.parts
    for part in parts:
        if part.startswith("Day_"):
            return "day"
        if part.startswith("Night_"):
            return "night"
    if "night" in parts:
        return "acdc_night"
    for condition in ("fog", "rain", "snow"):
        if condition in parts:
            return f"acdc_{condition}"
    return "unknown"


def image_id_from_record(record):
    if record.get("image_id") is not None:
        return str(record["image_id"])
    path = Path(record.get("img_path") or record.get("file_name"))
    return f"{path.parent.name}_{path.stem}"


def semseg_stats_from_logits(logits):
    logits = logits.astype(np.float32, copy=False)
    logits = logits - logits.max(axis=0, keepdims=True)
    exp = np.exp(logits)
    prob = exp / np.maximum(exp.sum(axis=0, keepdims=True), 1e-8)
    return semseg_stats_from_prob(prob)


def semseg_stats_from_prob(prob):
    prob = np.asarray(prob, dtype=np.float32)
    prob = np.maximum(prob, 1e-8)
    prob = prob / np.maximum(prob.sum(axis=0, keepdims=True), 1e-8)
    pred = prob.argmax(axis=0).astype(np.uint8, copy=False)
    top2 = np.partition(prob, kth=-2, axis=0)[-2:]
    top2.sort(axis=0)
    conf = top2[-1].astype(np.float16, copy=False)
    margin = (top2[-1] - top2[-2]).astype(np.float16, copy=False)
    entropy = (-(prob * np.log(np.clip(prob, 1e-8, 1.0))).sum(axis=0) / math.log(CLASS_COUNT))
    return {
        "pred": pred,
        "conf": conf,
        "margin": margin,
        "entropy": entropy.astype(np.float16, copy=False),
    }


class SegmentationStats:
    def __init__(self, num_classes=CLASS_COUNT):
        self.num_classes = int(num_classes)
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        self.pred_pixels = np.zeros(self.num_classes, dtype=np.int64)
        self.gt_pixels = np.zeros(self.num_classes, dtype=np.int64)
        self.conf_sum_pred = np.zeros(self.num_classes, dtype=np.float64)
        self.margin_sum_pred = np.zeros(self.num_classes, dtype=np.float64)
        self.entropy_sum_pred = np.zeros(self.num_classes, dtype=np.float64)
        self.correct_pixels_pred = np.zeros(self.num_classes, dtype=np.int64)

    def update(self, pred, label, conf=None, margin=None, entropy=None):
        pred = np.asarray(pred, dtype=np.int64)
        label = np.asarray(label, dtype=np.int64)
        valid = valid_label_mask(label) & (pred >= 0) & (pred < self.num_classes)
        if not np.any(valid):
            return
        indices = self.num_classes * label[valid] + pred[valid]
        self.matrix += np.bincount(indices, minlength=self.num_classes**2).reshape(
            self.num_classes,
            self.num_classes,
        )
        valid_pred = pred[valid]
        self.pred_pixels += np.bincount(valid_pred, minlength=self.num_classes)
        self.gt_pixels += np.bincount(label[valid], minlength=self.num_classes)
        correct = valid & (pred == label)
        if np.any(correct):
            self.correct_pixels_pred += np.bincount(pred[correct], minlength=self.num_classes)
        if conf is not None:
            self.conf_sum_pred += np.bincount(
                valid_pred,
                weights=np.asarray(conf, dtype=np.float64)[valid],
                minlength=self.num_classes,
            )
        if margin is not None:
            self.margin_sum_pred += np.bincount(
                valid_pred,
                weights=np.asarray(margin, dtype=np.float64)[valid],
                minlength=self.num_classes,
            )
        if entropy is not None:
            self.entropy_sum_pred += np.bincount(
                valid_pred,
                weights=np.asarray(entropy, dtype=np.float64)[valid],
                minlength=self.num_classes,
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
            "aAcc": float(100.0 * tp.sum() / total) if total > 0 else float("nan"),
            "iou": iou,
            "acc": acc,
            "tp": tp.astype(np.int64),
            "union": union.astype(np.int64),
        }

    def class_rows(self):
        metrics = self.metrics()
        rows = []
        for class_id, class_name in enumerate(CLASSES):
            pred_pixels = int(self.pred_pixels[class_id])
            mean_conf = float(self.conf_sum_pred[class_id] / pred_pixels) if pred_pixels else float("nan")
            mean_margin = float(self.margin_sum_pred[class_id] / pred_pixels) if pred_pixels else float("nan")
            mean_entropy = float(self.entropy_sum_pred[class_id] / pred_pixels) if pred_pixels else float("nan")
            rows.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "iou": float(100.0 * metrics["iou"][class_id]) if not np.isnan(metrics["iou"][class_id]) else "",
                    "acc": float(100.0 * metrics["acc"][class_id]) if not np.isnan(metrics["acc"][class_id]) else "",
                    "tp": int(metrics["tp"][class_id]),
                    "union": int(metrics["union"][class_id]),
                    "gt_pixels": int(self.gt_pixels[class_id]),
                    "pred_pixels": pred_pixels,
                    "correct_pred_pixels": int(self.correct_pixels_pred[class_id]),
                    "mean_conf_pred": mean_conf,
                    "mean_margin_pred": mean_margin,
                    "mean_entropy_pred": mean_entropy,
                }
            )
        return rows


def per_image_summary(pred, label, conf, margin, entropy):
    valid = valid_label_mask(label)
    meter = SegmentationStats()
    meter.update(pred, label, conf=conf, margin=margin, entropy=entropy)
    metrics = meter.metrics()
    correct = valid & (pred.astype(np.int64) == label)
    wrong = valid & ~correct
    return {
        "valid_pixels": int(valid.sum()),
        "mIoU": metrics["mIoU"],
        "mAcc": metrics["mAcc"],
        "aAcc": metrics["aAcc"],
        "mean_conf": float(np.asarray(conf, dtype=np.float32)[valid].mean()) if np.any(valid) else float("nan"),
        "mean_margin": float(np.asarray(margin, dtype=np.float32)[valid].mean()) if np.any(valid) else float("nan"),
        "mean_entropy": float(np.asarray(entropy, dtype=np.float32)[valid].mean()) if np.any(valid) else float("nan"),
        "correct_mean_conf": float(np.asarray(conf, dtype=np.float32)[correct].mean()) if np.any(correct) else float("nan"),
        "wrong_mean_conf": float(np.asarray(conf, dtype=np.float32)[wrong].mean()) if np.any(wrong) else float("nan"),
    }


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def save_feature_maps(path, pred, conf, margin, entropy):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        pred=np.asarray(pred, dtype=np.uint8),
        conf=np.asarray(conf, dtype=np.float16),
        margin=np.asarray(margin, dtype=np.float16),
        entropy=np.asarray(entropy, dtype=np.float16),
    )
