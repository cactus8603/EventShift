"""Semantic segmentation metrics."""

from __future__ import annotations

import numpy as np


def confusion_matrix(pred: np.ndarray, target: np.ndarray, num_classes: int = 19, ignore_index: int = 255) -> np.ndarray:
    mask = (target != ignore_index) & (target >= 0) & (target < num_classes)
    pred = pred[mask].astype(np.int64, copy=False)
    target = target[mask].astype(np.int64, copy=False)
    valid = (pred >= 0) & (pred < num_classes)
    ids = target[valid] * num_classes + pred[valid]
    return np.bincount(ids, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def intersection_union(hist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    intersection = np.diag(hist)
    union = hist.sum(axis=1) + hist.sum(axis=0) - intersection
    return intersection, union


def mean_iou(hist: np.ndarray) -> dict[str, float]:
    inter, union = intersection_union(hist)
    iou = inter / np.maximum(union, 1)
    acc = inter / np.maximum(hist.sum(axis=1), 1)
    return {
        "mIoU": float(np.nanmean(iou)),
        "mAcc": float(np.nanmean(acc)),
        "aAcc": float(inter.sum() / max(hist.sum(), 1)),
    }

