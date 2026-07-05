#!/usr/bin/env python
"""Estimate mIoU gain if event-supported errors are corrected.

This is an oracle/sensitivity diagnostic. It does not train a model and does
not postprocess predictions. It measures how much room exists if a future
event module can repair a fraction of the current model's mistakes inside
event-derived regions.
"""

import argparse
import copy
import json
import os
import sys
import importlib.util
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

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
from mask2former import MaskFormerSemanticDatasetMapper  # noqa: E402
from train_mask2former_cosec import register_cosec  # noqa: E402

from diagnose_confidence_ensemble_routing import (  # noqa: E402
    build_model,
    infer_mapped,
    normalize_scores,
    resize_stat,
    setup_cfg,
    top_conf_margin,
    valid_label_mask,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument(
        "--event-config",
        default="configs/Mask2Former_SwinL_CoSEC_DayExp4A_EarlyEventEdgeAdapter.yaml",
        help="Config used only to build mapper/event_stats.",
    )
    parser.add_argument("--dataset", default="cosec_day_val_event")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--boundary-radius", type=int, default=3)
    parser.add_argument("--low-margin-percentile", type=float, default=40.0)
    parser.add_argument("--high-entropy-percentile", type=float, default=80.0)
    parser.add_argument(
        "--rates",
        default="0,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1.0",
        help="Comma-separated repair rates for event-overlapped wrong pixels.",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def parse_rates(text):
    rates = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if value < 0.0 or value > 1.0:
            raise ValueError(f"repair rate must be in [0, 1], got {value}")
        rates.append(value)
    return sorted(set(rates))


def load_label(record):
    label = cv2.imread(record["sem_seg_file_name"], cv2.IMREAD_UNCHANGED)
    if label is None:
        raise RuntimeError(f"Could not read label: {record['sem_seg_file_name']}")
    if label.ndim == 3:
        label = label[:, :, 0]
    return label.astype(np.int64)


def entropy_from_prob(prob):
    entropy = -(prob * prob.clamp_min(1e-8).log()).sum(dim=0)
    return (entropy / np.log(float(prob.shape[0]))).cpu().numpy()


def boundary_mask(pred, radius):
    radius = int(radius)
    if radius <= 0:
        return np.zeros_like(pred, dtype=bool)
    pred = np.asarray(pred)
    padded = np.pad(pred, radius, mode="edge")
    center = padded[radius : radius + pred.shape[0], radius : radius + pred.shape[1]]
    boundary = np.zeros(pred.shape, dtype=bool)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            shifted = padded[
                radius + dy : radius + dy + pred.shape[0],
                radius + dx : radius + dx + pred.shape[1],
            ]
            boundary |= shifted != center
    return boundary


def percentile_region(values, valid, percentile, high):
    selected = values[valid]
    if selected.size == 0:
        return np.zeros_like(valid, dtype=bool)
    threshold = np.percentile(selected, float(percentile))
    return valid & ((values >= threshold) if high else (values <= threshold))


def confusion_matrix(pred, label, mask, num_classes):
    keep = mask & (label >= 0) & (label < num_classes) & (pred >= 0) & (pred < num_classes)
    indices = num_classes * label[keep].astype(np.int64) + pred[keep].astype(np.int64)
    return np.bincount(indices, minlength=num_classes**2).reshape(num_classes, num_classes)


def metrics_from_matrix(matrix):
    hist = matrix.astype(np.float64)
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
        "IoU": OrderedDict(
            (CLASSES[idx], None if np.isnan(value) else float(100.0 * value))
            for idx, value in enumerate(iou)
        ),
    }


def matrix_after_repair(base_matrix, correctable_matrix, repair_rate):
    repair_rate = float(repair_rate)
    correctable = correctable_matrix.astype(np.float64) * repair_rate
    repaired = base_matrix.astype(np.float64) - correctable
    for cls_id in range(correctable.shape[0]):
        repaired[cls_id, cls_id] += correctable[cls_id].sum()
    return repaired


def greedy_predicted_class_curve(base_matrix, correctable_matrix, pred_class_pixels):
    base_metrics = metrics_from_matrix(base_matrix)
    selected = []
    selected_correctable = np.zeros_like(correctable_matrix)
    remaining = set(range(len(CLASSES)))
    rows = []
    while remaining:
        best = None
        for pred_cls in remaining:
            candidate_correctable = selected_correctable.copy()
            candidate_correctable[:, pred_cls] += correctable_matrix[:, pred_cls]
            metrics = metrics_from_matrix(matrix_after_repair(base_matrix, candidate_correctable, 1.0))
            item = {
                "class": CLASSES[pred_cls],
                "pred_class_id": int(pred_cls),
                "mIoU": metrics["mIoU"],
                "delta_mIoU": metrics["mIoU"] - base_metrics["mIoU"],
                "incremental_delta_mIoU": (
                    metrics["mIoU"] - rows[-1]["mIoU"] if rows else metrics["mIoU"] - base_metrics["mIoU"]
                ),
                "correctable_wrong_pixels_added": int(correctable_matrix[:, pred_cls].sum()),
                "pred_class_region_pixels": int(pred_class_pixels[pred_cls]),
                "pred_class_error_density": (
                    float(correctable_matrix[:, pred_cls].sum() / pred_class_pixels[pred_cls])
                    if pred_class_pixels[pred_cls]
                    else 0.0
                ),
            }
            if best is None or item["mIoU"] > best["mIoU"]:
                best = item
        if best is None:
            break
        selected.append(best["pred_class_id"])
        selected_correctable[:, best["pred_class_id"]] += correctable_matrix[:, best["pred_class_id"]]
        remaining.remove(best["pred_class_id"])
        best["selected_classes"] = [CLASSES[idx] for idx in selected]
        best["selected_class_count"] = len(selected)
        rows.append(best)
    return rows


def summarize_region(
    name,
    base_matrix,
    correctable_matrix,
    region_pixels,
    valid_pixels,
    rates,
    pred_class_pixels,
):
    base_metrics = metrics_from_matrix(base_matrix)
    wrong_pixels = int(correctable_matrix.sum())
    curve = []
    for rate in rates:
        repaired_matrix = matrix_after_repair(base_matrix, correctable_matrix, rate)
        metrics = metrics_from_matrix(repaired_matrix)
        curve.append(
            {
                "repair_rate": float(rate),
                "corrected_pixels": float(wrong_pixels * rate),
                "mIoU": metrics["mIoU"],
                "delta_mIoU": metrics["mIoU"] - base_metrics["mIoU"],
                "mAcc": metrics["mAcc"],
                "delta_mAcc": metrics["mAcc"] - base_metrics["mAcc"],
                "aAcc": metrics["aAcc"],
                "delta_aAcc": metrics["aAcc"] - base_metrics["aAcc"],
            }
        )

    per_class = []
    for cls_id, cls_name in enumerate(CLASSES):
        wrong = int(correctable_matrix[cls_id].sum())
        if wrong == 0:
            continue
        class_correctable = np.zeros_like(correctable_matrix)
        class_correctable[cls_id] = correctable_matrix[cls_id]
        class_metrics = metrics_from_matrix(matrix_after_repair(base_matrix, class_correctable, 1.0))
        per_class.append(
            {
                "class": cls_name,
                "wrong_pixels": wrong,
                "oracle_delta_mIoU_if_class_fixed": class_metrics["mIoU"] - base_metrics["mIoU"],
            }
        )

    per_pred_class = []
    for cls_id, cls_name in enumerate(CLASSES):
        wrong = int(correctable_matrix[:, cls_id].sum())
        pixels = int(pred_class_pixels[cls_id])
        if pixels == 0 and wrong == 0:
            continue
        class_correctable = np.zeros_like(correctable_matrix)
        class_correctable[:, cls_id] = correctable_matrix[:, cls_id]
        class_metrics = metrics_from_matrix(matrix_after_repair(base_matrix, class_correctable, 1.0))
        per_pred_class.append(
            {
                "pred_class": cls_name,
                "pred_class_region_pixels": pixels,
                "correctable_wrong_pixels": wrong,
                "error_density_in_region": float(wrong / pixels) if pixels else 0.0,
                "oracle_delta_mIoU_if_pred_class_fixed": class_metrics["mIoU"] - base_metrics["mIoU"],
            }
        )

    return {
        "name": name,
        "region_pixels": int(region_pixels),
        "valid_pixels": int(valid_pixels),
        "region_coverage": float(region_pixels / valid_pixels) if valid_pixels else 0.0,
        "correctable_wrong_pixels": wrong_pixels,
        "correctable_ratio_of_region": float(wrong_pixels / region_pixels) if region_pixels else 0.0,
        "correctable_ratio_of_all_valid": float(wrong_pixels / valid_pixels) if valid_pixels else 0.0,
        "oracle_full_repair_mIoU": curve[-1]["mIoU"] if curve else base_metrics["mIoU"],
        "oracle_full_repair_delta_mIoU": curve[-1]["delta_mIoU"] if curve else 0.0,
        "curve": curve,
        "top_classes_by_oracle_delta": sorted(
            per_class,
            key=lambda item: item["oracle_delta_mIoU_if_class_fixed"],
            reverse=True,
        )[:19],
        "predicted_classes_by_error_density": sorted(
            per_pred_class,
            key=lambda item: (item["error_density_in_region"], item["correctable_wrong_pixels"]),
            reverse=True,
        )[:19],
        "predicted_classes_by_oracle_delta": sorted(
            per_pred_class,
            key=lambda item: item["oracle_delta_mIoU_if_pred_class_fixed"],
            reverse=True,
        )[:19],
        "greedy_predicted_class_curve": greedy_predicted_class_curve(
            base_matrix,
            correctable_matrix,
            pred_class_pixels,
        ),
    }


def main():
    args = parse_args()
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    register_cosec()
    rates = parse_rates(args.rates)

    model_cfg = setup_cfg(args.config, args.weights, args.device)
    event_cfg = setup_cfg(args.event_config, args.weights, args.device)
    mapper = MaskFormerSemanticDatasetMapper(event_cfg, False)
    model = build_model(model_cfg)

    records = DatasetCatalog.get(args.dataset)
    if args.limit is not None:
        records = records[: args.limit]

    num_classes = len(CLASSES)
    base_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    region_correctable = OrderedDict(
        (name, np.zeros((num_classes, num_classes), dtype=np.int64))
        for name in [
            "raw_event",
            "support",
            "raw_event_uncertain_boundary",
            "support_uncertain_boundary",
        ]
    )
    region_pixels = OrderedDict((name, 0) for name in region_correctable)
    region_pred_class_pixels = OrderedDict(
        (name, np.zeros(num_classes, dtype=np.int64)) for name in region_correctable
    )
    valid_pixels = 0

    iterator = records if args.quiet else tqdm(records, desc=args.dataset)
    for record in iterator:
        label = load_label(record)
        mapped = mapper(copy.deepcopy(record))
        event_stats = mapped["event_stats"].float()
        prob = normalize_scores(infer_mapped(model, mapped))
        pred = prob.argmax(dim=0).numpy()
        valid = valid_label_mask(label)

        raw_event = valid & (resize_stat(event_stats, 0, label.shape) > 0)
        support = valid & (resize_stat(event_stats, 3, label.shape) > 0)
        _, margin = top_conf_margin(prob)
        entropy = entropy_from_prob(prob)
        low_margin = percentile_region(margin, valid, args.low_margin_percentile, high=False)
        high_entropy = percentile_region(entropy, valid, args.high_entropy_percentile, high=True)
        pred_boundary = boundary_mask(pred, args.boundary_radius)
        uncertain_boundary = pred_boundary | low_margin | high_entropy

        regions = {
            "raw_event": raw_event,
            "support": support,
            "raw_event_uncertain_boundary": raw_event & uncertain_boundary,
            "support_uncertain_boundary": support & uncertain_boundary,
        }

        base_matrix += confusion_matrix(pred, label, valid, num_classes)
        valid_pixels += int(valid.sum())
        wrong = valid & (pred != label)
        for name, mask in regions.items():
            region_mask = valid & mask
            correctable_mask = region_mask & wrong
            region_pixels[name] += int(region_mask.sum())
            region_correctable[name] += confusion_matrix(pred, label, correctable_mask, num_classes)
            if region_mask.any():
                region_pred_class_pixels[name] += np.bincount(
                    pred[region_mask].astype(np.int64),
                    minlength=num_classes,
                )[:num_classes]

    base_metrics = metrics_from_matrix(base_matrix)
    regions = OrderedDict(
        (
            name,
            summarize_region(
                name,
                base_matrix,
                correctable,
                region_pixels[name],
                valid_pixels,
                rates,
                region_pred_class_pixels[name],
            ),
        )
        for name, correctable in region_correctable.items()
    )
    output = {
        "args": vars(args),
        "sample_count": len(records),
        "baseline": base_metrics,
        "regions": regions,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
        f.write("\n")

    print(f"Wrote oracle curve: {out_path}")
    print(
        "baseline: "
        f"mIoU={base_metrics['mIoU']:.4f}, "
        f"mAcc={base_metrics['mAcc']:.4f}, "
        f"aAcc={base_metrics['aAcc']:.4f}"
    )
    for name, region in regions.items():
        print(
            f"{name}: coverage={100*region['region_coverage']:.2f}%, "
            f"wrong={region['correctable_wrong_pixels']}, "
            f"oracle_mIoU={region['oracle_full_repair_mIoU']:.4f} "
            f"({region['oracle_full_repair_delta_mIoU']:+.4f})"
        )
        for item in region["curve"]:
            if item["repair_rate"] in {0.01, 0.02, 0.05, 0.1, 1.0}:
                print(
                    f"  repair={100*item['repair_rate']:.1f}% "
                    f"mIoU={item['mIoU']:.4f} "
                    f"delta={item['delta_mIoU']:+.4f}"
                )


if __name__ == "__main__":
    main()
