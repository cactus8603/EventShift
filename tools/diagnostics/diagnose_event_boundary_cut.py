#!/usr/bin/env python
"""Diagnose whether event correction should be restricted to boundaries."""

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
from detectron2.checkpoint import DetectionCheckpointer  # noqa: E402
from detectron2.config import get_cfg  # noqa: E402
from detectron2.data import DatasetCatalog  # noqa: E402
from detectron2.projects.deeplab import add_deeplab_config  # noqa: E402
from mask2former import MaskFormerSemanticDatasetMapper, add_maskformer2_config  # noqa: E402
from train_mask2former_cosec import CoSECTrainer, register_cosec  # noqa: E402


CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASSES)}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-config", default="configs/Mask2Former_SwinL_CoSEC_DayNight_Finetune.yaml")
    parser.add_argument("--rgb-weights", required=True)
    parser.add_argument("--event-config", default="configs/Mask2Former_SwinL_CoSEC_EventReliability_Exp2.yaml")
    parser.add_argument("--event-weights", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--event-classes", default="motorcycle,sky,pole,bicycle,sidewalk")
    parser.add_argument("--boundary-radii", default="1,3,5,9,15")
    parser.add_argument("--uncertain-percentiles", default="10,20,30")
    parser.add_argument("--out", required=True)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def parse_int_list(text):
    return [int(part) for part in text.split(",") if part.strip()]


def parse_float_list(text):
    return [float(part) for part in text.split(",") if part.strip()]


def parse_class_ids(text):
    names = [part.strip() for part in text.split(",") if part.strip()]
    unknown = [name for name in names if name not in CLASS_TO_ID]
    if unknown:
        raise ValueError(f"Unknown class names: {unknown}; valid classes: {list(CLASSES)}")
    return np.asarray([CLASS_TO_ID[name] for name in names], dtype=np.int64), names


def setup_cfg(config_file, weights, device):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.DEVICE = device
    cfg.freeze()
    return cfg


def build_model(cfg):
    model = CoSECTrainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS,
        resume=False,
    )
    model.eval()
    return model


def infer(model, mapper, record, return_mapped=False):
    with torch.no_grad():
        mapped = mapper(copy.deepcopy(record))
        outputs = model([mapped])[0]["sem_seg"].detach().cpu()
    if return_mapped:
        return outputs, mapped
    return outputs


def load_label(record):
    label = cv2.imread(record["sem_seg_file_name"], cv2.IMREAD_UNCHANGED)
    if label is None:
        raise RuntimeError(f"Could not read label: {record['sem_seg_file_name']}")
    if label.ndim == 3:
        label = label[:, :, 0]
    return label.astype(np.int64)


def resize_scores(scores, shape):
    if scores.shape[-2:] == shape:
        return scores
    return F.interpolate(
        scores.unsqueeze(0),
        size=shape,
        mode="bilinear",
        align_corners=False,
    )[0]


def resize_stat(mapped, channel, shape):
    stats = mapped["event_stats"][channel : channel + 1].float().unsqueeze(0)
    stats = F.interpolate(stats, size=shape, mode="nearest")[0, 0]
    return stats.cpu().numpy()


def valid_label_mask(label, ignore_label=255):
    return (label != ignore_label) & (label >= 0) & (label < len(CLASSES))


def semantic_boundary_band(label, radius, valid=None):
    if radius <= 0:
        return np.zeros(label.shape, dtype=bool)
    if valid is None:
        valid = np.ones(label.shape, dtype=bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    low = label.astype(np.float32, copy=True)
    high = label.astype(np.float32, copy=True)
    low[~valid] = -1000.0
    high[~valid] = 1000.0
    local_max = cv2.dilate(low, kernel)
    local_min = cv2.erode(high, kernel)
    return valid & (local_max != local_min)


def rgb_margin(scores):
    top2 = torch.topk(scores.float(), k=2, dim=0).values
    return (top2[0] - top2[1]).cpu().numpy()


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
            "IoU": {
                CLASSES[idx]: None if np.isnan(value) else float(100.0 * value)
                for idx, value in enumerate(iou)
            },
        }


def empty_mask_counts():
    return {
        "valid_pixels": 0,
        "mask_pixels": 0,
        "rgb_wrong": 0,
        "rgb_wrong_mask": 0,
        "event_repaired_mask": 0,
        "event_damaged_mask": 0,
        "selective_repaired_mask": 0,
        "selective_damaged_mask": 0,
        "selective_plus_road_repaired_mask": 0,
        "selective_plus_road_damaged_mask": 0,
    }


def update_mask_counts(
    counts,
    mask,
    rgb_pred,
    event_pred,
    selective_pred,
    selective_plus_road_pred,
    label,
    valid,
):
    mask = valid & mask
    rgb_wrong = valid & (rgb_pred != label)
    rgb_correct = valid & (rgb_pred == label)
    counts["valid_pixels"] += int(valid.sum())
    counts["mask_pixels"] += int(mask.sum())
    counts["rgb_wrong"] += int(rgb_wrong.sum())
    counts["rgb_wrong_mask"] += int((rgb_wrong & mask).sum())

    for prefix, pred in [
        ("event", event_pred),
        ("selective", selective_pred),
        ("selective_plus_road", selective_plus_road_pred),
    ]:
        method_correct = valid & (pred == label)
        method_wrong = valid & (pred != label)
        counts[f"{prefix}_repaired_mask"] += int((rgb_wrong & method_correct & mask).sum())
        counts[f"{prefix}_damaged_mask"] += int((rgb_correct & method_wrong & mask).sum())


def div(num, den):
    return float(num / den) if den else 0.0


def finalize_mask_counts(counts):
    out = {
        **counts,
        "mask_coverage": div(counts["mask_pixels"], counts["valid_pixels"]),
        "rgb_wrong_recall": div(counts["rgb_wrong_mask"], counts["rgb_wrong"]),
        "rgb_error_density_in_mask": div(counts["rgb_wrong_mask"], counts["mask_pixels"]),
    }
    rgb_correct_mask = counts["mask_pixels"] - counts["rgb_wrong_mask"]
    for prefix in ["event", "selective", "selective_plus_road"]:
        repaired = counts[f"{prefix}_repaired_mask"]
        damaged = counts[f"{prefix}_damaged_mask"]
        out[f"{prefix}_repair_rate_in_mask"] = div(repaired, counts["rgb_wrong_mask"])
        out[f"{prefix}_damage_rate_in_mask"] = div(damaged, rgb_correct_mask)
        out[f"{prefix}_net_in_mask"] = repaired - damaged
    return out


def top_metrics(metrics, topk=20):
    rows = [(values["mIoU"], name, values) for name, values in metrics.items()]
    rows.sort(reverse=True, key=lambda item: item[0])
    return [{"method": name, **values} for _, name, values in rows[:topk]]


def add_hybrid_meters(meters, mask_name):
    for prefix in ["event_on", "selective_on", "selective_plus_road_on"]:
        meters[f"{prefix}:{mask_name}"] = ConfusionMeter()


def main():
    args = parse_args()
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    register_cosec()

    target_ids, target_names = parse_class_ids(args.event_classes)
    plus_road_ids = np.unique(np.concatenate([target_ids, np.asarray([CLASS_TO_ID["road"]])]))
    radii = parse_int_list(args.boundary_radii)
    uncertain_percentiles = parse_float_list(args.uncertain_percentiles)

    rgb_cfg = setup_cfg(args.rgb_config, args.rgb_weights, args.device)
    event_cfg = setup_cfg(args.event_config, args.event_weights, args.device)
    rgb_mapper = MaskFormerSemanticDatasetMapper(rgb_cfg, False)
    event_mapper = MaskFormerSemanticDatasetMapper(event_cfg, False)
    rgb_model = build_model(rgb_cfg)
    event_model = build_model(event_cfg)

    base_meters = OrderedDict(
        (name, ConfusionMeter())
        for name in ["rgb_only", "event_only", "selective_default", "selective_default_plus_road"]
    )
    hybrid_meters = OrderedDict()
    mask_counts = OrderedDict()

    records = DatasetCatalog.get(args.dataset)
    if args.limit is not None:
        records = records[: args.limit]

    iterator = records if args.quiet else tqdm(records, desc=args.dataset)
    for record in iterator:
        label = load_label(record)
        valid = valid_label_mask(label)
        rgb_scores = resize_scores(infer(rgb_model, rgb_mapper, record), label.shape)
        event_scores, mapped_event = infer(event_model, event_mapper, record, return_mapped=True)
        event_scores = resize_scores(event_scores, label.shape)
        density = resize_stat(mapped_event, 0, label.shape)
        support = resize_stat(mapped_event, 3, label.shape) > 0
        raw_event = density > 0

        rgb_pred = rgb_scores.argmax(dim=0).numpy()
        event_pred = event_scores.argmax(dim=0).numpy()
        default_mask = np.isin(rgb_pred, target_ids) | np.isin(event_pred, target_ids)
        plus_road_mask = np.isin(rgb_pred, plus_road_ids) | np.isin(event_pred, plus_road_ids)
        selective_pred = np.where(default_mask, event_pred, rgb_pred)
        selective_plus_road_pred = np.where(plus_road_mask, event_pred, rgb_pred)
        predictions = {
            "rgb_only": rgb_pred,
            "event_only": event_pred,
            "selective_default": selective_pred,
            "selective_default_plus_road": selective_plus_road_pred,
        }
        for name, pred in predictions.items():
            base_meters[name].update(pred, label)

        margin = rgb_margin(rgb_scores)
        candidate_masks = OrderedDict()
        candidate_masks["raw_event"] = raw_event
        candidate_masks["support"] = support

        pred_boundaries = {}
        gt_boundaries = {}
        for radius in radii:
            gt_boundary = semantic_boundary_band(label, radius, valid)
            pred_boundary = semantic_boundary_band(rgb_pred, radius, valid)
            gt_boundaries[radius] = gt_boundary
            pred_boundaries[radius] = pred_boundary
            candidate_masks[f"gt_boundary_r{radius}"] = gt_boundary
            candidate_masks[f"gt_boundary_r{radius}_support"] = gt_boundary & support
            candidate_masks[f"pred_boundary_r{radius}"] = pred_boundary
            candidate_masks[f"pred_boundary_r{radius}_support"] = pred_boundary & support

        valid_margin = margin[valid]
        for percentile in uncertain_percentiles:
            threshold = np.percentile(valid_margin, percentile) if valid_margin.size else 0.0
            suffix = f"q{percentile:g}"
            uncertain = valid & (margin <= threshold)
            candidate_masks[f"uncertain_{suffix}"] = uncertain
            candidate_masks[f"uncertain_{suffix}_support"] = uncertain & support
            for radius in radii:
                pred_boundary = pred_boundaries[radius]
                candidate_masks[f"pred_boundary_r{radius}_uncertain_{suffix}_support"] = (
                    pred_boundary & uncertain & support
                )
                candidate_masks[f"pred_boundary_r{radius}_or_uncertain_{suffix}_support"] = (
                    (pred_boundary | uncertain) & support
                )

        for mask_name, mask in candidate_masks.items():
            if mask_name not in mask_counts:
                mask_counts[mask_name] = empty_mask_counts()
                add_hybrid_meters(hybrid_meters, mask_name)
            update_mask_counts(
                mask_counts[mask_name],
                mask,
                rgb_pred,
                event_pred,
                selective_pred,
                selective_plus_road_pred,
                label,
                valid,
            )
            mask = valid & mask
            hybrid_meters[f"event_on:{mask_name}"].update(np.where(mask, event_pred, rgb_pred), label)
            hybrid_meters[f"selective_on:{mask_name}"].update(np.where(mask, selective_pred, rgb_pred), label)
            hybrid_meters[f"selective_plus_road_on:{mask_name}"].update(
                np.where(mask, selective_plus_road_pred, rgb_pred),
                label,
            )

    base_metrics = OrderedDict((name, meter.metrics()) for name, meter in base_meters.items())
    hybrid_metrics = OrderedDict((name, meter.metrics()) for name, meter in hybrid_meters.items())
    masks = OrderedDict((name, finalize_mask_counts(counts)) for name, counts in mask_counts.items())
    output = {
        "args": vars(args),
        "event_classes": target_names,
        "sample_count": len(records),
        "base_metrics": base_metrics,
        "top_hybrid_metrics": top_metrics(hybrid_metrics),
        "hybrid_metrics": hybrid_metrics,
        "masks": masks,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote diagnostics: {out_path}")
    print("Base metrics:")
    for name, values in base_metrics.items():
        print(f"  {name}: mIoU={values['mIoU']:.4f}, aAcc={values['aAcc']:.4f}")
    print("Top boundary/event cut metrics:")
    for row in output["top_hybrid_metrics"][:10]:
        print(f"  {row['method']}: mIoU={row['mIoU']:.4f}, aAcc={row['aAcc']:.4f}")
    print("Representative masks:")
    for name in [
        "support",
        "gt_boundary_r3_support",
        "gt_boundary_r5_support",
        "pred_boundary_r3_support",
        "pred_boundary_r5_support",
        "pred_boundary_r5_or_uncertain_q20_support",
    ]:
        if name in masks:
            values = masks[name]
            print(
                f"  {name}: coverage={100*values['mask_coverage']:.2f}%, "
                f"rgb_wrong_recall={100*values['rgb_wrong_recall']:.2f}%, "
                f"error_density={100*values['rgb_error_density_in_mask']:.2f}%, "
                f"event_repair={100*values['event_repair_rate_in_mask']:.2f}%, "
                f"selective+road_repair={100*values['selective_plus_road_repair_rate_in_mask']:.2f}%"
            )


if __name__ == "__main__":
    main()
