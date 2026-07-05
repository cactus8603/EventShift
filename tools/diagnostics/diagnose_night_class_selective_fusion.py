#!/usr/bin/env python
"""Diagnose class-selective RGB/event fusion on CoSEC night validation."""

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
    parser.add_argument("--rgb-weights", default="work_dirs/swinL_seqholdout305_from_latest_night/best_model_cosec_night.pth")
    parser.add_argument("--event-config", default="configs/Mask2Former_SwinL_CoSEC_EventReliability_Exp2.yaml")
    parser.add_argument("--event-weights", default="work_dirs/exp2_swinL_seqholdout305_cosec_event_reliability_epoch_from_day65/best_model_cosec_night.pth")
    parser.add_argument("--dataset", default="cosec_night_val_event")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--event-classes",
        default="motorcycle,sky,pole,bicycle,sidewalk",
        help="Comma-separated classes where event branch improved over RGB baseline.",
    )
    parser.add_argument("--confidence-thresholds", default="0.05,0.10,0.15,0.20,0.25,0.30")
    parser.add_argument("--out", default="work_dirs/diagnostics/night_class_selective_fusion.json")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def parse_float_list(text):
    return [float(part) for part in text.split(",") if part.strip()]


def parse_class_list(text):
    names = [part.strip() for part in text.split(",") if part.strip()]
    unknown = [name for name in names if name not in CLASS_TO_ID]
    if unknown:
        raise ValueError(f"Unknown class names: {unknown}; valid classes: {list(CLASSES)}")
    return names


def class_ids(class_names):
    return np.asarray([CLASS_TO_ID[name] for name in class_names], dtype=np.int64)


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


def infer(model, mapper, record):
    with torch.no_grad():
        mapped = mapper(copy.deepcopy(record))
        outputs = model([mapped])[0]["sem_seg"].detach().cpu()
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


class ConfusionMeter:
    def __init__(self, num_classes=19, ignore_label=255):
        self.num_classes = int(num_classes)
        self.ignore_label = int(ignore_label)
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, pred, label):
        pred = np.asarray(pred, dtype=np.int64)
        label = np.asarray(label, dtype=np.int64)
        if pred.shape != label.shape:
            pred = cv2.resize(
                pred.astype(np.uint8),
                (label.shape[1], label.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.int64)
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
        per_class_iou = {
            CLASSES[idx]: None if np.isnan(value) else float(100.0 * value)
            for idx, value in enumerate(iou)
        }
        per_class_acc = {
            CLASSES[idx]: None if np.isnan(value) else float(100.0 * value)
            for idx, value in enumerate(acc)
        }
        return {
            "mIoU": float(100.0 * np.nanmean(iou)),
            "mAcc": float(100.0 * np.nanmean(acc)),
            "aAcc": float(100.0 * true_positive.sum() / total) if total > 0 else float("nan"),
            "IoU": per_class_iou,
            "Acc": per_class_acc,
        }


def make_class_sets(default_classes):
    default = list(default_classes)
    variants = OrderedDict()
    variants["default"] = default
    variants["no_sky"] = [name for name in default if name != "sky"]
    variants["mobile_thin"] = [name for name in default if name in {"motorcycle", "bicycle", "pole"}]
    variants["dynamic"] = [name for name in default if name in {"motorcycle", "bicycle"}]
    variants["default_plus_road"] = default + [name for name in ["road"] if name not in default]
    variants["default_plus_road_vegetation"] = default + [
        name for name in ["road", "vegetation"] if name not in default
    ]
    return variants


def method_names(class_sets, confidence_thresholds):
    names = ["rgb_only", "event_only"]
    for set_name in class_sets:
        names.extend(
            [
                f"event_pred_select:{set_name}",
                f"rgb_pred_select:{set_name}",
                f"either_pred_select:{set_name}",
            ],
        )
        for threshold in confidence_thresholds:
            names.extend(
                [
                    f"event_pred_select:{set_name}:event_conf@{threshold:g}",
                    f"event_pred_select:{set_name}:event_margin@{threshold:g}",
                    f"rgb_pred_select:{set_name}:event_conf@{threshold:g}",
                ],
            )
    return names


def top_methods(summary, topk=12):
    rows = [(values["mIoU"], name, values) for name, values in summary.items()]
    rows.sort(reverse=True, key=lambda item: item[0])
    return [{"method": name, **values} for _, name, values in rows[:topk]]


def main():
    args = parse_args()
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    register_cosec()

    default_classes = parse_class_list(args.event_classes)
    class_sets = make_class_sets(default_classes)
    confidence_thresholds = parse_float_list(args.confidence_thresholds)
    meters = OrderedDict((name, ConfusionMeter()) for name in method_names(class_sets, confidence_thresholds))
    usage = OrderedDict((name, {"event_pixels": 0, "total_pixels": 0}) for name in meters)

    rgb_cfg = setup_cfg(args.rgb_config, args.rgb_weights, args.device)
    event_cfg = setup_cfg(args.event_config, args.event_weights, args.device)
    rgb_mapper = MaskFormerSemanticDatasetMapper(rgb_cfg, False)
    event_mapper = MaskFormerSemanticDatasetMapper(event_cfg, False)
    rgb_model = build_model(rgb_cfg)
    event_model = build_model(event_cfg)

    records = DatasetCatalog.get(args.dataset)
    if args.limit is not None:
        records = records[: args.limit]

    iterator = records if args.quiet else tqdm(records, desc=args.dataset)
    for record in iterator:
        label = load_label(record)
        rgb_scores = resize_scores(infer(rgb_model, rgb_mapper, record), label.shape)
        event_scores = resize_scores(infer(event_model, event_mapper, record), label.shape)

        rgb_conf = rgb_scores.max(dim=0).values.numpy()
        event_conf = event_scores.max(dim=0).values.numpy()
        rgb_pred = rgb_scores.argmax(dim=0).numpy()
        event_pred = event_scores.argmax(dim=0).numpy()

        base_preds = {
            "rgb_only": rgb_pred,
            "event_only": event_pred,
        }
        for name, pred in base_preds.items():
            meters[name].update(pred, label)
            usage[name]["event_pixels"] += int(name == "event_only") * int(np.prod(label.shape))
            usage[name]["total_pixels"] += int(np.prod(label.shape))

        for set_name, names in class_sets.items():
            ids = class_ids(names)
            event_is_target = np.isin(event_pred, ids)
            rgb_is_target = np.isin(rgb_pred, ids)
            masks = {
                f"event_pred_select:{set_name}": event_is_target,
                f"rgb_pred_select:{set_name}": rgb_is_target,
                f"either_pred_select:{set_name}": event_is_target | rgb_is_target,
            }
            for threshold in confidence_thresholds:
                masks[f"event_pred_select:{set_name}:event_conf@{threshold:g}"] = (
                    event_is_target & (event_conf >= threshold)
                )
                masks[f"event_pred_select:{set_name}:event_margin@{threshold:g}"] = (
                    event_is_target & ((event_conf - rgb_conf) >= threshold)
                )
                masks[f"rgb_pred_select:{set_name}:event_conf@{threshold:g}"] = (
                    rgb_is_target & (event_conf >= threshold)
                )

            for name, use_event in masks.items():
                pred = np.where(use_event, event_pred, rgb_pred)
                meters[name].update(pred, label)
                usage[name]["event_pixels"] += int(use_event.sum())
                usage[name]["total_pixels"] += int(np.prod(label.shape))

    summary = OrderedDict()
    for name, meter in meters.items():
        values = meter.metrics()
        values["event_pixel_ratio"] = (
            usage[name]["event_pixels"] / usage[name]["total_pixels"]
            if usage[name]["total_pixels"]
            else 0.0
        )
        summary[name] = values

    output = {
        "args": vars(args),
        "classes": list(CLASSES),
        "class_sets": class_sets,
        "sample_count": len(records),
        "best_mIoU": top_methods(summary),
        "summary": summary,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, sort_keys=True, allow_nan=True)
        f.write("\n")

    print(f"Wrote diagnostics: {out_path}")
    print("Top night mIoU:")
    for row in output["best_mIoU"]:
        print(
            f"  {row['method']}: "
            f"mIoU={row['mIoU']:.4f}, mAcc={row['mAcc']:.4f}, "
            f"aAcc={row['aAcc']:.4f}, event_pixels={row['event_pixel_ratio']:.4f}",
        )


if __name__ == "__main__":
    main()
