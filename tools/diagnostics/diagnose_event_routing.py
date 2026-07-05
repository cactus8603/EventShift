#!/usr/bin/env python
"""Diagnose RGB/event routing strategies on CoSEC val splits."""

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

from detectron2.checkpoint import DetectionCheckpointer  # noqa: E402
from detectron2.config import get_cfg  # noqa: E402
from detectron2.projects.deeplab import add_deeplab_config  # noqa: E402
from detectron2.data import DatasetCatalog  # noqa: E402
from mask2former import MaskFormerSemanticDatasetMapper, add_maskformer2_config  # noqa: E402
from train_mask2former_cosec import CoSECTrainer, register_cosec  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-config", default="configs/Mask2Former_SwinL_CoSEC_DayNight_Finetune.yaml")
    parser.add_argument("--rgb-weights", default="work_dirs/swinL_seqholdout305_from_latest_night/best_model_cosec_day.pth")
    parser.add_argument("--event-config", default="configs/Mask2Former_SwinL_CoSEC_EventReliability_Exp2.yaml")
    parser.add_argument("--event-weights", default="work_dirs/exp2_swinL_seqholdout305_cosec_event_reliability_epoch_from_day65/best_model_cosec_night.pth")
    parser.add_argument("--datasets", nargs="+", default=["cosec_day_val_event", "cosec_night_val_event"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--brightness-thresholds", default="30,40,50,60,70,80,90,100,110,120,130,140")
    parser.add_argument("--confidence-thresholds", default="0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90")
    parser.add_argument("--support-thresholds", default="0.0,0.1,0.25,0.5")
    parser.add_argument("--out", default="work_dirs/diagnostics/event_routing_val.json")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def parse_float_list(text):
    return [float(part) for part in text.split(",") if part.strip()]


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
    return outputs, mapped


def load_label(record):
    label = cv2.imread(record["sem_seg_file_name"], cv2.IMREAD_UNCHANGED)
    if label is None:
        raise RuntimeError(f"Could not read label: {record['sem_seg_file_name']}")
    if label.ndim == 3:
        label = label[:, :, 0]
    return label.astype(np.int64)


def image_brightness(record):
    image = cv2.imread(record["file_name"], cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {record['file_name']}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.median(gray))


def event_support_map(mapped, output_shape):
    event_stats = mapped.get("event_stats")
    if event_stats is None:
        return np.zeros(output_shape, dtype=np.float32)
    support = event_stats[3:4].float().unsqueeze(0)
    support = F.interpolate(support, size=output_shape, mode="nearest")[0, 0]
    return support.cpu().numpy().astype(np.float32)


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

    def merge(self, other):
        self.matrix += other.matrix

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
        }


def new_meter_map(method_names):
    return OrderedDict((name, ConfusionMeter()) for name in method_names)


def method_names(brightness_thresholds, confidence_thresholds, support_thresholds):
    names = ["rgb_only", "event_only", "oracle_domain_route"]
    names.extend(f"brightness_route@{threshold:g}" for threshold in brightness_thresholds)
    for confidence_threshold in confidence_thresholds:
        names.append(f"confidence_route@{confidence_threshold:g}")
        for support_threshold in support_thresholds:
            if support_threshold > 0:
                names.append(f"confidence_support_route@{confidence_threshold:g}/{support_threshold:g}")
    return names


def summarize(split_meters):
    summary = OrderedDict()
    for split_name, meters in split_meters.items():
        summary[split_name] = OrderedDict((name, meter.metrics()) for name, meter in meters.items())

    overall = OrderedDict()
    for name in next(iter(split_meters.values())).keys():
        meter = ConfusionMeter()
        for meters in split_meters.values():
            meter.merge(meters[name])
        overall[name] = meter.metrics()
    summary["overall"] = overall
    return summary


def best_methods(summary, metric="mIoU", topk=8):
    rows = []
    for method, values in summary["overall"].items():
        rows.append((values[metric], method, values))
    rows.sort(reverse=True, key=lambda item: item[0])
    return [{"method": method, **values} for _, method, values in rows[:topk]]


def main():
    args = parse_args()
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    register_cosec()

    brightness_thresholds = parse_float_list(args.brightness_thresholds)
    confidence_thresholds = parse_float_list(args.confidence_thresholds)
    support_thresholds = parse_float_list(args.support_thresholds)
    names = method_names(brightness_thresholds, confidence_thresholds, support_thresholds)

    rgb_cfg = setup_cfg(args.rgb_config, args.rgb_weights, args.device)
    event_cfg = setup_cfg(args.event_config, args.event_weights, args.device)
    rgb_mapper = MaskFormerSemanticDatasetMapper(rgb_cfg, False)
    event_mapper = MaskFormerSemanticDatasetMapper(event_cfg, False)
    rgb_model = build_model(rgb_cfg)
    event_model = build_model(event_cfg)

    split_meters = OrderedDict()
    sample_counts = OrderedDict()
    brightness_records = []
    for dataset_name in args.datasets:
        records = DatasetCatalog.get(dataset_name)
        if args.limit is not None:
            records = records[: args.limit]
        split_meters[dataset_name] = new_meter_map(names)
        sample_counts[dataset_name] = len(records)

        iterator = records if args.quiet else tqdm(records, desc=dataset_name)
        for record in iterator:
            label = load_label(record)
            rgb_scores, _ = infer(rgb_model, rgb_mapper, record)
            event_scores, mapped_event = infer(event_model, event_mapper, record)
            if rgb_scores.shape[-2:] != label.shape:
                rgb_scores = F.interpolate(
                    rgb_scores.unsqueeze(0),
                    size=label.shape,
                    mode="bilinear",
                    align_corners=False,
                )[0]
            if event_scores.shape[-2:] != label.shape:
                event_scores = F.interpolate(
                    event_scores.unsqueeze(0),
                    size=label.shape,
                    mode="bilinear",
                    align_corners=False,
                )[0]

            rgb_pred = rgb_scores.argmax(dim=0).numpy()
            event_pred = event_scores.argmax(dim=0).numpy()
            rgb_prob = torch.softmax(rgb_scores, dim=0)
            rgb_conf = rgb_prob.max(dim=0).values.numpy()
            support = event_support_map(mapped_event, label.shape)
            brightness = image_brightness(record)
            is_night = "Night_" in record.get("image_id", "") or "/Night_" in record["file_name"]
            brightness_records.append(
                {
                    "dataset": dataset_name,
                    "image_id": record.get("image_id"),
                    "brightness": brightness,
                    "is_night": bool(is_night),
                }
            )

            meters = split_meters[dataset_name]
            meters["rgb_only"].update(rgb_pred, label)
            meters["event_only"].update(event_pred, label)
            meters["oracle_domain_route"].update(event_pred if is_night else rgb_pred, label)

            for threshold in brightness_thresholds:
                pred = event_pred if brightness < threshold else rgb_pred
                meters[f"brightness_route@{threshold:g}"].update(pred, label)

            for confidence_threshold in confidence_thresholds:
                use_event = rgb_conf < confidence_threshold
                pred = np.where(use_event, event_pred, rgb_pred)
                meters[f"confidence_route@{confidence_threshold:g}"].update(pred, label)
                for support_threshold in support_thresholds:
                    if support_threshold <= 0:
                        continue
                    use_event = (rgb_conf < confidence_threshold) & (support > support_threshold)
                    pred = np.where(use_event, event_pred, rgb_pred)
                    meters[f"confidence_support_route@{confidence_threshold:g}/{support_threshold:g}"].update(
                        pred,
                        label,
                    )

    summary = summarize(split_meters)
    output = {
        "args": vars(args),
        "sample_counts": sample_counts,
        "best_overall_mIoU": best_methods(summary, "mIoU"),
        "summary": summary,
        "brightness": brightness_records,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, sort_keys=True, allow_nan=True)
        f.write("\n")

    print(f"Wrote diagnostics: {out_path}")
    print("Top overall mIoU:")
    for row in output["best_overall_mIoU"]:
        print(
            f"  {row['method']}: "
            f"mIoU={row['mIoU']:.4f}, mAcc={row['mAcc']:.4f}, aAcc={row['aAcc']:.4f}",
        )


if __name__ == "__main__":
    main()
