#!/usr/bin/env python
"""Measure how many RGB errors are repaired by event predictions."""

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
    parser.add_argument("--out", required=True)
    parser.add_argument("--strip-rgb-event-fields", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def parse_class_ids(text):
    names = [part.strip() for part in text.split(",") if part.strip()]
    unknown = [name for name in names if name not in CLASS_TO_ID]
    if unknown:
        raise ValueError(f"Unknown class names: {unknown}")
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


def without_event_fields(record):
    stripped = copy.deepcopy(record)
    for key in ["event_h5", "event_old", "event_new"]:
        stripped.pop(key, None)
    return stripped


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


def empty_counts():
    return {
        "valid_pixels": 0,
        "raw_pixels": 0,
        "support_pixels": 0,
        "rgb_wrong": 0,
        "rgb_wrong_raw": 0,
        "rgb_wrong_support": 0,
        "repaired": 0,
        "repaired_raw": 0,
        "repaired_support": 0,
        "damaged": 0,
        "damaged_raw": 0,
        "damaged_support": 0,
        "method_wrong": 0,
        "method_wrong_raw": 0,
        "method_wrong_support": 0,
    }


def add_counts(counts, rgb_pred, method_pred, label, raw_event, support, ignore_label=255):
    valid = (label != ignore_label) & (label >= 0) & (label < len(CLASSES))
    raw = valid & (raw_event > 0)
    sup = valid & (support > 0)
    rgb_wrong = valid & (rgb_pred != label)
    rgb_correct = valid & (rgb_pred == label)
    method_wrong = valid & (method_pred != label)
    method_correct = valid & (method_pred == label)
    repaired = rgb_wrong & method_correct
    damaged = rgb_correct & method_wrong

    counts["valid_pixels"] += int(valid.sum())
    counts["raw_pixels"] += int(raw.sum())
    counts["support_pixels"] += int(sup.sum())
    counts["rgb_wrong"] += int(rgb_wrong.sum())
    counts["rgb_wrong_raw"] += int((rgb_wrong & raw).sum())
    counts["rgb_wrong_support"] += int((rgb_wrong & sup).sum())
    counts["repaired"] += int(repaired.sum())
    counts["repaired_raw"] += int((repaired & raw).sum())
    counts["repaired_support"] += int((repaired & sup).sum())
    counts["damaged"] += int(damaged.sum())
    counts["damaged_raw"] += int((damaged & raw).sum())
    counts["damaged_support"] += int((damaged & sup).sum())
    counts["method_wrong"] += int(method_wrong.sum())
    counts["method_wrong_raw"] += int((method_wrong & raw).sum())
    counts["method_wrong_support"] += int((method_wrong & sup).sum())


def div(num, den):
    return float(num / den) if den else 0.0


def finalize(counts):
    return {
        **counts,
        "raw_coverage": div(counts["raw_pixels"], counts["valid_pixels"]),
        "support_coverage": div(counts["support_pixels"], counts["valid_pixels"]),
        "rgb_error_rate": div(counts["rgb_wrong"], counts["valid_pixels"]),
        "method_error_rate": div(counts["method_wrong"], counts["valid_pixels"]),
        "rgb_wrong_raw_ratio": div(counts["rgb_wrong_raw"], counts["rgb_wrong"]),
        "rgb_wrong_support_ratio": div(counts["rgb_wrong_support"], counts["rgb_wrong"]),
        "repair_rate_all_rgb_errors": div(counts["repaired"], counts["rgb_wrong"]),
        "repair_rate_raw_rgb_errors": div(counts["repaired_raw"], counts["rgb_wrong_raw"]),
        "repair_rate_support_rgb_errors": div(counts["repaired_support"], counts["rgb_wrong_support"]),
        "damage_rate_all_rgb_correct": div(
            counts["damaged"],
            counts["valid_pixels"] - counts["rgb_wrong"],
        ),
        "net_repaired": counts["repaired"] - counts["damaged"],
        "net_repaired_raw": counts["repaired_raw"] - counts["damaged_raw"],
        "net_repaired_support": counts["repaired_support"] - counts["damaged_support"],
        "net_support_per_rgb_wrong_support": div(
            counts["repaired_support"] - counts["damaged_support"],
            counts["rgb_wrong_support"],
        ),
    }


def main():
    args = parse_args()
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    register_cosec()
    target_ids, target_names = parse_class_ids(args.event_classes)
    plus_road_ids = np.unique(np.concatenate([target_ids, np.asarray([CLASS_TO_ID["road"]])]))

    rgb_cfg = setup_cfg(args.rgb_config, args.rgb_weights, args.device)
    event_cfg = setup_cfg(args.event_config, args.event_weights, args.device)
    rgb_mapper = MaskFormerSemanticDatasetMapper(rgb_cfg, False)
    event_mapper = MaskFormerSemanticDatasetMapper(event_cfg, False)
    rgb_model = build_model(rgb_cfg)
    event_model = build_model(event_cfg)

    counts = OrderedDict(
        (name, empty_counts())
        for name in [
            "event_only",
            "selective_default",
            "selective_default_plus_road",
        ]
    )

    records = DatasetCatalog.get(args.dataset)
    if args.limit is not None:
        records = records[: args.limit]

    iterator = records if args.quiet else tqdm(records, desc=args.dataset)
    for record in iterator:
        label = load_label(record)
        rgb_record = without_event_fields(record) if args.strip_rgb_event_fields else record
        rgb_scores = resize_scores(infer(rgb_model, rgb_mapper, rgb_record), label.shape)
        event_scores, mapped_event = infer(event_model, event_mapper, record, return_mapped=True)
        event_scores = resize_scores(event_scores, label.shape)
        density = resize_stat(mapped_event, 0, label.shape)
        support = resize_stat(mapped_event, 3, label.shape)
        raw_event = density > 0

        rgb_pred = rgb_scores.argmax(dim=0).numpy()
        event_pred = event_scores.argmax(dim=0).numpy()
        default_mask = np.isin(rgb_pred, target_ids) | np.isin(event_pred, target_ids)
        plus_road_mask = np.isin(rgb_pred, plus_road_ids) | np.isin(event_pred, plus_road_ids)
        predictions = {
            "event_only": event_pred,
            "selective_default": np.where(default_mask, event_pred, rgb_pred),
            "selective_default_plus_road": np.where(plus_road_mask, event_pred, rgb_pred),
        }
        for name, pred in predictions.items():
            add_counts(counts[name], rgb_pred, pred, label, raw_event, support)

    output = {
        "args": vars(args),
        "event_classes": target_names,
        "sample_count": len(records),
        "methods": OrderedDict((name, finalize(value)) for name, value in counts.items()),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote diagnostics: {out_path}")
    print("Repair/damage summary:")
    for name, values in output["methods"].items():
        print(
            f"  {name}: "
            f"repair_support={100*values['repair_rate_support_rgb_errors']:.2f}%, "
            f"repair_raw={100*values['repair_rate_raw_rgb_errors']:.2f}%, "
            f"net_support={values['net_repaired_support']}, "
            f"method_error={100*values['method_error_rate']:.2f}%",
        )


if __name__ == "__main__":
    main()
