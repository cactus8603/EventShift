#!/usr/bin/env python
"""Measure how often prediction errors overlap event signal/support."""

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
    parser.add_argument("--event-classes", default="motorcycle,sky,pole,bicycle,sidewalk")
    parser.add_argument("--out", default="work_dirs/diagnostics/error_event_support_night.json")
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
        "total_valid": 0,
        "total_wrong": 0,
        "wrong_raw_event": 0,
        "wrong_support": 0,
        "wrong_no_raw_event": 0,
        "wrong_no_support": 0,
        "correct_raw_event": 0,
        "correct_support": 0,
        "total_raw_event": 0,
        "total_support": 0,
    }


def update_counts(counts, pred, label, raw_event, support, ignore_label=255):
    valid = (label != ignore_label) & (label >= 0) & (label < len(CLASSES))
    wrong = valid & (pred != label)
    correct = valid & (pred == label)
    raw = valid & (raw_event > 0)
    sup = valid & (support > 0)
    counts["total_valid"] += int(valid.sum())
    counts["total_wrong"] += int(wrong.sum())
    counts["wrong_raw_event"] += int((wrong & raw).sum())
    counts["wrong_support"] += int((wrong & sup).sum())
    counts["wrong_no_raw_event"] += int((wrong & ~raw).sum())
    counts["wrong_no_support"] += int((wrong & ~sup).sum())
    counts["correct_raw_event"] += int((correct & raw).sum())
    counts["correct_support"] += int((correct & sup).sum())
    counts["total_raw_event"] += int(raw.sum())
    counts["total_support"] += int(sup.sum())


def finalize_counts(counts):
    total_valid = max(counts["total_valid"], 1)
    total_wrong = max(counts["total_wrong"], 1)
    return {
        **counts,
        "error_rate": counts["total_wrong"] / total_valid,
        "wrong_raw_event_ratio": counts["wrong_raw_event"] / total_wrong,
        "wrong_support_ratio": counts["wrong_support"] / total_wrong,
        "wrong_no_raw_event_ratio": counts["wrong_no_raw_event"] / total_wrong,
        "wrong_no_support_ratio": counts["wrong_no_support"] / total_wrong,
        "raw_event_coverage": counts["total_raw_event"] / total_valid,
        "support_coverage": counts["total_support"] / total_valid,
    }


def main():
    args = parse_args()
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    register_cosec()
    target_ids, target_names = parse_class_ids(args.event_classes)

    rgb_cfg = setup_cfg(args.rgb_config, args.rgb_weights, args.device)
    event_cfg = setup_cfg(args.event_config, args.event_weights, args.device)
    rgb_mapper = MaskFormerSemanticDatasetMapper(rgb_cfg, False)
    event_mapper = MaskFormerSemanticDatasetMapper(event_cfg, False)
    rgb_model = build_model(rgb_cfg)
    event_model = build_model(event_cfg)

    method_counts = OrderedDict(
        (name, empty_counts())
        for name in [
            "rgb_only",
            "event_only",
            "selective_default",
            "selective_default_plus_road",
        ]
    )
    class_counts = OrderedDict((name, empty_counts()) for name in CLASSES)

    records = DatasetCatalog.get(args.dataset)
    if args.limit is not None:
        records = records[: args.limit]

    iterator = records if args.quiet else tqdm(records, desc=args.dataset)
    for record in iterator:
        label = load_label(record)
        rgb_scores = resize_scores(infer(rgb_model, rgb_mapper, record), label.shape)
        event_scores, mapped_event = infer(event_model, event_mapper, record, return_mapped=True)
        event_scores = resize_scores(event_scores, label.shape)
        density = resize_stat(mapped_event, 0, label.shape)
        support = resize_stat(mapped_event, 3, label.shape)
        raw_event = density > 0

        rgb_pred = rgb_scores.argmax(dim=0).numpy()
        event_pred = event_scores.argmax(dim=0).numpy()
        default_mask = np.isin(rgb_pred, target_ids) | np.isin(event_pred, target_ids)
        plus_road_ids = np.unique(np.concatenate([target_ids, np.asarray([CLASS_TO_ID["road"]])]))
        plus_road_mask = np.isin(rgb_pred, plus_road_ids) | np.isin(event_pred, plus_road_ids)
        predictions = {
            "rgb_only": rgb_pred,
            "event_only": event_pred,
            "selective_default": np.where(default_mask, event_pred, rgb_pred),
            "selective_default_plus_road": np.where(plus_road_mask, event_pred, rgb_pred),
        }

        for name, pred in predictions.items():
            update_counts(method_counts[name], pred, label, raw_event, support)

        best_pred = predictions["selective_default_plus_road"]
        for class_id, class_name in enumerate(CLASSES):
            class_label = np.where(label == class_id, label, 255)
            update_counts(class_counts[class_name], best_pred, class_label, raw_event, support)

    output = {
        "args": vars(args),
        "event_classes": target_names,
        "sample_count": len(records),
        "methods": OrderedDict((name, finalize_counts(counts)) for name, counts in method_counts.items()),
        "classes_for_selective_default_plus_road": OrderedDict(
            (name, finalize_counts(counts)) for name, counts in class_counts.items()
        ),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote diagnostics: {out_path}")
    print("Wrong-pixel event overlap:")
    for name, values in output["methods"].items():
        print(
            f"  {name}: error={100*values['error_rate']:.2f}%, "
            f"wrong_raw={100*values['wrong_raw_event_ratio']:.2f}%, "
            f"wrong_support={100*values['wrong_support_ratio']:.2f}%, "
            f"wrong_no_support={100*values['wrong_no_support_ratio']:.2f}%, "
            f"support_coverage={100*values['support_coverage']:.2f}%",
        )


if __name__ == "__main__":
    main()
