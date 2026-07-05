#!/usr/bin/env python
"""Diagnose event routing on top of a frozen RGB multi-scale TTA anchor."""

import argparse
import copy
import json
import os
import sys
import importlib.util
from collections import OrderedDict
from datetime import datetime
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
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--base-weights", required=True)
    parser.add_argument("--event-config", required=True)
    parser.add_argument("--event-weights", required=True)
    parser.add_argument("--dataset", default="cosec_night_val_event")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--scale-specs",
        default="s512:512:1200,s624:624:1200,s768:768:1400,s1024:1024:1600",
    )
    parser.add_argument("--scale-set", default="s512+s624+s768+s1024")
    parser.add_argument("--flip", action="store_true")
    parser.add_argument(
        "--route-classes",
        default="motorcycle,sky,pole,bicycle,sidewalk,road",
        help="Comma-separated RGB/event classes allowed to receive event correction.",
    )
    parser.add_argument("--base-conf-thresholds", default="0.50,0.60,0.70,0.80")
    parser.add_argument("--base-margin-thresholds", default="0.10,0.20,0.30,0.40")
    parser.add_argument("--event-conf-thresholds", default="0.30,0.40,0.50")
    parser.add_argument("--boundary-radii", default="0,1,3")
    parser.add_argument("--out", required=True)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def split_csv(text):
    return [part.strip() for part in str(text).split(",") if part.strip()]


def parse_float_list(text):
    return [float(part) for part in split_csv(text)]


def parse_int_list(text):
    return [int(part) for part in split_csv(text)]


def parse_class_ids(text):
    names = split_csv(text)
    unknown = [name for name in names if name not in CLASS_TO_ID]
    if unknown:
        raise ValueError(f"Unknown class names: {unknown}; valid={list(CLASSES)}")
    return names, np.asarray([CLASS_TO_ID[name] for name in names], dtype=np.int64)


def parse_scale_specs(text):
    specs = OrderedDict()
    for item in split_csv(text):
        name, min_size, max_size = item.split(":")
        specs[name] = {
            "name": name,
            "min_size": int(min_size),
            "max_size": int(max_size),
        }
    return specs


def setup_cfg(config_file, weights, device, min_size=None, max_size=None):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.DEVICE = device
    cfg.DATASETS.TEST = ()
    cfg.TEST.AUG.ENABLED = False
    if min_size is not None:
        cfg.INPUT.MIN_SIZE_TEST = int(min_size)
    if max_size is not None:
        cfg.INPUT.MAX_SIZE_TEST = int(max_size)
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


def load_label(record):
    label = cv2.imread(record["sem_seg_file_name"], cv2.IMREAD_UNCHANGED)
    if label is None:
        raise RuntimeError(f"Could not read label: {record['sem_seg_file_name']}")
    if label.ndim == 3:
        label = label[:, :, 0]
    return label.astype(np.int64)


def resize_scores(scores, shape):
    if tuple(scores.shape[-2:]) == tuple(shape):
        return scores
    return F.interpolate(
        scores.unsqueeze(0),
        size=shape,
        mode="bilinear",
        align_corners=False,
    )[0]


def normalize_scores(scores):
    prob = scores.float().clamp_min(1e-8)
    return prob / prob.sum(dim=0, keepdim=True).clamp_min(1e-8)


def infer_prob(model, mapped, label_shape, use_flip=False):
    with torch.no_grad():
        scores = model([dict(mapped)])[0]["sem_seg"].detach().cpu()
        if use_flip:
            flipped = dict(mapped)
            flipped["image"] = torch.flip(mapped["image"], dims=[2])
            for key in ["event", "event_edge", "event_stats"]:
                if key in flipped:
                    flipped[key] = torch.flip(flipped[key], dims=[2])
            for key, value in list(flipped.items()):
                if key.startswith("boundary_r") or key.startswith("class_boundary_r"):
                    flipped[key] = torch.flip(value, dims=[-1])
            flip_scores = model([flipped])[0]["sem_seg"].detach().cpu()
            scores = 0.5 * (scores + torch.flip(flip_scores, dims=[2]))
    return normalize_scores(resize_scores(scores, label_shape))


def without_event_fields(record):
    cleaned = copy.deepcopy(record)
    for key in ["event_h5", "event_old", "event_new"]:
        cleaned.pop(key, None)
    return cleaned


def build_scale_branches(args, specs, scale_names):
    branches = []
    for scale_name in scale_names:
        spec = specs[scale_name]
        cfg = setup_cfg(
            args.base_config,
            args.base_weights,
            args.device,
            spec["min_size"],
            spec["max_size"],
        )
        branches.append(
            {
                "name": scale_name,
                "mapper": MaskFormerSemanticDatasetMapper(cfg, False),
            }
        )
    return branches


def collect_tta_prob(args, model, branches, record, label_shape):
    avg = None
    used = 0
    base_record = without_event_fields(record)
    for branch in branches:
        mapped = branch["mapper"](copy.deepcopy(base_record))
        prob = infer_prob(model, mapped, label_shape, use_flip=args.flip)
        avg = prob if avg is None else avg + prob
        used += 1
    return avg / float(max(used, 1))


def top_conf_margin(prob):
    top2 = torch.topk(prob, k=2, dim=0).values
    return top2[0].numpy(), (top2[0] - top2[1]).numpy()


def resize_stat(event_stats, channel, shape):
    stat = event_stats[channel : channel + 1].float().unsqueeze(0)
    stat = F.interpolate(stat, size=shape, mode="nearest")[0, 0]
    return stat.cpu().numpy()


def valid_label_mask(label, ignore_label=255):
    return (label != ignore_label) & (label >= 0) & (label < len(CLASSES))


def pred_boundary(pred, radius):
    radius = int(radius)
    if radius <= 0:
        return np.ones_like(pred, dtype=bool)
    padded = np.pad(pred, radius, mode="edge")
    center = padded[radius : radius + pred.shape[0], radius : radius + pred.shape[1]]
    out = np.zeros(pred.shape, dtype=bool)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            shifted = padded[
                radius + dy : radius + dy + pred.shape[0],
                radius + dx : radius + dx + pred.shape[1],
            ]
            out |= shifted != center
    return out


class ConfusionMeter:
    def __init__(self, num_classes=19, ignore_label=255):
        self.num_classes = int(num_classes)
        self.ignore_label = int(ignore_label)
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, pred, label):
        keep = (label != self.ignore_label) & (label >= 0) & (label < self.num_classes)
        keep &= (pred >= 0) & (pred < self.num_classes)
        indices = self.num_classes * label[keep].astype(np.int64) + pred[keep].astype(np.int64)
        self.matrix += np.bincount(indices, minlength=self.num_classes**2).reshape(
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
            "aAcc": float(100.0 * tp.sum() / total) if total > 0 else float("nan"),
        }


def empty_counts():
    return {
        "valid_pixels": 0,
        "raw_event_pixels": 0,
        "support_pixels": 0,
        "base_wrong": 0,
        "repaired": 0,
        "damaged": 0,
        "changed": 0,
        "changed_event": 0,
    }


def add_counts(counts, base_pred, pred, label, valid, event_region):
    base_wrong = valid & (base_pred != label)
    base_correct = valid & (base_pred == label)
    repaired = base_wrong & (pred == label)
    damaged = base_correct & (pred != label)
    changed = valid & (base_pred != pred)
    counts["valid_pixels"] += int(valid.sum())
    counts["base_wrong"] += int(base_wrong.sum())
    counts["repaired"] += int(repaired.sum())
    counts["damaged"] += int(damaged.sum())
    counts["changed"] += int(changed.sum())
    counts["changed_event"] += int((changed & event_region).sum())


def finalize_counts(counts):
    valid = counts["valid_pixels"]
    return {
        **counts,
        "repair_rate": float(counts["repaired"] / counts["base_wrong"]) if counts["base_wrong"] else 0.0,
        "net_repaired": counts["repaired"] - counts["damaged"],
        "changed_rate": float(counts["changed"] / valid) if valid else 0.0,
        "changed_event_rate": (
            float(counts["changed_event"] / counts["changed"]) if counts["changed"] else 0.0
        ),
    }


def update_method(methods, name, pred, base_pred, label, valid, event_region):
    if name not in methods:
        methods[name] = {"meter": ConfusionMeter(num_classes=len(CLASSES)), "counts": empty_counts()}
    methods[name]["meter"].update(pred, label)
    add_counts(methods[name]["counts"], base_pred, pred, label, valid, event_region)


def main():
    args = parse_args()
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    register_cosec()

    route_names, route_ids = parse_class_ids(args.route_classes)
    scale_specs = parse_scale_specs(args.scale_specs)
    scale_names = [name.strip() for name in args.scale_set.split("+") if name.strip()]
    unknown_scales = [name for name in scale_names if name not in scale_specs]
    if unknown_scales:
        raise ValueError(f"Unknown scales in --scale-set: {unknown_scales}")

    event_cfg = setup_cfg(args.event_config, args.event_weights, args.device)
    event_mapper = MaskFormerSemanticDatasetMapper(event_cfg, False)
    event_model = build_model(event_cfg)
    base_cfg = setup_cfg(args.base_config, args.base_weights, args.device)
    base_model = build_model(base_cfg)
    scale_branches = build_scale_branches(args, scale_specs, scale_names)

    records = list(DatasetCatalog.get(args.dataset))
    if args.limit is not None:
        records = records[: args.limit]

    base_conf_thresholds = parse_float_list(args.base_conf_thresholds)
    base_margin_thresholds = parse_float_list(args.base_margin_thresholds)
    event_conf_thresholds = parse_float_list(args.event_conf_thresholds)
    boundary_radii = parse_int_list(args.boundary_radii)

    methods = OrderedDict()
    iterator = records if args.quiet else tqdm(records, desc=args.dataset)
    for record in iterator:
        label = load_label(record)
        valid = valid_label_mask(label)

        base_prob = collect_tta_prob(args, base_model, scale_branches, record, label.shape)
        base_pred = base_prob.argmax(dim=0).numpy()
        base_conf, base_margin = top_conf_margin(base_prob)

        event_mapped = event_mapper(copy.deepcopy(record))
        event_prob = infer_prob(event_model, event_mapped, label.shape, use_flip=False)
        event_pred = event_prob.argmax(dim=0).numpy()
        event_conf, event_margin = top_conf_margin(event_prob)

        event_stats = event_mapped["event_stats"].float()
        raw_event = valid & (resize_stat(event_stats, 0, label.shape) > 0)
        support = valid & (resize_stat(event_stats, 3, label.shape) > 0)
        event_regions = OrderedDict([("raw", raw_event), ("support", support)])

        update_method(methods, "rgb_tta", base_pred, base_pred, label, valid, raw_event | support)
        update_method(methods, "event_candidate", event_pred, base_pred, label, valid, raw_event | support)

        rgb_class = np.isin(base_pred, route_ids)
        event_class = np.isin(event_pred, route_ids)
        either_class = rgb_class | event_class

        for region_name, event_region in event_regions.items():
            for radius in boundary_radii:
                boundary = pred_boundary(base_pred, radius)
                base_mask = event_region & boundary
                for base_conf_thr in base_conf_thresholds:
                    for event_conf_thr in event_conf_thresholds:
                        route = (
                            base_mask
                            & either_class
                            & (base_conf >= base_conf_thr)
                            & (event_conf >= event_conf_thr)
                            & (event_pred != base_pred)
                        )
                        pred = np.where(route, event_pred, base_pred)
                        update_method(
                            methods,
                            (
                                f"either_cls_{region_name}_b{radius}"
                                f"_baseconf{base_conf_thr:g}_eventconf{event_conf_thr:g}"
                            ),
                            pred,
                            base_pred,
                            label,
                            valid,
                            event_region,
                        )

                    for base_margin_thr in base_margin_thresholds:
                        route = (
                            base_mask
                            & either_class
                            & (base_conf >= base_conf_thr)
                            & (base_margin >= base_margin_thr)
                            & (event_margin >= base_margin)
                            & (event_pred != base_pred)
                        )
                        pred = np.where(route, event_pred, base_pred)
                        update_method(
                            methods,
                            (
                                f"either_cls_{region_name}_b{radius}"
                                f"_baseconf{base_conf_thr:g}_basemargin{base_margin_thr:g}"
                                "_eventmargin_ge_base"
                            ),
                            pred,
                            base_pred,
                            label,
                            valid,
                            event_region,
                        )

                for event_conf_thr in event_conf_thresholds:
                    route = (
                        base_mask
                        & event_class
                        & (event_conf >= event_conf_thr)
                        & (event_margin >= base_margin)
                        & (event_pred != base_pred)
                    )
                    pred = np.where(route, event_pred, base_pred)
                    update_method(
                        methods,
                        f"event_cls_{region_name}_b{radius}_eventconf{event_conf_thr:g}_margin_ge_base",
                        pred,
                        base_pred,
                        label,
                        valid,
                        event_region,
                    )

        for value in methods.values():
            value["counts"]["raw_event_pixels"] += int(raw_event.sum())
            value["counts"]["support_pixels"] += int(support.sum())

    output = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "route_classes": route_names,
        "sample_count": len(records),
        "methods": OrderedDict(
            (
                name,
                {
                    **value["meter"].metrics(),
                    **finalize_counts(value["counts"]),
                },
            )
            for name, value in methods.items()
        ),
    }
    baseline = output["methods"]["rgb_tta"]["mIoU"]
    output["top_by_mIoU"] = [
        {"method": name, "delta_vs_rgb_tta": values["mIoU"] - baseline, **values}
        for name, values in sorted(
            output["methods"].items(),
            key=lambda item: item[1]["mIoU"],
            reverse=True,
        )[:30]
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote TTA/event routing diagnostic: {out_path}")
    print(f"RGB TTA baseline: mIoU={baseline:.4f}")
    for row in output["top_by_mIoU"][:10]:
        print(
            f"  {row['method']}: mIoU={row['mIoU']:.4f} "
            f"delta={row['delta_vs_rgb_tta']:+.4f} "
            f"changed={100*row['changed_rate']:.4f}% "
            f"net={row['net_repaired']}"
        )


if __name__ == "__main__":
    main()
