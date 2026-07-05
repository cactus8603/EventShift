#!/usr/bin/env python
"""Transition-pair repair diagnostics from saved semantic prediction RLE JSON."""

import argparse
import copy
import json
import os
import sys
import importlib.util
from collections import OrderedDict, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch.nn.functional as F
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
from detectron2.config import get_cfg  # noqa: E402
from detectron2.data import DatasetCatalog  # noqa: E402
from detectron2.projects.deeplab import add_deeplab_config  # noqa: E402
from mask2former import MaskFormerSemanticDatasetMapper, add_maskformer2_config  # noqa: E402
from train_mask2former_cosec import register_cosec  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-predictions", required=True)
    parser.add_argument("--new-predictions", required=True)
    parser.add_argument("--event-config", required=True)
    parser.add_argument("--dataset", default="cosec_day_val_event")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def setup_mapper(config_file):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.freeze()
    return MaskFormerSemanticDatasetMapper(cfg, False)


def load_prediction_index(path):
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    index = defaultdict(list)
    for row in rows:
        index[row["file_name"]].append(row)
        index[prediction_key(row["file_name"])].append(row)
    return index


def prediction_key(path):
    parts = Path(path).parts
    for idx, part in enumerate(parts):
        if part.startswith(("Day_", "Night_", "REAL_")):
            return str(Path(*parts[idx:]))
    return str(Path(path).name)


def decode_prediction(rows, shape):
    pred = np.full(shape, fill_value=255, dtype=np.int64)
    for row in rows:
        rle = dict(row["segmentation"])
        counts = rle.get("counts")
        if isinstance(counts, str):
            rle["counts"] = counts.encode("ascii")
        mask = mask_util.decode(rle).astype(bool)
        if mask.shape != shape:
            raise RuntimeError(f"Prediction mask shape mismatch: pred={mask.shape}, label={shape}")
        pred[mask] = int(row["category_id"])
    return pred


def load_label(record):
    label = cv2.imread(record["sem_seg_file_name"], cv2.IMREAD_UNCHANGED)
    if label is None:
        raise RuntimeError(f"Could not read label: {record['sem_seg_file_name']}")
    if label.ndim == 3:
        label = label[:, :, 0]
    return label.astype(np.int64)


def resize_stat(event_stats, channel, shape):
    stat = event_stats[channel : channel + 1].float().unsqueeze(0)
    return F.interpolate(stat, size=shape, mode="nearest")[0, 0].cpu().numpy()


def valid_label_mask(label, base_pred, new_pred, ignore_label=255):
    valid = (label != ignore_label) & (label >= 0) & (label < len(CLASSES))
    valid &= (base_pred >= 0) & (base_pred < len(CLASSES))
    valid &= (new_pred >= 0) & (new_pred < len(CLASSES))
    return valid


def pair_name(base_cls, new_cls):
    return f"{CLASSES[int(base_cls)]}->{CLASSES[int(new_cls)]}"


def empty_pair_counts():
    return {
        "changed": 0,
        "repaired": 0,
        "damaged": 0,
        "both_wrong": 0,
    }


def empty_region_counts():
    return {
        "valid_pixels": 0,
        "mask_pixels": 0,
        "changed": 0,
        "repaired": 0,
        "damaged": 0,
        "both_wrong": 0,
    }


def add_region(region_counts, pair_stats, name, region, base_pred, new_pred, label, valid):
    mask = valid & region
    changed = mask & (base_pred != new_pred)
    repaired = changed & (base_pred != label) & (new_pred == label)
    damaged = changed & (base_pred == label) & (new_pred != label)
    both_wrong = changed & (base_pred != label) & (new_pred != label)

    region_counts[name]["valid_pixels"] += int(valid.sum())
    region_counts[name]["mask_pixels"] += int(mask.sum())
    region_counts[name]["changed"] += int(changed.sum())
    region_counts[name]["repaired"] += int(repaired.sum())
    region_counts[name]["damaged"] += int(damaged.sum())
    region_counts[name]["both_wrong"] += int(both_wrong.sum())

    if not changed.any():
        return
    num_classes = len(CLASSES)
    changed_counts = np.bincount(
        num_classes * base_pred[changed] + new_pred[changed],
        minlength=num_classes**2,
    )
    repaired_counts = np.bincount(
        num_classes * base_pred[repaired] + new_pred[repaired],
        minlength=num_classes**2,
    )
    damaged_counts = np.bincount(
        num_classes * base_pred[damaged] + new_pred[damaged],
        minlength=num_classes**2,
    )
    both_wrong_counts = np.bincount(
        num_classes * base_pred[both_wrong] + new_pred[both_wrong],
        minlength=num_classes**2,
    )
    for idx in np.flatnonzero(changed_counts):
        item = pair_stats[name][pair_name(idx // num_classes, idx % num_classes)]
        item["changed"] += int(changed_counts[idx])
        item["repaired"] += int(repaired_counts[idx])
        item["damaged"] += int(damaged_counts[idx])
        item["both_wrong"] += int(both_wrong_counts[idx])


def div(num, den):
    return float(num / den) if den else 0.0


def finalize_counts(counts):
    out = dict(counts)
    out.update(
        {
            "mask_coverage": div(counts["mask_pixels"], counts["valid_pixels"]),
            "changed_rate_in_mask": div(counts["changed"], counts["mask_pixels"]),
            "repair_precision": div(counts["repaired"], counts["changed"]),
            "damage_rate": div(counts["damaged"], counts["changed"]),
            "both_wrong_rate": div(counts["both_wrong"], counts["changed"]),
            "net_repaired": counts["repaired"] - counts["damaged"],
        }
    )
    return out


def finalize_pair(pair, counts):
    out = dict(counts)
    out.update(
        {
            "pair": pair,
            "net_repaired": counts["repaired"] - counts["damaged"],
            "repair_precision": div(counts["repaired"], counts["changed"]),
            "damage_rate": div(counts["damaged"], counts["changed"]),
            "both_wrong_rate": div(counts["both_wrong"], counts["changed"]),
        }
    )
    return out


def main():
    args = parse_args()
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    register_cosec()
    mapper = setup_mapper(args.event_config)
    base_index = load_prediction_index(args.base_predictions)
    new_index = load_prediction_index(args.new_predictions)

    records = DatasetCatalog.get(args.dataset)
    if args.limit is not None:
        records = records[: args.limit]

    region_counts = defaultdict(empty_region_counts)
    pair_stats = defaultdict(lambda: defaultdict(empty_pair_counts))
    missing = []
    for record in records:
        file_name = record["file_name"]
        key = prediction_key(file_name)
        base_rows = base_index.get(file_name) or base_index.get(key)
        new_rows = new_index.get(file_name) or new_index.get(key)
        if not base_rows or not new_rows:
            missing.append(file_name)
            continue
        label = load_label(record)
        base_pred = decode_prediction(base_rows, label.shape)
        new_pred = decode_prediction(new_rows, label.shape)
        valid = valid_label_mask(label, base_pred, new_pred)

        mapped = mapper(copy.deepcopy(record))
        event_stats = mapped["event_stats"].float()
        raw_event = valid & (resize_stat(event_stats, 0, label.shape) > 0)
        support = valid & (resize_stat(event_stats, 3, label.shape) > 0)
        support_only = support & ~raw_event
        raw_only = raw_event & ~support
        event_union = raw_event | support

        regions = OrderedDict(
            [
                ("all", valid),
                ("event_union", event_union),
                ("raw_event", raw_event),
                ("support", support),
                ("raw_only", raw_only),
                ("support_only", support_only),
            ]
        )
        for name, region in regions.items():
            add_region(region_counts, pair_stats, name, region, base_pred, new_pred, label, valid)

    regions_out = OrderedDict()
    pairs_out = OrderedDict()
    for name in region_counts:
        rows = [finalize_pair(pair, counts) for pair, counts in pair_stats[name].items()]
        regions_out[name] = finalize_counts(region_counts[name])
        pairs_out[name] = {
            "top_positive_by_net": sorted(rows, key=lambda row: row["net_repaired"], reverse=True)[
                : args.top_k
            ],
            "top_negative_by_net": sorted(rows, key=lambda row: row["net_repaired"])[: args.top_k],
            "top_by_changed": sorted(rows, key=lambda row: row["changed"], reverse=True)[: args.top_k],
        }

    output = {
        "args": vars(args),
        "sample_count": len(records),
        "missing_prediction_count": len(missing),
        "missing_predictions": missing[:20],
        "classes": list(CLASSES),
        "regions": regions_out,
        "pairs": pairs_out,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote diagnostics: {out_path}")
    print(f"records={len(records)}, missing={len(missing)}")
    for name, counts in regions_out.items():
        print(
            f"{name}: changed={counts['changed']}, repaired={counts['repaired']}, "
            f"damaged={counts['damaged']}, net={counts['net_repaired']}"
        )
        for row in pairs_out[name]["top_positive_by_net"][:5]:
            print(
                f"  + {row['pair']}: net={row['net_repaired']}, "
                f"repair={row['repaired']}, damage={row['damaged']}, changed={row['changed']}"
            )
        for row in pairs_out[name]["top_negative_by_net"][:5]:
            print(
                f"  - {row['pair']}: net={row['net_repaired']}, "
                f"repair={row['repaired']}, damage={row['damaged']}, changed={row['changed']}"
            )


if __name__ == "__main__":
    main()
