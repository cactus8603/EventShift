#!/usr/bin/env python
"""Evaluate pair-whitelisted merges between two saved prediction JSON files."""

import argparse
import copy
import json
import os
import sys
import importlib.util
from collections import OrderedDict
from pathlib import Path

import numpy as np

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
from train_mask2former_cosec import register_cosec  # noqa: E402

from diagnose_pair_transition_from_predictions import (  # noqa: E402
    decode_prediction,
    load_label,
    load_prediction_index,
    prediction_key,
    resize_stat,
    setup_mapper,
    valid_label_mask,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-predictions", required=True)
    parser.add_argument("--new-predictions", required=True)
    parser.add_argument("--pair-diagnostics", required=True)
    parser.add_argument("--event-config", required=True)
    parser.add_argument("--dataset", default="cosec_night_val_event")
    parser.add_argument("--regions", default="all,event_union,raw_event,support")
    parser.add_argument("--top-ks", default="5,10,20,all")
    parser.add_argument("--min-net", type=int, default=1)
    parser.add_argument("--min-precision", type=float, default=0.0)
    parser.add_argument("--min-changed", type=int, default=1)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def split_csv(text):
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_top_ks(text):
    values = []
    for item in split_csv(text):
        values.append(item if item == "all" else int(item))
    return values


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
            "class_iou": {
                CLASSES[idx]: (None if np.isnan(value) else float(100.0 * value))
                for idx, value in enumerate(iou)
            },
        }


def pair_to_ids(pair):
    src, dst = pair.split("->", 1)
    return CLASSES.index(src), CLASSES.index(dst)


def allowed_pairs(pair_diag, region, top_k, min_net, min_precision, min_changed):
    rows = [
        row
        for row in pair_diag["pairs"][region]["top_positive_by_net"]
        if row["net_repaired"] >= min_net
        and row["repair_precision"] >= min_precision
        and row["changed"] >= min_changed
    ]
    if top_k != "all":
        rows = rows[: int(top_k)]
    return {pair_to_ids(row["pair"]) for row in rows}, rows


def make_regions(mapped, label_shape, valid):
    event_stats = mapped["event_stats"].float()
    raw_event = valid & (resize_stat(event_stats, 0, label_shape) > 0)
    support = valid & (resize_stat(event_stats, 3, label_shape) > 0)
    return OrderedDict(
        [
            ("all", valid),
            ("event_union", raw_event | support),
            ("raw_event", raw_event),
            ("support", support),
            ("raw_only", raw_event & ~support),
            ("support_only", support & ~raw_event),
        ]
    )


def pair_mask(base_pred, new_pred, pairs):
    mask = np.zeros(base_pred.shape, dtype=bool)
    for src, dst in pairs:
        mask |= (base_pred == src) & (new_pred == dst)
    return mask


def main():
    args = parse_args()
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    register_cosec()
    mapper = setup_mapper(args.event_config)
    base_index = load_prediction_index(args.base_predictions)
    new_index = load_prediction_index(args.new_predictions)
    pair_diag = json.load(open(args.pair_diagnostics, "r", encoding="utf-8"))

    regions = split_csv(args.regions)
    top_ks = parse_top_ks(args.top_ks)

    meters = OrderedDict()
    meters["base"] = ConfusionMeter(num_classes=len(CLASSES))
    meters["new"] = ConfusionMeter(num_classes=len(CLASSES))
    variant_meta = OrderedDict()
    for region in regions:
        for top_k in top_ks:
            variant = f"{region}_top{top_k}"
            meters[variant] = ConfusionMeter(num_classes=len(CLASSES))
            pairs, rows = allowed_pairs(
                pair_diag,
                region,
                top_k,
                args.min_net,
                args.min_precision,
                args.min_changed,
            )
            variant_meta[variant] = {
                "region": region,
                "top_k": top_k,
                "pair_count": len(pairs),
                "pairs": rows,
                "accepted_pixels": 0,
            }

    records = DatasetCatalog.get(args.dataset)
    missing = []
    for record in records:
        key = prediction_key(record["file_name"])
        base_rows = base_index.get(record["file_name"]) or base_index.get(key)
        new_rows = new_index.get(record["file_name"]) or new_index.get(key)
        if not base_rows or not new_rows:
            missing.append(record["file_name"])
            continue

        label = load_label(record)
        base_pred = decode_prediction(base_rows, label.shape)
        new_pred = decode_prediction(new_rows, label.shape)
        valid = valid_label_mask(label, base_pred, new_pred)
        mapped = mapper(copy.deepcopy(record))
        region_masks = make_regions(mapped, label.shape, valid)

        meters["base"].update(base_pred, label)
        meters["new"].update(new_pred, label)
        for variant, meta in variant_meta.items():
            pairs = {pair_to_ids(row["pair"]) for row in meta["pairs"]}
            accept = region_masks[meta["region"]] & pair_mask(base_pred, new_pred, pairs)
            merged = base_pred.copy()
            merged[accept] = new_pred[accept]
            meta["accepted_pixels"] += int(accept.sum())
            meters[variant].update(merged, label)

    results = OrderedDict()
    for name, meter in meters.items():
        results[name] = meter.metrics()
        if name in variant_meta:
            results[name].update(
                {
                    "accepted_pixels": variant_meta[name]["accepted_pixels"],
                    "pair_count": variant_meta[name]["pair_count"],
                    "region": variant_meta[name]["region"],
                    "top_k": variant_meta[name]["top_k"],
                }
            )

    output = {
        "args": vars(args),
        "sample_count": len(records),
        "missing_prediction_count": len(missing),
        "missing_predictions": missing[:20],
        "results": results,
        "variant_meta": variant_meta,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote merge evaluation: {out_path}")
    print(f"records={len(records)}, missing={len(missing)}")
    for name, values in sorted(results.items(), key=lambda item: item[1]["mIoU"], reverse=True):
        extra = ""
        if name in variant_meta:
            extra = f", pairs={values['pair_count']}, accepted={values['accepted_pixels']}"
        print(
            f"{name}: mIoU={values['mIoU']:.4f}, mAcc={values['mAcc']:.4f}, "
            f"aAcc={values['aAcc']:.4f}{extra}"
        )


if __name__ == "__main__":
    main()
