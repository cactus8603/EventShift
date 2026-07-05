#!/usr/bin/env python
"""Measure how much event-active support overlaps RGB segmentation errors."""

import argparse
import json
import os
import sys
import importlib.util
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
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

DEFAULT_HDF5_PLUGIN_PATH = Path(
    "/path/to/hdf5plugin/plugins"
)
if "HDF5_PLUGIN_PATH" not in os.environ and DEFAULT_HDF5_PLUGIN_PATH.exists():
    os.environ["HDF5_PLUGIN_PATH"] = str(DEFAULT_HDF5_PLUGIN_PATH)

from cosec_event_dataset import load_event_edge_representation  # noqa: E402
from cosec_finetune_splits import CLASSES  # noqa: E402
from detectron2.data import DatasetCatalog  # noqa: E402
from train_mask2former_cosec import register_cosec  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-json", required=True)
    parser.add_argument("--new-json", required=True)
    parser.add_argument("--dataset", default="dsec19_val_event")
    parser.add_argument("--out", required=True)
    parser.add_argument("--ignore-label", type=int, default=255)
    parser.add_argument("--event-radii-ms", nargs="*", type=int, default=[50])
    parser.add_argument("--percentiles", nargs="*", type=float, default=[50, 70, 80, 90])
    parser.add_argument("--dilate-radius", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def load_label(path):
    label = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if label is None:
        raise RuntimeError(f"Could not read label: {path}")
    if label.ndim == 3:
        label = label[:, :, 0]
    return label.astype(np.int64)


def valid_mask(label, ignore_label, num_classes):
    return (label != ignore_label) & (label >= 0) & (label < num_classes)


def load_prediction_json(path):
    grouped = {}
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    for row in rows:
        grouped.setdefault(row["file_name"], []).append(row)
    return grouped


def decode_prediction(rows, shape, fill_value=255):
    pred = np.full(shape, fill_value, dtype=np.int64)
    for row in rows:
        rle = dict(row["segmentation"])
        if isinstance(rle.get("counts"), str):
            rle["counts"] = rle["counts"].encode("ascii")
        mask = mask_util.decode(rle).astype(bool)
        if mask.shape != shape:
            raise RuntimeError(f"Mask shape mismatch: got {mask.shape}, expected {shape}")
        pred[mask] = int(row["category_id"])
    return pred


def event_score(record, image_shape, radii_ms):
    edge = load_event_edge_representation(record, image_shape, radii_ms)
    if edge.size == 0:
        return np.zeros(image_shape[:2], dtype=np.float32)
    edge_score_channels = edge[1::3]
    return np.max(edge_score_channels, axis=0).astype(np.float32)


def make_event_mask(score, valid, percentile, dilate_radius):
    active_values = score[valid & (score > 0)]
    if active_values.size == 0:
        return np.zeros(score.shape, dtype=bool), 0.0
    threshold = float(np.percentile(active_values, percentile))
    mask = valid & (score > threshold)
    if dilate_radius > 0 and mask.any():
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * int(dilate_radius) + 1, 2 * int(dilate_radius) + 1),
        )
        mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool) & valid
    return mask, threshold


def empty_counts():
    return {
        "valid_pixels": 0,
        "event_pixels": 0,
        "base_wrong": 0,
        "new_wrong": 0,
        "changed": 0,
        "repaired": 0,
        "damaged": 0,
        "event_base_wrong": 0,
        "event_new_wrong": 0,
        "event_changed": 0,
        "event_repaired": 0,
        "event_damaged": 0,
        "threshold_sum": 0.0,
        "threshold_count": 0,
    }


def add_counts(counts, valid, event, base_pred, new_pred, label, threshold):
    base_wrong = valid & (base_pred != label)
    new_wrong = valid & (new_pred != label)
    changed = valid & (base_pred != new_pred)
    repaired = base_wrong & (new_pred == label)
    damaged = valid & (base_pred == label) & (new_pred != label)

    counts["valid_pixels"] += int(valid.sum())
    counts["event_pixels"] += int(event.sum())
    counts["base_wrong"] += int(base_wrong.sum())
    counts["new_wrong"] += int(new_wrong.sum())
    counts["changed"] += int(changed.sum())
    counts["repaired"] += int(repaired.sum())
    counts["damaged"] += int(damaged.sum())
    counts["event_base_wrong"] += int((event & base_wrong).sum())
    counts["event_new_wrong"] += int((event & new_wrong).sum())
    counts["event_changed"] += int((event & changed).sum())
    counts["event_repaired"] += int((event & repaired).sum())
    counts["event_damaged"] += int((event & damaged).sum())
    counts["threshold_sum"] += float(threshold)
    counts["threshold_count"] += 1


def div(num, den):
    return float(num / den) if den else 0.0


def finalize_counts(counts):
    event_pixels = counts["event_pixels"]
    base_wrong = counts["base_wrong"]
    repaired = counts["repaired"]
    changed = counts["changed"]
    out = dict(counts)
    out.update(
        {
            "event_coverage_valid": div(event_pixels, counts["valid_pixels"]),
            "event_precision_for_base_error": div(counts["event_base_wrong"], event_pixels),
            "event_recall_of_base_error": div(counts["event_base_wrong"], base_wrong),
            "event_recall_of_repairs": div(counts["event_repaired"], repaired),
            "event_recall_of_changed_pixels": div(counts["event_changed"], changed),
            "event_repair_rate_of_base_wrong": div(counts["event_repaired"], counts["event_base_wrong"]),
            "event_damage_rate": div(counts["event_damaged"], event_pixels),
            "event_net_repaired": int(counts["event_repaired"] - counts["event_damaged"]),
            "global_repair_rate_of_base_wrong": div(repaired, base_wrong),
            "global_changed_rate": div(changed, counts["valid_pixels"]),
            "mean_threshold": div(counts["threshold_sum"], counts["threshold_count"]),
        }
    )
    return out


def main():
    args = parse_args()
    register_cosec()

    base_rows = load_prediction_json(args.base_json)
    new_rows = load_prediction_json(args.new_json)
    records = DatasetCatalog.get(args.dataset)
    if args.limit > 0:
        records = records[: args.limit]

    num_classes = len(CLASSES)
    counts = OrderedDict((str(percentile), empty_counts()) for percentile in args.percentiles)
    missing = []

    for index, record in enumerate(records, 1):
        file_name = record["file_name"]
        if file_name not in base_rows or file_name not in new_rows:
            missing.append(file_name)
            continue
        label = load_label(record["sem_seg_file_name"])
        valid = valid_mask(label, args.ignore_label, num_classes)
        score = event_score(record, label.shape, args.event_radii_ms)
        base_pred = decode_prediction(base_rows[file_name], label.shape, fill_value=args.ignore_label)
        new_pred = decode_prediction(new_rows[file_name], label.shape, fill_value=args.ignore_label)

        for percentile in args.percentiles:
            event, threshold = make_event_mask(score, valid, percentile, args.dilate_radius)
            add_counts(counts[str(percentile)], valid, event, base_pred, new_pred, label, threshold)

        if index % 250 == 0:
            print(f"processed {index}/{len(records)}", flush=True)

    output = {
        "args": vars(args),
        "records": len(records),
        "evaluated_records": len(records) - len(missing),
        "missing_records": missing[:20],
        "classes": CLASSES,
        "percentiles": OrderedDict((key, finalize_counts(value)) for key, value in counts.items()),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote {out_path}")
    for percentile, metrics in output["percentiles"].items():
        print(
            "p{p}: event_valid={ev:.2%} error_precision={prec:.2%} "
            "error_recall={rec:.2%} repair_recall={rr:.2%} "
            "changed_recall={cr:.2%} event_net={net}".format(
                p=percentile,
                ev=metrics["event_coverage_valid"],
                prec=metrics["event_precision_for_base_error"],
                rec=metrics["event_recall_of_base_error"],
                rr=metrics["event_recall_of_repairs"],
                cr=metrics["event_recall_of_changed_pixels"],
                net=metrics["event_net_repaired"],
            )
        )


if __name__ == "__main__":
    main()
