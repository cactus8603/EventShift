#!/usr/bin/env python
"""Scan temporal/spatial event calibration against semantic boundaries.

The event representation should be time-calibrated before being used as an edge
cue. This diagnostic searches event time offsets and small spatial shifts, then
scores event/RGB edge maps against segmentation GT boundaries on the validation
split.
"""

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

def _eventshift_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "configs").is_dir() and (parent / "third_party").is_dir():
            return parent
    return Path(__file__).resolve().parents[1]


ROOT = _eventshift_root()
sys.path.insert(0, str(ROOT / "tools"))

from cosec_event_dataset import load_cosec_event_dicts, load_event_edge_representation  # noqa: E402
from ensemble_feature_cache_common import resize_if_needed, safe_name, valid_label_mask  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        required=True,
        help="CoSEC event split, e.g. day_val, night_val, train, or cosec_day_val_event.",
    )
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--time-offsets-ms", default="-50,-25,-10,0,10,25,50")
    parser.add_argument("--window-radii-ms", default="25,50,100")
    parser.add_argument("--spatial-shifts", default="-6,0,6")
    parser.add_argument("--edge-percentiles", default="70,80,90")
    parser.add_argument("--gt-boundary-radius", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def parse_float_list(text):
    return [float(part) for part in str(text).split(",") if part.strip()]


def parse_int_list(text):
    return [int(part) for part in str(text).split(",") if part.strip()]


def normalize_split_name(text):
    text = str(text)
    if text.startswith("cosec_") and text.endswith("_event"):
        text = text[len("cosec_") : -len("_event")]
    return text


def load_label(path):
    label = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if label is None:
        raise RuntimeError(f"Could not read label: {path}")
    if label.ndim == 3:
        label = label[:, :, 0]
    return label.astype(np.int64, copy=False)


def semantic_boundary_band(label, radius, valid):
    if radius <= 0:
        return np.zeros(label.shape, dtype=bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    low = label.astype(np.float32, copy=True)
    high = label.astype(np.float32, copy=True)
    low[~valid] = -1000.0
    high[~valid] = 1000.0
    local_max = cv2.dilate(low, kernel)
    local_min = cv2.erode(high, kernel)
    return valid & (local_max != local_min)


def normalize_score(score):
    score = score.astype(np.float32, copy=True)
    score[~np.isfinite(score)] = 0
    max_value = float(score.max())
    if max_value > 1e-6:
        score /= max_value
    return score


def rgb_edge_score(image_path, shape):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    image = resize_if_needed(image, shape, cv2.INTER_LINEAR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return normalize_score(cv2.magnitude(gx, gy))


def event_edge_score(record, shape, window_radii_ms, time_offset_ms):
    channels = load_event_edge_representation(
        record,
        shape,
        window_radii_ms,
        time_offset_ms=float(time_offset_ms),
    )
    if channels.shape[0] == 0:
        return np.zeros(shape, dtype=np.float32)
    edge_channels = channels[1::3]
    if edge_channels.shape[0] == 0:
        edge_channels = channels
    return normalize_score(edge_channels.max(axis=0))


def shift_image(array, dx=0, dy=0, fill_value=0):
    dx = int(dx)
    dy = int(dy)
    if dx == 0 and dy == 0:
        return array
    shifted = np.full_like(array, fill_value)
    height, width = array.shape[:2]
    src_x0 = max(0, -dx)
    src_x1 = min(width, width - dx)
    dst_x0 = max(0, dx)
    dst_x1 = min(width, width + dx)
    src_y0 = max(0, -dy)
    src_y1 = min(height, height - dy)
    dst_y0 = max(0, dy)
    dst_y1 = min(height, height + dy)
    if src_x1 > src_x0 and src_y1 > src_y0:
        shifted[dst_y0:dst_y1, dst_x0:dst_x1] = array[src_y0:src_y1, src_x0:src_x1]
    return shifted


def threshold_score(score, percentile):
    nonzero = score[score > 0]
    if nonzero.size == 0:
        return np.zeros(score.shape, dtype=bool), 0.0
    threshold = float(np.percentile(nonzero, float(percentile)))
    return score >= threshold, threshold


def new_counts():
    return {
        "tp": 0,
        "pred": 0,
        "gt": 0,
        "valid": 0,
        "threshold_sum": 0.0,
        "threshold_count": 0,
    }


def update_counts(counts, edge, gt_boundary, valid, threshold):
    edge = edge & valid
    gt_boundary = gt_boundary & valid
    counts["tp"] += int((edge & gt_boundary).sum())
    counts["pred"] += int(edge.sum())
    counts["gt"] += int(gt_boundary.sum())
    counts["valid"] += int(valid.sum())
    counts["threshold_sum"] += float(threshold)
    counts["threshold_count"] += 1


def finalize_counts(counts):
    tp = int(counts["tp"])
    pred = int(counts["pred"])
    gt = int(counts["gt"])
    precision = tp / max(pred, 1)
    recall = tp / max(gt, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "tp": tp,
        "pred": pred,
        "gt": gt,
        "edge_coverage": float(pred / max(int(counts["valid"]), 1)),
        "mean_threshold": float(counts["threshold_sum"] / max(int(counts["threshold_count"]), 1)),
    }


def main():
    args = parse_args()
    split = normalize_split_name(args.split)
    records = list(load_cosec_event_dicts(split))
    if args.limit is not None:
        records = records[: int(args.limit)]
    if not records:
        raise RuntimeError(f"No event records found for split={split}")

    time_offsets_ms = parse_float_list(args.time_offsets_ms)
    radii_ms = parse_int_list(args.window_radii_ms)
    shift_values = parse_int_list(args.spatial_shifts)
    edge_percentiles = parse_float_list(args.edge_percentiles)
    shift_pairs = [(dx, dy) for dy in shift_values for dx in shift_values]

    counts = OrderedDict()
    rgb_counts = OrderedDict()

    for idx, record in enumerate(records):
        label = load_label(record["sem_seg_file_name"])
        valid = valid_label_mask(label)
        gt_boundary = semantic_boundary_band(label, args.gt_boundary_radius, valid)
        rgb_score = rgb_edge_score(record["file_name"], label.shape)

        for percentile in edge_percentiles:
            rgb_mask, rgb_threshold = threshold_score(rgb_score, percentile)
            key = f"rgb_p{percentile:g}"
            if key not in rgb_counts:
                rgb_counts[key] = new_counts()
            update_counts(rgb_counts[key], rgb_mask, gt_boundary, valid, rgb_threshold)

        for offset_ms in time_offsets_ms:
            base_event = event_edge_score(record, label.shape, radii_ms, offset_ms)
            for dx, dy in shift_pairs:
                shifted_event = shift_image(base_event, dx=dx, dy=dy)
                combined_max = np.maximum(shifted_event, rgb_score)
                combined_avg = normalize_score(0.5 * shifted_event + 0.5 * rgb_score)
                score_items = [
                    ("event", shifted_event),
                    ("event_rgb_max", combined_max),
                    ("event_rgb_avg", combined_avg),
                ]
                for score_name, score in score_items:
                    for percentile in edge_percentiles:
                        edge, threshold = threshold_score(score, percentile)
                        key = (
                            f"{score_name}_dt{offset_ms:g}_dx{dx}_dy{dy}_"
                            f"r{'-'.join(str(v) for v in radii_ms)}_p{percentile:g}"
                        )
                        if key not in counts:
                            counts[key] = new_counts()
                        update_counts(counts[key], edge, gt_boundary, valid, threshold)

        if not args.quiet and (idx + 1) % 20 == 0:
            print(f"processed {idx + 1}/{len(records)}")

    results = []
    for key, value in counts.items():
        summary = finalize_counts(value)
        parts = key.split("_")
        summary.update({"name": key})
        results.append(summary)
    results.sort(key=lambda item: item["f1"], reverse=True)

    rgb_results = [{"name": key, **finalize_counts(value)} for key, value in rgb_counts.items()]
    rgb_results.sort(key=lambda item: item["f1"], reverse=True)

    best_event = next((item for item in results if item["name"].startswith("event_dt")), None)
    best_event_rgb = next((item for item in results if item["name"].startswith("event_rgb")), None)

    payload = {
        "split": split,
        "sample_count": len(records),
        "time_offsets_ms": time_offsets_ms,
        "window_radii_ms": radii_ms,
        "spatial_shifts": shift_values,
        "edge_percentiles": edge_percentiles,
        "gt_boundary_radius": int(args.gt_boundary_radius),
        "best_rgb": rgb_results[0] if rgb_results else None,
        "best_event": best_event,
        "best_event_rgb": best_event_rgb,
        "top_rgb": rgb_results[:10],
        "top_results": results[:30],
        "results": results,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"split={split} samples={len(records)}")
    if rgb_results:
        print(
            f"best_rgb={rgb_results[0]['name']} "
            f"F1={100.0 * rgb_results[0]['f1']:.2f} "
            f"P={100.0 * rgb_results[0]['precision']:.2f} "
            f"R={100.0 * rgb_results[0]['recall']:.2f}"
        )
    for label, row in [("best_event", best_event), ("best_event_rgb", best_event_rgb)]:
        if row:
            print(
                f"{label}={row['name']} "
                f"F1={100.0 * row['f1']:.2f} "
                f"P={100.0 * row['precision']:.2f} "
                f"R={100.0 * row['recall']:.2f} "
                f"coverage={100.0 * row['edge_coverage']:.2f}"
            )
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
