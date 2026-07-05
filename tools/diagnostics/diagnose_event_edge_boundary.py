#!/usr/bin/env python
"""Diagnose event accumulation as an edge cue for semantic boundaries."""

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

from cosec_event_dataset import _event_slice, _h5  # noqa: E402
from cosec_finetune_splits import CLASSES  # noqa: E402
from detectron2.checkpoint import DetectionCheckpointer  # noqa: E402
from detectron2.config import get_cfg  # noqa: E402
from detectron2.data import DatasetCatalog  # noqa: E402
from detectron2.projects.deeplab import add_deeplab_config  # noqa: E402
from mask2former import MaskFormerSemanticDatasetMapper, add_maskformer2_config  # noqa: E402
from train_mask2former_cosec import CoSECTrainer, register_cosec  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-config", default="configs/Mask2Former_SwinL_CoSEC_DayNight_Finetune.yaml")
    parser.add_argument("--rgb-weights", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--time-radii-ms",
        default="25,50,100,200,400",
        help="Centered event accumulation half-window in ms. 50 means [t-50ms,t+50ms].",
    )
    parser.add_argument(
        "--edge-percentiles",
        default="70,80,90,95",
        help="Percentiles over nonzero event-edge scores.",
    )
    parser.add_argument(
        "--smooth-radii",
        default="0,2,4",
        help="Box-average radii applied to event density before thresholding.",
    )
    parser.add_argument("--gt-boundary-radii", default="1,3,5")
    parser.add_argument("--uncertain-percentiles", default="20,30")
    parser.add_argument("--local-average-radii", default="3,5")
    parser.add_argument("--out", required=True)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def parse_int_list(text):
    return [int(part) for part in text.split(",") if part.strip()]


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


def event_density(record, shape, radius_ms):
    height, width = shape
    center_us = int(record["event_old"][1])
    half_window = int(round(float(radius_ms) * 1000.0))
    start_us = center_us - half_window
    end_us = center_us + half_window
    events = _event_slice(_h5(record["event_h5"]), [start_us, end_us])
    density = np.zeros((height, width), dtype=np.float32)
    pos = np.zeros((height, width), dtype=np.float32)
    neg = np.zeros((height, width), dtype=np.float32)
    if events is None or events["x"].size == 0:
        return density, pos, neg

    x = events["x"]
    y = events["y"]
    p = events["p"]
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    if not np.any(valid):
        return density, pos, neg

    x = x[valid]
    y = y[valid]
    p = p[valid]
    np.add.at(density, (y, x), 1.0)
    np.add.at(pos, (y[p > 0], x[p > 0]), 1.0)
    np.add.at(neg, (y[p <= 0], x[p <= 0]), 1.0)
    return density, pos, neg


def event_edge_score(density, pos, neg, smooth_radius):
    score = np.log1p(density).astype(np.float32)
    polarity_total = pos + neg
    if polarity_total.any():
        polarity_balance = 1.0 - np.abs(pos - neg) / (polarity_total + 1e-6)
        # Keep dense event contours but downweight one-polarity isolated noise a little.
        score = score * (0.5 + 0.5 * polarity_balance.astype(np.float32))
    if smooth_radius > 0:
        kernel = 2 * int(smooth_radius) + 1
        score = cv2.blur(score, (kernel, kernel))
    return score


def threshold_event_edge(score, percentile):
    nonzero = score[score > 0]
    if nonzero.size == 0:
        return np.zeros(score.shape, dtype=bool), 0.0
    threshold = float(np.percentile(nonzero, percentile))
    return score >= threshold, threshold


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


def empty_edge_counts(gt_radii):
    counts = {
        "valid_pixels": 0,
        "edge_pixels": 0,
        "rgb_wrong": 0,
        "rgb_wrong_edge": 0,
    }
    for radius in gt_radii:
        counts[f"gt_boundary_r{radius}"] = 0
        counts[f"edge_gt_boundary_r{radius}"] = 0
    return counts


def update_edge_counts(counts, edge, rgb_pred, label, valid, gt_boundaries):
    edge = valid & edge
    rgb_wrong = valid & (rgb_pred != label)
    counts["valid_pixels"] += int(valid.sum())
    counts["edge_pixels"] += int(edge.sum())
    counts["rgb_wrong"] += int(rgb_wrong.sum())
    counts["rgb_wrong_edge"] += int((rgb_wrong & edge).sum())
    for radius, boundary in gt_boundaries.items():
        boundary = valid & boundary
        counts[f"gt_boundary_r{radius}"] += int(boundary.sum())
        counts[f"edge_gt_boundary_r{radius}"] += int((edge & boundary).sum())


def div(num, den):
    return float(num / den) if den else 0.0


def finalize_edge_counts(counts, gt_radii):
    out = {
        **counts,
        "edge_coverage": div(counts["edge_pixels"], counts["valid_pixels"]),
        "rgb_wrong_recall": div(counts["rgb_wrong_edge"], counts["rgb_wrong"]),
        "rgb_error_density_in_edge": div(counts["rgb_wrong_edge"], counts["edge_pixels"]),
    }
    for radius in gt_radii:
        precision = div(counts[f"edge_gt_boundary_r{radius}"], counts["edge_pixels"])
        recall = div(counts[f"edge_gt_boundary_r{radius}"], counts[f"gt_boundary_r{radius}"])
        out[f"precision_gt_boundary_r{radius}"] = precision
        out[f"recall_gt_boundary_r{radius}"] = recall
        out[f"f1_gt_boundary_r{radius}"] = div(2.0 * precision * recall, precision + recall)
    return out


def update_oracle_meter(meter, rgb_pred, label, mask, valid):
    pred = rgb_pred.copy()
    pred[valid & mask] = label[valid & mask]
    meter.update(pred, label)


def update_replacement_meter(meter, rgb_pred, replacement_pred, label, mask, valid):
    pred = rgb_pred.copy()
    pred[valid & mask] = replacement_pred[valid & mask]
    meter.update(pred, label)


def blurred_score_prediction(scores_np, radius):
    if radius <= 0:
        return np.argmax(scores_np, axis=0)
    kernel = 2 * int(radius) + 1
    blurred = np.stack([cv2.blur(scores_np[idx], (kernel, kernel)) for idx in range(scores_np.shape[0])])
    return np.argmax(blurred, axis=0)


def majority_prediction(pred, radius, num_classes):
    if radius <= 0:
        return pred
    kernel = 2 * int(radius) + 1
    votes = np.stack(
        [
            cv2.blur((pred == class_id).astype(np.float32), (kernel, kernel))
            for class_id in range(num_classes)
        ],
        axis=0,
    )
    return np.argmax(votes, axis=0)


def top_metrics(metrics, topk=20):
    rows = [(values["mIoU"], name, values) for name, values in metrics.items()]
    rows.sort(reverse=True, key=lambda item: item[0])
    return [{"method": name, **values} for _, name, values in rows[:topk]]


def best_by_time(edge_summary, metric_name):
    grouped = OrderedDict()
    for name, values in edge_summary.items():
        time_ms = values["time_radius_ms"]
        best = grouped.get(str(time_ms))
        score = values.get(metric_name, 0.0)
        if best is None or score > best[metric_name]:
            grouped[str(time_ms)] = {"edge": name, metric_name: score, **values}
    return grouped


def main():
    args = parse_args()
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    register_cosec()

    time_radii_ms = parse_int_list(args.time_radii_ms)
    edge_percentiles = parse_float_list(args.edge_percentiles)
    smooth_radii = parse_int_list(args.smooth_radii)
    gt_radii = parse_int_list(args.gt_boundary_radii)
    uncertain_percentiles = parse_float_list(args.uncertain_percentiles)
    local_average_radii = parse_int_list(args.local_average_radii)

    cfg = setup_cfg(args.rgb_config, args.rgb_weights, args.device)
    mapper = MaskFormerSemanticDatasetMapper(cfg, False)
    model = build_model(cfg)

    records = DatasetCatalog.get(args.dataset)
    if args.limit is not None:
        records = records[: args.limit]

    base_meter = ConfusionMeter()
    oracle_meters = OrderedDict()
    average_meters = OrderedDict()
    majority_meters = OrderedDict()
    edge_counts = OrderedDict()

    iterator = records if args.quiet else tqdm(records, desc=args.dataset)
    for record in iterator:
        label = load_label(record)
        valid = valid_label_mask(label)
        scores = resize_scores(infer(model, mapper, record), label.shape)
        scores_np = scores.float().numpy()
        rgb_pred = scores_np.argmax(axis=0)
        base_meter.update(rgb_pred, label)

        gt_boundaries = {
            radius: semantic_boundary_band(label, radius, valid)
            for radius in gt_radii
        }
        margin = rgb_margin(scores)
        valid_margin = margin[valid]
        uncertain_masks = OrderedDict()
        for percentile in uncertain_percentiles:
            threshold = np.percentile(valid_margin, percentile) if valid_margin.size else 0.0
            uncertain_masks[f"uncertain_q{percentile:g}"] = valid & (margin <= threshold)

        average_preds = {
            radius: blurred_score_prediction(scores_np, radius)
            for radius in local_average_radii
        }
        majority_preds = {
            radius: majority_prediction(rgb_pred, radius, len(CLASSES))
            for radius in local_average_radii
        }

        for time_ms in time_radii_ms:
            density, pos, neg = event_density(record, label.shape, time_ms)
            for smooth_radius in smooth_radii:
                score = event_edge_score(density, pos, neg, smooth_radius)
                for percentile in edge_percentiles:
                    edge, threshold = threshold_event_edge(score, percentile)
                    edge_name = f"event_t{time_ms}ms_s{smooth_radius}_p{percentile:g}"
                    if edge_name not in edge_counts:
                        edge_counts[edge_name] = empty_edge_counts(gt_radii)
                    update_edge_counts(edge_counts[edge_name], edge, rgb_pred, label, valid, gt_boundaries)

                    candidate_masks = OrderedDict()
                    candidate_masks[f"{edge_name}:edge"] = edge
                    for uncertain_name, uncertain in uncertain_masks.items():
                        candidate_masks[f"{edge_name}:edge_and_{uncertain_name}"] = edge & uncertain
                        candidate_masks[f"{edge_name}:edge_or_{uncertain_name}"] = edge | uncertain

                    for mask_name, mask in candidate_masks.items():
                        if mask_name not in oracle_meters:
                            oracle_meters[mask_name] = ConfusionMeter()
                        update_oracle_meter(oracle_meters[mask_name], rgb_pred, label, mask, valid)

                        for radius, pred in average_preds.items():
                            method_name = f"avg_logits_r{radius}:{mask_name}"
                            if method_name not in average_meters:
                                average_meters[method_name] = ConfusionMeter()
                            update_replacement_meter(
                                average_meters[method_name],
                                rgb_pred,
                                pred,
                                label,
                                mask,
                                valid,
                            )

                        for radius, pred in majority_preds.items():
                            method_name = f"majority_r{radius}:{mask_name}"
                            if method_name not in majority_meters:
                                majority_meters[method_name] = ConfusionMeter()
                            update_replacement_meter(
                                majority_meters[method_name],
                                rgb_pred,
                                pred,
                                label,
                                mask,
                                valid,
                            )

    base_metrics = base_meter.metrics()
    edge_summary = OrderedDict()
    for name, counts in edge_counts.items():
        values = finalize_edge_counts(counts, gt_radii)
        parts = name.split("_")
        values["time_radius_ms"] = int(parts[1][1:-2])
        values["smooth_radius"] = int(parts[2][1:])
        values["edge_percentile"] = float(parts[3][1:])
        edge_summary[name] = values

    oracle_metrics = OrderedDict((name, meter.metrics()) for name, meter in oracle_meters.items())
    average_metrics = OrderedDict((name, meter.metrics()) for name, meter in average_meters.items())
    majority_metrics = OrderedDict((name, meter.metrics()) for name, meter in majority_meters.items())
    output = {
        "args": vars(args),
        "sample_count": len(records),
        "base_metrics": base_metrics,
        "edge_summary": edge_summary,
        "best_edge_by_time_f1_r3": best_by_time(edge_summary, "f1_gt_boundary_r3"),
        "best_edge_by_time_error_density": best_by_time(edge_summary, "rgb_error_density_in_edge"),
        "top_oracle_metrics": top_metrics(oracle_metrics),
        "top_average_metrics": top_metrics(average_metrics),
        "top_majority_metrics": top_metrics(majority_metrics),
        "oracle_metrics": oracle_metrics,
        "average_metrics": average_metrics,
        "majority_metrics": majority_metrics,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote diagnostics: {out_path}")
    print(f"Base RGB: mIoU={base_metrics['mIoU']:.4f}, aAcc={base_metrics['aAcc']:.4f}")
    print("Best edge maps by time window using GT-boundary-r3 F1:")
    for time_ms, values in output["best_edge_by_time_f1_r3"].items():
        print(
            f"  t={time_ms}ms: {values['edge']} "
            f"F1={100*values['f1_gt_boundary_r3']:.2f}%, "
            f"coverage={100*values['edge_coverage']:.2f}%, "
            f"wrong_recall={100*values['rgb_wrong_recall']:.2f}%"
        )
    print("Top oracle event-edge candidate masks:")
    for row in output["top_oracle_metrics"][:8]:
        print(f"  {row['method']}: mIoU={row['mIoU']:.4f}, aAcc={row['aAcc']:.4f}")
    print("Top no-GT average/majority refinements:")
    for row in (output["top_average_metrics"][:4] + output["top_majority_metrics"][:4]):
        print(f"  {row['method']}: mIoU={row['mIoU']:.4f}, aAcc={row['aAcc']:.4f}")


if __name__ == "__main__":
    main()
