#!/usr/bin/env python
"""Diagnose event-edge constrained candidate-class routes on cached predictions.

This evaluates post-processing ideas without re-running model inference:

* route_all: paste candidate classes wherever the candidate predicts the routed class.
* event masks: only paste routed classes where event edge/density is active.
* boundary/uncertainty masks: optionally restrict the paste to RGB semantic
  boundaries and low-margin RGB pixels.
* guided filter: use event edge score as guidance to smooth/snap the candidate
  acceptance mask.

The script is intentionally diagnostic. Cached maps only store pred/conf/margin
instead of full class probabilities, so guided filtering is applied to the
candidate acceptance mask, not to the full probability volume.
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from ensemble_feature_cache_common import (
    CLASSES,
    SegmentationStats,
    load_label,
    resize_if_needed,
    safe_name,
    valid_label_mask,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", default="work_dirs/ensemble_feature_cache")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--event-dataset", required=True)
    parser.add_argument("--anchor-model", required=True)
    parser.add_argument("--route-json", required=True)
    parser.add_argument("--event-edge-cache-dir", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument(
        "--candidate-model",
        action="append",
        default=[],
        help="Candidate mapping in the form route_name=cache_model_name. Repeatable.",
    )
    parser.add_argument("--event-score-thresholds", default="0.0,0.15,0.25,0.35")
    parser.add_argument("--event-dilate-radii", default="0,2,4")
    parser.add_argument("--boundary-radii", default="2,4,8")
    parser.add_argument("--uncertain-percentiles", default="20,30")
    parser.add_argument("--guided-radii", default="4,8")
    parser.add_argument("--guided-eps", type=float, default=1e-3)
    parser.add_argument("--guided-thresholds", default="0.25,0.5")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def parse_mapping(items):
    parsed = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--candidate-model must be route_name=cache_model_name, got: {item}")
        key, value = item.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def parse_float_list(text):
    return [float(part) for part in text.split(",") if part.strip()]


def parse_int_list(text):
    return [int(part) for part in text.split(",") if part.strip()]


def read_rows(cache_root, model, dataset, limit=None):
    path = cache_root / model / f"{dataset}_per_image.csv"
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if limit is not None:
        rows = rows[: int(limit)]
    return rows


def map_paths(cache_root, model, dataset, limit=None):
    paths = sorted((cache_root / model / "maps" / dataset).glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No cached maps found for {model}:{dataset}")
    if limit is not None:
        paths = paths[: int(limit)]
    return paths


def load_map(path, shape):
    payload = np.load(path)
    pred = payload["pred"].astype(np.uint8, copy=False)
    conf = payload["conf"].astype(np.float32, copy=False)
    margin = payload["margin"].astype(np.float32, copy=False)
    entropy = payload["entropy"].astype(np.float32, copy=False)
    return {
        "pred": resize_if_needed(pred, shape, cv2.INTER_NEAREST).astype(np.uint8, copy=False),
        "conf": resize_if_needed(conf, shape, cv2.INTER_LINEAR).astype(np.float32, copy=False),
        "margin": resize_if_needed(margin, shape, cv2.INTER_LINEAR).astype(np.float32, copy=False),
        "entropy": resize_if_needed(entropy, shape, cv2.INTER_LINEAR).astype(np.float32, copy=False),
    }


def load_event_edge(event_root, event_dataset, image_id, shape):
    path = event_root / event_dataset / f"{safe_name(image_id)}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing event edge cache for {image_id}: {path}")
    payload = np.load(path)
    score = payload["score"].astype(np.float32, copy=False)
    if "mask" in payload:
        mask = payload["mask"].astype(np.uint8, copy=False) > 0
    else:
        mask = score > 0
    score = resize_if_needed(score, shape, cv2.INTER_LINEAR).astype(np.float32, copy=False)
    mask = resize_if_needed(mask.astype(np.uint8), shape, cv2.INTER_NEAREST).astype(bool, copy=False)
    return score, mask


def semantic_boundary_band(pred, radius):
    if radius <= 0:
        return np.zeros(pred.shape, dtype=bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    values = pred.astype(np.float32, copy=False)
    local_max = cv2.dilate(values, kernel)
    local_min = cv2.erode(values, kernel)
    return local_max != local_min


def dilate_bool(mask, radius):
    if radius <= 0:
        return mask.astype(bool, copy=False)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def guided_filter_gray(guide, src, radius, eps):
    """Single-channel guided filter using boxFilter means."""
    guide = guide.astype(np.float32, copy=False)
    src = src.astype(np.float32, copy=False)
    ksize = (2 * int(radius) + 1, 2 * int(radius) + 1)
    mean_i = cv2.boxFilter(guide, -1, ksize, normalize=True, borderType=cv2.BORDER_REFLECT)
    mean_p = cv2.boxFilter(src, -1, ksize, normalize=True, borderType=cv2.BORDER_REFLECT)
    corr_i = cv2.boxFilter(guide * guide, -1, ksize, normalize=True, borderType=cv2.BORDER_REFLECT)
    corr_ip = cv2.boxFilter(guide * src, -1, ksize, normalize=True, borderType=cv2.BORDER_REFLECT)
    var_i = corr_i - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p
    a = cov_ip / (var_i + float(eps))
    b = mean_p - a * mean_i
    mean_a = cv2.boxFilter(a, -1, ksize, normalize=True, borderType=cv2.BORDER_REFLECT)
    mean_b = cv2.boxFilter(b, -1, ksize, normalize=True, borderType=cv2.BORDER_REFLECT)
    return mean_a * guide + mean_b


def build_variants(event_thresholds, event_dilate_radii, boundary_radii, uncertain_percentiles, guided_radii, guided_thresholds):
    variants = [{"name": "anchor", "kind": "anchor"}, {"name": "route_all", "kind": "route"}]
    variants.append({"name": "event_cache_mask", "kind": "route", "event_mode": "cache", "event_dilate": 0})
    for radius in event_dilate_radii:
        if radius > 0:
            variants.append(
                {
                    "name": f"event_cache_mask_dil{radius}",
                    "kind": "route",
                    "event_mode": "cache",
                    "event_dilate": radius,
                }
            )
    for threshold in event_thresholds:
        variants.append(
            {
                "name": f"event_score_ge_{threshold:g}",
                "kind": "route",
                "event_mode": "score",
                "event_threshold": threshold,
                "event_dilate": 0,
            }
        )
    for boundary_radius in boundary_radii:
        variants.append(
            {
                "name": f"boundary{boundary_radius}",
                "kind": "route",
                "boundary_radius": boundary_radius,
            }
        )
        variants.append(
            {
                "name": f"event_cache_boundary{boundary_radius}",
                "kind": "route",
                "event_mode": "cache",
                "event_dilate": 0,
                "boundary_radius": boundary_radius,
            }
        )
        variants.append(
            {
                "name": f"event_cache_disagree{boundary_radius}",
                "kind": "route",
                "event_mode": "cache",
                "event_dilate": 0,
                "disagree_radius": boundary_radius,
            }
        )
        for q in uncertain_percentiles:
            variants.append(
                {
                    "name": f"event_cache_boundary{boundary_radius}_uncertain_q{q}",
                    "kind": "route",
                    "event_mode": "cache",
                    "event_dilate": 0,
                    "boundary_radius": boundary_radius,
                    "uncertain_percentile": q,
                }
            )
            variants.append(
                {
                    "name": f"event_any_boundary{boundary_radius}_uncertain_q{q}",
                    "kind": "route",
                    "event_mode": "score",
                    "event_threshold": 0.0,
                    "event_dilate": 0,
                    "boundary_radius": boundary_radius,
                    "uncertain_percentile": q,
                }
            )
    for radius in guided_radii:
        for threshold in guided_thresholds:
            variants.append(
                {
                    "name": f"guided_event_r{radius}_t{threshold:g}",
                    "kind": "guided",
                    "guided_radius": radius,
                    "guided_threshold": threshold,
                }
            )
            variants.append(
                {
                    "name": f"guided_event_r{radius}_t{threshold:g}_event_cache",
                    "kind": "guided",
                    "guided_radius": radius,
                    "guided_threshold": threshold,
                    "event_mode": "cache",
                    "event_dilate": 0,
                }
            )
    seen = set()
    unique = []
    for variant in variants:
        if variant["name"] in seen:
            continue
        seen.add(variant["name"])
        unique.append(variant)
    return unique


def make_event_mask(variant, event_score, event_cache_mask):
    mode = variant.get("event_mode")
    if not mode:
        return None
    if mode == "cache":
        mask = event_cache_mask.copy()
    elif mode == "score":
        mask = event_score > float(variant.get("event_threshold", 0.0))
    else:
        raise ValueError(f"Unknown event_mode: {mode}")
    return dilate_bool(mask, int(variant.get("event_dilate", 0)))


def class_route_prediction(anchor_pred, candidate_maps, routes, variant, event_score, event_cache_mask, anchor_stats, valid):
    if variant["kind"] == "anchor":
        return anchor_pred

    event_mask = make_event_mask(variant, event_score, event_cache_mask)
    boundary_radius = int(variant.get("boundary_radius", 0))
    disagree_radius = int(variant.get("disagree_radius", 0))
    uncertainty_q = variant.get("uncertain_percentile")
    uncertain_mask = None
    if uncertainty_q is not None:
        margin = anchor_stats["margin"]
        threshold = float(np.percentile(margin[valid], float(uncertainty_q))) if np.any(valid) else 0.0
        uncertain_mask = margin <= threshold

    anchor_boundary = None
    if boundary_radius > 0:
        anchor_boundary = semantic_boundary_band(anchor_pred, boundary_radius)

    guided_accept = None
    if variant["kind"] == "guided":
        route_seed = np.zeros(anchor_pred.shape, dtype=np.float32)
        for route in routes:
            candidate_pred = candidate_maps[route["candidate"]]["pred"]
            class_id = int(route["class_id"])
            route_seed[candidate_pred == class_id] = 1.0
        guide = np.clip(event_score, 0.0, 1.0)
        guided_accept = guided_filter_gray(
            guide,
            route_seed,
            int(variant.get("guided_radius", 4)),
            float(variant.get("guided_eps", 1e-3)),
        )
        guided_accept = guided_accept >= float(variant.get("guided_threshold", 0.5))

    merged = anchor_pred.copy()
    for route in routes:
        candidate = candidate_maps[route["candidate"]]["pred"]
        class_id = int(route["class_id"])
        take = candidate == class_id
        if variant["kind"] == "guided":
            take &= guided_accept
        if event_mask is not None:
            take &= event_mask
        if anchor_boundary is not None:
            take &= anchor_boundary | semantic_boundary_band(candidate, boundary_radius)
        if disagree_radius > 0:
            take &= dilate_bool(anchor_pred != candidate, disagree_radius)
        if uncertain_mask is not None:
            take &= uncertain_mask
        merged[take] = class_id
    return merged


def empty_counts():
    return {
        "changed_pixels": 0,
        "candidate_claimed_pixels": 0,
        "repaired_pixels": 0,
        "damaged_pixels": 0,
        "event_active_pixels": 0,
        "rgb_wrong_pixels": 0,
        "rgb_wrong_event_pixels": 0,
        "repaired_event_pixels": 0,
        "damaged_event_pixels": 0,
        "valid_pixels": 0,
        "total_pixels": 0,
    }


def update_counts(counts, anchor_pred, merged_pred, candidate_maps, routes, label, valid, event_cache_mask):
    changed = merged_pred != anchor_pred
    anchor_correct = valid & (anchor_pred == label)
    anchor_wrong = valid & ~anchor_correct
    merged_correct = valid & (merged_pred == label)
    repaired = anchor_wrong & merged_correct
    damaged = anchor_correct & ~merged_correct
    candidate_claimed = np.zeros(anchor_pred.shape, dtype=bool)
    for route in routes:
        candidate_pred = candidate_maps[route["candidate"]]["pred"]
        candidate_claimed |= candidate_pred == int(route["class_id"])

    counts["changed_pixels"] += int((changed & valid).sum())
    counts["candidate_claimed_pixels"] += int((candidate_claimed & valid).sum())
    counts["repaired_pixels"] += int(repaired.sum())
    counts["damaged_pixels"] += int(damaged.sum())
    counts["event_active_pixels"] += int((event_cache_mask & valid).sum())
    counts["rgb_wrong_pixels"] += int(anchor_wrong.sum())
    counts["rgb_wrong_event_pixels"] += int((anchor_wrong & event_cache_mask).sum())
    counts["repaired_event_pixels"] += int((repaired & event_cache_mask).sum())
    counts["damaged_event_pixels"] += int((damaged & event_cache_mask).sum())
    counts["valid_pixels"] += int(valid.sum())
    counts["total_pixels"] += int(valid.size)


def summarize_variant(name, variant, meter, anchor_metrics, counts, class_iou_anchor):
    metrics = meter.metrics()
    class_iou = metrics["iou"]
    per_class_delta = []
    for class_id, class_name in enumerate(CLASSES):
        current = class_iou[class_id]
        base = class_iou_anchor[class_id]
        if np.isnan(current) or np.isnan(base):
            delta = None
        else:
            delta = float(100.0 * (current - base))
        per_class_delta.append({"class_id": class_id, "class_name": class_name, "delta_iou": delta})
    valid_pixels = max(int(counts["valid_pixels"]), 1)
    changed = int(counts["changed_pixels"])
    repaired = int(counts["repaired_pixels"])
    damaged = int(counts["damaged_pixels"])
    event_active = int(counts["event_active_pixels"])
    rgb_wrong_event = int(counts["rgb_wrong_event_pixels"])
    return {
        "name": name,
        "spec": variant,
        "mIoU": metrics["mIoU"],
        "mAcc": metrics["mAcc"],
        "aAcc": metrics["aAcc"],
        "delta_mIoU": metrics["mIoU"] - anchor_metrics["mIoU"],
        "changed_pixels": changed,
        "changed_percent_valid": 100.0 * changed / valid_pixels,
        "repaired_pixels": repaired,
        "damaged_pixels": damaged,
        "net_repaired_pixels": repaired - damaged,
        "repair_damage_ratio": float(repaired / max(damaged, 1)),
        "event_active_pixels": event_active,
        "event_active_percent_valid": 100.0 * event_active / valid_pixels,
        "rgb_wrong_pixels": int(counts["rgb_wrong_pixels"]),
        "rgb_wrong_event_pixels": rgb_wrong_event,
        "rgb_wrong_event_percent_wrong": 100.0 * rgb_wrong_event / max(int(counts["rgb_wrong_pixels"]), 1),
        "repaired_event_pixels": int(counts["repaired_event_pixels"]),
        "damaged_event_pixels": int(counts["damaged_event_pixels"]),
        "event_repair_rate_on_rgb_wrong_event": 100.0 * int(counts["repaired_event_pixels"]) / max(rgb_wrong_event, 1),
        "per_class_delta_iou": per_class_delta,
    }


def main():
    args = parse_args()
    cache_root = Path(args.cache_root)
    event_root = Path(args.event_edge_cache_dir)
    candidates = parse_mapping(args.candidate_model)
    route_payload = json.load(open(args.route_json, "r", encoding="utf-8"))
    routes = route_payload["routes"]
    missing = sorted({route["candidate"] for route in routes} - set(candidates))
    if missing:
        raise ValueError(f"Missing --candidate-model entries for route names: {missing}")

    rows = read_rows(cache_root, args.anchor_model, args.dataset, args.limit)
    anchor_paths = map_paths(cache_root, args.anchor_model, args.dataset, args.limit)
    candidate_paths = {
        name: map_paths(cache_root, model, args.dataset, args.limit)
        for name, model in candidates.items()
    }
    if len(rows) != len(anchor_paths):
        raise ValueError(f"Count mismatch: rows={len(rows)} anchor_maps={len(anchor_paths)}")
    for name, paths in candidate_paths.items():
        if len(paths) != len(anchor_paths):
            raise ValueError(f"Count mismatch: {name} maps={len(paths)} anchor_maps={len(anchor_paths)}")

    variants = build_variants(
        parse_float_list(args.event_score_thresholds),
        parse_int_list(args.event_dilate_radii),
        parse_int_list(args.boundary_radii),
        parse_float_list(args.uncertain_percentiles),
        parse_int_list(args.guided_radii),
        parse_float_list(args.guided_thresholds),
    )
    meters = {variant["name"]: SegmentationStats() for variant in variants}
    counts = {variant["name"]: empty_counts() for variant in variants}

    for index, row in enumerate(rows):
        label = load_label(row["label_path"])
        valid = valid_label_mask(label)
        anchor_stats = load_map(anchor_paths[index], label.shape)
        anchor_pred = anchor_stats["pred"]
        candidate_maps = {
            name: load_map(paths[index], label.shape)
            for name, paths in candidate_paths.items()
        }
        event_score, event_cache_mask = load_event_edge(event_root, args.event_dataset, row["image_id"], label.shape)
        for variant in variants:
            merged = class_route_prediction(
                anchor_pred,
                candidate_maps,
                routes,
                variant,
                event_score,
                event_cache_mask,
                anchor_stats,
                valid,
            )
            name = variant["name"]
            meters[name].update(merged, label)
            update_counts(counts[name], anchor_pred, merged, candidate_maps, routes, label, valid, event_cache_mask)
        if (index + 1) % 25 == 0:
            print(f"processed {index + 1}/{len(rows)}")

    anchor_metrics = meters["anchor"].metrics()
    class_iou_anchor = anchor_metrics["iou"].copy()
    results = [
        summarize_variant(variant["name"], variant, meters[variant["name"]], anchor_metrics, counts[variant["name"]], class_iou_anchor)
        for variant in variants
    ]
    ranked = sorted(results, key=lambda item: item["mIoU"], reverse=True)
    payload = {
        "dataset": args.dataset,
        "event_dataset": args.event_dataset,
        "anchor_model": args.anchor_model,
        "route_json": str(Path(args.route_json).resolve()),
        "candidate_models": candidates,
        "event_edge_cache_dir": str(event_root.resolve()),
        "num_images": len(rows),
        "anchor": next(item for item in results if item["name"] == "anchor"),
        "ranked": ranked,
        "results": results,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"dataset={args.dataset} images={len(rows)}")
    print(f"anchor_mIoU={anchor_metrics['mIoU']:.4f}")
    for row in ranked[:12]:
        print(
            f"{row['name']:45s} "
            f"mIoU={row['mIoU']:.4f} delta={row['delta_mIoU']:+.4f} "
            f"changed={row['changed_percent_valid']:.4f}% "
            f"repair={row['repaired_pixels']} damage={row['damaged_pixels']}"
        )
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
