#!/usr/bin/env python
"""Evaluate lightweight CRF-style post-processing on cached segmentation maps.

The cache used by our current Swin-L diagnostics stores hard predictions plus
confidence summaries, not full 19-class logits. This script therefore evaluates
conservative CRF-style refinements that preserve the cached prediction as a
strong unary and use local/guided class support as the pairwise term.
"""

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from tqdm import tqdm

def _eventshift_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "configs").is_dir() and (parent / "third_party").is_dir():
            return parent
    return Path(__file__).resolve().parents[1]


ROOT = _eventshift_root()
sys.path.insert(0, str(ROOT / "tools"))

from ensemble_feature_cache_common import (  # noqa: E402
    CLASS_COUNT,
    CLASSES,
    SegmentationStats,
    load_label,
    resize_if_needed,
    safe_name,
)


DEFAULT_SPECS = (
    "cosec_day=swinl_day65_4352:cosec_day_val",
    "cosec_night=swinl_night50_4284:cosec_night_val",
    "acdc_night=swinl_acdc54_754_proxy:acdc_night_val",
)


@dataclass(frozen=True)
class Variant:
    name: str
    mode: str
    unary_base: float = 1.0
    unary_margin: float = 0.0
    pairwise_weight: float = 1.0
    radius: int = 3
    sigma: float = 1.2
    iterations: int = 1
    boundary_radius: int = 0
    uncertain_percentile: float = 100.0
    guided_eps: float = 1e-3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", default=str(ROOT / "work_dirs/ensemble_feature_cache"))
    parser.add_argument("--out-json", default=str(ROOT / "work_dirs/diagnostics/crf_postprocess.json"))
    parser.add_argument("--spec", action="append", default=[], help="name=model:dataset")
    parser.add_argument("--json-spec", action="append", default=[], help="name=/path/to/sem_seg_predictions.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--preset", choices=("quick", "final", "selected", "tune"), default="quick")
    parser.add_argument("--save-per-class", action="store_true")
    return parser.parse_args()


def parse_specs(items):
    specs = {}
    for item in items or DEFAULT_SPECS:
        if "=" not in item or ":" not in item:
            raise ValueError(f"--spec must be name=model:dataset, got: {item}")
        name, rest = item.split("=", 1)
        model, dataset = rest.split(":", 1)
        specs[name.strip()] = {"model": model.strip(), "dataset": dataset.strip()}
    return specs


def parse_json_specs(items):
    specs = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--json-spec must be name=/path/to/sem_seg_predictions.json, got: {item}")
        name, path = item.split("=", 1)
        specs[name.strip()] = Path(path.strip())
    return specs


def variants_for_preset(preset):
    quick = [
        Variant("spatial_u1.5_w0.6_r3_i1", "spatial", unary_base=1.5, pairwise_weight=0.6, radius=3),
        Variant("spatial_u1.0_w0.8_r3_i1_boundary2_q40", "spatial", unary_base=1.0, pairwise_weight=0.8, radius=3, boundary_radius=2, uncertain_percentile=40),
        Variant("spatial_u0.5_m1.5_w1.0_r5_i1_boundary2_q40", "spatial", unary_base=0.5, unary_margin=1.5, pairwise_weight=1.0, radius=5, sigma=1.8, boundary_radius=2, uncertain_percentile=40),
        Variant("guided_u1.0_w0.8_r4_i1_boundary2_q40", "guided", unary_base=1.0, pairwise_weight=0.8, radius=4, boundary_radius=2, uncertain_percentile=40),
    ]
    if preset == "quick":
        return quick
    final = [
        Variant("spatial_u1.5_w0.6_r3_i1", "spatial", unary_base=1.5, pairwise_weight=0.6, radius=3),
        Variant("spatial_u1.0_w0.8_r3_i1_boundary2_q40", "spatial", unary_base=1.0, pairwise_weight=0.8, radius=3, boundary_radius=2, uncertain_percentile=40),
        Variant("guided_u1.0_w0.8_r4_i1_boundary2_q40", "guided", unary_base=1.0, pairwise_weight=0.8, radius=4, boundary_radius=2, uncertain_percentile=40),
    ]
    if preset == "final":
        return final
    selected = [
        Variant("guided_u0.5_m1.5_w1.0_r6_i1_boundary2_q50", "guided", unary_base=0.5, unary_margin=1.5, pairwise_weight=1.0, radius=6, boundary_radius=2, uncertain_percentile=50),
        Variant("spatial_u0.25_m1.75_w1.2_r5_i1_boundary4_q50", "spatial", unary_base=0.25, unary_margin=1.75, pairwise_weight=1.2, radius=5, sigma=1.8, boundary_radius=4, uncertain_percentile=50),
        Variant("spatial_u0.5_m1.5_w1.0_r5_i1_boundary2_q40", "spatial", unary_base=0.5, unary_margin=1.5, pairwise_weight=1.0, radius=5, sigma=1.8, boundary_radius=2, uncertain_percentile=40),
    ]
    if preset == "selected":
        return selected
    return quick + [
        Variant("spatial_u1.0_w1.2_r5_i1", "spatial", unary_base=1.0, pairwise_weight=1.2, radius=5, sigma=1.8),
        Variant("spatial_u1.0_w1.0_r5_i2_boundary2_q50", "spatial", unary_base=1.0, pairwise_weight=1.0, radius=5, sigma=1.8, iterations=2, boundary_radius=2, uncertain_percentile=50),
        Variant("spatial_u0.25_m1.75_w1.2_r5_i1_boundary4_q50", "spatial", unary_base=0.25, unary_margin=1.75, pairwise_weight=1.2, radius=5, sigma=1.8, boundary_radius=4, uncertain_percentile=50),
        Variant("guided_u0.5_m1.5_w1.0_r6_i1_boundary2_q50", "guided", unary_base=0.5, unary_margin=1.5, pairwise_weight=1.0, radius=6, boundary_radius=2, uncertain_percentile=50),
    ]


def read_rows(cache_root, model, dataset, limit=None):
    path = cache_root / model / f"{dataset}_per_image.csv"
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if limit is not None:
        rows = rows[: int(limit)]
    return rows


def map_paths(cache_root, model, dataset, limit=None):
    paths = sorted((cache_root / model / "maps" / dataset).glob("*.npz"))
    if limit is not None:
        paths = paths[: int(limit)]
    if not paths:
        raise FileNotFoundError(f"No maps found for {model}:{dataset}")
    return paths


def load_cached_map(path, shape):
    payload = np.load(path)
    pred = resize_if_needed(payload["pred"], shape, cv2.INTER_NEAREST).astype(np.uint8, copy=False)
    margin = resize_if_needed(payload["margin"], shape, cv2.INTER_LINEAR).astype(np.float32, copy=False)
    return pred, margin


def label_path_from_image(path):
    path = Path(path)
    parts = list(path.parts)
    if "img_co_left" in parts:
        idx = parts.index("img_co_left")
        parts[idx] = "segment_co"
        return str(Path(*parts))
    if "rgb_anon" in parts:
        idx = parts.index("rgb_anon")
        parts[idx] = "gt"
        name = path.name.replace("_rgb_anon.png", "_gt_labelTrainIds.png")
        return str(Path(*parts[:-1]) / name)
    raise ValueError(f"Cannot infer label path from image path: {path}")


def read_json_predictions(path, limit=None):
    with Path(path).open("r", encoding="utf-8") as f:
        rows = json.load(f)
    by_file = {}
    for row in rows:
        by_file.setdefault(row["file_name"], []).append(row)
    file_names = sorted(by_file)
    if limit is not None:
        file_names = file_names[: int(limit)]
    return [(file_name, by_file[file_name]) for file_name in file_names]


def decode_json_pred(entries, shape):
    pred = np.full(shape, 255, dtype=np.uint8)
    for entry in entries:
        rle = entry["segmentation"]
        if isinstance(rle.get("counts"), str):
            rle = {"size": rle["size"], "counts": rle["counts"].encode("utf-8")}
        mask = mask_utils.decode(rle).astype(bool)
        if mask.shape != tuple(shape):
            mask = resize_if_needed(mask.astype(np.uint8), shape, cv2.INTER_NEAREST).astype(bool)
        pred[mask] = int(entry["category_id"])
    pred[pred == 255] = 0
    return pred


def load_guide(path, shape):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    image = resize_if_needed(image, shape, cv2.INTER_LINEAR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return gray


def semantic_boundary_band(pred, radius):
    if radius <= 0:
        return np.ones(pred.shape, dtype=bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    values = pred.astype(np.float32, copy=False)
    local_max = cv2.dilate(values, kernel)
    local_min = cv2.erode(values, kernel)
    return local_max != local_min


def uncertainty_mask(margin, percentile):
    if percentile >= 100:
        return np.ones(margin.shape, dtype=bool)
    threshold = np.percentile(margin.astype(np.float32, copy=False), float(percentile))
    return margin <= threshold


def margin_unary(pred, margin, variant):
    if variant.unary_margin <= 0:
        return np.full(pred.shape, float(variant.unary_base), dtype=np.float32)
    scale = np.percentile(margin.astype(np.float32, copy=False), 90)
    if not np.isfinite(scale) or scale <= 1e-8:
        norm = np.zeros(pred.shape, dtype=np.float32)
    else:
        norm = np.clip(margin.astype(np.float32, copy=False) / float(scale), 0.0, 1.0)
    return (float(variant.unary_base) + float(variant.unary_margin) * norm).astype(np.float32, copy=False)


def gaussian_message(mask, radius, sigma):
    ksize = 2 * int(radius) + 1
    return cv2.GaussianBlur(
        mask.astype(np.float32, copy=False),
        (ksize, ksize),
        float(sigma),
        borderType=cv2.BORDER_REFLECT,
    )


def precompute_guided(guide, radius):
    ksize = (2 * int(radius) + 1, 2 * int(radius) + 1)
    mean_i = cv2.boxFilter(guide, -1, ksize, normalize=True, borderType=cv2.BORDER_REFLECT)
    corr_i = cv2.boxFilter(guide * guide, -1, ksize, normalize=True, borderType=cv2.BORDER_REFLECT)
    return ksize, mean_i, corr_i - mean_i * mean_i


def guided_message(mask, guide, guided_precomputed, eps):
    ksize, mean_i, var_i = guided_precomputed
    src = mask.astype(np.float32, copy=False)
    mean_p = cv2.boxFilter(src, -1, ksize, normalize=True, borderType=cv2.BORDER_REFLECT)
    corr_ip = cv2.boxFilter(guide * src, -1, ksize, normalize=True, borderType=cv2.BORDER_REFLECT)
    cov_ip = corr_ip - mean_i * mean_p
    a = cov_ip / (var_i + float(eps))
    b = mean_p - a * mean_i
    mean_a = cv2.boxFilter(a, -1, ksize, normalize=True, borderType=cv2.BORDER_REFLECT)
    mean_b = cv2.boxFilter(b, -1, ksize, normalize=True, borderType=cv2.BORDER_REFLECT)
    return np.clip(mean_a * guide + mean_b, 0.0, 1.0)


def refine(pred0, margin, guide, variant):
    pred0 = pred0.astype(np.uint8, copy=False)
    pred = pred0.copy()
    unary = margin_unary(pred0, margin, variant)
    allow = semantic_boundary_band(pred0, variant.boundary_radius) & uncertainty_mask(margin, variant.uncertain_percentile)
    guided_precomputed = precompute_guided(guide, variant.radius) if variant.mode == "guided" else None

    for _ in range(int(variant.iterations)):
        best_score = np.full(pred.shape, -1e9, dtype=np.float32)
        best_label = pred.copy()
        for class_id in range(CLASS_COUNT):
            mask = pred == class_id
            if variant.mode == "guided":
                message = guided_message(mask, guide, guided_precomputed, variant.guided_eps)
            else:
                message = gaussian_message(mask, variant.radius, variant.sigma)
            score = float(variant.pairwise_weight) * message
            score = score + np.where(pred0 == class_id, unary, 0.0)
            update = score > best_score
            best_score[update] = score[update]
            best_label[update] = class_id
        pred = np.where(allow, best_label, pred0).astype(np.uint8, copy=False)
    return pred


def metric_summary(meter):
    values = meter.metrics()
    return {
        "mIoU": values["mIoU"],
        "mAcc": values["mAcc"],
        "aAcc": values["aAcc"],
    }


def evaluate_spec(cache_root, name, model, dataset, variants, limit=None, save_per_class=False):
    rows = read_rows(cache_root, model, dataset, limit=limit)
    paths = map_paths(cache_root, model, dataset, limit=limit)
    if len(rows) != len(paths):
        raise RuntimeError(f"Row/map count mismatch for {name}: {len(rows)} rows vs {len(paths)} maps")

    meters = {"baseline": SegmentationStats()}
    changed = {variant.name: 0 for variant in variants}
    valid_pixels_total = 0
    for variant in variants:
        meters[variant.name] = SegmentationStats()

    iterator = tqdm(list(zip(rows, paths)), desc=f"crf:{name}")
    for row, path in iterator:
        label = load_label(row["label_path"])
        pred, margin = load_cached_map(path, label.shape)
        guide = load_guide(row["img_path"], label.shape)
        meters["baseline"].update(pred, label)
        valid_pixels_total += int(((label >= 0) & (label < CLASS_COUNT)).sum())
        for variant in variants:
            refined = refine(pred, margin, guide, variant)
            changed[variant.name] += int((refined != pred).sum())
            meters[variant.name].update(refined, label)

    baseline = metric_summary(meters["baseline"])
    results = {
        "name": name,
        "model": model,
        "dataset": dataset,
        "samples": len(rows),
        "baseline": baseline,
        "variants": {},
    }
    for variant in variants:
        summary = metric_summary(meters[variant.name])
        summary["delta_mIoU"] = summary["mIoU"] - baseline["mIoU"]
        summary["delta_aAcc"] = summary["aAcc"] - baseline["aAcc"]
        summary["changed_pixel_ratio"] = changed[variant.name] / max(1, valid_pixels_total)
        summary["params"] = asdict(variant)
        if save_per_class:
            summary["per_class"] = meters[variant.name].class_rows()
        results["variants"][variant.name] = summary
    if save_per_class:
        results["baseline"]["per_class"] = meters["baseline"].class_rows()
    return results


def evaluate_json_spec(name, json_path, variants, limit=None, save_per_class=False):
    rows = read_json_predictions(json_path, limit=limit)
    meters = {"baseline": SegmentationStats()}
    changed = {variant.name: 0 for variant in variants}
    valid_pixels_total = 0
    for variant in variants:
        meters[variant.name] = SegmentationStats()

    iterator = tqdm(rows, desc=f"crf-json:{name}")
    for img_path, entries in iterator:
        label_path = label_path_from_image(img_path)
        label = load_label(label_path)
        pred = decode_json_pred(entries, label.shape)
        margin = np.zeros(label.shape, dtype=np.float32)
        guide = load_guide(img_path, label.shape)
        meters["baseline"].update(pred, label)
        valid_pixels_total += int(((label >= 0) & (label < CLASS_COUNT)).sum())
        for variant in variants:
            refined = refine(pred, margin, guide, variant)
            changed[variant.name] += int((refined != pred).sum())
            meters[variant.name].update(refined, label)

    baseline = metric_summary(meters["baseline"])
    results = {
        "name": name,
        "source": str(json_path),
        "source_type": "detectron2_sem_seg_predictions_json",
        "samples": len(rows),
        "baseline": baseline,
        "variants": {},
    }
    for variant in variants:
        summary = metric_summary(meters[variant.name])
        summary["delta_mIoU"] = summary["mIoU"] - baseline["mIoU"]
        summary["delta_aAcc"] = summary["aAcc"] - baseline["aAcc"]
        summary["changed_pixel_ratio"] = changed[variant.name] / max(1, valid_pixels_total)
        summary["params"] = asdict(variant)
        if save_per_class:
            summary["per_class"] = meters[variant.name].class_rows()
        results["variants"][variant.name] = summary
    if save_per_class:
        results["baseline"]["per_class"] = meters["baseline"].class_rows()
    return results


def best_variant(results):
    variants = results["variants"]
    if not variants:
        return None
    return max(variants.items(), key=lambda item: item[1]["mIoU"])


def main():
    args = parse_args()
    cache_root = Path(args.cache_root)
    variants = variants_for_preset(args.preset)
    json_specs = parse_json_specs(args.json_spec)
    specs = parse_specs(args.spec) if args.spec or not json_specs else {}

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "note": (
            "CRF-style postprocess from hard predictions. Cache specs use hard pred + margin. "
            "JSON specs use hard pred only, so margin-based uncertainty is unavailable."
        ),
        "limit": args.limit,
        "preset": args.preset,
        "classes": list(CLASSES),
        "specs": {},
    }
    for name, spec in specs.items():
        result = evaluate_spec(
            cache_root,
            name,
            spec["model"],
            spec["dataset"],
            variants,
            limit=args.limit,
            save_per_class=args.save_per_class,
        )
        best = best_variant(result)
        if best is not None:
            best_name, best_metrics = best
            result["best_variant"] = {
                "name": best_name,
                "mIoU": best_metrics["mIoU"],
                "delta_mIoU": best_metrics["delta_mIoU"],
                "aAcc": best_metrics["aAcc"],
                "delta_aAcc": best_metrics["delta_aAcc"],
                "changed_pixel_ratio": best_metrics["changed_pixel_ratio"],
            }
        payload["specs"][name] = result
    for name, json_path in json_specs.items():
        result = evaluate_json_spec(
            name,
            json_path,
            variants,
            limit=args.limit,
            save_per_class=args.save_per_class,
        )
        best = best_variant(result)
        if best is not None:
            best_name, best_metrics = best
            result["best_variant"] = {
                "name": best_name,
                "mIoU": best_metrics["mIoU"],
                "delta_mIoU": best_metrics["delta_mIoU"],
                "aAcc": best_metrics["aAcc"],
                "delta_aAcc": best_metrics["delta_aAcc"],
                "changed_pixel_ratio": best_metrics["changed_pixel_ratio"],
            }
        payload["specs"][name] = result

    out_path = Path(args.out_json)
    if args.limit is not None:
        stem = out_path.stem
        out_path = out_path.with_name(f"{stem}_limit{int(args.limit)}{out_path.suffix}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote {out_path}")
    for name, result in payload["specs"].items():
        base = result["baseline"]["mIoU"]
        best = result.get("best_variant")
        if best is None:
            continue
        print(
            f"{name}: baseline={base:.4f}, best={best['mIoU']:.4f}, "
            f"delta={best['delta_mIoU']:+.4f}, changed={100.0 * best['changed_pixel_ratio']:.3f}%"
        )


if __name__ == "__main__":
    main()
