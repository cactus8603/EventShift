#!/usr/bin/env python
"""Per-class confidence ensemble over cached segmentation feature maps."""

import argparse
import csv
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from ensemble_feature_cache_common import (
    CLASS_COUNT,
    CLASSES,
    SegmentationStats,
    load_label,
    per_image_summary,
    resize_if_needed,
    safe_name,
    save_feature_maps,
    write_csv,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-cache",
        action="append",
        required=True,
        help="Model cache in name=/path/to/cache/model_dir form. Repeat this flag.",
    )
    parser.add_argument("--dataset", required=True, help="Dataset key inside each model cache, e.g. cosec_day_val.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-maps", action="store_true")
    parser.add_argument("--save-png", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--iou-weight", type=float, default=0.5)
    parser.add_argument("--acc-weight", type=float, default=0.25)
    parser.add_argument("--precision-weight", type=float, default=0.25)
    parser.add_argument("--class-conf-weight", type=float, default=0.0)
    parser.add_argument("--prior-power", type=float, default=1.0)
    parser.add_argument("--conf-power", type=float, default=1.0)
    parser.add_argument("--margin-power", type=float, default=0.0)
    parser.add_argument("--entropy-power", type=float, default=0.0)
    parser.add_argument("--min-prior", type=float, default=0.05)
    parser.add_argument("--anchor-model", default="")
    parser.add_argument(
        "--anchor-score-bonus",
        type=float,
        default=1.0,
        help="Multiply anchor score by this value before winner selection.",
    )
    parser.add_argument(
        "--candidate-prior-gap",
        type=float,
        default=0.0,
        help=(
            "If >0 and an anchor is set, a non-anchor candidate can replace only when "
            "its prior for the candidate class exceeds the anchor prior for that same "
            "class by at least this gap."
        ),
    )
    parser.add_argument(
        "--anchor-keep-conf",
        type=float,
        default=0.0,
        help="If >0, keep anchor prediction where anchor confidence is at least this threshold.",
    )
    parser.add_argument(
        "--anchor-score-ratio",
        type=float,
        default=1.0,
        help="Require a non-anchor score to exceed anchor_score * ratio before replacing anchor.",
    )
    return parser.parse_args()


def parse_model_cache(text):
    if "=" not in text:
        raise ValueError(f"--model-cache must be name=/path, got: {text}")
    name, path = text.split("=", 1)
    name = name.strip()
    path = Path(path.strip())
    if not name:
        raise ValueError(f"Empty model name in --model-cache: {text}")
    if not path.is_dir():
        raise FileNotFoundError(path)
    return name, path


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def float_or_nan(value):
    if value in ("", None):
        return float("nan")
    return float(value)


def load_class_priors(model_dir, dataset, args):
    rows = read_csv(model_dir / f"{dataset}_per_class_iou.csv")
    if len(rows) != CLASS_COUNT:
        raise ValueError(f"Expected {CLASS_COUNT} class rows in {model_dir}, got {len(rows)}")

    priors = np.zeros(CLASS_COUNT, dtype=np.float32)
    diagnostics = []
    total_weight = args.iou_weight + args.acc_weight + args.precision_weight + args.class_conf_weight
    if total_weight <= 0:
        raise ValueError("At least one class prior weight must be positive.")

    for row in rows:
        class_id = int(row["class_id"])
        iou = float_or_nan(row.get("iou")) / 100.0
        acc = float_or_nan(row.get("acc")) / 100.0
        pred_pixels = int(row.get("pred_pixels") or 0)
        correct_pred_pixels = int(row.get("correct_pred_pixels") or 0)
        precision = correct_pred_pixels / pred_pixels if pred_pixels else float("nan")
        mean_conf = float_or_nan(row.get("mean_conf_pred"))
        terms = [
            (args.iou_weight, iou),
            (args.acc_weight, acc),
            (args.precision_weight, precision),
            (args.class_conf_weight, mean_conf),
        ]
        weighted_sum = 0.0
        used_weight = 0.0
        for weight, value in terms:
            if weight <= 0 or not np.isfinite(value):
                continue
            weighted_sum += weight * float(value)
            used_weight += weight
        prior = weighted_sum / used_weight if used_weight > 0 else args.min_prior
        prior = max(float(prior), float(args.min_prior))
        priors[class_id] = prior
        diagnostics.append(
            {
                "class_id": class_id,
                "class_name": row.get("class_name", CLASSES[class_id]),
                "iou": None if not np.isfinite(iou) else iou,
                "acc": None if not np.isfinite(acc) else acc,
                "precision": None if not np.isfinite(precision) else precision,
                "mean_conf": None if not np.isfinite(mean_conf) else mean_conf,
                "prior": prior,
                "pred_pixels": pred_pixels,
                "correct_pred_pixels": correct_pred_pixels,
            }
        )
    return priors, diagnostics


def load_model_cache(name, model_dir, dataset, args):
    per_image_path = model_dir / f"{dataset}_per_image.csv"
    map_dir = model_dir / "maps" / dataset
    if not per_image_path.is_file():
        raise FileNotFoundError(per_image_path)
    if not map_dir.is_dir():
        raise FileNotFoundError(f"Missing maps for {name}:{dataset}: {map_dir}")
    rows = read_csv(per_image_path)
    map_paths = sorted(map_dir.glob("*.npz"))
    if args.limit is not None:
        rows = rows[: args.limit]
        map_paths = map_paths[: args.limit]
    if len(rows) != len(map_paths):
        raise ValueError(f"{name}:{dataset} row/map count mismatch: {len(rows)} rows vs {len(map_paths)} maps")
    priors, prior_rows = load_class_priors(model_dir, dataset, args)
    return {
        "name": name,
        "dir": model_dir,
        "rows": rows,
        "maps": map_paths,
        "priors": priors,
        "prior_rows": prior_rows,
    }


def load_npz_map(path):
    data = np.load(path)
    return {
        "pred": data["pred"].astype(np.uint8, copy=False),
        "conf": data["conf"].astype(np.float32, copy=False),
        "margin": data["margin"].astype(np.float32, copy=False),
        "entropy": data["entropy"].astype(np.float32, copy=False),
    }


def score_map(feature, priors, args):
    pred = feature["pred"].astype(np.int64, copy=False)
    prior = priors[pred].astype(np.float32, copy=False)
    conf = np.clip(feature["conf"], 1e-6, 1.0)
    margin = np.clip(feature["margin"], 1e-6, 1.0)
    entropy_keep = np.clip(1.0 - feature["entropy"], 1e-6, 1.0)
    score = np.power(prior, args.prior_power) * np.power(conf, args.conf_power)
    if args.margin_power != 0:
        score *= np.power(margin, args.margin_power)
    if args.entropy_power != 0:
        score *= np.power(entropy_keep, args.entropy_power)
    return score.astype(np.float32, copy=False)


def validate_aligned_rows(models):
    base = models[0]["rows"]
    for model in models[1:]:
        for idx, (left, right) in enumerate(zip(base, model["rows"])):
            if left.get("image_id") != right.get("image_id"):
                raise ValueError(
                    f"Image order mismatch at index {idx}: "
                    f"{models[0]['name']}={left.get('image_id')} vs {model['name']}={right.get('image_id')}"
                )


def write_png(path, mask):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), mask.astype(np.uint8, copy=False)):
        raise RuntimeError(f"Could not write PNG: {path}")


def main():
    args = parse_args()
    model_specs = [parse_model_cache(item) for item in args.model_cache]
    if len(model_specs) < 2:
        raise ValueError("Need at least two --model-cache entries.")

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(out_dir)
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    models = [load_model_cache(name, path, args.dataset, args) for name, path in model_specs]
    validate_aligned_rows(models)
    model_names = [model["name"] for model in models]
    if args.anchor_model and args.anchor_model not in model_names:
        raise ValueError(f"--anchor-model must be one of {model_names}, got {args.anchor_model}")
    anchor_index = model_names.index(args.anchor_model) if args.anchor_model else None

    global_meter = SegmentationStats()
    per_image_rows = []
    winner_counter = Counter()
    winner_by_class = {class_name: Counter() for class_name in CLASSES}

    for image_index, row in enumerate(models[0]["rows"]):
        label = load_label(row["label_path"])
        features = []
        scores = []
        for model in models:
            feature = load_npz_map(model["maps"][image_index])
            for key, interpolation in [
                ("pred", cv2.INTER_NEAREST),
                ("conf", cv2.INTER_LINEAR),
                ("margin", cv2.INTER_LINEAR),
                ("entropy", cv2.INTER_LINEAR),
            ]:
                feature[key] = resize_if_needed(feature[key], label.shape, interpolation)
            features.append(feature)
            scores.append(score_map(feature, model["priors"], args))

        score_stack = np.stack(scores, axis=0)
        if anchor_index is not None:
            if args.anchor_score_bonus != 1.0:
                score_stack[anchor_index] *= float(args.anchor_score_bonus)
            if args.candidate_prior_gap > 0:
                for model_index, model in enumerate(models):
                    if model_index == anchor_index:
                        continue
                    candidate_pred = features[model_index]["pred"].astype(np.int64, copy=False)
                    candidate_prior = model["priors"][candidate_pred]
                    anchor_same_class_prior = models[anchor_index]["priors"][candidate_pred]
                    allowed = candidate_prior >= (
                        anchor_same_class_prior + float(args.candidate_prior_gap)
                    )
                    score_stack[model_index] = np.where(allowed, score_stack[model_index], -np.inf)
        winner = score_stack.argmax(axis=0)
        if anchor_index is not None:
            anchor_score = score_stack[anchor_index]
            best_score = score_stack.max(axis=0)
            keep_anchor = best_score < (anchor_score * float(args.anchor_score_ratio))
            if args.anchor_keep_conf > 0:
                keep_anchor |= features[anchor_index]["conf"] >= float(args.anchor_keep_conf)
            winner[keep_anchor] = anchor_index

        pred_stack = np.stack([feature["pred"] for feature in features], axis=0)
        conf_stack = np.stack([feature["conf"] for feature in features], axis=0)
        margin_stack = np.stack([feature["margin"] for feature in features], axis=0)
        entropy_stack = np.stack([feature["entropy"] for feature in features], axis=0)
        pred = np.take_along_axis(pred_stack, winner[None], axis=0)[0].astype(np.uint8, copy=False)
        conf = np.take_along_axis(conf_stack, winner[None], axis=0)[0].astype(np.float16, copy=False)
        margin = np.take_along_axis(margin_stack, winner[None], axis=0)[0].astype(np.float16, copy=False)
        entropy = np.take_along_axis(entropy_stack, winner[None], axis=0)[0].astype(np.float16, copy=False)

        for model_idx, model_name in enumerate(model_names):
            count = int(np.count_nonzero(winner == model_idx))
            winner_counter[model_name] += count
            for class_id, class_name in enumerate(CLASSES):
                class_count = int(np.count_nonzero((winner == model_idx) & (pred == class_id)))
                if class_count:
                    winner_by_class[class_name][model_name] += class_count

        global_meter.update(pred, label, conf=conf, margin=margin, entropy=entropy)
        summary = per_image_summary(pred, label, conf, margin, entropy)
        per_image_rows.append(
            {
                "dataset": args.dataset,
                "image_id": row["image_id"],
                "img_path": row["img_path"],
                "label_path": row["label_path"],
                **summary,
            }
        )

        image_id = safe_name(row["image_id"])
        if args.save_maps:
            save_feature_maps(out_dir / "maps" / args.dataset / f"{image_index:06d}_{image_id}.npz", pred, conf, margin, entropy)
        if args.save_png:
            write_png(out_dir / "png" / args.dataset / f"{image_index:06d}_{image_id}.png", pred)

    metrics = global_meter.metrics()
    class_rows = global_meter.class_rows()
    write_csv(out_dir / f"{args.dataset}_per_image.csv", per_image_rows)
    write_csv(out_dir / f"{args.dataset}_per_class_iou.csv", class_rows)

    prior_payload = {}
    for model in models:
        prior_payload[model["name"]] = model["prior_rows"]
    write_json(out_dir / "class_priors.json", prior_payload)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": args.dataset,
        "models": [
            {
                "name": model["name"],
                "cache_dir": str(model["dir"].resolve()),
            }
            for model in models
        ],
        "args": vars(args),
        "metrics": {
            "mIoU": metrics["mIoU"],
            "mAcc": metrics["mAcc"],
            "aAcc": metrics["aAcc"],
        },
        "winner_pixels": dict(winner_counter),
        "winner_pixels_by_class": {
            class_name: dict(counter) for class_name, counter in winner_by_class.items()
        },
    }
    write_json(out_dir / "summary.json", manifest)
    print(f"Wrote per-class confidence ensemble: {out_dir}")
    print(json.dumps(manifest["metrics"], indent=2, sort_keys=True))
    print(json.dumps(manifest["winner_pixels"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
