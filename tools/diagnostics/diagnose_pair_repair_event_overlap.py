#!/usr/bin/env python
"""Measure repaired and damaged prediction-transition pairs in event regions."""

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
from detectron2.data import DatasetCatalog  # noqa: E402
from mask2former import MaskFormerSemanticDatasetMapper  # noqa: E402
from train_mask2former_cosec import register_cosec  # noqa: E402

from diagnose_confidence_ensemble_routing import (  # noqa: E402
    build_model,
    infer_mapped,
    load_label,
    normalize_scores,
    resize_stat,
    setup_cfg,
    top_conf_margin,
    valid_label_mask,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--base-weights", required=True)
    parser.add_argument("--new-config", required=True)
    parser.add_argument("--new-weights", required=True)
    parser.add_argument("--event-config", default=None)
    parser.add_argument("--dataset", default="cosec_day_val_event")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--boundary-radius", type=int, default=3)
    parser.add_argument("--low-margin-percentile", type=float, default=40.0)
    parser.add_argument("--high-entropy-percentile", type=float, default=80.0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def div(num, den):
    return float(num / den) if den else 0.0


def entropy_from_prob(prob):
    entropy = -(prob * prob.clamp_min(1e-8).log()).sum(dim=0)
    return (entropy / np.log(float(prob.shape[0]))).cpu().numpy()


def boundary_mask(pred, radius):
    pred = np.asarray(pred)
    padded = np.pad(pred, int(radius), mode="edge")
    center = padded[radius : radius + pred.shape[0], radius : radius + pred.shape[1]]
    boundary = np.zeros(pred.shape, dtype=bool)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            shifted = padded[
                radius + dy : radius + dy + pred.shape[0],
                radius + dx : radius + dx + pred.shape[1],
            ]
            boundary |= shifted != center
    return boundary


def percentile_region(values, valid, percentile, high):
    data = values[valid]
    if data.size == 0:
        return np.zeros_like(valid, dtype=bool)
    threshold = np.percentile(data, float(percentile))
    return valid & ((values >= threshold) if high else (values <= threshold))


def empty_counts():
    return {
        "changed": 0,
        "repaired": 0,
        "damaged": 0,
        "wrong_to_wrong": 0,
        "correct_stayed_correct": 0,
    }


def add_pair(pair_stats, region, base_class, new_class, repaired, damaged, wrong_to_wrong):
    pair_key = f"{CLASSES[int(base_class)]}->{CLASSES[int(new_class)]}"
    stats = pair_stats[region][pair_key]
    stats["changed"] += 1
    stats["repaired"] += int(repaired)
    stats["damaged"] += int(damaged)
    stats["wrong_to_wrong"] += int(wrong_to_wrong)


def update_region(global_stats, pair_stats, region, mask, base_pred, new_pred, label, valid):
    region_mask = valid & mask
    changed = region_mask & (base_pred != new_pred)
    base_wrong = region_mask & (base_pred != label)
    base_correct = region_mask & (base_pred == label)
    repaired = changed & base_wrong & (new_pred == label)
    damaged = changed & base_correct & (new_pred != label)
    wrong_to_wrong = changed & base_wrong & (new_pred != label)

    stats = global_stats[region]
    stats["valid_pixels"] += int(valid.sum())
    stats["region_pixels"] += int(region_mask.sum())
    stats["base_wrong"] += int(base_wrong.sum())
    stats["base_correct"] += int(base_correct.sum())
    stats["changed"] += int(changed.sum())
    stats["repaired"] += int(repaired.sum())
    stats["damaged"] += int(damaged.sum())
    stats["wrong_to_wrong"] += int(wrong_to_wrong.sum())
    stats["net_repaired"] = stats["repaired"] - stats["damaged"]

    ys, xs = np.nonzero(changed)
    for y, x in zip(ys.tolist(), xs.tolist()):
        add_pair(
            pair_stats,
            region,
            base_pred[y, x],
            new_pred[y, x],
            bool(repaired[y, x]),
            bool(damaged[y, x]),
            bool(wrong_to_wrong[y, x]),
        )


def finalize_global(stats):
    output = dict(stats)
    output["net_repaired"] = stats["repaired"] - stats["damaged"]
    output["region_coverage"] = div(stats["region_pixels"], stats["valid_pixels"])
    output["changed_rate_in_region"] = div(stats["changed"], stats["region_pixels"])
    output["repair_rate_of_base_wrong"] = div(stats["repaired"], stats["base_wrong"])
    output["damage_rate_of_base_correct"] = div(stats["damaged"], stats["base_correct"])
    output["repair_precision_among_changes"] = div(stats["repaired"], stats["changed"])
    output["damage_rate_among_changes"] = div(stats["damaged"], stats["changed"])
    return output


def finalize_pair(stats):
    output = dict(stats)
    output["net_repaired"] = stats["repaired"] - stats["damaged"]
    output["repair_precision"] = div(stats["repaired"], stats["changed"])
    output["damage_rate"] = div(stats["damaged"], stats["changed"])
    return output


def main():
    args = parse_args()
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    register_cosec()

    event_config = args.event_config or args.new_config
    base_cfg = setup_cfg(args.base_config, args.base_weights, args.device)
    new_cfg = setup_cfg(args.new_config, args.new_weights, args.device)
    event_cfg = setup_cfg(event_config, args.new_weights, args.device)

    mapper = MaskFormerSemanticDatasetMapper(event_cfg, False)
    base_model = build_model(base_cfg)
    new_model = build_model(new_cfg)

    records = DatasetCatalog.get(args.dataset)
    if args.limit is not None:
        records = records[: args.limit]

    global_stats = OrderedDict()
    pair_stats = defaultdict(lambda: defaultdict(empty_counts))
    for name in ["all", "raw_event", "support", "support_uncertain_boundary"]:
        global_stats[name] = {
            "valid_pixels": 0,
            "region_pixels": 0,
            "base_wrong": 0,
            "base_correct": 0,
            "changed": 0,
            "repaired": 0,
            "damaged": 0,
            "wrong_to_wrong": 0,
            "net_repaired": 0,
        }

    iterator = records if args.quiet else tqdm(records, desc=args.dataset)
    for record in iterator:
        label = load_label(record)
        mapped = mapper(copy.deepcopy(record))
        event_stats = mapped["event_stats"].float()
        base_prob = normalize_scores(infer_mapped(base_model, mapped))
        new_prob = normalize_scores(infer_mapped(new_model, mapped))
        base_pred = base_prob.argmax(dim=0).numpy()
        new_pred = new_prob.argmax(dim=0).numpy()
        valid = valid_label_mask(label)

        raw_event = valid & (resize_stat(event_stats, 0, label.shape) > 0)
        support = valid & (resize_stat(event_stats, 3, label.shape) > 0)
        _, base_margin = top_conf_margin(base_prob)
        base_entropy = entropy_from_prob(base_prob)
        low_margin = percentile_region(base_margin, valid, args.low_margin_percentile, high=False)
        high_entropy = percentile_region(base_entropy, valid, args.high_entropy_percentile, high=True)
        base_boundary = boundary_mask(base_pred, args.boundary_radius)
        uncertain_boundary = support & (base_boundary | low_margin | high_entropy)

        regions = {
            "all": valid,
            "raw_event": raw_event,
            "support": support,
            "support_uncertain_boundary": uncertain_boundary,
        }
        for region, mask in regions.items():
            update_region(global_stats, pair_stats, region, mask, base_pred, new_pred, label, valid)

    output_pairs = OrderedDict()
    for region, pairs in pair_stats.items():
        finalized = [dict(pair=pair, **finalize_pair(stats)) for pair, stats in pairs.items()]
        output_pairs[region] = {
            "top_positive": sorted(finalized, key=lambda item: item["net_repaired"], reverse=True)[:50],
            "top_negative": sorted(finalized, key=lambda item: item["net_repaired"])[:50],
            "all_pairs": sorted(finalized, key=lambda item: (-item["changed"], item["pair"])),
        }

    output = {
        "args": vars(args),
        "sample_count": len(records),
        "regions": OrderedDict((name, finalize_global(stats)) for name, stats in global_stats.items()),
        "pairs": output_pairs,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote pair diagnostics: {out_path}")
    for region, stats in output["regions"].items():
        print(
            f"{region}: changed={stats['changed']} repaired={stats['repaired']} "
            f"damaged={stats['damaged']} net={stats['net_repaired']} "
            f"repair_precision={100 * stats['repair_precision_among_changes']:.2f}%"
        )
    print("Top support_uncertain_boundary positive pairs:")
    for item in output["pairs"]["support_uncertain_boundary"]["top_positive"][:10]:
        print(
            f"  {item['pair']}: changed={item['changed']} repaired={item['repaired']} "
            f"damaged={item['damaged']} net={item['net_repaired']}"
        )
    print("Top support_uncertain_boundary negative pairs:")
    for item in output["pairs"]["support_uncertain_boundary"]["top_negative"][:10]:
        print(
            f"  {item['pair']}: changed={item['changed']} repaired={item['repaired']} "
            f"damaged={item['damaged']} net={item['net_repaired']}"
        )


if __name__ == "__main__":
    main()
