#!/usr/bin/env python
"""Learn event transition pairs on train, then evaluate them on val.

This is a diagnostic for the hypothesis:

  strong RGB TTA anchor + event correction only for train-positive
  base_class -> event_class transitions in event-active regions.

It deliberately learns the whitelist from a training split, not from val, so a
positive val result is stronger evidence than another validation-greedy rule.
"""

import argparse
import copy
import json
import os
import sys
import importlib.util
from collections import OrderedDict, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
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

from diagnose_tta_event_class_routing import (  # noqa: E402
    ConfusionMeter,
    build_model,
    build_scale_branches,
    collect_tta_prob,
    infer_prob,
    load_label,
    parse_float_list,
    parse_int_list,
    parse_scale_specs,
    pred_boundary,
    resize_stat,
    setup_cfg,
    top_conf_margin,
    valid_label_mask,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--base-weights", required=True)
    parser.add_argument("--event-config", required=True)
    parser.add_argument("--event-weights", required=True)
    parser.add_argument("--train-dataset", default="cosec_night_train_event")
    parser.add_argument("--eval-dataset", default="cosec_night_val_event")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-limit", type=int, default=192)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument(
        "--scale-specs",
        default="s512:512:1200,s624:624:1200,s768:768:1400,s1024:1024:1600",
    )
    parser.add_argument("--scale-set", default="s512+s624+s768+s1024")
    parser.add_argument("--flip", action="store_true")
    parser.add_argument("--regions", default="raw,support,event_union")
    parser.add_argument("--boundary-radii", default="0,1,3")
    parser.add_argument("--base-conf-thresholds", default="0.0,0.6,0.8")
    parser.add_argument("--event-conf-thresholds", default="0.0,0.4,0.6")
    parser.add_argument("--margin-modes", default="none,event_ge_base")
    parser.add_argument("--min-net", type=int, default=200)
    parser.add_argument("--min-precision", type=float, default=0.45)
    parser.add_argument("--min-changed", type=int, default=500)
    parser.add_argument("--max-pairs", type=int, default=20)
    parser.add_argument("--out", required=True)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def split_csv(text):
    return [part.strip() for part in str(text).split(",") if part.strip()]


def pair_name(src, dst):
    return f"{CLASSES[int(src)]}->{CLASSES[int(dst)]}"


def pair_ids(name):
    src, dst = name.split("->", 1)
    return CLASSES.index(src), CLASSES.index(dst)


def empty_pair_counts():
    return {"changed": 0, "repaired": 0, "damaged": 0, "both_wrong": 0}


def add_pair_counts(stats, mask, base_pred, event_pred, label):
    if not mask.any():
        return
    changed = mask & (base_pred != event_pred)
    if not changed.any():
        return
    repaired = changed & (base_pred != label) & (event_pred == label)
    damaged = changed & (base_pred == label) & (event_pred != label)
    both_wrong = changed & (base_pred != label) & (event_pred != label)
    num_classes = len(CLASSES)
    changed_counts = np.bincount(
        num_classes * base_pred[changed].astype(np.int64) + event_pred[changed].astype(np.int64),
        minlength=num_classes**2,
    )
    repaired_counts = np.bincount(
        num_classes * base_pred[repaired].astype(np.int64) + event_pred[repaired].astype(np.int64),
        minlength=num_classes**2,
    )
    damaged_counts = np.bincount(
        num_classes * base_pred[damaged].astype(np.int64) + event_pred[damaged].astype(np.int64),
        minlength=num_classes**2,
    )
    both_wrong_counts = np.bincount(
        num_classes * base_pred[both_wrong].astype(np.int64) + event_pred[both_wrong].astype(np.int64),
        minlength=num_classes**2,
    )
    for idx in np.flatnonzero(changed_counts):
        item = stats[pair_name(idx // num_classes, idx % num_classes)]
        item["changed"] += int(changed_counts[idx])
        item["repaired"] += int(repaired_counts[idx])
        item["damaged"] += int(damaged_counts[idx])
        item["both_wrong"] += int(both_wrong_counts[idx])


def finalize_pair_counts(pair, counts):
    changed = counts["changed"]
    net = counts["repaired"] - counts["damaged"]
    return {
        "pair": pair,
        **counts,
        "net_repaired": net,
        "repair_precision": float(counts["repaired"] / changed) if changed else 0.0,
        "damage_rate": float(counts["damaged"] / changed) if changed else 0.0,
        "both_wrong_rate": float(counts["both_wrong"] / changed) if changed else 0.0,
    }


def select_pairs(pair_stats, min_net, min_precision, min_changed, max_pairs):
    rows = [
        finalize_pair_counts(pair, counts)
        for pair, counts in pair_stats.items()
        if counts["changed"] >= min_changed
    ]
    rows = [
        row
        for row in rows
        if row["net_repaired"] >= min_net and row["repair_precision"] >= min_precision
    ]
    rows.sort(key=lambda row: (row["net_repaired"], row["repair_precision"], row["changed"]), reverse=True)
    if max_pairs > 0:
        rows = rows[:max_pairs]
    return rows, {pair_ids(row["pair"]) for row in rows}


def make_regions(event_stats, label_shape, valid):
    raw = valid & (resize_stat(event_stats, 0, label_shape) > 0)
    support = valid & (resize_stat(event_stats, 3, label_shape) > 0)
    return OrderedDict(
        [
            ("raw", raw),
            ("support", support),
            ("event_union", raw | support),
        ]
    )


def build_contexts(args):
    scale_specs = parse_scale_specs(args.scale_specs)
    scale_names = [name.strip() for name in args.scale_set.split("+") if name.strip()]
    unknown = [name for name in scale_names if name not in scale_specs]
    if unknown:
        raise ValueError(f"Unknown scales in --scale-set: {unknown}")

    event_cfg = setup_cfg(args.event_config, args.event_weights, args.device)
    event_mapper = MaskFormerSemanticDatasetMapper(event_cfg, False)
    event_model = build_model(event_cfg)
    base_cfg = setup_cfg(args.base_config, args.base_weights, args.device)
    base_model = build_model(base_cfg)
    scale_branches = build_scale_branches(args, scale_specs, scale_names)
    return base_model, event_model, event_mapper, scale_branches


def infer_record(args, base_model, event_model, event_mapper, scale_branches, record):
    label = load_label(record)
    valid = valid_label_mask(label)
    base_prob = collect_tta_prob(args, base_model, scale_branches, record, label.shape)
    event_mapped = event_mapper(copy.deepcopy(record))
    event_prob = infer_prob(event_model, event_mapped, label.shape, use_flip=False)

    base_pred = base_prob.argmax(dim=0).numpy()
    event_pred = event_prob.argmax(dim=0).numpy()
    base_conf, base_margin = top_conf_margin(base_prob)
    event_conf, event_margin = top_conf_margin(event_prob)
    event_stats = event_mapped["event_stats"].float()
    regions = make_regions(event_stats, label.shape, valid)
    return {
        "label": label,
        "valid": valid,
        "base_pred": base_pred,
        "event_pred": event_pred,
        "base_conf": base_conf,
        "base_margin": base_margin,
        "event_conf": event_conf,
        "event_margin": event_margin,
        "regions": regions,
    }


def condition_mask(ctx, region_name, radius, base_conf_thr, event_conf_thr, margin_mode):
    mask = (
        ctx["valid"]
        & ctx["regions"][region_name]
        & pred_boundary(ctx["base_pred"], radius)
        & (ctx["base_conf"] >= base_conf_thr)
        & (ctx["event_conf"] >= event_conf_thr)
        & (ctx["base_pred"] != ctx["event_pred"])
    )
    if margin_mode == "event_ge_base":
        mask &= ctx["event_margin"] >= ctx["base_margin"]
    elif margin_mode != "none":
        raise ValueError(f"Unknown margin mode: {margin_mode}")
    return mask


def merge_with_pairs(ctx, condition, selected_pairs):
    if not selected_pairs:
        return ctx["base_pred"]
    accept = np.zeros(condition.shape, dtype=bool)
    for src, dst in selected_pairs:
        accept |= condition & (ctx["base_pred"] == src) & (ctx["event_pred"] == dst)
    merged = ctx["base_pred"].copy()
    merged[accept] = ctx["event_pred"][accept]
    return merged


def empty_route_counts():
    return {"changed": 0, "repaired": 0, "damaged": 0, "both_wrong": 0}


def add_route_counts(counts, ctx, pred):
    valid = ctx["valid"]
    base_pred = ctx["base_pred"]
    label = ctx["label"]
    changed = valid & (base_pred != pred)
    repaired = changed & (base_pred != label) & (pred == label)
    damaged = changed & (base_pred == label) & (pred != label)
    both_wrong = changed & (base_pred != label) & (pred != label)
    counts["changed"] += int(changed.sum())
    counts["repaired"] += int(repaired.sum())
    counts["damaged"] += int(damaged.sum())
    counts["both_wrong"] += int(both_wrong.sum())


def finalize_route_counts(counts):
    changed = counts["changed"]
    return {
        **counts,
        "net_repaired": counts["repaired"] - counts["damaged"],
        "repair_precision": float(counts["repaired"] / changed) if changed else 0.0,
        "damage_rate": float(counts["damaged"] / changed) if changed else 0.0,
        "both_wrong_rate": float(counts["both_wrong"] / changed) if changed else 0.0,
    }


def route_name(region, radius, base_conf_thr, event_conf_thr, margin_mode):
    return (
        f"{region}_b{radius}_baseconf{base_conf_thr:g}"
        f"_eventconf{event_conf_thr:g}_{margin_mode}"
    )


def main():
    args = parse_args()
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    register_cosec()

    regions = split_csv(args.regions)
    boundary_radii = parse_int_list(args.boundary_radii)
    base_conf_thresholds = parse_float_list(args.base_conf_thresholds)
    event_conf_thresholds = parse_float_list(args.event_conf_thresholds)
    margin_modes = split_csv(args.margin_modes)

    base_model, event_model, event_mapper, scale_branches = build_contexts(args)

    train_records = list(DatasetCatalog.get(args.train_dataset))
    eval_records = list(DatasetCatalog.get(args.eval_dataset))
    if args.train_limit is not None:
        train_records = train_records[: args.train_limit]
    if args.eval_limit is not None:
        eval_records = eval_records[: args.eval_limit]

    pair_stats_by_route = OrderedDict()
    route_specs = []
    for region in regions:
        for radius in boundary_radii:
            for base_conf_thr in base_conf_thresholds:
                for event_conf_thr in event_conf_thresholds:
                    for margin_mode in margin_modes:
                        name = route_name(region, radius, base_conf_thr, event_conf_thr, margin_mode)
                        route_specs.append((name, region, radius, base_conf_thr, event_conf_thr, margin_mode))
                        pair_stats_by_route[name] = defaultdict(empty_pair_counts)

    train_iter = train_records if args.quiet else tqdm(train_records, desc=args.train_dataset)
    for record in train_iter:
        ctx = infer_record(args, base_model, event_model, event_mapper, scale_branches, record)
        for name, region, radius, base_conf_thr, event_conf_thr, margin_mode in route_specs:
            cond = condition_mask(ctx, region, radius, base_conf_thr, event_conf_thr, margin_mode)
            add_pair_counts(pair_stats_by_route[name], cond, ctx["base_pred"], ctx["event_pred"], ctx["label"])

    selected_by_route = OrderedDict()
    for name, stats in pair_stats_by_route.items():
        rows, pairs = select_pairs(
            stats,
            args.min_net,
            args.min_precision,
            args.min_changed,
            args.max_pairs,
        )
        selected_by_route[name] = {"pairs": rows, "pair_ids": pairs}

    meters = OrderedDict()
    meters["rgb_tta"] = ConfusionMeter(num_classes=len(CLASSES))
    meters["event_candidate"] = ConfusionMeter(num_classes=len(CLASSES))
    counts = OrderedDict((name, empty_route_counts()) for name in selected_by_route)
    for name in selected_by_route:
        meters[name] = ConfusionMeter(num_classes=len(CLASSES))

    eval_iter = eval_records if args.quiet else tqdm(eval_records, desc=args.eval_dataset)
    for record in eval_iter:
        ctx = infer_record(args, base_model, event_model, event_mapper, scale_branches, record)
        meters["rgb_tta"].update(ctx["base_pred"], ctx["label"])
        meters["event_candidate"].update(ctx["event_pred"], ctx["label"])
        for name, region, radius, base_conf_thr, event_conf_thr, margin_mode in route_specs:
            cond = condition_mask(ctx, region, radius, base_conf_thr, event_conf_thr, margin_mode)
            pred = merge_with_pairs(ctx, cond, selected_by_route[name]["pair_ids"])
            meters[name].update(pred, ctx["label"])
            add_route_counts(counts[name], ctx, pred)

    results = OrderedDict()
    for name, meter in meters.items():
        results[name] = meter.metrics()
        if name in selected_by_route:
            results[name].update(finalize_route_counts(counts[name]))
            results[name]["selected_pair_count"] = len(selected_by_route[name]["pairs"])
            results[name]["selected_pairs"] = selected_by_route[name]["pairs"]

    baseline = results["rgb_tta"]["mIoU"]
    eval_routes = [
        {"method": name, "delta_vs_rgb_tta": values["mIoU"] - baseline, **values}
        for name, values in results.items()
        if name not in {"rgb_tta", "event_candidate"}
    ]
    eval_routes.sort(key=lambda row: row["mIoU"], reverse=True)

    output = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "train_sample_count": len(train_records),
        "eval_sample_count": len(eval_records),
        "results": results,
        "top_eval_routes": eval_routes[:30],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote train-learned event pair router diagnostic: {out_path}")
    print(
        f"RGB TTA={results['rgb_tta']['mIoU']:.4f}, "
        f"event_candidate={results['event_candidate']['mIoU']:.4f}"
    )
    for row in eval_routes[:10]:
        print(
            f"  {row['method']}: mIoU={row['mIoU']:.4f} "
            f"delta={row['delta_vs_rgb_tta']:+.4f} pairs={row['selected_pair_count']} "
            f"changed={row['changed']} net={row['net_repaired']}"
        )


if __name__ == "__main__":
    main()
