#!/usr/bin/env python
"""Full-val transition-pair repair diagnostics between two checkpoints."""

import argparse
import copy
import json
import math
import os
import sys
import importlib.util
from collections import OrderedDict, defaultdict
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
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--out", required=True)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def entropy_from_prob(prob):
    entropy = -(prob * prob.clamp_min(1e-8).log()).sum(dim=0)
    return (entropy / math.log(prob.shape[0])).cpu().numpy()


def percentile_mask(values, valid, q, low=True):
    selected = values[valid]
    if selected.size == 0:
        return np.zeros_like(valid, dtype=bool)
    threshold = np.percentile(selected, float(q))
    selected_mask = values <= threshold if low else values >= threshold
    return valid & selected_mask


def pair_name(base_cls, new_cls):
    return f"{CLASSES[int(base_cls)]}->{CLASSES[int(new_cls)]}"


def empty_pair_counts():
    return {
        "changed": 0,
        "repaired": 0,
        "damaged": 0,
        "both_wrong": 0,
        "wrong_to_wrong_changed_label": 0,
    }


def add_region_pair_counts(pair_stats, region_name, region, base_pred, new_pred, label, valid):
    changed = valid & region & (base_pred != new_pred)
    if not changed.any():
        return

    num_classes = len(CLASSES)
    pair_index = num_classes * base_pred[changed] + new_pred[changed]
    changed_counts = np.bincount(pair_index, minlength=num_classes**2)

    repaired = changed & (base_pred != label) & (new_pred == label)
    damaged = changed & (base_pred == label) & (new_pred != label)
    both_wrong = changed & (base_pred != label) & (new_pred != label)

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
        base_cls = idx // num_classes
        new_cls = idx % num_classes
        item = pair_stats[region_name][pair_name(base_cls, new_cls)]
        item["changed"] += int(changed_counts[idx])
        item["repaired"] += int(repaired_counts[idx])
        item["damaged"] += int(damaged_counts[idx])
        item["both_wrong"] += int(both_wrong_counts[idx])
        item["wrong_to_wrong_changed_label"] += int(both_wrong_counts[idx])


def empty_region_counts():
    return {
        "valid_pixels": 0,
        "mask_pixels": 0,
        "changed": 0,
        "repaired": 0,
        "damaged": 0,
        "both_wrong": 0,
    }


def add_region_counts(region_counts, region_name, region, base_pred, new_pred, label, valid):
    mask = valid & region
    changed = mask & (base_pred != new_pred)
    repaired = changed & (base_pred != label) & (new_pred == label)
    damaged = changed & (base_pred == label) & (new_pred != label)
    both_wrong = changed & (base_pred != label) & (new_pred != label)
    item = region_counts[region_name]
    item["valid_pixels"] += int(valid.sum())
    item["mask_pixels"] += int(mask.sum())
    item["changed"] += int(changed.sum())
    item["repaired"] += int(repaired.sum())
    item["damaged"] += int(damaged.sum())
    item["both_wrong"] += int(both_wrong.sum())


def div(num, den):
    return float(num / den) if den else 0.0


def finalize_pair(pair, counts):
    changed = counts["changed"]
    net = counts["repaired"] - counts["damaged"]
    return {
        "pair": pair,
        **counts,
        "net_repaired": net,
        "repair_precision": div(counts["repaired"], changed),
        "damage_rate": div(counts["damaged"], changed),
        "both_wrong_rate": div(counts["both_wrong"], changed),
    }


def finalize_region_counts(counts):
    out = dict(counts)
    out.update(
        {
            "mask_coverage": div(counts["mask_pixels"], counts["valid_pixels"]),
            "changed_rate_in_mask": div(counts["changed"], counts["mask_pixels"]),
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

    pair_stats = defaultdict(lambda: defaultdict(empty_pair_counts))
    region_counts = defaultdict(empty_region_counts)

    iterator = records if args.quiet else tqdm(records, desc=args.dataset)
    for record in iterator:
        label = load_label(record)
        mapped = mapper(copy.deepcopy(record))
        event_stats = mapped["event_stats"].float()
        valid = valid_label_mask(label)

        raw_event = valid & (resize_stat(event_stats, 0, label.shape) > 0)
        support = valid & (resize_stat(event_stats, 3, label.shape) > 0)

        base_prob = normalize_scores(infer_mapped(base_model, mapped))
        new_prob = normalize_scores(infer_mapped(new_model, mapped))
        base_pred = base_prob.argmax(dim=0).numpy()
        new_pred = new_prob.argmax(dim=0).numpy()

        _, base_margin = top_conf_margin(base_prob)
        base_entropy = entropy_from_prob(base_prob)
        lowmargin40 = percentile_mask(base_margin, valid, 40, low=True)
        highentropy80 = percentile_mask(base_entropy, valid, 80, low=False)

        regions = OrderedDict(
            [
                ("all", valid),
                ("raw_event", raw_event),
                ("support", support),
                ("support_lowmargin40", support & lowmargin40),
                ("support_highentropy80", support & highentropy80),
                ("raw_event_lowmargin40", raw_event & lowmargin40),
            ]
        )
        for region_name, region in regions.items():
            add_region_counts(region_counts, region_name, region, base_pred, new_pred, label, valid)
            add_region_pair_counts(pair_stats, region_name, region, base_pred, new_pred, label, valid)

    regions_out = OrderedDict()
    pairs_out = OrderedDict()
    for region_name in region_counts:
        rows = [
            finalize_pair(pair, counts)
            for pair, counts in pair_stats[region_name].items()
            if counts["changed"] > 0
        ]
        regions_out[region_name] = finalize_region_counts(region_counts[region_name])
        pairs_out[region_name] = {
            "top_positive_by_net": sorted(rows, key=lambda row: row["net_repaired"], reverse=True)[
                : args.top_k
            ],
            "top_negative_by_net": sorted(rows, key=lambda row: row["net_repaired"])[: args.top_k],
            "top_by_changed": sorted(rows, key=lambda row: row["changed"], reverse=True)[: args.top_k],
        }

    output = {
        "args": vars(args),
        "sample_count": len(records),
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
    for region_name, counts in regions_out.items():
        print(
            f"{region_name}: changed={counts['changed']}, repaired={counts['repaired']}, "
            f"damaged={counts['damaged']}, net={counts['net_repaired']}"
        )
        for row in pairs_out[region_name]["top_positive_by_net"][:5]:
            print(
                f"  + {row['pair']}: net={row['net_repaired']}, "
                f"repair={row['repaired']}, damage={row['damaged']}, changed={row['changed']}"
            )
        for row in pairs_out[region_name]["top_negative_by_net"][:5]:
            print(
                f"  - {row['pair']}: net={row['net_repaired']}, "
                f"repair={row['repaired']}, damage={row['damaged']}, changed={row['changed']}"
            )


if __name__ == "__main__":
    main()
