#!/usr/bin/env python
"""Small-grid day routing diagnostics for event-edge guided segmentation.

This intentionally does not train. It tests whether a candidate event model
should replace a frozen RGB baseline only on event-edge regions selected by
RGB uncertainty, candidate confidence, and class filters.
"""

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
    ConfusionMeter,
    add_counts,
    build_model,
    empty_counts,
    finalize_counts,
    infer_mapped,
    load_label,
    normalize_scores,
    resize_stat,
    setup_cfg,
    split_csv,
    top_conf_margin,
    update_method,
    valid_label_mask,
)


GOOD_CLASSES = ["road", "building", "pole", "person", "rider"]
RISKY_CLASSES = [
    "sidewalk",
    "traffic sign",
    "vegetation",
    "sky",
    "car",
    "motorcycle",
    "bicycle",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--base-weights", required=True)
    parser.add_argument("--candidate-configs", required=True)
    parser.add_argument("--candidate-weights", required=True)
    parser.add_argument("--candidate-names", default="")
    parser.add_argument("--event-config", required=True)
    parser.add_argument("--dataset", default="cosec_day_val_event")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--full-grid", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def percentile_mask(values, valid, q, low=True):
    selected = values[valid]
    if selected.size == 0:
        return np.zeros_like(valid, dtype=bool)
    threshold = np.percentile(selected, float(q))
    return valid & ((values <= threshold) if low else (values >= threshold))


def entropy_from_prob(prob):
    entropy = -(prob * prob.clamp_min(1e-8).log()).sum(dim=0)
    return (entropy / math.log(prob.shape[0])).cpu().numpy()


def class_mask(pred, names):
    ids = np.array([CLASSES.index(name) for name in names], dtype=np.int64)
    return np.isin(pred, ids)


def method_bucket(methods, name):
    if name not in methods:
        methods[name] = {"meter": ConfusionMeter(num_classes=len(CLASSES)), "counts": empty_counts()}
    return methods[name]


def update_oracle(methods, name, pred, base_pred, label, valid, support):
    update_method(methods, name, pred, base_pred, label, valid, support)


def route_pred(base_pred, cand_pred, route):
    return np.where(route, cand_pred, base_pred)


def pair_key(base_cls, cand_cls):
    return f"{CLASSES[int(base_cls)]}->{CLASSES[int(cand_cls)]}"


def add_pair_stats(pair_stats, cand_name, base_pred, cand_pred, label, valid, support):
    changed = valid & support & (base_pred != cand_pred)
    if not changed.any():
        return
    pair_index = len(CLASSES) * base_pred[changed] + cand_pred[changed]
    repaired_mask = changed & (base_pred != label) & (cand_pred == label)
    damaged_mask = changed & (base_pred == label) & (cand_pred != label)
    changed_counts = np.bincount(pair_index, minlength=len(CLASSES) ** 2)
    repaired_counts = np.bincount(
        len(CLASSES) * base_pred[repaired_mask] + cand_pred[repaired_mask],
        minlength=len(CLASSES) ** 2,
    )
    damaged_counts = np.bincount(
        len(CLASSES) * base_pred[damaged_mask] + cand_pred[damaged_mask],
        minlength=len(CLASSES) ** 2,
    )
    for idx in np.flatnonzero(changed_counts):
        base_cls = idx // len(CLASSES)
        cand_cls = idx % len(CLASSES)
        item = pair_stats[(cand_name, pair_key(base_cls, cand_cls))]
        item["changed"] += int(changed_counts[idx])
        item["repaired"] += int(repaired_counts[idx])
        item["damaged"] += int(damaged_counts[idx])


def finalize_methods(methods):
    output = OrderedDict()
    for name, value in methods.items():
        output[name] = {
            **value["meter"].metrics(),
            **finalize_counts(value["counts"]),
        }
    return output


def main():
    args = parse_args()
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    register_cosec()

    candidate_configs = split_csv(args.candidate_configs)
    candidate_weights = split_csv(args.candidate_weights)
    if len(candidate_configs) != len(candidate_weights):
        raise ValueError("--candidate-configs and --candidate-weights length mismatch")
    candidate_names = split_csv(args.candidate_names)
    if not candidate_names:
        candidate_names = [f"cand{idx}" for idx in range(len(candidate_configs))]
    if len(candidate_names) != len(candidate_configs):
        raise ValueError("--candidate-names length mismatch")

    base_cfg = setup_cfg(args.base_config, args.base_weights, args.device)
    event_cfg = setup_cfg(args.event_config, args.base_weights, args.device)
    candidate_cfgs = [
        setup_cfg(config, weights, args.device)
        for config, weights in zip(candidate_configs, candidate_weights)
    ]

    mapper = MaskFormerSemanticDatasetMapper(event_cfg, False)
    base_model = build_model(base_cfg)
    candidate_models = [build_model(cfg) for cfg in candidate_cfgs]

    records = DatasetCatalog.get(args.dataset)
    if args.limit is not None:
        records = records[: args.limit]

    methods = OrderedDict()
    pair_stats = defaultdict(lambda: {"changed": 0, "repaired": 0, "damaged": 0})
    iterator = records if args.quiet else tqdm(records, desc=args.dataset)
    for record in iterator:
        label = load_label(record)
        mapped = mapper(copy.deepcopy(record))
        event_stats = mapped["event_stats"].float()
        valid = valid_label_mask(label)
        raw_event = valid & (resize_stat(event_stats, 0, label.shape) > 0)
        support = valid & (resize_stat(event_stats, 3, label.shape) > 0)

        base_prob = normalize_scores(infer_mapped(base_model, mapped))
        cand_probs = [normalize_scores(infer_mapped(model, mapped)) for model in candidate_models]
        probs = [base_prob] + cand_probs
        preds = [prob.argmax(dim=0).numpy() for prob in probs]
        confs, margins = zip(*(top_conf_margin(prob) for prob in probs))
        entropies = [entropy_from_prob(prob) for prob in probs]

        base_pred = preds[0]
        base_conf = confs[0]
        base_margin = margins[0]
        base_entropy = entropies[0]
        base_good = class_mask(base_pred, GOOD_CLASSES)
        base_risky = class_mask(base_pred, RISKY_CLASSES)

        update_method(methods, "base", base_pred, base_pred, label, valid, support)

        if args.full_grid:
            event_regions = OrderedDict([("support", support), ("raw", raw_event)])
            uncertainty_masks = OrderedDict(
                [
                    ("all", valid),
                    ("lowmargin20", percentile_mask(base_margin, valid, 20, low=True)),
                    ("lowmargin40", percentile_mask(base_margin, valid, 40, low=True)),
                    ("lowconf20", percentile_mask(base_conf, valid, 20, low=True)),
                    ("highentropy80", percentile_mask(base_entropy, valid, 80, low=False)),
                    ("highentropy90", percentile_mask(base_entropy, valid, 90, low=False)),
                ]
            )
        else:
            event_regions = OrderedDict([("support", support)])
            uncertainty_masks = OrderedDict(
                [
                    ("all", valid),
                    ("lowmargin40", percentile_mask(base_margin, valid, 40, low=True)),
                    ("highentropy80", percentile_mask(base_entropy, valid, 80, low=False)),
                ]
            )

        for cand_idx, cand_name in enumerate(candidate_names, start=1):
            cand_pred = preds[cand_idx]
            cand_conf = confs[cand_idx]
            cand_margin = margins[cand_idx]
            cand_entropy = entropies[cand_idx]
            cand_good = class_mask(cand_pred, GOOD_CLASSES)
            cand_risky = class_mask(cand_pred, RISKY_CLASSES)
            changed = base_pred != cand_pred

            update_method(methods, cand_name, cand_pred, base_pred, label, valid, support)
            add_pair_stats(pair_stats, cand_name, base_pred, cand_pred, label, valid, support)

            if args.full_grid:
                class_masks = OrderedDict(
                    [
                        ("candgood", cand_good),
                        ("candgood_nobaserisky", cand_good & ~base_risky),
                        ("goodchange_nobaserisky", (cand_good | base_good) & ~base_risky),
                        ("candgood_fromrisky", cand_good & base_risky),
                        ("norisky_to_good", ~cand_risky & cand_good),
                    ]
                )
                trust_masks = OrderedDict(
                    [
                        ("changed", changed),
                        ("conf_ge_base_m02", changed & (cand_conf >= base_conf - 0.02)),
                        ("conf_ge_base", changed & (cand_conf >= base_conf)),
                        ("margin_ge_base", changed & (cand_margin >= base_margin)),
                        ("entropy_le_base", changed & (cand_entropy <= base_entropy)),
                    ]
                )
            else:
                class_masks = OrderedDict(
                    [
                        ("candgood_nobaserisky", cand_good & ~base_risky),
                        ("candgood_fromrisky", cand_good & base_risky),
                    ]
                )
                trust_masks = OrderedDict(
                    [
                        ("changed", changed),
                        ("conf_ge_base_m02", changed & (cand_conf >= base_conf - 0.02)),
                        ("margin_ge_base", changed & (cand_margin >= base_margin)),
                    ]
                )

            for region_name, region in event_regions.items():
                for unc_name, unc_mask in uncertainty_masks.items():
                    for class_name, cls_mask in class_masks.items():
                        for trust_name, trust_mask in trust_masks.items():
                            route = region & unc_mask & cls_mask & trust_mask
                            if not route.any():
                                continue
                            name = f"{cand_name}_{region_name}_{unc_name}_{class_name}_{trust_name}"
                            update_method(
                                methods,
                                name,
                                route_pred(base_pred, cand_pred, route),
                                base_pred,
                                label,
                                valid,
                                support,
                            )

            oracle_route = support & changed & (cand_pred == label) & (base_pred != label)
            update_oracle(
                methods,
                f"oracle_{cand_name}_support_repair_only",
                route_pred(base_pred, cand_pred, oracle_route),
                base_pred,
                label,
                valid,
                support,
            )

        if len(candidate_names) >= 2:
            cand_stack = np.stack(preds[1:], axis=0)
            for region_name, region in event_regions.items():
                for unc_name, unc_mask in uncertainty_masks.items():
                    vote_pred = np.full_like(base_pred, fill_value=-1)
                    vote_count = np.zeros_like(base_pred, dtype=np.int16)
                    for class_id in range(len(CLASSES)):
                        count = (cand_stack == class_id).sum(axis=0)
                        take = count > vote_count
                        vote_pred[take] = class_id
                        vote_count[take] = count[take]
                    vote_good = class_mask(vote_pred, GOOD_CLASSES)
                    route = (
                        region
                        & unc_mask
                        & (vote_count >= 2)
                        & vote_good
                        & ~base_risky
                        & (vote_pred != base_pred)
                    )
                    if route.any():
                        pred = route_pred(base_pred, vote_pred, route)
                        update_method(
                            methods,
                            f"vote2_{region_name}_{unc_name}_good_nobaserisky",
                            pred,
                            base_pred,
                            label,
                            valid,
                            support,
                        )

    method_output = finalize_methods(methods)
    non_oracle = {k: v for k, v in method_output.items() if not k.startswith("oracle_")}
    pair_output = []
    for (cand_name, pair), counts in pair_stats.items():
        net = counts["repaired"] - counts["damaged"]
        changed = counts["changed"]
        if changed == 0:
            continue
        pair_output.append(
            {
                "candidate": cand_name,
                "pair": pair,
                **counts,
                "net_repaired": net,
                "repair_precision": counts["repaired"] / changed,
                "damage_rate": counts["damaged"] / changed,
            }
        )
    pair_output.sort(key=lambda row: row["net_repaired"], reverse=True)

    output = {
        "args": vars(args),
        "sample_count": len(records),
        "good_classes": GOOD_CLASSES,
        "risky_classes": RISKY_CLASSES,
        "methods": method_output,
        "top_non_oracle_by_mIoU": [
            {"method": name, **values}
            for name, values in sorted(
                non_oracle.items(),
                key=lambda item: item[1]["mIoU"],
                reverse=True,
            )[:30]
        ],
        "top_by_mIoU": [
            {"method": name, **values}
            for name, values in sorted(
                method_output.items(),
                key=lambda item: item[1]["mIoU"],
                reverse=True,
            )[:30]
        ],
        "top_support_pairs_by_net": pair_output[:30],
        "worst_support_pairs_by_net": sorted(pair_output, key=lambda row: row["net_repaired"])[:30],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote diagnostics: {out_path}")
    print("Top non-oracle methods by mIoU:")
    for row in output["top_non_oracle_by_mIoU"][:10]:
        print(
            f"  {row['method']}: mIoU={row['mIoU']:.4f}, "
            f"net={row['net_repaired']}, repair={100 * row['repair_rate']:.2f}%, "
            f"changed={100 * row['changed_rate']:.4f}%"
        )
    print("Top support prediction pairs by net:")
    for row in output["top_support_pairs_by_net"][:10]:
        print(
            f"  {row['candidate']} {row['pair']}: net={row['net_repaired']}, "
            f"repair={row['repaired']}, damage={row['damaged']}, changed={row['changed']}"
        )


if __name__ == "__main__":
    main()
