#!/usr/bin/env python
"""Build a day Exp2 pair-aware config from pair repair diagnostics."""

import argparse
import json
from pathlib import Path

from cosec_finetune_splits import CLASSES


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-json", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--region", default="support_uncertain_boundary")
    parser.add_argument("--base-config", default="Mask2Former_SwinL_CoSEC_DayExp1C_StageC_Head.yaml")
    parser.add_argument("--weights", default="work_dirs/day-exp1c_stageC_head_from_stageB_strong_bs8/model_final.pth")
    parser.add_argument("--output-dir", default="work_dirs/day-exp2_event_score_pair_aware_from_stageC_head_bs8")
    parser.add_argument("--allow-min-net", type=int, default=50)
    parser.add_argument("--allow-min-repaired", type=int, default=80)
    parser.add_argument("--allow-min-precision", type=float, default=0.0)
    parser.add_argument("--suppress-max-net", type=int, default=-25)
    parser.add_argument("--suppress-min-damaged", type=int, default=40)
    parser.add_argument("--suppress-min-damage-rate", type=float, default=0.0)
    parser.add_argument("--pair-weight-default", type=float, default=None)
    parser.add_argument("--base-lr", type=float, default=0.000002)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--score-predictor", action="store_true")
    parser.add_argument("--hard-pair-gate", action="store_true")
    parser.add_argument("--hard-pair-suppress", action="store_true")
    parser.add_argument("--hard-pair-threshold", type=float, default=1.0)
    parser.add_argument("--hard-pair-include-identity", action="store_true")
    parser.add_argument("--event-active-source", default=None)
    parser.add_argument("--init-alpha", type=float, default=None)
    parser.add_argument("--gate-bias", type=float, default=None)
    parser.add_argument("--final-gate-scale", type=float, default=None)
    parser.add_argument("--topk-classes", type=int, default=None)
    parser.add_argument("--class-gate-all", action="store_true")
    parser.add_argument("--loss-boundary-ce-weight", type=float, default=1.0)
    parser.add_argument("--loss-uncertain-ce-weight", type=float, default=0.5)
    parser.add_argument("--repair-gate-bce-weight", type=float, default=0.05)
    parser.add_argument("--candidate-ce-weight", type=float, default=0.25)
    parser.add_argument("--preserve-kl-weight", type=float, default=4.0)
    parser.add_argument("--gate-invalid-weight", type=float, default=0.004)
    parser.add_argument("--delta-non-boundary-weight", type=float, default=0.0004)
    parser.add_argument("--pair-suppress-gate-weight", type=float, default=0.004)
    parser.add_argument("--pair-suppress-delta-weight", type=float, default=0.0004)
    parser.add_argument("--conservative", action="store_true")
    return parser.parse_args()


def matrix(default):
    return [[float(default) for _ in CLASSES] for _ in CLASSES]


def flatten(mat):
    return [mat[row][col] for row in range(len(CLASSES)) for col in range(len(CLASSES))]


def fmt_list(values, indent=6):
    pad = " " * indent
    lines = ["["]
    for start in range(0, len(values), len(CLASSES)):
        row = ", ".join(f"{value:.3g}" for value in values[start : start + len(CLASSES)])
        lines.append(f"{pad}{row},")
    lines.append(" " * (indent - 2) + "]")
    return "\n".join(lines)


def pair_ids(pair):
    left, right = pair.split("->", 1)
    return CLASSES.index(left), CLASSES.index(right)


def load_region_pairs(data, region):
    region_pairs = data["pairs"][region]
    if "all_pairs" in region_pairs:
        return list(region_pairs["all_pairs"])
    merged = {}
    for key in ("top_by_changed", "top_positive_by_net", "top_negative_by_net"):
        for item in region_pairs.get(key, []):
            merged[item["pair"]] = item
    return list(merged.values())


def main():
    args = parse_args()
    data = json.loads(Path(args.pair_json).read_text(encoding="utf-8"))
    pairs = load_region_pairs(data, args.region)

    default_pair_weight = (
        float(args.pair_weight_default)
        if args.pair_weight_default is not None
        else (0.35 if args.conservative else 0.5)
    )
    allow = matrix(default_pair_weight)
    suppress = matrix(0.0)
    selected_allow = []
    selected_suppress = []
    for item in pairs:
        src, dst = pair_ids(item["pair"])
        net = int(item["net_repaired"])
        repaired = int(item["repaired"])
        damaged = int(item["damaged"])
        changed = int(item["changed"])
        precision = repaired / max(changed, 1)
        if (
            net >= args.allow_min_net
            and repaired >= args.allow_min_repaired
            and precision >= args.allow_min_precision
        ):
            allow[src][dst] = 1.5 if precision >= 0.55 else 1.25
            selected_allow.append(item)
        damage_rate = damaged / max(changed, 1)
        if (
            net <= args.suppress_max_net
            and damaged >= args.suppress_min_damaged
            and damage_rate >= args.suppress_min_damage_rate
        ):
            suppress[src][dst] = 1.5 if damage_rate >= 0.65 else 1.0
            selected_suppress.append(item)

    # Same-class target should not be encouraged as a repair pair, but it should
    # stay neutral so preserve/Mask2Former losses can keep stable RGB regions.
    for idx in range(len(CLASSES)):
        allow[idx][idx] = 1.0

    score_enabled = "True" if args.score_predictor else "False"
    optional_lines = []
    if args.event_active_source is not None:
        optional_lines.append(f'    EVENT_ACTIVE_SOURCE: "{args.event_active_source}"')
    if args.init_alpha is not None:
        optional_lines.append(f"    INIT_ALPHA: {args.init_alpha}")
    if args.gate_bias is not None:
        optional_lines.append(f"    GATE_BIAS: {args.gate_bias}")
    if args.final_gate_scale is not None:
        optional_lines.append(f"    FINAL_GATE_SCALE: {args.final_gate_scale}")
    if args.topk_classes is not None:
        optional_lines.append(f"    TOPK_CLASSES: {args.topk_classes}")
    if args.class_gate_all:
        optional_lines.append(
            "    CLASS_GATE_WEIGHTS: ["
            + ", ".join("1.0" for _ in CLASSES)
            + "]"
        )
        optional_lines.append("    CLASS_GATE_LOSS_THRESHOLD: 0.0")
    optional_block = "\n".join(optional_lines)
    if optional_block:
        optional_block += "\n"
    text = f"""_BASE_: {args.base_config}
MODEL:
  WEIGHTS: "{args.weights}"
  TRAINABLE_PREFIXES:
    - "day_event_boundary_refiner."
  DAY_EVENT_BOUNDARY_REFINER:
{optional_block}    HARD_PAIR_GATE_ENABLED: {str(bool(args.hard_pair_gate))}
    HARD_PAIR_GATE_THRESHOLD: {args.hard_pair_threshold}
    HARD_PAIR_GATE_INCLUDE_IDENTITY: {str(bool(args.hard_pair_include_identity))}
    HARD_PAIR_SUPPRESS_ENABLED: {str(bool(args.hard_pair_suppress))}
    DETACH_BASE_PROB: True
    SKIP_MASK2FORMER_LOSS: True
    SCORE_PREDICTOR_ENABLED: {score_enabled}
    SCORE_BCE_WEIGHT: {0.04 if args.score_predictor else 0.0}
    SCORE_SPARSITY_WEIGHT: {0.002 if args.score_predictor else 0.0}
    SCORE_POSITIVE_WEIGHT: 2.0
    SCORE_NEGATIVE_WEIGHT: 1.0
    SELECTIVE_REPAIR_GATE_ENABLED: True
    LOSS_REPAIR_GATE_BCE_WEIGHT: {args.repair_gate_bce_weight}
    REPAIR_GATE_POSITIVE_WEIGHT: 5.0
    REPAIR_GATE_NEGATIVE_WEIGHT: 1.0
    REPAIR_REQUIRE_TARGET_IN_TOPK: True
    REPAIR_SUPERVISE_SCORE: False
    PAIR_AWARE_ENABLED: True
    PAIR_WEIGHT_DEFAULT: {default_pair_weight}
    LOSS_BOUNDARY_CE_WEIGHT: {args.loss_boundary_ce_weight}
    LOSS_UNCERTAIN_CE_WEIGHT: {args.loss_uncertain_ce_weight}
    LOSS_CANDIDATE_CE_WEIGHT: {args.candidate_ce_weight}
    LOSS_PRESERVE_KL_WEIGHT: {args.preserve_kl_weight}
    LOSS_GATE_INVALID_WEIGHT: {args.gate_invalid_weight}
    LOSS_DELTA_NON_BOUNDARY_WEIGHT: {args.delta_non_boundary_weight}
    LOSS_PAIR_SUPPRESS_GATE_WEIGHT: {args.pair_suppress_gate_weight}
    LOSS_PAIR_SUPPRESS_DELTA_WEIGHT: {args.pair_suppress_delta_weight}
    PAIR_ALLOW_WEIGHTS: {fmt_list(flatten(allow), indent=6)}
    PAIR_SUPPRESS_WEIGHTS: {fmt_list(flatten(suppress), indent=6)}
SOLVER:
  IMS_PER_BATCH: {args.batch_size}
  BASE_LR: {args.base_lr}
DATALOADER:
  NUM_WORKERS: {args.num_workers}
TRAIN:
  EPOCHS: {args.epochs}
OUTPUT_DIR: "{args.output_dir}"
"""

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")

    print(f"Wrote config: {out_path}")
    print("Allow pairs:")
    for item in selected_allow[:20]:
        print(f"  {item['pair']}: repaired={item['repaired']} damaged={item['damaged']} net={item['net_repaired']}")
    print("Suppress pairs:")
    for item in selected_suppress[:20]:
        print(f"  {item['pair']}: repaired={item['repaired']} damaged={item['damaged']} net={item['net_repaired']}")


if __name__ == "__main__":
    main()
