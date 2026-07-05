#!/usr/bin/env python
"""Summarize multi-scale oracle and routing headroom diagnostics."""

import argparse
import json
import math
from pathlib import Path


DEFAULT_SCALE_ENSEMBLE = "work_dirs/diagnostics/scale_ensemble_routing_dayonly65_43_s512_624_768_1024.json"
DEFAULT_DAY_CONF = "work_dirs/diagnostics/scale_confidence_gated_day_full.json"
DEFAULT_NIGHT_CONF = "work_dirs/diagnostics/scale_confidence_gated_night_full.json"
DEFAULT_TTA_DAYNIGHT = "work_dirs/diagnostics/scale_tta_sets_daynight_limit48_flip.json"
DEFAULT_TTA_NIGHT = "work_dirs/diagnostics/scale_tta_sets_night_limit96_flip.json"

SPLIT_NAMES = {
    "cosec_day_val": "Day",
    "cosec_night_val": "Night",
}


def valid_number(value):
    return isinstance(value, (int, float)) and not math.isnan(float(value))


def fmt(value, digits=4):
    if not valid_number(value):
        return ""
    return f"{float(value):.{digits}f}"


def fmt_pct(value):
    if not valid_number(value):
        return ""
    return f"{100.0 * float(value):.3f}%"


def read_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def method_by_name(methods, name):
    if isinstance(methods, dict):
        return methods.get(name)
    for row in methods:
        if row.get("method") == name:
            return row
    return None


def best_non_oracle_method(methods):
    banned = {"pixel_oracle_any_scale_correct", "image_oracle", "sequence_oracle"}
    rows = methods.values() if isinstance(methods, dict) else methods
    rows = [row for row in rows if row.get("method") not in banned]
    rows = [row for row in rows if valid_number(row.get("mIoU"))]
    return max(rows, key=lambda row: row["mIoU"]) if rows else None


def dataset_rows(scale_ensemble, conf_by_split):
    rows = []
    for dataset in scale_ensemble["datasets"]:
        dataset_name = dataset["dataset"]
        split = SPLIT_NAMES.get(dataset_name, dataset_name)
        methods = dataset["methods"]
        oracle = method_by_name(methods, "pixel_oracle_any_scale_correct")
        best_branch = best_non_oracle_method(methods)

        conf = conf_by_split.get(dataset_name)
        tta_anchor = method_by_name(conf.get("methods", {}), "anchor_tta4flip") if conf else None
        best_conf = conf.get("top_by_mIoU", [{}])[0] if conf else None

        rows.append(
            {
                "split": split,
                "sample_count": dataset.get("sample_count"),
                "tta_anchor": tta_anchor,
                "best_conf": best_conf,
                "best_branch": best_branch,
                "oracle": oracle,
            }
        )
    return rows


def print_main_table(rows):
    print("| Split | Samples | TTA Anchor | Best Rule | Rule Gain | Rule Changed | Pixel Oracle | Oracle Gain | Oracle Changed |")
    print("|---|---:|---:|---|---:|---:|---:|---:|---:|")
    for row in rows:
        tta = row["tta_anchor"]
        best_rule = row["best_conf"]
        oracle = row["oracle"]
        tta_miou = tta.get("mIoU") if tta else None
        best_rule_miou = best_rule.get("mIoU") if best_rule else None
        oracle_miou = oracle.get("mIoU") if oracle else None
        print(
            f"| {row['split']} | {row['sample_count']} | {fmt(tta_miou)} | "
            f"`{best_rule.get('method', '') if best_rule else ''}` | "
            f"{fmt(best_rule_miou - tta_miou if valid_number(best_rule_miou) and valid_number(tta_miou) else None)} | "
            f"{fmt_pct(best_rule.get('changed_rate') if best_rule else None)} | "
            f"{fmt(oracle_miou)} | "
            f"{fmt(oracle_miou - tta_miou if valid_number(oracle_miou) and valid_number(tta_miou) else None)} | "
            f"{fmt_pct(oracle.get('changed_rate') if oracle else None)} |"
        )


def print_oracle_class_table(rows, top_k):
    print()
    print("| Split | Class | TTA IoU | Oracle IoU | Gain |")
    print("|---|---|---:|---:|---:|")
    for row in rows:
        tta = row["tta_anchor"] or {}
        oracle = row["oracle"] or {}
        tta_iou = tta.get("class_iou", {})
        oracle_iou = oracle.get("class_iou", {})
        deltas = []
        for cls, oracle_value in oracle_iou.items():
            tta_value = tta_iou.get(cls)
            if valid_number(oracle_value) and valid_number(tta_value):
                deltas.append((cls, float(tta_value), float(oracle_value), float(oracle_value) - float(tta_value)))
        deltas.sort(key=lambda item: item[3], reverse=True)
        for cls, tta_value, oracle_value, delta in deltas[:top_k]:
            print(f"| {row['split']} | {cls} | {fmt(tta_value, 2)} | {fmt(oracle_value, 2)} | {fmt(delta, 2)} |")


def print_tta_set_table(tta_files):
    print()
    print("| Diagnostic | Dataset | Samples | Best TTA Set | mIoU | All Sets |")
    print("|---|---|---:|---|---:|---|")
    for path in tta_files:
        data = read_json(path)
        for dataset in data.get("datasets", []):
            rows = dataset.get("top_by_mIoU", [])
            if not rows:
                continue
            best = rows[0]
            all_sets = ", ".join(f"{row['method']}={row['mIoU']:.4f}" for row in rows)
            print(
                f"| `{Path(path).name}` | {dataset['dataset']} | {dataset['sample_count']} | "
                f"`{best['method']}` | {fmt(best['mIoU'])} | {all_sets} |"
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale-ensemble", default=DEFAULT_SCALE_ENSEMBLE)
    parser.add_argument("--day-conf", default=DEFAULT_DAY_CONF)
    parser.add_argument("--night-conf", default=DEFAULT_NIGHT_CONF)
    parser.add_argument("--tta-files", nargs="*", default=[DEFAULT_TTA_DAYNIGHT, DEFAULT_TTA_NIGHT])
    parser.add_argument("--top-k-classes", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    scale_ensemble = read_json(args.scale_ensemble)
    conf_by_split = {
        "cosec_day_val": read_json(args.day_conf),
        "cosec_night_val": read_json(args.night_conf),
    }
    rows = dataset_rows(scale_ensemble, conf_by_split)

    print("# Scale Headroom Summary")
    print()
    print_main_table(rows)
    print_oracle_class_table(rows, args.top_k_classes)
    print_tta_set_table(args.tta_files)


if __name__ == "__main__":
    main()
