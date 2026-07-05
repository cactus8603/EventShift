#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path


def _eventshift_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "configs").is_dir() and (parent / "third_party").is_dir():
            return parent
    return Path(__file__).resolve().parents[1]


ROOT = _eventshift_root()
sys.path.insert(0, str(ROOT / "tools"))

from acdc_dataset import (  # noqa: E402
    DEFAULT_ACDC_KFOLD_COUNT,
    acdc_root,
    build_acdc_kfold_sequence_sets,
    iter_acdc_sequence_infos,
    load_acdc_kfold_dicts,
)


def _stats_for_keys(keys, info_by_key):
    stats = {
        "sequences": len(keys),
        "frames": 0,
        "conditions": {},
    }
    for key in sorted(keys):
        info = info_by_key[key]
        condition = info["condition"]
        condition_stats = stats["conditions"].setdefault(condition, {"sequences": 0, "frames": 0})
        condition_stats["sequences"] += 1
        condition_stats["frames"] += int(info["record_count"])
        stats["frames"] += int(info["record_count"])
    return stats


def _fold_summary(condition, folds, fold_index, fold_sequences, info_by_key):
    train_records = load_acdc_kfold_dicts(condition, folds, fold_index, "train")
    val_records = load_acdc_kfold_dicts(condition, folds, fold_index, "val")
    train_keys = {f"{record['acdc_condition']}:{record['acdc_sequence']}" for record in train_records}
    val_keys = {f"{record['acdc_condition']}:{record['acdc_sequence']}" for record in val_records}
    prefix = f"acdc_{condition}_kfold{folds}_fold{fold_index}"
    return {
        "fold": fold_index,
        "datasets": {
            "train": f"{prefix}_train",
            "val": f"{prefix}_val",
        },
        "train": _stats_for_keys(train_keys, info_by_key),
        "val": _stats_for_keys(val_keys, info_by_key),
        "val_sequences": sorted(fold_sequences["val"]),
        "sequence_overlap": sorted(train_keys & val_keys),
        "frame_overlap_count": len(
            {(record["file_name"], record["sem_seg_file_name"]) for record in train_records}
            & {(record["file_name"], record["sem_seg_file_name"]) for record in val_records}
        ),
        "train_loader_frame_count": len(train_records),
        "val_loader_frame_count": len(val_records),
    }


def _format_stats(stats):
    parts = [f"{stats['sequences']} seq/{stats['frames']} fr"]
    for condition, condition_stats in sorted(stats["conditions"].items()):
        parts.append(f"{condition} {condition_stats['sequences']}/{condition_stats['frames']}")
    return " (" + ", ".join(parts) + ")"


def _print_human(payload):
    print(
        f"ACDC condition-aware sequence-level k-fold summary: "
        f"root={payload['root']} condition={payload['condition']} folds={payload['folds']}"
    )
    print("fold | train | val | overlap(seq/frame) | primary datasets")
    print("-" * 110)
    for summary in payload["folds_summary"]:
        overlap = f"{len(summary['sequence_overlap'])}/{summary['frame_overlap_count']}"
        print(
            f"{summary['fold']} | "
            f"{_format_stats(summary['train'])} | "
            f"{_format_stats(summary['val'])} | "
            f"{overlap} | "
            f"{summary['datasets']['train']} -> {summary['datasets']['val']}"
        )
        print(f"  val sequences: {', '.join(summary['val_sequences'])}")
    print(f"overlap check: {'PASS' if payload['overlap_ok'] else 'FAIL'}")
    print(f"overall check: {'PASS' if payload['ok'] else 'FAIL'}")


def main():
    parser = argparse.ArgumentParser(description="Build/check ACDC sequence-level k-fold splits.")
    parser.add_argument("--condition", default="night", help="ACDC condition, e.g. night or all.")
    parser.add_argument("--folds", type=int, default=DEFAULT_ACDC_KFOLD_COUNT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    info_by_key = {info["key"]: info for info in iter_acdc_sequence_infos(args.condition)}
    try:
        fold_sets = build_acdc_kfold_sequence_sets(args.condition, folds=args.folds)
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    summaries = [
        _fold_summary(args.condition, args.folds, fold_index, fold_sequences, info_by_key)
        for fold_index, fold_sequences in enumerate(fold_sets)
    ]
    overlap_ok = all(
        not summary["sequence_overlap"] and summary["frame_overlap_count"] == 0
        for summary in summaries
    )
    payload = {
        "root": str(acdc_root()),
        "condition": args.condition,
        "mode": "condition-aware sequence-level",
        "folds": args.folds,
        "sequence_count": len(info_by_key),
        "folds_summary": summaries,
        "overlap_ok": overlap_ok,
        "ok": overlap_ok,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_human(payload)
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
