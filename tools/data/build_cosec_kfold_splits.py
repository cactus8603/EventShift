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

from cosec_finetune_splits import (  # noqa: E402
    DEFAULT_COSEC_TRAIN_ROOT,
    DEFAULT_KFOLD_COUNT,
    SPLIT_DIR,
    build_cosec_kfold_sequence_sets,
    cosec_domain,
    iter_cosec_samples,
    iter_cosec_sequence_infos,
    mmseg_split_name,
)


def _sample_ids(root, split):
    return {
        mmseg_split_name(seq_name, frame_id)
        for seq_name, frame_id, _, _ in iter_cosec_samples(root=root, split=split)
    }


def _stats_for_sequences(sequences, info_by_seq):
    stats = {
        "sequences": len(sequences),
        "frames": 0,
        "day_sequences": 0,
        "day_frames": 0,
        "night_sequences": 0,
        "night_frames": 0,
    }
    for seq_name in sorted(sequences):
        info = info_by_seq[seq_name]
        frames = int(info["frame_count"])
        stats["frames"] += frames
        domain = info["domain"]
        stats[f"{domain}_sequences"] += 1
        stats[f"{domain}_frames"] += frames
    return stats


def _fold_summary(root, folds, fold_index, fold_sequences, info_by_seq):
    prefix = f"kfold{folds}_fold{fold_index}"
    train_split = f"{prefix}_train"
    val_split = f"{prefix}_val"
    train_ids = _sample_ids(root, train_split)
    val_ids = _sample_ids(root, val_split)
    train_sequences = {sample_id.split("/", 1)[0] for sample_id in train_ids}
    val_sequences = {sample_id.split("/", 1)[0] for sample_id in val_ids}
    train_stats = _stats_for_sequences(train_sequences, info_by_seq)
    val_stats = _stats_for_sequences(val_sequences, info_by_seq)
    return {
        "fold": fold_index,
        "datasets": {
            "train": f"cosec_{train_split}",
            "val": f"cosec_{val_split}",
            "day_train": f"cosec_{prefix}_day_train",
            "day_val": f"cosec_{prefix}_day_val",
            "night_train": f"cosec_{prefix}_night_train",
            "night_val": f"cosec_{prefix}_night_val",
            "train_event": f"cosec_{train_split}_event",
            "val_event": f"cosec_{val_split}_event",
        },
        "train": train_stats,
        "val": val_stats,
        "val_sequences": sorted(fold_sequences["val"]),
        "sequence_overlap": sorted(train_sequences & val_sequences),
        "frame_overlap_count": len(train_ids & val_ids),
        "missing_val_domains": [
            domain
            for domain in ("day", "night")
            if val_stats[f"{domain}_sequences"] == 0
        ],
        "train_loader_frame_count": len(train_ids),
        "val_loader_frame_count": len(val_ids),
    }


def _write_split_file(path, sample_ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample_id in sorted(sample_ids):
            f.write(f"{sample_id}\n")


def _write_split_files(root, folds, split_dir):
    split_dir = Path(split_dir)
    written = []
    for fold_index in range(folds):
        prefix = f"kfold{folds}_fold{fold_index}"
        for subset in ("train", "val"):
            split = f"{prefix}_{subset}"
            path = split_dir / f"{subset}_{prefix}.txt"
            _write_split_file(path, _sample_ids(root, split))
            written.append(str(path))
    return written


def _format_domain_stats(stats):
    return (
        f"{stats['sequences']} seq/{stats['frames']} fr "
        f"(day {stats['day_sequences']}/{stats['day_frames']}, "
        f"night {stats['night_sequences']}/{stats['night_frames']})"
    )


def _print_human(payload):
    print(
        f"CoSEC domain-aware sequence-level k-fold summary: "
        f"root={payload['root']} folds={payload['folds']}"
    )
    print("fold | train | val | overlap(seq/frame) | missing val domains | primary datasets")
    print("-" * 120)
    for summary in payload["folds_summary"]:
        overlap = f"{len(summary['sequence_overlap'])}/{summary['frame_overlap_count']}"
        missing_domains = ",".join(summary["missing_val_domains"]) if summary["missing_val_domains"] else "-"
        print(
            f"{summary['fold']} | "
            f"{_format_domain_stats(summary['train'])} | "
            f"{_format_domain_stats(summary['val'])} | "
            f"{overlap} | "
            f"{missing_domains} | "
            f"{summary['datasets']['train']} -> {summary['datasets']['val']}"
        )
        print(f"  val sequences: {', '.join(summary['val_sequences'])}")
    if payload["written_files"]:
        print("written split files:")
        for path in payload["written_files"]:
            print(f"  {path}")
    print(f"overlap check: {'PASS' if payload['overlap_ok'] else 'FAIL'}")
    print(f"val domain coverage check: {'PASS' if payload['val_domain_ok'] else 'FAIL'}")
    print(f"overall check: {'PASS' if payload['ok'] else 'FAIL'}")


def main():
    parser = argparse.ArgumentParser(description="Build/check CoSEC domain-aware sequence-level k-fold splits.")
    parser.add_argument("--root", default=str(DEFAULT_COSEC_TRAIN_ROOT), help="CoSEC train root with sequence dirs.")
    parser.add_argument("--folds", type=int, default=DEFAULT_KFOLD_COUNT, help="Number of sequence folds.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument("--write-splits", action="store_true", help="Write train/val frame-list txt files for audit.")
    parser.add_argument("--split-dir", default=str(SPLIT_DIR), help="Output directory for --write-splits.")
    args = parser.parse_args()

    root = Path(args.root)
    info_by_seq = {info["seq_name"]: info for info in iter_cosec_sequence_infos(root)}
    try:
        fold_sets = build_cosec_kfold_sequence_sets(root=root, folds=args.folds)
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    summaries = [
        _fold_summary(root, args.folds, fold_index, fold_sequences, info_by_seq)
        for fold_index, fold_sequences in enumerate(fold_sets)
    ]
    written_files = _write_split_files(root, args.folds, args.split_dir) if args.write_splits else []
    overlap_ok = all(
        not summary["sequence_overlap"] and summary["frame_overlap_count"] == 0
        for summary in summaries
    )
    val_domain_ok = all(not summary["missing_val_domains"] for summary in summaries)
    payload = {
        "root": str(root),
        "mode": "domain-aware sequence-level",
        "folds": args.folds,
        "sequence_count": len(info_by_seq),
        "day_sequence_count": sum(1 for info in info_by_seq.values() if cosec_domain(info["seq_name"]) == "day"),
        "night_sequence_count": sum(1 for info in info_by_seq.values() if cosec_domain(info["seq_name"]) == "night"),
        "folds_summary": summaries,
        "written_files": written_files,
        "overlap_ok": overlap_ok,
        "val_domain_ok": val_domain_ok,
        "ok": overlap_ok and val_domain_ok,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_human(payload)
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
