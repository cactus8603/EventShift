#!/usr/bin/env python
"""Summarize REAL pseudo-label support after event-based filtering."""

import argparse
import json
import sys
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image

def _eventshift_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "configs").is_dir() and (parent / "third_party").is_dir():
            return parent
    return Path(__file__).resolve().parents[1]


ROOT = _eventshift_root()
sys.path.insert(0, str(ROOT / "tools"))

from cosec_finetune_splits import CLASSES  # noqa: E402
from pseudo_dataset import IGNORE_LABEL, load_real_pool_pseudo_dicts  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=["swinl", "swinl_eventedge", "swinl_eventactive"])
    parser.add_argument("--threshold", type=int, default=224)
    parser.add_argument("--limit", type=int, default=600)
    parser.add_argument("--min-valid-fraction", type=float, default=0.01)
    parser.add_argument(
        "--out",
        default=str(ROOT / "work_dirs" / "diagnostics" / "real_event_pseudo_support.json"),
    )
    return parser.parse_args()


def _read_label(path):
    array = np.asarray(Image.open(path))
    if array.ndim == 3:
        array = array[:, :, 0]
    return array.astype(np.uint8, copy=False)


def summarize_variant(variant, threshold, limit, min_valid_fraction):
    records = list(
        load_real_pool_pseudo_dicts(
            variant,
            threshold,
            repeat=1,
            limit=limit,
            min_valid_fraction=min_valid_fraction,
        )
    )
    valid_fractions = []
    class_pixels = Counter()
    sequence_counts = Counter()
    total_valid = 0
    total_pixels = 0

    for record in records:
        label = _read_label(record["sem_seg_file_name"])
        valid = label != IGNORE_LABEL
        valid_fraction = float(valid.mean())
        valid_fractions.append(valid_fraction)
        total_valid += int(valid.sum())
        total_pixels += int(label.size)
        sequence_counts[record.get("real_sequence", "unknown")] += 1
        values, counts = np.unique(label[valid], return_counts=True)
        for value, count in zip(values.tolist(), counts.tolist()):
            class_pixels[int(value)] += int(count)

    valid_array = np.asarray(valid_fractions, dtype=np.float64)
    class_distribution = OrderedDict()
    for class_id, class_name in enumerate(CLASSES):
        count = class_pixels[class_id]
        class_distribution[class_name] = {
            "pixels": count,
            "fraction_of_valid": float(count / total_valid) if total_valid else 0.0,
        }

    return {
        "variant": variant,
        "threshold": int(threshold),
        "limit": int(limit),
        "min_valid_fraction": float(min_valid_fraction),
        "num_records": len(records),
        "num_sequences": len(sequence_counts),
        "total_valid_pixels": int(total_valid),
        "total_pixels": int(total_pixels),
        "valid_fraction_mean": float(valid_array.mean()) if valid_array.size else 0.0,
        "valid_fraction_median": float(np.median(valid_array)) if valid_array.size else 0.0,
        "valid_fraction_min": float(valid_array.min()) if valid_array.size else 0.0,
        "valid_fraction_max": float(valid_array.max()) if valid_array.size else 0.0,
        "sequence_counts": dict(sorted(sequence_counts.items())),
        "class_distribution": class_distribution,
    }


def write_markdown(report, out_path):
    lines = [
        "# REAL event pseudo-label support",
        "",
        "REAL_dataset/gt is RGB imagery; these masks are pseudo labels from prior_swinL_ft.",
        "Event variants use events.h5 only to select which pseudo-label pixels are reliable enough to train on.",
        "",
        "| variant | records | sequences | valid mean | valid median | valid min | valid max |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["variants"].values():
        lines.append(
            "| {variant} | {num_records} | {num_sequences} | {mean:.4f} | {median:.4f} | {minv:.4f} | {maxv:.4f} |".format(
                variant=item["variant"],
                num_records=item["num_records"],
                num_sequences=item["num_sequences"],
                mean=item["valid_fraction_mean"],
                median=item["valid_fraction_median"],
                minv=item["valid_fraction_min"],
                maxv=item["valid_fraction_max"],
            )
        )
    lines.extend(["", "## Top Classes"])
    for variant, item in report["variants"].items():
        top_classes = sorted(
            item["class_distribution"].items(),
            key=lambda kv: kv[1]["pixels"],
            reverse=True,
        )[:8]
        rendered = ", ".join(
            f"{name} {100.0 * stats['fraction_of_valid']:.1f}%"
            for name, stats in top_classes
            if stats["pixels"] > 0
        )
        lines.append(f"- `{variant}`: {rendered}")
    out_path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    report = {
        "threshold": int(args.threshold),
        "limit": int(args.limit),
        "min_valid_fraction": float(args.min_valid_fraction),
        "variants": OrderedDict(),
    }
    for variant in args.variants:
        report["variants"][variant] = summarize_variant(
            variant,
            args.threshold,
            args.limit,
            args.min_valid_fraction,
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    write_markdown(report, out_path)

    print(f"Wrote: {out_path}")
    print(f"Wrote: {out_path.with_suffix('.md')}")
    for variant, item in report["variants"].items():
        print(
            f"{variant}: records={item['num_records']} "
            f"valid_mean={item['valid_fraction_mean']:.4f} "
            f"valid_median={item['valid_fraction_median']:.4f}"
        )


if __name__ == "__main__":
    main()
