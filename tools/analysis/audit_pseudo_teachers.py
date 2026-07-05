#!/usr/bin/env python
"""Audit pseudo-label teacher independence before self-training.

This catches the bad case where a "consensus" pseudo dataset is actually built
from two identical mask folders, which behaves like single-teacher thresholding.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from cosec_finetune_splits import CLASSES


def _eventshift_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "configs").is_dir() and (parent / "third_party").is_dir():
            return parent
    return Path(__file__).resolve().parents[1]


ROOT = _eventshift_root()
IGNORE_LABEL = 255


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-root", default=str(ROOT / "data" / "test"))
    parser.add_argument("--domain", default="daynight", choices=["day", "night", "real", "daynight", "all"])
    parser.add_argument("--primary-label-dir", default="prior_swinL_ft")
    parser.add_argument("--primary-conf-dir", default="prior_swinL_ft_conf")
    parser.add_argument("--agree-label-dir", default="prior_mask2former_large_ft_cc_submission")
    parser.add_argument("--agree-conf-dir", default="prior_mask2former_large_ft_cc_submission_conf")
    parser.add_argument("--threshold", type=int, default=192)
    parser.add_argument("--limit-per-seq", type=int, default=0)
    parser.add_argument(
        "--output",
        default=str(ROOT / "work_dirs" / "diagnostics" / "pseudo_teacher_audit_daynight_consensus_conf192.json"),
    )
    return parser.parse_args()


def read_gray(path):
    array = np.asarray(Image.open(path))
    if array.ndim == 3:
        array = array[:, :, 0]
    return array


def prefixes_for_domain(domain):
    if domain == "day":
        return ("Day_",)
    if domain == "night":
        return ("Night_",)
    if domain == "real":
        return ("REAL_",)
    if domain == "daynight":
        return ("Day_", "Night_")
    if domain == "all":
        return ("Day_", "Night_", "REAL_")
    raise ValueError(domain)


def image_paths(seq_dir, limit_per_seq):
    paths = sorted((seq_dir / "img_co_left").glob("*.png"))
    if limit_per_seq and limit_per_seq > 0:
        paths = paths[:limit_per_seq]
    return paths


def pct(value, total):
    return float(value / total) if total else 0.0


def audit(args):
    test_root = Path(args.test_root)
    prefixes = prefixes_for_domain(args.domain)
    counts = Counter()
    class_hist = np.zeros(len(CLASSES), dtype=np.int64)
    examples = []
    per_sequence = []

    for seq_dir in sorted(path for path in test_root.iterdir() if path.is_dir() and path.name.startswith(prefixes)):
        seq_counts = Counter()
        for image_path in image_paths(seq_dir, args.limit_per_seq):
            stem = image_path.stem
            primary_label_path = seq_dir / args.primary_label_dir / f"{stem}.png"
            primary_conf_path = seq_dir / args.primary_conf_dir / f"{stem}.png"
            agree_label_path = seq_dir / args.agree_label_dir / f"{stem}.png"
            agree_conf_path = seq_dir / args.agree_conf_dir / f"{stem}.png"
            if not primary_label_path.exists() or not primary_conf_path.exists() or not agree_label_path.exists():
                counts["missing"] += 1
                seq_counts["missing"] += 1
                continue

            primary = read_gray(primary_label_path).astype(np.uint8, copy=False)
            agree = read_gray(agree_label_path).astype(np.uint8, copy=False)
            primary_conf = read_gray(primary_conf_path)
            agree_conf = read_gray(agree_conf_path) if agree_conf_path.exists() else None
            if primary.shape != agree.shape:
                counts["shape_mismatch"] += 1
                seq_counts["shape_mismatch"] += 1
                continue

            total = int(primary.size)
            same = primary == agree
            keep = primary_conf >= args.threshold
            if agree_conf is not None:
                keep &= agree_conf >= args.threshold
            keep &= same
            valid_class = (primary >= 0) & (primary < len(CLASSES))

            counts["images"] += 1
            counts["pixels"] += total
            counts["same_pixels"] += int(same.sum())
            counts["kept_pixels"] += int(keep.sum())
            counts["primary_conf_pixels"] += int((primary_conf >= args.threshold).sum())
            seq_counts["images"] += 1
            seq_counts["pixels"] += total
            seq_counts["same_pixels"] += int(same.sum())
            seq_counts["kept_pixels"] += int(keep.sum())
            seq_counts["primary_conf_pixels"] += int((primary_conf >= args.threshold).sum())

            if agree_conf is not None:
                conf_same = primary_conf == agree_conf
                counts["conf_pixels"] += total
                counts["same_conf_pixels"] += int(conf_same.sum())
                seq_counts["conf_pixels"] += total
                seq_counts["same_conf_pixels"] += int(conf_same.sum())

            class_hist += np.bincount(primary[valid_class].reshape(-1), minlength=len(CLASSES))[: len(CLASSES)]
            if len(examples) < 8 and pct(int(same.sum()), total) >= 0.9999:
                examples.append(f"{seq_dir.name}/{stem}.png")

        if seq_counts["images"]:
            per_sequence.append(
                {
                    "sequence": seq_dir.name,
                    "images": seq_counts["images"],
                    "label_agreement": pct(seq_counts["same_pixels"], seq_counts["pixels"]),
                    "kept_fraction": pct(seq_counts["kept_pixels"], seq_counts["pixels"]),
                    "primary_conf_fraction": pct(seq_counts["primary_conf_pixels"], seq_counts["pixels"]),
                    "confidence_agreement": pct(seq_counts["same_conf_pixels"], seq_counts["conf_pixels"]),
                }
            )

    hist_total = int(class_hist.sum())
    class_distribution = [
        {"class": name, "fraction": pct(int(class_hist[idx]), hist_total)}
        for idx, name in enumerate(CLASSES)
        if class_hist[idx] > 0
    ]
    class_distribution.sort(key=lambda row: row["fraction"], reverse=True)

    report = {
        "domain": args.domain,
        "primary_label_dir": args.primary_label_dir,
        "primary_conf_dir": args.primary_conf_dir,
        "agree_label_dir": args.agree_label_dir,
        "agree_conf_dir": args.agree_conf_dir,
        "threshold": args.threshold,
        "images": counts["images"],
        "missing": counts["missing"],
        "shape_mismatch": counts["shape_mismatch"],
        "label_agreement": pct(counts["same_pixels"], counts["pixels"]),
        "confidence_agreement": pct(counts["same_conf_pixels"], counts["conf_pixels"]),
        "primary_conf_fraction": pct(counts["primary_conf_pixels"], counts["pixels"]),
        "consensus_kept_fraction": pct(counts["kept_pixels"], counts["pixels"]),
        "identical_label_examples": examples,
        "top_classes": class_distribution[:10],
        "per_sequence": per_sequence,
    }
    return report


def write_report(report, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path = output_path.with_suffix(".md")
    lines = [
        "# Pseudo Teacher Audit",
        "",
        f"- domain: `{report['domain']}`",
        f"- primary: `{report['primary_label_dir']}` / `{report['primary_conf_dir']}`",
        f"- agree: `{report['agree_label_dir']}` / `{report['agree_conf_dir']}`",
        f"- threshold: `{report['threshold']}`",
        f"- images audited: {report['images']}",
        f"- label agreement: {report['label_agreement']:.6f}",
        f"- confidence agreement: {report['confidence_agreement']:.6f}",
        f"- primary confidence fraction: {report['primary_conf_fraction']:.6f}",
        f"- consensus kept fraction: {report['consensus_kept_fraction']:.6f}",
        "",
        "## Top Classes",
        "",
        "| class | fraction |",
        "|---|---:|",
    ]
    for row in report["top_classes"]:
        lines.append(f"| {row['class']} | {row['fraction']:.6f} |")
    lines.extend(["", "## Identical Label Examples", ""])
    for example in report["identical_label_examples"]:
        lines.append(f"- `{example}`")
    lines.append("")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def main():
    args = parse_args()
    report = audit(args)
    output_path = Path(args.output)
    md_path = write_report(report, output_path)
    print(f"Wrote {output_path}")
    print(f"Wrote {md_path}")
    print(
        "label_agreement="
        f"{report['label_agreement']:.6f}, confidence_agreement={report['confidence_agreement']:.6f}, "
        f"kept={report['consensus_kept_fraction']:.6f}"
    )


if __name__ == "__main__":
    main()
