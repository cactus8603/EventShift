#!/usr/bin/env python3
"""Analyze confidence of changed pixels between two REAL candidates."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from analyze_real_prediction_candidates import collect_masks, load_mask, parse_candidate
from cosec_finetune_splits import CLASSES


NUM_CLASSES = len(CLASSES)


def load_confidence(test_root, frame_key, conf_dir):
    seq, name = frame_key.split("/", 1)
    path = test_root / seq / conf_dir / name
    if not path.exists():
        return None
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.float32) / 255.0


def pct(value):
    return float(value) if np.isfinite(value) else 0.0


def summarize_pair(name_a, masks_a, name_b, masks_b, test_root, conf_dir, thresholds, topk):
    common = sorted(set(masks_a) & set(masks_b))
    changed_pixels = 0
    valid_pixels = 0
    conf_sum_all = 0.0
    conf_sum_changed = 0.0
    conf_sum_unchanged = 0.0
    conf_count_all = 0
    conf_count_changed = 0
    conf_count_unchanged = 0
    changed_under = Counter()
    transition_counts = Counter()
    transition_conf_sum = Counter()
    frame_rows = []
    missing_conf = []

    for key in common:
        mask_a = load_mask(masks_a[key])
        mask_b = load_mask(masks_b[key])
        if mask_a.shape != mask_b.shape:
            continue
        conf = load_confidence(test_root, key, conf_dir)
        if conf is None:
            missing_conf.append(key)
            continue
        if conf.shape != mask_a.shape:
            conf_img = Image.fromarray((conf * 255.0).astype(np.uint8))
            conf = np.asarray(
                conf_img.resize((mask_a.shape[1], mask_a.shape[0]), Image.Resampling.BILINEAR)
            ).astype(np.float32) / 255.0

        valid = (
            (mask_a >= 0)
            & (mask_a < NUM_CLASSES)
            & (mask_b >= 0)
            & (mask_b < NUM_CLASSES)
        )
        changed = valid & (mask_a != mask_b)
        unchanged = valid & (mask_a == mask_b)
        if not np.any(valid):
            continue

        valid_count = int(valid.sum())
        changed_count = int(changed.sum())
        valid_pixels += valid_count
        changed_pixels += changed_count
        conf_sum_all += float(conf[valid].sum())
        conf_count_all += valid_count
        if changed_count:
            changed_conf = conf[changed]
            conf_sum_changed += float(changed_conf.sum())
            conf_count_changed += changed_count
            for threshold in thresholds:
                changed_under[str(threshold)] += int((changed_conf < threshold).sum())
            pairs = mask_a[changed].astype(np.int64) * NUM_CLASSES + mask_b[changed].astype(np.int64)
            counts = np.bincount(pairs, minlength=NUM_CLASSES * NUM_CLASSES)
            for flat_idx, count in enumerate(counts):
                if count:
                    transition_counts[flat_idx] += int(count)
            for flat_idx in np.unique(pairs):
                transition_conf_sum[int(flat_idx)] += float(changed_conf[pairs == flat_idx].sum())
        if np.any(unchanged):
            conf_sum_unchanged += float(conf[unchanged].sum())
            conf_count_unchanged += int(unchanged.sum())

        high_changed = int((conf[changed] >= 0.75).sum()) if changed_count else 0
        frame_rows.append(
            {
                "frame": key,
                "changed_fraction": changed_count / valid_count,
                "changed_pixels": changed_count,
                "changed_conf_mean": pct(conf[changed].mean()) if changed_count else 0.0,
                "changed_conf_ge_075_fraction": high_changed / changed_count if changed_count else 0.0,
            }
        )

    transition_rows = []
    for flat_idx, count in transition_counts.most_common(topk):
        src = flat_idx // NUM_CLASSES
        dst = flat_idx % NUM_CLASSES
        transition_rows.append(
            {
                "from_class": CLASSES[src],
                "to_class": CLASSES[dst],
                "pixels": count,
                "fraction_of_changed": count / changed_pixels if changed_pixels else 0.0,
                "mean_old_confidence": transition_conf_sum[flat_idx] / count,
            }
        )

    frame_rows.sort(key=lambda row: row["changed_fraction"], reverse=True)
    return {
        "candidate_a": name_a,
        "candidate_b": name_b,
        "common_frames": len(common),
        "frames_with_confidence": len(common) - len(missing_conf),
        "missing_confidence_frames": missing_conf[:topk],
        "valid_pixels": valid_pixels,
        "changed_pixels": changed_pixels,
        "changed_fraction": changed_pixels / valid_pixels if valid_pixels else 0.0,
        "old_confidence_mean_all": conf_sum_all / conf_count_all if conf_count_all else 0.0,
        "old_confidence_mean_changed": conf_sum_changed / conf_count_changed
        if conf_count_changed
        else 0.0,
        "old_confidence_mean_unchanged": conf_sum_unchanged / conf_count_unchanged
        if conf_count_unchanged
        else 0.0,
        "changed_below_threshold": {
            str(threshold): changed_under[str(threshold)] / changed_pixels
            if changed_pixels
            else 0.0
            for threshold in thresholds
        },
        "top_transitions": transition_rows,
        "top_changed_frames": frame_rows[:topk],
    }


def write_markdown(path, report):
    lines = [
        "# REAL Candidate Confidence Analysis",
        "",
        "Confidence comes from the old prior confidence masks, not from GT.",
        "",
        f"Pair: `{report['candidate_a']}` -> `{report['candidate_b']}`",
        "",
        f"- common REAL frames: {report['common_frames']}",
        f"- frames with confidence: {report['frames_with_confidence']}",
        f"- changed pixels: {report['changed_fraction']:.2%} "
        f"({report['changed_pixels']:,}/{report['valid_pixels']:,})",
        f"- old confidence mean, all valid pixels: {report['old_confidence_mean_all']:.4f}",
        f"- old confidence mean, changed pixels: {report['old_confidence_mean_changed']:.4f}",
        f"- old confidence mean, unchanged pixels: {report['old_confidence_mean_unchanged']:.4f}",
        "",
        "## Changed Pixels Below Confidence Threshold",
    ]
    for threshold, value in report["changed_below_threshold"].items():
        lines.append(f"- < {float(threshold):.2f}: {value:.2%}")

    lines.append("")
    lines.append("## Top Transitions")
    for row in report["top_transitions"]:
        lines.append(
            f"- {row['from_class']} -> {row['to_class']}: "
            f"{row['fraction_of_changed']:.2%} of changed, "
            f"mean old conf {row['mean_old_confidence']:.4f}"
        )

    lines.append("")
    lines.append("## Most Changed Frames")
    for row in report["top_changed_frames"]:
        lines.append(
            f"- {row['frame']}: changed {row['changed_fraction']:.2%}, "
            f"mean changed conf {row['changed_conf_mean']:.4f}, "
            f"changed conf>=0.75 {row['changed_conf_ge_075_fraction']:.2%}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-a", type=parse_candidate, required=True)
    parser.add_argument("--candidate-b", type=parse_candidate, required=True)
    parser.add_argument("--test-root", type=Path, default=Path("data/test"))
    parser.add_argument(
        "--confidence-dir",
        default="prior_mask2former_large_ft_cc_submission_conf",
    )
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=12)
    args = parser.parse_args()

    name_a, path_a = args.candidate_a
    name_b, path_b = args.candidate_b
    masks_a, _ = collect_masks(path_a, real_only=True)
    masks_b, _ = collect_masks(path_b, real_only=True)
    if not masks_a or not masks_b:
        raise RuntimeError("Both candidates must contain REAL masks")

    report = summarize_pair(
        name_a,
        masks_a,
        name_b,
        masks_b,
        args.test_root,
        args.confidence_dir,
        thresholds=(0.25, 0.50, 0.625, 0.75, 0.875),
        topk=args.topk,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_markdown(args.out_md, report)
    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    print(
        f"{name_a}->{name_b}: changed={report['changed_fraction']:.2%}, "
        f"changed_conf={report['old_confidence_mean_changed']:.4f}, "
        f"unchanged_conf={report['old_confidence_mean_unchanged']:.4f}"
    )


if __name__ == "__main__":
    main()
