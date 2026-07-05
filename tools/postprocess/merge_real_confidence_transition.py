#!/usr/bin/env python3
"""Create REAL masks by accepting candidate changes under confidence/transition rules."""

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from analyze_real_candidate_confidence import load_confidence
from analyze_real_prediction_candidates import collect_masks, load_mask, parse_candidate
from cosec_finetune_splits import CLASSES


CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)


def parse_class(value):
    value = value.strip()
    if value.isdigit():
        idx = int(value)
        if 0 <= idx < NUM_CLASSES:
            return idx
    if value in CLASS_TO_ID:
        return CLASS_TO_ID[value]
    raise argparse.ArgumentTypeError(f"Unknown class: {value}")


def parse_transition(value):
    if "->" not in value:
        raise argparse.ArgumentTypeError(
            f"transition must use from->to format, got: {value}"
        )
    src, dst = value.split("->", 1)
    return parse_class(src), parse_class(dst)


def save_mask(mask, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8), mode="L").save(path)


def transition_mask(base, candidate, transitions):
    if not transitions:
        return np.ones_like(base, dtype=bool)
    allowed = np.zeros_like(base, dtype=bool)
    for src, dst in transitions:
        allowed |= (base == src) & (candidate == dst)
    return allowed


def class_mask(candidate, allow_to, deny_to):
    allowed = np.ones_like(candidate, dtype=bool)
    if allow_to:
        allowed &= np.isin(candidate, list(allow_to))
    if deny_to:
        allowed &= ~np.isin(candidate, list(deny_to))
    return allowed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=parse_candidate, required=True)
    parser.add_argument("--candidate", type=parse_candidate, required=True)
    parser.add_argument("--test-root", type=Path, default=Path("data/test"))
    parser.add_argument(
        "--confidence-dir",
        default="prior_mask2former_large_ft_cc_submission_conf",
    )
    parser.add_argument("--conf-lt", type=float, default=0.75)
    parser.add_argument("--allow-transition", action="append", type=parse_transition)
    parser.add_argument("--allow-to", action="append", type=parse_class)
    parser.add_argument("--deny-to", action="append", type=parse_class)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    base_name, base_path = args.base
    cand_name, cand_path = args.candidate
    base_masks, _ = collect_masks(base_path, real_only=True)
    cand_masks, _ = collect_masks(cand_path, real_only=True)
    common = sorted(set(base_masks) & set(cand_masks))
    if not common:
        raise RuntimeError("No common REAL masks found")

    out_dir = args.out_dir
    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    allow_transitions = set(args.allow_transition or [])
    allow_to = set(args.allow_to or [])
    deny_to = set(args.deny_to or [])

    stats = {
        "frames": 0,
        "valid_pixels": 0,
        "candidate_changed_pixels": 0,
        "accepted_pixels": 0,
        "accepted_transition_counts": Counter(),
        "candidate_transition_counts": Counter(),
        "missing_confidence": [],
    }

    for key in common:
        seq, name = key.split("/", 1)
        base = load_mask(base_masks[key])
        cand = load_mask(cand_masks[key])
        if base.shape != cand.shape:
            continue
        conf = load_confidence(args.test_root, key, args.confidence_dir)
        if conf is None:
            stats["missing_confidence"].append(key)
            conf = np.ones_like(base, dtype=np.float32)
        elif conf.shape != base.shape:
            conf_img = Image.fromarray((conf * 255.0).astype(np.uint8))
            conf = np.asarray(
                conf_img.resize((base.shape[1], base.shape[0]), Image.Resampling.BILINEAR)
            ).astype(np.float32) / 255.0

        valid = (base >= 0) & (base < NUM_CLASSES) & (cand >= 0) & (cand < NUM_CLASSES)
        changed = valid & (base != cand)
        accept = (
            changed
            & (conf < args.conf_lt)
            & transition_mask(base, cand, allow_transitions)
            & class_mask(cand, allow_to, deny_to)
        )
        merged = base.copy()
        merged[accept] = cand[accept]
        save_mask(merged, out_dir / seq / "segment_co" / name)

        stats["frames"] += 1
        stats["valid_pixels"] += int(valid.sum())
        stats["candidate_changed_pixels"] += int(changed.sum())
        stats["accepted_pixels"] += int(accept.sum())
        if np.any(changed):
            changed_pairs = base[changed].astype(np.int64) * NUM_CLASSES + cand[changed].astype(np.int64)
            for flat_idx, count in enumerate(np.bincount(changed_pairs, minlength=NUM_CLASSES**2)):
                if count:
                    stats["candidate_transition_counts"][flat_idx] += int(count)
        if np.any(accept):
            accepted_pairs = base[accept].astype(np.int64) * NUM_CLASSES + cand[accept].astype(np.int64)
            for flat_idx, count in enumerate(np.bincount(accepted_pairs, minlength=NUM_CLASSES**2)):
                if count:
                    stats["accepted_transition_counts"][flat_idx] += int(count)

    def transition_rows(counter):
        rows = []
        for flat_idx, count in counter.most_common(20):
            src = flat_idx // NUM_CLASSES
            dst = flat_idx % NUM_CLASSES
            rows.append(
                {
                    "from_class": CLASSES[src],
                    "to_class": CLASSES[dst],
                    "pixels": count,
                }
            )
        return rows

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "base_name": base_name,
        "base_path": str(base_path),
        "candidate_name": cand_name,
        "candidate_path": str(cand_path),
        "confidence_dir": args.confidence_dir,
        "conf_lt": args.conf_lt,
        "allow_transitions": [
            f"{CLASSES[src]}->{CLASSES[dst]}" for src, dst in sorted(allow_transitions)
        ],
        "allow_to": [CLASSES[idx] for idx in sorted(allow_to)],
        "deny_to": [CLASSES[idx] for idx in sorted(deny_to)],
        "stats": {
            "frames": stats["frames"],
            "valid_pixels": stats["valid_pixels"],
            "candidate_changed_pixels": stats["candidate_changed_pixels"],
            "candidate_changed_fraction": stats["candidate_changed_pixels"] / stats["valid_pixels"],
            "accepted_pixels": stats["accepted_pixels"],
            "accepted_fraction_of_valid": stats["accepted_pixels"] / stats["valid_pixels"],
            "accepted_fraction_of_candidate_changes": stats["accepted_pixels"]
            / max(stats["candidate_changed_pixels"], 1),
            "missing_confidence": stats["missing_confidence"],
            "candidate_top_transitions": transition_rows(stats["candidate_transition_counts"]),
            "accepted_top_transitions": transition_rows(stats["accepted_transition_counts"]),
        },
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"wrote {out_dir}")
    print(
        "accepted "
        f"{manifest['stats']['accepted_fraction_of_candidate_changes']:.2%} "
        "of candidate changes"
    )
    print(
        "accepted "
        f"{manifest['stats']['accepted_fraction_of_valid']:.2%} "
        "of valid REAL pixels"
    )


if __name__ == "__main__":
    main()
