#!/usr/bin/env python
"""Merge prediction dirs with conservative anchor-preserving agreement."""

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from ensemble_feature_cache_common import CLASSES


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-dir", required=True)
    parser.add_argument("--candidate-dir", action="append", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--allow-class", action="append", required=True)
    parser.add_argument("--sequences", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def class_id(value):
    text = str(value)
    if text.isdigit():
        idx = int(text)
        if idx < 0 or idx >= len(CLASSES):
            raise ValueError(f"Class id out of range: {idx}")
        return idx
    if text not in CLASSES:
        raise ValueError(f"Unknown class: {text}")
    return CLASSES.index(text)


def list_sequence_masks(root, sequences=None):
    root = Path(root)
    requested = set(sequences) if sequences else None
    mapping = {}
    for seq_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if requested is not None and seq_dir.name not in requested:
            continue
        seg_dir = seq_dir / "segment_co"
        if not seg_dir.is_dir():
            continue
        mapping[seq_dir.name] = {path.name: path for path in sorted(seg_dir.glob("*.png"))}
    return mapping


def read_mask(path):
    mask = np.asarray(Image.open(path))
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask.astype(np.uint8, copy=False)


def write_mask(path, mask):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8, copy=False)).save(path)


def safe_rate(count, total):
    return float(count / total) if total else 0.0


def merge_one(anchor, candidates, allowed_classes):
    if not candidates:
        raise ValueError("Need at least one candidate mask.")
    if any(mask.shape != anchor.shape for mask in candidates):
        raise ValueError(f"Shape mismatch: anchor={anchor.shape}, candidates={[mask.shape for mask in candidates]}")

    first = candidates[0]
    agree = np.ones(anchor.shape, dtype=bool)
    for mask in candidates[1:]:
        agree &= mask == first

    allowed = np.isin(first, sorted(allowed_classes))
    take = agree & allowed & (first != anchor)
    merged = anchor.copy()
    merged[take] = first[take]
    return merged, take, first


def main():
    args = parse_args()
    anchor_dir = Path(args.anchor_dir)
    candidate_dirs = [Path(path) for path in args.candidate_dir]
    out_dir = Path(args.out_dir)
    allowed_classes = {class_id(value) for value in args.allow_class}

    if not anchor_dir.is_dir():
        raise FileNotFoundError(anchor_dir)
    for candidate_dir in candidate_dirs:
        if not candidate_dir.is_dir():
            raise FileNotFoundError(candidate_dir)
    if len(candidate_dirs) < 2:
        raise ValueError("Need at least two --candidate-dir values for agreement.")
    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(out_dir)
        shutil.rmtree(out_dir)

    anchor_index = list_sequence_masks(anchor_dir, sequences=args.sequences)
    candidate_indexes = [list_sequence_masks(root, sequences=args.sequences) for root in candidate_dirs]
    common_sequences = sorted(set(anchor_index).intersection(*(set(index) for index in candidate_indexes)))
    if not common_sequences:
        raise ValueError("No common sequences across prediction dirs.")

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "anchor_dir": str(anchor_dir.resolve()),
        "candidate_dirs": [str(path.resolve()) for path in candidate_dirs],
        "out_dir": str(out_dir.resolve()),
        "allowed_classes": [CLASSES[idx] for idx in sorted(allowed_classes)],
        "sequences": {},
        "totals": {
            "files": 0,
            "pixels": 0,
            "changed_pixels": 0,
            "accepted_pixels": 0,
            "accepted_by_class": {CLASSES[idx]: 0 for idx in range(len(CLASSES))},
            "output_histogram": {CLASSES[idx]: 0 for idx in range(len(CLASSES))},
        },
    }

    for seq_name in common_sequences:
        mask_names = sorted(
            set(anchor_index[seq_name]).intersection(*(set(index[seq_name]) for index in candidate_indexes))
        )
        if not mask_names:
            raise ValueError(f"No common masks for sequence {seq_name}.")
        seq_stats = {
            "files": 0,
            "pixels": 0,
            "changed_pixels": 0,
            "accepted_pixels": 0,
            "accepted_by_class": {CLASSES[idx]: 0 for idx in range(len(CLASSES))},
        }
        for mask_name in mask_names:
            anchor = read_mask(anchor_index[seq_name][mask_name])
            candidates = [read_mask(index[seq_name][mask_name]) for index in candidate_indexes]
            merged, take, agreed = merge_one(anchor, candidates, allowed_classes)
            write_mask(out_dir / seq_name / "segment_co" / mask_name, merged)

            pixels = int(anchor.size)
            accepted_pixels = int(np.count_nonzero(take))
            seq_stats["files"] += 1
            seq_stats["pixels"] += pixels
            seq_stats["accepted_pixels"] += accepted_pixels
            seq_stats["changed_pixels"] += int(np.count_nonzero(merged != anchor))
            manifest["totals"]["files"] += 1
            manifest["totals"]["pixels"] += pixels
            manifest["totals"]["accepted_pixels"] += accepted_pixels
            manifest["totals"]["changed_pixels"] += int(np.count_nonzero(merged != anchor))

            for idx in range(len(CLASSES)):
                accepted = int(np.count_nonzero(take & (agreed == idx)))
                if accepted:
                    seq_stats["accepted_by_class"][CLASSES[idx]] += accepted
                    manifest["totals"]["accepted_by_class"][CLASSES[idx]] += accepted
            for label, count in Counter(merged.reshape(-1).tolist()).items():
                if 0 <= int(label) < len(CLASSES):
                    manifest["totals"]["output_histogram"][CLASSES[int(label)]] += int(count)

        seq_stats["changed_pixel_rate"] = safe_rate(seq_stats["changed_pixels"], seq_stats["pixels"])
        seq_stats["accepted_pixel_rate"] = safe_rate(seq_stats["accepted_pixels"], seq_stats["pixels"])
        manifest["sequences"][seq_name] = seq_stats

    totals = manifest["totals"]
    totals["changed_pixel_rate"] = safe_rate(totals["changed_pixels"], totals["pixels"])
    totals["accepted_pixel_rate"] = safe_rate(totals["accepted_pixels"], totals["pixels"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(totals, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
