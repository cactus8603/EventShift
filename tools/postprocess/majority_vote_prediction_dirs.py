#!/usr/bin/env python
"""Majority-vote semantic prediction directories with an optional tie anchor."""

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", action="append", required=True, help="Prediction dir to vote. Repeat this flag.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tie-dir", default=None, help="Prediction dir used when there is no strict majority.")
    parser.add_argument("--sequences", nargs="*", default=None, help="Optional sequence names to include.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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


def common_index(input_roots, sequences=None):
    indexes = [list_sequence_masks(root, sequences=sequences) for root in input_roots]
    common_sequences = sorted(set.intersection(*(set(index) for index in indexes)))
    if not common_sequences:
        raise ValueError("No common sequences across input dirs.")

    common = {}
    for seq_name in common_sequences:
        common_names = sorted(set.intersection(*(set(index[seq_name]) for index in indexes)))
        if not common_names:
            raise ValueError(f"No common masks for sequence {seq_name}.")
        common[seq_name] = common_names
    return indexes, common


def read_mask(path):
    mask = np.asarray(Image.open(path))
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask.astype(np.uint8, copy=False)


def write_mask(path, mask):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8, copy=False)).save(path)


def vote_masks(masks, tie_mask=None):
    base_shape = masks[0].shape
    if any(mask.shape != base_shape for mask in masks):
        raise ValueError(f"Shape mismatch in vote input: {[mask.shape for mask in masks]}")
    if tie_mask is not None and tie_mask.shape != base_shape:
        raise ValueError(f"Tie mask shape mismatch: {tie_mask.shape} vs {base_shape}")

    stack = np.stack(masks, axis=0)
    out = np.empty(base_shape, dtype=np.uint8)
    tie_pixels = np.zeros(base_shape, dtype=bool)
    changed_from_first = np.zeros(base_shape, dtype=bool)

    # The masks are 8-bit trainIds. Looping over labels is simpler and fast
    # enough for 102 REAL images.
    labels = np.unique(stack)
    counts = np.zeros((len(labels),) + base_shape, dtype=np.uint8)
    for label_idx, label in enumerate(labels):
        counts[label_idx] = np.count_nonzero(stack == label, axis=0)

    max_count = counts.max(axis=0)
    winner_count = np.count_nonzero(counts == max_count[None, ...], axis=0)
    tie_pixels = winner_count > 1
    winner_indices = counts.argmax(axis=0)
    out[:] = labels[winner_indices]

    if tie_mask is not None:
        out[tie_pixels] = tie_mask[tie_pixels]

    changed_from_first = out != masks[0]
    return out, {
        "pixels": int(out.size),
        "tie_pixels": int(np.count_nonzero(tie_pixels)),
        "tie_pixel_rate": float(np.count_nonzero(tie_pixels) / out.size),
        "changed_from_first_pixels": int(np.count_nonzero(changed_from_first)),
        "changed_from_first_rate": float(np.count_nonzero(changed_from_first) / out.size),
        "label_histogram": {str(int(label)): int(count) for label, count in Counter(out.reshape(-1).tolist()).items()},
    }


def main():
    args = parse_args()
    input_roots = [Path(path) for path in args.input_dir]
    out_dir = Path(args.out_dir)
    tie_root = Path(args.tie_dir) if args.tie_dir else None

    if len(input_roots) < 2:
        raise ValueError("Need at least two --input-dir values for voting.")
    for root in input_roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
    if tie_root is not None and not tie_root.is_dir():
        raise FileNotFoundError(tie_root)
    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(out_dir)
        shutil.rmtree(out_dir)

    indexes, common = common_index(input_roots, sequences=args.sequences)
    tie_index = list_sequence_masks(tie_root, sequences=args.sequences) if tie_root else {}

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_dirs": [str(root.resolve()) for root in input_roots],
        "tie_dir": str(tie_root.resolve()) if tie_root else None,
        "out_dir": str(out_dir.resolve()),
        "sequences": {},
        "totals": {
            "files": 0,
            "pixels": 0,
            "tie_pixels": 0,
            "changed_from_first_pixels": 0,
        },
    }

    for seq_name, mask_names in common.items():
        seq_stats = {
            "files": 0,
            "pixels": 0,
            "tie_pixels": 0,
            "changed_from_first_pixels": 0,
        }
        for mask_name in mask_names:
            masks = [read_mask(index[seq_name][mask_name]) for index in indexes]
            tie_mask = None
            if tie_root is not None:
                try:
                    tie_mask = read_mask(tie_index[seq_name][mask_name])
                except KeyError as error:
                    raise FileNotFoundError(f"Missing tie mask for {seq_name}/{mask_name}") from error
            voted, stats = vote_masks(masks, tie_mask=tie_mask)
            write_mask(out_dir / seq_name / "segment_co" / mask_name, voted)
            seq_stats["files"] += 1
            seq_stats["pixels"] += stats["pixels"]
            seq_stats["tie_pixels"] += stats["tie_pixels"]
            seq_stats["changed_from_first_pixels"] += stats["changed_from_first_pixels"]
        for key in ("files", "pixels", "tie_pixels", "changed_from_first_pixels"):
            manifest["totals"][key] += seq_stats[key]
        seq_stats["tie_pixel_rate"] = seq_stats["tie_pixels"] / seq_stats["pixels"] if seq_stats["pixels"] else 0.0
        seq_stats["changed_from_first_rate"] = (
            seq_stats["changed_from_first_pixels"] / seq_stats["pixels"] if seq_stats["pixels"] else 0.0
        )
        manifest["sequences"][seq_name] = seq_stats

    totals = manifest["totals"]
    totals["tie_pixel_rate"] = totals["tie_pixels"] / totals["pixels"] if totals["pixels"] else 0.0
    totals["changed_from_first_rate"] = (
        totals["changed_from_first_pixels"] / totals["pixels"] if totals["pixels"] else 0.0
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(manifest["totals"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
