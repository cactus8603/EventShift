#!/usr/bin/env python
"""Compose semantic masks by weighted voting over prediction directories."""

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        action="append",
        required=True,
        help="Prediction root. Repeat once per model.",
    )
    parser.add_argument(
        "--weight",
        action="append",
        type=float,
        required=True,
        help="Vote weight for the corresponding --input-dir. Repeat once per model.",
    )
    parser.add_argument(
        "--name",
        action="append",
        default=None,
        help="Optional model name for the corresponding --input-dir.",
    )
    parser.add_argument(
        "--tie-dir",
        default=None,
        help="Prediction root used when weighted scores tie. Defaults to the first input dir.",
    )
    parser.add_argument("--test-root", default="data/test")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def iter_images(test_root, sequences):
    test_root = Path(test_root)
    keep = set(sequences)
    for seq_dir in sorted(path for path in test_root.iterdir() if path.is_dir()):
        if seq_dir.name not in keep:
            continue
        img_dir = seq_dir / "img_co_left"
        if not img_dir.is_dir():
            raise FileNotFoundError(f"Missing image dir: {img_dir}")
        for img_path in sorted(img_dir.glob("*.png")):
            yield seq_dir.name, img_path.name


def read_mask(path):
    mask = np.asarray(Image.open(path))
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask.astype(np.uint8, copy=False)


def write_mask(path, mask):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8, copy=False)).save(path)


def weighted_vote(masks, weights, tie_mask):
    base_shape = masks[0].shape
    if any(mask.shape != base_shape for mask in masks):
        raise ValueError(f"Shape mismatch in vote input: {[mask.shape for mask in masks]}")
    if tie_mask.shape != base_shape:
        raise ValueError(f"Tie mask shape mismatch: {tie_mask.shape} vs {base_shape}")

    stack = np.stack(masks, axis=0)
    labels = np.unique(stack)
    scores = np.zeros((len(labels), *base_shape), dtype=np.float32)
    weight_arr = np.asarray(weights, dtype=np.float32)
    for label_idx, label in enumerate(labels):
        scores[label_idx] = ((stack == label) * weight_arr[:, None, None]).sum(axis=0)

    winner_idx = scores.argmax(axis=0)
    out = labels[winner_idx].astype(np.uint8, copy=False)
    top_score = scores.max(axis=0)
    tied = np.count_nonzero(np.isclose(scores, top_score[None, :, :], atol=1e-6), axis=0) > 1
    out[tied] = tie_mask[tied]
    return out, tied


def init_stats():
    return {
        "images": 0,
        "pixels": 0,
        "tie_pixels": 0,
        "changed_from_first_pixels": 0,
        "label_histogram": {},
    }


def add_item_stats(stats, out_mask, first_mask, tied):
    pixels = int(out_mask.size)
    tie_pixels = int(np.count_nonzero(tied))
    changed = int(np.count_nonzero(out_mask != first_mask))
    stats["images"] += 1
    stats["pixels"] += pixels
    stats["tie_pixels"] += tie_pixels
    stats["changed_from_first_pixels"] += changed
    hist = Counter(out_mask.reshape(-1).tolist())
    for label, count in hist.items():
        key = str(int(label))
        stats["label_histogram"][key] = stats["label_histogram"].get(key, 0) + int(count)


def add_rates(stats):
    pixels = stats["pixels"]
    stats["tie_rate"] = float(stats["tie_pixels"] / pixels) if pixels else 0.0
    stats["changed_from_first_rate"] = (
        float(stats["changed_from_first_pixels"] / pixels) if pixels else 0.0
    )


def main():
    args = parse_args()
    input_dirs = [Path(path) for path in args.input_dir]
    weights = args.weight
    names = args.name or []
    if len(input_dirs) != len(weights):
        raise ValueError("--input-dir and --weight must be repeated the same number of times")
    if names and len(names) != len(input_dirs):
        raise ValueError("--name must be repeated once per --input-dir when provided")
    if any(weight <= 0 for weight in weights):
        raise ValueError("All weights must be positive")
    if len(input_dirs) < 2:
        raise ValueError("Need at least two input dirs")
    for root in input_dirs:
        if not root.is_dir():
            raise FileNotFoundError(root)

    tie_dir = Path(args.tie_dir) if args.tie_dir else input_dirs[0]
    if not tie_dir.is_dir():
        raise FileNotFoundError(tie_dir)

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(out_dir)
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = init_stats()
    stats["sequences"] = {}
    for seq_name, image_name in tqdm(list(iter_images(args.test_root, args.sequences)), desc="Weighted vote"):
        input_paths = [root / seq_name / "segment_co" / image_name for root in input_dirs]
        tie_path = tie_dir / seq_name / "segment_co" / image_name
        missing = [str(path) for path in [*input_paths, tie_path] if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing masks for {seq_name}/{image_name}: {missing}")

        masks = [read_mask(path) for path in input_paths]
        tie_mask = read_mask(tie_path)
        out_mask, tied = weighted_vote(masks, weights, tie_mask)
        write_mask(out_dir / seq_name / "segment_co" / image_name, out_mask)

        add_item_stats(stats, out_mask, masks[0], tied)
        seq_stats = stats["sequences"].setdefault(seq_name, init_stats())
        add_item_stats(seq_stats, out_mask, masks[0], tied)

    add_rates(stats)
    for seq_stats in stats["sequences"].values():
        add_rates(seq_stats)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "inputs": [
            {
                "name": names[idx] if names else input_dirs[idx].name,
                "dir": str(input_dirs[idx].resolve()),
                "weight": float(weights[idx]),
            }
            for idx in range(len(input_dirs))
        ],
        "tie_dir": str(tie_dir.resolve()),
        "test_root": str(Path(args.test_root).resolve()),
        "out_dir": str(out_dir.resolve()),
        "sequences": args.sequences,
        "stats": stats,
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Wrote weighted-vote masks to: {out_dir}")
    print(json.dumps({"stats": {key: stats[key] for key in stats if key != "sequences"}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
