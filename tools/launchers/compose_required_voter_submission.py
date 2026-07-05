#!/usr/bin/env python
"""Compose masks by requiring support for one trusted voter's label.

For each pixel, start from a base/fallback prediction. Let the required voter
predict label R. If at least min_votes of the voter dirs predict R, output R;
otherwise keep the base label.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True, help="Fallback prediction root.")
    parser.add_argument(
        "--voter-dirs",
        nargs="+",
        required=True,
        help="Prediction roots counted as votes. Include base here if it should count.",
    )
    parser.add_argument("--required-dir", required=True, help="Trusted prediction root whose label must be supported.")
    parser.add_argument("--test-root", default="data/test")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--min-votes", type=int, default=3)
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
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"Could not read mask: {path}")
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return np.asarray(mask, dtype=np.uint8)


def add_rates(item):
    pixels = item["pixels"]
    item["eligible_rate"] = float(item["eligible_pixels"] / pixels) if pixels else 0.0
    item["changed_rate"] = float(item["changed_pixels"] / pixels) if pixels else 0.0
    item["fallback_rate"] = float(item["fallback_pixels"] / pixels) if pixels else 0.0


def main():
    args = parse_args()
    base_dir = Path(args.base_dir)
    voter_dirs = [Path(path) for path in args.voter_dirs]
    required_dir = Path(args.required_dir)
    out_dir = Path(args.out_dir)

    if args.min_votes < 1 or args.min_votes > len(voter_dirs):
        raise ValueError("--min-votes must be between 1 and the number of voter dirs")
    if out_dir.exists() and args.overwrite:
        import shutil

        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "images": 0,
        "pixels": 0,
        "eligible_pixels": 0,
        "changed_pixels": 0,
        "fallback_pixels": 0,
        "required_vote_hist": {str(idx): 0 for idx in range(len(voter_dirs) + 1)},
        "sequences": {},
    }

    image_items = list(iter_images(args.test_root, args.sequences))
    for seq_name, image_name in tqdm(image_items, desc="Required-voter compose"):
        base_path = base_dir / seq_name / "segment_co" / image_name
        voter_paths = [root / seq_name / "segment_co" / image_name for root in voter_dirs]
        required_path = required_dir / seq_name / "segment_co" / image_name
        missing = [str(path) for path in [base_path, required_path, *voter_paths] if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing masks for {seq_name}/{image_name}: {missing}")

        base_mask = read_mask(base_path)
        required_mask = read_mask(required_path)
        voter_masks = [read_mask(path) for path in voter_paths]
        bad_shapes = [
            str(path)
            for path, mask in zip([required_path, *voter_paths], [required_mask, *voter_masks])
            if mask.shape != base_mask.shape
        ]
        if bad_shapes:
            raise ValueError(f"Mask shape mismatch for {seq_name}/{image_name}: {bad_shapes}")

        stack = np.stack(voter_masks, axis=0)
        required_votes = np.count_nonzero(stack == required_mask[None, :, :], axis=0)
        eligible = required_votes >= args.min_votes
        out_mask = base_mask.copy()
        out_mask[eligible] = required_mask[eligible]
        changed = out_mask != base_mask

        dst_dir = out_dir / seq_name / "segment_co"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_path = dst_dir / image_name
        if not cv2.imwrite(str(dst_path), out_mask):
            raise RuntimeError(f"Could not write mask: {dst_path}")

        pixels = int(base_mask.size)
        eligible_pixels = int(np.count_nonzero(eligible))
        changed_pixels = int(np.count_nonzero(changed))
        hist = np.bincount(required_votes.reshape(-1), minlength=len(voter_dirs) + 1)

        stats["images"] += 1
        stats["pixels"] += pixels
        stats["eligible_pixels"] += eligible_pixels
        stats["changed_pixels"] += changed_pixels
        stats["fallback_pixels"] += pixels - eligible_pixels
        for idx, value in enumerate(hist[: len(voter_dirs) + 1]):
            stats["required_vote_hist"][str(idx)] += int(value)

        seq_stats = stats["sequences"].setdefault(
            seq_name,
            {"images": 0, "pixels": 0, "eligible_pixels": 0, "changed_pixels": 0, "fallback_pixels": 0},
        )
        seq_stats["images"] += 1
        seq_stats["pixels"] += pixels
        seq_stats["eligible_pixels"] += eligible_pixels
        seq_stats["changed_pixels"] += changed_pixels
        seq_stats["fallback_pixels"] += pixels - eligible_pixels

    add_rates(stats)
    for seq_stats in stats["sequences"].values():
        add_rates(seq_stats)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "base_dir": str(base_dir.resolve()),
        "voter_dirs": [str(path.resolve()) for path in voter_dirs],
        "required_dir": str(required_dir.resolve()),
        "test_root": str(Path(args.test_root).resolve()),
        "out_dir": str(out_dir.resolve()),
        "sequences": args.sequences,
        "min_votes": args.min_votes,
        "stats": stats,
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Wrote required-voter masks to: {out_dir}")
    print(json.dumps({"stats": stats}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
