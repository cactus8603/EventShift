#!/usr/bin/env python
"""Compose sequence masks by conservative hard voting across prediction dirs."""

import argparse
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-dirs",
        nargs="+",
        required=True,
        help="Prediction roots to vote. Each root contains SEQ/segment_co/*.png.",
    )
    parser.add_argument(
        "--anchor-dir",
        required=True,
        help="Fallback prediction root used when vote agreement is weak or tied.",
    )
    parser.add_argument("--test-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--num-classes", type=int, default=19)
    parser.add_argument(
        "--min-votes",
        type=int,
        default=4,
        help="Minimum top-class votes required to override the anchor pixel.",
    )
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


def vote_masks(candidate_masks, anchor_mask, num_classes, min_votes):
    stack = np.stack(candidate_masks, axis=0)
    if stack.ndim != 3:
        raise ValueError(f"Expected stacked masks to be 3-D, got {stack.shape}")
    if stack.shape[1:] != anchor_mask.shape:
        raise ValueError(f"Candidate shape {stack.shape[1:]} != anchor shape {anchor_mask.shape}")

    counts = np.empty((num_classes, *anchor_mask.shape), dtype=np.uint8)
    for cls_idx in range(num_classes):
        counts[cls_idx] = np.count_nonzero(stack == cls_idx, axis=0)

    top_class = counts.argmax(axis=0).astype(np.uint8)
    top_votes = counts.max(axis=0)
    tied = np.count_nonzero(counts == top_votes[None, :, :], axis=0) > 1
    use_vote = (top_votes >= min_votes) & ~tied
    out = anchor_mask.copy()
    out[use_vote] = top_class[use_vote]
    return out, use_vote, top_votes, tied


def main():
    args = parse_args()
    candidate_dirs = [Path(path) for path in args.candidate_dirs]
    anchor_dir = Path(args.anchor_dir)
    out_dir = Path(args.out_dir)

    if len(candidate_dirs) < 2:
        raise ValueError("--candidate-dirs must include at least two roots")
    if args.min_votes < 1 or args.min_votes > len(candidate_dirs):
        raise ValueError("--min-votes must be between 1 and the candidate count")
    if out_dir.exists() and args.overwrite:
        import shutil

        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_items = list(iter_images(args.test_root, args.sequences))
    stats = {
        "images": 0,
        "pixels": 0,
        "voted_pixels": 0,
        "fallback_pixels": 0,
        "tied_pixels": 0,
        "top_vote_hist": {str(idx): 0 for idx in range(len(candidate_dirs) + 1)},
        "sequences": {},
    }

    for seq_name, image_name in tqdm(image_items, desc="Hard vote masks"):
        anchor_path = anchor_dir / seq_name / "segment_co" / image_name
        candidate_paths = [root / seq_name / "segment_co" / image_name for root in candidate_dirs]
        missing = [str(path) for path in [anchor_path, *candidate_paths] if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing masks for {seq_name}/{image_name}: {missing}")

        anchor_mask = read_mask(anchor_path)
        candidate_masks = [read_mask(path) for path in candidate_paths]
        bad_shapes = [str(path) for path, mask in zip(candidate_paths, candidate_masks) if mask.shape != anchor_mask.shape]
        if bad_shapes:
            raise ValueError(
                f"Mask shape mismatch for {seq_name}/{image_name}; "
                f"anchor={anchor_mask.shape}, bad={bad_shapes}"
            )

        out_mask, use_vote, top_votes, tied = vote_masks(
            candidate_masks,
            anchor_mask,
            args.num_classes,
            args.min_votes,
        )

        dst_dir = out_dir / seq_name / "segment_co"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_path = dst_dir / image_name
        if not cv2.imwrite(str(dst_path), out_mask):
            raise RuntimeError(f"Could not write mask: {dst_path}")

        pixels = int(anchor_mask.size)
        voted = int(np.count_nonzero(use_vote))
        tied_pixels = int(np.count_nonzero(tied))
        stats["images"] += 1
        stats["pixels"] += pixels
        stats["voted_pixels"] += voted
        stats["fallback_pixels"] += pixels - voted
        stats["tied_pixels"] += tied_pixels
        hist = np.bincount(top_votes.reshape(-1), minlength=len(candidate_dirs) + 1)
        for idx, value in enumerate(hist[: len(candidate_dirs) + 1]):
            stats["top_vote_hist"][str(idx)] += int(value)
        seq_stats = stats["sequences"].setdefault(
            seq_name,
            {"images": 0, "pixels": 0, "voted_pixels": 0, "fallback_pixels": 0, "tied_pixels": 0},
        )
        seq_stats["images"] += 1
        seq_stats["pixels"] += pixels
        seq_stats["voted_pixels"] += voted
        seq_stats["fallback_pixels"] += pixels - voted
        seq_stats["tied_pixels"] += tied_pixels

    def add_rates(item):
        pixels = item["pixels"]
        item["voted_rate"] = float(item["voted_pixels"] / pixels) if pixels else 0.0
        item["fallback_rate"] = float(item["fallback_pixels"] / pixels) if pixels else 0.0
        item["tied_rate"] = float(item["tied_pixels"] / pixels) if pixels else 0.0

    add_rates(stats)
    for seq_stats in stats["sequences"].values():
        add_rates(seq_stats)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "candidate_dirs": [str(path.resolve()) for path in candidate_dirs],
        "anchor_dir": str(anchor_dir.resolve()),
        "test_root": str(Path(args.test_root).resolve()),
        "out_dir": str(out_dir.resolve()),
        "sequences": args.sequences,
        "num_classes": args.num_classes,
        "min_votes": args.min_votes,
        "stats": stats,
    }
    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Wrote hard-vote masks to: {out_dir}")
    print(json.dumps({"stats": stats}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
