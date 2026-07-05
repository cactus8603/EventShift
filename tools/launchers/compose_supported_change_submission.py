#!/usr/bin/env python
"""Apply high-scoring prediction changes only when a supporter agrees.

For each pixel, start from base. If high differs from base and at least one
supporter predicts the same class as high, use high. Otherwise keep base.
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
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--high-dir", required=True)
    parser.add_argument("--supporter-dirs", nargs="+", required=True)
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
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"Could not read mask: {path}")
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return np.asarray(mask, dtype=np.uint8)


def main():
    args = parse_args()
    base_dir = Path(args.base_dir)
    high_dir = Path(args.high_dir)
    supporter_dirs = [Path(path) for path in args.supporter_dirs]
    out_dir = Path(args.out_dir)

    if out_dir.exists() and args.overwrite:
        import shutil

        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "images": 0,
        "pixels": 0,
        "high_changed_pixels": 0,
        "applied_pixels": 0,
        "unsupported_pixels": 0,
        "sequences": {},
    }

    for seq_name, image_name in tqdm(list(iter_images(args.test_root, args.sequences)), desc="Supported changes"):
        base_path = base_dir / seq_name / "segment_co" / image_name
        high_path = high_dir / seq_name / "segment_co" / image_name
        supporter_paths = [root / seq_name / "segment_co" / image_name for root in supporter_dirs]
        missing = [str(path) for path in [base_path, high_path, *supporter_paths] if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing masks for {seq_name}/{image_name}: {missing}")

        base_mask = read_mask(base_path)
        high_mask = read_mask(high_path)
        supporter_masks = [read_mask(path) for path in supporter_paths]
        bad_shapes = [
            str(path)
            for path, mask in zip([high_path, *supporter_paths], [high_mask, *supporter_masks])
            if mask.shape != base_mask.shape
        ]
        if bad_shapes:
            raise ValueError(f"Mask shape mismatch for {seq_name}/{image_name}: {bad_shapes}")

        high_changed = high_mask != base_mask
        supporter_agrees = np.zeros(base_mask.shape, dtype=bool)
        for supporter_mask in supporter_masks:
            supporter_agrees |= supporter_mask == high_mask
        apply_change = high_changed & supporter_agrees

        out_mask = base_mask.copy()
        out_mask[apply_change] = high_mask[apply_change]

        dst_dir = out_dir / seq_name / "segment_co"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_path = dst_dir / image_name
        if not cv2.imwrite(str(dst_path), out_mask):
            raise RuntimeError(f"Could not write mask: {dst_path}")

        pixels = int(base_mask.size)
        high_changed_pixels = int(np.count_nonzero(high_changed))
        applied_pixels = int(np.count_nonzero(apply_change))
        unsupported_pixels = high_changed_pixels - applied_pixels

        stats["images"] += 1
        stats["pixels"] += pixels
        stats["high_changed_pixels"] += high_changed_pixels
        stats["applied_pixels"] += applied_pixels
        stats["unsupported_pixels"] += unsupported_pixels
        seq_stats = stats["sequences"].setdefault(
            seq_name,
            {
                "images": 0,
                "pixels": 0,
                "high_changed_pixels": 0,
                "applied_pixels": 0,
                "unsupported_pixels": 0,
            },
        )
        seq_stats["images"] += 1
        seq_stats["pixels"] += pixels
        seq_stats["high_changed_pixels"] += high_changed_pixels
        seq_stats["applied_pixels"] += applied_pixels
        seq_stats["unsupported_pixels"] += unsupported_pixels

    def add_rates(item):
        pixels = item["pixels"]
        high_changed = item["high_changed_pixels"]
        item["high_changed_rate"] = float(high_changed / pixels) if pixels else 0.0
        item["applied_rate"] = float(item["applied_pixels"] / pixels) if pixels else 0.0
        item["support_rate_of_high_changes"] = (
            float(item["applied_pixels"] / high_changed) if high_changed else 0.0
        )

    add_rates(stats)
    for seq_stats in stats["sequences"].values():
        add_rates(seq_stats)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "base_dir": str(base_dir.resolve()),
        "high_dir": str(high_dir.resolve()),
        "supporter_dirs": [str(path.resolve()) for path in supporter_dirs],
        "test_root": str(Path(args.test_root).resolve()),
        "out_dir": str(out_dir.resolve()),
        "sequences": args.sequences,
        "stats": stats,
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Wrote supported-change masks to: {out_dir}")
    print(json.dumps({"stats": stats}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
