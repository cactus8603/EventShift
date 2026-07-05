#!/usr/bin/env python
"""Copy an anchor prediction dir and paste selected candidate classes into it."""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

def _eventshift_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "configs").is_dir() and (parent / "third_party").is_dir():
            return parent
    return Path(__file__).resolve().parents[1]


ROOT = _eventshift_root()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-dir", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--class-id", action="append", type=int, required=True)
    parser.add_argument("--prefix", default=None, help="Optional sequence prefix filter, e.g. REAL_")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def iter_anchor_masks(anchor_dir, prefix=None):
    anchor_dir = Path(anchor_dir)
    for seq_dir in sorted(path for path in anchor_dir.iterdir() if path.is_dir()):
        if prefix and not seq_dir.name.startswith(prefix):
            continue
        seg_dir = seq_dir / "segment_co"
        if not seg_dir.is_dir():
            continue
        for mask_path in sorted(seg_dir.glob("*.png")):
            yield seq_dir.name, mask_path


def read_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"Could not read mask: {path}")
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask.astype(np.uint8, copy=False)


def write_mask(path, mask):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), mask):
        raise RuntimeError(f"Could not write mask: {path}")


def main():
    args = parse_args()
    anchor_dir = Path(args.anchor_dir)
    candidate_dir = Path(args.candidate_dir)
    out_dir = Path(args.out_dir)
    class_ids = sorted(set(args.class_id))

    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = {
        "files": 0,
        "changed_files": 0,
        "pixels": 0,
        "changed_pixels": 0,
        "candidate_class_pixels": {str(class_id): 0 for class_id in class_ids},
        "changed_pixels_by_class": {str(class_id): 0 for class_id in class_ids},
        "sequences": {},
    }
    items = list(iter_anchor_masks(anchor_dir, prefix=args.prefix))
    for seq_name, anchor_path in tqdm(items, desc="merge-class"):
        rel = anchor_path.relative_to(anchor_dir)
        candidate_path = candidate_dir / rel
        if not candidate_path.exists():
            raise FileNotFoundError(f"Missing candidate mask: {candidate_path}")
        anchor = read_mask(anchor_path)
        candidate = read_mask(candidate_path)
        if anchor.shape != candidate.shape:
            raise ValueError(f"Shape mismatch for {rel}: {anchor.shape} vs {candidate.shape}")

        merged = anchor.copy()
        for class_id in class_ids:
            take = candidate == class_id
            counts["candidate_class_pixels"][str(class_id)] += int(take.sum())
            counts["changed_pixels_by_class"][str(class_id)] += int((take & (anchor != class_id)).sum())
            merged[take] = class_id

        changed = int(np.count_nonzero(merged != anchor))
        counts["files"] += 1
        counts["changed_files"] += int(changed > 0)
        counts["pixels"] += int(anchor.size)
        counts["changed_pixels"] += changed
        seq = counts["sequences"].setdefault(seq_name, {"files": 0, "changed_files": 0, "changed_pixels": 0})
        seq["files"] += 1
        seq["changed_files"] += int(changed > 0)
        seq["changed_pixels"] += changed
        write_mask(out_dir / rel, merged)

    counts["changed_pixel_rate"] = counts["changed_pixels"] / max(1, counts["pixels"])
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "anchor_dir": str(anchor_dir.resolve()),
        "candidate_dir": str(candidate_dir.resolve()),
        "out_dir": str(out_dir.resolve()),
        "class_ids": class_ids,
        "prefix": args.prefix,
        "counts": counts,
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
