#!/usr/bin/env python
"""Merge PNG mask directories by rolling selected base classes back into a candidate."""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from cosec_finetune_splits import CLASSES


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--rollback-base-classes", default="building,wall,fence")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def split_classes(text):
    names = [item.strip() for item in text.split(",") if item.strip()]
    return names, {CLASSES.index(name) for name in names}


def read_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"Could not read mask: {path}")
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask


def write_mask(path, mask):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), mask):
        raise RuntimeError(f"Could not write mask: {path}")


def iter_masks(root):
    root = Path(root)
    for path in sorted(root.rglob("segment_co/*.png")):
        yield path.relative_to(root)


def main():
    args = parse_args()
    base_dir = Path(args.base_dir)
    candidate_dir = Path(args.candidate_dir)
    out_dir = Path(args.out_dir)
    class_names, rollback_classes = split_classes(args.rollback_base_classes)

    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    stats = {
        "files": 0,
        "pixels": 0,
        "changed_candidate_vs_base": 0,
        "rolled_back_pixels": 0,
        "rollback_by_class": {name: 0 for name in class_names},
        "missing_candidate": [],
    }
    class_name_by_id = {CLASSES.index(name): name for name in class_names}

    for rel_path in iter_masks(base_dir):
        base_path = base_dir / rel_path
        cand_path = candidate_dir / rel_path
        if not cand_path.is_file():
            stats["missing_candidate"].append(str(rel_path))
            continue
        base = read_mask(base_path)
        cand = read_mask(cand_path)
        if base.shape != cand.shape:
            raise RuntimeError(f"Shape mismatch for {rel_path}: base={base.shape}, cand={cand.shape}")
        changed = base != cand
        rollback = np.zeros(base.shape, dtype=bool)
        for class_id in rollback_classes:
            rollback |= base == class_id
        rollback &= changed
        merged = cand.copy()
        merged[rollback] = base[rollback]
        write_mask(out_dir / rel_path, merged)

        stats["files"] += 1
        stats["pixels"] += int(base.size)
        stats["changed_candidate_vs_base"] += int(changed.sum())
        stats["rolled_back_pixels"] += int(rollback.sum())
        for class_id, name in class_name_by_id.items():
            stats["rollback_by_class"][name] += int((rollback & (base == class_id)).sum())

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "base_dir": str(base_dir.resolve()),
        "candidate_dir": str(candidate_dir.resolve()),
        "out_dir": str(out_dir.resolve()),
        "rollback_base_classes": class_names,
        "stats": stats,
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote merged masks: {out_dir}")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
