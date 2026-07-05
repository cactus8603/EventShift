#!/usr/bin/env python3
"""Build a CityscapesDataset-compatible symlink view for full DSEC 19-class labels."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


DEFAULT_ROOT = Path(os.environ.get("DSEC_ROOT", "data/dsec"))
DEFAULT_OUT = Path(os.environ.get("DSEC_MMSEG_ROOT", "work_dirs/mmseg/dsec19_full_flat"))


def safe_symlink(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    source = source.resolve()
    if target.is_symlink() or target.exists():
        if target.resolve() == source:
            return
        raise FileExistsError(f"Refusing to overwrite {target}")
    target.symlink_to(source)


def build(root: Path, out_dir: Path) -> int:
    split_lines = []
    semantic_root = root / "train_semantic_segmentation"
    for label_dir in sorted(semantic_root.glob("*/19classes")):
        sequence = label_dir.parent.name
        for label_path in sorted(label_dir.glob("*.png")):
            frame = label_path.stem
            image_path = root / "train_image" / sequence / "images" / "left" / "rectified" / label_path.name
            if not image_path.exists():
                continue
            rel = f"{sequence}/{frame}"
            safe_symlink(image_path, out_dir / "images" / sequence / label_path.name)
            safe_symlink(label_path, out_dir / "labels" / sequence / label_path.name)
            split_lines.append(rel)

    split_path = out_dir / "splits" / "train_full.txt"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text("\n".join(split_lines) + ("\n" if split_lines else ""), encoding="utf-8")
    return len(split_lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsec-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    count = build(args.dsec_root, args.out_dir)
    print(f"{args.out_dir}: {count} DSEC19 full samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
