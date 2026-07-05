#!/usr/bin/env python3
"""Build mmseg split files for the local ACDC layout.

The ACDC files are stored as:
  rgb_anon/{condition}/{split}/{sequence}/{stem}_rgb_anon.png
  gt/{condition}/{split}/{sequence}/{stem}_gt_labelTrainIds.png

MMSeg's CityscapesDataset can pair them if each split line is:
  {condition}/{split}/{sequence}/{stem}
with image suffix ``_rgb_anon.png`` and label suffix
``_gt_labelTrainIds.png``.
"""

from __future__ import annotations

import argparse
from pathlib import Path


CONDITIONS = ("fog", "night", "rain", "snow")
SPLITS = ("train", "val")


def build_lines(acdc_root: Path, conditions: tuple[str, ...], splits: tuple[str, ...]) -> list[str]:
    lines: list[str] = []
    for condition in conditions:
        for split in splits:
            image_root = acdc_root / "rgb_anon" / condition / split
            label_root = acdc_root / "gt" / condition / split
            if not image_root.is_dir() or not label_root.is_dir():
                continue
            for image_path in sorted(image_root.glob("*/*_rgb_anon.png")):
                sequence = image_path.parent.name
                stem = image_path.name.removesuffix("_rgb_anon.png")
                label_path = label_root / sequence / f"{stem}_gt_labelTrainIds.png"
                if label_path.exists():
                    lines.append(f"{condition}/{split}/{sequence}/{stem}")
    return lines


def write_split(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"{path}: {len(lines)} samples")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--acdc-root",
        type=Path,
        default=Path("/work/u1621738/ebmv_eccv/MambaSeg/data/acdc"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/work/u1621738/ebmv_eccv/eccv_segment/swin_l/work_dirs/mmseg/acdc_splits"),
    )
    args = parser.parse_args()

    if not (args.acdc_root / "rgb_anon").is_dir() or not (args.acdc_root / "gt").is_dir():
        raise FileNotFoundError(f"Invalid ACDC root: {args.acdc_root}")

    for condition in CONDITIONS:
        for split in SPLITS:
            write_split(
                args.out_dir / f"{condition}_{split}.txt",
                build_lines(args.acdc_root, (condition,), (split,)),
            )
        write_split(
            args.out_dir / f"{condition}_trainval.txt",
            build_lines(args.acdc_root, (condition,), SPLITS),
        )

    for split in SPLITS:
        write_split(
            args.out_dir / f"all_{split}.txt",
            build_lines(args.acdc_root, CONDITIONS, (split,)),
        )
    write_split(args.out_dir / "all_trainval.txt", build_lines(args.acdc_root, CONDITIONS, SPLITS))


if __name__ == "__main__":
    main()
