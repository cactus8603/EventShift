#!/usr/bin/env python
"""Evaluate a prediction directory against label masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from eventshift.utils.metrics import confusion_matrix, mean_iou


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--num-classes", type=int, default=19)
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--suffix", default=".png")
    return parser.parse_args()


def read_mask(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.int64, copy=False)


def main() -> None:
    args = parse_args()
    pred_root = Path(args.pred)
    gt_root = Path(args.gt)
    hist = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)
    matched = 0
    for pred_path in sorted(pred_root.rglob(f"*{args.suffix}")):
        rel = pred_path.relative_to(pred_root)
        gt_path = gt_root / rel
        if not gt_path.exists():
            gt_path = gt_root / pred_path.name
        if not gt_path.exists():
            continue
        hist += confusion_matrix(read_mask(pred_path), read_mask(gt_path), args.num_classes, args.ignore_index)
        matched += 1
    result = mean_iou(hist)
    result["matched_images"] = matched
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

