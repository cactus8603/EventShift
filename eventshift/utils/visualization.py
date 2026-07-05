"""Visualization helpers for 19-class segmentation masks."""

from __future__ import annotations

import numpy as np


CITYSCAPES_PALETTE = np.asarray(
    [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ],
    dtype=np.uint8,
)


def colorize_label(label: np.ndarray, ignore_index: int = 255) -> np.ndarray:
    out = np.zeros((*label.shape[:2], 3), dtype=np.uint8)
    valid = (label != ignore_index) & (label >= 0) & (label < len(CITYSCAPES_PALETTE))
    out[valid] = CITYSCAPES_PALETTE[label[valid].astype(np.int64)]
    return out

