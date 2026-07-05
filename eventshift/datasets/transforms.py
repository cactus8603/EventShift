"""Small transform helpers shared by image, label, and event tensors."""

from __future__ import annotations

import cv2
import numpy as np


def resize_like(array: np.ndarray, shape: tuple[int, int], interpolation: int | None = None) -> np.ndarray:
    height, width = shape[:2]
    if array.shape[:2] == (height, width):
        return array
    if interpolation is None:
        interpolation = cv2.INTER_NEAREST if array.ndim == 2 else cv2.INTER_LINEAR
    return cv2.resize(array, (width, height), interpolation=interpolation)


def shift_image(array: np.ndarray, dx: int = 0, dy: int = 0, fill_value=0) -> np.ndarray:
    dx = int(dx)
    dy = int(dy)
    if dx == 0 and dy == 0:
        return array
    shifted = np.full_like(array, fill_value)
    height, width = array.shape[:2]
    src_x0 = max(0, -dx)
    src_x1 = min(width, width - dx)
    dst_x0 = max(0, dx)
    dst_x1 = min(width, width + dx)
    src_y0 = max(0, -dy)
    src_y1 = min(height, height - dy)
    dst_y0 = max(0, dy)
    dst_y1 = min(height, height + dy)
    if src_x1 > src_x0 and src_y1 > src_y0:
        shifted[dst_y0:dst_y1, dst_x0:dst_x1] = array[src_y0:src_y1, src_x0:src_x1]
    return shifted


def normalize_nonzero(array: np.ndarray) -> np.ndarray:
    out = array.astype(np.float32, copy=True)
    mask = out != 0
    if np.any(mask):
        values = out[mask]
        std = float(values.std())
        out[mask] = (values - float(values.mean())) / std if std > 1e-6 else values - float(values.mean())
    return out

