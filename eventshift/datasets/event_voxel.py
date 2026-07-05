"""Event H5 slicing and voxel / edge representation utilities."""

from __future__ import annotations

import math
from pathlib import Path

import h5py
import numpy as np

try:
    import hdf5plugin  # noqa: F401
except ImportError:  # pragma: no cover - optional runtime dependency
    hdf5plugin = None


_H5_CACHE: dict[str, h5py.File] = {}


def _h5(path: str | Path) -> h5py.File:
    key = str(path)
    handle = _H5_CACHE.get(key)
    if handle is None:
        handle = h5py.File(key, "r")
        _H5_CACHE[key] = handle
    return handle


def close_h5_cache() -> None:
    for handle in _H5_CACHE.values():
        handle.close()
    _H5_CACHE.clear()


def _event_group(h5_file: h5py.File):
    if "events" in h5_file and all(key in h5_file["events"] for key in ("x", "y", "t", "p")):
        return h5_file["events"]
    return h5_file


def _shift_time_window(time_window, time_offset_ms: float = 0.0) -> list[int]:
    offset_us = int(round(float(time_offset_ms) * 1000.0))
    return [int(value) + offset_us for value in time_window]


def _event_slice(h5_file: h5py.File, time_window) -> dict[str, np.ndarray] | None:
    start_us, end_us = [int(value) for value in time_window]
    if end_us <= start_us:
        return None

    group = _event_group(h5_file)
    t_offset = int(h5_file["t_offset"][()]) if "t_offset" in h5_file else 0
    query_start_us = start_us - t_offset
    query_end_us = end_us - t_offset
    if query_end_us <= query_start_us:
        return None

    if "ms_to_idx" in h5_file:
        ms_to_idx = h5_file["ms_to_idx"]
        start_ms = max(0, min(ms_to_idx.shape[0] - 1, int(math.floor(query_start_us / 1000.0))))
        end_ms = max(0, min(ms_to_idx.shape[0] - 1, int(math.ceil(query_end_us / 1000.0))))
        if end_ms <= start_ms:
            return None
        left = int(ms_to_idx[start_ms])
        right = int(ms_to_idx[end_ms])
    else:
        timestamps_all = np.asarray(group["t"])
        left = int(np.searchsorted(timestamps_all, query_start_us, side="left"))
        right = int(np.searchsorted(timestamps_all, query_end_us, side="right"))

    if right <= left:
        return None

    timestamps = np.asarray(group["t"][left:right])
    keep = (timestamps >= query_start_us) & (timestamps < query_end_us)
    if not np.any(keep):
        return None

    return {
        "x": np.asarray(group["x"][left:right])[keep].astype(np.int64),
        "y": np.asarray(group["y"][left:right])[keep].astype(np.int64),
        "t": (timestamps[keep] + t_offset).astype(np.float32),
        "p": np.asarray(group["p"][left:right])[keep].astype(np.int64),
    }


def _valid_xy(events: dict[str, np.ndarray], height: int, width: int):
    x = events["x"]
    y = events["y"]
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    return x[valid], y[valid], valid


def _accumulate_window(events, num_bins: int, height: int, width: int):
    voxel = np.zeros((num_bins, height, width), dtype=np.float32)
    density = np.zeros((height, width), dtype=np.float32)
    pos_count = np.zeros((height, width), dtype=np.float32)
    neg_count = np.zeros((height, width), dtype=np.float32)
    if events is None or events["x"].size == 0:
        return voxel, density, pos_count, neg_count

    x, y, valid = _valid_xy(events, height, width)
    if not np.any(valid):
        return voxel, density, pos_count, neg_count

    t = events["t"][valid]
    p = events["p"][valid]
    pol = np.where(p > 0, 1.0, -1.0).astype(np.float32)
    t0 = float(t[0])
    denom = max(float(t[-1]) - t0, 1.0)
    tb = (num_bins - 1) * (t - t0) / denom
    left_bin = np.floor(tb).astype(np.int64)
    right_bin = np.clip(left_bin + 1, 0, num_bins - 1)
    frac = (tb - left_bin).astype(np.float32)

    np.add.at(voxel, (left_bin, y, x), (1.0 - frac) * pol)
    np.add.at(voxel, (right_bin, y, x), frac * pol)
    np.add.at(density, (y, x), 1.0)
    np.add.at(pos_count, (y[p > 0], x[p > 0]), 1.0)
    np.add.at(neg_count, (y[p <= 0], x[p <= 0]), 1.0)
    return voxel, density, pos_count, neg_count


def _accumulate_counts(events, height: int, width: int):
    density = np.zeros((height, width), dtype=np.float32)
    pos_count = np.zeros((height, width), dtype=np.float32)
    neg_count = np.zeros((height, width), dtype=np.float32)
    if events is None or events["x"].size == 0:
        return density, pos_count, neg_count

    x, y, valid = _valid_xy(events, height, width)
    if not np.any(valid):
        return density, pos_count, neg_count

    p = events["p"][valid]
    np.add.at(density, (y, x), 1.0)
    np.add.at(pos_count, (y[p > 0], x[p > 0]), 1.0)
    np.add.at(neg_count, (y[p <= 0], x[p <= 0]), 1.0)
    return density, pos_count, neg_count


def _normalize_nonzero(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32, copy=True)
    max_value = float(array.max())
    if max_value > 1e-6:
        array /= max_value
    return array


def load_event_representation(dataset_dict: dict, image_shape, num_bins: int = 5, time_offset_ms: float = 0.0):
    height, width = image_shape[:2]
    h5_file = _h5(dataset_dict["event_h5"])
    old_events = _event_slice(h5_file, _shift_time_window(dataset_dict["event_old"], time_offset_ms))
    new_events = _event_slice(h5_file, _shift_time_window(dataset_dict["event_new"], time_offset_ms))

    old_voxel, old_density, old_pos, old_neg = _accumulate_window(old_events, num_bins, height, width)
    new_voxel, new_density, new_pos, new_neg = _accumulate_window(new_events, num_bins, height, width)

    event = np.concatenate([old_voxel, new_voxel], axis=0)
    nonzero = np.abs(event) > 0
    if np.any(nonzero):
        values = event[nonzero]
        std = float(values.std())
        event[nonzero] = (values - float(values.mean())) / std if std > 1e-6 else values - float(values.mean())

    aux = np.stack([old_density, new_density, old_pos + new_pos, old_neg + new_neg], axis=0)
    return event.astype(np.float32), aux.astype(np.float32)


def load_event_edge_representation(
    dataset_dict: dict,
    image_shape,
    window_radii_ms,
    time_offset_ms: float = 0.0,
):
    height, width = image_shape[:2]
    radii = [int(value) for value in window_radii_ms]
    if not radii:
        return np.zeros((0, height, width), dtype=np.float32)

    h5_file = _h5(dataset_dict["event_h5"])
    center_us = int(dataset_dict["event_old"][1]) + int(round(float(time_offset_ms) * 1000.0))
    channels = []
    for radius_ms in radii:
        half_window = int(round(float(radius_ms) * 1000.0))
        events = _event_slice(h5_file, [center_us - half_window, center_us + half_window])
        density, pos_count, neg_count = _accumulate_counts(events, height, width)
        density_log = np.log1p(density).astype(np.float32)
        polarity_total = pos_count + neg_count
        polarity_balance = np.zeros_like(density_log, dtype=np.float32)
        active = polarity_total > 0
        polarity_balance[active] = 1.0 - np.abs(pos_count[active] - neg_count[active]) / (
            polarity_total[active] + 1e-6
        )
        edge_score = density_log * (0.5 + 0.5 * polarity_balance)
        channels.extend(
            [
                _normalize_nonzero(density_log),
                _normalize_nonzero(edge_score),
                polarity_balance * (density > 0).astype(np.float32),
            ]
        )
    return np.stack(channels, axis=0).astype(np.float32)

