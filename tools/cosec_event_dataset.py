import json
import math
import os
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np

from cosec_finetune_splits import split_contains_sample

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    hdf5plugin = None


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
BRENET_ROOT = Path(os.environ.get("BRENET_ROOT", WORKSPACE_ROOT / "BRENet")).expanduser()
MANIFEST_PATH = Path(
    os.environ.get(
        "EVENTSHIFT_COSEC_MANIFEST",
        BRENET_ROOT / "projects" / "brenet_cosec" / "manifests" / "cosec_train_bidir_50ms.json",
    )
).expanduser()


@lru_cache(maxsize=1)
def _manifest_samples():
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return tuple(sample for sample in payload["samples"] if sample.get("valid", True))


def _resolve_brenet(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return BRENET_ROOT / path


def load_cosec_event_dicts(split):
    records = []
    for sample in _manifest_samples():
        seq_name = str(sample["sequence"])
        frame_id = int(sample["frame_id"])
        if not split_contains_sample(seq_name, frame_id, split):
            continue

        image_path = _resolve_brenet(sample["image"])
        label_path = _resolve_brenet(sample["label"])
        event_path = _resolve_brenet(sample["event_h5"])
        if not image_path.exists() or not label_path.exists() or not event_path.exists():
            continue

        records.append(
            {
                "file_name": str(image_path),
                "sem_seg_file_name": str(label_path),
                "image_id": f"{seq_name}_{frame_id:06d}",
                "event_h5": str(event_path),
                "event_old": [int(value) for value in sample["event_old"]],
                "event_new": [int(value) for value in sample["event_new"]],
            }
        )
    return records


_H5_CACHE = {}


def _h5(path):
    path = str(path)
    handle = _H5_CACHE.get(path)
    if handle is None:
        handle = h5py.File(path, "r")
        _H5_CACHE[path] = handle
    return handle


def _event_group(h5_file):
    if "events" in h5_file and all(key in h5_file["events"] for key in ("x", "y", "t", "p")):
        return h5_file["events"]
    return h5_file


def _event_slice(h5_file, time_window):
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
        timestamps = np.asarray(group["t"])
        left = int(np.searchsorted(timestamps, query_start_us, side="left"))
        right = int(np.searchsorted(timestamps, query_end_us, side="right"))

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


def _accumulate_window(events, num_bins, height, width):
    voxel = np.zeros((num_bins, height, width), dtype=np.float32)
    density = np.zeros((height, width), dtype=np.float32)
    pos_count = np.zeros((height, width), dtype=np.float32)
    neg_count = np.zeros((height, width), dtype=np.float32)
    if events is None or events["x"].size == 0:
        return voxel, density, pos_count, neg_count

    x = events["x"]
    y = events["y"]
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    if not np.any(valid):
        return voxel, density, pos_count, neg_count

    x = x[valid]
    y = y[valid]
    t = events["t"][valid]
    p = events["p"][valid]
    pol = np.where(p > 0, 1.0, -1.0).astype(np.float32)

    t0 = float(t[0])
    denom = max(float(t[-1]) - t0, 1.0)
    tb = (num_bins - 1) * (t - t0) / denom
    left_bin = np.floor(tb).astype(np.int64)
    right_bin = np.clip(left_bin + 1, 0, num_bins - 1)
    frac = (tb - left_bin).astype(np.float32)
    left_weight = (1.0 - frac) * pol
    right_weight = frac * pol

    np.add.at(voxel, (left_bin, y, x), left_weight)
    np.add.at(voxel, (right_bin, y, x), right_weight)
    np.add.at(density, (y, x), 1.0)
    np.add.at(pos_count, (y[p > 0], x[p > 0]), 1.0)
    np.add.at(neg_count, (y[p <= 0], x[p <= 0]), 1.0)
    return voxel, density, pos_count, neg_count


def _accumulate_counts(events, height, width):
    density = np.zeros((height, width), dtype=np.float32)
    pos_count = np.zeros((height, width), dtype=np.float32)
    neg_count = np.zeros((height, width), dtype=np.float32)
    if events is None or events["x"].size == 0:
        return density, pos_count, neg_count

    x = events["x"]
    y = events["y"]
    p = events["p"]
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    if not np.any(valid):
        return density, pos_count, neg_count

    x = x[valid]
    y = y[valid]
    p = p[valid]
    np.add.at(density, (y, x), 1.0)
    np.add.at(pos_count, (y[p > 0], x[p > 0]), 1.0)
    np.add.at(neg_count, (y[p <= 0], x[p <= 0]), 1.0)
    return density, pos_count, neg_count


def _normalize_nonzero(array):
    array = array.astype(np.float32, copy=True)
    max_value = float(array.max())
    if max_value > 1e-6:
        array /= max_value
    return array


def _shift_time_window(time_window, time_offset_ms=0.0):
    offset_us = int(round(float(time_offset_ms) * 1000.0))
    return [int(value) + offset_us for value in time_window]


def load_event_representation(dataset_dict, image_shape, num_bins=5, time_offset_ms=0.0):
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
        if std > 1e-6:
            event[nonzero] = (values - float(values.mean())) / std
        else:
            event[nonzero] = values - float(values.mean())

    aux = np.stack(
        [
            old_density,
            new_density,
            old_pos + new_pos,
            old_neg + new_neg,
        ],
        axis=0,
    ).astype(np.float32)
    return event.astype(np.float32), aux


def load_event_edge_representation(dataset_dict, image_shape, window_radii_ms, time_offset_ms=0.0):
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
        polarity_balance[active] = (
            1.0 - np.abs(pos_count[active] - neg_count[active]) / (polarity_total[active] + 1e-6)
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
