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
_DEFAULT_BRENET_ROOT = WORKSPACE_ROOT / "BRENet"
_EXPLICIT_BRENET_ROOT = os.environ.get("BRENET_ROOT")
BRENET_ROOT = Path(_EXPLICIT_BRENET_ROOT).expanduser() if _EXPLICIT_BRENET_ROOT else None
MANIFEST_PATH = Path(
    os.environ.get(
        "EVENTSHIFT_COSEC_MANIFEST",
        (BRENET_ROOT or _DEFAULT_BRENET_ROOT)
        / "projects"
        / "brenet_cosec"
        / "manifests"
        / "cosec_train_bidir_50ms.json",
    )
).expanduser()


@lru_cache(maxsize=1)
def _manifest_samples():
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return tuple(sample for sample in payload["samples"] if sample.get("valid", True))


def _infer_manifest_root(manifest_path):
    manifest_path = Path(manifest_path).expanduser().resolve()
    parent = manifest_path.parent
    if (
        parent.name == "manifests"
        and parent.parent.name == "brenet_cosec"
        and parent.parent.parent.name == "projects"
    ):
        return parent.parent.parent.parent
    return parent


def _candidate_roots():
    roots = []
    if BRENET_ROOT is not None:
        roots.append(BRENET_ROOT)
    roots.extend([_infer_manifest_root(MANIFEST_PATH), MANIFEST_PATH.parent, _DEFAULT_BRENET_ROOT])
    deduped = []
    seen = set()
    for root in roots:
        root = Path(root).expanduser()
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(root)
    return deduped


def _resolve_brenet(path):
    path = Path(path)
    if path.is_absolute():
        return path
    candidates = [root / path for root in _candidate_roots()]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


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
_DAT_CACHE = {}
_RECTIFY_MAP_CACHE = {}


def _h5(path):
    path = str(path)
    handle = _H5_CACHE.get(path)
    if handle is None:
        handle = h5py.File(path, "r")
        _H5_CACHE[path] = handle
    return handle


def _dat_arrays(t_path, xyp_path):
    key = (str(t_path), str(xyp_path))
    arrays = _DAT_CACHE.get(key)
    if arrays is not None:
        return arrays

    t_path = Path(t_path)
    xyp_path = Path(xyp_path)
    num_events = t_path.stat().st_size // np.dtype(np.int64).itemsize
    expected_xyp_bytes = num_events * 3 * np.dtype(np.uint16).itemsize
    if xyp_path.stat().st_size != expected_xyp_bytes:
        raise ValueError(
            f"DDD17 event xyp size does not match timestamps: "
            f"{xyp_path} has {xyp_path.stat().st_size} bytes, expected {expected_xyp_bytes}"
        )

    arrays = (
        np.memmap(t_path, dtype=np.int64, mode="r", shape=(num_events,)),
        np.memmap(xyp_path, dtype=np.uint16, mode="r", shape=(num_events, 3)),
    )
    _DAT_CACHE[key] = arrays
    return arrays


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

    timestamps = np.asarray(group["t"][left:right], dtype=np.int64)
    keep = (timestamps >= query_start_us) & (timestamps < query_end_us)
    if not np.any(keep):
        return None

    return {
        "x": np.asarray(group["x"][left:right])[keep].astype(np.int64),
        "y": np.asarray(group["y"][left:right])[keep].astype(np.int64),
        "t": (timestamps[keep].astype(np.float64) + float(t_offset)).astype(np.float32),
        "p": np.asarray(group["p"][left:right])[keep].astype(np.int64),
    }


def _event_slice_dat(dataset_dict, time_window):
    start_time, end_time = [int(value) for value in time_window]
    if end_time <= start_time:
        return None

    timestamps, xyp = _dat_arrays(dataset_dict["event_dat_t"], dataset_dict["event_dat_xyp"])
    left = int(np.searchsorted(timestamps, start_time, side="left"))
    right = int(np.searchsorted(timestamps, end_time, side="right"))
    if right <= left:
        return None

    window_timestamps = np.asarray(timestamps[left:right], dtype=np.int64)
    keep = (window_timestamps >= start_time) & (window_timestamps < end_time)
    if not np.any(keep):
        return None

    window_xyp = np.asarray(xyp[left:right][keep], dtype=np.int64)
    return {
        "x": window_xyp[:, 0],
        "y": window_xyp[:, 1],
        "t": window_timestamps[keep].astype(np.float64),
        "p": window_xyp[:, 2],
    }


def _rectify_map(path):
    path = str(path)
    rect_map = _RECTIFY_MAP_CACHE.get(path)
    if rect_map is None:
        handle = _h5(path)
        if "rectify_map" not in handle:
            raise KeyError(f"rectify_map dataset not found in {path}")
        rect_map = np.asarray(handle["rectify_map"], dtype=np.float32)
        _RECTIFY_MAP_CACHE[path] = rect_map
    return rect_map


def _rectify_events(events, dataset_dict):
    if events is None or events["x"].size == 0 or "event_rectify_map" not in dataset_dict:
        return events

    rect_map = _rectify_map(dataset_dict["event_rectify_map"])
    map_height, map_width = rect_map.shape[:2]
    x = events["x"].astype(np.int64, copy=False)
    y = events["y"].astype(np.int64, copy=False)
    valid = (x >= 0) & (x < map_width) & (y >= 0) & (y < map_height)
    if not np.any(valid):
        return {key: value[:0] for key, value in events.items()}

    coords = rect_map[y[valid], x[valid]]
    finite = np.isfinite(coords[:, 0]) & np.isfinite(coords[:, 1])
    if not np.any(finite):
        return {key: value[:0] for key, value in events.items()}

    keep_indices = np.nonzero(valid)[0][finite]
    rectified = {
        "x": np.rint(coords[finite, 0]).astype(np.int64),
        "y": np.rint(coords[finite, 1]).astype(np.int64),
        "t": events["t"][keep_indices],
        "p": events["p"][keep_indices],
    }
    return rectified


def _scale_events_to_image_shape(events, dataset_dict, height, width):
    if events is None or events["x"].size == 0:
        return events

    sensor_width = int(dataset_dict.get("event_sensor_width", width))
    sensor_height = int(dataset_dict.get("event_sensor_height", height))
    if sensor_width == width and sensor_height == height:
        return events

    scaled = dict(events)
    if sensor_width > 0 and sensor_width != width:
        scaled["x"] = np.floor(scaled["x"].astype(np.float64) * float(width) / float(sensor_width)).astype(np.int64)
    if sensor_height > 0 and sensor_height != height:
        scaled["y"] = np.floor(scaled["y"].astype(np.float64) * float(height) / float(sensor_height)).astype(np.int64)
    return scaled


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


def _shift_time_window(time_window, time_offset_ms=0.0, units_per_ms=1000):
    offset = int(round(float(time_offset_ms) * float(units_per_ms)))
    return [int(value) + offset for value in time_window]


def _slice_events_for_record(dataset_dict, time_window):
    if "event_h5" in dataset_dict:
        return _event_slice(_h5(dataset_dict["event_h5"]), time_window)
    if "event_dat_t" in dataset_dict and "event_dat_xyp" in dataset_dict:
        return _event_slice_dat(dataset_dict, time_window)
    return None


def load_event_representation(dataset_dict, image_shape, num_bins=5, time_offset_ms=0.0):
    height, width = image_shape[:2]
    units_per_ms = int(dataset_dict.get("event_time_units_per_ms", 1000))
    old_events = _slice_events_for_record(
        dataset_dict,
        _shift_time_window(dataset_dict["event_old"], time_offset_ms, units_per_ms),
    )
    new_events = _slice_events_for_record(
        dataset_dict,
        _shift_time_window(dataset_dict["event_new"], time_offset_ms, units_per_ms),
    )
    old_events = _rectify_events(old_events, dataset_dict)
    new_events = _rectify_events(new_events, dataset_dict)
    old_events = _scale_events_to_image_shape(old_events, dataset_dict, height, width)
    new_events = _scale_events_to_image_shape(new_events, dataset_dict, height, width)

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


def _dilate_binary_mask(mask, radius):
    radius = int(radius)
    if radius <= 0 or not mask.any():
        return mask
    padded = np.pad(mask.astype(bool), radius, mode="constant", constant_values=False)
    dilated = np.zeros_like(mask, dtype=bool)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            dilated |= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return dilated


def _event_support_from_density_temporal(
    density_log,
    temporal_balance,
    support_percentile=50.0,
    temporal_threshold=0.05,
    support_dilation=1,
):
    nonzero_density = density_log[density_log > 0]
    if nonzero_density.size > 0:
        threshold = np.percentile(nonzero_density, float(support_percentile))
        support = (density_log > threshold) & (temporal_balance > float(temporal_threshold))
    else:
        support = np.zeros_like(density_log, dtype=bool)
    support = _dilate_binary_mask(support, int(support_dilation))
    return support.astype(np.float32)


def load_event_edge_representation(
    dataset_dict,
    image_shape,
    window_radii_ms,
    time_offset_ms=0.0,
    representation="legacy",
    support_percentile=50.0,
    temporal_threshold=0.05,
    support_dilation=1,
):
    height, width = image_shape[:2]
    radii = [int(value) for value in window_radii_ms]
    if not radii:
        return np.zeros((0, height, width), dtype=np.float32)
    representation = str(representation).lower()
    if representation in {"default", "triplet", "3ch"}:
        representation = "legacy"
    if representation in {"rms_v2", "true_reliability", "multiwindow_reliability"}:
        representation = "reliability_v2"
    if representation not in {"legacy", "reliability_v2"}:
        raise ValueError(f"Unsupported event edge representation: {representation!r}")

    units_per_ms = int(dataset_dict.get("event_time_units_per_ms", 1000))
    center_time = int(dataset_dict["event_old"][1]) + int(round(float(time_offset_ms) * float(units_per_ms)))
    channels = []
    for radius_ms in radii:
        half_window = int(round(float(radius_ms) * float(units_per_ms)))
        old_events = _slice_events_for_record(dataset_dict, [center_time - half_window, center_time])
        new_events = _slice_events_for_record(dataset_dict, [center_time, center_time + half_window])
        old_events = _rectify_events(old_events, dataset_dict)
        new_events = _rectify_events(new_events, dataset_dict)
        old_events = _scale_events_to_image_shape(old_events, dataset_dict, height, width)
        new_events = _scale_events_to_image_shape(new_events, dataset_dict, height, width)
        old_density, old_pos, old_neg = _accumulate_counts(old_events, height, width)
        new_density, new_pos, new_neg = _accumulate_counts(new_events, height, width)
        density = old_density + new_density
        pos_count = old_pos + new_pos
        neg_count = old_neg + new_neg
        density_log = np.log1p(density).astype(np.float32)

        temporal_balance = np.zeros_like(density_log, dtype=np.float32)
        temporal_active = density > 0
        temporal_balance[temporal_active] = (
            1.0
            - np.abs(old_density[temporal_active] - new_density[temporal_active])
            / (density[temporal_active] + 1e-6)
        )

        polarity_total = pos_count + neg_count
        polarity_balance = np.zeros_like(density_log, dtype=np.float32)
        active = polarity_total > 0
        polarity_balance[active] = (
            1.0 - np.abs(pos_count[active] - neg_count[active]) / (polarity_total[active] + 1e-6)
        )
        active_mask = (density > 0).astype(np.float32)
        if representation == "reliability_v2":
            support = _event_support_from_density_temporal(
                density_log,
                temporal_balance,
                support_percentile=support_percentile,
                temporal_threshold=temporal_threshold,
                support_dilation=support_dilation,
            )
            edge_score = density_log
            channels.extend(
                [
                    _normalize_nonzero(density_log),
                    temporal_balance * active_mask,
                    polarity_balance * active_mask,
                    support,
                    _normalize_nonzero(edge_score),
                ]
            )
        else:
            edge_score = density_log * (0.5 + 0.5 * polarity_balance)
            channels.extend(
                [
                    _normalize_nonzero(density_log),
                    _normalize_nonzero(edge_score),
                    polarity_balance * active_mask,
                ]
            )

    return np.stack(channels, axis=0).astype(np.float32)
