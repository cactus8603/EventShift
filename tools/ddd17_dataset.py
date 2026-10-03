import os
from functools import lru_cache
from pathlib import Path

import numpy as np


DDD17_DEFAULT_ROOT = Path(
    os.environ.get(
        "DDD17_ROOT",
        "/data/ddd17/ess_preprocessed/ddd17_seg",
    )
).expanduser()

DDD17_CLASSES = (
    "flat",
    "construction_sky",
    "object",
    "nature",
    "human",
    "vehicle",
)

DDD17_PALETTE = (
    (128, 64, 128),
    (70, 70, 70),
    (0, 0, 142),
    (107, 142, 35),
    (220, 20, 60),
    (0, 0, 70),
)

DDD17_TRAIN_DIRS = ("dir0", "dir1", "dir3", "dir4")
DDD17_VAL_DIRS = ("dir6", "dir7")
DDD17_DIR5_DIRS = ("dir5",)
DDD17_FULL_LABELED_DIRS = DDD17_TRAIN_DIRS + DDD17_DIR5_DIRS + DDD17_VAL_DIRS
DDD17_DEFAULT_EVENT_WINDOW_MS = 50
DDD17_SENSOR_WIDTH = 346
DDD17_SENSOR_HEIGHT = 260
DDD17_EVENT_TIME_UNITS_PER_MS = 1_000_000


def _normalize_root(root):
    root = Path(root).expanduser()
    candidates = (
        root,
        root / "ess_preprocessed" / "ddd17_seg",
        root / "ddd17_seg",
    )
    for candidate in candidates:
        if (candidate / "data").is_dir():
            return candidate
    return candidates[0]


def ddd17_root(root=DDD17_DEFAULT_ROOT):
    root = _normalize_root(root)
    if not (root / "data").is_dir():
        raise FileNotFoundError(f"DDD17 data directory not found under: {root}")
    return root


def _split_dirs(split):
    aliases = {
        "train": DDD17_TRAIN_DIRS,
        "val": DDD17_VAL_DIRS,
        "full": DDD17_FULL_LABELED_DIRS,
        "full_labeled": DDD17_FULL_LABELED_DIRS,
        "dir5": DDD17_DIR5_DIRS,
    }
    try:
        return aliases[split]
    except KeyError as error:
        raise ValueError(f"Unknown DDD17 split: {split}") from error


def _frame_id_from_mask(mask_path):
    stem = Path(mask_path).stem
    if not stem.startswith("segmentation_"):
        raise ValueError(f"Unexpected DDD17 mask name: {mask_path}")
    return int(stem.rsplit("_", 1)[1])


def _image_candidates(img_dir, frame_id):
    return (
        img_dir / f"img_{frame_id:08d}.png",
        img_dir / f"img_{frame_id:010d}.png",
        img_dir / f"{frame_id:010d}.png",
        img_dir / f"{frame_id:08d}.png",
    )


def _image_for_frame(seq_dir, frame_id):
    img_dir = seq_dir / "imgs"
    for candidate in _image_candidates(img_dir, frame_id):
        if candidate.exists():
            return candidate
    matches = sorted(img_dir.glob(f"*{frame_id:08d}.png"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"DDD17 image not found for {seq_dir.name} frame {frame_id:08d}")


def _event_index_path(seq_dir, event_window_ms):
    requested = seq_dir / "index" / f"index_{int(event_window_ms)}ms.npy"
    if requested.exists():
        return requested
    fallback = seq_dir / "index" / f"index_{DDD17_DEFAULT_EVENT_WINDOW_MS}ms.npy"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"DDD17 event index not found under: {seq_dir / 'index'}")


@lru_cache(maxsize=None)
def _load_event_index(index_path):
    return np.load(str(index_path), mmap_mode="r")


def _add_event_fields(record, seq_dir, frame_id, event_window_ms):
    event_t_path = seq_dir / "events.dat.t"
    event_xyp_path = seq_dir / "events.dat.xyp"
    if not event_t_path.exists():
        raise FileNotFoundError(f"DDD17 event timestamps not found: {event_t_path}")
    if not event_xyp_path.exists():
        raise FileNotFoundError(f"DDD17 event xyp not found: {event_xyp_path}")

    index_path = _event_index_path(seq_dir, event_window_ms)
    index = _load_event_index(str(index_path))
    row = int(frame_id) - 1
    if row < 0 or row >= len(index):
        raise IndexError(
            f"DDD17 frame id out of event index range: sequence={seq_dir.name}, "
            f"frame_id={frame_id}, index_rows={len(index)}"
        )
    timestamp = int(index[row][0])
    half_window = int(round(float(event_window_ms) * DDD17_EVENT_TIME_UNITS_PER_MS))
    record.update(
        {
            "event_dat_t": str(event_t_path),
            "event_dat_xyp": str(event_xyp_path),
            "event_old": [timestamp - half_window, timestamp],
            "event_new": [timestamp, timestamp + half_window],
            "event_timestamp_ns": timestamp,
            "event_window_ms": int(event_window_ms),
            "event_index_file": str(index_path),
            "event_index_row": row,
            "event_sensor_width": DDD17_SENSOR_WIDTH,
            "event_sensor_height": DDD17_SENSOR_HEIGHT,
            "event_time_units_per_ms": DDD17_EVENT_TIME_UNITS_PER_MS,
        }
    )


def load_ddd17_labeled_dicts(
    split="train",
    root=DDD17_DEFAULT_ROOT,
    with_event=False,
    event_window_ms=DDD17_DEFAULT_EVENT_WINDOW_MS,
):
    root = ddd17_root(root)
    records = []
    for dir_name in _split_dirs(split):
        seq_dir = root / "data" / dir_name
        mask_dir = seq_dir / "segmentation_masks"
        if not mask_dir.is_dir():
            raise FileNotFoundError(f"DDD17 mask directory not found: {mask_dir}")
        for local_idx, mask_path in enumerate(sorted(mask_dir.glob("segmentation_*.png"))):
            frame_id = _frame_id_from_mask(mask_path)
            image_path = _image_for_frame(seq_dir, frame_id)
            record = {
                "file_name": str(image_path),
                "sem_seg_file_name": str(mask_path),
                "image_id": f"ddd17_{split}_{dir_name}_{frame_id:08d}",
                "sequence": dir_name,
                "frame_id": frame_id,
                "source": f"ddd17_{split}",
                "ddd17_local_index": local_idx,
            }
            if with_event:
                _add_event_fields(record, seq_dir, frame_id, event_window_ms)
            records.append(record)
    return records


def load_ddd17_train_dicts():
    return load_ddd17_labeled_dicts("train")


def load_ddd17_train_event_dicts():
    return load_ddd17_labeled_dicts("train", with_event=True)


def load_ddd17_val_dicts():
    return load_ddd17_labeled_dicts("val")


def load_ddd17_val_event_dicts():
    return load_ddd17_labeled_dicts("val", with_event=True)


def load_ddd17_full_labeled_dicts():
    return load_ddd17_labeled_dicts("full_labeled")


def load_ddd17_full_labeled_event_dicts():
    return load_ddd17_labeled_dicts("full_labeled", with_event=True)


def load_ddd17_dir5_dicts():
    return load_ddd17_labeled_dicts("dir5")


def load_ddd17_dir5_event_dicts():
    return load_ddd17_labeled_dicts("dir5", with_event=True)


if __name__ == "__main__":
    for split_name in ("train", "val", "dir5", "full_labeled"):
        records = load_ddd17_labeled_dicts(split_name)
        print(f"{split_name}: {len(records)}")
