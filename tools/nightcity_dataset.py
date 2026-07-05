"""NightCity dataset registration helpers for the Swin-L/Mask2Former pipeline."""

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image


def _eventshift_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "configs").is_dir() and (parent / "third_party").is_dir():
            return parent
    return Path(__file__).resolve().parents[1]


ROOT = _eventshift_root()
DEFAULT_NIGHTCITY_ROOTS = (
    Path(os.environ.get("NIGHTCITY_ROOT", "")),
    Path("/work/u1621738/ebmv_eccv/MambaSeg/data/nightcity"),
    ROOT / "data" / "nightcity",
)

# NightCity labels are Cityscapes labelIds. The current CoSEC/Swin-L mapper
# expects contiguous trainIds 0..18 and 255 for ignored labels.
CITYSCAPES_LABEL_ID_TO_TRAIN_ID = np.full(256, 255, dtype=np.uint8)
for label_id, train_id in {
    7: 0,    # road
    8: 1,    # sidewalk
    11: 2,   # building
    12: 3,   # wall
    13: 4,   # fence
    17: 5,   # pole
    19: 6,   # traffic light
    20: 7,   # traffic sign
    21: 8,   # vegetation
    22: 9,   # terrain
    23: 10,  # sky
    24: 11,  # person
    25: 12,  # rider
    26: 13,  # car
    27: 14,  # truck
    28: 15,  # bus
    31: 16,  # train
    32: 17,  # motorcycle
    33: 18,  # bicycle
}.items():
    CITYSCAPES_LABEL_ID_TO_TRAIN_ID[label_id] = train_id


def _valid_nightcity_root(path):
    return (
        path
        and (path / "NightCity-images" / "images").is_dir()
        and (path / "NightCity-label" / "label").is_dir()
    )


@lru_cache(maxsize=1)
def nightcity_root():
    for path in DEFAULT_NIGHTCITY_ROOTS:
        if _valid_nightcity_root(path):
            return path
    candidates = ", ".join(str(path) for path in DEFAULT_NIGHTCITY_ROOTS if str(path))
    raise FileNotFoundError(
        "NightCity root not found. Set NIGHTCITY_ROOT or place it under "
        f"{ROOT / 'data' / 'nightcity'}. Checked: {candidates}"
    )


def nightcity_label_cache_root():
    return ROOT / "work_dirs" / "cache" / "nightcity_trainIds" / "label"


def default_classdist_manifest():
    return ROOT / "work_dirs" / "manifests" / "nightcity_cosec_classdist_filtered.json"


def default_trainval_strict_classdist_manifest():
    return ROOT / "work_dirs" / "manifests" / "nightcity_trainval_cosec_classdist_strict_filtered.json"


def default_trainval_cosec_night_domain_patch_manifest():
    return ROOT / "work_dirs" / "manifests" / "nightcity_trainval_cosec_night_domain_patch_filtered_greedy.json"


def _image_dir(split):
    return nightcity_root() / "NightCity-images" / "images" / split


def _label_dir(split):
    return nightcity_root() / "NightCity-label" / "label" / split


def _cache_label_path(split, image_stem):
    return nightcity_label_cache_root() / split / f"{image_stem}_trainIds.png"


def _source_label_path(split, image_stem):
    return _label_dir(split) / f"{image_stem}_labelIds.png"


def _needs_update(src, dst):
    return not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime


def _write_train_id_label(src, dst):
    label = np.asarray(Image.open(src))
    if label.ndim == 3:
        label = label[:, :, 0]
    train_id_label = CITYSCAPES_LABEL_ID_TO_TRAIN_ID[label.astype(np.uint8, copy=False)]
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(train_id_label, mode="L").save(dst)


def _ensure_train_id_label(split, image_stem):
    src = _source_label_path(split, image_stem)
    if not src.exists():
        return None
    dst = _cache_label_path(split, image_stem)
    if _needs_update(src, dst):
        _write_train_id_label(src, dst)
    return dst


def _iter_split_records(split, limit=None):
    if split not in {"train", "val"}:
        raise ValueError(f"Unknown NightCity split: {split}")
    images = sorted(_image_dir(split).glob("*.png"))
    if limit is not None:
        images = images[: int(limit)]
    records = []
    for index, image_path in enumerate(images):
        label_path = _ensure_train_id_label(split, image_path.stem)
        if label_path is None:
            continue
        records.append(
            {
                "file_name": str(image_path),
                "sem_seg_file_name": str(label_path),
                "image_id": f"nightcity_{split}_{image_path.stem}",
                "nightcity_split": split,
                "nightcity_index": index,
            }
        )
    return records


@lru_cache(maxsize=None)
def _load_single_split(split):
    return tuple(_iter_split_records(split))


def load_nightcity_dicts(split="train", limit=None):
    if limit is not None:
        if split == "trainval":
            train_limit = int(limit)
            train_records = _iter_split_records("train", train_limit)
            if len(train_records) >= train_limit:
                return train_records[:train_limit]
            val_records = _iter_split_records("val", train_limit - len(train_records))
            return train_records + val_records
        return _iter_split_records(split, int(limit))
    if split == "trainval":
        records = list(_load_single_split("train")) + list(_load_single_split("val"))
    else:
        records = list(_load_single_split(split))
    return records


def load_nightcity_cosec_classdist_dicts(manifest_path=None, limit=None):
    manifest_path = Path(manifest_path) if manifest_path is not None else default_classdist_manifest()
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Filtered NightCity manifest not found: {manifest_path}. "
            "Run tools/filter_nightcity_by_cosec_distribution.py first."
        )
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    kept_entries = manifest.get("kept", [])
    if limit is not None:
        kept_entries = kept_entries[: int(limit)]

    records = []
    for index, entry in enumerate(kept_entries):
        records.append(
            {
                "file_name": entry["file_name"],
                "sem_seg_file_name": entry["sem_seg_file_name"],
                "image_id": entry["image_id"],
                "nightcity_split": entry.get("nightcity_split", "train"),
                "nightcity_filtered_index": index,
                "nightcity_classdist_score": entry.get("score"),
                "source": "nightcity_cosec_classdist",
            }
        )
    return records


def load_nightcity_trainval_cosec_classdist_strict_dicts(limit=None):
    return load_nightcity_cosec_classdist_dicts(default_trainval_strict_classdist_manifest(), limit=limit)


def load_nightcity_trainval_cosec_night_domain_patch_dicts(limit=None):
    records = load_nightcity_cosec_classdist_dicts(
        default_trainval_cosec_night_domain_patch_manifest(),
        limit=limit,
    )
    for record in records:
        record["source"] = "nightcity_cosec_night_domain_patch"
    return records
