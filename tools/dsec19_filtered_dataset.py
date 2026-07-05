import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from cosec_finetune_splits import WORKSPACE_ROOT


DSEC_ROOT = WORKSPACE_ROOT / "swin_l" / "data" / "dsec"
DSEC_FILTERED_630_MANIFEST = (
    WORKSPACE_ROOT
    / "BRENet"
    / "projects"
    / "brenet_cosec"
    / "manifests"
    / "dsec19_filtered_medium_more_630.json"
)
DSEC_DEFAULT_VAL_SEQUENCES = ("zurich_city_06_a", "zurich_city_07_a", "zurich_city_08_a")
DSEC_DEFAULT_EVENT_WINDOW_US = 50_000

DSEC_CLOSE_180_SEQUENCE_QUOTAS = (
    ("zurich_city_00_a", 80),
    ("zurich_city_04_a", 80),
    ("zurich_city_05_a", 20),
)

DSEC_CLOSE_240_SEQUENCE_QUOTAS = (
    ("zurich_city_00_a", 100),
    ("zurich_city_04_a", 110),
    ("zurich_city_05_a", 30),
)


def _evenly_spaced_subset(samples, keep_count):
    if keep_count >= len(samples):
        return list(samples)
    if keep_count <= 0:
        return []
    selected = []
    for rank in range(keep_count):
        index = int((rank + 0.5) * len(samples) / keep_count)
        selected.append(samples[min(index, len(samples) - 1)])
    return selected


def _load_manifest_samples(manifest_path):
    manifest_path = Path(manifest_path)
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)["samples"]


@lru_cache(maxsize=None)
def _load_sequence_timestamps(root, sequence):
    timestamps_path = Path(root) / "train_image" / sequence / "images" / "timestamps.txt"
    if not timestamps_path.exists():
        raise FileNotFoundError(f"DSEC timestamps not found: {timestamps_path}")
    with timestamps_path.open("r", encoding="utf-8") as f:
        return tuple(int(line.strip()) for line in f if line.strip())


def _add_event_fields(record, sample, root, event_window_us):
    sequence = sample["sequence"]
    frame_id = int(sample["frame_id"])
    timestamps = _load_sequence_timestamps(str(root), sequence)
    if frame_id >= len(timestamps):
        raise IndexError(
            f"DSEC timestamp index out of range: sequence={sequence}, frame_id={frame_id}, "
            f"timestamps={len(timestamps)}"
        )
    event_path = root / "train_event" / sequence / "events" / "left" / "events.h5"
    if not event_path.exists():
        raise FileNotFoundError(f"DSEC events not found: {event_path}")
    timestamp_us = int(timestamps[frame_id])
    record.update(
        {
            "event_h5": str(event_path),
            "event_old": [timestamp_us - int(event_window_us), timestamp_us],
            "event_new": [timestamp_us, timestamp_us + int(event_window_us)],
            "event_timestamp_us": timestamp_us,
        }
    )


def _samples_to_records(samples, root, source_name, with_event=False, event_window_us=DSEC_DEFAULT_EVENT_WINDOW_US):
    root = Path(root)
    records = []
    for idx, sample in enumerate(samples):
        img_path = root / sample["image"]
        label_path = root / sample["label"]
        if not img_path.exists():
            raise FileNotFoundError(f"DSEC image not found: {img_path}")
        if not label_path.exists():
            raise FileNotFoundError(f"DSEC label not found: {label_path}")
        record = {
            "file_name": str(img_path),
            "sem_seg_file_name": str(label_path),
            "image_id": f"dsec19_{source_name}_{sample['sequence']}_{int(sample['frame_id']):06d}_{idx:04d}",
            "sequence": sample["sequence"],
            "source": source_name,
        }
        if with_event:
            _add_event_fields(record, sample, root, event_window_us)
        records.append(record)
    return records


def load_dsec19_filtered_dicts(manifest_path=DSEC_FILTERED_630_MANIFEST, root=DSEC_ROOT):
    return _samples_to_records(
        _load_manifest_samples(manifest_path),
        root,
        "dsec19_filtered630",
    )


def load_dsec19_filtered_event_dicts(manifest_path=DSEC_FILTERED_630_MANIFEST, root=DSEC_ROOT):
    return _samples_to_records(
        _load_manifest_samples(manifest_path),
        root,
        "dsec19_filtered630_event",
        with_event=True,
    )


def _iter_full_dsec19_samples(root=DSEC_ROOT, include_val_sequences=True, only_val_sequences=False):
    root = Path(root)
    semantic_root = root / "train_semantic_segmentation"
    for label_dir in sorted(semantic_root.glob("*/19classes")):
        sequence = label_dir.parent.name
        if only_val_sequences and sequence not in DSEC_DEFAULT_VAL_SEQUENCES:
            continue
        if not include_val_sequences and sequence in DSEC_DEFAULT_VAL_SEQUENCES:
            continue
        for label_path in sorted(label_dir.glob("*.png")):
            frame_id = int(label_path.stem)
            image_path = root / "train_image" / sequence / "images" / "left" / "rectified" / label_path.name
            if not image_path.exists():
                continue
            yield {
                "sequence": sequence,
                "frame_id": frame_id,
                "image": str(image_path.relative_to(root)),
                "label": str(label_path.relative_to(root)),
            }


def load_dsec19_full_dicts(root=DSEC_ROOT, include_val_sequences=True):
    return _samples_to_records(
        list(_iter_full_dsec19_samples(root, include_val_sequences)),
        root,
        "dsec19_full",
    )


def load_dsec19_train_split_dicts(root=DSEC_ROOT):
    return _samples_to_records(
        list(_iter_full_dsec19_samples(root, include_val_sequences=False)),
        root,
        "dsec19_train_noval",
    )


def load_dsec19_val_dicts(root=DSEC_ROOT):
    return _samples_to_records(
        list(_iter_full_dsec19_samples(root, only_val_sequences=True)),
        root,
        "dsec19_val",
    )


def load_dsec19_train_split_event_dicts(root=DSEC_ROOT):
    return _samples_to_records(
        list(_iter_full_dsec19_samples(root, include_val_sequences=False)),
        root,
        "dsec19_train_noval_event",
        with_event=True,
    )


def load_dsec19_val_event_dicts(root=DSEC_ROOT):
    return _samples_to_records(
        list(_iter_full_dsec19_samples(root, only_val_sequences=True)),
        root,
        "dsec19_val_event",
        with_event=True,
    )


def load_dsec19_close_dicts(
    sequence_quotas=DSEC_CLOSE_180_SEQUENCE_QUOTAS,
    manifest_path=DSEC_FILTERED_630_MANIFEST,
    root=DSEC_ROOT,
    source_name="dsec19_close180",
):
    manifest_path = Path(manifest_path)
    samples_by_sequence = defaultdict(list)
    for sample in _load_manifest_samples(manifest_path):
        samples_by_sequence[sample["sequence"]].append(sample)

    selected = []
    for sequence, quota in sequence_quotas:
        selected.extend(_evenly_spaced_subset(samples_by_sequence[sequence], quota))
    return _samples_to_records(selected, root, source_name)


def load_dsec19_close180_dicts():
    return load_dsec19_close_dicts(
        DSEC_CLOSE_180_SEQUENCE_QUOTAS,
        source_name="dsec19_close180",
    )


def load_dsec19_close240_dicts():
    return load_dsec19_close_dicts(
        DSEC_CLOSE_240_SEQUENCE_QUOTAS,
        source_name="dsec19_close240",
    )
