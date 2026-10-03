import json
import math
import os
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageStat

from cosec_finetune_splits import WORKSPACE_ROOT


BRENET_ROOT = Path(os.environ.get("BRENET_ROOT", WORKSPACE_ROOT / "BRENet")).expanduser()
DSEC_ROOT = Path(os.environ.get("DSEC_ROOT", "/data/dsec")).expanduser()
DSEC_BRENET_ROOT = Path(
    os.environ.get("DSEC_BRENET_ROOT", "/data/dsec/BRENet")
).expanduser()
DSEC_FILTERED_630_MANIFEST = Path(
    os.environ.get(
        "DSEC_FILTERED_630_MANIFEST",
        BRENET_ROOT / "projects" / "brenet_cosec" / "manifests" / "dsec19_filtered_medium_more_630.json",
    )
).expanduser()
DSEC_DEFAULT_VAL_SEQUENCES = ("zurich_city_06_a", "zurich_city_07_a", "zurich_city_08_a")
DSEC_EXTERNAL_VAL_SEQUENCES = ("zurich_city_13_a", "zurich_city_14_c", "zurich_city_15_a")
DSEC_DEFAULT_EVENT_WINDOW_US = 50_000
DSEC_LOWLIGHT_VAL_QUANTILE = 0.25
DSEC_VAL_BRIGHTNESS_MANIFEST = Path(
    os.environ.get(
        "DSEC_VAL_BRIGHTNESS_MANIFEST",
        Path(__file__).resolve().parents[1]
        / "configs"
        / "generated"
        / "dsec19_eventshift"
        / "manifests"
        / "dsec19_val_brightness_q25.json",
    )
).expanduser()
DSEC11_EXTERNAL_VAL_MANIFEST = Path(
    os.environ.get(
        "DSEC11_EXTERNAL_VAL_MANIFEST",
        Path(__file__).resolve().parents[1]
        / "configs"
        / "generated"
        / "dsec19_eventshift"
        / "manifests"
        / "dsec11_external_val_protocols.json",
    )
).expanduser()

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

DSEC11_CLASSES = (
    "background",
    "building",
    "fence",
    "person",
    "pole",
    "road",
    "sidewalk",
    "vegetation",
    "car",
    "wall",
    "traffic sign",
)

DSEC11_PALETTE = (
    (0, 0, 0),
    (70, 70, 70),
    (190, 153, 153),
    (220, 20, 60),
    (153, 153, 153),
    (128, 64, 128),
    (244, 35, 232),
    (107, 142, 35),
    (0, 0, 142),
    (102, 102, 156),
    (220, 220, 0),
)

# Official DSEC-Semantic 19-to-11 grouping inferred from paired 19classes and
# 11classes labels. Class order for the 19-side follows Cityscapes/CoSEC.
DSEC19_TO_DSEC11 = (
    5,   # road -> road
    6,   # sidewalk -> sidewalk
    1,   # building -> building
    9,   # wall -> wall
    2,   # fence -> fence
    4,   # pole -> pole
    10,  # traffic light -> traffic sign
    10,  # traffic sign -> traffic sign
    7,   # vegetation -> vegetation
    7,   # terrain -> vegetation
    0,   # sky -> background
    3,   # person -> person
    3,   # rider -> person
    8,   # car -> car
    8,   # truck -> car
    8,   # bus -> car
    8,   # train -> car
    8,   # motorcycle -> car
    8,   # bicycle -> car
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
def _load_sequence_timestamps(root, sequence, dsec_split="train"):
    timestamps_path = Path(root) / f"{dsec_split}_image" / sequence / "images" / "timestamps.txt"
    if not timestamps_path.exists():
        raise FileNotFoundError(f"DSEC timestamps not found: {timestamps_path}")
    with timestamps_path.open("r", encoding="utf-8") as f:
        return tuple(int(line.strip()) for line in f if line.strip())


@lru_cache(maxsize=None)
def _load_brenet_sequence_timestamps(root, sequence, split="train"):
    timestamps_path = Path(root) / split / sequence / "images" / "timestamps.txt"
    if not timestamps_path.exists():
        raise FileNotFoundError(f"DSEC_BRENet timestamps not found: {timestamps_path}")
    with timestamps_path.open("r", encoding="utf-8") as f:
        return tuple(int(line.strip()) for line in f if line.strip())


def _add_event_fields(record, sample, root, event_window_us):
    root = Path(root)
    sequence = sample["sequence"]
    frame_id = int(sample["frame_id"])
    dsec_split = sample.get("dsec_split", "train")
    if "timestamp_us" in sample:
        timestamp_us = int(sample["timestamp_us"])
    else:
        timestamps = _load_sequence_timestamps(str(root), sequence, dsec_split)
        if frame_id >= len(timestamps):
            raise IndexError(
                f"DSEC timestamp index out of range: sequence={sequence}, frame_id={frame_id}, "
                f"timestamps={len(timestamps)}"
            )
        timestamp_us = int(timestamps[frame_id])

    if "event_h5" in sample:
        event_path = Path(sample["event_h5"])
        if not event_path.is_absolute():
            event_path = root / event_path
    else:
        event_path = root / f"{dsec_split}_event" / sequence / "events" / "left" / "events.h5"
    if not event_path.exists():
        raise FileNotFoundError(f"DSEC events not found: {event_path}")

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
        for key in (
            "frame_id",
            "luminance_y",
            "luminance_split",
            "luminance_quantile",
            "dsec_split",
            "benchmark_protocol",
            "timestamp_us",
            "image_crop_to_sem_seg",
            "event_sensor_width",
            "event_sensor_height",
            "event_time_units_per_ms",
            "event_rectify_map",
        ):
            if key in sample:
                record[key] = sample[key]
        if "event_rectify_map" in record:
            rectify_map_path = Path(record["event_rectify_map"])
            if not rectify_map_path.is_absolute():
                record["event_rectify_map"] = str(root / rectify_map_path)
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


def _iter_full_dsec19_samples(
    root=DSEC_ROOT,
    include_val_sequences=True,
    only_val_sequences=False,
    label_set="19classes",
):
    root = Path(root)
    semantic_root = root / "train_semantic_segmentation"
    for label_dir in sorted(semantic_root.glob(f"*/{label_set}")):
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
                "dsec_split": "train",
            }


def _with_dsec_label_set(samples, label_set):
    if label_set == "19classes":
        return [dict(sample) for sample in samples]
    remapped = []
    old_token = "/19classes/"
    new_token = f"/{label_set}/"
    for sample in samples:
        item = dict(sample)
        label = item["label"]
        if old_token not in label:
            raise ValueError(f"Expected DSEC label path to contain {old_token!r}: {label}")
        item["label"] = label.replace(old_token, new_token, 1)
        remapped.append(item)
    return remapped



def _iter_official_test_samples(root=DSEC_ROOT, label_set="11classes"):
    root = Path(root)
    semantic_root = root / "test_semantic_segmentation"
    for sequence in DSEC_EXTERNAL_VAL_SEQUENCES:
        label_dir = semantic_root / sequence / label_set
        for label_path in sorted(label_dir.glob("*.png")):
            frame_id = int(label_path.stem)
            image_path = root / "test_image" / sequence / "images" / "left" / "rectified" / label_path.name
            if not image_path.exists():
                continue
            yield {
                "sequence": sequence,
                "frame_id": frame_id,
                "image": str(image_path.relative_to(root)),
                "label": str(label_path.relative_to(root)),
                "dsec_split": "test",
            }


def _compute_external_val_protocol_splits(root=DSEC_ROOT):
    root = Path(root)
    splits = {name: [] for name in ("full", "brenet", "mambaseg", "common")}
    for sequence in DSEC_EXTERNAL_VAL_SEQUENCES:
        image_dir = root / "test_image" / sequence / "images" / "left" / "rectified"
        label_dir = root / "test_semantic_segmentation" / sequence / "11classes"
        image_names = sorted((path.name for path in image_dir.glob("*.png")), key=lambda name: int(Path(name).stem))
        label_names = sorted((path.name for path in label_dir.glob("*.png")), key=lambda name: int(Path(name).stem))
        paired_names = set(image_names) & set(label_names)

        full_names = [name for name in image_names if name in paired_names]
        brenet_names = [
            name
            for index, name in enumerate(image_names)
            if index > 5 and index < len(image_names) - 6 and name in paired_names
        ]
        trimmed_label_names = label_names[6:]
        mambaseg_names = [
            trimmed_label_names[index * 2]
            for index in range((len(trimmed_label_names) + 1) // 2)
            if index * 2 < len(trimmed_label_names) and trimmed_label_names[index * 2] in paired_names
        ]
        common_names = sorted(set(brenet_names) & set(mambaseg_names), key=lambda name: int(Path(name).stem))

        for split_name, names in (
            ("full", full_names),
            ("brenet", brenet_names),
            ("mambaseg", mambaseg_names),
            ("common", common_names),
        ):
            for name in names:
                frame_id = int(Path(name).stem)
                splits[split_name].append(
                    {
                        "sequence": sequence,
                        "frame_id": frame_id,
                        "image": str(
                            (root / "test_image" / sequence / "images" / "left" / "rectified" / name).relative_to(root)
                        ),
                        "label": str(
                            (root / "test_semantic_segmentation" / sequence / "11classes" / name).relative_to(root)
                        ),
                        "label19": str(
                            (root / "test_semantic_segmentation" / sequence / "19classes" / name).relative_to(root)
                        ),
                        "dsec_split": "test",
                        "benchmark_protocol": split_name,
                    }
                )
    return splits


def _load_external_val_protocol_samples(split_name, root=DSEC_ROOT, manifest_path=DSEC11_EXTERNAL_VAL_MANIFEST):
    manifest_path = Path(manifest_path)
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        splits = payload.get("splits", {})
        if split_name not in splits:
            raise KeyError(f"DSEC11 external split {split_name!r} not found in {manifest_path}")
        return [dict(sample) for sample in splits[split_name]]
    return _compute_external_val_protocol_splits(root)[split_name]

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



def _iter_brenet11_samples(
    root=DSEC_BRENET_ROOT,
    split="train",
    include_val_sequences=True,
    only_val_sequences=False,
):
    root = Path(root)
    split_root = root / split
    if not split_root.exists():
        raise FileNotFoundError(f"DSEC_BRENet split not found: {split_root}")
    for sequence_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
        sequence = sequence_dir.name
        if split == "train":
            if only_val_sequences and sequence not in DSEC_DEFAULT_VAL_SEQUENCES:
                continue
            if not include_val_sequences and sequence in DSEC_DEFAULT_VAL_SEQUENCES:
                continue
        label_dir = sequence_dir / "11classes"
        image_dir = sequence_dir / "images" / "left" / "ev_inf"
        event_path = sequence_dir / "events" / "left" / "events.h5"
        rectify_map_path = sequence_dir / "events" / "left" / "rectify_map.h5"
        if not label_dir.exists() or not image_dir.exists() or not event_path.exists() or not rectify_map_path.exists():
            continue
        timestamps = _load_brenet_sequence_timestamps(str(root), sequence, split)
        for label_path in sorted(label_dir.glob("*.png")):
            frame_id = int(label_path.stem)
            image_path = image_dir / label_path.name
            if not image_path.exists() or frame_id >= len(timestamps):
                continue
            yield {
                "sequence": sequence,
                "frame_id": frame_id,
                "image": str(image_path.relative_to(root)),
                "label": str(label_path.relative_to(root)),
                "dsec_split": split,
                "timestamp_us": int(timestamps[frame_id]),
                "event_h5": str(event_path.relative_to(root)),
                "event_rectify_map": str(rectify_map_path.relative_to(root)),
                "image_crop_to_sem_seg": True,
                "event_time_units_per_ms": 1000,
            }


def load_dsec_brenet11_train_full_dicts(root=DSEC_BRENET_ROOT):
    return _samples_to_records(
        list(_iter_brenet11_samples(root, split="train", include_val_sequences=True)),
        root,
        "dsec_brenet11_train_full",
    )


def load_dsec_brenet11_train_full_event_dicts(root=DSEC_BRENET_ROOT):
    return _samples_to_records(
        list(_iter_brenet11_samples(root, split="train", include_val_sequences=True)),
        root,
        "dsec_brenet11_train_full_event",
        with_event=True,
    )


def load_dsec_brenet11_train_split_dicts(root=DSEC_BRENET_ROOT):
    return _samples_to_records(
        list(_iter_brenet11_samples(root, split="train", include_val_sequences=False)),
        root,
        "dsec_brenet11_train_noval",
    )


def load_dsec_brenet11_train_split_event_dicts(root=DSEC_BRENET_ROOT):
    return _samples_to_records(
        list(_iter_brenet11_samples(root, split="train", include_val_sequences=False)),
        root,
        "dsec_brenet11_train_noval_event",
        with_event=True,
    )


def load_dsec_brenet11_val_dicts(root=DSEC_BRENET_ROOT):
    return _samples_to_records(
        list(_iter_brenet11_samples(root, split="train", only_val_sequences=True)),
        root,
        "dsec_brenet11_val",
    )


def load_dsec_brenet11_val_event_dicts(root=DSEC_BRENET_ROOT):
    return _samples_to_records(
        list(_iter_brenet11_samples(root, split="train", only_val_sequences=True)),
        root,
        "dsec_brenet11_val_event",
        with_event=True,
    )


def load_dsec_brenet11_test_dicts(root=DSEC_BRENET_ROOT):
    return _samples_to_records(
        list(_iter_brenet11_samples(root, split="test", include_val_sequences=True)),
        root,
        "dsec_brenet11_test",
    )


def load_dsec_brenet11_test_event_dicts(root=DSEC_BRENET_ROOT):
    return _samples_to_records(
        list(_iter_brenet11_samples(root, split="test", include_val_sequences=True)),
        root,
        "dsec_brenet11_test_event",
        with_event=True,
    )


def load_dsec11_train_full_dicts(root=DSEC_ROOT):
    return _samples_to_records(
        list(_iter_full_dsec19_samples(root, include_val_sequences=True, label_set="11classes")),
        root,
        "dsec11_train_full",
    )


def load_dsec11_train_full_event_dicts(root=DSEC_ROOT):
    return _samples_to_records(
        list(_iter_full_dsec19_samples(root, include_val_sequences=True, label_set="11classes")),
        root,
        "dsec11_train_full_event",
        with_event=True,
    )

def load_dsec11_train_split_dicts(root=DSEC_ROOT):
    return _samples_to_records(
        list(_iter_full_dsec19_samples(root, include_val_sequences=False, label_set="11classes")),
        root,
        "dsec11_train_noval",
    )


def load_dsec11_train_split_event_dicts(root=DSEC_ROOT):
    return _samples_to_records(
        list(_iter_full_dsec19_samples(root, include_val_sequences=False, label_set="11classes")),
        root,
        "dsec11_train_noval_event",
        with_event=True,
    )


def load_dsec11_val_dicts(root=DSEC_ROOT):
    return _samples_to_records(
        list(_iter_full_dsec19_samples(root, only_val_sequences=True, label_set="11classes")),
        root,
        "dsec11_val",
    )


def load_dsec11_val_event_dicts(root=DSEC_ROOT):
    return _samples_to_records(
        list(_iter_full_dsec19_samples(root, only_val_sequences=True, label_set="11classes")),
        root,
        "dsec11_val_event",
        with_event=True,
    )



def _external_protocol_samples_with_label_set(split_name, label_set, root=DSEC_ROOT):
    samples = _load_external_val_protocol_samples(split_name, root)
    if label_set == "11classes":
        return samples
    if label_set != "19classes":
        raise ValueError(f"Unsupported DSEC external label set: {label_set}")
    remapped = []
    for sample in samples:
        item = dict(sample)
        if "label19" in item:
            item["label"] = item["label19"]
        else:
            item["label"] = item["label"].replace("/11classes/", "/19classes/", 1)
        remapped.append(item)
    return remapped


def load_dsec19_external_full_val_dicts(root=DSEC_ROOT):
    return _samples_to_records(
        _external_protocol_samples_with_label_set("full", "19classes", root),
        root,
        "dsec19_external_full_val",
    )


def load_dsec19_external_full_val_event_dicts(root=DSEC_ROOT):
    return _samples_to_records(
        _external_protocol_samples_with_label_set("full", "19classes", root),
        root,
        "dsec19_external_full_val_event",
        with_event=True,
    )


def load_dsec19_external_brenet_val_dicts(root=DSEC_ROOT):
    return _samples_to_records(
        _external_protocol_samples_with_label_set("brenet", "19classes", root),
        root,
        "dsec19_external_brenet_val",
    )


def load_dsec19_external_brenet_val_event_dicts(root=DSEC_ROOT):
    return _samples_to_records(
        _external_protocol_samples_with_label_set("brenet", "19classes", root),
        root,
        "dsec19_external_brenet_val_event",
        with_event=True,
    )


def load_dsec19_external_mambaseg_val_dicts(root=DSEC_ROOT):
    return _samples_to_records(
        _external_protocol_samples_with_label_set("mambaseg", "19classes", root),
        root,
        "dsec19_external_mambaseg_val",
    )


def load_dsec19_external_mambaseg_val_event_dicts(root=DSEC_ROOT):
    return _samples_to_records(
        _external_protocol_samples_with_label_set("mambaseg", "19classes", root),
        root,
        "dsec19_external_mambaseg_val_event",
        with_event=True,
    )


def load_dsec19_external_common_val_dicts(root=DSEC_ROOT):
    return _samples_to_records(
        _external_protocol_samples_with_label_set("common", "19classes", root),
        root,
        "dsec19_external_common_val",
    )


def load_dsec19_external_common_val_event_dicts(root=DSEC_ROOT):
    return _samples_to_records(
        _external_protocol_samples_with_label_set("common", "19classes", root),
        root,
        "dsec19_external_common_val_event",
        with_event=True,
    )

def load_dsec11_external_full_val_dicts(root=DSEC_ROOT):
    return _samples_to_records(
        _load_external_val_protocol_samples("full", root),
        root,
        "dsec11_external_full_val",
    )


def load_dsec11_external_full_val_event_dicts(root=DSEC_ROOT):
    return _samples_to_records(
        _load_external_val_protocol_samples("full", root),
        root,
        "dsec11_external_full_val_event",
        with_event=True,
    )


def load_dsec11_external_brenet_val_dicts(root=DSEC_ROOT):
    return _samples_to_records(
        _load_external_val_protocol_samples("brenet", root),
        root,
        "dsec11_external_brenet_val",
    )


def load_dsec11_external_brenet_val_event_dicts(root=DSEC_ROOT):
    return _samples_to_records(
        _load_external_val_protocol_samples("brenet", root),
        root,
        "dsec11_external_brenet_val_event",
        with_event=True,
    )


def load_dsec11_external_mambaseg_val_dicts(root=DSEC_ROOT):
    return _samples_to_records(
        _load_external_val_protocol_samples("mambaseg", root),
        root,
        "dsec11_external_mambaseg_val",
    )


def load_dsec11_external_mambaseg_val_event_dicts(root=DSEC_ROOT):
    return _samples_to_records(
        _load_external_val_protocol_samples("mambaseg", root),
        root,
        "dsec11_external_mambaseg_val_event",
        with_event=True,
    )


def load_dsec11_external_common_val_dicts(root=DSEC_ROOT):
    return _samples_to_records(
        _load_external_val_protocol_samples("common", root),
        root,
        "dsec11_external_common_val",
    )


def load_dsec11_external_common_val_event_dicts(root=DSEC_ROOT):
    return _samples_to_records(
        _load_external_val_protocol_samples("common", root),
        root,
        "dsec11_external_common_val_event",
        with_event=True,
    )


@lru_cache(maxsize=None)
def _mean_image_luminance(image_path):
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        red, green, blue = ImageStat.Stat(rgb).mean[:3]
    return 0.299 * red + 0.587 * green + 0.114 * blue


def _load_dsec19_val_brightness_manifest_split(split_name, manifest_path=DSEC_VAL_BRIGHTNESS_MANIFEST):
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return None
    with manifest_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    splits = payload.get("splits", {})
    if split_name not in splits:
        raise KeyError(f"DSEC brightness split {split_name!r} not found in {manifest_path}")
    return [dict(sample) for sample in splits[split_name]]


def _dsec19_val_brightness_split_samples(root=DSEC_ROOT, lowlight=True, quantile=DSEC_LOWLIGHT_VAL_QUANTILE):
    split_name = "lowlight_q25" if lowlight else "remaining_q75"
    manifest_samples = _load_dsec19_val_brightness_manifest_split(split_name)
    if manifest_samples is not None:
        return manifest_samples

    root = Path(root)
    samples = []
    for sample in _iter_full_dsec19_samples(root, only_val_sequences=True):
        enriched = dict(sample)
        enriched["luminance_y"] = float(_mean_image_luminance(str(root / sample["image"])))
        enriched["luminance_quantile"] = float(quantile)
        samples.append(enriched)

    lowlight_keys = set()
    samples_by_sequence = defaultdict(list)
    for sample in samples:
        samples_by_sequence[sample["sequence"]].append(sample)

    for sequence, sequence_samples in samples_by_sequence.items():
        sorted_samples = sorted(
            sequence_samples,
            key=lambda item: (item["luminance_y"], int(item["frame_id"])),
        )
        keep_count = int(math.ceil(len(sorted_samples) * float(quantile)))
        for sample in sorted_samples[:keep_count]:
            lowlight_keys.add((sample["sequence"], int(sample["frame_id"])))

    selected = []
    for sample in samples:
        key = (sample["sequence"], int(sample["frame_id"]))
        is_lowlight = key in lowlight_keys
        if is_lowlight == bool(lowlight):
            enriched = dict(sample)
            enriched["luminance_split"] = split_name
            selected.append(enriched)
    return selected


def load_dsec19_val_lowlight_q25_dicts(root=DSEC_ROOT):
    return _samples_to_records(
        _dsec19_val_brightness_split_samples(root, lowlight=True),
        root,
        "dsec19_val_lowlight_q25",
    )


def load_dsec19_val_lowlight_q25_event_dicts(root=DSEC_ROOT):
    return _samples_to_records(
        _dsec19_val_brightness_split_samples(root, lowlight=True),
        root,
        "dsec19_val_lowlight_q25_event",
        with_event=True,
    )


def load_dsec19_val_remaining_q75_dicts(root=DSEC_ROOT):
    return _samples_to_records(
        _dsec19_val_brightness_split_samples(root, lowlight=False),
        root,
        "dsec19_val_remaining_q75",
    )


def load_dsec19_val_remaining_q75_event_dicts(root=DSEC_ROOT):
    return _samples_to_records(
        _dsec19_val_brightness_split_samples(root, lowlight=False),
        root,
        "dsec19_val_remaining_q75_event",
        with_event=True,
    )


def load_dsec11_val_lowlight_q25_dicts(root=DSEC_ROOT):
    return _samples_to_records(
        _with_dsec_label_set(_dsec19_val_brightness_split_samples(root, lowlight=True), "11classes"),
        root,
        "dsec11_val_lowlight_q25",
    )


def load_dsec11_val_lowlight_q25_event_dicts(root=DSEC_ROOT):
    return _samples_to_records(
        _with_dsec_label_set(_dsec19_val_brightness_split_samples(root, lowlight=True), "11classes"),
        root,
        "dsec11_val_lowlight_q25_event",
        with_event=True,
    )


def load_dsec11_val_remaining_q75_dicts(root=DSEC_ROOT):
    return _samples_to_records(
        _with_dsec_label_set(_dsec19_val_brightness_split_samples(root, lowlight=False), "11classes"),
        root,
        "dsec11_val_remaining_q75",
    )


def load_dsec11_val_remaining_q75_event_dicts(root=DSEC_ROOT):
    return _samples_to_records(
        _with_dsec_label_set(_dsec19_val_brightness_split_samples(root, lowlight=False), "11classes"),
        root,
        "dsec11_val_remaining_q75_event",
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
