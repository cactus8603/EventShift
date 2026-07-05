"""ACDC dataset registration helpers for the Swin-L/Mask2Former pipeline."""

import json
import os
from functools import lru_cache
from pathlib import Path


def _eventshift_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "configs").is_dir() and (parent / "third_party").is_dir():
            return parent
    return Path(__file__).resolve().parents[1]


ROOT = _eventshift_root()
WORKSPACE_ROOT = ROOT.parent
DEFAULT_ACDC_ROOTS = (
    Path(os.environ.get("ACDC_ROOT", "")),
    ROOT / "data" / "acdc",
)
DEFAULT_ACDC_SPLIT_DIRS = (
    Path(os.environ.get("ACDC_SPLIT_DIR", "")),
    WORKSPACE_ROOT / "unified_cosec_acdc" / "classcover_v1" / "splits" / "acdc",
    ROOT / "work_dirs" / "splits" / "acdc",
)
ACDC_CONDITIONS = ("fog", "night", "rain", "snow")
DEFAULT_ACDC_KFOLD_COUNT = 3


def _valid_acdc_root(path):
    return path and (path / "rgb_anon").is_dir() and (path / "gt").is_dir()


@lru_cache(maxsize=1)
def acdc_root():
    for path in DEFAULT_ACDC_ROOTS:
        if _valid_acdc_root(path):
            return path
    candidates = ", ".join(str(path) for path in DEFAULT_ACDC_ROOTS if str(path))
    raise FileNotFoundError(f"ACDC root not found. Checked: {candidates}")


def default_acdc_top50_manifest():
    return ROOT / "work_dirs" / "manifests" / "acdc_night_trainval_cosec_night_domain_patch_top50_filtered_greedy.json"


def _iter_acdc_records(conditions=("night",), splits=("train",), limit=None):
    root = acdc_root()
    records = []
    for condition in conditions:
        for split in splits:
            image_root = root / "rgb_anon" / condition / split
            label_root = root / "gt" / condition / split
            if not image_root.is_dir() or not label_root.is_dir():
                continue
            for image_path in sorted(image_root.glob("*/*_rgb_anon.png")):
                sequence = image_path.parent.name
                frame_stem = image_path.name.replace("_rgb_anon.png", "")
                label_path = label_root / sequence / f"{frame_stem}_gt_labelTrainIds.png"
                if not label_path.exists():
                    continue
                records.append(
                    {
                        "file_name": str(image_path),
                        "sem_seg_file_name": str(label_path),
                        "image_id": f"acdc_{condition}_{split}_{sequence}_{frame_stem}",
                        "acdc_condition": condition,
                        "acdc_split": split,
                        "acdc_sequence": sequence,
                        "acdc_frame": frame_stem,
                        "source": f"acdc_{condition}_{split}",
                    }
                )
                if limit is not None and len(records) >= int(limit):
                    return records
    return records


@lru_cache(maxsize=None)
def _load_acdc_cached(condition, split):
    return tuple(_iter_acdc_records((condition,), (split,)))


def _normalize_conditions(condition):
    if isinstance(condition, str):
        if condition == "all":
            return ACDC_CONDITIONS
        return (condition,)
    return tuple(condition)


def _normalize_splits(splits):
    if isinstance(splits, str):
        if splits == "trainval":
            return ("train", "val")
        return (splits,)
    return tuple(splits)


def load_acdc_dicts(condition="night", split="train", limit=None):
    conditions = _normalize_conditions(condition)
    splits = _normalize_splits(split)
    if len(conditions) == 1 and len(splits) == 1:
        records = list(_load_acdc_cached(conditions[0], splits[0]))
    else:
        records = _iter_acdc_records(conditions, splits)
    if limit is not None:
        records = records[: int(limit)]
    return records


def acdc_file_split_id(record):
    return "/".join(
        (
            record["acdc_condition"],
            record["acdc_split"],
            record["acdc_sequence"],
            record["acdc_frame"],
        )
    )


def iter_acdc_split_dirs():
    seen = set()
    for path in DEFAULT_ACDC_SPLIT_DIRS:
        if not path:
            continue
        resolved = Path(path)
        if resolved in seen or not resolved.is_dir():
            continue
        seen.add(resolved)
        yield resolved


def discover_acdc_file_split_prefixes():
    prefixes = set()
    for split_dir in iter_acdc_split_dirs():
        for path in split_dir.glob("*.txt"):
            stem = path.stem
            if stem.startswith("train_"):
                prefixes.add(stem[len("train_") :])
            elif stem.startswith("val_"):
                prefixes.add(stem[len("val_") :])
    return sorted(prefixes)


def _find_acdc_split_file(prefix, subset):
    name = f"{subset}_{prefix}.txt"
    for split_dir in iter_acdc_split_dirs():
        path = split_dir / name
        if path.exists():
            return path
    checked = ", ".join(str(path) for path in DEFAULT_ACDC_SPLIT_DIRS if str(path))
    raise FileNotFoundError(f"ACDC split file {name} not found. Checked: {checked}")


@lru_cache(maxsize=None)
def _load_acdc_file_split_ids(prefix, subset):
    path = _find_acdc_split_file(prefix, subset)
    with path.open("r", encoding="utf-8") as handle:
        return frozenset(line.strip() for line in handle if line.strip() and not line.lstrip().startswith("#"))


def load_acdc_file_split_dicts(prefix, subset="train", condition="all", limit=None):
    if subset not in {"train", "val"}:
        raise ValueError(f"Unknown ACDC file split subset: {subset}")
    selected_ids = _load_acdc_file_split_ids(prefix, subset)
    records = []
    for record in _iter_acdc_records(_normalize_conditions(condition), ("train", "val")):
        if acdc_file_split_id(record) not in selected_ids:
            continue
        item = dict(record)
        item["source"] = f"acdc_{prefix}_{condition}_{subset}"
        item["acdc_file_split_prefix"] = prefix
        item["acdc_file_split_subset"] = subset
        records.append(item)
        if limit is not None and len(records) >= int(limit):
            break
    return records


def iter_acdc_sequence_infos(condition="night", splits=("train", "val")):
    records = _iter_acdc_records(_normalize_conditions(condition), _normalize_splits(splits))
    grouped = {}
    for record in records:
        key = (record["acdc_condition"], record["acdc_sequence"])
        info = grouped.setdefault(
            key,
            {
                "condition": record["acdc_condition"],
                "sequence": record["acdc_sequence"],
                "record_count": 0,
                "splits": set(),
            },
        )
        info["record_count"] += 1
        info["splits"].add(record["acdc_split"])
    for condition_name, sequence_name in sorted(grouped):
        info = grouped[(condition_name, sequence_name)]
        yield {
            "condition": info["condition"],
            "sequence": info["sequence"],
            "record_count": info["record_count"],
            "splits": tuple(sorted(info["splits"])),
            "key": f"{info['condition']}:{info['sequence']}",
        }


@lru_cache(maxsize=None)
def _cached_acdc_kfold_sequence_sets(condition_key, split_key, folds):
    infos = tuple(iter_acdc_sequence_infos(condition_key, split_key))
    if len(infos) < folds:
        raise ValueError(f"Cannot build {folds}-fold ACDC split from only {len(infos)} sequences")

    val_sequence_sets = [set() for _ in range(folds)]
    for condition_name in condition_key:
        condition_infos = [info for info in infos if info["condition"] == condition_name]
        if not condition_infos:
            continue
        if len(condition_infos) < folds:
            raise ValueError(
                f"Cannot build condition-aware sequence-level {folds}-fold ACDC split: "
                f"{condition_name} has only {len(condition_infos)} sequences. "
                f"Use folds <= {len(condition_infos)} or fewer conditions."
            )
        fold_counts = [0 for _ in range(folds)]
        fold_seq_counts = [0 for _ in range(folds)]
        for info in sorted(condition_infos, key=lambda item: (-item["record_count"], item["key"])):
            fold_index = min(range(folds), key=lambda idx: (fold_counts[idx], fold_seq_counts[idx], idx))
            val_sequence_sets[fold_index].add(info["key"])
            fold_counts[fold_index] += int(info["record_count"])
            fold_seq_counts[fold_index] += 1

    all_sequences = frozenset(info["key"] for info in infos)
    return tuple(
        {
            "train": frozenset(all_sequences - frozenset(val_sequences)),
            "val": frozenset(val_sequences),
        }
        for val_sequences in val_sequence_sets
    )


def build_acdc_kfold_sequence_sets(condition="night", splits=("train", "val"), folds=DEFAULT_ACDC_KFOLD_COUNT):
    return _cached_acdc_kfold_sequence_sets(
        _normalize_conditions(condition),
        _normalize_splits(splits),
        int(folds),
    )


def load_acdc_kfold_dicts(condition="night", folds=DEFAULT_ACDC_KFOLD_COUNT, fold_index=0, subset="train", limit=None):
    folds = int(folds)
    fold_index = int(fold_index)
    if subset not in {"train", "val"}:
        raise ValueError(f"Unknown ACDC k-fold subset: {subset}")
    if fold_index < 0 or fold_index >= folds:
        raise ValueError(f"ACDC k-fold index out of range: {fold_index}/{folds}")

    conditions = _normalize_conditions(condition)
    splits = ("train", "val")
    sequence_sets = build_acdc_kfold_sequence_sets(conditions, splits, folds)[fold_index]
    selected_sequences = sequence_sets[subset]
    records = []
    for record in _iter_acdc_records(conditions, splits):
        key = f"{record['acdc_condition']}:{record['acdc_sequence']}"
        if key not in selected_sequences:
            continue
        item = dict(record)
        item["source"] = f"acdc_{condition}_kfold{folds}_fold{fold_index}_{subset}"
        item["acdc_kfold"] = folds
        item["acdc_fold"] = fold_index
        item["acdc_fold_subset"] = subset
        records.append(item)
        if limit is not None and len(records) >= int(limit):
            break
    return records


def load_acdc_night_top50_dicts(manifest_path=None, limit=None):
    manifest_path = Path(manifest_path) if manifest_path is not None else default_acdc_top50_manifest()
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Filtered ACDC manifest not found: {manifest_path}. "
            "Run tools/postprocess/filter_acdc_domain_patch.py first."
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
                "acdc_condition": entry.get("acdc_condition", "night"),
                "acdc_split": entry.get("acdc_split", "trainval"),
                "acdc_sequence": entry.get("acdc_sequence"),
                "acdc_filtered_index": index,
                "acdc_classdist_score": entry.get("score"),
                "source": "acdc_night_top50_cosec_night_domain_patch",
            }
        )
    return records


def load_acdc_night_top50_repeat_dicts(repeats=4, manifest_path=None, limit=None):
    """Repeat the filtered ACDC top50 set for small mixed-domain training."""
    base_records = load_acdc_night_top50_dicts(manifest_path=manifest_path, limit=limit)
    repeated = []
    for repeat_index in range(int(repeats)):
        for record in base_records:
            copied = dict(record)
            copied["image_id"] = f"{record['image_id']}_rep{repeat_index}"
            copied["acdc_repeat_index"] = repeat_index
            copied["source"] = f"acdc_night_top50_repeat{int(repeats)}"
            repeated.append(copied)
    return repeated
