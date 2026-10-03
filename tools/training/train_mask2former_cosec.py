#!/usr/bin/env python
import json
import math
import os
import shutil
import sys
import weakref
import importlib.util
from collections import OrderedDict
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np

# NumPy>=2 removed aliases that the vendored Detectron2 stack still uses.
for _np_name, _np_value in {"bool": bool, "int": int, "float": float, "str": str, "object": object}.items():
    if not hasattr(np, _np_name):
        setattr(np, _np_name, _np_value)

import torch
from PIL import Image

# Pillow>=10 removed a few legacy constants that vendored Detectron2 still
# references during import.
if not hasattr(Image, "LINEAR"):
    Image.LINEAR = Image.BILINEAR
if not hasattr(Image, "CUBIC"):
    Image.CUBIC = Image.BICUBIC
if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.LANCZOS

# Local experiment checkpoints are trusted artifacts produced by this repo.
# PyTorch>=2.6 defaults torch.load(weights_only=True), which rejects older
# Detectron2 checkpoints containing numpy scalar metadata.
_TORCH_LOAD = torch.load
def _torch_load_legacy_default(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _TORCH_LOAD(*args, **kwargs)
torch.load = _torch_load_legacy_default

def _eventshift_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "configs").is_dir() and (parent / "third_party").is_dir():
            return parent
    return Path(__file__).resolve().parents[1]


ROOT = _eventshift_root()
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "third_party" / "Mask2Former"))
if importlib.util.find_spec("detectron2") is None:
    sys.path.insert(0, str(ROOT / "third_party" / "detectron2"))

from acdc_dataset import (  # noqa: E402
    ACDC_CONDITIONS,
    DEFAULT_ACDC_KFOLD_COUNT,
    acdc_root,
    discover_acdc_file_split_prefixes,
    load_acdc_dicts,
    load_acdc_file_split_dicts,
    load_acdc_kfold_dicts,
    load_acdc_night_top50_dicts,
    load_acdc_night_top50_repeat_dicts,
)
from cosec_finetune_splits import (  # noqa: E402
    CLASSES,
    DEFAULT_KFOLD_COUNT,
    PALETTE,
    SPLIT_DIR,
    iter_cosec_samples,
)
from cosec_event_dataset import load_cosec_event_dicts  # noqa: E402
from dsec19_filtered_dataset import (  # noqa: E402
    DSEC11_CLASSES,
    DSEC11_PALETTE,
    DSEC19_TO_DSEC11,
    load_dsec19_close180_dicts,
    load_dsec19_close240_dicts,
    load_dsec19_filtered_event_dicts,
    load_dsec19_filtered_dicts,
    load_dsec19_full_dicts,
    load_dsec19_train_split_dicts,
    load_dsec19_train_split_event_dicts,
    load_dsec19_val_dicts,
    load_dsec19_val_event_dicts,
    load_dsec19_external_full_val_dicts,
    load_dsec19_external_full_val_event_dicts,
    load_dsec19_external_brenet_val_dicts,
    load_dsec19_external_brenet_val_event_dicts,
    load_dsec19_external_mambaseg_val_dicts,
    load_dsec19_external_mambaseg_val_event_dicts,
    load_dsec19_external_common_val_dicts,
    load_dsec19_external_common_val_event_dicts,
    load_dsec19_val_lowlight_q25_dicts,
    load_dsec19_val_lowlight_q25_event_dicts,
    load_dsec19_val_remaining_q75_dicts,
    load_dsec19_val_remaining_q75_event_dicts,
    load_dsec_brenet11_train_full_dicts,
    load_dsec_brenet11_train_full_event_dicts,
    load_dsec_brenet11_train_split_dicts,
    load_dsec_brenet11_train_split_event_dicts,
    load_dsec_brenet11_val_dicts,
    load_dsec_brenet11_val_event_dicts,
    load_dsec_brenet11_test_dicts,
    load_dsec_brenet11_test_event_dicts,
    load_dsec11_train_full_dicts,
    load_dsec11_train_full_event_dicts,
    load_dsec11_train_split_dicts,
    load_dsec11_train_split_event_dicts,
    load_dsec11_val_dicts,
    load_dsec11_val_event_dicts,
    load_dsec11_external_full_val_dicts,
    load_dsec11_external_full_val_event_dicts,
    load_dsec11_external_brenet_val_dicts,
    load_dsec11_external_brenet_val_event_dicts,
    load_dsec11_external_mambaseg_val_dicts,
    load_dsec11_external_mambaseg_val_event_dicts,
    load_dsec11_external_common_val_dicts,
    load_dsec11_external_common_val_event_dicts,
    load_dsec11_val_lowlight_q25_dicts,
    load_dsec11_val_lowlight_q25_event_dicts,
    load_dsec11_val_remaining_q75_dicts,
    load_dsec11_val_remaining_q75_event_dicts,
)
from ddd17_dataset import (  # noqa: E402
    DDD17_CLASSES,
    DDD17_PALETTE,
    ddd17_root,
    load_ddd17_dir5_dicts,
    load_ddd17_dir5_event_dicts,
    load_ddd17_full_labeled_dicts,
    load_ddd17_full_labeled_event_dicts,
    load_ddd17_train_dicts,
    load_ddd17_train_event_dicts,
    load_ddd17_val_dicts,
    load_ddd17_val_event_dicts,
)
from nightcity_dataset import (  # noqa: E402
    load_nightcity_cosec_classdist_dicts,
    load_nightcity_dicts,
    load_nightcity_trainval_cosec_classdist_strict_dicts,
    load_nightcity_trainval_cosec_night_domain_patch_dicts,
    nightcity_root,
)
from pseudo_dataset import (  # noqa: E402
    load_cosec_test_prediction_pseudo_dicts,
    load_cosec_test_pseudo_dicts,
    load_real_pool_pseudo_dicts,
)

from detectron2.checkpoint import DetectionCheckpointer  # noqa: E402
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader  # noqa: E402
from detectron2.engine import default_argument_parser, hooks, launch  # noqa: E402
from detectron2.evaluation import SemSegEvaluator  # noqa: E402
from detectron2.utils import comm  # noqa: E402
from detectron2.utils.events import CommonMetricPrinter, JSONWriter  # noqa: E402
from detectron2.utils.file_io import PathManager  # noqa: E402
from mask2former import MaskFormerSemanticDatasetMapper  # noqa: E402

from train_net import Trainer as Mask2FormerTrainer  # noqa: E402


CODE_BACKUP_PATHS = (
    "README.md",
    "configs",
    "tools",
    "third_party/Mask2Former/train_net.py",
    "third_party/Mask2Former/configs",
    "third_party/Mask2Former/mask2former",
    "third_party/detectron2/setup.py",
    "third_party/detectron2/setup.cfg",
    "third_party/detectron2/detectron2",
    "third_party/detectron2/projects",
)

CODE_BACKUP_IGNORE = shutil.ignore_patterns(
    "__pycache__",
    "*.pyc",
    ".git",
    ".pytest_cache",
    "build",
)

DAY_RARE_FOCUS_CLASSES = (
    "fence",
    "person",
    "motorcycle",
    "traffic sign",
    "bicycle",
    "rider",
)

NIGHT_RARE_FOCUS_CLASSES = (
    "traffic sign",
    "building",
    "motorcycle",
    "wall",
    "fence",
    "bicycle",
)

CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASSES)}


def _cosec_base_dataset_specs():
    specs = [
        ("cosec_train", "train"),
        ("cosec_day_train", "day_train"),
        ("cosec_night_train", "night_train"),
        ("cosec_day_val", "day_val"),
        ("cosec_night_val", "night_val"),
        ("cosec_train_event", "train"),
        ("cosec_day_train_event", "day_train"),
        ("cosec_night_train_event", "night_train"),
        ("cosec_day_val_event", "day_val"),
        ("cosec_night_val_event", "night_val"),
    ]
    for fold_index in range(DEFAULT_KFOLD_COUNT):
        for split in (
            "train",
            "val",
            "day_train",
            "day_val",
            "night_train",
            "night_val",
        ):
            kfold_split = f"kfold{DEFAULT_KFOLD_COUNT}_fold{fold_index}_{split}"
            specs.append((f"cosec_{kfold_split}", kfold_split))
            specs.append((f"cosec_{kfold_split}_event", kfold_split))
    for subset in ("train", "val"):
        for path in sorted(SPLIT_DIR.glob(f"{subset}_*.txt")):
            prefix = path.stem[len(subset) + 1 :]
            if not prefix:
                continue
            for domain in ("", "day_", "night_"):
                split = f"{prefix}_{domain}{subset}"
                specs.append((f"cosec_{split}", split))
                specs.append((f"cosec_{split}_event", split))
    deduped = OrderedDict()
    for name, split in specs:
        deduped[name] = split
    return list(deduped.items())


def backup_runtime_code(output_dir, args=None):
    backup_root = Path(output_dir) / "code_backup"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_dir = backup_root / timestamp
    snapshot_dir.mkdir(parents=True, exist_ok=False)

    copied = []
    for rel_path in CODE_BACKUP_PATHS:
        src = ROOT / rel_path
        dst = snapshot_dir / rel_path
        if not src.exists():
            copied.append({"path": rel_path, "copied": False, "reason": "missing"})
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, ignore=CODE_BACKUP_IGNORE)
        else:
            shutil.copy2(src, dst)
        copied.append({"path": rel_path, "copied": True})

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(ROOT),
        "argv": sys.argv,
        "args": vars(args) if args is not None else None,
        "copied": copied,
    }
    with (snapshot_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"[code_backup] saved runtime code to {snapshot_dir}", flush=True)
    return snapshot_dir



def _json_sanitize(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_sanitize(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_sanitize(child) for child in value]
    return value


def write_eval_results(output_dir, results):
    os.makedirs(output_dir, exist_ok=True)
    record = _json_sanitize(results)
    with open(os.path.join(output_dir, "eval_results.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    metrics_record = dict(record) if isinstance(record, dict) else {"eval_results": record}
    metrics_record["eval_only"] = True
    with open(os.path.join(output_dir, "metrics.json"), "a", encoding="utf-8") as f:
        json.dump(metrics_record, f, sort_keys=True, allow_nan=False)
        f.write("\n")

def should_skip_code_backup():
    return os.environ.get("SKIP_CODE_BACKUP", "").lower() in {"1", "true", "yes", "y"}


def load_cosec_dicts(split):
    records = []
    cosec_root = Path(os.environ.get("COSEC_ROOT", "/data/cosec/train")).expanduser()
    for idx, (seq_name, frame_id, img_path, label_path) in enumerate(iter_cosec_samples(cosec_root, split)):
        records.append(
            {
                "file_name": str(img_path),
                "sem_seg_file_name": str(label_path),
                "image_id": f"{seq_name}_{frame_id:06d}",
            }
        )
    return records


def _evenly_spaced_subset(records, keep_count):
    if keep_count >= len(records):
        return list(records)
    if keep_count <= 0:
        return []
    # Preserve coverage across all sorted day sequences without depending on randomness.
    selected = []
    for rank in range(keep_count):
        index = int((rank + 0.5) * len(records) / keep_count)
        selected.append(records[min(index, len(records) - 1)])
    return selected


def load_cosec_train_night_focus_day700_dicts():
    records = load_cosec_dicts("train")
    day_records = [record for record in records if record["image_id"].startswith("Day_")]
    night_records = [record for record in records if record["image_id"].startswith("Night_")]
    selected_day = _evenly_spaced_subset(day_records, 700)
    focused_records = selected_day + night_records
    for record in focused_records:
        record["source"] = "cosec_train_night_focus_day700"
    return focused_records


def load_cosec_kfold_dayextra_dsec180_dicts(fold_index, day_extra=250):
    split = f"kfold{DEFAULT_KFOLD_COUNT}_fold{fold_index}_train"
    day_split = f"kfold{DEFAULT_KFOLD_COUNT}_fold{fold_index}_day_train"
    base_records = load_cosec_dicts(split)
    day_records = load_cosec_dicts(day_split)
    day_extra_records = _evenly_spaced_subset(day_records, day_extra)
    dsec_records = load_dsec19_close180_dicts()

    output = []
    for record in base_records:
        item = dict(record)
        item["source"] = f"cosec_{split}_base"
        output.append(item)
    for record in day_extra_records:
        item = dict(record)
        item["source"] = f"cosec_{day_split}_extra{day_extra}"
        output.append(item)
    for record in dsec_records:
        item = dict(record)
        item["source"] = f"{record.get('source', 'dsec19_close180')}_kfold_aux"
        output.append(item)

    if comm.is_main_process():
        day_count = sum(1 for record in base_records if record["image_id"].startswith("Day_"))
        night_count = sum(1 for record in base_records if record["image_id"].startswith("Night_"))
        print(
            f"[kfold_dayextra_dsec] fold={fold_index} "
            f"base={len(base_records)} day={day_count} night={night_count} "
            f"day_extra={len(day_extra_records)} dsec={len(dsec_records)} total={len(output)}",
            flush=True,
        )
    return output


def _scene_from_cosec_image_id(image_id):
    seq_name = image_id.rsplit("_", 1)[0]
    parts = seq_name.split("_")
    if len(parts) < 2:
        return "unknown"
    return parts[1]


def _class_ids(class_names):
    missing = [name for name in class_names if name not in CLASS_TO_ID]
    if missing:
        raise KeyError(f"Unknown CoSEC class names: {missing}")
    return frozenset(CLASS_TO_ID[name] for name in class_names)


@lru_cache(maxsize=None)
def _label_class_ids(label_path):
    mask = np.asarray(Image.open(label_path))
    if mask.ndim == 3:
        # CoSEC GT is expected to be index masks. If a future source is RGB, keep
        # this path conservative and only match exact Cityscapes/CoSEC palette.
        out = np.full(mask.shape[:2], 255, dtype=np.uint8)
        rgb = mask[..., :3]
        for class_id, color in enumerate(PALETTE):
            out[np.all(rgb == np.asarray(color, dtype=rgb.dtype), axis=-1)] = class_id
        mask = out
    valid = (mask >= 0) & (mask < len(CLASSES))
    if not np.any(valid):
        return frozenset()
    return frozenset(int(value) for value in np.unique(mask[valid]))


def _record_has_any_class(record, target_ids):
    return bool(_label_class_ids(record["sem_seg_file_name"]) & target_ids)


def load_cosec_rare_focus_dicts(split, class_names, repeats=4):
    records = load_cosec_dicts(split)
    target_ids = _class_ids(class_names)
    focused = [record for record in records if _record_has_any_class(record, target_ids)]
    output = []
    for record in records:
        item = dict(record)
        item["source"] = f"cosec_{split}_rare_focus_base"
        output.append(item)
    for repeat_idx in range(max(0, repeats - 1)):
        for record in focused:
            item = dict(record)
            item["source"] = f"cosec_{split}_rare_focus_repeat{repeat_idx + 1}"
            item["rare_focus_classes"] = list(class_names)
            output.append(item)
    if comm.is_main_process():
        print(
            f"[rare_focus] split={split} classes={list(class_names)} "
            f"base={len(records)} focused={len(focused)} repeats={repeats} total={len(output)}",
            flush=True,
        )
    return output


def load_cosec_day_rare_focus_repeat4_dicts():
    return load_cosec_rare_focus_dicts("day_train", DAY_RARE_FOCUS_CLASSES, repeats=4)


def load_cosec_night_rare_focus_repeat4_dicts():
    return load_cosec_rare_focus_dicts("night_train", NIGHT_RARE_FOCUS_CLASSES, repeats=4)


def load_cosec_train_scene_diag_dicts(domain, per_scene=40):
    if domain not in {"day", "night"}:
        raise ValueError(f"Unknown CoSEC scene diagnostic domain: {domain}")
    split = f"{domain}_train"
    records_by_scene = OrderedDict()
    for record in load_cosec_dicts(split):
        scene = _scene_from_cosec_image_id(record["image_id"])
        records_by_scene.setdefault(scene, []).append(record)

    selected = []
    for scene, records in sorted(records_by_scene.items()):
        for record in _evenly_spaced_subset(records, per_scene):
            item = dict(record)
            item["source"] = f"cosec_{domain}_train_scene_diag"
            item["scene"] = scene
            selected.append(item)
    return selected


def register_cosec():
    for name, split in _cosec_base_dataset_specs():
        if name not in DatasetCatalog.list():
            if name.endswith("_event"):
                DatasetCatalog.register(name, lambda split=split: load_cosec_event_dicts(split))
            else:
                DatasetCatalog.register(name, lambda split=split: load_cosec_dicts(split))
        MetadataCatalog.get(name).set(
            stuff_classes=list(CLASSES),
            stuff_colors=[list(color) for color in PALETTE],
            evaluator_type="sem_seg",
            ignore_label=255,
        )
        split_parts = split.split("_", 1)
        if len(split_parts) == 2 and split_parts[0] in {"train", "val"}:
            alias_split = f"{split_parts[1]}_{split_parts[0]}"
            alias_name = f"cosec_{alias_split}{'_event' if name.endswith('_event') else ''}"
            if alias_name not in DatasetCatalog.list():
                if alias_name.endswith("_event"):
                    DatasetCatalog.register(alias_name, lambda split=split: load_cosec_event_dicts(split))
                else:
                    DatasetCatalog.register(alias_name, lambda split=split: load_cosec_dicts(split))
            MetadataCatalog.get(alias_name).set(
                stuff_classes=list(CLASSES),
                stuff_colors=[list(color) for color in PALETTE],
                evaluator_type="sem_seg",
                ignore_label=255,
            )

    for name, loader in [
        ("cosec_train_night_focus_day700", load_cosec_train_night_focus_day700_dicts),
        ("cosec_day_train_rare_focus_repeat4", load_cosec_day_rare_focus_repeat4_dicts),
        ("cosec_night_train_rare_focus_repeat4", load_cosec_night_rare_focus_repeat4_dicts),
        ("cosec_day_train_scene_diag", lambda: load_cosec_train_scene_diag_dicts("day", per_scene=40)),
        ("cosec_night_train_scene_diag", lambda: load_cosec_train_scene_diag_dicts("night", per_scene=80)),
        ("dsec19_train_filtered630", load_dsec19_filtered_dicts),
        ("dsec19_train_filtered630_event", load_dsec19_filtered_event_dicts),
        ("dsec19_train_full", load_dsec19_full_dicts),
        ("dsec19_train_noval", load_dsec19_train_split_dicts),
        ("dsec19_train_noval_event", load_dsec19_train_split_event_dicts),
        ("dsec19_val", load_dsec19_val_dicts),
        ("dsec19_val_event", load_dsec19_val_event_dicts),
        ("dsec19_external_full_val", load_dsec19_external_full_val_dicts),
        ("dsec19_external_full_val_event", load_dsec19_external_full_val_event_dicts),
        ("dsec19_external_brenet_val", load_dsec19_external_brenet_val_dicts),
        ("dsec19_external_brenet_val_event", load_dsec19_external_brenet_val_event_dicts),
        ("dsec19_external_mambaseg_val", load_dsec19_external_mambaseg_val_dicts),
        ("dsec19_external_mambaseg_val_event", load_dsec19_external_mambaseg_val_event_dicts),
        ("dsec19_external_common_val", load_dsec19_external_common_val_dicts),
        ("dsec19_external_common_val_event", load_dsec19_external_common_val_event_dicts),
        ("dsec19_val_lowlight_q25", load_dsec19_val_lowlight_q25_dicts),
        ("dsec19_val_lowlight_q25_event", load_dsec19_val_lowlight_q25_event_dicts),
        ("dsec19_val_remaining_q75", load_dsec19_val_remaining_q75_dicts),
        ("dsec19_val_remaining_q75_event", load_dsec19_val_remaining_q75_event_dicts),
        ("dsec19_train_close180", load_dsec19_close180_dicts),
        ("dsec19_train_close240", load_dsec19_close240_dicts),
        *(
            (
                f"cosec_kfold{DEFAULT_KFOLD_COUNT}_fold{fold_index}_train_dayextra250_dsec180",
                lambda fold_index=fold_index: load_cosec_kfold_dayextra_dsec180_dicts(fold_index),
            )
            for fold_index in range(DEFAULT_KFOLD_COUNT)
        ),
        ("acdc_all_train", lambda: load_acdc_dicts("all", "train")),
        ("acdc_all_val", lambda: load_acdc_dicts("all", "val")),
        ("acdc_night_train", lambda: load_acdc_dicts("night", "train")),
        ("acdc_night_val", lambda: load_acdc_dicts("night", "val")),
        ("acdc_night_trainval", lambda: load_acdc_dicts("night", "trainval")),
        *(
            (
                f"acdc_{condition}_kfold{DEFAULT_ACDC_KFOLD_COUNT}_fold{fold_index}_{subset}",
                lambda condition=condition, fold_index=fold_index, subset=subset: load_acdc_kfold_dicts(
                    condition,
                    DEFAULT_ACDC_KFOLD_COUNT,
                    fold_index,
                    subset,
                ),
            )
            for condition in ("night", "all")
            for fold_index in range(DEFAULT_ACDC_KFOLD_COUNT)
            for subset in ("train", "val")
        ),
        ("acdc_night_top50", load_acdc_night_top50_dicts),
        ("acdc_night_top50_repeat4", lambda: load_acdc_night_top50_repeat_dicts(4)),
        ("acdc_night_top50_repeat8", lambda: load_acdc_night_top50_repeat_dicts(8)),
        ("nightcity_train", lambda: load_nightcity_dicts("train")),
        ("nightcity_train_cosec_classdist", load_nightcity_cosec_classdist_dicts),
        ("nightcity_trainval_cosec_classdist_strict", load_nightcity_trainval_cosec_classdist_strict_dicts),
        ("nightcity_trainval_cosec_night_domain_patch", load_nightcity_trainval_cosec_night_domain_patch_dicts),
        ("nightcity_val", lambda: load_nightcity_dicts("val")),
        ("nightcity_trainval", lambda: load_nightcity_dicts("trainval")),
        (
            "cosec_test_daynight_pseudo_consensus_conf192",
            lambda: load_cosec_test_pseudo_dicts("daynight", "consensus", 192, repeat=1),
        ),
        (
            "cosec_test_night_pseudo_consensus_conf192",
            lambda: load_cosec_test_pseudo_dicts("night", "consensus", 192, repeat=2),
        ),
        (
            "cosec_test_real_pseudo_consensus_conf192",
            lambda: load_cosec_test_pseudo_dicts("real", "consensus", 192, repeat=1),
        ),
        (
            "cosec_test_daynight_pseudo_currentbest_tta_all",
            lambda: load_cosec_test_prediction_pseudo_dicts(
                "daynight",
                "swinL_day65_4352_tta5126247681024_daynight_acdc_proxy_real",
                "all",
                repeat=1,
                min_valid_fraction=0.99,
            ),
        ),
        (
            "cosec_test_daynight_pseudo_currentbest_tta_segformer_agree_conf192_limit384",
            lambda: load_cosec_test_prediction_pseudo_dicts(
                "daynight",
                "swinL_day65_4352_tta5126247681024_daynight_acdc_proxy_real",
                "segformer_agree_conf192",
                repeat=1,
                min_valid_fraction=0.01,
                limit=384,
            ),
        ),
        (
            "cosec_test_daynight_pseudo_currentbest_tta_segformer_agree_rare_boundary_conf192_limit384",
            lambda: load_cosec_test_prediction_pseudo_dicts(
                "daynight",
                "swinL_day65_4352_tta5126247681024_daynight_acdc_proxy_real",
                "segformer_agree_rare_boundary_conf192",
                repeat=1,
                min_valid_fraction=0.01,
                limit=384,
            ),
        ),
        (
            "cosec_test_daynight_pseudo_currentbest_tta_segformer_agree_gap_focus_conf192_limit384",
            lambda: load_cosec_test_prediction_pseudo_dicts(
                "daynight",
                "swinL_day65_4352_tta5126247681024_daynight_acdc_proxy_real",
                "segformer_agree_gap_focus_conf192",
                repeat=1,
                min_valid_fraction=0.01,
                limit=384,
            ),
        ),
        (
            "cosec_test_day_pseudo_currentbest_tta_segformer_agree_gap_focus_conf192_limit256",
            lambda: load_cosec_test_prediction_pseudo_dicts(
                "day",
                "swinL_day65_4352_tta5126247681024_daynight_acdc_proxy_real",
                "segformer_agree_gap_focus_conf192",
                repeat=1,
                min_valid_fraction=0.01,
                limit=256,
            ),
        ),
        (
            "cosec_test_night_pseudo_currentbest_tta_segformer_agree_gap_focus_conf192_limit192",
            lambda: load_cosec_test_prediction_pseudo_dicts(
                "night",
                "swinL_day65_4352_tta5126247681024_daynight_acdc_proxy_real",
                "segformer_agree_gap_focus_conf192",
                repeat=1,
                min_valid_fraction=0.001,
                limit=192,
            ),
        ),
        (
            "cosec_test_daynight_pseudo_segformer_consensus_conf192_limit256",
            lambda: load_cosec_test_pseudo_dicts(
                "daynight",
                "segformer_consensus",
                192,
                repeat=1,
                limit=256,
            ),
        ),
        (
            "cosec_test_night_pseudo_segformer_consensus_conf192_limit128",
            lambda: load_cosec_test_pseudo_dicts(
                "night",
                "segformer_consensus",
                192,
                repeat=1,
                limit=128,
            ),
        ),
        (
            "cosec_test_real_pseudo_segformer_consensus_conf192_limit73",
            lambda: load_cosec_test_pseudo_dicts(
                "real",
                "segformer_consensus",
                192,
                repeat=1,
                limit=73,
            ),
        ),
        (
            "cosec_test_daynight_pseudo_segformer_balcap_conf192_limit256",
            lambda: load_cosec_test_pseudo_dicts(
                "daynight",
                "segformer_balcap",
                192,
                repeat=1,
                limit=256,
            ),
        ),
        (
            "cosec_test_daynight_pseudo_segformer_rare_boundary_conf192_limit384",
            lambda: load_cosec_test_pseudo_dicts(
                "daynight",
                "segformer_rare_boundary",
                192,
                repeat=1,
                min_valid_fraction=0.01,
                limit=384,
            ),
        ),
        (
            "cosec_test_night_pseudo_segformer_rare_boundary_conf192_limit192",
            lambda: load_cosec_test_pseudo_dicts(
                "night",
                "segformer_rare_boundary",
                192,
                repeat=1,
                min_valid_fraction=0.01,
                limit=192,
            ),
        ),
        (
            "cosec_test_night_pseudo_segformer_balcap_conf192_limit128",
            lambda: load_cosec_test_pseudo_dicts(
                "night",
                "segformer_balcap",
                192,
                repeat=1,
                limit=128,
            ),
        ),
        (
            "cosec_test_real_pseudo_segformer_balcap_conf192_limit73",
            lambda: load_cosec_test_pseudo_dicts(
                "real",
                "segformer_balcap",
                192,
                repeat=1,
                limit=73,
            ),
        ),
        (
            "real_pool_pseudo_swinl_conf224",
            lambda: load_real_pool_pseudo_dicts("swinl", 224, repeat=1, limit=600),
        ),
        (
            "real_pool_pseudo_swinl_eventedge_conf224",
            lambda: load_real_pool_pseudo_dicts(
                "swinl_eventedge",
                224,
                repeat=1,
                limit=600,
                min_valid_fraction=0.01,
            ),
        ),
        (
            "real_pool_pseudo_swinl_eventactive_conf224",
            lambda: load_real_pool_pseudo_dicts(
                "swinl_eventactive",
                224,
                repeat=1,
                limit=600,
                min_valid_fraction=0.01,
            ),
        ),
        (
            "real_pool_pseudo_swinl_eventedge100_conf224",
            lambda: load_real_pool_pseudo_dicts(
                "swinl_eventedge100",
                224,
                repeat=1,
                limit=600,
                min_valid_fraction=0.01,
            ),
        ),
        (
            "real_pool_pseudo_swinl_segmentco_eventactive_conf224",
            lambda: load_real_pool_pseudo_dicts(
                "swinl_segmentco_eventactive",
                224,
                repeat=1,
                limit=600,
                min_valid_fraction=0.01,
            ),
        ),
        (
            "real_pool_pseudo_swinl_segmentco_conf224",
            lambda: load_real_pool_pseudo_dicts(
                "swinl_segmentco",
                224,
                repeat=1,
                limit=600,
                min_valid_fraction=0.01,
            ),
        ),
        (
            "real_pool_pseudo_swinl_segmentco_eventedge_conf224",
            lambda: load_real_pool_pseudo_dicts(
                "swinl_segmentco_eventedge",
                224,
                repeat=1,
                limit=600,
                min_valid_fraction=0.01,
            ),
        ),
    ]:
        if name not in DatasetCatalog.list():
            DatasetCatalog.register(name, loader)
        MetadataCatalog.get(name).set(
            stuff_classes=list(CLASSES),
            stuff_colors=[list(color) for color in PALETTE],
            evaluator_type="sem_seg",
            ignore_label=255,
        )

    dsec11_specs = [
        ("dsec11_train_full", load_dsec11_train_full_dicts, "sem_seg"),
        ("dsec11_train_full_event", load_dsec11_train_full_event_dicts, "sem_seg"),
        ("dsec11_train_noval", load_dsec11_train_split_dicts, "sem_seg"),
        ("dsec11_train_noval_event", load_dsec11_train_split_event_dicts, "sem_seg"),
        ("dsec11_val", load_dsec11_val_dicts, "sem_seg"),
        ("dsec11_val_event", load_dsec11_val_event_dicts, "sem_seg"),
        ("dsec11_external_full_val", load_dsec11_external_full_val_dicts, "sem_seg"),
        ("dsec11_external_full_val_event", load_dsec11_external_full_val_event_dicts, "sem_seg"),
        ("dsec11_external_brenet_val", load_dsec11_external_brenet_val_dicts, "sem_seg"),
        ("dsec11_external_brenet_val_event", load_dsec11_external_brenet_val_event_dicts, "sem_seg"),
        ("dsec11_external_mambaseg_val", load_dsec11_external_mambaseg_val_dicts, "sem_seg"),
        ("dsec11_external_mambaseg_val_event", load_dsec11_external_mambaseg_val_event_dicts, "sem_seg"),
        ("dsec11_external_common_val", load_dsec11_external_common_val_dicts, "sem_seg"),
        ("dsec11_external_common_val_event", load_dsec11_external_common_val_event_dicts, "sem_seg"),
        ("dsec11_val_lowlight_q25", load_dsec11_val_lowlight_q25_dicts, "sem_seg"),
        ("dsec11_val_lowlight_q25_event", load_dsec11_val_lowlight_q25_event_dicts, "sem_seg"),
        ("dsec11_val_remaining_q75", load_dsec11_val_remaining_q75_dicts, "sem_seg"),
        ("dsec11_val_remaining_q75_event", load_dsec11_val_remaining_q75_event_dicts, "sem_seg"),
        ("dsec11_val_from_dsec19", load_dsec11_val_dicts, "sem_seg_dsec19_to_dsec11"),
        ("dsec11_val_event_from_dsec19", load_dsec11_val_event_dicts, "sem_seg_dsec19_to_dsec11"),
        ("dsec11_external_full_val_from_dsec19", load_dsec11_external_full_val_dicts, "sem_seg_dsec19_to_dsec11"),
        ("dsec11_external_full_val_event_from_dsec19", load_dsec11_external_full_val_event_dicts, "sem_seg_dsec19_to_dsec11"),
        ("dsec11_external_brenet_val_from_dsec19", load_dsec11_external_brenet_val_dicts, "sem_seg_dsec19_to_dsec11"),
        ("dsec11_external_brenet_val_event_from_dsec19", load_dsec11_external_brenet_val_event_dicts, "sem_seg_dsec19_to_dsec11"),
        ("dsec11_external_mambaseg_val_from_dsec19", load_dsec11_external_mambaseg_val_dicts, "sem_seg_dsec19_to_dsec11"),
        ("dsec11_external_mambaseg_val_event_from_dsec19", load_dsec11_external_mambaseg_val_event_dicts, "sem_seg_dsec19_to_dsec11"),
        ("dsec11_external_common_val_from_dsec19", load_dsec11_external_common_val_dicts, "sem_seg_dsec19_to_dsec11"),
        ("dsec11_external_common_val_event_from_dsec19", load_dsec11_external_common_val_event_dicts, "sem_seg_dsec19_to_dsec11"),
        (
            "dsec11_val_lowlight_q25_from_dsec19",
            load_dsec11_val_lowlight_q25_dicts,
            "sem_seg_dsec19_to_dsec11",
        ),
        (
            "dsec11_val_lowlight_q25_event_from_dsec19",
            load_dsec11_val_lowlight_q25_event_dicts,
            "sem_seg_dsec19_to_dsec11",
        ),
        (
            "dsec11_val_remaining_q75_from_dsec19",
            load_dsec11_val_remaining_q75_dicts,
            "sem_seg_dsec19_to_dsec11",
        ),
        (
            "dsec11_val_remaining_q75_event_from_dsec19",
            load_dsec11_val_remaining_q75_event_dicts,
            "sem_seg_dsec19_to_dsec11",
        ),
    ]
    for name, loader, evaluator_type in dsec11_specs:
        if name not in DatasetCatalog.list():
            DatasetCatalog.register(name, loader)
        MetadataCatalog.get(name).set(
            stuff_classes=list(DSEC11_CLASSES),
            stuff_colors=[list(color) for color in DSEC11_PALETTE],
            evaluator_type=evaluator_type,
            ignore_label=255,
            dsec19_to_dsec11=list(DSEC19_TO_DSEC11),
            dsec19_classes=list(CLASSES),
        )

    dsec_brenet11_specs = [
        ("dsec_brenet11_train_full", load_dsec_brenet11_train_full_dicts, "sem_seg"),
        ("dsec_brenet11_train_full_event", load_dsec_brenet11_train_full_event_dicts, "sem_seg"),
        ("dsec_brenet11_train_noval", load_dsec_brenet11_train_split_dicts, "sem_seg"),
        ("dsec_brenet11_train_noval_event", load_dsec_brenet11_train_split_event_dicts, "sem_seg"),
        ("dsec_brenet11_val", load_dsec_brenet11_val_dicts, "sem_seg"),
        ("dsec_brenet11_val_event", load_dsec_brenet11_val_event_dicts, "sem_seg"),
        ("dsec_brenet11_test", load_dsec_brenet11_test_dicts, "sem_seg"),
        ("dsec_brenet11_test_event", load_dsec_brenet11_test_event_dicts, "sem_seg"),
    ]
    for name, loader, evaluator_type in dsec_brenet11_specs:
        if name not in DatasetCatalog.list():
            DatasetCatalog.register(name, loader)
        MetadataCatalog.get(name).set(
            stuff_classes=list(DSEC11_CLASSES),
            stuff_colors=[list(color) for color in DSEC11_PALETTE],
            evaluator_type=evaluator_type,
            ignore_label=255,
            dsec19_to_dsec11=list(DSEC19_TO_DSEC11),
            dsec19_classes=list(CLASSES),
        )

    for name, loader in [
        ("ddd17_train", load_ddd17_train_dicts),
        ("ddd17_train_event", load_ddd17_train_event_dicts),
        ("ddd17_val", load_ddd17_val_dicts),
        ("ddd17_val_event", load_ddd17_val_event_dicts),
        ("ddd17_full_labeled", load_ddd17_full_labeled_dicts),
        ("ddd17_full_labeled_event", load_ddd17_full_labeled_event_dicts),
        ("ddd17_dir5_labeled", load_ddd17_dir5_dicts),
        ("ddd17_dir5_labeled_event", load_ddd17_dir5_event_dicts),
    ]:
        if name not in DatasetCatalog.list():
            DatasetCatalog.register(name, loader)
        MetadataCatalog.get(name).set(
            stuff_classes=list(DDD17_CLASSES),
            stuff_colors=[list(color) for color in DDD17_PALETTE],
            evaluator_type="sem_seg",
            ignore_label=255,
        )

    for prefix in discover_acdc_file_split_prefixes():
        split_specs = [
            (f"acdc_{prefix}_train", lambda prefix=prefix: load_acdc_file_split_dicts(prefix, "train", "all")),
            (f"acdc_{prefix}_val", lambda prefix=prefix: load_acdc_file_split_dicts(prefix, "val", "all")),
            (f"acdc_{prefix}_all_train", lambda prefix=prefix: load_acdc_file_split_dicts(prefix, "train", "all")),
            (f"acdc_{prefix}_all_val", lambda prefix=prefix: load_acdc_file_split_dicts(prefix, "val", "all")),
        ]
        for condition in ACDC_CONDITIONS:
            split_specs.extend(
                [
                    (
                        f"acdc_{prefix}_{condition}_train",
                        lambda prefix=prefix, condition=condition: load_acdc_file_split_dicts(
                            prefix,
                            "train",
                            condition,
                        ),
                    ),
                    (
                        f"acdc_{prefix}_{condition}_val",
                        lambda prefix=prefix, condition=condition: load_acdc_file_split_dicts(
                            prefix,
                            "val",
                            condition,
                        ),
                    ),
                ]
            )
        for name, loader in split_specs:
            if name not in DatasetCatalog.list():
                DatasetCatalog.register(name, loader)
            MetadataCatalog.get(name).set(
                stuff_classes=list(CLASSES),
                stuff_colors=[list(color) for color in PALETTE],
                evaluator_type="sem_seg",
                ignore_label=255,
            )

    if comm.is_main_process():
        try:
            print(f"[acdc] root: {acdc_root()}", flush=True)
        except FileNotFoundError as error:
            print(f"[acdc] not registered from disk yet: {error}", flush=True)
        try:
            print(f"[nightcity] root: {nightcity_root()}", flush=True)
        except FileNotFoundError as error:
            print(f"[nightcity] not registered from disk yet: {error}", flush=True)
        try:
            print(f"[ddd17] root: {ddd17_root()}", flush=True)
        except FileNotFoundError as error:
            print(f"[ddd17] not registered from disk yet: {error}", flush=True)


def _kfold_validation_target(dataset_name):
    base_name = dataset_name[: -len("_event")] if dataset_name.endswith("_event") else dataset_name
    if not base_name.startswith("cosec_kfold"):
        return None
    if not (
        base_name.endswith("_val")
        or base_name.endswith("_day_val")
        or base_name.endswith("_night_val")
    ):
        return None
    tag = base_name[len("cosec_") :]
    return (
        tag,
        (f"{base_name}_event", base_name),
        f"best_model_cosec_{tag}",
    )


def _acdc_kfold_validation_target(dataset_name):
    if not dataset_name.startswith("acdc_") or "_kfold" not in dataset_name:
        return None
    if not dataset_name.endswith("_val"):
        return None
    return (
        dataset_name,
        (dataset_name,),
        f"best_model_{dataset_name}",
    )


def _parse_best_checkpoint_min_miou(raw_thresholds):
    thresholds = {}
    for item in raw_thresholds or []:
        if isinstance(item, str):
            if not item.strip():
                continue
            if ":" not in item:
                raise ValueError(f"Expected BEST_CHECKPOINT_MIN_MIOU item as 'tag:value', got {item!r}")
            tag, value = item.split(":", 1)
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            tag, value = item
        else:
            raise ValueError(f"Unsupported BEST_CHECKPOINT_MIN_MIOU item: {item!r}")
        thresholds[str(tag).strip()] = float(value)
    return thresholds


def _dataset_matches_best_tag(tag, dataset_name):
    base_name = dataset_name[: -len("_event")] if dataset_name.endswith("_event") else dataset_name
    if tag == "day":
        return base_name.startswith("cosec_") and base_name.endswith("_day_val")
    if tag == "night":
        return base_name.startswith("cosec_") and base_name.endswith("_night_val")
    if tag == "overall":
        return (
            base_name.startswith("cosec_")
            and base_name.endswith("_val")
            and not base_name.endswith("_day_val")
            and not base_name.endswith("_night_val")
        )
    if tag in {"acdc", "acdc_all"}:
        return base_name.startswith("acdc_") and base_name.endswith("_all_val")
    if tag == "acdc_night":
        return base_name.startswith("acdc_") and base_name.endswith("_night_val")
    if tag == "dsec19":
        return dataset_name in {
            "dsec19_val",
            "dsec19_val_event",
            "dsec19_external_full_val",
            "dsec19_external_full_val_event",
        }
    if tag == "dsec11":
        return dataset_name in {
            "dsec11_val",
            "dsec11_val_event",
            "dsec11_external_common_val",
            "dsec11_external_common_val_event",
            "dsec_brenet11_val",
            "dsec_brenet11_val_event",
        }
    if tag == "ddd17":
        return dataset_name in {"ddd17_val", "ddd17_val_event"}
    return False


class CoSECDetectionCheckpointer(DetectionCheckpointer):
    def __init__(self, *args, input_concat_channels=None, init_dsec11_head_from_dsec19=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_concat_channels = input_concat_channels
        self.init_dsec11_head_from_dsec19 = bool(init_dsec11_head_from_dsec19)

    def _load_model(self, checkpoint):
        if self.input_concat_channels:
            self._expand_input_concat_patch_embed(checkpoint)
        if self.init_dsec11_head_from_dsec19:
            self._remap_dsec19_class_embed_to_dsec11(checkpoint)
        return super()._load_model(checkpoint)

    @staticmethod
    def _remap_class_embed_tensor(source, target_shape):
        source_class_count = len(DSEC19_TO_DSEC11)
        target_class_count = len(DSEC11_CLASSES)
        if int(source.shape[0]) != source_class_count + 1:
            return None
        if int(target_shape[0]) != target_class_count + 1:
            return None
        if tuple(source.shape[1:]) != tuple(target_shape[1:]):
            return None
        if torch.is_tensor(source):
            target = source.new_zeros(tuple(target_shape))
            mean_kwargs = {"dim": 0}
        else:
            source = np.asarray(source)
            target = np.zeros(tuple(target_shape), dtype=source.dtype)
            mean_kwargs = {"axis": 0}
        for target_id in range(target_class_count):
            source_ids = [
                source_id
                for source_id, mapped_id in enumerate(DSEC19_TO_DSEC11)
                if int(mapped_id) == target_id
            ]
            if source_ids:
                target[target_id] = source[source_ids].mean(**mean_kwargs)
        target[target_class_count] = source[source_class_count]
        return target

    def _target_state_key(self, ckpt_key, current_state):
        if ckpt_key in current_state:
            return ckpt_key
        if ckpt_key.startswith("module.") and ckpt_key[len("module.") :] in current_state:
            return ckpt_key[len("module.") :]
        module_key = f"module.{ckpt_key}"
        if module_key in current_state:
            return module_key
        return None

    def _remap_dsec19_class_embed_to_dsec11(self, checkpoint):
        model_state = checkpoint.get("model")
        if not isinstance(model_state, dict):
            return
        current_state = self.model.state_dict()
        remapped = []
        for ckpt_key, source in list(model_state.items()):
            if not (ckpt_key.endswith("class_embed.weight") or ckpt_key.endswith("class_embed.bias")):
                continue
            if not hasattr(source, "shape"):
                continue
            target_key = self._target_state_key(ckpt_key, current_state)
            if target_key is None:
                continue
            target = current_state[target_key]
            remapped_tensor = self._remap_class_embed_tensor(source, target.shape)
            if remapped_tensor is None:
                continue
            if torch.is_tensor(remapped_tensor):
                remapped_tensor = remapped_tensor.cpu()
            model_state[ckpt_key] = remapped_tensor
            remapped.append(f"{ckpt_key}:{tuple(source.shape)}->{tuple(remapped_tensor.shape)}")
        if comm.is_main_process():
            if remapped:
                print("[dsec11_init] remapped DSEC19 class_embed rows: " + ", ".join(remapped), flush=True)
            else:
                print("[dsec11_init] no compatible DSEC19 class_embed tensors found to remap", flush=True)

    def _expand_input_concat_patch_embed(self, checkpoint):
        model_state = checkpoint.get("model")
        if not isinstance(model_state, dict):
            return
        current_state = self.model.state_dict()
        key_pairs = [
            ("backbone.patch_embed.proj.weight", "backbone.patch_embed.proj.weight"),
            ("backbone.bottom_up.patch_embed.proj.weight", "backbone.bottom_up.patch_embed.proj.weight"),
            ("module.backbone.patch_embed.proj.weight", "module.backbone.patch_embed.proj.weight"),
        ]
        for ckpt_key, model_key in key_pairs:
            if ckpt_key not in model_state:
                continue
            target = current_state.get(model_key)
            source = model_state[ckpt_key]
            if target is None or not hasattr(source, "shape"):
                continue
            if len(source.shape) != 4 or len(target.shape) != 4:
                continue
            if source.shape[1] == target.shape[1] or source.shape[1] != 3 or target.shape[1] <= 3:
                continue
            expanded = target.new_zeros(target.shape)
            expanded[:, :3, :, :] = source.to(dtype=expanded.dtype)
            model_state[ckpt_key] = expanded.cpu()
            if comm.is_main_process():
                print(
                    f"[input_concat] expanded {ckpt_key} from {tuple(source.shape)} "
                    f"to {tuple(expanded.shape)}; event channels zero-initialized",
                    flush=True,
                )
            return


class BestValidationCheckpointer(hooks.HookBase):
    DEFAULT_TARGETS = (
        ("day", ("cosec_day_val_event", "cosec_day_val"), "best_model_cosec_day"),
        ("night", ("cosec_night_val_event", "cosec_night_val"), "best_model_cosec_night"),
        (
            "overall",
            (
                "cosec_seqday_nightchunk15_val_event",
                "cosec_seqday_nightchunk15_val",
                "cosec_val_event",
                "cosec_val",
            ),
            "best_model_cosec_overall",
        ),
        ("acdc", ("acdc_all_val",), "best_model_acdc"),
        ("acdc_all", ("acdc_all_val",), "best_model_acdc_all"),
        ("acdc_night", ("acdc_night_val",), "best_model_acdc_night"),
        ("dsec19", ("dsec19_val_event", "dsec19_val"), "best_model_dsec19"),
        (
            "dsec11",
            ("dsec_brenet11_val_event", "dsec_brenet11_val", "dsec11_val_event", "dsec11_val"),
            "best_model_dsec11",
        ),
        ("ddd17", ("ddd17_val_event", "ddd17_val"), "best_model_ddd17"),
    )

    def __init__(self, output_dir=None, dataset_names=(), min_miou_thresholds=()):
        targets = list(self.DEFAULT_TARGETS)
        seen_tags = {tag for tag, _, _ in targets}
        self.dataset_names = tuple(dataset_names)
        for dataset_name in dataset_names:
            target = _kfold_validation_target(dataset_name) or _acdc_kfold_validation_target(dataset_name)
            if target is None or target[0] in seen_tags:
                continue
            targets.append(target)
            seen_tags.add(target[0])
        self.targets = tuple(targets)
        self.best = {tag: float("-inf") for tag, _, _ in self.targets}
        self.best["balance"] = float("-inf")
        self.min_miou_thresholds = _parse_best_checkpoint_min_miou(min_miou_thresholds)
        self._apply_min_miou_thresholds()
        self.last_seen_iter = -1
        self._load_previous_best(output_dir)
        self._apply_min_miou_thresholds()
        if self.min_miou_thresholds and comm.is_main_process():
            print(
                f"[best_checkpointer] minimum mIoU floors: {self.min_miou_thresholds}",
                flush=True,
            )

    def _apply_min_miou_thresholds(self):
        for tag, threshold in self.min_miou_thresholds.items():
            if tag in self.best and np.isfinite(threshold):
                self.best[tag] = max(self.best[tag], threshold)

    def _load_previous_best(self, output_dir):
        if not output_dir:
            return
        metrics_path = os.path.join(output_dir, "metrics.json")
        if not os.path.exists(metrics_path):
            return
        with open(metrics_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for tag in self.best:
                    value = record.get(f"best_{tag}_mIoU")
                    if value is not None:
                        self.best[tag] = max(self.best[tag], float(value))

    def _result_for_target(self, tag, dataset_names, results):
        dataset_result = {}
        for dataset_name in dataset_names:
            if dataset_name in results:
                dataset_result = results[dataset_name]
                break
            if len(self.dataset_names) == 1 and dataset_name == self.dataset_names[0]:
                dataset_result = results
                break
        if (
            not dataset_result
            and len(self.dataset_names) == 1
            and _dataset_matches_best_tag(tag, self.dataset_names[0])
        ):
            dataset_result = results
        if not dataset_result:
            for dataset_name, candidate_result in results.items():
                if _dataset_matches_best_tag(tag, dataset_name):
                    dataset_result = candidate_result
                    break
        return dataset_result

    @staticmethod
    def _miou_from_result(dataset_result):
        if not isinstance(dataset_result, dict) or not dataset_result:
            return None
        sem_seg_result = dataset_result.get("sem_seg", dataset_result)
        if not isinstance(sem_seg_result, dict):
            return None
        return sem_seg_result.get("mIoU")

    def _miou_for_tag(self, tag, results):
        for target_tag, dataset_names, _ in self.targets:
            if target_tag != tag:
                continue
            return self._miou_from_result(self._result_for_target(tag, dataset_names, results))
        return None

    def _maybe_save_balance_best(self, results):
        day_miou = self._miou_for_tag("day", results)
        night_miou = self._miou_for_tag("night", results)
        if day_miou is None or night_miou is None:
            return
        balance_miou = 0.5 * (float(day_miou) + float(night_miou))
        if balance_miou <= self.best["balance"]:
            return
        self.best["balance"] = balance_miou
        overall_miou = self._miou_for_tag("overall", results)
        checkpoint_state = {
            "iteration": self.trainer.iter,
            "best_balance_mIoU": balance_miou,
            "best_balance_day_mIoU": float(day_miou),
            "best_balance_night_mIoU": float(night_miou),
        }
        if overall_miou is not None:
            checkpoint_state["best_balance_overall_mIoU"] = float(overall_miou)
        self.trainer.checkpointer.save("best_model_cosec_balance", **checkpoint_state)
        self.trainer.storage.put_scalar("best_balance_mIoU", balance_miou, smoothing_hint=False)
        self.trainer.storage.put_scalar("best_balance_day_mIoU", float(day_miou), smoothing_hint=False)
        self.trainer.storage.put_scalar("best_balance_night_mIoU", float(night_miou), smoothing_hint=False)
        if overall_miou is not None:
            self.trainer.storage.put_scalar("best_balance_overall_mIoU", float(overall_miou), smoothing_hint=False)

    def _maybe_save_best(self):
        results = getattr(self.trainer, "_last_eval_results", None)
        if not results or self.trainer.iter == self.last_seen_iter:
            return
        self.last_seen_iter = self.trainer.iter
        for tag, dataset_names, checkpoint_name in self.targets:
            miou = self._miou_from_result(self._result_for_target(tag, dataset_names, results))
            if miou is None:
                continue
            if miou > self.best[tag]:
                self.best[tag] = miou
                self.trainer.checkpointer.save(
                    checkpoint_name,
                    iteration=self.trainer.iter,
                    **{f"best_{tag}_mIoU": miou},
                )
                self.trainer.storage.put_scalar(f"best_{tag}_mIoU", miou, smoothing_hint=False)
        self._maybe_save_balance_best(results)

    def after_step(self):
        self._maybe_save_best()

    def after_train(self):
        self._maybe_save_best()


class DSEC19ToDSEC11SemSegEvaluator(SemSegEvaluator):
    def __init__(self, dataset_name, distributed=True, output_dir=None):
        super().__init__(dataset_name, distributed=distributed, output_dir=output_dir)
        meta = MetadataCatalog.get(dataset_name)
        self._source_to_target = np.asarray(
            getattr(meta, "dsec19_to_dsec11", DSEC19_TO_DSEC11),
            dtype=np.int64,
        )

    def _map_prediction(self, pred, output_class_count):
        if int(output_class_count) == self._num_classes:
            mapped = pred.astype(np.int64, copy=True)
        else:
            mapped = np.full(pred.shape, self._num_classes, dtype=np.int64)
            valid = (pred >= 0) & (pred < len(self._source_to_target))
            mapped[valid] = self._source_to_target[pred[valid]]
        invalid = (mapped < 0) | (mapped >= self._num_classes)
        mapped[invalid] = self._num_classes
        return mapped

    def _map_sem_seg_scores(self, sem_seg):
        output_class_count = int(sem_seg.shape[0])
        if output_class_count == self._num_classes:
            return np.array(sem_seg.argmax(dim=0).to(self._cpu_device), dtype=np.int64)
        if output_class_count != len(self._source_to_target):
            raw_pred = np.array(sem_seg.argmax(dim=0).to(self._cpu_device), dtype=np.int64)
            return self._map_prediction(raw_pred, output_class_count)

        mapped_scores = sem_seg.new_zeros((self._num_classes, *sem_seg.shape[-2:]))
        for source_id, target_id in enumerate(self._source_to_target):
            target_id = int(target_id)
            if 0 <= target_id < self._num_classes:
                mapped_scores[target_id] += sem_seg[source_id]
        return np.array(mapped_scores.argmax(dim=0).to(self._cpu_device), dtype=np.int64)

    def process(self, inputs, outputs):
        for input_record, output in zip(inputs, outputs):
            sem_seg = output["sem_seg"]
            output_class_count = sem_seg.shape[0]
            pred = self._map_sem_seg_scores(sem_seg)

            with PathManager.open(self.input_file_to_gt_file[input_record["file_name"]], "rb") as handle:
                gt = np.array(Image.open(handle), dtype=np.int64)
            invalid_gt = (gt == self._ignore_label) | (gt < 0) | (gt >= self._num_classes)
            gt[invalid_gt] = self._num_classes

            self._conf_matrix += np.bincount(
                (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)

            self._predictions.extend(self.encode_json_sem_seg(pred, input_record["file_name"]))


class CoSECTrainer(Mask2FormerTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == "sem_seg_dsec19_to_dsec11":
            if output_folder is None:
                output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            return DSEC19ToDSEC11SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        return super().build_evaluator(cfg, dataset_name, output_folder)

    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.INPUT.EVENT.CONCAT_TO_IMAGE or cfg.MODEL.INIT_DSEC11_HEAD_FROM_DSEC19:
            checkpointer_kwargs = {
                "init_dsec11_head_from_dsec19": cfg.MODEL.INIT_DSEC11_HEAD_FROM_DSEC19,
            }
            if cfg.INPUT.EVENT.CONCAT_TO_IMAGE:
                checkpointer_kwargs["input_concat_channels"] = cfg.INPUT.EVENT.CONCAT_CHANNELS
            self.checkpointer = CoSECDetectionCheckpointer(
                self.model,
                cfg.OUTPUT_DIR,
                trainer=weakref.proxy(self),
                **checkpointer_kwargs,
            )

    @classmethod
    def build_model(cls, cfg):
        model = super().build_model(cfg)
        if cfg.MODEL.TRAINABLE_PREFIXES:
            for parameter in model.parameters():
                parameter.requires_grad = False
            trainable_count = 0
            trainable_prefixes = list(cfg.MODEL.TRAINABLE_PREFIXES)
            for name, parameter in model.named_parameters():
                if any(name.startswith(prefix) for prefix in trainable_prefixes):
                    parameter.requires_grad = True
                    trainable_count += parameter.numel()
            if comm.is_main_process():
                print(
                    f"[fine_tune] custom trainable prefixes enabled; "
                    f"prefixes={trainable_prefixes}, trainable parameters: {trainable_count}",
                    flush=True,
                )
        if cfg.MODEL.EVENT_FUSION.ENABLED and cfg.MODEL.EVENT_FUSION.TRAIN_ONLY_EVENT:
            for parameter in model.parameters():
                parameter.requires_grad = False
            trainable_count = 0
            for name, parameter in model.named_parameters():
                if name.startswith("event_fusion."):
                    parameter.requires_grad = True
                    trainable_count += parameter.numel()
            if comm.is_main_process():
                print(
                    f"[event_fusion] train-only-event enabled; "
                    f"trainable event_fusion parameters: {trainable_count}",
                    flush=True,
                )
        if cfg.MODEL.EVENT_EDGE.ENABLED and cfg.MODEL.EVENT_EDGE.TRAIN_ONLY_EDGE:
            for parameter in model.parameters():
                parameter.requires_grad = False
            trainable_count = 0
            for name, parameter in model.named_parameters():
                if name.startswith("event_edge_head."):
                    parameter.requires_grad = True
                    trainable_count += parameter.numel()
            if comm.is_main_process():
                print(
                    f"[event_edge] train-only-edge enabled; "
                    f"trainable event_edge_head parameters: {trainable_count}",
                    flush=True,
                )
        if cfg.MODEL.EVENT_EDGE_GUIDE.ENABLED and cfg.MODEL.EVENT_EDGE_GUIDE.TRAINABLE_PREFIXES:
            for parameter in model.parameters():
                parameter.requires_grad = False
            trainable_count = 0
            trainable_prefixes = list(cfg.MODEL.EVENT_EDGE_GUIDE.TRAINABLE_PREFIXES)
            for name, parameter in model.named_parameters():
                if any(name.startswith(prefix) for prefix in trainable_prefixes):
                    parameter.requires_grad = True
                    trainable_count += parameter.numel()
            if comm.is_main_process():
                print(
                    f"[event_edge_guide] custom trainable prefixes enabled; "
                    f"prefixes={trainable_prefixes}, trainable parameters: {trainable_count}",
                    flush=True,
                )
        elif cfg.MODEL.EVENT_EDGE_GUIDE.ENABLED and cfg.MODEL.EVENT_EDGE_GUIDE.TRAIN_ONLY_GUIDE:
            for parameter in model.parameters():
                parameter.requires_grad = False
            trainable_count = 0
            for name, parameter in model.named_parameters():
                should_train = name.startswith("event_edge_guide.")
                if cfg.MODEL.EVENT_EDGE_GUIDE.TRAIN_EDGE_HEAD:
                    should_train = should_train or name.startswith("event_edge_head.")
                if should_train:
                    parameter.requires_grad = True
                    trainable_count += parameter.numel()
            if comm.is_main_process():
                print(
                    f"[event_edge_guide] train-only-guide enabled; "
                    f"trainable event_edge_guide/event_edge_head parameters: {trainable_count}",
                    flush=True,
                )
        elif cfg.MODEL.EVENT_EDGE_GUIDE.ENABLED and cfg.MODEL.EVENT_EDGE_GUIDE.TRAIN_WITH_SEM_SEG_HEAD:
            for parameter in model.parameters():
                parameter.requires_grad = False
            trainable_count = 0
            trainable_prefixes = ["event_edge_guide.", "sem_seg_head."]
            if cfg.MODEL.EVENT_EDGE_GUIDE.TRAIN_EDGE_HEAD:
                trainable_prefixes.append("event_edge_head.")
            for name, parameter in model.named_parameters():
                if any(name.startswith(prefix) for prefix in trainable_prefixes):
                    parameter.requires_grad = True
                    trainable_count += parameter.numel()
            if comm.is_main_process():
                print(
                    f"[event_edge_guide] train-with-sem-seg-head enabled; "
                    f"frozen backbone, trainable prefixes={trainable_prefixes}, "
                    f"trainable parameters: {trainable_count}",
                    flush=True,
                )
        return model

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, False)
            return build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        return super().build_test_loader(cfg, dataset_name)

    def build_writers(self):
        return [
            CommonMetricPrinter(self.max_iter),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
        ]

    def build_hooks(self):
        ret = super().build_hooks()
        if self.cfg.TRAIN.DISABLE_PERIODIC_CHECKPOINT:
            ret = [hook for hook in ret if not isinstance(hook, hooks.PeriodicCheckpointer)]
        if self.cfg.DATASETS.TEST:
            ret.insert(
                -1,
                BestValidationCheckpointer(
                    self.cfg.OUTPUT_DIR,
                    self.cfg.DATASETS.TEST,
                    self.cfg.TRAIN.BEST_CHECKPOINT_MIN_MIOU,
                ),
            )
        return ret


def main(args):
    register_cosec()
    from train_net import setup  # noqa: WPS433

    cfg = setup(args)
    if comm.is_main_process() and not should_skip_code_backup():
        backup_runtime_code(cfg.OUTPUT_DIR, args)
    if args.eval_only:
        model = CoSECTrainer.build_model(cfg)
        checkpointer_cls = (
            CoSECDetectionCheckpointer
            if cfg.INPUT.EVENT.CONCAT_TO_IMAGE or cfg.MODEL.INIT_DSEC11_HEAD_FROM_DSEC19
            else DetectionCheckpointer
        )
        checkpointer_kwargs = {}
        if cfg.INPUT.EVENT.CONCAT_TO_IMAGE:
            checkpointer_kwargs["input_concat_channels"] = cfg.INPUT.EVENT.CONCAT_CHANNELS
        if cfg.MODEL.INIT_DSEC11_HEAD_FROM_DSEC19:
            checkpointer_kwargs["init_dsec11_head_from_dsec19"] = True
        checkpointer_cls(model, save_dir=cfg.OUTPUT_DIR, **checkpointer_kwargs).resume_or_load(
            cfg.MODEL.WEIGHTS,
            resume=args.resume,
        )
        res = CoSECTrainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(CoSECTrainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            write_eval_results(cfg.OUTPUT_DIR, res)
        return res

    trainer = CoSECTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
