#!/usr/bin/env python
import json
import os
import shutil
import sys
import importlib.util
from collections import OrderedDict
from datetime import datetime
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
    load_dsec19_close180_dicts,
    load_dsec19_close240_dicts,
    load_dsec19_filtered_event_dicts,
    load_dsec19_filtered_dicts,
    load_dsec19_full_dicts,
    load_dsec19_train_split_dicts,
    load_dsec19_train_split_event_dicts,
    load_dsec19_val_dicts,
    load_dsec19_val_event_dicts,
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
from detectron2.utils import comm  # noqa: E402
from detectron2.utils.events import CommonMetricPrinter, JSONWriter  # noqa: E402
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


def should_skip_code_backup():
    return os.environ.get("SKIP_CODE_BACKUP", "").lower() in {"1", "true", "yes", "y"}


def load_cosec_dicts(split):
    records = []
    for idx, (seq_name, frame_id, img_path, label_path) in enumerate(iter_cosec_samples(ROOT / "data" / "train", split)):
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
    if tag == "day":
        return dataset_name.startswith("cosec_") and dataset_name.endswith("_day_val")
    if tag == "night":
        return dataset_name.startswith("cosec_") and dataset_name.endswith("_night_val")
    if tag in {"acdc", "acdc_all"}:
        return dataset_name.startswith("acdc_") and dataset_name.endswith("_all_val")
    if tag == "acdc_night":
        return dataset_name.startswith("acdc_") and dataset_name.endswith("_night_val")
    if tag == "dsec19":
        return dataset_name in {"dsec19_val", "dsec19_val_event"}
    return False


class BestValidationCheckpointer(hooks.HookBase):
    DEFAULT_TARGETS = (
        ("day", ("cosec_day_val_event", "cosec_day_val"), "best_model_cosec_day"),
        ("night", ("cosec_night_val_event", "cosec_night_val"), "best_model_cosec_night"),
        ("acdc", ("acdc_all_val",), "best_model_acdc"),
        ("acdc_all", ("acdc_all_val",), "best_model_acdc_all"),
        ("acdc_night", ("acdc_night_val",), "best_model_acdc_night"),
        ("dsec19", ("dsec19_val_event", "dsec19_val"), "best_model_dsec19"),
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

    def _maybe_save_best(self):
        results = getattr(self.trainer, "_last_eval_results", None)
        if not results or self.trainer.iter == self.last_seen_iter:
            return
        self.last_seen_iter = self.trainer.iter
        for tag, dataset_names, checkpoint_name in self.targets:
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
            sem_seg_result = dataset_result.get("sem_seg", dataset_result)
            miou = sem_seg_result.get("mIoU")
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

    def after_step(self):
        self._maybe_save_best()

    def after_train(self):
        self._maybe_save_best()


class CoSECTrainer(Mask2FormerTrainer):
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
    if not args.eval_only and comm.is_main_process() and not should_skip_code_backup():
        backup_runtime_code(cfg.OUTPUT_DIR, args)
    if args.eval_only:
        model = CoSECTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS,
            resume=args.resume,
        )
        res = CoSECTrainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(CoSECTrainer.test_with_TTA(cfg, model))
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
