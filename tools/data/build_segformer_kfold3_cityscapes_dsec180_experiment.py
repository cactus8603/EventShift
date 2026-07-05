#!/usr/bin/env python3
"""Build an isolated SegFormer-B5 k-fold CoSEC+DSEC180 experiment.

The CoSEC fold membership intentionally reuses the Swin-L/Mask2Former k-fold
helpers so SegFormer trains on the same fold splits.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path


def _eventshift_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "configs").is_dir() and (parent / "third_party").is_dir():
            return parent
    return Path(__file__).resolve().parents[1]


ROOT = _eventshift_root()
WORKSPACE_ROOT = ROOT.parent
TOOLS_ROOT = ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from cosec_finetune_splits import (  # noqa: E402
    DEFAULT_KFOLD_COUNT,
    iter_cosec_samples,
    mmseg_split_name,
)
from dsec19_filtered_dataset import load_dsec19_close180_dicts  # noqa: E402


EXP_ROOT = ROOT / "experiments" / "segformer_b5_kfold3_cityscapes_dsec180"
DATA_ROOT = EXP_ROOT / "data"
COSEC_MMSEG_ROOT = Path("./data/cosec_mmseg")
CITYSCAPES_PRETRAIN = (
    "third_party/mmsegmentation/pretrained_model/"
    "segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth"
)
BASE_CONFIG = "./configs/SegFormer_B5_CoSEC_DayNight_Finetune.py"
MMSEG_ROOT = "third_party/mmsegmentation"
MAMBASEG_ROOT = "."
CONDA = "conda"
ENV_NAME = "mmseg"
FOLDS = DEFAULT_KFOLD_COUNT
DAY_EXTRA = 250
DSEC_SUBDIR = "DSEC_close180"
EPOCHS = 4


def safe_symlink(src: Path, dst: Path) -> None:
    src = src.resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink():
        current = Path(os.readlink(dst))
        if not current.is_absolute():
            current = (dst.parent / current).resolve()
        if current == src:
            return
        raise RuntimeError(f"Refusing to replace symlink {dst}: {current} != {src}")
    if dst.exists():
        if dst.resolve() == src:
            return
        raise RuntimeError(f"Refusing to replace existing path {dst}")
    dst.symlink_to(src, target_is_directory=src.is_dir())


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def evenly_spaced_subset(items: list[str], keep_count: int) -> list[str]:
    if keep_count >= len(items):
        return list(items)
    if keep_count <= 0:
        return []
    selected = []
    for rank in range(keep_count):
        index = int((rank + 0.5) * len(items) / keep_count)
        selected.append(items[min(index, len(items) - 1)])
    return selected


def cosec_split_names(split: str) -> list[str]:
    names = []
    for seq_name, frame_id, _img_path, _label_path in iter_cosec_samples(
        ROOT / "data" / "train", split
    ):
        names.append(mmseg_split_name(seq_name, frame_id))
    return names


def link_cosec_sequences() -> list[str]:
    linked = []
    for domain in ("images", "labels"):
        source_root = COSEC_MMSEG_ROOT / domain
        target_root = DATA_ROOT / domain
        target_root.mkdir(parents=True, exist_ok=True)
        for seq_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
            if not seq_dir.name.startswith(("Day_", "Night_")):
                continue
            safe_symlink(seq_dir, target_root / seq_dir.name)
            linked.append(f"{domain}/{seq_dir.name}")
    return linked


def dsec_stem(record: dict) -> str:
    return f"{DSEC_SUBDIR}/{record['image_id']}"


def write_cropped_dsec_image(src: Path, label: Path, dst: Path) -> None:
    from PIL import Image

    src = src.resolve()
    label = label.resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(label) as label_img:
        target_size = label_img.size

    if dst.is_symlink():
        dst.unlink()
    elif dst.exists():
        with Image.open(dst) as existing_img:
            if existing_img.size == target_size:
                return
        dst.unlink()

    with Image.open(src) as image:
        if image.size != target_size:
            image = image.crop((0, 0, target_size[0], target_size[1]))
        image.save(dst)


def link_dsec_close180() -> list[str]:
    stems = []
    for record in load_dsec19_close180_dicts():
        stem = dsec_stem(record)
        label_path = Path(record["sem_seg_file_name"])
        write_cropped_dsec_image(Path(record["file_name"]), label_path, DATA_ROOT / "images" / f"{stem}.png")
        safe_symlink(label_path, DATA_ROOT / "labels" / f"{stem}.png")
        stems.append(stem)
    return stems


def verify_pair(stem: str) -> None:
    img_path = DATA_ROOT / "images" / f"{stem}.png"
    label_path = DATA_ROOT / "labels" / f"{stem}.png"
    if not img_path.exists():
        raise FileNotFoundError(f"Missing image for split entry {stem}: {img_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Missing label for split entry {stem}: {label_path}")


def config_text(fold: int, train_count: int) -> str:
    max_iters = train_count * EPOCHS
    return f'''_base_ = "{BASE_CONFIG}"

# Isolated SegFormer-B5 k-fold run. CoSEC fold membership and DSEC180
# auxiliary samples mirror:
#   experiments/clean_swinl_kfold3_cityscapes_dsec180
data_root = "{DATA_ROOT}"
load_from = "{CITYSCAPES_PRETRAIN}"
work_dir = "{EXP_ROOT}/outputs/fold{fold}"

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        ann_file="{EXP_ROOT}/splits/fold{fold}_train.txt",
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        ann_file="{EXP_ROOT}/splits/fold{fold}_val.txt",
    ),
)
test_dataloader = val_dataloader

train_cfg = dict(type="IterBasedTrainLoop", max_iters={max_iters}, val_interval={train_count})
param_scheduler = [
    dict(type="LinearLR", start_factor=1e-6, by_epoch=False, begin=0, end=500),
    dict(type="PolyLR", eta_min=0.0, power=1.0, begin=500, end={max_iters}, by_epoch=False),
]

default_hooks = dict(
    checkpoint=dict(
        interval={train_count},
        max_keep_ckpts=2,
        save_best=["mIoU", "day_mIoU", "night_mIoU"],
        rule=["greater", "greater", "greater"],
    ),
    logger=dict(interval=50),
)
randomness = dict(seed={20260627 + fold}, deterministic=False)
'''


def run_script_text() -> str:
    return f'''#!/usr/bin/env bash
set -euo pipefail

ROOT="{ROOT}"
EXP_ROOT="{EXP_ROOT}"
MMSEG_ROOT="{MMSEG_ROOT}"
MAMBASEG_ROOT="{MAMBASEG_ROOT}"
CONDA="{CONDA}"
ENV_NAME="{ENV_NAME}"
GPU_ID="${{GPU_ID:-0}}"
FOLD="${{1:?Usage: $0 <fold>  # fold is 0, 1, or 2}}"

CFG="${{EXP_ROOT}}/configs/fold${{FOLD}}.py"
LOG_DIR="${{EXP_ROOT}}/logs"
PID_DIR="${{EXP_ROOT}}/pids"
mkdir -p "${{LOG_DIR}}" "${{PID_DIR}}"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${{LOG_DIR}}/fold${{FOLD}}_${{STAMP}}.log"
PID_FILE="${{PID_DIR}}/fold${{FOLD}}.pid"

export PYTHONPATH="${{MAMBASEG_ROOT}}:${{MMSEG_ROOT}}:${{ROOT}}:${{PYTHONPATH:-}}"
export PYTHONNOUSERSITE=1

setsid bash -c "
  echo \\$\\$ > '${{PID_FILE}}' &&
  cd '${{MAMBASEG_ROOT}}' &&
  export PYTHONPATH='${{PYTHONPATH}}' &&
  export PYTHONNOUSERSITE=1 &&
  CUDA_VISIBLE_DEVICES='${{GPU_ID}}' '${{CONDA}}' run --no-capture-output -n '${{ENV_NAME}}' \\
    python '${{MMSEG_ROOT}}/tools/train.py' '${{CFG}}'
" > "${{LOG_FILE}}" 2>&1 < /dev/null &

echo "Launched SegFormer fold ${{FOLD}} on GPU ${{GPU_ID}}"
echo "Config: ${{CFG}}"
echo "Log: ${{LOG_FILE}}"
echo "PID file: ${{PID_FILE}}"
echo "Work dir: ${{EXP_ROOT}}/outputs/fold${{FOLD}}"
'''


def readme_text(summary: dict) -> str:
    lines = [
        "# SegFormer-B5 KFold3 Cityscapes+DSEC180",
        "",
        "Isolated MMSeg experiment for SegFormer-B5 using the same CoSEC k-fold split logic as",
        "`experiments/clean_swinl_kfold3_cityscapes_dsec180`.",
        "",
        "- CoSEC fold membership is produced by `swin_l/tools/cosec_finetune_splits.py`.",
        f"- Each train split is `kfold3_fold{{i}}_train + {DAY_EXTRA}` evenly spaced day-train duplicates + DSEC close180.",
        "- CoSEC files and DSEC labels are linked under this experiment's own `data/` directory.",
        "- DSEC RGB images are copied after the same top-left crop-to-label-size alignment used by `DSECFlatSegmentation`.",
        "- Validation uses the fold's combined CoSEC day/night validation split; `CoSECDayNightIoUMetric` reports both.",
        "",
        "## Counts",
        "",
    ]
    for fold in range(FOLDS):
        fold_summary = summary["folds"][str(fold)]
        lines.append(
            f"- fold{fold}: train={fold_summary['train_total']} "
            f"(base={fold_summary['base_train']}, day_extra={fold_summary['day_extra']}, "
            f"dsec={fold_summary['dsec']}), day_val={fold_summary['day_val']}, "
            f"night_val={fold_summary['night_val']}, max_iters={fold_summary['max_iters']}"
        )
    lines.extend(
        [
            "",
            "## Launch",
            "",
            "```bash",
            "GPU_ID=0 ./run.sh 0",
            "GPU_ID=1 ./run.sh 1",
            "GPU_ID=0 ./run.sh 2",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    EXP_ROOT.mkdir(parents=True, exist_ok=True)
    for subdir in ("configs", "splits", "outputs", "logs", "pids", "data/images", "data/labels"):
        (EXP_ROOT / subdir).mkdir(parents=True, exist_ok=True)

    linked_cosec = link_cosec_sequences()
    dsec_names = link_dsec_close180()
    if len(dsec_names) != 180:
        raise RuntimeError(f"Expected 180 DSEC close samples, got {len(dsec_names)}")

    summary = {
        "experiment": str(EXP_ROOT),
        "data_root": str(DATA_ROOT),
        "cosec_mmseg_root": str(COSEC_MMSEG_ROOT),
        "linked_cosec_dirs": len(linked_cosec),
        "dsec": len(dsec_names),
        "folds": {},
    }

    for fold in range(FOLDS):
        split_prefix = f"kfold{FOLDS}_fold{fold}"
        base_train = cosec_split_names(f"{split_prefix}_train")
        day_train = cosec_split_names(f"{split_prefix}_day_train")
        day_extra = evenly_spaced_subset(day_train, DAY_EXTRA)
        day_val = cosec_split_names(f"{split_prefix}_day_val")
        night_val = cosec_split_names(f"{split_prefix}_night_val")
        val = day_val + night_val
        train = base_train + day_extra + dsec_names

        overlap = set(base_train) & set(val)
        if overlap:
            raise RuntimeError(f"CoSEC train/val overlap for fold{fold}: {sorted(overlap)[:5]}")

        for stem in set(train + val):
            verify_pair(stem)

        write_lines(EXP_ROOT / "splits" / f"fold{fold}_train.txt", train)
        write_lines(EXP_ROOT / "splits" / f"fold{fold}_day_val.txt", day_val)
        write_lines(EXP_ROOT / "splits" / f"fold{fold}_night_val.txt", night_val)
        write_lines(EXP_ROOT / "splits" / f"fold{fold}_val.txt", val)
        (EXP_ROOT / "configs" / f"fold{fold}.py").write_text(
            config_text(fold, len(train)), encoding="utf-8"
        )

        summary["folds"][str(fold)] = {
            "base_train": len(base_train),
            "day_train_pool": len(day_train),
            "day_extra": len(day_extra),
            "dsec": len(dsec_names),
            "train_total": len(train),
            "unique_train_total": len(set(train)),
            "day_val": len(day_val),
            "night_val": len(night_val),
            "val_total": len(val),
            "max_iters": len(train) * EPOCHS,
        }

    run_path = EXP_ROOT / "run.sh"
    run_path.write_text(run_script_text(), encoding="utf-8")
    run_path.chmod(run_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)

    (EXP_ROOT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (EXP_ROOT / "README.md").write_text(readme_text(summary), encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
