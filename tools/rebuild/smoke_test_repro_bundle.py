#!/usr/bin/env python3
"""Smoke-test a non-symlink repro bundle.

The checks are intentionally small:
1. load every checkpoint with torch.load(map_location="cpu");
2. build one representative model per framework and load its bundled weight.

This verifies that the packaged .pth files are readable and compatible with
the current repo/configs without running full validation.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle",
        default="work_dirs/submissions/repro_bundle_20260629_portable",
        help="Path to the portable repro bundle. Relative paths are resolved from swin_l/.",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="eccv_segment repo root. Defaults to the parent of this script's swin_l directory.",
    )
    parser.add_argument(
        "--checks",
        nargs="+",
        default=["torch-load"],
        choices=["torch-load", "m2f", "segformer", "maskdino"],
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    script_path = Path(__file__).resolve()
    swin_l_root = script_path.parents[1]
    repo_root = Path(args.repo_root).resolve() if args.repo_root else swin_l_root.parent
    bundle = Path(args.bundle)
    if not bundle.is_absolute():
        bundle = (repo_root / "swin_l" / bundle).resolve()
    return repo_root, repo_root / "swin_l", bundle


def make_detectron_args(config_file: Path, checkpoint: Path, output_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        config_file=str(config_file),
        eval_only=True,
        resume=False,
        num_gpus=0,
        num_machines=1,
        machine_rank=0,
        dist_url="tcp://127.0.0.1:49152",
        opts=[
            "MODEL.WEIGHTS",
            str(checkpoint),
            "MODEL.DEVICE",
            "cpu",
            "OUTPUT_DIR",
            str(output_dir),
            "DATASETS.TEST",
            "()",
            "TEST.AUG.ENABLED",
            "False",
            "DATALOADER.NUM_WORKERS",
            "0",
        ],
    )


def torch_load_all(bundle: Path) -> None:
    import torch

    checkpoint_dir = bundle / "checkpoints"
    paths = sorted(checkpoint_dir.glob("*.pth"))
    if not paths:
        raise FileNotFoundError(f"No .pth files found under {checkpoint_dir}")
    print(f"[torch-load] {len(paths)} checkpoints")
    for path in paths:
        obj = torch.load(str(path), map_location="cpu")
        if isinstance(obj, dict):
            keys = ",".join(list(obj.keys())[:5])
        else:
            keys = type(obj).__name__
        size_mib = path.stat().st_size / 1024 / 1024
        print(f"  OK {path.name} {size_mib:.1f} MiB keys={keys}")
        del obj
        gc.collect()


def smoke_m2f(repo_root: Path, swin_l_root: Path, bundle: Path) -> None:
    sys.path.insert(0, str(swin_l_root / "tools"))
    sys.path.insert(0, str(swin_l_root / "third_party" / "Mask2Former"))
    sys.path.insert(0, str(swin_l_root / "third_party" / "detectron2"))

    import train_mask2former_cosec as train_m2f
    from detectron2.checkpoint import DetectionCheckpointer
    from train_net import setup

    config = swin_l_root / "configs" / "Mask2Former_SwinL_FullDSEC_CoSEC_ACDC_UnifiedClassCover.yaml"
    checkpoint = bundle / "checkpoints" / "m2f_full_desc_selected_cosec_day.pth"
    with tempfile.TemporaryDirectory(prefix="m2f_smoke_") as tmp:
        args = make_detectron_args(config, checkpoint, Path(tmp))
        train_m2f.register_cosec()
        cfg = setup(args)
        model = train_m2f.CoSECTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=str(tmp)).resume_or_load(str(checkpoint), resume=False)
    print(f"[m2f] OK config={config.name} checkpoint={checkpoint.name}")


def smoke_maskdino(repo_root: Path, swin_l_root: Path, bundle: Path) -> None:
    maskdino_root = repo_root / "maskdino_swinl"
    sys.path.insert(0, str(maskdino_root / "tools"))
    sys.path.insert(0, str(maskdino_root))
    sys.path.insert(0, str(swin_l_root / "tools"))

    import train_maskdino_cosec as train_maskdino
    from detectron2.checkpoint import DetectionCheckpointer

    config = (
        maskdino_root
        / "configs"
        / "cosec"
        / "semantic-segmentation"
        / "maskdino_swinl_full_dsec_cosec_acdc_unified_classcover_v1.yaml"
    )
    checkpoint = bundle / "checkpoints" / "maskdino_full_desc_step1_cosec_day.pth"
    with tempfile.TemporaryDirectory(prefix="maskdino_smoke_") as tmp:
        args = make_detectron_args(config, checkpoint, Path(tmp))
        train_maskdino.register_cosec_dsec_acdc()
        cfg = train_maskdino.setup(args)
        model = train_maskdino.CoSECMaskDINOTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=str(tmp)).resume_or_load(str(checkpoint), resume=False)
    print(f"[maskdino] OK config={config.name} checkpoint={checkpoint.name}")


def smoke_segformer(repo_root: Path, swin_l_root: Path, bundle: Path) -> None:
    sys.path.insert(0, str(swin_l_root))
    sys.path.insert(0, str(swin_l_root / "tools"))

    from mmengine.config import Config
    from mmengine.runner import load_checkpoint
    from mmseg.models import build_segmentor
    from mmseg.utils import register_all_modules

    config = swin_l_root / "configs" / "mmseg" / "SegFormer_B5_FullDSEC_CoSEC_ACDC_Unified.py"
    checkpoint = bundle / "checkpoints" / "segformer_full_desc_selected_cosec_day.pth"
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(str(config))
    model = build_segmentor(cfg.model)
    load_checkpoint(model, str(checkpoint), map_location="cpu")
    print(f"[segformer] OK config={config.name} checkpoint={checkpoint.name}")


def main() -> None:
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    args = parse_args()
    repo_root, swin_l_root, bundle = resolve_paths(args)
    print(f"repo_root={repo_root}")
    print(f"bundle={bundle}")
    if "torch-load" in args.checks:
        torch_load_all(bundle)
    if "m2f" in args.checks:
        smoke_m2f(repo_root, swin_l_root, bundle)
    if "segformer" in args.checks:
        smoke_segformer(repo_root, swin_l_root, bundle)
    if "maskdino" in args.checks:
        smoke_maskdino(repo_root, swin_l_root, bundle)


if __name__ == "__main__":
    main()
