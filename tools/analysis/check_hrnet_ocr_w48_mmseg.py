#!/usr/bin/env python
import argparse
import gc
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.utils import import_modules_from_strings
from mmseg.registry import DATASETS, MODELS
from mmseg.utils import register_all_modules


ROOT = Path(".")
DEFAULT_CONFIGS = [
    ROOT / "configs/mmseg/HRNet_OCR_W48_CoSEC_Finetune.py",
    ROOT / "configs/mmseg/HRNet_OCR_W48_ACDC_Night_Finetune.py",
    ROOT / "configs/mmseg/HRNet_OCR_W48_ACDC_All_Finetune.py",
]


def _maybe_import_custom_modules(cfg):
    custom_imports = cfg.get("custom_imports", None)
    if custom_imports is not None:
        import_modules_from_strings(**custom_imports)


def check_config(path: Path, skip_model: bool) -> None:
    cfg = Config.fromfile(path)
    _maybe_import_custom_modules(cfg)

    print(f"\n== {path.name}")
    print(f"work_dir: {cfg.work_dir}")
    print(f"load_from: {cfg.load_from}")
    ckpt_path = Path(cfg.load_from)
    print(f"checkpoint_exists: {ckpt_path.is_file()} ({ckpt_path})")

    train_dataset = DATASETS.build(cfg.train_dataloader.dataset)
    val_dataset = DATASETS.build(cfg.val_dataloader.dataset)
    print(f"train_len: {len(train_dataset)}")
    print(f"val_len: {len(val_dataset)}")

    if not skip_model:
        model = MODELS.build(cfg.model)
        print(f"model: {model.__class__.__name__}")
        print(f"decode_heads: {len(model.decode_head)}")
        del model
        gc.collect()


def check_checkpoint(path: Path) -> None:
    print(f"\n== checkpoint")
    print(path)
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict):
        print(f"top_keys: {sorted(ckpt.keys())[:8]}")
        state_dict = ckpt.get("state_dict", {})
        print(f"state_dict_keys: {len(state_dict)}")
        meta = ckpt.get("meta", {})
        if isinstance(meta, dict):
            print(f"meta_keys: {sorted(meta.keys())[:8]}")
    else:
        print(f"type: {type(ckpt).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="*", type=Path, default=DEFAULT_CONFIGS)
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--skip-checkpoint", action="store_true")
    args = parser.parse_args()

    register_all_modules(init_default_scope=True)
    for config in args.configs:
        check_config(config, skip_model=args.skip_model)

    if not args.skip_checkpoint:
        cfg = Config.fromfile(args.configs[0])
        check_checkpoint(Path(cfg.load_from))


if __name__ == "__main__":
    main()
