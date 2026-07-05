#!/usr/bin/env python
"""Cache mmseg prediction confidence features for ensemble routing."""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.utils import import_modules_from_strings
from mmseg.apis import inference_model, init_model
from mmseg.registry import DATASETS
from mmseg.utils import register_all_modules
from tqdm import tqdm

def _eventshift_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "configs").is_dir() and (parent / "third_party").is_dir():
            return parent
    return Path(__file__).resolve().parents[1]


ROOT = _eventshift_root()
sys.path.insert(0, str(ROOT / "tools"))

from acdc_dataset import load_acdc_dicts  # noqa: E402
from cosec_finetune_splits import iter_cosec_samples  # noqa: E402
from ensemble_feature_cache_common import (  # noqa: E402
    SegmentationStats,
    image_id_from_record,
    load_label,
    per_image_summary,
    safe_name,
    save_feature_maps,
    semseg_stats_from_logits,
    split_name_from_path,
    write_csv,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--dataset-source", choices=["val", "test", "train"], default="val")
    parser.add_argument(
        "--record-datasets",
        nargs="*",
        default=None,
        help="Use project records instead of the mmseg config dataset. "
        "Choices: cosec_day_train, cosec_night_train, cosec_train, "
        "cosec_day_val, cosec_night_val, cosec_val, acdc_night_val, acdc_all_val.",
    )
    parser.add_argument("--out-root", default=str(ROOT / "work_dirs/ensemble_feature_cache"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-save-maps", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def maybe_import_custom_modules(cfg):
    custom_imports = cfg.get("custom_imports", None)
    if custom_imports is not None:
        import_modules_from_strings(**custom_imports)


def dataset_cfg_from_source(cfg, source):
    if source == "train":
        return cfg.train_dataloader.dataset
    if source == "test":
        return cfg.test_dataloader.dataset
    return cfg.val_dataloader.dataset


def data_info_to_record(data_info):
    img_path = data_info.get("img_path") or data_info.get("file_name")
    label_path = data_info.get("seg_map_path") or data_info.get("sem_seg_file_name")
    return {
        "file_name": str(img_path),
        "sem_seg_file_name": str(label_path),
        "image_id": data_info.get("img_id") or Path(str(img_path)).stem,
    }


def records_for_project_dataset(name):
    if name in {
        "cosec_day_train",
        "cosec_night_train",
        "cosec_train",
        "cosec_day_val",
        "cosec_night_val",
        "cosec_val",
    }:
        if name == "cosec_day_val":
            splits = ("day_val",)
        elif name == "cosec_night_val":
            splits = ("night_val",)
        elif name == "cosec_day_train":
            splits = ("day_train",)
        elif name == "cosec_night_train":
            splits = ("night_train",)
        elif name == "cosec_train":
            splits = ("day_train", "night_train")
        else:
            splits = ("day_val", "night_val")
        records = []
        for split in splits:
            for seq_name, frame_id, img_path, label_path in iter_cosec_samples(ROOT / "data/train", split):
                records.append(
                    {
                        "file_name": str(img_path),
                        "sem_seg_file_name": str(label_path),
                        "image_id": f"{seq_name}_{frame_id:06d}",
                    }
                )
        return records
    if name == "acdc_night_val":
        return load_acdc_dicts("night", "val")
    if name == "acdc_all_val":
        return load_acdc_dicts("all", "val")
    raise ValueError(f"Unknown --record-datasets entry: {name}")


def logits_from_result(result, shape):
    logits = result.seg_logits.data.detach().float().cpu()
    if tuple(logits.shape[-2:]) != tuple(shape):
        logits = F.interpolate(logits.unsqueeze(0), size=shape, mode="bilinear", align_corners=False)[0]
    return logits.numpy()


def process_records(model, records, dataset_name, out_dir, args):
    if args.limit is not None:
        records = records[: int(args.limit)]

    per_image_rows = []
    per_image_class_rows = []
    global_meter = SegmentationStats()
    split_meters = {}
    map_dir = out_dir / "maps" / dataset_name

    for index, record in enumerate(tqdm(records, desc=f"mmseg:{args.model_name}:{dataset_name}")):
        img_path = record["file_name"]
        label_path = record["sem_seg_file_name"]
        label = load_label(label_path)
        result = inference_model(model, img_path)
        logits = logits_from_result(result, label.shape)
        stats = semseg_stats_from_logits(logits)
        pred = stats["pred"].astype(np.uint8, copy=False)
        conf = stats["conf"].astype(np.float16, copy=False)
        margin = stats["margin"].astype(np.float16, copy=False)
        entropy = stats["entropy"].astype(np.float16, copy=False)

        if pred.shape != label.shape:
            pred = cv2.resize(pred, (label.shape[1], label.shape[0]), interpolation=cv2.INTER_NEAREST)
            conf = cv2.resize(conf, (label.shape[1], label.shape[0]), interpolation=cv2.INTER_LINEAR)
            margin = cv2.resize(margin, (label.shape[1], label.shape[0]), interpolation=cv2.INTER_LINEAR)
            entropy = cv2.resize(entropy, (label.shape[1], label.shape[0]), interpolation=cv2.INTER_LINEAR)

        image_id = image_id_from_record(record)
        split = split_name_from_path(img_path)
        summary = per_image_summary(pred, label, conf, margin, entropy)
        per_image_rows.append(
            {
                "dataset": dataset_name,
                "split": split,
                "image_id": image_id,
                "img_path": img_path,
                "label_path": label_path,
                **summary,
            }
        )

        global_meter.update(pred, label, conf=conf, margin=margin, entropy=entropy)
        split_meters.setdefault(split, SegmentationStats()).update(pred, label, conf=conf, margin=margin, entropy=entropy)

        per_class_meter = SegmentationStats()
        per_class_meter.update(pred, label, conf=conf, margin=margin, entropy=entropy)
        for class_row in per_class_meter.class_rows():
            class_row.update({"dataset": dataset_name, "split": split, "image_id": image_id})
            per_image_class_rows.append(class_row)

        if not args.no_save_maps:
            save_feature_maps(map_dir / f"{index:06d}_{safe_name(image_id)}.npz", pred, conf, margin, entropy)

    write_csv(out_dir / f"{dataset_name}_per_image.csv", per_image_rows)
    write_csv(out_dir / f"{dataset_name}_per_image_class_iou.csv", per_image_class_rows)
    write_csv(out_dir / f"{dataset_name}_per_class_iou.csv", global_meter.class_rows())
    for split, meter in split_meters.items():
        write_csv(out_dir / f"{dataset_name}_{safe_name(split)}_per_class_iou.csv", meter.class_rows())

    metrics = global_meter.metrics()
    split_metrics = {split: meter.metrics() for split, meter in split_meters.items()}
    return {
        "name": dataset_name,
        "samples": len(records),
        "mIoU": metrics["mIoU"],
        "mAcc": metrics["mAcc"],
        "aAcc": metrics["aAcc"],
        "splits": {
            split: {
                "mIoU": values["mIoU"],
                "mAcc": values["mAcc"],
                "aAcc": values["aAcc"],
            }
            for split, values in split_metrics.items()
        },
    }


def main():
    args = parse_args()
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    register_all_modules(init_default_scope=True)

    cfg = Config.fromfile(args.config_file)
    maybe_import_custom_modules(cfg)
    model = init_model(cfg, args.checkpoint, device=args.device)

    out_dir = Path(args.out_root) / safe_name(args.model_name)
    if out_dir.exists() and args.overwrite:
        import shutil

        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_summaries = []
    if args.record_datasets:
        for dataset_name in args.record_datasets:
            dataset_summaries.append(
                process_records(model, records_for_project_dataset(dataset_name), dataset_name, out_dir, args)
            )
    else:
        dataset = DATASETS.build(dataset_cfg_from_source(cfg, args.dataset_source))
        records = [data_info_to_record(data_info) for data_info in dataset.data_list]
        dataset_name = f"{Path(args.config_file).stem}_{args.dataset_source}"
        dataset_summaries.append(process_records(model, records, dataset_name, out_dir, args))

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "backend": "mmseg",
        "model_name": args.model_name,
        "config_file": str(Path(args.config_file).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "dataset_source": args.dataset_source,
        "record_datasets": args.record_datasets,
        "device": args.device,
        "limit": args.limit,
        "save_maps": not args.no_save_maps,
        "datasets": dataset_summaries,
    }
    write_json(out_dir / "summary.json", manifest)
    print(f"Wrote ensemble feature cache: {out_dir}")


if __name__ == "__main__":
    main()
