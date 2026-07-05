#!/usr/bin/env python
"""Cache mmseg hard predictions for class-route ensemble diagnostics."""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTHONNOUSERSITE", "1")
sys.path = [path for path in sys.path if "/.local/lib/python" not in path]

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.utils import import_modules_from_strings
from mmseg.apis import inference_model, init_model
from mmseg.utils import register_all_modules
from tqdm import tqdm

def _eventshift_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "configs").is_dir() and (parent / "third_party").is_dir():
            return parent
    return Path(__file__).resolve().parents[1]


ROOT = _eventshift_root()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from acdc_dataset import load_acdc_dicts, load_acdc_file_split_dicts  # noqa: E402
from cosec_finetune_splits import iter_cosec_samples  # noqa: E402
from ensemble_feature_cache_common import (  # noqa: E402
    SegmentationStats,
    image_id_from_record,
    load_label,
    resize_if_needed,
    safe_name,
    split_name_from_path,
    write_csv,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument(
        "--record-datasets",
        nargs="+",
        required=True,
        help="Choices: cosec_day_train, cosec_night_train, cosec_train, "
        "cosec_day_val, cosec_night_val, cosec_val, acdc_night_val, acdc_all_val.",
    )
    parser.add_argument("--out-root", default=str(ROOT / "work_dirs/ensemble_pred_cache"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--scale-specs",
        default="s512:512:1200,s624:624:1200,s768:768:1400,s1024:1024:1600",
        help="Named TTA scale specs: name:min_size:max_size.",
    )
    parser.add_argument("--scale-set", default="", help="TTA scale names joined by '+', e.g. s512+s624+s768+s1024.")
    parser.add_argument("--flip", action="store_true", help="Use horizontal flip for each TTA scale.")
    return parser.parse_args()


def maybe_import_custom_modules(cfg):
    custom_imports = cfg.get("custom_imports", None)
    if custom_imports is not None:
        import_modules_from_strings(**custom_imports)


def records_for_project_dataset(name):
    unified_cosec_prefix = "cosec_unified_classcover_v1_"
    if name.startswith(unified_cosec_prefix):
        split = "unified_classcover_v1_" + name[len(unified_cosec_prefix) :]
        records = []
        for seq_name, frame_id, img_path, label_path in iter_cosec_samples(ROOT / "data/train", split):
            records.append(
                {
                    "file_name": str(img_path),
                    "sem_seg_file_name": str(label_path),
                    "image_id": f"{seq_name}_{frame_id:06d}",
                }
            )
        return records
    if name == "acdc_unified_classcover_v1_all_val":
        return load_acdc_file_split_dicts("unified_classcover_v1", "val", "all")
    if name == "acdc_unified_classcover_v1_night_val":
        return load_acdc_file_split_dicts("unified_classcover_v1", "val", "night")
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


def pred_from_result(result, shape):
    pred = result.pred_sem_seg.data.detach().cpu().numpy()
    pred = np.squeeze(pred).astype(np.uint8, copy=False)
    return resize_if_needed(pred, shape, cv2.INTER_NEAREST).astype(np.uint8, copy=False)


def split_csv(text):
    return [part.strip() for part in str(text).split(",") if part.strip()]


def parse_scale_specs(text):
    specs = {}
    for item in split_csv(text):
        name, min_size, max_size = item.split(":")
        specs[name] = {"name": name, "min_size": int(min_size), "max_size": int(max_size)}
    return specs


def parse_scale_set(text):
    return [part.strip() for part in str(text).split("+") if part.strip()]


def resize_short_edge(image, min_size, max_size):
    height, width = image.shape[:2]
    scale = float(min_size) / float(min(height, width))
    if max(height, width) * scale > float(max_size):
        scale = float(max_size) / float(max(height, width))
    new_height = max(1, int(round(height * scale)))
    new_width = max(1, int(round(width * scale)))
    if (new_height, new_width) == (height, width):
        return image
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)


def logits_from_result(result):
    return result.seg_logits.data.detach().float().cpu()


def prob_from_logits(logits, shape):
    if tuple(logits.shape[-2:]) != tuple(shape):
        logits = F.interpolate(logits.unsqueeze(0), size=shape, mode="bilinear", align_corners=False)[0]
    prob = torch.softmax(logits, dim=0)
    return prob.cpu().numpy().astype(np.float32, copy=False)


def pred_from_tta(model, img_path, shape, args):
    scale_names = parse_scale_set(args.scale_set)
    if not scale_names:
        result = inference_model(model, img_path)
        return pred_from_result(result, shape)

    specs = parse_scale_specs(args.scale_specs)
    missing = [name for name in scale_names if name not in specs]
    if missing:
        raise ValueError(f"Unknown scale names in --scale-set: {missing}. Available: {sorted(specs)}")

    image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {img_path}")

    prob_sum = None
    count = 0
    for scale_name in scale_names:
        spec = specs[scale_name]
        resized = resize_short_edge(image, spec["min_size"], spec["max_size"])
        result = inference_model(model, resized)
        prob = prob_from_logits(logits_from_result(result), shape)
        prob_sum = prob if prob_sum is None else prob_sum + prob
        count += 1

        if args.flip:
            flipped = np.ascontiguousarray(resized[:, ::-1, :])
            result = inference_model(model, flipped)
            logits = torch.flip(logits_from_result(result), dims=[2])
            prob_sum += prob_from_logits(logits, shape)
            count += 1

    pred = (prob_sum / float(count)).argmax(axis=0)
    return pred.astype(np.uint8, copy=False)


def save_pred(path, pred):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, pred=np.asarray(pred, dtype=np.uint8))


def per_image_metrics(pred, label):
    meter = SegmentationStats()
    meter.update(pred, label)
    metrics = meter.metrics()
    return {
        "valid_pixels": int(((label != 255) & (label >= 0) & (label < 19)).sum()),
        "mIoU": metrics["mIoU"],
        "mAcc": metrics["mAcc"],
        "aAcc": metrics["aAcc"],
    }


def process_records(model, records, dataset_name, out_dir, args):
    if args.limit is not None:
        records = records[: int(args.limit)]

    per_image_rows = []
    per_image_class_rows = []
    global_meter = SegmentationStats()
    split_meters = {}
    map_dir = out_dir / "maps" / dataset_name

    for index, record in enumerate(tqdm(records, desc=f"mmseg-pred:{args.model_name}:{dataset_name}")):
        img_path = record["file_name"]
        label_path = record["sem_seg_file_name"]
        label = load_label(label_path)
        pred = pred_from_tta(model, img_path, label.shape, args)

        image_id = image_id_from_record(record)
        split = split_name_from_path(img_path)
        per_image_rows.append(
            {
                "dataset": dataset_name,
                "split": split,
                "image_id": image_id,
                "img_path": img_path,
                "label_path": label_path,
                **per_image_metrics(pred, label),
            }
        )

        global_meter.update(pred, label)
        split_meters.setdefault(split, SegmentationStats()).update(pred, label)

        per_class_meter = SegmentationStats()
        per_class_meter.update(pred, label)
        for class_row in per_class_meter.class_rows():
            class_row.update({"dataset": dataset_name, "split": split, "image_id": image_id})
            per_image_class_rows.append(class_row)

        save_pred(map_dir / f"{index:06d}_{safe_name(image_id)}.npz", pred)

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

    dataset_summaries = [
        process_records(model, records_for_project_dataset(dataset_name), dataset_name, out_dir, args)
        for dataset_name in args.record_datasets
    ]
    write_json(
        out_dir / "summary.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "backend": "mmseg",
            "cache_kind": "pred_only",
            "model_name": args.model_name,
            "config_file": str(Path(args.config_file).resolve()),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "record_datasets": args.record_datasets,
            "device": args.device,
            "limit": args.limit,
            "scale_set": args.scale_set,
            "flip": args.flip,
            "datasets": dataset_summaries,
        },
    )
    print(f"Wrote mmseg pred cache: {out_dir}")


if __name__ == "__main__":
    main()
