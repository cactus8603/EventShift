#!/usr/bin/env python
"""Convert Detectron2 sem_seg_predictions.json RLE output into pred cache."""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from tqdm import tqdm

def _eventshift_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "configs").is_dir() and (parent / "third_party").is_dir():
            return parent
    return Path(__file__).resolve().parents[1]


ROOT = _eventshift_root()
sys.path.insert(0, str(ROOT / "tools"))

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
    parser.add_argument("--json", required=True, help="Detectron2 sem_seg_predictions.json")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--out-root", default=str(ROOT / "work_dirs/ensemble_pred_cache"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def label_path_from_image(path):
    path = Path(path)
    parts = list(path.parts)
    if "img_co_left" in parts:
        idx = parts.index("img_co_left")
        parts[idx] = "segment_co"
        return str(Path(*parts))
    if "rgb_anon" in parts:
        idx = parts.index("rgb_anon")
        parts[idx] = "gt"
        name = path.name.replace("_rgb_anon.png", "_gt_labelTrainIds.png")
        return str(Path(*parts[:-1]) / name)
    raise ValueError(f"Cannot infer label path from image path: {path}")


def read_json_predictions(path, limit=None):
    with Path(path).open("r", encoding="utf-8") as f:
        rows = json.load(f)
    by_file = {}
    for row in rows:
        by_file.setdefault(row["file_name"], []).append(row)
    file_names = sorted(by_file)
    if limit is not None:
        file_names = file_names[: int(limit)]
    return [(file_name, by_file[file_name]) for file_name in file_names]


def decode_prediction(entries, shape):
    pred = np.full(shape, 255, dtype=np.uint8)
    for entry in entries:
        rle = entry["segmentation"]
        if isinstance(rle.get("counts"), str):
            rle = {"size": rle["size"], "counts": rle["counts"].encode("utf-8")}
        mask = mask_utils.decode(rle).astype(bool)
        if mask.shape != tuple(shape):
            mask = resize_if_needed(mask.astype(np.uint8), shape, cv2.INTER_NEAREST).astype(bool)
        pred[mask] = int(entry["category_id"])
    pred[pred == 255] = 0
    return pred


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


def write_empty_class_csv(out_dir, dataset_name):
    path = out_dir / f"{dataset_name}_per_image_class_iou.csv"
    if path.exists():
        return
    fieldnames = ["dataset", "split", "image_id", "class_id", "class_name", "iou", "acc", "tp", "union"]
    with path.open("w", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()


def main():
    args = parse_args()
    out_dir = Path(args.out_root) / safe_name(args.model_name)
    if out_dir.exists() and args.overwrite:
        import shutil

        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = read_json_predictions(args.json, limit=args.limit)
    per_image_rows = []
    per_image_class_rows = []
    global_meter = SegmentationStats()
    split_meters = {}
    map_dir = out_dir / "maps" / args.dataset_name

    for index, (file_name, entries) in enumerate(tqdm(items, desc=f"json-cache:{args.model_name}:{args.dataset_name}")):
        label_path = label_path_from_image(file_name)
        label = load_label(label_path)
        pred = decode_prediction(entries, label.shape)
        record = {"file_name": file_name, "sem_seg_file_name": label_path}
        image_id = image_id_from_record(record)
        split = split_name_from_path(file_name)
        per_image_rows.append(
            {
                "dataset": args.dataset_name,
                "split": split,
                "image_id": image_id,
                "img_path": file_name,
                "label_path": label_path,
                **per_image_metrics(pred, label),
            }
        )
        global_meter.update(pred, label)
        split_meters.setdefault(split, SegmentationStats()).update(pred, label)

        per_class_meter = SegmentationStats()
        per_class_meter.update(pred, label)
        for class_row in per_class_meter.class_rows():
            class_row.update({"dataset": args.dataset_name, "split": split, "image_id": image_id})
            per_image_class_rows.append(class_row)

        save_pred(map_dir / f"{index:06d}_{safe_name(image_id)}.npz", pred)

    write_csv(out_dir / f"{args.dataset_name}_per_image.csv", per_image_rows)
    write_csv(out_dir / f"{args.dataset_name}_per_image_class_iou.csv", per_image_class_rows)
    write_csv(out_dir / f"{args.dataset_name}_per_class_iou.csv", global_meter.class_rows())
    for split, meter in split_meters.items():
        write_csv(out_dir / f"{args.dataset_name}_{safe_name(split)}_per_class_iou.csv", meter.class_rows())

    metrics = global_meter.metrics()
    write_json(
        out_dir / "summary.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "backend": "detectron2_json",
            "cache_kind": "pred_only",
            "model_name": args.model_name,
            "json": str(Path(args.json).resolve()),
            "dataset_name": args.dataset_name,
            "limit": args.limit,
            "datasets": [
                {
                    "name": args.dataset_name,
                    "samples": len(items),
                    "mIoU": metrics["mIoU"],
                    "mAcc": metrics["mAcc"],
                    "aAcc": metrics["aAcc"],
                }
            ],
        },
    )
    print(f"Wrote Detectron2 JSON pred cache: {out_dir}")
    print(f"{args.dataset_name}: mIoU={metrics['mIoU']:.4f} mAcc={metrics['mAcc']:.4f} aAcc={metrics['aAcc']:.4f}")


if __name__ == "__main__":
    main()
