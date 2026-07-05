#!/usr/bin/env python
"""Cache Mask2Former/Swin-L prediction confidence features for ensemble routing."""

import argparse
import copy
import os
import sys
import importlib.util
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

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

from acdc_dataset import load_acdc_dicts  # noqa: E402
from ensemble_feature_cache_common import (  # noqa: E402
    SegmentationStats,
    image_id_from_record,
    load_label,
    per_image_summary,
    resize_if_needed,
    safe_name,
    save_feature_maps,
    semseg_stats_from_logits,
    semseg_stats_from_prob,
    split_name_from_path,
    valid_label_mask,
    write_csv,
    write_json,
)
from detectron2.checkpoint import DetectionCheckpointer  # noqa: E402
from detectron2.config import get_cfg  # noqa: E402
from detectron2.data import DatasetCatalog  # noqa: E402
from detectron2.projects.deeplab import add_deeplab_config  # noqa: E402
from mask2former import MaskFormerSemanticDatasetMapper, add_maskformer2_config  # noqa: E402
from train_mask2former_cosec import CoSECTrainer, register_cosec  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--out-root", default=str(ROOT / "work_dirs/ensemble_feature_cache"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-save-maps", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--scale-specs",
        default="s512:512:1200,s624:624:1200,s768:768:1400,s1024:1024:1600",
        help="Named TTA scale specs: name:min_size:max_size.",
    )
    parser.add_argument(
        "--scale-set",
        default="",
        help="Optional TTA scale names joined by '+', e.g. s512+s624+s768+s1024.",
    )
    parser.add_argument("--flip", action="store_true", help="Use horizontal flip inside each TTA scale.")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()


def split_csv(text):
    return [part.strip() for part in str(text).split(",") if part.strip()]


def parse_scale_specs(text):
    specs = OrderedDict()
    for item in split_csv(text):
        name, min_size, max_size = item.split(":")
        specs[name] = {"name": name, "min_size": int(min_size), "max_size": int(max_size)}
    return specs


def parse_scale_set(text):
    return [part.strip() for part in str(text).split("+") if part.strip()]


def setup_cfg(args, min_size=None, max_size=None):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    if args.opts:
        opts = args.opts[1:] if args.opts and args.opts[0] == "--" else args.opts
        cfg.merge_from_list(opts)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.DEVICE = args.device
    cfg.DATASETS.TEST = ()
    if min_size is not None:
        cfg.TEST.AUG.ENABLED = False
        cfg.INPUT.MIN_SIZE_TEST = int(min_size)
    if max_size is not None:
        cfg.INPUT.MAX_SIZE_TEST = int(max_size)
    cfg.freeze()
    return cfg


def ensure_extra_datasets():
    for name, condition, split in (
        ("acdc_all_val", "all", "val"),
        ("acdc_all_trainval", "all", "trainval"),
    ):
        if name not in DatasetCatalog.list():
            DatasetCatalog.register(name, lambda condition=condition, split=split: load_acdc_dicts(condition, split))


def build_model(cfg):
    model = CoSECTrainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=False)
    model.eval()
    return model


def infer_logits(model, mapped):
    with torch.no_grad():
        output = model([dict(mapped)])[0]["sem_seg"].detach().cpu().numpy()
    return output


def resize_scores(scores, shape):
    if tuple(scores.shape[-2:]) == tuple(shape):
        return scores
    return F.interpolate(
        scores.unsqueeze(0),
        size=shape,
        mode="bilinear",
        align_corners=False,
    )[0]


def normalize_scores(scores):
    prob = scores.float().clamp_min(1e-8)
    return prob / prob.sum(dim=0, keepdim=True).clamp_min(1e-8)


def infer_prob(model, mapped, label_shape, use_flip=False):
    with torch.no_grad():
        scores = model([dict(mapped)])[0]["sem_seg"].detach().cpu()
        if use_flip:
            flipped = dict(mapped)
            flipped["image"] = torch.flip(mapped["image"], dims=[2])
            flip_scores = model([flipped])[0]["sem_seg"].detach().cpu()
            scores = 0.5 * (scores + torch.flip(flip_scores, dims=[2]))
    return normalize_scores(resize_scores(scores, label_shape)).numpy()


def build_tta_mappers(args):
    scale_names = parse_scale_set(args.scale_set)
    if not scale_names:
        return []
    specs = parse_scale_specs(args.scale_specs)
    missing = [name for name in scale_names if name not in specs]
    if missing:
        raise ValueError(f"Unknown scale names in --scale-set: {missing}. Available: {list(specs)}")
    mappers = []
    for name in scale_names:
        spec = specs[name]
        cfg = setup_cfg(args, min_size=spec["min_size"], max_size=spec["max_size"])
        mappers.append((name, MaskFormerSemanticDatasetMapper(cfg, False)))
    return mappers


def process_dataset(model, mapper, dataset_name, out_dir, limit=None, save_maps=True, tta_mappers=None, use_flip=False):
    records = DatasetCatalog.get(dataset_name)
    if limit is not None:
        records = records[: int(limit)]

    per_image_rows = []
    per_image_class_rows = []
    global_meter = SegmentationStats()
    split_meters = {}
    map_dir = out_dir / "maps" / dataset_name

    iterator = tqdm(records, desc=f"swinl:{dataset_name}")
    for index, record in enumerate(iterator):
        image_id = image_id_from_record(record)
        img_path = record["file_name"]
        label_path = record["sem_seg_file_name"]
        label = load_label(label_path)
        if tta_mappers:
            prob_sum = None
            for _, tta_mapper in tta_mappers:
                mapped = tta_mapper(copy.deepcopy(record))
                prob = infer_prob(model, mapped, label.shape, use_flip=use_flip).astype(np.float32, copy=False)
                prob_sum = prob if prob_sum is None else prob_sum + prob
            stats = semseg_stats_from_prob(prob_sum / float(len(tta_mappers)))
        else:
            mapped = mapper(copy.deepcopy(record))
            logits = infer_logits(model, mapped)
            if logits.shape[1:] != label.shape:
                resized = []
                for class_id in range(logits.shape[0]):
                    resized.append(
                        cv2.resize(logits[class_id], (label.shape[1], label.shape[0]), interpolation=cv2.INTER_LINEAR)
                    )
                logits = np.stack(resized, axis=0)
            stats = semseg_stats_from_logits(logits)
        pred = resize_if_needed(stats["pred"], label.shape, cv2.INTER_NEAREST).astype(np.uint8, copy=False)
        conf = resize_if_needed(stats["conf"], label.shape, cv2.INTER_LINEAR).astype(np.float16, copy=False)
        margin = resize_if_needed(stats["margin"], label.shape, cv2.INTER_LINEAR).astype(np.float16, copy=False)
        entropy = resize_if_needed(stats["entropy"], label.shape, cv2.INTER_LINEAR).astype(np.float16, copy=False)

        split = split_name_from_path(img_path)
        summary = per_image_summary(pred, label, conf, margin, entropy)
        row = {
            "dataset": dataset_name,
            "split": split,
            "image_id": image_id,
            "img_path": img_path,
            "label_path": label_path,
            **summary,
        }
        per_image_rows.append(row)

        global_meter.update(pred, label, conf=conf, margin=margin, entropy=entropy)
        split_meters.setdefault(split, SegmentationStats()).update(pred, label, conf=conf, margin=margin, entropy=entropy)

        per_class_meter = SegmentationStats()
        per_class_meter.update(pred, label, conf=conf, margin=margin, entropy=entropy)
        for class_row in per_class_meter.class_rows():
            class_row.update({"dataset": dataset_name, "split": split, "image_id": image_id})
            per_image_class_rows.append(class_row)

        if save_maps:
            save_feature_maps(map_dir / f"{index:06d}_{safe_name(image_id)}.npz", pred, conf, margin, entropy)

    write_csv(out_dir / f"{dataset_name}_per_image.csv", per_image_rows)
    write_csv(out_dir / f"{dataset_name}_per_image_class_iou.csv", per_image_class_rows)
    write_csv(out_dir / f"{dataset_name}_per_class_iou.csv", global_meter.class_rows())
    for split, meter in split_meters.items():
        write_csv(out_dir / f"{dataset_name}_{safe_name(split)}_per_class_iou.csv", meter.class_rows())

    metrics = global_meter.metrics()
    split_metrics = {split: meter.metrics() for split, meter in split_meters.items()}
    return {
        "dataset": dataset_name,
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
    register_cosec()
    ensure_extra_datasets()
    cfg = setup_cfg(args)
    mapper = MaskFormerSemanticDatasetMapper(cfg, False)
    tta_mappers = build_tta_mappers(args)
    model = build_model(cfg)

    out_dir = Path(args.out_root) / safe_name(args.model_name)
    if out_dir.exists() and args.overwrite:
        import shutil

        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_summaries = []
    for dataset_name in args.datasets:
        dataset_summaries.append(
            process_dataset(
                model,
                mapper,
                dataset_name,
                out_dir,
                limit=args.limit,
                save_maps=not args.no_save_maps,
                tta_mappers=tta_mappers,
                use_flip=args.flip,
            )
        )

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "backend": "swinl_mask2former",
        "model_name": args.model_name,
        "config_file": str(Path(args.config_file).resolve()),
        "weights": str(Path(args.weights).resolve()),
        "device": args.device,
        "limit": args.limit,
        "save_maps": not args.no_save_maps,
        "tta": {
            "scale_specs": args.scale_specs,
            "scale_set": args.scale_set,
            "flip": args.flip,
        },
        "datasets": dataset_summaries,
    }
    write_json(out_dir / "summary.json", manifest)
    print(f"Wrote ensemble feature cache: {out_dir}")


if __name__ == "__main__":
    main()
