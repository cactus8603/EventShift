#!/usr/bin/env python
"""Diagnose multi-scale TTA scale sets on CoSEC validation subsets."""

import argparse
import copy
import json
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

from cosec_finetune_splits import CLASSES  # noqa: E402
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
    parser.add_argument(
        "--scale-specs",
        default="s384:384:1200,s512:512:1200,s624:624:1200,s768:768:1400,s896:896:1600,s1024:1024:1600",
    )
    parser.add_argument(
        "--scale-sets",
        default=(
            "tta3=s512+s768+s1024,"
            "tta4=s512+s624+s768+s1024,"
            "tta5mid=s512+s624+s768+s896+s1024,"
            "tta5small=s384+s512+s624+s768+s1024"
        ),
    )
    parser.add_argument("--datasets", default="cosec_day_val,cosec_night_val")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--flip", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def split_csv(text):
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_scale_specs(text):
    specs = OrderedDict()
    for item in split_csv(text):
        name, min_size, max_size = item.split(":")
        specs[name] = {"name": name, "min_size": int(min_size), "max_size": int(max_size)}
    return specs


def parse_scale_sets(text):
    sets = OrderedDict()
    for item in split_csv(text):
        name, scales = item.split("=", 1)
        sets[name] = [part.strip() for part in scales.split("+") if part.strip()]
    return sets


def setup_cfg(args, min_size, max_size):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.DEVICE = args.device
    cfg.DATASETS.TEST = ()
    cfg.TEST.AUG.ENABLED = False
    cfg.INPUT.MIN_SIZE_TEST = int(min_size)
    cfg.INPUT.MAX_SIZE_TEST = int(max_size)
    cfg.freeze()
    return cfg


def build_model(cfg):
    model = CoSECTrainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS,
        resume=False,
    )
    model.eval()
    return model


def load_label(record):
    label = cv2.imread(record["sem_seg_file_name"], cv2.IMREAD_UNCHANGED)
    if label is None:
        raise RuntimeError(f"Could not read label: {record['sem_seg_file_name']}")
    if label.ndim == 3:
        label = label[:, :, 0]
    return label.astype(np.int64)


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


def infer_prob(model, mapped, label_shape, use_flip):
    with torch.no_grad():
        scores = model([dict(mapped)])[0]["sem_seg"].detach().cpu()
        if use_flip:
            flipped = dict(mapped)
            flipped["image"] = torch.flip(mapped["image"], dims=[2])
            flip_scores = model([flipped])[0]["sem_seg"].detach().cpu()
            scores = 0.5 * (scores + torch.flip(flip_scores, dims=[2]))
    return normalize_scores(resize_scores(scores, label_shape)).to(torch.float16)


class ConfusionMeter:
    def __init__(self, num_classes=19, ignore_label=255):
        self.num_classes = int(num_classes)
        self.ignore_label = int(ignore_label)
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, pred, label):
        pred = np.asarray(pred, dtype=np.int64)
        label = np.asarray(label, dtype=np.int64)
        keep = (label != self.ignore_label) & (label >= 0) & (label < self.num_classes)
        keep &= (pred >= 0) & (pred < self.num_classes)
        indices = self.num_classes * label[keep] + pred[keep]
        self.matrix += np.bincount(indices, minlength=self.num_classes**2).reshape(
            self.num_classes,
            self.num_classes,
        )

    def metrics(self):
        hist = self.matrix.astype(np.float64)
        tp = np.diag(hist)
        pos_gt = hist.sum(axis=1)
        pos_pred = hist.sum(axis=0)
        union = pos_gt + pos_pred - tp
        iou = np.divide(tp, union, out=np.full_like(tp, np.nan), where=union > 0)
        acc = np.divide(tp, pos_gt, out=np.full_like(tp, np.nan), where=pos_gt > 0)
        total = hist.sum()
        return {
            "mIoU": float(100.0 * np.nanmean(iou)),
            "mAcc": float(100.0 * np.nanmean(acc)),
            "aAcc": float(100.0 * tp.sum() / total) if total > 0 else float("nan"),
        }


def collect_scale_probs(args, specs, records, labels):
    outputs = {}
    for name, spec in specs.items():
        print(
            f"[collect] {name} min={spec['min_size']} max={spec['max_size']} records={len(records)}",
            flush=True,
        )
        cfg = setup_cfg(args, spec["min_size"], spec["max_size"])
        mapper = MaskFormerSemanticDatasetMapper(cfg, False)
        model = build_model(cfg)
        probs = []
        for record, label in tqdm(list(zip(records, labels)), desc=f"collect-{name}"):
            mapped = mapper(copy.deepcopy(record))
            probs.append(infer_prob(model, mapped, label.shape, args.flip))
        outputs[name] = probs
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return outputs


def evaluate_set(probs_by_scale, scale_names, labels):
    meter = ConfusionMeter(num_classes=len(CLASSES))
    for image_idx, label in enumerate(labels):
        avg = None
        for scale_name in scale_names:
            prob = probs_by_scale[scale_name][image_idx].float()
            avg = prob if avg is None else avg + prob
        avg = avg / float(len(scale_names))
        pred = avg.argmax(dim=0).to(torch.uint8).numpy()
        meter.update(pred, label)
    return meter.metrics()


def analyze_dataset(args, dataset_name, scale_specs, scale_sets):
    records = list(DatasetCatalog.get(dataset_name))
    if args.limit is not None:
        records = records[: args.limit]
    labels = [load_label(record) for record in records]
    used = OrderedDict()
    for scales in scale_sets.values():
        for name in scales:
            used[name] = scale_specs[name]
    probs_by_scale = collect_scale_probs(args, used, records, labels)
    methods = OrderedDict()
    for set_name, scale_names in scale_sets.items():
        methods[set_name] = evaluate_set(probs_by_scale, scale_names, labels)
    return {
        "dataset": dataset_name,
        "sample_count": len(records),
        "methods": methods,
        "top_by_mIoU": [
            {"method": name, **values}
            for name, values in sorted(methods.items(), key=lambda item: item[1]["mIoU"], reverse=True)
        ],
    }


def write_markdown(output, out_path):
    lines = [
        "# Scale TTA Set Diagnostic",
        "",
        f"created_at: `{output['created_at']}`",
        f"config: `{output['args']['config_file']}`",
        f"weights: `{output['args']['weights']}`",
        f"flip: `{output['args']['flip']}`",
        "",
    ]
    for dataset in output["datasets"]:
        lines.extend(
            [
                f"## {dataset['dataset']}",
                "",
                "| Method | mIoU | mAcc | aAcc |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in dataset["top_by_mIoU"]:
            lines.append(f"| `{row['method']}` | {row['mIoU']:.4f} | {row['mAcc']:.4f} | {row['aAcc']:.4f} |")
        lines.append("")
    md_path = out_path.with_suffix(".md")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def main():
    args = parse_args()
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    register_cosec()
    scale_specs = parse_scale_specs(args.scale_specs)
    scale_sets = parse_scale_sets(args.scale_sets)
    output = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "scale_specs": scale_specs,
        "scale_sets": scale_sets,
        "datasets": [
            analyze_dataset(args, dataset_name, scale_specs, scale_sets)
            for dataset_name in split_csv(args.datasets)
        ],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, sort_keys=True)
        f.write("\n")
    md_path = write_markdown(output, out_path)
    print(f"Wrote diagnostics: {out_path}")
    print(f"Wrote summary: {md_path}")
    for dataset in output["datasets"]:
        print(f"[{dataset['dataset']}]")
        for row in dataset["top_by_mIoU"]:
            print(f"  {row['method']}: mIoU={row['mIoU']:.4f}")


if __name__ == "__main__":
    main()
