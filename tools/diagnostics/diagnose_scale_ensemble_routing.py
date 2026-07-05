#!/usr/bin/env python
"""Diagnose scale/flip routing for CoSEC validation splits."""

import argparse
import copy
import json
import os
import sys
import importlib.util
from collections import OrderedDict, defaultdict
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
        default="s624:624:1200,s512:512:1200,s768:768:1400,s1024:1024:1600",
        help="Comma-separated name:min_size:max_size entries.",
    )
    parser.add_argument("--datasets", default="cosec_day_val,cosec_night_val")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--flip", action="store_true", help="Average each scale with horizontal flip.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def split_csv(text):
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_scale_specs(text):
    specs = []
    for item in split_csv(text):
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError(f"Bad scale spec '{item}', expected name:min:max")
        name, min_size, max_size = parts
        specs.append({"name": name, "min_size": int(min_size), "max_size": int(max_size)})
    return specs


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


def valid_label_mask(label, ignore_label=255):
    return (label != ignore_label) & (label >= 0) & (label < len(CLASSES))


def sequence_name(record):
    image_id = str(record.get("image_id", "unknown"))
    parts = image_id.split("_")
    if len(parts) > 1 and parts[-1].isdigit():
        return "_".join(parts[:-1])
    return Path(record["file_name"]).parents[1].name


def normalize_scores(scores):
    prob = scores.float().clamp_min(1e-8)
    return prob / prob.sum(dim=0, keepdim=True).clamp_min(1e-8)


def resize_scores(scores, shape):
    if tuple(scores.shape[-2:]) == tuple(shape):
        return scores
    return F.interpolate(
        scores.unsqueeze(0),
        size=shape,
        mode="bilinear",
        align_corners=False,
    )[0]


def infer_scores(model, mapped, use_flip):
    with torch.no_grad():
        scores = model([dict(mapped)])[0]["sem_seg"].detach().cpu()
        if not use_flip:
            return scores
        flipped = dict(mapped)
        flipped["image"] = torch.flip(mapped["image"], dims=[2])
        flip_scores = model([flipped])[0]["sem_seg"].detach().cpu()
        flip_scores = torch.flip(flip_scores, dims=[2])
        return 0.5 * (scores + flip_scores)


def top_conf_margin(prob):
    top2 = torch.topk(prob, k=2, dim=0).values
    conf = top2[0].numpy().astype(np.float16, copy=False)
    margin = (top2[0] - top2[1]).numpy().astype(np.float16, copy=False)
    pred = prob.argmax(dim=0).numpy().astype(np.uint8, copy=False)
    return pred, conf, margin


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

    def class_iou(self):
        hist = self.matrix.astype(np.float64)
        true_positive = np.diag(hist)
        pos_gt = hist.sum(axis=1)
        pos_pred = hist.sum(axis=0)
        union = pos_gt + pos_pred - true_positive
        return np.divide(
            true_positive,
            union,
            out=np.full_like(true_positive, np.nan),
            where=union > 0,
        )

    def metrics(self):
        hist = self.matrix.astype(np.float64)
        true_positive = np.diag(hist)
        pos_gt = hist.sum(axis=1)
        pos_pred = hist.sum(axis=0)
        union = pos_gt + pos_pred - true_positive
        iou = np.divide(true_positive, union, out=np.full_like(true_positive, np.nan), where=union > 0)
        acc = np.divide(true_positive, pos_gt, out=np.full_like(true_positive, np.nan), where=pos_gt > 0)
        total = hist.sum()
        return {
            "mIoU": float(100.0 * np.nanmean(iou)),
            "mAcc": float(100.0 * np.nanmean(acc)),
            "aAcc": float(100.0 * true_positive.sum() / total) if total > 0 else float("nan"),
            "class_iou": {
                CLASSES[idx]: (None if np.isnan(value) else float(100.0 * value))
                for idx, value in enumerate(iou)
            },
        }


def empty_counts():
    return {
        "valid_pixels": 0,
        "base_wrong": 0,
        "repaired": 0,
        "damaged": 0,
        "changed": 0,
    }


def add_counts(counts, base_pred, pred, label):
    valid = valid_label_mask(label)
    base_wrong = valid & (base_pred != label)
    base_correct = valid & (base_pred == label)
    repaired = base_wrong & (pred == label)
    damaged = base_correct & (pred != label)
    changed = valid & (base_pred != pred)
    counts["valid_pixels"] += int(valid.sum())
    counts["base_wrong"] += int(base_wrong.sum())
    counts["repaired"] += int(repaired.sum())
    counts["damaged"] += int(damaged.sum())
    counts["changed"] += int(changed.sum())


def div(num, den):
    return float(num / den) if den else 0.0


def finalize_counts(counts):
    return {
        **counts,
        "repair_rate": div(counts["repaired"], counts["base_wrong"]),
        "net_repaired": counts["repaired"] - counts["damaged"],
        "changed_rate": div(counts["changed"], counts["valid_pixels"]),
    }


def evaluate_method(preds, labels, base_preds):
    meter = ConfusionMeter(num_classes=len(CLASSES))
    counts = empty_counts()
    for pred, label, base_pred in zip(preds, labels, base_preds):
        meter.update(pred, label)
        add_counts(counts, base_pred, pred, label)
    return {**meter.metrics(), **finalize_counts(counts)}


def image_miou(pred, label):
    meter = ConfusionMeter(num_classes=len(CLASSES))
    meter.update(pred, label)
    return meter.metrics()["mIoU"]


def collect_branch_outputs(args, spec, records, labels):
    cfg = setup_cfg(args, spec["min_size"], spec["max_size"])
    mapper = MaskFormerSemanticDatasetMapper(cfg, False)
    model = build_model(cfg)
    outputs = []
    iterator = records if args.quiet else tqdm(records, desc=spec["name"])
    for record, label in zip(iterator, labels):
        mapped = mapper(copy.deepcopy(record))
        scores = infer_scores(model, mapped, args.flip)
        prob = normalize_scores(resize_scores(scores, label.shape))
        pred, conf, margin = top_conf_margin(prob)
        outputs.append({"pred": pred, "conf": conf, "margin": margin})
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outputs


def choose_by_stack(outputs_by_branch, field, reducer):
    preds = []
    for image_idx in range(len(outputs_by_branch[0])):
        pred_stack = np.stack([branch[image_idx]["pred"] for branch in outputs_by_branch], axis=0)
        value_stack = np.stack([branch[image_idx][field] for branch in outputs_by_branch], axis=0)
        choice = reducer(value_stack)
        pred = np.take_along_axis(pred_stack, choice[None], axis=0)[0]
        preds.append(pred.astype(np.uint8, copy=False))
    return preds


def sequence_best_predictions(outputs_by_branch, labels, records):
    branch_count = len(outputs_by_branch)
    seq_meters = defaultdict(lambda: [ConfusionMeter(num_classes=len(CLASSES)) for _ in range(branch_count)])
    for image_idx, (record, label) in enumerate(zip(records, labels)):
        seq = sequence_name(record)
        for branch_idx, branch in enumerate(outputs_by_branch):
            seq_meters[seq][branch_idx].update(branch[image_idx]["pred"], label)

    seq_choice = {}
    for seq, meters in seq_meters.items():
        seq_scores = [meter.metrics()["mIoU"] for meter in meters]
        seq_choice[seq] = int(np.nanargmax(seq_scores))

    preds = []
    for image_idx, record in enumerate(records):
        seq = sequence_name(record)
        preds.append(outputs_by_branch[seq_choice[seq]][image_idx]["pred"])
    return preds, seq_choice


def image_oracle_predictions(outputs_by_branch, labels):
    preds = []
    choices = []
    for image_idx, label in enumerate(labels):
        scores = [
            image_miou(branch[image_idx]["pred"], label)
            for branch in outputs_by_branch
        ]
        choice = int(np.nanargmax(scores))
        choices.append(choice)
        preds.append(outputs_by_branch[choice][image_idx]["pred"])
    return preds, choices


def pixel_oracle_predictions(outputs_by_branch, labels):
    preds = []
    for image_idx, label in enumerate(labels):
        base = outputs_by_branch[0][image_idx]["pred"].copy()
        valid = valid_label_mask(label)
        correct_stack = np.stack(
            [(branch[image_idx]["pred"] == label) & valid for branch in outputs_by_branch],
            axis=0,
        )
        any_correct = correct_stack.any(axis=0)
        repaired = valid & (base != label) & any_correct
        base[repaired] = label[repaired].astype(np.uint8)
        preds.append(base)
    return preds


def base_pred_class_best_predictions(outputs_by_branch, labels):
    branch_meters = []
    for branch in outputs_by_branch:
        meter = ConfusionMeter(num_classes=len(CLASSES))
        for output, label in zip(branch, labels):
            meter.update(output["pred"], label)
        branch_meters.append(meter)

    class_iou = np.stack([meter.class_iou() for meter in branch_meters], axis=0)
    class_choice = np.nanargmax(np.where(np.isnan(class_iou), -1.0, class_iou), axis=0)
    preds = []
    for image_idx, label in enumerate(labels):
        base_pred = outputs_by_branch[0][image_idx]["pred"]
        routed = base_pred.copy()
        for class_id, branch_idx in enumerate(class_choice):
            mask = base_pred == class_id
            if mask.any():
                routed[mask] = outputs_by_branch[int(branch_idx)][image_idx]["pred"][mask]
        preds.append(routed)
    return preds, class_choice.tolist()


def analyze_dataset(args, dataset_name, scale_specs):
    records = DatasetCatalog.get(dataset_name)
    if args.limit is not None:
        records = records[: args.limit]
    labels = [load_label(record) for record in records]

    outputs_by_branch = [
        collect_branch_outputs(args, spec, records, labels)
        for spec in scale_specs
    ]
    branch_names = [spec["name"] for spec in scale_specs]
    base_preds = [output["pred"] for output in outputs_by_branch[0]]

    methods = OrderedDict()
    for name, branch in zip(branch_names, outputs_by_branch):
        methods[name] = evaluate_method([output["pred"] for output in branch], labels, base_preds)

    conf_preds = choose_by_stack(outputs_by_branch, "conf", lambda values: np.argmax(values, axis=0))
    methods["choose_highest_conf"] = evaluate_method(conf_preds, labels, base_preds)

    margin_preds = choose_by_stack(outputs_by_branch, "margin", lambda values: np.argmax(values, axis=0))
    methods["choose_highest_margin"] = evaluate_method(margin_preds, labels, base_preds)

    seq_preds, seq_choice = sequence_best_predictions(outputs_by_branch, labels, records)
    methods["sequence_oracle"] = evaluate_method(seq_preds, labels, base_preds)

    image_preds, image_choices = image_oracle_predictions(outputs_by_branch, labels)
    methods["image_oracle"] = evaluate_method(image_preds, labels, base_preds)

    pixel_preds = pixel_oracle_predictions(outputs_by_branch, labels)
    methods["pixel_oracle_any_scale_correct"] = evaluate_method(pixel_preds, labels, base_preds)

    class_preds, class_choice = base_pred_class_best_predictions(outputs_by_branch, labels)
    methods["base_pred_class_best_scale"] = evaluate_method(class_preds, labels, base_preds)

    return {
        "dataset": dataset_name,
        "sample_count": len(records),
        "branch_names": branch_names,
        "scale_specs": scale_specs,
        "methods": methods,
        "top_by_mIoU": [
            {"method": name, **values}
            for name, values in sorted(methods.items(), key=lambda item: item[1]["mIoU"], reverse=True)
        ],
        "sequence_choice": {seq: branch_names[idx] for seq, idx in sorted(seq_choice.items())},
        "base_pred_class_choice": {
            CLASSES[idx]: branch_names[branch_idx]
            for idx, branch_idx in enumerate(class_choice)
        },
        "image_choice_counts": {
            branch_names[idx]: int(sum(1 for choice in image_choices if choice == idx))
            for idx in range(len(branch_names))
        },
    }


def write_markdown(output, out_path):
    md_path = out_path.with_suffix(".md")
    lines = [
        "# Scale Ensemble Routing Diagnostic",
        "",
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
                "| Method | mIoU | mAcc | aAcc | Changed vs base | Repair rate | Net repaired |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in dataset["top_by_mIoU"][:12]:
            lines.append(
                f"| `{row['method']}` | {row['mIoU']:.4f} | {row['mAcc']:.4f} | "
                f"{row['aAcc']:.4f} | {100.0 * row['changed_rate']:.4f}% | "
                f"{100.0 * row['repair_rate']:.2f}% | {row['net_repaired']} |"
            )
        lines.extend(["", "Sequence oracle choices:", ""])
        for seq, name in dataset["sequence_choice"].items():
            lines.append(f"- `{seq}` -> `{name}`")
        lines.extend(["", "Base-pred-class routing choices:", ""])
        for cls_name, name in dataset["base_pred_class_choice"].items():
            lines.append(f"- `{cls_name}` -> `{name}`")
        lines.append("")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def main():
    args = parse_args()
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    register_cosec()

    scale_specs = parse_scale_specs(args.scale_specs)
    output = {
        "args": vars(args),
        "datasets": [
            analyze_dataset(args, dataset_name, scale_specs)
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
        for row in dataset["top_by_mIoU"][:8]:
            print(
                f"  {row['method']}: mIoU={row['mIoU']:.4f}, "
                f"changed={100.0 * row['changed_rate']:.3f}%, "
                f"repair={100.0 * row['repair_rate']:.2f}%"
            )


if __name__ == "__main__":
    main()
