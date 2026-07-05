#!/usr/bin/env python
"""Train a spatial student that fuses frozen multi-scale Mask2Former outputs."""

import argparse
import copy
import json
import os
import random
import sys
import importlib.util
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
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
        default="s624:624:1200,s768:768:1400",
        help="Comma-separated name:min_size:max_size entries.",
    )
    parser.add_argument("--train-dataset", default="cosec_train")
    parser.add_argument("--eval-datasets", default="cosec_day_val,cosec_night_val")
    parser.add_argument("--train-limit", type=int, default=16)
    parser.add_argument("--eval-limit", type=int, default=24)
    parser.add_argument("--crop-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--balanced-loss", action="store_true")
    parser.add_argument(
        "--teacher-preserve-kl-weight",
        type=float,
        default=0.0,
        help="KL weight that keeps the fused prediction close to the non-learned scale average.",
    )
    parser.add_argument(
        "--disagreement-loss-weight",
        type=float,
        default=0.0,
        help="Extra CE weight on pixels where scale predictions disagree.",
    )
    parser.add_argument(
        "--entropy-loss-weight",
        type=float,
        default=0.0,
        help="Extra CE weight on high-entropy average predictions.",
    )
    parser.add_argument(
        "--zero-init-output",
        action="store_true",
        help="Initialize the final scale-logit layer to zero so the student starts as scale averaging.",
    )
    parser.add_argument(
        "--preserve-average-outside-allowed",
        action="store_true",
        help="Use the non-learned scale average outside an uncertainty/boundary/disagreement mask.",
    )
    parser.add_argument(
        "--allowed-require-disagreement",
        action="store_true",
        help="Only allow the student to change pixels where at least one scale disagrees with the average prediction.",
    )
    parser.add_argument(
        "--allowed-boundary-radius",
        type=int,
        default=-1,
        help="If >=0, require average-prediction semantic boundary band with this dilation radius.",
    )
    parser.add_argument(
        "--allowed-margin-quantile",
        type=float,
        default=0.0,
        help="If >0, allow pixels in the lowest margin quantile of the average prediction.",
    )
    parser.add_argument(
        "--allowed-entropy-quantile",
        type=float,
        default=0.0,
        help="If >0, allow pixels in the highest entropy quantile of the average prediction.",
    )
    parser.add_argument(
        "--outside-allowed-loss-weight",
        type=float,
        default=0.05,
        help="When preserving average outside allowed regions, downweight outside pixels in CE denominator.",
    )
    parser.add_argument("--flip", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def split_csv(text):
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_scale_specs(text):
    specs = OrderedDict()
    for item in split_csv(text):
        name, min_size, max_size = item.split(":")
        specs[name] = {"name": name, "min_size": int(min_size), "max_size": int(max_size)}
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


def prepare_records(dataset_name, limit, seed):
    records = list(DatasetCatalog.get(dataset_name))
    rng = random.Random(seed)
    rng.shuffle(records)
    if limit is not None:
        records = records[:limit]
    return records


def collect_probs(args, scale_specs, records, labels):
    per_image = [
        {
            "probs": [],
            "label": torch.from_numpy(label.astype(np.int64)),
            "record": record,
        }
        for record, label in zip(records, labels)
    ]
    for spec in scale_specs.values():
        print(
            f"[collect] {spec['name']} min={spec['min_size']} max={spec['max_size']} "
            f"records={len(records)}",
            flush=True,
        )
        cfg = setup_cfg(args, spec["min_size"], spec["max_size"])
        mapper = MaskFormerSemanticDatasetMapper(cfg, False)
        model = build_model(cfg)
        for image_idx, (record, label) in enumerate(tqdm(list(zip(records, labels)), desc=f"collect-{spec['name']}")):
            mapped = mapper(copy.deepcopy(record))
            per_image[image_idx]["probs"].append(infer_prob(model, mapped, label.shape, args.flip))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for item in per_image:
        item["probs"] = torch.stack(item["probs"], dim=0).contiguous()  # K,C,H,W
    return per_image


def make_student_features(probs):
    # probs: B,K,C,H,W
    batch, scale_count, class_count, height, width = probs.shape
    flat_prob = probs.reshape(batch, scale_count * class_count, height, width)
    top2 = torch.topk(probs, k=2, dim=2).values
    conf = top2[:, :, 0]
    margin = top2[:, :, 0] - top2[:, :, 1]
    entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=2)
    entropy = entropy / np.log(class_count)
    return torch.cat([flat_prob, conf, margin, entropy], dim=1)


class SpatialScaleStudent(nn.Module):
    def __init__(self, scale_count, class_count, hidden_channels=64, zero_init_output=False):
        super().__init__()
        in_channels = scale_count * class_count + scale_count * 3
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, scale_count, kernel_size=1),
        )
        if zero_init_output:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, probs):
        feat = make_student_features(probs)
        return self.net(feat)


def semantic_boundary_band(pred, radius):
    # pred: B,H,W integer labels. Boundary is computed in label space and then dilated.
    edge = torch.zeros_like(pred, dtype=torch.bool)
    edge[:, 1:, :] |= pred[:, 1:, :] != pred[:, :-1, :]
    edge[:, :-1, :] |= pred[:, :-1, :] != pred[:, 1:, :]
    edge[:, :, 1:] |= pred[:, :, 1:] != pred[:, :, :-1]
    edge[:, :, :-1] |= pred[:, :, :-1] != pred[:, :, 1:]
    if radius <= 0:
        return edge
    kernel = 2 * int(radius) + 1
    dilated = F.max_pool2d(edge.float().unsqueeze(1), kernel_size=kernel, stride=1, padding=radius)
    return dilated[:, 0] > 0


def quantile_mask(values, quantile, high=False):
    quantile = float(quantile)
    if quantile <= 0:
        return torch.ones_like(values, dtype=torch.bool)
    quantile = min(max(quantile, 0.0), 1.0)
    if high:
        threshold = torch.quantile(values.detach().flatten().float(), 1.0 - quantile)
        return values >= threshold
    threshold = torch.quantile(values.detach().flatten().float(), quantile)
    return values <= threshold


def allowed_region_mask(
    probs,
    require_disagreement=False,
    boundary_radius=-1,
    margin_quantile=0.0,
    entropy_quantile=0.0,
):
    # probs: B,K,C,H,W. The average probability is the protected anchor.
    batch, scale_count, class_count, height, width = probs.shape
    mask = torch.ones((batch, height, width), dtype=torch.bool, device=probs.device)
    avg_prob = probs.mean(dim=1).clamp_min(1e-8)
    anchor_pred = avg_prob.argmax(dim=1)

    if require_disagreement:
        scale_preds = probs.argmax(dim=2)
        disagree = (scale_preds != anchor_pred[:, None]).any(dim=1)
        mask &= disagree

    if int(boundary_radius) >= 0:
        mask &= semantic_boundary_band(anchor_pred, int(boundary_radius))

    uncertainty_masks = []
    if float(margin_quantile) > 0:
        top2 = torch.topk(avg_prob, k=2, dim=1).values
        margin = top2[:, 0] - top2[:, 1]
        uncertainty_masks.append(quantile_mask(margin, margin_quantile, high=False))
    if float(entropy_quantile) > 0:
        entropy = -(avg_prob * avg_prob.log()).sum(dim=1) / np.log(class_count)
        uncertainty_masks.append(quantile_mask(entropy, entropy_quantile, high=True))
    if uncertainty_masks:
        uncertainty = torch.zeros_like(mask)
        for item in uncertainty_masks:
            uncertainty |= item
        mask &= uncertainty

    return mask


def fuse_probs(probs, scale_logits, preserve_average_mask=None):
    weights = torch.softmax(scale_logits, dim=1)
    fused = (weights[:, :, None] * probs).sum(dim=1).clamp_min(1e-8)
    if preserve_average_mask is None:
        return fused
    average = probs.mean(dim=1).clamp_min(1e-8)
    return torch.where(preserve_average_mask[:, None], fused, average)


def build_allowed_mask_from_args(args, probs):
    if not args.preserve_average_outside_allowed:
        return None
    return allowed_region_mask(
        probs,
        require_disagreement=args.allowed_require_disagreement,
        boundary_radius=args.allowed_boundary_radius,
        margin_quantile=args.allowed_margin_quantile,
        entropy_quantile=args.allowed_entropy_quantile,
    )


def scale_disagreement_weight(probs, labels, disagreement_weight=0.0, entropy_weight=0.0):
    # The non-learned scale average is already strong. Emphasize only pixels
    # where scales disagree or the average prediction is uncertain.
    valid = (labels != 255).float()
    sample_weight = torch.ones_like(labels, dtype=probs.dtype)
    if disagreement_weight > 0:
        scale_preds = probs.argmax(dim=2)  # B,K,H,W
        mode = torch.mode(scale_preds, dim=1).values[:, None]
        disagree = (scale_preds != mode).float().mean(dim=1)
        sample_weight = sample_weight + float(disagreement_weight) * disagree
    if entropy_weight > 0:
        avg_prob = probs.mean(dim=1).clamp_min(1e-8)
        entropy = -(avg_prob * avg_prob.log()).sum(dim=1) / np.log(probs.shape[2])
        sample_weight = sample_weight + float(entropy_weight) * entropy
    return sample_weight * valid


def weighted_nll_loss(log_prob, labels, sample_weight, class_weight=None):
    loss = F.nll_loss(
        log_prob,
        labels,
        ignore_index=255,
        weight=class_weight,
        reduction="none",
    )
    valid = labels != 255
    denom = sample_weight[valid].sum().clamp_min(1.0)
    return (loss * sample_weight)[valid].sum() / denom


def teacher_preserve_kl(fused, teacher, labels):
    valid = labels != 255
    kl = F.kl_div(
        fused.clamp_min(1e-8).log(),
        teacher.clamp_min(1e-8),
        reduction="none",
    ).sum(dim=1)
    if not valid.any():
        return kl.sum() * 0.0
    return kl[valid].mean()


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


def class_weights_from_items(items):
    counts = np.zeros(len(CLASSES), dtype=np.float64)
    for item in items:
        label = item["label"].numpy()
        keep = (label != 255) & (label >= 0) & (label < len(CLASSES))
        counts += np.bincount(label[keep].reshape(-1), minlength=len(CLASSES))
    freq = counts / max(float(counts.sum()), 1.0)
    weights = 1.0 / np.sqrt(np.maximum(freq, 1e-5))
    weights = weights / np.nanmean(weights[counts > 0])
    weights[counts == 0] = 0.0
    return torch.from_numpy(weights.astype(np.float32))


def sample_crop(item, crop_size, rng):
    probs = item["probs"]
    label = item["label"]
    _, _, height, width = probs.shape
    crop_h = min(crop_size, height)
    crop_w = min(crop_size, width)
    if height == crop_h:
        y0 = 0
    else:
        y0 = int(rng.integers(0, height - crop_h + 1))
    if width == crop_w:
        x0 = 0
    else:
        x0 = int(rng.integers(0, width - crop_w + 1))
    return (
        probs[:, :, y0 : y0 + crop_h, x0 : x0 + crop_w],
        label[y0 : y0 + crop_h, x0 : x0 + crop_w],
    )


def evaluate_branch(items, scale_idx):
    meter = ConfusionMeter(num_classes=len(CLASSES))
    for item in items:
        pred = item["probs"][scale_idx].argmax(dim=0).to(torch.uint8).numpy()
        meter.update(pred, item["label"].numpy())
    return meter.metrics()


def evaluate_average(items):
    meter = ConfusionMeter(num_classes=len(CLASSES))
    for item in items:
        pred = item["probs"].float().mean(dim=0).argmax(dim=0).to(torch.uint8).numpy()
        meter.update(pred, item["label"].numpy())
    return meter.metrics()


def evaluate_student(student, items, device, args):
    meter = ConfusionMeter(num_classes=len(CLASSES))
    choice_pixels = np.zeros(items[0]["probs"].shape[0], dtype=np.int64)
    weight_sums = np.zeros(items[0]["probs"].shape[0], dtype=np.float64)
    changed_pixels = 0
    allowed_pixels = 0
    pixel_count = 0
    student.eval()
    with torch.no_grad():
        for item in items:
            probs = item["probs"].float().unsqueeze(0).to(device)
            scale_logits = student(probs)
            scale_weights = torch.softmax(scale_logits, dim=1)
            allowed_mask = build_allowed_mask_from_args(args, probs)
            fused = fuse_probs(probs, scale_logits, allowed_mask)
            avg_pred = probs.mean(dim=1).argmax(dim=1)
            pred = fused.argmax(dim=1)[0].to(torch.uint8).cpu().numpy()
            choices = scale_logits.argmax(dim=1)[0].to(torch.int64).cpu().numpy()
            choice_pixels += np.bincount(choices.reshape(-1), minlength=len(choice_pixels))
            weight_sums += scale_weights.sum(dim=(0, 2, 3)).detach().cpu().numpy()
            changed_pixels += int((fused.argmax(dim=1) != avg_pred).sum().detach().cpu().item())
            if allowed_mask is not None:
                allowed_pixels += int(allowed_mask.sum().detach().cpu().item())
            else:
                allowed_pixels += int(scale_weights.shape[0] * scale_weights.shape[-2] * scale_weights.shape[-1])
            pixel_count += int(scale_weights.shape[0] * scale_weights.shape[-2] * scale_weights.shape[-1])
            meter.update(pred, item["label"].numpy())
    mean_weights = (weight_sums / max(pixel_count, 1)).tolist()
    return {
        **meter.metrics(),
        "choice_pixels": choice_pixels.tolist(),
        "mean_scale_weights": mean_weights,
        "allowed_pixel_rate": float(allowed_pixels / max(pixel_count, 1)),
        "changed_vs_average_pixel_rate": float(changed_pixels / max(pixel_count, 1)),
    }


def evaluate_all(student, eval_items_by_dataset, scale_names, device, args):
    output = OrderedDict()
    for dataset_name, items in eval_items_by_dataset.items():
        methods = OrderedDict()
        for idx, scale_name in enumerate(scale_names):
            methods[scale_name] = evaluate_branch(items, idx)
        methods["average"] = evaluate_average(items)
        methods["spatial_student"] = evaluate_student(student, items, device, args)
        output[dataset_name] = {
            "sample_count": len(items),
            "methods": methods,
            "top_by_mIoU": [
                {"method": name, **values}
                for name, values in sorted(methods.items(), key=lambda pair: pair[1]["mIoU"], reverse=True)
            ],
        }
    return output


def train_student(args, scale_specs, train_items, eval_items_by_dataset):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    scale_names = list(scale_specs.keys())
    student = SpatialScaleStudent(
        scale_count=len(scale_names),
        class_count=len(CLASSES),
        hidden_channels=args.hidden_channels,
        zero_init_output=args.zero_init_output,
    ).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    class_weight = class_weights_from_items(train_items).to(device) if args.balanced_loss else None
    rng = np.random.default_rng(args.seed)
    history = []

    for step in range(1, args.steps + 1):
        student.train()
        batch_probs = []
        batch_labels = []
        for _ in range(args.batch_size):
            item = train_items[int(rng.integers(0, len(train_items)))]
            probs, label = sample_crop(item, args.crop_size, rng)
            batch_probs.append(probs)
            batch_labels.append(label)
        probs = torch.stack(batch_probs).float().to(device)
        labels = torch.stack(batch_labels).long().to(device)
        scale_logits = student(probs)
        allowed_mask = build_allowed_mask_from_args(args, probs)
        fused = fuse_probs(probs, scale_logits, allowed_mask)
        sample_weight = scale_disagreement_weight(
            probs,
            labels,
            disagreement_weight=args.disagreement_loss_weight,
            entropy_weight=args.entropy_loss_weight,
        )
        if allowed_mask is not None:
            outside_weight = float(args.outside_allowed_loss_weight)
            sample_weight = torch.where(allowed_mask, sample_weight, sample_weight * outside_weight)
        loss_ce = weighted_nll_loss(fused.log(), labels, sample_weight, class_weight=class_weight)
        avg_prob = probs.mean(dim=1).clamp_min(1e-8)
        loss_kl = teacher_preserve_kl(fused, avg_prob, labels)
        loss = loss_ce + float(args.teacher_preserve_kl_weight) * loss_kl
        allowed_rate = 1.0 if allowed_mask is None else float(allowed_mask.float().mean().detach().cpu().item())
        changed_rate = float((fused.argmax(dim=1) != avg_prob.argmax(dim=1)).float().mean().detach().cpu().item())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            eval_result = evaluate_all(student, eval_items_by_dataset, scale_names, device, args)
            row = {
                "step": step,
                "loss": float(loss.detach().cpu().item()),
                "loss_ce": float(loss_ce.detach().cpu().item()),
                "loss_teacher_kl": float(loss_kl.detach().cpu().item()),
                "train_allowed_pixel_rate": allowed_rate,
                "train_changed_vs_average_pixel_rate": changed_rate,
                "eval": eval_result,
            }
            history.append(row)
            summary = []
            for dataset_name, dataset in eval_result.items():
                student_miou = dataset["methods"]["spatial_student"]["mIoU"]
                best = dataset["top_by_mIoU"][0]
                summary.append(
                    f"{dataset_name}: student={student_miou:.4f}, best={best['method']}:{best['mIoU']:.4f}"
                )
            print(f"[student] step {step}: loss={row['loss']:.5f}; " + " | ".join(summary), flush=True)
    return student, history


def write_markdown(output, out_dir):
    lines = [
        "# Spatial Scale Student Diagnostic",
        "",
        f"created_at: `{output['created_at']}`",
        f"config: `{output['args']['config_file']}`",
        f"weights: `{output['args']['weights']}`",
        f"scale_specs: `{output['args']['scale_specs']}`",
        f"flip: `{output['args']['flip']}`",
        f"balanced_loss: `{output['args']['balanced_loss']}`",
        f"preserve_average_outside_allowed: `{output['args']['preserve_average_outside_allowed']}`",
        f"allowed_require_disagreement: `{output['args']['allowed_require_disagreement']}`",
        f"allowed_boundary_radius: `{output['args']['allowed_boundary_radius']}`",
        f"allowed_margin_quantile: `{output['args']['allowed_margin_quantile']}`",
        f"allowed_entropy_quantile: `{output['args']['allowed_entropy_quantile']}`",
        "",
    ]
    for row in output["history"]:
        lines.extend(
            [
                "",
                f"## Step {row['step']}",
                "",
                f"loss: `{row['loss']:.6f}`",
                f"train_allowed_pixel_rate: `{row.get('train_allowed_pixel_rate', 1.0):.6f}`",
                f"train_changed_vs_average_pixel_rate: `{row.get('train_changed_vs_average_pixel_rate', 0.0):.6f}`",
                "",
            ]
        )
        for dataset_name, dataset in row["eval"].items():
            lines.extend(
                [
                    f"### {dataset_name}",
                    "",
                    "| Method | mIoU | mAcc | aAcc | Allowed | Changed vs avg | Choice pixels |",
                    "|---|---:|---:|---:|---:|---:|---|",
                ]
            )
            for result in dataset["top_by_mIoU"]:
                choice = result.get("choice_pixels")
                choice_text = "" if choice is None else ", ".join(
                    f"{name}:{value}" for name, value in zip(output["scale_names"], choice)
                )
                allowed = result.get("allowed_pixel_rate")
                changed = result.get("changed_vs_average_pixel_rate")
                lines.append(
                    f"| `{result['method']}` | {result['mIoU']:.4f} | "
                    f"{result['mAcc']:.4f} | {result['aAcc']:.4f} | "
                    f"{'' if allowed is None else f'{allowed:.4f}'} | "
                    f"{'' if changed is None else f'{changed:.4f}'} | {choice_text} |"
                )
                mean_weights = result.get("mean_scale_weights")
                if mean_weights is not None:
                    weight_text = ", ".join(
                        f"{name}:{value:.4f}" for name, value in zip(output["scale_names"], mean_weights)
                    )
                    lines.append(f"| `  mean weights` |  |  |  |  |  | {weight_text} |")
            lines.append("")
    md_path = Path(out_dir) / "summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def main():
    args = parse_args()
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    register_cosec()

    scale_specs = parse_scale_specs(args.scale_specs)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_records = prepare_records(args.train_dataset, args.train_limit, args.seed)
    train_labels = [load_label(record) for record in train_records]
    train_items = collect_probs(args, scale_specs, train_records, train_labels)

    eval_items_by_dataset = OrderedDict()
    for idx, dataset_name in enumerate(split_csv(args.eval_datasets)):
        records = list(DatasetCatalog.get(dataset_name))
        if args.eval_limit is not None:
            records = records[: args.eval_limit]
        labels = [load_label(record) for record in records]
        eval_items_by_dataset[dataset_name] = collect_probs(args, scale_specs, records, labels)

    student, history = train_student(args, scale_specs, train_items, eval_items_by_dataset)
    ckpt_path = out_dir / "scale_student.pth"
    torch.save(
        {
            "model": student.state_dict(),
            "args": vars(args),
            "scale_specs": list(scale_specs.values()),
            "classes": list(CLASSES),
        },
        ckpt_path,
    )
    output = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "scale_names": list(scale_specs.keys()),
        "checkpoint": str(ckpt_path),
        "history": history,
    }
    json_path = out_dir / "metrics.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, sort_keys=True)
        f.write("\n")
    md_path = write_markdown(output, out_dir)
    print(f"Wrote checkpoint: {ckpt_path}")
    print(f"Wrote metrics: {json_path}")
    print(f"Wrote summary: {md_path}")


if __name__ == "__main__":
    main()
