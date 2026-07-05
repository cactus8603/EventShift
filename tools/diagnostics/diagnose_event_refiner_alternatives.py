#!/usr/bin/env python
"""Proxy sanity checks for event modules that replace the current event gate."""

import argparse
import copy
import json
import os
import random
import sys
import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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


DEFAULT_CONFIG = "configs/Mask2Former_SwinL_CoSEC_EventEdgeGuide_Exp5e_Night50HybridPreserveDecoder.yaml"
MODULE_ALIASES = {
    "rgb_only": "rgb_only_uncertainty_refiner",
    "boundary": "boundary_logit_refiner",
    "cross": "event_rgb_cross_attention_adapter",
    "bias": "boundary_aware_decoder_bias",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--log-period", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--event-quantile", type=float, default=0.75)
    parser.add_argument("--uncertain-percentile", type=float, default=20.0)
    parser.add_argument("--delta-l2-weight", type=float, default=1e-4)
    parser.add_argument("--loss-type", choices=["pixel_ce", "class_balanced_ce"], default="pixel_ce")
    parser.add_argument("--include-event-support", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--modules",
        default="rgb_only,boundary,cross,bias",
        help="Comma-separated: rgb_only,boundary,cross,bias or full module names.",
    )
    parser.add_argument("--use-event-base", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    return parser.parse_args()


def setup_cfg(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def eval_inputs_for_mapped_batch(fixed_batch, use_event_base):
    eval_batch = []
    for item in fixed_batch:
        cloned = dict(item)
        cloned.pop("height", None)
        cloned.pop("width", None)
        if not use_event_base:
            for key in ["event", "event_edge", "event_stats"]:
                if key in cloned:
                    cloned[key] = torch.zeros_like(cloned[key])
        eval_batch.append(cloned)
    return eval_batch


def resize_scores_to_label(scores, label_shape):
    if tuple(scores.shape[-2:]) == tuple(label_shape):
        return scores
    return F.interpolate(
        scores.unsqueeze(0),
        size=tuple(label_shape),
        mode="bilinear",
        align_corners=False,
    )[0]


def scores_to_log_probs(scores):
    probs = scores.float().clamp_min(1e-8)
    probs = probs / probs.sum(dim=0, keepdim=True).clamp_min(1e-8)
    return probs.clamp_min(1e-8).log()


def stack_padded(tensors, value=0.0):
    max_h = max(tensor.shape[-2] for tensor in tensors)
    max_w = max(tensor.shape[-1] for tensor in tensors)
    padded = []
    for tensor in tensors:
        pad_h = max_h - tensor.shape[-2]
        pad_w = max_w - tensor.shape[-1]
        padded.append(F.pad(tensor, [0, pad_w, 0, pad_h], value=value))
    return torch.stack(padded, dim=0)


def summarize_segmentation(predictions, labels, num_classes, ignore_label=255, reference_predictions=None):
    intersection = np.zeros(num_classes, dtype=np.float64)
    pred_area = np.zeros(num_classes, dtype=np.float64)
    label_area = np.zeros(num_classes, dtype=np.float64)
    valid_pixels = 0
    correct_pixels = 0
    repaired_pixels = 0
    damaged_pixels = 0
    changed_pixels = 0
    reference_wrong_pixels = 0
    reference_correct_pixels = 0

    for idx, (pred, label) in enumerate(zip(predictions, labels)):
        valid = (label != ignore_label) & (label >= 0) & (label < num_classes)
        if not np.any(valid):
            continue
        pred_valid = pred[valid]
        label_valid = label[valid]
        valid_pixels += int(valid.sum())
        correct_pixels += int((pred_valid == label_valid).sum())
        for class_id in range(num_classes):
            pred_mask = pred_valid == class_id
            label_mask = label_valid == class_id
            intersection[class_id] += float((pred_mask & label_mask).sum())
            pred_area[class_id] += float(pred_mask.sum())
            label_area[class_id] += float(label_mask.sum())
        if reference_predictions is not None:
            reference = reference_predictions[idx][valid]
            reference_correct = reference == label_valid
            current_correct = pred_valid == label_valid
            repaired_pixels += int((~reference_correct & current_correct).sum())
            damaged_pixels += int((reference_correct & ~current_correct).sum())
            changed_pixels += int((reference != pred_valid).sum())
            reference_wrong_pixels += int((~reference_correct).sum())
            reference_correct_pixels += int(reference_correct.sum())

    union = pred_area + label_area - intersection
    iou = np.full(num_classes, np.nan, dtype=np.float64)
    acc = np.full(num_classes, np.nan, dtype=np.float64)
    np.divide(intersection, union, out=iou, where=union > 0)
    np.divide(intersection, label_area, out=acc, where=label_area > 0)

    metrics = {
        "batch_mIoU": float(np.nanmean(iou) * 100.0),
        "batch_mAcc": float(np.nanmean(acc) * 100.0),
        "batch_aAcc": float((correct_pixels / valid_pixels) * 100.0) if valid_pixels else 0.0,
        "batch_valid_pixels": int(valid_pixels),
    }
    if reference_predictions is not None:
        metrics.update(
            {
                "batch_repaired_pixels": int(repaired_pixels),
                "batch_damaged_pixels": int(damaged_pixels),
                "batch_net_repaired_pixels": int(repaired_pixels - damaged_pixels),
                "batch_changed_pixels": int(changed_pixels),
                "batch_repair_rate": (
                    float(repaired_pixels / reference_wrong_pixels) if reference_wrong_pixels else 0.0
                ),
                "batch_damage_rate": (
                    float(damaged_pixels / reference_correct_pixels) if reference_correct_pixels else 0.0
                ),
            }
        )
    return metrics


def evaluate_logits(logits, labels, reference_predictions=None):
    predictions = logits.argmax(dim=1).detach().cpu().numpy().astype(np.int64, copy=False)
    label_np = labels.detach().cpu().numpy().astype(np.int64, copy=False)
    return summarize_segmentation(
        list(predictions),
        list(label_np),
        len(CLASSES),
        reference_predictions=reference_predictions,
    )


@torch.no_grad()
def build_fixed_tensors(model, fixed_batch, use_event_base):
    was_training = model.training
    model.eval()
    outputs = model(eval_inputs_for_mapped_batch(fixed_batch, use_event_base))
    base_logits = []
    labels = []
    event_stats = []
    event_edges = []
    boundaries = []
    for output, mapped in zip(outputs, fixed_batch):
        label = mapped["sem_seg"].long()
        scores = resize_scores_to_label(output["sem_seg"].detach().cpu(), label.shape[-2:])
        base_logits.append(scores_to_log_probs(scores))
        labels.append(label)
        event_stats.append(mapped["event_stats"].float())
        event_edges.append(mapped["event_edge"].float())
        boundary_keys = sorted(key for key in mapped.keys() if key.startswith("boundary_r"))
        if boundary_keys:
            boundaries.append(mapped[boundary_keys[0]].float())
        else:
            boundaries.append(torch.zeros_like(label, dtype=torch.float32))
    if was_training:
        model.train()
    return {
        "base_logits": stack_padded(base_logits, value=0.0),
        "labels": stack_padded(labels, value=255),
        "event_stats": stack_padded(event_stats, value=0.0),
        "event_edges": stack_padded(event_edges, value=0.0),
        "target_boundary": stack_padded(boundaries, value=0.0).unsqueeze(1),
    }


def normalize_per_image(signal):
    flat = signal.flatten(1)
    min_value = flat.min(dim=1).values[:, None, None, None]
    max_value = flat.max(dim=1).values[:, None, None, None]
    return (signal - min_value) / (max_value - min_value).clamp_min(1e-6)


def quantile_threshold(signal, valid, quantile):
    thresholds = []
    for sig, keep in zip(signal[:, 0], valid[:, 0]):
        values = sig[keep & (sig > 0)]
        if values.numel() == 0:
            thresholds.append(sig.new_tensor(1.0))
        else:
            thresholds.append(torch.quantile(values.float(), quantile))
    return torch.stack(thresholds).view(-1, 1, 1, 1)


def percentile_threshold(signal, valid, percentile):
    thresholds = []
    q = max(0.0, min(1.0, percentile / 100.0))
    for sig, keep in zip(signal[:, 0], valid[:, 0]):
        values = sig[keep]
        if values.numel() == 0:
            thresholds.append(sig.new_tensor(0.0))
        else:
            thresholds.append(torch.quantile(values.float(), q))
    return torch.stack(thresholds).view(-1, 1, 1, 1)


def prepare_features(tensors, event_quantile, uncertain_percentile, include_event_support, device):
    base_logits = tensors["base_logits"].to(device)
    labels = tensors["labels"].to(device)
    event_stats = tensors["event_stats"].to(device)
    event_edges = tensors["event_edges"].to(device)
    target_boundary = tensors["target_boundary"].to(device)
    valid = ((labels != 255) & (labels >= 0) & (labels < len(CLASSES))).unsqueeze(1)

    base_probs = base_logits.exp().clamp_min(1e-8)
    top2 = base_probs.topk(k=2, dim=1).values
    confidence = top2[:, 0:1]
    margin = top2[:, 0:1] - top2[:, 1:2]
    entropy = -(base_probs * base_probs.clamp_min(1e-8).log()).sum(dim=1, keepdim=True)
    entropy = entropy / np.log(float(len(CLASSES)))

    event_edge_strength = event_edges.abs().amax(dim=1, keepdim=True) if event_edges.shape[1] > 0 else torch.zeros_like(confidence)
    event_density = event_stats[:, 0:1]
    event_support = event_stats[:, 3:4]
    event_signal = normalize_per_image(torch.maximum(event_density, event_edge_strength))
    event_threshold = quantile_threshold(event_signal, valid, event_quantile)
    event_mask = (event_signal >= event_threshold) & (event_signal > 0)
    uncertain_threshold = percentile_threshold(margin, valid, uncertain_percentile)
    uncertain_mask = margin <= uncertain_threshold
    event_or_uncertain = event_mask | uncertain_mask
    if include_event_support:
        event_or_uncertain = event_or_uncertain | (event_support > 0.5)
    event_or_uncertain = event_or_uncertain & valid
    uncertainty_only = uncertain_mask & valid

    rgb_aux = torch.cat([confidence, margin, entropy], dim=1)
    event_aux = torch.cat([event_stats, event_edges, event_signal, event_support], dim=1)
    return {
        "base_logits": base_logits,
        "base_probs": base_probs,
        "labels": labels,
        "valid": valid,
        "rgb_aux": rgb_aux,
        "event_aux": event_aux,
        "target_boundary": target_boundary,
        "masks": {
            "event_or_uncertain": event_or_uncertain.float(),
            "uncertainty_only": uncertainty_only.float(),
        },
    }


def mask_summary(mask, labels, base_predictions, target_boundary):
    valid = ((labels != 255) & (labels >= 0) & (labels < len(CLASSES))).unsqueeze(1)
    wrong = (base_predictions.unsqueeze(1) != labels.unsqueeze(1)) & valid
    correct = (base_predictions.unsqueeze(1) == labels.unsqueeze(1)) & valid
    mask_bool = mask.bool() & valid
    boundary = (target_boundary > 0.5) & valid
    valid_count = valid.sum().item()
    wrong_count = wrong.sum().item()
    correct_count = correct.sum().item()
    mask_count = mask_bool.sum().item()
    return {
        "mask_fraction": float(mask_count / max(valid_count, 1)),
        "base_error_recall": float((wrong & mask_bool).sum().item() / max(wrong_count, 1)),
        "base_error_density": float((wrong & mask_bool).sum().item() / max(mask_count, 1)),
        "base_correct_mask_fraction": float((correct & mask_bool).sum().item() / max(correct_count, 1)),
        "target_boundary_recall": float((boundary & mask_bool).sum().item() / max(boundary.sum().item(), 1)),
    }


class RGBOnlyUncertaintyRefiner(nn.Module):
    def __init__(self, num_classes, rgb_aux_channels, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(num_classes + rgb_aux_channels, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_classes, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, base_logits, base_probs, rgb_aux, event_aux, change_mask):
        delta = self.net(torch.cat([base_probs, rgb_aux], dim=1)) * change_mask
        return base_logits + delta, delta, {}


class BoundaryLogitRefiner(nn.Module):
    def __init__(self, num_classes, rgb_aux_channels, event_aux_channels, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(num_classes + rgb_aux_channels + event_aux_channels, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_classes, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, base_logits, base_probs, rgb_aux, event_aux, change_mask):
        features = torch.cat([base_probs, rgb_aux, event_aux], dim=1)
        delta = self.net(features) * change_mask
        return base_logits + delta, delta, {}


class EventRGBCrossAttentionAdapter(nn.Module):
    def __init__(self, num_classes, rgb_aux_channels, event_aux_channels, hidden_dim):
        super().__init__()
        self.rgb_proj = nn.Sequential(
            nn.Conv2d(num_classes + rgb_aux_channels, hidden_dim, 1),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.event_proj = nn.Sequential(
            nn.Conv2d(event_aux_channels, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.score = nn.Conv2d(hidden_dim * 3, hidden_dim, 1)
        self.out = nn.Conv2d(hidden_dim, num_classes, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, base_logits, base_probs, rgb_aux, event_aux, change_mask):
        rgb = self.rgb_proj(torch.cat([base_probs, rgb_aux], dim=1))
        event = self.event_proj(event_aux)
        attention = torch.sigmoid(self.score(torch.cat([rgb, event, rgb * event], dim=1)))
        fused = rgb + attention * event
        delta = self.out(fused) * change_mask
        return base_logits + delta, delta, {"adapter_attention_mean": attention.detach().mean()}


class BoundaryAwareDecoderBias(nn.Module):
    def __init__(self, num_classes, rgb_aux_channels, event_aux_channels, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(event_aux_channels + rgb_aux_channels, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_classes, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, base_logits, base_probs, rgb_aux, event_aux, change_mask):
        delta = self.net(torch.cat([event_aux, rgb_aux], dim=1)) * change_mask
        return base_logits + delta, delta, {}


def build_module(name, num_classes, rgb_aux_channels, event_aux_channels, hidden_dim):
    if name == "rgb_only_uncertainty_refiner":
        return RGBOnlyUncertaintyRefiner(num_classes, rgb_aux_channels, hidden_dim)
    if name == "boundary_logit_refiner":
        return BoundaryLogitRefiner(num_classes, rgb_aux_channels, event_aux_channels, hidden_dim)
    if name == "event_rgb_cross_attention_adapter":
        return EventRGBCrossAttentionAdapter(num_classes, rgb_aux_channels, event_aux_channels, hidden_dim)
    if name == "boundary_aware_decoder_bias":
        return BoundaryAwareDecoderBias(num_classes, rgb_aux_channels, event_aux_channels, hidden_dim)
    raise ValueError(f"Unknown module: {name}")


def parse_modules(text):
    names = []
    for raw_name in text.split(","):
        raw_name = raw_name.strip()
        if not raw_name:
            continue
        names.append(MODULE_ALIASES.get(raw_name, raw_name))
    return names


def masked_ce_loss(logits, labels, valid, loss_type):
    loss_map = F.cross_entropy(logits, labels.clamp(0, len(CLASSES) - 1), reduction="none")
    mask = valid[:, 0].float()
    if loss_type == "pixel_ce":
        return (loss_map * mask).sum() / mask.sum().clamp_min(1.0)

    class_losses = []
    for class_id in range(len(CLASSES)):
        class_mask = ((labels == class_id) & valid[:, 0]).float()
        if class_mask.sum() > 0:
            class_losses.append((loss_map * class_mask).sum() / class_mask.sum().clamp_min(1.0))
    if not class_losses:
        return (loss_map * mask).sum() / mask.sum().clamp_min(1.0)
    return torch.stack(class_losses).mean()


def train_proxy_module(name, module, features, args, reference_predictions):
    module.train()
    optimizer = torch.optim.AdamW(module.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if name == "rgb_only_uncertainty_refiner":
        change_mask = features["masks"]["uncertainty_only"]
    else:
        change_mask = features["masks"]["event_or_uncertain"]

    rows = []
    for iteration in range(args.max_iter):
        optimizer.zero_grad(set_to_none=True)
        refined_logits, delta, extra = module(
            features["base_logits"],
            features["base_probs"],
            features["rgb_aux"],
            features["event_aux"],
            change_mask,
        )
        ce = masked_ce_loss(refined_logits, features["labels"], features["valid"], args.loss_type)
        delta_l2 = ((delta.square() * change_mask).sum() / change_mask.sum().clamp_min(1.0))
        total_loss = ce + args.delta_l2_weight * delta_l2
        if not torch.isfinite(total_loss):
            raise FloatingPointError(f"non-finite loss for {name} at iter {iteration}: {total_loss}")
        total_loss.backward()
        optimizer.step()

        if iteration == 0 or iteration + 1 == args.max_iter or (iteration + 1) % args.log_period == 0:
            module.eval()
            with torch.no_grad():
                eval_logits, eval_delta, eval_extra = module(
                    features["base_logits"],
                    features["base_probs"],
                    features["rgb_aux"],
                    features["event_aux"],
                    change_mask,
                )
                metrics = evaluate_logits(
                    eval_logits,
                    features["labels"],
                    reference_predictions=reference_predictions,
                )
                row = {
                    "module": name,
                    "iteration": iteration + 1,
                    "total_loss": float(total_loss.detach().cpu()),
                    "ce_loss": float(ce.detach().cpu()),
                    "delta_l2": float(delta_l2.detach().cpu()),
                    "delta_abs_mean_in_mask": float(
                        ((eval_delta.abs() * change_mask).sum() / change_mask.sum().clamp_min(1.0)).detach().cpu()
                    ),
                    **{key: float(value.detach().cpu()) for key, value in {**extra, **eval_extra}.items()},
                    **metrics,
                }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            module.train()
    return rows


def main():
    args = parse_args()
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    register_cosec()
    cfg = setup_cfg(args)
    dataset_name = args.dataset or cfg.DATASETS.TRAIN[0]
    batch_size = args.batch_size or cfg.SOLVER.IMS_PER_BATCH
    records = DatasetCatalog.get(dataset_name)
    selected = records[args.start_index : args.start_index + batch_size]
    if len(selected) != batch_size:
        raise ValueError(f"requested batch_size={batch_size}, got {len(selected)} records")

    mapper = MaskFormerSemanticDatasetMapper(cfg, True)
    fixed_batch = [mapper(copy.deepcopy(record)) for record in selected]

    model = CoSECTrainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS,
        resume=False,
    )
    tensors = build_fixed_tensors(model, fixed_batch, use_event_base=args.use_event_base)
    device = next(model.parameters()).device
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    features = prepare_features(
        tensors,
        args.event_quantile,
        args.uncertain_percentile,
        args.include_event_support,
        device,
    )

    base_predictions = features["base_logits"].argmax(dim=1)
    reference_predictions = base_predictions.detach().cpu().numpy().astype(np.int64, copy=False)
    baseline = evaluate_logits(features["base_logits"], features["labels"])
    baseline_row = {
        "phase": "baseline",
        **baseline,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    module_names = parse_modules(args.modules)
    summary = {
        "config_file": args.config_file,
        "dataset": dataset_name,
        "batch_size": batch_size,
        "start_index": args.start_index,
        "max_iter": args.max_iter,
        "weights": cfg.MODEL.WEIGHTS,
        "use_event_base": args.use_event_base,
        "event_quantile": args.event_quantile,
        "uncertain_percentile": args.uncertain_percentile,
        "loss_type": args.loss_type,
        "include_event_support": args.include_event_support,
        "modules": module_names,
        "mask_summary": {
            "event_or_uncertain": mask_summary(
                features["masks"]["event_or_uncertain"],
                features["labels"],
                base_predictions,
                features["target_boundary"],
            ),
            "uncertainty_only": mask_summary(
                features["masks"]["uncertainty_only"],
                features["labels"],
                base_predictions,
                features["target_boundary"],
            ),
        },
    }
    print(json.dumps({"summary": summary}, sort_keys=True), flush=True)
    print(json.dumps(baseline_row, sort_keys=True), flush=True)

    rows = [{"summary": summary}, baseline_row]
    rgb_aux_channels = features["rgb_aux"].shape[1]
    event_aux_channels = features["event_aux"].shape[1]
    for module_name in module_names:
        torch.manual_seed(args.seed)
        module = build_module(
            module_name,
            len(CLASSES),
            rgb_aux_channels,
            event_aux_channels,
            args.hidden_dim,
        ).to(device)
        rows.extend(train_proxy_module(module_name, module, features, args, reference_predictions))
        del module
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"Wrote event refiner alternative diagnostic: {out_path}", flush=True)


if __name__ == "__main__":
    main()
