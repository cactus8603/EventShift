#!/usr/bin/env python
"""Repeat one fixed mapped batch to test whether a config can overfit."""

import argparse
import json
import os
import random
import sys
import importlib.util
from pathlib import Path

import numpy as np
import torch
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

from detectron2.checkpoint import DetectionCheckpointer  # noqa: E402
from detectron2.config import get_cfg  # noqa: E402
from detectron2.data import DatasetCatalog  # noqa: E402
from detectron2.projects.deeplab import add_deeplab_config  # noqa: E402
from detectron2.utils.events import EventStorage  # noqa: E402
from mask2former import MaskFormerSemanticDatasetMapper, add_maskformer2_config  # noqa: E402
from train_mask2former_cosec import CoSECTrainer, register_cosec  # noqa: E402
from cosec_finetune_splits import CLASSES  # noqa: E402


TRACKED_STORAGE_KEYS = [
    "loss_group/total",
    "loss_group/mask2former",
    "loss_group/edge",
    "loss_group/preserve_safe",
    "loss_group/gate_regularize",
    "event_preserve/loss",
    "event_preserve/safe_fraction",
    "event_edge/f1",
    "event_edge/precision",
    "event_edge/recall",
    "event_edge/prob_mean",
    "event_edge_guide/res2_gate_mean",
    "event_edge_guide/res2_learned_gate_mean",
    "event_edge_guide/res2_event_score_mean",
    "event_edge_guide/res2_event_score_high_fraction",
    "event_edge_guide/res2_event_score_low_fraction",
    "event_edge_guide/res2_alpha",
    "early_event_edge_adapter/gate_mean",
    "early_event_edge_adapter/gate_max",
    "early_event_edge_adapter/learned_gate_mean",
    "early_event_edge_adapter/edge_prob_mean",
    "early_event_edge_adapter/edge_prior_mean",
    "early_event_edge_adapter/reliability_mean",
    "early_event_edge_adapter/alpha",
    "early_event_edge_adapter/target_positive_fraction",
    "early_event_edge_adapter/target_negative_fraction",
    "early_event_edge_adapter/target_positive_before_class_policy_fraction",
    "early_event_edge_adapter/class_allowed_fraction",
    "early_event_edge_adapter/class_blocked_fraction",
    "early_event_edge_adapter/event_ok_fraction",
    "early_event_edge_adapter/error_fraction",
    "early_event_edge_adapter/uncertain_fraction",
    "early_event_edge_adapter/supervision_fraction",
    "early_event_edge_adapter/gate_on_positive",
    "early_event_edge_adapter/gate_on_negative",
    "early_event_edge_adapter/learned_gate_on_positive",
    "early_event_edge_adapter/learned_gate_on_negative",
    "day_event_boundary_refiner/event_active_fraction",
    "day_event_boundary_refiner/event_active_score_mean",
    "day_event_boundary_refiner/allowed_mean",
    "day_event_boundary_refiner/allowed_loss_mask",
    "day_event_boundary_refiner/allowed_hard",
    "day_event_boundary_refiner/gate_mean",
    "day_event_boundary_refiner/gate_max",
    "day_event_boundary_refiner/gate_active_001",
    "day_event_boundary_refiner/learned_gate_mean",
    "day_event_boundary_refiner/event_score_mean",
    "day_event_boundary_refiner/event_score_high_fraction",
    "day_event_boundary_refiner/repair_positive_fraction",
    "day_event_boundary_refiner/repair_negative_fraction",
    "day_event_boundary_refiner/repair_candidate_fraction",
    "day_event_boundary_refiner/repair_target_in_topk_fraction",
    "day_event_boundary_refiner/correction_mask_fraction",
    "day_event_boundary_refiner/repair_class_weight_mean",
    "day_event_boundary_refiner/boundary_loss_mask",
    "day_event_boundary_refiner/uncertain_loss_mask",
    "day_event_boundary_refiner/allowed_ce_base",
    "day_event_boundary_refiner/allowed_ce_candidate",
    "day_event_boundary_refiner/allowed_ce_final",
    "day_event_boundary_refiner/allowed_ce_candidate_delta",
    "day_event_boundary_refiner/allowed_ce_final_delta",
    "day_event_boundary_refiner/delta_abs_mean",
    "day_event_boundary_refiner/alpha",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--log-period", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
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


def scalarize_losses(losses):
    out = {}
    for name, value in losses.items():
        out[name] = float(value.detach().cpu())
    return out


def latest_storage_scalars(storage):
    latest = storage.latest()
    out = {}
    for key in TRACKED_STORAGE_KEYS:
        if key in latest:
            out[key] = float(latest[key][0])
    return out


def eval_inputs_for_mapped_batch(fixed_batch):
    eval_batch = []
    for item in fixed_batch:
        cloned = dict(item)
        # The mapper stores original height/width before train-time resize/pad.
        # Drop them so inference returns logits aligned with the mapped label.
        cloned.pop("height", None)
        cloned.pop("width", None)
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


@torch.no_grad()
def evaluate_fixed_batch(model, fixed_batch, num_classes, reference_predictions=None):
    was_training = model.training
    model.eval()
    outputs = model(eval_inputs_for_mapped_batch(fixed_batch))
    predictions = []
    labels = []
    for output, mapped in zip(outputs, fixed_batch):
        scores = resize_scores_to_label(output["sem_seg"].detach().cpu(), mapped["sem_seg"].shape[-2:])
        predictions.append(scores.argmax(dim=0).numpy().astype(np.int64, copy=False))
        labels.append(mapped["sem_seg"].detach().cpu().numpy().astype(np.int64, copy=False))
    metrics = summarize_segmentation(
        predictions,
        labels,
        num_classes,
        reference_predictions=reference_predictions,
    )
    if was_training:
        model.train()
    return metrics, predictions


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
    fixed_batch = [mapper(record) for record in selected]

    model = CoSECTrainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS,
        resume=False,
    )
    model.train()
    optimizer = CoSECTrainer.build_optimizer(cfg, model)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "config_file": args.config_file,
        "dataset": dataset_name,
        "batch_size": batch_size,
        "start_index": args.start_index,
        "max_iter": args.max_iter,
        "weights": cfg.MODEL.WEIGHTS,
        "output_dir": cfg.OUTPUT_DIR,
    }
    print(json.dumps({"summary": summary}, sort_keys=True), flush=True)

    with out_path.open("w", encoding="utf-8") as f, EventStorage(0) as storage:
        f.write(json.dumps({"summary": summary}, sort_keys=True) + "\n")
        initial_metrics, initial_predictions = evaluate_fixed_batch(
            model,
            fixed_batch,
            num_classes=len(CLASSES),
        )
        initial_row = {
            "iteration": 0,
            "phase": "before_train",
            **initial_metrics,
        }
        f.write(json.dumps(initial_row, sort_keys=True) + "\n")
        f.flush()
        print(json.dumps(initial_row, sort_keys=True), flush=True)
        for iteration in range(args.max_iter):
            storage.iter = iteration
            optimizer.zero_grad(set_to_none=True)
            losses = model(fixed_batch)
            total_loss = sum(losses.values())
            if not torch.isfinite(total_loss):
                raise FloatingPointError(f"non-finite loss at iter {iteration}: {total_loss}")
            total_loss.backward()
            optimizer.step()

            if (
                iteration == 0
                or iteration + 1 == args.max_iter
                or (iteration + 1) % args.log_period == 0
            ):
                batch_metrics, _ = evaluate_fixed_batch(
                    model,
                    fixed_batch,
                    num_classes=len(CLASSES),
                    reference_predictions=initial_predictions,
                )
                row = {
                    "iteration": iteration + 1,
                    "phase": "after_step",
                    "total_loss": float(total_loss.detach().cpu()),
                    **scalarize_losses(losses),
                    **latest_storage_scalars(storage),
                    **batch_metrics,
                }
                f.write(json.dumps(row, sort_keys=True) + "\n")
                f.flush()
                compact = {
                    key: row[key]
                    for key in [
                        "iteration",
                        "total_loss",
                        "loss_group/mask2former",
                        "loss_group/edge",
                        "loss_group/preserve_safe",
                        "event_edge/f1",
                        "event_edge_guide/res2_gate_mean",
                        "early_event_edge_adapter/gate_mean",
                        "early_event_edge_adapter/gate_on_positive",
                        "early_event_edge_adapter/target_positive_fraction",
                        "early_event_edge_adapter/event_ok_fraction",
                        "early_event_edge_adapter/alpha",
                        "batch_mIoU",
                        "batch_aAcc",
                        "batch_net_repaired_pixels",
                    ]
                    if key in row
                }
                print(json.dumps(compact, sort_keys=True), flush=True)

    print(f"Wrote one-batch overfit log: {out_path}", flush=True)


if __name__ == "__main__":
    main()
