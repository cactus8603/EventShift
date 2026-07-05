#!/usr/bin/env python
"""Evaluate confidence/event routing between Swin-L checkpoints."""

import argparse
import copy
import json
import os
import sys
import importlib.util
from collections import OrderedDict
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
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--base-weights", required=True)
    parser.add_argument("--candidate-configs", required=True)
    parser.add_argument("--candidate-weights", required=True)
    parser.add_argument("--candidate-names", default="")
    parser.add_argument("--event-config", required=True)
    parser.add_argument("--dataset", default="cosec_day_val_event")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def split_csv(text):
    return [part.strip() for part in text.split(",") if part.strip()]


def setup_cfg(config_file, weights, device):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.DEVICE = device
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


def infer_mapped(model, mapped):
    with torch.no_grad():
        return model([dict(mapped)])[0]["sem_seg"].detach().cpu()


def normalize_scores(scores):
    prob = scores.float().clamp_min(1e-8)
    return prob / prob.sum(dim=0, keepdim=True).clamp_min(1e-8)


def top_conf_margin(prob):
    top2 = torch.topk(prob, k=2, dim=0).values
    return top2[0].numpy(), (top2[0] - top2[1]).numpy()


def load_label(record):
    label = cv2.imread(record["sem_seg_file_name"], cv2.IMREAD_UNCHANGED)
    if label is None:
        raise RuntimeError(f"Could not read label: {record['sem_seg_file_name']}")
    if label.ndim == 3:
        label = label[:, :, 0]
    return label.astype(np.int64)


def resize_stat(event_stats, channel, shape):
    stat = event_stats[channel : channel + 1].float().unsqueeze(0)
    stat = F.interpolate(stat, size=shape, mode="nearest")[0, 0]
    return stat.cpu().numpy()


def valid_label_mask(label, ignore_label=255):
    return (label != ignore_label) & (label >= 0) & (label < len(CLASSES))


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
        }


def empty_counts():
    return {
        "valid_pixels": 0,
        "support_pixels": 0,
        "base_wrong": 0,
        "repaired": 0,
        "damaged": 0,
        "changed": 0,
        "changed_support": 0,
    }


def add_counts(counts, base_pred, pred, label, valid, support):
    base_wrong = valid & (base_pred != label)
    base_correct = valid & (base_pred == label)
    repaired = base_wrong & (pred == label)
    damaged = base_correct & (pred != label)
    changed = valid & (base_pred != pred)
    counts["valid_pixels"] += int(valid.sum())
    counts["support_pixels"] += int((valid & support).sum())
    counts["base_wrong"] += int(base_wrong.sum())
    counts["repaired"] += int(repaired.sum())
    counts["damaged"] += int(damaged.sum())
    counts["changed"] += int(changed.sum())
    counts["changed_support"] += int((changed & support).sum())


def div(num, den):
    return float(num / den) if den else 0.0


def finalize_counts(counts):
    return {
        **counts,
        "repair_rate": div(counts["repaired"], counts["base_wrong"]),
        "net_repaired": counts["repaired"] - counts["damaged"],
        "changed_rate": div(counts["changed"], counts["valid_pixels"]),
        "support_changed_rate": div(counts["changed_support"], counts["support_pixels"]),
    }


def update_method(methods, name, pred, base_pred, label, valid, support):
    if name not in methods:
        methods[name] = {"meter": ConfusionMeter(num_classes=len(CLASSES)), "counts": empty_counts()}
    methods[name]["meter"].update(pred, label)
    add_counts(methods[name]["counts"], base_pred, pred, label, valid, support)


def percentile_mask(margin, valid, q):
    values = margin[valid]
    if values.size == 0:
        return np.zeros_like(valid)
    threshold = np.percentile(values, float(q))
    return valid & (margin <= threshold)


def main():
    args = parse_args()
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    register_cosec()

    candidate_configs = split_csv(args.candidate_configs)
    candidate_weights = split_csv(args.candidate_weights)
    if len(candidate_configs) != len(candidate_weights):
        raise ValueError("--candidate-configs and --candidate-weights must have the same length")
    candidate_names = split_csv(args.candidate_names)
    if not candidate_names:
        candidate_names = [f"cand{idx}" for idx in range(len(candidate_configs))]
    if len(candidate_names) != len(candidate_configs):
        raise ValueError("--candidate-names length must match candidates")

    base_cfg = setup_cfg(args.base_config, args.base_weights, args.device)
    event_cfg = setup_cfg(args.event_config, args.base_weights, args.device)
    candidate_cfgs = [
        setup_cfg(config, weights, args.device)
        for config, weights in zip(candidate_configs, candidate_weights)
    ]
    mapper = MaskFormerSemanticDatasetMapper(event_cfg, False)
    base_model = build_model(base_cfg)
    candidate_models = [build_model(cfg) for cfg in candidate_cfgs]

    records = DatasetCatalog.get(args.dataset)
    if args.limit is not None:
        records = records[: args.limit]

    methods = OrderedDict()
    iterator = records if args.quiet else tqdm(records, desc=args.dataset)
    for record in iterator:
        label = load_label(record)
        mapped = mapper(copy.deepcopy(record))
        event_stats = mapped["event_stats"].float()
        valid = valid_label_mask(label)
        raw_event = valid & (resize_stat(event_stats, 0, label.shape) > 0)
        support = valid & (resize_stat(event_stats, 3, label.shape) > 0)

        base_prob = normalize_scores(infer_mapped(base_model, mapped))
        cand_probs = [normalize_scores(infer_mapped(model, mapped)) for model in candidate_models]
        probs = [base_prob] + cand_probs
        names = ["base"] + candidate_names
        preds = [prob.argmax(dim=0).numpy() for prob in probs]
        confs, margins = zip(*(top_conf_margin(prob) for prob in probs))
        base_pred = preds[0]
        base_margin = margins[0]

        update_method(methods, "base", base_pred, base_pred, label, valid, support)
        for name, pred in zip(candidate_names, preds[1:]):
            update_method(methods, name, pred, base_pred, label, valid, support)

        avg_prob = torch.stack(probs, dim=0).mean(dim=0)
        update_method(methods, "avg_all", avg_prob.argmax(dim=0).numpy(), base_pred, label, valid, support)

        conf_stack = np.stack(confs, axis=0)
        margin_stack = np.stack(margins, axis=0)
        pred_stack = np.stack(preds, axis=0)
        conf_choice = np.argmax(conf_stack, axis=0)
        margin_choice = np.argmax(margin_stack, axis=0)
        update_method(
            methods,
            "choose_highest_conf",
            np.take_along_axis(pred_stack, conf_choice[None], axis=0)[0],
            base_pred,
            label,
            valid,
            support,
        )
        update_method(
            methods,
            "choose_highest_margin",
            np.take_along_axis(pred_stack, margin_choice[None], axis=0)[0],
            base_pred,
            label,
            valid,
            support,
        )

        if cand_probs:
            cand_pred = preds[1]
            cand_conf = confs[1]
            cand_margin = margins[1]
            for region_name, region in [("raw", raw_event), ("support", support)]:
                for q in [10, 20, 30, 40]:
                    uncertain = percentile_mask(base_margin, valid, q)
                    base_mask = region & uncertain
                    for conf_delta in [-0.02, 0.0, 0.02]:
                        route = base_mask & (cand_conf >= confs[0] + conf_delta)
                        routed = np.where(route, cand_pred, base_pred)
                        update_method(
                            methods,
                            f"{candidate_names[0]}_on_{region_name}_q{q}_conf{conf_delta:+.2f}",
                            routed,
                            base_pred,
                            label,
                            valid,
                            support,
                        )
                    margin_route = base_mask & (cand_margin >= base_margin)
                    routed = np.where(margin_route, cand_pred, base_pred)
                    update_method(
                        methods,
                        f"{candidate_names[0]}_on_{region_name}_q{q}_margin_ge_base",
                        routed,
                        base_pred,
                        label,
                        valid,
                        support,
                    )

    output = {
        "args": vars(args),
        "sample_count": len(records),
        "methods": OrderedDict(
            (
                name,
                {
                    **value["meter"].metrics(),
                    **finalize_counts(value["counts"]),
                },
            )
            for name, value in methods.items()
        ),
    }
    output["top_by_mIoU"] = [
        {"method": name, **values}
        for name, values in sorted(
            output["methods"].items(),
            key=lambda item: item[1]["mIoU"],
            reverse=True,
        )[:20]
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote diagnostics: {out_path}")
    print("Top methods by mIoU:")
    for row in output["top_by_mIoU"][:10]:
        print(
            f"  {row['method']}: mIoU={row['mIoU']:.4f}, "
            f"repair={100*row['repair_rate']:.2f}%, "
            f"net={row['net_repaired']}, changed={100*row['changed_rate']:.4f}%"
        )


if __name__ == "__main__":
    main()
