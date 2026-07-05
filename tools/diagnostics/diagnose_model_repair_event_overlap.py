#!/usr/bin/env python
"""Compare two segmentation checkpoints and measure repair inside event regions."""

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
    parser.add_argument("--new-config", required=True)
    parser.add_argument("--new-weights", required=True)
    parser.add_argument(
        "--event-config",
        default=None,
        help="Config used only for event_stats mapping. Defaults to --new-config.",
    )
    parser.add_argument("--dataset", default="cosec_day_val_event")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


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


def infer(model, mapper, record):
    with torch.no_grad():
        mapped = mapper(copy.deepcopy(record))
        output = model([mapped])[0]["sem_seg"].detach().cpu()
    return output


def infer_mapped(model, mapped):
    # The model reads from the dict but should not mutate image/event tensors.
    with torch.no_grad():
        output = model([dict(mapped)])[0]["sem_seg"].detach().cpu()
    return output


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
            "IoU": {
                CLASSES[idx]: None if np.isnan(value) else float(100.0 * value)
                for idx, value in enumerate(iou)
            },
            "ACC": {
                CLASSES[idx]: None if np.isnan(value) else float(100.0 * value)
                for idx, value in enumerate(acc)
            },
        }


def empty_counts():
    return {
        "valid_pixels": 0,
        "mask_pixels": 0,
        "base_wrong": 0,
        "new_wrong": 0,
        "base_correct": 0,
        "repaired": 0,
        "damaged": 0,
        "still_wrong": 0,
        "changed": 0,
        "changed_repaired": 0,
        "changed_damaged": 0,
    }


def add_counts(counts, mask, base_pred, new_pred, label, valid):
    mask = valid & mask
    base_wrong = mask & (base_pred != label)
    new_wrong = mask & (new_pred != label)
    base_correct = mask & (base_pred == label)
    repaired = base_wrong & (new_pred == label)
    damaged = base_correct & new_wrong
    changed = mask & (base_pred != new_pred)

    counts["valid_pixels"] += int(valid.sum())
    counts["mask_pixels"] += int(mask.sum())
    counts["base_wrong"] += int(base_wrong.sum())
    counts["new_wrong"] += int(new_wrong.sum())
    counts["base_correct"] += int(base_correct.sum())
    counts["repaired"] += int(repaired.sum())
    counts["damaged"] += int(damaged.sum())
    counts["still_wrong"] += int((base_wrong & new_wrong).sum())
    counts["changed"] += int(changed.sum())
    counts["changed_repaired"] += int((changed & repaired).sum())
    counts["changed_damaged"] += int((changed & damaged).sum())


def div(num, den):
    return float(num / den) if den else 0.0


def finalize_counts(counts):
    out = dict(counts)
    out.update(
        {
            "mask_coverage": div(counts["mask_pixels"], counts["valid_pixels"]),
            "base_error_rate_in_mask": div(counts["base_wrong"], counts["mask_pixels"]),
            "new_error_rate_in_mask": div(counts["new_wrong"], counts["mask_pixels"]),
            "repair_rate_of_base_wrong": div(counts["repaired"], counts["base_wrong"]),
            "damage_rate_of_base_correct": div(counts["damaged"], counts["base_correct"]),
            "net_repaired": counts["repaired"] - counts["damaged"],
            "net_per_base_wrong": div(counts["repaired"] - counts["damaged"], counts["base_wrong"]),
            "changed_rate_in_mask": div(counts["changed"], counts["mask_pixels"]),
            "changed_repair_precision": div(counts["changed_repaired"], counts["changed"]),
            "changed_damage_rate": div(counts["changed_damaged"], counts["changed"]),
        }
    )
    return out


def class_counts_template():
    return OrderedDict((name, empty_counts()) for name in CLASSES)


def main():
    args = parse_args()
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    register_cosec()

    event_config = args.event_config or args.new_config
    base_cfg = setup_cfg(args.base_config, args.base_weights, args.device)
    new_cfg = setup_cfg(args.new_config, args.new_weights, args.device)
    event_cfg = setup_cfg(event_config, args.new_weights, args.device)

    map_mapper = MaskFormerSemanticDatasetMapper(event_cfg, False)
    base_model = build_model(base_cfg)
    new_model = build_model(new_cfg)

    global_counts = OrderedDict((name, empty_counts()) for name in ["all", "raw_event", "support"])
    class_counts = OrderedDict(
        (region, class_counts_template()) for region in ["all", "raw_event", "support"]
    )
    base_meter = ConfusionMeter(num_classes=len(CLASSES))
    new_meter = ConfusionMeter(num_classes=len(CLASSES))

    records = DatasetCatalog.get(args.dataset)
    if args.limit is not None:
        records = records[: args.limit]

    iterator = records if args.quiet else tqdm(records, desc=args.dataset)
    for record in iterator:
        label = load_label(record)
        mapped = map_mapper(copy.deepcopy(record))
        base_scores = infer_mapped(base_model, mapped)
        new_scores = infer_mapped(new_model, mapped)
        event_stats = mapped["event_stats"].float()

        base_pred = base_scores.argmax(dim=0).numpy()
        new_pred = new_scores.argmax(dim=0).numpy()
        if base_pred.shape != label.shape or new_pred.shape != label.shape:
            raise RuntimeError(
                f"Prediction/label shape mismatch for {record.get('image_id')}: "
                f"base={base_pred.shape}, new={new_pred.shape}, label={label.shape}"
            )

        valid = valid_label_mask(label)
        raw_event = valid & (resize_stat(event_stats, 0, label.shape) > 0)
        support = valid & (resize_stat(event_stats, 3, label.shape) > 0)
        masks = {
            "all": valid,
            "raw_event": raw_event,
            "support": support,
        }

        base_meter.update(base_pred, label)
        new_meter.update(new_pred, label)
        for region, mask in masks.items():
            add_counts(global_counts[region], mask, base_pred, new_pred, label, valid)
            for class_id, class_name in enumerate(CLASSES):
                class_mask = mask & (label == class_id)
                add_counts(class_counts[region][class_name], class_mask, base_pred, new_pred, label, valid)

    output = {
        "args": vars(args),
        "sample_count": len(records),
        "metrics": {
            "base": base_meter.metrics(),
            "new": new_meter.metrics(),
        },
        "regions": OrderedDict((name, finalize_counts(value)) for name, value in global_counts.items()),
        "classes": OrderedDict(
            (
                region,
                OrderedDict((name, finalize_counts(values)) for name, values in per_class.items()),
            )
            for region, per_class in class_counts.items()
        ),
    }
    output["metrics"]["delta_mIoU"] = output["metrics"]["new"]["mIoU"] - output["metrics"]["base"]["mIoU"]
    output["metrics"]["delta_mAcc"] = output["metrics"]["new"]["mAcc"] - output["metrics"]["base"]["mAcc"]
    output["metrics"]["delta_aAcc"] = output["metrics"]["new"]["aAcc"] - output["metrics"]["base"]["aAcc"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote diagnostics: {out_path}")
    print(
        "mIoU: "
        f"base={output['metrics']['base']['mIoU']:.4f}, "
        f"new={output['metrics']['new']['mIoU']:.4f}, "
        f"delta={output['metrics']['delta_mIoU']:+.4f}"
    )
    print("Repair/damage:")
    for region, values in output["regions"].items():
        print(
            f"  {region}: "
            f"coverage={100*values['mask_coverage']:.2f}%, "
            f"base_wrong={values['base_wrong']}, "
            f"repaired={values['repaired']} ({100*values['repair_rate_of_base_wrong']:.2f}%), "
            f"damaged={values['damaged']}, "
            f"net={values['net_repaired']}, "
            f"new_wrong={values['new_wrong']}"
        )


if __name__ == "__main__":
    main()
