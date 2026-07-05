#!/usr/bin/env python
"""Learn class-wise scale/TTA routing on train, then evaluate the fixed route."""

import argparse
import copy
import hashlib
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
    parser.add_argument("--train-dataset", required=True)
    parser.add_argument("--eval-dataset", required=True)
    parser.add_argument("--train-limit", type=int, default=64)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--scale-specs",
        default="s512:512:1200,s624:624:1200,s768:768:1400,s1024:1024:1600",
        help="Comma-separated scale definitions as name:min_size:max_size.",
    )
    parser.add_argument(
        "--branches",
        default=(
            "base624=s624,"
            "highres768=s768,"
            "tta3flip=s512+s768+s1024:flip,"
            "tta4flip=s512+s624+s768+s1024:flip"
        ),
        help=(
            "Comma-separated branch specs as name=scale+scale[:flip|noflip]. "
            "The flip modifier applies to every scale in that branch."
        ),
    )
    parser.add_argument("--anchor", default="tta4flip")
    parser.add_argument("--basis", default=None, help="Predicted-class basis branch; defaults to anchor.")
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--min-routed-pixels", type=int, default=500)
    parser.add_argument(
        "--combo-ablation-size",
        type=int,
        default=2,
        help="If >1, evaluate combinations of this many selected routes removed together.",
    )
    parser.add_argument(
        "--stats-cache-dir",
        default=None,
        help="Optional directory for cached route contribution/count stats.",
    )
    parser.add_argument(
        "--reuse-stats-cache",
        action="store_true",
        help="Reuse matching cached route stats when present.",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def split_csv(text):
    return [part.strip() for part in str(text).split(",") if part.strip()]


def parse_scale_specs(text):
    specs = OrderedDict()
    for item in split_csv(text):
        name, min_size, max_size = item.split(":")
        specs[name] = {
            "name": name,
            "min_size": int(min_size),
            "max_size": int(max_size),
        }
    return specs


def parse_branch_specs(text, scale_specs):
    branches = OrderedDict()
    for item in split_csv(text):
        if "=" not in item:
            raise ValueError(f"Bad branch spec '{item}', expected name=scales[:flip|noflip]")
        name, rhs = item.split("=", 1)
        name = name.strip()
        rhs = rhs.strip()
        modifier = "noflip"
        if ":" in rhs:
            rhs, modifier = rhs.rsplit(":", 1)
            modifier = modifier.strip().lower()
        if modifier not in {"flip", "noflip"}:
            raise ValueError(f"Unknown branch modifier '{modifier}' in '{item}'")
        scale_names = [part.strip() for part in rhs.split("+") if part.strip()]
        unknown = [scale_name for scale_name in scale_names if scale_name not in scale_specs]
        if unknown:
            raise ValueError(f"Unknown scale(s) in branch '{name}': {unknown}")
        if not name or not scale_names:
            raise ValueError(f"Bad branch spec: {item}")
        branches[name] = {
            "name": name,
            "scales": scale_names,
            "flip": modifier == "flip",
        }
    return branches


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


def without_event_fields(record):
    cleaned = copy.deepcopy(record)
    for key in ["event_h5", "event_old", "event_new"]:
        cleaned.pop(key, None)
    return cleaned


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


def valid_label_mask(label, ignore_label=255):
    return (label != ignore_label) & (label >= 0) & (label < len(CLASSES))


def confusion_from_arrays(label, pred, mask):
    keep = mask & valid_label_mask(label)
    keep &= (pred >= 0) & (pred < len(CLASSES))
    indices = len(CLASSES) * label[keep].astype(np.int64) + pred[keep].astype(np.int64)
    return np.bincount(indices, minlength=len(CLASSES) ** 2).reshape(len(CLASSES), len(CLASSES))


def metrics_from_matrix(matrix):
    hist = matrix.astype(np.float64)
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
        "class_iou": {
            CLASSES[idx]: (None if np.isnan(value) else float(100.0 * value))
            for idx, value in enumerate(iou)
        },
    }


def empty_counts_matrix(branch_count):
    return {
        "valid_pixels": np.zeros((len(CLASSES), branch_count), dtype=np.int64),
        "base_wrong": np.zeros((len(CLASSES), branch_count), dtype=np.int64),
        "repaired": np.zeros((len(CLASSES), branch_count), dtype=np.int64),
        "damaged": np.zeros((len(CLASSES), branch_count), dtype=np.int64),
        "changed": np.zeros((len(CLASSES), branch_count), dtype=np.int64),
    }


def add_counts(counts, class_id, branch_idx, base_pred, pred, label, mask):
    valid = mask & valid_label_mask(label)
    base_wrong = valid & (base_pred != label)
    base_correct = valid & (base_pred == label)
    counts["valid_pixels"][class_id, branch_idx] += int(valid.sum())
    counts["base_wrong"][class_id, branch_idx] += int(base_wrong.sum())
    counts["repaired"][class_id, branch_idx] += int((base_wrong & (pred == label)).sum())
    counts["damaged"][class_id, branch_idx] += int((base_correct & (pred != label)).sum())
    counts["changed"][class_id, branch_idx] += int((valid & (base_pred != pred)).sum())


def route_matrix(contributions, branch_for_class):
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    for class_id, branch_idx in enumerate(branch_for_class):
        matrix += contributions[class_id, branch_idx]
    return matrix


def route_counts(counts, branch_for_class, branch_names):
    total = {key: 0 for key in counts}
    routed_pixels_by_branch = {name: 0 for name in branch_names}
    for class_id, branch_idx in enumerate(branch_for_class):
        branch_name = branch_names[branch_idx]
        for key, value in counts.items():
            amount = int(value[class_id, branch_idx])
            total[key] += amount
            if key == "valid_pixels":
                routed_pixels_by_branch[branch_name] += amount
    valid = total["valid_pixels"]
    base_wrong = total["base_wrong"]
    return {
        **total,
        "net_repaired": total["repaired"] - total["damaged"],
        "repair_rate": float(total["repaired"] / base_wrong) if base_wrong else 0.0,
        "changed_rate": float(total["changed"] / valid) if valid else 0.0,
        "routed_pixels_by_branch": routed_pixels_by_branch,
    }


def evaluate_route(contributions, counts, branch_for_class, branch_names):
    return {
        **metrics_from_matrix(route_matrix(contributions, branch_for_class)),
        **route_counts(counts, branch_for_class, branch_names),
    }


def branch_use_keys(branches):
    keys = OrderedDict()
    for branch in branches.values():
        for scale_name in branch["scales"]:
            keys[(scale_name, branch["flip"])] = None
    return list(keys)


def stats_cache_metadata(args, dataset_name, branches, scale_specs, limit):
    return {
        "anchor": args.anchor,
        "basis": args.basis or args.anchor,
        "branch_names": list(branches),
        "branches": branches,
        "classes": list(CLASSES),
        "config_file": str(args.config_file),
        "dataset": dataset_name,
        "limit": limit,
        "scale_specs": scale_specs,
        "weights": str(args.weights),
    }


def stats_cache_path(args, dataset_name, branches, scale_specs, limit):
    if not args.stats_cache_dir:
        return None
    metadata = stats_cache_metadata(args, dataset_name, branches, scale_specs, limit)
    text = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    limit_text = "all" if limit is None else str(limit)
    filename = f"{dataset_name}_limit{limit_text}_{args.anchor}_{args.basis}_{digest}.npz"
    return Path(args.stats_cache_dir) / filename


def load_cached_stats(path, expected_branch_names):
    data = np.load(path, allow_pickle=False)
    metadata = json.loads(data["metadata_json"].item())
    branch_names = list(metadata["branch_names"])
    if branch_names != list(expected_branch_names):
        raise ValueError(f"Cache branch mismatch for {path}: {branch_names} != {list(expected_branch_names)}")
    counts = {
        key[len("count_") :]: data[key]
        for key in data.files
        if key.startswith("count_")
    }
    missing = set(empty_counts_matrix(len(branch_names))) - set(counts)
    if missing:
        raise ValueError(f"Cache missing count arrays {sorted(missing)}: {path}")
    return {
        "dataset": metadata["dataset"],
        "sample_count": int(data["sample_count"]),
        "contributions": data["contributions"],
        "counts": counts,
    }


def save_cached_stats(path, stats, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata_json": json.dumps(metadata, sort_keys=True),
        "sample_count": np.array(stats["sample_count"], dtype=np.int64),
        "contributions": stats["contributions"],
    }
    for key, value in stats["counts"].items():
        payload[f"count_{key}"] = value
    np.savez_compressed(path, **payload)


def build_inference_contexts(args, scale_specs, use_keys):
    first_scale_name, _ = use_keys[0]
    first_spec = scale_specs[first_scale_name]
    model_cfg = setup_cfg(args, first_spec["min_size"], first_spec["max_size"])
    model = build_model(model_cfg)
    mappers = {}
    for scale_name, use_flip in use_keys:
        spec = scale_specs[scale_name]
        mapper_cfg = setup_cfg(args, spec["min_size"], spec["max_size"])
        mappers[(scale_name, use_flip)] = MaskFormerSemanticDatasetMapper(mapper_cfg, False)
    return model, mappers


def collect_scale_probs(args, scale_specs, records, labels, use_keys, model, mappers):
    scale_probs = {}
    for scale_name, use_flip in use_keys:
        spec = scale_specs[scale_name]
        if not args.quiet:
            print(
                f"[collect] {scale_name} flip={use_flip} "
                f"min={spec['min_size']} max={spec['max_size']} n={len(records)}",
                flush=True,
            )
        mapper = mappers[(scale_name, use_flip)]
        probs = []
        iterator = zip(records, labels)
        if not args.quiet:
            iterator = tqdm(list(iterator), desc=f"{scale_name}-flip{int(use_flip)}")
        for record, label in iterator:
            mapped = mapper(without_event_fields(record))
            probs.append(infer_prob(model, mapped, label.shape, use_flip))
        scale_probs[(scale_name, use_flip)] = probs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return scale_probs


def branch_predictions(scale_probs, branches, image_idx):
    preds = OrderedDict()
    for branch_name, branch in branches.items():
        avg = None
        for scale_name in branch["scales"]:
            prob = scale_probs[(scale_name, branch["flip"])][image_idx].float()
            avg = prob if avg is None else avg + prob
        avg = avg / float(len(branch["scales"]))
        preds[branch_name] = avg.argmax(dim=0).numpy().astype(np.uint8, copy=False)
    return preds


def collect_dataset_stats(args, dataset_name, branches, scale_specs, limit):
    branch_names = list(branches)
    cache_path = stats_cache_path(args, dataset_name, branches, scale_specs, limit)
    if args.reuse_stats_cache and cache_path is not None and cache_path.is_file():
        if not args.quiet:
            print(f"[cache] loading {dataset_name}: {cache_path}", flush=True)
        return load_cached_stats(cache_path, branch_names)

    records = list(DatasetCatalog.get(dataset_name))
    if limit is not None:
        records = records[:limit]
    branch_count = len(branch_names)
    basis_name = args.basis or args.anchor
    contributions = np.zeros(
        (len(CLASSES), branch_count, len(CLASSES), len(CLASSES)),
        dtype=np.int64,
    )
    counts = empty_counts_matrix(branch_count)
    use_keys = branch_use_keys(branches)
    model, mappers = build_inference_contexts(args, scale_specs, use_keys)

    iterator = range(0, len(records), max(1, args.chunk_size))
    if not args.quiet:
        iterator = tqdm(list(iterator), desc=f"chunks-{dataset_name}")
    for start in iterator:
        chunk_records = records[start : start + max(1, args.chunk_size)]
        labels = [load_label(record) for record in chunk_records]
        scale_probs = collect_scale_probs(
            args,
            scale_specs,
            chunk_records,
            labels,
            use_keys,
            model,
            mappers,
        )
        for image_idx, label in enumerate(labels):
            preds = branch_predictions(scale_probs, branches, image_idx)
            basis_pred = preds[basis_name]
            base_pred = preds[args.anchor]
            for class_id in range(len(CLASSES)):
                region = basis_pred == class_id
                if not region.any():
                    continue
                for branch_idx, branch_name in enumerate(branch_names):
                    pred = preds[branch_name]
                    contributions[class_id, branch_idx] += confusion_from_arrays(label, pred, region)
                    add_counts(counts, class_id, branch_idx, base_pred, pred, label, region)
        del scale_probs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    stats = {
        "dataset": dataset_name,
        "sample_count": len(records),
        "contributions": contributions,
        "counts": counts,
    }
    if cache_path is not None:
        metadata = stats_cache_metadata(args, dataset_name, branches, scale_specs, limit)
        metadata["sample_count"] = len(records)
        save_cached_stats(cache_path, stats, metadata)
        if not args.quiet:
            print(f"[cache] wrote {dataset_name}: {cache_path}", flush=True)
    return stats


def select_routes(train_stats, branch_names, anchor_idx, min_delta, min_routed_pixels):
    anchor_route = [anchor_idx] * len(CLASSES)
    anchor_metrics = evaluate_route(
        train_stats["contributions"],
        train_stats["counts"],
        anchor_route,
        branch_names,
    )
    anchor_miou = anchor_metrics["mIoU"]
    selected = list(anchor_route)
    class_rows = []
    for class_id, class_name in enumerate(CLASSES):
        routed_pixels = int(train_stats["counts"]["valid_pixels"][class_id, anchor_idx])
        best = {
            "class": class_name,
            "class_id": class_id,
            "branch": branch_names[anchor_idx],
            "branch_idx": anchor_idx,
            "mIoU": anchor_miou,
            "delta_vs_anchor": 0.0,
            "routed_pixels": routed_pixels,
            "selected": False,
        }
        for branch_idx, branch_name in enumerate(branch_names):
            if branch_idx == anchor_idx:
                continue
            candidate_route = list(anchor_route)
            candidate_route[class_id] = branch_idx
            metrics = evaluate_route(
                train_stats["contributions"],
                train_stats["counts"],
                candidate_route,
                branch_names,
            )
            row = {
                "class": class_name,
                "class_id": class_id,
                "branch": branch_name,
                "branch_idx": branch_idx,
                "mIoU": metrics["mIoU"],
                "delta_vs_anchor": metrics["mIoU"] - anchor_miou,
                "routed_pixels": routed_pixels,
                "selected": False,
            }
            if row["delta_vs_anchor"] > best["delta_vs_anchor"]:
                best = row
        if best["branch_idx"] != anchor_idx:
            best["selected"] = (
                best["delta_vs_anchor"] > min_delta
                and best["routed_pixels"] >= min_routed_pixels
            )
        if best["selected"]:
            selected[class_id] = best["branch_idx"]
        class_rows.append(best)
    class_rows.sort(key=lambda item: item["delta_vs_anchor"], reverse=True)
    return selected, class_rows, anchor_metrics


def selected_branch_class_counts(stats, class_id, branch_idx):
    counts = {
        key: int(value[class_id, branch_idx])
        for key, value in stats["counts"].items()
    }
    valid = counts["valid_pixels"]
    base_wrong = counts["base_wrong"]
    counts["net_repaired"] = counts["repaired"] - counts["damaged"]
    counts["repair_rate"] = float(counts["repaired"] / base_wrong) if base_wrong else 0.0
    counts["changed_rate"] = float(counts["changed"] / valid) if valid else 0.0
    return counts


def selected_route_ablations(stats, branch_names, selected_route, anchor_idx, anchor_metrics=None, full_metrics=None):
    anchor_route = [anchor_idx] * len(CLASSES)
    if anchor_metrics is None:
        anchor_metrics = evaluate_route(stats["contributions"], stats["counts"], anchor_route, branch_names)
    if full_metrics is None:
        full_metrics = evaluate_route(stats["contributions"], stats["counts"], selected_route, branch_names)

    rows = []
    for class_id, branch_idx in enumerate(selected_route):
        if branch_idx == anchor_idx:
            continue
        ablated_route = list(selected_route)
        ablated_route[class_id] = anchor_idx
        metrics = evaluate_route(stats["contributions"], stats["counts"], ablated_route, branch_names)
        branch_counts = selected_branch_class_counts(stats, class_id, branch_idx)
        rows.append(
            {
                "removed_class": CLASSES[class_id],
                "removed_class_id": class_id,
                "removed_branch": branch_names[branch_idx],
                "removed_branch_idx": branch_idx,
                "mIoU": metrics["mIoU"],
                "delta_vs_anchor": metrics["mIoU"] - anchor_metrics["mIoU"],
                "delta_vs_full": metrics["mIoU"] - full_metrics["mIoU"],
                "changed_rate": metrics["changed_rate"],
                "changed": metrics["changed"],
                "net_repaired": metrics["net_repaired"],
                "repair_rate": metrics["repair_rate"],
                "selected_branch_counts": branch_counts,
            }
        )
    rows.sort(key=lambda item: item["delta_vs_full"], reverse=True)
    return rows


def selected_route_combo_ablations(
    stats,
    branch_names,
    selected_route,
    anchor_idx,
    combo_size,
    anchor_metrics=None,
    full_metrics=None,
):
    if combo_size <= 1:
        return []
    selected_class_ids = [
        class_id
        for class_id, branch_idx in enumerate(selected_route)
        if branch_idx != anchor_idx
    ]
    if len(selected_class_ids) < combo_size:
        return []

    from itertools import combinations

    anchor_route = [anchor_idx] * len(CLASSES)
    if anchor_metrics is None:
        anchor_metrics = evaluate_route(stats["contributions"], stats["counts"], anchor_route, branch_names)
    if full_metrics is None:
        full_metrics = evaluate_route(stats["contributions"], stats["counts"], selected_route, branch_names)

    rows = []
    for class_ids in combinations(selected_class_ids, combo_size):
        ablated_route = list(selected_route)
        removed = []
        branch_changed = 0
        branch_net_repaired = 0
        for class_id in class_ids:
            branch_idx = selected_route[class_id]
            ablated_route[class_id] = anchor_idx
            branch_counts = selected_branch_class_counts(stats, class_id, branch_idx)
            branch_changed += branch_counts["changed"]
            branch_net_repaired += branch_counts["net_repaired"]
            removed.append(
                {
                    "class": CLASSES[class_id],
                    "class_id": class_id,
                    "branch": branch_names[branch_idx],
                    "branch_idx": branch_idx,
                    "selected_branch_counts": branch_counts,
                }
            )
        metrics = evaluate_route(stats["contributions"], stats["counts"], ablated_route, branch_names)
        rows.append(
            {
                "removed": removed,
                "removed_classes": [item["class"] for item in removed],
                "removed_branches": [item["branch"] for item in removed],
                "mIoU": metrics["mIoU"],
                "delta_vs_anchor": metrics["mIoU"] - anchor_metrics["mIoU"],
                "delta_vs_full": metrics["mIoU"] - full_metrics["mIoU"],
                "changed_rate": metrics["changed_rate"],
                "changed": metrics["changed"],
                "net_repaired": metrics["net_repaired"],
                "repair_rate": metrics["repair_rate"],
                "removed_branch_changed": branch_changed,
                "removed_branch_net_repaired": branch_net_repaired,
            }
        )
    rows.sort(key=lambda item: item["delta_vs_full"], reverse=True)
    return rows


def named_routes(branch_for_class, branch_names, anchor_idx):
    return OrderedDict(
        (
            (CLASSES[class_id], branch_names[branch_idx])
            for class_id, branch_idx in enumerate(branch_for_class)
            if branch_idx != anchor_idx
        )
    )


def summarize_branches(stats, branch_names):
    out = OrderedDict()
    for branch_idx, branch_name in enumerate(branch_names):
        route = [branch_idx] * len(CLASSES)
        out[branch_name] = evaluate_route(stats["contributions"], stats["counts"], route, branch_names)
    return out


def write_markdown(output, out_path):
    md_path = out_path.with_suffix(".md")
    lines = [
        "# Train-Learned Scale Branch Router",
        "",
        f"created_at: `{output['created_at']}`",
        f"config: `{output['args']['config_file']}`",
        f"weights: `{output['args']['weights']}`",
        f"train: `{output['train']['dataset']}` n={output['train']['sample_count']}",
        f"eval: `{output['eval']['dataset']}` n={output['eval']['sample_count']}",
        f"anchor/basis: `{output['anchor']}` / `{output['basis']}`",
        "",
        "## Eval Result",
        "",
        "| Method | mIoU | mAcc | aAcc | Changed vs anchor | Repair rate | Net repaired |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ["anchor", "train_selected"]:
        row = output["eval"][name]
        lines.append(
            f"| `{name}` | {row['mIoU']:.4f} | {row['mAcc']:.4f} | {row['aAcc']:.4f} | "
            f"{100.0 * row['changed_rate']:.4f}% | {100.0 * row['repair_rate']:.2f}% | "
            f"{row['net_repaired']} |"
        )
    lines.extend(["", "Eval branch baselines:", ""])
    lines.extend(["| Branch | mIoU | mAcc | aAcc |", "|---|---:|---:|---:|"])
    for branch_name, row in output["eval"]["branches"].items():
        lines.append(f"| `{branch_name}` | {row['mIoU']:.4f} | {row['mAcc']:.4f} | {row['aAcc']:.4f} |")
    if output["eval"].get("selected_route_ablations"):
        lines.extend(["", "Eval selected-route ablations:", ""])
        lines.extend(
            [
                "| Removed class | Removed branch | mIoU | Delta vs full | Changed vs anchor | Net repaired | Branch changed | Branch net repaired |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in output["eval"]["selected_route_ablations"]:
            branch_counts = row["selected_branch_counts"]
            lines.append(
                f"| `{row['removed_class']}` | `{row['removed_branch']}` | {row['mIoU']:.4f} | "
                f"{row['delta_vs_full']:+.4f} | {100.0 * row['changed_rate']:.4f}% | "
                f"{row['net_repaired']} | {branch_counts['changed']} | {branch_counts['net_repaired']} |"
            )
    if output["eval"].get("selected_route_combo_ablations"):
        combo_size = output["args"].get("combo_ablation_size", 2)
        lines.extend(["", f"Eval selected-route {combo_size}-way ablations:", ""])
        lines.extend(
            [
                "| Removed classes | Removed branches | mIoU | Delta vs full | Changed vs anchor | Net repaired | Removed branch changed | Removed branch net repaired |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in output["eval"]["selected_route_combo_ablations"]:
            removed_classes = ", ".join(f"`{name}`" for name in row["removed_classes"])
            removed_branches = ", ".join(f"`{name}`" for name in row["removed_branches"])
            lines.append(
                f"| {removed_classes} | {removed_branches} | {row['mIoU']:.4f} | "
                f"{row['delta_vs_full']:+.4f} | {100.0 * row['changed_rate']:.4f}% | "
                f"{row['net_repaired']} | {row['removed_branch_changed']} | "
                f"{row['removed_branch_net_repaired']} |"
            )
    lines.extend(["", "Selected routes learned from train:", ""])
    if output["selected_routes"]:
        for class_name, branch_name in output["selected_routes"].items():
            lines.append(f"- `{class_name}` -> `{branch_name}`")
    else:
        lines.append("- none")
    lines.extend(["", "Top train class candidates:", ""])
    lines.extend(["| Class | Branch | Delta train mIoU | Routed pixels | Selected |", "|---|---|---:|---:|---|"])
    for row in output["train"]["class_candidates"][:12]:
        lines.append(
            f"| `{row['class']}` | `{row['branch']}` | {row['delta_vs_anchor']:.4f} | "
            f"{row['routed_pixels']} | {row['selected']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def jsonify_metrics(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: jsonify_metrics(subvalue) for key, subvalue in value.items()}
    if isinstance(value, list):
        return [jsonify_metrics(subvalue) for subvalue in value]
    return value


def main():
    args = parse_args()
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    register_cosec()

    scale_specs = parse_scale_specs(args.scale_specs)
    branches = parse_branch_specs(args.branches, scale_specs)
    branch_names = list(branches)
    if args.anchor not in branches:
        raise ValueError(f"Unknown anchor branch: {args.anchor}")
    args.basis = args.basis or args.anchor
    if args.basis not in branches:
        raise ValueError(f"Unknown basis branch: {args.basis}")
    anchor_idx = branch_names.index(args.anchor)

    train_stats = collect_dataset_stats(args, args.train_dataset, branches, scale_specs, args.train_limit)
    selected, class_rows, train_anchor = select_routes(
        train_stats,
        branch_names,
        anchor_idx,
        args.min_delta,
        args.min_routed_pixels,
    )
    eval_stats = collect_dataset_stats(args, args.eval_dataset, branches, scale_specs, args.eval_limit)
    anchor_route = [anchor_idx] * len(CLASSES)
    train_selected = evaluate_route(
        train_stats["contributions"],
        train_stats["counts"],
        selected,
        branch_names,
    )
    eval_anchor = evaluate_route(eval_stats["contributions"], eval_stats["counts"], anchor_route, branch_names)
    eval_selected = evaluate_route(eval_stats["contributions"], eval_stats["counts"], selected, branch_names)

    output = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "scale_specs": scale_specs,
        "branches": branches,
        "anchor": args.anchor,
        "basis": args.basis,
        "selected_routes": named_routes(selected, branch_names, anchor_idx),
        "train": {
            "dataset": train_stats["dataset"],
            "sample_count": train_stats["sample_count"],
            "anchor": train_anchor,
            "train_selected": train_selected,
            "selected_route_ablations": selected_route_ablations(
                train_stats,
                branch_names,
                selected,
                anchor_idx,
                train_anchor,
                train_selected,
            ),
            "selected_route_combo_ablations": selected_route_combo_ablations(
                train_stats,
                branch_names,
                selected,
                anchor_idx,
                args.combo_ablation_size,
                train_anchor,
                train_selected,
            ),
            "branches": summarize_branches(train_stats, branch_names),
            "class_candidates": class_rows,
        },
        "eval": {
            "dataset": eval_stats["dataset"],
            "sample_count": eval_stats["sample_count"],
            "anchor": eval_anchor,
            "train_selected": eval_selected,
            "selected_route_ablations": selected_route_ablations(
                eval_stats,
                branch_names,
                selected,
                anchor_idx,
                eval_anchor,
                eval_selected,
            ),
            "selected_route_combo_ablations": selected_route_combo_ablations(
                eval_stats,
                branch_names,
                selected,
                anchor_idx,
                args.combo_ablation_size,
                eval_anchor,
                eval_selected,
            ),
            "branches": summarize_branches(eval_stats, branch_names),
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(jsonify_metrics(output), handle, indent=2, sort_keys=True)
        handle.write("\n")
    md_path = write_markdown(jsonify_metrics(output), out_path)

    eval_anchor = output["eval"]["anchor"]["mIoU"]
    eval_selected = output["eval"]["train_selected"]["mIoU"]
    print(f"Wrote diagnostics: {out_path}")
    print(f"Wrote summary: {md_path}")
    print(f"Eval anchor {args.anchor}: mIoU={eval_anchor:.4f}")
    print(f"Eval train-selected route: mIoU={eval_selected:.4f} delta={eval_selected - eval_anchor:+.4f}")
    print("Selected routes:", dict(output["selected_routes"]))


if __name__ == "__main__":
    main()
