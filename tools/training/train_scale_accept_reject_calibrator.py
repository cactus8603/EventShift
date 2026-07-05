#!/usr/bin/env python
"""Train a conservative accept/reject calibrator for scale/TTA corrections.

The model keeps the TTA anchor everywhere by default.  It only replaces a pixel
with a candidate scale prediction when a small MLP accepts the correction inside
an explicitly allowed region.
"""

import argparse
import copy
import json
import math
import os
import random
import sys
import importlib.util
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

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

DEFAULT_CLASSES = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]


def load_classes():
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        from cosec_finetune_splits import CLASSES  # pylint: disable=import-outside-toplevel

        return list(CLASSES)
    except Exception:
        return list(DEFAULT_CLASSES)


CLASSES = load_classes()


def split_csv(text):
    return [part.strip() for part in str(text).split(",") if part.strip()]


def parse_scale_specs(text):
    specs = []
    for item in split_csv(text):
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError(f"Bad scale spec '{item}', expected name:min:max")
        name, min_size, max_size = parts
        specs.append({"name": name, "min_size": int(min_size), "max_size": int(max_size)})
    if not specs:
        raise ValueError("At least one scale spec is required.")
    return specs


def parse_branch_specs(text, scale_specs):
    scale_names = [spec["name"] for spec in scale_specs]
    branches = OrderedDict()
    for item in split_csv(text):
        if "=" not in item:
            raise ValueError(f"Bad branch spec '{item}', expected name=scale+scale")
        name, rhs = item.split("=", 1)
        name = name.strip()
        branch_scale_names = [part.strip() for part in rhs.split("+") if part.strip()]
        if not name or not branch_scale_names:
            raise ValueError(f"Bad branch spec '{item}'")
        unknown = [scale_name for scale_name in branch_scale_names if scale_name not in scale_names]
        if unknown:
            raise ValueError(f"Unknown scale(s) in branch '{name}': {unknown}")
        branches[name] = {
            "name": name,
            "scale_names": branch_scale_names,
            "scale_indices": [scale_names.index(scale_name) for scale_name in branch_scale_names],
        }
    if not branches:
        raise ValueError("At least one branch spec is required.")
    return branches


def parse_class_route(text, branch_names):
    route = {}
    for item in split_csv(text):
        if "->" not in item:
            raise ValueError(f"Bad class route '{item}', expected class->branch")
        class_token, branch_name = item.split("->", 1)
        class_id = class_id_from_token(class_token)
        branch_name = branch_name.strip()
        if branch_name not in branch_names:
            raise ValueError(f"Unknown branch '{branch_name}' in class route. Known: {branch_names}")
        route[class_id] = branch_names.index(branch_name)
    return route


def parse_thresholds(text):
    values = [float(part) for part in split_csv(text)]
    return values or [0.5]


def class_ids_from_csv(text):
    names = split_csv(text)
    if not names:
        return []
    ids = []
    for name in names:
        ids.append(class_id_from_token(name))
    return sorted(set(ids))


def class_id_from_token(token):
    token = str(token).strip()
    if token.isdigit():
        class_id = int(token)
        if class_id < 0 or class_id >= len(CLASSES):
            raise ValueError(f"Class id out of range: {token}")
        return class_id
    if token in CLASSES:
        return CLASSES.index(token)
    raise ValueError(f"Unknown class '{token}'. Known: {', '.join(CLASSES)}")


def parse_pair_ids(text):
    pairs = []
    for item in split_csv(text):
        if "->" not in item:
            raise ValueError(f"Bad pair '{item}', expected anchor->candidate")
        left, right = item.split("->", 1)
        pairs.append((class_id_from_token(left), class_id_from_token(right)))
    return sorted(set(pairs))


def pair_names(pairs):
    return [f"{CLASSES[anchor_id]}->{CLASSES[candidate_id]}" for anchor_id, candidate_id in pairs]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file")
    parser.add_argument("--weights")
    parser.add_argument(
        "--scale-specs",
        default="s512:512:1200,s624:624:1200,s768:768:1400,s1024:1024:1600",
        help="Comma-separated name:min_size:max_size entries.",
    )
    parser.add_argument("--train-dataset", default="cosec_train")
    parser.add_argument("--eval-datasets", default="cosec_day_val,cosec_night_val")
    parser.add_argument("--train-limit", type=int, default=384)
    parser.add_argument("--eval-limit", type=int, default=96)
    parser.add_argument("--pixels-per-image", type=int, default=8192)
    parser.add_argument("--batch-pixels", type=int, default=65536)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--class-embed-dim", type=int, default=8)
    parser.add_argument("--scale-embed-dim", type=int, default=4)
    parser.add_argument("--init-bias", type=float, default=-3.0)
    parser.add_argument(
        "--candidate-mode",
        choices=["highest_conf", "highest_margin", "fixed_scale", "class_route"],
        default="highest_conf",
    )
    parser.add_argument("--fixed-candidate-scale", default="s768")
    parser.add_argument(
        "--branch-specs",
        default="base624=s624,highres768=s768,tta3=s512+s768+s1024,tta4=s512+s624+s768+s1024",
        help=(
            "Comma-separated branch definitions used by --candidate-mode class_route. "
            "Example: highres768=s768,tta3=s512+s768+s1024,tta4=s512+s624+s768+s1024"
        ),
    )
    parser.add_argument(
        "--anchor-branch",
        default="tta4",
        help="Branch used as the frozen TTA anchor when --candidate-mode class_route is enabled.",
    )
    parser.add_argument(
        "--basis-branch",
        default="",
        help="Branch whose predicted class selects the class route; defaults to --anchor-branch.",
    )
    parser.add_argument(
        "--class-route",
        default="",
        help=(
            "Comma-separated class->branch route for --candidate-mode class_route. "
            "Classes not listed keep the anchor branch."
        ),
    )
    parser.add_argument(
        "--target-classes",
        default="",
        help="Comma-separated class names/ids to allow. Empty means all classes.",
    )
    parser.add_argument(
        "--target-match",
        choices=["anchor", "candidate", "either", "label"],
        default="either",
        help="Which prediction must be in target-classes. label is train/eval-GT only.",
    )
    parser.add_argument(
        "--allow-pairs",
        default="",
        help=(
            "Comma-separated anchor->candidate class pairs to allow. "
            "Empty means no pair whitelist."
        ),
    )
    parser.add_argument(
        "--deny-pairs",
        default="",
        help=(
            "Comma-separated anchor->candidate class pairs to suppress. "
            "Applied after allow-pairs and other region filters."
        ),
    )
    parser.add_argument("--lowmargin-q", type=float, default=-1.0)
    parser.add_argument("--highentropy-q", type=float, default=20.0)
    parser.add_argument(
        "--uncertainty-mode",
        choices=["none", "any", "all"],
        default="any",
        help="How to combine low-margin and high-entropy masks.",
    )
    parser.add_argument("--min-candidate-conf-delta", type=float, default=-1.0)
    parser.add_argument(
        "--use-semantic-boundary-features",
        action="store_true",
        help="Add candidate-boundary and scale-boundary-fraction scalar features.",
    )
    parser.add_argument(
        "--semantic-boundary-radius",
        type=int,
        default=3,
        help="Dilation radius for per-scale semantic boundary features.",
    )
    parser.add_argument(
        "--require-semantic-boundary",
        action="store_true",
        help="Restrict allowed pixels to candidate/scale semantic-boundary regions.",
    )
    parser.add_argument(
        "--semantic-boundary-source",
        choices=["candidate", "any_scale"],
        default="any_scale",
        help="Boundary source used when --require-semantic-boundary is enabled.",
    )
    parser.add_argument(
        "--use-event-edge-features",
        action="store_true",
        help="Add precomputed event-edge score as a scalar feature.",
    )
    parser.add_argument(
        "--require-event-edge",
        action="store_true",
        help="Restrict allowed pixels to precomputed event-edge support.",
    )
    parser.add_argument(
        "--event-edge-cache-dir",
        default="",
        help="Directory produced by build_cosec_event_edge_cache.py.",
    )
    parser.add_argument(
        "--event-edge-threshold",
        type=float,
        default=0.0,
        help="Minimum event-edge score required when --require-event-edge is enabled.",
    )
    pred_disagree = parser.add_mutually_exclusive_group()
    pred_disagree.add_argument("--require-pred-disagree", dest="require_pred_disagree", action="store_true")
    pred_disagree.add_argument("--no-require-pred-disagree", dest="require_pred_disagree", action="store_false")
    parser.set_defaults(require_pred_disagree=True)
    scale_disagree = parser.add_mutually_exclusive_group()
    scale_disagree.add_argument("--require-scale-disagree", dest="require_scale_disagree", action="store_true")
    scale_disagree.add_argument("--no-require-scale-disagree", dest="require_scale_disagree", action="store_false")
    parser.set_defaults(require_scale_disagree=False)
    parser.add_argument("--repair-weight", type=float, default=2.0)
    parser.add_argument("--damage-weight", type=float, default=1.0)
    parser.add_argument("--neutral-weight", type=float, default=0.02)
    parser.add_argument("--thresholds", default="0.3,0.5,0.7")
    parser.add_argument("--flip", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--reuse-train-matrix", action="store_true")
    parser.add_argument(
        "--train-matrix-path",
        default="",
        help="Optional path for the sampled train probability matrix. Defaults to out-dir/train_matrix.npz.",
    )
    parser.add_argument(
        "--eval-cache-dir",
        default="",
        help="Directory for full-resolution eval probability caches. Defaults to out-dir/eval_cache.",
    )
    parser.add_argument(
        "--reuse-eval-cache",
        action="store_true",
        help="Reuse existing per-scale eval .npy files when present.",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--synthetic-smoke",
        action="store_true",
        help="Run a CPU-only synthetic test without Detectron2/Mask2Former.",
    )
    return parser.parse_args()


def load_runtime():
    sys.path.insert(0, str(ROOT / "tools"))
    sys.path.insert(0, str(ROOT / "third_party" / "Mask2Former"))
    if importlib.util.find_spec("detectron2") is None:
    sys.path.insert(0, str(ROOT / "third_party" / "detectron2"))
    from detectron2.checkpoint import DetectionCheckpointer  # pylint: disable=import-outside-toplevel
    from detectron2.config import get_cfg  # pylint: disable=import-outside-toplevel
    from detectron2.data import DatasetCatalog  # pylint: disable=import-outside-toplevel
    from detectron2.projects.deeplab import add_deeplab_config  # pylint: disable=import-outside-toplevel
    from mask2former import MaskFormerSemanticDatasetMapper, add_maskformer2_config  # pylint: disable=import-outside-toplevel
    from train_mask2former_cosec import CoSECTrainer, register_cosec  # pylint: disable=import-outside-toplevel

    return SimpleNamespace(
        DetectionCheckpointer=DetectionCheckpointer,
        get_cfg=get_cfg,
        DatasetCatalog=DatasetCatalog,
        add_deeplab_config=add_deeplab_config,
        add_maskformer2_config=add_maskformer2_config,
        MaskFormerSemanticDatasetMapper=MaskFormerSemanticDatasetMapper,
        CoSECTrainer=CoSECTrainer,
        register_cosec=register_cosec,
    )


def setup_cfg(args, runtime, min_size, max_size):
    cfg = runtime.get_cfg()
    runtime.add_deeplab_config(cfg)
    runtime.add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.DEVICE = args.device
    cfg.DATASETS.TEST = ()
    cfg.TEST.AUG.ENABLED = False
    cfg.INPUT.MIN_SIZE_TEST = int(min_size)
    cfg.INPUT.MAX_SIZE_TEST = int(max_size)
    cfg.freeze()
    return cfg


def build_model(cfg, runtime):
    model = runtime.CoSECTrainer.build_model(cfg)
    runtime.DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS,
        resume=False,
    )
    model.eval()
    return model


def load_label(record):
    import cv2  # pylint: disable=import-outside-toplevel

    label = cv2.imread(record["sem_seg_file_name"], cv2.IMREAD_UNCHANGED)
    if label is None:
        raise RuntimeError(f"Could not read label: {record['sem_seg_file_name']}")
    if label.ndim == 3:
        label = label[:, :, 0]
    return label.astype(np.int64, copy=False)


def valid_label_mask(label, num_classes=None):
    count = int(num_classes or len(CLASSES))
    return (label != 255) & (label >= 0) & (label < count)


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


def infer_prob(model, mapper, record, shape, use_flip):
    mapped = mapper(copy.deepcopy(record))
    scores = infer_scores(model, mapped, use_flip)
    return normalize_scores(resize_scores(scores, shape))


def prob_top_stats(prob):
    prob = prob.astype(np.float32, copy=False)
    pred = prob.argmax(axis=-1).astype(np.int64, copy=False)
    part = np.partition(prob, kth=max(prob.shape[-1] - 2, 0), axis=-1)
    top1 = part[..., -1]
    top2 = part[..., -2] if prob.shape[-1] > 1 else np.zeros_like(top1)
    entropy = -(prob * np.log(np.clip(prob, 1e-8, 1.0))).sum(axis=-1)
    entropy /= math.log(prob.shape[-1]) if prob.shape[-1] > 1 else 1.0
    return {
        "pred": pred,
        "conf": top1.astype(np.float32, copy=False),
        "margin": (top1 - top2).astype(np.float32, copy=False),
        "entropy": entropy.astype(np.float32, copy=False),
    }


def semantic_boundary_band_np(pred, radius, valid=None):
    pred = np.asarray(pred)
    edge = np.zeros(pred.shape, dtype=bool)
    edge[1:, :] |= pred[1:, :] != pred[:-1, :]
    edge[:-1, :] |= pred[:-1, :] != pred[1:, :]
    edge[:, 1:] |= pred[:, 1:] != pred[:, :-1]
    edge[:, :-1] |= pred[:, :-1] != pred[:, 1:]
    if valid is not None:
        edge &= valid
    radius = int(radius)
    if radius <= 0:
        return edge
    import cv2  # pylint: disable=import-outside-toplevel

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    dilated = cv2.dilate(edge.astype(np.uint8), kernel) > 0
    if valid is not None:
        dilated &= valid
    return dilated


def scale_boundary_features_from_pred(scale_pred_maps, radius, valid=None):
    boundaries = []
    for pred in scale_pred_maps:
        boundaries.append(semantic_boundary_band_np(pred, radius=radius, valid=valid))
    return np.stack(boundaries, axis=0).astype(np.uint8, copy=False)


def branch_probabilities(scale_prob, scale_specs, args, scale_boundary=None):
    branches = parse_branch_specs(args.branch_specs, scale_specs)
    branch_probs = []
    branch_boundaries = []
    for branch in branches.values():
        indices = branch["scale_indices"]
        branch_probs.append(scale_prob[:, indices, :].mean(axis=1))
        if scale_boundary is not None:
            branch_boundaries.append(scale_boundary[:, indices].mean(axis=1))
    output = {"branches": branches, "prob": np.stack(branch_probs, axis=1)}
    if scale_boundary is not None:
        output["boundary"] = np.stack(branch_boundaries, axis=1)
    return output


def choose_candidate(scale_prob, scale_specs, args, scale_boundary=None, event_edge=None):
    # scale_prob: [N, K, C]
    if args.candidate_mode == "class_route":
        branch_pack = branch_probabilities(scale_prob, scale_specs, args, scale_boundary=scale_boundary)
        branch_names = list(branch_pack["branches"])
        anchor_branch = str(args.anchor_branch)
        basis_branch = str(args.basis_branch or args.anchor_branch)
        if anchor_branch not in branch_pack["branches"]:
            raise ValueError(f"Unknown anchor branch '{anchor_branch}'. Known: {branch_names}")
        if basis_branch not in branch_pack["branches"]:
            raise ValueError(f"Unknown basis branch '{basis_branch}'. Known: {branch_names}")
        anchor_idx = branch_names.index(anchor_branch)
        basis_idx = branch_names.index(basis_branch)
        branch_prob = branch_pack["prob"]
        branch_stats = prob_top_stats(branch_prob)
        branch_pred = branch_stats["pred"]
        anchor = prob_top_stats(branch_prob[:, anchor_idx, :])
        basis_pred = branch_pred[:, basis_idx]
        route_by_class = np.full(max(len(CLASSES), scale_prob.shape[-1]), anchor_idx, dtype=np.int64)
        for class_id, branch_idx in parse_class_route(args.class_route, branch_names).items():
            if class_id < len(route_by_class):
                route_by_class[class_id] = branch_idx
        scale_idx = route_by_class[np.clip(basis_pred, 0, len(route_by_class) - 1)]
        row_idx = np.arange(scale_prob.shape[0])
        candidate_prob = branch_prob[row_idx, scale_idx, :]
        candidate = prob_top_stats(candidate_prob)
        output = {
            "anchor": anchor,
            "candidate": candidate,
            "candidate_scale": scale_idx.astype(np.int64, copy=False),
            "scale_pred": branch_pred,
            "scale_disagree_count": (branch_pred != anchor["pred"][:, None]).sum(axis=1).astype(np.float32),
            "scale_disagree_frac": (branch_pred != anchor["pred"][:, None]).mean(axis=1).astype(np.float32),
        }
        if scale_boundary is not None:
            branch_boundary = branch_pack["boundary"].astype(np.float32, copy=False)
            output["candidate_boundary"] = branch_boundary[row_idx, scale_idx].astype(np.float32, copy=False)
            output["scale_boundary_frac"] = branch_boundary.mean(axis=1).astype(np.float32, copy=False)
        if event_edge is not None:
            output["event_edge_score"] = event_edge.astype(np.float32, copy=False)
        return output

    scale_stats = prob_top_stats(scale_prob)
    scale_pred = scale_stats["pred"]
    scale_conf = scale_stats["conf"]
    scale_margin = scale_stats["margin"]
    if args.candidate_mode == "highest_conf":
        scale_idx = scale_conf.argmax(axis=1).astype(np.int64, copy=False)
    elif args.candidate_mode == "highest_margin":
        scale_idx = scale_margin.argmax(axis=1).astype(np.int64, copy=False)
    else:
        names = [spec["name"] for spec in scale_specs]
        if args.fixed_candidate_scale not in names:
            raise ValueError(f"Unknown fixed candidate scale: {args.fixed_candidate_scale}")
        scale_idx = np.full(scale_prob.shape[0], names.index(args.fixed_candidate_scale), dtype=np.int64)
    row_idx = np.arange(scale_prob.shape[0])
    candidate_prob = scale_prob[row_idx, scale_idx, :]
    candidate = prob_top_stats(candidate_prob)
    anchor = prob_top_stats(scale_prob.mean(axis=1))
    output = {
        "anchor": anchor,
        "candidate": candidate,
        "candidate_scale": scale_idx,
        "scale_pred": scale_pred,
        "scale_disagree_count": (scale_pred != anchor["pred"][:, None]).sum(axis=1).astype(np.float32),
        "scale_disagree_frac": (scale_pred != anchor["pred"][:, None]).mean(axis=1).astype(np.float32),
    }
    if scale_boundary is not None:
        scale_boundary = scale_boundary.astype(np.float32, copy=False)
        output["candidate_boundary"] = scale_boundary[row_idx, scale_idx].astype(np.float32, copy=False)
        output["scale_boundary_frac"] = scale_boundary.mean(axis=1).astype(np.float32, copy=False)
    if event_edge is not None:
        output["event_edge_score"] = event_edge.astype(np.float32, copy=False)
    return output


def percentile_condition(values, valid, q, high=False):
    if q < 0:
        return None
    sampled = values[valid]
    if sampled.size == 0:
        return np.zeros_like(valid, dtype=bool)
    percentile = 100.0 - float(q) if high else float(q)
    threshold = np.percentile(sampled, percentile)
    if high:
        return valid & (values >= threshold)
    return valid & (values <= threshold)


def pair_membership(anchor_pred, candidate_pred, pairs):
    pair_mask = np.zeros(anchor_pred.shape, dtype=bool)
    for anchor_id, candidate_id in pairs:
        pair_mask |= (anchor_pred == anchor_id) & (candidate_pred == candidate_id)
    return pair_mask


def allowed_region(meta, label, args, target_class_ids, num_classes):
    anchor = meta["anchor"]
    candidate = meta["candidate"]
    valid = np.ones(anchor["pred"].shape, dtype=bool)
    if label is not None:
        valid = valid_label_mask(label, num_classes=num_classes).reshape(-1)

    mask = valid.copy()
    if args.require_pred_disagree:
        mask &= candidate["pred"] != anchor["pred"]
    if args.require_scale_disagree:
        mask &= meta["scale_disagree_count"] > 0
    if args.min_candidate_conf_delta > -0.999:
        mask &= candidate["conf"] >= anchor["conf"] + float(args.min_candidate_conf_delta)
    if args.require_semantic_boundary:
        if "candidate_boundary" not in meta:
            raise ValueError("--require-semantic-boundary requires --use-semantic-boundary-features.")
        if args.semantic_boundary_source == "candidate":
            mask &= meta["candidate_boundary"] > 0
        else:
            mask &= meta["scale_boundary_frac"] > 0
    if args.require_event_edge:
        if "event_edge_score" not in meta:
            raise ValueError("--require-event-edge requires --use-event-edge-features.")
        mask &= meta["event_edge_score"] > float(args.event_edge_threshold)

    allow_pair_ids = getattr(args, "allow_pair_ids", [])
    if allow_pair_ids:
        mask &= pair_membership(anchor["pred"], candidate["pred"], allow_pair_ids)

    if target_class_ids:
        ids = np.asarray(target_class_ids, dtype=np.int64)
        if args.target_match == "anchor":
            class_mask = np.isin(anchor["pred"], ids)
        elif args.target_match == "candidate":
            class_mask = np.isin(candidate["pred"], ids)
        elif args.target_match == "label":
            if label is None:
                raise ValueError("--target-match label requires labels.")
            class_mask = np.isin(label.reshape(-1), ids)
        else:
            class_mask = np.isin(anchor["pred"], ids) | np.isin(candidate["pred"], ids)
        mask &= class_mask

    if args.uncertainty_mode != "none":
        conditions = []
        low_margin = percentile_condition(anchor["margin"], valid, args.lowmargin_q, high=False)
        high_entropy = percentile_condition(anchor["entropy"], valid, args.highentropy_q, high=True)
        if low_margin is not None:
            conditions.append(low_margin)
        if high_entropy is not None:
            conditions.append(high_entropy)
        if conditions:
            if args.uncertainty_mode == "all":
                uncertainty = np.logical_and.reduce(conditions)
            else:
                uncertainty = np.logical_or.reduce(conditions)
            mask &= uncertainty

    deny_pair_ids = getattr(args, "deny_pair_ids", [])
    if deny_pair_ids:
        mask &= ~pair_membership(anchor["pred"], candidate["pred"], deny_pair_ids)
    return mask


def make_scalar_features(meta):
    anchor = meta["anchor"]
    candidate = meta["candidate"]
    features = [
        anchor["conf"],
        candidate["conf"],
        candidate["conf"] - anchor["conf"],
        anchor["margin"],
        candidate["margin"],
        candidate["margin"] - anchor["margin"],
        anchor["entropy"],
        candidate["entropy"],
        candidate["entropy"] - anchor["entropy"],
        meta["scale_disagree_frac"],
    ]
    if "candidate_boundary" in meta:
        features.extend([meta["candidate_boundary"], meta["scale_boundary_frac"]])
    if "event_edge_score" in meta:
        features.append(meta["event_edge_score"])
    return np.stack(features, axis=1).astype(np.float32, copy=False)


def make_examples_from_probs(scale_prob, target, scale_specs, args, target_class_ids):
    num_classes = scale_prob.shape[-1]
    scale_boundary = None
    if args.use_semantic_boundary_features:
        raise RuntimeError(
            "Internal error: semantic-boundary features require make_examples_from_matrix(), "
            "not make_examples_from_probs()."
        )
    meta = choose_candidate(scale_prob, scale_specs, args, scale_boundary=scale_boundary)
    target = target.reshape(-1).astype(np.int64, copy=False)
    allowed = allowed_region(meta, target, args, target_class_ids, num_classes=num_classes)
    anchor_pred = meta["anchor"]["pred"]
    candidate_pred = meta["candidate"]["pred"]
    valid = valid_label_mask(target, num_classes=num_classes)
    repair = valid & (anchor_pred != target) & (candidate_pred == target)
    damage = valid & (anchor_pred == target) & (candidate_pred != target)
    neutral = valid & ~(repair | damage)
    accept_target = repair.astype(np.float32)
    weight = np.zeros_like(accept_target, dtype=np.float32)
    weight[repair] = float(args.repair_weight)
    weight[damage] = float(args.damage_weight)
    weight[neutral] = float(args.neutral_weight)
    keep = allowed & (weight > 0)
    scalars = make_scalar_features(meta)
    return {
        "scalars": scalars[keep],
        "anchor_class": anchor_pred[keep].astype(np.int64, copy=False),
        "candidate_class": candidate_pred[keep].astype(np.int64, copy=False),
        "candidate_scale": meta["candidate_scale"][keep].astype(np.int64, copy=False),
        "target": accept_target[keep].astype(np.float32, copy=False),
        "weight": weight[keep].astype(np.float32, copy=False),
        "stats": {
            "samples": int(len(target)),
            "kept": int(keep.sum()),
            "allowed": int(allowed.sum()),
            "repair_positive": int((keep & repair).sum()),
            "damage_negative": int((keep & damage).sum()),
            "neutral_negative": int((keep & neutral).sum()),
        },
    }


def make_examples_from_matrix(matrix, scale_specs, args, target_class_ids):
    scale_boundary = None
    if args.use_semantic_boundary_features:
        if "boundary" not in matrix:
            raise RuntimeError(
                "Training matrix lacks semantic-boundary features. Rebuild without "
                "--reuse-train-matrix or disable --use-semantic-boundary-features."
            )
        scale_boundary = matrix["boundary"]
    num_classes = matrix["prob"].shape[-1]
    event_edge = None
    if args.use_event_edge_features:
        if "event_edge" not in matrix:
            raise RuntimeError(
                "Training matrix lacks event-edge features. Rebuild without "
                "--reuse-train-matrix or disable --use-event-edge-features."
            )
        event_edge = matrix["event_edge"]
    meta = choose_candidate(
        matrix["prob"],
        scale_specs,
        args,
        scale_boundary=scale_boundary,
        event_edge=event_edge,
    )
    target = matrix["target"].reshape(-1).astype(np.int64, copy=False)
    allowed = allowed_region(meta, target, args, target_class_ids, num_classes=num_classes)
    anchor_pred = meta["anchor"]["pred"]
    candidate_pred = meta["candidate"]["pred"]
    valid = valid_label_mask(target, num_classes=num_classes)
    repair = valid & (anchor_pred != target) & (candidate_pred == target)
    damage = valid & (anchor_pred == target) & (candidate_pred != target)
    neutral = valid & ~(repair | damage)
    accept_target = repair.astype(np.float32)
    weight = np.zeros_like(accept_target, dtype=np.float32)
    weight[repair] = float(args.repair_weight)
    weight[damage] = float(args.damage_weight)
    weight[neutral] = float(args.neutral_weight)
    keep = allowed & (weight > 0)
    scalars = make_scalar_features(meta)
    return {
        "scalars": scalars[keep],
        "anchor_class": anchor_pred[keep].astype(np.int64, copy=False),
        "candidate_class": candidate_pred[keep].astype(np.int64, copy=False),
        "candidate_scale": meta["candidate_scale"][keep].astype(np.int64, copy=False),
        "target": accept_target[keep].astype(np.float32, copy=False),
        "weight": weight[keep].astype(np.float32, copy=False),
        "stats": {
            "samples": int(len(target)),
            "kept": int(keep.sum()),
            "allowed": int(allowed.sum()),
            "repair_positive": int((keep & repair).sum()),
            "damage_negative": int((keep & damage).sum()),
            "neutral_negative": int((keep & neutral).sum()),
            "semantic_boundary_positive": int(
                (keep & (meta.get("scale_boundary_frac", np.zeros_like(keep, dtype=np.float32)) > 0)).sum()
            ),
            "event_edge_positive": int(
                (keep & (meta.get("event_edge_score", np.zeros_like(keep, dtype=np.float32)) > 0)).sum()
            ),
        },
    }


def sample_pixels(label, pixels_per_image, rng):
    valid = valid_label_mask(label)
    ys_all, xs_all = np.where(valid)
    if len(ys_all) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    per_class = max(64, pixels_per_image // max(1, len(CLASSES)))
    chosen = []
    for class_id in range(len(CLASSES)):
        ys, xs = np.where(valid & (label == class_id))
        if len(ys) == 0:
            continue
        take = min(per_class, len(ys))
        idx = rng.choice(len(ys), size=take, replace=False)
        chosen.extend(zip(ys[idx].tolist(), xs[idx].tolist()))

    remaining = max(0, pixels_per_image - len(chosen))
    if remaining:
        take = min(remaining, len(ys_all))
        idx = rng.choice(len(ys_all), size=take, replace=False)
        chosen.extend(zip(ys_all[idx].tolist(), xs_all[idx].tolist()))

    if len(chosen) > pixels_per_image:
        idx = rng.choice(len(chosen), size=pixels_per_image, replace=False)
        chosen = [chosen[int(i)] for i in idx]

    ys = np.asarray([item[0] for item in chosen], dtype=np.int64)
    xs = np.asarray([item[1] for item in chosen], dtype=np.int64)
    return ys, xs


def prepare_records(runtime, dataset_name, limit, seed):
    records = list(runtime.DatasetCatalog.get(dataset_name))
    rng = random.Random(seed)
    rng.shuffle(records)
    if limit is not None:
        records = records[:limit]
    return records


def event_edge_cache_path(cache_dir, dataset_name, record, image_idx):
    return Path(cache_dir) / dataset_name / f"{cache_record_stem(record, image_idx)}.npz"


def load_event_edge_cache(cache_dir, dataset_name, record, image_idx, shape):
    if not cache_dir:
        raise ValueError("--event-edge-cache-dir is required when event-edge features are enabled.")
    path = event_edge_cache_path(cache_dir, dataset_name, record, image_idx)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing event-edge cache: {path}. Build it with tools/build_cosec_event_edge_cache.py."
        )
    with np.load(path) as payload:
        if "score" in payload:
            score = payload["score"].astype(np.float32, copy=False)
        elif "mask" in payload:
            score = payload["mask"].astype(np.float32, copy=False)
        else:
            raise KeyError(f"Event-edge cache has no score/mask field: {path}")
    if tuple(score.shape) != tuple(shape):
        raise ValueError(f"Event-edge cache shape mismatch for {path}: {score.shape} vs {shape}")
    return score


def collect_sample_matrix(args, runtime, scale_specs, records, labels, coords):
    offsets = np.cumsum([0] + [len(item[0]) for item in coords])
    total = int(offsets[-1])
    scale_count = len(scale_specs)
    class_count = len(CLASSES)
    prob = np.zeros((total, scale_count, class_count), dtype=np.float16)
    boundary = (
        np.zeros((total, scale_count), dtype=np.uint8)
        if args.use_semantic_boundary_features
        else None
    )
    event_edge = np.zeros(total, dtype=np.float16) if args.use_event_edge_features else None
    target = np.concatenate(
        [label[ys, xs].astype(np.int64, copy=False) for label, (ys, xs) in zip(labels, coords)]
    )
    if event_edge is not None:
        for image_idx, (record, label, (ys, xs)) in enumerate(zip(records, labels, coords)):
            if len(ys) == 0:
                continue
            start, end = offsets[image_idx], offsets[image_idx + 1]
            score = load_event_edge_cache(
                args.event_edge_cache_dir,
                args.train_dataset,
                record,
                image_idx,
                label.shape,
            )
            event_edge[start:end] = score[ys, xs].astype(np.float16, copy=False)

    for scale_idx, spec in enumerate(scale_specs):
        print(
            f"[collect-train] scale {scale_idx + 1}/{len(scale_specs)} "
            f"{spec['name']} min={spec['min_size']} max={spec['max_size']} records={len(records)}",
            flush=True,
        )
        cfg = setup_cfg(args, runtime, spec["min_size"], spec["max_size"])
        mapper = runtime.MaskFormerSemanticDatasetMapper(cfg, False)
        model = build_model(cfg, runtime)
        iterator = list(zip(records, labels, coords))
        if not args.quiet:
            iterator = tqdm(iterator, desc=f"collect-train-{spec['name']}")
        for image_idx, (record, label, (ys, xs)) in enumerate(iterator):
            if len(ys) == 0:
                continue
            start, end = offsets[image_idx], offsets[image_idx + 1]
            scale_prob = infer_prob(model, mapper, record, label.shape, args.flip)
            prob[start:end, scale_idx, :] = (
                scale_prob[:, ys, xs].T.numpy().astype(np.float16, copy=False)
            )
            if boundary is not None:
                scale_pred = scale_prob.argmax(dim=0).numpy().astype(np.int64, copy=False)
                valid = valid_label_mask(label, num_classes=class_count)
                boundary_map = semantic_boundary_band_np(
                    scale_pred,
                    radius=args.semantic_boundary_radius,
                    valid=valid,
                )
                boundary[start:end, scale_idx] = boundary_map[ys, xs].astype(np.uint8, copy=False)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[collect-train] done {spec['name']}", flush=True)
    matrix = {"prob": prob, "target": target}
    if boundary is not None:
        matrix["boundary"] = boundary
    if event_edge is not None:
        matrix["event_edge"] = event_edge
    return matrix


class AcceptRejectCalibrator(nn.Module):
    def __init__(
        self,
        num_classes,
        num_scales,
        scalar_dim=10,
        class_embed_dim=8,
        scale_embed_dim=4,
        hidden_dim=64,
        init_bias=-3.0,
    ):
        super().__init__()
        self.anchor_embed = nn.Embedding(num_classes, class_embed_dim)
        self.candidate_embed = nn.Embedding(num_classes, class_embed_dim)
        self.scale_embed = nn.Embedding(num_scales, scale_embed_dim)
        in_dim = scalar_dim + 2 * class_embed_dim + scale_embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.constant_(self.mlp[-1].bias, float(init_bias))

    def forward(self, scalars, anchor_class, candidate_class, candidate_scale):
        feat = torch.cat(
            [
                scalars,
                self.anchor_embed(anchor_class.long()),
                self.candidate_embed(candidate_class.long()),
                self.scale_embed(candidate_scale.long()),
            ],
            dim=1,
        )
        return self.mlp(feat).squeeze(1)


def train_calibrator(args, examples, scale_specs, num_classes):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    num_sources = max(len(scale_specs), int(examples["candidate_scale"].max()) + 1)
    model = AcceptRejectCalibrator(
        num_classes=num_classes,
        num_scales=num_sources,
        scalar_dim=examples["scalars"].shape[1],
        class_embed_dim=args.class_embed_dim,
        scale_embed_dim=args.scale_embed_dim,
        hidden_dim=args.hidden_dim,
        init_bias=args.init_bias,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scalars = torch.from_numpy(examples["scalars"].astype(np.float32))
    anchor_class = torch.from_numpy(examples["anchor_class"].astype(np.int64))
    candidate_class = torch.from_numpy(examples["candidate_class"].astype(np.int64))
    candidate_scale = torch.from_numpy(examples["candidate_scale"].astype(np.int64))
    target = torch.from_numpy(examples["target"].astype(np.float32))
    weight = torch.from_numpy(examples["weight"].astype(np.float32))

    n = int(len(target))
    if n == 0:
        raise RuntimeError("No training examples survived the configured allowed-region filters.")

    history = []
    for epoch in range(args.epochs):
        order = torch.randperm(n)
        total_loss = 0.0
        total_weight = 0.0
        accepted = 0
        positives = 0
        for start in range(0, n, args.batch_pixels):
            idx = order[start : start + args.batch_pixels]
            batch_scalars = scalars[idx].to(device, non_blocking=True)
            batch_anchor = anchor_class[idx].to(device, non_blocking=True)
            batch_candidate = candidate_class[idx].to(device, non_blocking=True)
            batch_scale = candidate_scale[idx].to(device, non_blocking=True)
            batch_target = target[idx].to(device, non_blocking=True)
            batch_weight = weight[idx].to(device, non_blocking=True)
            logits = model(batch_scalars, batch_anchor, batch_candidate, batch_scale)
            loss_vec = F.binary_cross_entropy_with_logits(logits, batch_target, reduction="none")
            loss = (loss_vec * batch_weight).sum() / batch_weight.sum().clamp_min(1e-6)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float((loss_vec.detach() * batch_weight).sum().item())
            total_weight += float(batch_weight.sum().item())
            accepted += int((torch.sigmoid(logits.detach()) >= 0.5).sum().item())
            positives += int((batch_target >= 0.5).sum().item())

        row = {
            "epoch": epoch + 1,
            "loss": total_loss / max(total_weight, 1.0),
            "accept_rate_at_0.5": accepted / max(n, 1),
            "positive_rate": positives / max(n, 1),
        }
        history.append(row)
        print(
            f"[calibrator] epoch {row['epoch']}: loss={row['loss']:.5f}, "
            f"accept@0.5={100.0 * row['accept_rate_at_0.5']:.3f}%, "
            f"positive={100.0 * row['positive_rate']:.3f}%",
            flush=True,
        )
    return model, history


class ConfusionMeter:
    def __init__(self, num_classes=19):
        self.num_classes = int(num_classes)
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, pred, label):
        keep = valid_label_mask(label, num_classes=self.num_classes)
        keep &= (pred >= 0) & (pred < self.num_classes)
        indices = self.num_classes * label[keep].astype(np.int64) + pred[keep].astype(np.int64)
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
            "class_iou": {
                CLASSES[idx] if idx < len(CLASSES) else str(idx): (
                    None if np.isnan(value) else float(100.0 * value)
                )
                for idx, value in enumerate(iou)
            },
        }


def empty_counts():
    return {
        "valid_pixels": 0,
        "anchor_wrong": 0,
        "allowed_pixels": 0,
        "accepted_pixels": 0,
        "changed_pixels": 0,
        "repaired": 0,
        "damaged": 0,
    }


def add_counts(counts, anchor_pred, final_pred, label, allowed, accepted):
    valid = valid_label_mask(label, num_classes=len(CLASSES)).reshape(-1)
    anchor = anchor_pred.reshape(-1)
    final = final_pred.reshape(-1)
    target = label.reshape(-1)
    base_wrong = valid & (anchor != target)
    base_correct = valid & (anchor == target)
    changed = valid & (anchor != final)
    repaired = base_wrong & (final == target)
    damaged = base_correct & (final != target)
    counts["valid_pixels"] += int(valid.sum())
    counts["anchor_wrong"] += int(base_wrong.sum())
    counts["allowed_pixels"] += int((valid & allowed.reshape(-1)).sum())
    counts["accepted_pixels"] += int((valid & accepted.reshape(-1)).sum())
    counts["changed_pixels"] += int(changed.sum())
    counts["repaired"] += int(repaired.sum())
    counts["damaged"] += int(damaged.sum())


def finalize_counts(counts):
    valid = counts["valid_pixels"]
    wrong = counts["anchor_wrong"]
    return {
        **counts,
        "allowed_rate": float(counts["allowed_pixels"] / valid) if valid else 0.0,
        "accepted_rate": float(counts["accepted_pixels"] / valid) if valid else 0.0,
        "changed_rate": float(counts["changed_pixels"] / valid) if valid else 0.0,
        "repair_rate": float(counts["repaired"] / wrong) if wrong else 0.0,
        "net_repaired": int(counts["repaired"] - counts["damaged"]),
    }


def predict_accept(model, feature_pack, device, chunk_pixels=262144):
    count = len(feature_pack["scalars"])
    out = np.zeros(count, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, count, chunk_pixels):
            end = min(count, start + chunk_pixels)
            scalars = torch.from_numpy(feature_pack["scalars"][start:end]).to(device)
            anchor = torch.from_numpy(feature_pack["anchor_class"][start:end]).to(device)
            candidate = torch.from_numpy(feature_pack["candidate_class"][start:end]).to(device)
            scale = torch.from_numpy(feature_pack["candidate_scale"][start:end]).to(device)
            out[start:end] = torch.sigmoid(model(scalars, anchor, candidate, scale)).cpu().numpy()
    return out


def evaluate_prob_image(
    scale_prob,
    label,
    scale_specs,
    args,
    target_class_ids,
    model,
    thresholds,
    device,
    event_edge=None,
):
    height, width = label.shape
    flat_prob = scale_prob.transpose(2, 3, 0, 1).reshape(-1, scale_prob.shape[0], scale_prob.shape[1])
    scale_boundary = None
    if args.use_semantic_boundary_features:
        scale_pred_maps = scale_prob.argmax(axis=1).astype(np.int64, copy=False)
        valid = valid_label_mask(label, num_classes=scale_prob.shape[1])
        scale_boundary_maps = scale_boundary_features_from_pred(
            scale_pred_maps,
            radius=args.semantic_boundary_radius,
            valid=valid,
        )
        scale_boundary = scale_boundary_maps.reshape(scale_prob.shape[0], -1).T
    event_flat = event_edge.reshape(-1).astype(np.float32, copy=False) if event_edge is not None else None
    meta = choose_candidate(
        flat_prob,
        scale_specs,
        args,
        scale_boundary=scale_boundary,
        event_edge=event_flat,
    )
    allowed = allowed_region(meta, label, args, target_class_ids, num_classes=flat_prob.shape[-1])
    features = {
        "scalars": make_scalar_features(meta),
        "anchor_class": meta["anchor"]["pred"].astype(np.int64, copy=False),
        "candidate_class": meta["candidate"]["pred"].astype(np.int64, copy=False),
        "candidate_scale": meta["candidate_scale"].astype(np.int64, copy=False),
    }
    accept_score = predict_accept(model, features, device=device)
    anchor_pred = meta["anchor"]["pred"].reshape(height, width).astype(np.uint8, copy=False)
    candidate_pred = meta["candidate"]["pred"].reshape(height, width).astype(np.uint8, copy=False)

    outputs = OrderedDict()
    outputs["anchor_tta"] = {
        "pred": anchor_pred,
        "allowed": np.zeros((height, width), dtype=bool),
        "accepted": np.zeros((height, width), dtype=bool),
    }
    always = allowed.reshape(height, width)
    pred_candidate = anchor_pred.copy()
    pred_candidate[always] = candidate_pred[always]
    outputs["candidate_all_allowed"] = {
        "pred": pred_candidate,
        "allowed": always,
        "accepted": always,
    }
    for threshold in thresholds:
        accepted = (allowed & (accept_score >= float(threshold))).reshape(height, width)
        pred = anchor_pred.copy()
        pred[accepted] = candidate_pred[accepted]
        outputs[f"accept_threshold_{threshold:.3f}"] = {
            "pred": pred,
            "allowed": always,
            "accepted": accepted,
        }
    return outputs


def cache_record_stem(record, image_idx):
    if record.get("image_id"):
        stem = str(record["image_id"])
    else:
        stem = str(record.get("file_name", f"image_{image_idx:06d}"))
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)
    return safe


def eval_cache_path(cache_dir, dataset_name, scale_name, record, image_idx):
    return Path(cache_dir) / dataset_name / scale_name / f"{cache_record_stem(record, image_idx)}.npy"


def eval_cache_manifest_path(cache_dir, dataset_name):
    return Path(cache_dir) / dataset_name / "manifest.json"


def path_signature(path_text):
    if not path_text:
        return ""
    path = Path(path_text)
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def make_eval_cache_signature(args, scale_specs, dataset_name, records):
    return {
        "config_file": path_signature(args.config_file),
        "weights": path_signature(args.weights),
        "scale_specs": scale_specs,
        "flip": bool(args.flip),
        "dataset_name": dataset_name,
        "record_count": len(records),
        "records": [
            {
                "file_name": path_signature(record.get("file_name", "")),
                "sem_seg_file_name": path_signature(record.get("sem_seg_file_name", "")),
            }
            for record in records
        ],
    }


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def any_eval_cache_files(cache_dir, dataset_name):
    dataset_dir = Path(cache_dir) / dataset_name
    return dataset_dir.exists() and any(dataset_dir.glob("*/*.npy"))


def validate_or_prepare_eval_cache(args, scale_specs, dataset_name, records, cache_dir):
    manifest_path = eval_cache_manifest_path(cache_dir, dataset_name)
    signature = make_eval_cache_signature(args, scale_specs, dataset_name, records)
    if manifest_path.exists():
        existing = load_json(manifest_path)
        if existing.get("signature") != signature:
            message = (
                f"Eval cache manifest mismatch for {dataset_name}: {manifest_path}. "
                "Use a new --eval-cache-dir or rerun without --reuse-eval-cache."
            )
            if args.reuse_eval_cache:
                raise RuntimeError(message)
            print(f"[eval-cache] {message} Existing files may be overwritten.", flush=True)
    elif args.reuse_eval_cache and any_eval_cache_files(cache_dir, dataset_name):
        raise RuntimeError(
            f"Eval cache files exist without a manifest for {dataset_name}: "
            f"{Path(cache_dir) / dataset_name}. Use a new --eval-cache-dir or rerun "
            "without --reuse-eval-cache to rebuild a manifest."
        )
    return signature


def collect_eval_cache(args, runtime, scale_specs, dataset_name, records, cache_dir):
    cache_dir = Path(cache_dir)
    signature = validate_or_prepare_eval_cache(args, scale_specs, dataset_name, records, cache_dir)
    for scale_idx, spec in enumerate(scale_specs):
        scale_dir = cache_dir / dataset_name / spec["name"]
        scale_dir.mkdir(parents=True, exist_ok=True)
        missing = [
            image_idx
            for image_idx, record in enumerate(records)
            if not eval_cache_path(cache_dir, dataset_name, spec["name"], record, image_idx).exists()
        ]
        if args.reuse_eval_cache and not missing:
            print(
                f"[eval-cache] reuse {dataset_name}/{spec['name']} "
                f"({len(records)} cached files)",
                flush=True,
            )
            continue

        print(
            f"[eval-cache] collect {scale_idx + 1}/{len(scale_specs)} "
            f"{dataset_name}/{spec['name']} records={len(records)}",
            flush=True,
        )
        cfg = setup_cfg(args, runtime, spec["min_size"], spec["max_size"])
        mapper = runtime.MaskFormerSemanticDatasetMapper(cfg, False)
        scale_model = build_model(cfg, runtime)
        iterator = list(enumerate(records))
        if not args.quiet:
            iterator = tqdm(iterator, desc=f"cache-{dataset_name}-{spec['name']}")
        for image_idx, record in iterator:
            out_path = eval_cache_path(cache_dir, dataset_name, spec["name"], record, image_idx)
            if args.reuse_eval_cache and out_path.exists():
                continue
            label = load_label(record)
            prob = infer_prob(scale_model, mapper, record, label.shape, args.flip)
            np.save(out_path, prob.numpy().astype(np.float16, copy=False))
        del scale_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[eval-cache] done {dataset_name}/{spec['name']}", flush=True)
    write_json(
        eval_cache_manifest_path(cache_dir, dataset_name),
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "signature": signature,
        },
    )


def load_eval_scale_prob(cache_dir, dataset_name, scale_specs, record, image_idx):
    probs = []
    for spec in scale_specs:
        path = eval_cache_path(cache_dir, dataset_name, spec["name"], record, image_idx)
        if not path.exists():
            raise FileNotFoundError(f"Missing eval cache: {path}")
        probs.append(np.load(path).astype(np.float32, copy=False))
    return np.stack(probs, axis=0)


def evaluate_dataset(args, runtime, model, scale_specs, dataset_name, thresholds, target_class_ids, cache_dir):
    records = list(runtime.DatasetCatalog.get(dataset_name))
    if args.eval_limit is not None:
        records = records[: args.eval_limit]
    collect_eval_cache(args, runtime, scale_specs, dataset_name, records, cache_dir)
    methods = OrderedDict()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    iterator = list(enumerate(records))
    if not args.quiet:
        iterator = tqdm(iterator, desc=f"eval-{dataset_name}")
    for image_idx, record in iterator:
        label = load_label(record)
        scale_prob = load_eval_scale_prob(cache_dir, dataset_name, scale_specs, record, image_idx)
        event_edge = None
        if args.use_event_edge_features:
            event_edge = load_event_edge_cache(
                args.event_edge_cache_dir,
                dataset_name,
                record,
                image_idx,
                label.shape,
            )
        image_outputs = evaluate_prob_image(
            scale_prob,
            label,
            scale_specs,
            args,
            target_class_ids,
            model,
            thresholds,
            device,
            event_edge=event_edge,
        )
        for name, output in image_outputs.items():
            if name not in methods:
                methods[name] = {"meter": ConfusionMeter(num_classes=len(CLASSES)), "counts": empty_counts()}
            methods[name]["meter"].update(output["pred"], label)
            anchor = image_outputs["anchor_tta"]["pred"]
            add_counts(methods[name]["counts"], anchor, output["pred"], label, output["allowed"], output["accepted"])

    result = OrderedDict()
    for name, value in methods.items():
        result[name] = {
            **value["meter"].metrics(),
            **finalize_counts(value["counts"]),
        }
    return {
        "dataset": dataset_name,
        "sample_count": len(records),
        "methods": result,
        "top_by_mIoU": [
            {"method": name, **metrics}
            for name, metrics in sorted(result.items(), key=lambda item: item[1]["mIoU"], reverse=True)
        ],
    }


def write_markdown(output, out_dir):
    lines = [
        "# Scale Accept/Reject Calibrator",
        "",
        f"created_at: `{output['created_at']}`",
        f"weights: `{output['args'].get('weights')}`",
        f"scale_specs: `{output['args'].get('scale_specs')}`",
        f"candidate_mode: `{output['args'].get('candidate_mode')}`",
        f"target_classes: `{output['args'].get('target_classes')}`",
        "",
        "## Training Samples",
        "",
        "| Item | Count |",
        "|---|---:|",
    ]
    for key, value in output["train_examples"].items():
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Train", "", "| Epoch | Loss | Accept@0.5 | Positive |", "|---:|---:|---:|---:|"])
    for row in output["train_history"]:
        lines.append(
            f"| {row['epoch']} | {row['loss']:.5f} | "
            f"{100.0 * row['accept_rate_at_0.5']:.3f}% | {100.0 * row['positive_rate']:.3f}% |"
        )
    for dataset in output.get("datasets", []):
        lines.extend(
            [
                "",
                f"## {dataset['dataset']}",
                "",
                "| Method | mIoU | Changed | Accepted | Net Repaired |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in dataset["top_by_mIoU"]:
            lines.append(
                f"| `{row['method']}` | {row['mIoU']:.4f} | "
                f"{100.0 * row['changed_rate']:.4f}% | {100.0 * row['accepted_rate']:.4f}% | "
                f"{row['net_repaired']} |"
            )
    path = Path(out_dir) / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def make_synthetic_probs(seed=1337, n=6000, scale_count=4, class_count=5):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, class_count, size=n, dtype=np.int64)
    prob = rng.random((n, scale_count, class_count), dtype=np.float32)
    prob /= prob.sum(axis=2, keepdims=True)
    anchor_wrong = rng.random(n) < 0.35
    repairable = anchor_wrong & (rng.random(n) < 0.45)
    damaging = (~anchor_wrong) & (rng.random(n) < 0.10)

    for idx in range(n):
        true_class = int(labels[idx])
        wrong_class = int((true_class + 1 + rng.integers(0, class_count - 1)) % class_count)
        base_class = wrong_class if anchor_wrong[idx] else true_class
        for scale_idx in range(scale_count):
            pred_class = base_class
            if repairable[idx] and scale_idx == 2:
                pred_class = true_class
            if damaging[idx] and scale_idx == 2:
                pred_class = wrong_class
            prob[idx, scale_idx] *= 0.15
            prob[idx, scale_idx, pred_class] += 0.85
            prob[idx, scale_idx] /= prob[idx, scale_idx].sum()
    return prob.astype(np.float16), labels


def run_synthetic_smoke(args):
    scale_specs = parse_scale_specs(args.scale_specs)
    prob, labels = make_synthetic_probs(seed=args.seed, scale_count=len(scale_specs))
    target_class_ids = []
    matrix = {"prob": prob, "target": labels}
    if args.use_semantic_boundary_features:
        scale_pred_maps = prob.reshape(60, 100, len(scale_specs), prob.shape[-1])
        scale_pred_maps = scale_pred_maps.transpose(2, 0, 1, 3).argmax(axis=-1)
        boundary = scale_boundary_features_from_pred(
            scale_pred_maps,
            radius=args.semantic_boundary_radius,
        )
        matrix["boundary"] = boundary.reshape(len(scale_specs), -1).T
    event_edge = None
    if args.use_event_edge_features:
        meta_for_event = choose_candidate(prob, scale_specs, args)
        event_edge = ((meta_for_event["candidate"]["pred"] != meta_for_event["anchor"]["pred"]).astype(np.float32))
        matrix["event_edge"] = event_edge.astype(np.float16)
    examples = make_examples_from_matrix(matrix, scale_specs, args, target_class_ids)
    model, history = train_calibrator(args, examples, scale_specs, num_classes=prob.shape[-1])
    device = torch.device("cpu")
    output = evaluate_prob_image(
        prob[: labels.shape[0]].transpose(1, 2, 0).reshape(len(scale_specs), prob.shape[-1], 60, 100),
        labels.reshape(60, 100),
        scale_specs,
        args,
        target_class_ids,
        model,
        parse_thresholds(args.thresholds),
        device,
        event_edge=event_edge.reshape(60, 100) if event_edge is not None else None,
    )
    anchor = output["anchor_tta"]["pred"].reshape(-1)
    label_flat = labels.reshape(-1)
    metrics = {}
    for name, row in output.items():
        pred = row["pred"].reshape(-1)
        metrics[name] = {
            "accuracy": float((pred == label_flat).mean()),
            "changed": int((pred != anchor).sum()),
            "accepted": int(row["accepted"].sum()),
        }
    result = {
        "examples": examples["stats"],
        "history": history,
        "metrics": metrics,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


def main():
    args = parse_args()
    args.allow_pair_ids = parse_pair_ids(args.allow_pairs)
    args.deny_pair_ids = parse_pair_ids(args.deny_pairs)
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.synthetic_smoke:
        run_synthetic_smoke(args)
        return

    if not args.config_file or not args.weights:
        raise ValueError("--config-file and --weights are required unless --synthetic-smoke is used.")

    runtime = load_runtime()
    runtime.register_cosec()
    scale_specs = parse_scale_specs(args.scale_specs)
    target_class_ids = class_ids_from_csv(args.target_classes)
    thresholds = parse_thresholds(args.thresholds)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_cache_dir = Path(args.eval_cache_dir) if args.eval_cache_dir else out_dir / "eval_cache"

    matrix_path = Path(args.train_matrix_path) if args.train_matrix_path else out_dir / "train_matrix.npz"
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    if args.reuse_train_matrix and matrix_path.exists():
        print(f"[matrix] loading cached train matrix: {matrix_path}", flush=True)
        loaded = np.load(matrix_path)
        matrix = {name: loaded[name] for name in loaded.files}
    else:
        train_records = prepare_records(runtime, args.train_dataset, args.train_limit, args.seed)
        labels = [load_label(record) for record in train_records]
        rng = np.random.default_rng(args.seed)
        coords = [sample_pixels(label, args.pixels_per_image, rng) for label in labels]
        matrix = collect_sample_matrix(args, runtime, scale_specs, train_records, labels, coords)
        np.savez_compressed(matrix_path, **matrix)
        print(f"[matrix] wrote train matrix: {matrix_path}", flush=True)

    examples = make_examples_from_matrix(matrix, scale_specs, args, target_class_ids)
    model, history = train_calibrator(args, examples, scale_specs, num_classes=matrix["prob"].shape[-1])
    model_path = out_dir / "accept_reject_calibrator.pth"
    torch.save(
        {
            "model": model.state_dict(),
            "scale_specs": scale_specs,
            "classes": list(CLASSES),
            "args": vars(args),
        },
        model_path,
    )

    output = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "scale_specs": scale_specs,
        "branch_specs": parse_branch_specs(args.branch_specs, scale_specs)
        if args.candidate_mode == "class_route"
        else {},
        "target_class_ids": target_class_ids,
        "target_class_names": [CLASSES[idx] for idx in target_class_ids],
        "allow_pair_ids": args.allow_pair_ids,
        "allow_pair_names": pair_names(args.allow_pair_ids),
        "deny_pair_ids": args.deny_pair_ids,
        "deny_pair_names": pair_names(args.deny_pair_ids),
        "model_path": str(model_path),
        "eval_cache_dir": str(eval_cache_dir),
        "train_matrix_path": str(matrix_path),
        "train_examples": examples["stats"],
        "train_history": history,
        "datasets": [
            evaluate_dataset(
                args,
                runtime,
                model,
                scale_specs,
                dataset_name,
                thresholds,
                target_class_ids,
                eval_cache_dir,
            )
            for dataset_name in split_csv(args.eval_datasets)
        ],
    }
    json_path = out_dir / "metrics.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, sort_keys=True)
        f.write("\n")
    md_path = write_markdown(output, out_dir)
    print(f"Wrote calibrator: {model_path}")
    print(f"Wrote metrics: {json_path}")
    print(f"Wrote summary: {md_path}")
    for dataset in output["datasets"]:
        print(f"[{dataset['dataset']}]")
        for row in dataset["top_by_mIoU"][:8]:
            print(
                f"  {row['method']}: mIoU={row['mIoU']:.4f}, "
                f"changed={100.0 * row['changed_rate']:.4f}%, net={row['net_repaired']}"
            )


if __name__ == "__main__":
    main()
