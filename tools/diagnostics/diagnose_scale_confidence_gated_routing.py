#!/usr/bin/env python
"""Diagnose confidence-gated branch routing for CoSEC multi-scale predictions."""

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


DEFAULT_SCALES = OrderedDict(
    [
        ("s512", (512, 1200)),
        ("s624", (624, 1200)),
        ("s768", (768, 1400)),
        ("s1024", (1024, 1600)),
    ]
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--anchor", default="tta4flip")
    parser.add_argument("--basis", default="base624")
    parser.add_argument(
        "--fixed-route",
        action="append",
        default=[],
        help=(
            "Class route in class_name=branch_name form. Branches: base624, "
            "base624flip, highres768, highres768flip, tta3flip, tta4flip, tta4noflip."
        ),
    )
    parser.add_argument("--boundary-radii", default="1,3,5,9")
    parser.add_argument("--boundary-sources", default="anchor,basis,union,intersection")
    parser.add_argument("--uncertainty-qs", default="10,20,30,40,60")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def split_csv(text):
    return [part.strip() for part in str(text).split(",") if part.strip()]


def parse_ints(text):
    return [int(part) for part in split_csv(text)]


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
    return label.astype(np.int64, copy=False)


def valid_label_mask(label):
    return (label != 255) & (label >= 0) & (label < len(CLASSES))


def without_event_fields(record):
    cleaned = copy.deepcopy(record)
    for key in ["event_h5", "event_old", "event_new"]:
        cleaned.pop(key, None)
    return cleaned


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


def infer_prob(model, mapped, shape, use_flip):
    with torch.no_grad():
        scores = model([dict(mapped)])[0]["sem_seg"].detach().cpu()
        if use_flip:
            flipped = dict(mapped)
            flipped["image"] = torch.flip(mapped["image"], dims=[2])
            flip_scores = model([flipped])[0]["sem_seg"].detach().cpu()
            scores = 0.5 * (scores + torch.flip(flip_scores, dims=[2]))
    return normalize_scores(resize_scores(scores, shape))


def top_stats(prob):
    top2 = torch.topk(prob, k=2, dim=0).values
    conf = top2[0].numpy()
    margin = (top2[0] - top2[1]).numpy()
    pred = prob.argmax(dim=0).numpy().astype(np.uint8, copy=False)
    entropy = -(prob.clamp_min(1e-8) * prob.clamp_min(1e-8).log()).sum(dim=0)
    entropy = (entropy / np.log(prob.shape[0])).numpy()
    return pred, conf, margin, entropy


class ConfusionMeter:
    def __init__(self, num_classes=19):
        self.num_classes = int(num_classes)
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, pred, label):
        keep = valid_label_mask(label)
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
        "routed_pixels": 0,
    }


def add_counts(counts, anchor_pred, pred, label, route_mask):
    valid = valid_label_mask(label)
    base_wrong = valid & (anchor_pred != label)
    base_correct = valid & (anchor_pred == label)
    repaired = base_wrong & (pred == label)
    damaged = base_correct & (pred != label)
    changed = valid & (anchor_pred != pred)
    counts["valid_pixels"] += int(valid.sum())
    counts["base_wrong"] += int(base_wrong.sum())
    counts["repaired"] += int(repaired.sum())
    counts["damaged"] += int(damaged.sum())
    counts["changed"] += int(changed.sum())
    counts["routed_pixels"] += int((valid & route_mask).sum())


def finalize_counts(counts):
    valid = counts["valid_pixels"]
    wrong = counts["base_wrong"]
    return {
        **counts,
        "repair_rate": float(counts["repaired"] / wrong) if wrong else 0.0,
        "net_repaired": int(counts["repaired"] - counts["damaged"]),
        "changed_rate": float(counts["changed"] / valid) if valid else 0.0,
        "routed_rate": float(counts["routed_pixels"] / valid) if valid else 0.0,
    }


def parse_fixed_routes(route_specs):
    route = {}
    for spec in route_specs:
        if "=" not in spec:
            raise ValueError(f"Invalid fixed route: {spec}")
        class_name, branch_name = [part.strip() for part in spec.split("=", 1)]
        if class_name not in CLASSES:
            raise ValueError(f"Unknown class: {class_name}")
        route[CLASSES.index(class_name)] = branch_name
    return route


def percentile_mask(values, valid, q):
    sampled = values[valid]
    if sampled.size == 0:
        return np.zeros_like(valid)
    return valid & (values <= np.percentile(sampled, float(q)))


def branch_prob(branch_name, scale_probs_flip, scale_probs_noflip):
    if branch_name == "base624":
        return scale_probs_noflip["s624"]
    if branch_name == "base624flip":
        return scale_probs_flip["s624"]
    if branch_name == "highres768":
        return scale_probs_noflip["s768"]
    if branch_name == "highres768flip":
        return scale_probs_flip["s768"]
    if branch_name == "tta3flip":
        return torch.stack([scale_probs_flip[name] for name in ("s512", "s768", "s1024")]).mean(dim=0)
    if branch_name == "tta4flip":
        return torch.stack(list(scale_probs_flip.values())).mean(dim=0)
    if branch_name == "tta4noflip":
        return torch.stack(list(scale_probs_noflip.values())).mean(dim=0)
    raise ValueError(f"Unknown branch: {branch_name}")


def route_prediction(anchor_pred, basis_pred, branches, routes, gate):
    merged = anchor_pred.copy()
    routed = np.zeros(anchor_pred.shape, dtype=bool)
    for class_id, branch_name in routes.items():
        class_region = basis_pred == class_id
        if not class_region.any():
            continue
        take = class_region & gate(branch_name)
        if not take.any():
            continue
        merged[take] = branches[branch_name]["pred"][take]
        routed |= take
    return merged, routed


def update_method(methods, name, pred, anchor_pred, label, routed):
    if name not in methods:
        methods[name] = {"meter": ConfusionMeter(num_classes=len(CLASSES)), "counts": empty_counts()}
    methods[name]["meter"].update(pred, label)
    add_counts(methods[name]["counts"], anchor_pred, pred, label, routed)


def semantic_boundary_band(pred, radius, valid):
    if radius <= 0:
        return np.zeros(pred.shape, dtype=bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    low = pred.astype(np.float32, copy=True)
    high = pred.astype(np.float32, copy=True)
    low[~valid] = -1000.0
    high[~valid] = 1000.0
    local_max = cv2.dilate(low, kernel)
    local_min = cv2.erode(high, kernel)
    return valid & (local_max != local_min)


def boundary_for_source(source, anchor_boundary, basis_boundary):
    if source == "anchor":
        return anchor_boundary
    if source == "basis":
        return basis_boundary
    if source == "union":
        return anchor_boundary | basis_boundary
    if source == "intersection":
        return anchor_boundary & basis_boundary
    raise ValueError(f"Unknown boundary source: {source}")


def main():
    args = parse_args()
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    register_cosec()
    routes = parse_fixed_routes(args.fixed_route)
    if not routes:
        routes = {
            CLASSES.index("building"): "base624",
            CLASSES.index("pole"): "base624",
            CLASSES.index("vegetation"): "tta3flip",
        }

    records = list(DatasetCatalog.get(args.dataset))
    if args.limit is not None:
        records = records[: args.limit]

    models = {}
    mappers = {}
    for name, (min_size, max_size) in DEFAULT_SCALES.items():
        cfg = setup_cfg(args, min_size, max_size)
        models[name] = build_model(cfg)
        mappers[name] = MaskFormerSemanticDatasetMapper(cfg, False)

    methods = OrderedDict()
    iterator = records if args.quiet else tqdm(records, desc=args.dataset)
    for record in iterator:
        label = load_label(record)
        shape = label.shape
        valid = valid_label_mask(label)
        scale_probs_flip = OrderedDict()
        scale_probs_noflip = OrderedDict()
        for scale_name, model in models.items():
            mapped = mappers[scale_name](without_event_fields(record))
            scale_probs_noflip[scale_name] = infer_prob(model, mapped, shape, use_flip=False)
            scale_probs_flip[scale_name] = infer_prob(model, mapped, shape, use_flip=True)

        branch_names = [
            "base624",
            "base624flip",
            "highres768",
            "highres768flip",
            "tta3flip",
            "tta4flip",
            "tta4noflip",
        ]
        if args.anchor not in branch_names:
            raise ValueError(f"Unknown anchor branch: {args.anchor}")
        if args.basis not in branch_names:
            raise ValueError(f"Unknown basis branch: {args.basis}")
        branches = OrderedDict()
        for branch_name in branch_names:
            prob = branch_prob(branch_name, scale_probs_flip, scale_probs_noflip)
            pred, conf, margin, entropy = top_stats(prob)
            branches[branch_name] = {
                "prob": prob,
                "pred": pred,
                "conf": conf,
                "margin": margin,
                "entropy": entropy,
            }

        anchor = branches[args.anchor]
        basis = branches[args.basis]["pred"]
        update_method(
            methods,
            f"anchor_{args.anchor}",
            anchor["pred"],
            anchor["pred"],
            label,
            np.zeros(shape, dtype=bool),
        )
        for branch_name in branch_names:
            update_method(
                methods,
                f"branch_{branch_name}",
                branches[branch_name]["pred"],
                anchor["pred"],
                label,
                np.ones(shape, dtype=bool),
            )

        all_gate = lambda _branch_name: valid
        pred, routed = route_prediction(anchor["pred"], basis, branches, routes, all_gate)
        update_method(methods, "fixed_route_all", pred, anchor["pred"], label, routed)

        for delta in [-0.05, -0.02, 0.0, 0.02, 0.05]:
            pred, routed = route_prediction(
                anchor["pred"],
                basis,
                branches,
                routes,
                lambda b, d=delta: valid & (branches[b]["conf"] >= anchor["conf"] + d),
            )
            update_method(methods, f"fixed_conf_delta_{delta:+.2f}", pred, anchor["pred"], label, routed)

        for delta in [-0.02, 0.0, 0.02, 0.05]:
            pred, routed = route_prediction(
                anchor["pred"],
                basis,
                branches,
                routes,
                lambda b, d=delta: valid & (branches[b]["margin"] >= anchor["margin"] + d),
            )
            update_method(methods, f"fixed_margin_delta_{delta:+.2f}", pred, anchor["pred"], label, routed)

        for q in parse_ints(args.uncertainty_qs):
            uncertain_margin = percentile_mask(anchor["margin"], valid, q)
            high_entropy = valid & (anchor["entropy"] >= np.percentile(anchor["entropy"][valid], 100 - q))
            for region_name, region in [("lowmargin", uncertain_margin), ("highentropy", high_entropy)]:
                pred, routed = route_prediction(
                    anchor["pred"],
                    basis,
                    branches,
                    routes,
                    lambda b, r=region: r & (branches[b]["pred"] != anchor["pred"]),
                )
                update_method(methods, f"fixed_{region_name}_q{q}_disagree", pred, anchor["pred"], label, routed)
                pred, routed = route_prediction(
                    anchor["pred"],
                    basis,
                    branches,
                    routes,
                    lambda b, r=region: r & (branches[b]["conf"] >= anchor["conf"] - 0.02),
                )
                update_method(methods, f"fixed_{region_name}_q{q}_conf_ge_anchor_m002", pred, anchor["pred"], label, routed)

        for radius in parse_ints(args.boundary_radii):
            anchor_boundary = semantic_boundary_band(anchor["pred"], radius, valid)
            basis_boundary = semantic_boundary_band(basis, radius, valid)
            for source in split_csv(args.boundary_sources):
                boundary = boundary_for_source(source, anchor_boundary, basis_boundary)
                pred, routed = route_prediction(
                    anchor["pred"],
                    basis,
                    branches,
                    routes,
                    lambda b, r=boundary: r & (branches[b]["pred"] != anchor["pred"]),
                )
                update_method(
                    methods,
                    f"fixed_boundary_{source}_r{radius}_disagree",
                    pred,
                    anchor["pred"],
                    label,
                    routed,
                )
                for q in parse_ints(args.uncertainty_qs):
                    low_margin = percentile_mask(anchor["margin"], valid, q)
                    high_entropy = valid & (
                        anchor["entropy"] >= np.percentile(anchor["entropy"][valid], 100 - q)
                    )
                    for region_name, region in [("lowmargin", low_margin), ("highentropy", high_entropy)]:
                        pred, routed = route_prediction(
                            anchor["pred"],
                            basis,
                            branches,
                            routes,
                            lambda b, r=boundary, u=region: r
                            & u
                            & (branches[b]["pred"] != anchor["pred"]),
                        )
                        update_method(
                            methods,
                            f"fixed_boundary_{source}_r{radius}_{region_name}_q{q}_disagree",
                            pred,
                            anchor["pred"],
                            label,
                            routed,
                        )

    output = {
        "args": vars(args),
        "routes": {CLASSES[class_id]: branch for class_id, branch in routes.items()},
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
        {"method": name, **metrics}
        for name, metrics in sorted(
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

    print(f"Wrote scale confidence routing diagnostics: {out_path}")
    print("Top methods by mIoU:")
    for row in output["top_by_mIoU"][:12]:
        print(
            f"  {row['method']}: mIoU={row['mIoU']:.4f}, "
            f"changed={100*row['changed_rate']:.4f}%, routed={100*row['routed_rate']:.4f}%, "
            f"net={row['net_repaired']}"
        )


if __name__ == "__main__":
    main()
