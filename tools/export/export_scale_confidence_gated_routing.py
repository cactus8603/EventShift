#!/usr/bin/env python
"""Export CoSEC test masks with confidence-gated multi-branch routing."""

import argparse
import json
import os
import shutil
import sys
import importlib.util
import zipfile
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
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
from detectron2.config import get_cfg  # noqa: E402
from detectron2.engine import DefaultPredictor  # noqa: E402
from detectron2.projects.deeplab import add_deeplab_config  # noqa: E402
from mask2former import add_maskformer2_config  # noqa: E402


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
    parser.add_argument("--test-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--zip", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--gate",
        default="all",
        choices=[
            "all",
            "conf_delta",
            "margin_delta",
            "lowmargin_disagree",
            "highentropy_disagree",
            "lowmargin_conf_ge_anchor_m002",
            "highentropy_conf_ge_anchor_m002",
        ],
    )
    parser.add_argument("--delta", type=float, default=0.0)
    parser.add_argument("--q", type=float, default=10.0)
    parser.add_argument(
        "--route",
        action="append",
        default=[],
        help="Class route in class_name=branch_name form. Branches: base624, highres768, tta3flip, tta4flip, tta4noflip.",
    )
    return parser.parse_args()


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


def parse_routes(route_specs):
    routes = {}
    for spec in route_specs:
        if "=" not in spec:
            raise ValueError(f"Invalid route: {spec}")
        class_name, branch_name = [part.strip() for part in spec.split("=", 1)]
        if class_name not in CLASSES:
            raise ValueError(f"Unknown class: {class_name}")
        routes[CLASSES.index(class_name)] = branch_name
    if not routes:
        routes = {
            CLASSES.index("building"): "base624",
            CLASSES.index("pole"): "base624",
            CLASSES.index("vegetation"): "tta3flip",
        }
    return routes


def iter_test_images(test_root, sequences=None):
    test_root = Path(test_root)
    keep_sequences = set(sequences) if sequences else None
    for seq_dir in sorted(path for path in test_root.iterdir() if path.is_dir()):
        if keep_sequences is not None and seq_dir.name not in keep_sequences:
            continue
        img_dir = seq_dir / "img_co_left"
        if not img_dir.is_dir():
            continue
        for img_path in sorted(img_dir.glob("*.png")):
            yield seq_dir.name, img_path


def normalize_scores(scores):
    prob = scores.float().clamp_min(1e-8)
    return prob / prob.sum(dim=0, keepdim=True).clamp_min(1e-8)


def infer_prob(predictor, image, use_flip):
    with torch.no_grad():
        scores = predictor(image)["sem_seg"].detach().cpu()
        if use_flip:
            flipped_image = np.ascontiguousarray(image[:, ::-1])
            flip_scores = predictor(flipped_image)["sem_seg"].detach().cpu()
            scores = 0.5 * (scores + torch.flip(flip_scores, dims=[2]))
    return normalize_scores(scores)


def branch_prob(branch_name, scale_probs_flip, scale_probs_noflip):
    if branch_name == "base624":
        return scale_probs_noflip["s624"]
    if branch_name == "highres768":
        return scale_probs_noflip["s768"]
    if branch_name == "tta3flip":
        return torch.stack([scale_probs_flip[name] for name in ("s512", "s768", "s1024")]).mean(dim=0)
    if branch_name == "tta4flip":
        return torch.stack(list(scale_probs_flip.values())).mean(dim=0)
    if branch_name == "tta4noflip":
        return torch.stack(list(scale_probs_noflip.values())).mean(dim=0)
    raise ValueError(f"Unknown branch: {branch_name}")


def top_stats(prob):
    top2 = torch.topk(prob, k=2, dim=0).values
    conf = top2[0].numpy()
    margin = (top2[0] - top2[1]).numpy()
    pred = prob.argmax(dim=0).numpy().astype(np.uint8, copy=False)
    entropy = -(prob.clamp_min(1e-8) * prob.clamp_min(1e-8).log()).sum(dim=0)
    entropy = (entropy / np.log(prob.shape[0])).numpy()
    return {"pred": pred, "conf": conf, "margin": margin, "entropy": entropy}


def percentile_region(values, q, high=False):
    threshold = np.percentile(values, float(100.0 - q if high else q))
    return values >= threshold if high else values <= threshold


def gate_mask(args, branch, anchor):
    if args.gate == "all":
        return np.ones(anchor["pred"].shape, dtype=bool)
    if args.gate == "conf_delta":
        return branch["conf"] >= anchor["conf"] + args.delta
    if args.gate == "margin_delta":
        return branch["margin"] >= anchor["margin"] + args.delta
    if args.gate == "lowmargin_disagree":
        return percentile_region(anchor["margin"], args.q, high=False) & (branch["pred"] != anchor["pred"])
    if args.gate == "highentropy_disagree":
        return percentile_region(anchor["entropy"], args.q, high=True) & (branch["pred"] != anchor["pred"])
    if args.gate == "lowmargin_conf_ge_anchor_m002":
        return percentile_region(anchor["margin"], args.q, high=False) & (branch["conf"] >= anchor["conf"] - 0.02)
    if args.gate == "highentropy_conf_ge_anchor_m002":
        return percentile_region(anchor["entropy"], args.q, high=True) & (branch["conf"] >= anchor["conf"] - 0.02)
    raise ValueError(args.gate)


def route_prediction(anchor, basis_pred, branches, routes, args):
    merged = anchor["pred"].copy()
    route_stats = {"routed_pixels": 0, "changed_vs_anchor": 0, "by_class": {}, "by_branch": {}}
    for class_id, branch_name in routes.items():
        branch = branches[branch_name]
        take = (basis_pred == class_id) & gate_mask(args, branch, anchor)
        if not take.any():
            continue
        changed = take & (branch["pred"] != anchor["pred"])
        merged[take] = branch["pred"][take]
        class_name = CLASSES[class_id]
        route_stats["routed_pixels"] += int(take.sum())
        route_stats["changed_vs_anchor"] += int(changed.sum())
        route_stats["by_class"][class_name] = route_stats["by_class"].get(class_name, 0) + int(take.sum())
        route_stats["by_branch"][branch_name] = route_stats["by_branch"].get(branch_name, 0) + int(take.sum())
    return merged, route_stats


def zip_directory(src_dir, zip_path):
    src_dir = Path(src_dir)
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    max_path_len = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(src_dir.rglob("*")):
            if not path.is_file() or path.name == "manifest.json":
                continue
            rel_path = path.relative_to(src_dir)
            max_path_len = max(max_path_len, len(str(rel_path)))
            zf.write(path, rel_path)
            count += 1
    return {"entries": count, "max_path_len": max_path_len}


def main():
    args = parse_args()
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    routes = parse_routes(args.route)

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    predictors = OrderedDict()
    for scale_name, (min_size, max_size) in DEFAULT_SCALES.items():
        predictors[scale_name] = DefaultPredictor(setup_cfg(args, min_size, max_size))

    images = list(iter_test_images(args.test_root, args.sequences))
    if args.limit is not None:
        images = images[: args.limit]
    counts = {"total": 0, "sequences": {}}
    global_stats = {"routed_pixels": 0, "changed_vs_anchor": 0, "by_class": {}, "by_branch": {}}

    for seq_name, img_path in tqdm(images, desc="Export gated masks"):
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read image: {img_path}")

        scale_probs_flip = OrderedDict()
        scale_probs_noflip = OrderedDict()
        for scale_name, predictor in predictors.items():
            scale_probs_noflip[scale_name] = infer_prob(predictor, image, use_flip=False)
            scale_probs_flip[scale_name] = infer_prob(predictor, image, use_flip=True)

        branches = OrderedDict()
        for branch_name in ["base624", "highres768", "tta3flip", "tta4flip", "tta4noflip"]:
            branches[branch_name] = top_stats(branch_prob(branch_name, scale_probs_flip, scale_probs_noflip))

        anchor = branches["tta4flip"]
        pred, stats = route_prediction(anchor, branches["base624"]["pred"], branches, routes, args)
        for key in ("routed_pixels", "changed_vs_anchor"):
            global_stats[key] += stats[key]
        for key in ("by_class", "by_branch"):
            for name, value in stats[key].items():
                global_stats[key][name] = global_stats[key].get(name, 0) + value

        dst_dir = out_dir / seq_name / "segment_co"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_path = dst_dir / img_path.name
        if not cv2.imwrite(str(dst_path), pred):
            raise RuntimeError(f"Could not write mask: {dst_path}")
        counts["total"] += 1
        counts["sequences"][seq_name] = counts["sequences"].get(seq_name, 0) + 1

    zip_info = zip_directory(out_dir, args.zip) if args.zip else None
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_file": str(Path(args.config_file).resolve()),
        "weights": str(Path(args.weights).resolve()),
        "test_root": str(Path(args.test_root).resolve()),
        "out_dir": str(out_dir.resolve()),
        "zip": str(Path(args.zip).resolve()) if args.zip else None,
        "gate": args.gate,
        "delta": args.delta,
        "q": args.q,
        "routes": {CLASSES[class_id]: branch_name for class_id, branch_name in routes.items()},
        "counts": counts,
        "routing_stats": global_stats,
        "zip_info": zip_info,
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote gated masks: {out_dir}")
    if args.zip:
        print(f"Wrote zip: {args.zip}")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
