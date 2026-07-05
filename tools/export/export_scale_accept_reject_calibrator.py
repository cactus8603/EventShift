#!/usr/bin/env python
"""Export CoSEC test masks with a trained scale accept/reject calibrator."""

import argparse
import copy
import json
import os
import shutil
import sys
import importlib.util
import zipfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

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

from cosec_event_dataset import load_event_edge_representation  # noqa: E402
from train_scale_accept_reject_calibrator import (  # noqa: E402
    AcceptRejectCalibrator,
    CLASSES,
    allowed_region,
    choose_candidate,
    class_ids_from_csv,
    make_scalar_features,
    parse_pair_ids,
    parse_scale_specs,
    predict_accept,
    scale_boundary_features_from_pred,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default=None)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--calibrator", required=True)
    parser.add_argument(
        "--scale-specs",
        default=None,
        help="Optional override. Defaults to specs stored in the calibrator checkpoint.",
    )
    parser.add_argument("--test-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--zip", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--candidate-all",
        action="store_true",
        help="Accept every allowed candidate pixel instead of using the learned threshold.",
    )
    parser.add_argument(
        "--event-window-radii-ms",
        nargs="+",
        type=int,
        default=[25, 50],
        help="Event windows used when the calibrator expects event-edge features.",
    )
    parser.add_argument(
        "--missing-event",
        choices=["error", "zero"],
        default="error",
        help="How to handle missing test event files when event features are required.",
    )
    parser.add_argument("--chunk-pixels", type=int, default=262144)
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Extra detectron2 config options after --.",
    )
    return parser.parse_args()


def split_opts(opts):
    if not opts:
        return []
    return opts[1:] if opts and opts[0] == "--" else opts


def setup_cfg(args, min_size, max_size):
    sys.path.insert(0, str(ROOT / "third_party" / "Mask2Former"))
    if importlib.util.find_spec("detectron2") is None:
        sys.path.insert(0, str(ROOT / "third_party" / "detectron2"))
    from detectron2.config import get_cfg  # pylint: disable=import-outside-toplevel
    from detectron2.projects.deeplab import add_deeplab_config  # pylint: disable=import-outside-toplevel
    from mask2former import add_maskformer2_config  # pylint: disable=import-outside-toplevel

    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    opts = split_opts(args.opts)
    if opts:
        cfg.merge_from_list(opts)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.DEVICE = args.device
    cfg.DATASETS.TEST = ()
    cfg.TEST.AUG.ENABLED = False
    cfg.INPUT.MIN_SIZE_TEST = int(min_size)
    cfg.INPUT.MAX_SIZE_TEST = int(max_size)
    cfg.freeze()
    return cfg


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
            yield seq_dir, img_path


def read_sequence_timestamps(seq_dir):
    path = Path(seq_dir) / "timestamps.txt"
    if not path.exists():
        return None
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            values.append(int(float(line)))
    return values


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


def predict_scores(predictor, image, use_flip):
    with torch.no_grad():
        outputs = predictor(image)
        scores = outputs["sem_seg"].detach()
        if not use_flip:
            return scores
        flip_image = np.ascontiguousarray(image[:, ::-1])
        flip_outputs = predictor(flip_image)
        flip_scores = torch.flip(flip_outputs["sem_seg"].detach(), dims=[2])
        return 0.5 * (scores + flip_scores)


def event_edge_score(event_h5, timestamp_us, image_shape, radii_ms):
    record = {
        "event_h5": str(event_h5),
        "event_old": [int(timestamp_us), int(timestamp_us)],
        "event_new": [int(timestamp_us), int(timestamp_us)],
    }
    channels = load_event_edge_representation(record, image_shape, radii_ms)
    if channels.shape[0] == 0:
        return np.zeros(image_shape, dtype=np.float32)
    edge_channels = channels[1::3]
    if edge_channels.shape[0] == 0:
        edge_channels = channels
    score = edge_channels.max(axis=0).astype(np.float32, copy=False)
    max_value = float(score.max())
    if max_value > 1e-6:
        score = score / max_value
    return score.astype(np.float32, copy=False)


def load_calibrator(path, device):
    ckpt = torch.load(path, map_location="cpu")
    saved_args = dict(ckpt.get("args", {}))
    state = ckpt["model"]
    class_embed_dim = int(state["anchor_embed.weight"].shape[1])
    scale_embed_dim = int(state["scale_embed.weight"].shape[1])
    hidden_dim = int(state["mlp.0.weight"].shape[0])
    scalar_dim = int(state["mlp.0.weight"].shape[1] - 2 * class_embed_dim - scale_embed_dim)
    model = AcceptRejectCalibrator(
        num_classes=int(state["anchor_embed.weight"].shape[0]),
        num_scales=int(state["scale_embed.weight"].shape[0]),
        scalar_dim=scalar_dim,
        class_embed_dim=class_embed_dim,
        scale_embed_dim=scale_embed_dim,
        hidden_dim=hidden_dim,
        init_bias=float(saved_args.get("init_bias", -3.0)),
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, ckpt, SimpleNamespace(**saved_args)


def ensure_calibrator_args(calib_args):
    if not hasattr(calib_args, "allow_pair_ids"):
        calib_args.allow_pair_ids = parse_pair_ids(getattr(calib_args, "allow_pairs", ""))
    if not hasattr(calib_args, "deny_pair_ids"):
        calib_args.deny_pair_ids = parse_pair_ids(getattr(calib_args, "deny_pairs", ""))
    if not hasattr(calib_args, "basis_branch"):
        calib_args.basis_branch = ""
    return calib_args


def route_image(scale_prob, event_edge, model, calib_args, scale_specs, threshold, candidate_all, device, chunk_pixels):
    height, width = scale_prob.shape[-2:]
    flat_prob = scale_prob.transpose(2, 3, 0, 1).reshape(-1, scale_prob.shape[0], scale_prob.shape[1])
    scale_boundary = None
    if getattr(calib_args, "use_semantic_boundary_features", False):
        scale_pred_maps = scale_prob.argmax(axis=1).astype(np.int64, copy=False)
        scale_boundary_maps = scale_boundary_features_from_pred(
            scale_pred_maps,
            radius=getattr(calib_args, "semantic_boundary_radius", 3),
        )
        scale_boundary = scale_boundary_maps.reshape(scale_prob.shape[0], -1).T

    event_flat = None
    if getattr(calib_args, "use_event_edge_features", False):
        if event_edge is None:
            event_flat = np.zeros(height * width, dtype=np.float32)
        else:
            event_flat = event_edge.reshape(-1).astype(np.float32, copy=False)

    meta = choose_candidate(
        flat_prob,
        scale_specs,
        calib_args,
        scale_boundary=scale_boundary,
        event_edge=event_flat,
    )
    target_class_ids = class_ids_from_csv(getattr(calib_args, "target_classes", ""))
    allowed = allowed_region(meta, None, calib_args, target_class_ids, num_classes=flat_prob.shape[-1])
    anchor_pred = meta["anchor"]["pred"].reshape(height, width).astype(np.uint8, copy=False)
    candidate_pred = meta["candidate"]["pred"].reshape(height, width).astype(np.uint8, copy=False)

    if candidate_all:
        accepted = allowed
        accept_score = np.zeros_like(allowed, dtype=np.float32)
    else:
        if threshold is None:
            raise ValueError("--threshold is required unless --candidate-all is set.")
        features = {
            "scalars": make_scalar_features(meta),
            "anchor_class": meta["anchor"]["pred"].astype(np.int64, copy=False),
            "candidate_class": meta["candidate"]["pred"].astype(np.int64, copy=False),
            "candidate_scale": meta["candidate_scale"].astype(np.int64, copy=False),
        }
        accept_score = predict_accept(model, features, device=device, chunk_pixels=chunk_pixels)
        accepted = allowed & (accept_score >= float(threshold))

    pred = anchor_pred.copy()
    accepted_2d = accepted.reshape(height, width)
    pred[accepted_2d] = candidate_pred[accepted_2d]
    return {
        "pred": pred,
        "anchor": anchor_pred,
        "candidate": candidate_pred,
        "allowed": allowed.reshape(height, width),
        "accepted": accepted_2d,
        "accept_score": accept_score.reshape(height, width),
    }


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
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, ckpt, calib_args = load_calibrator(args.calibrator, device)
    calib_args = ensure_calibrator_args(calib_args)
    if args.config_file:
        calib_args.config_file = args.config_file
    if args.weights:
        calib_args.weights = args.weights
    if not getattr(calib_args, "config_file", None) or not getattr(calib_args, "weights", None):
        raise ValueError("--config-file and --weights are required if the calibrator did not store them.")
    calib_args.device = args.device
    calib_args.config_file = str(calib_args.config_file)
    calib_args.weights = str(calib_args.weights)
    scale_specs = parse_scale_specs(args.scale_specs) if args.scale_specs else list(ckpt["scale_specs"])

    out_dir = Path(args.out_dir)
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = list(iter_test_images(args.test_root, args.sequences))
    if args.limit is not None:
        images = images[: args.limit]
    if not images:
        raise ValueError("No test images matched the requested sequences.")

    timestamp_cache = {}
    loaded_images = []
    for seq_dir, img_path in images:
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read image: {img_path}")
        seq_name = seq_dir.name
        if seq_name not in timestamp_cache:
            timestamp_cache[seq_name] = read_sequence_timestamps(seq_dir)
        frame_idx = int(img_path.stem)
        timestamp = None
        timestamps = timestamp_cache[seq_name]
        if timestamps is not None and frame_idx < len(timestamps):
            timestamp = int(timestamps[frame_idx])
        loaded_images.append(
            {
                "seq_dir": seq_dir,
                "seq_name": seq_name,
                "img_path": img_path,
                "frame_idx": frame_idx,
                "timestamp": timestamp,
                "image": image,
                "shape": image.shape[:2],
            }
        )

    from detectron2.engine import DefaultPredictor  # pylint: disable=import-outside-toplevel

    all_probs = [[] for _ in loaded_images]
    for spec in scale_specs:
        export_args = copy.copy(args)
        export_args.config_file = calib_args.config_file
        export_args.weights = calib_args.weights
        cfg = setup_cfg(export_args, spec["min_size"], spec["max_size"])
        predictor = DefaultPredictor(cfg)
        for image_idx, item in enumerate(tqdm(loaded_images, desc=f"export-{spec['name']}")):
            scores = predict_scores(predictor, item["image"], getattr(calib_args, "flip", False))
            prob = normalize_scores(resize_scores(scores, item["shape"]))
            all_probs[image_idx].append(prob.to(torch.float16).cpu().numpy())
        del predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    counts = {
        "total": 0,
        "skipped": 0,
        "sequences": {},
        "pixels": 0,
        "allowed_pixels": 0,
        "accepted_pixels": 0,
        "changed_vs_anchor": 0,
        "missing_event": 0,
    }
    by_sequence = {}
    for item, probs in tqdm(list(zip(loaded_images, all_probs)), desc="accept-write"):
        seq_name = item["seq_name"]
        dst_dir = out_dir / seq_name / "segment_co"
        dst_path = dst_dir / item["img_path"].name
        if args.skip_existing and dst_path.exists():
            counts["skipped"] += 1
            continue

        event_edge = None
        if getattr(calib_args, "use_event_edge_features", False):
            event_h5 = item["seq_dir"] / "events_co_left.h5"
            if event_h5.exists() and item["timestamp"] is not None:
                event_edge = event_edge_score(event_h5, item["timestamp"], item["shape"], args.event_window_radii_ms)
            elif args.missing_event == "zero":
                counts["missing_event"] += 1
                event_edge = np.zeros(item["shape"], dtype=np.float32)
            else:
                raise FileNotFoundError(f"Missing event data/timestamp for {item['img_path']}")

        scale_prob = np.stack(probs, axis=0).astype(np.float32, copy=False)
        routed = route_image(
            scale_prob,
            event_edge,
            model,
            calib_args,
            scale_specs,
            args.threshold,
            args.candidate_all,
            device,
            args.chunk_pixels,
        )

        dst_dir.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(dst_path), routed["pred"]):
            raise RuntimeError(f"Could not write mask: {dst_path}")

        pixels = int(routed["pred"].size)
        allowed_count = int(routed["allowed"].sum())
        accepted_count = int(routed["accepted"].sum())
        changed_count = int((routed["pred"] != routed["anchor"]).sum())
        counts["total"] += 1
        counts["pixels"] += pixels
        counts["allowed_pixels"] += allowed_count
        counts["accepted_pixels"] += accepted_count
        counts["changed_vs_anchor"] += changed_count
        counts["sequences"][seq_name] = counts["sequences"].get(seq_name, 0) + 1
        seq_stats = by_sequence.setdefault(
            seq_name,
            {"files": 0, "pixels": 0, "allowed_pixels": 0, "accepted_pixels": 0, "changed_vs_anchor": 0},
        )
        seq_stats["files"] += 1
        seq_stats["pixels"] += pixels
        seq_stats["allowed_pixels"] += allowed_count
        seq_stats["accepted_pixels"] += accepted_count
        seq_stats["changed_vs_anchor"] += changed_count

    zip_info = zip_directory(out_dir, args.zip) if args.zip else None
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(ROOT),
        "config_file": str(Path(calib_args.config_file).resolve()),
        "weights": str(Path(calib_args.weights).resolve()),
        "calibrator": str(Path(args.calibrator).resolve()),
        "calibrator_args": vars(calib_args),
        "test_root": str(Path(args.test_root).resolve()),
        "out_dir": str(out_dir.resolve()),
        "zip": str(Path(args.zip).resolve()) if args.zip else None,
        "threshold": args.threshold,
        "candidate_all": bool(args.candidate_all),
        "scale_specs": scale_specs,
        "event_window_radii_ms": [int(v) for v in args.event_window_radii_ms],
        "counts": counts,
        "by_sequence": by_sequence,
        "zip_info": zip_info,
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    if args.zip:
        print(f"Wrote zip: {args.zip}")
    print(json.dumps({"counts": counts, "zip_info": zip_info}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
