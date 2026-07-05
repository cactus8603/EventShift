#!/usr/bin/env python
"""Export CoSEC test masks with a trained lightweight scale router."""

import argparse
import json
import os
import shutil
import sys
import importlib.util
import zipfile
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
from detectron2.config import get_cfg  # noqa: E402
from detectron2.engine import DefaultPredictor  # noqa: E402
from detectron2.projects.deeplab import add_deeplab_config  # noqa: E402
from mask2former import add_maskformer2_config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--router", required=True)
    parser.add_argument(
        "--scale-specs",
        default=None,
        help="Optional override. Comma-separated name:min_size:max_size entries.",
    )
    parser.add_argument("--test-root", default="data/test")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--zip", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--flip", action="store_true", help="Average each scale with horizontal flip.")
    parser.add_argument("--chunk-pixels", type=int, default=262144)
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Extra detectron2 config options after --.",
    )
    return parser.parse_args()


def split_csv(text):
    return [part.strip() for part in text.split(",") if part.strip()]


def parse_scale_specs(text):
    specs = []
    for item in split_csv(text):
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError(f"Bad scale spec '{item}', expected name:min:max")
        name, min_size, max_size = parts
        specs.append({"name": name, "min_size": int(min_size), "max_size": int(max_size)})
    return specs


class ScaleRouter(nn.Module):
    def __init__(self, num_scales, num_classes, class_embed_dim=8, hidden_dim=64):
        super().__init__()
        self.num_scales = int(num_scales)
        self.num_classes = int(num_classes)
        self.class_embed = nn.Embedding(num_classes, class_embed_dim)
        in_dim = 3 + num_scales + class_embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, pred, conf, margin, entropy):
        scale_eye = torch.eye(self.num_scales, device=conf.device, dtype=conf.dtype)
        scale_feat = scale_eye.unsqueeze(0).expand(conf.shape[0], -1, -1)
        cls_feat = self.class_embed(pred.long())
        scalar_feat = torch.stack([conf, margin, entropy], dim=-1)
        feat = torch.cat([scalar_feat, scale_feat, cls_feat], dim=-1)
        return self.mlp(feat).squeeze(-1)


def load_router(path, device):
    ckpt = torch.load(path, map_location="cpu")
    args = ckpt.get("args", {})
    scale_specs = ckpt["scale_specs"]
    class_embed_dim = int(args.get("class_embed_dim", 8))
    hidden_dim = int(args.get("hidden_dim", 64))
    router = ScaleRouter(
        num_scales=len(scale_specs),
        num_classes=len(CLASSES),
        class_embed_dim=class_embed_dim,
        hidden_dim=hidden_dim,
    )
    router.load_state_dict(ckpt["model"])
    router.to(device)
    router.eval()
    return router, scale_specs, ckpt


def setup_cfg(args, min_size, max_size):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    if args.opts:
        opts = args.opts[1:] if args.opts and args.opts[0] == "--" else args.opts
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
            yield seq_dir.name, img_path


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


def predict_scores(predictor, image, cfg, use_flip):
    with torch.no_grad():
        outputs = predictor(image)
        scores = outputs["sem_seg"].detach()
        if not use_flip:
            return scores
        flip_image = np.ascontiguousarray(image[:, ::-1])
        flip_outputs = predictor(flip_image)
        flip_scores = torch.flip(flip_outputs["sem_seg"].detach(), dims=[2])
        return 0.5 * (scores + flip_scores)


def branch_stats(scores, image_shape):
    prob = normalize_scores(resize_scores(scores, image_shape))
    top2 = torch.topk(prob, k=2, dim=0).values
    entropy = -(prob * prob.clamp_min(1e-8).log()).sum(dim=0) / np.log(len(CLASSES))
    return {
        "pred": prob.argmax(dim=0).to(torch.uint8).cpu().numpy(),
        "conf": top2[0].to(torch.float16).cpu().numpy(),
        "margin": (top2[0] - top2[1]).to(torch.float16).cpu().numpy(),
        "entropy": entropy.to(torch.float16).cpu().numpy(),
    }


def route_image(router, branch_outputs, device, chunk_pixels):
    pred_stack = np.stack([item["pred"] for item in branch_outputs], axis=0)
    conf_stack = np.stack([item["conf"] for item in branch_outputs], axis=0)
    margin_stack = np.stack([item["margin"] for item in branch_outputs], axis=0)
    entropy_stack = np.stack([item["entropy"] for item in branch_outputs], axis=0)
    scale_count, height, width = pred_stack.shape
    flat_count = height * width
    choice = np.zeros(flat_count, dtype=np.uint8)
    with torch.no_grad():
        for start in range(0, flat_count, chunk_pixels):
            end = min(flat_count, start + chunk_pixels)
            batch_pred = torch.from_numpy(pred_stack.reshape(scale_count, -1)[:, start:end].T.astype(np.int64)).to(device)
            batch_conf = torch.from_numpy(conf_stack.reshape(scale_count, -1)[:, start:end].T.astype(np.float32)).to(device)
            batch_margin = torch.from_numpy(margin_stack.reshape(scale_count, -1)[:, start:end].T.astype(np.float32)).to(device)
            batch_entropy = torch.from_numpy(entropy_stack.reshape(scale_count, -1)[:, start:end].T.astype(np.float32)).to(device)
            logits = router(batch_pred, batch_conf, batch_margin, batch_entropy)
            choice[start:end] = logits.argmax(dim=1).to(torch.uint8).cpu().numpy()
    flat_pred = pred_stack.reshape(scale_count, -1)
    routed = flat_pred[choice.astype(np.int64), np.arange(flat_count)]
    return routed.reshape(height, width).astype(np.uint8, copy=False), choice.reshape(height, width)


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


def write_manifest(out_dir, args, scale_specs, counts, routing_stats, zip_info, router_ckpt):
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(ROOT),
        "config_file": str(Path(args.config_file).resolve()),
        "weights": str(Path(args.weights).resolve()),
        "router": str(Path(args.router).resolve()),
        "router_train_args": router_ckpt.get("args", {}),
        "test_root": str(Path(args.test_root).resolve()),
        "out_dir": str(Path(args.out_dir).resolve()),
        "zip": str(Path(args.zip).resolve()) if args.zip else None,
        "device": args.device,
        "sequences": args.sequences,
        "scale_specs": scale_specs,
        "flip": bool(args.flip),
        "routing": "learned_scale_router",
        "counts": counts,
        "routing_stats": routing_stats,
        "zip_info": zip_info,
    }
    with (Path(out_dir) / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")


def main():
    args = parse_args()
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    router, router_scale_specs, router_ckpt = load_router(args.router, device)
    scale_specs = parse_scale_specs(args.scale_specs) if args.scale_specs else router_scale_specs

    out_dir = Path(args.out_dir)
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = list(iter_test_images(args.test_root, args.sequences))
    if args.limit is not None:
        images = images[: args.limit]
    if not images:
        raise ValueError("No test images matched the requested sequences.")

    loaded_images = []
    for seq_name, img_path in images:
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read image: {img_path}")
        loaded_images.append({"seq_name": seq_name, "img_path": img_path, "image": image})

    all_outputs = [[] for _ in loaded_images]
    for spec in scale_specs:
        cfg = setup_cfg(args, spec["min_size"], spec["max_size"])
        predictor = DefaultPredictor(cfg)
        for image_idx, item in enumerate(tqdm(loaded_images, desc=f"export-{spec['name']}")):
            scores = predict_scores(predictor, item["image"], cfg, args.flip)
            all_outputs[image_idx].append(branch_stats(scores, item["image"].shape[:2]))
        del predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    branch_names = [spec["name"] for spec in scale_specs]
    counts = {"total": 0, "sequences": {}, "skipped": 0}
    routing_stats = {"by_sequence": {}, "global_choice_pixels": {name: 0 for name in branch_names}}
    for item, branch_outputs in tqdm(list(zip(loaded_images, all_outputs)), desc="route-write"):
        seq_name = item["seq_name"]
        dst_dir = out_dir / seq_name / "segment_co"
        dst_path = dst_dir / item["img_path"].name
        if args.skip_existing and dst_path.exists():
            counts["skipped"] += 1
            continue
        pred, choice = route_image(router, branch_outputs, device, args.chunk_pixels)
        dst_dir.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(dst_path), pred):
            raise RuntimeError(f"Could not write mask: {dst_path}")
        counts["total"] += 1
        counts["sequences"][seq_name] = counts["sequences"].get(seq_name, 0) + 1

        hist = np.bincount(choice.reshape(-1), minlength=len(branch_names))
        seq_stats = routing_stats["by_sequence"].setdefault(seq_name, {})
        for idx, value in enumerate(hist):
            name = branch_names[idx]
            seq_stats[name] = seq_stats.get(name, 0) + int(value)
            routing_stats["global_choice_pixels"][name] += int(value)

    zip_info = None
    if args.zip:
        zip_info = zip_directory(out_dir, args.zip)
        print(f"Wrote zip: {args.zip}")
        print(f"zip_entries: {zip_info['entries']}")
        print(f"max_zip_path_len: {zip_info['max_path_len']}")
    write_manifest(out_dir, args, scale_specs, counts, routing_stats, zip_info, router_ckpt)
    print(json.dumps({"counts": counts, "routing_stats": routing_stats}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
