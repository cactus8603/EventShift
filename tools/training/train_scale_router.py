#!/usr/bin/env python
"""Train a lightweight per-pixel router over frozen Mask2Former scale outputs."""

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
    parser.add_argument(
        "--scale-specs",
        default="s512:512:1200,s624:624:1200,s768:768:1400,s1024:1024:1600",
        help="Comma-separated name:min_size:max_size entries.",
    )
    parser.add_argument("--train-dataset", default="cosec_train")
    parser.add_argument("--eval-datasets", default="cosec_day_val,cosec_night_val")
    parser.add_argument("--train-limit", type=int, default=384)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--pixels-per-image", type=int, default=8192)
    parser.add_argument("--batch-pixels", type=int, default=65536)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--class-embed-dim", type=int, default=8)
    parser.add_argument("--nonrepair-weight", type=float, default=0.15)
    parser.add_argument("--agree-weight", type=float, default=0.05)
    parser.add_argument("--flip", action="store_true", help="Average each scale with horizontal flip.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--reuse-train-matrix", action="store_true")
    parser.add_argument("--quiet", action="store_true")
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
    return label.astype(np.int64)


def valid_label_mask(label, ignore_label=255):
    return (label != ignore_label) & (label >= 0) & (label < len(CLASSES))


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


def branch_stats(prob):
    top2 = torch.topk(prob, k=2, dim=0).values
    conf = top2[0].numpy().astype(np.float16, copy=False)
    margin = (top2[0] - top2[1]).numpy().astype(np.float16, copy=False)
    entropy = (-(prob * prob.clamp_min(1e-8).log()).sum(dim=0) / math.log(len(CLASSES)))
    pred = prob.argmax(dim=0).numpy().astype(np.uint8, copy=False)
    return {
        "pred": pred,
        "conf": conf,
        "margin": margin,
        "entropy": entropy.numpy().astype(np.float16, copy=False),
    }


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
        # pred/conf/margin/entropy: [N, K]
        scale_eye = torch.eye(self.num_scales, device=conf.device, dtype=conf.dtype)
        scale_feat = scale_eye.unsqueeze(0).expand(conf.shape[0], -1, -1)
        cls_feat = self.class_embed(pred.long())
        scalar_feat = torch.stack([conf, margin, entropy], dim=-1)
        feat = torch.cat([scalar_feat, scale_feat, cls_feat], dim=-1)
        return self.mlp(feat).squeeze(-1)


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


def prepare_records(dataset_name, limit, seed):
    records = list(DatasetCatalog.get(dataset_name))
    rng = random.Random(seed)
    rng.shuffle(records)
    if limit is not None:
        records = records[:limit]
    return records


def collect_sample_matrix(args, scale_specs, records, labels, coords):
    offsets = np.cumsum([0] + [len(item[0]) for item in coords])
    total = int(offsets[-1])
    scale_count = len(scale_specs)
    pred = np.zeros((total, scale_count), dtype=np.uint8)
    conf = np.zeros((total, scale_count), dtype=np.float16)
    margin = np.zeros((total, scale_count), dtype=np.float16)
    entropy = np.zeros((total, scale_count), dtype=np.float16)
    target = np.concatenate(
        [label[ys, xs].astype(np.int64, copy=False) for label, (ys, xs) in zip(labels, coords)]
    )

    for scale_idx, spec in enumerate(scale_specs):
        print(
            f"[collect-train] scale {scale_idx + 1}/{len(scale_specs)} "
            f"{spec['name']} min={spec['min_size']} max={spec['max_size']} records={len(records)}",
            flush=True,
        )
        cfg = setup_cfg(args, spec["min_size"], spec["max_size"])
        mapper = MaskFormerSemanticDatasetMapper(cfg, False)
        model = build_model(cfg)
        iterator = zip(records, labels, coords)
        if not args.quiet:
            iterator = tqdm(list(iterator), desc=f"collect-train-{spec['name']}")
        for image_idx, (record, label, (ys, xs)) in enumerate(iterator):
            if len(ys) == 0:
                continue
            start, end = offsets[image_idx], offsets[image_idx + 1]
            mapped = mapper(copy.deepcopy(record))
            scores = infer_scores(model, mapped, args.flip)
            prob = normalize_scores(resize_scores(scores, label.shape))
            stats = branch_stats(prob)
            pred[start:end, scale_idx] = stats["pred"][ys, xs]
            conf[start:end, scale_idx] = stats["conf"][ys, xs]
            margin[start:end, scale_idx] = stats["margin"][ys, xs]
            entropy[start:end, scale_idx] = stats["entropy"][ys, xs]
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[collect-train] done {spec['name']}", flush=True)

    return {
        "pred": pred,
        "conf": conf,
        "margin": margin,
        "entropy": entropy,
        "target": target,
    }


def make_router_targets(matrix, nonrepair_weight, agree_weight):
    pred = matrix["pred"].astype(np.int64)
    conf = matrix["conf"].astype(np.float32)
    target_class = matrix["target"].astype(np.int64)
    correct = pred == target_class[:, None]
    any_correct = correct.any(axis=1)

    masked_conf = np.where(correct, conf, -1.0)
    target_branch = masked_conf.argmax(axis=1).astype(np.int64)
    fallback = conf.argmax(axis=1).astype(np.int64)
    target_branch = np.where(any_correct, target_branch, fallback)

    disagree = (pred != pred[:, :1]).any(axis=1)
    weight = np.ones(len(target_branch), dtype=np.float32)
    weight[~any_correct] = float(nonrepair_weight)
    weight[~disagree] = np.minimum(weight[~disagree], float(agree_weight))
    weight = np.clip(weight, 0.0, 1.0)
    return target_branch, weight


def train_router(args, matrix, scale_specs):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    router = ScaleRouter(
        num_scales=len(scale_specs),
        num_classes=len(CLASSES),
        class_embed_dim=args.class_embed_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(router.parameters(), lr=args.lr, weight_decay=1e-4)

    target_branch, weight = make_router_targets(
        matrix,
        nonrepair_weight=args.nonrepair_weight,
        agree_weight=args.agree_weight,
    )
    pred = torch.from_numpy(matrix["pred"].astype(np.int64))
    conf = torch.from_numpy(matrix["conf"].astype(np.float32))
    margin = torch.from_numpy(matrix["margin"].astype(np.float32))
    entropy = torch.from_numpy(matrix["entropy"].astype(np.float32))
    target = torch.from_numpy(target_branch)
    sample_weight = torch.from_numpy(weight)

    n = len(target_branch)
    history = []
    for epoch in range(args.epochs):
        order = torch.randperm(n)
        total_loss = 0.0
        total_weight = 0.0
        correct = 0
        seen = 0
        for start in range(0, n, args.batch_pixels):
            idx = order[start : start + args.batch_pixels]
            batch_pred = pred[idx].to(device, non_blocking=True)
            batch_conf = conf[idx].to(device, non_blocking=True)
            batch_margin = margin[idx].to(device, non_blocking=True)
            batch_entropy = entropy[idx].to(device, non_blocking=True)
            batch_target = target[idx].to(device, non_blocking=True)
            batch_weight = sample_weight[idx].to(device, non_blocking=True)

            logits = router(batch_pred, batch_conf, batch_margin, batch_entropy)
            loss_vec = F.cross_entropy(logits, batch_target, reduction="none")
            loss = (loss_vec * batch_weight).sum() / batch_weight.sum().clamp_min(1e-6)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float((loss_vec.detach() * batch_weight).sum().item())
            total_weight += float(batch_weight.sum().item())
            correct += int((logits.argmax(dim=1) == batch_target).sum().item())
            seen += int(len(idx))

        row = {
            "epoch": epoch + 1,
            "loss": total_loss / max(total_weight, 1.0),
            "target_acc": correct / max(seen, 1),
        }
        history.append(row)
        print(
            f"[router] epoch {row['epoch']}: loss={row['loss']:.5f}, "
            f"target_acc={100.0 * row['target_acc']:.2f}%",
            flush=True,
        )
    return router, history


def collect_eval_outputs(args, spec, records, labels):
    print(
        f"[collect-eval] {spec['name']} min={spec['min_size']} max={spec['max_size']} "
        f"records={len(records)}",
        flush=True,
    )
    cfg = setup_cfg(args, spec["min_size"], spec["max_size"])
    mapper = MaskFormerSemanticDatasetMapper(cfg, False)
    model = build_model(cfg)
    outputs = []
    iterator = records if args.quiet else tqdm(records, desc=f"eval-{spec['name']}")
    for record, label in zip(iterator, labels):
        mapped = mapper(copy.deepcopy(record))
        scores = infer_scores(model, mapped, args.flip)
        prob = normalize_scores(resize_scores(scores, label.shape))
        outputs.append(branch_stats(prob))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[collect-eval] done {spec['name']}", flush=True)
    return outputs


def evaluate_predictions(preds, labels):
    meter = ConfusionMeter(num_classes=len(CLASSES))
    for pred, label in zip(preds, labels):
        meter.update(pred, label)
    return meter.metrics()


def route_image(router, outputs_by_branch, image_idx, device, chunk_pixels=262144):
    pred_stack = np.stack([branch[image_idx]["pred"] for branch in outputs_by_branch], axis=0)
    conf_stack = np.stack([branch[image_idx]["conf"] for branch in outputs_by_branch], axis=0)
    margin_stack = np.stack([branch[image_idx]["margin"] for branch in outputs_by_branch], axis=0)
    entropy_stack = np.stack([branch[image_idx]["entropy"] for branch in outputs_by_branch], axis=0)
    scale_count, height, width = pred_stack.shape
    flat_count = height * width
    chosen = np.zeros(flat_count, dtype=np.uint8)
    router.eval()
    with torch.no_grad():
        for start in range(0, flat_count, chunk_pixels):
            end = min(flat_count, start + chunk_pixels)
            batch_pred = torch.from_numpy(pred_stack.reshape(scale_count, -1)[:, start:end].T.astype(np.int64)).to(device)
            batch_conf = torch.from_numpy(conf_stack.reshape(scale_count, -1)[:, start:end].T.astype(np.float32)).to(device)
            batch_margin = torch.from_numpy(margin_stack.reshape(scale_count, -1)[:, start:end].T.astype(np.float32)).to(device)
            batch_entropy = torch.from_numpy(entropy_stack.reshape(scale_count, -1)[:, start:end].T.astype(np.float32)).to(device)
            scores = router(batch_pred, batch_conf, batch_margin, batch_entropy)
            chosen[start:end] = scores.argmax(dim=1).to(torch.uint8).cpu().numpy()
    flat_pred = pred_stack.reshape(scale_count, -1)
    routed = flat_pred[chosen.astype(np.int64), np.arange(flat_count)]
    return routed.reshape(height, width).astype(np.uint8, copy=False), chosen.reshape(height, width)


def evaluate_router(args, router, scale_specs, dataset_name):
    records = list(DatasetCatalog.get(dataset_name))
    if args.eval_limit is not None:
        records = records[: args.eval_limit]
    labels = [load_label(record) for record in records]
    outputs_by_branch = [
        collect_eval_outputs(args, spec, records, labels)
        for spec in scale_specs
    ]
    branch_names = [spec["name"] for spec in scale_specs]
    methods = OrderedDict()
    for name, branch in zip(branch_names, outputs_by_branch):
        methods[name] = evaluate_predictions([output["pred"] for output in branch], labels)

    conf_preds = []
    router_preds = []
    choice_pixels = {name: 0 for name in branch_names}
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    for image_idx in range(len(records)):
        pred_stack = np.stack([branch[image_idx]["pred"] for branch in outputs_by_branch], axis=0)
        conf_stack = np.stack([branch[image_idx]["conf"] for branch in outputs_by_branch], axis=0)
        conf_choice = conf_stack.argmax(axis=0)
        conf_preds.append(
            np.take_along_axis(pred_stack, conf_choice[None], axis=0)[0].astype(np.uint8, copy=False)
        )
        routed, choice = route_image(router, outputs_by_branch, image_idx, device)
        router_preds.append(routed)
        hist = np.bincount(choice.reshape(-1), minlength=len(branch_names))
        for idx, value in enumerate(hist):
            choice_pixels[branch_names[idx]] += int(value)

    methods["choose_highest_conf"] = evaluate_predictions(conf_preds, labels)
    methods["learned_router"] = evaluate_predictions(router_preds, labels)
    return {
        "dataset": dataset_name,
        "sample_count": len(records),
        "methods": methods,
        "top_by_mIoU": [
            {"method": name, **values}
            for name, values in sorted(methods.items(), key=lambda item: item[1]["mIoU"], reverse=True)
        ],
        "router_choice_pixels": choice_pixels,
    }


def write_markdown(output, out_dir):
    lines = [
        "# Scale Router Diagnostic",
        "",
        f"created_at: `{output['created_at']}`",
        f"config: `{output['args']['config_file']}`",
        f"weights: `{output['args']['weights']}`",
        f"flip: `{output['args']['flip']}`",
        "",
        "## Train",
        "",
        f"- dataset: `{output['args']['train_dataset']}`",
        f"- train_limit: `{output['args']['train_limit']}`",
        f"- pixels_per_image: `{output['args']['pixels_per_image']}`",
        "",
        "| Epoch | Loss | Target Acc |",
        "|---:|---:|---:|",
    ]
    for row in output["train_history"]:
        lines.append(f"| {row['epoch']} | {row['loss']:.5f} | {100.0 * row['target_acc']:.2f}% |")
    for dataset in output["datasets"]:
        lines.extend(
            [
                "",
                f"## {dataset['dataset']}",
                "",
                "| Method | mIoU | mAcc | aAcc |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in dataset["top_by_mIoU"]:
            lines.append(f"| `{row['method']}` | {row['mIoU']:.4f} | {row['mAcc']:.4f} | {row['aAcc']:.4f} |")
        lines.extend(["", "Router choice pixels:", ""])
        for name, count in dataset["router_choice_pixels"].items():
            lines.append(f"- `{name}`: {count}")
    md_path = Path(out_dir) / "summary.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def main():
    args = parse_args()
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    register_cosec()

    scale_specs = parse_scale_specs(args.scale_specs)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = out_dir / "train_matrix.npz"
    if args.reuse_train_matrix and matrix_path.exists():
        print(f"[matrix] loading cached train matrix: {matrix_path}", flush=True)
        loaded = np.load(matrix_path)
        matrix = {name: loaded[name] for name in loaded.files}
    else:
        train_records = prepare_records(args.train_dataset, args.train_limit, args.seed)
        labels = [load_label(record) for record in train_records]
        rng = np.random.default_rng(args.seed)
        coords = [sample_pixels(label, args.pixels_per_image, rng) for label in labels]
        matrix = collect_sample_matrix(args, scale_specs, train_records, labels, coords)
        np.savez_compressed(matrix_path, **matrix)
        print(f"[matrix] wrote train matrix: {matrix_path}", flush=True)

    router, history = train_router(args, matrix, scale_specs)
    router_path = out_dir / "scale_router.pth"
    torch.save(
        {
            "model": router.state_dict(),
            "scale_specs": scale_specs,
            "classes": list(CLASSES),
            "args": vars(args),
        },
        router_path,
    )

    output = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "scale_specs": scale_specs,
        "router_path": str(router_path),
        "train_history": history,
        "datasets": [
            evaluate_router(args, router, scale_specs, dataset_name)
            for dataset_name in split_csv(args.eval_datasets)
        ],
    }
    json_path = out_dir / "metrics.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, sort_keys=True)
        f.write("\n")
    md_path = write_markdown(output, out_dir)
    print(f"Wrote router: {router_path}")
    print(f"Wrote metrics: {json_path}")
    print(f"Wrote summary: {md_path}")
    for dataset in output["datasets"]:
        print(f"[{dataset['dataset']}]")
        for row in dataset["top_by_mIoU"][:8]:
            print(f"  {row['method']}: mIoU={row['mIoU']:.4f}")


if __name__ == "__main__":
    main()
