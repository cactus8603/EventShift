#!/usr/bin/env python
"""Probe DayEventBoundaryRefiner alpha/gate-bias scales on one fixed batch."""

import argparse
import json
import random
import sys
import importlib.util
from pathlib import Path

import numpy as np
import torch

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
from mask2former import MaskFormerSemanticDatasetMapper, add_maskformer2_config  # noqa: E402
from train_mask2former_cosec import CoSECTrainer, register_cosec  # noqa: E402
from cosec_finetune_splits import CLASSES  # noqa: E402
from overfit_one_batch import evaluate_fixed_batch  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--combo",
        action="append",
        default=[],
        help="alpha,gate_bias pair. Example: --combo 0.12,-1.5",
    )
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


def parse_combos(specs):
    if not specs:
        specs = ["current,current", "0.12,-1.5", "0.25,-1.0", "0.50,-0.5", "0.75,0.0"]
    combos = []
    for spec in specs:
        left, right = [part.strip() for part in spec.split(",", 1)]
        alpha = None if left == "current" else float(left)
        gate_bias = None if right == "current" else float(right)
        combos.append((alpha, gate_bias, spec))
    return combos


def set_refiner_scale(model, alpha, gate_bias):
    refiner = getattr(model, "day_event_boundary_refiner", None)
    if refiner is None:
        raise AttributeError("model has no day_event_boundary_refiner")
    if alpha is not None:
        refiner.alpha.data.fill_(float(alpha))
    if gate_bias is not None:
        refiner.gate_head.bias.data.fill_(float(gate_bias))
    return refiner


def stats_dict(refiner):
    out = {}
    for key, value in getattr(refiner, "last_stats", {}).items():
        try:
            out[key] = float(value.detach().cpu())
        except Exception:
            pass
    return out


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    register_cosec()
    cfg = setup_cfg(args)
    records = DatasetCatalog.get(args.dataset)
    selected = records[args.start_index : args.start_index + args.batch_size]
    if len(selected) != args.batch_size:
        raise ValueError(f"requested batch_size={args.batch_size}, got {len(selected)}")
    mapper = MaskFormerSemanticDatasetMapper(cfg, False)
    fixed_batch = [mapper(record) for record in selected]

    model = CoSECTrainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS,
        resume=False,
    )
    model.eval()

    combos = parse_combos(args.combo)
    rows = []
    reference_predictions = None
    for idx, (alpha, gate_bias, spec) in enumerate(combos):
        refiner = set_refiner_scale(model, alpha, gate_bias)
        metrics, predictions = evaluate_fixed_batch(
            model,
            fixed_batch,
            num_classes=len(CLASSES),
            reference_predictions=reference_predictions,
        )
        if reference_predictions is None:
            reference_predictions = predictions
        row = {
            "combo": spec,
            "alpha": float(refiner.alpha.detach().cpu()),
            "gate_bias": float(refiner.gate_head.bias.detach().cpu().reshape(-1)[0]),
            **metrics,
            "refiner": stats_dict(refiner),
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config_file": args.config_file,
                "dataset": args.dataset,
                "batch_size": args.batch_size,
                "start_index": args.start_index,
                "weights": cfg.MODEL.WEIGHTS,
                "rows": rows,
            },
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")
    print(f"Wrote event refiner scale probe: {out_path}", flush=True)


if __name__ == "__main__":
    main()
