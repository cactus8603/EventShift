#!/usr/bin/env python
"""Adjust selected DayEventBoundaryRefiner parameters in a checkpoint."""

import argparse
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--gate-bias", type=float, default=None)
    parser.add_argument("--score-bias", type=float, default=None)
    return parser.parse_args()


def set_scalar(model, key, value):
    if key not in model:
        raise KeyError(f"Missing checkpoint key: {key}")
    tensor = model[key]
    if tensor.numel() != 1:
        raise ValueError(f"Expected scalar-like tensor for {key}, got shape {tuple(tensor.shape)}")
    model[key] = tensor.new_tensor(float(value)).reshape_as(tensor)


def fill_tensor(model, key, value):
    if key not in model:
        raise KeyError(f"Missing checkpoint key: {key}")
    tensor = model[key]
    model[key] = tensor.new_full(tensor.shape, float(value))


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if "model" not in checkpoint:
        raise KeyError("checkpoint does not contain a 'model' key")
    model = checkpoint["model"]

    changes = {}
    if args.alpha is not None:
        key = "day_event_boundary_refiner.alpha"
        changes[key] = {
            "old": float(model[key].reshape(-1)[0]),
            "new": float(args.alpha),
        }
        set_scalar(model, key, args.alpha)
    if args.gate_bias is not None:
        key = "day_event_boundary_refiner.gate_head.bias"
        changes[key] = {
            "old": [float(value) for value in model[key].reshape(-1)],
            "new": float(args.gate_bias),
        }
        fill_tensor(model, key, args.gate_bias)
    if args.score_bias is not None:
        key = "day_event_boundary_refiner.score_predictor.3.bias"
        changes[key] = {
            "old": [float(value) for value in model[key].reshape(-1)],
            "new": float(args.score_bias),
        }
        fill_tensor(model, key, args.score_bias)

    checkpoint.setdefault("retune_meta", {})
    checkpoint["retune_meta"] = {
        **checkpoint["retune_meta"],
        "source_checkpoint": str(Path(args.checkpoint).resolve()),
        "changes": changes,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, out_path)
    print(f"Wrote retuned checkpoint: {out_path}")
    for key, row in changes.items():
        print(f"{key}: {row['old']} -> {row['new']}")


if __name__ == "__main__":
    main()
