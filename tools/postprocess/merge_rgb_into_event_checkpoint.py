#!/usr/bin/env python
"""Build a hybrid checkpoint: RGB semantic weights + event modules."""

import argparse
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-checkpoint", required=True)
    parser.add_argument("--rgb-checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--rgb-prefixes",
        default="backbone.,sem_seg_head.,criterion.",
        help="Comma-separated model key prefixes copied from the RGB checkpoint.",
    )
    return parser.parse_args()


def checkpoint_model(checkpoint):
    if "model" not in checkpoint:
        raise KeyError("checkpoint does not contain a 'model' key")
    return checkpoint["model"]


def main():
    args = parse_args()
    prefixes = tuple(part.strip() for part in args.rgb_prefixes.split(",") if part.strip())
    if not prefixes:
        raise ValueError("at least one RGB prefix is required")

    event_checkpoint = torch.load(args.event_checkpoint, map_location="cpu")
    rgb_checkpoint = torch.load(args.rgb_checkpoint, map_location="cpu")
    event_model = checkpoint_model(event_checkpoint)
    rgb_model = checkpoint_model(rgb_checkpoint)

    copied = []
    skipped_missing = []
    skipped_shape = []
    for key, value in rgb_model.items():
        if not key.startswith(prefixes):
            continue
        if key not in event_model:
            skipped_missing.append(key)
            continue
        if tuple(event_model[key].shape) != tuple(value.shape):
            skipped_shape.append(key)
            continue
        event_model[key] = value
        copied.append(key)

    output = {
        "model": event_model,
        "__author__": "merge_rgb_into_event_checkpoint.py",
        "hybrid_meta": {
            "event_checkpoint": args.event_checkpoint,
            "rgb_checkpoint": args.rgb_checkpoint,
            "rgb_prefixes": list(prefixes),
            "copied_keys": len(copied),
            "skipped_missing": skipped_missing,
            "skipped_shape": skipped_shape,
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, out_path)
    print(f"Wrote {out_path}")
    print(f"Copied RGB keys: {len(copied)}")
    if skipped_missing:
        print(f"Skipped missing keys: {len(skipped_missing)}")
    if skipped_shape:
        print(f"Skipped shape-mismatch keys: {len(skipped_shape)}")


if __name__ == "__main__":
    main()
