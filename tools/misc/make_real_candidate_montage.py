#!/usr/bin/env python3
"""Build qualitative montages for REAL prediction candidates."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from analyze_real_prediction_candidates import (
    collect_masks,
    load_mask,
    parse_candidate,
)
from cosec_finetune_splits import PALETTE


def colorize(mask):
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for idx, color in enumerate(PALETTE):
        out[mask == idx] = color
    out[(mask < 0) | (mask >= len(PALETTE))] = (0, 0, 0)
    return Image.fromarray(out, mode="RGB")


def fit_cell(image, width, height, label):
    image = image.convert("RGB")
    image = ImageOps.contain(image, (width, height - 24), method=Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (width, height), (20, 20, 20))
    x = (width - image.width) // 2
    canvas.paste(image, (x, 24))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width, 23), fill=(35, 35, 35))
    draw.text((6, 5), label, fill=(240, 240, 240))
    return canvas


def load_rgb(test_root, frame_key):
    seq, name = frame_key.split("/", 1)
    path = test_root / seq / "img_co_left" / name
    if not path.exists():
        return None
    return Image.open(path).convert("RGB")


def change_overlay(rgb, mask_a, mask_b):
    rgb_arr = np.asarray(rgb.resize((mask_a.shape[1], mask_a.shape[0]))).astype(np.float32)
    dim = (rgb_arr * 0.35).astype(np.uint8)
    changed = mask_a != mask_b
    dim[changed] = (255, 40, 40)
    return Image.fromarray(dim, mode="RGB")


def select_frames(analysis_json, pair, topn):
    with analysis_json.open("r", encoding="utf-8") as f:
        report = json.load(f)
    a_name, b_name = pair.split(":", 1)
    for item in report.get("pairwise", []):
        if item.get("candidate_a") == a_name and item.get("candidate_b") == b_name:
            return [row["frame"] for row in item.get("top_changed_frames", [])[:topn]]
    raise RuntimeError(f"Pair not found in analysis JSON: {pair}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-json", type=Path, required=True)
    parser.add_argument("--pair", required=True, help="candidate_a:candidate_b")
    parser.add_argument("--candidate", action="append", type=parse_candidate, required=True)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--topn", type=int, default=5)
    parser.add_argument("--cell-width", type=int, default=300)
    parser.add_argument("--cell-height", type=int, default=210)
    args = parser.parse_args()

    frames = select_frames(args.analysis_json, args.pair, args.topn)
    candidate_masks = {}
    for name, path in args.candidate:
        masks, _ = collect_masks(path, real_only=True)
        candidate_masks[name] = masks

    pair_a, pair_b = args.pair.split(":", 1)
    columns = ["rgb"] + [name for name, _ in args.candidate] + [f"changed {pair_a}->{pair_b}"]
    rows = []
    for frame in frames:
        rgb = load_rgb(args.test_root, frame)
        if rgb is None:
            continue
        row_cells = [fit_cell(rgb, args.cell_width, args.cell_height, frame)]
        loaded_masks = {}
        for name, _ in args.candidate:
            mask_path = candidate_masks[name].get(frame)
            if mask_path is None:
                continue
            mask = load_mask(mask_path)
            loaded_masks[name] = mask
            row_cells.append(fit_cell(colorize(mask), args.cell_width, args.cell_height, name))
        if pair_a in loaded_masks and pair_b in loaded_masks:
            overlay = change_overlay(rgb, loaded_masks[pair_a], loaded_masks[pair_b])
            row_cells.append(
                fit_cell(overlay, args.cell_width, args.cell_height, columns[-1])
            )
        rows.append(row_cells)

    if not rows:
        raise RuntimeError("No montage rows created")

    width = max(len(row) for row in rows) * args.cell_width
    height = len(rows) * args.cell_height
    montage = Image.new("RGB", (width, height), (12, 12, 12))
    for y, row in enumerate(rows):
        for x, cell in enumerate(row):
            montage.paste(cell, (x * args.cell_width, y * args.cell_height))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    montage.save(args.out)
    print(f"wrote {args.out}")
    print("frames:")
    for frame in frames:
        print(f"  {frame}")


if __name__ == "__main__":
    main()
