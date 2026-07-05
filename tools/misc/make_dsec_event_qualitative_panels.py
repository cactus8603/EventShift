#!/usr/bin/env python
"""Create DSEC qualitative panels from cached semantic prediction JSON files."""

import argparse
import json
import os
import sys
import importlib.util
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_util

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

DEFAULT_HDF5_PLUGIN_PATH = Path(
    "/path/to/hdf5plugin/plugins"
)
if "HDF5_PLUGIN_PATH" not in os.environ and DEFAULT_HDF5_PLUGIN_PATH.exists():
    os.environ["HDF5_PLUGIN_PATH"] = str(DEFAULT_HDF5_PLUGIN_PATH)

from cosec_event_dataset import load_event_edge_representation  # noqa: E402
from detectron2.data import DatasetCatalog  # noqa: E402
from train_mask2former_cosec import register_cosec  # noqa: E402


CITYSCAPES_PALETTE = np.array(
    [
        (128, 64, 128),   # road
        (244, 35, 232),   # sidewalk
        (70, 70, 70),     # building
        (102, 102, 156),  # wall
        (190, 153, 153),  # fence
        (153, 153, 153),  # pole
        (250, 170, 30),   # traffic light
        (220, 220, 0),    # traffic sign
        (107, 142, 35),   # vegetation
        (152, 251, 152),  # terrain
        (70, 130, 180),   # sky
        (220, 20, 60),    # person
        (255, 0, 0),      # rider
        (0, 0, 142),      # car
        (0, 0, 70),       # truck
        (0, 60, 100),     # bus
        (0, 80, 100),     # train
        (0, 0, 230),      # motorcycle
        (119, 11, 32),    # bicycle
    ],
    dtype=np.uint8,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dsec19_val_event")
    parser.add_argument("--rgb-json", required=True)
    parser.add_argument("--full-event-json", required=True)
    parser.add_argument("--naive-json", required=True)
    parser.add_argument("--no-reliability-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--tile-width", type=int, default=420)
    parser.add_argument("--ignore-label", type=int, default=255)
    parser.add_argument("--event-radii-ms", nargs="*", type=int, default=[50])
    parser.add_argument("--no-event-edge", action="store_true")
    return parser.parse_args()


def load_prediction_json(path):
    grouped = {}
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    for row in rows:
        grouped.setdefault(row["file_name"], []).append(row)
    return grouped


def decode_prediction(rows, shape, fill_value=255):
    pred = np.full(shape, fill_value, dtype=np.int64)
    for row in rows:
        rle = dict(row["segmentation"])
        if isinstance(rle.get("counts"), str):
            rle["counts"] = rle["counts"].encode("ascii")
        mask = mask_util.decode(rle).astype(bool)
        if mask.shape != shape:
            raise RuntimeError(f"Mask shape mismatch: got {mask.shape}, expected {shape}")
        pred[mask] = int(row["category_id"])
    return pred


def load_label(path):
    label = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if label is None:
        raise RuntimeError(f"Could not read label: {path}")
    if label.ndim == 3:
        label = label[:, :, 0]
    return label.astype(np.int64)


def valid_mask(label, ignore_label):
    return (label != ignore_label) & (label >= 0) & (label < len(CITYSCAPES_PALETTE))


def colorize_label(label, invalid=(0, 0, 0)):
    out = np.zeros((*label.shape, 3), dtype=np.uint8)
    out[:] = invalid
    keep = (label >= 0) & (label < len(CITYSCAPES_PALETTE))
    out[keep] = CITYSCAPES_PALETTE[label[keep]]
    return out


def repair_map(base_pred, new_pred, label, valid):
    out = np.zeros((*label.shape, 3), dtype=np.uint8)
    out[:] = (24, 24, 24)
    changed = valid & (base_pred != new_pred)
    repaired = valid & (base_pred != label) & (new_pred == label)
    damaged = valid & (base_pred == label) & (new_pred != label)
    neutral = changed & ~(repaired | damaged)
    out[neutral] = (230, 190, 60)
    out[repaired] = (40, 210, 90)
    out[damaged] = (220, 45, 45)
    return out


def event_edge_tile(record, image_shape, radii_ms):
    channels = load_event_edge_representation(record, image_shape, radii_ms)
    if channels.size == 0:
        return np.zeros((*image_shape[:2], 3), dtype=np.uint8)
    # For each radius the channels are density, edge_score, polarity. Use edge_score.
    edge = channels[1]
    edge = edge.astype(np.float32)
    if float(edge.max()) > 1e-6:
        edge = edge / float(edge.max())
    edge_u8 = np.clip(edge * 255.0, 0, 255).astype(np.uint8)
    edge_u8 = cv2.GaussianBlur(edge_u8, (3, 3), 0)
    return cv2.applyColorMap(edge_u8, cv2.COLORMAP_INFERNO)


def resize_tile(tile, width):
    h, w = tile.shape[:2]
    scale = float(width) / float(w)
    height = max(1, int(round(h * scale)))
    return cv2.resize(tile, (width, height), interpolation=cv2.INTER_NEAREST)


def add_title(tile, title):
    pad = 34
    out = np.zeros((tile.shape[0] + pad, tile.shape[1], 3), dtype=np.uint8)
    out[:] = (28, 28, 28)
    out[pad:] = tile
    cv2.putText(out, title, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 2, cv2.LINE_AA)
    return out


def make_grid(tiles, cols=4):
    rows = []
    for start in range(0, len(tiles), cols):
        row = tiles[start : start + cols]
        max_h = max(tile.shape[0] for tile in row)
        padded = []
        for tile in row:
            if tile.shape[0] < max_h:
                extra = np.zeros((max_h - tile.shape[0], tile.shape[1], 3), dtype=np.uint8)
                extra[:] = (28, 28, 28)
                tile = np.concatenate([tile, extra], axis=0)
            padded.append(tile)
        rows.append(np.concatenate(padded, axis=1))
    max_w = max(row.shape[1] for row in rows)
    padded_rows = []
    for row in rows:
        if row.shape[1] < max_w:
            extra = np.zeros((row.shape[0], max_w - row.shape[1], 3), dtype=np.uint8)
            extra[:] = (28, 28, 28)
            row = np.concatenate([row, extra], axis=1)
        padded_rows.append(row)
    return np.concatenate(padded_rows, axis=0)


def score_record(record, rows):
    label = load_label(record["sem_seg_file_name"])
    valid = valid_mask(label, args.ignore_label)
    base = decode_prediction(rows["rgb"][record["file_name"]], label.shape, args.ignore_label)
    full = decode_prediction(rows["full"][record["file_name"]], label.shape, args.ignore_label)
    repaired = valid & (base != label) & (full == label)
    damaged = valid & (base == label) & (full != label)
    changed = valid & (base != full)
    return {
        "file_name": record["file_name"],
        "label": record["sem_seg_file_name"],
        "image_id": record.get("image_id", ""),
        "sequence": record.get("sequence", ""),
        "net_repaired": int(repaired.sum() - damaged.sum()),
        "repaired": int(repaired.sum()),
        "damaged": int(damaged.sum()),
        "changed": int(changed.sum()),
        "changed_rate": float(changed.sum() / max(1, valid.sum())),
    }


def build_panel(record, rows, out_path, tile_width, include_event_edge=True, event_radii_ms=None):
    rgb_bgr = cv2.imread(record["file_name"], cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise RuntimeError(f"Could not read image: {record['file_name']}")
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    label = load_label(record["sem_seg_file_name"])
    valid = valid_mask(label, args.ignore_label)

    base = decode_prediction(rows["rgb"][record["file_name"]], label.shape, args.ignore_label)
    full = decode_prediction(rows["full"][record["file_name"]], label.shape, args.ignore_label)
    naive = decode_prediction(rows["naive"][record["file_name"]], label.shape, args.ignore_label)
    no_rel = decode_prediction(rows["no_rel"][record["file_name"]], label.shape, args.ignore_label)

    tiles = [
        ("RGB", rgb),
        ("GT", colorize_label(label)),
        ("F1 RGB", colorize_label(base)),
        ("F2 full event", colorize_label(full)),
        ("F3 naive", colorize_label(naive)),
        ("F4 no reliability", colorize_label(no_rel)),
        ("F2 repair map", repair_map(base, full, label, valid)),
    ]
    if include_event_edge:
        try:
            tiles.insert(1, ("Event edge", event_edge_tile(record, rgb.shape, event_radii_ms or [50])))
        except Exception as exc:  # Keep qualitative export usable even if H5 plugin is unavailable.
            blank = np.zeros_like(rgb)
            cv2.putText(blank, f"event unavailable: {type(exc).__name__}", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            tiles.insert(1, ("Event edge", blank))

    rendered = []
    for title, tile in tiles:
        tile = resize_tile(tile, tile_width)
        # cv2.imwrite expects BGR.
        if title in {"RGB", "GT", "F1 RGB", "F2 full event", "F3 naive", "F4 no reliability", "F2 repair map"}:
            tile = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
        rendered.append(add_title(tile, title))
    panel = make_grid(rendered, cols=4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), panel)


def main(args):
    register_cosec()
    records = DatasetCatalog.get(args.dataset)
    rows = {
        "rgb": load_prediction_json(args.rgb_json),
        "full": load_prediction_json(args.full_event_json),
        "naive": load_prediction_json(args.naive_json),
        "no_rel": load_prediction_json(args.no_reliability_json),
    }

    usable = []
    for record in records:
        file_name = record["file_name"]
        if all(file_name in group for group in rows.values()):
            usable.append((score_record(record, rows), record))
    usable.sort(key=lambda item: item[0]["net_repaired"], reverse=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for rank, (score, record) in enumerate(usable[: args.top_k], start=1):
        out_path = out_dir / f"{rank:02d}_{Path(record['file_name']).stem}_{score['net_repaired']:+d}.png"
        build_panel(
            record,
            rows,
            out_path,
            tile_width=args.tile_width,
            include_event_edge=not args.no_event_edge,
            event_radii_ms=args.event_radii_ms,
        )
        score["panel"] = str(out_path)
        index.append(score)

    with (out_dir / "index.json").open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    print(f"Wrote {len(index)} panels to {out_dir}")
    if index:
        best = index[0]
        print(
            "best:",
            best["panel"],
            "net_repaired=",
            best["net_repaired"],
            "repaired=",
            best["repaired"],
            "damaged=",
            best["damaged"],
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
