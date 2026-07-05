#!/usr/bin/env python
"""Build per-image event-edge score caches for CoSEC event datasets."""

import argparse
import json
import sys
import importlib.util
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
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

from cosec_event_dataset import load_event_edge_representation  # noqa: E402
from detectron2.data import DatasetCatalog  # noqa: E402
from train_mask2former_cosec import register_cosec  # noqa: E402


def split_csv(text):
    return [part.strip() for part in str(text).split(",") if part.strip()]


def cache_record_stem(record, _image_idx=None):
    if record.get("image_id"):
        stem = str(record["image_id"])
    else:
        stem = str(record.get("file_name", "image"))
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)
    return safe


def cache_path(out_dir, dataset_name, record, image_idx):
    return Path(out_dir) / dataset_name / f"{cache_record_stem(record, image_idx)}.npz"


def read_label_shape(record):
    label = cv2.imread(record["sem_seg_file_name"], cv2.IMREAD_UNCHANGED)
    if label is None:
        raise RuntimeError(f"Could not read label: {record['sem_seg_file_name']}")
    return label.shape[:2]


def shift_image(array, dx=0, dy=0, fill_value=0):
    dx = int(dx)
    dy = int(dy)
    if dx == 0 and dy == 0:
        return array
    shifted = np.full_like(array, fill_value)
    height, width = array.shape[:2]
    src_x0 = max(0, -dx)
    src_x1 = min(width, width - dx)
    dst_x0 = max(0, dx)
    dst_x1 = min(width, width + dx)
    src_y0 = max(0, -dy)
    src_y1 = min(height, height - dy)
    dst_y0 = max(0, dy)
    dst_y1 = min(height, height + dy)
    if src_x1 > src_x0 and src_y1 > src_y0:
        shifted[dst_y0:dst_y1, dst_x0:dst_x1] = array[src_y0:src_y1, src_x0:src_x1]
    return shifted


def event_edge_score(record, shape, radii, time_offset_ms=0.0, spatial_shift=(0, 0)):
    channels = load_event_edge_representation(
        record,
        shape,
        radii,
        time_offset_ms=float(time_offset_ms),
    )
    if channels.shape[0] == 0:
        return np.zeros(shape, dtype=np.float32)
    # load_event_edge_representation returns [density, edge_score, polarity] per radius.
    edge_channels = channels[1::3]
    if edge_channels.shape[0] == 0:
        edge_channels = channels
    score = edge_channels.max(axis=0).astype(np.float32, copy=False)
    max_value = float(score.max())
    if max_value > 1e-6:
        score = score / max_value
    dx, dy = [int(value) for value in spatial_shift]
    score = shift_image(score, dx=dx, dy=dy, fill_value=0)
    return score.astype(np.float32, copy=False)


def threshold_score(score, percentile):
    nonzero = score[score > 0]
    if nonzero.size == 0:
        return np.zeros(score.shape, dtype=np.uint8), 0.0
    threshold = float(np.percentile(nonzero, float(percentile)))
    return (score >= threshold).astype(np.uint8), threshold


def path_signature(path_text):
    path = Path(path_text)
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def write_manifest(out_dir, dataset_name, records, args, count):
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_name": dataset_name,
        "record_count": len(records),
        "written_count": count,
        "window_radii_ms": [int(v) for v in args.window_radii_ms],
        "time_offset_ms": float(args.time_offset_ms),
        "spatial_shift_xy": [int(value) for value in args.spatial_shift],
        "percentile": float(args.percentile),
        "records": [
            {
                "file_name": path_signature(record.get("file_name", "")),
                "sem_seg_file_name": path_signature(record.get("sem_seg_file_name", "")),
                "event_h5": path_signature(record.get("event_h5", "")),
            }
            for record in records
        ],
    }
    path = Path(out_dir) / dataset_name / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def build_dataset_cache(dataset_name, args):
    records = list(DatasetCatalog.get(dataset_name))
    if args.limit is not None:
        records = records[: args.limit]
    out_dir = Path(args.out_dir) / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    iterator = records if args.quiet else tqdm(records, desc=dataset_name)
    written = 0
    for image_idx, record in enumerate(iterator):
        out_path = cache_path(args.out_dir, dataset_name, record, image_idx)
        if args.reuse and out_path.exists():
            continue
        for key in ("event_h5", "event_old", "event_new"):
            if key not in record:
                raise KeyError(f"{dataset_name} record lacks {key}; use *_event datasets.")
        shape = read_label_shape(record)
        score = event_edge_score(
            record,
            shape,
            args.window_radii_ms,
            time_offset_ms=args.time_offset_ms,
            spatial_shift=args.spatial_shift,
        )
        mask, threshold = threshold_score(score, args.percentile)
        np.savez_compressed(
            out_path,
            score=score.astype(np.float16, copy=False),
            mask=mask,
            threshold=np.asarray(threshold, dtype=np.float32),
        )
        written += 1
    manifest_path = write_manifest(args.out_dir, dataset_name, records, args, written)
    return {
        "dataset": dataset_name,
        "records": len(records),
        "written": written,
        "manifest": str(manifest_path),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        default="cosec_train_event,cosec_day_val_event,cosec_night_val_event",
        help="Comma-separated event dataset names.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--window-radii-ms", nargs="+", type=int, default=[25, 50])
    parser.add_argument(
        "--time-offset-ms",
        type=float,
        default=0.0,
        help="Temporal calibration offset added to event windows. Positive uses later events.",
    )
    parser.add_argument(
        "--spatial-shift",
        nargs=2,
        type=int,
        default=[0, 0],
        metavar=("DX", "DY"),
        help="Spatial calibration shift applied after event accumulation. Positive dx/dy move right/down.",
    )
    parser.add_argument("--percentile", type=float, default=80.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    register_cosec()
    rows = [build_dataset_cache(dataset_name, args) for dataset_name in split_csv(args.datasets)]
    print("# Event-edge cache")
    for row in rows:
        print(
            f"{row['dataset']}: records={row['records']} written={row['written']} "
            f"manifest={row['manifest']}"
        )


if __name__ == "__main__":
    main()
