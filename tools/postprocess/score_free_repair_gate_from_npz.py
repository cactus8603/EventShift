#!/usr/bin/env python3
"""Learn transition repair gates from validation prediction npz maps.

This is intentionally score-free with respect to hidden submissions: it only
uses validation ground truth to decide which base->candidate transitions are
repair-positive.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import importlib.util
from collections import Counter, OrderedDict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

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
from detectron2.data import DatasetCatalog  # noqa: E402
from train_mask2former_cosec import register_cosec  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--base-map-dir", required=True)
    parser.add_argument("--candidate-map-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow-pairs-out", default=None)
    parser.add_argument("--min-net", type=int, default=1)
    parser.add_argument("--min-precision", type=float, default=0.50)
    parser.add_argument("--min-changed", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=80)
    parser.add_argument(
        "--component-min-boundary5-rate",
        type=float,
        default=None,
        help="If set, evaluate merged val predictions with a component boundary gate.",
    )
    parser.add_argument(
        "--component-max-area",
        type=int,
        default=None,
        help="Also keep components up to this area when component gating is enabled.",
    )
    return parser.parse_args()


def read_label(path: str) -> np.ndarray:
    label = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if label is None:
        raise RuntimeError(f"Could not read label: {path}")
    if label.ndim == 3:
        label = label[:, :, 0]
    return label.astype(np.int64, copy=False)


def read_pred(path: Path) -> np.ndarray:
    data = np.load(path)
    if "pred" not in data.files:
        raise KeyError(f"{path} does not contain a 'pred' array; keys={data.files}")
    return np.asarray(data["pred"], dtype=np.int64)


def valid_mask(label: np.ndarray, *preds: np.ndarray) -> np.ndarray:
    valid = (label >= 0) & (label < len(CLASSES)) & (label != 255)
    for pred in preds:
        valid &= (pred >= 0) & (pred < len(CLASSES))
    return valid


def pair_name(pair_id: int) -> str:
    src = pair_id // len(CLASSES)
    dst = pair_id % len(CLASSES)
    return f"{CLASSES[src]}->{CLASSES[dst]}"


def semantic_boundary(mask: np.ndarray) -> np.ndarray:
    boundary = np.zeros(mask.shape, dtype=bool)
    boundary[:-1, :] |= mask[:-1, :] != mask[1:, :]
    boundary[1:, :] |= mask[:-1, :] != mask[1:, :]
    boundary[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    boundary[:, 1:] |= mask[:, :-1] != mask[:, 1:]
    return boundary


def boundary5(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    structure = np.ones((11, 11), dtype=bool)
    return ndimage.binary_dilation(semantic_boundary(mask_a) | semantic_boundary(mask_b), structure=structure)


def component_gate(
    take: np.ndarray,
    base: np.ndarray,
    candidate: np.ndarray,
    min_boundary5_rate: float | None,
    max_area: int | None,
) -> np.ndarray:
    if min_boundary5_rate is None and max_area is None:
        return take
    labels, count = ndimage.label(take, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return take
    b5 = boundary5(base, candidate)
    kept = np.zeros_like(take, dtype=bool)
    for label_id, slc in enumerate(ndimage.find_objects(labels), start=1):
        if slc is None:
            continue
        comp = labels[slc] == label_id
        area = int(comp.sum())
        b_rate = float((comp & b5[slc]).sum() / area) if area else 0.0
        keep_by_boundary = min_boundary5_rate is not None and b_rate >= min_boundary5_rate
        keep_by_area = max_area is not None and area <= max_area
        if keep_by_boundary or keep_by_area:
            kept[slc] |= comp
    return kept


class ConfusionMeter:
    def __init__(self, num_classes: int = 19):
        self.num_classes = int(num_classes)
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, pred: np.ndarray, label: np.ndarray) -> None:
        keep = valid_mask(label, pred)
        indices = self.num_classes * label[keep].astype(np.int64) + pred[keep].astype(np.int64)
        self.matrix += np.bincount(indices, minlength=self.num_classes**2).reshape(
            self.num_classes,
            self.num_classes,
        )

    def metrics(self) -> dict:
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


def finalize_pair(pair_id: int, counts: dict[str, int]) -> dict:
    changed = counts["changed"]
    repaired = counts["repaired"]
    damaged = counts["damaged"]
    return {
        "pair": pair_name(pair_id),
        "pair_id": int(pair_id),
        "changed": int(changed),
        "repaired": int(repaired),
        "damaged": int(damaged),
        "both_wrong": int(counts["both_wrong"]),
        "net_repaired": int(repaired - damaged),
        "repair_precision": float(repaired / changed) if changed else 0.0,
        "damage_rate": float(damaged / changed) if changed else 0.0,
        "both_wrong_rate": float(counts["both_wrong"] / changed) if changed else 0.0,
    }


def npz_files(path: Path) -> list[Path]:
    files = sorted(path.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found under {path}")
    return files


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    register_cosec()

    records = DatasetCatalog.get(args.dataset)
    base_files = npz_files(Path(args.base_map_dir))
    cand_files = npz_files(Path(args.candidate_map_dir))
    if len(base_files) != len(records) or len(cand_files) != len(records):
        raise SystemExit(
            "Map count must match dataset record count for ordered alignment: "
            f"dataset={len(records)}, base={len(base_files)}, candidate={len(cand_files)}"
        )

    pair_counts: dict[int, dict[str, int]] = {
        idx: {"changed": 0, "repaired": 0, "damaged": 0, "both_wrong": 0}
        for idx in range(len(CLASSES) ** 2)
    }
    total = Counter()
    meters = OrderedDict(
        [
            ("base", ConfusionMeter(len(CLASSES))),
            ("candidate", ConfusionMeter(len(CLASSES))),
            ("merged_allowed_pairs", ConfusionMeter(len(CLASSES))),
        ]
    )
    allowed_pair_ids: set[int] = set()
    per_image = []

    loaded = []
    for record, base_path, cand_path in zip(records, base_files, cand_files):
        label = read_label(record["sem_seg_file_name"])
        base = read_pred(base_path)
        cand = read_pred(cand_path)
        if base.shape != label.shape or cand.shape != label.shape:
            raise ValueError(
                f"Shape mismatch for {record['file_name']}: "
                f"label={label.shape}, base={base.shape}, candidate={cand.shape}"
            )
        valid = valid_mask(label, base, cand)
        changed = valid & (base != cand)
        repaired = changed & (base != label) & (cand == label)
        damaged = changed & (base == label) & (cand != label)
        both_wrong = changed & (base != label) & (cand != label)
        pair_ids = base.astype(np.int64) * len(CLASSES) + cand.astype(np.int64)

        total["valid"] += int(valid.sum())
        total["changed"] += int(changed.sum())
        total["repaired"] += int(repaired.sum())
        total["damaged"] += int(damaged.sum())
        total["both_wrong"] += int(both_wrong.sum())

        for mask_name, mask in [
            ("changed", changed),
            ("repaired", repaired),
            ("damaged", damaged),
            ("both_wrong", both_wrong),
        ]:
            if mask.any():
                values, counts = np.unique(pair_ids[mask], return_counts=True)
                for pair_id, count in zip(values, counts):
                    pair_counts[int(pair_id)][mask_name] += int(count)

        meters["base"].update(base, label)
        meters["candidate"].update(cand, label)
        loaded.append((record, base, cand, label, valid, pair_ids))
        per_image.append(
            {
                "file_name": record["file_name"],
                "base_map": str(base_path),
                "candidate_map": str(cand_path),
                "changed": int(changed.sum()),
                "repaired": int(repaired.sum()),
                "damaged": int(damaged.sum()),
                "net_repaired": int(repaired.sum() - damaged.sum()),
            }
        )

    rows = [finalize_pair(pair_id, counts) for pair_id, counts in pair_counts.items() if counts["changed"]]
    rows_by_net = sorted(rows, key=lambda row: (row["net_repaired"], row["repair_precision"], row["changed"]), reverse=True)
    allowed_rows = [
        row
        for row in rows_by_net
        if row["net_repaired"] >= args.min_net
        and row["repair_precision"] >= args.min_precision
        and row["changed"] >= args.min_changed
    ]
    allowed_pair_ids = {int(row["pair_id"]) for row in allowed_rows}

    accepted_pixels = 0
    for _record, base, cand, label, valid, pair_ids in loaded:
        accept = valid & (base != cand) & np.isin(pair_ids, list(allowed_pair_ids))
        accept = component_gate(
            accept,
            base,
            cand,
            args.component_min_boundary5_rate,
            args.component_max_area,
        )
        merged = base.copy()
        merged[accept] = cand[accept]
        accepted_pixels += int(accept.sum())
        meters["merged_allowed_pairs"].update(merged, label)

    results = OrderedDict((name, meter.metrics()) for name, meter in meters.items())
    results["merged_allowed_pairs"]["accepted_pixels"] = int(accepted_pixels)
    results["merged_allowed_pairs"]["pair_count"] = len(allowed_rows)

    output = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "dataset_size": len(records),
        "classes": list(CLASSES),
        "totals": {
            "valid": int(total["valid"]),
            "changed": int(total["changed"]),
            "repaired": int(total["repaired"]),
            "damaged": int(total["damaged"]),
            "both_wrong": int(total["both_wrong"]),
            "net_repaired": int(total["repaired"] - total["damaged"]),
            "repair_precision": float(total["repaired"] / total["changed"]) if total["changed"] else 0.0,
        },
        "thresholds": {
            "min_net": args.min_net,
            "min_precision": args.min_precision,
            "min_changed": args.min_changed,
            "component_min_boundary5_rate": args.component_min_boundary5_rate,
            "component_max_area": args.component_max_area,
        },
        "results": results,
        "allowed_pairs": allowed_rows,
        "allow_pairs_csv": ",".join(row["pair"] for row in allowed_rows),
        "top_positive_by_net": rows_by_net[: args.top_k],
        "top_negative_by_net": sorted(rows, key=lambda row: (row["net_repaired"], -row["changed"]))[: args.top_k],
        "top_by_changed": sorted(rows, key=lambda row: row["changed"], reverse=True)[: args.top_k],
        "per_image_top_negative": sorted(per_image, key=lambda row: row["net_repaired"])[:20],
        "per_image_top_positive": sorted(per_image, key=lambda row: row["net_repaired"], reverse=True)[:20],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.allow_pairs_out:
        allow_path = Path(args.allow_pairs_out)
        allow_path.parent.mkdir(parents=True, exist_ok=True)
        allow_path.write_text(output["allow_pairs_csv"] + "\n", encoding="utf-8")

    print(f"Wrote score-free repair gate: {out_path}")
    print(
        f"candidate changed={total['changed']} repaired={total['repaired']} "
        f"damaged={total['damaged']} net={total['repaired'] - total['damaged']}"
    )
    print(
        f"allowed_pairs={len(allowed_rows)} accepted_pixels={accepted_pixels} "
        f"merged_mIoU={results['merged_allowed_pairs']['mIoU']:.4f} "
        f"base_mIoU={results['base']['mIoU']:.4f} candidate_mIoU={results['candidate']['mIoU']:.4f}"
    )
    for row in allowed_rows[:10]:
        print(
            f"  {row['pair']}: net={row['net_repaired']} "
            f"repair={row['repaired']} damage={row['damaged']} changed={row['changed']} "
            f"precision={row['repair_precision']:.3f}"
        )


if __name__ == "__main__":
    main()
