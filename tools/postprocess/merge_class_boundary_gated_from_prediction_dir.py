#!/usr/bin/env python
"""Paste selected candidate classes into an anchor dir with boundary gates."""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-dir", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument(
        "--support-dir",
        default=None,
        help="Optional third prediction dir. If set, candidate pixels are kept only when support predicts the same class.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--class-id", action="append", type=int, required=True)
    parser.add_argument("--prefix", default=None, help="Optional sequence prefix filter, e.g. Night_")
    parser.add_argument("--boundary-radius", type=int, default=5)
    parser.add_argument(
        "--boundary-source",
        choices=["anchor", "candidate", "either"],
        default="anchor",
        help="Prediction map used to build the semantic boundary band.",
    )
    parser.add_argument(
        "--gate-mode",
        choices=["none", "pixel", "component"],
        default="component",
        help="pixel keeps only pixels in the boundary band; component accepts/rejects connected changes.",
    )
    parser.add_argument(
        "--component-min-boundary-rate",
        type=float,
        default=0.6,
        help="For component mode, accept a component when this share lies in the boundary band.",
    )
    parser.add_argument(
        "--component-max-area",
        type=int,
        default=0,
        help="For component mode, also accept components no larger than this area. 0 disables.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def iter_anchor_masks(anchor_dir, prefix=None):
    anchor_dir = Path(anchor_dir)
    for seq_dir in sorted(path for path in anchor_dir.iterdir() if path.is_dir()):
        if prefix and not seq_dir.name.startswith(prefix):
            continue
        seg_dir = seq_dir / "segment_co"
        if not seg_dir.is_dir():
            continue
        for mask_path in sorted(seg_dir.glob("*.png")):
            yield seq_dir.name, mask_path


def read_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"Could not read mask: {path}")
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask.astype(np.uint8, copy=False)


def write_mask(path, mask):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), mask):
        raise RuntimeError(f"Could not write mask: {path}")


def semantic_boundary_band(pred, radius):
    edge = np.zeros(pred.shape, dtype=bool)
    edge[:, 1:] |= pred[:, 1:] != pred[:, :-1]
    edge[:, :-1] |= pred[:, 1:] != pred[:, :-1]
    edge[1:, :] |= pred[1:, :] != pred[:-1, :]
    edge[:-1, :] |= pred[1:, :] != pred[:-1, :]
    if radius <= 0:
        return edge
    kernel = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)
    return cv2.dilate(edge.astype(np.uint8), kernel, iterations=1) > 0


def boundary_for_source(anchor, candidate, radius, source):
    anchor_boundary = semantic_boundary_band(anchor, radius)
    if source == "anchor":
        return anchor_boundary
    candidate_boundary = semantic_boundary_band(candidate, radius)
    if source == "candidate":
        return candidate_boundary
    return anchor_boundary | candidate_boundary


def component_gate(raw_take, boundary, min_boundary_rate, max_area):
    if not np.any(raw_take):
        return raw_take, []
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        raw_take.astype(np.uint8),
        connectivity=8,
    )
    keep = np.zeros(raw_take.shape, dtype=bool)
    components = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        comp = labels == label
        boundary_pixels = int(np.count_nonzero(comp & boundary))
        boundary_rate = boundary_pixels / max(1, area)
        accepted = boundary_rate >= min_boundary_rate or (max_area > 0 and area <= max_area)
        if accepted:
            keep[comp] = True
        components.append(
            {
                "area": area,
                "boundary_pixels": boundary_pixels,
                "boundary_rate": boundary_rate,
                "accepted": bool(accepted),
            }
        )
    return keep, components


def main():
    args = parse_args()
    anchor_dir = Path(args.anchor_dir)
    candidate_dir = Path(args.candidate_dir)
    support_dir = Path(args.support_dir) if args.support_dir else None
    out_dir = Path(args.out_dir)
    class_ids = sorted(set(args.class_id))

    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = {
        "files": 0,
        "changed_files": 0,
        "pixels": 0,
        "candidate_class_pixels": {str(class_id): 0 for class_id in class_ids},
        "raw_changed_pixels_by_class": {str(class_id): 0 for class_id in class_ids},
        "support_kept_pixels_by_class": {str(class_id): 0 for class_id in class_ids},
        "kept_changed_pixels_by_class": {str(class_id): 0 for class_id in class_ids},
        "raw_changed_pixels": 0,
        "support_kept_pixels": 0,
        "kept_changed_pixels": 0,
        "accepted_components": 0,
        "rejected_components": 0,
        "sequences": {},
    }

    items = list(iter_anchor_masks(anchor_dir, prefix=args.prefix))
    for seq_name, anchor_path in tqdm(items, desc="boundary-gated-merge"):
        rel = anchor_path.relative_to(anchor_dir)
        candidate_path = candidate_dir / rel
        if not candidate_path.exists():
            raise FileNotFoundError(f"Missing candidate mask: {candidate_path}")
        anchor = read_mask(anchor_path)
        candidate = read_mask(candidate_path)
        if anchor.shape != candidate.shape:
            raise ValueError(f"Shape mismatch for {rel}: {anchor.shape} vs {candidate.shape}")
        support = None
        if support_dir is not None:
            support_path = support_dir / rel
            if not support_path.exists():
                raise FileNotFoundError(f"Missing support mask: {support_path}")
            support = read_mask(support_path)
            if anchor.shape != support.shape:
                raise ValueError(f"Shape mismatch for {rel}: {anchor.shape} vs {support.shape}")

        boundary = None
        if args.gate_mode != "none":
            boundary = boundary_for_source(
                anchor,
                candidate,
                radius=args.boundary_radius,
                source=args.boundary_source,
            )
        merged = anchor.copy()
        seq = counts["sequences"].setdefault(
            seq_name,
            {"files": 0, "changed_files": 0, "raw_changed_pixels": 0, "kept_changed_pixels": 0},
        )

        for class_id in class_ids:
            class_key = str(class_id)
            candidate_class = candidate == class_id
            raw_take = candidate_class & (anchor != class_id)
            counts["candidate_class_pixels"][class_key] += int(np.count_nonzero(candidate_class))
            raw_count = int(np.count_nonzero(raw_take))
            counts["raw_changed_pixels_by_class"][class_key] += raw_count
            counts["raw_changed_pixels"] += raw_count
            seq["raw_changed_pixels"] += raw_count

            if support is not None:
                raw_take = raw_take & (support == class_id)
            support_count = int(np.count_nonzero(raw_take))
            counts["support_kept_pixels_by_class"][class_key] += support_count
            counts["support_kept_pixels"] += support_count

            if args.gate_mode == "none":
                keep = raw_take
            elif args.gate_mode == "pixel":
                keep = raw_take & boundary
            else:
                keep, components = component_gate(
                    raw_take,
                    boundary,
                    min_boundary_rate=args.component_min_boundary_rate,
                    max_area=args.component_max_area,
                )
                counts["accepted_components"] += sum(1 for comp in components if comp["accepted"])
                counts["rejected_components"] += sum(1 for comp in components if not comp["accepted"])

            kept_count = int(np.count_nonzero(keep))
            counts["kept_changed_pixels_by_class"][class_key] += kept_count
            counts["kept_changed_pixels"] += kept_count
            seq["kept_changed_pixels"] += kept_count
            merged[keep] = class_id

        changed = int(np.count_nonzero(merged != anchor))
        counts["files"] += 1
        counts["changed_files"] += int(changed > 0)
        counts["pixels"] += int(anchor.size)
        seq["files"] += 1
        seq["changed_files"] += int(changed > 0)
        write_mask(out_dir / rel, merged)

    counts["raw_changed_pixel_rate"] = counts["raw_changed_pixels"] / max(1, counts["pixels"])
    counts["support_kept_share_of_raw"] = counts["support_kept_pixels"] / max(1, counts["raw_changed_pixels"])
    counts["support_kept_pixel_rate"] = counts["support_kept_pixels"] / max(1, counts["pixels"])
    counts["kept_changed_pixel_rate"] = counts["kept_changed_pixels"] / max(1, counts["pixels"])
    counts["kept_share_of_raw"] = counts["kept_changed_pixels"] / max(1, counts["raw_changed_pixels"])
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "anchor_dir": str(anchor_dir.resolve()),
        "candidate_dir": str(candidate_dir.resolve()),
        "support_dir": str(support_dir.resolve()) if support_dir is not None else None,
        "out_dir": str(out_dir.resolve()),
        "class_ids": class_ids,
        "prefix": args.prefix,
        "boundary_radius": args.boundary_radius,
        "boundary_source": args.boundary_source,
        "gate_mode": args.gate_mode,
        "component_min_boundary_rate": args.component_min_boundary_rate,
        "component_max_area": args.component_max_area,
        "counts": counts,
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
