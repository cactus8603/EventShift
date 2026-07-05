#!/usr/bin/env python
"""Merge PNG mask directories using class-wise routes from a basis prediction."""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

from cosec_finetune_splits import CLASSES


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basis-dir", required=True)
    parser.add_argument("--anchor-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--branch",
        action="append",
        default=[],
        help="Branch mapping in the form name=/path/to/prediction_dir. Repeatable.",
    )
    parser.add_argument(
        "--route",
        action="append",
        default=[],
        help="Class route in the form class_name=branch_name. Repeatable.",
    )
    parser.add_argument(
        "--boundary-source",
        default="none",
        choices=["none", "anchor", "basis", "union", "intersection"],
        help="Optionally limit routing to a semantic-boundary band.",
    )
    parser.add_argument(
        "--boundary-radius",
        type=int,
        default=0,
        help="Boundary band radius in pixels when --boundary-source is not none.",
    )
    parser.add_argument(
        "--protect-anchor-class",
        action="append",
        default=[],
        help="Do not route pixels whose anchor prediction is this class. Repeatable.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_key_value(items, kind):
    parsed = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"{kind} must use key=value format, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(f"{kind} has empty key/value: {item}")
        if key in parsed:
            raise ValueError(f"Duplicate {kind} key: {key}")
        parsed[key] = value
    return parsed


def parse_class_names(class_names):
    class_ids = []
    for class_name in class_names:
        if class_name not in CLASSES:
            raise ValueError(f"Unknown class: {class_name}. Valid classes: {', '.join(CLASSES)}")
        class_ids.append(CLASSES.index(class_name))
    return class_ids


def read_mask(path):
    if cv2 is not None:
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError(f"Could not read mask: {path}")
    else:
        mask = np.asarray(Image.open(path))
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask


def write_mask(path, mask):
    path.parent.mkdir(parents=True, exist_ok=True)
    if cv2 is not None:
        if not cv2.imwrite(str(path), mask):
            raise RuntimeError(f"Could not write mask: {path}")
    else:
        Image.fromarray(mask.astype(np.uint8, copy=False)).save(path)


def iter_masks(root):
    root = Path(root)
    for path in sorted(root.rglob("segment_co/*.png")):
        yield path.relative_to(root)


def semantic_boundary_band(mask, radius):
    if radius <= 0:
        return np.zeros(mask.shape, dtype=bool)
    if cv2 is None:
        raise RuntimeError("Boundary routing requires OpenCV; rerun with --boundary-source none or install cv2.")
    valid = (mask >= 0) & (mask < len(CLASSES))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    low = mask.astype(np.float32, copy=True)
    high = mask.astype(np.float32, copy=True)
    low[~valid] = -1000.0
    high[~valid] = 1000.0
    local_max = cv2.dilate(low, kernel)
    local_min = cv2.erode(high, kernel)
    return valid & (local_max != local_min)


def boundary_gate(source, radius, anchor, basis):
    if source == "none":
        return np.ones(anchor.shape, dtype=bool)
    anchor_boundary = semantic_boundary_band(anchor, radius)
    basis_boundary = semantic_boundary_band(basis, radius)
    if source == "anchor":
        return anchor_boundary
    if source == "basis":
        return basis_boundary
    if source == "union":
        return anchor_boundary | basis_boundary
    if source == "intersection":
        return anchor_boundary & basis_boundary
    raise ValueError(f"Unknown boundary source: {source}")


def main():
    args = parse_args()
    basis_dir = Path(args.basis_dir)
    anchor_dir = Path(args.anchor_dir)
    out_dir = Path(args.out_dir)
    branches = {name: Path(path) for name, path in parse_key_value(args.branch, "branch").items()}
    routes_by_name = parse_key_value(args.route, "route")
    protect_anchor_ids = parse_class_names(args.protect_anchor_class)

    if "anchor" in branches:
        raise ValueError("The branch name 'anchor' is reserved.")
    branches["anchor"] = anchor_dir

    routes = {}
    for class_name, branch_name in routes_by_name.items():
        if class_name not in CLASSES:
            raise ValueError(f"Unknown class: {class_name}. Valid classes: {', '.join(CLASSES)}")
        if branch_name not in branches:
            raise ValueError(f"Unknown branch for {class_name}: {branch_name}")
        routes[CLASSES.index(class_name)] = branch_name

    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    stats = {
        "files": 0,
        "pixels": 0,
        "changed_vs_anchor": 0,
        "routed_pixels": 0,
        "routed_pixels_by_class": {class_name: 0 for class_name in routes_by_name},
        "routed_pixels_by_branch": {name: 0 for name in branches if name != "anchor"},
        "changed_vs_anchor_by_branch": {name: 0 for name in branches if name != "anchor"},
        "missing_branch_masks": [],
    }

    for rel_path in iter_masks(anchor_dir):
        basis_path = basis_dir / rel_path
        if not basis_path.is_file():
            raise FileNotFoundError(f"Missing basis mask for {rel_path}: {basis_path}")

        basis = read_mask(basis_path)
        anchor = read_mask(anchor_dir / rel_path)
        if basis.shape != anchor.shape:
            raise RuntimeError(f"Shape mismatch for {rel_path}: basis={basis.shape}, anchor={anchor.shape}")

        merged = anchor.copy()
        gate = boundary_gate(args.boundary_source, args.boundary_radius, anchor, basis)
        if protect_anchor_ids:
            gate = gate & ~np.isin(anchor, protect_anchor_ids)
        for class_id, branch_name in routes.items():
            branch_path = branches[branch_name] / rel_path
            if not branch_path.is_file():
                stats["missing_branch_masks"].append({"branch": branch_name, "path": str(rel_path)})
                continue
            branch_mask = read_mask(branch_path)
            if branch_mask.shape != anchor.shape:
                raise RuntimeError(
                    f"Shape mismatch for {rel_path}: branch={branch_name} "
                    f"shape={branch_mask.shape}, anchor={anchor.shape}"
                )
            routed = (basis == class_id) & gate
            routed_count = int(routed.sum())
            if routed_count == 0:
                continue
            changed = routed & (branch_mask != anchor)
            merged[routed] = branch_mask[routed]
            class_name = CLASSES[class_id]
            changed_count = int(changed.sum())
            stats["routed_pixels"] += routed_count
            stats["routed_pixels_by_class"][class_name] += routed_count
            stats["routed_pixels_by_branch"][branch_name] += routed_count
            stats["changed_vs_anchor_by_branch"][branch_name] += changed_count

        changed_vs_anchor = merged != anchor
        stats["files"] += 1
        stats["pixels"] += int(anchor.size)
        stats["changed_vs_anchor"] += int(changed_vs_anchor.sum())
        write_mask(out_dir / rel_path, merged)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "basis_dir": str(basis_dir.resolve()),
        "anchor_dir": str(anchor_dir.resolve()),
        "out_dir": str(out_dir.resolve()),
        "branches": {name: str(path.resolve()) for name, path in branches.items()},
        "routes": routes_by_name,
        "boundary_source": args.boundary_source,
        "boundary_radius": args.boundary_radius,
        "protect_anchor_class": list(args.protect_anchor_class),
        "stats": stats,
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote routed masks: {out_dir}")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
