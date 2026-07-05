#!/usr/bin/env python
"""Merge prediction PNG dirs by letting candidates claim selected classes."""

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
    parser.add_argument("--anchor-dir", required=True)
    parser.add_argument("--route-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Candidate mapping in the form name=/path/to/prediction_dir. Repeatable.",
    )
    parser.add_argument(
        "--sequences-prefix",
        action="append",
        default=[],
        help="Only process sequences whose name starts with this prefix. Repeatable.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_candidate(items):
    parsed = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--candidate must be name=dir, got: {item}")
        name, path = item.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"Empty candidate name/path: {item}")
        if name in parsed:
            raise ValueError(f"Duplicate candidate name: {name}")
        parsed[name] = Path(path)
    return parsed


def read_mask(path):
    if cv2 is not None:
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError(f"Could not read mask: {path}")
    else:
        mask = np.asarray(Image.open(path))
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask.astype(np.uint8, copy=False)


def write_mask(path, mask):
    path.parent.mkdir(parents=True, exist_ok=True)
    if cv2 is not None:
        if not cv2.imwrite(str(path), mask):
            raise RuntimeError(f"Could not write mask: {path}")
    else:
        Image.fromarray(mask.astype(np.uint8, copy=False)).save(path)


def iter_masks(root, prefixes):
    root = Path(root)
    for path in sorted(root.rglob("segment_co/*.png")):
        seq_name = path.parent.parent.name
        if prefixes and not any(seq_name.startswith(prefix) for prefix in prefixes):
            continue
        yield path.relative_to(root)


def load_routes(path):
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    routes = []
    for route in payload.get("routes", []):
        class_id = int(route["class_id"])
        class_name = route.get("class_name") or CLASSES[class_id]
        if class_id < 0 or class_id >= len(CLASSES):
            raise ValueError(f"Invalid class id in route: {route}")
        if class_name != CLASSES[class_id]:
            raise ValueError(f"Route class mismatch: {route}")
        routes.append(route)
    routes.sort(key=lambda item: float(item.get("gain") or 0.0), reverse=True)
    return payload, routes


def main():
    args = parse_args()
    anchor_dir = Path(args.anchor_dir)
    out_dir = Path(args.out_dir)
    candidates = parse_candidate(args.candidate)
    route_payload, routes = load_routes(args.route_json)

    missing_candidates = sorted({route["candidate"] for route in routes} - set(candidates))
    if missing_candidates:
        raise ValueError(f"Missing --candidate mappings for: {missing_candidates}")

    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    stats = {
        "files": 0,
        "pixels": 0,
        "changed_vs_anchor": 0,
        "claimed_pixels": 0,
        "claimed_pixels_by_class": {route["class_name"]: 0 for route in routes},
        "changed_vs_anchor_by_class": {route["class_name"]: 0 for route in routes},
        "claimed_pixels_by_candidate": {name: 0 for name in candidates},
        "changed_vs_anchor_by_candidate": {name: 0 for name in candidates},
        "priority_overwrites": 0,
    }

    candidate_cache = {}
    for rel_path in iter_masks(anchor_dir, args.sequences_prefix):
        anchor = read_mask(anchor_dir / rel_path)
        merged = anchor.copy()
        claimed = np.zeros(anchor.shape, dtype=bool)

        for route in routes:
            class_id = int(route["class_id"])
            class_name = route["class_name"]
            candidate_name = route["candidate"]
            candidate_dir = candidates[candidate_name]
            cache_key = (candidate_name, rel_path)
            if cache_key not in candidate_cache:
                candidate_path = candidate_dir / rel_path
                if not candidate_path.is_file():
                    raise FileNotFoundError(f"Missing candidate mask: {candidate_path}")
                candidate_cache[cache_key] = read_mask(candidate_path)
            candidate = candidate_cache[cache_key]
            if candidate.shape != anchor.shape:
                raise RuntimeError(
                    f"Shape mismatch for {rel_path}: candidate={candidate.shape}, anchor={anchor.shape}"
                )
            claim = candidate == class_id
            if not np.any(claim):
                continue
            stats["priority_overwrites"] += int((claim & claimed).sum())
            changed = claim & (anchor != class_id)
            merged[claim] = class_id
            claimed |= claim

            claim_count = int(claim.sum())
            changed_count = int(changed.sum())
            stats["claimed_pixels"] += claim_count
            stats["claimed_pixels_by_class"][class_name] += claim_count
            stats["changed_vs_anchor_by_class"][class_name] += changed_count
            stats["claimed_pixels_by_candidate"][candidate_name] += claim_count
            stats["changed_vs_anchor_by_candidate"][candidate_name] += changed_count

        stats["files"] += 1
        stats["pixels"] += int(anchor.size)
        stats["changed_vs_anchor"] += int((merged != anchor).sum())
        write_mask(out_dir / rel_path, merged)
        candidate_cache.clear()

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "anchor_dir": str(anchor_dir.resolve()),
        "route_json": str(Path(args.route_json).resolve()),
        "out_dir": str(out_dir.resolve()),
        "candidates": {name: str(path.resolve()) for name, path in candidates.items()},
        "sequences_prefix": args.sequences_prefix,
        "route_source": route_payload,
        "stats": stats,
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote merged prediction dir: {out_dir}")
    print(json.dumps({"routes": len(routes), "stats": stats}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
