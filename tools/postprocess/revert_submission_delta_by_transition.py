#!/usr/bin/env python3
"""Revert selected class-transition deltas from a candidate submission.

This is the inverse of a transition allow-list filter: start from the candidate
submission, then restore base labels for selected risky transitions in selected
domains. It is useful when a candidate is mostly good, but a previous hidden
negative probe suggests that a few transition families should be suppressed.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from PIL import Image
from scipy import ndimage


CLASSES = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Reference submission zip.")
    parser.add_argument("--candidate", required=True, help="Candidate submission zip.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--zip", required=True, dest="zip_path")
    parser.add_argument("--summary", required=True)
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["Night"],
        help="Domain prefixes to edit, e.g. Night or Day REAL.",
    )
    parser.add_argument(
        "--deny-pairs",
        required=True,
        help="Comma-separated class transitions to revert, e.g. fence->wall,wall->sidewalk.",
    )
    parser.add_argument(
        "--revert-component-min-area",
        type=int,
        default=None,
        help="Only revert denied components with at least this area.",
    )
    parser.add_argument(
        "--revert-component-max-boundary5-rate",
        type=float,
        default=None,
        help="Only revert denied components whose boundary5 rate is at most this value.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def class_id(text: str) -> int:
    text = text.strip()
    if text.isdigit():
        idx = int(text)
        if 0 <= idx < len(CLASSES):
            return idx
    if text in CLASSES:
        return CLASSES.index(text)
    raise ValueError(f"Unknown class: {text!r}")


def parse_pairs(text: str) -> set[int]:
    out: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "->" not in part:
            raise ValueError(f"Expected from->to pair, got {part!r}")
        src, dst = part.split("->", 1)
        out.add(class_id(src) * len(CLASSES) + class_id(dst))
    if not out:
        raise ValueError("No deny pairs parsed")
    return out


def pair_name(pair_id: int) -> str:
    src = pair_id // len(CLASSES)
    dst = pair_id % len(CLASSES)
    return f"{CLASSES[src]}->{CLASSES[dst]}"


def read_png(zf: ZipFile, name: str) -> np.ndarray:
    with zf.open(name) as handle:
        arr = np.array(Image.open(handle))
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.uint8, copy=False)


def write_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8, copy=False)).save(path)


def domain_of(name: str) -> str:
    return name.split("/", 1)[0].split("_", 1)[0]


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


def component_revert_gate(
    denied: np.ndarray,
    base: np.ndarray,
    candidate: np.ndarray,
    min_area: int | None,
    max_boundary5_rate: float | None,
) -> tuple[np.ndarray, dict[str, int]]:
    if min_area is None and max_boundary5_rate is None:
        return denied, {}

    labels, count = ndimage.label(denied, structure=np.ones((3, 3), dtype=np.uint8))
    stats = {
        "denied_component_count": int(count),
        "denied_component_reverted_count": 0,
        "denied_component_kept_count": 0,
        "denied_component_kept_pixels": 0,
    }
    if count == 0:
        return denied, stats

    b5 = boundary5(base, candidate)
    revert = np.zeros_like(denied, dtype=bool)
    for label_id, slc in enumerate(ndimage.find_objects(labels), start=1):
        if slc is None:
            continue
        comp = labels[slc] == label_id
        area = int(comp.sum())
        b_rate = float((comp & b5[slc]).sum() / area) if area else 0.0
        pass_area = min_area is None or area >= min_area
        pass_boundary = max_boundary5_rate is None or b_rate <= max_boundary5_rate
        if pass_area and pass_boundary:
            revert[slc] |= comp
            stats["denied_component_reverted_count"] += 1
        else:
            stats["denied_component_kept_count"] += 1
            stats["denied_component_kept_pixels"] += area
    return revert, stats


def top_pairs(counter: Counter[int], top_k: int = 30) -> list[dict[str, int | str]]:
    return [
        {"transition": pair_name(pair_id), "pixels": int(count)}
        for pair_id, count in counter.most_common(top_k)
    ]


def main() -> None:
    args = parse_args()
    base_zip = Path(args.base)
    cand_zip = Path(args.candidate)
    out_dir = Path(args.out_dir)
    out_zip = Path(args.zip_path)
    summary_path = Path(args.summary)

    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"{out_dir} exists and is not empty; pass --overwrite")
    if out_zip.exists() and not args.overwrite:
        raise SystemExit(f"{out_zip} exists; pass --overwrite")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    deny_pairs = parse_pairs(args.deny_pairs)
    wanted_domains = set(args.domains)
    stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    denied_pair_stats: dict[str, Counter[int]] = defaultdict(Counter)
    missing_candidate: list[str] = []

    with ZipFile(base_zip) as base_zf, ZipFile(cand_zip) as cand_zf:
        base_names = sorted(n for n in base_zf.namelist() if n.endswith(".png") and not n.endswith("/"))
        cand_names = set(cand_zf.namelist())
        for name in base_names:
            base = read_png(base_zf, name)
            if name not in cand_names:
                missing_candidate.append(name)
                final = base.copy()
            else:
                cand = read_png(cand_zf, name)
                if cand.shape != base.shape:
                    raise ValueError(f"Shape mismatch for {name}: {base.shape} vs {cand.shape}")
                final = cand.copy()
                dom = domain_of(name)
                stats[dom]["images"] += 1
                stats[dom]["pixels"] += int(base.size)
                changed = cand != base
                stats[dom]["candidate_delta_pixels"] += int(changed.sum())

                if dom in wanted_domains and changed.any():
                    pair_ids = base.astype(np.int64) * len(CLASSES) + cand.astype(np.int64)
                    denied = changed & np.isin(pair_ids, list(deny_pairs))
                    denied_before_component_gate = int(denied.sum())
                    denied, component_stats = component_revert_gate(
                        denied,
                        base,
                        cand,
                        args.revert_component_min_area,
                        args.revert_component_max_boundary5_rate,
                    )
                    final[denied] = base[denied]
                    stats[dom]["denied_pixels_before_component_gate"] += denied_before_component_gate
                    stats[dom]["reverted_pixels"] += int(denied.sum())
                    stats[dom]["kept_candidate_delta_pixels"] += int((final != base).sum())
                    for key, value in component_stats.items():
                        stats[dom][key] += int(value)
                    if denied_before_component_gate:
                        denied_ids = pair_ids[changed & np.isin(pair_ids, list(deny_pairs))]
                        denied_pair_stats[dom].update(denied_ids.tolist())
                elif dom in wanted_domains:
                    stats[dom]["kept_candidate_delta_pixels"] += int(changed.sum())

            write_png(out_dir / name, final)

    with ZipFile(out_zip, "w", compression=ZIP_DEFLATED) as zf:
        for path in sorted(out_dir.rglob("*.png")):
            zf.write(path, path.relative_to(out_dir).as_posix())

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "base": str(base_zip.resolve()),
        "candidate": str(cand_zip.resolve()),
        "out_dir": str(out_dir.resolve()),
        "zip": str(out_zip.resolve()),
        "domains": args.domains,
        "deny_pairs": [pair_name(pair_id) for pair_id in sorted(deny_pairs)],
        "revert_component_min_area": args.revert_component_min_area,
        "revert_component_max_boundary5_rate": args.revert_component_max_boundary5_rate,
        "missing_candidate_entries": missing_candidate,
        "stats": {k: dict(v) for k, v in sorted(stats.items())},
        "top_denied_pairs": {k: top_pairs(v) for k, v in sorted(denied_pair_stats.items())},
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
