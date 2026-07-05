#!/usr/bin/env python3
"""Filter a candidate submission delta by class transition and component shape."""

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
    parser.add_argument("--base", required=True, help="Reference submission zip or prediction dir.")
    parser.add_argument("--candidate", required=True, help="Candidate submission zip or prediction dir.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--zip", required=True, dest="zip_path")
    parser.add_argument("--summary", required=True)
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["Day", "Night", "REAL"],
        help="Domain prefixes to filter, e.g. Day Night REAL.",
    )
    parser.add_argument(
        "--allow-pairs",
        required=True,
        help="Comma-separated class transitions, e.g. road->sidewalk,building->vegetation.",
    )
    parser.add_argument(
        "--component-min-boundary5-rate",
        type=float,
        default=None,
        help="Keep components with at least this boundary5 rate after transition filtering.",
    )
    parser.add_argument(
        "--component-max-area",
        type=int,
        default=None,
        help="Also keep components up to this area even if boundary rate is lower.",
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
        raise ValueError("No allow pairs parsed")
    return out


def pair_name(pair_id: int) -> str:
    src = pair_id // len(CLASSES)
    dst = pair_id % len(CLASSES)
    return f"{CLASSES[src]}->{CLASSES[dst]}"


def is_zip(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".zip"


class MaskSource:
    def __init__(self, path: str):
        self.path = Path(path)
        self.zf = ZipFile(self.path) if is_zip(self.path) else None

    def close(self) -> None:
        if self.zf is not None:
            self.zf.close()

    def entries(self) -> list[str]:
        if self.zf is not None:
            return sorted(n for n in self.zf.namelist() if n.endswith(".png") and not n.endswith("/"))
        return sorted(p.relative_to(self.path).as_posix() for p in self.path.rglob("*.png"))

    def has(self, name: str) -> bool:
        if self.zf is not None:
            return name in set(self.zf.namelist())
        return (self.path / name).is_file()

    def read(self, name: str) -> np.ndarray:
        if self.zf is not None:
            with self.zf.open(name) as handle:
                arr = np.array(Image.open(handle))
        else:
            arr = np.array(Image.open(self.path / name))
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


def component_gate(
    take: np.ndarray,
    base: np.ndarray,
    candidate: np.ndarray,
    min_boundary5_rate: float | None,
    max_area: int | None,
) -> tuple[np.ndarray, dict[str, int]]:
    if min_boundary5_rate is None and max_area is None:
        return take, {}

    labels, count = ndimage.label(take, structure=np.ones((3, 3), dtype=np.uint8))
    stats = {
        "component_count": int(count),
        "component_kept_count": 0,
        "component_removed_count": 0,
        "component_removed_pixels": 0,
    }
    if count == 0:
        return take, stats

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
            stats["component_kept_count"] += 1
        else:
            stats["component_removed_count"] += 1
            stats["component_removed_pixels"] += area
    return kept, stats


def top_pairs(counter: Counter[int], top_k: int = 30) -> list[dict[str, int | str]]:
    return [
        {"transition": pair_name(pair_id), "pixels": int(count)}
        for pair_id, count in counter.most_common(top_k)
    ]


def main() -> None:
    args = parse_args()
    base_src = MaskSource(args.base)
    cand_src = MaskSource(args.candidate)
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

    allow_pairs = parse_pairs(args.allow_pairs)
    wanted = set(args.domains)
    stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    pair_stats: dict[str, Counter[int]] = defaultdict(Counter)
    missing_candidate: list[str] = []

    try:
        base_entries = base_src.entries()
        cand_entries = set(cand_src.entries())
        for name in base_entries:
            base = base_src.read(name)
            final = base.copy()
            dom = domain_of(name)
            stats[dom]["images"] += 1
            stats[dom]["pixels"] += int(base.size)

            if dom in wanted:
                if name not in cand_entries:
                    missing_candidate.append(name)
                else:
                    cand = cand_src.read(name)
                    if cand.shape != base.shape:
                        raise ValueError(f"Shape mismatch for {name}: {base.shape} vs {cand.shape}")
                    changed = cand != base
                    pair_ids = base.astype(np.int64) * len(CLASSES) + cand.astype(np.int64)
                    pair_allowed = changed & np.isin(pair_ids, list(allow_pairs))
                    pre_component_pixels = int(pair_allowed.sum())
                    take, component_stats = component_gate(
                        pair_allowed,
                        base,
                        cand,
                        args.component_min_boundary5_rate,
                        args.component_max_area,
                    )
                    final[take] = cand[take]

                    stats[dom]["candidate_delta_pixels"] += int(changed.sum())
                    stats[dom]["pair_allowed_pixels"] += pre_component_pixels
                    stats[dom]["applied_pixels"] += int(take.sum())
                    stats[dom]["pair_blocked_pixels"] += int(changed.sum()) - pre_component_pixels
                    stats[dom]["component_removed_pixels"] += int(pre_component_pixels) - int(take.sum())
                    for key, value in component_stats.items():
                        stats[dom][key] += int(value)

                    if changed.any():
                        pair_stats[f"{dom}_candidate"].update(
                            dict(zip(*np.unique(pair_ids[changed], return_counts=True)))
                        )
                    if take.any():
                        pair_stats[f"{dom}_applied"].update(
                            dict(zip(*np.unique(pair_ids[take], return_counts=True)))
                        )

            write_png(out_dir / name, final)

        with ZipFile(out_zip, "w", compression=ZIP_DEFLATED) as zf:
            for path in sorted(out_dir.rglob("*.png")):
                zf.write(path, path.relative_to(out_dir).as_posix())

        summary = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "base": str(Path(args.base).resolve()),
            "candidate": str(Path(args.candidate).resolve()),
            "out_dir": str(out_dir.resolve()),
            "zip": str(out_zip.resolve()),
            "domains": args.domains,
            "allow_pairs": [pair_name(pair_id) for pair_id in sorted(allow_pairs)],
            "component_min_boundary5_rate": args.component_min_boundary5_rate,
            "component_max_area": args.component_max_area,
            "missing_candidate_entries": missing_candidate,
            "stats": {k: dict(v) for k, v in sorted(stats.items())},
            "top_pairs": {k: top_pairs(v) for k, v in sorted(pair_stats.items())},
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        base_src.close()
        cand_src.close()


if __name__ == "__main__":
    main()
