#!/usr/bin/env python3
"""Transfer a controlled submission delta from one zip onto another."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from PIL import Image
from scipy import ndimage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-base", required=True, help="Zip before a known useful delta.")
    parser.add_argument("--old-candidate", required=True, help="Zip after a known useful delta.")
    parser.add_argument("--new-base", required=True, help="Zip to patch.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--zip", required=True, dest="zip_path")
    parser.add_argument("--summary", required=True)
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["Night"],
        help="Path prefixes to transfer, e.g. Night or Day REAL.",
    )
    parser.add_argument(
        "--require-new-equals-old-base",
        action="store_true",
        help="Only transfer pixels where the new base still equals the old base.",
    )
    parser.add_argument(
        "--component-min-boundary5-rate",
        type=float,
        default=None,
        help="Keep transfer components with at least this boundary5 rate.",
    )
    parser.add_argument(
        "--component-max-area",
        type=int,
        default=None,
        help="Also keep transfer components up to this area, even if boundary rate is lower.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_png(zf: ZipFile, name: str) -> np.ndarray:
    with zf.open(name) as f:
        return np.array(Image.open(f))


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
    transfer: np.ndarray,
    old_cand: np.ndarray,
    new_arr: np.ndarray,
    min_boundary5_rate: float | None,
    max_area: int | None,
) -> tuple[np.ndarray, dict[str, int]]:
    if min_boundary5_rate is None and max_area is None:
        return transfer, {}

    labels, count = ndimage.label(transfer, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return transfer, {
            "component_count": 0,
            "component_kept_count": 0,
            "component_removed_count": 0,
            "component_removed_pixels": 0,
        }

    b5 = boundary5(new_arr, old_cand)
    kept = np.zeros_like(transfer, dtype=bool)
    removed_pixels = 0
    kept_count = 0

    objects = ndimage.find_objects(labels)
    for label_id, slc in enumerate(objects, start=1):
        if slc is None:
            continue
        comp = labels[slc] == label_id
        area = int(comp.sum())
        b_rate = float((comp & b5[slc]).sum()) / area if area else 0.0
        keep_by_boundary = min_boundary5_rate is not None and b_rate >= min_boundary5_rate
        keep_by_area = max_area is not None and area <= max_area
        if keep_by_boundary or keep_by_area:
            kept[slc] |= comp
            kept_count += 1
        else:
            removed_pixels += area

    return kept, {
        "component_count": int(count),
        "component_kept_count": int(kept_count),
        "component_removed_count": int(count - kept_count),
        "component_removed_pixels": int(removed_pixels),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_zip = Path(args.zip_path)
    summary_path = Path(args.summary)

    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"{out_dir} exists and is not empty; pass --overwrite")
    if out_zip.exists() and not args.overwrite:
        raise SystemExit(f"{out_zip} exists; pass --overwrite")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    wanted = set(args.domains)
    stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    missing_old: list[str] = []

    with ZipFile(args.old_base) as old_base_zf, ZipFile(args.old_candidate) as old_cand_zf, ZipFile(
        args.new_base
    ) as new_base_zf:
        old_base_names = set(old_base_zf.namelist())
        old_cand_names = set(old_cand_zf.namelist())
        new_names = sorted(n for n in new_base_zf.namelist() if not n.endswith("/"))

        for name in new_names:
            new_arr = read_png(new_base_zf, name)
            final = new_arr.copy()
            dom = domain_of(name)
            stats[dom]["images"] += 1
            stats[dom]["pixels"] += int(new_arr.size)

            if dom in wanted:
                if name not in old_base_names or name not in old_cand_names:
                    missing_old.append(name)
                else:
                    old_base = read_png(old_base_zf, name)
                    old_cand = read_png(old_cand_zf, name)
                    if old_base.shape != new_arr.shape or old_cand.shape != new_arr.shape:
                        raise ValueError(f"shape mismatch for {name}")

                    old_delta = old_cand != old_base
                    compatible = new_arr == old_base
                    if args.require_new_equals_old_base:
                        transfer = old_delta & compatible
                    else:
                        transfer = old_delta
                    transfer_before_component_gate = int(transfer.sum())
                    transfer, component_stats = component_gate(
                        transfer,
                        old_cand,
                        new_arr,
                        args.component_min_boundary5_rate,
                        args.component_max_area,
                    )

                    final[transfer] = old_cand[transfer]
                    stats[dom]["old_delta_pixels"] += int(old_delta.sum())
                    stats[dom]["compatible_old_delta_pixels"] += int((old_delta & compatible).sum())
                    stats[dom]["conflict_old_delta_pixels"] += int((old_delta & ~compatible).sum())
                    stats[dom]["pre_component_gate_transfer_pixels"] += transfer_before_component_gate
                    stats[dom]["transferred_pixels"] += int(transfer.sum())
                    for key, value in component_stats.items():
                        stats[dom][key] += int(value)
                    stats[dom]["changed_vs_new_pixels"] += int((final != new_arr).sum())

            write_png(out_dir / name, final)

    with ZipFile(out_zip, "w", compression=ZIP_DEFLATED) as zf:
        for path in sorted(out_dir.rglob("*.png")):
            zf.write(path, path.relative_to(out_dir).as_posix())

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "old_base": str(Path(args.old_base).resolve()),
        "old_candidate": str(Path(args.old_candidate).resolve()),
        "new_base": str(Path(args.new_base).resolve()),
        "out_dir": str(out_dir.resolve()),
        "zip": str(out_zip.resolve()),
        "domains": args.domains,
        "require_new_equals_old_base": args.require_new_equals_old_base,
        "component_min_boundary5_rate": args.component_min_boundary5_rate,
        "component_max_area": args.component_max_area,
        "missing_old_entries": missing_old,
        "stats": {k: dict(v) for k, v in sorted(stats.items())},
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
