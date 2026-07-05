#!/usr/bin/env python
"""Compose a CoSEC submission from separate Day, Night, and REAL prediction dirs."""

import argparse
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path


EXPECTED_PREFIX_COUNTS = {
    "Day_": 574,
    "Night_": 306,
    "REAL_": 102,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--day-dir", required=True)
    parser.add_argument("--night-dir", required=True)
    parser.add_argument("--real-dir", required=True)
    parser.add_argument("--test-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--zip", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-count-mismatch", action="store_true")
    return parser.parse_args()


def iter_test_sequences(test_root):
    test_root = Path(test_root)
    for seq_dir in sorted(path for path in test_root.iterdir() if path.is_dir()):
        img_dir = seq_dir / "img_co_left"
        if not img_dir.is_dir():
            continue
        image_count = len(list(img_dir.glob("*.png")))
        yield seq_dir.name, image_count


def source_for_sequence(seq_name, day_dir, night_dir, real_dir):
    if seq_name.startswith("Day_"):
        return day_dir, "day"
    if seq_name.startswith("Night_"):
        return night_dir, "night"
    if seq_name.startswith("REAL_"):
        return real_dir, "real"
    raise ValueError(f"Unknown test sequence prefix: {seq_name}")


def copy_sequence_masks(seq_name, source_root, out_dir, expected_count):
    source_seg = source_root / seq_name / "segment_co"
    if not source_seg.is_dir():
        raise FileNotFoundError(f"Missing source masks for {seq_name}: {source_seg}")
    masks = sorted(source_seg.glob("*.png"))
    if len(masks) != expected_count:
        raise ValueError(
            f"Mask count mismatch for {seq_name}: source has {len(masks)}, "
            f"test has {expected_count}"
        )
    dst_seg = out_dir / seq_name / "segment_co"
    dst_seg.mkdir(parents=True, exist_ok=True)
    for src_path in masks:
        shutil.copy2(src_path, dst_seg / src_path.name)
    return len(masks), str(source_seg.resolve())


def zip_prediction_dir(src_dir, zip_path):
    src_dir = Path(src_dir)
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    max_path_len = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(src_dir.rglob("*")):
            if not path.is_file() or path.name == "manifest.json":
                continue
            rel_path = path.relative_to(src_dir)
            max_path_len = max(max_path_len, len(str(rel_path)))
            zf.write(path, rel_path)
            count += 1
    return {"entries": count, "max_path_len": max_path_len}


def main():
    args = parse_args()
    day_dir = Path(args.day_dir)
    night_dir = Path(args.night_dir)
    real_dir = Path(args.real_dir)
    out_dir = Path(args.out_dir)

    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    counts = {
        "total": 0,
        "by_bucket": {"day": 0, "night": 0, "real": 0},
        "by_prefix": {"Day_": 0, "Night_": 0, "REAL_": 0},
        "sequences": {},
    }
    for seq_name, expected_count in iter_test_sequences(args.test_root):
        source_root, bucket = source_for_sequence(seq_name, day_dir, night_dir, real_dir)
        try:
            copied, source = copy_sequence_masks(seq_name, source_root, out_dir, expected_count)
        except ValueError:
            if not args.allow_count_mismatch:
                raise
            source_seg = source_root / seq_name / "segment_co"
            masks = sorted(source_seg.glob("*.png"))
            dst_seg = out_dir / seq_name / "segment_co"
            dst_seg.mkdir(parents=True, exist_ok=True)
            for src_path in masks:
                shutil.copy2(src_path, dst_seg / src_path.name)
            copied = len(masks)
            source = str(source_seg.resolve())
        counts["total"] += copied
        counts["by_bucket"][bucket] += copied
        for prefix in counts["by_prefix"]:
            if seq_name.startswith(prefix):
                counts["by_prefix"][prefix] += copied
                break
        counts["sequences"][seq_name] = {
            "bucket": bucket,
            "count": copied,
            "expected_count": expected_count,
            "source": source,
        }

    if not args.allow_count_mismatch:
        for prefix, expected in EXPECTED_PREFIX_COUNTS.items():
            actual = counts["by_prefix"][prefix]
            if actual != expected:
                raise ValueError(f"{prefix} count mismatch: expected {expected}, got {actual}")
        if counts["total"] != sum(EXPECTED_PREFIX_COUNTS.values()):
            raise ValueError(f"Total count mismatch: got {counts['total']}")

    zip_info = zip_prediction_dir(out_dir, args.zip)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "day_dir": str(day_dir.resolve()),
        "night_dir": str(night_dir.resolve()),
        "real_dir": str(real_dir.resolve()),
        "test_root": str(Path(args.test_root).resolve()),
        "out_dir": str(out_dir.resolve()),
        "zip": str(Path(args.zip).resolve()),
        "counts": counts,
        "zip_info": zip_info,
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote composed predictions to: {out_dir}")
    print(f"Wrote zip: {args.zip}")
    print(json.dumps({"counts": counts, "zip_info": zip_info}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
