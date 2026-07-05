#!/usr/bin/env python
"""Merge an existing submission directory with newly exported REAL predictions."""

import argparse
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True, help="Existing complete prediction directory.")
    parser.add_argument("--real-dir", required=True, help="Prediction directory containing REAL sequences.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--zip", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def copy_tree_files(src_dir, dst_dir):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Missing source directory: {src_dir}")
    count = 0
    for src_path in sorted(path for path in src_dir.rglob("*") if path.is_file()):
        rel_path = src_path.relative_to(src_dir)
        dst_path = dst_dir / rel_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)
        count += 1
    return count


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
    base_dir = Path(args.base_dir)
    real_dir = Path(args.real_dir)
    out_dir = Path(args.out_dir)

    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    counts = {"total": 0, "real_new": 0, "base_kept": 0, "sequences": {}}
    for seq_dir in sorted(path for path in base_dir.iterdir() if path.is_dir()):
        seq_name = seq_dir.name
        base_seg = seq_dir / "segment_co"
        real_seg = real_dir / seq_name / "segment_co"
        if seq_name.startswith("REAL_") and real_seg.is_dir():
            source = real_seg
            bucket = "real_new"
        else:
            source = base_seg
            bucket = "base_kept"
        dst_seg = out_dir / seq_name / "segment_co"
        count = copy_tree_files(source, dst_seg)
        counts[bucket] += count
        counts["total"] += count
        counts["sequences"][seq_name] = {
            "bucket": bucket,
            "count": count,
            "source": str(source.resolve()),
        }

    zip_info = zip_prediction_dir(out_dir, args.zip)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "base_dir": str(base_dir.resolve()),
        "real_dir": str(real_dir.resolve()),
        "out_dir": str(out_dir.resolve()),
        "zip": str(Path(args.zip).resolve()),
        "counts": counts,
        "zip_info": zip_info,
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote merged predictions to: {out_dir}")
    print(f"Wrote zip: {args.zip}")
    print(json.dumps({"counts": counts, "zip_info": zip_info}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
