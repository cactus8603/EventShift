#!/usr/bin/env python3
# Build a CoSEC bidirectional event manifest from the official sequence layout.

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_timestamps(path: Path) -> list[int]:
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            values.append(int(line))
    return values


def maybe_rel(path: Path, base: Path | None) -> str:
    path = path.resolve()
    if base is None:
        return str(path)
    try:
        return str(path.relative_to(base.resolve()))
    except ValueError:
        return str(path)


def build_manifest(cosec_root: Path, out_path: Path, window_us: int, relative_to: Path | None) -> dict:
    cosec_root = cosec_root.resolve()
    samples = []
    per_sequence = {}
    missing = []
    for seq_dir in sorted(p for p in cosec_root.iterdir() if p.is_dir()):
        image_dir = seq_dir / "img_co_left"
        label_dir = seq_dir / "segment_co"
        event_h5 = seq_dir / "events_co_left.h5"
        timestamps_path = seq_dir / "timestamps.txt"
        required = [image_dir, label_dir, event_h5, timestamps_path]
        if not all(p.exists() for p in required):
            missing.append({"sequence": seq_dir.name, "missing": [str(p) for p in required if not p.exists()]})
            continue
        timestamps = read_timestamps(timestamps_path)
        seq_count = 0
        for image_path in sorted(image_dir.glob("*.png")):
            frame_id = int(image_path.stem)
            label_path = label_dir / image_path.name
            if frame_id >= len(timestamps) or not label_path.exists():
                missing.append({
                    "sequence": seq_dir.name,
                    "frame_id": frame_id,
                    "missing": [str(label_path)] if not label_path.exists() else ["timestamp"],
                })
                continue
            timestamp_us = int(timestamps[frame_id])
            samples.append({
                "sequence": seq_dir.name,
                "frame_id": frame_id,
                "image": maybe_rel(image_path, relative_to),
                "label": maybe_rel(label_path, relative_to),
                "event_h5": maybe_rel(event_h5, relative_to),
                "event_old": [timestamp_us - window_us, timestamp_us],
                "event_new": [timestamp_us, timestamp_us + window_us],
                "timestamp_us": timestamp_us,
                "valid": True,
            })
            seq_count += 1
        per_sequence[seq_dir.name] = seq_count
    payload = {
        "format": "eventshift_cosec_event_manifest_v1",
        "cosec_root": str(cosec_root),
        "window_us": int(window_us),
        "path_mode": "relative" if relative_to is not None else "absolute",
        "relative_to": str(relative_to.resolve()) if relative_to is not None else None,
        "num_samples": len(samples),
        "per_sequence": per_sequence,
        "missing": missing,
        "samples": samples,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cosec-root", type=Path, required=True, help="CoSEC train root containing Day_*/Night_* sequences")
    parser.add_argument("--out", type=Path, required=True, help="Output manifest JSON path")
    parser.add_argument("--window-us", type=int, default=50000, help="Bidirectional event window size in microseconds")
    parser.add_argument("--relative-to", type=Path, default=None, help="Store paths relative to this root instead of absolute paths")
    args = parser.parse_args()
    payload = build_manifest(args.cosec_root, args.out, args.window_us, args.relative_to)
    num_samples = payload["num_samples"]
    missing_count = len(payload["missing"])
    print(f"wrote {args.out}: {num_samples} samples, {missing_count} missing entries")
    for seq, count in payload["per_sequence"].items():
        print(f"  {seq}: {count}")
    return 0 if not payload["missing"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
