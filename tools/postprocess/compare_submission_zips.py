#!/usr/bin/env python
"""Compare two CoSEC submission zips by mask names and changed pixels."""

import argparse
import json
import struct
import zipfile
import zlib
from collections import defaultdict
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None


DOMAIN_PREFIXES = {
    "day": "Day_",
    "night": "Night_",
    "real": "REAL_",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Reference submission zip.")
    parser.add_argument("--candidate", required=True, help="Candidate submission zip.")
    parser.add_argument("--out", default=None, help="Optional JSON report path.")
    return parser.parse_args()


def list_png_entries(zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        return sorted(
            name
            for name in zf.namelist()
            if name.endswith(".png") and not name.endswith("/")
        )


def unfilter_scanline(filter_type, raw, prior, bpp):
    out = bytearray(raw)
    for idx in range(len(out)):
        left = out[idx - bpp] if idx >= bpp else 0
        up = prior[idx] if prior is not None else 0
        up_left = prior[idx - bpp] if prior is not None and idx >= bpp else 0
        if filter_type == 0:
            value = out[idx]
        elif filter_type == 1:
            value = out[idx] + left
        elif filter_type == 2:
            value = out[idx] + up
        elif filter_type == 3:
            value = out[idx] + ((left + up) // 2)
        elif filter_type == 4:
            predictor = left + up - up_left
            pa = abs(predictor - left)
            pb = abs(predictor - up)
            pc = abs(predictor - up_left)
            if pa <= pb and pa <= pc:
                value = out[idx] + left
            elif pb <= pc:
                value = out[idx] + up
            else:
                value = out[idx] + up_left
        else:
            raise ValueError(f"Unsupported PNG filter type: {filter_type}")
        out[idx] = value & 0xFF
    return bytes(out)


def decode_png_mask(data, name):
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"Not a PNG file: {name}")
    pos = 8
    width = height = bit_depth = color_type = interlace = None
    idat_parts = []
    while pos < len(data):
        if pos + 8 > len(data):
            raise ValueError(f"Truncated PNG chunk header: {name}")
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        chunk_type = data[pos + 4 : pos + 8]
        chunk_data = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, _, _, interlace = struct.unpack(
                ">IIBBBBB",
                chunk_data,
            )
        elif chunk_type == b"IDAT":
            idat_parts.append(chunk_data)
        elif chunk_type == b"IEND":
            break

    if width is None or height is None:
        raise ValueError(f"Missing IHDR: {name}")
    if bit_depth != 8:
        raise ValueError(f"Unsupported PNG bit depth {bit_depth} in {name}")
    if interlace != 0:
        raise ValueError(f"Unsupported interlaced PNG in {name}")

    channels_by_type = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
    if color_type not in channels_by_type:
        raise ValueError(f"Unsupported PNG color type {color_type} in {name}")
    channels = channels_by_type[color_type]
    row_size = width * channels
    raw = zlib.decompress(b"".join(idat_parts))
    rows = []
    prior = None
    offset = 0
    for _ in range(height):
        filter_type = raw[offset]
        offset += 1
        scanline = raw[offset : offset + row_size]
        offset += row_size
        row = unfilter_scanline(filter_type, scanline, prior, channels)
        rows.append(row)
        prior = row
    if channels == 1:
        return (height, width), b"".join(rows)
    first_channel_rows = [row[0::channels] for row in rows]
    return (height, width), b"".join(first_channel_rows)


def read_mask(zf, name):
    if cv2 is not None and np is not None:
        data = np.frombuffer(zf.read(name), dtype=np.uint8)
        mask = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError(f"Could not decode {name}")
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        return mask.shape, mask
    return decode_png_mask(zf.read(name), name)


def read_png_shape_header(zf, name):
    with zf.open(name) as handle:
        header = handle.read(24)
    if not header.startswith(b"\x89PNG\r\n\x1a\n") or header[12:16] != b"IHDR":
        raise ValueError(f"Not a PNG file or missing IHDR: {name}")
    width, height = struct.unpack(">II", header[16:24])
    return height, width


def changed_pixel_count(base_mask, candidate_mask):
    if np is not None and not isinstance(base_mask, bytes):
        return int(np.count_nonzero(base_mask != candidate_mask)), int(base_mask.size)
    return (
        sum(1 for lhs, rhs in zip(base_mask, candidate_mask) if lhs != rhs),
        len(base_mask),
    )


def domain_for_name(name):
    for domain, prefix in DOMAIN_PREFIXES.items():
        if name.startswith(prefix):
            return domain
    return "other"


def empty_stats():
    return {
        "files": 0,
        "changed_files": 0,
        "pixels": 0,
        "changed_pixels": 0,
    }


def finalize_stats(stats):
    out = dict(stats)
    pixels = out["pixels"]
    files = out["files"]
    out["changed_pixel_rate"] = float(out["changed_pixels"] / pixels) if pixels else 0.0
    out["changed_file_rate"] = float(out["changed_files"] / files) if files else 0.0
    return out


def main():
    args = parse_args()
    base_path = Path(args.base)
    candidate_path = Path(args.candidate)
    base_entries = list_png_entries(base_path)
    candidate_entries = list_png_entries(candidate_path)
    base_set = set(base_entries)
    candidate_set = set(candidate_entries)
    common = sorted(base_set & candidate_set)

    report = {
        "base": str(base_path.resolve()),
        "candidate": str(candidate_path.resolve()),
        "base_entries": len(base_entries),
        "candidate_entries": len(candidate_entries),
        "common_entries": len(common),
        "missing_in_candidate": sorted(base_set - candidate_set),
        "extra_in_candidate": sorted(candidate_set - base_set),
        "by_domain": defaultdict(empty_stats),
        "total": empty_stats(),
    }

    with zipfile.ZipFile(base_path) as base_zf, zipfile.ZipFile(candidate_path) as candidate_zf:
        for name in common:
            base_info = base_zf.getinfo(name)
            candidate_info = candidate_zf.getinfo(name)
            if (
                base_info.CRC == candidate_info.CRC
                and base_info.file_size == candidate_info.file_size
            ):
                height, width = read_png_shape_header(base_zf, name)
                changed = 0
                pixels = height * width
            else:
                base_shape, base_mask = read_mask(base_zf, name)
                candidate_shape, candidate_mask = read_mask(candidate_zf, name)
                if base_shape != candidate_shape:
                    raise ValueError(f"Shape mismatch for {name}: {base_shape} vs {candidate_shape}")
                if isinstance(base_mask, bytes) and len(base_mask) != len(candidate_mask):
                    raise ValueError(f"Decoded byte length mismatch for {name}")
                changed, pixels = changed_pixel_count(base_mask, candidate_mask)
            domain = domain_for_name(name)
            for bucket in (report["by_domain"][domain], report["total"]):
                bucket["files"] += 1
                bucket["pixels"] += pixels
                bucket["changed_pixels"] += changed
                if changed:
                    bucket["changed_files"] += 1

    report["by_domain"] = {
        domain: finalize_stats(stats)
        for domain, stats in sorted(report["by_domain"].items())
    }
    report["total"] = finalize_stats(report["total"])

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
