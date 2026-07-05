#!/usr/bin/env python3
"""Compare REAL prediction candidates without ground truth.

This tool is intentionally label-free: REAL test masks do not have GT, so it
only reports candidate class distributions and pairwise changes.
"""

import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image

from cosec_finetune_splits import CLASSES, PALETTE


NUM_CLASSES = len(CLASSES)


def parse_candidate(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"candidate must use name=path format, got: {value}"
        )
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError(f"empty candidate name in: {value}")
    return name, Path(path).expanduser().resolve()


def read_manifest(path):
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        return None
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_sequence_sources(candidate_path, real_only=True):
    manifest = read_manifest(candidate_path)
    if manifest is not None:
        sequences = manifest.get("counts", {}).get("sequences", {})
        if sequences:
            for seq, detail in sorted(sequences.items()):
                if real_only and not seq.startswith("REAL_"):
                    continue
                if isinstance(detail, dict) and detail.get("source"):
                    source = Path(detail["source"])
                else:
                    source = candidate_path / seq / "segment_co"
                if source.is_dir():
                    yield seq, source
            return

    for seq_dir in sorted(candidate_path.glob("REAL_*" if real_only else "*")):
        if not seq_dir.is_dir():
            continue
        source = seq_dir / "segment_co"
        if source.is_dir():
            yield seq_dir.name, source


def collect_masks(candidate_path, real_only=True):
    masks = {}
    missing_sources = []
    for seq, source in iter_sequence_sources(candidate_path, real_only=real_only):
        pngs = sorted(source.glob("*.png"))
        if not pngs:
            missing_sources.append(str(source))
            continue
        for png_path in pngs:
            masks[f"{seq}/{png_path.name}"] = png_path
    return masks, missing_sources


def load_mask(path):
    arr = np.asarray(Image.open(path))
    if arr.ndim == 2:
        return arr.astype(np.int64, copy=False)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        rgb = arr[..., :3]
        out = np.full(rgb.shape[:2], 255, dtype=np.int64)
        for idx, color in enumerate(PALETTE):
            matches = np.all(rgb == np.asarray(color, dtype=rgb.dtype), axis=-1)
            out[matches] = idx
        return out
    raise ValueError(f"Unsupported mask shape {arr.shape} for {path}")


def class_histogram(paths):
    hist = np.zeros(NUM_CLASSES, dtype=np.int64)
    ignored = 0
    pixels = 0
    shapes = Counter()
    for path in paths:
        mask = load_mask(path)
        shapes[str(tuple(mask.shape))] += 1
        pixels += int(mask.size)
        valid = (mask >= 0) & (mask < NUM_CLASSES)
        ignored += int(mask.size - valid.sum())
        hist += np.bincount(mask[valid].ravel(), minlength=NUM_CLASSES)
    return hist, pixels, ignored, shapes


def top_classes(hist, topk):
    total = int(hist.sum())
    rows = []
    for idx in np.argsort(hist)[::-1][:topk]:
        count = int(hist[idx])
        if count == 0:
            continue
        rows.append(
            {
                "class_id": int(idx),
                "class_name": CLASSES[idx],
                "pixels": count,
                "fraction": count / total if total else 0.0,
            }
        )
    return rows


def candidate_summary(name, path, masks, topk):
    hist, pixels, ignored, shapes = class_histogram(masks.values())
    return {
        "name": name,
        "path": str(path),
        "frames": len(masks),
        "total_pixels": pixels,
        "valid_pixels": int(hist.sum()),
        "ignored_pixels": ignored,
        "ignored_fraction": ignored / pixels if pixels else 0.0,
        "shapes": dict(shapes),
        "class_histogram": {
            CLASSES[idx]: int(value) for idx, value in enumerate(hist.tolist())
        },
        "class_fraction": {
            CLASSES[idx]: float(value / hist.sum()) if hist.sum() else 0.0
            for idx, value in enumerate(hist.tolist())
        },
        "top_classes": top_classes(hist, topk),
    }


def compare_pair(a_name, a_masks, b_name, b_masks, topk):
    common = sorted(set(a_masks) & set(b_masks))
    transitions = np.zeros(NUM_CLASSES * NUM_CLASSES, dtype=np.int64)
    hist_a = np.zeros(NUM_CLASSES, dtype=np.int64)
    hist_b = np.zeros(NUM_CLASSES, dtype=np.int64)
    changed = 0
    valid_pixels = 0
    skipped_shape = []
    frame_changes = []

    for key in common:
        a = load_mask(a_masks[key])
        b = load_mask(b_masks[key])
        if a.shape != b.shape:
            skipped_shape.append(
                {"frame": key, "shape_a": tuple(a.shape), "shape_b": tuple(b.shape)}
            )
            continue
        valid = (a >= 0) & (a < NUM_CLASSES) & (b >= 0) & (b < NUM_CLASSES)
        if not np.any(valid):
            continue
        av = a[valid].ravel()
        bv = b[valid].ravel()
        per_frame_valid = int(av.size)
        per_frame_changed = int(np.count_nonzero(av != bv))
        valid_pixels += per_frame_valid
        changed += per_frame_changed
        hist_a += np.bincount(av, minlength=NUM_CLASSES)
        hist_b += np.bincount(bv, minlength=NUM_CLASSES)
        transitions += np.bincount(av * NUM_CLASSES + bv, minlength=NUM_CLASSES**2)
        frame_changes.append(
            {
                "frame": key,
                "valid_pixels": per_frame_valid,
                "changed_pixels": per_frame_changed,
                "changed_fraction": per_frame_changed / per_frame_valid,
            }
        )

    transition_rows = []
    for flat_idx in np.argsort(transitions)[::-1]:
        count = int(transitions[flat_idx])
        if count == 0:
            break
        src = int(flat_idx // NUM_CLASSES)
        dst = int(flat_idx % NUM_CLASSES)
        if src == dst:
            continue
        transition_rows.append(
            {
                "from_class_id": src,
                "from_class_name": CLASSES[src],
                "to_class_id": dst,
                "to_class_name": CLASSES[dst],
                "pixels": count,
                "fraction_of_valid": count / valid_pixels if valid_pixels else 0.0,
                "fraction_of_changed": count / changed if changed else 0.0,
            }
        )
        if len(transition_rows) >= topk:
            break

    class_delta_rows = []
    for idx in range(NUM_CLASSES):
        delta = int(hist_b[idx] - hist_a[idx])
        if delta == 0:
            continue
        class_delta_rows.append(
            {
                "class_id": idx,
                "class_name": CLASSES[idx],
                "delta_pixels": delta,
                "delta_fraction_of_valid": delta / valid_pixels if valid_pixels else 0.0,
            }
        )
    class_delta_rows.sort(key=lambda row: abs(row["delta_pixels"]), reverse=True)

    frame_changes.sort(key=lambda row: row["changed_fraction"], reverse=True)
    return {
        "candidate_a": a_name,
        "candidate_b": b_name,
        "common_frames": len(common),
        "valid_pixels": valid_pixels,
        "changed_pixels": changed,
        "changed_fraction": changed / valid_pixels if valid_pixels else 0.0,
        "agreement_fraction": 1.0 - (changed / valid_pixels) if valid_pixels else 0.0,
        "skipped_shape_mismatch": skipped_shape[:topk],
        "top_transitions": transition_rows,
        "top_class_deltas": class_delta_rows[:topk],
        "top_changed_frames": frame_changes[:topk],
    }


def write_markdown(path, report, topk):
    lines = [
        "# REAL Prediction Candidate Analysis",
        "",
        "REAL has no GT here, so these numbers compare prediction masks only.",
        "",
        "## Candidates",
    ]
    for item in report["candidates"]:
        lines.append("")
        lines.append(f"### {item['name']}")
        lines.append(f"- path: `{item['path']}`")
        lines.append(f"- frames: {item['frames']}")
        lines.append(f"- valid pixels: {item['valid_pixels']:,}")
        if item["ignored_pixels"]:
            lines.append(
                f"- ignored pixels: {item['ignored_pixels']:,} "
                f"({item['ignored_fraction']:.4%})"
            )
        lines.append("- top classes:")
        for row in item["top_classes"][:topk]:
            lines.append(
                f"  - {row['class_name']}: {row['fraction']:.2%} "
                f"({row['pixels']:,})"
            )

    lines.append("")
    lines.append("## Pairwise Changes")
    for pair in report["pairwise"]:
        lines.append("")
        lines.append(f"### {pair['candidate_a']} -> {pair['candidate_b']}")
        lines.append(f"- common frames: {pair['common_frames']}")
        lines.append(f"- agreement: {pair['agreement_fraction']:.2%}")
        lines.append(
            f"- changed: {pair['changed_fraction']:.2%} "
            f"({pair['changed_pixels']:,}/{pair['valid_pixels']:,})"
        )
        if pair["top_transitions"]:
            lines.append("- top transitions:")
            for row in pair["top_transitions"][:topk]:
                lines.append(
                    f"  - {row['from_class_name']} -> {row['to_class_name']}: "
                    f"{row['fraction_of_changed']:.2%} of changed "
                    f"({row['pixels']:,})"
                )
        if pair["top_class_deltas"]:
            lines.append("- largest class-count deltas:")
            for row in pair["top_class_deltas"][:topk]:
                sign = "+" if row["delta_pixels"] > 0 else ""
                lines.append(
                    f"  - {row['class_name']}: {sign}{row['delta_pixels']:,} "
                    f"({sign}{row['delta_fraction_of_valid']:.2%})"
                )
        if pair["top_changed_frames"]:
            lines.append("- most changed frames:")
            for row in pair["top_changed_frames"][: min(5, topk)]:
                lines.append(
                    f"  - {row['frame']}: {row['changed_fraction']:.2%}"
                )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        action="append",
        type=parse_candidate,
        required=True,
        help="Candidate in name=path format. Path may be a raw prediction dir or a manifest dir.",
    )
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=12)
    parser.add_argument("--include-non-real", action="store_true")
    args = parser.parse_args()

    mask_maps = {}
    candidates = []
    missing_sources = {}
    for name, path in args.candidate:
        masks, missing = collect_masks(path, real_only=not args.include_non_real)
        if not masks:
            raise RuntimeError(f"No masks found for candidate {name}: {path}")
        mask_maps[name] = masks
        missing_sources[name] = missing
        candidates.append(candidate_summary(name, path, masks, args.topk))

    pairwise = []
    for (a_name, _), (b_name, _) in combinations(args.candidate, 2):
        pairwise.append(
            compare_pair(
                a_name,
                mask_maps[a_name],
                b_name,
                mask_maps[b_name],
                args.topk,
            )
        )

    report = {
        "classes": list(CLASSES),
        "real_only": not args.include_non_real,
        "candidates": candidates,
        "missing_sources": missing_sources,
        "pairwise": pairwise,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_markdown(args.out_md, report, args.topk)

    print(f"wrote {args.out_json}")
    print(f"wrote {args.out_md}")
    for item in candidates:
        top = ", ".join(
            f"{row['class_name']} {row['fraction']:.1%}"
            for row in item["top_classes"][:3]
        )
        print(f"{item['name']}: frames={item['frames']} top={top}")
    for pair in pairwise:
        print(
            f"{pair['candidate_a']} -> {pair['candidate_b']}: "
            f"agreement={pair['agreement_fraction']:.2%}, "
            f"changed={pair['changed_fraction']:.2%}"
        )


if __name__ == "__main__":
    main()
