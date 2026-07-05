#!/usr/bin/env python
"""Explain semantic changes between two submission zips or prediction dirs.

The report is meant to answer whether a candidate mostly changes boundary
pixels, or whether it rewrites whole semantic objects/interiors.
"""

import argparse
import json
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


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

DOMAIN_PREFIXES = {
    "day": "Day_",
    "night": "Night_",
    "real": "REAL_",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Reference zip or prediction dir.")
    parser.add_argument("--candidate", required=True, help="Candidate zip or prediction dir.")
    parser.add_argument("--out", required=True, help="Output JSON report.")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--component-top-k", type=int, default=40)
    parser.add_argument("--boundary-radii", default="2,5")
    parser.add_argument(
        "--boundary-source",
        choices=["base", "candidate", "either"],
        default="base",
        help="Prediction map used to build semantic boundary bands.",
    )
    parser.add_argument("--small-component-area", type=int, default=5000)
    parser.add_argument("--boundary-like-rate", type=float, default=0.75)
    parser.add_argument("--mixed-rate", type=float, default=0.40)
    return parser.parse_args()


def parse_radii(text):
    radii = sorted({int(part) for part in text.split(",") if part.strip()})
    if not radii:
        raise ValueError("At least one boundary radius is required.")
    return radii


def is_zip(path):
    return Path(path).is_file() and str(path).lower().endswith(".zip")


class MaskSource:
    def __init__(self, path):
        self.path = Path(path)
        self.zf = zipfile.ZipFile(self.path) if is_zip(self.path) else None

    def close(self):
        if self.zf is not None:
            self.zf.close()

    def entries(self):
        if self.zf is not None:
            return sorted(
                name
                for name in self.zf.namelist()
                if name.endswith(".png") and not name.endswith("/")
            )
        return sorted(
            path.relative_to(self.path).as_posix()
            for path in self.path.rglob("*.png")
            if path.is_file()
        )

    def read(self, rel):
        if self.zf is not None:
            data = np.frombuffer(self.zf.read(rel), dtype=np.uint8)
            mask = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        else:
            mask = cv2.imread(str(self.path / rel), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError(f"Could not read mask {rel} from {self.path}")
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        return mask.astype(np.uint8, copy=False)


def domain_for_name(name):
    first = Path(name).parts[0] if Path(name).parts else name
    for domain, prefix in DOMAIN_PREFIXES.items():
        if first.startswith(prefix):
            return domain
    return "other"


def seq_for_name(name):
    parts = Path(name).parts
    return parts[0] if parts else "unknown"


def semantic_boundary_band(pred, radius):
    edge = np.zeros(pred.shape, dtype=bool)
    edge[:, 1:] |= pred[:, 1:] != pred[:, :-1]
    edge[:, :-1] |= pred[:, 1:] != pred[:, :-1]
    edge[1:, :] |= pred[1:, :] != pred[:-1, :]
    edge[:-1, :] |= pred[1:, :] != pred[:-1, :]
    if radius <= 0:
        return edge
    kernel = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=np.uint8)
    return cv2.dilate(edge.astype(np.uint8), kernel, iterations=1) > 0


def boundary_bands(base, candidate, radii, source):
    out = {}
    for radius in radii:
        base_band = semantic_boundary_band(base, radius)
        if source == "base":
            out[radius] = base_band
            continue
        candidate_band = semantic_boundary_band(candidate, radius)
        if source == "candidate":
            out[radius] = candidate_band
        else:
            out[radius] = base_band | candidate_band
    return out


def empty_bucket():
    return {
        "images": 0,
        "pixels": 0,
        "changed": 0,
        "boundary": defaultdict(int),
    }


def update_bucket(bucket, pixels, changed, boundary_counts):
    bucket["images"] += 1
    bucket["pixels"] += int(pixels)
    bucket["changed"] += int(changed)
    for radius, count in boundary_counts.items():
        bucket["boundary"][str(radius)] += int(count)


def finalize_bucket(bucket, radii):
    out = {
        "images": int(bucket["images"]),
        "pixels": int(bucket["pixels"]),
        "changed": int(bucket["changed"]),
        "changed_rate": float(bucket["changed"] / bucket["pixels"]) if bucket["pixels"] else 0.0,
    }
    for radius in radii:
        boundary = int(bucket["boundary"].get(str(radius), 0))
        out[f"boundary{radius}"] = boundary
        out[f"boundary{radius}_rate_of_changed"] = (
            float(boundary / bucket["changed"]) if bucket["changed"] else 0.0
        )
    max_radius = max(radii)
    max_boundary = int(bucket["boundary"].get(str(max_radius), 0))
    interior = int(bucket["changed"] - max_boundary)
    out[f"interior{max_radius}"] = interior
    out[f"interior{max_radius}_rate_of_changed"] = (
        float(interior / bucket["changed"]) if bucket["changed"] else 0.0
    )
    return out


def class_rows(counter, key_name, total_changed, top_k):
    rows = []
    for class_id, count in counter.most_common(top_k):
        class_id = int(class_id)
        rows.append(
            {
                "class_id": class_id,
                "class": CLASSES[class_id] if 0 <= class_id < len(CLASSES) else str(class_id),
                key_name: int(count),
                "share_of_changed": float(count / total_changed) if total_changed else 0.0,
            }
        )
    return rows


def transition_name(pair_idx):
    num_classes = len(CLASSES)
    src = int(pair_idx // num_classes)
    dst = int(pair_idx % num_classes)
    src_name = CLASSES[src] if 0 <= src < len(CLASSES) else str(src)
    dst_name = CLASSES[dst] if 0 <= dst < len(CLASSES) else str(dst)
    return src, dst, f"{src_name}->{dst_name}"


def transition_rows(pair_counter, pair_boundary_counter, max_radius, total_changed, top_k):
    rows = []
    for pair_idx, count in pair_counter.most_common(top_k):
        src, dst, pair = transition_name(pair_idx)
        boundary = int(pair_boundary_counter.get(pair_idx, 0))
        interior = int(count - boundary)
        rows.append(
            {
                "from_class_id": src,
                "to_class_id": dst,
                "transition": pair,
                "pixels": int(count),
                "share_of_changed": float(count / total_changed) if total_changed else 0.0,
                f"boundary{max_radius}": boundary,
                f"boundary{max_radius}_share": float(boundary / count) if count else 0.0,
                f"interior{max_radius}": interior,
                f"interior{max_radius}_share": float(interior / count) if count else 0.0,
            }
        )
    return rows


def component_type(area, boundary_rate, args):
    if boundary_rate >= args.boundary_like_rate:
        return "boundary_like"
    if area <= args.small_component_area:
        return "interior_like_small"
    if boundary_rate >= args.mixed_rate:
        return "mixed"
    return "object_or_interior_like"


def collect_components(name, changed, base, candidate, boundary, args):
    if not np.any(changed):
        return []
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        changed.astype(np.uint8),
        connectivity=8,
    )
    components = []
    num_classes = len(CLASSES)
    domain = domain_for_name(name)
    seq = seq_for_name(name)
    image = Path(name).name
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        comp = labels == label
        boundary_pixels = int(np.count_nonzero(comp & boundary))
        boundary_rate = boundary_pixels / area
        pair_ids = num_classes * base[comp].astype(np.int64) + candidate[comp].astype(np.int64)
        pair_counts = np.bincount(pair_ids, minlength=num_classes**2)
        dominant_idx = int(np.argmax(pair_counts))
        _, _, dominant = transition_name(dominant_idx)
        dominant_pixels = int(pair_counts[dominant_idx])
        components.append(
            {
                "bucket": domain,
                "seq": seq,
                "image": image,
                "area": area,
                "bbox": [
                    int(stats[label, cv2.CC_STAT_LEFT]),
                    int(stats[label, cv2.CC_STAT_TOP]),
                    int(stats[label, cv2.CC_STAT_WIDTH]),
                    int(stats[label, cv2.CC_STAT_HEIGHT]),
                ],
                "boundary_rate": float(boundary_rate),
                "interior_rate": float(1.0 - boundary_rate),
                "dominant_transition": dominant,
                "dominant_transition_pixels": dominant_pixels,
                "dominant_transition_rate": float(dominant_pixels / area),
                "type": component_type(area, boundary_rate, args),
            }
        )
    return components


def count_pairs(base, candidate, mask, boundary_mask):
    num_classes = len(CLASSES)
    pair_ids = num_classes * base[mask].astype(np.int64) + candidate[mask].astype(np.int64)
    counts = np.bincount(pair_ids, minlength=num_classes**2)
    boundary_pair_ids = (
        num_classes * base[mask & boundary_mask].astype(np.int64)
        + candidate[mask & boundary_mask].astype(np.int64)
    )
    boundary_counts = np.bincount(boundary_pair_ids, minlength=num_classes**2)
    return counts, boundary_counts


def main():
    args = parse_args()
    radii = parse_radii(args.boundary_radii)
    max_radius = max(radii)
    base_source = MaskSource(args.base)
    candidate_source = MaskSource(args.candidate)
    try:
        base_entries = set(base_source.entries())
        candidate_entries = set(candidate_source.entries())
        common = sorted(base_entries & candidate_entries)

        total = empty_bucket()
        by_domain = defaultdict(empty_bucket)
        by_sequence = defaultdict(empty_bucket)
        from_counter = Counter()
        to_counter = Counter()
        pair_counter = Counter()
        pair_boundary_counter = Counter()
        components = []
        invalid_changed_pixels = 0

        for rel in common:
            base = base_source.read(rel)
            candidate = candidate_source.read(rel)
            if base.shape != candidate.shape:
                raise ValueError(f"Shape mismatch for {rel}: {base.shape} vs {candidate.shape}")

            changed_all = base != candidate
            valid = (
                (base >= 0)
                & (base < len(CLASSES))
                & (candidate >= 0)
                & (candidate < len(CLASSES))
            )
            changed = changed_all & valid
            invalid_changed_pixels += int(np.count_nonzero(changed_all & ~valid))

            bands = boundary_bands(base, candidate, radii, args.boundary_source)
            boundary_counts = {
                radius: int(np.count_nonzero(changed & bands[radius])) for radius in radii
            }
            changed_count = int(np.count_nonzero(changed))
            pixels = int(base.size)

            update_bucket(total, pixels, changed_count, boundary_counts)
            update_bucket(by_domain[domain_for_name(rel)], pixels, changed_count, boundary_counts)
            update_bucket(by_sequence[seq_for_name(rel)], pixels, changed_count, boundary_counts)

            if changed_count:
                from_counter.update(base[changed].astype(int).tolist())
                to_counter.update(candidate[changed].astype(int).tolist())
                pair_counts, boundary_pair_counts = count_pairs(
                    base,
                    candidate,
                    changed,
                    bands[max_radius],
                )
                for idx in np.flatnonzero(pair_counts):
                    pair_counter[int(idx)] += int(pair_counts[idx])
                for idx in np.flatnonzero(boundary_pair_counts):
                    pair_boundary_counter[int(idx)] += int(boundary_pair_counts[idx])
                components.extend(
                    collect_components(rel, changed, base, candidate, bands[max_radius], args)
                )

        total_out = finalize_bucket(total, radii)
        components.sort(key=lambda item: item["area"], reverse=True)
        component_type_counts = Counter(item["type"] for item in components)
        report = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "base": str(Path(args.base).resolve()),
            "candidate": str(Path(args.candidate).resolve()),
            "base_entries": len(base_entries),
            "candidate_entries": len(candidate_entries),
            "common_entries": len(common),
            "missing_in_candidate": sorted(base_entries - candidate_entries),
            "extra_in_candidate": sorted(candidate_entries - base_entries),
            "classes": CLASSES,
            "num_classes": len(CLASSES),
            "boundary_source": args.boundary_source,
            "boundary_radii": radii,
            "changed_pixels": total_out["changed"],
            "changed_rate": total_out["changed_rate"],
            "invalid_changed_pixels": invalid_changed_pixels,
            "bucket_stats": {
                domain: finalize_bucket(bucket, radii)
                for domain, bucket in sorted(by_domain.items())
            },
            "seq_stats": {
                seq: finalize_bucket(bucket, radii) for seq, bucket in sorted(by_sequence.items())
            },
            "from_class": class_rows(
                from_counter,
                "removed_pixels",
                total_out["changed"],
                args.top_k,
            ),
            "to_class": class_rows(
                to_counter,
                "added_pixels",
                total_out["changed"],
                args.top_k,
            ),
            "top_transitions": transition_rows(
                pair_counter,
                pair_boundary_counter,
                max_radius,
                total_out["changed"],
                args.top_k,
            ),
            "component_type_counts": dict(sorted(component_type_counts.items())),
            "top_components": components[: args.component_top_k],
        }
    finally:
        base_source.close()
        candidate_source.close()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
