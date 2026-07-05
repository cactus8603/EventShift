#!/usr/bin/env python
"""Filter NightCity images whose class distribution is far from CoSEC."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

def _eventshift_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "configs").is_dir() and (parent / "third_party").is_dir():
            return parent
    return Path(__file__).resolve().parents[1]


ROOT = _eventshift_root()
sys.path.insert(0, str(ROOT / "tools"))

from cosec_finetune_splits import CLASSES, iter_cosec_samples  # noqa: E402
from nightcity_dataset import load_nightcity_dicts  # noqa: E402


NUM_CLASSES = len(CLASSES)
IGNORE_LABEL = 255


def label_hist(label_path):
    label = np.asarray(Image.open(label_path))
    if label.ndim == 3:
        label = label[:, :, 0]
    valid = label[label != IGNORE_LABEL]
    hist = np.bincount(valid.astype(np.int64), minlength=NUM_CLASSES)
    return hist[:NUM_CLASSES].astype(np.float64)


def normalize(hist, eps=1e-12):
    total = float(hist.sum())
    if total <= 0:
        return np.full(NUM_CLASSES, 1.0 / NUM_CLASSES, dtype=np.float64)
    return (hist + eps) / (total + eps * len(hist))


def js_divergence_matrix(query, reference):
    """Return JS divergence for every query/reference pair.

    query: [N, C], reference: [M, C].
    """
    query = np.asarray(query, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    mix = 0.5 * (query[:, None, :] + reference[None, :, :])
    q_kl = np.sum(query[:, None, :] * (np.log(query[:, None, :]) - np.log(mix)), axis=2)
    r_kl = np.sum(reference[None, :, :] * (np.log(reference[None, :, :]) - np.log(mix)), axis=2)
    return 0.5 * (q_kl + r_kl)


def js_divergence_vector(query, reference):
    mix = 0.5 * (query + reference)
    return 0.5 * (
        np.sum(query * (np.log(query) - np.log(mix)), axis=1)
        + np.sum(reference * (np.log(reference) - np.log(mix)), axis=1)
    )


def load_cosec_train_records(limit=None):
    records = []
    for seq_name, frame_id, image_path, label_path in iter_cosec_samples(ROOT / "data" / "train", "train"):
        records.append(
            {
                "image_id": f"{seq_name}_{frame_id:06d}",
                "file_name": str(image_path),
                "sem_seg_file_name": str(label_path),
            }
        )
        if limit is not None and len(records) >= limit:
            break
    return records


def compute_hists(records):
    hists = []
    dists = []
    for record in records:
        hist = label_hist(record["sem_seg_file_name"])
        hists.append(hist)
        dists.append(normalize(hist))
    return np.stack(hists, axis=0), np.stack(dists, axis=0)


def top_classes(dist, topk=5):
    indices = np.argsort(-dist)[:topk]
    return [{"class": CLASSES[index], "ratio": float(dist[index])} for index in indices if dist[index] > 0]


def make_manifest(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cosec_records = load_cosec_train_records(args.cosec_limit)
    nightcity_records = load_nightcity_dicts(args.nightcity_split, limit=args.nightcity_limit)

    cosec_hists, cosec_dists = compute_hists(cosec_records)
    nightcity_hists, nightcity_dists = compute_hists(nightcity_records)

    cosec_global = normalize(cosec_hists.sum(axis=0))
    nightcity_global = normalize(nightcity_hists.sum(axis=0))

    pair_js = js_divergence_matrix(nightcity_dists, cosec_dists)
    nearest_js = pair_js.min(axis=1)
    nearest_index = pair_js.argmin(axis=1)
    global_js = js_divergence_vector(nightcity_dists, np.broadcast_to(cosec_global, nightcity_dists.shape))

    rare_mask = cosec_global < args.rare_threshold
    rare_mass = nightcity_dists[:, rare_mask].sum(axis=1)
    score = nearest_js + args.global_weight * global_js + args.rare_weight * rare_mass

    order = np.argsort(score)
    keep_count = int(round(len(order) * args.keep_ratio))
    keep_count = max(1, min(len(order), keep_count))
    keep_indices = set(order[:keep_count].tolist())

    entries = []
    for index, record in enumerate(nightcity_records):
        dist = nightcity_dists[index]
        entries.append(
            {
                "rank": int(np.where(order == index)[0][0] + 1),
                "keep": index in keep_indices,
                "score": float(score[index]),
                "nearest_js": float(nearest_js[index]),
                "global_js": float(global_js[index]),
                "rare_mass": float(rare_mass[index]),
                "image_id": record["image_id"],
                "file_name": record["file_name"],
                "sem_seg_file_name": record["sem_seg_file_name"],
                "nightcity_split": record.get("nightcity_split", args.nightcity_split),
                "nearest_cosec_image_id": cosec_records[int(nearest_index[index])]["image_id"],
                "top_classes": top_classes(dist, args.topk),
            }
        )

    entries.sort(key=lambda item: item["score"])
    kept_entries = [entry for entry in entries if entry["keep"]]
    rejected_entries = [entry for entry in entries if not entry["keep"]]

    kept_indices_sorted = [nightcity_records.index(next(r for r in nightcity_records if r["image_id"] == entry["image_id"])) for entry in kept_entries]
    kept_global = normalize(nightcity_hists[kept_indices_sorted].sum(axis=0))

    quantiles = {}
    for ratio in [0.25, 0.33, 0.5, 0.67, 0.75]:
        count = max(1, min(len(order), int(round(len(order) * ratio))))
        quantiles[f"keep_{int(round(ratio * 100))}"] = {
            "count": count,
            "score_threshold": float(score[order[count - 1]]),
        }

    manifest = {
        "method": "nightcity_class_distribution_filter",
        "classes": list(CLASSES),
        "cosec_records": len(cosec_records),
        "nightcity_split": args.nightcity_split,
        "nightcity_records": len(nightcity_records),
        "keep_ratio": args.keep_ratio,
        "keep_count": keep_count,
        "reject_count": len(nightcity_records) - keep_count,
        "score": {
            "formula": "nearest_js + global_weight * global_js + rare_weight * rare_mass",
            "global_weight": args.global_weight,
            "rare_weight": args.rare_weight,
            "rare_threshold": args.rare_threshold,
        },
        "quantiles": quantiles,
        "cosec_global_distribution": {CLASSES[i]: float(cosec_global[i]) for i in range(NUM_CLASSES)},
        "nightcity_global_distribution": {CLASSES[i]: float(nightcity_global[i]) for i in range(NUM_CLASSES)},
        "kept_nightcity_global_distribution": {CLASSES[i]: float(kept_global[i]) for i in range(NUM_CLASSES)},
        "kept": kept_entries,
        "rejected": rejected_entries,
    }

    keep_label = str(args.keep_ratio).replace(".", "p")
    split_label = "" if args.nightcity_split == "train" else f"{args.nightcity_split}_"
    output_path = output_dir / f"nightcity_{split_label}cosec_classdist_keep{keep_label}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    summary_path = output_dir / f"nightcity_{split_label}cosec_classdist_keep{keep_label}_summary.md"
    write_summary(summary_path, manifest)

    latest_name = args.latest_name
    if latest_name is None:
        latest_name = "nightcity_cosec_classdist_filtered.json" if args.nightcity_split == "train" else f"nightcity_{args.nightcity_split}_cosec_classdist_filtered.json"
    latest_path = output_dir / latest_name
    with latest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(f"wrote manifest: {output_path}")
    print(f"wrote summary: {summary_path}")
    print(f"wrote latest copy: {latest_path}")
    print(f"kept {keep_count}/{len(nightcity_records)} NightCity {args.nightcity_split} images")
    print("class distribution distance:")
    print(f"  all NightCity JS(global): {float(js_divergence_vector(nightcity_global[None, :], cosec_global[None, :])[0]):.6f}")
    print(f"  kept NightCity JS(global): {float(js_divergence_vector(kept_global[None, :], cosec_global[None, :])[0]):.6f}")
    return output_path


def write_summary(path, manifest):
    rows = []
    for cls in manifest["classes"]:
        rows.append(
            (
                cls,
                manifest["cosec_global_distribution"][cls],
                manifest["nightcity_global_distribution"][cls],
                manifest["kept_nightcity_global_distribution"][cls],
            )
        )
    largest_shift = sorted(rows, key=lambda item: abs(item[2] - item[1]), reverse=True)[:8]
    largest_kept_shift = sorted(rows, key=lambda item: abs(item[3] - item[1]), reverse=True)[:8]

    lines = [
        "# NightCity Class-Distribution Filter",
        "",
        f"- CoSEC train records: {manifest['cosec_records']}",
        f"- NightCity split: {manifest.get('nightcity_split', 'train')}",
        f"- NightCity records: {manifest['nightcity_records']}",
        f"- Kept NightCity records: {manifest['keep_count']}",
        f"- Rejected NightCity records: {manifest['reject_count']}",
        f"- Keep ratio: {manifest['keep_ratio']}",
        "",
        "## Score",
        "",
        f"`{manifest['score']['formula']}`",
        "",
        "## Keep Quantiles",
        "",
        "| Setting | Count | Score threshold |",
        "| --- | ---: | ---: |",
    ]
    for name, value in manifest["quantiles"].items():
        lines.append(f"| `{name}` | {value['count']} | {value['score_threshold']:.6f} |")
    lines.extend(
        [
            "",
            "## Largest Original NightCity Shifts",
            "",
            "| Class | CoSEC train | NightCity train | Kept NightCity |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for cls, cosec_ratio, nightcity_ratio, kept_ratio in largest_shift:
        lines.append(f"| {cls} | {cosec_ratio:.4f} | {nightcity_ratio:.4f} | {kept_ratio:.4f} |")
    lines.extend(
        [
            "",
            "## Largest Remaining Kept Shifts",
            "",
            "| Class | CoSEC train | NightCity train | Kept NightCity |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for cls, cosec_ratio, nightcity_ratio, kept_ratio in largest_kept_shift:
        lines.append(f"| {cls} | {cosec_ratio:.4f} | {nightcity_ratio:.4f} | {kept_ratio:.4f} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep-ratio", type=float, default=0.5)
    parser.add_argument("--nightcity-split", choices=["train", "val", "trainval"], default="train")
    parser.add_argument("--global-weight", type=float, default=0.25)
    parser.add_argument("--rare-weight", type=float, default=0.5)
    parser.add_argument("--rare-threshold", type=float, default=0.001)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--cosec-limit", type=int, default=None)
    parser.add_argument("--nightcity-limit", type=int, default=None)
    parser.add_argument("--output-dir", default=str(ROOT / "work_dirs" / "manifests"))
    parser.add_argument("--latest-name", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    make_manifest(parse_args())
