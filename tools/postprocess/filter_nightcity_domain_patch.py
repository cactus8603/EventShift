#!/usr/bin/env python
"""Select a tiny NightCity subset as a CoSEC-night domain patch.

This filter is intentionally stricter than the class-distribution filter.  It
uses CoSEC train-night frames as the reference and ranks NightCity images by
both semantic-label distribution and simple low-light image statistics.
"""

import argparse
import json
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
from filter_nightcity_by_cosec_distribution import (  # noqa: E402
    compute_hists,
    js_divergence_matrix,
    js_divergence_vector,
    normalize,
    top_classes,
)
from nightcity_dataset import load_nightcity_dicts  # noqa: E402


NUM_CLASSES = len(CLASSES)


def load_cosec_train_records(domain="night"):
    records = []
    for seq_name, frame_id, image_path, label_path in iter_cosec_samples(ROOT / "data" / "train", "train"):
        is_night = seq_name.startswith("Night_")
        if domain == "night" and not is_night:
            continue
        if domain == "day" and is_night:
            continue
        records.append(
            {
                "image_id": f"{seq_name}_{frame_id:06d}",
                "file_name": str(image_path),
                "sem_seg_file_name": str(label_path),
                "cosec_seq": seq_name,
                "cosec_frame": int(frame_id),
            }
        )
    return records


def image_domain_feature(image_path, resize=128):
    image = Image.open(image_path).convert("RGB")
    if resize and max(image.size) > resize:
        resample = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
        image.thumbnail((resize, resize), resample)
    array = np.asarray(image, dtype=np.float32) / 255.0
    channels = array.reshape(-1, 3)
    lum = 0.2126 * channels[:, 0] + 0.7152 * channels[:, 1] + 0.0722 * channels[:, 2]
    max_channel = channels.max(axis=1)
    min_channel = channels.min(axis=1)
    saturation = np.zeros_like(max_channel)
    np.divide(max_channel - min_channel, max_channel, out=saturation, where=max_channel > 1e-6)
    p10, p50, p90 = np.percentile(lum, [10, 50, 90])
    return np.array(
        [
            channels[:, 0].mean(),
            channels[:, 1].mean(),
            channels[:, 2].mean(),
            channels[:, 0].std(),
            channels[:, 1].std(),
            channels[:, 2].std(),
            lum.mean(),
            lum.std(),
            p10,
            p50,
            p90,
            (lum < 0.15).mean(),
            (lum < 0.25).mean(),
            (lum > 0.65).mean(),
            saturation.mean(),
            saturation.std(),
        ],
        dtype=np.float64,
    )


def compute_image_features(records, resize=128):
    return np.stack([image_domain_feature(record["file_name"], resize=resize) for record in records], axis=0)


def chunked_min_l2(query, reference, chunk_size=512):
    out = []
    nearest = []
    for start in range(0, len(query), chunk_size):
        stop = min(len(query), start + chunk_size)
        diff = query[start:stop, None, :] - reference[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        out.append(dist.min(axis=1))
        nearest.append(dist.argmin(axis=1))
    return np.concatenate(out, axis=0), np.concatenate(nearest, axis=0)


def robust_unit(values):
    values = np.asarray(values, dtype=np.float64)
    p90 = np.percentile(values, 90)
    if p90 <= 1e-12:
        return values
    return values / p90


def class_gap_table(classes, reference, candidates, kept):
    rows = []
    for index, class_name in enumerate(classes):
        rows.append(
            {
                "class": class_name,
                "reference": float(reference[index]),
                "candidate_all": float(candidates[index]),
                "kept": float(kept[index]),
                "gap_all": float(abs(candidates[index] - reference[index])),
                "gap_kept": float(abs(kept[index] - reference[index])),
            }
        )
    return rows


def select_topk(score, keep_count):
    order = np.argsort(score)
    return order[:keep_count], order


def select_greedy_global_match(
    base_score,
    candidate_hists,
    candidate_dists,
    candidate_img,
    reference_global,
    reference_img_mean,
    keep_count,
    pool_size=800,
    base_weight=0.2,
    class_global_weight=1.0,
    image_global_weight=1.0,
):
    base_order = np.argsort(base_score)
    pool = base_order[: min(len(base_order), int(pool_size))]
    selected = []
    remaining = pool.copy()
    hist_sum = np.zeros(candidate_hists.shape[1], dtype=np.float64)
    img_sum = np.zeros(candidate_img.shape[1], dtype=np.float64)

    full_global = normalize(candidate_hists.sum(axis=0))
    full_class_gap = float(js_divergence_vector(full_global[None, :], reference_global[None, :])[0])
    full_class_gap = max(full_class_gap, 1e-12)
    full_image_gap = float(np.linalg.norm(candidate_img.mean(axis=0) - reference_img_mean))
    full_image_gap = max(full_image_gap, 1e-12)
    base_unit = robust_unit(base_score)

    for _ in range(min(keep_count, len(remaining))):
        trial_hists = hist_sum[None, :] + candidate_hists[remaining]
        trial_dists = (trial_hists + 1e-12) / (trial_hists.sum(axis=1, keepdims=True) + 1e-12 * trial_hists.shape[1])
        class_gap = js_divergence_vector(trial_dists, np.broadcast_to(reference_global, trial_dists.shape))

        trial_img_mean = (img_sum[None, :] + candidate_img[remaining]) / (len(selected) + 1)
        image_gap = np.sqrt(np.sum((trial_img_mean - reference_img_mean[None, :]) ** 2, axis=1))

        objective = (
            base_weight * base_unit[remaining]
            + class_global_weight * (class_gap / full_class_gap)
            + image_global_weight * (image_gap / full_image_gap)
        )
        best_pos = int(np.argmin(objective))
        best_index = int(remaining[best_pos])
        selected.append(best_index)
        hist_sum += candidate_hists[best_index]
        img_sum += candidate_img[best_index]
        remaining = np.delete(remaining, best_pos)

    selected = np.asarray(selected, dtype=np.int64)
    final_order = np.concatenate([selected, np.asarray([idx for idx in base_order if idx not in set(selected)], dtype=np.int64)])
    return selected, final_order


def mean_feature_dict(names, feature):
    return {name: float(value) for name, value in zip(names, feature)}


def make_manifest(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_records = load_cosec_train_records(args.reference_domain)
    nightcity_records = load_nightcity_dicts(args.nightcity_split, limit=args.nightcity_limit)

    ref_hists, ref_dists = compute_hists(reference_records)
    nc_hists, nc_dists = compute_hists(nightcity_records)
    ref_global = normalize(ref_hists.sum(axis=0))
    nc_global = normalize(nc_hists.sum(axis=0))

    class_js_matrix = js_divergence_matrix(nc_dists, ref_dists)
    nearest_class_js = class_js_matrix.min(axis=1)
    nearest_class_index = class_js_matrix.argmin(axis=1)
    global_class_js = js_divergence_vector(nc_dists, np.broadcast_to(ref_global, nc_dists.shape))
    rare_mask = ref_global < args.rare_threshold
    rare_mass = nc_dists[:, rare_mask].sum(axis=1)

    ref_img = compute_image_features(reference_records, resize=args.resize)
    nc_img = compute_image_features(nightcity_records, resize=args.resize)
    all_img = np.concatenate([ref_img, nc_img], axis=0)
    img_center = all_img.mean(axis=0)
    img_scale = all_img.std(axis=0) + 1e-6
    ref_img_z = (ref_img - img_center) / img_scale
    nc_img_z = (nc_img - img_center) / img_scale
    nearest_image_l2, nearest_image_index = chunked_min_l2(nc_img_z, ref_img_z, chunk_size=args.chunk_size)
    global_image_l2 = np.sqrt(np.sum((nc_img_z - ref_img_z.mean(axis=0)[None, :]) ** 2, axis=1))

    score = (
        args.class_weight * robust_unit(nearest_class_js)
        + args.global_class_weight * robust_unit(global_class_js)
        + args.image_weight * robust_unit(nearest_image_l2)
        + args.global_image_weight * robust_unit(global_image_l2)
        + args.rare_weight * robust_unit(rare_mass)
    )

    keep_count = args.keep_count
    if keep_count is None:
        keep_count = int(round(len(score) * args.keep_ratio))
    keep_count = max(1, min(len(score), int(keep_count)))
    if args.selection_mode == "top":
        selected_order, order = select_topk(score, keep_count)
    elif args.selection_mode == "greedy":
        selected_order, order = select_greedy_global_match(
            score,
            nc_hists,
            nc_dists,
            nc_img,
            ref_global,
            ref_img.mean(axis=0),
            keep_count,
            pool_size=args.greedy_pool_size,
            base_weight=args.greedy_base_weight,
            class_global_weight=args.greedy_class_global_weight,
            image_global_weight=args.greedy_image_global_weight,
        )
    else:
        raise ValueError(f"Unknown selection mode: {args.selection_mode}")
    keep_indices = set(selected_order.tolist())

    entries = []
    for index, record in enumerate(nightcity_records):
        ref_class_record = reference_records[int(nearest_class_index[index])]
        ref_image_record = reference_records[int(nearest_image_index[index])]
        entries.append(
            {
                "rank": int(np.where(order == index)[0][0] + 1),
                "keep": index in keep_indices,
                "selection_mode": args.selection_mode,
                "score": float(score[index]),
                "nearest_class_js": float(nearest_class_js[index]),
                "global_class_js": float(global_class_js[index]),
                "rare_mass": float(rare_mass[index]),
                "nearest_image_l2": float(nearest_image_l2[index]),
                "global_image_l2": float(global_image_l2[index]),
                "image_id": record["image_id"],
                "file_name": record["file_name"],
                "sem_seg_file_name": record["sem_seg_file_name"],
                "nightcity_split": record.get("nightcity_split", args.nightcity_split),
                "nearest_class_cosec_image_id": ref_class_record["image_id"],
                "nearest_image_cosec_image_id": ref_image_record["image_id"],
                "top_classes": top_classes(nc_dists[index], args.topk),
                "image_feature": mean_feature_dict(FEATURE_NAMES, nc_img[index]),
            }
        )

    entries.sort(key=lambda item: item["rank"])
    kept_entries = [entry for entry in entries if entry["keep"]]
    kept_indices_sorted = [nightcity_records.index(next(r for r in nightcity_records if r["image_id"] == entry["image_id"])) for entry in kept_entries]
    kept_hists = nc_hists[kept_indices_sorted]
    kept_img = nc_img[kept_indices_sorted]
    kept_global = normalize(kept_hists.sum(axis=0))

    manifest = {
        "method": "nightcity_cosec_night_domain_patch_filter",
        "classes": list(CLASSES),
        "reference_domain": args.reference_domain,
        "reference_records": len(reference_records),
        "nightcity_split": args.nightcity_split,
        "nightcity_records": len(nightcity_records),
        "keep_count": keep_count,
        "reject_count": len(nightcity_records) - keep_count,
        "selection_mode": args.selection_mode,
        "score": {
            "formula": (
                "class_weight*nearest_class_js_p90 + global_class_weight*global_class_js_p90 "
                "+ image_weight*nearest_image_l2_p90 + global_image_weight*global_image_l2_p90 "
                "+ rare_weight*rare_mass_p90"
            ),
            "class_weight": args.class_weight,
            "global_class_weight": args.global_class_weight,
            "image_weight": args.image_weight,
            "global_image_weight": args.global_image_weight,
            "rare_weight": args.rare_weight,
            "rare_threshold": args.rare_threshold,
            "image_feature": FEATURE_NAMES,
            "greedy_pool_size": args.greedy_pool_size,
            "greedy_base_weight": args.greedy_base_weight,
            "greedy_class_global_weight": args.greedy_class_global_weight,
            "greedy_image_global_weight": args.greedy_image_global_weight,
        },
        "reference_global_distribution": {CLASSES[i]: float(ref_global[i]) for i in range(NUM_CLASSES)},
        "nightcity_global_distribution": {CLASSES[i]: float(nc_global[i]) for i in range(NUM_CLASSES)},
        "kept_nightcity_global_distribution": {CLASSES[i]: float(kept_global[i]) for i in range(NUM_CLASSES)},
        "class_gap_table": class_gap_table(CLASSES, ref_global, nc_global, kept_global),
        "image_statistics": {
            "feature": FEATURE_NAMES,
            "reference_mean": mean_feature_dict(FEATURE_NAMES, ref_img.mean(axis=0)),
            "nightcity_mean": mean_feature_dict(FEATURE_NAMES, nc_img.mean(axis=0)),
            "kept_mean": mean_feature_dict(FEATURE_NAMES, kept_img.mean(axis=0)),
            "l2_mean_nightcity_vs_reference": float(np.linalg.norm(nc_img.mean(axis=0) - ref_img.mean(axis=0))),
            "l2_mean_kept_vs_reference": float(np.linalg.norm(kept_img.mean(axis=0) - ref_img.mean(axis=0))),
        },
        "kept": kept_entries,
        "rejected": [entry for entry in entries if not entry["keep"]],
    }

    output_path = output_dir / args.output_name
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    summary_path = output_path.with_suffix("").with_name(output_path.stem + "_summary.md")
    write_summary(summary_path, manifest)

    if args.latest_name:
        latest_path = output_dir / args.latest_name
        with latest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
            f.write("\n")
    else:
        latest_path = None

    print(f"wrote manifest: {output_path}")
    print(f"wrote summary: {summary_path}")
    if latest_path:
        print(f"wrote latest copy: {latest_path}")
    print(f"kept {keep_count}/{len(nightcity_records)} NightCity {args.nightcity_split} images")
    print("class global JS:")
    print(f"  all NightCity vs reference: {float(js_divergence_vector(nc_global[None, :], ref_global[None, :])[0]):.6f}")
    print(f"  kept vs reference:          {float(js_divergence_vector(kept_global[None, :], ref_global[None, :])[0]):.6f}")
    print("image-stat L2 gap:")
    print(f"  all NightCity vs reference: {manifest['image_statistics']['l2_mean_nightcity_vs_reference']:.6f}")
    print(f"  kept vs reference:          {manifest['image_statistics']['l2_mean_kept_vs_reference']:.6f}")
    return output_path


FEATURE_NAMES = (
    "rgb_mean_r",
    "rgb_mean_g",
    "rgb_mean_b",
    "rgb_std_r",
    "rgb_std_g",
    "rgb_std_b",
    "lum_mean",
    "lum_std",
    "lum_p10",
    "lum_p50",
    "lum_p90",
    "dark_ratio_015",
    "dark_ratio_025",
    "bright_ratio_065",
    "saturation_mean",
    "saturation_std",
)


def write_summary(path, manifest):
    class_rows = sorted(manifest["class_gap_table"], key=lambda item: item["gap_kept"], reverse=True)[:10]
    kept_preview = manifest["kept"][:20]
    img = manifest["image_statistics"]
    lines = [
        "# NightCity Domain-Patch Filter",
        "",
        f"- Reference: CoSEC train `{manifest['reference_domain']}`",
        f"- Reference records: {manifest['reference_records']}",
        f"- NightCity split: {manifest['nightcity_split']}",
        f"- NightCity records: {manifest['nightcity_records']}",
        f"- Kept records: {manifest['keep_count']}",
        f"- Selection mode: `{manifest.get('selection_mode', 'top')}`",
        f"- Rejected records: {manifest['reject_count']}",
        "",
        "## Gap Summary",
        "",
        f"- Image-stat L2 full NightCity -> reference: {img['l2_mean_nightcity_vs_reference']:.6f}",
        f"- Image-stat L2 kept -> reference: {img['l2_mean_kept_vs_reference']:.6f}",
        "",
        "## Image Statistics",
        "",
        "| Feature | CoSEC reference | NightCity all | Kept |",
        "| --- | ---: | ---: | ---: |",
    ]
    for feature in FEATURE_NAMES:
        lines.append(
            f"| {feature} | {img['reference_mean'][feature]:.6f} | "
            f"{img['nightcity_mean'][feature]:.6f} | {img['kept_mean'][feature]:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Largest Remaining Class Gaps",
            "",
            "| Class | CoSEC reference | NightCity all | Kept | Kept abs gap |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in class_rows:
        lines.append(
            f"| {row['class']} | {row['reference']:.4f} | {row['candidate_all']:.4f} | "
            f"{row['kept']:.4f} | {row['gap_kept']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Top Selected Images",
            "",
            "| Rank | Split | Image ID | Score | Class JS | Image L2 | Top classes |",
            "| ---: | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for item in kept_preview:
        top = ", ".join(f"{entry['class']} {entry['ratio']:.2f}" for entry in item["top_classes"][:3])
        lines.append(
            f"| {item['rank']} | {item['nightcity_split']} | `{item['image_id']}` | "
            f"{item['score']:.4f} | {item['nearest_class_js']:.4f} | "
            f"{item['nearest_image_l2']:.4f} | {top} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nightcity-split", choices=["train", "val", "trainval"], default="trainval")
    parser.add_argument("--reference-domain", choices=["night", "day", "all"], default="night")
    parser.add_argument("--keep-count", type=int, default=80)
    parser.add_argument("--keep-ratio", type=float, default=0.02)
    parser.add_argument("--class-weight", type=float, default=1.0)
    parser.add_argument("--global-class-weight", type=float, default=0.25)
    parser.add_argument("--image-weight", type=float, default=1.0)
    parser.add_argument("--global-image-weight", type=float, default=0.5)
    parser.add_argument("--rare-weight", type=float, default=0.5)
    parser.add_argument("--rare-threshold", type=float, default=0.001)
    parser.add_argument("--resize", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--selection-mode", choices=["top", "greedy"], default="top")
    parser.add_argument("--greedy-pool-size", type=int, default=1000)
    parser.add_argument("--greedy-base-weight", type=float, default=0.2)
    parser.add_argument("--greedy-class-global-weight", type=float, default=1.0)
    parser.add_argument("--greedy-image-global-weight", type=float, default=1.0)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--nightcity-limit", type=int, default=None)
    parser.add_argument("--output-dir", default=str(ROOT / "work_dirs" / "manifests"))
    parser.add_argument(
        "--output-name",
        default="nightcity_trainval_cosec_night_domain_patch_top80.json",
    )
    parser.add_argument(
        "--latest-name",
        default="nightcity_trainval_cosec_night_domain_patch_filtered.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    make_manifest(parse_args())
