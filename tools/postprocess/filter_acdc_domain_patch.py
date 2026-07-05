#!/usr/bin/env python
"""Select an ACDC subset as a CoSEC-night domain patch."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

def _eventshift_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "configs").is_dir() and (parent / "third_party").is_dir():
            return parent
    return Path(__file__).resolve().parents[1]


ROOT = _eventshift_root()
sys.path.insert(0, str(ROOT / "tools"))

from cosec_finetune_splits import CLASSES  # noqa: E402
from filter_nightcity_by_cosec_distribution import (  # noqa: E402
    compute_hists,
    js_divergence_matrix,
    js_divergence_vector,
    normalize,
    top_classes,
)
from filter_nightcity_domain_patch import (  # noqa: E402
    FEATURE_NAMES,
    chunked_min_l2,
    class_gap_table,
    compute_image_features,
    load_cosec_train_records,
    mean_feature_dict,
    robust_unit,
    select_greedy_global_match,
    select_topk,
)


NUM_CLASSES = len(CLASSES)
DEFAULT_ACDC_ROOTS = (
    Path(os.environ.get("ACDC_ROOT", "")),
    Path("/work/u1621738/ebmv_eccv/MambaSeg/data/acdc"),
    ROOT / "data" / "acdc",
)


def _valid_acdc_root(path):
    return path and (path / "rgb_anon").is_dir() and (path / "gt").is_dir()


def acdc_root():
    for path in DEFAULT_ACDC_ROOTS:
        if _valid_acdc_root(path):
            return path
    candidates = ", ".join(str(path) for path in DEFAULT_ACDC_ROOTS if str(path))
    raise FileNotFoundError(f"ACDC root not found. Checked: {candidates}")


def _parse_csv(value, choices):
    if value == "all":
        return tuple(choices)
    parts = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = sorted(set(parts) - set(choices))
    if unknown:
        raise ValueError(f"Unknown value(s): {unknown}; choices={choices} or all")
    return parts


def load_acdc_records(conditions=("night",), splits=("train", "val"), limit=None):
    root = acdc_root()
    records = []
    for condition in conditions:
        for split in splits:
            image_root = root / "rgb_anon" / condition / split
            label_root = root / "gt" / condition / split
            if not image_root.is_dir() or not label_root.is_dir():
                continue
            for image_path in sorted(image_root.glob("*/*_rgb_anon.png")):
                sequence = image_path.parent.name
                frame_stem = image_path.name.replace("_rgb_anon.png", "")
                label_path = label_root / sequence / f"{frame_stem}_gt_labelTrainIds.png"
                if not label_path.exists():
                    continue
                records.append(
                    {
                        "file_name": str(image_path),
                        "sem_seg_file_name": str(label_path),
                        "image_id": f"acdc_{condition}_{split}_{frame_stem}",
                        "acdc_condition": condition,
                        "acdc_split": split,
                        "acdc_sequence": sequence,
                    }
                )
                if limit is not None and len(records) >= int(limit):
                    return records
    return records


def make_manifest(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    conditions = _parse_csv(args.conditions, ("fog", "night", "rain", "snow"))
    splits = _parse_csv(args.splits, ("train", "val"))
    reference_records = load_cosec_train_records(args.reference_domain)
    acdc_records = load_acdc_records(conditions, splits, limit=args.acdc_limit)
    if not acdc_records:
        raise ValueError("No ACDC records found for the requested conditions/splits.")

    ref_hists, ref_dists = compute_hists(reference_records)
    acdc_hists, acdc_dists = compute_hists(acdc_records)
    ref_global = normalize(ref_hists.sum(axis=0))
    acdc_global = normalize(acdc_hists.sum(axis=0))

    class_js_matrix = js_divergence_matrix(acdc_dists, ref_dists)
    nearest_class_js = class_js_matrix.min(axis=1)
    nearest_class_index = class_js_matrix.argmin(axis=1)
    global_class_js = js_divergence_vector(acdc_dists, np.broadcast_to(ref_global, acdc_dists.shape))
    rare_mask = ref_global < args.rare_threshold
    rare_mass = acdc_dists[:, rare_mask].sum(axis=1)

    ref_img = compute_image_features(reference_records, resize=args.resize)
    acdc_img = compute_image_features(acdc_records, resize=args.resize)
    all_img = np.concatenate([ref_img, acdc_img], axis=0)
    img_center = all_img.mean(axis=0)
    img_scale = all_img.std(axis=0) + 1e-6
    ref_img_z = (ref_img - img_center) / img_scale
    acdc_img_z = (acdc_img - img_center) / img_scale
    nearest_image_l2, nearest_image_index = chunked_min_l2(acdc_img_z, ref_img_z, chunk_size=args.chunk_size)
    global_image_l2 = np.sqrt(np.sum((acdc_img_z - ref_img_z.mean(axis=0)[None, :]) ** 2, axis=1))

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
            acdc_hists,
            acdc_dists,
            acdc_img,
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
    for index, record in enumerate(acdc_records):
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
                "acdc_condition": record["acdc_condition"],
                "acdc_split": record["acdc_split"],
                "acdc_sequence": record["acdc_sequence"],
                "nearest_class_cosec_image_id": ref_class_record["image_id"],
                "nearest_image_cosec_image_id": ref_image_record["image_id"],
                "top_classes": top_classes(acdc_dists[index], args.topk),
                "image_feature": mean_feature_dict(FEATURE_NAMES, acdc_img[index]),
            }
        )

    entries.sort(key=lambda item: item["rank"])
    kept_entries = [entry for entry in entries if entry["keep"]]
    kept_indices_sorted = [
        acdc_records.index(next(record for record in acdc_records if record["image_id"] == entry["image_id"]))
        for entry in kept_entries
    ]
    kept_hists = acdc_hists[kept_indices_sorted]
    kept_img = acdc_img[kept_indices_sorted]
    kept_global = normalize(kept_hists.sum(axis=0))

    full_class_js = float(js_divergence_vector(acdc_global[None, :], ref_global[None, :])[0])
    kept_class_js = float(js_divergence_vector(kept_global[None, :], ref_global[None, :])[0])

    manifest = {
        "method": "acdc_cosec_night_domain_patch_filter",
        "classes": list(CLASSES),
        "reference_domain": args.reference_domain,
        "reference_records": len(reference_records),
        "acdc_root": str(acdc_root()),
        "acdc_conditions": list(conditions),
        "acdc_splits": list(splits),
        "acdc_records": len(acdc_records),
        "keep_count": keep_count,
        "reject_count": len(acdc_records) - keep_count,
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
        "acdc_global_distribution": {CLASSES[i]: float(acdc_global[i]) for i in range(NUM_CLASSES)},
        "kept_acdc_global_distribution": {CLASSES[i]: float(kept_global[i]) for i in range(NUM_CLASSES)},
        "class_gap_table": class_gap_table(CLASSES, ref_global, acdc_global, kept_global),
        "class_global_js_acdc_vs_reference": full_class_js,
        "class_global_js_kept_vs_reference": kept_class_js,
        "image_statistics": {
            "feature": FEATURE_NAMES,
            "reference_mean": mean_feature_dict(FEATURE_NAMES, ref_img.mean(axis=0)),
            "acdc_mean": mean_feature_dict(FEATURE_NAMES, acdc_img.mean(axis=0)),
            "kept_mean": mean_feature_dict(FEATURE_NAMES, kept_img.mean(axis=0)),
            "l2_mean_acdc_vs_reference": float(np.linalg.norm(acdc_img.mean(axis=0) - ref_img.mean(axis=0))),
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
    print(f"kept {keep_count}/{len(acdc_records)} ACDC images")
    print("class global JS:")
    print(f"  all ACDC vs reference: {full_class_js:.6f}")
    print(f"  kept vs reference:     {kept_class_js:.6f}")
    print("image-stat L2 gap:")
    print(f"  all ACDC vs reference: {manifest['image_statistics']['l2_mean_acdc_vs_reference']:.6f}")
    print(f"  kept vs reference:     {manifest['image_statistics']['l2_mean_kept_vs_reference']:.6f}")
    return output_path


def write_summary(path, manifest):
    class_rows = sorted(manifest["class_gap_table"], key=lambda item: item["gap_kept"], reverse=True)[:10]
    kept_preview = manifest["kept"][:20]
    img = manifest["image_statistics"]
    lines = [
        "# ACDC Domain-Patch Filter",
        "",
        f"- Reference: CoSEC train `{manifest['reference_domain']}`",
        f"- Reference records: {manifest['reference_records']}",
        f"- ACDC conditions: {', '.join(manifest['acdc_conditions'])}",
        f"- ACDC splits: {', '.join(manifest['acdc_splits'])}",
        f"- ACDC records: {manifest['acdc_records']}",
        f"- Kept records: {manifest['keep_count']}",
        f"- Selection mode: `{manifest.get('selection_mode', 'top')}`",
        f"- Rejected records: {manifest['reject_count']}",
        "",
        "## Gap Summary",
        "",
        f"- Class JS full ACDC -> reference: {manifest['class_global_js_acdc_vs_reference']:.6f}",
        f"- Class JS kept -> reference: {manifest['class_global_js_kept_vs_reference']:.6f}",
        f"- Image-stat L2 full ACDC -> reference: {img['l2_mean_acdc_vs_reference']:.6f}",
        f"- Image-stat L2 kept -> reference: {img['l2_mean_kept_vs_reference']:.6f}",
        "",
        "## Image Statistics",
        "",
        "| Feature | CoSEC reference | ACDC all | Kept |",
        "| --- | ---: | ---: | ---: |",
    ]
    for feature in FEATURE_NAMES:
        lines.append(
            f"| {feature} | {img['reference_mean'][feature]:.6f} | "
            f"{img['acdc_mean'][feature]:.6f} | {img['kept_mean'][feature]:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Largest Remaining Class Gaps",
            "",
            "| Class | CoSEC reference | ACDC all | Kept | Kept abs gap |",
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
            "| Rank | Condition | Split | Image ID | Score | Class JS | Image L2 | Top classes |",
            "| ---: | --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for item in kept_preview:
        top = ", ".join(f"{entry['class']} {entry['ratio']:.2f}" for entry in item["top_classes"][:3])
        lines.append(
            f"| {item['rank']} | {item['acdc_condition']} | {item['acdc_split']} | "
            f"`{item['image_id']}` | {item['score']:.4f} | "
            f"{item['nearest_class_js']:.4f} | {item['nearest_image_l2']:.4f} | {top} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", default="night", help="Comma-separated ACDC conditions or all.")
    parser.add_argument("--splits", default="train,val", help="Comma-separated ACDC splits or all.")
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
    parser.add_argument("--selection-mode", choices=["top", "greedy"], default="greedy")
    parser.add_argument("--greedy-pool-size", type=int, default=400)
    parser.add_argument("--greedy-base-weight", type=float, default=0.2)
    parser.add_argument("--greedy-class-global-weight", type=float, default=1.0)
    parser.add_argument("--greedy-image-global-weight", type=float, default=1.0)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--acdc-limit", type=int, default=None)
    parser.add_argument("--output-dir", default=str(ROOT / "work_dirs" / "manifests"))
    parser.add_argument(
        "--output-name",
        default="acdc_night_trainval_cosec_night_domain_patch_top80_greedy.json",
    )
    parser.add_argument(
        "--latest-name",
        default="acdc_night_trainval_cosec_night_domain_patch_filtered_greedy.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    make_manifest(parse_args())
