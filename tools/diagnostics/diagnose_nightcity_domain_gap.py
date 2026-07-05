#!/usr/bin/env python
"""Compare CoSEC, full NightCity, and filtered NightCity domain gaps."""

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
    load_cosec_train_records,
    normalize,
)
from nightcity_dataset import load_nightcity_cosec_classdist_dicts, load_nightcity_dicts  # noqa: E402


def load_records(name):
    if name == "cosec_train":
        return load_cosec_train_records()
    if name == "nightcity_train":
        return load_nightcity_dicts("train")
    if name == "nightcity_filtered":
        return load_nightcity_cosec_classdist_dicts()
    raise ValueError(f"Unknown record set: {name}")


def image_stats(records, max_samples=None):
    selected = records if max_samples is None else records[: int(max_samples)]
    values = []
    for record in selected:
        image = np.asarray(Image.open(record["file_name"]).convert("RGB"), dtype=np.float32) / 255.0
        channels = image.reshape(-1, 3)
        luminance = 0.2126 * channels[:, 0] + 0.7152 * channels[:, 1] + 0.0722 * channels[:, 2]
        values.append(
            np.concatenate(
                [
                    channels.mean(axis=0),
                    channels.std(axis=0),
                    np.array([luminance.mean(), luminance.std()], dtype=np.float32),
                ]
            )
        )
    return np.stack(values, axis=0)


def mean_distance(a, b):
    return float(np.linalg.norm(a.mean(axis=0) - b.mean(axis=0)))


def nearest_js_values(source_dists, target_dists, chunk_size=128):
    chunks = []
    for start in range(0, len(source_dists), chunk_size):
        stop = min(len(source_dists), start + chunk_size)
        chunks.append(js_divergence_matrix(source_dists[start:stop], target_dists).min(axis=1))
    return np.concatenate(chunks, axis=0)


def summarize_nearest_js(source_dists, target_dists, chunk_size=128):
    nearest = nearest_js_values(source_dists, target_dists, chunk_size=chunk_size)
    return {
        "mean": float(nearest.mean()),
        "median": float(np.median(nearest)),
        "p75": float(np.percentile(nearest, 75)),
        "p90": float(np.percentile(nearest, 90)),
    }


def main(args):
    cosec_records = load_records("cosec_train")
    nightcity_records = load_records("nightcity_train")
    filtered_records = load_records("nightcity_filtered")

    cosec_hists, cosec_dists = compute_hists(cosec_records)
    nightcity_hists, nightcity_dists = compute_hists(nightcity_records)
    filtered_hists, filtered_dists = compute_hists(filtered_records)

    cosec_global = normalize(cosec_hists.sum(axis=0))
    nightcity_global = normalize(nightcity_hists.sum(axis=0))
    filtered_global = normalize(filtered_hists.sum(axis=0))

    cosec_img = image_stats(cosec_records, args.image_stat_limit)
    nightcity_img = image_stats(nightcity_records, args.image_stat_limit)
    filtered_img = image_stats(filtered_records, args.image_stat_limit)

    result = {
        "records": {
            "cosec_train": len(cosec_records),
            "nightcity_train": len(nightcity_records),
            "nightcity_filtered": len(filtered_records),
        },
        "class_distribution": {
            "global_js_nightcity_vs_cosec": float(
                js_divergence_vector(nightcity_global[None, :], cosec_global[None, :])[0]
            ),
            "global_js_filtered_vs_cosec": float(
                js_divergence_vector(filtered_global[None, :], cosec_global[None, :])[0]
            ),
            "nearest_js_nightcity_to_cosec": summarize_nearest_js(
                nightcity_dists, cosec_dists, chunk_size=args.chunk_size
            ),
            "nearest_js_filtered_to_cosec": summarize_nearest_js(
                filtered_dists, cosec_dists, chunk_size=args.chunk_size
            ),
        },
        "image_statistics": {
            "feature": "RGB channel mean/std plus luminance mean/std",
            "samples_per_set": args.image_stat_limit or "all",
            "l2_mean_nightcity_vs_cosec": mean_distance(nightcity_img, cosec_img),
            "l2_mean_filtered_vs_cosec": mean_distance(filtered_img, cosec_img),
            "mean_vectors": {
                "cosec_train": cosec_img.mean(axis=0).tolist(),
                "nightcity_train": nightcity_img.mean(axis=0).tolist(),
                "nightcity_filtered": filtered_img.mean(axis=0).tolist(),
            },
        },
        "class_distribution_table": [
            {
                "class": class_name,
                "cosec_train": float(cosec_global[index]),
                "nightcity_train": float(nightcity_global[index]),
                "nightcity_filtered": float(filtered_global[index]),
                "abs_gap_full": float(abs(nightcity_global[index] - cosec_global[index])),
                "abs_gap_filtered": float(abs(filtered_global[index] - cosec_global[index])),
            }
            for index, class_name in enumerate(CLASSES)
        ],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        f.write("\n")

    print(f"wrote {output}")
    print("records:", result["records"])
    print("class global JS:")
    print(f"  full NightCity vs CoSEC: {result['class_distribution']['global_js_nightcity_vs_cosec']:.6f}")
    print(f"  filtered vs CoSEC:      {result['class_distribution']['global_js_filtered_vs_cosec']:.6f}")
    print("nearest-image class JS mean:")
    print(f"  full NightCity -> CoSEC: {result['class_distribution']['nearest_js_nightcity_to_cosec']['mean']:.6f}")
    print(f"  filtered -> CoSEC:       {result['class_distribution']['nearest_js_filtered_to_cosec']['mean']:.6f}")
    print("RGB-stat L2 gap:")
    print(f"  full NightCity vs CoSEC: {result['image_statistics']['l2_mean_nightcity_vs_cosec']:.6f}")
    print(f"  filtered vs CoSEC:      {result['image_statistics']['l2_mean_filtered_vs_cosec']:.6f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(ROOT / "work_dirs" / "diagnostics" / "nightcity_domain_gap_classdist25.json"),
    )
    parser.add_argument(
        "--image-stat-limit",
        type=int,
        default=None,
        help="Optional per-set cap for image-stat diagnostics. Class statistics always use all records.",
    )
    parser.add_argument("--chunk-size", type=int, default=128)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
