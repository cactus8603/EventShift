#!/usr/bin/env python
"""Compare semantic class distributions for registered training mixes."""

import argparse
import json
import sys
import importlib.util
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
sys.path.insert(0, str(ROOT / "third_party" / "Mask2Former"))
if importlib.util.find_spec("detectron2") is None:
    sys.path.insert(0, str(ROOT / "third_party" / "detectron2"))

from cosec_finetune_splits import CLASSES  # noqa: E402
from filter_nightcity_by_cosec_distribution import js_divergence_vector, normalize  # noqa: E402
from train_mask2former_cosec import register_cosec  # noqa: E402

from detectron2.data import DatasetCatalog  # noqa: E402


NUM_CLASSES = len(CLASSES)


def label_hist(path):
    label = np.asarray(Image.open(path))
    if label.ndim == 3:
        label = label[:, :, 0]
    valid = (label >= 0) & (label < NUM_CLASSES)
    return np.bincount(label[valid].astype(np.int64), minlength=NUM_CLASSES).astype(np.float64)


def distribution(records):
    hist = np.zeros(NUM_CLASSES, dtype=np.float64)
    valid_records = 0
    for record in records:
        label_path = record.get("sem_seg_file_name")
        if not label_path:
            continue
        hist += label_hist(label_path)
        valid_records += 1
    return normalize(hist), valid_records


def js(a, b):
    return float(js_divergence_vector(a[None, :], b[None, :])[0])


def top_classes(dist, topk=8):
    order = np.argsort(-dist)[:topk]
    return ", ".join(f"{CLASSES[index]} {dist[index]:.3f}" for index in order if dist[index] > 0)


def summarize_dataset(name, references):
    records = DatasetCatalog.get(name)
    dist, valid_records = distribution(records)
    return {
        "name": name,
        "records": len(records),
        "valid_label_records": valid_records,
        "top_classes": top_classes(dist),
        "js": {ref_name: js(dist, ref_dist) for ref_name, ref_dist in references.items()},
        "distribution": {CLASSES[index]: float(dist[index]) for index in range(NUM_CLASSES)},
    }


def write_markdown(path, summaries, references):
    lines = [
        "# Dataset Mix Distribution Diagnostic",
        "",
        "## References",
        "",
        "| Reference | Top classes |",
        "|---|---|",
    ]
    for name, dist in references.items():
        lines.append(f"| {name} | {top_classes(dist)} |")

    lines.extend(
        [
            "",
            "## Datasets",
            "",
            "| Dataset | Records | Valid label records | JS vs day train | JS vs night train | JS vs day val | JS vs night val | Top classes |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for summary in summaries:
        values = summary["js"]
        lines.append(
            f"| {summary['name']} | {summary['records']} | {summary['valid_label_records']} | "
            f"{values.get('cosec_day_train', float('nan')):.6f} | "
            f"{values.get('cosec_night_train', float('nan')):.6f} | "
            f"{values.get('cosec_day_val', float('nan')):.6f} | "
            f"{values.get('cosec_night_val', float('nan')):.6f} | "
            f"{summary['top_classes']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--output", default=str(ROOT / "work_dirs" / "diagnostics" / "dataset_mix_distribution.json"))
    args = parser.parse_args()

    register_cosec()
    reference_names = ["cosec_day_train", "cosec_night_train", "cosec_day_val", "cosec_night_val"]
    references = {}
    for name in reference_names:
        references[name] = distribution(DatasetCatalog.get(name))[0]

    summaries = [summarize_dataset(name, references) for name in args.datasets]
    output_json = Path(args.output)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump({"references": reference_names, "datasets": summaries}, handle, indent=2)
        handle.write("\n")
    output_md = output_json.with_suffix(".md")
    write_markdown(output_md, summaries, references)
    print(f"wrote: {output_json}")
    print(f"wrote: {output_md}")


if __name__ == "__main__":
    main()
