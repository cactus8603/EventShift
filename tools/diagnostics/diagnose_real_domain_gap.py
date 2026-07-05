#!/usr/bin/env python
"""Diagnose REAL-domain proximity to CoSEC, ACDC, and NightCity."""

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

from cosec_finetune_splits import CLASSES  # noqa: E402
from filter_nightcity_by_cosec_distribution import (  # noqa: E402
    js_divergence_vector,
    label_hist,
    normalize,
)
from filter_nightcity_domain_patch import (  # noqa: E402
    FEATURE_NAMES,
    compute_image_features,
    load_cosec_train_records,
    mean_feature_dict,
)


NUM_CLASSES = len(CLASSES)
IGNORE_LABEL = 255


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def real_test_records(test_root):
    records = []
    for seq_dir in sorted(Path(test_root).glob("REAL_*")):
        image_dir = seq_dir / "img_co_left"
        if not image_dir.is_dir():
            continue
        for image_path in sorted(image_dir.glob("*.png")):
            records.append(
                {
                    "file_name": str(image_path),
                    "image_id": f"{seq_dir.name}_{image_path.stem}",
                    "sequence": seq_dir.name,
                }
            )
    return records


def real_pool_records(real_root, image_subdir="gt"):
    records = []
    for seq_dir in sorted(Path(real_root).glob("*")):
        image_dir = seq_dir / image_subdir
        if not image_dir.is_dir():
            continue
        for image_path in sorted(image_dir.glob("*.png")):
            records.append(
                {
                    "file_name": str(image_path),
                    "image_id": f"realpool_{seq_dir.name}_{image_path.stem}",
                    "sequence": seq_dir.name,
                }
            )
    return records


def evenly_spaced_records(records, limit=None):
    if limit is None or int(limit) <= 0 or int(limit) >= len(records):
        return list(records)
    keep_count = int(limit)
    selected = []
    for rank in range(keep_count):
        index = int((rank + 0.5) * len(records) / keep_count)
        selected.append(records[min(index, len(records) - 1)])
    return selected


def manifest_kept_records(manifest, label_key):
    records = []
    for item in manifest["kept"]:
        records.append(
            {
                "file_name": item["file_name"],
                "sem_seg_file_name": item.get("sem_seg_file_name"),
                "image_id": item["image_id"],
                "source": label_key,
            }
        )
    return records


def stack_features(records, resize):
    if not records:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float64)
    return compute_image_features(records, resize=resize)


def l2_mean(a, b):
    return float(np.linalg.norm(a.mean(axis=0) - b.mean(axis=0)))


def nearest_l2(query, reference):
    diff = query[:, None, :] - reference[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    return dist.min(axis=1)


def summarize_feature(records, features):
    if len(records) == 0:
        return {
            "count": 0,
            "mean": {},
            "sequence_count": 0,
        }
    return {
        "count": len(records),
        "sequence_count": len({record.get("sequence", "") for record in records}),
        "mean": mean_feature_dict(FEATURE_NAMES, features.mean(axis=0)),
    }


def pseudo_records_from_real_test(test_root, subdir):
    records = []
    for seq_dir in sorted(Path(test_root).glob("REAL_*")):
        label_dir = seq_dir / subdir
        if not label_dir.is_dir():
            continue
        for label_path in sorted(label_dir.glob("*.png")):
            records.append(
                {
                    "sem_seg_file_name": str(label_path),
                    "image_id": f"{seq_dir.name}_{label_path.stem}",
                    "sequence": seq_dir.name,
                }
            )
    return records


def pseudo_records_from_real_pool(real_root, subdir):
    records = []
    for seq_dir in sorted(Path(real_root).glob("*")):
        label_dir = seq_dir / subdir
        if not label_dir.is_dir():
            continue
        for label_path in sorted(label_dir.glob("*.png")):
            records.append(
                {
                    "sem_seg_file_name": str(label_path),
                    "image_id": f"realpool_{seq_dir.name}_{label_path.stem}",
                    "sequence": seq_dir.name,
                }
            )
    return records


def global_distribution(records):
    hist = np.zeros(NUM_CLASSES, dtype=np.float64)
    for record in records:
        hist += label_hist(record["sem_seg_file_name"])
    return normalize(hist)


def confidence_summary(records, conf_subdir):
    values = []
    high_192 = 0
    high_224 = 0
    total = 0
    for record in records:
        label_path = Path(record["sem_seg_file_name"])
        conf_path = label_path.parent.parent / conf_subdir / label_path.name
        if not conf_path.exists():
            continue
        conf = np.asarray(Image.open(conf_path))
        if conf.ndim == 3:
            conf = conf[:, :, 0]
        values.append(conf.reshape(-1))
        total += conf.size
        high_192 += int((conf >= 192).sum())
        high_224 += int((conf >= 224).sum())
    if not values:
        return {}
    values = np.concatenate(values).astype(np.float64)
    return {
        "count": int(values.size),
        "mean": float(values.mean() / 255.0),
        "p10": float(np.percentile(values, 10) / 255.0),
        "p50": float(np.percentile(values, 50) / 255.0),
        "p90": float(np.percentile(values, 90) / 255.0),
        "ratio_ge_192": float(high_192 / max(1, total)),
        "ratio_ge_224": float(high_224 / max(1, total)),
    }


def js(dist_a, dist_b):
    return float(js_divergence_vector(dist_a[None, :], dist_b[None, :])[0])


def class_dist_dict(dist):
    return {CLASSES[index]: float(dist[index]) for index in range(NUM_CLASSES)}


def top_class_text(dist, topk=8):
    order = np.argsort(-dist)[:topk]
    return ", ".join(f"{CLASSES[index]} {dist[index]:.3f}" for index in order if dist[index] > 0)


def write_summary(path, report):
    lines = [
        "# REAL Domain Gap Diagnostic",
        "",
        "## Image-Domain Gap",
        "",
        "| Dataset | Count | Seq | L2 vs CoSEC night | L2 vs REAL test | lum mean | dark<0.25 | saturation mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, info in report["image_domains"].items():
        mean = info["mean"]
        lines.append(
            f"| {name} | {info['count']} | {info['sequence_count']} | "
            f"{info.get('l2_vs_cosec_night', math.nan):.6f} | "
            f"{info.get('l2_vs_real_test', math.nan):.6f} | "
            f"{mean.get('lum_mean', math.nan):.6f} | "
            f"{mean.get('dark_ratio_025', math.nan):.6f} | "
            f"{mean.get('saturation_mean', math.nan):.6f} |"
        )

    lines.extend(
        [
            "",
            "## Pseudo-Label Class Distribution",
            "",
            "| Source | Count | JS vs CoSEC night | JS vs ACDC top50 | JS vs REAL test SwinL prior | Confidence mean | Conf >= 224 | Top classes |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for name, info in report["pseudo_domains"].items():
        conf = info.get("confidence", {})
        lines.append(
            f"| {name} | {info['count']} | "
            f"{info.get('js_vs_cosec_night', math.nan):.6f} | "
            f"{info.get('js_vs_acdc_top50', math.nan):.6f} | "
            f"{info.get('js_vs_real_test_swinl_prior', math.nan):.6f} | "
            f"{conf.get('mean', math.nan):.4f} | "
            f"{conf.get('ratio_ge_224', math.nan):.4f} | "
            f"{info['top_classes']} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `REAL_dataset/*/gt` is RGB imagery, not semantic ground truth.",
            "- REAL can be used for pseudo-label/self-training only; do not treat `gt` as supervised labels.",
            "- ACDC top50 is semantically closer to the REAL Swin-L prior than CoSEC night is, but REAL test imagery is darker than ACDC.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cosec_records = load_cosec_train_records("night")
    real_test = real_test_records(args.test_root)
    real_pool = evenly_spaced_records(
        real_pool_records(args.real_root, image_subdir=args.real_pool_image_subdir),
        args.real_pool_limit,
    )
    acdc_top50 = manifest_kept_records(load_json(args.acdc_top50_manifest), "acdc_top50")
    nightcity_top80 = manifest_kept_records(load_json(args.nightcity_top80_manifest), "nightcity_top80")

    feature_records = {
        "CoSEC train night": cosec_records,
        "REAL test": real_test,
        "REAL_dataset RGB": real_pool,
        "ACDC night top50": acdc_top50,
        "NightCity top80": nightcity_top80,
    }
    features = {name: stack_features(records, args.resize) for name, records in feature_records.items()}

    report = {
        "created_for": "REAL unlabeled training/domain-gap analysis",
        "image_domains": {},
        "pseudo_domains": {},
    }
    cosec_feat = features["CoSEC train night"]
    real_test_feat = features["REAL test"]
    for name, feats in features.items():
        info = summarize_feature(feature_records[name], feats)
        if len(feats) and len(cosec_feat):
            info["l2_vs_cosec_night"] = l2_mean(feats, cosec_feat)
        if len(feats) and len(real_test_feat):
            info["l2_vs_real_test"] = l2_mean(feats, real_test_feat)
            nearest = nearest_l2(real_test_feat, feats)
            info["real_test_nearest_l2_mean"] = float(nearest.mean())
            info["real_test_nearest_l2_p50"] = float(np.percentile(nearest, 50))
        report["image_domains"][name] = info

    pseudo_sources = {
        "REAL test prior_swinL_ft": (
            pseudo_records_from_real_test(args.test_root, "prior_swinL_ft"),
            "prior_swinL_ft_conf",
        ),
        "REAL test prior_mask2former_large": (
            pseudo_records_from_real_test(args.test_root, "prior_mask2former_large"),
            "prior_mask2former_large_ft_cc_submission_conf",
        ),
        "REAL_dataset prior_swinL_ft": (
            evenly_spaced_records(
                pseudo_records_from_real_pool(args.real_root, "prior_swinL_ft"),
                args.real_pool_limit,
            ),
            "prior_swinL_ft_conf",
        ),
        "REAL_dataset prior_segformer": (
            evenly_spaced_records(
                pseudo_records_from_real_pool(args.real_root, "prior_segformer"),
                args.real_pool_limit,
            ),
            "prior_swinL_ft_conf",
        ),
    }
    cosec_dist = global_distribution(cosec_records)
    acdc_dist = global_distribution(acdc_top50)
    real_test_swinl_dist = None
    for name, (records, conf_subdir) in pseudo_sources.items():
        if not records:
            continue
        dist = global_distribution(records)
        if name == "REAL test prior_swinL_ft":
            real_test_swinl_dist = dist
        report["pseudo_domains"][name] = {
            "count": len(records),
            "distribution": class_dist_dict(dist),
            "js_vs_cosec_night": js(dist, cosec_dist),
            "js_vs_acdc_top50": js(dist, acdc_dist),
            "top_classes": top_class_text(dist),
            "confidence": confidence_summary(records, conf_subdir),
        }
    if real_test_swinl_dist is not None:
        for info in report["pseudo_domains"].values():
            dist = np.array([info["distribution"][cls] for cls in CLASSES], dtype=np.float64)
            info["js_vs_real_test_swinl_prior"] = js(dist, real_test_swinl_dist)

    output_json = output_dir / args.output_name
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    output_md = output_json.with_suffix(".md")
    write_summary(output_md, report)
    print(f"wrote: {output_json}")
    print(f"wrote: {output_md}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-root", default=str(ROOT / "data" / "test"))
    parser.add_argument("--real-root", default=str(ROOT / "data" / "REAL_dataset"))
    parser.add_argument("--real-pool-image-subdir", default="gt")
    parser.add_argument("--real-pool-limit", type=int, default=600)
    parser.add_argument(
        "--acdc-top50-manifest",
        default=str(ROOT / "work_dirs" / "manifests" / "acdc_night_trainval_cosec_night_domain_patch_top50_greedy.json"),
    )
    parser.add_argument(
        "--nightcity-top80-manifest",
        default=str(ROOT / "work_dirs" / "manifests" / "nightcity_trainval_cosec_night_domain_patch_top80_greedy.json"),
    )
    parser.add_argument("--resize", type=int, default=128)
    parser.add_argument("--output-dir", default=str(ROOT / "work_dirs" / "diagnostics"))
    parser.add_argument("--output-name", default="real_domain_gap_diagnostic.json")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
