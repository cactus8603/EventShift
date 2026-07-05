#!/usr/bin/env python
"""Print record counts and valid-pixel coverage for pseudo datasets."""

import argparse
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
from pseudo_dataset import (  # noqa: E402
    load_cosec_test_prediction_pseudo_dicts,
    load_cosec_test_pseudo_dicts,
    load_real_pool_pseudo_dicts,
)


def read_label(path):
    array = np.asarray(Image.open(path))
    if array.ndim == 3:
        array = array[:, :, 0]
    return array


def summarize(name, records):
    records = list(records)
    fractions = [float(record.get("pseudo_valid_fraction", 0.0)) for record in records]
    if not fractions:
        print(f"{name}: count=0")
        return
    hist = np.zeros(len(CLASSES), dtype=np.int64)
    for record in records:
        label = read_label(record["sem_seg_file_name"])
        valid = (label >= 0) & (label < len(CLASSES))
        hist += np.bincount(label[valid].reshape(-1), minlength=len(CLASSES))[: len(CLASSES)]
    total = int(hist.sum())
    top_classes = []
    if total:
        order = np.argsort(-hist)[:8]
        top_classes = [f"{CLASSES[idx]}={hist[idx] / total:.3f}" for idx in order if hist[idx] > 0]
    print(
        f"{name}: count={len(records)} "
        f"valid_mean={sum(fractions) / len(fractions):.4f} "
        f"valid_min={min(fractions):.4f} "
        f"valid_max={max(fractions):.4f} "
        f"top_classes={', '.join(top_classes)}",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=int, default=192)
    parser.add_argument("--real-pool-threshold", type=int, default=224)
    args = parser.parse_args()

    summarize(
        f"cosec_test_daynight_pseudo_consensus_conf{args.threshold}",
        load_cosec_test_pseudo_dicts("daynight", "consensus", args.threshold, repeat=1),
    )
    summarize(
        f"cosec_test_night_pseudo_consensus_conf{args.threshold}_repeat2",
        load_cosec_test_pseudo_dicts("night", "consensus", args.threshold, repeat=2),
    )
    summarize(
        f"cosec_test_real_pseudo_consensus_conf{args.threshold}",
        load_cosec_test_pseudo_dicts("real", "consensus", args.threshold, repeat=1),
    )
    summarize(
        f"cosec_test_daynight_pseudo_segformer_consensus_conf{args.threshold}_limit256",
        load_cosec_test_pseudo_dicts("daynight", "segformer_consensus", args.threshold, repeat=1, limit=256),
    )
    summarize(
        f"cosec_test_night_pseudo_segformer_consensus_conf{args.threshold}_limit128",
        load_cosec_test_pseudo_dicts("night", "segformer_consensus", args.threshold, repeat=1, limit=128),
    )
    summarize(
        f"cosec_test_real_pseudo_segformer_consensus_conf{args.threshold}_limit73",
        load_cosec_test_pseudo_dicts("real", "segformer_consensus", args.threshold, repeat=1, limit=73),
    )
    summarize(
        f"cosec_test_daynight_pseudo_segformer_balcap_conf{args.threshold}_limit256",
        load_cosec_test_pseudo_dicts("daynight", "segformer_balcap", args.threshold, repeat=1, limit=256),
    )
    summarize(
        f"cosec_test_night_pseudo_segformer_balcap_conf{args.threshold}_limit128",
        load_cosec_test_pseudo_dicts("night", "segformer_balcap", args.threshold, repeat=1, limit=128),
    )
    summarize(
        f"cosec_test_real_pseudo_segformer_balcap_conf{args.threshold}_limit73",
        load_cosec_test_pseudo_dicts("real", "segformer_balcap", args.threshold, repeat=1, limit=73),
    )
    summarize(
        f"real_pool_pseudo_swinl_conf{args.real_pool_threshold}",
        load_real_pool_pseudo_dicts("swinl", args.real_pool_threshold, repeat=1, limit=600),
    )
    summarize(
        "cosec_test_daynight_pseudo_currentbest_tta_all",
        load_cosec_test_prediction_pseudo_dicts(
            "daynight",
            "swinL_day65_4352_tta5126247681024_daynight_acdc_proxy_real",
            "all",
            repeat=1,
            min_valid_fraction=0.99,
        ),
    )
    summarize(
        "cosec_test_daynight_pseudo_currentbest_tta_segformer_agree_conf192_limit384",
        load_cosec_test_prediction_pseudo_dicts(
            "daynight",
            "swinL_day65_4352_tta5126247681024_daynight_acdc_proxy_real",
            "segformer_agree_conf192",
            repeat=1,
            min_valid_fraction=0.01,
            limit=384,
        ),
    )
    summarize(
        "cosec_test_daynight_pseudo_currentbest_tta_segformer_agree_rare_boundary_conf192_limit384",
        load_cosec_test_prediction_pseudo_dicts(
            "daynight",
            "swinL_day65_4352_tta5126247681024_daynight_acdc_proxy_real",
            "segformer_agree_rare_boundary_conf192",
            repeat=1,
            min_valid_fraction=0.01,
            limit=384,
        ),
    )
    summarize(
        "cosec_test_daynight_pseudo_currentbest_tta_segformer_agree_gap_focus_conf192_limit384",
        load_cosec_test_prediction_pseudo_dicts(
            "daynight",
            "swinL_day65_4352_tta5126247681024_daynight_acdc_proxy_real",
            "segformer_agree_gap_focus_conf192",
            repeat=1,
            min_valid_fraction=0.01,
            limit=384,
        ),
    )
    summarize(
        "cosec_test_day_pseudo_currentbest_tta_segformer_agree_gap_focus_conf192_limit256",
        load_cosec_test_prediction_pseudo_dicts(
            "day",
            "swinL_day65_4352_tta5126247681024_daynight_acdc_proxy_real",
            "segformer_agree_gap_focus_conf192",
            repeat=1,
            min_valid_fraction=0.01,
            limit=256,
        ),
    )
    summarize(
        "cosec_test_night_pseudo_currentbest_tta_segformer_agree_gap_focus_conf192_limit192",
        load_cosec_test_prediction_pseudo_dicts(
            "night",
            "swinL_day65_4352_tta5126247681024_daynight_acdc_proxy_real",
            "segformer_agree_gap_focus_conf192",
            repeat=1,
            min_valid_fraction=0.001,
            limit=192,
        ),
    )


if __name__ == "__main__":
    main()
