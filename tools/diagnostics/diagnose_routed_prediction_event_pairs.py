#!/usr/bin/env python
"""Diagnose event-region repair/damage for class-routed prediction branches."""

import argparse
import copy
import json
import os
import sys
import importlib.util
from collections import OrderedDict, defaultdict
from pathlib import Path

import cv2
import numpy as np

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
from detectron2.data import DatasetCatalog  # noqa: E402
from train_mask2former_cosec import register_cosec  # noqa: E402

from diagnose_pair_transition_from_predictions import (  # noqa: E402
    decode_prediction,
    load_label,
    load_prediction_index,
    prediction_key,
    resize_stat,
    setup_mapper,
    valid_label_mask,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--event-config", required=True)
    parser.add_argument(
        "--branch",
        action="append",
        required=True,
        help="Branch spec in name=/path/to/sem_seg_predictions.json form.",
    )
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--basis", required=True)
    parser.add_argument(
        "--fixed-route",
        action="append",
        required=True,
        help="Class route in class_name=branch_name form. Repeatable.",
    )
    parser.add_argument(
        "--boundary-source",
        default="none",
        choices=["none", "anchor", "basis", "union", "intersection"],
    )
    parser.add_argument("--boundary-radius", type=int, default=0)
    parser.add_argument(
        "--protect-anchor-class",
        action="append",
        default=[],
        help="Do not route pixels whose anchor prediction is this class.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def parse_branch_specs(specs):
    branches = OrderedDict()
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid branch spec: {spec}")
        name, path = spec.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Empty branch name in {spec}")
        if name in branches:
            raise ValueError(f"Duplicate branch name: {name}")
        branches[name] = str(Path(path).expanduser())
    return branches


def parse_routes(route_specs, branches):
    routes = OrderedDict()
    for spec in route_specs:
        if "=" not in spec:
            raise ValueError(f"Invalid route: {spec}")
        class_name, branch_name = [part.strip() for part in spec.split("=", 1)]
        if class_name not in CLASSES:
            raise ValueError(f"Unknown class: {class_name}")
        if branch_name not in branches:
            raise ValueError(f"Unknown branch for {class_name}: {branch_name}")
        routes[CLASSES.index(class_name)] = branch_name
    return routes


def parse_class_names(class_names):
    ids = []
    for class_name in class_names:
        if class_name not in CLASSES:
            raise ValueError(f"Unknown class: {class_name}")
        ids.append(CLASSES.index(class_name))
    return ids


def semantic_boundary_band(pred, radius, valid):
    if radius <= 0:
        return np.zeros(pred.shape, dtype=bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    low = pred.astype(np.float32, copy=True)
    high = pred.astype(np.float32, copy=True)
    low[~valid] = -1000.0
    high[~valid] = 1000.0
    local_max = cv2.dilate(low, kernel)
    local_min = cv2.erode(high, kernel)
    return valid & (local_max != local_min)


def boundary_gate(source, radius, anchor_pred, basis_pred, valid):
    if source == "none":
        return valid
    anchor_boundary = semantic_boundary_band(anchor_pred, radius, valid)
    basis_boundary = semantic_boundary_band(basis_pred, radius, valid)
    if source == "anchor":
        return anchor_boundary
    if source == "basis":
        return basis_boundary
    if source == "union":
        return anchor_boundary | basis_boundary
    if source == "intersection":
        return anchor_boundary & basis_boundary
    raise ValueError(f"Unknown boundary source: {source}")


def route_prediction(anchor_pred, basis_pred, preds, routes, gate):
    merged = anchor_pred.copy()
    routed = np.zeros(anchor_pred.shape, dtype=bool)
    for class_id, branch_name in routes.items():
        take = (basis_pred == class_id) & gate
        if not take.any():
            continue
        branch_pred = preds[branch_name]
        merged[take] = branch_pred[take]
        routed |= take
    return merged, routed


class ConfusionMeter:
    def __init__(self, num_classes=19):
        self.num_classes = int(num_classes)
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, pred, label):
        keep = (label != 255) & (label >= 0) & (label < self.num_classes)
        keep &= (pred >= 0) & (pred < self.num_classes)
        indices = self.num_classes * label[keep].astype(np.int64) + pred[keep].astype(np.int64)
        self.matrix += np.bincount(indices, minlength=self.num_classes**2).reshape(
            self.num_classes,
            self.num_classes,
        )

    def metrics(self):
        hist = self.matrix.astype(np.float64)
        tp = np.diag(hist)
        pos_gt = hist.sum(axis=1)
        pos_pred = hist.sum(axis=0)
        union = pos_gt + pos_pred - tp
        iou = np.divide(tp, union, out=np.full_like(tp, np.nan), where=union > 0)
        acc = np.divide(tp, pos_gt, out=np.full_like(tp, np.nan), where=pos_gt > 0)
        total = hist.sum()
        return {
            "mIoU": float(100.0 * np.nanmean(iou)),
            "mAcc": float(100.0 * np.nanmean(acc)),
            "aAcc": float(100.0 * tp.sum() / total) if total > 0 else float("nan"),
            "class_iou": {
                CLASSES[idx]: (None if np.isnan(value) else float(100.0 * value))
                for idx, value in enumerate(iou)
            },
        }


def empty_region_counts():
    return {
        "valid_pixels": 0,
        "region_pixels": 0,
        "routed_pixels": 0,
        "changed": 0,
        "repaired": 0,
        "damaged": 0,
        "both_wrong": 0,
        "anchor_wrong": 0,
    }


def empty_pair_counts():
    return {"changed": 0, "repaired": 0, "damaged": 0, "both_wrong": 0}


def div(num, den):
    return float(num / den) if den else 0.0


def pair_name(src, dst):
    return f"{CLASSES[int(src)]}->{CLASSES[int(dst)]}"


def update_region(region_counts, pair_counts, name, region, routed, anchor_pred, pred, label, valid):
    mask = valid & region
    changed = mask & (anchor_pred != pred)
    repaired = changed & (anchor_pred != label) & (pred == label)
    damaged = changed & (anchor_pred == label) & (pred != label)
    both_wrong = changed & (anchor_pred != label) & (pred != label)

    counts = region_counts[name]
    counts["valid_pixels"] += int(valid.sum())
    counts["region_pixels"] += int(mask.sum())
    counts["routed_pixels"] += int((mask & routed).sum())
    counts["changed"] += int(changed.sum())
    counts["repaired"] += int(repaired.sum())
    counts["damaged"] += int(damaged.sum())
    counts["both_wrong"] += int(both_wrong.sum())
    counts["anchor_wrong"] += int((mask & (anchor_pred != label)).sum())

    num_classes = len(CLASSES)
    if not changed.any():
        return
    changed_counts = np.bincount(
        num_classes * anchor_pred[changed].astype(np.int64) + pred[changed].astype(np.int64),
        minlength=num_classes**2,
    )
    repaired_counts = np.bincount(
        num_classes * anchor_pred[repaired].astype(np.int64) + pred[repaired].astype(np.int64),
        minlength=num_classes**2,
    )
    damaged_counts = np.bincount(
        num_classes * anchor_pred[damaged].astype(np.int64) + pred[damaged].astype(np.int64),
        minlength=num_classes**2,
    )
    both_wrong_counts = np.bincount(
        num_classes * anchor_pred[both_wrong].astype(np.int64) + pred[both_wrong].astype(np.int64),
        minlength=num_classes**2,
    )
    for idx in np.flatnonzero(changed_counts):
        row = pair_counts[name][pair_name(idx // num_classes, idx % num_classes)]
        row["changed"] += int(changed_counts[idx])
        row["repaired"] += int(repaired_counts[idx])
        row["damaged"] += int(damaged_counts[idx])
        row["both_wrong"] += int(both_wrong_counts[idx])


def finalize_region(counts):
    out = dict(counts)
    out["net_repaired"] = counts["repaired"] - counts["damaged"]
    out["region_coverage"] = div(counts["region_pixels"], counts["valid_pixels"])
    out["routed_rate_in_region"] = div(counts["routed_pixels"], counts["region_pixels"])
    out["changed_rate_in_region"] = div(counts["changed"], counts["region_pixels"])
    out["repair_rate_of_anchor_wrong"] = div(counts["repaired"], counts["anchor_wrong"])
    out["repair_precision"] = div(counts["repaired"], counts["changed"])
    out["damage_rate"] = div(counts["damaged"], counts["changed"])
    out["both_wrong_rate"] = div(counts["both_wrong"], counts["changed"])
    return out


def finalize_pair(pair, counts):
    out = dict(counts)
    out["pair"] = pair
    out["net_repaired"] = counts["repaired"] - counts["damaged"]
    out["repair_precision"] = div(counts["repaired"], counts["changed"])
    out["damage_rate"] = div(counts["damaged"], counts["changed"])
    return out


def decode_records(records, branch_indices, branch_names):
    decoded = []
    missing = []
    for record in records:
        key = prediction_key(record["file_name"])
        label = load_label(record).astype(np.int64, copy=False)
        preds = OrderedDict()
        for name in branch_names:
            rows = branch_indices[name].get(record["file_name"]) or branch_indices[name].get(key)
            if not rows:
                missing.append({"file_name": record["file_name"], "branch": name})
                continue
            preds[name] = decode_prediction(rows, label.shape).astype(np.int64, copy=False)
        if len(preds) == len(branch_names):
            decoded.append({"record": record, "label": label, "preds": preds})
    return decoded, missing


def main():
    args = parse_args()
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    register_cosec()

    branches = parse_branch_specs(args.branch)
    if args.anchor not in branches:
        raise ValueError(f"Unknown anchor branch: {args.anchor}")
    if args.basis not in branches:
        raise ValueError(f"Unknown basis branch: {args.basis}")
    routes = parse_routes(args.fixed_route, branches)
    protect_anchor_ids = parse_class_names(args.protect_anchor_class)
    mapper = setup_mapper(args.event_config)

    records = list(DatasetCatalog.get(args.dataset))
    if args.limit is not None:
        records = records[: args.limit]

    branch_indices = {name: load_prediction_index(path) for name, path in branches.items()}
    decoded, missing = decode_records(records, branch_indices, list(branches))
    if missing:
        raise RuntimeError(f"Missing predictions: {len(missing)}; first={missing[:5]}")

    anchor_meter = ConfusionMeter(num_classes=len(CLASSES))
    routed_meter = ConfusionMeter(num_classes=len(CLASSES))
    region_counts = defaultdict(empty_region_counts)
    pair_counts = defaultdict(lambda: defaultdict(empty_pair_counts))

    for item in decoded:
        record = item["record"]
        label = item["label"]
        preds = item["preds"]
        anchor_pred = preds[args.anchor]
        basis_pred = preds[args.basis]
        valid = valid_label_mask(label, anchor_pred, basis_pred)
        gate = boundary_gate(args.boundary_source, args.boundary_radius, anchor_pred, basis_pred, valid)
        if protect_anchor_ids:
            gate = gate & ~np.isin(anchor_pred, protect_anchor_ids)
        routed_pred, routed = route_prediction(anchor_pred, basis_pred, preds, routes, gate)

        mapped = mapper(copy.deepcopy(record))
        event_stats = mapped["event_stats"].float()
        raw_event = valid & (resize_stat(event_stats, 0, label.shape) > 0)
        support = valid & (resize_stat(event_stats, 3, label.shape) > 0)
        event_union = raw_event | support

        regions = OrderedDict(
            [
                ("all", valid),
                ("event_union", event_union),
                ("raw_event", raw_event),
                ("support", support),
                ("raw_only", raw_event & ~support),
                ("support_only", support & ~raw_event),
            ]
        )
        anchor_meter.update(anchor_pred, label)
        routed_meter.update(routed_pred, label)
        for name, region in regions.items():
            update_region(
                region_counts,
                pair_counts,
                name,
                region,
                routed,
                anchor_pred,
                routed_pred,
                label,
                valid,
            )

    regions_out = OrderedDict((name, finalize_region(counts)) for name, counts in region_counts.items())
    pairs_out = OrderedDict()
    for name, pairs in pair_counts.items():
        rows = [finalize_pair(pair, counts) for pair, counts in pairs.items()]
        pairs_out[name] = {
            "top_positive_by_net": sorted(rows, key=lambda row: row["net_repaired"], reverse=True)[
                : args.top_k
            ],
            "top_negative_by_net": sorted(rows, key=lambda row: row["net_repaired"])[: args.top_k],
            "top_by_changed": sorted(rows, key=lambda row: row["changed"], reverse=True)[: args.top_k],
        }

    output = OrderedDict(
        [
            ("args", vars(args)),
            ("sample_count", len(decoded)),
            ("classes", list(CLASSES)),
            ("routes", {CLASSES[class_id]: branch for class_id, branch in routes.items()}),
            ("anchor_metrics", anchor_meter.metrics()),
            ("routed_metrics", routed_meter.metrics()),
            (
                "delta_mIoU",
                routed_meter.metrics()["mIoU"] - anchor_meter.metrics()["mIoU"],
            ),
            ("regions", regions_out),
            ("pairs", pairs_out),
        ]
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote routed event-pair diagnostics: {out_path}")
    print(
        f"anchor={output['anchor_metrics']['mIoU']:.4f} "
        f"routed={output['routed_metrics']['mIoU']:.4f} "
        f"delta={output['delta_mIoU']:+.4f}"
    )
    for name, counts in regions_out.items():
        print(
            f"{name}: changed={counts['changed']} repaired={counts['repaired']} "
            f"damaged={counts['damaged']} net={counts['net_repaired']} "
            f"precision={100.0 * counts['repair_precision']:.2f}%"
        )


if __name__ == "__main__":
    main()
