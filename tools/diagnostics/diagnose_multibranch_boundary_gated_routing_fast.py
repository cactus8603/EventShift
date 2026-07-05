#!/usr/bin/env python
"""Diagnose boundary-gated class routing across saved prediction JSON branches."""

import argparse
import json
import os
import sys
import importlib.util
from collections import OrderedDict
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
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
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
    parser.add_argument("--boundary-radii", default="1,3,5,9,15")
    parser.add_argument(
        "--boundary-sources",
        default="anchor,basis,union,intersection",
        help="Comma-separated sources: anchor,basis,union,intersection.",
    )
    parser.add_argument(
        "--protect-anchor-class",
        action="append",
        default=[],
        help="Do not route pixels whose anchor prediction is this class. Repeatable.",
    )
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def split_csv(text):
    return [part.strip() for part in str(text).split(",") if part.strip()]


def parse_ints(text):
    return [int(part) for part in split_csv(text)]


def parse_branch_specs(specs):
    branches = OrderedDict()
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid branch spec: {spec}")
        name, path = spec.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Empty branch name: {spec}")
        if name in branches:
            raise ValueError(f"Duplicate branch name: {name}")
        branches[name] = str(Path(path).expanduser())
    return branches


def parse_fixed_routes(route_specs, branches):
    route = OrderedDict()
    for spec in route_specs:
        if "=" not in spec:
            raise ValueError(f"Invalid route: {spec}")
        class_name, branch_name = [part.strip() for part in spec.split("=", 1)]
        if class_name not in CLASSES:
            raise ValueError(f"Unknown class: {class_name}")
        if branch_name not in branches:
            raise ValueError(f"Unknown branch for {class_name}: {branch_name}")
        route[CLASSES.index(class_name)] = branch_name
    return route


def parse_class_names(class_names):
    class_ids = []
    for class_name in class_names:
        if class_name not in CLASSES:
            raise ValueError(f"Unknown class: {class_name}")
        class_ids.append(CLASSES.index(class_name))
    return class_ids


def valid_label_mask(label):
    return (label != 255) & (label >= 0) & (label < len(CLASSES))


def semantic_boundary_band(pred, radius, valid=None):
    if radius <= 0:
        return np.zeros(pred.shape, dtype=bool)
    if valid is None:
        valid = np.ones(pred.shape, dtype=bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    low = pred.astype(np.float32, copy=True)
    high = pred.astype(np.float32, copy=True)
    low[~valid] = -1000.0
    high[~valid] = 1000.0
    local_max = cv2.dilate(low, kernel)
    local_min = cv2.erode(high, kernel)
    return valid & (local_max != local_min)


class ConfusionMeter:
    def __init__(self, num_classes=19):
        self.num_classes = int(num_classes)
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, pred, label):
        keep = valid_label_mask(label)
        keep &= (pred >= 0) & (pred < self.num_classes)
        indices = self.num_classes * label[keep].astype(np.int64) + pred[keep].astype(np.int64)
        self.matrix += np.bincount(indices, minlength=self.num_classes**2).reshape(
            self.num_classes,
            self.num_classes,
        )

    def metrics(self):
        hist = self.matrix.astype(np.float64)
        true_positive = np.diag(hist)
        pos_gt = hist.sum(axis=1)
        pos_pred = hist.sum(axis=0)
        union = pos_gt + pos_pred - true_positive
        iou = np.divide(
            true_positive,
            union,
            out=np.full_like(true_positive, np.nan),
            where=union > 0,
        )
        acc = np.divide(
            true_positive,
            pos_gt,
            out=np.full_like(true_positive, np.nan),
            where=pos_gt > 0,
        )
        total = hist.sum()
        return {
            "mIoU": float(100.0 * np.nanmean(iou)),
            "mAcc": float(100.0 * np.nanmean(acc)),
            "aAcc": float(100.0 * true_positive.sum() / total) if total > 0 else float("nan"),
            "class_iou": {
                CLASSES[idx]: (None if np.isnan(value) else float(100.0 * value))
                for idx, value in enumerate(iou)
            },
        }


def empty_counts():
    return {
        "valid_pixels": 0,
        "anchor_wrong": 0,
        "repaired": 0,
        "damaged": 0,
        "changed": 0,
        "routed_pixels": 0,
    }


def update_counts(counts, anchor_pred, pred, label, routed):
    valid = valid_label_mask(label)
    anchor_wrong = valid & (anchor_pred != label)
    anchor_correct = valid & (anchor_pred == label)
    counts["valid_pixels"] += int(valid.sum())
    counts["anchor_wrong"] += int(anchor_wrong.sum())
    counts["repaired"] += int((anchor_wrong & (pred == label)).sum())
    counts["damaged"] += int((anchor_correct & (pred != label)).sum())
    counts["changed"] += int((valid & (anchor_pred != pred)).sum())
    counts["routed_pixels"] += int((valid & routed).sum())


def finalize_counts(counts):
    valid = max(int(counts["valid_pixels"]), 1)
    wrong = max(int(counts["anchor_wrong"]), 1)
    return {
        **counts,
        "repair_rate": float(counts["repaired"] / wrong),
        "net_repaired": int(counts["repaired"] - counts["damaged"]),
        "changed_rate": float(counts["changed"] / valid),
        "routed_rate": float(counts["routed_pixels"] / valid),
    }


def decode_records(records, branch_indices, branch_names):
    decoded = []
    missing = []
    for record in records:
        key = prediction_key(record["file_name"])
        rows_by_branch = {}
        for name in branch_names:
            rows = branch_indices[name].get(record["file_name"]) or branch_indices[name].get(key)
            if not rows:
                missing.append({"file_name": record["file_name"], "branch": name})
                continue
            rows_by_branch[name] = rows
        if len(rows_by_branch) != len(branch_names):
            continue
        label = load_label(record).astype(np.uint8, copy=False)
        preds = OrderedDict()
        for name in branch_names:
            preds[name] = decode_prediction(rows_by_branch[name], label.shape).astype(np.uint8, copy=False)
        decoded.append({"label": label, "preds": preds, "record": record})
    return decoded, missing


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


def update_method(methods, name, pred, anchor_pred, label, routed):
    if name not in methods:
        methods[name] = {"meter": ConfusionMeter(num_classes=len(CLASSES)), "counts": empty_counts()}
    methods[name]["meter"].update(pred, label)
    update_counts(methods[name]["counts"], anchor_pred, pred, label, routed)


def boundary_for_source(source, anchor_boundary, basis_boundary):
    if source == "anchor":
        return anchor_boundary
    if source == "basis":
        return basis_boundary
    if source == "union":
        return anchor_boundary | basis_boundary
    if source == "intersection":
        return anchor_boundary & basis_boundary
    raise ValueError(f"Unknown boundary source: {source}")


def scene_from_record(record):
    if record.get("scene"):
        return record["scene"]
    parts = Path(record["file_name"]).parts
    seq_name = next(
        (part for part in parts if part.startswith(("Day_", "Night_", "REAL_"))),
        "",
    )
    pieces = seq_name.split("_")
    return pieces[1] if len(pieces) > 1 else "unknown"


def update_all_methods(
    methods,
    item,
    anchor_name,
    basis_name,
    branch_names,
    routes,
    radii,
    sources,
    protect_anchor_ids,
):
    label = item["label"]
    preds = item["preds"]
    anchor_pred = preds[anchor_name]
    basis_pred = preds[basis_name]
    valid = valid_label_mask(label)

    update_method(
        methods,
        f"anchor_{anchor_name}",
        anchor_pred,
        anchor_pred,
        label,
        np.zeros(label.shape, dtype=bool),
    )
    for branch_name in branch_names:
        update_method(
            methods,
            f"branch_{branch_name}",
            preds[branch_name],
            anchor_pred,
            label,
            valid,
        )

    protect_gate = valid
    if protect_anchor_ids:
        protect_gate = protect_gate & ~np.isin(anchor_pred, protect_anchor_ids)

    pred, routed = route_prediction(anchor_pred, basis_pred, preds, routes, protect_gate)
    update_method(methods, "fixed_route_all", pred, anchor_pred, label, routed)

    for radius in radii:
        anchor_boundary = semantic_boundary_band(anchor_pred, radius, valid)
        basis_boundary = semantic_boundary_band(basis_pred, radius, valid)
        for source in sources:
            gate = boundary_for_source(source, anchor_boundary, basis_boundary) & protect_gate
            pred, routed = route_prediction(anchor_pred, basis_pred, preds, routes, gate)
            update_method(
                methods,
                f"fixed_route_{source}_boundary_r{radius}",
                pred,
                anchor_pred,
                label,
                routed,
            )


def finalize_method_dict(methods):
    return OrderedDict(
        (
            name,
            {
                **value["meter"].metrics(),
                **finalize_counts(value["counts"]),
            },
        )
        for name, value in methods.items()
    )


def write_markdown(output, out_path):
    md_path = out_path.with_suffix(".md")
    lines = [
        "# Multibranch Boundary-Gated Routing Diagnostic",
        "",
        f"dataset: `{output['args']['dataset']}`",
        f"anchor: `{output['args']['anchor']}`",
        f"basis: `{output['args']['basis']}`",
        "",
        "routes:",
        "",
    ]
    for class_name, branch_name in output["routes"].items():
        lines.append(f"- `{class_name}` -> `{branch_name}`")
    lines.extend(
        [
            "",
            "| Method | mIoU | mAcc | aAcc | Changed | Routed | Repair rate | Net repaired |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in output["top_by_mIoU"]:
        lines.append(
            f"| `{row['method']}` | {row['mIoU']:.4f} | {row['mAcc']:.4f} | "
            f"{row['aAcc']:.4f} | {100.0 * row['changed_rate']:.4f}% | "
            f"{100.0 * row['routed_rate']:.4f}% | {100.0 * row['repair_rate']:.2f}% | "
            f"{row['net_repaired']} |"
        )
    if output.get("group_metrics"):
        lines.extend(["", "## Scene Groups", ""])
        key_methods = [
            f"anchor_{output['args']['anchor']}",
            "fixed_route_all",
        ]
        for row in output["top_by_mIoU"][:3]:
            if row["method"].startswith("fixed_route_") and row["method"] not in key_methods:
                key_methods.append(row["method"])
        for group_name, group in output["group_metrics"].items():
            lines.extend(
                [
                    "",
                    f"### {group_name}",
                    "",
                    "| Method | mIoU | Changed | Routed | Net repaired |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            for method in key_methods:
                if method not in group["methods"]:
                    continue
                row = group["methods"][method]
                lines.append(
                    f"| `{method}` | {row['mIoU']:.4f} | "
                    f"{100.0 * row['changed_rate']:.4f}% | "
                    f"{100.0 * row['routed_rate']:.4f}% | {row['net_repaired']} |"
                )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def main():
    args = parse_args()
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    register_cosec()

    branches = parse_branch_specs(args.branch)
    branch_names = list(branches)
    if args.anchor not in branches:
        raise ValueError(f"Unknown anchor: {args.anchor}")
    if args.basis not in branches:
        raise ValueError(f"Unknown basis: {args.basis}")
    routes = parse_fixed_routes(args.fixed_route, branches)
    protect_anchor_ids = parse_class_names(args.protect_anchor_class)
    radii = parse_ints(args.boundary_radii)
    sources = split_csv(args.boundary_sources)

    branch_indices = {name: load_prediction_index(path) for name, path in branches.items()}
    records = list(DatasetCatalog.get(args.dataset))
    decoded, missing = decode_records(records, branch_indices, branch_names)
    if missing:
        raise RuntimeError(f"Missing predictions: {len(missing)}; first={missing[:5]}")

    methods = OrderedDict()
    group_methods = OrderedDict()
    for item in decoded:
        update_all_methods(
            methods,
            item,
            args.anchor,
            args.basis,
            branch_names,
            routes,
            radii,
            sources,
            protect_anchor_ids,
        )
        scene = scene_from_record(item["record"])
        if scene not in group_methods:
            group_methods[scene] = OrderedDict()
        update_all_methods(
            group_methods[scene],
            item,
            args.anchor,
            args.basis,
            branch_names,
            routes,
            radii,
            sources,
            protect_anchor_ids,
        )

    output = {
        "args": vars(args),
        "sample_count": len(decoded),
        "routes": {CLASSES[class_id]: branch for class_id, branch in routes.items()},
        "methods": finalize_method_dict(methods),
        "group_metrics": OrderedDict(
            (
                group_name,
                {
                    "methods": finalize_method_dict(values),
                    "sample_count": sum(
                        1 for item in decoded if scene_from_record(item["record"]) == group_name
                    ),
                },
            )
            for group_name, values in group_methods.items()
        ),
    }
    output["top_by_mIoU"] = [
        {"method": name, **metrics}
        for name, metrics in sorted(
            output["methods"].items(),
            key=lambda item: item[1]["mIoU"],
            reverse=True,
        )
    ]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, sort_keys=True)
        f.write("\n")
    md_path = write_markdown(output, out_path)

    print(f"Wrote boundary-gated diagnostics: {out_path}")
    print(f"Wrote summary: {md_path}")
    for row in output["top_by_mIoU"][:12]:
        print(
            f"  {row['method']}: mIoU={row['mIoU']:.4f}, "
            f"changed={100.0 * row['changed_rate']:.4f}%, "
            f"routed={100.0 * row['routed_rate']:.4f}%, net={row['net_repaired']}"
        )


if __name__ == "__main__":
    main()
