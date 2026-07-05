#!/usr/bin/env python
"""Fast class-wise routing diagnostics across saved prediction JSON branches."""

import argparse
import json
import os
import sys
import importlib.util
from collections import OrderedDict
from pathlib import Path

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
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument(
        "--fixed-route",
        action="append",
        default=[],
        help="Optional fixed route in class_name=branch_name form. Repeatable.",
    )
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
            raise ValueError(f"Empty branch name: {spec}")
        if name in branches:
            raise ValueError(f"Duplicate branch name: {name}")
        branches[name] = str(Path(path).expanduser())
    return branches


def confusion_from_arrays(label, pred, mask, num_classes):
    keep = mask & (label != 255) & (label >= 0) & (label < num_classes)
    keep &= (pred >= 0) & (pred < num_classes)
    indices = num_classes * label[keep].astype(np.int64) + pred[keep].astype(np.int64)
    return np.bincount(indices, minlength=num_classes**2).reshape(num_classes, num_classes)


def metrics_from_matrix(matrix):
    hist = matrix.astype(np.float64)
    true_positive = np.diag(hist)
    pos_gt = hist.sum(axis=1)
    pos_pred = hist.sum(axis=0)
    union = pos_gt + pos_pred - true_positive
    iou = np.divide(true_positive, union, out=np.full_like(true_positive, np.nan), where=union > 0)
    acc = np.divide(true_positive, pos_gt, out=np.full_like(true_positive, np.nan), where=pos_gt > 0)
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


def decode_branch_predictions(records, branch_indices, branch_names):
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


def build_contributions(decoded, branch_names, basis_name):
    num_records = len(decoded)
    num_classes = len(CLASSES)
    num_branches = len(branch_names)
    contributions = np.zeros(
        (num_records, num_classes, num_branches, num_classes, num_classes),
        dtype=np.int64,
    )
    routed_pixels = np.zeros((num_records, num_classes), dtype=np.int64)
    for record_idx, item in enumerate(decoded):
        label = item["label"]
        basis = item["preds"][basis_name]
        for class_id in range(num_classes):
            region = basis == class_id
            routed_pixels[record_idx, class_id] = int(region.sum())
            if not region.any():
                continue
            for branch_idx, branch_name in enumerate(branch_names):
                contributions[record_idx, class_id, branch_idx] = confusion_from_arrays(
                    label,
                    item["preds"][branch_name],
                    region,
                    num_classes,
                )
    return contributions, routed_pixels


def route_matrix(contributions, indices, branch_for_class):
    num_classes = len(CLASSES)
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    selected = contributions[indices]
    for class_id, branch_idx in enumerate(branch_for_class):
        matrix += selected[:, class_id, branch_idx].sum(axis=0)
    return matrix


def branch_matrix(contributions, indices, branch_idx):
    branch_for_class = [branch_idx] * len(CLASSES)
    return route_matrix(contributions, indices, branch_for_class)


def route_pixel_counts(routed_pixels, indices, branch_for_class, branch_names):
    out = {name: 0 for name in branch_names}
    selected = routed_pixels[indices]
    for class_id, branch_idx in enumerate(branch_for_class):
        out[branch_names[branch_idx]] += int(selected[:, class_id].sum())
    return out


def select_routes(contributions, routed_pixels, indices, branch_names, anchor_idx, min_delta):
    anchor_assign = [anchor_idx] * len(CLASSES)
    anchor_matrix = route_matrix(contributions, indices, anchor_assign)
    anchor_metrics = metrics_from_matrix(anchor_matrix)
    anchor_miou = anchor_metrics["mIoU"]
    class_rows = []
    selected = list(anchor_assign)

    class_anchor = contributions[indices, :, anchor_idx].sum(axis=0)
    for class_id, class_name in enumerate(CLASSES):
        best = {
            "class": class_name,
            "class_id": class_id,
            "branch": branch_names[anchor_idx],
            "branch_idx": anchor_idx,
            "mIoU": anchor_miou,
            "delta_vs_anchor": 0.0,
            "routed_pixels": 0,
        }
        for branch_idx, branch_name in enumerate(branch_names):
            if branch_idx == anchor_idx:
                continue
            matrix = anchor_matrix - class_anchor[class_id]
            matrix = matrix + contributions[indices, class_id, branch_idx].sum(axis=0)
            metrics = metrics_from_matrix(matrix)
            delta = metrics["mIoU"] - anchor_miou
            routed = int(routed_pixels[indices, class_id].sum())
            row = {
                "class": class_name,
                "class_id": class_id,
                "branch": branch_name,
                "branch_idx": branch_idx,
                "mIoU": metrics["mIoU"],
                "delta_vs_anchor": delta,
                "routed_pixels": routed,
            }
            if delta > best["delta_vs_anchor"]:
                best = row
        class_rows.append(best)
        if best["branch_idx"] != anchor_idx and best["delta_vs_anchor"] > min_delta:
            selected[class_id] = best["branch_idx"]

    class_rows.sort(key=lambda row: row["delta_vs_anchor"], reverse=True)
    return selected, class_rows, anchor_metrics


def named_routes(branch_for_class, branch_names, anchor_idx):
    return {
        CLASSES[class_id]: branch_names[branch_idx]
        for class_id, branch_idx in enumerate(branch_for_class)
        if branch_idx != anchor_idx
    }


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


def parse_fixed_routes(route_specs, branch_names, anchor_idx):
    branch_lookup = {name: idx for idx, name in enumerate(branch_names)}
    fixed = [anchor_idx] * len(CLASSES)
    for spec in route_specs:
        if "=" not in spec:
            raise ValueError(f"Invalid fixed route: {spec}")
        class_name, branch_name = spec.split("=", 1)
        class_name = class_name.strip()
        branch_name = branch_name.strip()
        if class_name not in CLASSES:
            raise ValueError(f"Unknown fixed-route class: {class_name}")
        if branch_name not in branch_lookup:
            raise ValueError(f"Unknown fixed-route branch: {branch_name}")
        fixed[CLASSES.index(class_name)] = branch_lookup[branch_name]
    return fixed


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
    anchor_idx = branch_names.index(args.anchor)

    branch_indices = {name: load_prediction_index(path) for name, path in branches.items()}
    records = DatasetCatalog.get(args.dataset)
    decoded, missing = decode_branch_predictions(records, branch_indices, branch_names)
    if missing:
        raise RuntimeError(f"Missing predictions: {len(missing)}; first={missing[:5]}")
    contributions, routed_pixels = build_contributions(decoded, branch_names, args.basis)

    all_indices = np.arange(len(decoded), dtype=np.int64)
    branch_metrics = OrderedDict()
    for branch_idx, branch_name in enumerate(branch_names):
        matrix = branch_matrix(contributions, all_indices, branch_idx)
        branch_metrics[branch_name] = metrics_from_matrix(matrix)

    selected_all, class_rows_all, anchor_metrics_all = select_routes(
        contributions,
        routed_pixels,
        all_indices,
        branch_names,
        anchor_idx,
        args.min_delta,
    )
    selected_all_matrix = route_matrix(contributions, all_indices, selected_all)
    selected_all_metrics = metrics_from_matrix(selected_all_matrix)
    selected_all_metrics["routed_pixels_by_branch"] = route_pixel_counts(
        routed_pixels,
        all_indices,
        selected_all,
        branch_names,
    )
    fixed_routes = None
    fixed_all_metrics = None
    if args.fixed_route:
        fixed_routes = parse_fixed_routes(args.fixed_route, branch_names, anchor_idx)
        fixed_all_matrix = route_matrix(contributions, all_indices, fixed_routes)
        fixed_all_metrics = metrics_from_matrix(fixed_all_matrix)
        fixed_all_metrics["routed_pixels_by_branch"] = route_pixel_counts(
            routed_pixels,
            all_indices,
            fixed_routes,
            branch_names,
        )

    group_metrics = OrderedDict()
    groups = OrderedDict()
    for idx, item in enumerate(decoded):
        groups.setdefault(scene_from_record(item["record"]), []).append(idx)
    for group_name, indices_list in groups.items():
        group_indices = np.array(indices_list, dtype=np.int64)
        group_row = OrderedDict()
        group_row["count"] = int(len(group_indices))
        group_row["anchor"] = metrics_from_matrix(branch_matrix(contributions, group_indices, anchor_idx))
        group_row["selected"] = metrics_from_matrix(route_matrix(contributions, group_indices, selected_all))
        if fixed_routes is not None:
            group_row["fixed"] = metrics_from_matrix(route_matrix(contributions, group_indices, fixed_routes))
        group_row["branches"] = OrderedDict()
        for branch_idx, branch_name in enumerate(branch_names):
            group_row["branches"][branch_name] = metrics_from_matrix(
                branch_matrix(contributions, group_indices, branch_idx)
            )
        group_metrics[group_name] = group_row

    folds = []
    for fold in range(args.folds):
        test_indices = np.array([idx for idx in all_indices if idx % args.folds == fold], dtype=np.int64)
        train_indices = np.array([idx for idx in all_indices if idx % args.folds != fold], dtype=np.int64)
        selected, class_rows, train_anchor = select_routes(
            contributions,
            routed_pixels,
            train_indices,
            branch_names,
            anchor_idx,
            args.min_delta,
        )
        test_anchor = metrics_from_matrix(branch_matrix(contributions, test_indices, anchor_idx))
        test_selected_matrix = route_matrix(contributions, test_indices, selected)
        test_selected = metrics_from_matrix(test_selected_matrix)
        test_selected["routed_pixels_by_branch"] = route_pixel_counts(
            routed_pixels,
            test_indices,
            selected,
            branch_names,
        )
        fold_result = OrderedDict(
            [
                ("fold", fold),
                ("train_count", int(len(train_indices))),
                ("test_count", int(len(test_indices))),
                ("selected_routes", named_routes(selected, branch_names, anchor_idx)),
                ("train_anchor_mIoU", train_anchor["mIoU"]),
                ("train_class_candidates", class_rows),
                ("test_anchor", test_anchor),
                ("test_selected", test_selected),
            ]
        )
        if fixed_routes is not None:
            test_fixed_matrix = route_matrix(contributions, test_indices, fixed_routes)
            test_fixed = metrics_from_matrix(test_fixed_matrix)
            test_fixed["routed_pixels_by_branch"] = route_pixel_counts(
                routed_pixels,
                test_indices,
                fixed_routes,
                branch_names,
            )
            fold_result["fixed_routes"] = named_routes(fixed_routes, branch_names, anchor_idx)
            fold_result["test_fixed"] = test_fixed
        for branch_idx, branch_name in enumerate(branch_names):
            fold_result[f"test_branch_{branch_name}"] = metrics_from_matrix(
                branch_matrix(contributions, test_indices, branch_idx)
            )
        folds.append(fold_result)

    def avg(key, metric="mIoU"):
        return float(np.mean([fold[key][metric] for fold in folds]))

    summary = {
        "avg_test_anchor_mIoU": avg("test_anchor"),
        "avg_test_selected_mIoU": avg("test_selected"),
        "selected_minus_anchor": avg("test_selected") - avg("test_anchor"),
    }
    if fixed_routes is not None:
        summary["avg_test_fixed_mIoU"] = avg("test_fixed")
        summary["fixed_minus_anchor"] = avg("test_fixed") - avg("test_anchor")
    for branch_name in branch_names:
        summary[f"avg_test_branch_{branch_name}_mIoU"] = avg(f"test_branch_{branch_name}")

    output = {
        "args": vars(args),
        "branches": branches,
        "classes": list(CLASSES),
        "sample_count": len(decoded),
        "branch_metrics": branch_metrics,
        "anchor_metrics_all": anchor_metrics_all,
        "selected_routes_all": named_routes(selected_all, branch_names, anchor_idx),
        "selected_class_candidates_all": class_rows_all,
        "overall_selected": selected_all_metrics,
        "fixed_routes_all": (
            None if fixed_routes is None else named_routes(fixed_routes, branch_names, anchor_idx)
        ),
        "overall_fixed": fixed_all_metrics,
        "group_metrics": group_metrics,
        "folds": folds,
        "summary": summary,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Wrote multibranch class routing: {out_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("overall branches:")
    for name, metrics in sorted(branch_metrics.items(), key=lambda item: item[1]["mIoU"], reverse=True):
        print(f"  {name}: mIoU={metrics['mIoU']:.4f}, mAcc={metrics['mAcc']:.4f}, aAcc={metrics['aAcc']:.4f}")
    print(
        "overall selected: "
        f"mIoU={selected_all_metrics['mIoU']:.4f}, routes={output['selected_routes_all']}"
    )
    if fixed_all_metrics is not None:
        print(
            "overall fixed: "
            f"mIoU={fixed_all_metrics['mIoU']:.4f}, routes={output['fixed_routes_all']}"
        )
    print("group metrics:")
    for group_name, row in group_metrics.items():
        message = (
            f"  {group_name} n={row['count']} "
            f"anchor={row['anchor']['mIoU']:.4f} "
            f"selected={row['selected']['mIoU']:.4f}"
        )
        if fixed_routes is not None:
            message += f" fixed={row['fixed']['mIoU']:.4f}"
        print(message)
    for fold in folds:
        message = (
            f"fold {fold['fold']}: anchor={fold['test_anchor']['mIoU']:.4f} "
            f"selected={fold['test_selected']['mIoU']:.4f} routes={fold['selected_routes']}"
        )
        if fixed_routes is not None:
            message += f" fixed={fold['test_fixed']['mIoU']:.4f}"
        print(message)


if __name__ == "__main__":
    main()
