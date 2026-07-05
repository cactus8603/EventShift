#!/usr/bin/env python
"""Cross-validate class-wise routing across multiple saved prediction JSON files."""

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
from evaluate_class_rule_prediction_merge import ConfusionMeter  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--branch",
        action="append",
        required=True,
        help="Branch spec in name=/path/to/sem_seg_predictions.json form.",
    )
    parser.add_argument("--anchor", required=True, help="Default branch to start from.")
    parser.add_argument(
        "--basis",
        required=True,
        help="Branch whose predicted class defines the per-class routing regions.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.0)
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
            raise ValueError(f"Empty branch name in spec: {spec}")
        if name in branches:
            raise ValueError(f"Duplicate branch name: {name}")
        branches[name] = str(Path(path).expanduser())
    return branches


def metric_for_predictions(items, indices, pred_getter):
    meter = ConfusionMeter(num_classes=len(CLASSES))
    for idx in indices:
        item = items[idx]
        meter.update(pred_getter(item), item["label"])
    return meter.metrics()


def route_prediction(item, anchor_name, basis_name, class_to_branch):
    pred = item["preds"][anchor_name].copy()
    basis = item["preds"][basis_name]
    for class_id, branch_name in class_to_branch.items():
        if branch_name == anchor_name:
            continue
        pred[basis == class_id] = item["preds"][branch_name][basis == class_id]
    return pred


def evaluate_route(items, indices, anchor_name, basis_name, class_to_branch):
    pixels = {name: 0 for name in set(class_to_branch.values())}

    def get_pred(item):
        pred = item["preds"][anchor_name].copy()
        basis = item["preds"][basis_name]
        for class_id, branch_name in class_to_branch.items():
            if branch_name == anchor_name:
                continue
            mask = basis == class_id
            pixels[branch_name] = pixels.get(branch_name, 0) + int(mask.sum())
            pred[mask] = item["preds"][branch_name][mask]
        return pred

    metrics = metric_for_predictions(items, indices, get_pred)
    metrics["routed_pixels_by_branch"] = {name: int(count) for name, count in sorted(pixels.items())}
    return metrics


def decode_items(records, branch_indices, branch_names):
    items = []
    missing = []
    for record in records:
        key = prediction_key(record["file_name"])
        rows_by_branch = {}
        for name in branch_names:
            index = branch_indices[name]
            rows = index.get(record["file_name"]) or index.get(key)
            if not rows:
                missing.append({"file_name": record["file_name"], "branch": name})
                continue
            rows_by_branch[name] = rows
        if len(rows_by_branch) != len(branch_names):
            continue
        label = load_label(record).astype(np.uint8, copy=False)
        preds = {
            name: decode_prediction(rows_by_branch[name], label.shape).astype(np.uint8, copy=False)
            for name in branch_names
        }
        items.append({"record": record, "label": label, "preds": preds})
    return items, missing


def select_class_routes(items, train_indices, branch_names, anchor_name, basis_name, min_delta):
    anchor_metrics = metric_for_predictions(
        items,
        train_indices,
        lambda item: item["preds"][anchor_name],
    )
    class_rows = []
    selected = {}
    for class_id, class_name in enumerate(CLASSES):
        best = {
            "class_id": class_id,
            "class": class_name,
            "branch": anchor_name,
            "mIoU": anchor_metrics["mIoU"],
            "delta_vs_anchor": 0.0,
            "routed_pixels": 0,
        }
        for branch_name in branch_names:
            if branch_name == anchor_name:
                continue
            route = {class_id: branch_name}
            metrics = evaluate_route(items, train_indices, anchor_name, basis_name, route)
            delta = metrics["mIoU"] - anchor_metrics["mIoU"]
            routed_pixels = metrics["routed_pixels_by_branch"].get(branch_name, 0)
            row = {
                "class_id": class_id,
                "class": class_name,
                "branch": branch_name,
                "mIoU": metrics["mIoU"],
                "delta_vs_anchor": delta,
                "routed_pixels": routed_pixels,
            }
            if delta > best["delta_vs_anchor"]:
                best = row
        class_rows.append(best)
        if best["branch"] != anchor_name and best["delta_vs_anchor"] > min_delta:
            selected[class_id] = best["branch"]
    class_rows.sort(key=lambda row: row["delta_vs_anchor"], reverse=True)
    return selected, class_rows, anchor_metrics


def main():
    args = parse_args()
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    register_cosec()
    branches = parse_branch_specs(args.branch)
    branch_names = list(branches)
    if args.anchor not in branches:
        raise ValueError(f"Anchor branch not found: {args.anchor}")
    if args.basis not in branches:
        raise ValueError(f"Basis branch not found: {args.basis}")

    branch_indices = {name: load_prediction_index(path) for name, path in branches.items()}
    records = DatasetCatalog.get(args.dataset)
    items, missing = decode_items(records, branch_indices, branch_names)
    if missing:
        raise RuntimeError(f"Missing predictions: {len(missing)}; first={missing[:5]}")

    all_indices = list(range(len(items)))
    branch_metrics = OrderedDict(
        (
            name,
            metric_for_predictions(items, all_indices, lambda item, branch=name: item["preds"][branch]),
        )
        for name in branch_names
    )

    selected_all, class_rows_all, anchor_train_all = select_class_routes(
        items,
        all_indices,
        branch_names,
        args.anchor,
        args.basis,
        args.min_delta,
    )
    overall_selected = evaluate_route(items, all_indices, args.anchor, args.basis, selected_all)

    folds = []
    for fold in range(args.folds):
        test_indices = [idx for idx in all_indices if idx % args.folds == fold]
        train_indices = [idx for idx in all_indices if idx % args.folds != fold]
        selected, class_rows, train_anchor = select_class_routes(
            items,
            train_indices,
            branch_names,
            args.anchor,
            args.basis,
            args.min_delta,
        )
        fold_result = OrderedDict(
            [
                ("fold", fold),
                ("train_count", len(train_indices)),
                ("test_count", len(test_indices)),
                ("selected_routes", {CLASSES[k]: v for k, v in sorted(selected.items())}),
                ("train_anchor_mIoU", train_anchor["mIoU"]),
                ("train_class_candidates", class_rows),
                (
                    "test_anchor",
                    metric_for_predictions(
                        items,
                        test_indices,
                        lambda item: item["preds"][args.anchor],
                    ),
                ),
                (
                    "test_selected",
                    evaluate_route(items, test_indices, args.anchor, args.basis, selected),
                ),
            ]
        )
        for name in branch_names:
            fold_result[f"test_branch_{name}"] = metric_for_predictions(
                items,
                test_indices,
                lambda item, branch=name: item["preds"][branch],
            )
        folds.append(fold_result)

    def avg(key, metric="mIoU"):
        return float(np.mean([fold[key][metric] for fold in folds]))

    summary = {
        "avg_test_anchor_mIoU": avg("test_anchor"),
        "avg_test_selected_mIoU": avg("test_selected"),
        "selected_minus_anchor": avg("test_selected") - avg("test_anchor"),
    }
    for name in branch_names:
        summary[f"avg_test_branch_{name}_mIoU"] = avg(f"test_branch_{name}")

    output = {
        "args": vars(args),
        "branches": branches,
        "classes": list(CLASSES),
        "sample_count": len(items),
        "branch_metrics": branch_metrics,
        "selected_routes_all": {CLASSES[k]: v for k, v in sorted(selected_all.items())},
        "selected_class_candidates_all": class_rows_all,
        "anchor_train_all_mIoU": anchor_train_all["mIoU"],
        "overall_selected": overall_selected,
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
        f"mIoU={overall_selected['mIoU']:.4f}, routes={output['selected_routes_all']}"
    )
    for fold in folds:
        print(
            f"fold {fold['fold']}: anchor={fold['test_anchor']['mIoU']:.4f} "
            f"selected={fold['test_selected']['mIoU']:.4f} routes={fold['selected_routes']}"
        )


if __name__ == "__main__":
    main()
