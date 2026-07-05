#!/usr/bin/env python
"""Search anchor-predicted classes to protect in boundary-gated routing."""

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

from diagnose_multibranch_boundary_gated_routing_fast import (  # noqa: E402
    boundary_for_source,
    decode_records,
    parse_branch_specs,
    parse_fixed_routes,
    semantic_boundary_band,
    valid_label_mask,
)
from diagnose_pair_transition_from_predictions import load_prediction_index  # noqa: E402


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
    parser.add_argument(
        "--boundary-source",
        required=True,
        choices=["anchor", "basis", "union", "intersection"],
    )
    parser.add_argument("--boundary-radius", type=int, required=True)
    parser.add_argument("--max-protect", type=int, default=6)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument(
        "--selection-mode",
        choices=["aggregate", "group_stable"],
        default="aggregate",
        help=(
            "aggregate keeps the original greedy search. group_stable also "
            "requires each selected protect class to be non-harmful for every "
            "scene/group tracked by the dataset records."
        ),
    )
    parser.add_argument(
        "--group-min-delta",
        type=float,
        default=0.0,
        help="Minimum per-group delta vs the current protected matrix when using group_stable.",
    )
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def confusion_from_arrays(label, pred, mask, num_classes):
    keep = mask & valid_label_mask(label)
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


def add_scene_metrics(scene_store, scene, key, matrix):
    scene_store.setdefault(scene, {})[key] = scene_store.setdefault(scene, {}).get(
        key,
        np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64),
    ) + matrix


def build_matrices(decoded, args, routes):
    num_classes = len(CLASSES)
    anchor_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    routed_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    protect_delta = np.zeros((num_classes, num_classes, num_classes), dtype=np.int64)
    protect_stats = [
        {
            "anchor_class": CLASSES[class_id],
            "protected_routed_pixels": 0,
            "protected_changed_pixels": 0,
        }
        for class_id in range(num_classes)
    ]
    totals = {
        "valid_pixels": 0,
        "routed_pixels": 0,
        "changed_vs_anchor": 0,
    }
    scenes = OrderedDict()

    for item in decoded:
        label = item["label"]
        preds = item["preds"]
        anchor_pred = preds[args.anchor]
        basis_pred = preds[args.basis]
        valid = valid_label_mask(label)
        anchor_boundary = semantic_boundary_band(anchor_pred, args.boundary_radius, valid)
        basis_boundary = semantic_boundary_band(basis_pred, args.boundary_radius, valid)
        gate = boundary_for_source(args.boundary_source, anchor_boundary, basis_boundary)
        routed_pred, routed_mask = route_prediction(anchor_pred, basis_pred, preds, routes, gate)
        changed = valid & (routed_pred != anchor_pred)
        scene = scene_from_record(item["record"])

        anchor_item_matrix = confusion_from_arrays(label, anchor_pred, valid, num_classes)
        routed_item_matrix = confusion_from_arrays(label, routed_pred, valid, num_classes)
        anchor_matrix += anchor_item_matrix
        routed_matrix += routed_item_matrix
        add_scene_metrics(scenes, scene, "anchor", anchor_item_matrix)
        add_scene_metrics(scenes, scene, "routed", routed_item_matrix)

        totals["valid_pixels"] += int(valid.sum())
        totals["routed_pixels"] += int((valid & routed_mask).sum())
        totals["changed_vs_anchor"] += int(changed.sum())

        for class_id in range(num_classes):
            protect_region = valid & routed_mask & (anchor_pred == class_id)
            if not protect_region.any():
                continue
            routed_region_matrix = confusion_from_arrays(label, routed_pred, protect_region, num_classes)
            anchor_region_matrix = confusion_from_arrays(label, anchor_pred, protect_region, num_classes)
            delta = anchor_region_matrix - routed_region_matrix
            protect_delta[class_id] += delta
            protect_stats[class_id]["protected_routed_pixels"] += int(protect_region.sum())
            protect_stats[class_id]["protected_changed_pixels"] += int((protect_region & changed).sum())
            add_scene_metrics(scenes, scene, f"delta_{class_id}", delta)

    return anchor_matrix, routed_matrix, protect_delta, protect_stats, totals, scenes


def group_current_metrics(scene_current_matrices):
    return {
        scene: metrics_from_matrix(matrix)
        for scene, matrix in scene_current_matrices.items()
    }


def candidate_group_deltas(scenes, scene_current_matrices, current_group_metrics, class_id):
    deltas = OrderedDict()
    for scene, matrices in scenes.items():
        candidate_matrix = scene_current_matrices[scene] + matrices.get(
            f"delta_{class_id}",
            np.zeros_like(scene_current_matrices[scene]),
        )
        candidate_metrics = metrics_from_matrix(candidate_matrix)
        deltas[scene] = {
            "mIoU": candidate_metrics["mIoU"],
            "delta_vs_current": candidate_metrics["mIoU"] - current_group_metrics[scene]["mIoU"],
        }
    return deltas


def greedy_search(
    routed_matrix,
    protect_delta,
    max_protect,
    min_delta,
    scenes=None,
    selection_mode="aggregate",
    group_min_delta=0.0,
):
    selected = []
    current_matrix = routed_matrix.copy()
    current_metrics = metrics_from_matrix(current_matrix)
    candidates = set(range(len(CLASSES)))
    steps = []
    routed_metrics = metrics_from_matrix(routed_matrix)
    scene_current_matrices = OrderedDict()
    if selection_mode == "group_stable":
        if not scenes:
            raise ValueError("group_stable selection requires scene/group matrices")
        for scene, matrices in scenes.items():
            scene_current_matrices[scene] = matrices["routed"].copy()

    for _ in range(max(0, int(max_protect))):
        best = None
        current_group_metrics = (
            group_current_metrics(scene_current_matrices)
            if selection_mode == "group_stable"
            else None
        )
        for class_id in sorted(candidates):
            candidate_matrix = current_matrix + protect_delta[class_id]
            candidate_metrics = metrics_from_matrix(candidate_matrix)
            delta = candidate_metrics["mIoU"] - current_metrics["mIoU"]
            group_deltas = OrderedDict()
            if selection_mode == "group_stable":
                group_deltas = candidate_group_deltas(
                    scenes,
                    scene_current_matrices,
                    current_group_metrics,
                    class_id,
                )
                if any(
                    row["delta_vs_current"] < float(group_min_delta)
                    for row in group_deltas.values()
                ):
                    continue
            row = {
                "anchor_class": CLASSES[class_id],
                "class_id": class_id,
                "mIoU": candidate_metrics["mIoU"],
                "delta_vs_current": delta,
                "delta_vs_unprotected": candidate_metrics["mIoU"] - routed_metrics["mIoU"],
            }
            if group_deltas:
                row["group_deltas"] = group_deltas
            if best is None or row["delta_vs_current"] > best["delta_vs_current"]:
                best = row
        if best is None or best["delta_vs_current"] <= float(min_delta):
            break
        selected.append(best["class_id"])
        candidates.remove(best["class_id"])
        current_matrix = current_matrix + protect_delta[best["class_id"]]
        current_metrics = metrics_from_matrix(current_matrix)
        if selection_mode == "group_stable":
            for scene, matrices in scenes.items():
                scene_current_matrices[scene] = scene_current_matrices[scene] + matrices.get(
                    f"delta_{best['class_id']}",
                    np.zeros_like(scene_current_matrices[scene]),
                )
        best["metrics"] = current_metrics
        steps.append(best)
    return selected, current_matrix, current_metrics, steps


def evaluate_single_protects(routed_matrix, protect_delta, protect_stats):
    base_miou = metrics_from_matrix(routed_matrix)["mIoU"]
    rows = []
    for class_id, class_name in enumerate(CLASSES):
        matrix = routed_matrix + protect_delta[class_id]
        metrics = metrics_from_matrix(matrix)
        rows.append(
            {
                **protect_stats[class_id],
                "anchor_class": class_name,
                "class_id": class_id,
                "mIoU": metrics["mIoU"],
                "delta_vs_unprotected": metrics["mIoU"] - base_miou,
            }
        )
    rows.sort(key=lambda row: row["delta_vs_unprotected"], reverse=True)
    return rows


def scene_summary(scenes, selected):
    out = OrderedDict()
    for scene, matrices in scenes.items():
        anchor_metrics = metrics_from_matrix(matrices["anchor"])
        routed_metrics = metrics_from_matrix(matrices["routed"])
        selected_matrix = matrices["routed"].copy()
        for class_id in selected:
            selected_matrix += matrices.get(
                f"delta_{class_id}",
                np.zeros_like(selected_matrix),
            )
        selected_metrics = metrics_from_matrix(selected_matrix)
        out[scene] = {
            "anchor": anchor_metrics,
            "unprotected": routed_metrics,
            "protected": selected_metrics,
            "protected_minus_anchor": selected_metrics["mIoU"] - anchor_metrics["mIoU"],
            "protected_minus_unprotected": selected_metrics["mIoU"] - routed_metrics["mIoU"],
        }
    return out


def write_markdown(output, out_path):
    md_path = out_path.with_suffix(".md")
    lines = [
        "# Boundary Protect-Class Search",
        "",
        f"dataset: `{output['args']['dataset']}`",
        f"boundary: `{output['args']['boundary_source']} r{output['args']['boundary_radius']}`",
        f"selection mode: `{output['args']['selection_mode']}`",
        "",
        "| Method | mIoU | Delta vs anchor | Delta vs unprotected |",
        "|---|---:|---:|---:|",
        f"| `anchor` | {output['anchor']['mIoU']:.4f} | 0.0000 |  |",
        f"| `unprotected` | {output['unprotected']['mIoU']:.4f} | "
        f"{output['unprotected']['mIoU'] - output['anchor']['mIoU']:+.4f} | 0.0000 |",
        f"| `greedy_protected` | {output['greedy']['metrics']['mIoU']:.4f} | "
        f"{output['greedy']['metrics']['mIoU'] - output['anchor']['mIoU']:+.4f} | "
        f"{output['greedy']['metrics']['mIoU'] - output['unprotected']['mIoU']:+.4f} |",
        "",
        "Greedy protected classes:",
        "",
    ]
    for row in output["greedy"]["steps"]:
        text = (
            f"- `{row['anchor_class']}`: mIoU {row['mIoU']:.4f}, "
            f"delta vs current {row['delta_vs_current']:+.4f}"
        )
        if row.get("group_deltas"):
            group_text = ", ".join(
                f"{scene} {group_row['delta_vs_current']:+.4f}"
                for scene, group_row in row["group_deltas"].items()
            )
            text += f" ({group_text})"
        lines.append(text)
    lines.extend(
        [
            "",
            "Top single-class protects:",
            "",
            "| Anchor class | mIoU | Delta | Routed px | Changed px |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in output["single_class"][:12]:
        lines.append(
            f"| `{row['anchor_class']}` | {row['mIoU']:.4f} | "
            f"{row['delta_vs_unprotected']:+.4f} | "
            f"{row['protected_routed_pixels']} | {row['protected_changed_pixels']} |"
        )
    if output.get("scenes"):
        lines.extend(["", "Scene summary:", "", "| Scene | Anchor | Unprotected | Protected |", "|---|---:|---:|---:|"])
        for scene, row in output["scenes"].items():
            lines.append(
                f"| `{scene}` | {row['anchor']['mIoU']:.4f} | "
                f"{row['unprotected']['mIoU']:.4f} | {row['protected']['mIoU']:.4f} |"
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
    branch_indices = {name: load_prediction_index(path) for name, path in branches.items()}
    records = list(DatasetCatalog.get(args.dataset))
    decoded, missing = decode_records(records, branch_indices, branch_names)
    if missing:
        raise RuntimeError(f"Missing predictions: {len(missing)}; first={missing[:5]}")

    anchor_matrix, routed_matrix, protect_delta, protect_stats, totals, scenes = build_matrices(
        decoded,
        args,
        routes,
    )
    anchor_metrics = metrics_from_matrix(anchor_matrix)
    routed_metrics = metrics_from_matrix(routed_matrix)
    single_rows = evaluate_single_protects(routed_matrix, protect_delta, protect_stats)
    selected, selected_matrix, selected_metrics, steps = greedy_search(
        routed_matrix,
        protect_delta,
        args.max_protect,
        args.min_delta,
        scenes=scenes,
        selection_mode=args.selection_mode,
        group_min_delta=args.group_min_delta,
    )
    output = {
        "args": vars(args),
        "sample_count": len(decoded),
        "routes": {CLASSES[class_id]: branch for class_id, branch in routes.items()},
        "totals": totals,
        "anchor": anchor_metrics,
        "unprotected": routed_metrics,
        "single_class": single_rows,
        "greedy": {
            "protected_classes": [CLASSES[class_id] for class_id in selected],
            "metrics": selected_metrics,
            "steps": steps,
        },
        "scenes": scene_summary(scenes, selected),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, sort_keys=True)
        f.write("\n")
    md_path = write_markdown(output, out_path)
    print(f"Wrote protect-class search: {out_path}")
    print(f"Wrote summary: {md_path}")
    print(
        "anchor={:.4f} unprotected={:.4f} protected={:.4f} classes={}".format(
            anchor_metrics["mIoU"],
            routed_metrics["mIoU"],
            selected_metrics["mIoU"],
            output["greedy"]["protected_classes"],
        )
    )
    for row in single_rows[:8]:
        print(
            f"  protect {row['anchor_class']}: mIoU={row['mIoU']:.4f}, "
            f"delta={row['delta_vs_unprotected']:+.4f}"
        )


if __name__ == "__main__":
    main()
