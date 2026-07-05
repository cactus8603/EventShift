#!/usr/bin/env python
"""Derive fold-consensus class routes from multibranch routing diagnostics."""

import argparse
import json
from collections import Counter, OrderedDict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic", required=True, help="diagnose_multibranch_class_routing_fast JSON.")
    parser.add_argument("--min-support", type=int, default=None, help="Minimum fold count for a route.")
    parser.add_argument("--min-fraction", type=float, default=None, help="Minimum fraction of folds for a route.")
    parser.add_argument("--out", default=None)
    parser.add_argument("--print-fixed-route-args", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    with Path(args.diagnostic).open("r", encoding="utf-8") as f:
        report = json.load(f)

    folds = report.get("folds", [])
    if not folds:
        raise ValueError(f"No folds found in {args.diagnostic}")

    fold_count = len(folds)
    if args.min_support is None:
        min_support = fold_count if args.min_fraction is None else int(args.min_fraction * fold_count + 0.999999)
    else:
        min_support = args.min_support
    if min_support < 1 or min_support > fold_count:
        raise ValueError(f"min-support must be within [1, {fold_count}], got {min_support}")

    classes = report.get("classes") or []
    route_counts = OrderedDict()
    route_support = []
    for class_name in classes:
        counter = Counter()
        for fold in folds:
            routes = fold.get("selected_routes", {}) or {}
            branch_name = routes.get(class_name)
            if branch_name:
                counter[branch_name] += 1
        if not counter:
            continue
        branch_name, support = counter.most_common(1)[0]
        row = {
            "class": class_name,
            "branch": branch_name,
            "support": int(support),
            "fold_count": fold_count,
            "fraction": float(support / fold_count),
            "counts": dict(counter),
        }
        route_support.append(row)
        if support >= min_support:
            route_counts[class_name] = branch_name

    summary = {
        "diagnostic": str(Path(args.diagnostic).resolve()),
        "fold_count": fold_count,
        "min_support": min_support,
        "routes": route_counts,
        "route_count": len(route_counts),
        "route_support": route_support,
        "source_summary": report.get("summary", {}),
        "source_overall_selected_mIoU": (report.get("overall_selected") or {}).get("mIoU"),
        "source_overall_fixed_mIoU": (report.get("overall_fixed") or {}).get("mIoU"),
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
            f.write("\n")

    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.print_fixed_route_args:
        print(" ".join(f"--fixed-route {name}={branch}" for name, branch in route_counts.items()))


if __name__ == "__main__":
    main()
