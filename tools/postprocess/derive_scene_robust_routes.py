#!/usr/bin/env python
"""Derive class routes that agree across multiple routing diagnostics.

This is intentionally conservative: a class route is kept only when every
diagnostic selects the same non-anchor branch and the class delta is positive
enough in every diagnostic.
"""

import argparse
import json
from collections import OrderedDict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diagnostic",
        action="append",
        required=True,
        help="diagnostic_name=/path/to/diagnose_multibranch_class_routing_fast.json",
    )
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--min-routed-pixels", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--print-fixed-route-args", action="store_true")
    return parser.parse_args()


def parse_specs(specs):
    out = OrderedDict()
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid diagnostic spec: {spec}")
        name, path = spec.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Empty diagnostic name in {spec}")
        if name in out:
            raise ValueError(f"Duplicate diagnostic name: {name}")
        out[name] = Path(path)
    return out


def candidate_lookup(report):
    rows = report.get("selected_class_candidates_all") or []
    return {row["class"]: row for row in rows}


def main():
    args = parse_args()
    specs = parse_specs(args.diagnostic)
    reports = OrderedDict()
    lookups = OrderedDict()
    classes = None
    for name, path in specs.items():
        with path.open("r", encoding="utf-8") as f:
            report = json.load(f)
        reports[name] = report
        lookups[name] = candidate_lookup(report)
        report_classes = report.get("classes") or []
        if classes is None:
            classes = report_classes
        elif report_classes != classes:
            raise ValueError(f"Class list differs in {path}")

    robust_routes = OrderedDict()
    rejected = []
    for class_name in classes:
        rows = OrderedDict()
        for diag_name, lookup in lookups.items():
            row = lookup.get(class_name)
            if not row:
                rows[diag_name] = None
            else:
                rows[diag_name] = {
                    "branch": row.get("branch"),
                    "delta_vs_anchor": float(row.get("delta_vs_anchor", 0.0)),
                    "routed_pixels": int(row.get("routed_pixels", 0)),
                    "mIoU": row.get("mIoU"),
                }

        available = [row for row in rows.values() if row is not None]
        reason = None
        if len(available) != len(rows):
            reason = "missing_candidate"
        else:
            branches = {row["branch"] for row in available}
            if len(branches) != 1:
                reason = "branch_disagreement"
            elif any(row["delta_vs_anchor"] <= args.min_delta for row in available):
                reason = "delta_too_small"
            elif any(row["routed_pixels"] < args.min_routed_pixels for row in available):
                reason = "routed_pixels_too_small"

        if reason is None:
            robust_routes[class_name] = available[0]["branch"]
        else:
            rejected.append({"class": class_name, "reason": reason, "diagnostics": rows})

    output = OrderedDict(
        [
            ("diagnostics", {name: str(path) for name, path in specs.items()}),
            ("min_delta", args.min_delta),
            ("min_routed_pixels", args.min_routed_pixels),
            ("routes", robust_routes),
            ("route_count", len(robust_routes)),
            ("rejected", rejected),
            (
                "source_summaries",
                {name: report.get("summary", {}) for name, report in reports.items()},
            ),
        ]
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, sort_keys=True)
        f.write("\n")

    print(json.dumps(output, indent=2, sort_keys=True))
    if args.print_fixed_route_args:
        print(" ".join(f"--fixed-route {name}={branch}" for name, branch in robust_routes.items()))


if __name__ == "__main__":
    main()
