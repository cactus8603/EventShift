#!/usr/bin/env python
"""Summarize scale accept/reject calibration diagnostics."""

import argparse
import json
import math
from pathlib import Path


def valid_number(value):
    return isinstance(value, (int, float)) and not math.isnan(float(value))


def fmt(value, digits=4):
    if not valid_number(value):
        return ""
    return f"{float(value):.{digits}f}"


def fmt_pct(value):
    if not valid_number(value):
        return ""
    return f"{100.0 * float(value):.3f}%"


def read_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def metrics_path(path):
    path = Path(path)
    return path if path.name == "metrics.json" else path / "metrics.json"


def method_rows(methods):
    if isinstance(methods, dict):
        for name, values in methods.items():
            row = dict(values)
            row.setdefault("method", name)
            yield row
    else:
        yield from methods


def method_by_name(methods, name):
    for row in method_rows(methods):
        if row.get("method") == name:
            return row
    return None


def threshold_methods(methods):
    rows = [row for row in method_rows(methods) if str(row.get("method", "")).startswith("accept_threshold_")]
    return [row for row in rows if valid_number(row.get("mIoU"))]


def best_threshold(methods):
    rows = threshold_methods(methods)
    if not rows:
        return None
    return max(rows, key=lambda row: float(row["mIoU"]))


def class_delta_rows(anchor, best, top_k):
    anchor_iou = anchor.get("class_iou", {}) if anchor else {}
    best_iou = best.get("class_iou", {}) if best else {}
    rows = []
    for cls, value in best_iou.items():
        anchor_value = anchor_iou.get(cls)
        if valid_number(value) and valid_number(anchor_value):
            rows.append((cls, float(anchor_value), float(value), float(value) - float(anchor_value)))
    rows.sort(key=lambda item: item[3], reverse=True)
    gains = [row for row in rows if row[3] > 0]
    losses = [row for row in sorted(rows, key=lambda item: item[3]) if row[3] < 0]
    return gains[:top_k], losses[:top_k]


def decision(best, anchor, candidate_all, min_gain, max_changed):
    if not best or not anchor:
        return "missing"
    gain = float(best.get("mIoU", float("nan"))) - float(anchor.get("mIoU", float("nan")))
    changed = float(best.get("changed_rate", 0.0))
    candidate_gain = None
    if candidate_all and valid_number(candidate_all.get("mIoU")) and valid_number(anchor.get("mIoU")):
        candidate_gain = float(candidate_all["mIoU"]) - float(anchor["mIoU"])
    if gain >= min_gain and changed <= max_changed:
        return "promote"
    if gain > 0.0:
        return "watch"
    if candidate_gain is not None and candidate_gain > 0.0:
        return "features-too-weak"
    return "stop"


def summarize_run(path, args):
    data = read_json(metrics_path(path))
    rows = []
    for dataset in data.get("datasets", []):
        methods = dataset.get("methods", {})
        anchor = method_by_name(methods, "anchor_tta")
        candidate_all = method_by_name(methods, "candidate_all_allowed")
        best = best_threshold(methods)
        if not anchor:
            continue
        anchor_miou = anchor.get("mIoU")
        best_miou = best.get("mIoU") if best else None
        candidate_miou = candidate_all.get("mIoU") if candidate_all else None
        rows.append(
            {
                "run": Path(path).name if Path(path).name != "metrics.json" else Path(path).parent.name,
                "path": str(metrics_path(path)),
                "dataset": dataset.get("dataset"),
                "sample_count": dataset.get("sample_count"),
                "anchor": anchor,
                "candidate_all": candidate_all,
                "best": best,
                "anchor_miou": anchor_miou,
                "candidate_gain": (
                    float(candidate_miou) - float(anchor_miou)
                    if valid_number(candidate_miou) and valid_number(anchor_miou)
                    else None
                ),
                "best_gain": (
                    float(best_miou) - float(anchor_miou)
                    if valid_number(best_miou) and valid_number(anchor_miou)
                    else None
                ),
                "decision": decision(best, anchor, candidate_all, args.min_gain, args.max_changed_rate),
            }
        )
    return rows


def print_table(rows):
    print(
        "| Decision | Run | Dataset | Samples | Anchor | Candidate-All | Cand Gain | "
        "Best Threshold | Best mIoU | Gain | Changed | Accepted | Net Repair |"
    )
    print("|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|")
    for row in rows:
        best = row["best"] or {}
        candidate = row["candidate_all"] or {}
        print(
            f"| {row['decision']} | `{row['run']}` | {row['dataset']} | {row['sample_count']} | "
            f"{fmt(row['anchor_miou'])} | {fmt(candidate.get('mIoU'))} | {fmt(row['candidate_gain'])} | "
            f"`{best.get('method', '')}` | {fmt(best.get('mIoU'))} | {fmt(row['best_gain'])} | "
            f"{fmt_pct(best.get('changed_rate'))} | {fmt_pct(best.get('accepted_rate'))} | "
            f"{best.get('net_repaired', '')} |"
        )


def print_class_tables(rows, top_k):
    if top_k <= 0:
        return
    for row in rows:
        best = row["best"]
        if not best:
            continue
        gains, losses = class_delta_rows(row["anchor"], best, top_k)
        print()
        print(f"## {row['run']} / {row['dataset']} / {best.get('method')}")
        print()
        print("| Top Gain Class | Anchor IoU | Best IoU | Delta |")
        print("|---|---:|---:|---:|")
        for cls, anchor_value, best_value, delta in gains:
            print(f"| {cls} | {fmt(anchor_value, 2)} | {fmt(best_value, 2)} | {fmt(delta, 2)} |")
        print()
        print("| Top Loss Class | Anchor IoU | Best IoU | Delta |")
        print("|---|---:|---:|---:|")
        for cls, anchor_value, best_value, delta in losses:
            print(f"| {cls} | {fmt(anchor_value, 2)} | {fmt(best_value, 2)} | {fmt(delta, 2)} |")


def collect_rows(paths, args):
    rows = []
    for path in paths:
        try:
            rows.extend(summarize_run(path, args))
        except FileNotFoundError as exc:
            if args.ignore_missing:
                print(f"[warn] missing metrics: {exc}")
                continue
            raise
    rows.sort(
        key=lambda row: (
            {"promote": 0, "watch": 1, "features-too-weak": 2, "stop": 3, "missing": 4}.get(row["decision"], 9),
            -(float(row["best_gain"]) if valid_number(row["best_gain"]) else -999.0),
        )
    )
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="Scale-calib output dirs or metrics.json files.")
    parser.add_argument("--min-gain", type=float, default=0.2)
    parser.add_argument("--max-changed-rate", type=float, default=0.01)
    parser.add_argument("--top-k-classes", type=int, default=5)
    parser.add_argument("--ignore-missing", action="store_true")
    parser.add_argument("--out", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = collect_rows(args.paths, args)
    lines = []

    import io
    import contextlib

    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        print("# Scale-Calib Result Summary")
        print()
        print_table(rows)
        print_class_tables(rows, args.top_k_classes)
    text = stream.getvalue()
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
