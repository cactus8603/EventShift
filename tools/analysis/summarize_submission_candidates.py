#!/usr/bin/env python
"""Summarize generated submission zips, known scores, and diffs to current best."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--zip-dir",
        type=Path,
        default=Path("work_dirs/submissions/submission_zips"),
    )
    parser.add_argument(
        "--composed-dir",
        type=Path,
        default=Path("work_dirs/submissions/composed"),
    )
    parser.add_argument(
        "--diagnostics-dir",
        type=Path,
        default=Path("work_dirs/diagnostics"),
    )
    parser.add_argument(
        "--known-scores",
        type=Path,
        default=Path("docs/known_submission_scores.json"),
    )
    parser.add_argument("--current-best", default="current_best.zip")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--recent", type=int, default=18)
    return parser.parse_args()


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def collect_manifests(composed_dir):
    manifests = {}
    for path in composed_dir.glob("*/manifest.json"):
        data = load_json(path)
        zip_name = path.parent.name + ".zip"
        manifests[zip_name] = data
    return manifests


def diff_priority(data, current_best):
    base = Path(data.get("base", "")).name
    if base == current_best:
        return 0
    if current_best in base:
        return 1
    return 2


def collect_diffs(diagnostics_dir, current_best):
    diffs = {}
    paths = list(diagnostics_dir.glob("submission_diff_*.json"))
    paths.extend((diagnostics_dir / "submission_diffs").glob("*.json"))
    for path in paths:
        data = load_json(path)
        candidate = data.get("candidate")
        if not candidate:
            continue
        name = Path(candidate).name
        by_domain = data.get("by_domain", {})
        if not by_domain and data.get("summary"):
            summary = data["summary"]
            by_domain = {}
            for src_key, dst_key in (("Day", "day"), ("Night", "night"), ("REAL", "real")):
                row = summary.get(src_key)
                if not row:
                    continue
                by_domain[dst_key] = {
                    "changed_pixels": row.get("changed_pixels"),
                    "changed_pixel_rate": row.get("changed_rate"),
                    "changed_files": row.get("changed_files"),
                    "files": row.get("files"),
                    "pixels": row.get("pixels"),
                }
        candidate_diff = {
            "path": str(path),
            "by_domain": by_domain,
            "total": data.get("total", data.get("summary", {}).get("total", {})),
            "priority": diff_priority(data, current_best),
        }
        old = diffs.get(name)
        if old is None or candidate_diff["priority"] < old["priority"]:
            diffs[name] = candidate_diff
    return diffs


def fmt_score(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}" if value > 1 else f"{value:.10g}"
    return str(value)


def fmt_rate(value):
    if value is None:
        return ""
    return f"{100.0 * value:.4f}%"


def domain_rate(diff, domain):
    if not diff:
        return None
    return diff.get("by_domain", {}).get(domain, {}).get("changed_pixel_rate")


def domain_pixels(diff, domain):
    if not diff:
        return None
    return diff.get("by_domain", {}).get(domain, {}).get("changed_pixels")


def risk_label(name, known, diff):
    note = (known.get("note") or "").lower()
    method = (known.get("method") or "").lower()
    day_rate = domain_rate(diff, "day") or 0.0
    night_rate = domain_rate(diff, "night") or 0.0
    real_rate = domain_rate(diff, "real") or 0.0
    if name == "current_best.zip" or "current best" in note:
        return "anchor"
    if known.get("codabench_miou") is not None:
        return "submitted"
    if day_rate == 0 and night_rate == 0 and real_rate > 0:
        return "real-only probe"
    if real_rate == 0 and (day_rate > 0 or night_rate > 0):
        if "greedy" in method or "protect_search" in name:
            return "val-derived route"
        return "day/night probe"
    if day_rate > 0 and night_rate > 0 and real_rate > 0:
        return "combined probe"
    return "generated"


def submission_recommendation(row):
    name = row["name"]
    if row["codabench_miou"] is not None:
        return None

    day_rate = row["day_changed_rate"] or 0.0
    night_rate = row["night_changed_rate"] or 0.0
    real_rate = row["real_changed_rate"] or 0.0
    has_diff = any(
        value is not None
        for value in (
            row["day_changed_rate"],
            row["night_changed_rate"],
            row["real_changed_rate"],
        )
    )
    if not has_diff:
        return (90, "needs current_best diff first")

    if "night_trainlearnedscale64" in name and day_rate == 0 and real_rate == 0:
        return (
            1,
            "train-learned Night scale route; Day/REAL unchanged and Night val is repair-positive",
        )

    if day_rate == 0 and night_rate == 0 and real_rate > 0:
        if "acdc54_754_acdconly_tta5126247681024_real" in name:
            return (3, "new best ACDC54.754 REAL-only TTA ablation")
        if "acdc54_754_acdconly_raw_real" in name:
            return (1, "new best ACDC54.754 clean REAL-only raw proxy probe")
        if "acdc54_7349_tta_real" in name:
            return (4, "previous ACDC54.7349 REAL-only TTA ablation")
        if "acdc_segmentco_eventedge54_70_tta_real" in name:
            return (5, "eventedge REAL-only TTA ablation")
        if "acdc54_7349_real" in name:
            return (2, "previous ACDC54.7349 clean REAL-only raw proxy probe")
        if "acdc_segmentco_eventedge54_70" in name:
            return (3, "clean REAL-only eventedge proxy probe")
        if "acdc_proxy_tta_real" in name:
            return (6, "old ACDC54.58 REAL-only TTA ablation")
        if "coretrans" in name:
            return (7, "very conservative REAL-only transition probe")
        return (8, "clean REAL-only probe")

    if "scene_robust_daynight_acdc54_7349_real" in name:
        return (10, "combined low-footprint scene-robust + newer REAL probe")

    if real_rate == 0 and day_rate <= 0.002 and night_rate <= 0.007 and (
        day_rate > 0 or night_rate > 0
    ):
        return (11, "low-footprint scene/fold-robust Day/Night probe")

    if real_rate == 0 and day_rate > 0 and night_rate == 0 and day_rate <= 0.003:
        return (20, "Day-only low-footprint probe")

    if real_rate == 0 and night_rate > 0 and day_rate == 0 and night_rate <= 0.014:
        return (30, "Night-only conservative probe, but hidden risk remains")

    if row["risk"] == "val-derived route":
        return (70, "validation-derived route; deprioritize after hidden drops")

    return (60, "generated probe; review manually")


def row_for_zip(path, known_scores, manifests, diffs):
    name = path.name
    known = known_scores.get(name, {})
    manifest = manifests.get(name, {})
    diff = diffs.get(name, {})
    stat = path.stat()
    row = {
        "name": name,
        "mtime": stat.st_mtime,
        "size": stat.st_size,
        "codabench_miou": known.get("codabench_miou"),
        "local_day_miou": known.get("local_day_miou"),
        "local_night_miou": known.get("local_night_miou"),
        "method": known.get("method") or manifest.get("description") or "",
        "note": known.get("note") or "",
        "risk": risk_label(name, known, diff),
        "diff_path": diff.get("path", ""),
        "day_changed_rate": domain_rate(diff, "day"),
        "night_changed_rate": domain_rate(diff, "night"),
        "real_changed_rate": domain_rate(diff, "real"),
        "day_changed_pixels": domain_pixels(diff, "day"),
        "night_changed_pixels": domain_pixels(diff, "night"),
        "real_changed_pixels": domain_pixels(diff, "real"),
    }
    recommendation = submission_recommendation(row)
    row["recommendation_priority"] = recommendation[0] if recommendation else None
    row["recommendation"] = recommendation[1] if recommendation else ""
    return row


def sort_key(row):
    score = row["codabench_miou"]
    has_score = score is not None
    return (0 if has_score else 1, -(score or 0.0), -row["mtime"])


def render_markdown(rows, recent, current_best):
    scored = [row for row in rows if row["codabench_miou"] is not None]
    generated = [row for row in rows if row["codabench_miou"] is None]
    generated = sorted(generated, key=lambda row: row["mtime"], reverse=True)[:recent]
    queue = sorted(
        [row for row in rows if row["recommendation_priority"] is not None],
        key=lambda row: (row["recommendation_priority"], -row["mtime"]),
    )[:12]
    out = [
        "# Submission Candidate Summary",
        "",
        f"Anchor: `{current_best}`",
        "",
        "## Recommended Next Queue",
        "",
        "Heuristic: prefer train-learned low-footprint Day/Night routes or "
        "unsubmitted REAL-only probes; deprioritize validation-derived Day/Night "
        "routing because two local-positive routes already dropped on hidden.",
        "",
        "| Priority | Zip | Reason | Day Δpix | Night Δpix | REAL Δpix |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for row in queue:
        out.append(
            "| {priority} | `{name}` | {reason} | {day} | {night} | {real} |".format(
                priority=row["recommendation_priority"],
                name=row["name"],
                reason=row["recommendation"],
                day=fmt_rate(row["day_changed_rate"]),
                night=fmt_rate(row["night_changed_rate"]),
                real=fmt_rate(row["real_changed_rate"]),
            )
        )
    out.extend([
        "",
        "## Submitted / Scored",
        "",
        "| Zip | Codabench mIoU | Local Day | Local Night | Risk | Note |",
        "|---|---:|---:|---:|---|---|",
    ])
    for row in sorted(scored, key=sort_key):
        out.append(
            "| `{name}` | {score} | {day} | {night} | {risk} | {note} |".format(
                name=row["name"],
                score=fmt_score(row["codabench_miou"]),
                day=fmt_score(row["local_day_miou"]),
                night=fmt_score(row["local_night_miou"]),
                risk=row["risk"],
                note=row["note"],
            )
        )
    out.extend(
        [
            "",
            f"## Recent Generated Candidates (Newest {recent})",
            "",
            "| Zip | Risk | Day Δpix | Night Δpix | REAL Δpix | Diff report |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in generated:
        out.append(
            "| `{name}` | {risk} | {day} | {night} | {real} | {diff} |".format(
                name=row["name"],
                risk=row["risk"],
                day=fmt_rate(row["day_changed_rate"]),
                night=fmt_rate(row["night_changed_rate"]),
                real=fmt_rate(row["real_changed_rate"]),
                diff=f"`{row['diff_path']}`" if row["diff_path"] else "",
            )
        )
    out.append("")
    return "\n".join(out)


def main():
    args = parse_args()
    known_scores = load_json(args.known_scores)
    manifests = collect_manifests(args.composed_dir)
    diffs = collect_diffs(args.diagnostics_dir, args.current_best)
    rows = [
        row_for_zip(path, known_scores, manifests, diffs)
        for path in sorted(args.zip_dir.glob("*.zip"))
    ]
    text = render_markdown(rows, args.recent, args.current_best)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
