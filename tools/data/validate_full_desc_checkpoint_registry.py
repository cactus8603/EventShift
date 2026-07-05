#!/usr/bin/env python
"""Validate the full-DESC + CoSEC + ACDC checkpoint registry."""

import argparse
import csv
import json
import sys
from pathlib import Path


DEFAULT_REGISTRY = Path(
    "/work/u1621738/ebmv_eccv/eccv_segment/"
    "unified_cosec_acdc/classcover_v1/checkpoints/full_desc_cosec_acdc"
)

MANIFEST_FIELDS = ("model", "stage", "domain", "status", "link", "source")
SELECTED_FIELDS = (
    "model",
    "domain",
    "selected_stage",
    "miou_percent",
    "iteration",
    "selected_link",
    "source_registry",
    "size_bytes",
    "note",
)
SELECTED_DOMAIN_TO_MANIFEST = {
    "cosec_day": "day",
    "cosec_night": "night",
    "acdc_all": "acdc",
    "acdc_night": "acdc_night",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def read_tsv(path, fields):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = [field for field in fields if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
        return list(reader)


def path_exists(path_text):
    return Path(path_text).exists()


def validate_manifest(rows):
    errors = []
    counts = {"rows": len(rows), "symlink": 0, "missing": 0, "selected": 0}
    seen = set()
    for idx, row in enumerate(rows, start=2):
        key = (row["model"], row["stage"], row["domain"], row["link"])
        if key in seen:
            errors.append(f"manifest row {idx}: duplicate key {key}")
        seen.add(key)

        status = row["status"]
        if status == "symlink":
            counts["symlink"] += 1
            if not path_exists(row["link"]):
                errors.append(f"manifest row {idx}: missing link {row['link']}")
            if not path_exists(row["source"]):
                errors.append(f"manifest row {idx}: missing source {row['source']}")
        elif status == "missing":
            counts["missing"] += 1
        else:
            errors.append(f"manifest row {idx}: unsupported status {status}")

        if row["stage"] == "selected":
            counts["selected"] += 1
    return errors, counts


def validate_selected(rows, manifest_rows):
    errors = []
    counts = {"rows": len(rows)}
    manifest_selected = {
        (row["model"], row["domain"]): row
        for row in manifest_rows
        if row["stage"] == "selected" and row["status"] == "symlink"
    }

    for idx, row in enumerate(rows, start=2):
        domain = row["domain"]
        if domain not in SELECTED_DOMAIN_TO_MANIFEST:
            errors.append(f"selected row {idx}: unknown domain {domain}")
            continue

        for field in ("selected_link", "source_registry"):
            if not path_exists(row[field]):
                errors.append(f"selected row {idx}: missing {field} {row[field]}")

        selected_path = Path(row["selected_link"])
        if selected_path.exists():
            actual_size = selected_path.stat().st_size
            try:
                expected_size = int(row["size_bytes"])
            except ValueError:
                errors.append(f"selected row {idx}: invalid size_bytes {row['size_bytes']}")
            else:
                if actual_size != expected_size:
                    errors.append(
                        f"selected row {idx}: size mismatch for {selected_path}: "
                        f"{actual_size} != {expected_size}"
                    )

        manifest_domain = SELECTED_DOMAIN_TO_MANIFEST[domain]
        manifest_key = (row["model"], manifest_domain)
        manifest_row = manifest_selected.get(manifest_key)
        if manifest_row is None:
            errors.append(f"selected row {idx}: missing manifest selected row {manifest_key}")
            continue
        if Path(manifest_row["link"]) != Path(row["selected_link"]):
            errors.append(
                f"selected row {idx}: manifest link mismatch for {manifest_key}: "
                f"{manifest_row['link']} != {row['selected_link']}"
            )
        if Path(manifest_row["source"]) != Path(row["source_registry"]):
            errors.append(
                f"selected row {idx}: manifest source mismatch for {manifest_key}: "
                f"{manifest_row['source']} != {row['source_registry']}"
            )

    if len(manifest_selected) != len(rows):
        errors.append(
            f"selected count mismatch: manifest has {len(manifest_selected)}, "
            f"SELECTED_BEST.tsv has {len(rows)}"
        )
    return errors, counts


def main():
    args = parse_args()
    manifest_path = args.registry / "MANIFEST.tsv"
    selected_path = args.registry / "SELECTED_BEST.tsv"

    manifest_rows = read_tsv(manifest_path, MANIFEST_FIELDS)
    selected_rows = read_tsv(selected_path, SELECTED_FIELDS)

    errors = []
    manifest_errors, manifest_counts = validate_manifest(manifest_rows)
    selected_errors, selected_counts = validate_selected(selected_rows, manifest_rows)
    errors.extend(manifest_errors)
    errors.extend(selected_errors)

    report = {
        "ok": not errors,
        "registry": str(args.registry),
        "manifest": manifest_counts,
        "selected": selected_counts,
        "errors": errors,
    }

    if args.json:
        print(json.dumps(report, indent=2))
    elif errors:
        for error in errors:
            print(error, file=sys.stderr)
    else:
        print(
            "OK: manifest rows={rows}, symlinks={symlink}, selected={selected}, "
            "selected_table_rows={selected_rows}".format(
                rows=manifest_counts["rows"],
                symlink=manifest_counts["symlink"],
                selected=manifest_counts["selected"],
                selected_rows=selected_counts["rows"],
            )
        )

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
