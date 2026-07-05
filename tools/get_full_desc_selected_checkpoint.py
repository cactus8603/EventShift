#!/usr/bin/env python
"""Query selected full-DESC + CoSEC + ACDC checkpoints."""

import argparse
import csv
import json
import sys
from pathlib import Path


DEFAULT_REGISTRY = Path(
    "/work/u1621738/ebmv_eccv/eccv_segment/"
    "unified_cosec_acdc/classcover_v1/checkpoints/full_desc_cosec_acdc"
)

MODEL_ALIASES = {
    "m2f": "mask2former",
    "mask2former": "mask2former",
    "mask2former_swinl": "mask2former",
    "swinl": "mask2former",
    "seg": "segformer",
    "segformer": "segformer",
    "segformer_b5": "segformer",
    "dino": "maskdino",
    "maskdino": "maskdino",
    "maskdino_swinl": "maskdino",
    "brenet": "brenet",
    "brenet_b2": "brenet",
}

DOMAIN_ALIASES = {
    "day": "cosec_day",
    "cosec_day": "cosec_day",
    "night": "cosec_night",
    "cosec_night": "cosec_night",
    "acdc": "acdc_all",
    "acdc_all": "acdc_all",
    "real": "acdc_all",
    "acdc_night": "acdc_night",
}

FIELDS = (
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--table", type=Path, default=None)
    parser.add_argument("--model", default=None, help="Model name or alias.")
    parser.add_argument("--domain", default=None, help="Domain name or alias.")
    parser.add_argument(
        "--field",
        default="selected_link",
        choices=FIELDS,
        help="Single field to print for a unique query.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON rows.")
    parser.add_argument("--list", action="store_true", help="Print matching TSV rows.")
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Do not verify selected_link/source_registry paths exist.",
    )
    return parser.parse_args()


def normalize(value, aliases, kind):
    if value is None:
        return None
    key = value.strip().lower().replace("-", "_")
    if key not in aliases:
        valid = ", ".join(sorted(aliases))
        raise ValueError(f"Unknown {kind}: {value}. Valid aliases: {valid}")
    return aliases[key]


def load_rows(table_path):
    with table_path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = [field for field in FIELDS if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{table_path} is missing columns: {', '.join(missing)}")
        return list(reader)


def filter_rows(rows, model, domain):
    out = rows
    if model is not None:
        out = [row for row in out if row["model"] == model]
    if domain is not None:
        out = [row for row in out if row["domain"] == domain]
    return out


def check_paths(rows):
    errors = []
    for row in rows:
        for field in ("selected_link", "source_registry"):
            path = Path(row[field])
            if not path.exists():
                errors.append(f"{row['model']} {row['domain']} missing {field}: {path}")
    if errors:
        raise FileNotFoundError("\n".join(errors))


def print_tsv(rows):
    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS, delimiter="\t")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row[field] for field in FIELDS})


def main():
    args = parse_args()
    table_path = args.table or (args.registry / "SELECTED_BEST.tsv")
    model = normalize(args.model, MODEL_ALIASES, "model")
    domain = normalize(args.domain, DOMAIN_ALIASES, "domain")

    rows = filter_rows(load_rows(table_path), model, domain)
    if not rows:
        raise SystemExit("No selected checkpoint matched the query.")

    if not args.no_check:
        check_paths(rows)

    if args.json:
        print(json.dumps(rows if args.list or len(rows) != 1 else rows[0], indent=2))
        return

    if args.list or len(rows) != 1:
        print_tsv(rows)
        return

    print(rows[0][args.field])


if __name__ == "__main__":
    main()
