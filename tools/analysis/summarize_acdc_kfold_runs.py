#!/usr/bin/env python
import argparse
import json
from pathlib import Path


def _read_records(metrics_path):
    records = []
    if not metrics_path.exists():
        return records
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _best(records, key):
    best_record = None
    best_value = None
    for record in records:
        value = record.get(key)
        if value is None:
            continue
        value = float(value)
        if best_value is None or value > best_value:
            best_value = value
            best_record = record
    if best_record is None:
        return None
    return {
        "value": best_value,
        "iteration": best_record.get("iteration"),
    }


def summarize_run(run_dir):
    run_dir = Path(run_dir)
    records = _read_records(run_dir / "metrics.json")
    dataset_keys = sorted(
        {
            key.rsplit("/sem_seg/mIoU", 1)[0]
            for record in records
            for key in record
            if key.endswith("/sem_seg/mIoU")
        }
    )
    best_by_dataset = {
        dataset: _best(records, f"{dataset}/sem_seg/mIoU")
        for dataset in dataset_keys
    }
    best_scalars = {
        key: _best(records, key)
        for key in sorted({key for record in records for key in record if key.startswith("best_") and key.endswith("_mIoU")})
    }
    return {
        "run_dir": str(run_dir),
        "records": len(records),
        "best_by_dataset": best_by_dataset,
        "best_scalars": best_scalars,
        "checkpoints": sorted(path.name for path in run_dir.glob("best_model*.pth")),
    }


def _format_best(item):
    if not item:
        return "-"
    return f"{item['value']:.4f}@{item['iteration']}"


def print_human(summaries):
    print("| Run | Dataset best mIoU | Saved best scalars | Best checkpoints |")
    print("|---|---|---|---|")
    for summary in summaries:
        dataset_text = "<br>".join(
            f"{dataset}: {_format_best(item)}"
            for dataset, item in summary["best_by_dataset"].items()
        ) or "-"
        scalar_text = "<br>".join(
            f"{key}: {_format_best(item)}"
            for key, item in summary["best_scalars"].items()
        ) or "-"
        ckpt_text = "<br>".join(summary["checkpoints"]) or "-"
        print(f"| `{summary['run_dir']}` | {dataset_text} | {scalar_text} | {ckpt_text} |")


def main():
    parser = argparse.ArgumentParser(description="Summarize ACDC k-fold run metrics.")
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    summaries = [summarize_run(run_dir) for run_dir in args.run_dirs]
    if args.json:
        text = json.dumps(summaries, indent=2, sort_keys=True) + "\n"
    else:
        from io import StringIO

        buffer = StringIO()
        import sys

        old_stdout = sys.stdout
        try:
            sys.stdout = buffer
            print_human(summaries)
        finally:
            sys.stdout = old_stdout
        text = buffer.getvalue()

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
