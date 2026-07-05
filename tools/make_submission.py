#!/usr/bin/env python
"""Create a submission zip from a prediction directory."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--zip", required=True, dest="zip_path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_dir = Path(args.pred_dir)
    zip_path = Path(args.zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as zf:
        for path in sorted(pred_dir.rglob("*.png")):
            zf.write(path, path.relative_to(pred_dir).as_posix())
    print(zip_path)


if __name__ == "__main__":
    main()

