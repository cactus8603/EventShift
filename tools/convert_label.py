#!/usr/bin/env python
"""Copy or remap label masks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--identity", action="store_true", help="Keep labels unchanged.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.src)
    dst = Path(args.dst)
    for path in sorted(src.rglob("*.png")):
        rel = path.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        arr = np.array(Image.open(path))
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        Image.fromarray(arr.astype(np.uint8, copy=False)).save(out)


if __name__ == "__main__":
    main()

