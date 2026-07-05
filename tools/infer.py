#!/usr/bin/env python
"""EventShift inference entry point for backend exporters."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from eventshift.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--test-root", default=os.environ.get("TEST_ROOT"))
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    return parser.parse_args()


def repo_root() -> Path:
    return Path(os.environ.get("EVENTSHIFT_ROOT", Path(__file__).resolve().parents[1])).resolve()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    root = repo_root()
    backend_config = Path(cfg.get("model", {}).get("backend_config", "")).resolve()
    weights = Path(args.weights or cfg.get("checkpoints", {}).get("init_weights", "")).resolve()
    out_dir = Path(args.out_dir or cfg.get("train", {}).get("output_dir", root / "outputs" / "infer")).resolve()
    exporter = root / "tools" / "export" / "export_mask2former_submission.py"
    command = [
        sys.executable,
        str(exporter),
        "--config-file",
        str(backend_config),
        "--weights",
        str(weights),
        "--out-dir",
        str(out_dir),
    ]
    if args.test_root:
        command.extend(["--test-root", str(Path(args.test_root).resolve())])
    if args.opts:
        command.extend(args.opts)
    print(" ".join(shlex.quote(part) for part in command))
    if args.execute:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
