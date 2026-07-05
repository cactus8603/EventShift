#!/usr/bin/env python
"""EventShift inference entry point for backend exporters."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from eventshift.utils.config import add_data_path_args, apply_data_path_args, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    add_data_path_args(parser, include_test_root=False)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--test-root", default=None, help="Root directory for CoSEC test sequences.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--execute", action="store_true")
    args, opts = parser.parse_known_args()
    if opts and opts[0] == "--":
        opts = opts[1:]
    args.opts = opts
    return args


def repo_root() -> Path:
    return Path(os.environ.get("EVENTSHIFT_ROOT", Path(__file__).resolve().parents[1])).resolve()


def format_command(command: list[str], env_overrides: dict[str, str]) -> str:
    parts = [f"{key}={shlex.quote(value)}" for key, value in sorted(env_overrides.items())]
    parts.extend(shlex.quote(part) for part in command)
    return " ".join(parts)


def main() -> None:
    args = parse_args()
    env_overrides = apply_data_path_args(args)
    cfg = load_config(args.config)
    root = repo_root()
    backend_config = Path(cfg.get("model", {}).get("backend_config", "")).resolve()
    weights = Path(args.weights or cfg.get("checkpoints", {}).get("init_weights", "")).resolve()
    out_dir = Path(args.out_dir or cfg.get("train", {}).get("output_dir", root / "outputs" / "infer")).resolve()
    test_root = args.test_root or os.environ.get("TEST_ROOT")
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
    if test_root:
        command.extend(["--test-root", str(Path(test_root).resolve())])
    if args.opts:
        command.extend(args.opts)
    print(format_command(command, env_overrides))
    if args.execute:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
