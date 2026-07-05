#!/usr/bin/env python
"""EventShift training entry point.

By default this prints the backend command. Pass `--execute` to run it.
"""

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
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=None)
    add_data_path_args(parser)
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


def build_mask2former_command(cfg: dict, args: argparse.Namespace) -> list[str]:
    root = repo_root()
    backend_config = Path(cfg.get("model", {}).get("backend_config", "")).resolve()
    train_script = root / "tools" / "training" / "train_mask2former_cosec.py"
    num_gpus = args.num_gpus or int(cfg.get("train", {}).get("num_gpus", 1))
    command = [
        sys.executable,
        str(train_script),
        "--num-gpus",
        str(num_gpus),
        "--config-file",
        str(backend_config),
    ]
    if args.opts:
        command.extend(args.opts)
    return command


def main() -> None:
    args = parse_args()
    env_overrides = apply_data_path_args(args)
    cfg = load_config(args.config)
    backend = cfg.get("model", {}).get("backend", "native")
    if backend != "mask2former":
        raise SystemExit(f"Unsupported backend for this entry point: {backend!r}")
    command = build_mask2former_command(cfg, args)
    print(format_command(command, env_overrides))
    if args.execute:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
