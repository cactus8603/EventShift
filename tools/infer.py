#!/usr/bin/env python
"""EventShift inference entry point for backend exporters."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from eventshift.backends import backend_pythonpath, get_export_backend
from eventshift.utils.config import (
    add_data_path_args,
    apply_data_path_args,
    compose_configs,
    eventshift_config_path,
)


DEFAULT_CONFIG = "configs/eventshift/cosec_eventshift.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="EventShift YAML config. Legacy flat configs remain supported.")
    parser.add_argument("--model", default=None, help="Model family under configs/eventshift/models, e.g. mask2former or segformer.")
    parser.add_argument("--variant", default=None, help="Variant under configs/eventshift/variants/<model>.")
    add_data_path_args(parser, include_test_root=True)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--progress-desc", default=None)
    parser.add_argument("--execute", action="store_true")
    args, opts = parser.parse_known_args()
    if opts and opts[0] == "--":
        opts = opts[1:]
    args.opts = opts
    return args


def repo_root() -> Path:
    return Path(os.environ.get("EVENTSHIFT_ROOT", Path(__file__).resolve().parents[1])).resolve()


def as_path(value: str | Path, root: Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def merged_pythonpath(root: Path, backend_name: str) -> str:
    backend = get_export_backend(backend_name)
    parts = [str(root), backend_pythonpath(root, backend)]
    if os.environ.get("PYTHONPATH"):
        parts.append(os.environ["PYTHONPATH"])
    return ":".join(part for part in parts if part)


def format_command(command: list[str], env_overrides: dict[str, str]) -> str:
    parts = [f"{key}={shlex.quote(value)}" for key, value in sorted(env_overrides.items())]
    parts.extend(shlex.quote(part) for part in command)
    return " ".join(parts)


def load_selected_config(args: argparse.Namespace) -> tuple[dict[str, Any], list[Path]]:
    config = args.config
    if not config and not args.model and not args.variant:
        config = DEFAULT_CONFIG
    paths = eventshift_config_path(config=config, model=args.model, variant=args.variant)
    return compose_configs(paths), paths


def build_export_command(cfg: dict[str, Any], args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    root = repo_root()
    model_cfg = cfg.get("model", {})
    inference_cfg = cfg.get("inference", {})
    data_cfg = cfg.get("data", {})
    checkpoint_cfg = cfg.get("checkpoints", {})

    backend_name = model_cfg.get("exporter") or model_cfg.get("backend")
    if not backend_name:
        raise SystemExit("Config is missing model.exporter/model.backend")
    backend = get_export_backend(str(backend_name))

    backend_config = model_cfg.get("backend_config")
    if not backend_config:
        raise SystemExit("Config is missing model.backend_config")
    weights = args.weights or checkpoint_cfg.get("init_weights") or model_cfg.get("weights")
    if not weights:
        raise SystemExit("Weights are required. Pass --weights or set checkpoints.init_weights in the config.")

    out_dir = args.out_dir or inference_cfg.get("output_dir") or cfg.get("train", {}).get("output_dir")
    if not out_dir:
        out_dir = root / "outputs" / "infer"
    test_root = args.test_root or data_cfg.get("test_root") or os.environ.get("TEST_ROOT")
    if not test_root:
        raise SystemExit("--test-root is required for submission inference")

    command = [
        sys.executable,
        str((root / backend.script).resolve()),
        backend.config_flag,
        str(as_path(backend_config, root)),
        backend.weights_flag,
        str(as_path(weights, root)),
        "--test-root",
        str(as_path(test_root, root)),
        "--out-dir",
        str(as_path(out_dir, root)),
    ]

    device = args.device or cfg.get("runtime", {}).get("device")
    if device:
        command.extend(["--device", str(device)])

    sequences = args.sequences or inference_cfg.get("sequences")
    if sequences:
        command.extend(["--sequences", *as_list(sequences)])

    progress_desc = args.progress_desc or inference_cfg.get("progress_desc") or model_cfg.get("name")
    if progress_desc:
        command.extend(["--progress-desc", str(progress_desc)])

    command.extend(as_list(inference_cfg.get("extra_args")))
    if args.opts:
        command.extend(args.opts)

    env = {"PYTHONPATH": merged_pythonpath(root, backend.name)}
    return command, env


def main() -> None:
    args = parse_args()
    data_env = apply_data_path_args(args)
    cfg, paths = load_selected_config(args)
    command, backend_env = build_export_command(cfg, args)
    display_env = {**backend_env, **data_env}
    print("config_files=" + ",".join(str(path) for path in paths))
    print(format_command(command, display_env))
    if args.execute:
        env = os.environ.copy()
        env.update(backend_env)
        subprocess.run(command, check=True, env=env)


if __name__ == "__main__":
    main()
