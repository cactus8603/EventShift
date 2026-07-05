"""YAML config loading with simple environment-variable expansion."""

from __future__ import annotations

import os
import re
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - environment setup guard
    yaml = None


_ENV_PATTERN = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")


def expand_env(value: Any) -> Any:
    if isinstance(value, str):
        def repl(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            return os.environ.get(name, default or "")

        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, list):
        return [expand_env(item) for item in value]
    if isinstance(value, tuple):
        return tuple(expand_env(item) for item in value)
    if isinstance(value, dict):
        return {key: expand_env(item) for key, item in value.items()}
    return value


_DATA_PATH_ARGS = (
    ("cosec_root", "COSEC_ROOT", "--cosec-root", "Root directory for the CoSEC dataset."),
    ("brenet_root", "BRENET_ROOT", "--brenet-root", "Root directory for BRENet and CoSEC event assets."),
    ("dsec_root", "DSEC_ROOT", "--dsec-root", "Root directory for the DSEC dataset."),
    ("acdc_root", "ACDC_ROOT", "--acdc-root", "Root directory for the ACDC dataset."),
    ("cosec_manifest", "EVENTSHIFT_COSEC_MANIFEST", "--cosec-manifest", "Path to the CoSEC event manifest JSON."),
    ("test_root", "TEST_ROOT", "--test-root", "Root directory for CoSEC test sequences."),
)


def add_data_path_args(parser: ArgumentParser, include_test_root: bool = True) -> None:
    """Add common dataset path overrides to an argparse parser."""

    for attr, _env_name, flag, help_text in _DATA_PATH_ARGS:
        if attr == "test_root" and not include_test_root:
            continue
        parser.add_argument(flag, dest=attr, default=None, help=help_text)


def apply_data_path_args(args: Any, include_test_root: bool = True) -> dict[str, str]:
    """Apply dataset path args as environment overrides before config loading."""

    overrides = {}
    for attr, env_name, _flag, _help_text in _DATA_PATH_ARGS:
        if attr == "test_root" and not include_test_root:
            continue
        value = getattr(args, attr, None)
        if value is None or value == "":
            continue
        path = str(Path(value).expanduser())
        os.environ[env_name] = path
        overrides[env_name] = path
    return overrides


def load_config(path: str | Path) -> dict:
    if yaml is None:
        raise SystemExit(
            "PyYAML is required to read EventShift configs. "
            "Install the environment with `conda env create -f environment.yml` "
            "or run `pip install -r requirements.txt`."
        )
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return expand_env(data)

