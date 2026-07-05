"""YAML config loading with simple environment-variable expansion."""

from __future__ import annotations

import os
import re
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

