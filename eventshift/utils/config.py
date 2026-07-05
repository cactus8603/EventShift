"""YAML config loading with environment expansion and lightweight inheritance."""

from __future__ import annotations

import copy
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
_CONFIG_BASE_KEYS = ("_base_", "bases", "extends")


def repo_root() -> Path:
    return Path(os.environ.get("EVENTSHIFT_ROOT", Path(__file__).resolve().parents[2])).resolve()


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


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries without mutating either input."""

    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _read_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise SystemExit(
            "PyYAML is required to read EventShift configs. "
            "Install the environment with `conda env create -f environment.yml` "
            "or run `pip install -r requirements.txt`."
        )
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def _resolve_relative(path: str | Path, base_dir: Path) -> Path:
    path = Path(os.path.expandvars(str(path))).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _default_config_path(kind: str, name: str, model: str | None = None) -> Path:
    config_dir = repo_root() / "configs" / "eventshift"
    if kind == "base":
        return (config_dir / f"{name}.yaml").resolve()
    if kind in {"dataset", "datasets"}:
        return (config_dir / "datasets" / f"{name}.yaml").resolve()
    if kind in {"model", "models"}:
        return (config_dir / "models" / f"{name}.yaml").resolve()
    if kind in {"variant", "variants"}:
        if model:
            candidate = config_dir / "variants" / model / f"{name}.yaml"
            if candidate.exists():
                return candidate.resolve()
        return (config_dir / "variants" / f"{name}.yaml").resolve()
    if kind in {"recipe", "recipes"}:
        return (config_dir / "recipes" / f"{name}.yaml").resolve()
    return (config_dir / f"{kind}" / f"{name}.yaml").resolve()


def _resolve_default(entry: Any, current_dir: Path) -> list[Path]:
    if isinstance(entry, str):
        if "/" in entry or entry.endswith((".yaml", ".yml")):
            return [_resolve_relative(entry, current_dir)]
        return [_default_config_path("base", entry)]
    if not isinstance(entry, dict):
        raise ValueError(f"Unsupported config default entry: {entry!r}")

    paths: list[Path] = []
    model_name = entry.get("model") or entry.get("models")
    for kind, value in entry.items():
        if value is None or kind == "optional":
            continue
        for item in _as_list(value):
            path = _default_config_path(kind, str(item), model=str(model_name) if model_name else None)
            if entry.get("optional") and not path.exists():
                continue
            paths.append(path)
    return paths


def _load_config_tree(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ValueError(f"Circular config inheritance detected: {chain}")
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    data = _read_yaml(path)
    base_paths: list[Path] = []
    for key in _CONFIG_BASE_KEYS:
        base_paths.extend(_resolve_relative(item, path.parent) for item in _as_list(data.pop(key, None)))
    for entry in _as_list(data.pop("defaults", None)):
        base_paths.extend(_resolve_default(entry, path.parent))

    merged: dict[str, Any] = {}
    for base_path in base_paths:
        merged = deep_merge(merged, _load_config_tree(base_path, (*stack, path)))
    return deep_merge(merged, data)


_DATA_PATH_ARGS = (
    ("cosec_root", "COSEC_ROOT", "--cosec-root", "Root directory for the CoSEC dataset."),
    ("brenet_root", "BRENET_ROOT", "--brenet-root", "Optional fallback root for legacy manifests with relative event paths."),
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
    return expand_env(_load_config_tree(Path(path)))


def compose_configs(paths: list[str | Path]) -> dict:
    merged: dict[str, Any] = {}
    for path in paths:
        merged = deep_merge(merged, _load_config_tree(Path(path)))
    return expand_env(merged)


def eventshift_config_path(
    *,
    config: str | Path | None = None,
    model: str | None = None,
    variant: str | None = None,
) -> list[Path]:
    """Resolve config/model/variant CLI selections to config files."""

    paths: list[Path] = []
    if config:
        paths.append(_resolve_relative(config, repo_root()))
    elif model or variant:
        paths.append(_default_config_path("base", "base"))

    if model:
        paths.append(_default_config_path("model", model))
    if variant:
        if not model:
            raise ValueError("--variant requires --model when --config is not a full recipe/config")
        paths.append(_default_config_path("variant", variant, model=model))
    return paths
