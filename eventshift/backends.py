"""Backend registry for EventShift command wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExportBackend:
    name: str
    script: str
    config_flag: str
    weights_flag: str
    pythonpath: tuple[str, ...] = ()


EXPORT_BACKENDS = {
    "mask2former": ExportBackend(
        name="mask2former",
        script="tools/export/export_mask2former_submission.py",
        config_flag="--config-file",
        weights_flag="--weights",
        pythonpath=("third_party/Mask2Former", "third_party/detectron2"),
    ),
    "mmseg": ExportBackend(
        name="mmseg",
        script="tools/export/export_mmseg_submission.py",
        config_flag="--config-file",
        weights_flag="--checkpoint",
        pythonpath=("tools", "third_party/mmsegmentation"),
    ),
    "segformer": ExportBackend(
        name="segformer",
        script="tools/export/export_mmseg_submission.py",
        config_flag="--config-file",
        weights_flag="--checkpoint",
        pythonpath=("tools", "third_party/mmsegmentation"),
    ),
}


def get_export_backend(name: str) -> ExportBackend:
    try:
        return EXPORT_BACKENDS[name]
    except KeyError as exc:
        available = ", ".join(sorted(EXPORT_BACKENDS))
        raise SystemExit(f"Unsupported inference backend: {name!r}. Available: {available}") from exc


def backend_pythonpath(root: Path, backend: ExportBackend) -> str:
    return ":".join(str((root / item).resolve()) for item in backend.pythonpath)
