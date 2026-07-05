"""Checkpoint load/save wrappers."""

from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(path: str | Path, model, optimizer=None, **metadata) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(), "metadata": metadata}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(path: str | Path, model=None, optimizer=None, map_location="cpu") -> dict:
    payload = torch.load(path, map_location=map_location)
    if model is not None:
        model.load_state_dict(payload.get("model", payload), strict=False)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload

