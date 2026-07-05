"""Minimal distributed helpers."""

from __future__ import annotations

import os

import torch


def is_dist_avail_and_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def get_rank() -> int:
    if not is_dist_avail_and_initialized():
        return 0
    return torch.distributed.get_rank()


def is_main_process() -> bool:
    return get_rank() == 0


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))

