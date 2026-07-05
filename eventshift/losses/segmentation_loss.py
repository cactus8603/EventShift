"""Semantic segmentation loss helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = 255) -> torch.Tensor:
    return F.cross_entropy(logits, target.long(), ignore_index=ignore_index)

