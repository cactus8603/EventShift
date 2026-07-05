"""Binary edge supervision losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    target = target.float()
    inter = (prob * target).sum(dim=(-2, -1))
    denom = prob.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    return 1.0 - ((2.0 * inter + eps) / (denom + eps)).mean()


def binary_edge_loss(logits: torch.Tensor, target: torch.Tensor, bce_weight: float = 1.0, dice_weight: float = 1.0):
    target = target.float()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    return bce_weight * bce + dice_weight * dice_loss(logits, target)

