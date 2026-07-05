"""Segmentation decoder."""

from __future__ import annotations

import torch
from torch import nn


class SegmentationDecoder(nn.Module):
    def __init__(self, channels: int = 64, num_classes: int = 19):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

