"""RGB-event feature fusion blocks."""

from __future__ import annotations

import torch
from torch import nn


class ReliabilityGatedFusion(nn.Module):
    def __init__(self, channels: int = 64, stat_channels: int = 4):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2 + stat_channels, channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        rgb_feature: torch.Tensor,
        event_feature: torch.Tensor,
        event_stats: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if event_stats is None:
            event_stats = torch.zeros(
                rgb_feature.shape[0],
                4,
                rgb_feature.shape[-2],
                rgb_feature.shape[-1],
                device=rgb_feature.device,
                dtype=rgb_feature.dtype,
            )
        gate = self.gate(torch.cat([rgb_feature, event_feature, event_stats], dim=1))
        return rgb_feature + gate * event_feature

