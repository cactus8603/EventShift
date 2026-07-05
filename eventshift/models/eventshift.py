"""Lightweight EventShift reference model.

The production 04111 runs still use the legacy Mask2Former/MMSeg code. This
module gives the cleaned repo a small native model for smoke tests and future
experiments.
"""

from __future__ import annotations

import torch
from torch import nn

from .decoder import SegmentationDecoder
from .event_encoder import EventEncoder
from .fusion import ReliabilityGatedFusion
from .rgb_encoder import RGBEncoder


class EventShiftNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 19,
        event_channels: int = 10,
        stat_channels: int = 4,
        channels: int = 64,
        use_event: bool = True,
    ):
        super().__init__()
        self.use_event = use_event
        self.rgb_encoder = RGBEncoder(channels=channels)
        self.event_encoder = EventEncoder(event_channels, channels=channels) if use_event else None
        self.fusion = ReliabilityGatedFusion(channels=channels, stat_channels=stat_channels) if use_event else None
        self.decoder = SegmentationDecoder(channels=channels, num_classes=num_classes)

    def forward(
        self,
        image: torch.Tensor,
        event: torch.Tensor | None = None,
        event_stats: torch.Tensor | None = None,
    ) -> torch.Tensor:
        feature = self.rgb_encoder(image)
        if self.use_event and event is not None and self.event_encoder is not None and self.fusion is not None:
            feature = self.fusion(feature, self.event_encoder(event), event_stats)
        return self.decoder(feature)

