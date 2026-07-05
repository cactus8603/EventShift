# Copyright (c) Facebook, Inc. and its affiliates.
import math
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.events import get_event_storage
from detectron2.utils.memory import retry_if_cuda_oom

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher


def _group_count(channels):
    for group_count in (32, 16, 8, 4, 2):
        if channels % group_count == 0:
            return group_count
    return 1


def _sum_losses(losses, keys):
    total = None
    found = False
    for key in keys:
        value = losses.get(key)
        if value is None:
            continue
        found = True
        total = value if total is None else total + value
    return total, found


class EventStageFusion(nn.Module):
    def __init__(
        self,
        event_channels,
        stat_channels,
        feat_channels,
        hidden_dim,
        init_alpha,
        gate_bias,
        use_reliability_prior,
        reliability_density_power,
        reliability_temporal_power,
        reliability_polarity_power,
        reliability_floor,
        reliability_gain,
    ):
        super().__init__()
        hidden_dim = min(int(hidden_dim), int(feat_channels))
        self.event_encoder = nn.Sequential(
            nn.Conv2d(event_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(hidden_dim), hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(hidden_dim), hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, feat_channels, kernel_size=1),
        )
        self.gate = nn.Conv2d(feat_channels * 2 + stat_channels, 1, kernel_size=1)
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))
        self.use_reliability_prior = bool(use_reliability_prior)
        self.reliability_density_power = float(reliability_density_power)
        self.reliability_temporal_power = float(reliability_temporal_power)
        self.reliability_polarity_power = float(reliability_polarity_power)
        self.reliability_floor = float(reliability_floor)
        self.reliability_gain = float(reliability_gain)
        nn.init.zeros_(self.event_encoder[-1].weight)
        nn.init.zeros_(self.event_encoder[-1].bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, float(gate_bias))

    def _event_reliability(self, stage_stats):
        density = stage_stats[:, 0:1].clamp_min(0.0)
        temporal = stage_stats[:, 1:2].clamp(0.0, 1.0)
        polarity = stage_stats[:, 2:3].clamp(0.0, 1.0)
        support = stage_stats[:, 3:4].clamp(0.0, 1.0)
        density_max = density.flatten(2).amax(dim=2).view(-1, 1, 1, 1).clamp_min(1e-6)
        density = (density / density_max).clamp(0.0, 1.0)

        reliability = support
        if self.reliability_density_power > 0:
            reliability = reliability * density.clamp_min(1e-4).pow(self.reliability_density_power)
        if self.reliability_temporal_power > 0:
            reliability = reliability * temporal.clamp_min(1e-4).pow(self.reliability_temporal_power)
        if self.reliability_polarity_power > 0:
            reliability = reliability * polarity.clamp_min(1e-4).pow(self.reliability_polarity_power)
        return reliability.clamp(0.0, 1.0), support

    def forward(self, feature, event, event_stats):
        size = feature.shape[-2:]
        event = F.interpolate(event, size=size, mode="bilinear", align_corners=False)
        smooth_stats = F.interpolate(event_stats[:, :3], size=size, mode="bilinear", align_corners=False)
        support = F.interpolate(event_stats[:, 3:4], size=size, mode="nearest")
        stage_stats = torch.cat([smooth_stats, support], dim=1)
        reliability, support = self._event_reliability(stage_stats)
        gate_prior = support.to(dtype=feature.dtype)
        event_input = event.to(dtype=feature.dtype)
        if self.use_reliability_prior:
            floor = min(max(self.reliability_floor, 0.0), 1.0)
            gate_prior = support.to(dtype=feature.dtype) * (
                floor + (1.0 - floor) * reliability.to(dtype=feature.dtype)
            )
            event_scale = gate_prior * (1.0 + self.reliability_gain * reliability.to(dtype=feature.dtype))
            event_input = event_input * event_scale
        event_delta = self.event_encoder(event_input)
        usefulness = torch.sigmoid(self.gate(torch.cat([feature, event_delta, stage_stats.to(dtype=feature.dtype)], dim=1)))
        final_gate = usefulness * gate_prior
        aux = {
            "final_gate": final_gate,
            "reliability": reliability.to(dtype=feature.dtype),
            "support": support.to(dtype=feature.dtype),
            "invalid": (support * (1.0 - reliability)).to(dtype=feature.dtype),
        }
        return feature + self.alpha.clamp(0.0, 1.0) * final_gate * event_delta, aux


class EventUsefulnessFusion(nn.Module):
    def __init__(
        self,
        output_shapes,
        *,
        event_channels,
        stat_channels,
        stages,
        hidden_dim,
        init_alpha,
        gate_bias,
        use_reliability_prior,
        reliability_density_power,
        reliability_temporal_power,
        reliability_polarity_power,
        reliability_floor,
        reliability_gain,
        gate_sparsity_weight,
        gate_invalid_weight,
        log_gate_stats,
    ):
        super().__init__()
        self.event_channels = int(event_channels)
        self.stat_channels = int(stat_channels)
        self.stages = tuple(stages)
        self.gate_sparsity_weight = float(gate_sparsity_weight)
        self.gate_invalid_weight = float(gate_invalid_weight)
        self.log_gate_stats = bool(log_gate_stats)
        self.stage_fusions = nn.ModuleDict(
            {
                stage: EventStageFusion(
                    self.event_channels,
                    self.stat_channels,
                    output_shapes[stage].channels,
                    hidden_dim,
                    init_alpha,
                    gate_bias,
                    use_reliability_prior,
                    reliability_density_power,
                    reliability_temporal_power,
                    reliability_polarity_power,
                    reliability_floor,
                    reliability_gain,
                )
                for stage in self.stages
            }
        )
        self.last_gate_stats = {}
        self.aux_losses = {}

    def forward(self, features, event, event_stats):
        if event is None or event_stats is None:
            self.aux_losses = {}
            return features
        features = dict(features)
        gate_stats = {}
        sparse_losses = []
        invalid_losses = []
        for stage, fusion in self.stage_fusions.items():
            if stage not in features:
                continue
            features[stage], aux = fusion(features[stage], event, event_stats)
            final_gate = aux["final_gate"]
            reliability = aux["reliability"]
            support = aux["support"]
            invalid = aux["invalid"]
            gate_stats[stage] = {
                "mean": final_gate.detach().mean(),
                "max": final_gate.detach().max(),
                "support_mean": support.detach().mean(),
                "reliability_mean": reliability.detach().mean(),
                "invalid_mean": invalid.detach().mean(),
                "alpha": fusion.alpha.detach().clamp(0.0, 1.0),
            }
            if self.gate_sparsity_weight > 0:
                sparse_losses.append(final_gate.mean())
            if self.gate_invalid_weight > 0:
                invalid_losses.append((final_gate * invalid).sum() / invalid.sum().clamp_min(1.0))
        self.last_gate_stats = gate_stats
        self.aux_losses = {}
        if sparse_losses:
            self.aux_losses["loss_event_gate_sparse"] = (
                torch.stack(sparse_losses).mean() * self.gate_sparsity_weight
            )
        if invalid_losses:
            self.aux_losses["loss_event_gate_invalid"] = (
                torch.stack(invalid_losses).mean() * self.gate_invalid_weight
            )
        return features

    def extra_losses(self):
        return dict(self.aux_losses)

    def put_gate_stats(self):
        if not self.log_gate_stats or not self.last_gate_stats:
            return
        try:
            storage = get_event_storage()
        except AssertionError:
            return
        for stage, stats in self.last_gate_stats.items():
            for name, value in stats.items():
                storage.put_scalar(f"event_fusion/{stage}_{name}", value.item(), smoothing_hint=False)


class EventEdgeSemanticHead(nn.Module):
    def __init__(
        self,
        output_shapes,
        *,
        in_channels,
        hidden_dim,
        use_rgb_feature,
        rgb_feature,
        detach_rgb_feature,
        output_stride,
        primary_boundary_radius,
        bce_weight,
        dice_weight,
        pos_weight_max,
        log_edge_stats,
        train_only_edge,
        class_aware,
        num_classes,
        class_boundary_radius,
        class_bce_weight,
        class_pos_weight_max,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_dim = int(hidden_dim)
        self.use_rgb_feature = bool(use_rgb_feature)
        self.rgb_feature = str(rgb_feature)
        self.detach_rgb_feature = bool(detach_rgb_feature)
        self.output_stride = max(1, int(output_stride))
        self.primary_boundary_radius = int(primary_boundary_radius)
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.pos_weight_max = float(pos_weight_max)
        self.log_edge_stats = bool(log_edge_stats)
        self.train_only_edge = bool(train_only_edge)
        self.class_aware = bool(class_aware)
        self.num_classes = int(num_classes)
        self.class_boundary_radius = int(class_boundary_radius)
        self.class_bce_weight = float(class_bce_weight)
        self.class_pos_weight_max = float(class_pos_weight_max)

        self.event_encoder = nn.Sequential(
            nn.Conv2d(self.in_channels, self.hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(self.hidden_dim), self.hidden_dim),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(self.hidden_dim), self.hidden_dim),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(self.hidden_dim), self.hidden_dim),
            nn.GELU(),
        )
        if self.use_rgb_feature:
            if self.rgb_feature not in output_shapes:
                raise ValueError(f"Unknown RGB feature for event edge head: {self.rgb_feature}")
            self.rgb_proj = nn.Sequential(
                nn.Conv2d(output_shapes[self.rgb_feature].channels, self.hidden_dim, kernel_size=1),
                nn.GroupNorm(_group_count(self.hidden_dim), self.hidden_dim),
                nn.GELU(),
            )
            fuse_channels = self.hidden_dim * 2
        else:
            self.rgb_proj = None
            fuse_channels = self.hidden_dim
        self.fuse = nn.Sequential(
            nn.Conv2d(fuse_channels, self.hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(self.hidden_dim), self.hidden_dim),
            nn.GELU(),
        )
        self.edge_logits = nn.Conv2d(self.hidden_dim, 1, kernel_size=1)
        self.class_edge_logits = (
            nn.Conv2d(self.hidden_dim, self.num_classes, kernel_size=1) if self.class_aware else None
        )
        self.last_edge_stats = {}

    def _downsample_event_edge(self, event_edge):
        if self.output_stride <= 1:
            return event_edge
        return F.avg_pool2d(
            event_edge,
            kernel_size=self.output_stride,
            stride=self.output_stride,
            ceil_mode=True,
        )

    def forward(self, features, event_edge, boundary_target=None, class_boundary_target=None):
        if event_edge is None or event_edge.shape[1] == 0:
            self.last_edge_stats = {}
            return None, {}

        x = self._downsample_event_edge(event_edge.to(dtype=next(self.parameters()).dtype))
        edge_feature = self.event_encoder(x)
        if self.use_rgb_feature:
            rgb_feature = features[self.rgb_feature]
            if self.detach_rgb_feature:
                rgb_feature = rgb_feature.detach()
            rgb_feature = self.rgb_proj(rgb_feature.to(dtype=edge_feature.dtype))
            rgb_feature = F.interpolate(
                rgb_feature,
                size=edge_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            edge_feature = torch.cat([edge_feature, rgb_feature], dim=1)
        edge_feature = self.fuse(edge_feature)
        logits = self.edge_logits(edge_feature)
        class_logits = self.class_edge_logits(edge_feature) if self.class_edge_logits is not None else None

        losses = {}
        stats = {}
        if boundary_target is not None:
            target = boundary_target.to(device=logits.device, dtype=logits.dtype)
            if target.ndim == 3:
                target = target[:, None]
            if target.shape[-2:] != logits.shape[-2:]:
                target = F.interpolate(target, size=logits.shape[-2:], mode="nearest")
            target = target.clamp(0.0, 1.0)
            positive = target.sum()
            negative = target.numel() - positive
            pos_weight = (negative / positive.clamp_min(1.0)).clamp(1.0, self.pos_weight_max)
            bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
            prob = logits.sigmoid()
            intersection = (prob * target).sum(dim=(1, 2, 3))
            denominator = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
            dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0))
            losses["loss_event_edge_bce"] = bce * self.bce_weight
            losses["loss_event_edge_dice"] = dice.mean() * self.dice_weight
            with torch.no_grad():
                pred = prob > 0.5
                true = target > 0.5
                tp = (pred & true).sum().float()
                fp = (pred & ~true).sum().float()
                fn = (~pred & true).sum().float()
                precision = tp / (tp + fp).clamp_min(1.0)
                recall = tp / (tp + fn).clamp_min(1.0)
                f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-6)
                stats.update({
                    "prob_mean": prob.mean().detach(),
                    "target_mean": target.mean().detach(),
                    "precision": precision.detach(),
                    "recall": recall.detach(),
                    "f1": f1.detach(),
                    "pos_weight": pos_weight.detach(),
                })
        if class_logits is not None and class_boundary_target is not None:
            class_target = class_boundary_target.to(device=class_logits.device, dtype=class_logits.dtype)
            if class_target.shape[-2:] != class_logits.shape[-2:]:
                class_target = F.interpolate(class_target, size=class_logits.shape[-2:], mode="nearest")
            class_target = class_target.clamp(0.0, 1.0)
            positive = class_target.sum(dim=(0, 2, 3))
            negative = class_target.numel() / max(1, self.num_classes) - positive
            pos_weight = (negative / positive.clamp_min(1.0)).clamp(1.0, self.class_pos_weight_max)
            pos_weight = pos_weight.to(dtype=class_logits.dtype)
            raw_bce = F.binary_cross_entropy_with_logits(class_logits, class_target, reduction="none")
            class_weight = torch.where(
                class_target > 0.5,
                pos_weight.view(1, -1, 1, 1),
                torch.ones_like(class_target),
            )
            losses["loss_event_class_edge_bce"] = (raw_bce * class_weight).mean() * self.class_bce_weight
            with torch.no_grad():
                prob = class_logits.sigmoid()
                pred = prob > 0.5
                true = class_target > 0.5
                tp = (pred & true).sum().float()
                fp = (pred & ~true).sum().float()
                fn = (~pred & true).sum().float()
                precision = tp / (tp + fp).clamp_min(1.0)
                recall = tp / (tp + fn).clamp_min(1.0)
                f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-6)
                stats.update({
                    "class_prob_mean": prob.mean().detach(),
                    "class_target_mean": class_target.mean().detach(),
                    "class_precision": precision.detach(),
                    "class_recall": recall.detach(),
                    "class_f1": f1.detach(),
                    "class_pos_weight": pos_weight.mean().detach(),
                })
        self.last_edge_stats = stats
        outputs = {
            "logits": logits,
            "prob": logits.sigmoid(),
            "feature": edge_feature,
            "class_logits": class_logits,
        }
        return outputs, losses

    def put_edge_stats(self):
        if not self.log_edge_stats or not self.last_edge_stats:
            return
        try:
            storage = get_event_storage()
        except AssertionError:
            return
        for name, value in self.last_edge_stats.items():
            storage.put_scalar(f"event_edge/{name}", value.item(), smoothing_hint=False)


class EventEdgeGuidedAdapter(nn.Module):
    def __init__(
        self,
        output_shapes,
        *,
        stage,
        edge_channels,
        init_alpha,
        gate_bias,
        edge_prob_power,
        use_event_reliability,
        reliability_floor,
        low_light_enabled,
        low_light_gain,
        low_light_luma_threshold,
        low_light_contrast_threshold,
        score_predictor_enabled,
        score_sparsity_weight,
        gate_sparsity_weight,
        non_boundary_gate_weight,
        log_stats,
    ):
        super().__init__()
        self.stage = str(stage)
        if self.stage not in output_shapes:
            raise ValueError(f"Unknown event-edge guide stage: {self.stage}")
        self.feature_channels = int(output_shapes[self.stage].channels)
        self.edge_prob_power = float(edge_prob_power)
        self.use_event_reliability = bool(use_event_reliability)
        self.reliability_floor = float(reliability_floor)
        self.low_light_enabled = bool(low_light_enabled)
        self.low_light_gain = float(low_light_gain)
        self.low_light_luma_threshold = float(low_light_luma_threshold)
        self.low_light_contrast_threshold = float(low_light_contrast_threshold)
        self.score_predictor_enabled = bool(score_predictor_enabled)
        self.score_sparsity_weight = float(score_sparsity_weight)
        self.gate_sparsity_weight = float(gate_sparsity_weight)
        self.non_boundary_gate_weight = float(non_boundary_gate_weight)
        self.log_stats = bool(log_stats)

        self.delta_proj = nn.Sequential(
            nn.Conv2d(edge_channels, self.feature_channels, kernel_size=1),
            nn.GroupNorm(_group_count(self.feature_channels), self.feature_channels),
            nn.GELU(),
            nn.Conv2d(self.feature_channels, self.feature_channels, kernel_size=3, padding=1),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(self.feature_channels * 2 + 3, self.feature_channels // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(self.feature_channels // 2, 1, kernel_size=1),
        )
        if self.score_predictor_enabled:
            self.score_predictor = nn.Sequential(
                nn.Conv2d(edge_channels, edge_channels, kernel_size=3, padding=1, groups=edge_channels),
                nn.GELU(),
                nn.Conv2d(edge_channels, 1, kernel_size=1),
                nn.Sigmoid(),
            )
        else:
            self.score_predictor = None
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))
        nn.init.zeros_(self.delta_proj[-1].weight)
        nn.init.zeros_(self.delta_proj[-1].bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, float(gate_bias))
        if self.score_predictor is not None:
            nn.init.zeros_(self.score_predictor[2].weight)
            nn.init.zeros_(self.score_predictor[2].bias)
        self.last_stats = {}
        self.aux_losses = {}

    def _event_reliability(self, event_stats, size, dtype):
        if event_stats is None or event_stats.shape[1] < 4:
            return None
        smooth_stats = F.interpolate(event_stats[:, :3], size=size, mode="bilinear", align_corners=False)
        support = F.interpolate(event_stats[:, 3:4], size=size, mode="nearest")
        density = smooth_stats[:, 0:1].clamp_min(0.0)
        temporal = smooth_stats[:, 1:2].clamp(0.0, 1.0)
        polarity = smooth_stats[:, 2:3].clamp(0.0, 1.0)
        density_max = density.flatten(2).amax(dim=2).view(-1, 1, 1, 1).clamp_min(1e-6)
        density = (density / density_max).clamp(0.0, 1.0)
        reliability = support * density.sqrt() * temporal.clamp_min(1e-4) * polarity.clamp_min(1e-4).sqrt()
        floor = min(max(self.reliability_floor, 0.0), 1.0)
        reliability = support * (floor + (1.0 - floor) * reliability.clamp(0.0, 1.0))
        return reliability.to(dtype=dtype)

    def _low_light_score(self, raw_images, size, dtype):
        if raw_images is None or not self.low_light_enabled:
            return None
        rgb = raw_images.to(dtype=torch.float32) / 255.0
        r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
        luma = (0.299 * r + 0.587 * g + 0.114 * b).clamp(0.0, 1.0)
        local_mean = F.avg_pool2d(luma, kernel_size=7, stride=1, padding=3)
        contrast = (luma - local_mean).abs()
        luma_score = (self.low_light_luma_threshold - luma) / max(self.low_light_luma_threshold, 1e-6)
        contrast_score = (self.low_light_contrast_threshold - contrast) / max(
            self.low_light_contrast_threshold,
            1e-6,
        )
        low_light = (0.7 * luma_score + 0.3 * contrast_score).clamp(0.0, 1.0)
        return F.interpolate(low_light, size=size, mode="bilinear", align_corners=False).to(dtype=dtype)

    def forward(self, features, edge_outputs, event_stats=None, raw_images=None, boundary_target=None):
        self.aux_losses = {}
        if edge_outputs is None or self.stage not in features:
            self.last_stats = {}
            return features

        features = dict(features)
        feature = features[self.stage]
        size = feature.shape[-2:]
        dtype = feature.dtype
        edge_feature = edge_outputs["feature"].to(dtype=dtype)
        edge_feature = F.interpolate(edge_feature, size=size, mode="bilinear", align_corners=False)
        edge_prob = edge_outputs["prob"].to(dtype=dtype)
        edge_prob = F.interpolate(edge_prob, size=size, mode="bilinear", align_corners=False).clamp(0.0, 1.0)
        if self.score_predictor is not None:
            event_score = self.score_predictor(edge_feature).to(dtype=dtype).clamp(0.0, 1.0)
        else:
            event_score = torch.ones_like(edge_prob)
        event_delta = self.delta_proj(edge_feature)

        reliability = self._event_reliability(event_stats, size, dtype) if self.use_event_reliability else None
        if reliability is None:
            reliability = torch.ones_like(edge_prob)
        low_light = self._low_light_score(raw_images, size, dtype)
        if low_light is None:
            low_light = torch.zeros_like(edge_prob)

        edge_prior = edge_prob.clamp_min(1e-4).pow(max(self.edge_prob_power, 0.0))
        low_light_boost = (1.0 + self.low_light_gain * low_light).clamp(0.0, 2.0)
        gate_input = torch.cat([feature, event_delta, edge_prob, reliability, low_light], dim=1)
        learned_gate = torch.sigmoid(self.gate(gate_input))
        final_gate = (learned_gate * event_score * edge_prior * reliability * low_light_boost).clamp(0.0, 1.0)
        features[self.stage] = feature + self.alpha.clamp(0.0, 1.0) * final_gate * event_delta

        losses = {}
        if self.score_predictor is not None and self.score_sparsity_weight > 0:
            losses["loss_event_edge_score_sparse"] = event_score.mean() * self.score_sparsity_weight
        if self.gate_sparsity_weight > 0:
            losses["loss_event_edge_guide_sparse"] = final_gate.mean() * self.gate_sparsity_weight
        if self.non_boundary_gate_weight > 0 and boundary_target is not None:
            target = boundary_target.to(device=final_gate.device, dtype=final_gate.dtype)
            if target.ndim == 3:
                target = target[:, None]
            target = F.interpolate(target, size=size, mode="nearest").clamp(0.0, 1.0)
            losses["loss_event_edge_guide_non_boundary"] = (
                (final_gate * (1.0 - target)).mean() * self.non_boundary_gate_weight
            )
        self.aux_losses = losses
        with torch.no_grad():
            self.last_stats = {
                "gate_mean": final_gate.mean().detach(),
                "gate_max": final_gate.max().detach(),
                "learned_gate_mean": learned_gate.mean().detach(),
                "event_score_mean": event_score.mean().detach(),
                "event_score_max": event_score.max().detach(),
                "event_score_low_fraction": (event_score < 0.02).float().mean().detach(),
                "event_score_high_fraction": (event_score > 0.98).float().mean().detach(),
                "edge_prior_mean": edge_prior.mean().detach(),
                "reliability_mean": reliability.mean().detach(),
                "low_light_mean": low_light.mean().detach(),
                "alpha": self.alpha.detach().clamp(0.0, 1.0),
            }
        return features

    def extra_losses(self):
        return dict(self.aux_losses)

    def put_guide_stats(self):
        if not self.log_stats or not self.last_stats:
            return
        try:
            storage = get_event_storage()
        except AssertionError:
            return
        for name, value in self.last_stats.items():
            storage.put_scalar(f"event_edge_guide/{self.stage}_{name}", value.item(), smoothing_hint=False)


class EarlyEventEdgeAdapter(nn.Module):
    def __init__(
        self,
        *,
        feature_channels,
        event_channels,
        stat_channels,
        hidden_dim,
        init_alpha,
        alpha_max,
        gate_bias,
        edge_prob_power,
        use_event_reliability,
        reliability_floor,
        reliability_threshold,
        event_edge_threshold,
        confidence_threshold,
        margin_threshold,
        entropy_threshold,
        require_boundary,
        boundary_radius,
        gate_bce_weight,
        gate_positive_weight,
        gate_negative_weight,
        gate_supervision_detach,
        gate_supervision_target,
        allow_pred_classes,
        deny_pred_classes,
        allow_target_classes,
        deny_target_classes,
        log_stats,
        ignore_label,
        num_classes,
    ):
        super().__init__()
        self.feature_channels = int(feature_channels)
        self.event_channels = int(event_channels)
        self.stat_channels = int(stat_channels)
        self.hidden_dim = int(hidden_dim)
        self.alpha_max = float(alpha_max)
        self.edge_prob_power = float(edge_prob_power)
        self.use_event_reliability = bool(use_event_reliability)
        self.reliability_floor = float(reliability_floor)
        self.reliability_threshold = float(reliability_threshold)
        self.event_edge_threshold = float(event_edge_threshold)
        self.confidence_threshold = float(confidence_threshold)
        self.margin_threshold = float(margin_threshold)
        self.entropy_threshold = float(entropy_threshold)
        self.require_boundary = bool(require_boundary)
        self.boundary_radius = max(1, int(boundary_radius))
        self.gate_bce_weight = float(gate_bce_weight)
        self.gate_positive_weight = float(gate_positive_weight)
        self.gate_negative_weight = float(gate_negative_weight)
        self.gate_supervision_detach = bool(gate_supervision_detach)
        self.gate_supervision_target = str(gate_supervision_target).lower()
        if self.gate_supervision_target not in {"final", "learned"}:
            raise ValueError(
                "gate_supervision_target must be 'final' or 'learned', "
                f"got {gate_supervision_target!r}"
            )
        self.log_stats = bool(log_stats)
        self.ignore_label = int(ignore_label)
        self.num_classes = int(num_classes)
        self.allow_pred_classes = self._normalize_class_ids(allow_pred_classes)
        self.deny_pred_classes = self._normalize_class_ids(deny_pred_classes)
        self.allow_target_classes = self._normalize_class_ids(allow_target_classes)
        self.deny_target_classes = self._normalize_class_ids(deny_target_classes)

        self.event_encoder = nn.Sequential(
            nn.Conv2d(self.event_channels, self.hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(self.hidden_dim), self.hidden_dim),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(self.hidden_dim), self.hidden_dim),
            nn.GELU(),
        )
        self.delta_proj = nn.Conv2d(self.hidden_dim, self.feature_channels, kernel_size=1)
        self.gate = nn.Sequential(
            nn.Conv2d(self.feature_channels + self.hidden_dim + 2, self.hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, 1, kernel_size=1),
        )
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))
        nn.init.zeros_(self.delta_proj.weight)
        nn.init.zeros_(self.delta_proj.bias)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, float(gate_bias))
        self.last_stats = {}
        self._cache = {}

    def _normalize_class_ids(self, class_ids):
        normalized = []
        for class_id in class_ids:
            value = int(class_id)
            if value < 0 or value >= self.num_classes:
                continue
            normalized.append(value)
        return tuple(sorted(set(normalized)))

    def _class_membership(self, labels, class_ids):
        if not class_ids:
            return torch.zeros_like(labels, dtype=torch.bool)
        ids = torch.as_tensor(class_ids, device=labels.device, dtype=labels.dtype).view(1, -1, 1, 1)
        return (labels == ids).any(dim=1, keepdim=True)

    def _class_policy_masks(self, prediction, target, valid):
        allowed = torch.ones_like(valid, dtype=torch.bool)
        blocked = torch.zeros_like(valid, dtype=torch.bool)
        if self.allow_pred_classes:
            allowed = allowed & self._class_membership(prediction, self.allow_pred_classes)
        if self.allow_target_classes:
            allowed = allowed & self._class_membership(target, self.allow_target_classes)
        if self.deny_pred_classes:
            blocked = blocked | self._class_membership(prediction, self.deny_pred_classes)
        if self.deny_target_classes:
            blocked = blocked | self._class_membership(target, self.deny_target_classes)
        allowed = allowed & valid
        blocked = blocked & valid
        return allowed, blocked

    def _event_edge_prior(self, event_edge, size, dtype):
        event_edge = F.interpolate(event_edge, size=size, mode="bilinear", align_corners=False)
        magnitude = event_edge.abs().mean(dim=1, keepdim=True)
        scale = magnitude.flatten(2).amax(dim=2).view(-1, 1, 1, 1).clamp_min(1e-6)
        edge_prob = (magnitude / scale).clamp(0.0, 1.0).to(dtype=dtype)
        edge_prior = edge_prob.clamp_min(1e-4).pow(max(self.edge_prob_power, 0.0))
        return event_edge.to(dtype=dtype), edge_prob, edge_prior

    def _event_reliability(self, event_stats, size, dtype, batch_size, device):
        if (
            not self.use_event_reliability
            or event_stats is None
            or event_stats.shape[1] < self.stat_channels
            or self.stat_channels < 4
        ):
            return torch.ones((batch_size, 1, size[0], size[1]), device=device, dtype=dtype)
        smooth_stats = F.interpolate(event_stats[:, :3], size=size, mode="bilinear", align_corners=False)
        support = F.interpolate(event_stats[:, 3:4], size=size, mode="nearest")
        density = smooth_stats[:, 0:1].clamp_min(0.0)
        temporal = smooth_stats[:, 1:2].clamp(0.0, 1.0)
        polarity = smooth_stats[:, 2:3].clamp(0.0, 1.0)
        density_scale = density.flatten(2).amax(dim=2).view(-1, 1, 1, 1).clamp_min(1e-6)
        density = (density / density_scale).clamp(0.0, 1.0)
        reliability = support * density.sqrt() * temporal.clamp_min(1e-4) * polarity.clamp_min(1e-4).sqrt()
        floor = min(max(self.reliability_floor, 0.0), 1.0)
        reliability = support * (floor + (1.0 - floor) * reliability.clamp(0.0, 1.0))
        return reliability.to(dtype=dtype)

    def forward(self, feature, event_edge=None, event_stats=None):
        self._cache = {}
        if event_edge is None:
            self.last_stats = {}
            return feature

        size = feature.shape[-2:]
        dtype = feature.dtype
        event_edge, edge_prob, edge_prior = self._event_edge_prior(event_edge, size, dtype)
        reliability = self._event_reliability(
            event_stats,
            size,
            dtype,
            feature.shape[0],
            feature.device,
        )
        event_feature = self.event_encoder(event_edge)
        delta = self.delta_proj(event_feature)
        gate_input = torch.cat([feature, event_feature, edge_prior, reliability], dim=1)
        learned_gate = torch.sigmoid(self.gate(gate_input))
        final_gate = (learned_gate * edge_prior * reliability).clamp(0.0, 1.0)
        alpha = self.alpha.clamp(0.0, self.alpha_max)
        self._cache = {
            "final_gate": final_gate,
            "learned_gate": learned_gate,
            "edge_prob": edge_prob,
            "edge_prior": edge_prior,
            "reliability": reliability,
            "alpha": alpha,
        }

        with torch.no_grad():
            self.last_stats = {
                "gate_mean": final_gate.mean().detach(),
                "gate_max": final_gate.max().detach(),
                "learned_gate_mean": learned_gate.mean().detach(),
                "edge_prob_mean": edge_prob.mean().detach(),
                "edge_prior_mean": edge_prior.mean().detach(),
                "reliability_mean": reliability.mean().detach(),
                "alpha": alpha.detach(),
            }
        return feature + alpha * final_gate * delta

    def gate_supervision_loss(self, outputs, sem_seg_targets, boundary_targets=None):
        if self.gate_bce_weight <= 0 or not self._cache:
            return {}
        final_gate = self._cache["final_gate"]
        learned_gate = self._cache["learned_gate"]
        edge_prior = self._cache["edge_prior"]
        reliability = self._cache["reliability"]
        size = final_gate.shape[-2:]
        dtype = final_gate.dtype
        device = final_gate.device

        with torch.no_grad():
            mask_cls = F.softmax(outputs["pred_logits"], dim=-1)[..., :-1]
            mask_pred = outputs["pred_masks"].sigmoid()
            sem_scores = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred).clamp_min(1e-8)
            sem_prob = sem_scores / sem_scores.sum(dim=1, keepdim=True).clamp_min(1e-8)
            sem_prob = F.interpolate(sem_prob, size=size, mode="bilinear", align_corners=False)
            sem_prob = sem_prob / sem_prob.sum(dim=1, keepdim=True).clamp_min(1e-8)
            if self.gate_supervision_detach:
                sem_prob = sem_prob.detach()
            topk = sem_prob.topk(k=min(2, sem_prob.shape[1]), dim=1).values
            confidence = topk[:, 0:1]
            if topk.shape[1] > 1:
                margin = topk[:, 0:1] - topk[:, 1:2]
            else:
                margin = torch.ones_like(confidence)
            entropy = -(sem_prob * sem_prob.clamp_min(1e-8).log()).sum(dim=1, keepdim=True)
            entropy = entropy / max(math.log(max(self.num_classes, 2)), 1e-6)
            prediction = sem_prob.argmax(dim=1, keepdim=True)

            target = sem_seg_targets.to(device=device)
            if target.ndim == 3:
                target = target[:, None].float()
            target = F.interpolate(target.float(), size=size, mode="nearest").long()
            valid = target != self.ignore_label
            error = (prediction != target) & valid
            uncertain = (
                (confidence < self.confidence_threshold)
                | (margin < self.margin_threshold)
                | (entropy > self.entropy_threshold)
            )
            event_ok = (
                (edge_prior >= self.event_edge_threshold)
                & (reliability >= self.reliability_threshold)
            )
            if boundary_targets is not None:
                boundary = boundary_targets.to(device=device)
                if boundary.ndim == 3:
                    boundary = boundary[:, None]
                boundary = F.interpolate(boundary.float(), size=size, mode="nearest") > 0.5
            else:
                boundary = torch.ones_like(event_ok, dtype=torch.bool)
            positive_before_class_policy = event_ok & valid & (error | uncertain)
            if self.require_boundary:
                positive_before_class_policy = positive_before_class_policy & boundary
            class_allowed, class_blocked = self._class_policy_masks(prediction, target, valid)
            positive = positive_before_class_policy & class_allowed & ~class_blocked
            negative = valid & event_ok & ~positive
            supervision = positive | negative
            gate_target = positive.to(dtype=dtype)
            gate_weight = (
                positive.to(dtype=dtype) * self.gate_positive_weight
                + negative.to(dtype=dtype) * self.gate_negative_weight
            )

        with torch.cuda.amp.autocast(enabled=False):
            supervised_gate = learned_gate if self.gate_supervision_target == "learned" else final_gate
            gate_prob = supervised_gate.float().clamp(1e-6, 1.0 - 1e-6)
            gate_target = gate_target.float()
            gate_weight = gate_weight.float()
            supervision_weight = supervision.float()
            bce = F.binary_cross_entropy(gate_prob, gate_target, reduction="none")
            denom = (gate_weight * supervision_weight).sum().clamp_min(1.0)
            loss = (bce * gate_weight * supervision_weight).sum() / denom

        with torch.no_grad():
            stats = {
                "target_positive_fraction": positive.float().mean().detach(),
                "target_negative_fraction": negative.float().mean().detach(),
                "target_positive_before_class_policy_fraction": positive_before_class_policy.float().mean().detach(),
                "class_allowed_fraction": class_allowed.float().mean().detach(),
                "class_blocked_fraction": class_blocked.float().mean().detach(),
                "event_ok_fraction": event_ok.float().mean().detach(),
                "error_fraction": error.float().mean().detach(),
                "uncertain_fraction": uncertain.float().mean().detach(),
                "supervision_fraction": supervision.float().mean().detach(),
                "gate_on_positive": (
                    final_gate.detach()[positive].mean()
                    if positive.any()
                    else torch.zeros((), device=device, dtype=dtype)
                ),
                "gate_on_negative": (
                    final_gate.detach()[negative].mean()
                    if negative.any()
                    else torch.zeros((), device=device, dtype=dtype)
                ),
                "learned_gate_on_positive": (
                    learned_gate.detach()[positive].mean()
                    if positive.any()
                    else torch.zeros((), device=device, dtype=dtype)
                ),
                "learned_gate_on_negative": (
                    learned_gate.detach()[negative].mean()
                    if negative.any()
                    else torch.zeros((), device=device, dtype=dtype)
                ),
            }
            self.last_stats.update(stats)
        return {"loss_early_event_gate_bce": loss * self.gate_bce_weight}

    def put_stats(self):
        if not self.log_stats or not self.last_stats:
            return
        try:
            storage = get_event_storage()
        except AssertionError:
            return
        for name, value in self.last_stats.items():
            storage.put_scalar(f"early_event_edge_adapter/{name}", value.item(), smoothing_hint=False)


class DayEventBoundaryRefiner(nn.Module):
    def __init__(
        self,
        *,
        num_classes,
        edge_channels,
        hidden_dim,
        init_alpha,
        gate_bias,
        final_gate_scale,
        topk_classes,
        tau_margin,
        entropy_threshold,
        uncertain_fraction,
        edge_threshold,
        edge_prob_power,
        boundary_radius,
        boundary_weight,
        use_event_reliability,
        use_rgb_uncertainty,
        use_semantic_boundary,
        require_uncertain_boundary_intersection,
        require_event_active,
        event_active_source,
        event_active_threshold,
        preserve_confidence_threshold,
        preserve_margin_threshold,
        min_confidence_for_correction,
        min_margin_for_correction,
        loss_boundary_ce_weight,
        loss_uncertain_ce_weight,
        loss_preserve_kl_weight,
        loss_gate_sparse_weight,
        loss_gate_invalid_weight,
        loss_delta_non_boundary_weight,
        loss_candidate_ce_weight,
        candidate_ce_repair_positive_only,
        candidate_logit_scale,
        loss_allowed_only,
        loss_allowed_soft_threshold,
        class_gate_weights,
        class_gate_loss_threshold,
        edge_only_correction,
        score_predictor_enabled,
        score_bce_weight,
        score_sparsity_weight,
        score_positive_weight,
        score_negative_weight,
        selective_repair_gate_enabled,
        loss_repair_gate_bce_weight,
        repair_gate_positive_weight,
        repair_gate_negative_weight,
        repair_require_target_in_topk,
        repair_supervise_score,
        repair_class_weights,
        pair_aware_enabled,
        pair_allow_weights,
        pair_suppress_weights,
        pair_weight_default,
        hard_pair_gate_enabled,
        hard_pair_gate_threshold,
        hard_pair_gate_include_identity,
        hard_pair_suppress_enabled,
        loss_pair_suppress_gate_weight,
        loss_pair_suppress_delta_weight,
        detach_base_prob,
        skip_mask2former_loss,
        log_stats,
        ignore_label,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.edge_channels = int(edge_channels)
        self.hidden_dim = int(hidden_dim)
        self.final_gate_scale = float(final_gate_scale)
        self.topk_classes = max(1, int(topk_classes))
        self.tau_margin = float(tau_margin)
        self.entropy_threshold = float(entropy_threshold)
        self.uncertain_fraction = float(uncertain_fraction)
        self.edge_threshold = float(edge_threshold)
        self.edge_prob_power = float(edge_prob_power)
        self.boundary_radius = max(1, int(boundary_radius))
        self.boundary_weight = float(boundary_weight)
        self.use_event_reliability = bool(use_event_reliability)
        self.use_rgb_uncertainty = bool(use_rgb_uncertainty)
        self.use_semantic_boundary = bool(use_semantic_boundary)
        self.require_uncertain_boundary_intersection = bool(
            require_uncertain_boundary_intersection
        )
        self.require_event_active = bool(require_event_active)
        self.event_active_source = str(event_active_source).lower()
        self.event_active_threshold = float(event_active_threshold)
        self.preserve_confidence_threshold = float(preserve_confidence_threshold)
        self.preserve_margin_threshold = float(preserve_margin_threshold)
        self.min_confidence_for_correction = float(min_confidence_for_correction)
        self.min_margin_for_correction = float(min_margin_for_correction)
        self.loss_boundary_ce_weight = float(loss_boundary_ce_weight)
        self.loss_uncertain_ce_weight = float(loss_uncertain_ce_weight)
        self.loss_preserve_kl_weight = float(loss_preserve_kl_weight)
        self.loss_gate_sparse_weight = float(loss_gate_sparse_weight)
        self.loss_gate_invalid_weight = float(loss_gate_invalid_weight)
        self.loss_delta_non_boundary_weight = float(loss_delta_non_boundary_weight)
        self.loss_candidate_ce_weight = float(loss_candidate_ce_weight)
        self.candidate_ce_repair_positive_only = bool(candidate_ce_repair_positive_only)
        self.candidate_logit_scale = float(candidate_logit_scale)
        self.loss_allowed_only = bool(loss_allowed_only)
        self.loss_allowed_soft_threshold = float(loss_allowed_soft_threshold)
        if class_gate_weights and len(class_gate_weights) == self.num_classes:
            self.class_gate_weights = tuple(float(value) for value in class_gate_weights)
        else:
            self.class_gate_weights = tuple()
        self.class_gate_loss_threshold = float(class_gate_loss_threshold)
        self.edge_only_correction = bool(edge_only_correction)
        self.score_predictor_enabled = bool(score_predictor_enabled)
        self.score_bce_weight = float(score_bce_weight)
        self.score_sparsity_weight = float(score_sparsity_weight)
        self.score_positive_weight = float(score_positive_weight)
        self.score_negative_weight = float(score_negative_weight)
        self.selective_repair_gate_enabled = bool(selective_repair_gate_enabled)
        self.loss_repair_gate_bce_weight = float(loss_repair_gate_bce_weight)
        self.repair_gate_positive_weight = float(repair_gate_positive_weight)
        self.repair_gate_negative_weight = float(repair_gate_negative_weight)
        self.repair_require_target_in_topk = bool(repair_require_target_in_topk)
        self.repair_supervise_score = bool(repair_supervise_score)
        if repair_class_weights and len(repair_class_weights) == self.num_classes:
            self.repair_class_weights = tuple(float(value) for value in repair_class_weights)
        else:
            self.repair_class_weights = tuple()
        self.pair_aware_enabled = bool(pair_aware_enabled)
        self.pair_weight_default = float(pair_weight_default)
        self.hard_pair_gate_enabled = bool(hard_pair_gate_enabled)
        self.hard_pair_gate_threshold = float(hard_pair_gate_threshold)
        self.hard_pair_gate_include_identity = bool(hard_pair_gate_include_identity)
        self.hard_pair_suppress_enabled = bool(hard_pair_suppress_enabled)
        self.loss_pair_suppress_gate_weight = float(loss_pair_suppress_gate_weight)
        self.loss_pair_suppress_delta_weight = float(loss_pair_suppress_delta_weight)
        self.detach_base_prob = bool(detach_base_prob)
        self.skip_mask2former_loss = bool(skip_mask2former_loss)
        self.log_stats = bool(log_stats)
        self.ignore_label = int(ignore_label)

        context_channels = 12
        in_channels = self.edge_channels + context_channels
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, self.hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(self.hidden_dim), self.hidden_dim),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(self.hidden_dim), self.hidden_dim),
            nn.GELU(),
        )
        self.delta_head = nn.Conv2d(self.hidden_dim, self.num_classes, kernel_size=1)
        self.gate_head = nn.Conv2d(self.hidden_dim, 1, kernel_size=1)
        self.score_predictor = None
        if self.score_predictor_enabled:
            self.score_predictor = nn.Sequential(
                nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
                nn.GroupNorm(_group_count(self.hidden_dim), self.hidden_dim),
                nn.GELU(),
                nn.Conv2d(self.hidden_dim, 1, kernel_size=1),
            )
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, float(gate_bias))
        if self.score_predictor is not None:
            nn.init.zeros_(self.score_predictor[-1].weight)
            nn.init.constant_(self.score_predictor[-1].bias, 4.0)
        pair_count = self.num_classes * self.num_classes
        if self.pair_aware_enabled and len(pair_allow_weights) == pair_count:
            pair_allow = torch.tensor(pair_allow_weights, dtype=torch.float32).view(
                self.num_classes,
                self.num_classes,
            )
        else:
            pair_allow = torch.empty(0, dtype=torch.float32)
        if self.pair_aware_enabled and len(pair_suppress_weights) == pair_count:
            pair_suppress = torch.tensor(pair_suppress_weights, dtype=torch.float32).view(
                self.num_classes,
                self.num_classes,
            )
        else:
            pair_suppress = torch.empty(0, dtype=torch.float32)
        self.register_buffer("pair_allow_weights", pair_allow, persistent=False)
        self.register_buffer("pair_suppress_weights", pair_suppress, persistent=False)
        self.last_stats = {}

    def _event_context(self, event_stats, size, dtype, batch_size):
        if event_stats is None or event_stats.shape[1] < 4:
            zeros = torch.zeros((batch_size, 4, size[0], size[1]), dtype=dtype, device=self.alpha.device)
            return zeros[:, 0:1], zeros
        event_stats = event_stats.to(device=self.alpha.device, dtype=dtype)
        smooth_stats = F.interpolate(event_stats[:, :3], size=size, mode="bilinear", align_corners=False)
        support = F.interpolate(event_stats[:, 3:4], size=size, mode="nearest").clamp(0.0, 1.0)
        density = smooth_stats[:, 0:1].clamp_min(0.0)
        temporal = smooth_stats[:, 1:2].clamp(0.0, 1.0)
        polarity = smooth_stats[:, 2:3].clamp(0.0, 1.0)
        density_max = density.flatten(2).amax(dim=2).view(-1, 1, 1, 1).clamp_min(1e-6)
        density = (density / density_max).clamp(0.0, 1.0)
        reliability = support * density.sqrt() * temporal.clamp_min(1e-4) * polarity.clamp_min(1e-4).sqrt()
        event_context = torch.cat([density, temporal, polarity, support], dim=1).to(dtype=dtype)
        return reliability.to(dtype=dtype).clamp(0.0, 1.0), event_context

    def _event_active_mask(self, event_context):
        density = event_context[:, 0:1].clamp(0.0, 1.0)
        support = event_context[:, 3:4].clamp(0.0, 1.0)
        source = self.event_active_source
        if source in {"support", "mask"}:
            active_score = support
        elif source in {"density_or_support", "union", "either"}:
            active_score = torch.maximum(density, support)
        elif source in {"density_and_support", "intersection", "both"}:
            active_score = torch.minimum(density, support)
        else:
            active_score = density
        if not self.require_event_active:
            return torch.ones_like(active_score), active_score
        return (active_score > self.event_active_threshold).to(dtype=event_context.dtype), active_score

    def _top_uncertain_fraction_mask(self, score):
        fraction = float(self.uncertain_fraction)
        if not (0.0 < fraction < 1.0):
            return torch.ones_like(score)
        flat = score.flatten(1)
        total = flat.shape[1]
        keep = max(1, min(total, int(total * fraction + 0.5)))
        _, indices = flat.topk(keep, dim=1, largest=True, sorted=False)
        mask = torch.zeros_like(flat)
        mask.scatter_(1, indices, 1.0)
        return mask.view_as(score)

    def _pred_boundary(self, pred):
        pred = pred[:, None].to(dtype=torch.float32)
        kernel_size = 2 * self.boundary_radius + 1
        local_max = F.max_pool2d(pred, kernel_size=kernel_size, stride=1, padding=self.boundary_radius)
        local_min = -F.max_pool2d(-pred, kernel_size=kernel_size, stride=1, padding=self.boundary_radius)
        return (local_max != local_min).to(dtype=pred.dtype)

    def _resize_boundary(self, boundary_targets, size, dtype):
        if boundary_targets is None:
            return None
        boundary = boundary_targets.to(device=self.alpha.device, dtype=dtype)
        if boundary.ndim == 3:
            boundary = boundary[:, None]
        return F.interpolate(boundary, size=size, mode="nearest").clamp(0.0, 1.0)

    def _resize_sem_seg(self, sem_seg_targets, size):
        if sem_seg_targets is None:
            return None
        target = sem_seg_targets[:, None].to(device=self.alpha.device, dtype=torch.float32)
        target = F.interpolate(target, size=size, mode="nearest")[:, 0]
        return target.to(dtype=torch.long)

    def _class_gate(self, pred, dtype):
        if not self.class_gate_weights:
            return torch.ones((pred.shape[0], 1, pred.shape[1], pred.shape[2]), device=pred.device, dtype=dtype)
        weights = torch.tensor(self.class_gate_weights, device=pred.device, dtype=dtype)
        return weights[pred].unsqueeze(1).clamp(0.0, 4.0)

    def _target_class_weight(self, target, dtype):
        if target is None:
            return None
        if not self.repair_class_weights:
            shape = (target.shape[0], 1, target.shape[1], target.shape[2])
            return torch.ones(shape, device=target.device, dtype=dtype)
        weights = torch.tensor(self.repair_class_weights, device=target.device, dtype=dtype)
        valid = target.ne(self.ignore_label) & target.ge(0) & target.lt(self.num_classes)
        safe_target = target.clamp(0, self.num_classes - 1)
        class_weight = weights[safe_target].unsqueeze(1).clamp(0.0, 16.0)
        return torch.where(valid[:, None], class_weight, torch.ones_like(class_weight))

    def _pair_allow_weight(self, pred, target, dtype):
        if self.pair_allow_weights.numel() == 0 or target is None:
            return torch.ones((pred.shape[0], 1, pred.shape[1], pred.shape[2]), device=pred.device, dtype=dtype)
        valid = target.ne(self.ignore_label) & target.ge(0) & target.lt(self.num_classes)
        safe_target = target.clamp(0, self.num_classes - 1)
        weights = self.pair_allow_weights.to(device=pred.device, dtype=dtype)[pred, safe_target]
        default = weights.new_full(weights.shape, self.pair_weight_default)
        return torch.where(valid, weights, default).unsqueeze(1).clamp(0.0, 8.0)

    def _pair_suppress_weight(self, pred, topk_mask, dtype):
        if self.pair_suppress_weights.numel() == 0:
            return torch.zeros((pred.shape[0], 1, pred.shape[1], pred.shape[2]), device=pred.device, dtype=dtype)
        suppress = self.pair_suppress_weights.to(device=pred.device, dtype=dtype)[pred]
        suppress = suppress.permute(0, 3, 1, 2).contiguous()
        suppress = (suppress * topk_mask).amax(dim=1, keepdim=True)
        return suppress.clamp(0.0, 8.0)

    def _hard_pair_correction_mask(self, pred, topk_mask, dtype):
        correction_mask = topk_mask
        if self.hard_pair_gate_enabled and self.pair_allow_weights.numel() > 0:
            allow = self.pair_allow_weights.to(device=pred.device, dtype=dtype)[pred]
            allow = allow.permute(0, 3, 1, 2).contiguous()
            if not self.hard_pair_gate_include_identity:
                pred_one_hot = torch.zeros_like(topk_mask)
                pred_one_hot.scatter_(1, pred[:, None], 1.0)
                allow = allow * (1.0 - pred_one_hot)
            allow_mask = (allow >= self.hard_pair_gate_threshold).to(dtype=dtype)
            correction_mask = correction_mask * allow_mask
        if self.hard_pair_suppress_enabled and self.pair_suppress_weights.numel() > 0:
            suppress = self.pair_suppress_weights.to(device=pred.device, dtype=dtype)[pred]
            suppress = suppress.permute(0, 3, 1, 2).contiguous()
            suppress_mask = (suppress > 0.0).to(dtype=dtype)
            correction_mask = correction_mask * (1.0 - suppress_mask)
        return correction_mask.clamp(0.0, 1.0)

    def _masked_ce(self, logits, target, mask):
        if target is None or mask is None:
            return logits.sum() * 0.0
        valid = target.ne(self.ignore_label)
        mask = mask[:, 0].bool() & valid
        if not mask.any():
            return logits.sum() * 0.0
        ce = F.cross_entropy(logits.float(), target.clamp_min(0), ignore_index=self.ignore_label, reduction="none")
        return (ce * mask.to(dtype=ce.dtype)).sum() / mask.sum().clamp_min(1).to(dtype=ce.dtype)

    def _masked_ce_weighted(self, logits, target, mask, weight_map=None):
        if target is None or mask is None:
            return logits.sum() * 0.0
        valid = target.ne(self.ignore_label)
        mask = mask[:, 0].bool() & valid
        if not mask.any():
            return logits.sum() * 0.0
        ce = F.cross_entropy(logits.float(), target.clamp_min(0), ignore_index=self.ignore_label, reduction="none")
        weight = mask.to(dtype=ce.dtype)
        if weight_map is not None:
            weight = weight * weight_map[:, 0].to(dtype=ce.dtype).clamp_min(0.0)
        denom = weight.sum().clamp_min(1.0)
        return (ce * weight).sum() / denom

    def forward(self, outputs, edge_outputs, event_stats=None, boundary_targets=None, sem_seg_targets=None):
        base_prob = self._semantic_probs_from_outputs(outputs)
        if self.detach_base_prob:
            base_prob = base_prob.detach()
        size = base_prob.shape[-2:]
        dtype = base_prob.dtype
        device = base_prob.device

        edge_feature = edge_outputs["feature"].to(device=device, dtype=dtype)
        edge_feature = F.interpolate(edge_feature, size=size, mode="bilinear", align_corners=False)
        edge_prob = edge_outputs["prob"].to(device=device, dtype=dtype)
        edge_prob = F.interpolate(edge_prob, size=size, mode="bilinear", align_corners=False).clamp(0.0, 1.0)
        reliability, event_context = self._event_context(event_stats, size, dtype, base_prob.shape[0])
        if reliability.shape[0] == 1 and reliability.shape[0] != base_prob.shape[0]:
            reliability = reliability.expand(base_prob.shape[0], -1, -1, -1)
        if event_context.shape[0] == 1 and event_context.shape[0] != base_prob.shape[0]:
            event_context = event_context.expand(base_prob.shape[0], -1, -1, -1)
        event_active_mask, event_active_score = self._event_active_mask(event_context)

        topk = base_prob.topk(k=min(2, base_prob.shape[1]), dim=1)
        top1_prob = topk.values[:, 0:1]
        if topk.values.shape[1] > 1:
            top2_prob = topk.values[:, 1:2]
        else:
            top2_prob = torch.zeros_like(top1_prob)
        margin = top1_prob - top2_prob
        rgb_uncertain = ((self.tau_margin - margin) / max(self.tau_margin, 1e-6)).clamp(0.0, 1.0)
        rgb_entropy = -(base_prob * base_prob.clamp_min(1e-8).log()).sum(dim=1, keepdim=True)
        rgb_entropy = (rgb_entropy / torch.log(base_prob.new_tensor(float(base_prob.shape[1])))).clamp(0.0, 1.0)
        rgb_entropy_uncertain = (
            (rgb_entropy - self.entropy_threshold) / max(1.0 - self.entropy_threshold, 1e-6)
        ).clamp(0.0, 1.0)
        pred = base_prob.argmax(dim=1)
        rgb_boundary = self._pred_boundary(pred).to(device=device, dtype=dtype)
        class_gate = self._class_gate(pred, dtype)
        if self.class_gate_loss_threshold > 0:
            class_gate_loss_mask = (class_gate >= self.class_gate_loss_threshold).to(dtype=dtype)
        else:
            class_gate_loss_mask = torch.ones_like(class_gate)

        edge_prior = edge_prob.clamp_min(1e-4).pow(max(self.edge_prob_power, 0.0))
        rgb_confidence_score = torch.maximum(rgb_uncertain, rgb_entropy_uncertain)
        if self.use_rgb_uncertainty and self.use_semantic_boundary:
            rgb_instability_score = torch.maximum(
                rgb_confidence_score,
                self.boundary_weight * rgb_boundary,
            ).clamp(0.0, 1.0)
            rgb_uncertain_fraction_mask = self._top_uncertain_fraction_mask(rgb_instability_score)
            rgb_confidence_prior = (rgb_instability_score * rgb_uncertain_fraction_mask).clamp(0.0, 1.0)
            if self.require_uncertain_boundary_intersection:
                rgb_prior = (rgb_confidence_prior * rgb_boundary).clamp(0.0, 1.0)
            else:
                rgb_prior = rgb_confidence_prior
            if self.edge_only_correction and not self.require_uncertain_boundary_intersection:
                rgb_prior = (rgb_prior * rgb_boundary).clamp(0.0, 1.0)
        elif self.use_rgb_uncertainty:
            rgb_instability_score = rgb_confidence_score.clamp(0.0, 1.0)
            rgb_uncertain_fraction_mask = self._top_uncertain_fraction_mask(rgb_instability_score)
            rgb_confidence_prior = (rgb_instability_score * rgb_uncertain_fraction_mask).clamp(0.0, 1.0)
            rgb_prior = rgb_confidence_prior
        elif self.use_semantic_boundary:
            rgb_instability_score = rgb_boundary.clamp(0.0, 1.0)
            rgb_uncertain_fraction_mask = torch.ones_like(rgb_boundary)
            rgb_confidence_prior = rgb_boundary.clamp(0.0, 1.0)
            rgb_prior = rgb_confidence_prior
        else:
            rgb_instability_score = torch.ones_like(rgb_confidence_score)
            rgb_uncertain_fraction_mask = torch.ones_like(rgb_confidence_score)
            rgb_confidence_prior = torch.ones_like(rgb_confidence_score)
            rgb_prior = rgb_confidence_prior
        reliability_prior = reliability if self.use_event_reliability else torch.ones_like(reliability)
        allowed_soft = (edge_prior * reliability_prior * rgb_prior * class_gate).clamp(0.0, 1.0)
        if self.min_confidence_for_correction > 0:
            allowed_soft = allowed_soft * (top1_prob >= self.min_confidence_for_correction).to(dtype=dtype)
        if self.min_margin_for_correction > 0:
            allowed_soft = allowed_soft * (margin >= self.min_margin_for_correction).to(dtype=dtype)
        if self.require_event_active:
            allowed_soft = allowed_soft * event_active_mask

        context = torch.cat(
            [
                edge_prob,
                reliability,
                event_context,
                rgb_uncertain,
                rgb_entropy_uncertain,
                rgb_boundary,
                margin.clamp(0.0, 1.0),
                top1_prob.clamp(0.0, 1.0),
                allowed_soft,
            ],
            dim=1,
        )
        hidden = self.refine(torch.cat([edge_feature, context.to(dtype=edge_feature.dtype)], dim=1))
        if self.score_predictor is not None:
            score_logits = self.score_predictor(hidden)
            event_score = torch.sigmoid(score_logits).to(dtype=dtype)
        else:
            score_logits = None
            event_score = torch.ones_like(edge_prob)
        gate_logits = self.gate_head(hidden)
        learned_gate = torch.sigmoid(gate_logits)
        gate = (learned_gate * allowed_soft * event_score * event_active_mask).clamp(0.0, 1.0)
        delta_logits = self.delta_head(hidden)

        topk_indices = base_prob.topk(k=min(self.topk_classes, base_prob.shape[1]), dim=1).indices
        topk_mask = torch.zeros_like(base_prob)
        topk_mask.scatter_(1, topk_indices, 1.0)
        correction_mask = self._hard_pair_correction_mask(pred, topk_mask, dtype)
        pair_suppress_weight = self._pair_suppress_weight(pred, topk_mask, dtype)

        base_logits = base_prob.clamp_min(1e-8).log()
        effective_gate = (self.final_gate_scale * gate).clamp(0.0, 1.0)
        final_logits = base_logits + self.alpha.clamp(0.0, 1.0) * effective_gate * correction_mask * delta_logits
        candidate_logits = base_logits + self.candidate_logit_scale * correction_mask * delta_logits
        final_prob = F.softmax(final_logits.float(), dim=1).to(dtype=dtype)

        losses = {}
        if sem_seg_targets is not None:
            target = self._resize_sem_seg(sem_seg_targets, size)
            gt_boundary = self._resize_boundary(boundary_targets, size, dtype)
            pair_allow_weight = self._pair_allow_weight(pred, target, dtype)
            target_class_weight = self._target_class_weight(target, dtype)
            if self.use_event_reliability:
                event_nontrivial = ((edge_prob >= self.edge_threshold) | (reliability > 0.0)).to(dtype=dtype)
            else:
                event_nontrivial = (edge_prob >= self.edge_threshold).to(dtype=dtype)
            if self.require_event_active:
                event_nontrivial = event_nontrivial * event_active_mask
            if self.use_semantic_boundary:
                if gt_boundary is None:
                    boundary_mask = rgb_boundary * event_nontrivial
                else:
                    boundary_mask = torch.maximum(gt_boundary, rgb_boundary) * event_nontrivial
            else:
                boundary_mask = event_nontrivial
            if self.use_rgb_uncertainty:
                uncertain_mask = (rgb_confidence_prior > 0.0).to(dtype=dtype) * event_nontrivial
            else:
                uncertain_mask = event_nontrivial
            safe_bool = (
                (top1_prob >= self.preserve_confidence_threshold)
                & (margin >= self.preserve_margin_threshold)
                & (edge_prob < self.edge_threshold)
            )
            if self.use_semantic_boundary:
                safe_bool = safe_bool & (rgb_boundary < 0.5)
            safe_mask = safe_bool.to(dtype=dtype)
            allowed_hard_bool = edge_prob >= self.edge_threshold
            if self.use_event_reliability:
                allowed_hard_bool = allowed_hard_bool & (reliability > 0.0)
            if self.require_event_active:
                allowed_hard_bool = allowed_hard_bool & (event_active_mask > 0.0)
            if self.use_rgb_uncertainty and self.use_semantic_boundary:
                if self.require_uncertain_boundary_intersection:
                    allowed_rgb_bool = (rgb_confidence_prior > 0.0) & (rgb_boundary > 0.5)
                else:
                    allowed_rgb_bool = (rgb_confidence_prior > 0.0) | (rgb_boundary > 0.5)
            elif self.use_rgb_uncertainty:
                allowed_rgb_bool = rgb_confidence_prior > 0.0
            elif self.use_semantic_boundary:
                allowed_rgb_bool = rgb_boundary > 0.5
            else:
                allowed_rgb_bool = torch.ones_like(allowed_hard_bool, dtype=torch.bool)
            if self.min_confidence_for_correction > 0:
                allowed_rgb_bool = allowed_rgb_bool & (top1_prob >= self.min_confidence_for_correction)
            if self.min_margin_for_correction > 0:
                allowed_rgb_bool = allowed_rgb_bool & (margin >= self.min_margin_for_correction)
            allowed_hard = (allowed_hard_bool & allowed_rgb_bool).to(dtype=dtype) * class_gate_loss_mask
            if self.loss_allowed_soft_threshold > 0:
                allowed_loss_mask = (allowed_soft >= self.loss_allowed_soft_threshold).to(dtype=dtype)
            else:
                allowed_loss_mask = (allowed_soft > 0.0).to(dtype=dtype)
            allowed_loss_mask = allowed_loss_mask * class_gate_loss_mask
            if self.loss_allowed_only:
                boundary_mask = boundary_mask * allowed_loss_mask
                uncertain_mask = uncertain_mask * allowed_loss_mask
            change_region = torch.maximum(boundary_mask, uncertain_mask).clamp(0.0, 1.0)
            valid_target = target.ne(self.ignore_label) & target.ge(0) & target.lt(self.num_classes)
            safe_target = target.clamp(0, self.num_classes - 1)
            target_in_topk = topk_mask.gather(1, safe_target[:, None]).to(dtype=dtype)
            target_in_correction = correction_mask.gather(1, safe_target[:, None]).to(dtype=dtype)
            if self.hard_pair_gate_enabled or self.hard_pair_suppress_enabled:
                target_candidate_ok = target_in_correction > 0.5
            else:
                target_candidate_ok = target_in_topk > 0.5
            if not self.repair_require_target_in_topk:
                target_candidate_ok = valid_target[:, None]
            base_correct = (pred == target) & valid_target
            base_wrong = (~base_correct) & valid_target
            if self.use_semantic_boundary:
                repair_candidate = (boundary_mask * class_gate_loss_mask).clamp(0.0, 1.0)
            else:
                repair_candidate = (change_region * class_gate_loss_mask).clamp(0.0, 1.0)
            pair_suppress_bool = pair_suppress_weight > 0
            repair_positive_bool = (
                (repair_candidate > 0.0)
                & base_wrong[:, None]
                & target_candidate_ok
            )
            repair_negative_bool = (
                (repair_candidate > 0.0)
                & (
                    base_correct[:, None]
                    | (~target_candidate_ok)
                    | pair_suppress_bool
                )
                & (~repair_positive_bool)
            )
            repair_positive = repair_positive_bool.to(dtype=dtype)
            repair_negative = repair_negative_bool.to(dtype=dtype)
            repair_target = repair_positive
            repair_loss_mask = torch.maximum(repair_positive, repair_negative).clamp(0.0, 1.0)

            if self.loss_boundary_ce_weight > 0:
                losses["loss_day_boundary_ce"] = (
                    self._masked_ce_weighted(final_logits, target, boundary_mask, pair_allow_weight)
                    * self.loss_boundary_ce_weight
                )
            if self.loss_uncertain_ce_weight > 0:
                losses["loss_day_uncertain_ce"] = (
                    self._masked_ce_weighted(final_logits, target, uncertain_mask, pair_allow_weight)
                    * self.loss_uncertain_ce_weight
                )
            if self.loss_candidate_ce_weight > 0:
                candidate_ce_mask = repair_positive if self.candidate_ce_repair_positive_only else change_region
                losses["loss_day_candidate_ce"] = (
                    self._masked_ce_weighted(
                        candidate_logits,
                        target,
                        candidate_ce_mask,
                        pair_allow_weight * target_class_weight,
                    )
                    * self.loss_candidate_ce_weight
                )
            if self.selective_repair_gate_enabled and self.loss_repair_gate_bce_weight > 0:
                repair_positive_weight = (
                    self.repair_gate_positive_weight * target_class_weight
                ).to(dtype=repair_positive.dtype)
                repair_weight = torch.where(
                    repair_positive > 0.5,
                    repair_positive_weight,
                    repair_positive.new_full(repair_positive.shape, self.repair_gate_negative_weight),
                )
                repair_weight = repair_weight * repair_loss_mask
                repair_gate_bce = F.binary_cross_entropy_with_logits(
                    gate_logits.float(),
                    repair_target.float(),
                    reduction="none",
                )
                losses["loss_day_repair_gate_bce"] = (
                    (repair_gate_bce * repair_weight.float()).sum()
                    / repair_weight.float().sum().clamp_min(1.0)
                    * self.loss_repair_gate_bce_weight
                )
            if score_logits is not None and self.score_bce_weight > 0:
                if self.use_rgb_uncertainty and self.use_semantic_boundary:
                    if self.require_uncertain_boundary_intersection:
                        score_candidate_prior = (rgb_confidence_prior * rgb_boundary).clamp(0.0, 1.0)
                    else:
                        score_candidate_prior = torch.maximum(rgb_confidence_prior, rgb_boundary)
                elif self.use_rgb_uncertainty:
                    score_candidate_prior = rgb_confidence_prior
                elif self.use_semantic_boundary:
                    score_candidate_prior = rgb_boundary
                else:
                    score_candidate_prior = torch.ones_like(event_nontrivial)
                if self.selective_repair_gate_enabled and self.repair_supervise_score:
                    score_positive = repair_positive
                    score_negative = torch.maximum(repair_negative, safe_mask)
                elif gt_boundary is None or not self.use_semantic_boundary:
                    score_positive = event_nontrivial * score_candidate_prior
                    score_negative = safe_mask
                else:
                    score_candidate = event_nontrivial * score_candidate_prior
                    score_positive = score_candidate * gt_boundary
                    score_negative = torch.maximum(safe_mask, score_candidate * (1.0 - gt_boundary))
                score_loss_mask = torch.maximum(score_positive, score_negative).clamp(0.0, 1.0)
                score_weight = torch.where(
                    score_positive > 0.5,
                    self.score_positive_weight * target_class_weight.to(dtype=score_positive.dtype),
                    score_positive.new_full(score_positive.shape, self.score_negative_weight),
                )
                score_weight = score_weight * score_loss_mask
                score_bce = F.binary_cross_entropy_with_logits(
                    score_logits.float(),
                    score_positive.float().clamp(0.0, 1.0),
                    reduction="none",
                )
                losses["loss_day_event_score_bce"] = (
                    (score_bce * score_weight.float()).sum()
                    / score_weight.float().sum().clamp_min(1.0)
                    * self.score_bce_weight
                )
            if self.score_sparsity_weight > 0:
                losses["loss_day_event_score_sparse"] = (
                    (event_score * event_nontrivial).sum()
                    / event_nontrivial.sum().clamp_min(1.0)
                    * self.score_sparsity_weight
                )
            if self.loss_preserve_kl_weight > 0:
                kl_map = F.kl_div(
                    final_prob.clamp_min(1e-8).log(),
                    base_prob.clamp_min(1e-8),
                    reduction="none",
                ).sum(dim=1, keepdim=True)
                denom = safe_mask.sum().clamp_min(1.0)
                losses["loss_day_preserve_kl"] = (
                    (kl_map * safe_mask).sum() / denom * self.loss_preserve_kl_weight
                )
            if self.loss_gate_sparse_weight > 0:
                losses["loss_day_gate_sparse"] = gate.mean() * self.loss_gate_sparse_weight
            if self.loss_gate_invalid_weight > 0:
                invalid = 1.0 - allowed_hard
                losses["loss_day_gate_invalid"] = (
                    (gate * invalid).sum() / invalid.sum().clamp_min(1.0) * self.loss_gate_invalid_weight
                )
            if self.use_rgb_uncertainty and self.use_semantic_boundary:
                if self.require_uncertain_boundary_intersection:
                    pair_suppress_prior = (rgb_confidence_prior * rgb_boundary).clamp(0.0, 1.0)
                else:
                    pair_suppress_prior = torch.maximum(rgb_confidence_prior, rgb_boundary)
            elif self.use_rgb_uncertainty:
                pair_suppress_prior = rgb_confidence_prior
            elif self.use_semantic_boundary:
                pair_suppress_prior = rgb_boundary
            else:
                pair_suppress_prior = torch.ones_like(event_nontrivial)
            pair_suppress_region = (event_nontrivial * pair_suppress_prior).clamp(0.0, 1.0)
            if self.loss_pair_suppress_gate_weight > 0:
                denom = (pair_suppress_weight * pair_suppress_region).sum().clamp_min(1.0)
                losses["loss_day_pair_suppress_gate"] = (
                    (gate * pair_suppress_weight * pair_suppress_region).sum()
                    / denom
                    * self.loss_pair_suppress_gate_weight
                )
            if self.loss_pair_suppress_delta_weight > 0:
                topk_count = topk_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
                delta_topk_abs = (delta_logits.abs() * topk_mask).sum(dim=1, keepdim=True) / topk_count
                denom = (pair_suppress_weight * pair_suppress_region).sum().clamp_min(1.0)
                losses["loss_day_pair_suppress_delta"] = (
                    (delta_topk_abs * pair_suppress_weight * pair_suppress_region).sum()
                    / denom
                    * self.loss_pair_suppress_delta_weight
                )
            if self.loss_delta_non_boundary_weight > 0:
                non_change = 1.0 - change_region
                losses["loss_day_delta_non_boundary"] = (
                    (delta_logits.abs() * non_change).mean() * self.loss_delta_non_boundary_weight
                )

            with torch.no_grad():
                allowed_ce_base = self._masked_ce(base_logits, target, change_region)
                allowed_ce_final = self._masked_ce(final_logits, target, change_region)
                allowed_ce_candidate = self._masked_ce(candidate_logits, target, change_region)
                repair_positive_denom = repair_positive.sum().clamp_min(1.0)
                repair_negative_denom = repair_negative.sum().clamp_min(1.0)
                gate_repair_positive_mean = (gate * repair_positive).sum() / repair_positive_denom
                gate_repair_negative_mean = (gate * repair_negative).sum() / repair_negative_denom
                effective_gate_repair_positive_mean = (
                    effective_gate * repair_positive
                ).sum() / repair_positive_denom
                effective_gate_repair_negative_mean = (
                    effective_gate * repair_negative
                ).sum() / repair_negative_denom
                learned_gate_repair_positive_mean = (
                    learned_gate * repair_positive
                ).sum() / repair_positive_denom
                learned_gate_repair_negative_mean = (
                    learned_gate * repair_negative
                ).sum() / repair_negative_denom
                candidate_delta = allowed_ce_candidate - allowed_ce_base
                final_delta = allowed_ce_final - allowed_ce_base
                final_to_candidate_delta_ratio = final_delta.abs() / candidate_delta.abs().clamp_min(1e-8)
                self.last_stats = {
                    "gate_mean": gate.mean().detach(),
                    "gate_max": gate.max().detach(),
                    "effective_gate_mean": effective_gate.mean().detach(),
                    "effective_gate_max": effective_gate.max().detach(),
                    "final_gate_scale": base_prob.new_tensor(self.final_gate_scale),
                    "gate_active_001": (gate > 0.01).float().mean().detach(),
                    "learned_gate_mean": learned_gate.mean().detach(),
                    "gate_repair_positive_mean": gate_repair_positive_mean.detach(),
                    "gate_repair_negative_mean": gate_repair_negative_mean.detach(),
                    "effective_gate_repair_positive_mean": effective_gate_repair_positive_mean.detach(),
                    "effective_gate_repair_negative_mean": effective_gate_repair_negative_mean.detach(),
                    "learned_gate_repair_positive_mean": learned_gate_repair_positive_mean.detach(),
                    "learned_gate_repair_negative_mean": learned_gate_repair_negative_mean.detach(),
                    "repair_positive_fraction": repair_positive.mean().detach(),
                    "repair_negative_fraction": repair_negative.mean().detach(),
                    "repair_loss_mask": repair_loss_mask.mean().detach(),
                    "repair_target_in_topk_fraction": target_in_topk.mean().detach(),
                    "repair_target_in_correction_fraction": target_in_correction.mean().detach(),
                    "repair_candidate_fraction": repair_candidate.mean().detach(),
                    "candidate_ce_repair_positive_only": base_prob.new_tensor(
                        float(self.candidate_ce_repair_positive_only)
                    ),
                    "base_wrong_fraction": base_wrong.float().mean().detach(),
                    "event_score_mean": event_score.mean().detach(),
                    "event_score_max": event_score.max().detach(),
                    "event_score_low_fraction": (event_score < 0.02).float().mean().detach(),
                    "event_score_high_fraction": (event_score > 0.98).float().mean().detach(),
                    "allowed_mean": allowed_soft.mean().detach(),
                    "use_event_reliability": base_prob.new_tensor(float(self.use_event_reliability)),
                    "use_rgb_uncertainty": base_prob.new_tensor(float(self.use_rgb_uncertainty)),
                    "use_semantic_boundary": base_prob.new_tensor(float(self.use_semantic_boundary)),
                    "require_uncertain_boundary_intersection": base_prob.new_tensor(
                        float(self.require_uncertain_boundary_intersection)
                    ),
                    "require_event_active": base_prob.new_tensor(float(self.require_event_active)),
                    "min_confidence_for_correction": base_prob.new_tensor(
                        self.min_confidence_for_correction
                    ),
                    "min_margin_for_correction": base_prob.new_tensor(
                        self.min_margin_for_correction
                    ),
                    "event_active_fraction": event_active_mask.mean().detach(),
                    "event_active_score_mean": event_active_score.mean().detach(),
                    "gate_inactive_mean": (
                        (gate * (1.0 - event_active_mask)).sum()
                        / (1.0 - event_active_mask).sum().clamp_min(1.0)
                    ).detach(),
                    "class_gate_mean": class_gate.mean().detach(),
                    "class_gate_loss_mask": class_gate_loss_mask.mean().detach(),
                    "repair_class_weight_mean": target_class_weight.mean().detach(),
                    "pair_allow_weight_mean": pair_allow_weight.mean().detach(),
                    "pair_allow_active_fraction": (pair_allow_weight > self.pair_weight_default).float().mean().detach(),
                    "pair_suppress_weight_mean": pair_suppress_weight.mean().detach(),
                    "pair_suppress_active_fraction": (pair_suppress_weight > 0).float().mean().detach(),
                    "correction_mask_fraction": (correction_mask > 0.0).float().mean().detach(),
                    "edge_prob_mean": edge_prob.mean().detach(),
                    "reliability_mean": reliability.mean().detach(),
                    "event_density_mean": event_context[:, 0:1].mean().detach(),
                    "event_temporal_mean": event_context[:, 1:2].mean().detach(),
                    "event_polarity_mean": event_context[:, 2:3].mean().detach(),
                    "event_support_mean": event_context[:, 3:4].mean().detach(),
                    "rgb_uncertain_mean": rgb_uncertain.mean().detach(),
                    "rgb_entropy_mean": rgb_entropy.mean().detach(),
                    "rgb_entropy_uncertain_mean": rgb_entropy_uncertain.mean().detach(),
                    "rgb_uncertain_fraction": rgb_uncertain_fraction_mask.mean().detach(),
                    "rgb_boundary_mean": rgb_boundary.mean().detach(),
                    "boundary_loss_mask": boundary_mask.mean().detach(),
                    "uncertain_loss_mask": uncertain_mask.mean().detach(),
                    "allowed_loss_mask": allowed_loss_mask.mean().detach(),
                    "allowed_hard": allowed_hard.mean().detach(),
                    "allowed_ce_base": allowed_ce_base.detach(),
                    "allowed_ce_final": allowed_ce_final.detach(),
                    "allowed_ce_candidate": allowed_ce_candidate.detach(),
                    "allowed_ce_candidate_delta": candidate_delta.detach(),
                    "allowed_ce_final_delta": final_delta.detach(),
                    "final_to_candidate_delta_ratio": final_to_candidate_delta_ratio.detach(),
                    "safe_mask": safe_mask.mean().detach(),
                    "delta_abs_mean": delta_logits.abs().mean().detach(),
                    "alpha": self.alpha.detach().clamp(0.0, 1.0),
                }
        else:
            with torch.no_grad():
                self.last_stats = {
                    "gate_mean": gate.mean().detach(),
                    "gate_max": gate.max().detach(),
                    "effective_gate_mean": effective_gate.mean().detach(),
                    "effective_gate_max": effective_gate.max().detach(),
                    "final_gate_scale": base_prob.new_tensor(self.final_gate_scale),
                    "gate_active_001": (gate > 0.01).float().mean().detach(),
                    "learned_gate_mean": learned_gate.mean().detach(),
                    "event_score_mean": event_score.mean().detach(),
                    "event_score_max": event_score.max().detach(),
                    "event_score_low_fraction": (event_score < 0.02).float().mean().detach(),
                    "event_score_high_fraction": (event_score > 0.98).float().mean().detach(),
                    "allowed_mean": allowed_soft.mean().detach(),
                    "use_event_reliability": base_prob.new_tensor(float(self.use_event_reliability)),
                    "use_rgb_uncertainty": base_prob.new_tensor(float(self.use_rgb_uncertainty)),
                    "use_semantic_boundary": base_prob.new_tensor(float(self.use_semantic_boundary)),
                    "require_uncertain_boundary_intersection": base_prob.new_tensor(
                        float(self.require_uncertain_boundary_intersection)
                    ),
                    "require_event_active": base_prob.new_tensor(float(self.require_event_active)),
                    "min_confidence_for_correction": base_prob.new_tensor(
                        self.min_confidence_for_correction
                    ),
                    "min_margin_for_correction": base_prob.new_tensor(
                        self.min_margin_for_correction
                    ),
                    "event_active_fraction": event_active_mask.mean().detach(),
                    "event_active_score_mean": event_active_score.mean().detach(),
                    "gate_inactive_mean": (
                        (gate * (1.0 - event_active_mask)).sum()
                        / (1.0 - event_active_mask).sum().clamp_min(1.0)
                    ).detach(),
                    "class_gate_mean": class_gate.mean().detach(),
                    "class_gate_loss_mask": class_gate_loss_mask.mean().detach(),
                    "pair_suppress_weight_mean": pair_suppress_weight.mean().detach(),
                    "pair_suppress_active_fraction": (pair_suppress_weight > 0).float().mean().detach(),
                    "correction_mask_fraction": (correction_mask > 0.0).float().mean().detach(),
                    "edge_prob_mean": edge_prob.mean().detach(),
                    "reliability_mean": reliability.mean().detach(),
                    "event_density_mean": event_context[:, 0:1].mean().detach(),
                    "event_temporal_mean": event_context[:, 1:2].mean().detach(),
                    "event_polarity_mean": event_context[:, 2:3].mean().detach(),
                    "event_support_mean": event_context[:, 3:4].mean().detach(),
                    "rgb_uncertain_mean": rgb_uncertain.mean().detach(),
                    "rgb_entropy_mean": rgb_entropy.mean().detach(),
                    "rgb_entropy_uncertain_mean": rgb_entropy_uncertain.mean().detach(),
                    "rgb_uncertain_fraction": rgb_uncertain_fraction_mask.mean().detach(),
                    "rgb_boundary_mean": rgb_boundary.mean().detach(),
                    "allowed_loss_mask": (allowed_soft > 0.0).to(dtype=dtype).mean().detach(),
                    "delta_abs_mean": delta_logits.abs().mean().detach(),
                    "alpha": self.alpha.detach().clamp(0.0, 1.0),
                }

        return {
            "logits": final_logits,
            "prob": final_prob,
            "gate": gate,
            "delta_logits": delta_logits,
            "losses": losses,
        }

    def _semantic_probs_from_outputs(self, outputs):
        mask_cls = F.softmax(outputs["pred_logits"], dim=-1)[..., :-1]
        mask_pred = outputs["pred_masks"].sigmoid()
        sem_scores = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred).clamp_min(1e-8)
        return sem_scores / sem_scores.sum(dim=1, keepdim=True).clamp_min(1e-8)

    def put_stats(self):
        if not self.log_stats or not self.last_stats:
            return
        try:
            storage = get_event_storage()
        except AssertionError:
            return
        for name, value in self.last_stats.items():
            storage.put_scalar(f"day_event_boundary_refiner/{name}", value.item(), smoothing_hint=False)


@META_ARCH_REGISTRY.register()
class MaskFormer(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        event_fusion: nn.Module,
        event_edge_head: nn.Module,
        event_edge_guide: nn.Module,
        early_event_edge_adapter: nn.Module,
        day_event_boundary_refiner: nn.Module,
        event_preserve_enabled: bool,
        event_preserve_weight: float,
        event_preserve_confidence_threshold: float,
        event_preserve_margin_threshold: float,
        event_preserve_event_edge_threshold: float,
        event_preserve_use_boundary_target: bool,
        event_preserve_use_event_edge: bool,
        event_preserve_log_stats: bool,
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.event_fusion = event_fusion
        self.event_edge_head = event_edge_head
        self.event_edge_guide = event_edge_guide
        self.early_event_edge_adapter = early_event_edge_adapter
        self.day_event_boundary_refiner = day_event_boundary_refiner
        self.event_preserve_enabled = bool(event_preserve_enabled)
        self.event_preserve_weight = float(event_preserve_weight)
        self.event_preserve_confidence_threshold = float(event_preserve_confidence_threshold)
        self.event_preserve_margin_threshold = float(event_preserve_margin_threshold)
        self.event_preserve_event_edge_threshold = float(event_preserve_event_edge_threshold)
        self.event_preserve_use_boundary_target = bool(event_preserve_use_boundary_target)
        self.event_preserve_use_event_edge = bool(event_preserve_use_event_edge)
        self.event_preserve_log_stats = bool(event_preserve_log_stats)
        self._last_event_preserve_stats = {}

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        backbone_shapes = backbone.output_shape()
        sem_seg_head = build_sem_seg_head(cfg, backbone_shapes)
        event_fusion = None
        if cfg.MODEL.EVENT_FUSION.ENABLED:
            event_fusion = EventUsefulnessFusion(
                backbone_shapes,
                event_channels=cfg.MODEL.EVENT_FUSION.IN_CHANNELS,
                stat_channels=cfg.MODEL.EVENT_FUSION.STAT_CHANNELS,
                stages=cfg.MODEL.EVENT_FUSION.STAGES,
                hidden_dim=cfg.MODEL.EVENT_FUSION.HIDDEN_DIM,
                init_alpha=cfg.MODEL.EVENT_FUSION.INIT_ALPHA,
                gate_bias=cfg.MODEL.EVENT_FUSION.GATE_BIAS,
                use_reliability_prior=cfg.MODEL.EVENT_FUSION.USE_RELIABILITY_PRIOR,
                reliability_density_power=cfg.MODEL.EVENT_FUSION.RELIABILITY_DENSITY_POWER,
                reliability_temporal_power=cfg.MODEL.EVENT_FUSION.RELIABILITY_TEMPORAL_POWER,
                reliability_polarity_power=cfg.MODEL.EVENT_FUSION.RELIABILITY_POLARITY_POWER,
                reliability_floor=cfg.MODEL.EVENT_FUSION.RELIABILITY_FLOOR,
                reliability_gain=cfg.MODEL.EVENT_FUSION.RELIABILITY_GAIN,
                gate_sparsity_weight=cfg.MODEL.EVENT_FUSION.GATE_SPARSITY_WEIGHT,
                gate_invalid_weight=cfg.MODEL.EVENT_FUSION.GATE_INVALID_WEIGHT,
                log_gate_stats=cfg.MODEL.EVENT_FUSION.LOG_GATE_STATS,
            )
        event_edge_head = None
        if cfg.MODEL.EVENT_EDGE.ENABLED:
            event_edge_head = EventEdgeSemanticHead(
                backbone_shapes,
                in_channels=cfg.MODEL.EVENT_EDGE.IN_CHANNELS,
                hidden_dim=cfg.MODEL.EVENT_EDGE.HIDDEN_DIM,
                use_rgb_feature=cfg.MODEL.EVENT_EDGE.USE_RGB_FEATURE,
                rgb_feature=cfg.MODEL.EVENT_EDGE.RGB_FEATURE,
                detach_rgb_feature=cfg.MODEL.EVENT_EDGE.DETACH_RGB_FEATURE,
                output_stride=cfg.MODEL.EVENT_EDGE.OUTPUT_STRIDE,
                primary_boundary_radius=cfg.MODEL.EVENT_EDGE.PRIMARY_BOUNDARY_RADIUS,
                bce_weight=cfg.MODEL.EVENT_EDGE.BCE_WEIGHT,
                dice_weight=cfg.MODEL.EVENT_EDGE.DICE_WEIGHT,
                pos_weight_max=cfg.MODEL.EVENT_EDGE.POS_WEIGHT_MAX,
                log_edge_stats=cfg.MODEL.EVENT_EDGE.LOG_EDGE_STATS,
                train_only_edge=cfg.MODEL.EVENT_EDGE.TRAIN_ONLY_EDGE,
                class_aware=cfg.MODEL.EVENT_EDGE.CLASS_AWARE,
                num_classes=cfg.MODEL.EVENT_EDGE.NUM_CLASSES,
                class_boundary_radius=cfg.MODEL.EVENT_EDGE.CLASS_BOUNDARY_RADIUS,
                class_bce_weight=cfg.MODEL.EVENT_EDGE.CLASS_BCE_WEIGHT,
                class_pos_weight_max=cfg.MODEL.EVENT_EDGE.CLASS_POS_WEIGHT_MAX,
            )
        event_edge_guide = None
        if cfg.MODEL.EVENT_EDGE_GUIDE.ENABLED:
            event_edge_guide = EventEdgeGuidedAdapter(
                backbone_shapes,
                stage=cfg.MODEL.EVENT_EDGE_GUIDE.STAGE,
                edge_channels=cfg.MODEL.EVENT_EDGE.HIDDEN_DIM,
                init_alpha=cfg.MODEL.EVENT_EDGE_GUIDE.INIT_ALPHA,
                gate_bias=cfg.MODEL.EVENT_EDGE_GUIDE.GATE_BIAS,
                edge_prob_power=cfg.MODEL.EVENT_EDGE_GUIDE.EDGE_PROB_POWER,
                use_event_reliability=cfg.MODEL.EVENT_EDGE_GUIDE.USE_EVENT_RELIABILITY,
                reliability_floor=cfg.MODEL.EVENT_EDGE_GUIDE.RELIABILITY_FLOOR,
                low_light_enabled=cfg.MODEL.EVENT_EDGE_GUIDE.LOW_LIGHT_ENABLED,
                low_light_gain=cfg.MODEL.EVENT_EDGE_GUIDE.LOW_LIGHT_GAIN,
                low_light_luma_threshold=cfg.MODEL.EVENT_EDGE_GUIDE.LOW_LIGHT_LUMA_THRESHOLD,
                low_light_contrast_threshold=cfg.MODEL.EVENT_EDGE_GUIDE.LOW_LIGHT_CONTRAST_THRESHOLD,
                score_predictor_enabled=cfg.MODEL.EVENT_EDGE_GUIDE.SCORE_PREDICTOR_ENABLED,
                score_sparsity_weight=cfg.MODEL.EVENT_EDGE_GUIDE.SCORE_SPARSITY_WEIGHT,
                gate_sparsity_weight=cfg.MODEL.EVENT_EDGE_GUIDE.GATE_SPARSITY_WEIGHT,
                non_boundary_gate_weight=cfg.MODEL.EVENT_EDGE_GUIDE.NON_BOUNDARY_GATE_WEIGHT,
                log_stats=cfg.MODEL.EVENT_EDGE_GUIDE.LOG_STATS,
            )
        early_event_edge_adapter = None
        if cfg.MODEL.EARLY_EVENT_EDGE_ADAPTER.ENABLED:
            early_event_edge_adapter = EarlyEventEdgeAdapter(
                feature_channels=backbone_shapes["res2"].channels,
                event_channels=cfg.MODEL.EARLY_EVENT_EDGE_ADAPTER.IN_CHANNELS,
                stat_channels=cfg.MODEL.EARLY_EVENT_EDGE_ADAPTER.STAT_CHANNELS,
                hidden_dim=cfg.MODEL.EARLY_EVENT_EDGE_ADAPTER.HIDDEN_DIM,
                init_alpha=cfg.MODEL.EARLY_EVENT_EDGE_ADAPTER.INIT_ALPHA,
                alpha_max=cfg.MODEL.EARLY_EVENT_EDGE_ADAPTER.ALPHA_MAX,
                gate_bias=cfg.MODEL.EARLY_EVENT_EDGE_ADAPTER.GATE_BIAS,
                edge_prob_power=cfg.MODEL.EARLY_EVENT_EDGE_ADAPTER.EDGE_PROB_POWER,
                use_event_reliability=cfg.MODEL.EARLY_EVENT_EDGE_ADAPTER.USE_EVENT_RELIABILITY,
                reliability_floor=cfg.MODEL.EARLY_EVENT_EDGE_ADAPTER.RELIABILITY_FLOOR,
                reliability_threshold=cfg.MODEL.EARLY_EVENT_EDGE_ADAPTER.RELIABILITY_THRESHOLD,
                event_edge_threshold=cfg.MODEL.EARLY_EVENT_EDGE_ADAPTER.EVENT_EDGE_THRESHOLD,
                confidence_threshold=cfg.MODEL.EARLY_EVENT_EDGE_ADAPTER.CONFIDENCE_THRESHOLD,
                margin_threshold=cfg.MODEL.EARLY_EVENT_EDGE_ADAPTER.MARGIN_THRESHOLD,
                entropy_threshold=cfg.MODEL.EARLY_EVENT_EDGE_ADAPTER.ENTROPY_THRESHOLD,
                require_boundary=cfg.MODEL.EARLY_EVENT_EDGE_ADAPTER.REQUIRE_BOUNDARY,
                boundary_radius=cfg.MODEL.EARLY_EVENT_EDGE_ADAPTER.BOUNDARY_RADIUS,
                gate_bce_weight=cfg.MODEL.EARLY_EVENT_EDGE_ADAPTER.GATE_BCE_WEIGHT,
                gate_positive_weight=cfg.MODEL.EARLY_EVENT_EDGE_ADAPTER.GATE_POSITIVE_WEIGHT,
                gate_negative_weight=cfg.MODEL.EARLY_EVENT_EDGE_ADAPTER.GATE_NEGATIVE_WEIGHT,
                gate_supervision_detach=cfg.MODEL.EARLY_EVENT_EDGE_ADAPTER.GATE_SUPERVISION_DETACH,
                gate_supervision_target=cfg.MODEL.EARLY_EVENT_EDGE_ADAPTER.GATE_SUPERVISION_TARGET,
                allow_pred_classes=cfg.MODEL.EARLY_EVENT_EDGE_ADAPTER.ALLOW_PRED_CLASSES,
                deny_pred_classes=cfg.MODEL.EARLY_EVENT_EDGE_ADAPTER.DENY_PRED_CLASSES,
                allow_target_classes=cfg.MODEL.EARLY_EVENT_EDGE_ADAPTER.ALLOW_TARGET_CLASSES,
                deny_target_classes=cfg.MODEL.EARLY_EVENT_EDGE_ADAPTER.DENY_TARGET_CLASSES,
                log_stats=cfg.MODEL.EARLY_EVENT_EDGE_ADAPTER.LOG_STATS,
                ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                num_classes=sem_seg_head.num_classes,
            )
        day_event_boundary_refiner = None
        if cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.ENABLED:
            day_event_boundary_refiner = DayEventBoundaryRefiner(
                num_classes=cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.NUM_CLASSES,
                edge_channels=cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.EDGE_CHANNELS,
                hidden_dim=cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.HIDDEN_DIM,
                init_alpha=cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.INIT_ALPHA,
                gate_bias=cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.GATE_BIAS,
                final_gate_scale=cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.FINAL_GATE_SCALE,
                topk_classes=cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.TOPK_CLASSES,
                tau_margin=cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.TAU_MARGIN,
                entropy_threshold=cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.ENTROPY_THRESHOLD,
                uncertain_fraction=cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.UNCERTAIN_FRACTION,
                edge_threshold=cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.EDGE_THRESHOLD,
                edge_prob_power=cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.EDGE_PROB_POWER,
                boundary_radius=cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.BOUNDARY_RADIUS,
                boundary_weight=cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.BOUNDARY_WEIGHT,
                use_event_reliability=cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.USE_EVENT_RELIABILITY,
                use_rgb_uncertainty=cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.USE_RGB_UNCERTAINTY,
                use_semantic_boundary=cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.USE_SEMANTIC_BOUNDARY,
                require_uncertain_boundary_intersection=(
                    cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.REQUIRE_UNCERTAIN_BOUNDARY_INTERSECTION
                ),
                require_event_active=cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.REQUIRE_EVENT_ACTIVE,
                event_active_source=cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.EVENT_ACTIVE_SOURCE,
                event_active_threshold=cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.EVENT_ACTIVE_THRESHOLD,
                preserve_confidence_threshold=(
                    cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.PRESERVE_CONFIDENCE_THRESHOLD
                ),
                preserve_margin_threshold=cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.PRESERVE_MARGIN_THRESHOLD,
                min_confidence_for_correction=(
                    cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.MIN_CONFIDENCE_FOR_CORRECTION
                ),
                min_margin_for_correction=(
                    cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.MIN_MARGIN_FOR_CORRECTION
                ),
                loss_boundary_ce_weight=cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.LOSS_BOUNDARY_CE_WEIGHT,
                loss_uncertain_ce_weight=cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.LOSS_UNCERTAIN_CE_WEIGHT,
                loss_preserve_kl_weight=cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.LOSS_PRESERVE_KL_WEIGHT,
                loss_gate_sparse_weight=cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.LOSS_GATE_SPARSE_WEIGHT,
                loss_gate_invalid_weight=cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.LOSS_GATE_INVALID_WEIGHT,
                loss_delta_non_boundary_weight=(
                    cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.LOSS_DELTA_NON_BOUNDARY_WEIGHT
                ),
                loss_candidate_ce_weight=cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.LOSS_CANDIDATE_CE_WEIGHT,
                candidate_ce_repair_positive_only=(
                    cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.CANDIDATE_CE_REPAIR_POSITIVE_ONLY
                ),
                candidate_logit_scale=cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.CANDIDATE_LOGIT_SCALE,
                loss_allowed_only=cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.LOSS_ALLOWED_ONLY,
                loss_allowed_soft_threshold=(
                    cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.LOSS_ALLOWED_SOFT_THRESHOLD
                ),
                class_gate_weights=cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.CLASS_GATE_WEIGHTS,
                class_gate_loss_threshold=(
                    cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.CLASS_GATE_LOSS_THRESHOLD
                ),
                score_predictor_enabled=cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.SCORE_PREDICTOR_ENABLED,
                score_bce_weight=cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.SCORE_BCE_WEIGHT,
                score_sparsity_weight=cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.SCORE_SPARSITY_WEIGHT,
                score_positive_weight=cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.SCORE_POSITIVE_WEIGHT,
                score_negative_weight=cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.SCORE_NEGATIVE_WEIGHT,
                edge_only_correction=cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.EDGE_ONLY_CORRECTION,
                selective_repair_gate_enabled=(
                    cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.SELECTIVE_REPAIR_GATE_ENABLED
                ),
                loss_repair_gate_bce_weight=(
                    cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.LOSS_REPAIR_GATE_BCE_WEIGHT
                ),
                repair_gate_positive_weight=(
                    cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.REPAIR_GATE_POSITIVE_WEIGHT
                ),
                repair_gate_negative_weight=(
                    cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.REPAIR_GATE_NEGATIVE_WEIGHT
                ),
                repair_require_target_in_topk=(
                    cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.REPAIR_REQUIRE_TARGET_IN_TOPK
                ),
                repair_supervise_score=cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.REPAIR_SUPERVISE_SCORE,
                repair_class_weights=cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.REPAIR_CLASS_WEIGHTS,
                pair_aware_enabled=cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.PAIR_AWARE_ENABLED,
                pair_allow_weights=cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.PAIR_ALLOW_WEIGHTS,
                pair_suppress_weights=cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.PAIR_SUPPRESS_WEIGHTS,
                pair_weight_default=cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.PAIR_WEIGHT_DEFAULT,
                hard_pair_gate_enabled=(
                    cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.HARD_PAIR_GATE_ENABLED
                ),
                hard_pair_gate_threshold=(
                    cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.HARD_PAIR_GATE_THRESHOLD
                ),
                hard_pair_gate_include_identity=(
                    cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.HARD_PAIR_GATE_INCLUDE_IDENTITY
                ),
                hard_pair_suppress_enabled=(
                    cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.HARD_PAIR_SUPPRESS_ENABLED
                ),
                loss_pair_suppress_gate_weight=(
                    cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.LOSS_PAIR_SUPPRESS_GATE_WEIGHT
                ),
                loss_pair_suppress_delta_weight=(
                    cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.LOSS_PAIR_SUPPRESS_DELTA_WEIGHT
                ),
                detach_base_prob=cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.DETACH_BASE_PROB,
                skip_mask2former_loss=cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.SKIP_MASK2FORMER_LOSS,
                log_stats=cfg.MODEL.DAY_EVENT_BOUNDARY_REFINER.LOG_STATS,
                ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            )

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "event_fusion": event_fusion,
            "event_edge_head": event_edge_head,
            "event_edge_guide": event_edge_guide,
            "early_event_edge_adapter": early_event_edge_adapter,
            "day_event_boundary_refiner": day_event_boundary_refiner,
            "event_preserve_enabled": cfg.MODEL.EVENT_PRESERVE.ENABLED,
            "event_preserve_weight": cfg.MODEL.EVENT_PRESERVE.WEIGHT,
            "event_preserve_confidence_threshold": cfg.MODEL.EVENT_PRESERVE.CONFIDENCE_THRESHOLD,
            "event_preserve_margin_threshold": cfg.MODEL.EVENT_PRESERVE.MARGIN_THRESHOLD,
            "event_preserve_event_edge_threshold": cfg.MODEL.EVENT_PRESERVE.EVENT_EDGE_THRESHOLD,
            "event_preserve_use_boundary_target": cfg.MODEL.EVENT_PRESERVE.USE_BOUNDARY_TARGET,
            "event_preserve_use_event_edge": cfg.MODEL.EVENT_PRESERVE.USE_EVENT_EDGE,
            "event_preserve_log_stats": cfg.MODEL.EVENT_PRESERVE.LOG_STATS,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        raw_images = [x["image"].to(self.device).float() for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in raw_images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        raw_image_list = ImageList.from_tensors(raw_images, self.size_divisibility)

        event_stats_for_guide = None
        early_event_edges = None
        if self.early_event_edge_adapter is not None:
            event_edges = [
                x.get(
                    "event_edge",
                    torch.zeros(
                        (
                            self.early_event_edge_adapter.event_channels,
                            x["image"].shape[-2],
                            x["image"].shape[-1],
                        ),
                        dtype=torch.float32,
                    ),
                ).to(self.device)
                for x in batched_inputs
            ]
            event_stats = [
                x.get(
                    "event_stats",
                    torch.zeros(
                        (
                            self.early_event_edge_adapter.stat_channels,
                            x["image"].shape[-2],
                            x["image"].shape[-1],
                        ),
                        dtype=torch.float32,
                    ),
                ).to(self.device)
                for x in batched_inputs
            ]
            early_event_edges = ImageList.from_tensors(event_edges, self.size_divisibility).tensor
            event_stats_for_guide = ImageList.from_tensors(event_stats, self.size_divisibility).tensor
            features = self.backbone(
                images.tensor,
                early_event_adapter=self.early_event_edge_adapter,
                event_edge=early_event_edges,
                event_stats=event_stats_for_guide,
            )
        else:
            features = self.backbone(images.tensor)
        if self.event_fusion is not None:
            events = [
                x.get(
                    "event",
                    torch.zeros(
                        (self.event_fusion.event_channels, x["image"].shape[-2], x["image"].shape[-1]),
                        dtype=torch.float32,
                    ),
                ).to(self.device)
                for x in batched_inputs
            ]
            event_stats = [
                x.get(
                    "event_stats",
                    torch.zeros(
                        (self.event_fusion.stat_channels, x["image"].shape[-2], x["image"].shape[-1]),
                        dtype=torch.float32,
                    ),
                ).to(self.device)
                for x in batched_inputs
            ]
            events = ImageList.from_tensors(events, self.size_divisibility).tensor
            event_stats = ImageList.from_tensors(event_stats, self.size_divisibility).tensor
            event_stats_for_guide = event_stats
            features = self.event_fusion(features, events, event_stats)
            if self.training:
                self.event_fusion.put_gate_stats()
        event_edge_losses = {}
        edge_outputs = None
        boundary_targets = None
        early_boundary_targets = None
        if self.training and self.early_event_edge_adapter is not None:
            boundary_key = f"boundary_r{self.early_event_edge_adapter.boundary_radius}"
            early_boundary_targets = [
                x.get(
                    boundary_key,
                    torch.zeros(
                        (x["image"].shape[-2], x["image"].shape[-1]),
                        dtype=torch.float32,
                    ),
                ).to(self.device)
                for x in batched_inputs
            ]
            early_boundary_targets = ImageList.from_tensors(
                early_boundary_targets,
                self.size_divisibility,
            ).tensor
        if self.event_edge_head is not None:
            event_edges = [
                x.get(
                    "event_edge",
                    torch.zeros(
                        (
                            self.event_edge_head.in_channels,
                            x["image"].shape[-2],
                            x["image"].shape[-1],
                        ),
                        dtype=torch.float32,
                    ),
                ).to(self.device)
                for x in batched_inputs
            ]
            event_edges = ImageList.from_tensors(event_edges, self.size_divisibility).tensor
            class_boundary_targets = None
            if self.training:
                boundary_key = f"boundary_r{self.event_edge_head.primary_boundary_radius}"
                boundary_targets = [
                    x.get(
                        boundary_key,
                        torch.zeros(
                            (x["image"].shape[-2], x["image"].shape[-1]),
                            dtype=torch.float32,
                        ),
                    ).to(self.device)
                    for x in batched_inputs
                ]
                boundary_targets = ImageList.from_tensors(boundary_targets, self.size_divisibility).tensor
                if self.event_edge_head.class_aware:
                    class_boundary_key = f"class_boundary_r{self.event_edge_head.class_boundary_radius}"
                    class_boundary_targets = [
                        x.get(
                            class_boundary_key,
                            torch.zeros(
                                (
                                    self.event_edge_head.num_classes,
                                    x["image"].shape[-2],
                                    x["image"].shape[-1],
                                ),
                                dtype=torch.float32,
                            ),
                        ).to(self.device)
                        for x in batched_inputs
                    ]
                    class_boundary_targets = ImageList.from_tensors(
                        class_boundary_targets,
                        self.size_divisibility,
                    ).tensor
            if early_boundary_targets is not None:
                early_boundary_targets = boundary_targets
            edge_outputs, event_edge_losses = self.event_edge_head(
                features,
                event_edges,
                boundary_targets,
                class_boundary_targets,
            )
            if self.training:
                self.event_edge_head.put_edge_stats()
        preserve_teacher_prob = None
        if (
            self.training
            and self.event_preserve_enabled
            and self.event_edge_guide is not None
            and edge_outputs is not None
            and self.event_preserve_weight > 0
        ):
            preserve_teacher_prob = self._event_preserve_teacher_prob(features)
        if self.event_edge_guide is not None and edge_outputs is not None:
            if event_stats_for_guide is None:
                event_stats = [
                    x.get(
                        "event_stats",
                        torch.zeros(
                            (4, x["image"].shape[-2], x["image"].shape[-1]),
                            dtype=torch.float32,
                        ),
                    ).to(self.device)
                    for x in batched_inputs
                ]
                event_stats_for_guide = ImageList.from_tensors(event_stats, self.size_divisibility).tensor
            features = self.event_edge_guide(
                features,
                edge_outputs,
                event_stats=event_stats_for_guide,
                raw_images=raw_image_list.tensor,
                boundary_target=boundary_targets,
            )
            if self.training:
                self.event_edge_guide.put_guide_stats()
        outputs = self.sem_seg_head(features)
        early_event_losses = {}
        if self.training and self.early_event_edge_adapter is not None:
            sem_seg_targets = [
                x.get(
                    "sem_seg",
                    torch.full(
                        x["image"].shape[-2:],
                        fill_value=self.early_event_edge_adapter.ignore_label,
                        dtype=torch.long,
                    ),
                ).to(self.device)
                for x in batched_inputs
            ]
            sem_seg_targets = ImageList.from_tensors(
                sem_seg_targets,
                self.size_divisibility,
                pad_value=self.early_event_edge_adapter.ignore_label,
            ).tensor
            early_event_losses = self.early_event_edge_adapter.gate_supervision_loss(
                outputs,
                sem_seg_targets,
                boundary_targets=early_boundary_targets,
            )
            self.early_event_edge_adapter.put_stats()
        day_refiner_losses = {}
        sem_seg_override = None
        if self.day_event_boundary_refiner is not None and edge_outputs is not None:
            if event_stats_for_guide is None:
                event_stats = [
                    x.get(
                        "event_stats",
                        torch.zeros(
                            (4, x["image"].shape[-2], x["image"].shape[-1]),
                            dtype=torch.float32,
                        ),
                    ).to(self.device)
                    for x in batched_inputs
                ]
                event_stats_for_guide = ImageList.from_tensors(event_stats, self.size_divisibility).tensor
            if self.training:
                sem_seg_targets = [
                    x.get(
                        "sem_seg",
                        torch.full(
                            x["image"].shape[-2:],
                            fill_value=self.day_event_boundary_refiner.ignore_label,
                            dtype=torch.long,
                        ),
                    ).to(self.device)
                    for x in batched_inputs
                ]
                sem_seg_targets = ImageList.from_tensors(
                    sem_seg_targets,
                    self.size_divisibility,
                    pad_value=self.day_event_boundary_refiner.ignore_label,
                ).tensor
            else:
                sem_seg_targets = None
            refiner_outputs = dict(outputs)
            if refiner_outputs["pred_masks"].shape[-2:] != images.tensor.shape[-2:]:
                refiner_outputs["pred_masks"] = F.interpolate(
                    outputs["pred_masks"],
                    size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                    mode="bilinear",
                    align_corners=False,
                )
            day_refiner_outputs = self.day_event_boundary_refiner(
                refiner_outputs,
                edge_outputs,
                event_stats=event_stats_for_guide,
                boundary_targets=boundary_targets,
                sem_seg_targets=sem_seg_targets,
            )
            sem_seg_override = day_refiner_outputs["prob"]
            day_refiner_losses = day_refiner_outputs["losses"]
            if self.training:
                self.day_event_boundary_refiner.put_stats()

        if self.training:
            # mask classification target
            if (
                self.day_event_boundary_refiner is not None
                and self.day_event_boundary_refiner.skip_mask2former_loss
            ):
                losses = {}
                mask2former_loss_keys = tuple()
            elif "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
                # bipartite matching-based loss
                losses = self.criterion(outputs, targets)

                for k in list(losses.keys()):
                    if k in self.criterion.weight_dict:
                        losses[k] *= self.criterion.weight_dict[k]
                    else:
                        # remove this loss if not specified in `weight_dict`
                        losses.pop(k)
                mask2former_loss_keys = tuple(losses.keys())
            else:
                losses = {}
                mask2former_loss_keys = tuple()
            if self.event_fusion is not None:
                losses.update(self.event_fusion.extra_losses())
            if event_edge_losses:
                losses.update(event_edge_losses)
            if self.event_edge_guide is not None:
                losses.update(self.event_edge_guide.extra_losses())
            if early_event_losses:
                losses.update(early_event_losses)
            if day_refiner_losses:
                losses.update(day_refiner_losses)
            if preserve_teacher_prob is not None:
                preserve_losses = self._event_preserve_losses(
                    outputs,
                    preserve_teacher_prob,
                    boundary_targets=boundary_targets,
                    edge_outputs=edge_outputs,
                )
                losses.update(preserve_losses)
                self._put_event_preserve_stats()
            self._put_loss_group_stats(losses, mask2former_loss_keys)
            return losses
        else:
            if sem_seg_override is not None:
                processed_results = []
                for sem_seg_result, input_per_image, image_size in zip(
                    sem_seg_override,
                    batched_inputs,
                    images.image_sizes,
                ):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    if self.sem_seg_postprocess_before_inference:
                        sem_seg_result = retry_if_cuda_oom(sem_seg_postprocess)(
                            sem_seg_result,
                            image_size,
                            height,
                            width,
                        )
                    elif sem_seg_result.shape[-2:] != (height, width):
                        sem_seg_result = retry_if_cuda_oom(sem_seg_postprocess)(
                            sem_seg_result,
                            image_size,
                            height,
                            width,
                        )
                    processed_results.append({"sem_seg": sem_seg_result})
                return processed_results
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            del outputs

            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})

                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)

                # semantic segmentation inference
                if self.semantic_on:
                    r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                    processed_results[-1]["sem_seg"] = r

                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["panoptic_seg"] = panoptic_r
                
                # instance segmentation inference
                if self.instance_on:
                    instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["instances"] = instance_r

            return processed_results

    def _semantic_probs_from_outputs(self, outputs):
        mask_cls = F.softmax(outputs["pred_logits"], dim=-1)[..., :-1]
        mask_pred = outputs["pred_masks"].sigmoid()
        sem_scores = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred).clamp_min(1e-8)
        return sem_scores / sem_scores.sum(dim=1, keepdim=True).clamp_min(1e-8)

    def _event_preserve_teacher_prob(self, features):
        was_training = self.sem_seg_head.training
        self.sem_seg_head.eval()
        with torch.no_grad():
            outputs = self.sem_seg_head(features)
            teacher_prob = self._semantic_probs_from_outputs(outputs).detach()
        if was_training:
            self.sem_seg_head.train()
        return teacher_prob

    def _event_preserve_safe_mask(self, teacher_prob, boundary_targets=None, edge_outputs=None):
        topk = teacher_prob.topk(k=min(2, teacher_prob.shape[1]), dim=1).values
        confidence = topk[:, 0:1]
        if topk.shape[1] > 1:
            margin = topk[:, 0:1] - topk[:, 1:2]
        else:
            margin = torch.ones_like(confidence)
        safe = (
            (confidence >= self.event_preserve_confidence_threshold)
            & (margin >= self.event_preserve_margin_threshold)
        )

        if self.event_preserve_use_boundary_target and boundary_targets is not None:
            boundary = boundary_targets.to(device=teacher_prob.device, dtype=teacher_prob.dtype)
            if boundary.ndim == 3:
                boundary = boundary[:, None]
            boundary = F.interpolate(boundary, size=teacher_prob.shape[-2:], mode="nearest").clamp(0.0, 1.0)
            safe = safe & (boundary < 0.5)
        else:
            boundary = None

        if self.event_preserve_use_event_edge and edge_outputs is not None:
            edge_prob = edge_outputs["prob"].detach().to(device=teacher_prob.device, dtype=teacher_prob.dtype)
            edge_prob = F.interpolate(
                edge_prob,
                size=teacher_prob.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).clamp(0.0, 1.0)
            safe = safe & (edge_prob < self.event_preserve_event_edge_threshold)
        else:
            edge_prob = None

        with torch.no_grad():
            stats = {
                "teacher_confidence_mean": confidence.mean().detach(),
                "teacher_margin_mean": margin.mean().detach(),
                "safe_fraction": safe.float().mean().detach(),
            }
            if boundary is not None:
                stats["boundary_fraction"] = (boundary >= 0.5).float().mean().detach()
            if edge_prob is not None:
                stats["event_edge_fraction"] = (
                    edge_prob >= self.event_preserve_event_edge_threshold
                ).float().mean().detach()
        return safe.to(dtype=teacher_prob.dtype), stats

    def _event_preserve_losses(self, outputs, teacher_prob, boundary_targets=None, edge_outputs=None):
        student_prob = self._semantic_probs_from_outputs(outputs)
        if teacher_prob.shape[-2:] != student_prob.shape[-2:]:
            teacher_prob = F.interpolate(
                teacher_prob,
                size=student_prob.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            teacher_prob = teacher_prob / teacher_prob.sum(dim=1, keepdim=True).clamp_min(1e-8)
        safe_mask, stats = self._event_preserve_safe_mask(
            teacher_prob,
            boundary_targets=boundary_targets,
            edge_outputs=edge_outputs,
        )
        kl_map = F.kl_div(
            student_prob.clamp_min(1e-8).log(),
            teacher_prob.clamp_min(1e-8),
            reduction="none",
        ).sum(dim=1, keepdim=True)
        denom = safe_mask.sum().clamp_min(1.0)
        loss = (kl_map * safe_mask).sum() / denom
        self._last_event_preserve_stats = {
            **stats,
            "loss": loss.detach(),
            "safe_pixels": denom.detach(),
        }
        return {"loss_event_preserve_safe": loss * self.event_preserve_weight}

    def _put_event_preserve_stats(self):
        if not self.event_preserve_log_stats or not self._last_event_preserve_stats:
            return
        try:
            storage = get_event_storage()
        except AssertionError:
            return
        for name, value in self._last_event_preserve_stats.items():
            storage.put_scalar(f"event_preserve/{name}", value.item(), smoothing_hint=False)

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets

    def _put_loss_group_stats(self, losses, mask2former_loss_keys):
        if not losses:
            return
        try:
            storage = get_event_storage()
        except AssertionError:
            return

        groups = {
            "mask2former": tuple(mask2former_loss_keys),
            "edge": (
                "loss_event_edge_bce",
                "loss_event_edge_dice",
                "loss_event_class_edge_bce",
            ),
            "boundary_sem": (
                "loss_boundary_sem",
                "loss_event_boundary_sem",
                "loss_refine_boundary_sem",
                "loss_day_boundary_ce",
                "loss_day_uncertain_ce",
            ),
            "preserve_safe": (
                "loss_preserve_safe",
                "loss_event_preserve_safe",
                "loss_refine_preserve_safe",
                "loss_day_preserve_kl",
            ),
            "gate_regularize": (
                "loss_event_gate_sparse",
                "loss_event_gate_invalid",
                "loss_event_edge_score_sparse",
                "loss_event_edge_guide_sparse",
                "loss_event_edge_guide_non_boundary",
                "loss_gate_regularize",
                "loss_gate_day_suppression",
                "loss_day_event_score_bce",
                "loss_day_event_score_sparse",
                "loss_day_repair_gate_bce",
                "loss_day_gate_sparse",
                "loss_day_gate_invalid",
                "loss_day_pair_suppress_gate",
                "loss_day_pair_suppress_delta",
                "loss_day_delta_non_boundary",
                "loss_early_event_gate_bce",
            ),
        }

        reference = next(iter(losses.values()))
        zero = reference.detach().new_zeros(())
        total = zero
        for value in losses.values():
            total = total + value.detach()
        storage.put_scalar("loss_group/total", total.item(), smoothing_hint=True)

        for name, keys in groups.items():
            value, found = _sum_losses(losses, keys)
            if found:
                value = value.detach()
            else:
                value = zero
            storage.put_scalar(f"loss_group/{name}", value.item(), smoothing_hint=True)

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // self.sem_seg_head.num_classes
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices]

        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result
