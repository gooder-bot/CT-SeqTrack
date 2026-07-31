"""Observation-first proposal fusion for CT-SeqTrack v2."""

import math

import torch
from torch import nn


def _logit(probability):
    probability = min(max(float(probability), 1e-6), 1.0 - 1e-6)
    return math.log(probability / (1.0 - probability))


class ProposalFusionGate(nn.Module):
    """Predict a bounded interpolation weight for a motion proposal.

    The observation proposal always remains the identity path.  The gate only
    controls a separately norm-bounded innovation toward the continuous-time
    prior.  Its final layer starts at a small, observation-biased constant,
    which avoids the unstable 50/50 fusion used by many generic gates.
    """

    def __init__(
            self,
            observation_dim=256,
            dynamics_dim=128,
            observation_stats_dim=5,
            hidden_dim=64,
            context_dim=16,
            max_alpha=0.75,
            init_alpha=0.25,
            time_scale=0.5,
            detach_context=True):
        super().__init__()
        self.max_alpha = float(max_alpha)
        self.init_alpha = float(init_alpha)
        self.time_scale = max(float(time_scale), 1e-6)
        self.detach_context = bool(detach_context)
        if not 0.0 < self.max_alpha <= 1.0:
            raise ValueError("ct_fusion_max_alpha must be in (0, 1]")
        if not 0.0 <= self.init_alpha < self.max_alpha:
            raise ValueError(
                "ct_fusion_init_alpha must be in [0, ct_fusion_max_alpha)")

        self.observation_projection = nn.Sequential(
            nn.Linear(int(observation_dim), int(context_dim)),
            nn.ReLU(),
        )
        self.dynamics_projection = nn.Sequential(
            nn.Linear(int(dynamics_dim), int(context_dim)),
            nn.ReLU(),
        )
        scalar_dim = int(observation_stats_dim) + 4
        self.gate = nn.Sequential(
            nn.Linear(2 * int(context_dim) + scalar_dim, int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), 1),
        )

        final = self.gate[-1]
        nn.init.zeros_(final.weight)
        initial_ratio = max(self.init_alpha / self.max_alpha, 1e-6)
        nn.init.constant_(final.bias, _logit(initial_ratio))

    @staticmethod
    def _batch_scalar(value, reference, default=0.0):
        batch_size = reference.shape[0]
        if value is None:
            return reference.new_full((batch_size, 1), float(default))
        if not torch.is_tensor(value):
            value = torch.as_tensor(
                value, device=reference.device, dtype=reference.dtype)
        value = value.to(device=reference.device, dtype=reference.dtype)
        # Validation is recursive and uses batch size 1. Its scalar fields can
        # therefore arrive as [1], [1, 1], or a rank-0 tensor. Flatten before
        # broadcasting so repeat() never receives fewer dimensions than the
        # input tensor.
        value = value.reshape(-1)
        if value.numel() == 1:
            value = value.repeat(batch_size)
        elif value.numel() != batch_size:
            raise ValueError(
                "fusion scalar must contain either one value or one value "
                f"per batch item, got {value.numel()} for batch {batch_size}")
        return value.reshape(batch_size, 1)

    def forward(
            self,
            observation_feature,
            dynamics_feature,
            observation_displacement,
            dynamics_displacement,
            observation_stats,
            dynamics_valid,
            current_delta_t,
            search_expansion_ratio=None):
        if observation_displacement.shape != dynamics_displacement.shape:
            raise ValueError("observation and dynamics proposals must match")

        obs_context = observation_feature
        dyn_context = dynamics_feature
        stats_context = observation_stats
        if self.detach_context:
            obs_context = obs_context.detach()
            dyn_context = dyn_context.detach()
            stats_context = stats_context.detach()

        disagreement = torch.linalg.norm(
            dynamics_displacement - observation_displacement.detach(),
            dim=1,
            keepdim=True,
        )
        query_gap = self._batch_scalar(
            current_delta_t, observation_displacement,
            default=self.time_scale,
        ) / self.time_scale
        expansion_ratio = self._batch_scalar(
            search_expansion_ratio, observation_displacement, default=0.0)
        valid = self._batch_scalar(
            dynamics_valid, observation_displacement, default=0.0)
        valid = (valid > 0).to(observation_displacement.dtype)

        gate_input = torch.cat((
            self.observation_projection(obs_context),
            self.dynamics_projection(dyn_context),
            stats_context,
            torch.log1p(disagreement),
            query_gap,
            expansion_ratio,
            valid,
        ), dim=1)
        gate_input = torch.nan_to_num(
            gate_input, nan=0.0, posinf=0.0, neginf=0.0)
        nominal_alpha = self.max_alpha * torch.sigmoid(self.gate(gate_input))
        effective_alpha = nominal_alpha * valid
        return effective_alpha, {
            # Keep the learned gate output separate from runtime safety masks.
            # The caller records the coefficient that was actually applied
            # after validity and warmup as ``ct_fusion_alpha_applied``.
            "ct_fusion_alpha": nominal_alpha.squeeze(1),
            "ct_fusion_disagreement": disagreement.squeeze(1),
            "ct_fusion_search_ratio": expansion_ratio.squeeze(1),
            "ct_fusion_valid": valid.squeeze(1),
        }


class ReliabilityGatedProposalFusion(nn.Module):
    """Fuse a physical xy prior after observation-based box refinement.

    Every context tensor is detached before entering the gate.  Consequently,
    the fused-box objective can update only this module and cannot perturb the
    observation tracker or turn the physical prior into an anchor-error head.
    """

    def __init__(
            self,
            observation_dim=256,
            motion_dim=128,
            observation_stats_dim=5,
            hidden_dim=64,
            context_dim=32,
            max_alpha=0.5,
            init_probability=0.01,
            radius_base=0.25,
            radius_per_second=0.5,
            radius_max=1.25,
            time_scale=0.5):
        super().__init__()
        self.max_alpha = float(max_alpha)
        self.radius_base = float(radius_base)
        self.radius_per_second = float(radius_per_second)
        self.radius_max = float(radius_max)
        self.time_scale = max(float(time_scale), 1e-6)
        if not 0.0 < self.max_alpha <= 1.0:
            raise ValueError("motion_v3_alpha_max must be in (0, 1]")
        if not 0.0 < float(init_probability) < 1.0:
            raise ValueError("motion_v3_gate_init_probability must be in (0, 1)")
        if self.radius_base < 0 or self.radius_per_second < 0:
            raise ValueError("motion_v3 fusion radii must be non-negative")
        if self.radius_max <= 0 or self.radius_base > self.radius_max:
            raise ValueError("motion_v3_radius_max must cover radius_base")

        self.observation_projection = nn.Sequential(
            nn.Linear(int(observation_dim), int(context_dim)),
            nn.ReLU(inplace=True),
        )
        self.motion_projection = nn.Sequential(
            nn.Linear(int(motion_dim), int(context_dim)),
            nn.ReLU(inplace=True),
        )
        # obs stats, prior velocity xy, innovation xy/norm, query dt,
        # gap ratio, history-valid ratio, and prior-valid flag.
        scalar_dim = int(observation_stats_dim) + 9
        self.gate = nn.Sequential(
            nn.Linear(2 * int(context_dim) + scalar_dim, int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, _logit(init_probability))

    @staticmethod
    def _batch_scalar(value, reference, default=0.0):
        batch_size = reference.shape[0]
        if value is None:
            return reference.new_full((batch_size, 1), float(default))
        if not torch.is_tensor(value):
            value = torch.as_tensor(
                value, device=reference.device, dtype=reference.dtype)
        value = value.to(device=reference.device, dtype=reference.dtype).reshape(-1)
        if value.numel() == 1:
            value = value.repeat(batch_size)
        elif value.numel() != batch_size:
            raise ValueError(
                "motion_v3 scalar must contain one value or one per batch item")
        return value.reshape(batch_size, 1)

    @staticmethod
    def _clip_norm(vector, max_norm, eps=1e-6):
        norm = torch.linalg.norm(vector, dim=1, keepdim=True)
        scale = torch.minimum(
            torch.ones_like(norm), max_norm / torch.clamp(norm, min=eps))
        return vector * scale, norm

    def forward(
            self,
            observation_box,
            observation_feature,
            observation_stats,
            motion_feature,
            prior_xy,
            prior_velocity_xy,
            prior_valid,
            gap_ratio,
            history_valid_ratio,
            current_delta_t,
            enabled_scale=1.0):
        if observation_box.dim() != 2 or observation_box.shape[1] != 4:
            raise ValueError("observation_box must have shape [B,4]")
        if prior_xy.shape != observation_box[:, :2].shape:
            raise ValueError("motion prior must have shape [B,2]")
        enabled_scale = float(enabled_scale)
        if not 0.0 <= enabled_scale <= 1.0:
            raise ValueError("motion_v3 enabled_scale must be in [0,1]")

        observation_detached = observation_box.detach()
        observation_xy = observation_detached[:, :2]
        prior_xy = prior_xy.detach()
        prior_velocity_xy = prior_velocity_xy.detach()
        motion_feature = motion_feature.detach()
        observation_feature = observation_feature.detach()
        observation_stats = observation_stats.detach()

        # Any non-finite auxiliary evidence disables the correction for that
        # sample.  The observation proposal remains the exact identity path.
        auxiliary_finite = (
            torch.isfinite(prior_xy).all(dim=1)
            & torch.isfinite(prior_velocity_xy).all(dim=1)
            & torch.isfinite(motion_feature).all(dim=1)
            & torch.isfinite(observation_feature).all(dim=1)
            & torch.isfinite(observation_stats).all(dim=1)
        ).reshape(-1, 1)
        safe_prior_xy = torch.where(
            torch.isfinite(prior_xy), prior_xy, observation_xy)
        prior_velocity_xy = torch.nan_to_num(
            prior_velocity_xy, nan=0.0, posinf=0.0, neginf=0.0)
        motion_feature = torch.nan_to_num(
            motion_feature, nan=0.0, posinf=0.0, neginf=0.0)
        observation_feature = torch.nan_to_num(
            observation_feature, nan=0.0, posinf=0.0, neginf=0.0)
        observation_stats = torch.nan_to_num(
            observation_stats, nan=0.0, posinf=0.0, neginf=0.0)

        valid = self._batch_scalar(
            prior_valid, observation_xy, default=0.0)
        valid = (
            (valid > 0) & auxiliary_finite
        ).to(observation_xy.dtype)
        query_dt = self._batch_scalar(
            current_delta_t, observation_xy, default=self.time_scale)
        query_dt_finite = torch.isfinite(query_dt)
        query_dt = torch.nan_to_num(
            query_dt, nan=self.time_scale,
            posinf=self.time_scale, neginf=self.time_scale)
        query_dt = torch.clamp(query_dt, min=1e-6)
        gap_ratio = self._batch_scalar(
            gap_ratio, observation_xy, default=1.0)
        history_valid_ratio = self._batch_scalar(
            history_valid_ratio, observation_xy, default=0.0)
        scalar_finite = (
            query_dt_finite
            & torch.isfinite(gap_ratio)
            & torch.isfinite(history_valid_ratio)
        )
        valid = valid * scalar_finite.to(observation_xy.dtype)
        gap_ratio = torch.nan_to_num(
            gap_ratio, nan=1.0, posinf=1.0, neginf=1.0)
        history_valid_ratio = torch.nan_to_num(
            history_valid_ratio, nan=0.0, posinf=0.0, neginf=0.0)

        innovation_raw = safe_prior_xy - observation_xy
        radius = torch.clamp(
            self.radius_base + self.radius_per_second * query_dt,
            min=self.radius_base,
            max=self.radius_max,
        )
        innovation, innovation_norm = self._clip_norm(
            innovation_raw, radius)
        gate_input = torch.cat((
            self.observation_projection(observation_feature),
            self.motion_projection(motion_feature),
            observation_stats,
            prior_velocity_xy,
            innovation_raw,
            torch.log1p(innovation_norm),
            torch.log1p(query_dt / self.time_scale),
            gap_ratio,
            history_valid_ratio,
            valid,
        ), dim=1)
        gate_input = torch.nan_to_num(
            gate_input, nan=0.0, posinf=0.0, neginf=0.0)
        probability = torch.sigmoid(self.gate(gate_input))
        nominal_alpha = self.max_alpha * probability
        applied_alpha = nominal_alpha * valid * enabled_scale
        correction_xy = applied_alpha * innovation
        final_xy = observation_xy + correction_xy
        final_box = torch.cat((final_xy, observation_detached[:, 2:]), dim=1)
        return final_box, {
            "motion_gate_probability": probability.squeeze(1),
            "motion_gate_alpha": applied_alpha.squeeze(1),
            "motion_gate_nominal_alpha": nominal_alpha.squeeze(1),
            "motion_correction_xy": correction_xy,
            "motion_innovation_xy": innovation_raw,
            "motion_innovation_norm": innovation_norm.squeeze(1),
            "motion_innovation_clipped_norm": torch.linalg.norm(
                innovation, dim=1),
            "motion_fusion_radius": radius.squeeze(1),
            "motion_fusion_clip_rate": (
                innovation_norm > radius).to(observation_xy.dtype).squeeze(1),
            "motion_fusion_valid": valid.squeeze(1),
            "motion_fusion_enabled_scale": observation_xy.new_tensor(
                enabled_scale),
        }
