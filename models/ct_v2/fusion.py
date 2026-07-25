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
        if value.numel() == 1:
            value = value.repeat(batch_size)
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
        alpha = self.max_alpha * torch.sigmoid(self.gate(gate_input))
        alpha = alpha * valid
        return alpha, {
            "ct_fusion_alpha": alpha.squeeze(1),
            "ct_fusion_disagreement": disagreement.squeeze(1),
            "ct_fusion_search_ratio": expansion_ratio.squeeze(1),
            "ct_fusion_valid": valid.squeeze(1),
        }
