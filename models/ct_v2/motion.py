"""Ordered, observation-safe trajectory components for CT-SeqTrack v2."""

import math

import torch
from torch import nn

from models.dynamics import DynamicsEncoder, wrap_angle


class ContinuousTimeMotionEncoder(DynamicsEncoder):
    """Legacy CT-v2 encoder kept for historical checkpoint compatibility."""


class OrderedTrajectoryEncoder(nn.Module):
    """Encode recent-to-old box transitions in their true chronological order.

    A bounded kinematic state supplies a useful cold-start prediction.  A GRU
    learns only the residual, and its prediction head is zero initialized.
    Unlike the legacy mean/max pooling, reversing the acceleration pattern
    changes the representation and the predicted residual.
    """

    def __init__(
            self,
            hidden_dim=128,
            step_dim=64,
            eps=1e-3,
            time_scale=0.5,
            residual_velocity_scale=4.0,
            initial_sigma=0.5):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.step_dim = int(step_dim)
        self.eps = float(eps)
        self.time_scale = max(float(time_scale), self.eps)
        self.residual_velocity_scale = float(residual_velocity_scale)
        if self.hidden_dim <= 0 or self.step_dim <= 0:
            raise ValueError("trajectory encoder dimensions must be positive")
        if self.residual_velocity_scale <= 0:
            raise ValueError("trajectory residual scale must be positive")
        if initial_sigma <= 0:
            raise ValueError("trajectory initial sigma must be positive")

        # velocity xyz, displacement xyz, sin/cos dyaw, angular velocity,
        # log gap, query/gap ratio and transition-valid flag.
        self.step_projection = nn.Sequential(
            nn.Linear(12, self.step_dim),
            nn.LayerNorm(self.step_dim),
            nn.ReLU(inplace=True),
        )
        self.gru = nn.GRU(
            input_size=self.step_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.context = nn.Sequential(
            nn.Linear(self.hidden_dim + 2, self.hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.rate_residual_head = nn.Linear(self.hidden_dim, 4)
        self.log_sigma_head = nn.Linear(self.hidden_dim, 4)
        nn.init.zeros_(self.rate_residual_head.weight)
        nn.init.zeros_(self.rate_residual_head.bias)
        nn.init.zeros_(self.log_sigma_head.weight)
        nn.init.constant_(self.log_sigma_head.bias, math.log(initial_sigma))

    def _format_time(self, value, batch_size, reference):
        if value is None:
            return reference.new_full((batch_size,), self.time_scale)
        if not torch.is_tensor(value):
            value = torch.as_tensor(
                value, device=reference.device, dtype=reference.dtype)
        value = value.to(device=reference.device, dtype=reference.dtype)
        if value.numel() == 1:
            value = value.reshape(1).repeat(batch_size)
        return torch.clamp(value.reshape(batch_size), min=self.eps)

    def forward_trajectory(
            self, ref_boxs, delta_t, valid_mask, current_delta_t=None):
        if ref_boxs.dim() != 3 or ref_boxs.shape[-1] != 4:
            raise ValueError("ref_boxs must have shape [B,H,4]")
        batch_size, history_length, _ = ref_boxs.shape
        if delta_t.dim() == 1:
            delta_t = delta_t.unsqueeze(0)
        delta_t = delta_t.to(device=ref_boxs.device, dtype=ref_boxs.dtype)
        valid_mask = valid_mask.to(
            device=ref_boxs.device, dtype=ref_boxs.dtype)
        query_gap = self._format_time(
            current_delta_t, batch_size, ref_boxs)

        if history_length < 2:
            zeros3 = ref_boxs.new_zeros((batch_size, 3))
            zeros1 = ref_boxs.new_zeros((batch_size, 1))
            feature = ref_boxs.new_zeros((batch_size, self.hidden_dim))
            return {
                "feature": feature,
                "velocity": zeros3,
                "yaw_rate": zeros1.squeeze(1),
                "displacement": zeros3,
                "yaw_displacement": zeros1.squeeze(1),
                "trajectory_displacement": ref_boxs.new_zeros((batch_size, 4)),
                "kinematic_displacement": ref_boxs.new_zeros((batch_size, 4)),
                "log_sigma": ref_boxs.new_full(
                    (batch_size, 4), self.log_sigma_head.bias[0].item()),
                "valid": zeros1,
                "gap_ratio": query_gap / self.time_scale,
            }

        if delta_t.shape[1] < history_length:
            pad = delta_t[:, -1:].expand(
                -1, history_length - delta_t.shape[1])
            delta_t = torch.cat((delta_t, pad), dim=1)
        pair_gap = torch.clamp(
            delta_t[:, 1:history_length], min=self.eps)
        pair_valid = (
            valid_mask[:, :-1] > 0
        ) & (
            valid_mask[:, 1:] > 0
        )
        pair_valid_f = pair_valid.to(ref_boxs.dtype)

        newer = ref_boxs[:, :-1]
        older = ref_boxs[:, 1:]
        displacement = newer[:, :, :3] - older[:, :, :3]
        velocity = displacement / pair_gap.unsqueeze(-1)
        yaw_delta = wrap_angle(newer[:, :, 3] - older[:, :, 3])
        yaw_rate = yaw_delta / pair_gap
        query_ratio = (
            query_gap.unsqueeze(1) / pair_gap
        ).expand_as(pair_gap)
        step_features = torch.cat((
            velocity,
            displacement,
            torch.sin(yaw_delta).unsqueeze(-1),
            (torch.cos(yaw_delta) - 1.0).unsqueeze(-1),
            yaw_rate.unsqueeze(-1),
            torch.log1p(pair_gap / self.time_scale).unsqueeze(-1),
            query_ratio.unsqueeze(-1),
            pair_valid_f.unsqueeze(-1),
        ), dim=-1)
        step_features = torch.nan_to_num(
            step_features, nan=0.0, posinf=0.0, neginf=0.0)
        step_features = step_features * pair_valid_f.unsqueeze(-1)

        # Input boxes are recent-to-old; GRU consumes oldest-to-newest.
        chronological = torch.flip(step_features, dims=(1,))
        projected = self.step_projection(chronological)
        projected = projected * torch.flip(
            pair_valid_f, dims=(1,)).unsqueeze(-1)
        ordered_output, _ = self.gru(projected)
        ordered_state = ordered_output[:, -1]

        transition_count = pair_valid_f.sum(dim=1)
        nominal_gap = (
            (pair_gap * pair_valid_f).sum(dim=1)
            / torch.clamp(transition_count, min=1.0)
        )
        gap_ratio = query_gap / torch.clamp(
            nominal_gap, min=self.eps)
        context = self.context(torch.cat((
            ordered_state,
            torch.log1p(query_gap / self.time_scale).unsqueeze(1),
            torch.log1p(gap_ratio).unsqueeze(1),
        ), dim=1))

        # The first transition is the most recent one.  A valid trajectory in
        # this project always has a valid recent pair; invalid rows are zeroed.
        base_rate = torch.cat((
            velocity[:, 0],
            yaw_rate[:, 0:1],
        ), dim=1)
        residual_rate = self.residual_velocity_scale * torch.tanh(
            self.rate_residual_head(context))
        valid = (transition_count > 0).to(ref_boxs.dtype).unsqueeze(1)
        rate = (base_rate + residual_rate) * valid
        trajectory_displacement = rate * query_gap.unsqueeze(1)
        log_sigma = torch.clamp(
            self.log_sigma_head(context), min=-4.0, max=2.5)
        kinematic_displacement = base_rate * query_gap.unsqueeze(1) * valid
        return {
            "feature": context * valid,
            "velocity": rate[:, :3],
            "yaw_rate": rate[:, 3],
            "displacement": trajectory_displacement[:, :3],
            "yaw_displacement": trajectory_displacement[:, 3],
            "trajectory_displacement": trajectory_displacement,
            "kinematic_displacement": kinematic_displacement,
            "log_sigma": log_sigma,
            "valid": valid,
            "gap_ratio": gap_ratio,
        }

    def forward(self, ref_boxs, delta_t, valid_mask, current_delta_t=None):
        """Keep the legacy four-tensor contract for lightweight callers."""
        result = self.forward_trajectory(
            ref_boxs, delta_t, valid_mask, current_delta_t=current_delta_t)
        return (
            result["feature"],
            result["velocity"],
            result["displacement"],
            result["valid"],
        )


class TrajectoryPointEncoder(nn.Module):
    """Encode second-crop points without invalid-zero BatchNorm pollution."""

    def __init__(self, input_dim=5, hidden_dim=64, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(int(input_dim), 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, int(hidden_dim), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(int(hidden_dim), int(output_dim)),
            nn.ReLU(inplace=True),
        )

    def forward(self, points):
        return self.net(points)


class ZeroInitTrajectoryAdapter(nn.Module):
    """Inject ordered trajectory and second-crop evidence as a safe residual."""

    def __init__(
            self,
            feature_dim=256,
            trajectory_dim=128,
            search_dim=64,
            hidden_dim=128,
            normal_scale=0.1,
            gap_trigger=1.5):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.normal_scale = float(normal_scale)
        self.gap_trigger = float(gap_trigger)
        if not 0.0 <= self.normal_scale <= 1.0:
            raise ValueError("trajectory adapter normal scale must be in [0,1]")
        if self.gap_trigger <= 1.0:
            raise ValueError("trajectory adapter gap trigger must be > 1")
        scalar_dim = 6  # four log-sigmas, gap ratio, second-crop validity
        self.net = nn.Sequential(
            nn.Linear(
                self.feature_dim + int(trajectory_dim)
                + int(search_dim) + scalar_dim,
                int(hidden_dim),
            ),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), self.feature_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
            self,
            observation_feature,
            trajectory_feature,
            trajectory_search_feature,
            log_sigma,
            gap_ratio,
            trajectory_valid,
            trajectory_search_valid,
            enabled_scale=1.0):
        enabled_scale = float(enabled_scale)
        if not 0.0 <= enabled_scale <= 1.0:
            raise ValueError("trajectory adapter enabled scale must be in [0,1]")
        batch_size = observation_feature.shape[0]
        gap_ratio = gap_ratio.to(
            device=observation_feature.device,
            dtype=observation_feature.dtype,
        ).reshape(batch_size, 1)
        trajectory_valid = trajectory_valid.to(
            device=observation_feature.device,
            dtype=observation_feature.dtype,
        ).reshape(batch_size, 1)
        trajectory_search_valid = trajectory_search_valid.to(
            device=observation_feature.device,
            dtype=observation_feature.dtype,
        ).reshape(batch_size, 1)
        gap_activation = torch.clamp(
            (gap_ratio - 1.0) / (self.gap_trigger - 1.0),
            min=0.0,
            max=1.0,
        )
        sample_scale = (
            self.normal_scale
            + (1.0 - self.normal_scale) * gap_activation
        ) * trajectory_valid * enabled_scale
        adapter_input = torch.cat((
            observation_feature,
            trajectory_feature,
            trajectory_search_feature,
            log_sigma.to(
                device=observation_feature.device,
                dtype=observation_feature.dtype),
            gap_ratio,
            trajectory_search_valid,
        ), dim=1)
        correction = self.net(adapter_input) * sample_scale
        adapted = observation_feature + correction
        return adapted, {
            "trajectory_adapter_correction": correction,
            "trajectory_adapter_norm": torch.linalg.norm(correction, dim=1),
            "trajectory_adapter_scale": sample_scale.squeeze(1),
            "trajectory_gap_activation": gap_activation.squeeze(1),
        }
