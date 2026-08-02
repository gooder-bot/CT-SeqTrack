"""Ordered, observation-safe trajectory components for CT-SeqTrack v2."""

import math

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

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


class OrderedPhysicalMotionEncoder(nn.Module):
    """Predict candidate-independent xy motion from an ordered box history.

    Boxes are supplied newest-to-oldest, matching the tracker contract.  The
    GRU consumes transitions oldest-to-newest, while a causal latest-velocity
    extrapolation provides a useful zero-initialized cold start.  Physical time
    is structural: transition velocities divide by their measured gaps and the
    predicted rate is integrated over the query gap.
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
            raise ValueError("physical motion encoder dimensions must be positive")
        if self.residual_velocity_scale <= 0:
            raise ValueError("physical residual velocity scale must be positive")
        if initial_sigma <= 0:
            raise ValueError("physical initial sigma must be positive")

        # xy velocity, xy displacement, sin/cos yaw change, log pair gap,
        # query/pair ratio, and transition-valid flag.
        self.step_projection = nn.Sequential(
            nn.Linear(9, self.step_dim),
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
        self.velocity_residual_head = nn.Linear(self.hidden_dim, 2)
        self.log_sigma_head = nn.Linear(self.hidden_dim, 2)
        nn.init.zeros_(self.velocity_residual_head.weight)
        nn.init.zeros_(self.velocity_residual_head.bias)
        nn.init.zeros_(self.log_sigma_head.weight)
        nn.init.constant_(self.log_sigma_head.bias, math.log(initial_sigma))

    def _format_query_gap(self, value, batch_size, reference):
        if value is None:
            value = reference.new_full((batch_size,), self.time_scale)
        elif not torch.is_tensor(value):
            value = torch.as_tensor(
                value, device=reference.device, dtype=reference.dtype)
        value = value.to(device=reference.device, dtype=reference.dtype)
        if value.numel() == 1:
            value = value.reshape(1).repeat(batch_size)
        elif value.numel() != batch_size:
            raise ValueError(
                "current_delta_t must contain one value or one per batch item")
        finite = torch.isfinite(value.reshape(batch_size))
        value = torch.nan_to_num(
            value.reshape(batch_size), nan=self.time_scale,
            posinf=self.time_scale, neginf=self.time_scale)
        return torch.clamp(value, min=self.eps), finite

    def forward(
            self, ref_boxs, delta_t, valid_mask, current_delta_t=None):
        if ref_boxs.dim() != 3 or ref_boxs.shape[-1] != 4:
            raise ValueError("motion ref_boxs must have shape [B,H,4]")
        batch_size, history_length, _ = ref_boxs.shape
        if delta_t.dim() == 1:
            delta_t = delta_t.unsqueeze(0)
        delta_t = delta_t.to(device=ref_boxs.device, dtype=ref_boxs.dtype)
        if delta_t.dim() != 2 or delta_t.shape[0] != batch_size:
            raise ValueError("motion delta_t must have shape [B,H]")
        valid_mask = valid_mask.to(
            device=ref_boxs.device, dtype=ref_boxs.dtype)
        if valid_mask.dim() == 1:
            valid_mask = valid_mask.unsqueeze(0)
        if valid_mask.shape != ref_boxs.shape[:2]:
            raise ValueError("motion valid_mask must have shape [B,H]")
        query_gap, query_finite = self._format_query_gap(
            current_delta_t, batch_size, ref_boxs)

        # PyTorch 2.0's Tensor.all accepts one dimension at a time; flattening
        # preserves the per-sample finite check and remains compatible with
        # newer releases.
        finite_row = torch.isfinite(ref_boxs).flatten(1).all(dim=1)
        finite_row = finite_row & torch.isfinite(delta_t).all(dim=1)
        finite_row = finite_row & query_finite
        safe_boxs = torch.nan_to_num(
            ref_boxs, nan=0.0, posinf=0.0, neginf=0.0)
        safe_delta_t = torch.nan_to_num(
            delta_t, nan=self.time_scale,
            posinf=self.time_scale, neginf=self.time_scale)

        zeros_xy = ref_boxs.new_zeros((batch_size, 2))
        zeros_feature = ref_boxs.new_zeros((batch_size, self.hidden_dim))
        initial_log_sigma = ref_boxs.new_full(
            (batch_size, 2), self.log_sigma_head.bias[0].item())
        if history_length < 2:
            return {
                "feature": zeros_feature,
                "velocity_xy": zeros_xy,
                "prior_xy": zeros_xy,
                "kinematic_prior_xy": zeros_xy,
                "log_sigma_xy": initial_log_sigma,
                "valid": ref_boxs.new_zeros((batch_size,)),
                "gap_ratio": ref_boxs.new_ones((batch_size,)),
            }

        if safe_delta_t.shape[1] < history_length:
            pad = safe_delta_t[:, -1:].expand(
                -1, history_length - safe_delta_t.shape[1])
            safe_delta_t = torch.cat((safe_delta_t, pad), dim=1)
        pair_gap = torch.clamp(
            safe_delta_t[:, 1:history_length], min=self.eps)
        pair_valid = (
            (valid_mask[:, :-1] > 0)
            & (valid_mask[:, 1:] > 0)
        )
        pair_valid_f = pair_valid.to(ref_boxs.dtype)

        newer = safe_boxs[:, :-1]
        older = safe_boxs[:, 1:]
        displacement_xy = newer[:, :, :2] - older[:, :, :2]
        velocity_xy = displacement_xy / pair_gap.unsqueeze(-1)
        yaw_delta = wrap_angle(newer[:, :, 3] - older[:, :, 3])
        query_ratio = (
            query_gap.unsqueeze(1) / pair_gap
        ).expand_as(pair_gap)
        step_features = torch.cat((
            velocity_xy,
            displacement_xy,
            torch.sin(yaw_delta).unsqueeze(-1),
            (torch.cos(yaw_delta) - 1.0).unsqueeze(-1),
            torch.log1p(pair_gap / self.time_scale).unsqueeze(-1),
            query_ratio.unsqueeze(-1),
            pair_valid_f.unsqueeze(-1),
        ), dim=-1)
        step_features = torch.nan_to_num(
            step_features, nan=0.0, posinf=0.0, neginf=0.0)
        step_features = step_features * pair_valid_f.unsqueeze(-1)

        chronological = torch.flip(step_features, dims=(1,))
        projected = self.step_projection(chronological)
        chronological_valid = torch.flip(pair_valid, dims=(1,))
        # A zero vector is not a no-op for a GRU with biases.  Compact valid
        # transitions before packing so padded history cannot alter the state.
        projected = projected * chronological_valid.unsqueeze(-1)
        compact_indices = torch.argsort(
            (~chronological_valid).to(torch.int64),
            dim=1,
            stable=True,
        )
        compact_projected = torch.gather(
            projected,
            dim=1,
            index=compact_indices.unsqueeze(-1).expand_as(projected),
        )
        transition_count = pair_valid_f.sum(dim=1)
        packed_projected = pack_padded_sequence(
            compact_projected,
            lengths=torch.clamp(
                transition_count, min=1).to(torch.long).cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, ordered_hidden = self.gru(packed_projected)
        ordered_state = ordered_hidden[-1]

        nominal_gap = (
            (pair_gap * pair_valid_f).sum(dim=1)
            / torch.clamp(transition_count, min=1.0)
        )
        gap_ratio_raw = query_gap / torch.clamp(nominal_gap, min=self.eps)
        context = self.context(torch.cat((
            ordered_state,
            torch.log1p(query_gap / self.time_scale).unsqueeze(1),
            torch.log1p(gap_ratio_raw).unsqueeze(1),
        ), dim=1))

        recent_pair_valid = pair_valid[:, 0]
        valid = (recent_pair_valid & finite_row).to(ref_boxs.dtype)
        base_velocity = velocity_xy[:, 0]
        residual_velocity = self.residual_velocity_scale * torch.tanh(
            self.velocity_residual_head(context))
        predicted_velocity = (
            base_velocity + residual_velocity) * valid.unsqueeze(1)
        prior_xy = predicted_velocity * query_gap.unsqueeze(1)
        kinematic_prior_xy = (
            base_velocity * query_gap.unsqueeze(1) * valid.unsqueeze(1))
        log_sigma_xy = torch.clamp(
            self.log_sigma_head(context), min=-4.0, max=2.5)
        gap_ratio = torch.where(
            valid > 0, gap_ratio_raw, torch.ones_like(gap_ratio_raw))
        return {
            "feature": context * valid.unsqueeze(1),
            "velocity_xy": predicted_velocity,
            "prior_xy": prior_xy,
            "kinematic_prior_xy": kinematic_prior_xy,
            "log_sigma_xy": log_sigma_xy,
            "valid": valid,
            "gap_ratio": gap_ratio,
        }


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


class TrajectorySearchEvidence(nn.Module):
    """Encode masked endpoint-crop points into soft foreground evidence."""

    def __init__(
            self,
            point_dim=9,
            feature_dim=128,
            observation_dim=256,
            motion_dim=128,
            observation_stats_dim=5,
            max_vote_offset=4.0,
            eps=1e-6):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.max_vote_offset = float(max_vote_offset)
        self.eps = float(eps)
        if self.feature_dim <= 0 or self.max_vote_offset <= 0:
            raise ValueError("search evidence dimensions/scales must be positive")

        self.point_mlp = nn.Sequential(
            nn.Linear(int(point_dim), 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.ReLU(inplace=True),
        )
        self.extension_source_embedding = nn.Parameter(
            torch.zeros(self.feature_dim))
        # query dt, gap ratio, sigma parallel/perpendicular, available count
        context_input_dim = (
            int(observation_dim) + int(motion_dim)
            + int(observation_stats_dim) + 5)
        self.context_projection = nn.Sequential(
            nn.Linear(context_input_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.ReLU(inplace=True),
        )
        self.film_scale = nn.Linear(self.feature_dim, self.feature_dim)
        self.film_shift = nn.Linear(self.feature_dim, self.feature_dim)
        self.targetness_head = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.vote_head = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )
        self.confidence_head = nn.Linear(self.feature_dim + 2, 1)

    @staticmethod
    def _batch_scalar(value, reference, default=0.0):
        batch_size = reference.shape[0]
        if value is None:
            return reference.new_full((batch_size, 1), float(default))
        if not torch.is_tensor(value):
            value = torch.as_tensor(
                value, device=reference.device, dtype=reference.dtype)
        value = value.to(
            device=reference.device, dtype=reference.dtype).reshape(-1)
        if value.numel() == 1:
            value = value.repeat(batch_size)
        elif value.numel() != batch_size:
            raise ValueError("search scalar must contain one value per sample")
        return value.reshape(batch_size, 1)

    def forward(
            self,
            point_inputs,
            point_xy,
            point_valid_mask,
            geometry_valid,
            observation_feature,
            motion_feature,
            observation_stats,
            query_delta_t,
            gap_ratio,
            sigma_parallel,
            sigma_perpendicular,
            available_count=None):
        if point_inputs.dim() != 3 or point_inputs.shape[-1] != 9:
            raise ValueError("search point_inputs must have shape [B,N,9]")
        if point_xy.shape != point_inputs.shape[:2] + (2,):
            raise ValueError("search point_xy must have shape [B,N,2]")
        batch_size, point_count, _ = point_inputs.shape
        valid = point_valid_mask.to(
            device=point_inputs.device, dtype=point_inputs.dtype)
        if valid.shape != (batch_size, point_count):
            raise ValueError("search point mask must have shape [B,N]")
        geometry_valid = self._batch_scalar(
            geometry_valid, point_inputs, default=0.0)
        geometry_valid = (geometry_valid > 0).to(point_inputs.dtype)
        valid = (valid > 0).to(point_inputs.dtype)
        valid = valid * geometry_valid

        scalar_context = torch.cat((
            self._batch_scalar(query_delta_t, point_inputs, default=0.1),
            self._batch_scalar(gap_ratio, point_inputs, default=1.0),
            self._batch_scalar(sigma_parallel, point_inputs, default=0.0),
            self._batch_scalar(
                sigma_perpendicular, point_inputs, default=0.0),
            torch.log1p(torch.clamp(self._batch_scalar(
                available_count, point_inputs, default=0.0), min=0.0)) / 8.0,
        ), dim=1)
        context_input = torch.cat((
            observation_feature.detach(),
            motion_feature.detach(),
            observation_stats.detach(),
            scalar_context,
        ), dim=1)
        context = self.context_projection(torch.nan_to_num(
            context_input, nan=0.0, posinf=0.0, neginf=0.0))

        point_feature = self.point_mlp(torch.nan_to_num(
            point_inputs, nan=0.0, posinf=0.0, neginf=0.0))
        point_feature = point_feature + self.extension_source_embedding.view(
            1, 1, -1)
        film_scale = torch.sigmoid(self.film_scale(context)).unsqueeze(1)
        film_shift = self.film_shift(context).unsqueeze(1)
        point_feature = point_feature * (1.0 + film_scale) + film_shift

        targetness_logits = self.targetness_head(
            point_feature).squeeze(-1)
        targetness = torch.sigmoid(targetness_logits)
        weights = targetness * valid
        targetness_mass = weights.sum(dim=1)
        denominator = torch.clamp(
            targetness_mass.unsqueeze(1), min=self.eps)
        search_evidence_token = (
            point_feature * weights.unsqueeze(-1)
        ).sum(dim=1) / denominator

        vote_offsets = self.max_vote_offset * torch.tanh(
            self.vote_head(point_feature))
        point_center_votes = point_xy + vote_offsets
        search_proposal_xy = (
            point_center_votes * weights.unsqueeze(-1)
        ).sum(dim=1) / denominator

        valid_count = valid.sum(dim=1)
        mean_targetness = targetness_mass / torch.clamp(
            valid_count, min=1.0)
        probability = torch.clamp(
            targetness, min=self.eps, max=1.0 - self.eps)
        point_entropy = -(
            probability * torch.log(probability)
            + (1.0 - probability) * torch.log(1.0 - probability))
        entropy = (point_entropy * valid).sum(dim=1) / torch.clamp(
            valid_count, min=1.0)
        confidence_logit = self.confidence_head(torch.cat((
            search_evidence_token,
            mean_targetness.unsqueeze(1),
            entropy.unsqueeze(1),
        ), dim=1)).squeeze(1)
        search_confidence = torch.sigmoid(confidence_logit)
        row_valid = (valid_count >= 3).to(point_inputs.dtype)
        search_confidence = search_confidence * row_valid
        search_evidence_token = search_evidence_token * row_valid.unsqueeze(1)
        search_proposal_xy = search_proposal_xy * row_valid.unsqueeze(1)
        return {
            "search_evidence_token": search_evidence_token,
            "search_proposal_xy": search_proposal_xy,
            "search_confidence": search_confidence,
            "search_confidence_logit": confidence_logit,
            "search_targetness_logits": targetness_logits,
            "search_targetness": targetness,
            "search_targetness_mass": targetness_mass,
            "search_targetness_mean": mean_targetness,
            "search_targetness_entropy": entropy,
            "search_vote_offsets": vote_offsets,
            "search_point_center_votes": point_center_votes,
            "search_candidate_valid": row_valid,
        }


class JointProposalFusion(nn.Module):
    """Observation-default bounded fusion of observation/motion/search xy."""

    def __init__(
            self,
            observation_dim=256,
            motion_dim=128,
            search_dim=128,
            observation_stats_dim=5,
            context_dim=32,
            hidden_dim=96,
            observation_bias=4.6,
            radius_base=0.5,
            radius_per_second=0.5,
            radius_max=2.0,
            normal_aux_mass=0.5,
            gap_aux_mass=0.8,
            eps=1e-6):
        super().__init__()
        self.radius_base = float(radius_base)
        self.radius_per_second = float(radius_per_second)
        self.radius_max = float(radius_max)
        self.normal_aux_mass = float(normal_aux_mass)
        self.gap_aux_mass = float(gap_aux_mass)
        self.eps = float(eps)
        if not 0.0 <= self.normal_aux_mass <= self.gap_aux_mass <= 1.0:
            raise ValueError("joint fusion auxiliary mass bounds are invalid")
        if self.radius_base < 0 or self.radius_max <= 0:
            raise ValueError("joint fusion radii must be positive")
        self.observation_projection = nn.Sequential(
            nn.Linear(int(observation_dim), int(context_dim)), nn.ReLU())
        self.motion_projection = nn.Sequential(
            nn.Linear(int(motion_dim), int(context_dim)), nn.ReLU())
        self.search_projection = nn.Sequential(
            nn.Linear(int(search_dim), int(context_dim)), nn.ReLU())
        # observation stats; motion sigma xy/history validity; search
        # confidence/mass/entropy; three pair distances; query dt/gap ratio.
        scalar_dim = int(observation_stats_dim) + 2 + 1 + 3 + 3 + 2
        self.gate = nn.Sequential(
            nn.Linear(3 * int(context_dim) + scalar_dim, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), 3),
        )
        nn.init.zeros_(self.gate[-1].weight)
        with torch.no_grad():
            self.gate[-1].bias.copy_(torch.tensor(
                [float(observation_bias), 0.0, 0.0]))

    @staticmethod
    def _batch_scalar(value, reference, default=0.0):
        batch_size = reference.shape[0]
        if value is None:
            return reference.new_full((batch_size, 1), float(default))
        if not torch.is_tensor(value):
            value = torch.as_tensor(
                value, device=reference.device, dtype=reference.dtype)
        value = value.to(
            device=reference.device, dtype=reference.dtype).reshape(-1)
        if value.numel() == 1:
            value = value.repeat(batch_size)
        elif value.numel() != batch_size:
            raise ValueError("joint fusion scalar must contain one per sample")
        return value.reshape(batch_size, 1)

    @staticmethod
    def _clip_norm(vector, radius, eps=1e-6):
        norm = torch.linalg.norm(vector, dim=1, keepdim=True)
        scale = torch.minimum(
            torch.ones_like(norm), radius / torch.clamp(norm, min=eps))
        return vector * scale

    def forward(
            self,
            observation_box,
            observation_feature,
            observation_stats,
            motion_feature,
            motion_proposal_xy,
            motion_log_sigma_xy,
            motion_valid,
            history_valid_ratio,
            search_evidence_token,
            search_proposal_xy,
            search_confidence,
            search_targetness_mass,
            search_entropy,
            search_valid,
            query_delta_t,
            gap_ratio,
            enabled_scale=1.0):
        if observation_box.dim() != 2 or observation_box.shape[1] != 4:
            raise ValueError("observation_box must have shape [B,4]")
        enabled_scale = float(enabled_scale)
        if not 0.0 <= enabled_scale <= 1.0:
            raise ValueError("joint fusion scale must be in [0,1]")
        if enabled_scale == 0.0:
            batch_size = observation_box.shape[0]
            diagnostics = {
                "joint_gate_logits": observation_box.new_zeros(
                    (batch_size, 3)),
                "joint_gate_probability": torch.cat((
                    observation_box.new_ones((batch_size, 1)),
                    observation_box.new_zeros((batch_size, 2))), dim=1),
                "joint_gate_applied_probability": torch.cat((
                    observation_box.new_ones((batch_size, 1)),
                    observation_box.new_zeros((batch_size, 2))), dim=1),
                "joint_correction_xy": observation_box.new_zeros(
                    (batch_size, 2)),
                "joint_fusion_radius": observation_box.new_zeros(
                    (batch_size,)),
                "joint_motion_valid": observation_box.new_zeros(
                    (batch_size,)),
                "joint_search_valid": observation_box.new_zeros(
                    (batch_size,)),
                "joint_fusion_enabled_scale": observation_box.new_tensor(0.0),
            }
            return observation_box, diagnostics

        observation = observation_box.detach()
        observation_xy = observation[:, :2]
        motion_xy = motion_proposal_xy.detach()
        search_xy = search_proposal_xy.detach()
        observation_feature = observation_feature.detach()
        observation_stats = observation_stats.detach()
        motion_feature = motion_feature.detach()
        motion_log_sigma_xy = motion_log_sigma_xy.detach()
        search_evidence_token = search_evidence_token.detach()
        search_confidence = search_confidence.detach()
        search_targetness_mass = search_targetness_mass.detach()
        search_entropy = search_entropy.detach()

        motion_valid = (self._batch_scalar(
            motion_valid, observation_xy) > 0)
        search_valid = (self._batch_scalar(
            search_valid, observation_xy) > 0)
        motion_finite = (
            torch.isfinite(motion_xy).all(dim=1, keepdim=True)
            & torch.isfinite(motion_feature).all(dim=1, keepdim=True)
            & torch.isfinite(motion_log_sigma_xy).all(dim=1, keepdim=True))
        search_finite = (
            torch.isfinite(search_xy).all(dim=1, keepdim=True)
            & torch.isfinite(search_evidence_token).all(dim=1, keepdim=True)
            & torch.isfinite(search_confidence).reshape(-1, 1))
        motion_valid = motion_valid & motion_finite
        search_valid = search_valid & search_finite
        motion_xy = torch.where(
            torch.isfinite(motion_xy), motion_xy, observation_xy)
        search_xy = torch.where(
            torch.isfinite(search_xy), search_xy, observation_xy)

        query_dt = torch.clamp(torch.nan_to_num(self._batch_scalar(
            query_delta_t, observation_xy, default=0.1), nan=0.1), min=0.0)
        gap_ratio = torch.nan_to_num(self._batch_scalar(
            gap_ratio, observation_xy, default=1.0), nan=1.0)
        history_valid_ratio = torch.nan_to_num(self._batch_scalar(
            history_valid_ratio, observation_xy), nan=0.0)
        search_confidence_column = torch.clamp(
            torch.nan_to_num(search_confidence.reshape(-1, 1), nan=0.0),
            min=0.0, max=1.0)
        search_mass_column = torch.log1p(torch.clamp(
            torch.nan_to_num(search_targetness_mass.reshape(-1, 1)), min=0.0))
        search_entropy_column = torch.nan_to_num(
            search_entropy.reshape(-1, 1), nan=0.0)

        obs_motion = torch.linalg.norm(
            motion_xy - observation_xy, dim=1, keepdim=True)
        obs_search = torch.linalg.norm(
            search_xy - observation_xy, dim=1, keepdim=True)
        motion_search = torch.linalg.norm(
            motion_xy - search_xy, dim=1, keepdim=True)
        gate_input = torch.cat((
            self.observation_projection(torch.nan_to_num(
                observation_feature, nan=0.0, posinf=0.0, neginf=0.0)),
            self.motion_projection(torch.nan_to_num(
                motion_feature, nan=0.0, posinf=0.0, neginf=0.0)),
            self.search_projection(torch.nan_to_num(
                search_evidence_token, nan=0.0, posinf=0.0, neginf=0.0)),
            torch.nan_to_num(
                observation_stats, nan=0.0, posinf=0.0, neginf=0.0),
            torch.nan_to_num(
                motion_log_sigma_xy, nan=0.0, posinf=0.0, neginf=0.0),
            history_valid_ratio,
            search_confidence_column,
            search_mass_column,
            search_entropy_column,
            obs_motion,
            obs_search,
            motion_search,
            query_dt,
            gap_ratio,
        ), dim=1)
        logits = self.gate(gate_input)
        logits = logits.clone()
        logits[:, 1] = torch.where(
            motion_valid.squeeze(1), logits[:, 1],
            logits.new_full((), float("-inf")))
        logits[:, 2] = torch.where(
            search_valid.squeeze(1),
            logits[:, 2] + torch.log(
                search_confidence_column.squeeze(1) + self.eps),
            logits.new_full((), float("-inf")))
        probability = torch.softmax(logits, dim=1)

        max_aux_mass = self.normal_aux_mass + (
            self.gap_aux_mass - self.normal_aux_mass
        ) * torch.clamp(gap_ratio - 1.0, min=0.0, max=1.0)
        aux_probability = probability[:, 1:]
        aux_mass = aux_probability.sum(dim=1, keepdim=True)
        aux_scale = torch.minimum(
            torch.ones_like(aux_mass),
            max_aux_mass / torch.clamp(aux_mass, min=self.eps))
        applied_aux = aux_probability * aux_scale
        applied_probability = torch.cat((
            1.0 - applied_aux.sum(dim=1, keepdim=True), applied_aux), dim=1)

        radius = torch.clamp(
            self.radius_base + self.radius_per_second * query_dt,
            max=self.radius_max)
        motion_residual = self._clip_norm(
            motion_xy - observation_xy, radius, eps=self.eps)
        search_residual = self._clip_norm(
            search_xy - observation_xy, radius, eps=self.eps)
        correction = (
            applied_aux[:, 0:1] * motion_residual
            + applied_aux[:, 1:2] * search_residual)
        correction = self._clip_norm(correction, radius, eps=self.eps)
        correction = correction * enabled_scale
        candidate_xy = observation_xy + correction
        any_valid = (motion_valid | search_valid).expand_as(candidate_xy)
        final_xy = torch.where(any_valid, candidate_xy, observation_xy)
        final_box = torch.cat((final_xy, observation[:, 2:]), dim=1)
        return final_box, {
            "joint_gate_logits": logits,
            "joint_gate_probability": probability,
            "joint_gate_applied_probability": applied_probability,
            "joint_correction_xy": torch.where(
                any_valid, correction, torch.zeros_like(correction)),
            "joint_fusion_radius": radius.squeeze(1),
            "joint_motion_valid": motion_valid.to(
                observation_xy.dtype).squeeze(1),
            "joint_search_valid": search_valid.to(
                observation_xy.dtype).squeeze(1),
            "joint_fusion_enabled_scale": observation_xy.new_tensor(
                enabled_scale),
        }


class TrajectorySearchEvidenceV21(nn.Module):
    """Observation-queried, source-aware evidence at a trajectory endpoint."""

    def __init__(
            self,
            point_dim=9,
            feature_dim=128,
            query_dim=32,
            observation_dim=256,
            motion_dim=128,
            observation_stats_dim=5,
            max_vote_offset=4.0,
            pool_temperature=0.5,
            eps=1e-6):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.query_dim = int(query_dim)
        self.max_vote_offset = float(max_vote_offset)
        self.pool_temperature = float(pool_temperature)
        self.eps = float(eps)
        if min(self.feature_dim, self.query_dim) <= 0:
            raise ValueError("search v2.1 feature dimensions must be positive")
        if self.max_vote_offset <= 0 or self.pool_temperature <= 0:
            raise ValueError("search v2.1 scales must be positive")

        self.point_mlp = nn.Sequential(
            nn.Linear(int(point_dim), 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.ReLU(inplace=True),
        )
        self.source_embedding = nn.Embedding(2, self.feature_dim)
        # dt, gap, two sigmas, three source counts, motion-context validity.
        scalar_dim = 8
        context_input_dim = (
            int(observation_dim) + int(motion_dim)
            + int(observation_stats_dim) + scalar_dim)
        self.context_projection = nn.Sequential(
            nn.Linear(context_input_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.ReLU(inplace=True),
        )
        self.film_scale = nn.Linear(self.feature_dim, self.feature_dim)
        self.film_shift = nn.Linear(self.feature_dim, self.feature_dim)

        query_input_dim = int(observation_dim) + int(observation_stats_dim)
        self.query_projection = nn.Sequential(
            nn.Linear(query_input_dim, self.query_dim),
            nn.LayerNorm(self.query_dim),
        )
        self.key_projection = nn.Linear(self.feature_dim, self.query_dim)
        self.key_norm = nn.LayerNorm(self.query_dim)
        self.query_value_projection = nn.Linear(
            self.query_dim, self.feature_dim)
        self.query_norm = nn.LayerNorm(self.feature_dim)
        self.local_targetness_head = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.vote_head = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

    @staticmethod
    def _batch_scalar(value, reference, default=0.0):
        batch_size = reference.shape[0]
        if value is None:
            return reference.new_full((batch_size, 1), float(default))
        if not torch.is_tensor(value):
            value = torch.as_tensor(
                value, device=reference.device, dtype=reference.dtype)
        value = value.to(
            device=reference.device, dtype=reference.dtype).reshape(-1)
        if value.numel() == 1:
            value = value.repeat(batch_size)
        elif value.numel() != batch_size:
            raise ValueError("search v2.1 scalar must contain one per sample")
        return value.reshape(batch_size, 1)

    def _masked_softmax(self, logits, valid):
        mask = valid > 0
        scaled = logits / self.pool_temperature
        scaled = scaled.masked_fill(~mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(scaled, dim=1) * valid
        return weights / torch.clamp(
            weights.sum(dim=1, keepdim=True), min=self.eps)

    def forward(
            self,
            point_inputs,
            point_xy,
            point_valid_mask,
            point_source,
            geometry_valid,
            observation_feature,
            motion_feature,
            motion_context_valid,
            observation_stats,
            query_delta_t,
            gap_ratio,
            sigma_parallel,
            sigma_perpendicular,
            available_count=None,
            extension_count=None,
            overlap_count=None):
        if point_inputs.dim() != 3 or point_inputs.shape[-1] != 9:
            raise ValueError("search v2.1 point inputs must have shape [B,N,9]")
        if point_xy.shape != point_inputs.shape[:2] + (2,):
            raise ValueError("search v2.1 point xy must have shape [B,N,2]")
        batch_size, point_count, _ = point_inputs.shape
        valid = point_valid_mask.to(
            device=point_inputs.device, dtype=point_inputs.dtype)
        if valid.shape != (batch_size, point_count):
            raise ValueError("search v2.1 point mask must have shape [B,N]")
        source = point_source.to(
            device=point_inputs.device, dtype=torch.long)
        if source.shape != (batch_size, point_count):
            raise ValueError("search v2.1 point source must have shape [B,N]")
        if bool(torch.any((source < 0) | (source > 1)).item()):
            raise ValueError("search v2.1 point source must be 0 or 1")

        geometry = (self._batch_scalar(
            geometry_valid, point_inputs) > 0).to(point_inputs.dtype)
        valid = (valid > 0).to(point_inputs.dtype) * geometry
        motion_context_valid = self._batch_scalar(
            motion_context_valid, point_inputs, default=0.0)
        available = self._batch_scalar(
            available_count, point_inputs, default=0.0)
        extension = self._batch_scalar(
            extension_count, point_inputs, default=0.0)
        overlap = self._batch_scalar(
            overlap_count, point_inputs, default=0.0)
        scalar_context = torch.cat((
            self._batch_scalar(query_delta_t, point_inputs, default=0.1),
            self._batch_scalar(gap_ratio, point_inputs, default=1.0),
            self._batch_scalar(sigma_parallel, point_inputs),
            self._batch_scalar(sigma_perpendicular, point_inputs),
            torch.log1p(torch.clamp(available, min=0.0)) / 8.0,
            torch.log1p(torch.clamp(extension, min=0.0)) / 8.0,
            torch.log1p(torch.clamp(overlap, min=0.0)) / 8.0,
            motion_context_valid,
        ), dim=1)
        observation_feature = observation_feature.detach()
        motion_feature = motion_feature.detach()
        observation_stats = observation_stats.detach()
        context_input = torch.cat((
            observation_feature,
            motion_feature,
            observation_stats,
            scalar_context,
        ), dim=1)
        context = self.context_projection(torch.nan_to_num(
            context_input, nan=0.0, posinf=0.0, neginf=0.0))

        point_feature = self.point_mlp(torch.nan_to_num(
            point_inputs, nan=0.0, posinf=0.0, neginf=0.0))
        point_feature = point_feature + self.source_embedding(source)
        film_scale = torch.tanh(self.film_scale(context)).unsqueeze(1)
        film_shift = self.film_shift(context).unsqueeze(1)
        point_feature = point_feature * (1.0 + film_scale) + film_shift

        query_input = torch.cat((
            observation_feature, observation_stats), dim=1)
        query = self.query_projection(torch.nan_to_num(
            query_input, nan=0.0, posinf=0.0, neginf=0.0))
        key = self.key_norm(self.key_projection(point_feature))
        match_logits = (
            key * query.unsqueeze(1)).sum(dim=2) / math.sqrt(
                float(self.query_dim))
        query_value = self.query_value_projection(query).unsqueeze(1)
        point_feature = self.query_norm(
            point_feature
            + torch.sigmoid(match_logits).unsqueeze(2) * query_value)

        local_logits = self.local_targetness_head(
            point_feature).squeeze(-1)
        targetness_logits = local_logits + match_logits
        targetness = torch.sigmoid(targetness_logits)
        pool_weights = self._masked_softmax(targetness_logits, valid)
        evidence_token = (
            point_feature * pool_weights.unsqueeze(2)).sum(dim=1)

        vote_offsets = self.max_vote_offset * torch.tanh(
            self.vote_head(point_feature))
        point_center_votes = point_xy + vote_offsets
        proposal_xy = (
            point_center_votes * pool_weights.unsqueeze(2)).sum(dim=1)

        valid_count = valid.sum(dim=1)
        row_valid = (valid_count >= 3).to(point_inputs.dtype)
        targetness_mass = (targetness * valid).sum(dim=1)
        targetness_mean = targetness_mass / torch.clamp(
            valid_count, min=1.0)
        masked_targetness = targetness.masked_fill(
            valid <= 0, -1.0)
        targetness_max = torch.clamp(
            masked_targetness.max(dim=1).values, min=0.0)
        probability = torch.clamp(
            targetness, min=self.eps, max=1.0 - self.eps)
        point_entropy = -(
            probability * torch.log(probability)
            + (1.0 - probability) * torch.log(1.0 - probability))
        entropy = (point_entropy * valid).sum(dim=1) / torch.clamp(
            valid_count, min=1.0)
        effective_sample_size = 1.0 / torch.clamp(
            pool_weights.pow(2).sum(dim=1), min=self.eps)
        extension_weight_ratio = (
            pool_weights * (source == 1).to(pool_weights.dtype)).sum(dim=1)

        evidence_token = evidence_token * row_valid.unsqueeze(1)
        proposal_xy = proposal_xy * row_valid.unsqueeze(1)
        pool_weights = pool_weights * row_valid.unsqueeze(1)
        return {
            "search_v21_evidence_token": evidence_token,
            "search_v21_proposal_xy": proposal_xy,
            "search_v21_match_logits": match_logits,
            "search_v21_local_targetness_logits": local_logits,
            "search_v21_targetness_logits": targetness_logits,
            "search_v21_targetness": targetness,
            "search_v21_pool_weights": pool_weights,
            "search_v21_targetness_mass": targetness_mass,
            "search_v21_targetness_mean": targetness_mean,
            "search_v21_targetness_max": targetness_max,
            "search_v21_targetness_entropy": entropy,
            "search_v21_effective_sample_size": effective_sample_size,
            "search_v21_extension_weight_ratio": extension_weight_ratio,
            "search_v21_vote_offsets": vote_offsets,
            "search_v21_point_center_votes": point_center_votes,
            "search_v21_candidate_valid": row_valid,
        }


class AdvantageGatedProposalFusion(nn.Module):
    """Independent relative-advantage gates with observation fallback."""

    def __init__(
            self,
            observation_dim=256,
            motion_dim=128,
            search_dim=128,
            observation_stats_dim=5,
            context_dim=32,
            hidden_dim=96,
            init_help_probability=0.02,
            radius_base=0.5,
            radius_per_second=0.5,
            radius_max=2.0,
            normal_aux_mass=0.5,
            gap_aux_mass=0.8,
            eps=1e-6):
        super().__init__()
        self.radius_base = float(radius_base)
        self.radius_per_second = float(radius_per_second)
        self.radius_max = float(radius_max)
        self.normal_aux_mass = float(normal_aux_mass)
        self.gap_aux_mass = float(gap_aux_mass)
        self.eps = float(eps)
        init_help_probability = float(init_help_probability)
        if not 0.0 < init_help_probability < 1.0:
            raise ValueError("initial help probability must be in (0,1)")
        if not 0.0 <= self.normal_aux_mass <= self.gap_aux_mass <= 1.0:
            raise ValueError("advantage fusion auxiliary bounds are invalid")
        if self.radius_base < 0 or self.radius_max <= 0:
            raise ValueError("advantage fusion radii must be positive")

        self.observation_projection = nn.Sequential(
            nn.Linear(int(observation_dim), int(context_dim)), nn.ReLU())
        self.motion_projection = nn.Sequential(
            nn.Linear(int(motion_dim), int(context_dim)), nn.ReLU())
        self.search_projection = nn.Sequential(
            nn.Linear(int(search_dim), int(context_dim)), nn.ReLU())
        # obs stats (5), motion sigma/history (3), search evidence (8),
        # proposal disagreements (3), query dt/gap (2).
        scalar_dim = int(observation_stats_dim) + 3 + 8 + 3 + 2
        self.trunk = nn.Sequential(
            nn.Linear(3 * int(context_dim) + scalar_dim, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.ReLU(inplace=True),
        )
        self.help_head = nn.Linear(int(hidden_dim), 2)
        self.step_head = nn.Linear(int(hidden_dim), 2)
        nn.init.zeros_(self.help_head.weight)
        nn.init.constant_(
            self.help_head.bias,
            math.log(init_help_probability / (1.0 - init_help_probability)))
        nn.init.zeros_(self.step_head.weight)
        nn.init.zeros_(self.step_head.bias)

    @staticmethod
    def _batch_scalar(value, reference, default=0.0):
        batch_size = reference.shape[0]
        if value is None:
            return reference.new_full((batch_size, 1), float(default))
        if not torch.is_tensor(value):
            value = torch.as_tensor(
                value, device=reference.device, dtype=reference.dtype)
        value = value.to(
            device=reference.device, dtype=reference.dtype).reshape(-1)
        if value.numel() == 1:
            value = value.repeat(batch_size)
        elif value.numel() != batch_size:
            raise ValueError("advantage scalar must contain one per sample")
        return value.reshape(batch_size, 1)

    @staticmethod
    def _clip_norm(vector, radius, eps=1e-6):
        norm = torch.linalg.norm(vector, dim=1, keepdim=True)
        scale = torch.minimum(
            torch.ones_like(norm), radius / torch.clamp(norm, min=eps))
        return vector * scale

    def forward(
            self,
            observation_box,
            observation_feature,
            observation_stats,
            motion_feature,
            motion_proposal_xy,
            motion_log_sigma_xy,
            motion_valid,
            history_valid_ratio,
            search_evidence_token,
            search_proposal_xy,
            search_valid,
            search_targetness_mean,
            search_targetness_max,
            search_entropy,
            search_effective_sample_size,
            search_extension_weight_ratio,
            search_available_count,
            search_extension_count,
            search_overlap_count,
            query_delta_t,
            gap_ratio,
            enabled_scale=1.0):
        if observation_box.dim() != 2 or observation_box.shape[1] != 4:
            raise ValueError("observation box must have shape [B,4]")
        enabled_scale = float(enabled_scale)
        if not 0.0 <= enabled_scale <= 1.0:
            raise ValueError("advantage fusion scale must be in [0,1]")
        batch_size = observation_box.shape[0]
        if enabled_scale == 0.0:
            return observation_box, {
                "advantage_help_logits": observation_box.new_zeros(
                    (batch_size, 2)),
                "advantage_step_logits": observation_box.new_zeros(
                    (batch_size, 2)),
                "advantage_help_probability": observation_box.new_zeros(
                    (batch_size, 2)),
                "advantage_step_ratio": observation_box.new_zeros(
                    (batch_size, 2)),
                "advantage_raw_weight": observation_box.new_zeros(
                    (batch_size, 2)),
                "advantage_applied_weight": observation_box.new_zeros(
                    (batch_size, 2)),
                "advantage_applied_probability": torch.cat((
                    observation_box.new_ones((batch_size, 1)),
                    observation_box.new_zeros((batch_size, 2))), dim=1),
                "advantage_correction_xy": observation_box.new_zeros(
                    (batch_size, 2)),
                "advantage_fusion_radius": observation_box.new_zeros(
                    (batch_size,)),
                "advantage_motion_valid": observation_box.new_zeros(
                    (batch_size,)),
                "advantage_search_valid": observation_box.new_zeros(
                    (batch_size,)),
                "advantage_fusion_enabled_scale":
                    observation_box.new_tensor(0.0),
            }

        observation = observation_box.detach()
        observation_xy = observation[:, :2]
        observation_feature = observation_feature.detach()
        observation_stats = observation_stats.detach()
        motion_feature = motion_feature.detach()
        motion_xy = motion_proposal_xy.detach()
        motion_log_sigma_xy = motion_log_sigma_xy.detach()
        search_evidence_token = search_evidence_token.detach()
        search_xy = search_proposal_xy.detach()

        motion_valid = self._batch_scalar(
            motion_valid, observation_xy) > 0
        search_valid = self._batch_scalar(
            search_valid, observation_xy) > 0
        motion_finite = (
            torch.isfinite(motion_xy).all(dim=1, keepdim=True)
            & torch.isfinite(motion_feature).all(dim=1, keepdim=True)
            & torch.isfinite(motion_log_sigma_xy).all(dim=1, keepdim=True))
        search_finite = (
            torch.isfinite(search_xy).all(dim=1, keepdim=True)
            & torch.isfinite(search_evidence_token).all(dim=1, keepdim=True))
        motion_valid = motion_valid & motion_finite
        search_valid = search_valid & search_finite
        motion_xy = torch.where(
            torch.isfinite(motion_xy), motion_xy, observation_xy)
        search_xy = torch.where(
            torch.isfinite(search_xy), search_xy, observation_xy)

        query_dt = torch.clamp(torch.nan_to_num(self._batch_scalar(
            query_delta_t, observation_xy, default=0.1), nan=0.1), min=0.0)
        gap_ratio = torch.nan_to_num(self._batch_scalar(
            gap_ratio, observation_xy, default=1.0), nan=1.0)
        history_valid_ratio = torch.nan_to_num(self._batch_scalar(
            history_valid_ratio, observation_xy), nan=0.0)
        search_scalars = [
            search_targetness_mean,
            search_targetness_max,
            search_entropy,
            search_effective_sample_size,
            search_extension_weight_ratio,
        ]
        search_scalars = [torch.nan_to_num(self._batch_scalar(
            value, observation_xy), nan=0.0) for value in search_scalars]
        count_scalars = [
            search_available_count, search_extension_count,
            search_overlap_count,
        ]
        count_scalars = [
            torch.log1p(torch.clamp(torch.nan_to_num(self._batch_scalar(
                value, observation_xy), nan=0.0), min=0.0)) / 8.0
            for value in count_scalars
        ]

        obs_motion = torch.linalg.norm(
            motion_xy - observation_xy, dim=1, keepdim=True)
        obs_search = torch.linalg.norm(
            search_xy - observation_xy, dim=1, keepdim=True)
        motion_search = torch.linalg.norm(
            motion_xy - search_xy, dim=1, keepdim=True)
        gate_input = torch.cat((
            self.observation_projection(torch.nan_to_num(
                observation_feature, nan=0.0, posinf=0.0, neginf=0.0)),
            self.motion_projection(torch.nan_to_num(
                motion_feature, nan=0.0, posinf=0.0, neginf=0.0)),
            self.search_projection(torch.nan_to_num(
                search_evidence_token, nan=0.0, posinf=0.0, neginf=0.0)),
            torch.nan_to_num(
                observation_stats, nan=0.0, posinf=0.0, neginf=0.0),
            torch.nan_to_num(
                motion_log_sigma_xy, nan=0.0, posinf=0.0, neginf=0.0),
            history_valid_ratio,
            *search_scalars,
            *count_scalars,
            obs_motion,
            obs_search,
            motion_search,
            query_dt,
            gap_ratio,
        ), dim=1)
        hidden = self.trunk(gate_input)
        help_logits = self.help_head(hidden)
        step_logits = self.step_head(hidden)
        help_probability = torch.sigmoid(help_logits)
        step_ratio = torch.sigmoid(step_logits)
        candidate_valid = torch.cat((motion_valid, search_valid), dim=1)
        raw_weight = (
            help_probability * step_ratio
            * candidate_valid.to(help_probability.dtype))

        max_aux_mass = self.normal_aux_mass + (
            self.gap_aux_mass - self.normal_aux_mass
        ) * torch.clamp(gap_ratio - 1.0, min=0.0, max=1.0)
        raw_mass = raw_weight.sum(dim=1, keepdim=True)
        mass_scale = torch.minimum(
            torch.ones_like(raw_mass),
            max_aux_mass / torch.clamp(raw_mass, min=self.eps))
        applied_weight = raw_weight * mass_scale

        radius = torch.clamp(
            self.radius_base + self.radius_per_second * query_dt,
            max=self.radius_max)
        motion_residual = self._clip_norm(
            motion_xy - observation_xy, radius, eps=self.eps)
        search_residual = self._clip_norm(
            search_xy - observation_xy, radius, eps=self.eps)
        correction = (
            applied_weight[:, 0:1] * motion_residual
            + applied_weight[:, 1:2] * search_residual)
        correction = self._clip_norm(correction, radius, eps=self.eps)
        correction = correction * enabled_scale
        candidate_xy = observation_xy + correction
        any_valid = (motion_valid | search_valid).expand_as(candidate_xy)
        final_xy = torch.where(any_valid, candidate_xy, observation_xy)
        final_box = torch.cat((final_xy, observation[:, 2:]), dim=1)
        applied_probability = torch.cat((
            1.0 - applied_weight.sum(dim=1, keepdim=True),
            applied_weight,
        ), dim=1)
        return final_box, {
            "advantage_help_logits": help_logits,
            "advantage_step_logits": step_logits,
            "advantage_help_probability": help_probability,
            "advantage_step_ratio": step_ratio,
            "advantage_raw_weight": raw_weight,
            "advantage_applied_weight": applied_weight,
            "advantage_applied_probability": applied_probability,
            "advantage_motion_residual_xy": motion_residual,
            "advantage_search_residual_xy": search_residual,
            "advantage_correction_xy": torch.where(
                any_valid, correction, torch.zeros_like(correction)),
            "advantage_fusion_radius": radius.squeeze(1),
            "advantage_motion_valid": motion_valid.to(
                observation_xy.dtype).squeeze(1),
            "advantage_search_valid": search_valid.to(
                observation_xy.dtype).squeeze(1),
            "advantage_fusion_enabled_scale": observation_xy.new_tensor(
                enabled_scale),
        }


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
