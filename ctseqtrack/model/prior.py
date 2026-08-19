"""Canonical v25 B1 ordered physical-time motion prior."""

import math

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from ctseqtrack.model.cfc import FullGatedCfCCell


def wrap_angle(angle):
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def motion_aligned_axes(velocity_xy, min_speed=0.2, eps=1e-6):
    """Return stable parallel/perpendicular axes for planar motion."""
    if velocity_xy.dim() != 2 or velocity_xy.shape[1] != 2:
        raise ValueError("velocity_xy must have shape [B,2]")
    speed = torch.linalg.norm(velocity_xy, dim=1, keepdim=True)
    fallback = torch.zeros_like(velocity_xy)
    fallback[:, 0] = 1.0
    direction = torch.where(
        (speed >= float(min_speed)).expand_as(velocity_xy),
        velocity_xy / torch.clamp(speed, min=float(eps)),
        fallback,
    )
    perpendicular = torch.stack((-direction[:, 1], direction[:, 0]), dim=1)
    return direction, perpendicular, speed.squeeze(1)


def motion_aligned_covariance(
    velocity_xy, log_sigma_parallel_perp, min_speed=0.2, eps=1e-6, direction_xy=None
):
    """Convert motion-frame standard deviations to an XY covariance."""
    if log_sigma_parallel_perp.shape != velocity_xy.shape:
        raise ValueError("log_sigma_parallel_perp must match velocity_xy")
    default_direction, _, speed = motion_aligned_axes(
        velocity_xy, min_speed=min_speed, eps=eps
    )
    if direction_xy is None:
        direction = default_direction
    else:
        if direction_xy.shape != velocity_xy.shape:
            raise ValueError("direction_xy must match velocity_xy")
        direction_norm = torch.linalg.norm(direction_xy, dim=1, keepdim=True)
        direction = torch.where(
            (direction_norm > float(eps)).expand_as(direction_xy),
            direction_xy / torch.clamp(direction_norm, min=float(eps)),
            default_direction,
        )
    perpendicular = torch.stack((-direction[:, 1], direction[:, 0]), dim=1)
    log_sigma = log_sigma_parallel_perp
    low_speed = speed < float(min_speed)
    isotropic = log_sigma.mean(dim=1, keepdim=True).expand_as(log_sigma)
    log_sigma = torch.where(low_speed.unsqueeze(1), isotropic, log_sigma)
    variance = torch.exp(2.0 * log_sigma)
    covariance = variance[:, 0, None, None] * direction.unsqueeze(
        2
    ) * direction.unsqueeze(1) + variance[:, 1, None, None] * perpendicular.unsqueeze(
        2
    ) * perpendicular.unsqueeze(
        1
    )
    marginal_log_sigma = 0.5 * torch.log(
        torch.clamp(torch.diagonal(covariance, dim1=1, dim2=2), min=float(eps))
    )
    return {
        "log_sigma_parallel_perp": log_sigma,
        "covariance_xy": covariance,
        "motion_direction_xy": direction,
        "log_sigma_xy": marginal_log_sigma,
        "motion_speed": speed,
    }


def physical_motion_uncertainty_loss(
    mean_xy, target_xy, log_sigma_parallel_perp, motion_direction_xy, valid, eps=1e-6
):
    """Masked-ready robust mean and heteroscedastic Gaussian terms.

    Returned losses are per sample.  Callers own weighting and masked
    reduction so this helper can also be used by calibration diagnostics.
    """
    if mean_xy.shape != target_xy.shape or mean_xy.shape[-1] != 2:
        raise ValueError("mean_xy and target_xy must both have shape [B,2]")
    if log_sigma_parallel_perp.shape != mean_xy.shape:
        raise ValueError("motion log sigma must have shape [B,2]")
    if motion_direction_xy.shape != mean_xy.shape:
        raise ValueError("motion direction must have shape [B,2]")
    perpendicular = torch.stack(
        (-motion_direction_xy[:, 1], motion_direction_xy[:, 0]), dim=1
    )
    error = target_xy - mean_xy
    aligned_error = torch.stack(
        (
            (error * motion_direction_xy).sum(dim=1),
            (error * perpendicular).sum(dim=1),
        ),
        dim=1,
    )
    safe_log_sigma = torch.clamp(
        torch.nan_to_num(log_sigma_parallel_perp), min=-4.0, max=2.5
    )
    nll_per_axis = 0.5 * (
        aligned_error.pow(2) * torch.exp(-2.0 * safe_log_sigma) + 2.0 * safe_log_sigma
    )
    robust_mean = torch.nn.functional.smooth_l1_loss(
        mean_xy, target_xy, reduction="none"
    ).mean(dim=1)
    valid = valid.to(device=mean_xy.device, dtype=mean_xy.dtype).reshape(-1)
    finite = (
        torch.isfinite(aligned_error).all(dim=1)
        & torch.isfinite(nll_per_axis).all(dim=1)
    ).to(mean_xy.dtype)
    return {
        "mean_per_sample": robust_mean,
        "nll_per_sample": nll_per_axis.sum(dim=1),
        "aligned_error": aligned_error,
        "valid": valid * finite,
    }


class OrderedPhysicalMotionEncoder(nn.Module):
    """Predict candidate-independent xy motion from an ordered box history.

    Boxes are supplied newest-to-oldest, matching the tracker contract.  The
    The selected temporal backend consumes transitions oldest-to-newest, while
    a causal latest-velocity extrapolation provides a useful zero-initialized
    cold start. Physical time is structural: transition velocities divide by
    their measured gaps and the predicted rate is integrated over the query gap.
    """

    def __init__(
        self,
        hidden_dim=128,
        step_dim=64,
        eps=1e-3,
        time_scale=0.5,
        residual_velocity_scale=4.0,
        initial_sigma=0.5,
        motion_aligned_uncertainty=False,
        min_direction_speed=0.2,
        shared_kinematic_anchor=False,
        max_acceleration=8.0,
        max_displacement=12.0,
        acceleration_weight=0.5,
        temporal_backend="gru",
        cfc_backbone_units=105,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.step_dim = int(step_dim)
        self.eps = float(eps)
        self.time_scale = max(float(time_scale), self.eps)
        self.residual_velocity_scale = float(residual_velocity_scale)
        self.motion_aligned_uncertainty = bool(motion_aligned_uncertainty)
        self.min_direction_speed = float(min_direction_speed)
        self.shared_kinematic_anchor = bool(shared_kinematic_anchor)
        self.max_acceleration = float(max_acceleration)
        self.max_displacement = float(max_displacement)
        self.acceleration_weight = float(acceleration_weight)
        self.temporal_backend = str(temporal_backend).strip().lower()
        self.cfc_backbone_units = int(cfc_backbone_units)
        if self.hidden_dim <= 0 or self.step_dim <= 0:
            raise ValueError("physical motion encoder dimensions must be positive")
        if self.residual_velocity_scale <= 0:
            raise ValueError("physical residual velocity scale must be positive")
        if initial_sigma <= 0:
            raise ValueError("physical initial sigma must be positive")
        if self.min_direction_speed < 0:
            raise ValueError("minimum direction speed must be non-negative")
        if self.max_acceleration <= 0 or self.max_displacement <= 0:
            raise ValueError(
                "shared-anchor acceleration/displacement caps must be positive"
            )
        if not 0.0 <= self.acceleration_weight <= 1.0:
            raise ValueError("shared-anchor acceleration weight must be in [0,1]")
        if self.temporal_backend not in ("gru", "cfc"):
            raise ValueError("B1 temporal backend must be gru or cfc")
        if self.cfc_backbone_units <= 0:
            raise ValueError("CfC backbone units must be positive")

        # xy velocity, xy displacement, sin/cos yaw change, log pair gap,
        # query/pair ratio, and transition-valid flag.
        self.step_projection = nn.Sequential(
            nn.Linear(9, self.step_dim),
            nn.LayerNorm(self.step_dim),
            nn.ReLU(inplace=True),
        )
        if self.temporal_backend == "gru":
            # Preserve the historical attribute and state-dict keys exactly.
            self.gru = nn.GRU(
                input_size=self.step_dim,
                hidden_size=self.hidden_dim,
                num_layers=1,
                batch_first=True,
            )
        else:
            self.cfc = FullGatedCfCCell(
                input_size=self.step_dim,
                hidden_size=self.hidden_dim,
                backbone_units=self.cfc_backbone_units,
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
        # Historical B1-v3 instances keep this non-persistent.  The new
        # calibrated path stores it in checkpoints without breaking strict
        # loading of old state dictionaries.
        self.register_buffer(
            "log_sigma_calibration",
            torch.zeros(2),
            persistent=self.motion_aligned_uncertainty,
        )

    def set_uncertainty_calibration(self, log_scale):
        value = torch.as_tensor(
            log_scale,
            device=self.log_sigma_calibration.device,
            dtype=self.log_sigma_calibration.dtype,
        ).reshape(-1)
        if value.numel() == 1:
            value = value.repeat(2)
        if value.numel() != 2 or not bool(torch.isfinite(value).all()):
            raise ValueError("calibration log scale must contain two finite values")
        self.log_sigma_calibration.copy_(value)

    def _uncertainty_outputs(self, basis_velocity_xy, raw_log_sigma, direction_xy=None):
        if not self.motion_aligned_uncertainty:
            direction, _, speed = motion_aligned_axes(
                basis_velocity_xy, min_speed=max(self.eps, 1e-6), eps=self.eps
            )
            if direction_xy is not None:
                direction_norm = torch.linalg.norm(direction_xy, dim=1, keepdim=True)
                direction = torch.where(
                    (direction_norm > self.eps).expand_as(direction_xy),
                    direction_xy / torch.clamp(direction_norm, min=self.eps),
                    direction,
                )
            return {
                "log_sigma_parallel_perp": raw_log_sigma,
                "covariance_xy": torch.diag_embed(torch.exp(2.0 * raw_log_sigma)),
                "motion_direction_xy": direction,
                "log_sigma_xy": raw_log_sigma,
                "motion_speed": speed,
            }
        calibrated = torch.clamp(
            raw_log_sigma
            + self.log_sigma_calibration.to(
                device=raw_log_sigma.device, dtype=raw_log_sigma.dtype
            ).unsqueeze(0),
            min=-4.0,
            max=2.5,
        )
        return motion_aligned_covariance(
            basis_velocity_xy,
            calibrated,
            min_speed=self.min_direction_speed,
            eps=self.eps,
            direction_xy=direction_xy,
        )

    def _format_query_gap(self, value, batch_size, reference):
        if value is None:
            value = reference.new_full((batch_size,), self.time_scale)
        elif not torch.is_tensor(value):
            value = torch.as_tensor(
                value, device=reference.device, dtype=reference.dtype
            )
        value = value.to(device=reference.device, dtype=reference.dtype)
        if value.numel() == 1:
            value = value.reshape(1).repeat(batch_size)
        elif value.numel() != batch_size:
            raise ValueError(
                "current_delta_t must contain one value or one per batch item"
            )
        finite = torch.isfinite(value.reshape(batch_size))
        value = torch.nan_to_num(
            value.reshape(batch_size),
            nan=self.time_scale,
            posinf=self.time_scale,
            neginf=self.time_scale,
        )
        return torch.clamp(value, min=self.eps), finite

    def _encode_transitions(
        self, projected, chronological_valid, chronological_pair_gap
    ):
        """Aggregate valid oldest-to-newest transitions with the chosen backend."""
        if self.temporal_backend == "cfc":
            hidden = projected.new_zeros((projected.shape[0], self.hidden_dim))
            normalized_gap = chronological_pair_gap / self.time_scale
            for index in range(projected.shape[1]):
                updated = self.cfc(
                    projected[:, index], hidden, normalized_gap[:, index]
                )
                valid = chronological_valid[:, index].unsqueeze(1)
                hidden = torch.where(valid, updated, hidden)
            return hidden

        # A zero vector is not a no-op for a GRU with biases. Compact valid
        # transitions before packing so padded history cannot alter the state.
        chronological_index = torch.arange(
            chronological_valid.shape[1],
            device=chronological_valid.device,
            dtype=torch.int64,
        ).unsqueeze(0)
        compact_key = chronological_index + (
            (~chronological_valid).to(torch.int64) * chronological_valid.shape[1]
        )
        compact_indices = torch.argsort(compact_key, dim=1)
        compact_projected = torch.gather(
            projected,
            dim=1,
            index=compact_indices.unsqueeze(-1).expand_as(projected),
        )
        transition_count = chronological_valid.to(projected.dtype).sum(dim=1)
        packed_projected = pack_padded_sequence(
            compact_projected,
            lengths=torch.clamp(transition_count, min=1).to(torch.long).cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, ordered_hidden = self.gru(packed_projected)
        return ordered_hidden[-1]

    def kinematic_fallback(self, ref_boxs, delta_t, valid_mask, current_delta_t=None):
        """Parameter-free fallback used by the strict ``-B1`` ablation."""
        if ref_boxs.dim() != 3 or ref_boxs.shape[-1] != 4:
            raise ValueError("motion ref_boxs must have shape [B,H,4]")
        batch_size, history_length, _ = ref_boxs.shape
        if history_length < 2:
            raise ValueError("kinematic fallback requires at least two boxes")
        if delta_t.dim() == 1:
            delta_t = delta_t.unsqueeze(0)
        delta_t = delta_t.to(device=ref_boxs.device, dtype=ref_boxs.dtype)
        valid_mask = valid_mask.to(device=ref_boxs.device, dtype=ref_boxs.dtype)
        if valid_mask.dim() == 1:
            valid_mask = valid_mask.unsqueeze(0)
        if valid_mask.shape != ref_boxs.shape[:2]:
            raise ValueError("motion valid_mask must have shape [B,H]")
        query_gap, query_finite = self._format_query_gap(
            current_delta_t, batch_size, ref_boxs
        )
        safe_boxs = torch.nan_to_num(ref_boxs, nan=0.0, posinf=0.0, neginf=0.0)
        safe_delta_t = torch.nan_to_num(
            delta_t, nan=self.time_scale, posinf=self.time_scale, neginf=self.time_scale
        )
        if safe_delta_t.shape[1] < history_length:
            safe_delta_t = torch.cat(
                (
                    safe_delta_t,
                    safe_delta_t[:, -1:].expand(
                        -1, history_length - safe_delta_t.shape[1]
                    ),
                ),
                dim=1,
            )
        pair_gap = torch.clamp(safe_delta_t[:, 1:history_length], min=self.eps)
        pair_valid = (valid_mask[:, :-1] > 0) & (valid_mask[:, 1:] > 0)
        finite_row = (
            torch.isfinite(ref_boxs).flatten(1).all(dim=1)
            & torch.isfinite(delta_t).all(dim=1)
            & query_finite
        )
        valid = (pair_valid[:, 0] & finite_row).to(ref_boxs.dtype)
        velocity_xy = (
            safe_boxs[:, :-1, :2] - safe_boxs[:, 1:, :2]
        ) / pair_gap.unsqueeze(2)
        base_velocity = velocity_xy[:, 0] * valid.unsqueeze(1)
        pair_valid_f = pair_valid.to(ref_boxs.dtype)
        transition_count = pair_valid_f.sum(dim=1)
        nominal_gap = (pair_gap * pair_valid_f).sum(dim=1) / torch.clamp(
            transition_count, min=1.0
        )
        gap_ratio_raw = query_gap / torch.clamp(nominal_gap, min=self.eps)

        older_pair_valid = (
            pair_valid[:, 1]
            if pair_valid.shape[1] > 1
            else torch.zeros_like(pair_valid[:, 0])
        )
        older_velocity = (
            velocity_xy[:, 1]
            if velocity_xy.shape[1] > 1
            else torch.zeros_like(base_velocity)
        )
        acceleration_gap = torch.clamp(
            0.5
            * (
                pair_gap[:, 0]
                + (pair_gap[:, 1] if pair_gap.shape[1] > 1 else pair_gap[:, 0])
            ),
            min=self.eps,
        )
        acceleration = (base_velocity - older_velocity) / acceleration_gap.unsqueeze(1)
        acceleration = acceleration * older_pair_valid.to(ref_boxs.dtype).unsqueeze(1)
        acceleration_norm = torch.linalg.norm(acceleration, dim=1, keepdim=True)
        acceleration = acceleration * torch.clamp(
            self.max_acceleration / torch.clamp(acceleration_norm, min=self.eps),
            max=1.0,
        )
        kinematic_xy = base_velocity * query_gap.unsqueeze(
            1
        ) + self.acceleration_weight * 0.5 * acceleration * query_gap.pow(2).unsqueeze(
            1
        )
        displacement_norm = torch.linalg.norm(kinematic_xy, dim=1, keepdim=True)
        kinematic_xy = (
            kinematic_xy
            * torch.clamp(
                self.max_displacement / torch.clamp(displacement_norm, min=self.eps),
                max=1.0,
            )
            * valid.unsqueeze(1)
        )

        velocity_spread = torch.linalg.norm(base_velocity - older_velocity, dim=1)
        velocity_spread = velocity_spread * older_pair_valid.to(ref_boxs.dtype)
        envelope = torch.stack(
            (
                torch.clamp(
                    0.25 + 0.25 * query_gap + 0.5 * velocity_spread * query_gap, max=4.0
                ),
                torch.clamp(
                    0.20 + 0.15 * query_gap + 0.25 * velocity_spread * query_gap,
                    max=3.0,
                ),
            ),
            dim=1,
        )
        direction, _, _ = motion_aligned_axes(
            kinematic_xy, min_speed=self.min_direction_speed, eps=self.eps
        )
        log_sigma = torch.log(torch.clamp(envelope, min=0.1))
        uncertainty = motion_aligned_covariance(
            base_velocity,
            log_sigma,
            min_speed=self.min_direction_speed,
            eps=self.eps,
            direction_xy=direction,
        )
        zeros_xy = torch.zeros_like(kinematic_xy)
        gap_ratio = torch.where(
            valid > 0, gap_ratio_raw, torch.ones_like(gap_ratio_raw)
        )
        return {
            "feature": ref_boxs.new_zeros((batch_size, self.hidden_dim)),
            "velocity_xy": kinematic_xy
            / torch.clamp(query_gap.unsqueeze(1), min=self.eps),
            "basis_velocity_xy": base_velocity,
            "prior_xy": kinematic_xy,
            "mu_xy": kinematic_xy,
            "kinematic_prior_xy": kinematic_xy,
            "residual_unit_parallel_perp": zeros_xy,
            "residual_xy": zeros_xy,
            "envelope_parallel_perp": envelope,
            **uncertainty,
            "direction_xy": uncertainty["motion_direction_xy"],
            "valid": valid,
            "gap_ratio": gap_ratio,
            "source_id": torch.zeros(
                batch_size, device=ref_boxs.device, dtype=torch.long
            ),
        }

    def forward(self, ref_boxs, delta_t, valid_mask, current_delta_t=None):
        if ref_boxs.dim() != 3 or ref_boxs.shape[-1] != 4:
            raise ValueError("motion ref_boxs must have shape [B,H,4]")
        batch_size, history_length, _ = ref_boxs.shape
        if delta_t.dim() == 1:
            delta_t = delta_t.unsqueeze(0)
        delta_t = delta_t.to(device=ref_boxs.device, dtype=ref_boxs.dtype)
        if delta_t.dim() != 2 or delta_t.shape[0] != batch_size:
            raise ValueError("motion delta_t must have shape [B,H]")
        valid_mask = valid_mask.to(device=ref_boxs.device, dtype=ref_boxs.dtype)
        if valid_mask.dim() == 1:
            valid_mask = valid_mask.unsqueeze(0)
        if valid_mask.shape != ref_boxs.shape[:2]:
            raise ValueError("motion valid_mask must have shape [B,H]")
        query_gap, query_finite = self._format_query_gap(
            current_delta_t, batch_size, ref_boxs
        )

        # PyTorch 2.0's Tensor.all accepts one dimension at a time; flattening
        # preserves the per-sample finite check and remains compatible with
        # newer releases.
        finite_row = torch.isfinite(ref_boxs).flatten(1).all(dim=1)
        finite_row = finite_row & torch.isfinite(delta_t).all(dim=1)
        finite_row = finite_row & query_finite
        safe_boxs = torch.nan_to_num(ref_boxs, nan=0.0, posinf=0.0, neginf=0.0)
        safe_delta_t = torch.nan_to_num(
            delta_t, nan=self.time_scale, posinf=self.time_scale, neginf=self.time_scale
        )

        zeros_xy = ref_boxs.new_zeros((batch_size, 2))
        zeros_feature = ref_boxs.new_zeros((batch_size, self.hidden_dim))
        initial_log_sigma = ref_boxs.new_full(
            (batch_size, 2), self.log_sigma_head.bias[0].item()
        )
        if history_length < 2:
            uncertainty = self._uncertainty_outputs(zeros_xy, initial_log_sigma)
            return {
                "feature": zeros_feature,
                "velocity_xy": zeros_xy,
                "basis_velocity_xy": zeros_xy,
                "prior_xy": zeros_xy,
                "mu_xy": zeros_xy,
                "kinematic_prior_xy": zeros_xy,
                "residual_unit_parallel_perp": zeros_xy,
                "residual_xy": zeros_xy,
                "envelope_parallel_perp": zeros_xy,
                **uncertainty,
                "direction_xy": uncertainty["motion_direction_xy"],
                "valid": ref_boxs.new_zeros((batch_size,)),
                "gap_ratio": ref_boxs.new_ones((batch_size,)),
                "source_id": torch.zeros(
                    batch_size, device=ref_boxs.device, dtype=torch.long
                ),
            }

        if safe_delta_t.shape[1] < history_length:
            pad = safe_delta_t[:, -1:].expand(
                -1, history_length - safe_delta_t.shape[1]
            )
            safe_delta_t = torch.cat((safe_delta_t, pad), dim=1)
        pair_gap = torch.clamp(safe_delta_t[:, 1:history_length], min=self.eps)
        pair_valid = (valid_mask[:, :-1] > 0) & (valid_mask[:, 1:] > 0)
        pair_valid_f = pair_valid.to(ref_boxs.dtype)

        newer = safe_boxs[:, :-1]
        older = safe_boxs[:, 1:]
        displacement_xy = newer[:, :, :2] - older[:, :, :2]
        velocity_xy = displacement_xy / pair_gap.unsqueeze(-1)
        yaw_delta = wrap_angle(newer[:, :, 3] - older[:, :, 3])
        query_ratio = (query_gap.unsqueeze(1) / pair_gap).expand_as(pair_gap)
        step_features = torch.cat(
            (
                velocity_xy,
                displacement_xy,
                torch.sin(yaw_delta).unsqueeze(-1),
                (torch.cos(yaw_delta) - 1.0).unsqueeze(-1),
                torch.log1p(pair_gap / self.time_scale).unsqueeze(-1),
                query_ratio.unsqueeze(-1),
                pair_valid_f.unsqueeze(-1),
            ),
            dim=-1,
        )
        step_features = torch.nan_to_num(step_features, nan=0.0, posinf=0.0, neginf=0.0)
        step_features = step_features * pair_valid_f.unsqueeze(-1)

        chronological = torch.flip(step_features, dims=(1,))
        projected = self.step_projection(chronological)
        chronological_valid = torch.flip(pair_valid, dims=(1,))
        projected = projected * chronological_valid.unsqueeze(-1)
        transition_count = pair_valid_f.sum(dim=1)
        ordered_state = self._encode_transitions(
            projected,
            chronological_valid,
            torch.flip(pair_gap, dims=(1,)),
        )

        nominal_gap = (pair_gap * pair_valid_f).sum(dim=1) / torch.clamp(
            transition_count, min=1.0
        )
        gap_ratio_raw = query_gap / torch.clamp(nominal_gap, min=self.eps)
        context = self.context(
            torch.cat(
                (
                    ordered_state,
                    torch.log1p(query_gap / self.time_scale).unsqueeze(1),
                    torch.log1p(gap_ratio_raw).unsqueeze(1),
                ),
                dim=1,
            )
        )

        recent_pair_valid = pair_valid[:, 0]
        valid = (recent_pair_valid & finite_row).to(ref_boxs.dtype)
        base_velocity = velocity_xy[:, 0] * valid.unsqueeze(1)
        if self.shared_kinematic_anchor:
            # The deterministic anchor mirrors ctseqtrack.data.search exactly: the
            # newest velocity is primary and the next valid transition only
            # supplies a bounded, deliberately under-weighted acceleration.
            older_pair_valid = (
                pair_valid[:, 1]
                if pair_valid.shape[1] > 1
                else torch.zeros_like(recent_pair_valid)
            )
            older_velocity = (
                velocity_xy[:, 1]
                if velocity_xy.shape[1] > 1
                else torch.zeros_like(base_velocity)
            )
            acceleration_gap = torch.clamp(
                0.5
                * (
                    pair_gap[:, 0]
                    + (pair_gap[:, 1] if pair_gap.shape[1] > 1 else pair_gap[:, 0])
                ),
                min=self.eps,
            )
            acceleration = (
                base_velocity - older_velocity
            ) / acceleration_gap.unsqueeze(1)
            acceleration = acceleration * older_pair_valid.to(ref_boxs.dtype).unsqueeze(
                1
            )
            acceleration_norm = torch.linalg.norm(acceleration, dim=1, keepdim=True)
            acceleration = acceleration * torch.clamp(
                self.max_acceleration / torch.clamp(acceleration_norm, min=self.eps),
                max=1.0,
            )
            kinematic_prior_xy = base_velocity * query_gap.unsqueeze(
                1
            ) + self.acceleration_weight * 0.5 * acceleration * query_gap.pow(
                2
            ).unsqueeze(
                1
            )
            displacement_norm = torch.linalg.norm(
                kinematic_prior_xy, dim=1, keepdim=True
            )
            kinematic_prior_xy = (
                kinematic_prior_xy
                * torch.clamp(
                    self.max_displacement
                    / torch.clamp(displacement_norm, min=self.eps),
                    max=1.0,
                )
                * valid.unsqueeze(1)
            )

            velocity_spread = torch.linalg.norm(base_velocity - older_velocity, dim=1)
            velocity_spread = velocity_spread * older_pair_valid.to(ref_boxs.dtype)
            envelope_parallel = torch.clamp(
                0.25 + 0.25 * query_gap + 0.5 * velocity_spread * query_gap,
                max=4.0,
            )
            envelope_perpendicular = torch.clamp(
                0.20 + 0.15 * query_gap + 0.25 * velocity_spread * query_gap,
                max=3.0,
            )
            envelope = torch.stack((envelope_parallel, envelope_perpendicular), dim=1)

            anchor_norm = torch.linalg.norm(kinematic_prior_xy, dim=1, keepdim=True)
            base_direction, _, _ = motion_aligned_axes(
                base_velocity,
                min_speed=self.min_direction_speed,
                eps=self.eps,
            )
            direction = torch.where(
                (anchor_norm > self.eps).expand_as(kinematic_prior_xy),
                kinematic_prior_xy / torch.clamp(anchor_norm, min=self.eps),
                base_direction,
            )
            perpendicular = torch.stack((-direction[:, 1], direction[:, 0]), dim=1)
            residual_unit = torch.tanh(
                self.velocity_residual_head(context)
            ) * valid.unsqueeze(1)
            residual_xy = direction * (
                residual_unit[:, 0:1] * envelope_parallel.unsqueeze(1)
            ) + perpendicular * (
                residual_unit[:, 1:2] * envelope_perpendicular.unsqueeze(1)
            )
            prior_xy = (kinematic_prior_xy + residual_xy) * valid.unsqueeze(1)
            predicted_velocity = prior_xy / torch.clamp(
                query_gap.unsqueeze(1), min=self.eps
            )

            raw_sigma = self.log_sigma_head(context)
            bounded_sigma = 0.1 + (envelope - 0.1) * torch.sigmoid(raw_sigma)
            bounded_sigma = torch.clamp(bounded_sigma, min=0.1)
            raw_log_sigma = torch.log(bounded_sigma)
            uncertainty = motion_aligned_covariance(
                base_velocity,
                raw_log_sigma,
                min_speed=self.min_direction_speed,
                eps=self.eps,
                direction_xy=direction,
            )
        else:
            residual_velocity = self.residual_velocity_scale * torch.tanh(
                self.velocity_residual_head(context)
            )
            predicted_velocity = (base_velocity + residual_velocity) * valid.unsqueeze(
                1
            )
            prior_xy = predicted_velocity * query_gap.unsqueeze(1)
            kinematic_prior_xy = (
                base_velocity * query_gap.unsqueeze(1) * valid.unsqueeze(1)
            )
            raw_log_sigma = torch.clamp(self.log_sigma_head(context), min=-4.0, max=2.5)
            envelope = ref_boxs.new_zeros((batch_size, 2))
            residual_unit = ref_boxs.new_zeros((batch_size, 2))
            residual_xy = prior_xy - kinematic_prior_xy
        base_direction, _, base_speed = motion_aligned_axes(
            base_velocity,
            min_speed=self.min_direction_speed,
            eps=self.eps,
        )
        if not self.shared_kinematic_anchor:
            displacement_norm = torch.linalg.norm(prior_xy, dim=1, keepdim=True)
            displacement_direction = torch.where(
                (displacement_norm > self.eps).expand_as(prior_xy),
                prior_xy / torch.clamp(displacement_norm, min=self.eps),
                base_direction,
            )
            direction = torch.where(
                (base_speed < self.min_direction_speed).unsqueeze(1),
                displacement_direction,
                base_direction,
            )
            uncertainty = self._uncertainty_outputs(
                base_velocity, raw_log_sigma, direction_xy=direction
            )
        gap_ratio = torch.where(
            valid > 0, gap_ratio_raw, torch.ones_like(gap_ratio_raw)
        )
        return {
            "feature": context * valid.unsqueeze(1),
            "velocity_xy": predicted_velocity,
            "basis_velocity_xy": base_velocity,
            "prior_xy": prior_xy,
            "mu_xy": prior_xy,
            "kinematic_prior_xy": kinematic_prior_xy,
            "residual_unit_parallel_perp": residual_unit,
            "residual_xy": residual_xy,
            "envelope_parallel_perp": envelope,
            **uncertainty,
            "direction_xy": uncertainty["motion_direction_xy"],
            "valid": valid,
            "gap_ratio": gap_ratio,
            "source_id": (valid > 0).to(torch.long),
        }


B1PhysicalTimePrior = OrderedPhysicalMotionEncoder

__all__ = [
    "B1PhysicalTimePrior",
    "OrderedPhysicalMotionEncoder",
    "motion_aligned_axes",
    "motion_aligned_covariance",
    "physical_motion_uncertainty_loss",
]
