import math

import torch
from torch import nn


def wrap_angle(angle):
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def clamp_vector_norm(vectors, max_norm, eps=1e-6):
    """Clamp the L2 norm of the last dimension without changing direction."""
    if max_norm is None or float(max_norm) <= 0:
        norms = torch.linalg.norm(vectors, dim=-1, keepdim=True)
        clamp_mask = torch.zeros_like(norms, dtype=torch.bool)
        return vectors, norms, clamp_mask

    max_norm = float(max_norm)
    norms = torch.linalg.norm(vectors, dim=-1, keepdim=True)
    scale = torch.clamp(max_norm / norms.clamp_min(float(eps)), max=1.0)
    clamped = vectors * scale
    clamp_mask = norms > max_norm
    return clamped, norms, clamp_mask


def build_innovation_radius(current_delta_t, base_radius=0.5,
                            radius_per_second=0.5, max_radius=2.0):
    """Build the explicit time-dependent M2 correction bound ``R(delta_t)``."""
    radius = (
        float(base_radius)
        + float(radius_per_second) * torch.clamp(current_delta_t, min=0.0)
    )
    radius = torch.clamp(radius, min=0.0)
    if max_radius is not None and float(max_radius) > 0:
        radius = torch.clamp(radius, max=float(max_radius))
    return radius


def apply_proposal_innovation(
        observation_displacement,
        dynamics_displacement,
        dynamics_valid,
        current_delta_t,
        alpha=0.0,
        enabled_scale=1.0,
        base_radius=0.5,
        radius_per_second=0.5,
        max_radius=2.0,
        eps=1e-6):
    """Move the observation proposal toward dynamics with a bounded innovation.

    The observation term is detached only inside the innovation, matching
    ``d_dyn - stopgrad(d_obs)``.  The final addition still gives the observation
    branch its ordinary identity gradient.  Zero alpha/scale and invalid
    dynamics are exact fallbacks, not sigmoid approximations.
    """
    if observation_displacement.shape != dynamics_displacement.shape:
        raise ValueError("observation and dynamics displacement shapes must match")
    if observation_displacement.dim() != 2 or observation_displacement.shape[1] != 3:
        raise ValueError("proposal displacements must have shape [B, 3]")

    device = observation_displacement.device
    dtype = observation_displacement.dtype
    batch_size = observation_displacement.shape[0]
    if not torch.is_tensor(current_delta_t):
        current_delta_t = torch.as_tensor(current_delta_t, device=device, dtype=dtype)
    current_delta_t = current_delta_t.to(device=device, dtype=dtype)
    if current_delta_t.numel() == 1:
        current_delta_t = current_delta_t.repeat(batch_size)
    current_delta_t = current_delta_t.reshape(batch_size, 1)

    if torch.is_tensor(alpha):
        alpha_tensor = alpha.to(device=device, dtype=dtype)
        if alpha_tensor.numel() == 1:
            alpha_tensor = alpha_tensor.repeat(batch_size)
        alpha_tensor = alpha_tensor.reshape(batch_size, 1)
    else:
        alpha_tensor = observation_displacement.new_full((batch_size, 1), float(alpha))
    if torch.any((alpha_tensor < 0.0) | (alpha_tensor > 1.0)):
        raise ValueError("proposal innovation alpha must be in [0, 1]")
    enabled_scale = float(enabled_scale)
    if not 0.0 <= enabled_scale <= 1.0:
        raise ValueError("proposal innovation enabled_scale must be in [0, 1]")

    if dynamics_valid is None:
        valid = observation_displacement.new_ones((batch_size, 1))
    else:
        valid = dynamics_valid.to(device=device, dtype=dtype).reshape(batch_size, 1)
        valid = (valid > 0).to(dtype)
    effective_alpha = alpha_tensor * enabled_scale * valid

    raw_innovation = dynamics_displacement - observation_displacement.detach()
    raw_norm = torch.linalg.norm(raw_innovation, dim=1, keepdim=True)
    radius = build_innovation_radius(
        current_delta_t,
        base_radius=base_radius,
        radius_per_second=radius_per_second,
        max_radius=max_radius,
    )
    norm_scale = torch.clamp(radius / raw_norm.clamp_min(float(eps)), max=1.0)
    clamped_innovation = raw_innovation * norm_scale
    clamp_mask = raw_norm > radius

    # ``where`` avoids 0 * NaN turning an invalid/disabled fallback into NaN.
    applied_innovation = torch.where(
        effective_alpha > 0,
        effective_alpha * clamped_innovation,
        torch.zeros_like(clamped_innovation),
    )
    final_displacement = observation_displacement + applied_innovation
    applied_norm = torch.linalg.norm(applied_innovation, dim=1, keepdim=True)
    aux = {
        "dynamics_innovation_raw": raw_innovation,
        "dynamics_innovation_clamped": clamped_innovation,
        "dynamics_innovation_applied": applied_innovation,
        "dynamics_innovation_raw_norm": raw_norm.squeeze(1),
        "dynamics_innovation_clamped_norm": torch.linalg.norm(
            clamped_innovation, dim=1),
        "dynamics_innovation_applied_norm": applied_norm.squeeze(1),
        "dynamics_innovation_radius": radius.squeeze(1),
        "dynamics_innovation_alpha": effective_alpha.squeeze(1),
        "dynamics_innovation_clamp_mask": clamp_mask.squeeze(1).to(dtype),
        "dynamics_innovation_applied_mask": (applied_norm > float(eps)).squeeze(1).to(dtype),
        "dynamics_innovation_invalid_fallback": (valid <= 0).squeeze(1).to(dtype),
    }
    return final_displacement, aux


class DynamicsResidualGate(nn.Module):
    """Small bounded gate for an observation-first dynamics residual.

    The last layer is initialized so that the initial alpha is close to
    ``init_alpha``. This keeps a freshly enabled dynamics branch from taking
    over the observation prediction before the reliability signal is learned.
    """

    def __init__(self, stats_dim=6, hidden_dim=16, max_alpha=0.2, init_alpha=0.0):
        super().__init__()
        self.max_alpha = float(max_alpha)
        if not 0.0 < self.max_alpha <= 1.0:
            raise ValueError("max_alpha must be in (0, 1].")

        hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(int(stats_dim), hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        output = self.net[-1]
        nn.init.zeros_(output.weight)
        init_ratio = float(init_alpha) / self.max_alpha
        init_ratio = min(max(init_ratio, 1e-4), 1.0 - 1e-4)
        nn.init.constant_(output.bias, math.log(init_ratio / (1.0 - init_ratio)))

    def forward(self, stats, dynamics_valid):
        alpha = self.max_alpha * torch.sigmoid(self.net(stats))
        if dynamics_valid is not None:
            dynamics_valid = dynamics_valid.to(device=alpha.device, dtype=alpha.dtype)
            if dynamics_valid.dim() == 1:
                dynamics_valid = dynamics_valid.unsqueeze(1)
            alpha = alpha * dynamics_valid
        return alpha


class ZeroInitPhysicalTimeAdapter(nn.Module):
    """Zero-output adapter that injects physical time without replacing order time."""

    def __init__(self, feature_dim=256, dynamics_dim=128, hidden_dim=128,
                 time_scale=0.5):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.dynamics_dim = int(dynamics_dim)
        self.time_scale = float(time_scale)
        if self.time_scale <= 0:
            raise ValueError("physical-time adapter time_scale must be positive")
        self.net = nn.Sequential(
            nn.Linear(self.feature_dim + self.dynamics_dim + 2, int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), self.feature_dim),
        )
        # This is an exact structural no-op at initialization.  Unlike a
        # sigmoid with a large negative bias, every correction element is zero.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, point_feature, z_dyn, current_delta_t, dynamics_valid,
                enabled_scale=1.0):
        batch_size = point_feature.shape[0]
        enabled_scale = float(enabled_scale)
        if not 0.0 <= enabled_scale <= 1.0:
            raise ValueError("physical-time adapter enabled_scale must be in [0, 1]")
        if enabled_scale == 0.0:
            correction = torch.zeros_like(point_feature)
            return point_feature, {
                "physical_time_adapter_correction": correction,
                "physical_time_adapter_norm": correction.new_zeros((batch_size,)),
                "physical_time_adapter_scale": correction.new_tensor(0.0),
            }

        if not torch.is_tensor(current_delta_t):
            current_delta_t = torch.as_tensor(
                current_delta_t, device=point_feature.device, dtype=point_feature.dtype)
        current_delta_t = current_delta_t.to(
            device=point_feature.device, dtype=point_feature.dtype)
        if current_delta_t.numel() == 1:
            current_delta_t = current_delta_t.repeat(batch_size)
        current_delta_t = current_delta_t.reshape(batch_size, 1)
        z_dyn = z_dyn.to(device=point_feature.device, dtype=point_feature.dtype)
        tau = torch.sign(current_delta_t) * torch.log1p(
            torch.abs(current_delta_t) / self.time_scale)
        time_features = torch.cat((tau, current_delta_t / self.time_scale), dim=1)
        adapter_input = torch.cat((point_feature, z_dyn, time_features), dim=1)
        correction = self.net(adapter_input)
        if dynamics_valid is not None:
            valid = dynamics_valid.to(
                device=point_feature.device, dtype=point_feature.dtype).reshape(batch_size, 1)
            correction = correction * (valid > 0).to(point_feature.dtype)
        correction = correction * enabled_scale
        adapted = point_feature + correction
        return adapted, {
            "physical_time_adapter_correction": correction,
            "physical_time_adapter_norm": torch.linalg.norm(correction, dim=1),
            "physical_time_adapter_scale": correction.new_tensor(enabled_scale),
        }


class DynamicsEncoder(nn.Module):
    def __init__(self, hidden_dim=128, eps=1e-3, use_query_gap=True):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.eps = float(eps)
        self.use_query_gap = bool(use_query_gap)

        dyn_dim = 10
        step_dim = 64
        query_dim = 2 if self.use_query_gap else 0
        self.per_step_mlp = nn.Sequential(
            nn.Linear(dyn_dim, step_dim),
            nn.ReLU(inplace=True),
            nn.Linear(step_dim, step_dim),
            nn.ReLU(inplace=True),
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(step_dim * 2 + query_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.velocity_head = nn.Linear(self.hidden_dim, 3)

    def _pad_delta_t(self, delta_t, hist_num):
        if delta_t.dim() == 1:
            delta_t = delta_t.unsqueeze(0)

        if delta_t.shape[1] >= hist_num:
            return delta_t[:, :hist_num]

        pad_count = hist_num - delta_t.shape[1]
        if delta_t.shape[1] > 0:
            pad_value = delta_t[:, -1:]
        else:
            pad_value = delta_t.new_full((delta_t.shape[0], 1), self.eps)
        return torch.cat((delta_t, pad_value.expand(-1, pad_count)), dim=1)

    def _format_current_delta_t(self, current_delta_t, delta_t, batch_size):
        if current_delta_t is None:
            current_delta_t = delta_t[:, 0]
        elif not torch.is_tensor(current_delta_t):
            current_delta_t = torch.as_tensor(
                current_delta_t, device=delta_t.device, dtype=delta_t.dtype)
        else:
            current_delta_t = current_delta_t.to(device=delta_t.device, dtype=delta_t.dtype)

        if current_delta_t.dim() == 0:
            current_delta_t = current_delta_t.repeat(batch_size)
        return current_delta_t.reshape(batch_size)

    def forward(self, ref_boxs, delta_t, valid_mask, current_delta_t=None):
        """
        Args:
            ref_boxs: B,H,4 boxes ordered from recent history to older history.
            delta_t: B,H positive gaps. delta_t[:, 1] is the gap between
                ref_boxs[:, 0] and ref_boxs[:, 1].
            valid_mask: B,H history validity mask.
            current_delta_t: optional B current query gap from ref_boxs[:, 0]
                to the current frame. When omitted, delta_t[:, 0] is used.

        Returns:
            z_dyn: B,hidden_dim
            velocity_pred: B,3
            displacement_pred: B,3, velocity_pred scaled by current_delta_t
            has_transition: B,1
        """
        B, H, _ = ref_boxs.shape
        if H < 2:
            z_dyn = ref_boxs.new_zeros((B, self.hidden_dim))
            velocity_pred = ref_boxs.new_zeros((B, 3))
            displacement_pred = ref_boxs.new_zeros((B, 3))
            has_transition = ref_boxs.new_zeros((B, 1))
            return z_dyn, velocity_pred, displacement_pred, has_transition

        delta_t = self._pad_delta_t(delta_t.to(device=ref_boxs.device, dtype=ref_boxs.dtype), H)
        valid_mask = valid_mask.to(device=ref_boxs.device, dtype=ref_boxs.dtype)
        current_gap = torch.clamp(
            self._format_current_delta_t(current_delta_t, delta_t, B),
            min=self.eps,
        )

        newer = ref_boxs[:, :-1, :]
        older = ref_boxs[:, 1:, :]
        gap = torch.clamp(delta_t[:, 1:H], min=self.eps)

        displacement = newer[:, :, :3] - older[:, :, :3]
        velocity = displacement / gap.unsqueeze(-1)
        angle_delta = wrap_angle(newer[:, :, 3] - older[:, :, 3])
        angular_velocity = angle_delta / gap
        speed = torch.linalg.norm(velocity, dim=-1)
        transition_mask = valid_mask[:, :-1] * valid_mask[:, 1:]

        dyn_features = torch.cat(
            (
                displacement,
                velocity,
                angular_velocity.unsqueeze(-1),
                speed.unsqueeze(-1),
                gap.unsqueeze(-1),
                transition_mask.unsqueeze(-1),
            ),
            dim=-1,
        )

        step_features = self.per_step_mlp(dyn_features)
        masked_step_features = step_features * transition_mask.unsqueeze(-1)

        valid_count = transition_mask.sum(dim=1, keepdim=True)
        has_transition = (valid_count > 0).to(ref_boxs.dtype)
        mean_features = masked_step_features.sum(dim=1) / valid_count.clamp_min(1.0)

        neg_inf = torch.finfo(step_features.dtype).min
        max_features = step_features.masked_fill(transition_mask.unsqueeze(-1) <= 0, neg_inf).max(dim=1).values
        max_features = torch.where(has_transition > 0, max_features, torch.zeros_like(max_features))

        pooled_features = torch.cat((mean_features, max_features), dim=-1)
        if self.use_query_gap:
            recent_gap = torch.clamp(delta_t[:, 0], min=self.eps)
            query_features = torch.stack((current_gap, current_gap / recent_gap), dim=1)
            query_features = torch.nan_to_num(query_features, nan=0.0, posinf=0.0, neginf=0.0)
            pooled_features = torch.cat((pooled_features, query_features), dim=-1)

        z_dyn = self.global_mlp(pooled_features) * has_transition
        velocity_pred = self.velocity_head(z_dyn) * has_transition
        displacement_pred = velocity_pred * current_gap.unsqueeze(-1) * has_transition
        return z_dyn, velocity_pred, displacement_pred, has_transition
