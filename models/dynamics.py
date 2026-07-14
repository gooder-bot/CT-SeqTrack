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
