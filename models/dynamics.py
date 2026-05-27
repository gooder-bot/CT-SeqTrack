import torch
from torch import nn


def wrap_angle(angle):
    return torch.atan2(torch.sin(angle), torch.cos(angle))


class DynamicsEncoder(nn.Module):
    def __init__(self, hidden_dim=128, eps=1e-3):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.eps = float(eps)

        dyn_dim = 10
        step_dim = 64
        self.per_step_mlp = nn.Sequential(
            nn.Linear(dyn_dim, step_dim),
            nn.ReLU(inplace=True),
            nn.Linear(step_dim, step_dim),
            nn.ReLU(inplace=True),
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(step_dim * 2, self.hidden_dim),
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

    def forward(self, ref_boxs, delta_t, valid_mask):
        """
        Args:
            ref_boxs: B,H,4 boxes ordered from recent history to older history.
            delta_t: B,H positive gaps. delta_t[:, 1] is the gap between
                ref_boxs[:, 0] and ref_boxs[:, 1].
            valid_mask: B,H history validity mask.

        Returns:
            z_dyn: B,hidden_dim
            velocity_pred: B,3
            has_transition: B,1
        """
        B, H, _ = ref_boxs.shape
        if H < 2:
            z_dyn = ref_boxs.new_zeros((B, self.hidden_dim))
            velocity_pred = ref_boxs.new_zeros((B, 3))
            has_transition = ref_boxs.new_zeros((B, 1))
            return z_dyn, velocity_pred, has_transition

        delta_t = self._pad_delta_t(delta_t.to(device=ref_boxs.device, dtype=ref_boxs.dtype), H)
        valid_mask = valid_mask.to(device=ref_boxs.device, dtype=ref_boxs.dtype)

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
        z_dyn = self.global_mlp(pooled_features) * has_transition
        velocity_pred = self.velocity_head(z_dyn) * has_transition
        return z_dyn, velocity_pred, has_transition
