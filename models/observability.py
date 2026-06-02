import torch
from torch import nn


class ObservabilityGate(nn.Module):
    def __init__(
        self,
        feature_dim=256,
        dynamics_dim=128,
        stats_dim=5,
        hidden_dim=64,
        init_obs_bias=1.0,
        min_dyn_valid=0.5,
        eps=1e-6,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.dynamics_dim = int(dynamics_dim)
        self.stats_dim = int(stats_dim)
        self.hidden_dim = int(hidden_dim)
        self.min_dyn_valid = float(min_dyn_valid)
        self.eps = float(eps)

        self.dyn_proj = nn.Linear(self.dynamics_dim, self.feature_dim)
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.stats_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, 2),
        )
        self._init_gate(float(init_obs_bias))

    def _init_gate(self, init_obs_bias):
        final = self.gate_mlp[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        final.bias.data[0] = init_obs_bias

    def compute_alpha(self, obs_stats, dynamics_valid, dtype=None, device=None):
        if device is None:
            device = obs_stats.device
        if dtype is None:
            dtype = obs_stats.dtype

        obs_stats = obs_stats.to(device=device, dtype=dtype)
        dynamics_valid = dynamics_valid.to(device=device, dtype=dtype)
        logits = self.gate_mlp(obs_stats)
        alpha = torch.softmax(logits, dim=1)

        dyn_valid = (dynamics_valid.view(-1, 1) >= self.min_dyn_valid).to(dtype)
        alpha_dyn = alpha[:, 1:2] * dyn_valid
        alpha_obs = alpha[:, 0:1]
        alpha_sum = (alpha_obs + alpha_dyn).clamp_min(self.eps)
        alpha = torch.cat((alpha_obs / alpha_sum, alpha_dyn / alpha_sum), dim=1)
        entropy = -(alpha * torch.log(alpha.clamp_min(self.eps))).sum(dim=1)

        aux = {
            "obs_alpha": alpha,
            "obs_alpha_obs": alpha[:, 0],
            "obs_alpha_dyn": alpha[:, 1],
            "obs_gate_entropy": entropy,
        }
        return alpha, aux

    def forward(self, point_feature, z_dyn, obs_stats, dynamics_valid):
        z_dyn = z_dyn.to(device=point_feature.device, dtype=point_feature.dtype)
        alpha, aux = self.compute_alpha(
            obs_stats,
            dynamics_valid,
            dtype=point_feature.dtype,
            device=point_feature.device,
        )

        z_dyn_proj = self.dyn_proj(z_dyn)
        fused_feature = alpha[:, 0:1] * point_feature + alpha[:, 1:2] * z_dyn_proj
        return fused_feature, aux
