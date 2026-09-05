"""v27 有界动作效用：连续证据特征、单个 q > tau 决策。"""
from __future__ import annotations

import math
import torch
from torch import nn

from utils.action_calibration_v27 import normalize_policy


def bounded_residual_xy(observation_box, raw_box, query_delta_t,
                        radius_base=.5, radius_per_second=.5, radius_max=2.):
    """B2 bounded-always、B3 和监督共用同一个实际动作。"""
    observation = observation_box.detach()
    raw = raw_box.detach()
    dt = query_delta_t.detach().reshape(-1)
    finite = torch.isfinite(raw).all(1) & torch.isfinite(observation).all(1)
    finite = finite & torch.isfinite(dt) & (dt >= 0)
    dt = torch.nan_to_num(dt, nan=0., posinf=0., neginf=0.).clamp_min(0)
    radius = (radius_base + radius_per_second * dt).clamp(max=radius_max)
    residual = torch.nan_to_num(raw[:, :2] - observation[:, :2],
                                nan=0., posinf=0., neginf=0.)
    norm = torch.linalg.norm(residual, dim=1)
    scale = (radius / norm.clamp_min(1e-6)).clamp(max=1.)
    bounded = residual * scale[:, None]
    bounded = torch.where(finite[:, None], bounded, torch.zeros_like(bounded))
    return bounded, {"residual": residual, "radius": radius, "scale": scale,
                     "norm": norm, "finite": finite}


class B3UtilityUpdater(nn.Module):
    def __init__(self, observation_stats_dim=5, hidden_dim=64,
                 presence_threshold=.5, decision_threshold=0.,
                 radius_base=.5, radius_per_second=.5, radius_max=2.,
                 require_calibration=False, consensus_features=True,
                 helpful_init_probability=.05, harmful_init_probability=.5,
                 mode_summary_dim=4):
        super().__init__()
        if radius_base <= 0 or radius_per_second < 0 or radius_max <= 0:
            raise ValueError("invalid action radius")
        self.radius_base, self.radius_per_second, self.radius_max = (
            float(radius_base), float(radius_per_second), float(radius_max))
        self.require_calibration = bool(require_calibration)
        self.consensus_features = bool(consensus_features)
        self.mode_summary_dim = int(mode_summary_dim)
        self.observation_stats_dim = int(observation_stats_dim)
        self.register_buffer("decision_threshold", torch.tensor(float(decision_threshold)), persistent=False)
        self.register_buffer("calibrated", torch.tensor(not require_calibration), persistent=False)
        self.action_policy = {"kind": "threshold", "threshold": float(decision_threshold)}
        self.evidence_projection = nn.Sequential(nn.Linear(128, 32), nn.GELU())
        point_dim = 6 + (7 if consensus_features else 0)
        input_dim = 32 + 2 + 2 + 3 + 3 + 2 + 3 + self.observation_stats_dim + point_dim + 4 + self.mode_summary_dim
        self.risk_trunk = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU())
        self.helpful_head, self.harmful_head = nn.Linear(hidden_dim, 1), nn.Linear(hidden_dim, 1)
        self.expected_success_gain_head = nn.Linear(hidden_dim, 1)
        self.expected_precision_gain_head = nn.Linear(hidden_dim, 1)
        for head, probability in ((self.helpful_head, helpful_init_probability),
                                  (self.harmful_head, harmful_init_probability)):
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, math.log(probability / (1 - probability)))
        for head in (self.expected_success_gain_head, self.expected_precision_gain_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    @torch.no_grad()
    def install_policy(self, policy):
        self.action_policy = normalize_policy(policy)
        self.decision_threshold.fill_(self.action_policy.get("threshold", 0.))
        self.calibrated.fill_(True)

    def install_calibration(self, presence_threshold=None, action_threshold=None,
                            action_policy=None):
        if action_policy is not None:
            self.install_policy(action_policy)
        else:
            threshold = action_threshold if action_threshold is not None else presence_threshold
            self.install_policy({"kind": "threshold", "threshold": float(threshold)})

    def forward(self, observation_box, raw_box, availability, base_evidence,
                extension_evidence, base_presence_probability,
                extension_presence_probability, observation_stats,
                b1_sigma_parallel_perp, query_delta_t, gap_ratio,
                recursive_age=None, enabled=True, coarse_box=None,
                b1_center_xy=None, targetness_entropy=None, normalized_ess=None,
                extension_point_count=None, extension_voxel_count=None,
                targetness_mean=None, targetness_max=None, vote_consistency=None,
                vote_covariance_xx=None, vote_covariance_xy=None,
                vote_covariance_yy=None, vote_inlier_ratio=None,
                vote_candidate_margin=None, compatible_hypothesis_count=None,
                h1_utility_logit=None, h1_expected_gain=None, mode_summary=None,
                **unused):
        observation, raw = observation_box.detach(), raw_box.detach()
        batch = len(observation)
        bounded, geometry = bounded_residual_xy(observation, raw, query_delta_t,
            self.radius_base, self.radius_per_second, self.radius_max)
        def column(value, default=0.):
            if value is None:
                return observation.new_full((batch, 1), default)
            return value.detach().reshape(batch, 1).to(observation)
        def disagreement(first, second):
            delta = second[:, :2] - first[:, :2]
            return torch.cat((delta, torch.linalg.norm(delta, dim=1, keepdim=True)), 1)
        coarse = observation if coarse_box is None else coarse_box.detach()
        prior = observation[:, :2] if b1_center_xy is None else b1_center_xy.detach()
        yaw_delta = coarse[:, 3] - observation[:, 3]
        evidence = self.evidence_projection(torch.cat((base_evidence.detach(), extension_evidence.detach()), 1))
        point = [column(targetness_entropy), column(normalized_ess),
                 torch.log1p(column(extension_point_count).clamp_min(0)),
                 torch.log1p(column(extension_voxel_count).clamp_min(0)),
                 column(targetness_mean), column(targetness_max)]
        if self.consensus_features:
            xy = column(vote_covariance_xy)
            point.extend([column(vote_consistency),
                          torch.log1p(column(vote_covariance_xx).clamp_min(0)),
                          xy.sign() * torch.log1p(xy.abs()),
                          torch.log1p(column(vote_covariance_yy).clamp_min(0)),
                          column(vote_inlier_ratio), column(vote_candidate_margin),
                          column(compatible_hypothesis_count).clamp(0, 3) / 3])
        if mode_summary is None:
            mode_summary = observation.new_zeros((batch, self.mode_summary_dim))
        mode_summary = mode_summary.detach().reshape(batch, self.mode_summary_dim).to(observation)
        radius = geometry["radius"].clamp_min(1e-6)
        # Four explicit executed-action features: normalized dx/dy, raw/radius, clipping fraction.
        action_features = torch.cat((bounded / radius[:, None],
            (geometry["norm"] / radius)[:, None], (1 - geometry["scale"])[:, None]), 1)
        features = torch.cat([
            evidence, column(base_presence_probability), column(extension_presence_probability),
            torch.linalg.norm(coarse[:, :2] - observation[:, :2], dim=1)[:, None],
            torch.atan2(yaw_delta.sin(), yaw_delta.cos()).abs()[:, None],
            disagreement(observation, prior), disagreement(observation, raw),
            b1_sigma_parallel_perp.detach().clamp_min(.1).log(),
            torch.log1p(column(query_delta_t).clamp_min(0)),
            torch.log1p(column(gap_ratio).clamp_min(0)),
            torch.log1p(column(recursive_age).clamp_min(0)),
            observation_stats.detach(), *point, action_features, mode_summary], 1)
        features_finite = torch.isfinite(features).all(1)
        hidden = self.risk_trunk(torch.nan_to_num(features, nan=0., posinf=0., neginf=0.))
        help_logit, harm_logit = self.helpful_head(hidden).squeeze(1), self.harmful_head(hidden).squeeze(1)
        success_gain = self.expected_success_gain_head(hidden).squeeze(1).tanh()
        precision_gain = self.expected_precision_gain_head(hidden).squeeze(1).tanh()
        score = (success_gain + precision_gain) / 2
        valid = (availability.detach().reshape(-1) > 0) & geometry["finite"] & features_finite & torch.isfinite(score)
        kind = self.action_policy["kind"]
        if kind == "always":
            decision = torch.ones_like(valid)
        elif kind == "never":
            decision = torch.zeros_like(valid)
        else:
            decision = score > self.decision_threshold.to(score)
        deployable = bool(enabled) and (bool(self.calibrated) or not self.require_calibration)
        applied = valid & decision & deployable
        final = torch.cat((torch.where(applied[:, None], observation[:, :2] + bounded,
                                     observation[:, :2]), observation[:, 2:]), 1)
        return final, {
            "ct_b3_help_logit": help_logit, "ct_b3_harm_logit": harm_logit,
            "ct_b3_help_probability": help_logit.sigmoid(), "ct_b3_harm_probability": harm_logit.sigmoid(),
            "ct_b3_expected_success_gain": success_gain, "ct_b3_expected_precision_gain": precision_gain,
            # Old structural contracts use these names; v27 loss must use the explicit names above.
            "ct_b3_expected_center_gain": precision_gain, "ct_b3_expected_iou_gain": success_gain,
            "ct_b3_action_score": score, "ct_b3_calibrated": score.new_full((batch,), float(bool(self.calibrated))),
            "ct_b3_h3_residual": score, "ct_b3_h3_utility": score,
            "ct_b3_final_gate": applied.to(score), "ct_router_logit": score,
            "ct_router_gate": score, "ct_router_applied_gate": applied.to(score),
            "ct_router_evidence_valid": valid.to(score), "ct_router_bounded_residual_xy": bounded,
            "ct_router_residual_xy": geometry["residual"], "ct_router_radius": geometry["radius"],
            "ct_router_clip_rate": (geometry["norm"] > geometry["radius"]).to(score),
            "ct_router_soft_box": final, "ct_b3_mode_summary": mode_summary,
            "ct_b3_bounded_action_features": action_features,
        }
