"""v30 共享六动作效用头；B0/B1/B2 的全部输入均停止梯度。"""
from __future__ import annotations

import math
import torch
from torch import nn

from models.ct_v2.action_v27 import bounded_residual_xy
from models.ct_v2.pipeline_contracts import EvidenceModeSet
from utils.v30_policy import choose_mode_action, normalize_mode_policy


def mode_action_geometry(observation_box, evidence_modes, query_delta_t,
                         radius_base=.5, radius_per_second=.5, radius_max=2.):
    """固定顺序 m0半/m0全/m1半/m1全/m2半/m2全，同状态生成真实框。"""
    observation = observation_box.detach()
    modes = evidence_modes.detached()
    batch, width = observation.shape
    if modes.centers_xy.shape != (batch, 3, 2) or modes.valid.shape != (batch, 3):
        raise ValueError('v30 action geometry requires three aligned modes')
    raw = observation[:, None].expand(-1, 3, -1).clone()
    raw[..., :2] = modes.centers_xy
    bounded, geometry = bounded_residual_xy(
        observation[:, None].expand_as(raw).reshape(-1, width), raw.reshape(-1, width),
        query_delta_t.detach().reshape(batch, 1).expand(-1, 3).reshape(-1),
        radius_base, radius_per_second, radius_max)
    amplitude = observation.new_tensor([.5, 1.])
    residual = (bounded.reshape(batch, 3, 1, 2) * amplitude[None, None, :, None]).reshape(batch, 6, 2)
    actions = observation[:, None].expand(-1, 6, -1).clone()
    actions[..., :2] += residual
    valid = (modes.valid & geometry['finite'].reshape(batch, 3)).repeat_interleave(2, 1)
    valid &= torch.linalg.norm(residual, dim=-1) > 1e-6
    return dict(action_boxes=actions, action_valid=valid, residual=residual,
                radius=geometry['radius'].reshape(batch, 3)[:, 0],
                raw_residual=geometry['residual'].reshape(batch, 3, 2).repeat_interleave(2, 1),
                raw_norm=geometry['norm'].reshape(batch, 3).repeat_interleave(2, 1),
                scale=geometry['scale'].reshape(batch, 3).repeat_interleave(2, 1),
                amplitude=amplitude.repeat(3)[None].expand(batch, -1))


class B3ModeUtilityUpdater(nn.Module):
    def __init__(self, observation_stats_dim=5, hidden_dim=64,
                 decision_threshold=0., radius_base=.5, radius_per_second=.5,
                 radius_max=2., require_calibration=False,
                 helpful_init_probability=.05, harmful_init_probability=.5, **unused):
        super().__init__()
        if radius_base <= 0 or radius_per_second < 0 or radius_max <= 0:
            raise ValueError('invalid action radius')
        self.radius_base, self.radius_per_second, self.radius_max = radius_base, radius_per_second, radius_max
        self.require_calibration = bool(require_calibration)
        self.observation_stats_dim = int(observation_stats_dim)
        self.register_buffer('decision_threshold', torch.tensor(float(decision_threshold)), persistent=False)
        self.register_buffer('calibrated', torch.tensor(not require_calibration), persistent=False)
        self.action_policy = {'kind': 'threshold', 'threshold': float(decision_threshold)}
        self.evidence_projection = nn.Sequential(nn.Linear(128, 32), nn.GELU())
        # context(12+stats), global point(6), mode quality/covariance(9), raw(3), action(5)
        self.risk_trunk = nn.Sequential(nn.Linear(67 + self.observation_stats_dim, hidden_dim), nn.GELU())
        self.helpful_head, self.harmful_head = nn.Linear(hidden_dim, 1), nn.Linear(hidden_dim, 1)
        self.expected_success_gain_head = nn.Linear(hidden_dim, 1)
        self.expected_precision_gain_head = nn.Linear(hidden_dim, 1)
        self.expected_distance_gain_head = nn.Linear(hidden_dim, 1)
        self.expected_continuous_iou_gain_head = nn.Linear(hidden_dim, 1)
        for head, probability in ((self.helpful_head, helpful_init_probability),
                                  (self.harmful_head, harmful_init_probability)):
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, math.log(probability / (1 - probability)))
        for head in (self.expected_success_gain_head, self.expected_precision_gain_head,
                     self.expected_distance_gain_head, self.expected_continuous_iou_gain_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    @torch.no_grad()
    def install_policy(self, policy):
        self.action_policy = normalize_mode_policy(policy)
        self.decision_threshold.fill_(self.action_policy.get('threshold', 0.))
        self.calibrated.fill_(True)

    def install_calibration(self, presence_threshold=None, action_threshold=None, action_policy=None):
        self.install_policy(action_policy if action_policy is not None else {
            'kind': 'threshold', 'threshold': float(action_threshold if action_threshold is not None else presence_threshold)})

    def forward(self, observation_box, evidence_modes, base_evidence,
                base_presence_probability, extension_presence_probability,
                observation_stats, b1_sigma_parallel_perp, query_delta_t,
                gap_ratio, recursive_age=None, enabled=True, coarse_box=None,
                b1_center_xy=None, targetness_entropy=None, normalized_ess=None,
                extension_point_count=None, extension_voxel_count=None,
                targetness_mean=None, targetness_max=None, **unused):
        if not isinstance(evidence_modes, EvidenceModeSet):
            raise TypeError('v30 B3 requires the complete EvidenceModeSet')
        observation, modes = observation_box.detach(), evidence_modes.detached()
        batch = len(observation)
        geometry = mode_action_geometry(observation, modes, query_delta_t,
            self.radius_base, self.radius_per_second, self.radius_max)
        def column(value, default=0.):
            return (observation.new_full((batch, 1), default) if value is None
                    else value.detach().reshape(batch, 1).to(observation))
        def expand(value):
            return value[:, None].expand(-1, 6, -1)
        coarse = observation if coarse_box is None else coarse_box.detach()
        prior = observation[:, :2] if b1_center_xy is None else b1_center_xy.detach()
        prior_delta = prior - observation[:, :2]
        yaw = coarse[:, 3] - observation[:, 3]
        evidence_input = torch.cat((base_evidence.detach()[:, None].expand(-1, 6, -1),
                                    modes.features.repeat_interleave(2, 1)), -1)
        evidence_finite = torch.isfinite(evidence_input).all(-1)
        evidence = self.evidence_projection(torch.nan_to_num(evidence_input, nan=0., posinf=0., neginf=0.))
        context = torch.cat((column(base_presence_probability), column(extension_presence_probability),
            torch.linalg.norm(coarse[:, :2] - observation[:, :2], dim=-1)[:, None],
            torch.atan2(yaw.sin(), yaw.cos()).abs()[:, None], prior_delta,
            torch.linalg.norm(prior_delta, dim=-1)[:, None],
            b1_sigma_parallel_perp.detach().clamp_min(1e-6).log(),
            torch.log1p(column(query_delta_t).clamp_min(0)),
            torch.log1p(column(gap_ratio).clamp_min(0)),
            torch.log1p(column(recursive_age).clamp_min(0)), observation_stats.detach()), -1)
        global_point = torch.cat((column(targetness_entropy), column(normalized_ess),
            torch.log1p(column(extension_point_count).clamp_min(0)),
            torch.log1p(column(extension_voxel_count).clamp_min(0)),
            column(targetness_mean), column(targetness_max)), -1)
        covariance = modes.covariance_xy
        total = modes.member_mask.any(1).sum(1).clamp_min(1)
        mode_stats = torch.stack((torch.log1p(modes.mass), torch.log1p(modes.unique_count.to(observation)),
            modes.targetness_mean, modes.identity / 2, modes.quality,
            torch.log1p(covariance[..., 0, 0].clamp_min(0)),
            covariance[..., 0, 1].sign() * torch.log1p(covariance[..., 0, 1].abs()),
            torch.log1p(covariance[..., 1, 1].clamp_min(0)),
            modes.unique_count.to(observation) / total[:, None]), -1).repeat_interleave(2, 1)
        raw = torch.cat((geometry['raw_residual'], geometry['raw_norm'][..., None]), -1)
        radius = geometry['radius'].clamp_min(1e-6)
        action_features = torch.cat((geometry['residual'] / radius[:, None, None],
            (geometry['raw_norm'] / radius[:, None])[..., None],
            (1 - geometry['scale'])[..., None], geometry['amplitude'][..., None]), -1)
        features = torch.cat((evidence, expand(context), expand(global_point), mode_stats, raw, action_features), -1)
        valid = geometry['action_valid'] & torch.isfinite(features).all(-1) & evidence_finite
        hidden = self.risk_trunk(torch.nan_to_num(features, nan=0., posinf=0., neginf=0.))
        help_logit, harm_logit = self.helpful_head(hidden).squeeze(-1), self.harmful_head(hidden).squeeze(-1)
        success = self.expected_success_gain_head(hidden).squeeze(-1).tanh()
        precision = self.expected_precision_gain_head(hidden).squeeze(-1).tanh()
        distance = self.expected_distance_gain_head(hidden).squeeze(-1).tanh()
        iou = self.expected_continuous_iou_gain_head(hidden).squeeze(-1).tanh()
        scores = .5 * (success + precision)
        deployable = bool(enabled) and (bool(self.calibrated) or not self.require_calibration)
        selected = choose_mode_action(observation, geometry['action_boxes'], valid, scores,
            self.action_policy if deployable else {'kind': 'never'}, evidence_top_index=modes.evidence_top_index)
        # 标量兼容接口明确对应实际选中动作；回退时对应 best-q 同状态候选。
        index = torch.where(selected['applied'], selected['chosen_action_index'], selected['best_action_index']).clamp_min(0)
        rows = torch.arange(batch, device=observation.device)
        pick = lambda value: value[rows, index]
        score = pick(scores)
        output = {
            'ct_b3_action_boxes': geometry['action_boxes'], 'ct_b3_action_valid': valid,
            'ct_b3_action_scores': scores, 'ct_b3_action_success_gain': success,
            'ct_b3_action_precision_gain': precision, 'ct_b3_action_distance_gain': distance,
            'ct_b3_action_iou_gain': iou, 'ct_b3_action_help_logit': help_logit,
            'ct_b3_action_harm_logit': harm_logit, 'ct_b3_action_residual_xy': geometry['residual'],
            'ct_b3_chosen_action_index': selected['chosen_action_index'],
            'ct_b3_best_action_index': selected['best_action_index'],
            'ct_b3_diagnostic_action_index': index,
            'ct_b3_max_action_score': selected['max_action_score'],
            'ct_b3_help_logit': pick(help_logit), 'ct_b3_harm_logit': pick(harm_logit),
            'ct_b3_help_probability': pick(help_logit).sigmoid(), 'ct_b3_harm_probability': pick(harm_logit).sigmoid(),
            'ct_b3_expected_success_gain': pick(success), 'ct_b3_expected_precision_gain': pick(precision),
            'ct_b3_expected_center_gain': pick(precision), 'ct_b3_expected_iou_gain': pick(success),
            'ct_b3_action_score': score, 'ct_b3_calibrated': score.new_full((batch,), float(bool(self.calibrated))),
            'ct_b3_h3_residual': score, 'ct_b3_h3_utility': score,
            'ct_b3_final_gate': selected['applied'].to(score), 'ct_router_logit': score,
            'ct_router_gate': score, 'ct_router_applied_gate': selected['applied'].to(score),
            'ct_router_evidence_valid': valid.any(1).to(score),
            'ct_router_bounded_residual_xy': pick(geometry['residual']),
            'ct_router_residual_xy': pick(geometry['raw_residual']), 'ct_router_radius': geometry['radius'],
            'ct_router_clip_rate': (pick(geometry['raw_norm']) > geometry['radius']).to(score),
            'ct_router_soft_box': selected['final_box'],
            'ct_b3_mode_summary': pick(mode_stats)[..., :4],
            'ct_b3_bounded_action_features': pick(action_features),
            'ct_b3_action_raw_residual_xy': geometry['raw_residual'],
            'ct_b3_action_raw_norm': geometry['raw_norm'],
            'ct_b3_all_action_features': action_features,
            'ct_b3_all_mode_summary': mode_stats[..., :4],
        }
        from utils.v30_action_output import apply_selection_output
        apply_selection_output(output, observation, selected)
        return selected['final_box'], output
