"""Compact jointly-trained B1/B2/B3 coupling for CT-SeqTrack.

The module deliberately keeps the stable SeqTrack3D observation path outside
the search branch.  Motion contributes a normalized residual score, while the
point cloud remains responsible for the actual raw Search proposal.
"""

import math

import numpy as np
import torch
from torch import nn


def _batch_scalar(value, reference):
    value = value.to(device=reference.device, dtype=reference.dtype)
    if value.numel() == 1:
        value = value.reshape(1).expand(reference.shape[0])
    return value.reshape(reference.shape[0])


def counterfactual_query_targets(
        raw_obs_xy, raw_motion_xy, target_xy, margin=0.05):
    """Label whether the alpha=1 arm beats alpha=0 outside a dead zone."""
    margin = float(margin)
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError("counterfactual query margin must be non-negative")
    raw_obs_xy = raw_obs_xy.detach()
    raw_motion_xy = raw_motion_xy.detach()
    target_xy = target_xy.detach()
    obs_error = torch.linalg.norm(raw_obs_xy - target_xy, dim=1)
    motion_error = torch.linalg.norm(raw_motion_xy - target_xy, dim=1)
    helpful = motion_error + margin < obs_error
    harmful = obs_error + margin < motion_error
    ambiguous = ~(helpful | harmful)
    return {
        "obs_error": obs_error,
        "motion_error": motion_error,
        "helpful": helpful,
        "harmful": harmful,
        "ambiguous": ambiguous,
        "target": helpful.to(target_xy.dtype),
        "valid": (helpful | harmful).to(target_xy.dtype),
        "uplift": obs_error - motion_error,
    }


def calibrate_joint_router_threshold(
        probabilities,
        h3_gain,
        valid,
        minimum_threshold=0.5,
        min_precision=0.75,
        max_harm_rate=0.05,
        min_coverage=0.05,
        max_coverage=0.25,
        helpful_margin=0.15):
    """Choose a conservative scalar threshold on held-out tracklets only."""
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    gains = np.asarray(h3_gain, dtype=np.float64).reshape(-1)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    if not (probabilities.shape == gains.shape == valid.shape):
        raise ValueError("router calibration arrays must have equal length")
    finite = valid & np.isfinite(probabilities) & np.isfinite(gains)
    if not finite.any():
        raise RuntimeError("router calibration contains no valid H=3 rows")
    thresholds = np.unique(np.concatenate((
        np.asarray([float(minimum_threshold)], dtype=np.float64),
        probabilities[finite & (probabilities >= minimum_threshold)],
    )))
    candidates = []
    denominator = max(1, len(probabilities))
    for threshold in thresholds:
        chosen = finite & (probabilities >= threshold)
        count = int(chosen.sum())
        coverage = count / float(denominator)
        if count == 0:
            continue
        precision = float(np.mean(gains[chosen] > float(helpful_margin)))
        harm_rate = float(np.mean(
            gains[chosen] < -float(helpful_margin)))
        if (min_coverage <= coverage <= max_coverage
                and precision >= min_precision
                and harm_rate <= max_harm_rate):
            candidates.append((
                coverage, precision, -harm_rate, float(threshold), count))
    if not candidates:
        raise RuntimeError(
            "no router threshold satisfies precision/harm/coverage guardrails")
    coverage, precision, negative_harm, threshold, count = max(candidates)
    return {
        "threshold": threshold,
        "coverage": coverage,
        "helpful_precision": precision,
        "harm_rate": -negative_harm,
        "intervention_count": count,
    }


class JointFullSearchRefiner(nn.Module):
    """Dual-reference soft geometry with reliability-gated query residual."""

    def __init__(
            self,
            point_dim=5,
            feature_dim=128,
            query_dim=64,
            motion_dim=128,
            observation_stats_dim=5,
            max_vote_offset=4.0,
            motion_dropout=0.1,
            gate_init_probability=0.05,
            query_gate_scale=1.0,
            presence_init_probability=0.1,
            presence_threshold=0.5,
            mahalanobis_clip=25.0,
            use_reliability_gate=True,
            contract_version=1,
            presence_hard_gate=True):
        super().__init__()
        self.point_dim = int(point_dim)
        self.feature_dim = int(feature_dim)
        self.query_dim = int(query_dim)
        self.motion_dim = int(motion_dim)
        self.observation_stats_dim = int(observation_stats_dim)
        self.max_vote_offset = float(max_vote_offset)
        self.motion_dropout = float(motion_dropout)
        self.mahalanobis_clip = float(mahalanobis_clip)
        self.query_gate_scale = float(query_gate_scale)
        self.presence_threshold = float(presence_threshold)
        self.use_reliability_gate = bool(use_reliability_gate)
        self.contract_version = int(contract_version)
        self.presence_hard_gate = bool(presence_hard_gate)
        if self.contract_version not in (1, 2):
            raise ValueError("joint search contract version must be 1 or 2")
        if min(self.point_dim, self.feature_dim, self.query_dim,
               self.motion_dim) <= 0:
            raise ValueError("joint Full feature dimensions must be positive")
        if not 0.0 <= self.motion_dropout < 1.0:
            raise ValueError("joint Full motion dropout must be in [0,1)")
        if not 0.0 < gate_init_probability < 1.0:
            raise ValueError("query gate initialization must be in (0,1)")
        if self.query_gate_scale < 0.0:
            raise ValueError("query gate scale must be non-negative")
        if not 0.0 < presence_init_probability < 1.0:
            raise ValueError("presence initialization must be in (0,1)")

        def feature_mlp(input_dim):
            return nn.Sequential(
                nn.Linear(input_dim, self.feature_dim),
                nn.LayerNorm(self.feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.feature_dim, self.feature_dim),
            )

        self.point_encoder = feature_mlp(self.point_dim)
        self.kinematic_geometry = feature_mlp(3)
        self.learned_geometry = feature_mlp(4)
        # residual unit, normalized sigma, log dt, gap ratio, history ratio
        self.motion_context = feature_mlp(7)
        self.support_source_embedding = nn.Embedding(2, self.feature_dim)
        # 0=baseline, 1=endpoint, 2=tube.  Baseline remains reserved even
        # though useful-support mode never substitutes the B0 crop into Search.
        self.branch_source_embedding = nn.Embedding(3, self.feature_dim)
        self.point_norm = nn.LayerNorm(self.feature_dim)

        self.query_observation_norm = nn.LayerNorm(self.query_dim)
        self.motion_query_projection = nn.Linear(
            self.motion_dim + 4, self.query_dim)
        self.motion_query_norm = nn.LayerNorm(self.query_dim)
        # q_search stays in q_obs space; there is deliberately no Search-only
        # normalization after adding the residual.
        # dt, gap, history, residual(2), sigma/M(2), observation stats
        gate_dim = (
            11 + self.observation_stats_dim
            if self.contract_version >= 2
            else 7 + self.observation_stats_dim)
        self.query_gate = nn.Sequential(
            nn.Linear(gate_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.query_gate[-1].weight)
        nn.init.constant_(
            self.query_gate[-1].bias,
            math.log(gate_init_probability / (1.0 - gate_init_probability)),
        )
        nn.init.zeros_(self.motion_query_projection.weight)
        nn.init.zeros_(self.motion_query_projection.bias)

        self.point_key = nn.Linear(self.feature_dim, self.query_dim)
        self.local_targetness = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.vote_head = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )
        if self.contract_version >= 2:
            self.presence_head = nn.Sequential(
                nn.Linear(2 * self.feature_dim + 3, self.feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.feature_dim, 1),
            )
            presence_output = self.presence_head[-1]
        else:
            self.presence_head = nn.Linear(self.feature_dim + 4, 1)
            presence_output = self.presence_head
        nn.init.zeros_(presence_output.weight)
        nn.init.constant_(
            presence_output.bias,
            math.log(presence_init_probability / (
                1.0 - presence_init_probability)),
        )

    def forward(
            self,
            encoded_points,
            point_xy,
            point_valid_mask,
            point_source,
            observation_query,
            observation_stats,
            motion_feature,
            kinematic_xy,
            learned_xy,
            residual_unit,
            sigma_parallel_perp,
            envelope_parallel_perp,
            direction_xy,
            motion_valid,
            query_delta_t,
            gap_ratio,
            history_valid_ratio,
            point_branch_source=None,
            search_support_valid=None):
        """Run the Search refiner only on the structurally useful sub-batch."""
        if encoded_points.dim() != 3:
            raise ValueError("joint Full points must have shape [B,N,point_dim]")
        batch_size, point_count, _ = encoded_points.shape
        if search_support_valid is None:
            support_mask = torch.ones(
                batch_size, dtype=torch.bool, device=encoded_points.device)
        else:
            support_mask = _batch_scalar(
                search_support_valid, encoded_points) > 0
        active_index = torch.nonzero(support_mask, as_tuple=False).flatten()
        if active_index.numel() == batch_size:
            return self._forward_valid(
                encoded_points, point_xy, point_valid_mask, point_source,
                observation_query, observation_stats, motion_feature,
                kinematic_xy, learned_xy, residual_unit,
                sigma_parallel_perp, envelope_parallel_perp, direction_xy,
                motion_valid, query_delta_t, gap_ratio, history_valid_ratio,
                point_branch_source=point_branch_source,
                search_support_valid=search_support_valid)

        reference = encoded_points
        q_observation = self.query_observation_norm(
            observation_query.detach())
        defaults = {
            "ct_query_observation": q_observation,
            "ct_query_residual": reference.new_zeros(
                (batch_size, self.query_dim)),
            "ct_query_search_internal": q_observation,
            "ct_query_search": q_observation,
            "ct_query_gate_logit": reference.new_zeros((batch_size,)),
            "ct_query_gate_probability": reference.new_full(
                (batch_size,), 0.5),
            "ct_query_gate_internal": reference.new_zeros((batch_size,)),
            "ct_query_gate": reference.new_zeros((batch_size,)),
            "ct_query_shift_norm": reference.new_zeros((batch_size,)),
            "ct_search_targetness_logits": reference.new_full(
                (batch_size, point_count), -20.0),
            "ct_search_targetness_logits_obs": reference.new_full(
                (batch_size, point_count), -20.0),
            "ct_search_targetness_logits_motion": reference.new_full(
                (batch_size, point_count), -20.0),
            "ct_search_point_votes": torch.nan_to_num(
                point_xy, nan=0.0, posinf=0.0, neginf=0.0),
            "ct_search_raw_xy": reference.new_zeros((batch_size, 2)),
            "ct_search_raw_obs_xy": reference.new_zeros((batch_size, 2)),
            "ct_search_raw_motion_xy": reference.new_zeros((batch_size, 2)),
            "ct_search_raw_alpha_xy": reference.new_zeros((batch_size, 2)),
            "ct_search_candidate_valid": reference.new_zeros((batch_size,)),
            "ct_search_structural_valid": reference.new_zeros((batch_size,)),
            "ct_search_new_support_valid": reference.new_zeros((batch_size,)),
            "ct_search_support_valid": support_mask.to(reference.dtype),
            "ct_search_available": reference.new_zeros((batch_size,)),
            "ct_search_effective": reference.new_zeros((batch_size,)),
            "ct_search_evidence": reference.new_zeros(
                (batch_size, self.feature_dim)),
            "ct_search_targetness_mean": reference.new_zeros((batch_size,)),
            "ct_search_targetness_max": reference.new_zeros((batch_size,)),
            "ct_search_targetness_entropy": reference.new_zeros((batch_size,)),
            "ct_search_normalized_ess": reference.new_zeros((batch_size,)),
            "ct_search_finite_point_count": reference.new_zeros((batch_size,)),
            "ct_search_extension_selected_count": reference.new_zeros(
                (batch_size,)),
            "ct_search_extension_feature_mean": reference.new_zeros(
                (batch_size, self.feature_dim)),
            "ct_search_extension_feature_max": reference.new_zeros(
                (batch_size, self.feature_dim)),
            "ct_search_extension_local_mean": reference.new_zeros(
                (batch_size,)),
            "ct_search_extension_local_max": reference.new_zeros(
                (batch_size,)),
            "ct_search_extension_mass_ratio": reference.new_zeros(
                (batch_size,)),
            "ct_search_extension_vote_rms": reference.new_full(
                (batch_size,), float("inf")),
            "ct_search_presence_logit": reference.new_zeros((batch_size,)),
            "ct_search_presence_probability": reference.new_zeros(
                (batch_size,)),
            "ct_motion_residual_saturation": reference.new_zeros(
                (batch_size,)),
        }
        if active_index.numel() == 0:
            return defaults

        def select(value):
            return value.index_select(0, active_index)

        active_output = self._forward_valid(
            select(encoded_points), select(point_xy),
            select(point_valid_mask), select(point_source),
            select(observation_query), select(observation_stats),
            select(motion_feature), select(kinematic_xy), select(learned_xy),
            select(residual_unit), select(sigma_parallel_perp),
            select(envelope_parallel_perp), select(direction_xy),
            select(motion_valid), select(query_delta_t), select(gap_ratio),
            select(history_valid_ratio),
            point_branch_source=(
                None if point_branch_source is None
                else select(point_branch_source)),
            search_support_valid=(
                None if search_support_valid is None
                else select(search_support_valid)),
        )
        for key, value in active_output.items():
            defaults[key] = defaults[key].index_copy(0, active_index, value)
        return defaults

    def _forward_valid(
            self,
            encoded_points,
            point_xy,
            point_valid_mask,
            point_source,
            observation_query,
            observation_stats,
            motion_feature,
            kinematic_xy,
            learned_xy,
            residual_unit,
            sigma_parallel_perp,
            envelope_parallel_perp,
            direction_xy,
            motion_valid,
            query_delta_t,
            gap_ratio,
            history_valid_ratio,
            point_branch_source=None,
            search_support_valid=None):
        if encoded_points.dim() != 3 or encoded_points.shape[-1] != self.point_dim:
            raise ValueError("joint Full points must have shape [B,N,point_dim]")
        batch_size, point_count, _ = encoded_points.shape
        if point_xy.shape != (batch_size, point_count, 2):
            raise ValueError("joint Full point xy has the wrong shape")
        if observation_query.shape != (batch_size, self.query_dim):
            raise ValueError("joint Full observation query has the wrong shape")

        reference = encoded_points
        finite_points = (
            torch.isfinite(encoded_points).all(dim=2)
            & torch.isfinite(point_xy).all(dim=2))
        point_mask = (
            point_valid_mask.to(device=reference.device).reshape(
                batch_size, point_count) > 0)
        if search_support_valid is None:
            search_support_valid = reference.new_ones((batch_size,))
        support_valid = (_batch_scalar(
            search_support_valid, reference) > 0)
        point_mask = point_mask & finite_points & support_valid.unsqueeze(1)
        mask_f = point_mask.to(reference.dtype)
        safe_points = torch.nan_to_num(
            encoded_points, nan=0.0, posinf=0.0, neginf=0.0)
        safe_xy = torch.nan_to_num(
            point_xy, nan=0.0, posinf=0.0, neginf=0.0)
        support_source = point_source.to(
            device=reference.device, dtype=torch.long).reshape(
                batch_size, point_count)
        support_source = torch.clamp(support_source, min=0, max=1)
        if point_branch_source is None:
            point_branch_source = torch.zeros_like(point_source)
        branch_source = point_branch_source.to(
            device=reference.device, dtype=torch.long).reshape(
                batch_size, point_count)
        branch_source = torch.clamp(branch_source, min=0, max=2)

        motion_valid = _batch_scalar(motion_valid, reference)
        finite_motion = (
            torch.isfinite(motion_feature).all(dim=1)
            & torch.isfinite(kinematic_xy).all(dim=1)
            & torch.isfinite(learned_xy).all(dim=1)
            & torch.isfinite(residual_unit).all(dim=1)
            & torch.isfinite(sigma_parallel_perp).all(dim=1)
            & torch.isfinite(envelope_parallel_perp).all(dim=1)
            & torch.isfinite(direction_xy).all(dim=1))
        motion_valid = (
            (motion_valid > 0) & finite_motion).to(reference.dtype)
        safe_motion = torch.nan_to_num(
            motion_feature.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        safe_kinematic = torch.nan_to_num(
            kinematic_xy.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        safe_learned = torch.nan_to_num(
            learned_xy.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        safe_residual = torch.clamp(torch.nan_to_num(
            residual_unit.detach(), nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
        safe_residual = safe_residual * motion_valid.unsqueeze(1)
        envelope = torch.clamp(torch.nan_to_num(
            envelope_parallel_perp.detach(), nan=0.1, posinf=4.0, neginf=0.1),
            min=0.1)
        sigma = torch.clamp(torch.nan_to_num(
            sigma_parallel_perp.detach(), nan=0.1, posinf=4.0, neginf=0.1),
            min=0.1)
        sigma_ratio = torch.clamp(
            sigma / envelope, min=0.0, max=4.0
        ) * motion_valid.unsqueeze(1)
        direction = torch.nan_to_num(
            direction_xy.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        direction_norm = torch.linalg.norm(direction, dim=1, keepdim=True)
        default_direction = torch.zeros_like(direction)
        default_direction[:, 0] = 1.0
        direction = torch.where(
            (direction_norm > 1e-6).expand_as(direction),
            direction / torch.clamp(direction_norm, min=1e-6),
            default_direction)
        perpendicular = torch.stack((-direction[:, 1], direction[:, 0]), dim=1)

        def aligned_geometry(center, scales):
            delta = safe_xy - center.unsqueeze(1)
            longitudinal = (
                delta * direction.unsqueeze(1)).sum(dim=2) / scales[:, 0:1]
            lateral = (
                delta * perpendicular.unsqueeze(1)).sum(dim=2) / scales[:, 1:2]
            return longitudinal, lateral

        kin_long, kin_lat = aligned_geometry(safe_kinematic, envelope)
        learned_long, learned_lat = aligned_geometry(safe_learned, sigma)
        kin_distance = torch.sqrt(torch.clamp(
            kin_long.pow(2) + kin_lat.pow(2), min=0.0))
        mahalanobis = torch.clamp(
            learned_long.pow(2) + learned_lat.pow(2),
            min=0.0, max=self.mahalanobis_clip)
        disagreement = torch.linalg.norm(safe_residual, dim=1, keepdim=True)
        learned_geometry = torch.stack((
            learned_long, learned_lat, mahalanobis,
            disagreement.expand(-1, point_count)), dim=2)
        kinematic_geometry = torch.stack((
            kin_long, kin_lat, kin_distance), dim=2)

        query_dt = torch.clamp(_batch_scalar(
            query_delta_t, reference), min=1e-3)
        gap = torch.clamp(_batch_scalar(gap_ratio, reference), min=0.0)
        history = torch.clamp(_batch_scalar(
            history_valid_ratio, reference), min=0.0, max=1.0)
        context = torch.cat((
            safe_residual,
            sigma_ratio,
            torch.log1p(query_dt).unsqueeze(1),
            torch.log1p(gap).unsqueeze(1),
            history.unsqueeze(1),
        ), dim=1)
        point_feature = (
            self.point_encoder(safe_points)
            + self.kinematic_geometry(kinematic_geometry)
            + self.learned_geometry(learned_geometry)
            * motion_valid.reshape(batch_size, 1, 1)
            + self.motion_context(context).unsqueeze(1)
            + self.support_source_embedding(support_source)
            + self.branch_source_embedding(branch_source))
        point_feature = torch.relu(self.point_norm(point_feature))
        point_feature = point_feature * mask_f.unsqueeze(2)

        q0 = self.query_observation_norm(observation_query.detach())
        query_motion_input = torch.cat((
            safe_motion, safe_residual,
            torch.log(torch.clamp(sigma, min=0.1))), dim=1)
        query_residual = self.motion_query_norm(
            self.motion_query_projection(query_motion_input))

        key = self.point_key(point_feature)
        observation_score = (
            key * q0.unsqueeze(1)).sum(dim=2) / math.sqrt(self.query_dim)
        residual_score = (
            key * query_residual.unsqueeze(1)).sum(dim=2) / math.sqrt(
                self.query_dim)
        local_score = self.local_targetness(point_feature).squeeze(2)

        extension_bool = point_mask & (support_source == 1)
        extension_mask = extension_bool.to(reference.dtype)
        extension_count = extension_mask.sum(dim=1)
        extension_denominator = torch.clamp(
            extension_count.unsqueeze(1), min=1.0)
        extension_feature_mean = (
            point_feature * extension_mask.unsqueeze(2)).sum(dim=1)
        extension_feature_mean = (
            extension_feature_mean / extension_denominator)
        extension_feature_max = point_feature.masked_fill(
            ~extension_bool.unsqueeze(2), float('-inf')).max(dim=1)[0]
        extension_feature_max = torch.where(
            (extension_count > 0).unsqueeze(1), extension_feature_max,
            torch.zeros_like(extension_feature_max))
        extension_local_mean = (
            local_score * extension_mask).sum(dim=1) / torch.clamp(
                extension_count, min=1.0)
        extension_local_max = local_score.masked_fill(
            ~extension_bool, float('-inf')).max(dim=1)[0]
        extension_local_max = torch.where(
            extension_count > 0, extension_local_max,
            torch.zeros_like(extension_local_max))

        if self.contract_version >= 2:
            # Presence observes extension-only, query-independent evidence.
            # It cannot be dominated by overlapping B0 points and cannot
            # inherit the predicted alpha query it is later used to explain.
            presence_input = torch.cat((
                extension_feature_mean,
                extension_feature_max,
                torch.log1p(extension_count).unsqueeze(1),
                extension_local_mean.unsqueeze(1),
                extension_local_max.unsqueeze(1),
            ), dim=1)
            presence_logit = self.presence_head(presence_input).squeeze(1)
            presence_probability = torch.sigmoid(presence_logit)
        else:
            # The legacy presence input is formed below after the alpha arm
            # aggregation, preserving contract-v1 behavior exactly.
            presence_logit = None
            presence_probability = None

        safe_obs_stats = torch.nan_to_num(
            observation_stats.detach().to(reference.dtype),
            nan=0.0, posinf=0.0, neginf=0.0)
        gate_parts = [
            torch.log1p(query_dt).unsqueeze(1),
            torch.log1p(gap).unsqueeze(1),
            history.unsqueeze(1),
            safe_residual,
            sigma_ratio,
            safe_obs_stats,
        ]
        if self.contract_version >= 2:
            gate_parts.extend((
                presence_probability.detach().unsqueeze(1),
                torch.log1p(extension_count).detach().unsqueeze(1),
                extension_local_mean.detach().unsqueeze(1),
                extension_local_max.detach().unsqueeze(1),
            ))
        gate_input = torch.cat(gate_parts, dim=1)
        query_gate_logit = self.query_gate(gate_input).squeeze(1)
        query_gate_probability = torch.sigmoid(query_gate_logit)
        query_gate = (
            query_gate_probability * motion_valid
            * support_valid.to(reference.dtype)
            if self.use_reliability_gate
            else motion_valid * support_valid.to(reference.dtype))
        query_gate = query_gate * self.query_gate_scale
        if (self.use_reliability_gate and self.training
                and self.motion_dropout > 0):
            keep = (torch.rand_like(query_gate) >= self.motion_dropout).to(
                query_gate.dtype)
            query_gate = query_gate * keep
        q_search = q0 + query_gate.unsqueeze(1) * query_residual

        base_logits = local_score + observation_score
        obs_logits = base_logits
        motion_logits = base_logits + residual_score
        alpha_logits = base_logits + query_gate.unsqueeze(1) * residual_score
        invalid_logits = alpha_logits.new_full(alpha_logits.shape, -20.0)
        obs_logits = torch.where(point_mask, obs_logits, invalid_logits)
        motion_logits = torch.where(point_mask, motion_logits, invalid_logits)
        alpha_logits = torch.where(point_mask, alpha_logits, invalid_logits)

        vote_offset = self.max_vote_offset * torch.tanh(
            self.vote_head(point_feature))
        point_votes = safe_xy + vote_offset
        finite_count = mask_f.sum(dim=1)

        def aggregate(logits):
            scores = torch.sigmoid(logits) * mask_f
            score_sum = scores.sum(dim=1, keepdim=True)
            raw_xy = (
                scores.unsqueeze(2) * point_votes).sum(dim=1) / torch.clamp(
                    score_sum, min=1e-6)
            evidence_value = (
                scores.unsqueeze(2) * point_feature).sum(dim=1)
            evidence_value = evidence_value / torch.clamp(
                score_sum, min=1e-6)
            probability = scores / torch.clamp(score_sum, min=1e-6)
            entropy_value = -(
                probability * torch.log(torch.clamp(probability, min=1e-8))
            ).sum(dim=1)
            ess_value = 1.0 / torch.clamp(
                probability.pow(2).sum(dim=1)
                * torch.clamp(finite_count, min=1.0), min=1e-6)
            mean_value = scores.sum(dim=1) / torch.clamp(
                finite_count, min=1.0)
            max_value = scores.max(dim=1)[0]
            return (scores, score_sum, raw_xy, evidence_value,
                    entropy_value, ess_value, mean_value, max_value)

        (_, _, raw_obs_xy, _, _, _, _, _) = aggregate(obs_logits)
        (_, _, raw_motion_xy, _, _, _, _, _) = aggregate(motion_logits)
        (targetness, weight_sum, raw_alpha_xy, evidence, entropy,
         normalized_ess, targetness_mean, targetness_max) = aggregate(
             alpha_logits)
        structural_valid = (
            (finite_count >= 3)
            & torch.isfinite(raw_alpha_xy).all(dim=1))
        raw_obs_xy = torch.nan_to_num(
            raw_obs_xy, nan=0.0, posinf=0.0, neginf=0.0)
        raw_motion_xy = torch.nan_to_num(
            raw_motion_xy, nan=0.0, posinf=0.0, neginf=0.0)
        raw_alpha_xy = torch.nan_to_num(
            raw_alpha_xy, nan=0.0, posinf=0.0, neginf=0.0)

        extension_mass = (targetness * extension_mask).sum(dim=1)
        extension_mass_ratio = extension_mass / torch.clamp(
            weight_sum.squeeze(1), min=1e-6)
        extension_weights = targetness * extension_mask
        extension_weight_sum = extension_weights.sum(dim=1, keepdim=True)
        extension_vote_center = (
            extension_weights.unsqueeze(2) * point_votes).sum(dim=1)
        extension_vote_center = extension_vote_center / torch.clamp(
            extension_weight_sum, min=1e-6)
        extension_vote_variance = (
            extension_weights * torch.sum(
                (point_votes - extension_vote_center.unsqueeze(1)).pow(2),
                dim=2)).sum(dim=1) / torch.clamp(
                    extension_weight_sum.squeeze(1), min=1e-6)
        extension_vote_rms = torch.sqrt(torch.clamp(
            extension_vote_variance, min=0.0))
        extension_vote_rms = torch.where(
            extension_weight_sum.squeeze(1) > 0,
            extension_vote_rms,
            torch.full_like(extension_vote_rms, float('inf')))
        new_support_valid = (extension_count > 0) & support_valid
        candidate_valid = (
            structural_valid & new_support_valid).to(reference.dtype)

        if self.contract_version < 2:
            presence_input = torch.cat((
                evidence,
                targetness_mean.unsqueeze(1),
                targetness_max.unsqueeze(1),
                entropy.unsqueeze(1),
                normalized_ess.unsqueeze(1),
            ), dim=1)
            presence_logit = self.presence_head(presence_input).squeeze(1)
            presence_probability = torch.sigmoid(presence_logit)
        presence_probability = (
            presence_probability * support_valid.to(reference.dtype))
        search_available = candidate_valid > 0
        if self.presence_hard_gate:
            search_effective = (
                search_available
                & (presence_probability >= self.presence_threshold))
        else:
            search_effective = search_available
        effective_query_gate = query_gate * search_effective.to(
            reference.dtype)
        deployed_query = torch.where(
            search_effective.unsqueeze(1), q_search, q0)

        return {
            "ct_query_observation": q0,
            "ct_query_residual": query_residual,
            "ct_query_search_internal": q_search,
            "ct_query_search": deployed_query,
            "ct_query_gate_logit": query_gate_logit,
            "ct_query_gate_probability": query_gate_probability,
            "ct_query_gate_internal": query_gate,
            "ct_query_gate": effective_query_gate,
            "ct_query_shift_norm": torch.linalg.norm(
                deployed_query - q0, dim=1) / math.sqrt(self.query_dim),
            "ct_search_targetness_logits": alpha_logits,
            "ct_search_targetness_logits_obs": obs_logits,
            "ct_search_targetness_logits_motion": motion_logits,
            "ct_search_point_votes": point_votes,
            "ct_search_raw_xy": raw_alpha_xy,
            "ct_search_raw_obs_xy": raw_obs_xy,
            "ct_search_raw_motion_xy": raw_motion_xy,
            "ct_search_raw_alpha_xy": raw_alpha_xy,
            "ct_search_candidate_valid": candidate_valid,
            "ct_search_structural_valid": structural_valid.to(
                reference.dtype),
            "ct_search_new_support_valid": new_support_valid.to(
                reference.dtype),
            "ct_search_support_valid": support_valid.to(reference.dtype),
            "ct_search_available": search_available.to(reference.dtype),
            "ct_search_effective": search_effective.to(reference.dtype),
            "ct_search_evidence": evidence,
            "ct_search_targetness_mean": targetness_mean,
            "ct_search_targetness_max": targetness_max,
            "ct_search_targetness_entropy": entropy,
            "ct_search_normalized_ess": normalized_ess,
            "ct_search_finite_point_count": finite_count,
            "ct_search_extension_selected_count": extension_count,
            "ct_search_extension_feature_mean": extension_feature_mean,
            "ct_search_extension_feature_max": extension_feature_max,
            "ct_search_extension_local_mean": extension_local_mean,
            "ct_search_extension_local_max": extension_local_max,
            "ct_search_extension_mass_ratio": extension_mass_ratio,
            "ct_search_extension_vote_rms": extension_vote_rms,
            "ct_search_presence_logit": presence_logit,
            "ct_search_presence_probability": presence_probability,
            "ct_motion_residual_saturation": (
                safe_residual.abs() >= 0.98).to(reference.dtype).mean(dim=1),
        }


class JointScalarResidualRouter(nn.Module):
    """Hard observation/Search action router with detached candidate inputs."""

    def __init__(
            self,
            observation_stats_dim=5,
            hidden_dim=64,
            init_probability=0.01,
            decision_threshold=0.5,
            extension_mass_threshold=0.25,
            presence_threshold=0.5,
            radius_base=0.5,
            radius_per_second=0.5,
            radius_max=2.0,
            contract_version=1,
            presence_hard_gate=True):
        super().__init__()
        self.observation_stats_dim = int(observation_stats_dim)
        self.contract_version = int(contract_version)
        self.presence_hard_gate = bool(presence_hard_gate)
        if self.contract_version not in (1, 2):
            raise ValueError("joint router contract version must be 1 or 2")
        self.radius_base = float(radius_base)
        self.radius_per_second = float(radius_per_second)
        self.radius_max = float(radius_max)
        self.extension_mass_threshold = float(extension_mass_threshold)
        self.presence_threshold = float(presence_threshold)
        self.register_buffer(
            "decision_threshold",
            torch.tensor(float(decision_threshold), dtype=torch.float32))
        scalar_dim = self.observation_stats_dim + (
            19 if self.contract_version >= 2 else 12)
        self.gate = nn.Sequential(
            nn.Linear(scalar_dim, int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(
            self.gate[-1].bias,
            math.log(init_probability / (1.0 - init_probability)),
        )

    def set_decision_threshold(self, threshold):
        threshold = float(threshold)
        if not math.isfinite(threshold) or not 0.5 <= threshold <= 1.0:
            raise ValueError(
                "joint router threshold must be finite and in [0.5,1]")
        self.decision_threshold.copy_(self.decision_threshold.new_tensor(
            threshold))

    def forward(
            self,
            observation_box,
            raw_search_xy,
            candidate_valid,
            observation_stats,
            targetness_mean,
            targetness_max,
            targetness_entropy,
            normalized_ess,
            query_gate,
            query_delta_t,
            gap_ratio,
            enabled=True,
            extension_mass_ratio=None,
            extension_vote_rms=None,
            presence_probability=None,
            total_point_count=None,
            extension_point_count=None,
            extension_voxels=None,
            coverage_need=None,
            quality_valid=None,
            proposal_valid=None,
            predicted_displacement=None):
        reference = observation_box
        candidate_valid = torch.nan_to_num(_batch_scalar(
            candidate_valid, reference).detach(), nan=0.0,
            posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
        query_dt = torch.clamp(torch.nan_to_num(_batch_scalar(
            query_delta_t, reference).detach(), nan=0.0,
            posinf=0.0, neginf=0.0), min=0.0)
        gap = torch.clamp(torch.nan_to_num(_batch_scalar(
            gap_ratio, reference).detach(), nan=0.0,
            posinf=0.0, neginf=0.0), min=0.0)
        observation_xy = observation_box[:, :2].detach()
        raw_xy = raw_search_xy.detach()
        residual = torch.nan_to_num(
            raw_xy - observation_xy, nan=0.0, posinf=0.0, neginf=0.0)
        residual_norm = torch.linalg.norm(residual, dim=1)
        if extension_mass_ratio is None:
            extension_mass_ratio = reference.new_ones((reference.shape[0],))
        if extension_vote_rms is None:
            extension_vote_rms = reference.new_zeros((reference.shape[0],))
        if presence_probability is None:
            presence_probability = reference.new_ones((reference.shape[0],))
        extension_mass = torch.nan_to_num(_batch_scalar(
            extension_mass_ratio, reference).detach(), nan=0.0,
            posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
        vote_rms = _batch_scalar(
            extension_vote_rms, reference).detach()
        vote_rms = torch.nan_to_num(
            vote_rms, nan=1e6, posinf=1e6, neginf=1e6)
        presence = torch.nan_to_num(_batch_scalar(
            presence_probability, reference).detach(), nan=0.0,
            posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
        feature_parts = [
            torch.nan_to_num(observation_stats.detach(), nan=0.0,
                             posinf=0.0, neginf=0.0),
            _batch_scalar(targetness_mean, reference).detach().unsqueeze(1),
            _batch_scalar(targetness_max, reference).detach().unsqueeze(1),
            _batch_scalar(targetness_entropy, reference).detach().unsqueeze(1),
            _batch_scalar(normalized_ess, reference).detach().unsqueeze(1),
            _batch_scalar(query_gate, reference).detach().unsqueeze(1),
            torch.log1p(query_dt).unsqueeze(1),
            torch.log1p(gap).unsqueeze(1),
            residual_norm.unsqueeze(1),
            candidate_valid.unsqueeze(1),
            extension_mass.unsqueeze(1),
            vote_rms.unsqueeze(1),
            presence.unsqueeze(1),
        ]
        if self.contract_version >= 2:
            def diagnostic_feature(value, default=0.0, log_scale=False):
                if value is None:
                    result = reference.new_full(
                        (reference.shape[0],), float(default))
                else:
                    result = torch.nan_to_num(_batch_scalar(
                        value, reference).detach(), nan=float(default),
                        posinf=float(default), neginf=float(default))
                if log_scale:
                    result = torch.log1p(torch.clamp(result, min=0.0))
                return result.unsqueeze(1)

            # These quantities inform the learned router but never veto a
            # structurally valid Search candidate under contract v2.
            feature_parts.extend((
                diagnostic_feature(total_point_count, log_scale=True),
                diagnostic_feature(extension_point_count, log_scale=True),
                diagnostic_feature(extension_voxels, log_scale=True),
                diagnostic_feature(coverage_need),
                diagnostic_feature(quality_valid),
                diagnostic_feature(proposal_valid),
                diagnostic_feature(predicted_displacement, log_scale=True),
            ))
        scalar_features = torch.cat(feature_parts, dim=1)
        scalar_features = torch.nan_to_num(
            scalar_features, nan=0.0, posinf=0.0, neginf=0.0)
        router_logit = self.gate(scalar_features).squeeze(1)
        learned_probability = torch.sigmoid(router_logit)
        radius = torch.clamp(
            self.radius_base + self.radius_per_second * query_dt,
            max=self.radius_max)
        residual_scale = torch.clamp(
            radius / torch.clamp(residual_norm, min=1e-6), max=1.0)
        bounded_residual = residual * residual_scale.unsqueeze(1)
        evidence_valid = (
            (candidate_valid > 0) & torch.isfinite(raw_xy).all(dim=1))
        if self.contract_version < 2:
            evidence_valid = (
                evidence_valid
                & (presence >= self.presence_threshold)
                & (extension_mass >= self.extension_mass_threshold)
                & (vote_rms <= radius))
        elif self.presence_hard_gate:
            evidence_valid = (
                evidence_valid
                & (presence >= self.presence_threshold))
        if bool(enabled):
            applied_action = (
                evidence_valid
                & (learned_probability >= self.decision_threshold.to(
                    device=reference.device,
                    dtype=learned_probability.dtype)))
        else:
            # -B3 is a forced-Search action ablation, but it may never bypass
            # the evidence-valid safety contract.
            applied_action = evidence_valid
        applied_gate = applied_action.to(reference.dtype)
        final_xy = observation_xy + applied_gate.unsqueeze(1) * bounded_residual
        soft_xy = observation_xy + learned_probability.unsqueeze(1) * bounded_residual
        final_box = torch.cat((final_xy, observation_box[:, 2:]), dim=1)
        soft_box = torch.cat((soft_xy, observation_box[:, 2:].detach()), dim=1)
        return final_box, {
            "ct_router_logit": router_logit,
            "ct_router_gate": learned_probability,
            "ct_router_applied_gate": applied_gate,
            "ct_router_evidence_valid": evidence_valid.to(reference.dtype),
            "ct_router_soft_box": soft_box,
            "ct_router_radius": radius,
            "ct_router_residual_xy": residual,
            "ct_router_bounded_residual_xy": bounded_residual,
            "ct_router_clip_rate": (residual_norm > radius).to(
                reference.dtype),
        }
