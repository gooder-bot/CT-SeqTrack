"""Compact jointly-trained B1/B2/B3 coupling for CT-SeqTrack.

The module deliberately keeps the stable SeqTrack3D observation path outside
the search branch.  Motion contributes a normalized residual score, while the
point cloud remains responsible for the actual raw Search proposal.
"""

import math

import torch
from torch import nn


def _batch_scalar(value, reference):
    value = value.to(device=reference.device, dtype=reference.dtype)
    if value.numel() == 1:
        value = value.reshape(1).expand(reference.shape[0])
    return value.reshape(reference.shape[0])


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
            mahalanobis_clip=25.0,
            use_reliability_gate=True):
        super().__init__()
        self.point_dim = int(point_dim)
        self.feature_dim = int(feature_dim)
        self.query_dim = int(query_dim)
        self.motion_dim = int(motion_dim)
        self.observation_stats_dim = int(observation_stats_dim)
        self.max_vote_offset = float(max_vote_offset)
        self.motion_dropout = float(motion_dropout)
        self.mahalanobis_clip = float(mahalanobis_clip)
        self.use_reliability_gate = bool(use_reliability_gate)
        if min(self.point_dim, self.feature_dim, self.query_dim,
               self.motion_dim) <= 0:
            raise ValueError("joint Full feature dimensions must be positive")
        if not 0.0 <= self.motion_dropout < 1.0:
            raise ValueError("joint Full motion dropout must be in [0,1)")
        if not 0.0 < gate_init_probability < 1.0:
            raise ValueError("query gate initialization must be in (0,1)")

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
        self.source_embedding = nn.Embedding(2, self.feature_dim)
        self.point_norm = nn.LayerNorm(self.feature_dim)

        self.query_observation_norm = nn.LayerNorm(self.query_dim)
        self.motion_query_projection = nn.Linear(
            self.motion_dim + 4, self.query_dim)
        self.motion_query_norm = nn.LayerNorm(self.query_dim)
        self.search_query_norm = nn.LayerNorm(self.query_dim)
        # dt, gap, history, residual(2), sigma/M(2), observation stats
        gate_dim = 7 + self.observation_stats_dim
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
            history_valid_ratio):
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
        point_mask = point_mask & finite_points
        mask_f = point_mask.to(reference.dtype)
        safe_points = torch.nan_to_num(
            encoded_points, nan=0.0, posinf=0.0, neginf=0.0)
        safe_xy = torch.nan_to_num(
            point_xy, nan=0.0, posinf=0.0, neginf=0.0)
        source = point_source.to(
            device=reference.device, dtype=torch.long).reshape(
                batch_size, point_count)
        source = torch.clamp(source, min=0, max=1)

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
            motion_feature, nan=0.0, posinf=0.0, neginf=0.0)
        safe_kinematic = torch.nan_to_num(
            kinematic_xy, nan=0.0, posinf=0.0, neginf=0.0)
        safe_learned = torch.nan_to_num(
            learned_xy, nan=0.0, posinf=0.0, neginf=0.0)
        safe_residual = torch.clamp(torch.nan_to_num(
            residual_unit, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
        safe_residual = safe_residual * motion_valid.unsqueeze(1)
        envelope = torch.clamp(torch.nan_to_num(
            envelope_parallel_perp, nan=0.1, posinf=4.0, neginf=0.1),
            min=0.1)
        sigma = torch.clamp(torch.nan_to_num(
            sigma_parallel_perp, nan=0.1, posinf=4.0, neginf=0.1),
            min=0.1)
        sigma_ratio = torch.clamp(
            sigma / envelope, min=0.0, max=4.0
        ) * motion_valid.unsqueeze(1)
        direction = torch.nan_to_num(
            direction_xy, nan=0.0, posinf=0.0, neginf=0.0)
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
            + self.source_embedding(source))
        point_feature = torch.relu(self.point_norm(point_feature))
        point_feature = point_feature * mask_f.unsqueeze(2)

        q0 = self.query_observation_norm(observation_query.detach())
        query_motion_input = torch.cat((
            safe_motion, safe_residual,
            torch.log(torch.clamp(sigma, min=0.1))), dim=1)
        query_residual = self.motion_query_norm(
            self.motion_query_projection(query_motion_input))
        safe_obs_stats = torch.nan_to_num(
            observation_stats.detach().to(reference.dtype),
            nan=0.0, posinf=0.0, neginf=0.0)
        gate_input = torch.cat((
            torch.log1p(query_dt).unsqueeze(1),
            torch.log1p(gap).unsqueeze(1),
            history.unsqueeze(1),
            safe_residual,
            sigma_ratio,
            safe_obs_stats,
        ), dim=1)
        query_gate_logit = self.query_gate(gate_input).squeeze(1)
        query_gate = (
            torch.sigmoid(query_gate_logit) * motion_valid
            if self.use_reliability_gate else motion_valid)
        if (self.use_reliability_gate and self.training
                and self.motion_dropout > 0):
            keep = (torch.rand_like(query_gate) >= self.motion_dropout).to(
                query_gate.dtype)
            query_gate = query_gate * keep
        q_search = self.search_query_norm(
            q0 + query_gate.unsqueeze(1) * query_residual)

        key = self.point_key(point_feature)
        observation_score = (
            key * q0.unsqueeze(1)).sum(dim=2) / math.sqrt(self.query_dim)
        residual_score = (
            key * query_residual.unsqueeze(1)).sum(dim=2) / math.sqrt(
                self.query_dim)
        local_score = self.local_targetness(point_feature).squeeze(2)
        targetness_logits = (
            local_score + observation_score
            + query_gate.unsqueeze(1) * residual_score)
        targetness_logits = torch.where(
            point_mask, targetness_logits,
            targetness_logits.new_full(targetness_logits.shape, -20.0))

        vote_offset = self.max_vote_offset * torch.tanh(
            self.vote_head(point_feature))
        point_votes = safe_xy + vote_offset
        targetness = torch.sigmoid(targetness_logits) * mask_f
        weight_sum = targetness.sum(dim=1, keepdim=True)
        raw_search_xy = (
            targetness.unsqueeze(2) * point_votes).sum(dim=1) / torch.clamp(
                weight_sum, min=1e-6)
        finite_count = mask_f.sum(dim=1)
        candidate_valid = (
            (finite_count >= 3)
            & torch.isfinite(raw_search_xy).all(dim=1)).to(reference.dtype)
        raw_search_xy = torch.nan_to_num(
            raw_search_xy, nan=0.0, posinf=0.0, neginf=0.0)

        evidence = (
            targetness.unsqueeze(2) * point_feature).sum(dim=1) / torch.clamp(
                weight_sum, min=1e-6)
        probability = targetness / torch.clamp(weight_sum, min=1e-6)
        entropy = -(
            probability * torch.log(torch.clamp(probability, min=1e-8))
        ).sum(dim=1)
        normalized_ess = 1.0 / torch.clamp(
            probability.pow(2).sum(dim=1) * torch.clamp(finite_count, min=1.0),
            min=1e-6)
        valid_denominator = torch.clamp(finite_count, min=1.0)
        targetness_mean = targetness.sum(dim=1) / valid_denominator
        targetness_max = targetness.max(dim=1)[0]

        return {
            "ct_query_observation": q0,
            "ct_query_residual": query_residual,
            "ct_query_search": q_search,
            "ct_query_gate_logit": query_gate_logit,
            "ct_query_gate": query_gate,
            "ct_query_shift_norm": torch.linalg.norm(
                q_search - q0, dim=1) / math.sqrt(self.query_dim),
            "ct_search_targetness_logits": targetness_logits,
            "ct_search_point_votes": point_votes,
            "ct_search_raw_xy": raw_search_xy,
            "ct_search_candidate_valid": candidate_valid,
            "ct_search_evidence": evidence,
            "ct_search_targetness_mean": targetness_mean,
            "ct_search_targetness_max": targetness_max,
            "ct_search_targetness_entropy": entropy,
            "ct_search_normalized_ess": normalized_ess,
            "ct_search_finite_point_count": finite_count,
            "ct_motion_residual_saturation": (
                safe_residual.abs() >= 0.98).to(reference.dtype).mean(dim=1),
        }


class JointScalarResidualRouter(nn.Module):
    """Observation-versus-raw-Search scalar gate with a time-aware radius."""

    def __init__(
            self,
            observation_stats_dim=5,
            hidden_dim=64,
            init_probability=0.05,
            radius_base=0.5,
            radius_per_second=0.5,
            radius_max=2.0):
        super().__init__()
        self.observation_stats_dim = int(observation_stats_dim)
        self.radius_base = float(radius_base)
        self.radius_per_second = float(radius_per_second)
        self.radius_max = float(radius_max)
        scalar_dim = self.observation_stats_dim + 9
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
            enabled=True):
        reference = observation_box
        candidate_valid = _batch_scalar(
            candidate_valid, reference).detach().clamp(0.0, 1.0)
        query_dt = torch.clamp(_batch_scalar(
            query_delta_t, reference).detach(), min=0.0)
        gap = torch.clamp(_batch_scalar(
            gap_ratio, reference).detach(), min=0.0)
        observation_xy = observation_box[:, :2].detach()
        raw_xy = raw_search_xy.detach()
        residual = torch.nan_to_num(
            raw_xy - observation_xy, nan=0.0, posinf=0.0, neginf=0.0)
        residual_norm = torch.linalg.norm(residual, dim=1)
        scalar_features = torch.cat((
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
        ), dim=1)
        router_logit = self.gate(scalar_features).squeeze(1)
        learned_gate = torch.sigmoid(router_logit) * candidate_valid
        applied_gate = (
            learned_gate if bool(enabled) else candidate_valid)
        radius = torch.clamp(
            self.radius_base + self.radius_per_second * query_dt,
            max=self.radius_max)
        residual_scale = torch.clamp(
            radius / torch.clamp(residual_norm, min=1e-6), max=1.0)
        bounded_residual = residual * residual_scale.unsqueeze(1)
        final_xy = observation_xy + applied_gate.unsqueeze(1) * bounded_residual
        final_box = torch.cat((final_xy, observation_box[:, 2:]), dim=1)
        return final_box, {
            "ct_router_logit": router_logit,
            "ct_router_gate": learned_gate,
            "ct_router_applied_gate": applied_gate,
            "ct_router_radius": radius,
            "ct_router_residual_xy": residual,
            "ct_router_bounded_residual_xy": bounded_residual,
            "ct_router_clip_rate": (residual_norm > radius).to(
                reference.dtype),
        }
