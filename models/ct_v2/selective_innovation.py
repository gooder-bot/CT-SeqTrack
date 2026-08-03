"""Motion-conditioned search refinement and signed-horizon routing.

This module is deliberately separate from the B2-v2.1 advantage fusion and
the first CRPA prototype.  Candidate production, recursive counterfactual
labelling, and selective routing have distinct training boundaries here.
"""

import hashlib
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


SELECTIVE_ROLLOUT_SCHEMA = "ct_seqtrack.selective_rollout.v2"
SELECTIVE_ROUTER_SCHEMA = "ct_seqtrack.signed_horizon_router.v2"


def _clip_vector_norm(vector, radius, eps=1e-6):
    norm = torch.linalg.norm(vector, dim=-1, keepdim=True)
    scale = torch.minimum(
        torch.ones_like(norm), radius / torch.clamp(norm, min=eps))
    return vector * scale


class MotionConditionedSearchRefiner(nn.Module):
    """Use endpoint point evidence to refine, rather than compete with, B1.

    ``point_mlp``, query, targetness, and vote submodules intentionally retain
    the B2-v2.1 names and shapes so their weights can be migrated exactly.  The
    new relative-to-motion geometry is injected through a separate adapter.
    """

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
            presence_threshold=0.5,
            radius_base=0.5,
            radius_per_second=0.5,
            radius_max=2.0,
            eps=1e-6):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.query_dim = int(query_dim)
        self.max_vote_offset = float(max_vote_offset)
        self.pool_temperature = float(pool_temperature)
        self.presence_threshold = float(presence_threshold)
        self.radius_base = float(radius_base)
        self.radius_per_second = float(radius_per_second)
        self.radius_max = float(radius_max)
        self.eps = float(eps)
        if min(self.feature_dim, self.query_dim) <= 0:
            raise ValueError("B2-v2.2 feature dimensions must be positive")
        if self.max_vote_offset <= 0 or self.pool_temperature <= 0:
            raise ValueError("B2-v2.2 vote and pooling scales must be positive")
        if not 0.0 <= self.presence_threshold <= 1.0:
            raise ValueError("B2-v2.2 presence threshold must be in [0,1]")
        if self.radius_base < 0.0 or self.radius_max <= 0.0:
            raise ValueError("B2-v2.2 refinement radii must be valid")

        # These layers match TrajectorySearchEvidenceV21 exactly.
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

        # New v2.2-only paths.
        self.motion_geometry_mlp = nn.Sequential(
            nn.Linear(2, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.feature_dim),
        )
        self.source_fusion = nn.Sequential(
            nn.Linear(3 * self.feature_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.ReLU(inplace=True),
        )
        self.presence_head = nn.Sequential(
            nn.Linear(3 * self.feature_dim + 2, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, 1),
        )
        nn.init.zeros_(self.presence_head[-1].weight)
        nn.init.constant_(
            self.presence_head[-1].bias, math.log(0.1 / 0.9))
        # Non-migrated adapters start as exact no-ops so the transferred
        # v2.1 point/query/targetness/vote path is not randomly corrupted.
        nn.init.zeros_(self.film_scale.weight)
        nn.init.zeros_(self.film_scale.bias)
        nn.init.zeros_(self.film_shift.weight)
        nn.init.zeros_(self.film_shift.bias)
        nn.init.zeros_(self.motion_geometry_mlp[-1].weight)
        nn.init.zeros_(self.motion_geometry_mlp[-1].bias)

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
            raise ValueError("B2-v2.2 scalar must contain one value per sample")
        return value.reshape(batch_size, 1)

    def _masked_softmax(self, logits, valid):
        mask = valid > 0
        scaled = logits / self.pool_temperature
        scaled = scaled.masked_fill(~mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(scaled, dim=1) * mask.to(logits.dtype)
        denominator = weights.sum(dim=1, keepdim=True)
        return torch.where(
            denominator > 0,
            weights / torch.clamp(denominator, min=self.eps),
            torch.zeros_like(weights),
        )

    def forward(
            self,
            point_inputs,
            point_xy,
            delta_to_motion,
            point_valid_mask,
            point_source,
            geometry_valid,
            support_anchor_xy,
            observation_feature,
            motion_feature,
            motion_proposal_xy,
            motion_valid,
            observation_stats,
            query_delta_t,
            gap_ratio,
            sigma_parallel,
            sigma_perpendicular,
            available_count=None,
            extension_count=None,
            overlap_count=None):
        if point_inputs.dim() != 3 or point_inputs.shape[-1] != 9:
            raise ValueError("B2-v2.2 point inputs must have shape [B,N,9]")
        if point_xy.shape != point_inputs.shape[:2] + (2,):
            raise ValueError("B2-v2.2 point xy must have shape [B,N,2]")
        if delta_to_motion.shape != point_xy.shape:
            raise ValueError("relative-to-motion geometry must match point xy")
        batch_size, point_count, _ = point_inputs.shape
        valid = point_valid_mask.to(
            device=point_inputs.device, dtype=point_inputs.dtype)
        if valid.shape != (batch_size, point_count):
            raise ValueError("B2-v2.2 point mask must have shape [B,N]")
        source = point_source.to(
            device=point_inputs.device, dtype=torch.long)
        if source.shape != (batch_size, point_count):
            raise ValueError("B2-v2.2 point source must have shape [B,N]")
        if bool(torch.any((source < 0) | (source > 1)).item()):
            raise ValueError("B2-v2.2 point source must be 0 or 1")

        geometry = (self._batch_scalar(
            geometry_valid, point_inputs) > 0).to(point_inputs.dtype)
        valid = (valid > 0).to(point_inputs.dtype) * geometry
        motion_valid_column = (self._batch_scalar(
            motion_valid, point_inputs) > 0).to(point_inputs.dtype)
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
            motion_valid_column,
        ), dim=1)

        observation_feature = observation_feature.detach()
        motion_feature = motion_feature.detach()
        observation_stats = observation_stats.detach()
        motion_proposal_xy = motion_proposal_xy.detach()
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
        point_feature = (
            point_feature
            + self.source_embedding(source)
            + self.motion_geometry_mlp(torch.nan_to_num(
                delta_to_motion.detach(),
                nan=0.0, posinf=0.0, neginf=0.0)))
        film_scale = torch.tanh(self.film_scale(context)).unsqueeze(1)
        film_shift = self.film_shift(context).unsqueeze(1)
        point_feature = point_feature * (1.0 + film_scale) + film_shift

        query = self.query_projection(torch.nan_to_num(torch.cat((
            observation_feature, observation_stats), dim=1)))
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
        overlap_mask = valid * (source == 0).to(valid.dtype)
        extension_mask = valid * (source == 1).to(valid.dtype)
        overlap_weights = self._masked_softmax(
            targetness_logits, overlap_mask)
        extension_weights = self._masked_softmax(
            targetness_logits, extension_mask)
        overlap_token = (
            point_feature * overlap_weights.unsqueeze(2)).sum(dim=1)
        extension_token = (
            point_feature * extension_weights.unsqueeze(2)).sum(dim=1)
        evidence_token = self.source_fusion(torch.cat((
            overlap_token, extension_token, context), dim=1))

        vote_offsets = self.max_vote_offset * torch.tanh(
            self.vote_head(point_feature))
        point_center_votes = point_xy + vote_offsets
        raw_proposal_xy = (
            point_center_votes * pool_weights.unsqueeze(2)).sum(dim=1)

        valid_count = valid.sum(dim=1)
        point_row_valid = (valid_count >= 3).to(point_inputs.dtype)
        candidate_available = (
            point_row_valid * geometry.squeeze(1)
            * motion_valid_column.squeeze(1))
        source_present = torch.stack((
            (overlap_mask.sum(dim=1) > 0).to(point_inputs.dtype),
            (extension_mask.sum(dim=1) > 0).to(point_inputs.dtype),
        ), dim=1)
        presence_logit = self.presence_head(torch.cat((
            overlap_token, extension_token, context, source_present),
            dim=1)).squeeze(1)
        presence_probability = torch.sigmoid(presence_logit)
        presence_probability = presence_probability * candidate_available
        presence_valid = (
            presence_probability >= self.presence_threshold).to(
                point_inputs.dtype)
        candidate_valid = candidate_available * presence_valid

        query_dt = self._batch_scalar(
            query_delta_t, point_inputs, default=0.1)
        refinement_radius = torch.clamp(
            self.radius_base + self.radius_per_second * query_dt,
            max=self.radius_max)
        refinement_residual = _clip_vector_norm(
            raw_proposal_xy - motion_proposal_xy,
            refinement_radius,
            eps=self.eps,
        )
        refined_xy = motion_proposal_xy + refinement_residual

        targetness_mass = (targetness * valid).sum(dim=1)
        targetness_mean = targetness_mass / torch.clamp(
            valid_count, min=1.0)
        masked_targetness = targetness.masked_fill(valid <= 0, -1.0)
        targetness_max = torch.clamp(
            masked_targetness.max(dim=1).values, min=0.0)
        probability = torch.clamp(
            targetness, min=self.eps, max=1.0 - self.eps)
        point_entropy = -(
            probability * torch.log(probability)
            + (1.0 - probability) * torch.log(1.0 - probability))
        entropy = (point_entropy * valid).sum(dim=1) / torch.clamp(
            valid_count, min=1.0)
        squared_mass = pool_weights.pow(2).sum(dim=1)
        raw_ess = torch.where(
            point_row_valid > 0,
            1.0 / torch.clamp(squared_mass, min=self.eps),
            torch.zeros_like(squared_mass),
        )
        normalized_ess = torch.where(
            point_row_valid > 0,
            torch.clamp(raw_ess / torch.clamp(valid_count, min=1.0), 0.0, 1.0),
            torch.zeros_like(raw_ess),
        )
        extension_weight_ratio = (
            pool_weights * (source == 1).to(pool_weights.dtype)).sum(dim=1)

        point_row = point_row_valid.unsqueeze(1)
        evidence_token = evidence_token * candidate_available.unsqueeze(1)
        raw_proposal_xy = raw_proposal_xy * point_row
        refined_xy = refined_xy * candidate_available.unsqueeze(1)
        refinement_residual = (
            refinement_residual * candidate_available.unsqueeze(1))
        targetness_mass = targetness_mass * point_row_valid
        targetness_mean = targetness_mean * point_row_valid
        targetness_max = targetness_max * point_row_valid
        entropy = entropy * point_row_valid
        extension_weight_ratio = extension_weight_ratio * point_row_valid

        return {
            "search_support_anchor_xy": support_anchor_xy.detach(),
            "search_raw_vote_xy": raw_proposal_xy,
            "motion_search_refined_xy": refined_xy,
            "motion_search_refinement_residual_xy": refinement_residual,
            "search_presence_logit": presence_logit,
            "search_presence_probability": presence_probability,
            "search_overlap_token": overlap_token * point_row,
            "search_extension_token": extension_token * point_row,
            "search_normalized_ess": normalized_ess,
            "search_raw_ess": raw_ess,
            "motion_search_candidate_available": candidate_available,
            "motion_search_candidate_valid": candidate_valid,
            "search_v22_evidence_token": evidence_token,
            "search_v22_match_logits": match_logits,
            "search_v22_local_targetness_logits": local_logits,
            "search_v22_targetness_logits": targetness_logits,
            "search_v22_targetness": targetness,
            "search_v22_pool_weights": pool_weights * point_row,
            "search_v22_overlap_pool_weights": overlap_weights * point_row,
            "search_v22_extension_pool_weights": extension_weights * point_row,
            "search_v22_targetness_mass": targetness_mass,
            "search_v22_targetness_mean": targetness_mean,
            "search_v22_targetness_max": targetness_max,
            "search_v22_targetness_entropy": entropy,
            "search_v22_extension_weight_ratio": extension_weight_ratio,
            "search_v22_vote_offsets": vote_offsets,
            "search_v22_point_center_votes": point_center_votes,
            "search_v22_valid_count": valid_count,
            "search_v22_refinement_radius": refinement_radius.squeeze(1),
        }


class SignedHorizonInnovationRouter(nn.Module):
    """Observation-anchored top-1 router trained on signed H-step gains."""

    STEP_RATIOS = (0.25, 0.5, 1.0)
    SCALAR_FEATURE_NAMES = (
        "obs_log_points", "obs_log_foreground", "obs_mean_foreground",
        "obs_history_valid_ratio", "obs_delta_t_ratio", "obs_entropy",
        "obs_refinement_x", "obs_refinement_y", "obs_refinement_norm",
        "motion_log_sigma_x", "motion_log_sigma_y", "motion_history_valid",
        "search_presence", "search_targetness_mean",
        "search_targetness_max", "search_targetness_entropy",
        "search_normalized_ess", "search_extension_weight_ratio",
        "search_log_available", "search_log_extension",
        "search_log_overlap", "motion_residual_x", "motion_residual_y",
        "motion_residual_norm", "motion_search_residual_x",
        "motion_search_residual_y", "motion_search_residual_norm",
        "candidate_distance", "candidate_cosine", "support_motion_distance",
        "raw_motion_distance", "query_delta_t", "gap_ratio",
        "motion_valid", "motion_search_valid",
    )

    def __init__(
            self,
            observation_dim=256,
            motion_dim=128,
            search_dim=128,
            observation_stats_dim=5,
            context_dim=32,
            hidden_dim=96,
            gain_threshold=0.0,
            radius_base=0.5,
            radius_per_second=0.5,
            radius_max=2.0,
            normal_step_cap=0.20,
            gap_step_cap=0.35,
            eps=1e-6):
        super().__init__()
        self.observation_dim = int(observation_dim)
        self.motion_dim = int(motion_dim)
        self.search_dim = int(search_dim)
        self.observation_stats_dim = int(observation_stats_dim)
        self.scalar_dim = len(self.SCALAR_FEATURE_NAMES)
        self.radius_base = float(radius_base)
        self.radius_per_second = float(radius_per_second)
        self.radius_max = float(radius_max)
        self.normal_step_cap = float(normal_step_cap)
        self.gap_step_cap = float(gap_step_cap)
        self.eps = float(eps)
        if self.observation_stats_dim != 5:
            raise ValueError("signed router requires five observation stats")
        if not 0.0 <= self.normal_step_cap <= self.gap_step_cap <= 1.0:
            raise ValueError("signed router step caps are invalid")

        def projection(input_dim):
            return nn.Sequential(
                nn.LayerNorm(int(input_dim)),
                nn.Linear(int(input_dim), int(context_dim)),
                nn.ReLU(inplace=True),
            )

        self.observation_projection = projection(self.observation_dim)
        self.motion_projection = projection(self.motion_dim)
        self.search_projection = projection(self.search_dim)
        self.trunk = nn.Sequential(
            nn.Linear(3 * int(context_dim) + self.scalar_dim,
                      int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.ReLU(inplace=True),
        )
        self.median_gain_head = nn.Linear(int(hidden_dim), 2)
        self.gain_spread_head = nn.Linear(int(hidden_dim), 2)
        self.step_head = nn.Linear(int(hidden_dim), 2 * len(self.STEP_RATIOS))
        nn.init.zeros_(self.median_gain_head.weight)
        nn.init.zeros_(self.median_gain_head.bias)
        nn.init.zeros_(self.gain_spread_head.weight)
        nn.init.constant_(
            self.gain_spread_head.bias, math.log(math.expm1(0.05)))
        nn.init.zeros_(self.step_head.weight)
        nn.init.zeros_(self.step_head.bias)
        self.register_buffer(
            "scalar_feature_mean", torch.zeros(self.scalar_dim))
        self.register_buffer(
            "scalar_feature_std", torch.ones(self.scalar_dim))
        self.register_buffer(
            "calibrated_gain_threshold",
            torch.tensor(float(gain_threshold), dtype=torch.float32))
        self.register_buffer(
            "step_ratio_values",
            torch.tensor(self.STEP_RATIOS, dtype=torch.float32))

    @property
    def export_feature_dim(self):
        return (
            self.observation_dim + self.motion_dim
            + self.search_dim + self.scalar_dim)

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
            raise ValueError("signed router scalar must contain one per sample")
        return value.reshape(batch_size, 1)

    def set_scalar_normalization(self, mean, std):
        mean = torch.as_tensor(
            mean, device=self.scalar_feature_mean.device,
            dtype=self.scalar_feature_mean.dtype).reshape(-1)
        std = torch.as_tensor(
            std, device=self.scalar_feature_std.device,
            dtype=self.scalar_feature_std.dtype).reshape(-1)
        if mean.numel() != self.scalar_dim or std.numel() != self.scalar_dim:
            raise ValueError("signed router scalar normalization width mismatch")
        self.scalar_feature_mean.copy_(mean)
        self.scalar_feature_std.copy_(torch.clamp(std, min=1e-4))

    def set_gain_threshold(self, value):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("signed router threshold must be finite")
        self.calibrated_gain_threshold.fill_(value)

    def _predict(self, observation_feature, motion_feature, search_feature,
                 scalar_features):
        normalized_scalar = (
            scalar_features - self.scalar_feature_mean.unsqueeze(0)
        ) / torch.clamp(self.scalar_feature_std.unsqueeze(0), min=1e-4)
        hidden = self.trunk(torch.cat((
            self.observation_projection(observation_feature),
            self.motion_projection(motion_feature),
            self.search_projection(search_feature),
            torch.nan_to_num(normalized_scalar),
        ), dim=1))
        q50 = self.median_gain_head(hidden)
        q10 = q50 - F.softplus(self.gain_spread_head(hidden))
        step_logits = self.step_head(hidden).reshape(
            -1, 2, len(self.STEP_RATIOS))
        return q10, q50, step_logits

    def predict_export_features(self, exported_features):
        if exported_features.dim() != 2:
            raise ValueError("router features must have shape [B,D]")
        if exported_features.shape[1] != self.export_feature_dim:
            raise ValueError("router feature width mismatch")
        obs_end = self.observation_dim
        motion_end = obs_end + self.motion_dim
        search_end = motion_end + self.search_dim
        q10, q50, step_logits = self._predict(
            exported_features[:, :obs_end],
            exported_features[:, obs_end:motion_end],
            exported_features[:, motion_end:search_end],
            exported_features[:, search_end:],
        )
        return {
            "q10": q10,
            "q50": q50,
            "step_logits": step_logits,
        }

    def forward(
            self,
            observation_box,
            observation_feature,
            observation_stats,
            observation_entropy,
            observation_refinement_xy,
            motion_feature,
            motion_proposal_xy,
            motion_log_sigma_xy,
            motion_valid,
            history_valid_ratio,
            search_feature,
            motion_search_xy,
            motion_search_valid,
            search_presence,
            search_targetness_mean,
            search_targetness_max,
            search_targetness_entropy,
            search_normalized_ess,
            search_extension_weight_ratio,
            search_available_count,
            search_extension_count,
            search_overlap_count,
            search_support_anchor_xy,
            search_raw_vote_xy,
            query_delta_t,
            gap_ratio,
            enabled_scale=1.0,
            forced_candidate=None,
            forced_step_ratio=None):
        observation_box = observation_box.detach()
        observation_feature = observation_feature.detach()
        observation_stats = observation_stats.detach()
        motion_feature = motion_feature.detach()
        search_feature = search_feature.detach()
        def detached(value):
            return value.detach() if torch.is_tensor(value) else value

        observation_entropy = detached(observation_entropy)
        observation_refinement_xy = detached(observation_refinement_xy)
        motion_log_sigma_xy = detached(motion_log_sigma_xy)
        motion_valid = detached(motion_valid)
        history_valid_ratio = detached(history_valid_ratio)
        motion_search_valid = detached(motion_search_valid)
        search_presence = detached(search_presence)
        search_targetness_mean = detached(search_targetness_mean)
        search_targetness_max = detached(search_targetness_max)
        search_targetness_entropy = detached(search_targetness_entropy)
        search_normalized_ess = detached(search_normalized_ess)
        search_extension_weight_ratio = detached(
            search_extension_weight_ratio)
        search_available_count = detached(search_available_count)
        search_extension_count = detached(search_extension_count)
        search_overlap_count = detached(search_overlap_count)
        query_delta_t = detached(query_delta_t)
        gap_ratio = detached(gap_ratio)
        motion_proposal_xy = motion_proposal_xy.detach()
        motion_search_xy = motion_search_xy.detach()
        search_support_anchor_xy = search_support_anchor_xy.detach()
        search_raw_vote_xy = search_raw_vote_xy.detach()
        reference = observation_box[:, :2]
        batch_size = reference.shape[0]

        dt = self._batch_scalar(query_delta_t, reference, default=0.1)
        gap = self._batch_scalar(gap_ratio, reference, default=1.0)
        radius = torch.clamp(
            self.radius_base + self.radius_per_second * dt,
            max=self.radius_max)
        motion_residual = _clip_vector_norm(
            motion_proposal_xy - reference, radius, self.eps)
        motion_search_residual = _clip_vector_norm(
            motion_search_xy - reference, radius, self.eps)
        motion_valid_column = (
            self._batch_scalar(motion_valid, reference) > 0)
        motion_search_valid_column = (
            self._batch_scalar(motion_search_valid, reference) > 0)
        safe_radius = torch.clamp(radius, min=self.eps)
        motion_norm = torch.linalg.norm(
            motion_residual, dim=1, keepdim=True)
        motion_search_norm = torch.linalg.norm(
            motion_search_residual, dim=1, keepdim=True)
        candidate_delta = motion_proposal_xy - motion_search_xy
        candidate_distance = torch.linalg.norm(
            candidate_delta, dim=1, keepdim=True)
        cosine = (
            motion_residual * motion_search_residual).sum(
                dim=1, keepdim=True) / torch.clamp(
                    motion_norm * motion_search_norm, min=self.eps)
        refinement = torch.nan_to_num(
            observation_refinement_xy.detach()) / safe_radius
        scalar_features = torch.cat((
            observation_stats.detach(),
            self._batch_scalar(observation_entropy, reference),
            refinement,
            torch.linalg.norm(refinement, dim=1, keepdim=True),
            torch.nan_to_num(motion_log_sigma_xy.detach()),
            self._batch_scalar(history_valid_ratio, reference),
            self._batch_scalar(search_presence, reference),
            self._batch_scalar(search_targetness_mean, reference),
            self._batch_scalar(search_targetness_max, reference),
            self._batch_scalar(search_targetness_entropy, reference),
            self._batch_scalar(search_normalized_ess, reference),
            self._batch_scalar(search_extension_weight_ratio, reference),
            torch.log1p(torch.clamp(self._batch_scalar(
                search_available_count, reference), min=0.0)) / 8.0,
            torch.log1p(torch.clamp(self._batch_scalar(
                search_extension_count, reference), min=0.0)) / 8.0,
            torch.log1p(torch.clamp(self._batch_scalar(
                search_overlap_count, reference), min=0.0)) / 8.0,
            motion_residual / safe_radius,
            motion_norm / safe_radius,
            motion_search_residual / safe_radius,
            motion_search_norm / safe_radius,
            candidate_distance / safe_radius,
            torch.clamp(cosine, -1.0, 1.0),
            torch.linalg.norm(
                search_support_anchor_xy - motion_proposal_xy,
                dim=1, keepdim=True) / safe_radius,
            torch.linalg.norm(
                search_raw_vote_xy - motion_proposal_xy,
                dim=1, keepdim=True) / safe_radius,
            dt,
            gap,
            motion_valid_column.to(reference.dtype),
            motion_search_valid_column.to(reference.dtype),
        ), dim=1)
        if scalar_features.shape[1] != self.scalar_dim:
            raise RuntimeError(
                "signed router scalar feature contract changed: "
                f"{scalar_features.shape[1]} != {self.scalar_dim}")
        exported_features = torch.cat((
            observation_feature,
            motion_feature,
            search_feature,
            torch.nan_to_num(scalar_features),
        ), dim=1)
        q10, q50, step_logits = self._predict(
            observation_feature, motion_feature, search_feature,
            scalar_features)

        valid = torch.cat((
            motion_valid_column, motion_search_valid_column), dim=1)
        masked_q10 = q10.masked_fill(
            ~valid, torch.finfo(q10.dtype).min)
        best_q10, selected_index = masked_q10.max(dim=1)
        any_valid = valid.any(dim=1)
        selected_step_class = step_logits.argmax(dim=2).gather(
            1, selected_index.unsqueeze(1)).squeeze(1)
        step_ratios = self.step_ratio_values.to(
            device=reference.device, dtype=reference.dtype)
        selected_step_ratio = step_ratios[selected_step_class]
        threshold = self.calibrated_gain_threshold.to(
            device=reference.device, dtype=reference.dtype)
        intervene = any_valid & (best_q10 > threshold)

        if forced_candidate is not None:
            forced = self._batch_scalar(
                forced_candidate, reference, default=-1.0
            ).reshape(-1).to(torch.long)
            requested = forced >= 0
            clamped = torch.clamp(forced, 0, 1)
            forced_valid = valid.gather(
                1, clamped.unsqueeze(1)).squeeze(1)
            selected_index = torch.where(requested, clamped, selected_index)
            intervene = requested & forced_valid
            if forced_step_ratio is not None:
                forced_ratio = self._batch_scalar(
                    forced_step_ratio, reference, default=0.25).reshape(-1)
                allowed_distance = torch.abs(
                    forced_ratio.unsqueeze(1)
                    - step_ratios.unsqueeze(0)).min(dim=1).values
                if bool(torch.any(allowed_distance > 1e-6).item()):
                    raise ValueError(
                        "forced step ratio must be 0.25, 0.5, or 1.0")
                selected_step_ratio = torch.where(
                    requested, forced_ratio, selected_step_ratio)

        enabled_scale = float(enabled_scale)
        if not 0.0 <= enabled_scale <= 1.0:
            raise ValueError("signed router enabled_scale must be in [0,1]")
        intervene = intervene & (enabled_scale > 0.0)
        candidates = torch.stack((
            motion_residual, motion_search_residual), dim=1)
        selected_residual = candidates.gather(
            1,
            selected_index.reshape(batch_size, 1, 1).expand(-1, 1, 2),
        ).squeeze(1)
        gap_state = gap.reshape(-1) > 1.0 + self.eps
        step_cap = torch.where(
            gap_state,
            reference.new_full((batch_size,), self.gap_step_cap),
            reference.new_full((batch_size,), self.normal_step_cap),
        )
        alpha = torch.clamp(selected_step_ratio, 0.0, 1.0) * step_cap
        alpha = alpha * intervene.to(reference.dtype) * enabled_scale
        correction = selected_residual * alpha.unsqueeze(1)
        final_xy = torch.where(
            intervene.unsqueeze(1), reference + correction, reference)
        final_box = torch.cat((final_xy, observation_box[:, 2:]), dim=1)
        selected_candidate = torch.where(
            intervene, selected_index + 1, torch.zeros_like(selected_index))

        return final_box, {
            "signed_router_features": exported_features,
            "signed_gain_quantiles": torch.stack((q10, q50), dim=2),
            "signed_step_logits": step_logits,
            "signed_candidate_residual_xy": candidates,
            "signed_candidate_valid": valid.to(reference.dtype),
            "signed_selected_candidate": selected_candidate,
            "signed_selected_candidate_index": selected_index,
            "signed_selected_step_ratio": selected_step_ratio,
            "signed_step_cap": step_cap,
            "signed_gain_threshold": threshold,
            "signed_abstained": (~intervene).to(reference.dtype),
            "signed_applied_alpha": alpha,
            "signed_correction_xy": correction,
            "signed_fusion_radius": radius.squeeze(1),
        }


def pinball_loss(prediction, target, quantile):
    error = target - prediction
    return torch.maximum(
        float(quantile) * error, (float(quantile) - 1.0) * error)


def signed_horizon_router_loss(
        prediction,
        signed_gain,
        candidate_valid,
        step_supervision_margin=0.02,
        q10_weight=1.0,
        q50_weight=0.5,
        step_weight=0.2):
    """Train quantiles on the best signed H-step gain for each candidate."""
    if signed_gain.dim() != 3 or signed_gain.shape[1:] != (2, 3):
        raise ValueError("signed_gain must have shape [B,2,3]")
    valid = candidate_valid.to(signed_gain.dtype)
    best_gain, best_step = signed_gain.max(dim=2)
    q10 = prediction["q10"]
    q50 = prediction["q50"]
    denominator = torch.clamp(valid.sum(), min=1.0)
    loss_q10 = (pinball_loss(q10, best_gain, 0.10) * valid).sum(
        ) / denominator
    loss_q50 = (pinball_loss(q50, best_gain, 0.50) * valid).sum(
        ) / denominator
    step_mask = valid * (best_gain > float(
        step_supervision_margin)).to(valid.dtype)
    step_error = F.cross_entropy(
        prediction["step_logits"].reshape(-1, 3),
        best_step.reshape(-1),
        reduction="none",
    ).reshape_as(step_mask)
    loss_step = (step_error * step_mask).sum() / torch.clamp(
        step_mask.sum(), min=1.0)
    total = (
        float(q10_weight) * loss_q10
        + float(q50_weight) * loss_q50
        + float(step_weight) * loss_step)
    return {
        "loss": total,
        "loss_q10": loss_q10,
        "loss_q50": loss_q50,
        "loss_step": loss_step,
        "best_signed_gain": best_gain,
        "best_step_class": best_step,
    }


def discounted_tracking_cost(ious, distances, gamma=0.8):
    """Joint Success/Precision proxy used by three-frame rollouts."""
    ious = np.asarray(ious, dtype=np.float64).reshape(-1)
    distances = np.asarray(distances, dtype=np.float64).reshape(-1)
    if ious.shape != distances.shape or ious.size == 0:
        raise ValueError("IoU and distance rollout arrays must be non-empty")
    weights = np.power(float(gamma), np.arange(ious.size, dtype=np.float64))
    frame_cost = (
        0.5 * (1.0 - np.clip(ious, 0.0, 1.0))
        + 0.5 * np.minimum(np.maximum(distances, 0.0) / 2.0, 1.0))
    return float(np.sum(weights * frame_cost) / np.sum(weights))


def stable_tracklet_partition(tracklet_key, seed=42):
    """Stable 70/15/15 train/dev/calibration partition by whole tracklet."""
    digest = hashlib.sha256(
        f"{int(seed)}::{str(tracklet_key)}".encode("utf-8")
    ).digest()
    value = int.from_bytes(digest[:8], "big") / float(2 ** 64)
    if value < 0.70:
        return "train"
    if value < 0.85:
        return "dev"
    return "calibration"


def calibrate_gain_threshold(
        q10,
        signed_gain,
        candidate_valid,
        step_class=None,
        min_precision=0.75,
        max_harm_rate=0.10,
        min_coverage=0.05,
        max_coverage=0.25,
        helpful_margin=0.02):
    """Choose one conservative threshold without touching mini_val."""
    q10 = np.asarray(q10, dtype=np.float64)
    gains = np.asarray(signed_gain, dtype=np.float64)
    valid = np.asarray(candidate_valid, dtype=bool)
    if q10.ndim != 2 or q10.shape[1] != 2:
        raise ValueError("q10 must have shape [N,2]")
    if gains.shape != (q10.shape[0], 2, 3):
        raise ValueError("signed_gain must have shape [N,2,3]")
    if valid.shape != q10.shape:
        raise ValueError("candidate_valid must match q10")
    masked = np.where(valid, q10, -np.inf)
    selected = np.argmax(masked, axis=1)
    scores = masked[np.arange(masked.shape[0]), selected]
    if step_class is None:
        applied_gain = gains.max(axis=2)
    else:
        step_class = np.asarray(step_class, dtype=np.int64)
        if step_class.shape != q10.shape:
            raise ValueError("step_class must match q10")
        applied_gain = np.take_along_axis(
            gains, step_class[..., None], axis=2).squeeze(2)
    selected_gain = applied_gain[np.arange(applied_gain.shape[0]), selected]
    finite_scores = scores[np.isfinite(scores)]
    if finite_scores.size == 0:
        raise RuntimeError("calibration contains no valid candidates")
    thresholds = np.unique(np.concatenate((
        np.nextafter(finite_scores.min(), -np.inf).reshape(1),
        finite_scores,
        np.nextafter(finite_scores.max(), np.inf).reshape(1),
    )))
    candidates = []
    total = max(1, q10.shape[0])
    for threshold in thresholds:
        chosen = np.isfinite(scores) & (scores > threshold)
        count = int(chosen.sum())
        coverage = count / float(total)
        if count == 0:
            precision = 1.0
            harm_rate = 0.0
        else:
            precision = float(np.mean(
                selected_gain[chosen] > float(helpful_margin)))
            harm_rate = float(np.mean(selected_gain[chosen] < 0.0))
        if (float(min_coverage) <= coverage <= float(max_coverage)
                and precision >= float(min_precision)
                and harm_rate <= float(max_harm_rate)):
            candidates.append((
                coverage, precision, -harm_rate, float(threshold), count))
    if not candidates:
        raise RuntimeError(
            "no calibration threshold satisfies precision/harm/coverage "
            "guardrails")
    coverage, precision, neg_harm, threshold, count = max(candidates)
    return {
        "threshold": threshold,
        "coverage": coverage,
        "helpful_precision": precision,
        "harm_rate": -neg_harm,
        "selected_count": count,
    }
