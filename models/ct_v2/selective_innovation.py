"""Motion-conditioned search refinement and signed-horizon routing.

This module is deliberately separate from the B2-v2.1 advantage fusion and
the first CRPA prototype.  Candidate production, recursive counterfactual
labelling, and selective routing have distinct training boundaries here.
"""

import hashlib
import json
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


SELECTIVE_ROLLOUT_SCHEMA = "ct_seqtrack.selective_rollout.v2"
SELECTIVE_ROUTER_SCHEMA = "ct_seqtrack.signed_horizon_router.v2"
SELECTIVE_V3_ROLLOUT_SCHEMA = "ct_seqtrack.selective_rollout.v3"
SELECTIVE_V3_ROUTER_SCHEMA = "ct_seqtrack.action_router.v3"
SELECTIVE_V4_ROLLOUT_SCHEMA = "ct_seqtrack.selective_rollout.v4"
SELECTIVE_V4_ROUTER_SCHEMA = "ct_seqtrack.action_router.v4"
B2_V3_PROTECTED_PREFIXES = (
    "seg_pointnet.", "mini_pointnet.", "motion_mlp.",
    "motion_state_mlp.", "feature_pointnet.", "Transformer.",
    "physical_motion_encoder.", "state_aligned_search_refiner.",
    "asymmetric_dual_query.",
)


def _tensor_prefixes_hash(state, prefixes):
    digest = hashlib.sha256()
    keys = sorted(
        key for key in state
        if any(key.startswith(prefix) for prefix in prefixes))
    for key in keys:
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest(), keys


def validate_trainable_parameter_prefixes(
        named_parameters, allowed_prefixes, required_prefixes):
    named_parameters = list(named_parameters)
    allowed_prefixes = tuple(allowed_prefixes)
    required_prefixes = set(required_prefixes)
    unexpected = sorted(
        name for name, parameter in named_parameters
        if parameter.requires_grad and not name.startswith(allowed_prefixes))
    if unexpected:
        raise RuntimeError(
            "trainable parameters escaped the formal prefix contract: "
            + ", ".join(unexpected[:20]))
    actual = {
        prefix for prefix in allowed_prefixes
        if any(name.startswith(prefix) and parameter.requires_grad
               for name, parameter in named_parameters)}
    if actual != required_prefixes:
        raise RuntimeError(
            f"formal trainable prefixes are incomplete: "
            f"expected={sorted(required_prefixes)}, actual={sorted(actual)}")
    return actual


def require_nonzero_finite_gradient(named_parameters, prefix):
    gradients = [
        parameter.grad.detach()
        for name, parameter in named_parameters
        if name.startswith(prefix) and parameter.requires_grad
        and parameter.grad is not None]
    if not gradients:
        raise RuntimeError(f"no gradients for trainable prefix {prefix}")
    if (not all(bool(torch.isfinite(gradient).all().item())
                for gradient in gradients)
            or not any(bool(torch.count_nonzero(gradient).item())
                       for gradient in gradients)):
        raise RuntimeError(
            f"invalid or zero gradients for trainable prefix {prefix}")
    return True


def validate_b2_v3_router_package(checkpoint, router=None):
    """Reject cold, uncalibrated, or tampered final-router checkpoints."""
    package = checkpoint.get("b2_v3_router_package")
    if not isinstance(package, dict):
        raise RuntimeError(
            "B2-v3 selective evaluation requires a checkpoint created by "
            "package_b2_v3_checkpoint.py")
    package_schema = package.get("schema")
    if package_schema not in (
            "ct_seqtrack.selective_checkpoint.v3",
            "ct_seqtrack.selective_checkpoint.v4"):
        raise RuntimeError("unsupported B2-v3 router package schema")
    if package_schema == "ct_seqtrack.selective_checkpoint.v4":
        for key in (
                "feature_schema", "feature_schema_hash",
                "scalar_normalization", "action_names", "step_ratios",
                "candidate_checkpoint_sha256", "candidate_config_sha256",
                "promotion_manifest_sha256"):
            if package.get(key) is None:
                raise RuntimeError(f"B3 v4 router package lacks {key}")
        if router is None:
            raise RuntimeError(
                "B3 v4 package validation requires the runtime router")
        if package.get("feature_schema") != router.feature_schema:
            raise RuntimeError("B3 v4 feature names/order mismatch")
        if package.get(
                "feature_schema_hash") != router.feature_schema_hash:
            raise RuntimeError("B3 v4 feature schema hash mismatch")
        if package.get("action_names") != router.action_names:
            raise RuntimeError("B3 v4 action order mismatch")
        if package.get("step_ratios") != list(router.STEP_RATIOS):
            raise RuntimeError("B3 v4 step ratio mismatch")
    calibration = package.get("calibration")
    if (not isinstance(calibration, dict)
            or calibration.get("status") != "passed"
            or calibration.get("partition") != "calibration"):
        raise RuntimeError("B2-v3 selective checkpoint lacks final calibration")
    if (package_schema == "ct_seqtrack.selective_checkpoint.v4"
            and (calibration.get("method") != "recursive_tracklet_scan_v4"
                 or calibration.get("final_recursive") is not True
                 or int(calibration.get("intervention_count", 0)) <= 0
                 or float(calibration.get("harm_rate", 1.0)) > 0.05)):
        raise RuntimeError(
            "B3 v4 requires a non-empty safe recursive tracklet calibration")
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict):
        raise RuntimeError("B2-v3 package has no state_dict")
    if package_schema == "ct_seqtrack.selective_checkpoint.v4":
        normalization = package["scalar_normalization"]
        normalization_keys = {
            "mean": "scalar_feature_mean",
            "std": "scalar_feature_std",
            "p1": "scalar_clip_low",
            "p99": "scalar_clip_high",
        }
        for metadata_key, state_suffix in normalization_keys.items():
            value = state.get(
                "action_consistent_router_v3." + state_suffix)
            expected = normalization.get(metadata_key)
            if value is None or expected is None:
                raise RuntimeError(
                    f"B3 v4 normalization lacks {metadata_key}")
            expected_tensor = torch.as_tensor(
                expected, dtype=value.dtype, device=value.device)
            if (expected_tensor.shape != value.shape
                    or not torch.equal(value, expected_tensor)):
                raise RuntimeError(
                    f"B3 v4 normalization mismatch for {metadata_key}")
    nonfinite = sorted(
        key for key, value in state.items()
        if torch.is_tensor(value)
        and (value.is_floating_point() or value.is_complex())
        and not bool(torch.isfinite(value).all().item()))
    if nonfinite:
        raise RuntimeError(
            "B2-v3 package contains non-finite tensors: "
            + ", ".join(nonfinite[:20]))
    protected_hash, protected_keys = _tensor_prefixes_hash(
        state, B2_V3_PROTECTED_PREFIXES)
    if (not protected_keys
            or protected_hash != package.get("protected_prefix_hash")):
        raise RuntimeError("B2-v3 packaged B0/B1/refiner hash is invalid")
    router_keys = sorted(
        key for key in state
        if key.startswith("action_consistent_router_v3."))
    try:
        router_tensor_count = int(package.get("router_tensor_count", -1))
    except (TypeError, ValueError) as error:
        raise RuntimeError("B2-v3 router tensor count is invalid") from error
    if not router_keys or len(router_keys) != router_tensor_count:
        raise RuntimeError("B2-v3 packaged router key set is invalid")
    threshold = state.get(
        "action_consistent_router_v3.calibrated_gain_threshold")
    expected_threshold = calibration.get("threshold")
    try:
        threshold_matches = (
            threshold is not None
            and expected_threshold is not None
            and bool(torch.isfinite(threshold.detach()).all().item())
            and abs(float(threshold.detach().cpu().reshape(-1)[0])
                    - float(expected_threshold)) <= 1e-6)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "B2-v3 packaged router threshold is invalid") from error
    if not threshold_matches:
        raise RuntimeError(
            "B2-v3 packaged router threshold is inconsistent")
    return package


def _clip_vector_norm(vector, radius, eps=1e-6):
    norm = torch.linalg.norm(vector, dim=-1, keepdim=True)
    scale = torch.minimum(
        torch.ones_like(norm), radius / torch.clamp(norm, min=eps))
    return vector * scale


class AsymmetricDualQueryAdapter(nn.Module):
    """Build a motion-guided search query without modifying q_obs.

    The residual head is exactly zero initialized.  Consequently a newly
    constructed adapter returns a bit-identical detached observation query,
    while B1 invalid rows continue to do so after training.
    """

    def __init__(
            self,
            observation_dim=64,
            motion_dim=128,
            hidden_dim=128,
            gate_max=0.5):
        super().__init__()
        self.observation_dim = int(observation_dim)
        self.motion_dim = int(motion_dim)
        self.gate_max = float(gate_max)
        if min(self.observation_dim, self.motion_dim, int(hidden_dim)) <= 0:
            raise ValueError("dual-query dimensions must be positive")
        if not 0.0 <= self.gate_max <= 1.0:
            raise ValueError("dual-query gate_max must be in [0,1]")
        self.residual = nn.Sequential(
            nn.LayerNorm(self.motion_dim + 4),
            nn.Linear(self.motion_dim + 4, int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), self.observation_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(4, int(hidden_dim) // 2),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim) // 2, 1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    @staticmethod
    def _column(value, reference, default=0.0):
        batch_size = reference.shape[0]
        if value is None:
            return reference.new_full((batch_size, 1), float(default))
        value = torch.as_tensor(
            value, device=reference.device,
            dtype=reference.dtype).reshape(-1)
        if value.numel() == 1:
            value = value.repeat(batch_size)
        if value.numel() != batch_size:
            raise ValueError("dual-query scalar must contain one value per row")
        return value.reshape(batch_size, 1)

    def forward(
            self, observation_query, motion_feature,
            log_sigma_parallel_perp, query_delta_t, gap_ratio,
            motion_valid):
        if (observation_query.dim() != 2
                or observation_query.shape[1] != self.observation_dim):
            raise ValueError("observation_query has the wrong shape")
        if (motion_feature.dim() != 2
                or motion_feature.shape[1] != self.motion_dim):
            raise ValueError("motion_feature has the wrong shape")
        observation = observation_query.detach()
        motion = motion_feature.detach()
        sigma = torch.nan_to_num(
            log_sigma_parallel_perp.detach().to(observation),
            nan=0.0, posinf=2.5, neginf=-4.0)
        if sigma.shape != (observation.shape[0], 2):
            raise ValueError("motion sigma must have shape [B,2]")
        dt = self._column(query_delta_t, observation, default=0.1)
        gap = self._column(gap_ratio, observation, default=1.0)
        scalar = torch.cat((sigma, dt, gap), dim=1)
        valid = (
            self._column(motion_valid, observation) > 0).to(
                observation.dtype)
        gate = (
            self.gate_max * torch.sigmoid(self.gate(scalar)) * valid)
        residual = self.residual(torch.cat((motion, scalar), dim=1))
        search_query = observation + gate * residual
        return search_query, {
            "dual_query_residual": residual,
            "dual_query_gate": gate.squeeze(1),
            "dual_query_delta_norm": torch.linalg.norm(
                search_query - observation, dim=1),
        }


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
            query_observation_dim=None,
            require_motion_valid=True,
            max_vote_offset=4.0,
            pool_temperature=0.5,
            presence_threshold=0.5,
            radius_base=0.5,
            radius_per_second=0.5,
            radius_max=2.0,
            eps=1e-6):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.point_dim = int(point_dim)
        self.query_dim = int(query_dim)
        self.query_observation_dim = int(
            observation_dim if query_observation_dim is None
            else query_observation_dim)
        self.require_motion_valid = bool(require_motion_valid)
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
        query_input_dim = (
            self.query_observation_dim + int(observation_stats_dim))
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

    def _filter_finite_points(
            self, valid, point_inputs, point_xy, delta_to_motion):
        """Compatibility hook: v2.2 keeps its original validity behavior."""
        return valid

    def _vote_point_xy(self, point_xy):
        return point_xy

    def _refinement_motion_xy(self, motion_proposal_xy):
        return motion_proposal_xy

    def _structural_finite(
            self, point_inputs, motion_proposal_xy, support_anchor_xy):
        """Compatibility hook: v2.2 keeps its original validity behavior."""
        return point_inputs.new_ones((point_inputs.shape[0],))

    def _compose_evidence(self, overlap_token, extension_token, context):
        return self.source_fusion(torch.cat((
            overlap_token, extension_token, context), dim=1))

    def _presence_inputs(
            self, overlap_token, extension_token, context, source_present):
        return torch.cat((
            overlap_token, extension_token, context, source_present), dim=1)

    def _candidate_valid(self, candidate_available, presence_probability):
        presence_valid = (
            presence_probability >= self.presence_threshold).to(
                candidate_available.dtype)
        return candidate_available * presence_valid

    @property
    def evidence_output_key(self):
        return "search_v22_evidence_token"

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
            overlap_count=None,
            query_feature=None):
        if (point_inputs.dim() != 3
                or point_inputs.shape[-1] != self.point_dim):
            raise ValueError(
                "B2 point inputs must have shape [B,N,point_dim]")
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
        valid = self._filter_finite_points(
            valid, point_inputs, point_xy, delta_to_motion)

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
        if query_feature is None:
            query_feature = observation_feature
        # An explicit query is produced by the trainable asymmetric adapter.
        # Its B0/B1 inputs are already detached inside that adapter, so
        # detaching again here would silently sever all B2 loss gradients to
        # the adapter itself.
        if query_feature.shape != (
                batch_size, self.query_observation_dim):
            raise ValueError("B2 query_feature has the wrong shape")
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
            query_feature, observation_stats), dim=1)))
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
        evidence_token = self._compose_evidence(
            overlap_token, extension_token, context)

        vote_offsets = self.max_vote_offset * torch.tanh(
            self.vote_head(point_feature))
        point_center_votes = self._vote_point_xy(point_xy) + vote_offsets
        raw_proposal_xy = (
            point_center_votes * pool_weights.unsqueeze(2)).sum(dim=1)

        valid_count = valid.sum(dim=1)
        point_row_valid = (valid_count >= 3).to(point_inputs.dtype)
        candidate_available = (
            point_row_valid * geometry.squeeze(1)
            * self._structural_finite(
                point_inputs, motion_proposal_xy, support_anchor_xy))
        if self.require_motion_valid:
            candidate_available = (
                candidate_available * motion_valid_column.squeeze(1))
        source_present = torch.stack((
            (overlap_mask.sum(dim=1) > 0).to(point_inputs.dtype),
            (extension_mask.sum(dim=1) > 0).to(point_inputs.dtype),
        ), dim=1)
        presence_logit = self.presence_head(self._presence_inputs(
            overlap_token, extension_token, context,
            source_present)).squeeze(1)
        presence_probability = torch.sigmoid(presence_logit)
        presence_probability = presence_probability * candidate_available
        candidate_valid = self._candidate_valid(
            candidate_available, presence_probability)

        query_dt = self._batch_scalar(
            query_delta_t, point_inputs, default=0.1)
        refinement_radius = torch.clamp(
            self.radius_base + self.radius_per_second * query_dt,
            max=self.radius_max)
        safe_motion_proposal_xy = self._refinement_motion_xy(
            motion_proposal_xy)
        refinement_residual = _clip_vector_norm(
            raw_proposal_xy - safe_motion_proposal_xy,
            refinement_radius,
            eps=self.eps,
        )
        refined_xy = safe_motion_proposal_xy + refinement_residual

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

        output = {
            "search_support_anchor_xy": support_anchor_xy.detach(),
            "search_raw_vote_xy": raw_proposal_xy,
            "motion_search_refined_xy": refined_xy,
            "motion_search_refinement_residual_xy": refinement_residual,
            "search_presence_logit": presence_logit,
            "search_presence_probability": presence_probability,
            "search_overlap_token": overlap_token * point_row,
            "search_extension_token": extension_token * point_row,
            "search_context_token": (
                context * candidate_available.unsqueeze(1)),
            "search_normalized_ess": normalized_ess,
            "search_raw_ess": raw_ess,
            "motion_search_candidate_available": candidate_available,
            "motion_search_candidate_valid": candidate_valid,
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
            "search_source_present": source_present,
        }
        output[self.evidence_output_key] = evidence_token
        return output


class StateAlignedSearchRefiner(MotionConditionedSearchRefiner):
    """B2-v3 search evidence aligned with B1's exact causal state.

    The migrated point/query/targetness/vote modules retain their v2.1 names
    and shapes.  Unlike v2.2, the router-facing evidence is the supervised
    structured concatenation itself; there is no randomly frozen fusion layer.
    """

    def __init__(self, *args, predict_utility=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.predict_utility = bool(predict_utility)
        del self.source_fusion
        self.presence_head = nn.Sequential(
            nn.Linear(3 * self.feature_dim + 2, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, 1),
        )
        nn.init.zeros_(self.presence_head[-1].weight)
        nn.init.constant_(
            self.presence_head[-1].bias, math.log(0.1 / 0.9))
        if self.predict_utility:
            self.utility_head = nn.Sequential(
                nn.Linear(3 * self.feature_dim + 2, self.feature_dim),
                nn.LayerNorm(self.feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.feature_dim, 1),
            )
            nn.init.zeros_(self.utility_head[-1].weight)
            nn.init.constant_(
                self.utility_head[-1].bias, math.log(0.1 / 0.9))

    @property
    def evidence_output_key(self):
        return "search_v3_evidence_components"

    def _filter_finite_points(
            self, valid, point_inputs, point_xy, delta_to_motion):
        finite = (
            torch.isfinite(point_inputs).all(dim=2)
            & torch.isfinite(point_xy).all(dim=2)
            & torch.isfinite(delta_to_motion).all(dim=2))
        return valid * finite.to(valid.dtype)

    def _vote_point_xy(self, point_xy):
        return torch.nan_to_num(
            point_xy, nan=0.0, posinf=0.0, neginf=0.0)

    def _refinement_motion_xy(self, motion_proposal_xy):
        return torch.nan_to_num(
            motion_proposal_xy, nan=0.0, posinf=0.0, neginf=0.0)

    def _structural_finite(
            self, point_inputs, motion_proposal_xy, support_anchor_xy):
        finite = (
            torch.isfinite(motion_proposal_xy).all(dim=1)
            & torch.isfinite(support_anchor_xy).all(dim=1))
        return finite.to(point_inputs.dtype)

    def _compose_evidence(self, overlap_token, extension_token, context):
        return torch.cat((overlap_token, extension_token, context), dim=1)

    def _candidate_valid(self, candidate_available, presence_probability):
        # Presence is a learned feature and loss target, never a hard gate.
        return candidate_available

    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        structural = output[
            "motion_search_candidate_valid"].unsqueeze(1)
        output.update({
            "search_v3_overlap_token": (
                output["search_overlap_token"] * structural),
            "search_v3_extension_token": (
                output["search_extension_token"] * structural),
            "search_v3_motion_observation_context":
                output["search_context_token"],
            "search_v3_presence_logit": output["search_presence_logit"],
            "search_v3_presence_probability":
                output["search_presence_probability"],
            "search_v3_raw_vote_xy": output["search_raw_vote_xy"],
            "motion_search_v3_refined_xy":
                output["motion_search_refined_xy"],
            "motion_search_v3_candidate_structural_valid":
                output["motion_search_candidate_valid"],
            "search_v3_match_logits": output["search_v22_match_logits"],
            "search_v3_targetness_logits":
                output["search_v22_targetness_logits"],
            "search_v3_targetness_mean":
                output["search_v22_targetness_mean"],
            "search_v3_targetness_max":
                output["search_v22_targetness_max"],
            "search_v3_targetness_entropy":
                output["search_v22_targetness_entropy"],
            "search_v3_extension_weight_ratio":
                output["search_v22_extension_weight_ratio"],
            "search_v3_point_center_votes":
                output["search_v22_point_center_votes"],
            "search_v3_normalized_ess": output["search_normalized_ess"],
            "search_v3_raw_ess": output["search_raw_ess"],
        })
        if self.predict_utility:
            utility_inputs = torch.cat((
                output["search_v3_overlap_token"],
                output["search_v3_extension_token"],
                output["search_v3_motion_observation_context"],
                output["search_source_present"],
            ), dim=1)
            utility_logit = self.utility_head(utility_inputs).squeeze(1)
            structural = output[
                "motion_search_v3_candidate_structural_valid"]
            output.update({
                "search_v3_utility_logit": utility_logit,
                "search_v3_utility_probability": (
                    torch.sigmoid(utility_logit) * structural),
            })
        return output


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
        # Candidate actions retain their raw observation-anchored residuals.
        # The single safety limiter is applied only after the exact source and
        # step action has been selected below.
        motion_residual = motion_proposal_xy - reference
        motion_search_residual = motion_search_xy - reference
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


class ActionConsistentInnovationRouter(SignedHorizonInnovationRouter):
    """Select and execute the same candidate/step action that q10 scores."""

    POLICY_OBSERVATION = -2
    POLICY_AUTO = -1
    POLICY_MOTION = 0
    POLICY_SEARCH = 1
    POLICY_REFINED = POLICY_SEARCH  # read-only alias for v3 artifacts

    def __init__(self, *args, search_dim=384, scalar_only=False,
                 use_utility_feature=False, **kwargs):
        super().__init__(*args, search_dim=search_dim, **kwargs)
        self.scalar_only = bool(scalar_only)
        self.use_utility_feature = bool(use_utility_feature)
        self.scalar_feature_names = list(self.SCALAR_FEATURE_NAMES)
        if self.use_utility_feature:
            self.scalar_feature_names.extend((
                "search_utility", "support_truncated"))
            old_input = self.trunk[0]
            self.scalar_dim = len(self.scalar_feature_names)
            self.trunk[0] = nn.Linear(
                old_input.in_features + 2, old_input.out_features)
            self.scalar_feature_mean = torch.zeros(self.scalar_dim)
            self.scalar_feature_std = torch.ones(self.scalar_dim)
            self.register_buffer(
                "scalar_clip_low",
                torch.full((self.scalar_dim,), float("-inf")))
            self.register_buffer(
                "scalar_clip_high",
                torch.full((self.scalar_dim,), float("inf")))
        hidden_dim = self.median_gain_head.in_features
        action_count = 2 * len(self.STEP_RATIOS)
        self.median_gain_head = nn.Linear(hidden_dim, action_count)
        self.gain_spread_head = nn.Linear(hidden_dim, action_count)
        del self.step_head
        nn.init.zeros_(self.median_gain_head.weight)
        nn.init.zeros_(self.median_gain_head.bias)
        nn.init.zeros_(self.gain_spread_head.weight)
        nn.init.constant_(
            self.gain_spread_head.bias, math.log(math.expm1(0.05)))

    @property
    def action_names(self):
        return [
            f"{source}@{ratio:g}"
            for source in ("MOTION", "SEARCH")
            for ratio in self.STEP_RATIOS
        ]

    @property
    def feature_schema(self):
        return {
            "observation_dim": self.observation_dim,
            "motion_dim": self.motion_dim,
            "search_dim": self.search_dim,
            "scalar_feature_names": list(self.scalar_feature_names),
            "scalar_dim": self.scalar_dim,
            "feature_dim": self.export_feature_dim,
            "scalar_only": self.scalar_only,
            "use_utility_feature": self.use_utility_feature,
            "action_names": self.action_names,
            "step_ratios": list(self.STEP_RATIOS),
        }

    @property
    def feature_schema_hash(self):
        encoded = json.dumps(
            self.feature_schema, sort_keys=True,
            separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def set_scalar_clipping(self, low, high):
        if not hasattr(self, "scalar_clip_low"):
            raise RuntimeError(
                "scalar clipping belongs to the formal v4 router schema")
        low = torch.as_tensor(
            low, device=self.scalar_clip_low.device,
            dtype=self.scalar_clip_low.dtype).reshape(-1)
        high = torch.as_tensor(
            high, device=self.scalar_clip_high.device,
            dtype=self.scalar_clip_high.dtype).reshape(-1)
        if (low.numel() != self.scalar_dim
                or high.numel() != self.scalar_dim
                or not bool(torch.isfinite(low).all())
                or not bool(torch.isfinite(high).all())
                or bool(torch.any(low > high))):
            raise ValueError("router p1/p99 clipping contract is invalid")
        self.scalar_clip_low.copy_(low)
        self.scalar_clip_high.copy_(high)

    def _predict(self, observation_feature, motion_feature, search_feature,
                 scalar_features):
        if self.scalar_only:
            observation_feature = torch.zeros_like(observation_feature)
            motion_feature = torch.zeros_like(motion_feature)
            search_feature = torch.zeros_like(search_feature)
        if hasattr(self, "scalar_clip_low"):
            scalar_features = torch.maximum(
                scalar_features, self.scalar_clip_low.unsqueeze(0))
            scalar_features = torch.minimum(
                scalar_features, self.scalar_clip_high.unsqueeze(0))
        normalized_scalar = (
            scalar_features - self.scalar_feature_mean.unsqueeze(0)
        ) / torch.clamp(self.scalar_feature_std.unsqueeze(0), min=1e-4)
        hidden = self.trunk(torch.cat((
            self.observation_projection(observation_feature),
            self.motion_projection(motion_feature),
            self.search_projection(search_feature),
            torch.nan_to_num(normalized_scalar),
        ), dim=1))
        q50 = self.median_gain_head(hidden).reshape(-1, 2, 3)
        q10 = q50 - F.softplus(
            self.gain_spread_head(hidden)).reshape(-1, 2, 3)
        return q10, q50

    def predict_export_features(self, exported_features):
        if exported_features.dim() != 2:
            raise ValueError("router features must have shape [B,D]")
        if exported_features.shape[1] != self.export_feature_dim:
            raise ValueError("router feature width mismatch")
        obs_end = self.observation_dim
        motion_end = obs_end + self.motion_dim
        search_end = motion_end + self.search_dim
        q10, q50 = self._predict(
            exported_features[:, :obs_end],
            exported_features[:, obs_end:motion_end],
            exported_features[:, motion_end:search_end],
            exported_features[:, search_end:],
        )
        return {"q10": q10, "q50": q50}

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
            policy_override=None,
            forced_step_ratio=None,
            action_allowed_mask=None,
            search_utility=None,
            support_truncated=None):
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
        search_utility = detached(search_utility)
        support_truncated = detached(support_truncated)
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
        # Candidate semantics stay raw.  Execution applies the chosen ratio
        # first and the normal/gap safety cap exactly once below.
        motion_residual = motion_proposal_xy - reference
        motion_search_residual = motion_search_xy - reference
        motion_valid_column = (
            self._batch_scalar(motion_valid, reference) > 0)
        motion_search_valid_column = (
            self._batch_scalar(motion_search_valid, reference) > 0)
        safe_radius = torch.clamp(radius, min=self.eps)
        motion_norm = torch.linalg.norm(
            motion_residual, dim=1, keepdim=True)
        motion_search_norm = torch.linalg.norm(
            motion_search_residual, dim=1, keepdim=True)
        candidate_distance = torch.linalg.norm(
            motion_proposal_xy - motion_search_xy, dim=1, keepdim=True)
        cosine = (
            motion_residual * motion_search_residual).sum(
                dim=1, keepdim=True) / torch.clamp(
                    motion_norm * motion_search_norm, min=self.eps)
        refinement = torch.nan_to_num(
            observation_refinement_xy) / safe_radius
        scalar_columns = [
            observation_stats,
            self._batch_scalar(observation_entropy, reference),
            refinement,
            torch.linalg.norm(refinement, dim=1, keepdim=True),
            torch.nan_to_num(motion_log_sigma_xy),
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
        ]
        if self.use_utility_feature:
            scalar_columns.extend((
                self._batch_scalar(search_utility, reference),
                self._batch_scalar(support_truncated, reference),
            ))
        scalar_features = torch.cat(scalar_columns, dim=1)
        if scalar_features.shape[1] != self.scalar_dim:
            raise RuntimeError(
                "action router scalar feature contract changed: "
                f"{scalar_features.shape[1]} != {self.scalar_dim}")
        exported_features = torch.cat((
            observation_feature,
            motion_feature,
            search_feature,
            torch.nan_to_num(scalar_features),
        ), dim=1)
        q10, q50 = self._predict(
            observation_feature, motion_feature, search_feature,
            scalar_features)

        candidate_valid = torch.cat((
            motion_valid_column, motion_search_valid_column), dim=1)
        if action_allowed_mask is None:
            action_allowed = torch.ones_like(candidate_valid)
        else:
            action_allowed = action_allowed_mask.to(
                device=reference.device, dtype=reference.dtype)
            if action_allowed.shape != candidate_valid.shape:
                raise ValueError(
                    "action_allowed_mask must have shape [B,2]")
            action_allowed = action_allowed > 0
        selectable_candidate = candidate_valid & action_allowed
        action_valid = selectable_candidate.unsqueeze(2).expand(-1, -1, 3)
        masked_q10 = q10.masked_fill(
            ~action_valid, torch.finfo(q10.dtype).min)
        flat_q10 = masked_q10.reshape(batch_size, -1)
        best_q10, flat_action = flat_q10.max(dim=1)
        selected_index = torch.div(flat_action, 3, rounding_mode='floor')
        selected_step_class = flat_action.remainder(3)
        any_valid = action_valid.reshape(batch_size, -1).any(dim=1)
        step_ratios = self.step_ratio_values.to(
            device=reference.device, dtype=reference.dtype)
        selected_step_ratio = step_ratios[selected_step_class]
        threshold = self.calibrated_gain_threshold.to(
            device=reference.device, dtype=reference.dtype)
        intervene = any_valid & (best_q10 > threshold)

        if policy_override is None:
            policy = reference.new_full(
                (batch_size,), self.POLICY_AUTO, dtype=torch.long)
        else:
            policy = self._batch_scalar(
                policy_override, reference,
                default=self.POLICY_AUTO).reshape(-1).to(torch.long)
        allowed_policy = (
            (policy == self.POLICY_OBSERVATION)
            | (policy == self.POLICY_AUTO)
            | (policy == self.POLICY_MOTION)
            | (policy == self.POLICY_SEARCH))
        if not bool(torch.all(allowed_policy).item()):
            raise ValueError("invalid B2-v3 policy override")
        forced = policy >= 0
        observation_only = policy == self.POLICY_OBSERVATION
        if bool(torch.any(forced).item()):
            if forced_step_ratio is None:
                raise ValueError(
                    "forced motion/search policy requires an explicit step")
            forced_ratio = self._batch_scalar(
                forced_step_ratio, reference, default=0.25).reshape(-1)
            distance = torch.abs(
                forced_ratio.unsqueeze(1)
                - step_ratios.unsqueeze(0))
            forced_step = distance.argmin(dim=1)
            if bool(torch.any(
                    distance.min(dim=1).values[forced] > 1e-6).item()):
                raise ValueError("forced step must be 0.25, 0.5, or 1.0")
            forced_candidate = torch.clamp(policy, 0, 1)
            forced_valid = selectable_candidate.gather(
                1, forced_candidate.unsqueeze(1)).squeeze(1)
            selected_index = torch.where(
                forced, forced_candidate, selected_index)
            selected_step_class = torch.where(
                forced, forced_step, selected_step_class)
            selected_step_ratio = step_ratios[selected_step_class]
            intervene = torch.where(forced, forced_valid, intervene)
        intervene = intervene & ~observation_only

        enabled_scale = float(enabled_scale)
        if not 0.0 <= enabled_scale <= 1.0:
            raise ValueError("action router enabled_scale must be in [0,1]")
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
        requested_correction = (
            selected_residual * selected_step_ratio.unsqueeze(1))
        correction = _clip_vector_norm(
            requested_correction, step_cap.unsqueeze(1), self.eps)
        correction = (
            correction * intervene.to(reference.dtype).unsqueeze(1)
            * enabled_scale)
        residual_norm = torch.linalg.norm(
            selected_residual, dim=1).clamp_min(self.eps)
        alpha = torch.linalg.norm(correction, dim=1) / residual_norm
        final_xy = torch.where(
            intervene.unsqueeze(1), reference + correction, reference)
        final_box = torch.cat((final_xy, observation_box[:, 2:]), dim=1)
        selected_candidate = torch.where(
            intervene, selected_index + 1, torch.zeros_like(selected_index))

        return final_box, {
            "router_v3_features": exported_features,
            "router_v3_gain_q10": q10,
            "router_v3_gain_q50": q50,
            "router_v3_action_valid": action_valid.to(reference.dtype),
            "router_v3_candidate_valid": candidate_valid.to(reference.dtype),
            "router_v3_action_allowed": action_allowed.to(reference.dtype),
            "router_v3_candidate_residual_xy": candidates,
            "router_v3_selected_candidate": selected_candidate,
            "router_v3_selected_candidate_index": selected_index,
            "router_v3_selected_step_index": selected_step_class,
            "router_v3_selected_step_ratio": selected_step_ratio,
            "router_v3_policy_override": policy,
            "router_v3_step_cap": step_cap,
            "router_v3_gain_threshold": threshold,
            "router_v3_abstained": (~intervene).to(reference.dtype),
            "router_v3_applied_alpha": alpha,
            "router_v3_correction_xy": correction,
            "router_v3_fusion_radius": radius.squeeze(1),
            "router_v3_scalar_only": reference.new_tensor(
                float(self.scalar_only)),
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


def action_consistent_router_loss(
        prediction,
        signed_gain,
        candidate_valid,
        q10_weight=1.0,
        q50_weight=0.5):
    """Supervise each of the six executable actions with its own gain."""
    if signed_gain.dim() != 3 or signed_gain.shape[1:] != (2, 3):
        raise ValueError("signed_gain must have shape [B,2,3]")
    q10 = prediction["q10"]
    q50 = prediction["q50"]
    if q10.shape != signed_gain.shape or q50.shape != signed_gain.shape:
        raise ValueError("router quantiles must match all six action gains")
    valid = candidate_valid.to(signed_gain.dtype)
    if valid.shape != signed_gain.shape[:2]:
        raise ValueError("candidate_valid must have shape [B,2]")
    action_valid = valid.unsqueeze(2).expand_as(signed_gain)
    denominator = torch.clamp(action_valid.sum(), min=1.0)
    loss_q10 = (
        pinball_loss(q10, signed_gain, 0.10) * action_valid
    ).sum() / denominator
    loss_q50 = (
        pinball_loss(q50, signed_gain, 0.50) * action_valid
    ).sum() / denominator
    total = float(q10_weight) * loss_q10 + float(q50_weight) * loss_q50
    return {
        "loss": total,
        "loss_q10": loss_q10,
        "loss_q50": loss_q50,
        "action_valid": action_valid,
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
        max_harm_rate=0.05,
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


def calibrate_action_threshold(
        q10,
        signed_gain,
        candidate_valid,
        min_precision=0.75,
        max_harm_rate=0.10,
        min_coverage=0.05,
        max_coverage=0.25,
        helpful_margin=0.02,
        min_selected_count=100):
    """Calibrate against the exact action selected and later executed."""
    q10 = np.asarray(q10, dtype=np.float64)
    gains = np.asarray(signed_gain, dtype=np.float64)
    valid = np.asarray(candidate_valid, dtype=bool)
    if q10.ndim != 3 or q10.shape[1:] != (2, 3):
        raise ValueError("q10 must have shape [N,2,3]")
    if gains.shape != q10.shape:
        raise ValueError("signed_gain must match q10")
    if valid.shape != q10.shape[:2]:
        raise ValueError("candidate_valid must have shape [N,2]")
    action_valid = np.repeat(valid[:, :, None], 3, axis=2)
    masked = np.where(action_valid, q10, -np.inf).reshape(q10.shape[0], -1)
    selected_action = np.argmax(masked, axis=1)
    scores = masked[np.arange(masked.shape[0]), selected_action]
    flat_gain = gains.reshape(gains.shape[0], -1)
    selected_gain = flat_gain[np.arange(flat_gain.shape[0]), selected_action]
    finite_scores = scores[np.isfinite(scores)]
    if finite_scores.size == 0:
        raise RuntimeError("calibration contains no valid actions")
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
        if (count >= int(min_selected_count)
                and float(min_coverage) <= coverage <= float(max_coverage)
                and precision >= float(min_precision)
                and harm_rate <= float(max_harm_rate)):
            candidates.append((
                coverage, precision, -harm_rate, float(threshold), count))
    if not candidates:
        raise RuntimeError(
            "no action threshold satisfies count/precision/harm/coverage "
            "guardrails")
    coverage, precision, neg_harm, threshold, count = max(candidates)
    return {
        "threshold": threshold,
        "coverage": coverage,
        "helpful_precision": precision,
        "harm_rate": -neg_harm,
        "selected_count": count,
    }
