"""Deployment-aligned B2 evidence acquisition and memory conditioning.

Contract v3 deliberately keeps B0 point tokens as context only.  Extension
tokens are the sole queries and the sole source of recovery votes.
"""

import math

import torch
from torch import nn


def _masked_mean(features, mask):
    weights = mask.to(features.dtype).unsqueeze(-1)
    return (features * weights).sum(dim=1) / torch.clamp(weights.sum(dim=1), min=1.0)


def _masked_max(features, mask):
    values = features.masked_fill(~mask.unsqueeze(-1), float("-inf"))
    result = values.max(dim=1).values
    return torch.where(mask.any(dim=1, keepdim=True), result, torch.zeros_like(result))


def _fps_indices(xyz, count):
    """Deterministic farthest-point indices for one finite point set."""
    if xyz.numel() == 0 or count <= 0:
        return torch.empty(0, dtype=torch.long, device=xyz.device)
    count = min(int(count), int(xyz.shape[0]))
    centroid = xyz.mean(dim=0, keepdim=True)
    distance_to_centroid = torch.sum((xyz - centroid).pow(2), dim=1)
    first = int(torch.argmax(distance_to_centroid).item())
    selected = [first]
    min_distance = torch.sum((xyz - xyz[first]).pow(2), dim=1)
    for _ in range(1, count):
        index = int(torch.argmax(min_distance).item())
        selected.append(index)
        candidate_distance = torch.sum((xyz - xyz[index]).pow(2), dim=1)
        min_distance = torch.minimum(min_distance, candidate_distance)
    return torch.as_tensor(selected, dtype=torch.long, device=xyz.device)


def build_box_memory_tokens(
    history_features,
    history_points,
    history_boxes,
    box_size,
    history_valid_mask,
    foreground_tokens=8,
    context_tokens=4,
    context_scale=2.0,
    history_timestamps=None,
    current_timestamp=None,
    current_box=None,
    return_metadata=False,
):
    """Select causal foreground/context tokens from predicted history boxes.

    No segmentation labels, current GT, or future frames are consumed.  Each
    history frame contributes a fixed padded block, which prevents one dense
    frame from dominating memory.
    """
    if history_features.dim() != 4 or history_features.shape[-1] != 64:
        raise ValueError("history features must have shape [B,H,N,64]")
    if history_points.shape[:3] != history_features.shape[:3]:
        raise ValueError("history points/features must be point aligned")
    batch_size, history_count, _, feature_dim = history_features.shape
    per_frame = int(foreground_tokens) + int(context_tokens)
    tokens = history_features.new_zeros(
        (batch_size, history_count * per_frame, feature_dim)
    )
    valid = torch.zeros(
        (batch_size, history_count * per_frame),
        dtype=torch.bool,
        device=history_features.device,
    )
    # local xyz, physical age, relative yaw sin/cos, inside/context role and
    # history-frame identity.  Metadata is geometric and never learned from
    # labels.
    metadata = history_features.new_zeros((batch_size, history_count * per_frame, 8))
    points_xyz = history_points[..., :3].detach()
    features = history_features.detach()
    boxes = history_boxes.detach()
    sizes = box_size.detach()
    history_valid = history_valid_mask.reshape(batch_size, history_count).to(torch.bool)
    if history_timestamps is None:
        history_timestamps = history_features.new_zeros((batch_size, history_count))
    else:
        history_timestamps = torch.as_tensor(
            history_timestamps,
            device=history_features.device,
            dtype=history_features.dtype,
        ).reshape(batch_size, history_count)
    if current_timestamp is None:
        current_timestamp = history_timestamps.max(dim=1).values
    else:
        current_timestamp = torch.as_tensor(
            current_timestamp,
            device=history_features.device,
            dtype=history_features.dtype,
        ).reshape(batch_size)
    if current_box is None:
        current_box = history_features.new_zeros((batch_size, 4))
    else:
        current_box = torch.as_tensor(
            current_box, device=history_features.device, dtype=history_features.dtype
        ).reshape(batch_size, 4)

    for batch_index in range(batch_size):
        size_xy = torch.clamp(sizes[batch_index, :2], min=1e-3)
        for history_index in range(history_count):
            if not bool(history_valid[batch_index, history_index]):
                continue
            xyz = points_xyz[batch_index, history_index]
            finite = torch.isfinite(xyz).all(dim=1)
            box = boxes[batch_index, history_index]
            delta = xyz[:, :2] - box[:2]
            yaw = box[3]
            cosine, sine = torch.cos(yaw), torch.sin(yaw)
            local_x = cosine * delta[:, 0] + sine * delta[:, 1]
            local_y = -sine * delta[:, 0] + cosine * delta[:, 1]
            normalized_signed = torch.stack(
                (
                    2.0 * local_x / size_xy[0],
                    2.0 * local_y / size_xy[1],
                    2.0
                    * (xyz[:, 2] - box[2])
                    / torch.clamp(sizes[batch_index, 2], min=1e-3),
                ),
                dim=1,
            )
            normalized = normalized_signed.abs()
            inside = finite & (normalized <= 1.0).all(dim=1)
            neighborhood = (
                finite & (normalized <= float(context_scale)).all(dim=1) & ~inside
            )

            foreground_rows = torch.nonzero(inside, as_tuple=False).flatten()
            context_rows = torch.nonzero(neighborhood, as_tuple=False).flatten()
            foreground_pick = (
                foreground_rows.index_select(
                    0,
                    _fps_indices(
                        xyz.index_select(0, foreground_rows), foreground_tokens
                    ),
                )
                if foreground_rows.numel()
                else (foreground_rows)
            )
            context_pick = (
                context_rows.index_select(
                    0, _fps_indices(xyz.index_select(0, context_rows), context_tokens)
                )
                if context_rows.numel()
                else context_rows
            )
            block_start = history_index * per_frame
            for offset, rows in (
                (0, foreground_pick),
                (int(foreground_tokens), context_pick),
            ):
                take = min(
                    int(rows.numel()),
                    int(foreground_tokens if offset == 0 else context_tokens),
                )
                if take == 0:
                    continue
                output_rows = slice(block_start + offset, block_start + offset + take)
                tokens[batch_index, output_rows] = features[
                    batch_index, history_index
                ].index_select(0, rows[:take])
                valid[batch_index, output_rows] = True
                selected = rows[:take]
                physical_age = torch.clamp(
                    current_timestamp[batch_index]
                    - history_timestamps[batch_index, history_index],
                    min=0.0,
                )
                relative_yaw = current_box[batch_index, 3] - box[3]
                metadata[batch_index, output_rows, :3] = normalized_signed.index_select(
                    0, selected
                )
                metadata[batch_index, output_rows, 3] = torch.log1p(physical_age)
                metadata[batch_index, output_rows, 4] = torch.sin(relative_yaw)
                metadata[batch_index, output_rows, 5] = torch.cos(relative_yaw)
                metadata[batch_index, output_rows, 6] = float(offset == 0)
                metadata[batch_index, output_rows, 7] = float(history_index + 1) / max(
                    float(history_count), 1.0
                )
    if return_metadata:
        return tokens, valid, metadata
    return tokens, valid


def apply_memory_control(tokens, valid, metadata, mode):
    """Apply the pre-registered memory ablation without channel shuffling."""
    mode = str(mode).strip().lower()
    if mode in ("none", "empty"):
        return (
            torch.zeros_like(tokens),
            torch.zeros_like(valid),
            torch.zeros_like(metadata),
        )
    if mode == "real":
        return tokens, valid, metadata
    if mode == "time_misaligned":
        if tokens.shape[1] % 3:
            raise ValueError("time-misaligned control requires three blocks")
        block_size = tokens.shape[1] // 3
        return tokens, valid, torch.roll(metadata, shifts=block_size, dims=1)
    raise ValueError("memory mode must be none, empty, real or time_misaligned")


def extension_target_bearing_mask(
    structural_available, extension_labels, extension_valid_mask
):
    """Rows permitted to supervise an extension-derived raw candidate."""
    presence = ((extension_labels * extension_valid_mask).sum(dim=1) > 0).to(
        structural_available.dtype
    )
    return structural_available.reshape(-1) * presence


class B2EvidenceAcquirer(nn.Module):
    """B2 extension-only evidence acquirer.

    This module owns targetness, voting, presence and the raw candidate.  It
    deliberately has no utility or action head; deciding whether applying the
    candidate is safe belongs exclusively to B3.
    """

    def __init__(
        self,
        feature_dim=64,
        num_heads=4,
        max_vote_offset=4.0,
        attention_dropout=0.0,
        presence_init_probability=0.1,
        presence_threshold=0.5,
        utility_init_probability=None,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_heads = int(num_heads)
        self.max_vote_offset = float(max_vote_offset)
        self.presence_threshold = float(presence_threshold)
        if self.feature_dim != 64:
            raise ValueError("contract-v3 B0 point features are fixed at 64d")
        if self.feature_dim % self.num_heads:
            raise ValueError("attention heads must divide the feature dimension")
        if float(attention_dropout) != 0.0:
            raise ValueError("contract-v3 attention dropout must remain zero")

        def mlp(input_dim, output_dim=64):
            return nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Linear(64, output_dim),
            )

        self.extension_encoder = mlp(5)
        # longitudinal, lateral, log(dt), gap ratio, B1 validity
        self.geometry_encoder = mlp(5)
        self.source_embedding = nn.Embedding(4, 64)
        self.memory_metadata_encoder = mlp(8)
        self.query_norm = nn.LayerNorm(64)
        self.kv_norm = nn.LayerNorm(64)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=64, num_heads=4, dropout=0.0, batch_first=True
        )
        self.attention_residual_gate = nn.Parameter(torch.zeros(64))
        self.output_norm = nn.LayerNorm(64)

        self.targetness_head = nn.Sequential(
            nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1)
        )
        self.vote_head = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 2))
        self.base_presence_head = nn.Sequential(
            nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1)
        )
        self.extension_presence_head = nn.Sequential(
            nn.Linear(130, 64), nn.GELU(), nn.Linear(64, 1)
        )
        for head, probability in (
            (self.base_presence_head[-1], presence_init_probability),
            (self.extension_presence_head[-1], presence_init_probability),
        ):
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, math.log(probability / (1.0 - probability)))

    def forward(
        self,
        extension_points,
        extension_valid_mask,
        extension_source,
        current_base_features,
        current_base_valid_mask,
        memory_tokens,
        memory_valid_mask,
        observation_box,
        observation_stats,
        b1_center_xy,
        b1_sigma_parallel_perp,
        b1_direction_xy,
        b1_valid,
        query_delta_t,
        gap_ratio,
        recursive_age=None,
        memory_metadata=None,
    ):
        batch_size, extension_count, _ = extension_points.shape
        if current_base_features.shape != (batch_size, 1024, self.feature_dim):
            raise ValueError("current base features must be [B,1024,64]")
        if memory_tokens.shape[1:] != (36, self.feature_dim):
            raise ValueError("history memory must be [B,36,64]")
        extension_mask = extension_valid_mask.reshape(batch_size, extension_count).to(
            torch.bool
        )
        extension_mask &= torch.isfinite(extension_points).all(dim=2)
        base_mask = current_base_valid_mask.reshape(batch_size, 1024).to(torch.bool)
        base_mask &= torch.isfinite(current_base_features).all(dim=2)
        memory_mask = memory_valid_mask.reshape(batch_size, 36).to(torch.bool)
        if memory_metadata is None:
            memory_metadata = memory_tokens.new_zeros((batch_size, 36, 8))
        if memory_metadata.shape != (batch_size, 36, 8):
            raise ValueError("history memory metadata must be [B,36,8]")

        safe_points = torch.nan_to_num(extension_points)
        direction = torch.nan_to_num(b1_direction_xy.detach())
        direction_norm = torch.linalg.norm(direction, dim=1, keepdim=True)
        default_direction = torch.zeros_like(direction)
        default_direction[:, 0] = 1.0
        direction = torch.where(
            direction_norm > 1e-6,
            direction / torch.clamp(direction_norm, min=1e-6),
            default_direction,
        )
        perpendicular = torch.stack((-direction[:, 1], direction[:, 0]), 1)
        delta = safe_points[..., :2] - b1_center_xy.detach().unsqueeze(1)
        sigma = torch.clamp(
            torch.nan_to_num(b1_sigma_parallel_perp.detach(), nan=1.0), min=0.1
        )
        longitudinal = (delta * direction.unsqueeze(1)).sum(2) / sigma[:, 0:1]
        lateral = (delta * perpendicular.unsqueeze(1)).sum(2) / sigma[:, 1:2]
        dt = torch.clamp(query_delta_t.reshape(batch_size), min=0.0)
        gap = torch.clamp(gap_ratio.reshape(batch_size), min=0.0)
        b1_valid_f = b1_valid.reshape(batch_size).to(safe_points.dtype)
        geometry = torch.stack(
            (
                longitudinal,
                lateral,
                torch.log1p(dt).unsqueeze(1).expand(-1, extension_count),
                gap.unsqueeze(1).expand(-1, extension_count),
                b1_valid_f.unsqueeze(1).expand(-1, extension_count),
            ),
            dim=2,
        )
        source = (
            extension_source.reshape(batch_size, extension_count).long().clamp(0, 3)
        )
        extension_feature = (
            self.extension_encoder(safe_points)
            + self.geometry_encoder(geometry)
            + self.source_embedding(source)
        )
        extension_feature = extension_feature * extension_mask.unsqueeze(2)

        base_features = current_base_features.detach()
        memory_features = memory_tokens.detach() + self.memory_metadata_encoder(
            memory_metadata.detach()
        )
        memory_features = memory_features * memory_mask.unsqueeze(2)
        kv = torch.cat((base_features, memory_features), dim=1)
        kv_valid = torch.cat((base_mask, memory_mask), dim=1)
        no_context = ~kv_valid.any(dim=1)
        if bool(no_context.any()):
            kv = kv.clone()
            kv_valid = kv_valid.clone()
            kv[no_context, 0] = 0.0
            kv_valid[no_context, 0] = True
        attention_output, attention_weights = self.cross_attention(
            self.query_norm(extension_feature),
            self.kv_norm(kv),
            self.kv_norm(kv),
            key_padding_mask=~kv_valid,
            need_weights=True,
            average_attn_weights=False,
        )
        enriched = self.output_norm(
            extension_feature
            + attention_output * self.attention_residual_gate.view(1, 1, -1)
        )
        enriched = enriched * extension_mask.unsqueeze(2)

        logits = self.targetness_head(enriched).squeeze(2)
        logits = logits.masked_fill(~extension_mask, -20.0)
        scores = torch.sigmoid(logits) * extension_mask.to(logits.dtype)
        votes = safe_points[..., :2] + self.max_vote_offset * torch.tanh(
            self.vote_head(enriched)
        )
        weight_sum = scores.sum(dim=1, keepdim=True)
        raw_xy = (scores.unsqueeze(2) * votes).sum(dim=1) / torch.clamp(
            weight_sum, min=1e-6
        )
        extension_point_count = extension_mask.sum(dim=1)
        availability = (
            (b1_valid_f > 0)
            & (extension_point_count > 0)
            & torch.isfinite(raw_xy).all(dim=1)
        )
        observation_xy = observation_box[:, :2].detach()
        raw_xy = torch.where(availability.unsqueeze(1), raw_xy, observation_xy)
        raw_box = torch.cat((raw_xy, observation_box[:, 2:].detach()), dim=1)

        base_mean = _masked_mean(base_features, base_mask)
        base_max = _masked_max(base_features, base_mask)
        extension_mean = _masked_mean(enriched, extension_mask)
        extension_max = _masked_max(enriched, extension_mask)
        base_presence_logit = self.base_presence_head(
            torch.cat((base_mean, base_max), dim=1)
        ).squeeze(1)
        extension_presence_logit = self.extension_presence_head(
            torch.cat(
                (
                    extension_mean,
                    extension_max,
                    torch.log1p(extension_point_count.to(enriched.dtype)).unsqueeze(1),
                    b1_valid_f.unsqueeze(1),
                ),
                dim=1,
            )
        ).squeeze(1)
        base_presence_probability = torch.sigmoid(base_presence_logit)
        extension_presence_probability = torch.sigmoid(
            extension_presence_logit
        ) * availability.to(enriched.dtype)
        evidence_present = extension_presence_probability >= self.presence_threshold
        candidate_valid = availability & evidence_present

        probability = scores / torch.clamp(weight_sum, min=1e-6)
        entropy = -(probability * torch.log(torch.clamp(probability, min=1e-8))).sum(
            dim=1
        )
        targetness_mean = scores.sum(dim=1) / torch.clamp(
            extension_point_count.to(scores.dtype), min=1.0
        )
        targetness_max = scores.max(dim=1).values
        normalized_ess = 1.0 / torch.clamp(
            probability.pow(2).sum(dim=1)
            * torch.clamp(extension_point_count.to(scores.dtype), min=1.0),
            min=1e-6,
        )
        return {
            "b0_current_base_features_detached": base_features,
            "ct_memory_tokens": memory_tokens,
            "ct_memory_valid_mask": memory_mask,
            "ct_memory_metadata": memory_metadata,
            "ct_extension_query_features": enriched,
            "ct_cross_attention_weights": attention_weights,
            "ct_b2_available": availability.to(enriched.dtype),
            "ct_b2_base_presence_logit": base_presence_logit,
            "ct_b2_base_presence_probability": base_presence_probability,
            "ct_b2_base_evidence": base_mean,
            "ct_b2_extension_presence_logit": extension_presence_logit,
            "ct_b2_extension_presence_probability": extension_presence_probability,
            "ct_b2_extension_evidence": extension_mean,
            "ct_b2_raw_box": raw_box,
            "ct_b2_no_extension_box": observation_box.detach(),
            "ct_search_targetness_logits": logits,
            "ct_search_point_votes": votes,
            "ct_search_unmasked_raw_xy": raw_xy,
            "ct_search_raw_xy": raw_xy,
            "ct_search_candidate_valid": candidate_valid.to(enriched.dtype),
            "ct_search_structural_valid": availability.to(enriched.dtype),
            "ct_search_new_support_valid": (extension_point_count > 0).to(
                enriched.dtype
            ),
            "ct_search_support_valid": availability.to(enriched.dtype),
            "ct_search_available": availability.to(enriched.dtype),
            "ct_search_effective": availability.to(enriched.dtype),
            "ct_search_presence_logit": extension_presence_logit,
            "ct_search_presence_probability": extension_presence_probability,
            "ct_search_targetness_mean": targetness_mean,
            "ct_search_targetness_max": targetness_max,
            "ct_search_targetness_entropy": entropy,
            "ct_search_normalized_ess": normalized_ess,
            "ct_search_extension_selected_count": extension_point_count.to(
                enriched.dtype
            ),
        }


class B3SelectiveUpdater(nn.Module):
    """B3 action-level reliability model with exact observation fallback.

    Helpful and harmful risks are predicted separately.  A correction is
    deployable only when structural evidence, calibrated presence, calibrated
    action reliability and the physical residual bound all pass.  B2 inputs
    are detached here as a second line of defence in addition to isolated
    optimizers.
    """

    def __init__(
        self,
        observation_stats_dim=5,
        hidden_dim=64,
        presence_threshold=0.5,
        decision_threshold=0.5,
        radius_base=0.5,
        radius_per_second=0.5,
        radius_max=2.0,
        require_calibration=False,
        helpful_init_probability=0.05,
        harmful_init_probability=0.5,
    ):
        super().__init__()
        self.require_calibration = bool(require_calibration)
        self.radius_base = float(radius_base)
        self.radius_per_second = float(radius_per_second)
        self.radius_max = float(radius_max)
        self.register_buffer(
            "presence_threshold",
            torch.tensor(float(presence_threshold)),
            persistent=False,
        )
        self.register_buffer(
            "decision_threshold",
            torch.tensor(float(decision_threshold)),
            persistent=False,
        )
        self.register_buffer(
            "calibrated",
            torch.tensor(not self.require_calibration, dtype=torch.bool),
            persistent=False,
        )
        self.evidence_projection = nn.Sequential(nn.Linear(128, 32), nn.GELU())
        # evidence(32), presence(2), coarse/refined agreement(2),
        # observation/prior disagreement(3), observation/evidence
        # disagreement(3), uncertainty(2), dt/gap/age(3), observation
        # statistics(5), point evidence(6).
        input_dim = 32 + 2 + 2 + 3 + 3 + 2 + 3 + int(observation_stats_dim) + 6
        self.risk_trunk = nn.Sequential(
            nn.Linear(input_dim, int(hidden_dim)),
            nn.GELU(),
        )
        self.helpful_head = nn.Linear(int(hidden_dim), 1)
        self.harmful_head = nn.Linear(int(hidden_dim), 1)
        self.expected_center_gain_head = nn.Linear(int(hidden_dim), 1)
        self.expected_iou_gain_head = nn.Linear(int(hidden_dim), 1)
        for head, probability in (
            (self.helpful_head, helpful_init_probability),
            (self.harmful_head, harmful_init_probability),
        ):
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, math.log(probability / (1.0 - probability)))
        for head in (self.expected_center_gain_head, self.expected_iou_gain_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    @torch.no_grad()
    def install_calibration(self, presence_threshold, action_threshold):
        """Install thresholds from a validated external artifact."""
        values = torch.as_tensor(
            [presence_threshold, action_threshold], dtype=torch.float64
        )
        if not bool(torch.isfinite(values).all()):
            raise ValueError("calibration thresholds must be finite")
        if not bool(((values >= 0.0) & (values <= 1.0)).all()):
            raise ValueError("calibration thresholds must be in [0, 1]")
        self.presence_threshold.copy_(values[0].to(self.presence_threshold))
        self.decision_threshold.copy_(values[1].to(self.decision_threshold))
        self.calibrated.fill_(True)

    @staticmethod
    def _box_disagreement(first, second):
        delta = second[:, :2] - first[:, :2]
        return torch.cat((delta, torch.linalg.norm(delta, dim=1, keepdim=True)), dim=1)

    def forward(
        self,
        observation_box,
        raw_box,
        availability,
        base_evidence,
        extension_evidence,
        base_presence_probability,
        extension_presence_probability,
        observation_stats,
        b1_sigma_parallel_perp,
        query_delta_t,
        gap_ratio,
        recursive_age=None,
        enabled=True,
        coarse_box=None,
        b1_center_xy=None,
        targetness_entropy=None,
        normalized_ess=None,
        extension_point_count=None,
        extension_voxel_count=None,
        targetness_mean=None,
        targetness_max=None,
        h1_utility_logit=None,
        h1_expected_gain=None,
    ):
        observation = observation_box.detach()
        raw = raw_box.detach()
        batch_size = observation.shape[0]
        dt = query_delta_t.reshape(batch_size).detach().clamp(min=0.0)
        gap = gap_ratio.reshape(batch_size).detach().clamp(min=0.0)
        if recursive_age is None:
            recursive_age = torch.zeros_like(dt)
        age = recursive_age.reshape(batch_size).detach().clamp(min=0.0)
        raw_finite = torch.isfinite(raw).all(dim=1)
        residual = torch.nan_to_num(
            raw[:, :2] - observation[:, :2], nan=0.0, posinf=0.0, neginf=0.0
        )
        residual_norm = torch.linalg.norm(residual, dim=1)
        evidence = self.evidence_projection(
            torch.cat((base_evidence.detach(), extension_evidence.detach()), dim=1)
        )
        if coarse_box is None:
            coarse_box = observation
        coarse_box = coarse_box.detach()
        coarse_center_gap = torch.linalg.norm(
            coarse_box[:, :2] - observation[:, :2], dim=1
        )
        coarse_yaw_gap = torch.atan2(
            torch.sin(coarse_box[:, 3] - observation[:, 3]),
            torch.cos(coarse_box[:, 3] - observation[:, 3]),
        ).abs()
        coarse_agreement = torch.stack((coarse_center_gap, coarse_yaw_gap), dim=1)
        if b1_center_xy is None:
            b1_center_xy = observation[:, :2]
        prior_box = torch.cat(
            (b1_center_xy.detach(), observation[:, 2:].detach()), dim=1
        )

        def column(value, default=0.0):
            if value is None:
                return observation.new_full((batch_size, 1), float(default))
            return torch.nan_to_num(value.detach().reshape(batch_size, 1))

        point_evidence = torch.cat(
            (
                column(targetness_entropy),
                column(normalized_ess),
                torch.log1p(column(extension_point_count).clamp(min=0.0)),
                torch.log1p(column(extension_voxel_count).clamp(min=0.0)),
                column(targetness_mean),
                column(targetness_max),
            ),
            dim=1,
        )
        features = torch.cat(
            (
                evidence,
                base_presence_probability.detach().unsqueeze(1),
                extension_presence_probability.detach().unsqueeze(1),
                coarse_agreement,
                self._box_disagreement(observation, prior_box),
                self._box_disagreement(observation, raw),
                torch.log(torch.clamp(b1_sigma_parallel_perp.detach(), min=0.1)),
                torch.log1p(dt).unsqueeze(1),
                torch.log1p(gap).unsqueeze(1),
                torch.log1p(age).unsqueeze(1),
                torch.nan_to_num(observation_stats.detach()),
                point_evidence,
            ),
            dim=1,
        )
        risk_feature = self.risk_trunk(torch.nan_to_num(features))
        helpful_logit = self.helpful_head(risk_feature).squeeze(1)
        harmful_logit = self.harmful_head(risk_feature).squeeze(1)
        expected_center_gain = self.expected_center_gain_head(risk_feature).squeeze(1)
        expected_iou_gain = self.expected_iou_gain_head(risk_feature).squeeze(1)
        helpful_probability = torch.sigmoid(helpful_logit)
        harmful_probability = torch.sigmoid(harmful_logit)
        probability = helpful_probability * (1.0 - harmful_probability)
        apply_logit = helpful_logit - harmful_logit
        radius = torch.clamp(
            self.radius_base + self.radius_per_second * dt, max=self.radius_max
        )
        residual_scale = torch.clamp(
            radius / torch.clamp(residual_norm, min=1e-6), max=1.0
        )
        bounded_residual = residual * residual_scale.unsqueeze(1)
        evidence_valid = (
            (availability.reshape(batch_size) > 0)
            & (
                extension_presence_probability
                >= self.presence_threshold.to(extension_presence_probability)
            )
            & raw_finite
            & torch.isfinite(features).all(dim=1)
        )
        deployable = bool(enabled) and (
            bool(self.calibrated) or not self.require_calibration
        )
        action = (
            evidence_valid
            & deployable
            & (probability >= self.decision_threshold.to(probability))
        )
        final_xy = torch.where(
            action.unsqueeze(1),
            observation[:, :2] + bounded_residual,
            observation[:, :2],
        )
        final_box = torch.cat((final_xy, observation[:, 2:]), dim=1)
        return final_box, {
            "ct_b3_help_logit": helpful_logit,
            "ct_b3_harm_logit": harmful_logit,
            "ct_b3_help_probability": helpful_probability,
            "ct_b3_harm_probability": harmful_probability,
            "ct_b3_expected_center_gain": expected_center_gain,
            "ct_b3_expected_iou_gain": expected_iou_gain,
            "ct_b3_action_score": probability,
            "ct_b3_calibrated": observation.new_full(
                (batch_size,), float(bool(self.calibrated))
            ),
            # v23 compatibility aliases.
            "ct_b3_h3_residual": apply_logit,
            "ct_b3_h3_utility": apply_logit,
            "ct_b3_final_gate": action.to(observation.dtype),
            "ct_router_logit": apply_logit,
            "ct_router_gate": probability,
            "ct_router_applied_gate": action.to(observation.dtype),
            "ct_router_evidence_valid": evidence_valid.to(observation.dtype),
            "ct_router_bounded_residual_xy": bounded_residual,
            "ct_router_residual_xy": residual,
            "ct_router_radius": radius,
            "ct_router_clip_rate": (residual_norm > radius).to(observation.dtype),
            "ct_router_soft_box": torch.cat(
                (
                    observation[:, :2] + probability.unsqueeze(1) * bounded_residual,
                    observation[:, 2:],
                ),
                dim=1,
            ),
        }
