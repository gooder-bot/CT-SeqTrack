"""Deployment-aligned B2 evidence acquisition and memory conditioning.

Contract v3 deliberately keeps B0 point tokens as context only.  Extension
tokens are the sole queries and the sole source of recovery votes.
"""

import math

import torch
from torch import nn


def _masked_mean(features, mask):
    weights = mask.to(features.dtype).unsqueeze(-1)
    return (features * weights).sum(dim=1) / torch.clamp(
        weights.sum(dim=1), min=1.0)


def _masked_max(features, mask):
    values = features.masked_fill(~mask.unsqueeze(-1), float("-inf"))
    result = values.max(dim=1).values
    return torch.where(
        mask.any(dim=1, keepdim=True), result, torch.zeros_like(result))


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
        context_scale=2.0):
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
        (batch_size, history_count * per_frame, feature_dim))
    valid = torch.zeros(
        (batch_size, history_count * per_frame), dtype=torch.bool,
        device=history_features.device)
    points_xyz = history_points[..., :3].detach()
    features = history_features.detach()
    boxes = history_boxes.detach()
    sizes = box_size.detach()
    history_valid = history_valid_mask.reshape(
        batch_size, history_count).to(torch.bool)

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
            normalized = torch.stack((
                2.0 * local_x / size_xy[0],
                2.0 * local_y / size_xy[1],
                2.0 * (xyz[:, 2] - box[2])
                / torch.clamp(sizes[batch_index, 2], min=1e-3),
            ), dim=1).abs()
            inside = finite & (normalized <= 1.0).all(dim=1)
            neighborhood = finite & (normalized <= float(
                context_scale)).all(dim=1) & ~inside

            foreground_rows = torch.nonzero(
                inside, as_tuple=False).flatten()
            context_rows = torch.nonzero(
                neighborhood, as_tuple=False).flatten()
            foreground_pick = foreground_rows.index_select(
                0, _fps_indices(
                    xyz.index_select(0, foreground_rows),
                    foreground_tokens)) if foreground_rows.numel() else (
                        foreground_rows)
            context_pick = context_rows.index_select(
                0, _fps_indices(
                    xyz.index_select(0, context_rows),
                    context_tokens)) if context_rows.numel() else context_rows
            block_start = history_index * per_frame
            for offset, rows in ((0, foreground_pick),
                                 (int(foreground_tokens), context_pick)):
                take = min(
                    int(rows.numel()),
                    int(foreground_tokens if offset == 0 else context_tokens))
                if take == 0:
                    continue
                output_rows = slice(
                    block_start + offset, block_start + offset + take)
                tokens[batch_index, output_rows] = features[
                    batch_index, history_index].index_select(0, rows[:take])
                valid[batch_index, output_rows] = True
    return tokens, valid


class ExtensionMemorySearchRefiner(nn.Module):
    """Extension-Q/current-base+history-memory-KV recovery refiner."""

    def __init__(
            self,
            feature_dim=64,
            num_heads=4,
            max_vote_offset=4.0,
            attention_dropout=0.0,
            presence_init_probability=0.1,
            utility_init_probability=0.05):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_heads = int(num_heads)
        self.max_vote_offset = float(max_vote_offset)
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
        self.query_norm = nn.LayerNorm(64)
        self.kv_norm = nn.LayerNorm(64)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=64, num_heads=4, dropout=0.0, batch_first=True)
        self.attention_residual_gate = nn.Parameter(torch.zeros(64))
        self.output_norm = nn.LayerNorm(64)

        self.targetness_head = nn.Sequential(
            nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))
        self.vote_head = nn.Sequential(
            nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 2))
        self.base_presence_head = nn.Sequential(
            nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))
        self.extension_presence_head = nn.Sequential(
            nn.Linear(130, 64), nn.GELU(), nn.Linear(64, 1))
        # base/ext evidence, residual xy, observation stats, sigma, dt, age
        self.utility_trunk = nn.Sequential(
            nn.Linear(64 * 2 + 2 + 5 + 2 + 2, 64),
            nn.GELU(),
        )
        self.utility_logit_head = nn.Linear(64, 1)
        self.expected_gain_head = nn.Linear(64, 1)

        for head, probability in (
                (self.base_presence_head[-1], presence_init_probability),
                (self.extension_presence_head[-1], presence_init_probability),
                (self.utility_logit_head, utility_init_probability)):
            nn.init.zeros_(head.weight)
            nn.init.constant_(
                head.bias, math.log(probability / (1.0 - probability)))
        nn.init.zeros_(self.expected_gain_head.weight)
        nn.init.zeros_(self.expected_gain_head.bias)

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
            recursive_age=None):
        batch_size, extension_count, _ = extension_points.shape
        if current_base_features.shape != (
                batch_size, 1024, self.feature_dim):
            raise ValueError("current base features must be [B,1024,64]")
        if memory_tokens.shape[1:] != (36, self.feature_dim):
            raise ValueError("history memory must be [B,36,64]")
        extension_mask = extension_valid_mask.reshape(
            batch_size, extension_count).to(torch.bool)
        extension_mask &= torch.isfinite(extension_points).all(dim=2)
        base_mask = current_base_valid_mask.reshape(
            batch_size, 1024).to(torch.bool)
        base_mask &= torch.isfinite(current_base_features).all(dim=2)
        memory_mask = memory_valid_mask.reshape(
            batch_size, 36).to(torch.bool)

        safe_points = torch.nan_to_num(extension_points)
        direction = torch.nan_to_num(b1_direction_xy.detach())
        direction_norm = torch.linalg.norm(direction, dim=1, keepdim=True)
        default_direction = torch.zeros_like(direction)
        default_direction[:, 0] = 1.0
        direction = torch.where(
            direction_norm > 1e-6,
            direction / torch.clamp(direction_norm, min=1e-6),
            default_direction)
        perpendicular = torch.stack((-direction[:, 1], direction[:, 0]), 1)
        delta = safe_points[..., :2] - b1_center_xy.detach().unsqueeze(1)
        sigma = torch.clamp(
            torch.nan_to_num(b1_sigma_parallel_perp.detach(), nan=1.0),
            min=0.1)
        longitudinal = (
            delta * direction.unsqueeze(1)).sum(2) / sigma[:, 0:1]
        lateral = (
            delta * perpendicular.unsqueeze(1)).sum(2) / sigma[:, 1:2]
        dt = torch.clamp(query_delta_t.reshape(batch_size), min=0.0)
        gap = torch.clamp(gap_ratio.reshape(batch_size), min=0.0)
        b1_valid_f = b1_valid.reshape(batch_size).to(safe_points.dtype)
        geometry = torch.stack((
            longitudinal,
            lateral,
            torch.log1p(dt).unsqueeze(1).expand(-1, extension_count),
            gap.unsqueeze(1).expand(-1, extension_count),
            b1_valid_f.unsqueeze(1).expand(-1, extension_count),
        ), dim=2)
        source = extension_source.reshape(
            batch_size, extension_count).long().clamp(0, 3)
        extension_feature = (
            self.extension_encoder(safe_points)
            + self.geometry_encoder(geometry)
            + self.source_embedding(source))
        extension_feature = extension_feature * extension_mask.unsqueeze(2)

        base_features = current_base_features.detach()
        memory_features = memory_tokens.detach()
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
            + attention_output * self.attention_residual_gate.view(1, 1, -1))
        enriched = enriched * extension_mask.unsqueeze(2)

        logits = self.targetness_head(enriched).squeeze(2)
        logits = logits.masked_fill(~extension_mask, -20.0)
        scores = torch.sigmoid(logits) * extension_mask.to(logits.dtype)
        votes = safe_points[..., :2] + self.max_vote_offset * torch.tanh(
            self.vote_head(enriched))
        weight_sum = scores.sum(dim=1, keepdim=True)
        raw_xy = (scores.unsqueeze(2) * votes).sum(dim=1) / torch.clamp(
            weight_sum, min=1e-6)
        extension_point_count = extension_mask.sum(dim=1)
        availability = (
            (b1_valid_f > 0)
            & (extension_point_count > 0)
            & torch.isfinite(raw_xy).all(dim=1))
        observation_xy = observation_box[:, :2].detach()
        raw_xy = torch.where(availability.unsqueeze(1), raw_xy, observation_xy)
        raw_box = torch.cat((raw_xy, observation_box[:, 2:].detach()), dim=1)

        base_mean = _masked_mean(base_features, base_mask)
        base_max = _masked_max(base_features, base_mask)
        extension_mean = _masked_mean(enriched, extension_mask)
        extension_max = _masked_max(enriched, extension_mask)
        base_presence_logit = self.base_presence_head(torch.cat((
            base_mean, base_max), dim=1)).squeeze(1)
        extension_presence_logit = self.extension_presence_head(torch.cat((
            extension_mean,
            extension_max,
            torch.log1p(extension_point_count.to(enriched.dtype)).unsqueeze(1),
            b1_valid_f.unsqueeze(1),
        ), dim=1)).squeeze(1)
        base_presence_probability = torch.sigmoid(base_presence_logit)
        extension_presence_probability = (
            torch.sigmoid(extension_presence_logit)
            * availability.to(enriched.dtype))

        if recursive_age is None:
            recursive_age = torch.zeros_like(dt)
        utility_input = torch.cat((
            base_mean,
            extension_mean,
            raw_xy - observation_xy,
            torch.nan_to_num(observation_stats.detach()),
            torch.log(torch.clamp(sigma, min=0.1)),
            torch.log1p(dt).unsqueeze(1),
            torch.log1p(torch.clamp(
                recursive_age.reshape(batch_size), min=0.0)).unsqueeze(1),
        ), dim=1)
        utility_feature = self.utility_trunk(utility_input)
        utility_logit = self.utility_logit_head(utility_feature).squeeze(1)
        expected_gain = self.expected_gain_head(utility_feature).squeeze(1)

        probability = scores / torch.clamp(weight_sum, min=1e-6)
        entropy = -(probability * torch.log(torch.clamp(
            probability, min=1e-8))).sum(dim=1)
        targetness_mean = scores.sum(dim=1) / torch.clamp(
            extension_point_count.to(scores.dtype), min=1.0)
        targetness_max = scores.max(dim=1).values
        normalized_ess = 1.0 / torch.clamp(
            probability.pow(2).sum(dim=1)
            * torch.clamp(extension_point_count.to(scores.dtype), min=1.0),
            min=1e-6)
        return {
            "b0_current_base_features_detached": base_features,
            "ct_memory_tokens": memory_tokens,
            "ct_memory_valid_mask": memory_mask,
            "ct_extension_query_features": enriched,
            "ct_cross_attention_weights": attention_weights,
            "ct_b2_available": availability.to(enriched.dtype),
            "ct_b2_base_presence_logit": base_presence_logit,
            "ct_b2_base_presence_probability": base_presence_probability,
            "ct_b2_base_evidence": base_mean,
            "ct_b2_extension_presence_logit": extension_presence_logit,
            "ct_b2_extension_presence_probability":
                extension_presence_probability,
            "ct_b2_extension_evidence": extension_mean,
            "ct_b2_utility_logit": utility_logit,
            "ct_b2_expected_gain": expected_gain,
            "ct_b2_raw_box": raw_box,
            "ct_search_targetness_logits": logits,
            "ct_search_point_votes": votes,
            "ct_search_unmasked_raw_xy": raw_xy,
            "ct_search_raw_xy": raw_xy,
            "ct_search_candidate_valid": availability.to(enriched.dtype),
            "ct_search_structural_valid": availability.to(enriched.dtype),
            "ct_search_new_support_valid": (
                extension_point_count > 0).to(enriched.dtype),
            "ct_search_support_valid": availability.to(enriched.dtype),
            "ct_search_available": availability.to(enriched.dtype),
            "ct_search_effective": availability.to(enriched.dtype),
            "ct_search_presence_logit": extension_presence_logit,
            "ct_search_presence_probability":
                extension_presence_probability,
            "ct_search_targetness_mean": targetness_mean,
            "ct_search_targetness_max": targetness_max,
            "ct_search_targetness_entropy": entropy,
            "ct_search_normalized_ess": normalized_ess,
            "ct_search_extension_selected_count":
                extension_point_count.to(enriched.dtype),
        }


class H3UtilityResidualRouter(nn.Module):
    """Zero-init H3 safety correction on top of the B2 H1 utility logit."""

    def __init__(
            self,
            observation_stats_dim=5,
            hidden_dim=64,
            presence_threshold=0.5,
            decision_threshold=0.5,
            radius_base=0.5,
            radius_per_second=0.5,
            radius_max=2.0):
        super().__init__()
        self.presence_threshold = float(presence_threshold)
        self.radius_base = float(radius_base)
        self.radius_per_second = float(radius_per_second)
        self.radius_max = float(radius_max)
        self.register_buffer(
            "decision_threshold", torch.tensor(float(decision_threshold)))
        self.evidence_projection = nn.Sequential(
            nn.Linear(128, 32), nn.GELU())
        # projected base/extension evidence(32), presence(2),
        # utility/gain(2), residual(3), uncertainty(2), dt/gap/age(3),
        # observation statistics.
        input_dim = 32 + 2 + 2 + 3 + 2 + 3 + int(observation_stats_dim)
        self.residual_head = nn.Sequential(
            nn.Linear(input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def forward(
            self,
            observation_box,
            raw_box,
            availability,
            base_evidence,
            extension_evidence,
            base_presence_probability,
            extension_presence_probability,
            h1_utility_logit,
            h1_expected_gain,
            observation_stats,
            b1_sigma_parallel_perp,
            query_delta_t,
            gap_ratio,
            recursive_age=None,
            enabled=True):
        observation = observation_box.detach()
        raw = raw_box.detach()
        batch_size = observation.shape[0]
        dt = query_delta_t.reshape(batch_size).detach().clamp(min=0.0)
        gap = gap_ratio.reshape(batch_size).detach().clamp(min=0.0)
        if recursive_age is None:
            recursive_age = torch.zeros_like(dt)
        age = recursive_age.reshape(batch_size).detach().clamp(min=0.0)
        residual = raw[:, :2] - observation[:, :2]
        residual_norm = torch.linalg.norm(residual, dim=1)
        evidence = self.evidence_projection(torch.cat((
            base_evidence.detach(), extension_evidence.detach()), dim=1))
        features = torch.cat((
            evidence,
            base_presence_probability.detach().unsqueeze(1),
            extension_presence_probability.detach().unsqueeze(1),
            h1_utility_logit.detach().unsqueeze(1),
            h1_expected_gain.detach().unsqueeze(1),
            residual,
            residual_norm.unsqueeze(1),
            torch.log(torch.clamp(
                b1_sigma_parallel_perp.detach(), min=0.1)),
            torch.log1p(dt).unsqueeze(1),
            torch.log1p(gap).unsqueeze(1),
            torch.log1p(age).unsqueeze(1),
            torch.nan_to_num(observation_stats.detach()),
        ), dim=1)
        h3_residual = self.residual_head(features).squeeze(1)
        if not bool(enabled):
            h3_residual = torch.zeros_like(h3_residual)
        apply_logit = h1_utility_logit + h3_residual
        probability = torch.sigmoid(apply_logit)
        radius = torch.clamp(
            self.radius_base + self.radius_per_second * dt,
            max=self.radius_max)
        residual_scale = torch.clamp(
            radius / torch.clamp(residual_norm, min=1e-6), max=1.0)
        bounded_residual = residual * residual_scale.unsqueeze(1)
        evidence_valid = (
            (availability.reshape(batch_size) > 0)
            & (extension_presence_probability >= self.presence_threshold)
            & torch.isfinite(raw).all(dim=1))
        action = evidence_valid & (
            probability >= self.decision_threshold.to(probability))
        final_xy = observation[:, :2] + action.to(
            observation.dtype).unsqueeze(1) * bounded_residual
        final_box = torch.cat((final_xy, observation[:, 2:]), dim=1)
        return final_box, {
            "ct_b3_h3_residual": h3_residual,
            "ct_b3_h3_utility": apply_logit,
            "ct_b3_final_gate": action.to(observation.dtype),
            "ct_router_logit": apply_logit,
            "ct_router_gate": probability,
            "ct_router_applied_gate": action.to(observation.dtype),
            "ct_router_evidence_valid": evidence_valid.to(observation.dtype),
            "ct_router_bounded_residual_xy": bounded_residual,
            "ct_router_residual_xy": residual,
            "ct_router_radius": radius,
            "ct_router_clip_rate": (residual_norm > radius).to(
                observation.dtype),
            "ct_router_soft_box": torch.cat((
                observation[:, :2] + probability.unsqueeze(1)
                * bounded_residual,
                observation[:, 2:],
            ), dim=1),
        }
