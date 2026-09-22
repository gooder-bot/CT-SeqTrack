"""v32 共享七框 decoder：局部几何 query，公开输出仍为世界轴框。

B0 的四帧 source 仍由 SeqTrack 原 local/global encoder 编码。
证据与记忆只接在 cross-attention 的 K/V；候选之间没有 self-attention。
"""
from __future__ import annotations

from dataclasses import replace
from typing import Mapping

import torch
from torch import Tensor, nn

from models.attn.Models import Encoder, Decoder
from .contracts import DecoderOutput, EvidenceHypotheses, ObservationFeatures, PriorContext
from .observation import box_corners_xyz
from .geometry import anchor_yaw, rotate_xy, transform_boxes


def _masked_source(value: Tensor, valid: Tensor, name: str) -> Tensor:
    if value.shape[:-1] != valid.shape:
        raise ValueError(f"{name} feature and validity shapes disagree")
    if not bool(torch.isfinite(value[valid]).all()):
        raise ValueError(f"valid {name} features must be finite")
    return torch.where(valid[..., None], value, 0.)


class SharedHypothesisDecoder(nn.Module):
    """一个共享 fine pose head，不假设 source 帧数等于 query 框数。"""

    def __init__(self, prior_dim: int = 128, dropout: float = .2):
        super().__init__()
        self.prior_dim = int(prior_dim)
        self.d_model = 64
        self.source_projection = nn.Linear(128, 64)
        self.corner_projection = nn.Linear(4, 64)
        # 保留 SeqTrack 注意力定义：每头 d_k=d_v=64，非 64/4。
        options = dict(d_word_vec=64, n_layers=3, n_head=4, d_k=64, d_v=64,
                       d_model=64, d_inner=512, pad_idx=1, dropout=float(dropout))
        self.encoder = Encoder(**options)
        self.encoder_global = Encoder(**options)
        self.decoder = Decoder(**options)
        self.extension_projection = nn.Linear(64, 64)
        self.memory_projection = nn.Linear(64, 64)
        # context + delta_xy + log_sigma + direction + log(dt) + valid + quality4。
        self.prior_adapter = nn.Sequential(nn.Linear(self.prior_dim + 12, 64),
                                           nn.GELU(), nn.Linear(64, 64))
        self.box_projection = nn.Linear(8 * 64, 64)
        self.pose_head = nn.Linear(64, 5)
        self.quality_head = nn.Linear(64, 1)
        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)
        nn.init.zeros_(self.prior_adapter[-1].weight)
        nn.init.zeros_(self.prior_adapter[-1].bias)
        nn.init.zeros_(self.pose_head.weight)
        with torch.no_grad():
            self.pose_head.bias.copy_(torch.tensor([0., 0., 0., 0., 1.]))
        nn.init.zeros_(self.quality_head.weight)
        nn.init.zeros_(self.quality_head.bias)

    @staticmethod
    def _frame_times(batch: Mapping[str, Tensor], reference: Tensor) -> Tensor:
        times = batch.get('frame_times')
        if times is None:
            history_times = batch.get('history_times')
            if history_times is None:
                raise KeyError("decoder requires frame_times or history_times, including empty frames")
            history_times = torch.as_tensor(history_times, device=reference.device, dtype=reference.dtype)
            if history_times.shape != (len(reference), 3):
                raise ValueError("history_times must be [B,3]")
            times = torch.cat((history_times, reference.new_zeros(len(reference), 1)), dim=1)
        times = torch.as_tensor(times, device=reference.device, dtype=reference.dtype)
        if times.shape != (len(reference), 4) or not bool(torch.isfinite(times).all()):
            raise ValueError("frame_times must be finite [B,4]")
        return times.detach()

    def encode_sources(self, observation: ObservationFeatures, history_valid: Tensor):
        source, valid = observation.source_tokens, observation.source_valid.bool()
        if source.ndim != 4 or source.shape[1] != 4 or source.shape[-1] != 128:
            raise ValueError("B0 source_tokens must be [B,4,T,128]")
        batch, frames, tokens, _ = source.shape
        if tokens < 1 or valid.shape != source.shape[:-1]:
            raise ValueError("source_valid must match positive B0 token slots")
        exists = torch.cat((history_valid, torch.ones_like(history_valid[:, :1])), dim=1)
        valid = valid & exists[..., None]
        source = self.source_projection(_masked_source(source, valid, 'B0'))
        source = torch.where(valid[..., None], source, 0.)
        local_valid = valid.reshape(batch * frames, tokens)
        local, = self.encoder(source.reshape(batch * frames, tokens, 64),
            src_mask=local_valid[:, :, None] & local_valid[:, None, :],
            query_valid=local_valid[..., None], zero_invalid_rows=True)
        flat_valid = valid.reshape(batch, frames * tokens)
        global_source, = self.encoder_global(source.reshape(batch, frames * tokens, 64),
            src_mask=flat_valid[:, :, None] & flat_valid[:, None, :],
            query_valid=flat_valid[..., None], zero_invalid_rows=True, global_feature=True)
        return (torch.cat((local.reshape(batch, frames * tokens, 64), global_source), dim=1),
                torch.cat((flat_valid, flat_valid), dim=1))

    @staticmethod
    def _seeds(observation: ObservationFeatures, evidence: EvidenceHypotheses,
               history_boxes: Tensor, history_valid: Tensor, prior: PriorContext):
        batch = len(observation.coarse_box)
        centers = evidence.centers_xyz
        if centers.shape != (batch, 3, 3) or evidence.mode_valid.shape != (batch, 3):
            raise ValueError("decoder requires three XYZ evidence hypotheses")
        extension_valid = evidence.point_valid.bool()
        if evidence.members.shape != (batch, 3, extension_valid.shape[1]):
            raise ValueError("fixed mode members must match extension point slots")
        members = evidence.members.detach().bool() & extension_valid[:, None]
        mode_valid = evidence.mode_valid.detach().bool() & members.any(-1)
        prior_box = prior.box.detach().to(observation.coarse_box)
        if prior_box.shape != (batch, 4) or not bool(torch.isfinite(prior_box).all()):
            raise ValueError("prior.box must be a finite fallback [B,4]")
        if not bool(torch.isfinite(centers[mode_valid]).all()):
            raise ValueError("valid evidence centers must be finite")
        coarse = observation.coarse_box
        if coarse.shape != (batch, 4) or not bool(torch.isfinite(coarse).all()):
            raise ValueError("coarse_box must be finite [B,4]")
        effective_coarse = torch.where(observation.sequence_valid[:, None].bool(), coarse, prior_box)
        centers = torch.where(mode_valid[..., None], centers, 0.)
        yaw = effective_coarse[:, None, 3:4].detach().expand(-1, 3, -1)
        mode_seeds = torch.cat((centers, yaw), dim=-1)
        history = torch.where(history_valid[..., None], history_boxes, 0.).detach()
        # 输出直接加 live center；角点 query 单独 detach，避免几何梯度旁路。
        output_seeds = torch.cat((history, effective_coarse[:, None], mode_seeds), dim=1)
        corner_seeds = torch.cat((history, effective_coarse[:, None], mode_seeds.detach()), dim=1)
        current_valid = torch.cat((torch.ones_like(mode_valid[:, :1]), mode_valid), dim=1)
        query_valid = torch.cat((history_valid, current_valid), dim=1)
        return output_seeds, corner_seeds, query_valid, members, prior_box

    def _prior_embedding(self, observation: ObservationFeatures, prior: PriorContext,
                         effective_coarse: Tensor, size: Tensor, times: Tensor) -> Tensor:
        batch = len(effective_coarse)
        valid = prior.valid.detach().to(effective_coarse.device).bool().reshape(batch)
        if prior.feature.shape != (batch, self.prior_dim):
            raise ValueError("prior.feature has the wrong width")
        context_valid = (valid if prior.context_valid is None else
                         prior.context_valid.detach().to(effective_coarse.device).bool().reshape(batch))
        feature = _masked_source(prior.feature, context_valid, 'prior')
        scale = torch.linalg.vector_norm(size[:, :2], dim=-1, keepdim=True).clamp_min(1e-6)
        delta = (prior.box[:, :2].detach() - effective_coarse[:, :2].detach()) / scale
        gap = (times[:, -1] - times[:, -2]).abs().clamp_min(1e-6)
        summary = torch.cat((delta, prior.log_sigma.detach(), prior.direction_xy.detach(),
                             torch.log1p(gap)[:, None], valid[:, None].to(feature),
                             observation.quality.detach()), dim=-1)
        summary = torch.nan_to_num(summary, nan=0., posinf=0., neginf=0.).detach()
        adapter = self.prior_adapter(torch.cat((feature, summary), dim=-1))
        return torch.where(context_valid[:, None], adapter, 0.)

    def forward(self, observation: ObservationFeatures, evidence: EvidenceHypotheses,
                batch: Mapping[str, Tensor], prior: PriorContext) -> DecoderOutput:
        reference = observation.coarse_box
        batch_size = len(reference)
        history_boxes = batch['history_boxes'].to(reference)
        history_valid = batch['history_valid'].to(reference.device).bool()
        size = batch['box_size'].to(reference)
        if history_boxes.shape != (batch_size, 3, 4) or history_valid.shape != (batch_size, 3):
            raise ValueError("decoder needs three history boxes and their validity")
        if size.shape != (batch_size, 3) or not bool(torch.isfinite(size).all()) or not bool((size > 0).all()):
            raise ValueError("box_size must be finite positive [B,3] L/W/H")
        if (observation.current_valid.shape != (batch_size,)
                or observation.sequence_valid.shape != (batch_size,)
                or observation.quality.shape != (batch_size, 4)):
            raise ValueError("observation current_valid/sequence_valid/quality shape mismatch")
        if not bool(torch.isfinite(history_boxes[history_valid]).all()):
            raise ValueError("valid history boxes must be finite")
        times = self._frame_times(batch, reference)
        seeds, corner_seeds, query_valid, members, prior_box = self._seeds(
            observation, evidence, history_boxes, history_valid, prior)
        # 只创建 decoder 的几何视图；B2 的输入、votes/covariance 和监督不换轴。
        # 模式 query 在 _seeds 已 detach，输出 seed 的 live center 梯度仍保留。
        yaw_anchor = anchor_yaw(batch, reference)
        local_seeds = transform_boxes(seeds, yaw_anchor)
        local_corners = transform_boxes(corner_seeds, yaw_anchor)
        local_prior = replace(prior,
            box=transform_boxes(prior.box.detach().to(reference), yaw_anchor),
            direction_xy=rotate_xy(prior.direction_xy.detach().to(reference), yaw_anchor))
        corners = box_corners_xyz(local_corners, size[:, None])
        query_times = torch.cat((times[:, :3], times[:, -1:].expand(-1, 4)), dim=1)
        corner_input = torch.cat((corners, query_times[:, :, None, None].expand(-1, -1, 8, -1)), dim=-1)
        queries = self.corner_projection(corner_input.reshape(batch_size, 56, 4))
        prior_embedding = self._prior_embedding(observation, local_prior, local_seeds[:, 3], size, times)
        prior_by_query = torch.cat((torch.zeros_like(prior_embedding[:, None]).expand(-1, 3, -1),
                                    prior_embedding[:, None].expand(-1, 4, -1)), dim=1)
        queries = queries + prior_by_query.repeat_interleave(8, dim=1)
        source, source_valid = self.encode_sources(observation, history_valid)
        extension_valid, memory_valid = evidence.point_valid.bool(), evidence.memory_valid.bool()
        extension = self.extension_projection(_masked_source(evidence.point_features, extension_valid, 'extension'))
        memory = self.memory_projection(_masked_source(evidence.memory_features, memory_valid, 'memory'))
        extension = torch.where(extension_valid[..., None], extension, 0.)
        memory = torch.where(memory_valid[..., None], memory, 0.)
        kv = torch.cat((source, extension, memory), dim=1)
        # B0/current q0 可读全部 extension；历史不读新增当前点；每个 mode只读成员。
        extension_allowed = torch.cat((
            torch.zeros((batch_size, 3, extension_valid.shape[1]), device=reference.device, dtype=torch.bool),
            extension_valid[:, None], members), dim=1)
        allowed = torch.cat((source_valid[:, None].expand(-1, 7, -1), extension_allowed,
                             memory_valid[:, None].expand(-1, 7, -1)), dim=-1)
        allowed = (allowed & query_valid[:, :, None]).repeat_interleave(8, dim=1)
        corner_valid = query_valid.repeat_interleave(8, dim=1)
        decoded, _ = self.decoder(queries, None, kv, allowed,
            query_valid=corner_valid[..., None], zero_invalid_rows=True)
        features = self.box_projection(decoded.reshape(batch_size, 7, 8 * 64))
        features = torch.where(query_valid[..., None], features, 0.)
        residual = self.pose_head(features)
        sine, cosine = residual[..., 3], residual[..., 4]
        nonzero = sine.square() + cosine.square() > 1e-12
        angle = torch.atan2(torch.where(nonzero, sine, 0.), torch.where(nonzero, cosine, 1.))
        yaw = local_seeds[..., 3] + angle
        yaw = torch.atan2(yaw.sin(), yaw.cos())
        boxes = torch.cat((local_seeds[..., :3] + residual[..., :3], yaw[..., None]), dim=-1)
        boxes = transform_boxes(boxes, yaw_anchor, to_world=True)
        boxes = torch.where(query_valid[..., None], boxes, 0.)
        current_boxes = boxes[:, 3:]
        # 历史真实点可预测当前框，但不会变成当前测量/记忆写入资格。
        # B0 全空而 extension 存在时维持原规则：q0=prior，模式仍可竞争。
        q0 = torch.where(observation.sequence_valid[:, None].bool(), current_boxes[:, 0], prior_box)
        current_boxes = torch.cat((q0[:, None], current_boxes[:, 1:]), dim=1)
        no_sequence_points = ~observation.sequence_valid.bool() & ~extension_valid.any(-1)
        current_boxes = torch.where(no_sequence_points[:, None, None], prior_box[:, None], current_boxes)
        current_valid = query_valid[:, 3:].clone()
        current_valid[:, 1:] = current_valid[:, 1:] & ~no_sequence_points[:, None]
        quality = self.quality_head(features[:, 3:]).squeeze(-1)
        quality = quality.masked_fill(~current_valid, -20.)
        return DecoderOutput(hypothesis_boxes=current_boxes, quality_logits=quality,
            hypothesis_valid=current_valid, history_boxes=boxes[:, :3], decoder_features=features[:, 3:])
