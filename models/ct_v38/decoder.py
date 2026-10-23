"""v38 紧凑四框 decoder：三个历史框与 q0，仅读取 B0 local/global K/V。"""
from __future__ import annotations

from dataclasses import replace
import math
from typing import Mapping

import torch
from torch import Tensor, nn

from models.attn.Models import Encoder, Decoder
from .contracts import DecoderOutput, ObservationFeatures, PriorContext
from .observation import box_corners_xyz
from .geometry import anchor_yaw, rotate_xy, transform_boxes
from .history_context import history_descriptor
from .local_observation import LocalObservationReader
from utils.feature_precision import feature_autocast, feature_fp32


def _masked_source(value: Tensor, valid: Tensor, name: str) -> Tensor:
    if value.shape[:-1] != valid.shape:
        raise ValueError(f"{name} feature and validity shapes disagree")
    if not bool(torch.isfinite(value[valid]).all()):
        raise ValueError(f"valid {name} features must be finite")
    return torch.where(valid[..., None], value, 0.)


def _consume_removed_projection_rng(*, xavier: bool):
    """仅初始化时保留 14c3b2f 有效参数的抽样顺序，不注册退役参数。

    原两个 64→64 投影先执行 Linear 默认初始化，之后其权重再次执行
    Xavier。两个阶段都消费对应随机数；临时张量不进入网络或优化器。
    """
    for _ in range(2):
        if xavier:
            nn.init.xavier_uniform_(torch.empty(64, 64))
        else:
            # 精确复用 Linear.reset_parameters 的抽样，只有临时普通张量。
            weight = torch.empty(64, 64)
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            nn.init.uniform_(torch.empty(64), -1 / math.sqrt(64), 1 / math.sqrt(64))


class SharedHypothesisDecoder(nn.Module):
    """共享 fine pose head；四个输出框均有八角点 query。"""

    def __init__(self, prior_dim: int = 128, dropout: float = .2, *,
                 query_context: bool = False, time_scale: float = .5,
                 initial_context: bool = False, local_observation: bool = False):
        super().__init__()
        self.execution_profile = 'legacy'
        self.query_context = bool(query_context)
        self.initial_context = bool(initial_context)
        self.local_observation = bool(local_observation)
        self.context_time_scale = float(time_scale)
        if self.query_context and self.context_time_scale <= 0:
            raise ValueError('v34 context time_scale must be positive')
        self.prior_dim = int(prior_dim)
        self.d_model = 64
        self.source_projection = nn.Linear(128, 64)
        # XYZ/time + 输出参考中心：q0 以 anchor 为原点，history 以 seed 为原点。
        # 共享 head 从角点输入区分两类中心回归目标。
        self.corner_projection = nn.Linear(7, 64)
        # 保留 SeqTrack 注意力定义：每头 d_k=d_v=64，非 64/4。
        options = dict(d_word_vec=64, n_layers=3, n_head=4, d_k=64, d_v=64,
                       d_model=64, d_inner=512, pad_idx=1, dropout=float(dropout))
        self.encoder = Encoder(**options)
        self.encoder_global = Encoder(**options)
        self.decoder = Decoder(**options)
        _consume_removed_projection_rng(xavier=False)
        # context + delta_xy + log_sigma + direction + log(dt) + valid + quality4。
        self.prior_adapter = nn.Sequential(nn.Linear(self.prior_dim + 12, 64),
                                           nn.GELU(), nn.Linear(64, 64))
        self.box_projection = nn.Linear(8 * 64, 64)
        self.pose_head = nn.Linear(64, 5)
        self.quality_head = nn.Linear(64, 1)
        for name, parameter in self.named_parameters():
            if name == 'prior_adapter.0.weight':
                _consume_removed_projection_rng(xavier=True)
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)
        nn.init.zeros_(self.prior_adapter[-1].weight)
        nn.init.zeros_(self.prior_adapter[-1].bias)
        nn.init.zeros_(self.pose_head.weight)
        with torch.no_grad():
            self.pose_head.bias.copy_(torch.tensor([0., 0., 0., 0., 1.]))
        nn.init.zeros_(self.quality_head.weight)
        nn.init.zeros_(self.quality_head.bias)
        if self.query_context:
            # 必须放在旧参数初始化之后；不消费后续 B1/B2 的公共 CPU RNG。
            # v33 不构造这些层。新分支无 BN/dropout，零出口保持原初始函数。
            with torch.random.fork_rng(devices=[]):
                self.history_context = nn.Sequential(nn.Linear(33 if self.initial_context else 30, 64),
                                                     nn.LayerNorm(64), nn.GELU())
                self.coarse_context = nn.Sequential(nn.Linear(259, 64), nn.LayerNorm(64), nn.GELU())
                self.context_fusion = nn.Linear(128, 64)
                for module in (self.history_context, self.coarse_context):
                    nn.init.xavier_uniform_(module[0].weight)
                    nn.init.zeros_(module[0].bias)
                nn.init.zeros_(self.context_fusion.weight)
                nn.init.zeros_(self.context_fusion.bias)
        if self.local_observation:
            with torch.random.fork_rng(devices=[]):
                self.local_reader = LocalObservationReader()

    def _query_context(self, observation: ObservationFeatures, batch: Mapping[str, Tensor],
                       local_history: Tensor, history_valid: Tensor, size: Tensor) -> Tensor:
        """固定历史槽保留时间/几何配对；仅 pooled 语义允许同帧反传。"""
        b = len(size)
        pooled, support, current = (observation.coarse_features, observation.history_support,
                                    observation.current_support)
        if pooled is None or support is None or current is None:
            raise ValueError('v34 query context requires coarse_features and observation support')
        if pooled.shape != (b, 256) or support.shape != (b, 3, 3) or current.shape != (b, 3):
            raise ValueError('v34 query context has invalid pooled/support shapes')
        # frame_times是真实物理时间；共用纯函数与coarse保持相同槽语义。
        descriptor = history_descriptor(batch, local_history, history_valid, size, support,
                                        self.context_time_scale, include_initial=self.initial_context)
        present = observation.sequence_valid.bool()
        current_input = torch.cat((pooled, current.detach()), dim=-1)
        current_input = torch.where(present[:, None], current_input, 0.)
        if not bool(torch.isfinite(current_input).all()):
            raise ValueError('v34 valid coarse context must be finite')
        history_feature = self.history_context(descriptor)
        coarse_feature = self.coarse_context(current_input)
        coarse_feature = torch.where(present[:, None], coarse_feature, 0.)
        return self.context_fusion(torch.cat((history_feature, coarse_feature), dim=-1))

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
        bf16 = self.execution_profile == 'optimized_bf16'
        with feature_autocast(source, enabled=bf16):
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
        local = feature_fp32(local, enabled=bf16)
        global_source = feature_fp32(global_source, enabled=bf16)
        return (torch.cat((local.reshape(batch, frames * tokens, 64), global_source), dim=1),
                torch.cat((flat_valid, flat_valid), dim=1))

    @staticmethod
    def _seeds(observation: ObservationFeatures, history_boxes: Tensor,
               history_valid: Tensor, prior: PriorContext):
        batch = len(observation.coarse_box)
        prior_box = prior.box.detach().to(observation.coarse_box)
        if prior_box.shape != (batch, 4) or not bool(torch.isfinite(prior_box).all()):
            raise ValueError("prior.box must be a finite fallback [B,4]")
        coarse = observation.coarse_box
        if coarse.shape != (batch, 4) or not bool(torch.isfinite(coarse).all()):
            raise ValueError("coarse_box must be finite [B,4]")
        effective_coarse = torch.where(observation.sequence_valid[:, None].bool(), coarse, prior_box)
        history = torch.where(history_valid[..., None], history_boxes, 0.).detach()
        seeds = torch.cat((history, effective_coarse[:, None]), dim=1)
        # q0 query 永远存在；是否是真实观测由 sequence_valid 单独判定。
        query_valid = torch.cat((history_valid, torch.ones_like(history_valid[:, :1])), dim=1)
        return seeds, query_valid, prior_box

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

    def forward(self, observation: ObservationFeatures, batch: Mapping[str, Tensor], prior: PriorContext) -> DecoderOutput:
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
        seeds, query_valid, prior_box = self._seeds(
            observation, history_boxes, history_valid, prior)
        # 几何视图使用 anchor-local 轴；公开输出仍为世界轴。
        yaw_anchor = anchor_yaw(batch, reference)
        local_seeds = transform_boxes(seeds, yaw_anchor)
        local_corners = transform_boxes(seeds, yaw_anchor)
        # q0 直接预测 anchor-local 最终中心；历史框输出 seed 残差。
        output_reference = torch.cat((local_seeds[:, :3, :3],
            torch.zeros_like(local_seeds[:, 3:4, :3])), dim=1)
        local_prior = replace(prior,
            box=transform_boxes(prior.box.detach().to(reference), yaw_anchor),
            direction_xy=rotate_xy(prior.direction_xy.detach().to(reference), yaw_anchor))
        corners = box_corners_xyz(local_corners, size[:, None])
        query_times = torch.cat((times[:, :3], times[:, -1:]), dim=1)
        corner_input = torch.cat((corners,
            query_times[:, :, None, None].expand(-1, -1, 8, -1),
            output_reference.detach()[:, :, None].expand(-1, -1, 8, -1)), dim=-1)
        queries = self.corner_projection(corner_input.reshape(batch_size, 32, 7))
        prior_embedding = self._prior_embedding(observation, local_prior, local_seeds[:, 3], size, times)
        prior_by_query = torch.cat((torch.zeros_like(prior_embedding[:, None]).expand(-1, 3, -1),
                                    prior_embedding[:, None]), dim=1)
        queries = queries + prior_by_query.repeat_interleave(8, dim=1)
        context_norm = None
        if self.query_context:
            context = self._query_context(observation, batch, local_seeds[:, :3], history_valid, size)
            # 三个历史重建 query 不读当前上下文；q0 保持原上下文。
            context_by_query = torch.cat((torch.zeros_like(context[:, None]).expand(-1, 3, -1),
                                         context[:, None]), dim=1)
            queries = queries + context_by_query.repeat_interleave(8, dim=1)
            context_norm = context.detach().norm(dim=-1)
        local_diagnostics = {}
        if self.local_observation:
            local_increment, local_diagnostics = self.local_reader(
                observation, batch, seeds[:, 3:], query_valid[:, 3:])
            history_increment = local_increment.new_zeros(batch_size, 3, 8, 64)
            queries = queries + torch.cat((history_increment, local_increment), dim=1).reshape(batch_size, 32, 64)
        kv, source_valid = self.encode_sources(observation, history_valid)
        allowed = (source_valid[:, None].expand(-1, 4, -1)
                   & query_valid[:, :, None]).repeat_interleave(8, dim=1)
        corner_valid = query_valid.repeat_interleave(8, dim=1)
        bf16 = self.execution_profile == 'optimized_bf16'
        with feature_autocast(queries, enabled=bf16):
            decoded, _ = self.decoder(queries, None, kv, allowed,
                query_valid=corner_valid[..., None], zero_invalid_rows=True)
        decoded = feature_fp32(decoded, enabled=bf16)
        features = self.box_projection(decoded.reshape(batch_size, 4, 8 * 64))
        features = torch.where(query_valid[..., None], features, 0.)
        residual = self.pose_head(features)
        sine, cosine = residual[..., 3], residual[..., 4]
        nonzero = sine.square() + cosine.square() > 1e-12
        angle = torch.atan2(torch.where(nonzero, sine, 0.), torch.where(nonzero, cosine, 1.))
        yaw = local_seeds[..., 3] + angle
        yaw = torch.atan2(yaw.sin(), yaw.cos())
        boxes = torch.cat((output_reference + residual[..., :3], yaw[..., None]), dim=-1)
        boxes = transform_boxes(boxes, yaw_anchor, to_world=True)
        boxes = torch.where(query_valid[..., None], boxes, 0.)
        # 历史真实点可预测当前框；全空序列的 q0 输出原 prior。
        q0 = torch.where(observation.sequence_valid[:, None].bool(), boxes[:, 3], prior_box)
        # q0 quality 保留空序列监督，不能按 sequence_valid 屏蔽。
        quality = self.quality_head(features[:, 3]).squeeze(-1)
        return DecoderOutput(q0_box=q0, q0_quality_logits=quality,
            history_boxes=boxes[:, :3], query_context_norm=context_norm, **local_diagnostics)
