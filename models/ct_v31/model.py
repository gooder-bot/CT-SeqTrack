"""v31 薄组合：单次物理前向、一个最终框头、一次联合损失。"""

from dataclasses import replace

import torch
from torch import nn

from .config import normalize_config
from .contracts import EvidenceHypotheses, TrackOutput
from .prior import PhysicalTimePrior, empty_prior
from .observation import B0Observation, unique_valid_mask
from .decoder import SharedHypothesisDecoder
from .evidence import B2IdentityEvidence
from .losses import compute_losses


def canonicalize_batch(batch):
    """统一 raw-ID 有效性，防止 feature/监督使用不同的重复点分母。"""
    result = dict(batch)
    points = batch['points']
    if points.ndim != 4 or points.shape[1] != 4 or points.shape[-1] != 5:
        raise ValueError('v31 points must be [B,4,N,5]')
    # 历史不存在时其点槽也不存在；先合并存在性再去重，使 B0/B1/损失
    # 使用同一真实测量集合，而非仅由 observation 在内部额外过滤。
    valid = B0Observation.measurement_mask(batch)
    if valid.shape != points.shape[:-1]:
        raise ValueError('point_valid must align with points')
    if not bool(torch.isfinite(points.masked_select(valid[..., None].expand_as(points))).all()):
        raise ValueError('real v31 point observations must be finite')
    result['point_valid'] = valid
    result['points'] = points.masked_fill(~valid[..., None], 0.)
    if 'extension_valid' in batch:
        ext_valid = unique_valid_mask(batch['extension_valid'], batch.get('extension_ids'))
        result['extension_valid'] = ext_valid
        result['extension_points'] = batch['extension_points'].masked_fill(~ext_valid[..., None], 0.)
    return result


def select_hypothesis(decoder):
    """argmax 首索引打破并列，q0 永远是第零候选。"""
    if not bool(decoder.hypothesis_valid[:, 0].all()):
        raise ValueError('the main hypothesis must always exist, including prior-only frames')
    quality = decoder.quality_logits.sigmoid()
    score = quality.masked_fill(~decoder.hypothesis_valid, -torch.inf)
    if not bool(torch.isfinite(score[:, 0]).all()):
        raise FloatingPointError('nonfinite main hypothesis quality')
    if not bool(torch.isfinite(decoder.hypothesis_boxes).all()):
        raise FloatingPointError('nonfinite v31 box prediction')
    selected = score.argmax(1)
    row = torch.arange(len(selected), device=selected.device)
    return decoder.hypothesis_boxes[row, selected], selected, quality[row, selected]


class JointTracker(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = normalize_config(config)
        arm = self.config.v31_arm
        self.enable_b1 = arm != 'b0'
        self.enable_b2 = arm in ('b1_b2', 'full')
        self.enable_b3 = arm == 'full'
        # 公共模块首先构造，启用 B1/B2 不消耗公共初始化之前的 RNG。
        self.observation = B0Observation(token_count=128)
        self.decoder = SharedHypothesisDecoder(prior_dim=128, dropout=.2)
        self.prior = (PhysicalTimePrior(self.config.time_scale, self.config.v31_temporal_backend)
                      if self.enable_b1 else None)
        self.evidence = B2IdentityEvidence() if self.enable_b2 else None

    def plan_prior(self, batch):
        prepared = canonicalize_batch(batch)
        return self.prior(prepared) if self.prior is not None else empty_prior(prepared)

    def forward(self, batch, prior=None):
        prepared = canonicalize_batch(batch)
        if prior is None:
            prior = self.plan_prior(prepared)
        observation = self.observation(prepared)
        evidence = (self.evidence(observation, prepared, prior) if self.evidence is not None
                    else EvidenceHypotheses.empty(observation.coarse_box))
        decoder_evidence = (evidence if self.enable_b3 else
                            replace(evidence, mode_valid=torch.zeros_like(evidence.mode_valid)))
        decoder = self.decoder(observation, decoder_evidence, prepared, prior)
        accepted, selected, quality = select_hypothesis(decoder)
        return TrackOutput(accepted, selected, quality, prior, observation, evidence, decoder)

    def compute_losses(self, batch, output):
        return compute_losses(canonicalize_batch(batch), output, enable_b1=self.enable_b1,
                              enable_b2=self.enable_b2, enable_b3=self.enable_b3)
