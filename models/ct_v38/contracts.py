"""v38 三候选与两项决策；位置效用、旧质量和可信写入不得混用。"""
from dataclasses import dataclass
from typing import Optional

from torch import Tensor
from models.ct_v31.contracts import DecoderOutput, ObservationFeatures, PriorContext, EvidenceHypotheses


@dataclass
class CandidateSet:
    boxes: Tensor                    # [B,3,4] real q0 / raw prior / trusted CV fallback
    valid: Tensor                    # [B,3] 去重后候选
    raw_valid: Tensor                # [B,3] 来源资格，不把 q0 fallback 当观测
    alias_of: Tensor                 # [B,3] -1 独立，其他为先保留的候选索引
    q0_is_fallback: Tensor
    q0_quality_logits: Tensor
    decoder: DecoderOutput           # 原强 B1 四槽输出只供原损失/兼容日志


@dataclass
class VerificationOutput:
    logits: Tensor                   # [B,3,3] S / P / T；prop 的 T 不监督、不使用
    support_mask: Tensor             # 当前唯一 raw ID、最终框内、predFG>=.5
    support_ids: Tensor
    support_xyz: Tensor
    support_probabilities: Tensor
    authorized: Tensor
    metadata: Tensor                 # [B,3,45]
    commit_support_mask: Optional[Tensor] = None
    support_source: Optional[Tensor] = None
    local_neighbor_count: Optional[Tensor] = None
    local_selected_count: Optional[Tensor] = None
    local_mean_support: Optional[Tensor] = None
    local_delta_norm: Optional[Tensor] = None

    @property
    def utility(self):
        return self.logits[..., :2].sigmoid().mean(-1)


@dataclass
class SelectionDecision:
    selected_index: Tensor
    accepted_box: Tensor
    output_utility: Tensor
    source: Tensor                   # 0=q0, 1=prior, 2=fallback


@dataclass
class CommitDecision:
    trusted_score: Tensor            # 选中真实 q0 的 T；传播恒为 0
    legacy_q0_quality: Tensor         # 原 q0 quality，未改目标/梯度
    support_mask: Tensor             # [B,N] 只有选中真实 q0 的拟提交集合
    eligible: Tensor                # 真实选中 q0 & T>=.5 & unique support>=3
    source: Tensor


@dataclass
class TrackOutput:
    accepted_box: Tensor
    selected_index: Tensor
    selected_quality: Tensor        # 旧 host 兼容量，不是 T
    prior: PriorContext
    observation: ObservationFeatures
    evidence: EvidenceHypotheses
    decoder: DecoderOutput
    candidates: CandidateSet
    verification: Optional[VerificationOutput]
    selection_decision: SelectionDecision
    commit_decision: CommitDecision
    legacy_q0_quality: Tensor
    output_utility: Tensor
    commit_quality: Tensor           # [B,3] Tq0 / 0 / 0
    q0_is_fallback: Tensor

    @property
    def hypothesis_boxes(self):
        return self.candidates.boxes

    @property
    def hypothesis_valid(self):
        return self.candidates.valid

    @property
    def quality_logits(self):
        return self.decoder.quality_logits
