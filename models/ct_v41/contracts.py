"""v41 四候选与测量来源；位置效用、旧质量和可信写入不得混用。"""
from dataclasses import dataclass
from typing import Optional

from torch import Tensor


@dataclass
class PriorContext:
    feature: Tensor                 # [B,128] 可微时序表示
    mean_xy: Tensor                 # [B,2] 物理位移，非锚点纠偏
    log_sigma: Tensor               # [B,2] 平行/垂直 log std
    valid: Tensor                   # [B] 合法物理速度对
    acquisition_fraction: Tensor    # [B,2] 共用获取控制 u in [0,1]
    direction_xy: Tensor            # [B,2]
    kinematic_xy: Tensor            # [B,2]
    envelope: Tensor                # [B,2]
    unit_residual: Tensor           # [B,2]
    box: Tensor                     # [B,4] detached 物理端点/可信速度 fallback
    context_valid: Optional[Tensor] = None  # [B] 时序表示存在；独立于速度对有效性

@dataclass
class ObservationFeatures:
    coarse_box: Tensor              # [B,4]
    point_features: Tensor          # [B,4,N,64]
    source_tokens: Tensor           # [B,4,T,128]
    source_valid: Tensor            # [B,4,T]
    segmentation_logits: Tensor     # [B,4,N,2]
    bc_prediction: Tensor           # [B,4,N,9]
    foreground_probability: Tensor  # [B,4,N]
    quality: Tensor                 # [B,4] 当前/历史点量、熵和有效性摘要
    current_valid: Tensor           # [B] 当前 B0 至少一个真实唯一点
    sequence_valid: Tensor          # [B] 四帧中至少一个真实唯一点，允许仅历史预测
    coarse_features: Optional[Tensor] = None  # v34 [B,256] live Mini pooled 语义
    history_support: Optional[Tensor] = None  # v34 [B,3,3] detached 点量/框内软 FG 量/熵
    current_support: Optional[Tensor] = None  # v34 [B,3] detached 点量/全 crop 软 FG 量/熵


@dataclass
class DecoderOutput:
    q0_box: Tensor                  # [B,4] 世界轴最终框；无序列点时为 prior
    q0_quality_logits: Tensor       # [B] 原强 B1 quality，空序列仍监督
    history_boxes: Tensor           # [B,3,4]
    query_context_norm: Optional[Tensor] = None  # [B] detached
    local_neighbor_count: Optional[Tensor] = None  # [B,8] detached
    local_selected_count: Optional[Tensor] = None  # [B,8] detached
    local_mean_support: Optional[Tensor] = None  # [B,8] detached
    local_delta_norm: Optional[Tensor] = None  # [B,8] detached


@dataclass
class CandidateSet:
    boxes: Tensor                    # [B,4,4] q0 / prior / trusted CV / current-point R
    valid: Tensor                    # [B,4] 旧三候选内部去重；R 保留测量来源
    raw_valid: Tensor                # [B,4] 来源资格，不把 q0 fallback 当观测
    alias_of: Tensor                 # [B,4] -1 独立，其他为先保留的候选索引
    q0_is_fallback: Tensor
    q0_quality_logits: Tensor
    measurement_candidate: Tensor    # [B,4] q0/R 来源资格；不是 trusted 判定


@dataclass
class RecoveryOutput:
    box: Tensor                     # [B,4] 当前点修正后的公开框，XYZ 可微
    valid: Tensor                   # [B] seed 合法且实际读取至少一个当前唯一点
    seed_box: Tensor                # [B,4] capped trusted seed，不读取 GT
    normalized_delta: Tensor        # [B,3] seed-local、每轴一个 L/W/H 范围
    point_mask: Tensor              # [B,N] 实际局部读取点并集，原始输入顺序


@dataclass
class VerificationOutput:
    logits: Tensor                   # [B,4,3] S / P / T；prop 的 T 不监督、不使用
    support_mask: Tensor             # 当前唯一 raw ID、最终框内、predFG>=.5
    metadata: Tensor                 # [B,4,46]
    local_neighbor_count: Optional[Tensor] = None
    local_selected_count: Optional[Tensor] = None
    local_mean_support: Optional[Tensor] = None
    local_delta_norm: Optional[Tensor] = None
    local_point_mask: Optional[Tensor] = None

    @property
    def utility(self):
        return self.logits[..., :2].sigmoid().mean(-1)


@dataclass
class SelectionDecision:
    selected_index: Tensor
    accepted_box: Tensor
    output_utility: Tensor
    source: Tensor                   # 0=q0, 1=prior, 2=fallback, 3=R


@dataclass
class CommitDecision:
    trusted_score: Tensor            # 选中 q0/R 的 T；传播恒为 0
    legacy_q0_quality: Tensor         # 原 q0 quality，未改目标/梯度
    support_mask: Tensor             # [B,N] 选中测量候选最终框内的拟提交集合
    eligible: Tensor                # 选中 q0/R & T>=.5 & unique support>=3
    source: Tensor


@dataclass
class TrackOutput:
    accepted_box: Tensor
    selected_index: Tensor
    selected_quality: Tensor        # accepted 递推的旧 quality 输入，不是 T
    prior: PriorContext
    observation: ObservationFeatures
    decoder: DecoderOutput
    candidates: CandidateSet
    verification: Optional[VerificationOutput]
    selection_decision: SelectionDecision
    commit_decision: CommitDecision
    legacy_q0_quality: Tensor
    output_utility: Tensor
    commit_quality: Tensor           # [B,4] Tq0 / 0 / 0 / TR
    q0_is_fallback: Tensor
    recovery: Optional[RecoveryOutput] = None
