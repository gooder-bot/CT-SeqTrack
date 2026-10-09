"""v37 候选与提交合同；旧 q0 质量和输出效用严格分开。"""
from dataclasses import dataclass
from typing import Optional

from torch import Tensor
from models.ct_v31.contracts import DecoderOutput, ObservationFeatures, PriorContext
from models.ct_v36.contracts import EvidenceHypotheses
from .identity import SCHEMA, RUNTIME_KEY


@dataclass
class CandidateSet:
    boxes: Tensor                   # [B,4,4] q0 / prior / mode0 / mode1
    valid: Tensor                   # q0=prior 的别名不参与 Full 竞争
    history_boxes: Tensor
    features: Tensor
    seed_boxes: Tensor              # [B,2,4] live mode center
    q0_quality_logits: Tensor
    q0_is_fallback: Tensor
    decoder: DecoderOutput          # 旧诊断布局 q0 / mode0 / mode1 / invalid
    pose_residual: Optional[Tensor] = None
    physical_context: Optional[Tensor] = None
    q0_pose_raw: Optional[Tensor] = None

    @property
    def mode_residual(self):
        return self.pose_residual

    @property
    def q0_raw_pose(self):
        return self.q0_pose_raw


@dataclass
class VerificationOutput:
    logits: Tensor                  # [B,4,3] S / P / mode提交质量
    support_mask: Tensor            # 最终框内预测 FG，尚未应用提交政策
    support_ids: Tensor
    support_xyz: Tensor
    support_probabilities: Tensor
    authorized: Tensor
    commit_support_mask: Optional[Tensor] = None
    support_source: Optional[Tensor] = None
    metadata: Optional[Tensor] = None
    local_neighbor_count: Optional[Tensor] = None
    local_selected_count: Optional[Tensor] = None
    local_mean_support: Optional[Tensor] = None
    local_delta_norm: Optional[Tensor] = None

    @property
    def utility(self):
        return self.logits[..., :2].sigmoid().mean(-1)


@dataclass
class TrackOutput:
    accepted_box: Tensor
    selected_index: Tensor
    selected_quality: Tensor        # [B] 真正用于提交的几何质量
    prior: PriorContext
    observation: ObservationFeatures
    evidence: EvidenceHypotheses
    decoder: DecoderOutput
    candidates: CandidateSet
    verification: VerificationOutput
    output_utility: Tensor          # [B] 被选择候选的 S/P 均值
    commit_quality: Tensor          # [B,4] q0旧头 / 0 / 两个新Qmode
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
