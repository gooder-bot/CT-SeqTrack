"""v32 张量接口。公开框为 anchor 平移后的世界轴 XYZ + 绝对 yaw(rad)。

box_size 始终为物体轴 length/width/height；历史按最旧到最新排列。
raw ID 只在同一原始帧内有意义。padding 的 ID=-1、valid=False。
记忆点在各自保存框的物体坐标中规范化，不能当作当前世界坐标。
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

SCHEMA = "ct_seqtrack.joint_identity.v32"


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


@dataclass
class EvidenceHypotheses:
    centers_xyz: Tensor             # [B,3,3] live votes/weights 聚合
    mode_valid: Tensor              # [B,3]
    members: Tensor                 # [B,3,256] 离散固定成员
    covariance_xy: Tensor           # [B,3,2,2] 仅作 detached 几何摘要
    point_features: Tensor          # [B,256,64] pre-vote，不包含 live vote 坐标
    point_xyz: Tensor               # [B,256,3] 真实原始点坐标
    point_valid: Tensor             # [B,256]
    point_ids: Tensor               # [B,256]
    point_indices: Tensor           # [B,256] 原 768 池索引；无效槽 -1
    identity_logits: Tensor         # [B,768]
    selected_identity_logits: Tensor  # [B,256]
    vote_xyz: Tensor                # [B,256,3]
    reliability_logits: Tensor      # [B,256]
    memory_features: Tensor         # [B,36,64] 当前参数重编码
    memory_valid: Tensor            # [B,36]

    @classmethod
    def empty(cls, reference: Tensor):
        b, device = reference.shape[0], reference.device
        z = lambda *shape: reference.new_zeros((b, *shape))
        mask = lambda *shape: torch.zeros((b, *shape), device=device, dtype=torch.bool)
        ids = lambda *shape: torch.full((b, *shape), -1, device=device, dtype=torch.long)
        return cls(z(3, 3), mask(3), mask(3, 256), z(3, 2, 2),
                   z(256, 64), z(256, 3), mask(256), ids(256), ids(256),
                   z(768), z(256), z(256, 3), z(256), z(36, 64), mask(36))


@dataclass
class DecoderOutput:
    hypothesis_boxes: Tensor        # [B,4,4] q0, mode0, mode1, mode2
    quality_logits: Tensor          # [B,4]
    hypothesis_valid: Tensor        # [B,4]
    history_boxes: Tensor           # [B,3,4]
    decoder_features: Tensor        # [B,4,64]


@dataclass
class TrackOutput:
    accepted_box: Tensor            # [B,4] 唯一候选选择输出
    selected_index: Tensor          # [B]
    selected_quality: Tensor        # [B]
    prior: PriorContext
    observation: ObservationFeatures
    evidence: EvidenceHypotheses
    decoder: DecoderOutput

    @property
    def hypothesis_boxes(self):
        return self.decoder.hypothesis_boxes

    @property
    def quality_logits(self):
        return self.decoder.quality_logits

    @property
    def hypothesis_valid(self):
        return self.decoder.hypothesis_valid
