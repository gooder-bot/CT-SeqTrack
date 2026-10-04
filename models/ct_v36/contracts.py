"""v36 权限接口。事件/锚世界框使用 float64；网络公开框为平移局部世界轴。"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import Tensor
from models.ct_v31.contracts import ObservationFeatures

SCHEMA = 'ct_seqtrack.joint_identity.v36'
RUNTIME_KEY = 'ct_v36_runtime'


@dataclass
class MeasurementEvent:
    box: np.ndarray
    frame: int
    sensor_time: float
    effective_time: float
    geometry: np.ndarray
    source: int = 0
    initial: bool = False
    segment: int = 0
    edge_valid: bool = False


@dataclass
class PriorPrediction:
    feature: Tensor
    event_feature: Tensor
    box: Tensor
    world_box: Tensor
    mean_xy: Tensor
    origin_world: Tensor
    origin_frame: Tensor
    horizon: Tensor
    velocity_age: Tensor
    axes: Tensor
    prior_valid: Tensor
    motion_valid: Tensor
    range_anchor: Tensor
    range_rate: Tensor
    range_xy: Tensor
    range_for_loss: Tensor
    geometry: Tensor
    initial: Tensor
    velocity: Tensor
    acceleration: Tensor
    tau: Tensor
    kinematic_xy: Optional[Tensor] = None
    envelope: Optional[Tensor] = None
    unit_residual: Optional[Tensor] = None
    legacy_valid: Optional[Tensor] = None

    @property
    def valid(self):
        return self.motion_valid

    @property
    def direction_xy(self):
        return self.axes[..., 0]


@dataclass
class EvidenceHypotheses:
    centers_xyz: Tensor
    mode_valid: Tensor
    members: Tensor
    covariance_xy: Tensor
    point_features: Tensor
    point_xyz: Tensor
    point_valid: Tensor
    point_ids: Tensor
    point_indices: Tensor
    identity_logits: Tensor
    selected_identity_logits: Tensor
    vote_xyz: Tensor
    reliability_logits: Tensor
    memory_features: Tensor
    memory_valid: Tensor

    @classmethod
    def empty(cls, reference):
        b, device = len(reference), reference.device
        z = lambda *s: reference.new_zeros((b, *s))
        mask = lambda *s: torch.zeros((b, *s), device=device, dtype=torch.bool)
        ids = lambda *s: torch.full((b, *s), -1, device=device, dtype=torch.long)
        return cls(z(2, 3), mask(2), mask(2, 256), z(2, 2, 2), z(256, 64),
                   z(256, 3), mask(256), ids(256), ids(256), z(768), z(256),
                   z(256, 3), z(256), z(36, 64), mask(36))


@dataclass
class CandidateSet:
    boxes: Tensor
    valid: Tensor
    history_boxes: Tensor
    features: Tensor
    seed_boxes: Tensor

    @property
    def hypothesis_boxes(self):
        return self.boxes

    @property
    def hypothesis_valid(self):
        return self.valid


@dataclass
class VerificationOutput:
    logits: Tensor
    support_mask: Tensor
    support_ids: Tensor
    support_xyz: Tensor
    support_probabilities: Tensor
    authorized: Tensor

    @property
    def utility(self):
        return self.logits[..., :2].sigmoid().mean(-1)

    @property
    def measurement_score(self):
        score = self.logits[..., 2].sigmoid()
        return score * score.new_tensor([1., 0., 1., 1.])


@dataclass
class CommitDecision:
    output_index: Tensor
    measurement_index: Tensor
    memory_eligible: Tensor


@dataclass
class TrackOutput:
    accepted_box: Tensor
    selected_index: Tensor
    selected_quality: Tensor
    prior: PriorPrediction
    observation: ObservationFeatures
    evidence: EvidenceHypotheses
    candidates: CandidateSet
    verification: VerificationOutput
    decision: CommitDecision

    @property
    def decoder(self):
        return self.candidates

    @property
    def hypothesis_boxes(self):
        return self.candidates.boxes

    @property
    def hypothesis_valid(self):
        return self.candidates.valid

    @property
    def measurement_index(self):
        return self.decision.measurement_index
