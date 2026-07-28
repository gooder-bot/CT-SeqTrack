"""Focused building blocks for the paper-facing CT-SeqTrack v2 path."""

from models.ct_v2.fusion import ProposalFusionGate
from models.ct_v2.motion import ContinuousTimeMotionEncoder
from models.ct_v2.point_feature_consistency import (
    PointFeatureTemporalConsistencyLoss,
    canonicalize_points,
    chronological_frame_indices,
)
from utils.ct_search import (
    build_time_guided_search_box,
    stratified_search_sample,
)

__all__ = [
    "ContinuousTimeMotionEncoder",
    "ProposalFusionGate",
    "PointFeatureTemporalConsistencyLoss",
    "canonicalize_points",
    "chronological_frame_indices",
    "build_time_guided_search_box",
    "stratified_search_sample",
]
