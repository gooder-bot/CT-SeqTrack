"""Focused building blocks for the paper-facing CT-SeqTrack v2 path."""

from models.ct_v2.fusion import ProposalFusionGate
from models.ct_v2.motion import (
    ContinuousTimeMotionEncoder,
    OrderedTrajectoryEncoder,
    TrajectoryPointEncoder,
    ZeroInitTrajectoryAdapter,
)
from models.ct_v2.point_feature_consistency import (
    PointFeatureTemporalConsistencyLoss,
    canonicalize_points,
    chronological_frame_indices,
)
from utils.ct_search import (
    build_ordered_trajectory_search_box,
    build_time_guided_search_box,
    sample_search_extension,
    stratified_search_sample,
)

__all__ = [
    "ContinuousTimeMotionEncoder",
    "OrderedTrajectoryEncoder",
    "TrajectoryPointEncoder",
    "ZeroInitTrajectoryAdapter",
    "ProposalFusionGate",
    "PointFeatureTemporalConsistencyLoss",
    "canonicalize_points",
    "chronological_frame_indices",
    "build_ordered_trajectory_search_box",
    "build_time_guided_search_box",
    "sample_search_extension",
    "stratified_search_sample",
]
