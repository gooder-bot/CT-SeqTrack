"""Focused building blocks for the paper-facing CT-SeqTrack v2 path."""

from models.ct_v2.fusion import (
    ProposalFusionGate,
    ReliabilityGatedProposalFusion,
)
from models.ct_v2.motion import (
    AdvantageGatedProposalFusion,
    ContinuousTimeMotionEncoder,
    JointProposalFusion,
    OrderedPhysicalMotionEncoder,
    OrderedTrajectoryEncoder,
    TrajectorySearchEvidence,
    TrajectorySearchEvidenceV21,
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
    build_trajectory_endpoint_search_box,
    build_time_guided_search_box,
    sample_padded_search_extension,
    sample_source_aware_endpoint_points,
    sample_search_extension,
    stratified_search_sample,
)

__all__ = [
    "AdvantageGatedProposalFusion",
    "ContinuousTimeMotionEncoder",
    "JointProposalFusion",
    "OrderedPhysicalMotionEncoder",
    "OrderedTrajectoryEncoder",
    "TrajectorySearchEvidence",
    "TrajectorySearchEvidenceV21",
    "TrajectoryPointEncoder",
    "ZeroInitTrajectoryAdapter",
    "ProposalFusionGate",
    "ReliabilityGatedProposalFusion",
    "PointFeatureTemporalConsistencyLoss",
    "canonicalize_points",
    "chronological_frame_indices",
    "build_ordered_trajectory_search_box",
    "build_trajectory_endpoint_search_box",
    "build_time_guided_search_box",
    "sample_padded_search_extension",
    "sample_source_aware_endpoint_points",
    "sample_search_extension",
    "stratified_search_sample",
]
