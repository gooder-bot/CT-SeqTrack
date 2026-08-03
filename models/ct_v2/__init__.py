"""Focused building blocks for the paper-facing CT-SeqTrack v2 path."""

from models.ct_v2.fusion import (
    ProposalFusionGate,
    ReliabilityGatedProposalFusion,
)
from models.ct_v2.motion import (
    AdvantageGatedProposalFusion,
    ClosedLoopRiskAwareProposalRouter,
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
from models.ct_v2.crpa import (
    CRPA_ROLLOUT_SCHEMA,
    CRPA_ROUTER_SCHEMA,
    counterfactual_gain_targets,
    crpa_router_loss,
    stable_tracklet_partition,
)
from models.ct_v2.selective_innovation import (
    MotionConditionedSearchRefiner,
    SELECTIVE_ROLLOUT_SCHEMA,
    SELECTIVE_ROUTER_SCHEMA,
    SignedHorizonInnovationRouter,
    calibrate_gain_threshold,
    discounted_tracking_cost,
    signed_horizon_router_loss,
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
    "ClosedLoopRiskAwareProposalRouter",
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
    "CRPA_ROLLOUT_SCHEMA",
    "CRPA_ROUTER_SCHEMA",
    "counterfactual_gain_targets",
    "crpa_router_loss",
    "stable_tracklet_partition",
    "MotionConditionedSearchRefiner",
    "SELECTIVE_ROLLOUT_SCHEMA",
    "SELECTIVE_ROUTER_SCHEMA",
    "SignedHorizonInnovationRouter",
    "calibrate_gain_threshold",
    "discounted_tracking_cost",
    "signed_horizon_router_loss",
]
