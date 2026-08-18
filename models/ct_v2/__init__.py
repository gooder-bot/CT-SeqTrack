"""Paper-facing CT-SeqTrack B0--B3 API plus isolated experimental B4."""

from models.ct_v2.evidence_memory import (
    apply_memory_control,
    build_box_memory_tokens,
    extension_target_bearing_mask,
    B2EvidenceAcquirer,
    B3SelectiveUpdater,
)
from models.ct_v2.pipeline_contracts import (
    ObservationOutput,
    MotionPriorOutput,
    EvidenceOutput,
    DecisionOutput,
    reexpress_motion_prior,
    motion_prior_covariance_xy,
    validate_motion_prior_support_alignment,
)
from models.ct_v2.pipeline import B0Observation, B1PhysicalTimePrior
from models.ct_v2.decoder_token_consistency import (
    DecoderTokenConsistencyLoss,
    GradientRatioWeightSelector,
)
from models.ct_v2.point_feature_consistency import (
    PointFeatureTemporalConsistencyLoss,
    canonicalize_points,
    chronological_frame_indices,
)


__all__ = [
    "apply_memory_control",
    "build_box_memory_tokens",
    "extension_target_bearing_mask",
    "B0Observation",
    "B1PhysicalTimePrior",
    "B2EvidenceAcquirer",
    "B3SelectiveUpdater",
    "ObservationOutput",
    "MotionPriorOutput",
    "EvidenceOutput",
    "DecisionOutput",
    "reexpress_motion_prior",
    "motion_prior_covariance_xy",
    "validate_motion_prior_support_alignment",
    # B4 is experimental-only and disabled in every formal configuration.
    "DecoderTokenConsistencyLoss",
    "GradientRatioWeightSelector",
    "PointFeatureTemporalConsistencyLoss",
    "canonicalize_points",
    "chronological_frame_indices",
]
