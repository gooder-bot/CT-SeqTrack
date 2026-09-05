from datasets import points_utils
from models import base_model
from models.backbone.pointnet import MiniPointNet, SegPointNet, FeaturePointNet
from models.attn.Models import Seq2SeqFormer

import copy
import contextlib
import hashlib
import json
import math
import numpy as np
import subprocess
import time
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate

from datasets.sampler import motion_processing_mf

from torchmetrics import Accuracy

from datasets.misc_utils import (
    build_effective_time_fields,
    build_time_fields,
    create_corner_timestamps_from_deltas,
    get_history_frame_ids_and_masks,
    get_last_n_bounding_boxes,
    get_tensor_corners_batch,
    normalize_dynamics_time_mode,
)
from models.dynamics import (
    DynamicsEncoder,
    DynamicsResidualGate,
    ZeroInitPhysicalTimeAdapter,
    apply_proposal_innovation,
    clamp_vector_norm,
)
from models.observability import ObservabilityGate
from models.time_encoding import TimeEncoding
from models.ct_v2.motion import (
    AdvantageGatedProposalFusion,
    ClosedLoopRiskAwareProposalRouter,
    ContinuousTimeMotionEncoder,
    JointProposalFusion,
    OrderedTrajectoryEncoder,
    TrajectorySearchEvidence,
    TrajectorySearchEvidenceV21,
    TrajectoryPointEncoder,
    ZeroInitTrajectoryAdapter,
    acquisition_margin_pinball_loss,
    physical_motion_uncertainty_loss,
    recursive_gap_age_balanced_mean,
)
from models.ct_v2.fusion import (
    ProposalFusionGate,
    ReliabilityGatedProposalFusion,
)
from models.ct_v2.joint_full import (
    counterfactual_query_targets,
    JointFullSearchRefiner,
    JointScalarResidualRouter,
)
from models.ct_v2.evidence_memory import (
    build_box_memory_tokens,
    apply_memory_control,
    extension_target_bearing_mask,
    B2EvidenceAcquirer,
    B3SelectiveUpdater,
)
from models.ct_v2.pipeline import B0Observation, B1PhysicalTimePrior
from models.ct_v2.decoder_token_consistency import (
    DecoderTokenConsistencyLoss,
    GradientRatioWeightSelector,
)
from models.ct_v2.point_feature_consistency import (
    PointFeatureTemporalConsistencyLoss,
)
from models.ct_v2.selective_innovation import (
    ActionConsistentInnovationRouter,
    AsymmetricDualQueryAdapter,
    MotionConditionedSearchRefiner,
    StateAlignedSearchRefiner,
    SignedHorizonInnovationRouter,
    validate_b2_v3_router_package,
    validate_trainable_parameter_prefixes,
    require_nonzero_finite_gradient,
)
from models.ct_v2.contracts import (
    build_search_usable_mask,
    resolve_observation_delta_t,
)
from models.ct_v2.pipeline_contracts import (
    MotionPriorOutput,
    EvidenceOutput,
    DecisionOutput,
    reexpress_motion_prior,
    validate_motion_prior_support_alignment,
)
from utils.replay_cache import (
    B0_STATE_PREFIXES,
    B1_STATE_PREFIXES,
    b1_calibration_config_sha256,
    b2_candidate_config_sha256,
    replay_config_sha256,
    sha256_json,
    tensor_prefixes_sha256,
    validate_b1_calibration_state,
    validate_replay_cache_manifest,
)
from utils.training_isolation import (
    CheckpointableRNG,
    advance_lightning_manual_transaction,
    assert_disjoint_parameter_sets,
    candidate_stratified_mean,
    capture_global_rng_state,
    contract_v3_action_probability,
    freeze_batchnorm_running_stats,
    isolated_constructor_rng,
    partition_named_parameter_groups,
    partition_candidate_view_items,
    restore_global_rng_state,
    update_cumulative_binary_class_balance,
    weighted_candidate_sum,
)
from utils.online_contract import (
    build_online_resume_contract,
    online_candidate_state_consistent,
)
from utils.metrics import estimateOverlap
from utils.action_calibration import require_selective_calibration
from utils.recursive_state import (
    build_recursive_input_contract,
    commit_canonical_prediction,
    RecursiveTrackState,
    rotating_rollout_horizon,
)
from utils.sampling_utils import stable_uint32_seed

# import vis_tool as vt


def _build_binary_segmentation_accuracy():
    """Create the metric across both legacy and current torchmetrics APIs."""
    try:
        return Accuracy(task='multiclass', num_classes=2, average='none')
    except (TypeError, AssertionError):
        # torchmetrics <=0.9 has no ``task`` dispatcher and otherwise treats
        # the first unknown keyword as part of the legacy constructor path.
        return Accuracy(num_classes=2, average='none')

class SEQTRACK3D(base_model.MotionBaseModelMF):
    B2_V3_FROZEN_PREFIXES = (
        "seg_pointnet.", "mini_pointnet.", "motion_mlp.",
        "motion_state_mlp.", "feature_pointnet.", "Transformer.",
        "physical_motion_encoder.",
    )

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.hist_num = getattr(config, 'hist_num', 1)
        self.seg_acc = _build_binary_segmentation_accuracy()
        self.ct_enable_v27 = bool(getattr(config, 'ct_enable_v27', False))

        self.box_aware = getattr(config, 'box_aware', False)
        self.use_motion_cls = getattr(config, 'use_motion_cls', True)
        self.use_dynamics_encoder = getattr(config, 'use_dynamics_encoder', False)
        if bool(getattr(config, 'dynamics_use_acceleration', False)):
            raise ValueError(
                "dynamics_use_acceleration is not implemented or consumed by "
                "DynamicsEncoder; keep it false to avoid a misleading ablation.")
        self.use_observability_gate = getattr(config, 'use_observability_gate', False)
        self.use_ct_v2 = bool(getattr(config, 'use_ct_v2', False))
        self.use_ct_joint_full = bool(getattr(
            config, 'use_ct_joint_full', False))
        self.ct_joint_contract_version = int(getattr(
            config, 'ct_joint_contract_version', 1))
        if self.ct_joint_contract_version not in (1, 2, 3):
            raise ValueError("ct_joint_contract_version must be 1, 2 or 3")
        self.ct_enable_b1 = bool(getattr(config, 'ct_enable_b1', True))
        self.ct_enable_b2 = bool(getattr(config, 'ct_enable_b2', True))
        self.ct_enable_b3 = bool(getattr(config, 'ct_enable_b3', True))
        self.ct_enable_v26_recovery = bool(getattr(
            config, 'ct_enable_v26_recovery', False))
        self.ct_initialization_policy = str(getattr(
            config, 'ct_initialization_policy',
            getattr(config, 'ct_b0_initialization_policy', 'legacy')))
        if (self.ct_joint_contract_version >= 3 and self.ct_enable_b3
                and not bool(getattr(
                    config, 'ct_b2_promotion_passed', False))
                and self.ct_initialization_policy != 'scratch_only'):
            raise ValueError(
                "contract-v3 B3 requires ct_b2_promotion_passed=True")
        self.ct_separate_optimizers = bool(getattr(
            config, 'ct_separate_optimizers', False))
        self.ct_runtime_protocol = str(getattr(
            config, 'ct_runtime_protocol', 'legacy')).strip().lower()
        self.ct_optimizer_topology = str(getattr(
            config, 'ct_optimizer_topology',
            'isolated_manual' if self.ct_separate_optimizers
            else 'legacy_single')).strip().lower()
        self.ct_unified_auto = (
            self.ct_runtime_protocol == 'safe_seqtrack_auto_v1'
            and self.ct_optimizer_topology == 'unified_auto')
        self.ct_cuda_stage_audit = bool(getattr(
            config, 'ct_cuda_stage_audit', False))
        self._ct_cuda_stage_current = {}
        self._ct_cuda_stage_max = {}
        self.ct_observation_fingerprint_steps = int(getattr(
            config, 'ct_observation_fingerprint_steps', 0))
        self._ct_observation_batch_fingerprints = []
        if self.ct_unified_auto and self.ct_separate_optimizers:
            raise ValueError(
                "safe unified automatic optimization cannot enable "
                "ct_separate_optimizers")
        self.ct_manual_amp_enabled = bool(getattr(
            config, 'ct_manual_amp_enabled', False))
        self.ct_b0_rng_shift_control = bool(getattr(
            config, 'ct_b0_rng_shift_control', False))
        self.ct_b0_candidate_views = int(getattr(
            config, 'ct_b0_candidate_views', 1))
        self.ct_b0_candidate_weights = tuple(float(value) for value in getattr(
            config, 'ct_b0_candidate_weights', [1.0]))
        self.ct_b2_candidate_views = int(getattr(
            config, 'ct_b2_candidate_views', 1))
        if self.ct_b0_candidate_views <= 0:
            raise ValueError("ct_b0_candidate_views must be positive")
        if len(self.ct_b0_candidate_weights) != self.ct_b0_candidate_views:
            raise ValueError(
                "ct_b0_candidate_weights must match ct_b0_candidate_views")
        if (any(value < 0.0 for value in self.ct_b0_candidate_weights)
                or not math.isclose(
                    sum(self.ct_b0_candidate_weights), 1.0,
                    rel_tol=0.0, abs_tol=5e-6)):
            raise ValueError(
                "ct_b0_candidate_weights must be non-negative and sum to 1")
        if self.ct_b2_candidate_views != 1:
            raise ValueError("CT-SeqTrack B2 accepts exactly one canonical view")
        self.ct_targetness_class_weight_source = str(getattr(
            config, 'ct_targetness_class_weight_source',
            'online_canonical_preflight')).strip().lower()
        initial_positive_weight = float(getattr(
            config, 'ct_targetness_positive_weight', 1.0))
        initial_negative_weight = float(getattr(
            config, 'ct_targetness_negative_weight', 1.0))
        if (self.ct_joint_contract_version >= 3 and self.ct_enable_b2
                and self.ct_targetness_class_weight_source
                != 'online_canonical_preflight'):
            raise ValueError(
                "contract-v3 B2 targetness balancing must use the cumulative "
                "canonical training population")
        if (not math.isfinite(initial_positive_weight)
                or not math.isfinite(initial_negative_weight)
                or initial_positive_weight <= 0.0
                or initial_negative_weight <= 0.0):
            raise ValueError("initial targetness class weights must be positive")
        if self.ct_joint_contract_version >= 3 and self.ct_enable_b2:
            self.register_buffer(
                'ct_targetness_running_positive_points',
                torch.zeros((), dtype=torch.float64))
            self.register_buffer(
                'ct_targetness_running_negative_points',
                torch.zeros((), dtype=torch.float64))
            self.register_buffer(
                'ct_targetness_running_positive_weight',
                torch.tensor(initial_positive_weight, dtype=torch.float64))
            self.register_buffer(
                'ct_targetness_running_negative_weight',
                torch.tensor(initial_negative_weight, dtype=torch.float64))
        if self.ct_separate_optimizers:
            self.automatic_optimization = False
        elif self.ct_unified_auto:
            self.automatic_optimization = True
        self.ct_plugin_rng = (
            CheckpointableRNG(
                int(getattr(config, 'seed', 42) or 42) + 24001)
            if (self.use_ct_joint_full
                and self.ct_joint_contract_version >= 3) else None)
        self.ct_auxiliary_rng = (
            CheckpointableRNG(
                int(getattr(config, 'seed', 42) or 42) + 24002)
            if (self.ct_joint_contract_version >= 3
                and not self.ct_unified_auto
                and self.ct_b0_candidate_views > 1) else None)
        self.ct_memory_control_rng = (
            CheckpointableRNG(
                int(getattr(config, 'seed', 42) or 42) + 24003)
            if (self.use_ct_joint_full
                and self.ct_joint_contract_version >= 3) else None)
        self._ct_scalers = {}
        # Compatibility attributes for old checkpoint inspectors.
        self._ct_b0_scaler = None
        self._ct_plugin_scaler = None
        self._ct_pending_scaler_state = None
        if self.ct_separate_optimizers or self.ct_unified_auto:
            for module_name in ('b0', 'b1', 'b2', 'b3'):
                self.register_buffer(
                    f'ct_{module_name}_update_step',
                    torch.zeros((), dtype=torch.long))
            self.register_buffer(
                'ct_plugin_update_step', torch.zeros((), dtype=torch.long))
        self._ct_epoch_boundary_complete = False
        method_promotion = getattr(
            config, 'ct_b2_method_promotion_manifest', None)
        if isinstance(method_promotion, dict):
            self._ct_b2_method_promotion = copy.deepcopy(method_promotion)
        preflight = getattr(
            config, 'ct_acquisition_preflight_manifest', None)
        if isinstance(preflight, dict):
            self._ct_acquisition_preflight = copy.deepcopy(preflight)
        self.ct_enable_shared_motion_anchor = bool(getattr(
            config, 'ct_enable_shared_motion_anchor', True))
        self.ct_enable_dynamic_residual_bound = bool(getattr(
            config, 'ct_enable_dynamic_residual_bound', True))
        self.ct_enable_query_reliability_gate = bool(getattr(
            config, 'ct_enable_query_reliability_gate', True))
        self.ct_query_dim = int(getattr(config, 'ct_query_dim', 64))
        if self.ct_query_dim <= 0:
            raise ValueError("ct_query_dim must be positive")
        self.ct_expansion_point_count = int(getattr(
            config, 'ct_expansion_point_count', 256))
        self.ct_endpoint_quota = int(getattr(
            config, 'ct_endpoint_quota', 128))
        self.ct_tube_quota = int(getattr(config, 'ct_tube_quota', 128))
        self.ct_corridor_quota = int(getattr(
            config, 'ct_corridor_quota', 0))
        self.use_b1motion_v3 = bool(getattr(
            config, 'use_b1motion_v3', False))
        if self.use_ct_joint_full:
            if not self.use_b1motion_v3:
                raise ValueError("CT joint Full requires use_b1motion_v3=True")
            expected_expansion_count = (
                self.ct_endpoint_quota + self.ct_tube_quota
                + (self.ct_corridor_quota
                   if self.ct_enable_v26_recovery else 0))
            if self.ct_expansion_point_count != expected_expansion_count:
                raise ValueError(
                    "CT acquisition quotas must equal expansion point count")
            if min(self.ct_endpoint_quota, self.ct_tube_quota) <= 0:
                raise ValueError("CT endpoint/tube quotas must be positive")
            if (self.ct_enable_v26_recovery
                    and self.ct_corridor_quota <= 0):
                raise ValueError("v26 corridor quota must be positive")
            if not (self.ct_enable_shared_motion_anchor
                    and self.ct_enable_dynamic_residual_bound):
                raise ValueError(
                    "CT joint Full requires shared anchor and dynamic bound")
        self.use_calibrated_motion_uncertainty = bool(getattr(
            config, 'use_calibrated_motion_uncertainty', False))
        self.require_b1_calibration_passed = bool(getattr(
            config, 'require_b1_calibration_passed', False))
        self.require_b1_calibration_artifact = bool(getattr(
            config, 'require_b1_calibration_artifact', False))
        if (self.use_calibrated_motion_uncertainty
                and not self.use_b1motion_v3):
            raise ValueError(
                "calibrated motion uncertainty requires B1motion-v3")
        self.use_motion_v3_legacy_fusion = bool(getattr(
            config, 'use_motion_v3_legacy_fusion', True))
        self.use_search_evidence_v2 = bool(getattr(
            config, 'use_search_evidence_v2', False))
        self.use_joint_proposal_fusion = bool(getattr(
            config, 'use_joint_proposal_fusion', False))
        self.use_search_evidence_v21 = bool(getattr(
            config, 'use_search_evidence_v21', False))
        self.use_advantage_proposal_fusion = bool(getattr(
            config, 'use_advantage_proposal_fusion', False))
        self.use_b3_risk_router = bool(getattr(
            config, 'use_b3_risk_router', False))
        self.use_motion_conditioned_search_v22 = bool(getattr(
            config, 'use_motion_conditioned_search_v22', False))
        self.use_signed_horizon_router = bool(getattr(
            config, 'use_signed_horizon_router', False))
        self.use_motion_conditioned_search_v3 = bool(getattr(
            config, 'use_motion_conditioned_search_v3', False))
        self.use_asymmetric_dual_query = bool(getattr(
            config, 'use_asymmetric_dual_query', False))
        self.use_raw_search_candidate = bool(getattr(
            config, 'use_raw_search_candidate', False))
        self.use_b1_prepass_support = bool(getattr(
            config, 'use_b1_prepass_support', False))
        if (self.use_ct_joint_full
                and self.ct_joint_contract_version >= 2
                and not self.ct_enable_b1
                and self.use_b1_prepass_support):
            raise ValueError(
                "-B1/B0 must disable B1 prepass support so Search geometry "
                "uses the deterministic kinematic fallback")
        self.use_recursive_replay_cache = bool(getattr(
            config, 'use_recursive_replay_cache', False))
        self.use_uncertainty_geometry = bool(getattr(
            config, 'use_uncertainty_geometry', False))
        if self.use_ct_joint_full and any((
                self.use_search_evidence_v2,
                self.use_search_evidence_v21,
                self.use_motion_conditioned_search_v22,
                self.use_motion_conditioned_search_v3,
                bool(getattr(config, 'use_action_consistent_router_v3', False)),
                (self.use_b1_prepass_support
                 and self.ct_joint_contract_version < 2),
                self.use_recursive_replay_cache)):
            raise ValueError(
                "CT joint Full is isolated from legacy B2/B3/prepass/replay paths")
        if ((self.use_asymmetric_dual_query
             or self.use_raw_search_candidate
             or self.use_b1_prepass_support
             or self.use_uncertainty_geometry)
                and not self.use_motion_conditioned_search_v3
                and not (self.use_ct_joint_full
                         and self.ct_joint_contract_version >= 2)):
            raise ValueError(
                "new B1/B2 coupling requires motion-conditioned search v3")
        self.use_action_consistent_router_v3 = bool(getattr(
            config, 'use_action_consistent_router_v3', False))
        self.b2_v3_freeze_candidate_producers = bool(getattr(
            config, 'b2_v3_freeze_candidate_producers', False))
        self.b2_v3_require_packaged_router = bool(getattr(
            config, 'b2_v3_require_packaged_router', False))
        self.v22_freeze_candidate_producers = bool(getattr(
            config, 'v22_freeze_candidate_producers', False))
        self.signed_router_enabled_scale = float(getattr(
            config, 'signed_router_enabled_scale', 1.0))
        if not 0.0 <= self.signed_router_enabled_scale <= 1.0:
            raise ValueError("signed_router_enabled_scale must be in [0,1]")
        self.router_v3_enabled_scale = float(getattr(
            config, 'router_v3_enabled_scale', 1.0))
        if not 0.0 <= self.router_v3_enabled_scale <= 1.0:
            raise ValueError("router_v3_enabled_scale must be in [0,1]")
        self.b3_enabled_scale = float(getattr(
            config, 'b3_enabled_scale', 1.0))
        if not 0.0 <= self.b3_enabled_scale <= 1.0:
            raise ValueError("b3_enabled_scale must be in [0,1]")
        self.proposal_inference_mode = str(getattr(
            config, 'proposal_inference_mode', 'full')).strip().lower()
        if self.proposal_inference_mode not in (
                'obs', 'obs_motion', 'obs_search', 'full',
                'obs_motion_search', 'full_selective',
                'obs_only', 'obs_vs_motion', 'obs_vs_refined',
                'obs_vs_all', 'observation', 'motion', 'raw_search',
                'legacy_clipped', 'selective', 'bounded_always'):
            raise ValueError(
                "proposal_inference_mode must be obs, obs_motion, "
                "obs_search, full, obs_motion_search, full_selective, or "
                "one of observation/motion/raw_search/legacy_clipped/selective")
        self.use_ordered_trajectory_encoder = bool(getattr(
            config, 'use_ordered_trajectory_encoder', False))
        self.use_trajectory_search = bool(getattr(
            config, 'use_trajectory_search', False))
        self.use_trajectory_adapter = bool(getattr(
            config, 'use_trajectory_adapter', False))
        self.use_time_guided_search = bool(getattr(
            config, 'use_time_guided_search', False))
        self.use_physical_time_adapter = bool(
            getattr(config, 'use_physical_time_adapter', False))
        self.use_point_feature_tc = bool(getattr(
            config, 'use_point_feature_tc', False))
        self.use_decoder_token_consistency = bool(getattr(
            config, 'use_decoder_token_consistency', False))
        self.use_b4_paired_views = bool(getattr(
            config, 'use_b4_paired_views', False))
        self.pftc_weight = float(getattr(config, 'pftc_weight', 1.0))
        self.pftc_ramp_epochs = int(getattr(
            config, 'pftc_ramp_epochs', 5))
        self.pftc_time_field = str(getattr(
            config, 'pftc_time_field', 'effective')).strip().lower()
        if self.pftc_time_field not in ('effective', 'real'):
            raise ValueError("pftc_time_field must be 'effective' or 'real'")
        if self.pftc_weight < 0:
            raise ValueError("pftc_weight must be non-negative")
        if self.pftc_ramp_epochs < 0:
            raise ValueError("pftc_ramp_epochs must be non-negative")
        if self.use_point_feature_tc and self.use_b4_paired_views:
            raise ValueError(
                "PFTC and B4 paired decoder consistency are separate "
                "experimental objectives")
        if self.use_decoder_token_consistency:
            if not self.use_b4_paired_views:
                raise ValueError(
                    "decoder-token consistency requires b4_paired_views")
            if self.use_point_feature_tc:
                raise ValueError(
                    "decoder-token consistency and PFTC are exclusive")
        self.obs_gate_fusion_mode = str(getattr(config, 'obs_gate_fusion_mode', 'feature')).lower()
        self.obs_gate_fusion_mode = self.obs_gate_fusion_mode.replace('-', '_')
        if self.obs_gate_fusion_mode == 'conf_res':
            self.obs_gate_fusion_mode = 'confidence_residual'
        if self.obs_gate_fusion_mode not in ('feature', 'confidence_residual'):
            raise ValueError("obs_gate_fusion_mode must be 'feature' or 'confidence_residual'.")
        self.obs_gate_residual_scale = float(getattr(config, 'obs_gate_residual_scale', 0.1))
        self.obs_gate_max_dyn_alpha = float(getattr(config, 'obs_gate_max_dyn_alpha', 0.2))
        self.dynamics_motion_mode = str(getattr(config, 'dynamics_motion_mode', 'feature')).lower()
        self.dynamics_motion_mode = self.dynamics_motion_mode.replace('-', '_')
        if self.dynamics_motion_mode in ('residual_limited', 'bounded_residual'):
            self.dynamics_motion_mode = 'residual'
        if self.dynamics_motion_mode in ('innovation', 'bounded_innovation'):
            self.dynamics_motion_mode = 'proposal_innovation'
        if self.dynamics_motion_mode in (
                'ordered_trajectory', 'trajectory', 'trajectory_residual'):
            self.dynamics_motion_mode = 'trajectory_adapter'
        if self.dynamics_motion_mode not in (
                'feature', 'residual', 'proposal_innovation',
                'trajectory_adapter'):
            raise ValueError(
                "dynamics_motion_mode must be 'feature', 'residual', "
                "'proposal_innovation', or 'trajectory_adapter'.")
        if (self.dynamics_motion_mode in ('residual', 'proposal_innovation')
                and not self.use_dynamics_encoder):
            raise ValueError(
                f"dynamics_motion_mode='{self.dynamics_motion_mode}' requires "
                "use_dynamics_encoder=True.")
        if self.use_observability_gate and not self.use_dynamics_encoder:
            raise ValueError("use_observability_gate=True requires use_dynamics_encoder=True.")
        if (self.use_observability_gate
                and self.dynamics_motion_mode in ('residual', 'proposal_innovation')):
            raise ValueError(
                "Observation-first dynamics correction cannot be combined with "
                "ObservabilityGate. Keep dynamics_motion_mode='feature' for gate experiments.")
        if self.use_physical_time_adapter and not self.use_dynamics_encoder:
            raise ValueError(
                "use_physical_time_adapter=True requires use_dynamics_encoder=True.")
        if self.use_ordered_trajectory_encoder and not self.use_dynamics_encoder:
            raise ValueError(
                "ordered trajectory encoding requires use_dynamics_encoder=True")
        if self.use_trajectory_adapter and not self.use_ordered_trajectory_encoder:
            raise ValueError(
                "trajectory adapter requires use_ordered_trajectory_encoder=True")
        if self.use_trajectory_search and not self.use_ordered_trajectory_encoder:
            raise ValueError(
                "trajectory pre-crop search requires the ordered trajectory path")
        if (self.use_physical_time_adapter
                and self.dynamics_motion_mode != 'proposal_innovation'):
            raise ValueError(
                "The zero-init physical-time adapter is currently isolated to "
                "dynamics_motion_mode='proposal_innovation'.")
        self.ct_fusion_mode = str(
            getattr(config, 'ct_fusion_mode', 'adaptive')).strip().lower()
        if self.ct_fusion_mode not in ('fixed', 'adaptive'):
            raise ValueError("ct_fusion_mode must be 'fixed' or 'adaptive'")
        if self.use_ct_v2:
            if not self.use_dynamics_encoder:
                raise ValueError("use_ct_v2=True requires use_dynamics_encoder=True")
            if self.dynamics_motion_mode not in (
                    'proposal_innovation', 'trajectory_adapter'):
                raise ValueError(
                    "CT-SeqTrack v2 requires "
                    "proposal innovation or the ordered trajectory adapter")
            if (self.dynamics_motion_mode == 'trajectory_adapter'
                    and not self.use_ordered_trajectory_encoder):
                raise ValueError(
                    "trajectory_adapter mode requires the ordered encoder")
            incompatible = {
                'use_observability_gate': self.use_observability_gate,
                'use_physical_time_adapter': self.use_physical_time_adapter,
                'use_b4_paired_views': self.use_b4_paired_views,
            }
            enabled = [name for name, value in incompatible.items() if value]
            if enabled:
                raise ValueError(
                    "CT-SeqTrack v2 keeps legacy modules disabled: "
                    + ", ".join(enabled))
        if self.use_b1motion_v3:
            incompatible_v3 = {
                'use_ct_v2': self.use_ct_v2,
                'use_dynamics_encoder': self.use_dynamics_encoder,
                'use_ordered_trajectory_encoder':
                    self.use_ordered_trajectory_encoder,
                'use_trajectory_adapter': self.use_trajectory_adapter,
                'use_trajectory_search': self.use_trajectory_search,
                'use_time_guided_search': self.use_time_guided_search,
                'use_observability_gate': self.use_observability_gate,
                'use_physical_time_adapter': self.use_physical_time_adapter,
                'use_b4_paired_views': self.use_b4_paired_views,
                'use_point_feature_tc': self.use_point_feature_tc,
            }
            enabled = [
                name for name, value in incompatible_v3.items() if value]
            if enabled:
                raise ValueError(
                    "B1motion-v3 is an isolated post-Transformer plugin; "
                    "disable incompatible modules: " + ", ".join(enabled))
        if self.use_search_evidence_v2 and not self.use_b1motion_v3:
            raise ValueError(
                "Search Evidence v2 requires the B1motion-v3 physical prior")
        if self.use_joint_proposal_fusion and not self.use_search_evidence_v2:
            raise ValueError(
                "joint proposal fusion requires Search Evidence v2")
        if (self.use_search_evidence_v2
                and self.use_motion_v3_legacy_fusion):
            raise ValueError(
                "B2-v2 disables the legacy two-candidate motion fusion")
        if self.use_search_evidence_v2 and self.use_search_evidence_v21:
            raise ValueError(
                "Search Evidence v2 and v2.1 are mutually exclusive")
        if (self.use_motion_conditioned_search_v22
                and (self.use_search_evidence_v2
                     or self.use_search_evidence_v21
                     or self.use_motion_conditioned_search_v3)):
            raise ValueError(
                "B2-v2.2 is isolated from Search Evidence v2/v2.1")
        if (self.use_motion_conditioned_search_v22
                and not self.use_b1motion_v3):
            raise ValueError(
                "B2-v2.2 requires the B1motion-v3 physical prior")
        if (self.use_signed_horizon_router
                and not self.use_motion_conditioned_search_v22):
            raise ValueError(
                "signed-horizon routing requires B2-v2.2")
        if (self.use_motion_conditioned_search_v3
                and (self.use_search_evidence_v2
                     or self.use_search_evidence_v21
                     or self.use_motion_conditioned_search_v22)):
            raise ValueError("B2-v3 is isolated from all older B2 branches")
        if (self.use_motion_conditioned_search_v3
                and not self.use_b1motion_v3):
            raise ValueError("B2-v3 requires the B1motion-v3 physical prior")
        if (self.use_action_consistent_router_v3
                and not self.use_motion_conditioned_search_v3):
            raise ValueError("action-consistent routing requires B2-v3")
        if (self.b2_v3_require_packaged_router
                and not (self.use_motion_conditioned_search_v3
                         and self.use_action_consistent_router_v3)):
            raise ValueError(
                "packaged-router enforcement requires action-consistent B2-v3")
        if (self.use_motion_conditioned_search_v3
                and self.proposal_inference_mode not in (
                    'obs_only', 'obs_vs_motion', 'obs_vs_refined',
                    'obs_vs_all', 'observation', 'motion', 'raw_search',
                    'legacy_clipped', 'selective', 'bounded_always')):
            raise ValueError(
                "B2-v3 mode must be a legacy obs_vs_* mode or one of "
                "observation/motion/raw_search/legacy_clipped/selective")
        if (self.use_advantage_proposal_fusion
                and not self.use_search_evidence_v21):
            raise ValueError(
                "advantage proposal fusion requires Search Evidence v2.1")
        if self.use_b3_risk_router and not self.use_search_evidence_v21:
            raise ValueError(
                "B3 CRPA requires Search Evidence v2.1")
        if self.use_b3_risk_router and self.use_advantage_proposal_fusion:
            raise ValueError(
                "B3 CRPA and advantage proposal fusion are mutually exclusive")
        if (self.use_search_evidence_v21
                and not (self.use_advantage_proposal_fusion
                         or self.use_b3_risk_router)):
            raise ValueError(
                "Search Evidence v2.1 requires advantage fusion or B3 CRPA")
        if (self.use_search_evidence_v21 and self.use_b1motion_v3
                and self.use_motion_v3_legacy_fusion):
            raise ValueError(
                "B2-v2.1 disables the legacy two-candidate motion fusion")
        # Time-guided search changes only the data-side crop and fixed token
        # allocation.  It is intentionally allowed without the CT-v2 motion
        # encoder so a baseline + search-only arm keeps exactly the baseline
        # model parameters and isolates the search contribution.
        self.dynamics_hidden_dim = int(getattr(config, 'dynamics_hidden_dim', 128))
        self.dynamics_residual_scale = float(getattr(config, 'dynamics_residual_scale', 0.1))
        self.dynamics_max_residual_norm = float(
            getattr(config, 'dynamics_max_residual_norm', 1.0))
        self.dynamics_warmup_epoch = int(getattr(config, 'dynamics_warmup_epoch', 0))
        self.dynamics_long_gap_only = bool(getattr(config, 'dynamics_long_gap_only', False))
        self.dynamics_min_delta_t = float(getattr(config, 'dynamics_min_delta_t', 0.0))
        self.dynamics_sparse_only = bool(getattr(config, 'dynamics_sparse_only', False))
        self.dynamics_sparse_point_threshold = float(
            getattr(config, 'dynamics_sparse_point_threshold', 128.0))
        self.dynamics_max_alpha = float(getattr(config, 'dynamics_max_alpha', 0.2))
        self.dynamics_residual_detach_stats = bool(
            getattr(config, 'dynamics_residual_detach_stats', True))
        self.physical_time_adapter_scale = float(
            getattr(config, 'physical_time_adapter_scale', 1.0))
        self.trajectory_adapter_scale = float(getattr(
            config, 'trajectory_adapter_scale', 1.0))
        self.trajectory_adapter_warmup_epoch = int(getattr(
            config, 'trajectory_adapter_warmup_epoch', 0))
        self.dynamics_innovation_alpha = float(
            getattr(config, 'dynamics_innovation_alpha', 0.0))
        self.dynamics_innovation_scale = float(
            getattr(config, 'dynamics_innovation_scale', 1.0))
        self.dynamics_innovation_radius_base = float(
            getattr(config, 'dynamics_innovation_radius_base', 0.5))
        self.dynamics_innovation_radius_per_second = float(
            getattr(config, 'dynamics_innovation_radius_per_second', 0.5))
        self.dynamics_innovation_radius_max = float(
            getattr(config, 'dynamics_innovation_radius_max', 2.0))
        self.dynamics_innovation_warmup_epoch = int(getattr(
            config, 'dynamics_innovation_warmup_epoch', self.dynamics_warmup_epoch))
        self.physical_time_adapter_warmup_epoch = int(getattr(
            config,
            'physical_time_adapter_warmup_epoch',
            self.dynamics_innovation_warmup_epoch,
        ))
        self.dynamics_innovation_disable_on_empty_search = bool(getattr(
            config, 'dynamics_innovation_disable_on_empty_search', False))
        self.motion_v3_warmup_epoch = int(getattr(
            config, 'motion_v3_warmup_epoch', 10))
        self.motion_v3_fusion_scale = float(getattr(
            config, 'motion_v3_fusion_scale', 1.0))
        self.motion_v3_prior_weight = float(getattr(
            config, 'motion_v3_prior_weight', 0.1))
        self.motion_v3_aux_prior_weight = float(getattr(
            config, 'motion_v3_aux_prior_weight', 0.1))
        self.motion_v3_nll_weight = float(getattr(
            config, 'motion_v3_nll_weight', 0.0))
        self.motion_v3_aux_nll_weight = float(getattr(
            config, 'motion_v3_aux_nll_weight',
            self.motion_v3_nll_weight))
        if self.ct_enable_v27 and self.ct_enable_b2:
            for suffix in ('positive_points', 'negative_points', 'positive_weight', 'negative_weight'):
                self.register_buffer('ct_relation_running_' + suffix,
                    getattr(self, 'ct_targetness_running_' + suffix).clone())
        self.ct_acquisition_margin_weight = float(getattr(
            config, 'ct_acquisition_margin_weight', 0.0))
        self.ct_acquisition_margin_quantile = float(getattr(
            config, 'ct_acquisition_margin_quantile', 0.90))
        self.motion_v3_beta_nll_beta = float(getattr(
            config, 'motion_v3_beta_nll_beta', 0.5))
        self.motion_v3_tail_direction_weight = float(getattr(
            config, 'motion_v3_tail_direction_weight', 0.25))
        self.motion_v3_tail_direction_margin = float(getattr(
            config, 'motion_v3_tail_direction_margin', 0.9))
        self.motion_v3_sigma_error_cap = float(getattr(
            config, 'ct_motion_max_displacement', 12.0))
        self.motion_v3_fused_weight = float(getattr(
            config, 'motion_v3_fused_weight', 1.0))
        self.motion_v3_gate_weight = float(getattr(
            config, 'motion_v3_gate_weight', 0.1))
        self.motion_v3_help_margin = float(getattr(
            config, 'motion_v3_help_margin', 0.05))
        self.motion_v3_aux_query_gaps = tuple(int(value) for value in getattr(
            config, 'motion_v3_aux_query_gaps', (2, 4)))
        self.search_v2_targetness_weight = float(getattr(
            config, 'search_v2_targetness_weight', 0.2))
        self.search_v2_vote_weight = float(getattr(
            config, 'search_v2_vote_weight', 1.0))
        self.search_v2_proposal_weight = float(getattr(
            config, 'search_v2_proposal_weight', 1.0))
        self.search_v2_confidence_weight = float(getattr(
            config, 'search_v2_confidence_weight', 0.1))
        self.search_v2_focal_alpha = float(getattr(
            config, 'search_v2_focal_alpha', 0.75))
        self.search_v2_focal_gamma = float(getattr(
            config, 'search_v2_focal_gamma', 2.0))
        self.joint_fused_weight = float(getattr(
            config, 'joint_fused_weight', 1.0))
        self.joint_gate_weight = float(getattr(
            config, 'joint_gate_weight', 0.1))
        self.joint_help_margin = float(getattr(
            config, 'joint_help_margin', 0.05))
        self.joint_fusion_warmup_epochs = int(getattr(
            config, 'joint_fusion_warmup_epochs', 10))
        self.joint_fusion_ramp_epochs = int(getattr(
            config, 'joint_fusion_ramp_epochs', 10))
        self.search_v21_match_weight = float(getattr(
            config, 'search_v21_match_weight', 0.1))
        self.search_v21_targetness_weight = float(getattr(
            config, 'search_v21_targetness_weight', 0.2))
        self.search_v21_vote_weight = float(getattr(
            config, 'search_v21_vote_weight', 1.0))
        self.search_v21_proposal_weight = float(getattr(
            config, 'search_v21_proposal_weight', 1.0))
        self.search_v21_focal_alpha = float(getattr(
            config, 'search_v21_focal_alpha', 0.75))
        self.search_v21_focal_gamma = float(getattr(
            config, 'search_v21_focal_gamma', 2.0))
        self.search_v22_match_weight = float(getattr(
            config, 'search_v22_match_weight', 0.1))
        self.search_v22_targetness_weight = float(getattr(
            config, 'search_v22_targetness_weight', 0.2))
        self.search_v22_vote_weight = float(getattr(
            config, 'search_v22_vote_weight', 1.0))
        self.search_v22_raw_proposal_weight = float(getattr(
            config, 'search_v22_raw_proposal_weight', 1.0))
        self.search_v22_refined_proposal_weight = float(getattr(
            config, 'search_v22_refined_proposal_weight', 1.0))
        self.search_v22_presence_weight = float(getattr(
            config, 'search_v22_presence_weight', 0.2))
        self.search_v22_focal_alpha = float(getattr(
            config, 'search_v22_focal_alpha', 0.75))
        self.search_v22_focal_gamma = float(getattr(
            config, 'search_v22_focal_gamma', 2.0))
        self.search_v3_match_weight = float(getattr(
            config, 'search_v3_match_weight', 0.1))
        self.search_v3_targetness_weight = float(getattr(
            config, 'search_v3_targetness_weight', 0.2))
        self.search_v3_vote_weight = float(getattr(
            config, 'search_v3_vote_weight', 1.0))
        self.search_v3_raw_proposal_weight = float(getattr(
            config, 'search_v3_raw_proposal_weight', 1.0))
        self.search_v3_refined_proposal_weight = float(getattr(
            config, 'search_v3_refined_proposal_weight', 1.0))
        self.search_v3_presence_weight = float(getattr(
            config, 'search_v3_presence_weight', 0.2))
        self.search_v3_utility_weight = float(getattr(
            config, 'search_v3_utility_weight', 0.0))
        self.search_v3_utility_margin = float(getattr(
            config, 'search_v3_utility_margin', 0.05))
        self.search_v3_focal_alpha = float(getattr(
            config, 'search_v3_focal_alpha', 0.75))
        self.search_v3_focal_gamma = float(getattr(
            config, 'search_v3_focal_gamma', 2.0))
        self.ct_query_gate_weight = float(getattr(
            config, 'ct_query_gate_weight', 0.05))
        self.ct_query_counterfactual_supervision = bool(getattr(
            config, 'ct_query_counterfactual_supervision', False))
        self.ct_query_counterfactual_margin = float(getattr(
            config, 'ct_query_counterfactual_margin', 0.05))
        self.ct_presence_balanced_loss = bool(getattr(
            config, 'ct_presence_balanced_loss', False))
        self.ct_targetness_weight = float(getattr(
            config, 'ct_targetness_weight', 0.2))
        self.ct_relation_weight = float(getattr(
            config, 'ct_relation_weight', 0.0))
        self.ct_vote_weight = float(getattr(config, 'ct_vote_weight', 1.0))
        self.ct_raw_search_weight = float(getattr(
            config, 'ct_raw_search_weight', 1.0))
        self.ct_presence_weight = float(getattr(
            config, 'ct_presence_weight', 0.2))
        self.ct_router_weight = float(getattr(
            config, 'ct_router_weight', 0.2))
        self.ct_correction_weight = float(getattr(
            config, 'ct_correction_weight', 1.0))
        self.ct_router_help_margin = float(getattr(
            config, 'ct_router_help_margin', 0.05))
        self.ct_router_h3_margin = float(getattr(
            config, 'ct_router_h3_margin', 0.15))
        self.ct_focal_alpha = float(getattr(
            config, 'ct_focal_alpha', 0.75))
        self.ct_focal_gamma = float(getattr(
            config, 'ct_focal_gamma', 2.0))
        self.advantage_fused_weight = float(getattr(
            config, 'advantage_fused_weight', 1.0))
        self.advantage_help_weight = float(getattr(
            config, 'advantage_help_weight', 0.2))
        self.advantage_step_weight = float(getattr(
            config, 'advantage_step_weight', 0.1))
        self.advantage_help_alpha_motion = float(getattr(
            config, 'advantage_help_alpha_motion', 0.5))
        self.advantage_help_alpha_search = float(getattr(
            config, 'advantage_help_alpha_search', 0.75))
        self.advantage_help_gamma = float(getattr(
            config, 'advantage_help_gamma', 2.0))
        self.advantage_help_margin = float(getattr(
            config, 'advantage_help_margin', 0.05))
        self.advantage_fusion_warmup_epochs = int(getattr(
            config, 'advantage_fusion_warmup_epochs', 10))
        self.advantage_fusion_ramp_epochs = int(getattr(
            config, 'advantage_fusion_ramp_epochs', 10))
        if self.dynamics_residual_scale < 0:
            raise ValueError("dynamics_residual_scale must be non-negative.")
        if self.dynamics_max_residual_norm <= 0:
            raise ValueError("dynamics_max_residual_norm must be positive.")
        if not 0.0 <= self.physical_time_adapter_scale <= 1.0:
            raise ValueError("physical_time_adapter_scale must be in [0, 1].")
        if not 0.0 <= self.trajectory_adapter_scale <= 1.0:
            raise ValueError("trajectory_adapter_scale must be in [0, 1].")
        if self.trajectory_adapter_warmup_epoch < 0:
            raise ValueError("trajectory_adapter_warmup_epoch must be non-negative")
        if self.physical_time_adapter_warmup_epoch < 0:
            raise ValueError("physical_time_adapter_warmup_epoch must be non-negative.")
        if not 0.0 <= self.dynamics_innovation_alpha <= 1.0:
            raise ValueError("dynamics_innovation_alpha must be in [0, 1].")
        if not 0.0 <= self.dynamics_innovation_scale <= 1.0:
            raise ValueError("dynamics_innovation_scale must be in [0, 1].")
        if self.dynamics_innovation_radius_base < 0:
            raise ValueError("dynamics_innovation_radius_base must be non-negative.")
        if self.dynamics_innovation_radius_per_second < 0:
            raise ValueError("dynamics_innovation_radius_per_second must be non-negative.")
        if self.dynamics_innovation_radius_max <= 0:
            raise ValueError("dynamics_innovation_radius_max must be positive.")
        if self.motion_v3_warmup_epoch < 0:
            raise ValueError("motion_v3_warmup_epoch must be non-negative")
        if not 0.0 <= self.motion_v3_fusion_scale <= 1.0:
            raise ValueError("motion_v3_fusion_scale must be in [0, 1]")
        if min(
                self.motion_v3_prior_weight,
                self.motion_v3_aux_prior_weight,
                self.motion_v3_nll_weight,
                self.motion_v3_aux_nll_weight,
                self.motion_v3_fused_weight,
                self.motion_v3_gate_weight,
                self.motion_v3_help_margin) < 0:
            raise ValueError("B1motion-v3 loss settings must be non-negative")
        if (not self.motion_v3_aux_query_gaps
                or any(value <= 0 for value in
                       self.motion_v3_aux_query_gaps)):
            raise ValueError("B1motion-v3 auxiliary query gaps must be positive")
        if not 0.0 <= self.motion_v3_beta_nll_beta <= 1.0:
            raise ValueError("motion_v3_beta_nll_beta must be in [0,1]")
        if (self.ct_acquisition_margin_weight < 0.0
                or not 0.0 < self.ct_acquisition_margin_quantile < 1.0):
            raise ValueError("v26 acquisition margin loss settings are invalid")
        if self.ct_relation_weight < 0.0:
            raise ValueError("relation loss weight must be non-negative")
        if (self.motion_v3_tail_direction_weight < 0
                or not 0.0 <= self.motion_v3_tail_direction_margin <= 1.0):
            raise ValueError("B1 tail direction settings are invalid")
        if min(
                self.search_v2_targetness_weight,
                self.search_v2_vote_weight,
                self.search_v2_proposal_weight,
                self.search_v2_confidence_weight,
                self.search_v2_focal_gamma,
                self.joint_fused_weight,
                self.joint_gate_weight,
                self.joint_help_margin) < 0:
            raise ValueError("B2-v2 loss settings must be non-negative")
        if not 0.0 <= self.search_v2_focal_alpha <= 1.0:
            raise ValueError("search_v2_focal_alpha must be in [0,1]")
        if (self.joint_fusion_warmup_epochs < 0
                or self.joint_fusion_ramp_epochs < 0):
            raise ValueError("joint fusion schedule must be non-negative")
        if min(
                self.search_v21_match_weight,
                self.search_v21_targetness_weight,
                self.search_v21_vote_weight,
                self.search_v21_proposal_weight,
                self.search_v21_focal_gamma,
                self.advantage_fused_weight,
                self.advantage_help_weight,
                self.advantage_step_weight,
                self.advantage_help_gamma,
                self.advantage_help_margin) < 0:
            raise ValueError("B2-v2.1 loss settings must be non-negative")
        if not all(0.0 <= value <= 1.0 for value in (
                self.search_v21_focal_alpha,
                self.advantage_help_alpha_motion,
                self.advantage_help_alpha_search)):
            raise ValueError("B2-v2.1 focal alpha values must be in [0,1]")
        if (self.advantage_fusion_warmup_epochs < 0
                or self.advantage_fusion_ramp_epochs < 0):
            raise ValueError(
                "advantage fusion schedule must be non-negative")
        if min(
                self.search_v22_match_weight,
                self.search_v22_targetness_weight,
                self.search_v22_vote_weight,
                self.search_v22_raw_proposal_weight,
                self.search_v22_refined_proposal_weight,
                self.search_v22_presence_weight,
                self.search_v22_focal_gamma) < 0:
            raise ValueError("B2-v2.2 loss settings must be non-negative")
        if not 0.0 <= self.search_v22_focal_alpha <= 1.0:
            raise ValueError("search_v22_focal_alpha must be in [0,1]")
        if min(
                self.search_v3_match_weight,
                self.search_v3_targetness_weight,
                self.search_v3_vote_weight,
                self.search_v3_raw_proposal_weight,
                self.search_v3_refined_proposal_weight,
                self.search_v3_presence_weight,
                self.search_v3_utility_weight,
                self.search_v3_utility_margin,
                self.search_v3_focal_gamma) < 0:
            raise ValueError("B2-v3 loss settings must be non-negative")
        if not 0.0 <= self.search_v3_focal_alpha <= 1.0:
            raise ValueError("search_v3_focal_alpha must be in [0,1]")
        if min(
                self.ct_query_gate_weight,
                self.ct_targetness_weight,
                self.ct_vote_weight,
                self.ct_raw_search_weight,
                self.ct_presence_weight,
                self.ct_router_weight,
                self.ct_correction_weight,
                self.ct_router_help_margin,
                self.ct_router_h3_margin,
                self.ct_focal_gamma,
                self.ct_query_counterfactual_margin) < 0:
            raise ValueError("CT joint Full loss settings must be non-negative")
        if not 0.0 <= self.ct_focal_alpha <= 1.0:
            raise ValueError("ct_focal_alpha must be in [0,1]")
        default_time_scale = getattr(config, 'default_time_step', getattr(config, 'time_step', 0.5))
        self.time_encoder = TimeEncoding(
            mode=getattr(config, 'time_encoding', 'raw'),
            scale=getattr(config, 'time_scale', default_time_scale),
            clip=getattr(config, 'time_clip', 4.0),
            fourier_bands=getattr(config, 'time_fourier_bands', 4),
            hidden_dim=getattr(config, 'time_hidden_dim', 16),
            output_scale=getattr(config, 'time_output_scale', getattr(config, 'pseudo_time_step', 0.1)),
        )
        self.seg_pointnet = SegPointNet(input_channel=3 + 1 + 1 + (9 if self.box_aware else 0),
                                        per_point_mlp1=[64, 64, 64, 128, 1024],
                                        per_point_mlp2=[512, 256, 128, 128],
                                        output_size=2 + (9 if self.box_aware else 0))
        self.mini_pointnet = MiniPointNet(input_channel=3 + 1 + (9 if self.box_aware else 0),
                                          per_point_mlp=[64, 128, 256, 512],
                                          hidden_mlp=[512, 256],
                                          output_size=-1)

        if self.use_motion_cls:
            self.motion_state_mlp = nn.Sequential(nn.Linear(256, 128),
                                                  nn.BatchNorm1d(128),
                                                  nn.ReLU(),
                                                  nn.Linear(128, 128),
                                                  nn.BatchNorm1d(128),
                                                  nn.ReLU(),
                                                  nn.Linear(128, 2))
            self.motion_acc = _build_binary_segmentation_accuracy()

        motion_feature_dim = 256
        if self.use_dynamics_encoder:
            if not self.use_ordered_trajectory_encoder:
                dynamics_encoder_class = (
                    ContinuousTimeMotionEncoder if self.use_ct_v2 else DynamicsEncoder)
                self.dynamics_encoder = dynamics_encoder_class(
                    hidden_dim=self.dynamics_hidden_dim,
                    eps=getattr(config, 'dynamics_eps', 1e-3),
                    use_query_gap=getattr(config, 'dynamics_use_query_gap', True),
                )
            if (not self.use_ordered_trajectory_encoder
                    and self.use_ct_v2 and self.ct_fusion_mode == 'adaptive'):
                self.ct_proposal_fusion = ProposalFusionGate(
                    observation_dim=256,
                    dynamics_dim=self.dynamics_hidden_dim,
                    observation_stats_dim=5,
                    hidden_dim=getattr(config, 'ct_fusion_hidden_dim', 64),
                    context_dim=getattr(config, 'ct_fusion_context_dim', 16),
                    max_alpha=getattr(config, 'ct_fusion_max_alpha', 0.75),
                    init_alpha=getattr(config, 'ct_fusion_init_alpha', 0.25),
                    time_scale=getattr(
                        config, 'time_scale', default_time_scale),
                    detach_context=getattr(
                        config, 'ct_fusion_detach_context', True),
                )
            if (not self.use_ordered_trajectory_encoder
                    and self.use_physical_time_adapter):
                self.physical_time_adapter = ZeroInitPhysicalTimeAdapter(
                    feature_dim=256,
                    dynamics_dim=self.dynamics_hidden_dim,
                    hidden_dim=getattr(config, 'physical_time_adapter_hidden_dim', 128),
                    time_scale=getattr(config, 'time_scale', default_time_scale),
                )
            if self.use_ordered_trajectory_encoder:
                # Constructed after all baseline modules below so enabling B1
                # cannot alter shared-layer initialization through RNG usage.
                pass
            elif self.dynamics_motion_mode == 'residual':
                self.dynamics_residual_gate = DynamicsResidualGate(
                    stats_dim=6,
                    hidden_dim=getattr(config, 'dynamics_residual_gate_hidden_dim', 16),
                    max_alpha=self.dynamics_max_alpha,
                    init_alpha=getattr(config, 'dynamics_residual_init_alpha', 0.0),
                )
            elif self.dynamics_motion_mode == 'proposal_innovation':
                # M2 keeps the observation head at 256-D.  The only permitted
                # feature change is the separately switchable zero-init M1 adapter.
                pass
            elif self.use_observability_gate:
                self.observability_gate = ObservabilityGate(
                    feature_dim=256,
                    dynamics_dim=self.dynamics_hidden_dim,
                    stats_dim=getattr(config, 'obs_gate_num_stats', 5),
                    hidden_dim=getattr(config, 'obs_gate_hidden_dim', 64),
                    init_obs_bias=getattr(config, 'obs_gate_init_obs_bias', 1.0),
                    min_dyn_valid=getattr(config, 'obs_gate_min_dyn_valid', 0.5),
                )
            else:
                motion_feature_dim += self.dynamics_hidden_dim

        self.motion_mlp = nn.Sequential(nn.Linear(motion_feature_dim, 128),
                                        nn.BatchNorm1d(128),
                                        nn.ReLU(),
                                        nn.Linear(128, 128),
                                        nn.BatchNorm1d(128),
                                        nn.ReLU(),
                                        nn.Linear(128, 4))

        self.feature_pointnet = FeaturePointNet(
            input_channel=3 + 1 + 1 + (9 if self.box_aware else 0),
            per_point_mlp1=[64, 64, 64, 128, 1024],
            per_point_mlp2=[512, 256, 128, 128],
            output_size=128)
        if self.use_point_feature_tc:
            self.point_feature_tc = PointFeatureTemporalConsistencyLoss(
                distance_threshold=float(getattr(
                    config, 'pftc_distance_threshold', 0.3)),
                min_correspondences=int(getattr(
                    config, 'pftc_min_correspondences', 3)),
                time_weighting=bool(getattr(
                    config, 'pftc_time_weighting', True)),
                time_scale=float(getattr(
                    config, 'pftc_time_scale', 0.5)),
                time_weight_min=float(getattr(
                    config, 'pftc_time_weight_min', 0.5)),
                time_weight_max=float(getattr(
                    config, 'pftc_time_weight_max', 3.0)),
                degrees=bool(getattr(config, 'degrees', False)),
            )

        self.Transformer = Seq2SeqFormer(d_word_vec=64, d_model=64, d_inner=512,
            n_layers=3, n_head=4, d_k=64, d_v=64, n_position = 1024*4)
        if self.use_decoder_token_consistency:
            self.decoder_token_consistency = DecoderTokenConsistencyLoss(
                input_dim=64,
                projection_dim=int(getattr(
                    config, 'decoder_tc_projection_dim', 64)),
                hidden_dim=int(getattr(
                    config, 'decoder_tc_hidden_dim', 128)),
                teacher_momentum=float(getattr(
                    config, 'decoder_tc_teacher_momentum', 0.996)),
                invariance_weight=float(getattr(
                    config, 'decoder_tc_invariance_weight', 1.0)),
                variance_weight=float(getattr(
                    config, 'decoder_tc_variance_weight', 1.0)),
                covariance_weight=float(getattr(
                    config, 'decoder_tc_covariance_weight', 0.04)),
            )
            self.decoder_tc_weight_selector = GradientRatioWeightSelector(
                candidates=tuple(float(value) for value in getattr(
                    config, 'decoder_tc_weight_candidates',
                    (0.001, 0.003, 0.01))),
                audit_batches=int(getattr(
                    config, 'decoder_tc_gradient_audit_batches', 200)),
                target_ratio=float(getattr(
                    config, 'decoder_tc_target_gradient_ratio', 0.075)),
            )

        if self.use_ordered_trajectory_encoder:
            self.dynamics_encoder = OrderedTrajectoryEncoder(
                hidden_dim=self.dynamics_hidden_dim,
                step_dim=int(getattr(config, 'trajectory_step_dim', 64)),
                eps=float(getattr(config, 'dynamics_eps', 1e-3)),
                time_scale=float(getattr(
                    config, 'time_scale', default_time_scale)),
                residual_velocity_scale=float(getattr(
                    config, 'trajectory_residual_velocity_scale', 4.0)),
                initial_sigma=float(getattr(
                    config, 'trajectory_initial_sigma', 0.5)),
            )
            self.trajectory_search_encoder = TrajectoryPointEncoder(
                input_dim=5, hidden_dim=64, output_dim=64)
            if self.use_trajectory_adapter:
                self.trajectory_adapter = ZeroInitTrajectoryAdapter(
                    feature_dim=256,
                    trajectory_dim=self.dynamics_hidden_dim,
                    search_dim=64,
                    hidden_dim=int(getattr(
                        config, 'trajectory_adapter_hidden_dim', 128)),
                    normal_scale=float(getattr(
                        config, 'trajectory_adapter_normal_scale', 0.1)),
                    gap_trigger=float(getattr(
                        config, 'trajectory_adapter_gap_trigger', 1.5)),
                )

        if self.use_b1motion_v3:
            # Keep v3 after every B0 module so enabling the plugin cannot
            # consume RNG before any shared parameter is initialized.
            with isolated_constructor_rng(
                    int(getattr(config, 'seed', 42) or 42), 'b1.motion'):
                self.physical_motion_encoder = B1PhysicalTimePrior(
                enable_v27=self.ct_enable_v27,
                initialization_seed=int(getattr(config, 'seed', 42) or 42),
                hidden_dim=int(getattr(
                    config, 'motion_v3_hidden_dim', 128)),
                step_dim=int(getattr(
                    config, 'motion_v3_step_dim', 64)),
                eps=float(getattr(config, 'motion_v3_eps', 1e-3)),
                time_scale=float(getattr(
                    config, 'time_scale', default_time_scale)),
                residual_velocity_scale=float(getattr(
                    config, 'motion_v3_residual_velocity_scale', 4.0)),
                initial_sigma=float(getattr(
                    config, 'motion_v3_initial_sigma', 0.5)),
                motion_aligned_uncertainty=(
                    self.use_calibrated_motion_uncertainty),
                min_direction_speed=float(getattr(
                    config, 'motion_v3_min_direction_speed', 0.2)),
                shared_kinematic_anchor=(
                    self.use_ct_joint_full
                    and self.ct_enable_shared_motion_anchor
                    and self.ct_enable_dynamic_residual_bound),
                max_acceleration=float(getattr(
                    config, 'ct_motion_max_acceleration', 8.0)),
                max_displacement=float(getattr(
                    config, 'ct_motion_max_displacement', 12.0)),
                acceleration_weight=float(getattr(
                    config, 'ct_motion_acceleration_weight', 0.5)),
                temporal_backend=str(getattr(
                    config, 'motion_v3_temporal_backend', 'gru')),
                cfc_backbone_units=int(getattr(
                    config, 'motion_v3_cfc_backbone_units', 105)),
                adaptive_acquisition_margin=bool(getattr(
                    config, 'ct_adaptive_acquisition_margin', False)),
                acquisition_margin_min=tuple(getattr(
                    config, 'ct_acquisition_margin_min', (2.0, 1.0))),
                acquisition_margin_max=tuple(getattr(
                    config, 'ct_acquisition_margin_max', (6.0, 3.0))),
                acquisition_margin_bias=float(getattr(
                    config, 'ct_acquisition_margin_bias', -8.0)),
                log_sigma_min=float(getattr(
                    config, 'motion_v3_log_sigma_min',
                    -2.302585092994046)),
                log_sigma_max=float(getattr(
                    config, 'motion_v3_log_sigma_max', 2.5)),
                )
            if self.use_motion_v3_legacy_fusion:
                with isolated_constructor_rng(
                        int(getattr(config, 'seed', 42) or 42),
                        'b1.legacy_fusion'):
                    self.motion_v3_fusion = ReliabilityGatedProposalFusion(
                    observation_dim=256,
                    motion_dim=int(getattr(
                        config, 'motion_v3_hidden_dim', 128)),
                    observation_stats_dim=5,
                    hidden_dim=int(getattr(
                        config, 'motion_v3_gate_hidden_dim', 64)),
                    context_dim=int(getattr(
                        config, 'motion_v3_gate_context_dim', 32)),
                    max_alpha=float(getattr(
                        config, 'motion_v3_alpha_max', 0.5)),
                    init_probability=float(getattr(
                        config, 'motion_v3_gate_init_probability', 0.01)),
                    radius_base=float(getattr(
                        config, 'motion_v3_radius_base', 0.25)),
                    radius_per_second=float(getattr(
                        config, 'motion_v3_radius_per_second', 0.5)),
                    radius_max=float(getattr(
                        config, 'motion_v3_radius_max', 1.25)),
                    time_scale=float(getattr(
                        config, 'time_scale', default_time_scale)),
                    )
            if self.use_ct_joint_full:
                self.b0_observation_contract = B0Observation()
                plugin_seed = int(getattr(config, 'seed', 42) or 42)
                plugin_cuda_devices = (
                    list(range(torch.cuda.device_count()))
                    if torch.cuda.is_available() else [])
                with torch.random.fork_rng(devices=plugin_cuda_devices):
                    if self.ct_joint_contract_version >= 3:
                        if self.ct_enable_b2:
                            with isolated_constructor_rng(
                                    plugin_seed, 'b2.evidence_acquirer'):
                                self.ct_joint_search_refiner = (
                                    B2EvidenceAcquirer(
                                    v27_enabled=self.ct_enable_v27,
                                    exploration_seed=plugin_seed,
                                    feature_dim=64,
                                    num_heads=4,
                                    max_vote_offset=float(getattr(
                                        config,
                                        'ct_search_max_vote_offset', 4.0)),
                                    attention_dropout=float(getattr(
                                        config,
                                        'ct_memory_attention_dropout', 0.0)),
                                    presence_init_probability=float(getattr(
                                        config,
                                        'ct_search_presence_init_probability',
                                        0.1)),
                                    presence_threshold=float(getattr(
                                        config,
                                        'ct_search_presence_threshold', 0.5)),
                                    relation_aware_sampling=bool(getattr(
                                        config,
                                        'ct_relation_aware_sampling', False)),
                                    relation_topk=int(getattr(
                                        config, 'ct_relation_topk', 128)),
                                    coverage_count=int(getattr(
                                        config, 'ct_relation_coverage_count', 96)),
                                    exploration_count=int(getattr(
                                        config,
                                        'ct_relation_exploration_count', 32)),
                                    robust_consensus_voting=bool(getattr(
                                        config,
                                        'ct_robust_consensus_voting', False)),
                                    ))
                        if self.ct_enable_b3:
                            with isolated_constructor_rng(
                                    plugin_seed, 'b3.selective_updater'):
                                from models.ct_v2.action_v27 import B3UtilityUpdater
                                router_class = B3UtilityUpdater if self.ct_enable_v27 else B3SelectiveUpdater
                                self.ct_joint_router = router_class(
                                observation_stats_dim=5,
                                hidden_dim=int(getattr(
                                    config, 'ct_router_hidden_dim', 64)),
                                presence_threshold=float(getattr(
                                    config,
                                    'ct_search_presence_threshold', 0.5)),
                                decision_threshold=float(getattr(
                                    config, 'ct_router_threshold', 0.5)),
                                radius_base=float(getattr(
                                    config, 'ct_router_radius_base', 0.5)),
                                radius_per_second=float(getattr(
                                    config,
                                    'ct_router_radius_per_second', 0.5)),
                                radius_max=float(getattr(
                                    config, 'ct_router_radius_max', 2.0)),
                                require_calibration=bool(getattr(
                                    config, 'ct_require_action_calibration',
                                    False)),
                                consensus_features=bool(getattr(
                                    config, 'ct_b3_consensus_features', False)),
                                )
                    else:
                        torch.manual_seed(plugin_seed + 24001)
                        self.ct_joint_search_refiner = JointFullSearchRefiner(
                        point_dim=5,
                        feature_dim=int(getattr(
                            config, 'ct_search_feature_dim', 128)),
                        query_dim=self.ct_query_dim,
                        motion_dim=int(getattr(
                            config, 'motion_v3_hidden_dim', 128)),
                        observation_stats_dim=5,
                        max_vote_offset=float(getattr(
                            config, 'ct_search_max_vote_offset', 4.0)),
                        motion_dropout=float(getattr(
                            config, 'ct_query_motion_dropout', 0.1)),
                gate_init_probability=float(getattr(
                    config, 'ct_query_gate_init_probability', 0.05)),
                query_gate_scale=float(getattr(
                    config, 'ct_query_gate_scale', 1.0)),
                presence_init_probability=float(getattr(
                            config, 'ct_search_presence_init_probability', 0.1)),
                        presence_threshold=float(getattr(
                            config, 'ct_search_presence_threshold', 0.5)),
                        mahalanobis_clip=float(getattr(
                            config, 'ct_mahalanobis_clip', 25.0)),
                        use_reliability_gate=(
                            self.ct_enable_query_reliability_gate),
                        contract_version=self.ct_joint_contract_version,
                        presence_hard_gate=bool(getattr(
                            config, 'ct_presence_hard_gate', True)),
                    )
                        self.ct_joint_router = JointScalarResidualRouter(
                        observation_stats_dim=5,
                        hidden_dim=int(getattr(
                            config, 'ct_router_hidden_dim', 64)),
                        init_probability=float(getattr(
                            config, 'ct_router_init_probability', 0.01)),
                        decision_threshold=float(getattr(
                            config, 'ct_router_threshold', 0.5)),
                extension_mass_threshold=float(getattr(
                    config, 'ct_router_extension_mass_threshold', 0.25)),
                presence_threshold=float(getattr(
                    config, 'ct_search_presence_threshold', 0.5)),
                radius_base=float(getattr(
                            config, 'ct_router_radius_base', 0.5)),
                        radius_per_second=float(getattr(
                            config, 'ct_router_radius_per_second', 0.5)),
                        radius_max=float(getattr(
                            config, 'ct_router_radius_max', 2.0)),
                        contract_version=self.ct_joint_contract_version,
                        presence_hard_gate=bool(getattr(
                            config, 'ct_presence_hard_gate', True)),
                        )

        if self.use_search_evidence_v2:
            # These modules are created only after all shared B0 parameters,
            # preserving tensor-identical B0 initialization under a fixed seed.
            self.search_evidence_v2 = TrajectorySearchEvidence(
                point_dim=9,
                feature_dim=int(getattr(
                    config, 'search_v2_feature_dim', 128)),
                observation_dim=256,
                motion_dim=int(getattr(
                    config, 'motion_v3_hidden_dim', 128)),
                observation_stats_dim=5,
                max_vote_offset=float(getattr(
                    config, 'search_v2_max_vote_offset', 4.0)),
            )
            if self.use_joint_proposal_fusion:
                self.joint_proposal_fusion = JointProposalFusion(
                    observation_dim=256,
                    motion_dim=int(getattr(
                        config, 'motion_v3_hidden_dim', 128)),
                    search_dim=int(getattr(
                        config, 'search_v2_feature_dim', 128)),
                    observation_stats_dim=5,
                    context_dim=int(getattr(
                        config, 'joint_gate_context_dim', 32)),
                    hidden_dim=int(getattr(
                        config, 'joint_gate_hidden_dim', 96)),
                    observation_bias=float(getattr(
                        config, 'joint_gate_observation_bias', 4.6)),
                    radius_base=float(getattr(
                        config, 'joint_radius_base', 0.5)),
                    radius_per_second=float(getattr(
                        config, 'joint_radius_per_second', 0.5)),
                    radius_max=float(getattr(
                        config, 'joint_radius_max', 2.0)),
                    normal_aux_mass=float(getattr(
                        config, 'joint_normal_aux_mass', 0.5)),
                    gap_aux_mass=float(getattr(
                        config, 'joint_gap_aux_mass', 0.8)),
                )

        if self.use_search_evidence_v21:
            # Isolated plugin RNG makes Search-v2.1 initialization identical
            # in search-only and motion+search configurations while preserving
            # every already-created B0/B1 parameter and the caller RNG state.
            plugin_seed = int(getattr(config, 'seed', 42) or 42)
            plugin_cuda_devices = (
                list(range(torch.cuda.device_count()))
                if torch.cuda.is_available() else [])
            with torch.random.fork_rng(devices=plugin_cuda_devices):
                torch.manual_seed(plugin_seed + 21001)
                self.search_evidence_v21 = TrajectorySearchEvidenceV21(
                    point_dim=9,
                    feature_dim=int(getattr(
                        config, 'search_v21_feature_dim', 128)),
                    query_dim=int(getattr(
                        config, 'search_v21_query_dim', 32)),
                    observation_dim=256,
                    motion_dim=int(getattr(
                        config, 'motion_v3_hidden_dim', 128)),
                    observation_stats_dim=5,
                    max_vote_offset=float(getattr(
                        config, 'search_v21_max_vote_offset', 4.0)),
                    pool_temperature=float(getattr(
                        config, 'search_v21_pool_temperature', 0.5)),
                )
            if self.use_advantage_proposal_fusion:
                with torch.random.fork_rng(devices=plugin_cuda_devices):
                    torch.manual_seed(plugin_seed + 21002)
                    self.advantage_proposal_fusion = (
                    AdvantageGatedProposalFusion(
                        observation_dim=256,
                        motion_dim=int(getattr(
                            config, 'motion_v3_hidden_dim', 128)),
                        search_dim=int(getattr(
                            config, 'search_v21_feature_dim', 128)),
                        observation_stats_dim=5,
                        context_dim=int(getattr(
                            config, 'advantage_gate_context_dim', 32)),
                        hidden_dim=int(getattr(
                            config, 'advantage_gate_hidden_dim', 96)),
                        init_help_probability=float(getattr(
                            config, 'advantage_init_help_probability', 0.02)),
                        radius_base=float(getattr(
                            config, 'advantage_radius_base', 0.5)),
                        radius_per_second=float(getattr(
                            config, 'advantage_radius_per_second', 0.5)),
                        radius_max=float(getattr(
                            config, 'advantage_radius_max', 2.0)),
                        normal_aux_mass=float(getattr(
                            config, 'advantage_normal_aux_mass', 0.5)),
                        gap_aux_mass=float(getattr(
                            config, 'advantage_gap_aux_mass', 0.8)),
                    ))
            if self.use_b3_risk_router:
                with torch.random.fork_rng(devices=plugin_cuda_devices):
                    torch.manual_seed(plugin_seed + 21003)
                    self.b3_risk_router = ClosedLoopRiskAwareProposalRouter(
                        observation_dim=256,
                        motion_dim=int(getattr(
                            config, 'motion_v3_hidden_dim', 128)),
                        search_dim=int(getattr(
                            config, 'search_v21_feature_dim', 128)),
                        observation_stats_dim=5,
                        context_dim=int(getattr(
                            config, 'b3_router_context_dim', 32)),
                        hidden_dim=int(getattr(
                            config, 'b3_router_hidden_dim', 96)),
                        gain_threshold=float(getattr(
                            config, 'b3_gain_threshold', 0.0)),
                        radius_base=float(getattr(
                            config, 'b3_radius_base', 0.5)),
                        radius_per_second=float(getattr(
                            config, 'b3_radius_per_second', 0.5)),
                        radius_max=float(getattr(
                            config, 'b3_radius_max', 2.0)),
                        normal_step_cap=float(getattr(
                            config, 'b3_normal_step_cap', 0.35)),
                        gap_step_cap=float(getattr(
                            config, 'b3_gap_step_cap', 0.60)),
                    )

        if self.use_motion_conditioned_search_v22:
            plugin_seed = int(getattr(config, 'seed', 42) or 42)
            plugin_cuda_devices = (
                list(range(torch.cuda.device_count()))
                if torch.cuda.is_available() else [])
            with torch.random.fork_rng(devices=plugin_cuda_devices):
                torch.manual_seed(plugin_seed + 22001)
                self.motion_conditioned_search_refiner = (
                    MotionConditionedSearchRefiner(
                        point_dim=9,
                        feature_dim=int(getattr(
                            config, 'search_v22_feature_dim', 128)),
                        query_dim=int(getattr(
                            config, 'search_v22_query_dim', 32)),
                        observation_dim=256,
                        motion_dim=int(getattr(
                            config, 'motion_v3_hidden_dim', 128)),
                        observation_stats_dim=5,
                        max_vote_offset=float(getattr(
                            config, 'search_v22_max_vote_offset', 4.0)),
                        pool_temperature=float(getattr(
                            config, 'search_v22_pool_temperature', 0.5)),
                        presence_threshold=float(getattr(
                            config, 'search_v22_presence_threshold', 0.5)),
                        radius_base=float(getattr(
                            config, 'search_v22_radius_base', 0.5)),
                        radius_per_second=float(getattr(
                            config, 'search_v22_radius_per_second', 0.5)),
                        radius_max=float(getattr(
                            config, 'search_v22_radius_max', 2.0)),
                    ))
            if self.use_signed_horizon_router:
                with torch.random.fork_rng(devices=plugin_cuda_devices):
                    torch.manual_seed(plugin_seed + 22002)
                    self.signed_horizon_router = (
                        SignedHorizonInnovationRouter(
                            observation_dim=256,
                            motion_dim=int(getattr(
                                config, 'motion_v3_hidden_dim', 128)),
                            search_dim=int(getattr(
                                config, 'search_v22_feature_dim', 128)),
                            observation_stats_dim=5,
                            context_dim=int(getattr(
                                config, 'signed_router_context_dim', 32)),
                            hidden_dim=int(getattr(
                                config, 'signed_router_hidden_dim', 96)),
                            gain_threshold=float(getattr(
                                config, 'signed_gain_threshold', 0.0)),
                            radius_base=float(getattr(
                                config, 'signed_radius_base', 0.5)),
                            radius_per_second=float(getattr(
                                config, 'signed_radius_per_second', 0.5)),
                            radius_max=float(getattr(
                                config, 'signed_radius_max', 2.0)),
                            normal_step_cap=float(getattr(
                                config, 'signed_normal_step_cap', 0.20)),
                            gap_step_cap=float(getattr(
                                config, 'signed_gap_step_cap', 0.35)),
                        ))

        if self.use_motion_conditioned_search_v3:
            plugin_seed = int(getattr(config, 'seed', 42) or 42)
            plugin_cuda_devices = (
                list(range(torch.cuda.device_count()))
                if torch.cuda.is_available() else [])
            with torch.random.fork_rng(devices=plugin_cuda_devices):
                torch.manual_seed(plugin_seed + 23001)
                self.state_aligned_search_refiner = StateAlignedSearchRefiner(
                    point_dim=(10 if self.use_uncertainty_geometry else 9),
                    feature_dim=int(getattr(
                        config, 'search_v3_feature_dim', 128)),
                    query_dim=int(getattr(
                        config, 'search_v3_query_dim', 32)),
                    observation_dim=256,
                    query_observation_dim=(
                        64 if self.use_asymmetric_dual_query else 256),
                    require_motion_valid=(
                        not self.use_raw_search_candidate),
                    motion_dim=int(getattr(
                        config, 'motion_v3_hidden_dim', 128)),
                    observation_stats_dim=5,
                    max_vote_offset=float(getattr(
                        config, 'search_v3_max_vote_offset', 4.0)),
                    pool_temperature=float(getattr(
                        config, 'search_v3_pool_temperature', 0.5)),
                    radius_base=float(getattr(
                        config, 'search_v3_radius_base', 0.5)),
                    radius_per_second=float(getattr(
                        config, 'search_v3_radius_per_second', 0.5)),
                    radius_max=float(getattr(
                        config, 'search_v3_radius_max', 2.0)),
                    predict_utility=bool(getattr(
                        config, 'search_v3_predict_utility',
                        self.use_raw_search_candidate)),
                )
            if self.use_asymmetric_dual_query:
                with torch.random.fork_rng(devices=plugin_cuda_devices):
                    torch.manual_seed(plugin_seed + 23003)
                    self.asymmetric_dual_query = AsymmetricDualQueryAdapter(
                        observation_dim=64,
                        motion_dim=int(getattr(
                            config, 'motion_v3_hidden_dim', 128)),
                        hidden_dim=int(getattr(
                            config, 'dual_query_hidden_dim', 128)),
                        gate_max=float(getattr(
                            config, 'dual_query_gate_max', 0.5)),
                    )
            if self.use_action_consistent_router_v3:
                with torch.random.fork_rng(devices=plugin_cuda_devices):
                    torch.manual_seed(plugin_seed + 23002)
                    self.action_consistent_router_v3 = (
                        ActionConsistentInnovationRouter(
                            observation_dim=256,
                            motion_dim=int(getattr(
                                config, 'motion_v3_hidden_dim', 128)),
                            search_dim=3 * int(getattr(
                                config, 'search_v3_feature_dim', 128)),
                            observation_stats_dim=5,
                            context_dim=int(getattr(
                                config, 'router_v3_context_dim', 32)),
                            hidden_dim=int(getattr(
                                config, 'router_v3_hidden_dim', 96)),
                            gain_threshold=float(getattr(
                                config, 'router_v3_gain_threshold', 0.0)),
                            radius_base=float(getattr(
                                config, 'router_v3_radius_base', 0.5)),
                            radius_per_second=float(getattr(
                                config, 'router_v3_radius_per_second', 0.5)),
                            radius_max=float(getattr(
                                config, 'router_v3_radius_max', 2.0)),
                            normal_step_cap=float(getattr(
                                config, 'router_v3_normal_step_cap', 0.20)),
                            gap_step_cap=float(getattr(
                                config, 'router_v3_gap_step_cap', 0.35)),
                            scalar_only=bool(getattr(
                                config, 'router_v3_scalar_only', False)),
                            use_utility_feature=bool(getattr(
                                config, 'router_v3_use_utility_feature',
                                False)),
                        ))

        if self.v22_freeze_candidate_producers:
            for parameter in self.parameters():
                parameter.requires_grad_(False)
            for parameter in (
                    self.motion_conditioned_search_refiner.parameters()):
                parameter.requires_grad_(True)
            self._apply_v22_frozen_module_modes()
        if self.b2_v3_freeze_candidate_producers:
            for parameter in self.parameters():
                parameter.requires_grad_(False)
            for parameter in self.state_aligned_search_refiner.parameters():
                parameter.requires_grad_(True)
            if self.use_asymmetric_dual_query:
                for parameter in self.asymmetric_dual_query.parameters():
                    parameter.requires_grad_(True)
            self._apply_v3_frozen_module_modes()
        if (self.ct_joint_contract_version >= 3
                and self.ct_initialization_policy == 'scratch_only'):
            module_flags = (
                ('physical_motion_encoder', self.ct_enable_b1),
                ('ct_joint_search_refiner', self.ct_enable_b2),
                ('ct_joint_router', self.ct_enable_b3),
            )
            for module_name, enabled in module_flags:
                module = getattr(self, module_name, None)
                if module is not None:
                    for parameter in module.parameters():
                        parameter.requires_grad_(bool(enabled))

    def _apply_v22_frozen_module_modes(self):
        if not getattr(self, 'v22_freeze_candidate_producers', False):
            return
        for name, module in self.named_children():
            if name == 'motion_conditioned_search_refiner':
                module.train(self.training)
            else:
                module.eval()

    def _apply_v3_frozen_module_modes(self):
        if not getattr(self, 'b2_v3_freeze_candidate_producers', False):
            return
        trainable_modules = {'state_aligned_search_refiner'}
        if getattr(self, 'use_asymmetric_dual_query', False):
            trainable_modules.add('asymmetric_dual_query')
        for name, module in self.named_children():
            if name in trainable_modules:
                module.train(self.training)
            else:
                module.eval()

    def train(self, mode=True):
        super().train(mode)
        self._apply_v22_frozen_module_modes()
        self._apply_v3_frozen_module_modes()
        return self

    def on_fit_start(self):
        if self.b2_v3_freeze_candidate_producers:
            allowed_prefixes = (
                'state_aligned_search_refiner.',
                'asymmetric_dual_query.',
            )
            required_prefixes = {'state_aligned_search_refiner.'}
            if self.use_asymmetric_dual_query:
                required_prefixes.add('asymmetric_dual_query.')
            validate_trainable_parameter_prefixes(
                self.named_parameters(), allowed_prefixes,
                required_prefixes)
            if not hasattr(self, '_b2_v3_frozen_reference_hashes'):
                raise RuntimeError(
                    "B2-v3 training requires strict initialization hashes; "
                    "use --init_checkpoint with a v3 composed checkpoint")
            self._verify_b2_v3_frozen_hashes()
            self._b2_v3_verified_optimizer_steps = 0
            self._b2_v3_verified_dual_query_gradient = False

    def on_after_backward(self):
        self._ct_record_cuda_stage('backward')
        if (not self.b2_v3_freeze_candidate_producers
                or not self.use_asymmetric_dual_query
                or bool(getattr(
                    self, '_b2_v3_verified_dual_query_gradient', False))):
            return
        require_nonzero_finite_gradient(
            self.named_parameters(), 'asymmetric_dual_query.')
        self._b2_v3_verified_dual_query_gradient = True

    @staticmethod
    def _state_prefix_hash(state, prefix):
        digest = hashlib.sha256()
        for key in sorted(key for key in state if key.startswith(prefix)):
            tensor = state[key].detach().cpu().contiguous()
            digest.update(key.encode('utf-8'))
            digest.update(str(tensor.dtype).encode('ascii'))
            digest.update(str(tuple(tensor.shape)).encode('ascii'))
            digest.update(tensor.numpy().tobytes())
        return digest.hexdigest()

    def _ct_module_parameter_hash(self, module_name):
        digest = hashlib.sha256()
        parameters = getattr(
            self, '_ct_named_parameters_by_module', {}).get(module_name, ())
        for parameter_name, parameter in sorted(parameters):
            tensor = parameter.detach().cpu().contiguous()
            digest.update(parameter_name.encode('utf-8'))
            digest.update(str(tensor.dtype).encode('ascii'))
            digest.update(str(tuple(tensor.shape)).encode('ascii'))
            digest.update(tensor.numpy().tobytes())
        return digest.hexdigest()

    def _ct_optimizer_state_hash(self, module_name, optimizer):
        """Hash Adam state for one named module without parameter IDs."""
        digest = hashlib.sha256()
        parameters = getattr(
            self, '_ct_named_parameters_by_module', {}).get(module_name, ())
        for parameter_name, parameter in sorted(parameters):
            digest.update(parameter_name.encode('utf-8'))
            state = optimizer.state.get(parameter, {})
            if not state:
                digest.update(b'<uninitialized>')
                continue
            for state_name in sorted(state):
                digest.update(str(state_name).encode('utf-8'))
                value = state[state_name]
                if torch.is_tensor(value):
                    tensor = value.detach().cpu().contiguous()
                    digest.update(str(tensor.dtype).encode('ascii'))
                    digest.update(str(tuple(tensor.shape)).encode('ascii'))
                    digest.update(tensor.numpy().tobytes())
                else:
                    digest.update(repr(value).encode('utf-8'))
        return digest.hexdigest()

    @torch.no_grad()
    def _verify_b2_v3_frozen_hashes(self):
        if sorted(self._b2_v3_frozen_reference_hashes) != sorted(
                self.B2_V3_FROZEN_PREFIXES):
            raise RuntimeError(
                "B2-v3 frozen hash manifest does not cover complete B0/B1")
        state = self.state_dict()
        for prefix, expected in (
                self._b2_v3_frozen_reference_hashes.items()):
            actual = self._state_prefix_hash(state, prefix)
            if actual != expected:
                raise RuntimeError(
                    f"B2-v3 frozen prefix changed after initialization: "
                    f"{prefix}")

    def on_save_checkpoint(self, checkpoint):
        formal_resume = (
            bool(getattr(
                self.config, 'ct_online_recursive_training', False))
            or bool(getattr(
                self.config, 'ct_formal_resume_contract', False)))
        if formal_resume:
            checkpoint['ct_online_resume_contract'] = (
                build_online_resume_contract(self.config))
            checkpoint['ct_global_rng_state'] = capture_global_rng_state()
            batch_progress = getattr(
                getattr(getattr(
                    getattr(self, '_trainer', None), 'fit_loop', None),
                    'epoch_loop', None), 'batch_progress', None)
            checkpoint['ct_epoch_boundary_complete'] = bool(
                self._ct_epoch_boundary_complete
                or (not self.ct_enable_v27
                    and getattr(batch_progress, 'is_last_batch', False)))
        if hasattr(self, '_ct_b2_method_promotion'):
            checkpoint['ct_b2_method_promotion'] = copy.deepcopy(
                self._ct_b2_method_promotion)
        if hasattr(self, '_ct_acquisition_preflight'):
            checkpoint['ct_acquisition_preflight'] = copy.deepcopy(
                self._ct_acquisition_preflight)
        if hasattr(self, '_ct_b2_promotion'):
            checkpoint['ct_b2_promotion'] = copy.deepcopy(
                self._ct_b2_promotion)
        if hasattr(self, '_b2_v3_init_provenance'):
            checkpoint['b2_v3_init'] = copy.deepcopy(
                self._b2_v3_init_provenance)
        if hasattr(self, '_b2_v3_frozen_reference_hashes'):
            checkpoint['b2_v3_frozen_reference_hashes'] = dict(
                self._b2_v3_frozen_reference_hashes)
        if hasattr(self, '_b1_uncertainty_calibration'):
            checkpoint['b1_uncertainty_calibration'] = copy.deepcopy(
                self._b1_uncertainty_calibration)
        if (self.use_motion_conditioned_search_v3
                and self.use_asymmetric_dual_query):
            checkpoint['b2_v3_candidate_config_sha256'] = (
                b2_candidate_config_sha256(self.config))
        if self.ct_separate_optimizers:
            checkpoint['ct_isolated_scalers'] = {
                name: scaler.state_dict()
                for name, scaler in self._ct_scalers.items()}
        if self.ct_separate_optimizers or self.ct_unified_auto:
            optimizer_lrs = {}
            trainer_optimizers = list(getattr(
                getattr(self, '_trainer', None), 'optimizers', []))
            if self.ct_unified_auto and trainer_optimizers:
                optimizer_lrs = {
                    str(group.get('name', f'group_{index}')):
                    [float(group['lr'])]
                    for index, group in enumerate(
                        trainer_optimizers[0].param_groups)}
            else:
                for name, optimizer in zip(
                        getattr(self, '_ct_optimizer_names', ()),
                        trainer_optimizers):
                    optimizer_lrs[name] = [
                        float(group['lr'])
                        for group in optimizer.param_groups]
            module_hashes = {}
            for name, parameters in getattr(
                    self, '_ct_named_parameters_by_module', {}).items():
                digest = hashlib.sha256()
                for parameter_name, parameter in sorted(parameters):
                    tensor = parameter.detach().cpu().contiguous()
                    digest.update(parameter_name.encode('utf-8'))
                    digest.update(str(tensor.dtype).encode('ascii'))
                    digest.update(str(tuple(tensor.shape)).encode('ascii'))
                    digest.update(tensor.numpy().tobytes())
                module_hashes[name] = digest.hexdigest()
            checkpoint['ct_module_audit'] = {
                'schema': 'ct_seqtrack.module_audit.v2',
                'epoch': int(getattr(self, 'current_epoch', 0)),
                'runtime_protocol': self.ct_runtime_protocol,
                'optimizer_topology': self.ct_optimizer_topology,
                'parameter_groups': list(getattr(
                    self, '_ct_optimizer_names', ())),
                'parameter_sha256': module_hashes,
                'optimizer_lr': optimizer_lrs,
                'update_steps': {
                    name: int(getattr(
                        self, f'ct_{name}_update_step').item())
                    for name in getattr(self, '_ct_optimizer_names', ())},
                'last_gradient_norm': dict(getattr(
                    self, '_ct_last_gradient_norm', {})),
                'max_gradient_norm': dict(getattr(
                    self, '_ct_max_gradient_norm', {})),
                'active_frozen_parameters': [
                    name for name, parameter in self.named_parameters()
                    if not parameter.requires_grad and (
                        not self._ct_any_plugin_parameter(name)
                        or {
                            'b1': self.ct_enable_b1,
                            'b2': self.ct_enable_b2,
                            'b3': self.ct_enable_b3,
                        }[self._ct_plugin_group(name)])],
            }
            checkpoint['ct_b0_prefix_hashes'] = copy.deepcopy(getattr(
                self, '_ct_b0_prefix_hashes', {}))
            checkpoint['ct_b0_optimizer_state_hashes'] = copy.deepcopy(
                getattr(self, '_ct_b0_optimizer_state_hashes', {}))
            checkpoint['ct_cuda_stage_audit'] = copy.deepcopy(getattr(
                self, '_ct_cuda_stage_max', {}))
            checkpoint['ct_observation_batch_fingerprints'] = copy.deepcopy(
                getattr(self, '_ct_observation_batch_fingerprints', []))
            resolved = json.dumps(
                dict(self.config), sort_keys=True,
                separators=(',', ':'), default=str)
            checkpoint['ct_runtime_identity'] = {
                'schema': 'ct_seqtrack.runtime_identity.v1',
                'runtime_protocol': self.ct_runtime_protocol,
                'resolved_config_sha256': hashlib.sha256(
                    resolved.encode('utf-8')).hexdigest(),
                'observation_rng_mode': str(getattr(
                    self.config, 'ct_observation_rng_mode', 'legacy')),
                'validation_rng_mode': str(getattr(
                    self.config, 'ct_validation_rng_mode', 'legacy')),
                'candidate_policy': str(getattr(
                    self.config, 'ct_candidate_policy', 'legacy')),
                'candidate_weights': list(self.ct_b0_candidate_weights),
                'rng_manifest': {
                    'seed': int(getattr(self.config, 'seed', 42) or 42),
                    'epoch': int(getattr(self, 'current_epoch', 0)),
                    'roles': [
                        'observation-shuffle', 'candidate', 'points',
                        'validation'],
                },
            }

    def on_load_checkpoint(self, checkpoint):
        self._ct_b0_prefix_hashes = copy.deepcopy(
            checkpoint.get('ct_b0_prefix_hashes', {}))
        self._ct_b0_optimizer_state_hashes = copy.deepcopy(
            checkpoint.get('ct_b0_optimizer_state_hashes', {}))
        self._ct_observation_batch_fingerprints = copy.deepcopy(
            checkpoint.get('ct_observation_batch_fingerprints', []))
        if ((bool(getattr(
                self.config, 'ct_online_recursive_training', False))
             or bool(getattr(
                self.config, 'ct_formal_resume_contract', False)))
                and not bool(getattr(self.config, 'test', False))):
            # Lightning performs additional setup between checkpoint loading
            # and the first resumed batch.  Restore at on_train_epoch_start,
            # after that setup, so dropout sees the exact saved stream.
            rng_state = checkpoint.get('ct_global_rng_state')
            if (not isinstance(rng_state, dict)
                    or rng_state.get('schema')
                    != 'ct_seqtrack.global_rng.v1'):
                raise ValueError(
                    'exact online resume requires '
                    'ct_seqtrack.global_rng.v1')
            self._ct_pending_global_rng_state = copy.deepcopy(rng_state)
        optional_preflight = checkpoint.get('ct_acquisition_preflight')
        if isinstance(optional_preflight, dict):
            self._ct_acquisition_preflight = copy.deepcopy(optional_preflight)
        if self.ct_separate_optimizers:
            self._ct_pending_scaler_state = copy.deepcopy(
                checkpoint.get('ct_isolated_scalers'))
        if isinstance(checkpoint.get('b2_v3_init'), dict):
            self._b2_v3_init_provenance = copy.deepcopy(
                checkpoint['b2_v3_init'])
        if (self.ct_joint_contract_version >= 3 and self.ct_enable_b3
                and self.ct_initialization_policy != 'scratch_only'):
            promotion = checkpoint.get('ct_b2_promotion')
            if (not isinstance(promotion, dict)
                    or promotion.get('schema')
                    != 'ct_seqtrack.b2_evidence_promotion.v4'
                    or not bool(promotion.get('passed'))):
                raise RuntimeError(
                    "contract-v3 B3 requires a promoted B2 checkpoint")
            self._ct_b2_promotion = copy.deepcopy(promotion)
        optional_method_promotion = checkpoint.get('ct_b2_method_promotion')
        if isinstance(optional_method_promotion, dict):
            self._ct_b2_method_promotion = copy.deepcopy(
                optional_method_promotion)
        if (self.ct_joint_contract_version >= 3 and self.ct_enable_b3
                and self.ct_initialization_policy == 'scratch_only'):
            final_promotion = checkpoint.get('ct_b2_promotion')
            if (isinstance(final_promotion, dict)
                    and final_promotion.get('schema')
                    == 'ct_seqtrack.b2_evidence_promotion.v4'
                    and bool(final_promotion.get('passed'))):
                self._ct_b2_promotion = copy.deepcopy(final_promotion)
        calibration = checkpoint.get('b1_uncertainty_calibration')
        if isinstance(calibration, dict):
            validate_b1_calibration_state(
                calibration, checkpoint.get('state_dict', {}))
        if self.require_b1_calibration_artifact:
            if (not isinstance(calibration, dict)
                    or calibration.get('schema')
                    != 'ct_seqtrack.b1_uncertainty_calibration.v3'
                    or len(calibration.get(
                        'fixed_margin_parallel_perpendicular_95', [])) != 2):
                raise RuntimeError(
                    "this configuration requires a verified v3 B1 "
                    "calibration artifact with fixed residual margins")
            if (self.ct_joint_contract_version >= 3
                    and len(calibration.get(
                        'standardized_abs_residual_q90_parallel_perpendicular',
                        [])) != 2):
                raise RuntimeError(
                    "contract-v3 calibration lacks standardized residual q90")
            source = calibration.get('source_artifact', {})
            evaluation = calibration.get('evaluation_artifact', {})
            if (source.get('partition') != 'calibration'
                    or evaluation.get('partition') != 'dev'
                    or evaluation.get('dataset') != source.get('dataset')
                    or evaluation.get('split') != source.get('split')
                    or evaluation.get('b1_config_sha256') != source.get(
                        'b1_config_sha256')
                    or source.get('dataset') != str(getattr(
                        self.config, 'dataset', 'unknown'))
                    or source.get('split') != str(getattr(
                        self.config, 'train_split', 'train'))
                    or source.get('b1_config_sha256')
                    != b1_calibration_config_sha256(self.config)):
                raise RuntimeError(
                    "B1 calibration dataset/partition/config mismatch")
        if self.require_b1_calibration_passed:
            if (not isinstance(calibration, dict)
                    or not bool(calibration.get(
                        'promotion', {}).get('passed'))):
                raise RuntimeError(
                    "this configuration requires a promoted B1 calibration")
        if isinstance(calibration, dict):
            self._b1_uncertainty_calibration = copy.deepcopy(calibration)
            standardized_q90 = calibration.get(
                'standardized_abs_residual_q90_parallel_perpendicular')
            if (isinstance(standardized_q90, (list, tuple))
                    and len(standardized_q90) == 2):
                self.config[
                    'search_v3_standardized_residual_q90_parallel_perpendicular'
                ] = [float(value) for value in standardized_q90]
        if bool(getattr(
                self.config, 'require_b2_candidate_config_contract', False)):
            stored_candidate_config = checkpoint.get(
                'b2_v3_candidate_config_sha256')
            expected_candidate_config = b2_candidate_config_sha256(
                self.config)
            if (not stored_candidate_config
                    or stored_candidate_config != expected_candidate_config):
                raise RuntimeError(
                    "B2 candidate checkpoint/config contract mismatch")
        if self.use_recursive_replay_cache:
            cache_dir = getattr(
                self.config, 'recursive_replay_cache_dir', None)
            state = checkpoint.get('state_dict', {})
            if not cache_dir or not isinstance(state, dict):
                raise RuntimeError(
                    "formal replay resume lacks cache/state identity")
            completed = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=Path(__file__).resolve().parents[1],
                check=True, capture_output=True, text=True)
            validate_replay_cache_manifest(cache_dir, expected_manifest={
                'dataset': str(getattr(
                    self.config, 'dataset', 'unknown')),
                'split': str(getattr(
                    self.config, 'train_split', 'train')),
                'replay_config_sha256': replay_config_sha256(self.config),
                'commit': completed.stdout.strip(),
                'b0_state_sha256': tensor_prefixes_sha256(
                    state, B0_STATE_PREFIXES),
                'b1_state_sha256': tensor_prefixes_sha256(
                    state, B1_STATE_PREFIXES),
                'b1_calibration_sha256': sha256_json(calibration),
            })
        if self.b2_v3_require_packaged_router:
            validate_b2_v3_router_package(
                checkpoint, router=self.action_consistent_router_v3)
        if self.b2_v3_freeze_candidate_producers:
            stored_hashes = checkpoint.get(
                'b2_v3_frozen_reference_hashes')
            if stored_hashes is not None:
                self._b2_v3_frozen_reference_hashes = dict(stored_hashes)
    def on_before_optimizer_step(self, optimizer):
        if self.ct_unified_auto:
            active = set()
            norms = {}
            for name, named_parameters in getattr(
                    self, '_ct_named_parameters_by_module', {}).items():
                gradients = [
                    parameter.grad.detach()
                    for _, parameter in named_parameters
                    if parameter.grad is not None]
                if gradients:
                    active.add(name)
                    norms[name] = float(torch.linalg.vector_norm(torch.stack([
                        torch.linalg.vector_norm(value)
                        for value in gradients])).detach().cpu())
            self._ct_pending_auto_updates = active
            previous = dict(getattr(self, '_ct_last_gradient_norm', {}))
            previous.update(norms)
            self._ct_last_gradient_norm = previous
            maximum = dict(getattr(self, '_ct_max_gradient_norm', {}))
            for name, value in norms.items():
                maximum[name] = max(float(value), float(maximum.get(name, 0.0)))
            self._ct_max_gradient_norm = maximum

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.ct_unified_auto:
            active = set(getattr(
                self, '_ct_pending_auto_updates', set()))
            for name in active:
                getattr(self, f'ct_{name}_update_step').add_(1)
                setattr(self, f'_ct_{name}_updated_this_epoch', True)
            if active & {'b1', 'b2', 'b3'}:
                self.ct_plugin_update_step.add_(1)
                self._ct_plugin_updated_this_epoch = True
            if 'b0' in active:
                update_step = int(self.ct_b0_update_step.item())
                if update_step in (1, 100):
                    self._ct_b0_prefix_hashes[f'step_{update_step}'] = (
                        self._ct_module_parameter_hash('b0'))
                    optimizer = self.trainer.optimizers[0]
                    self._ct_b0_optimizer_state_hashes[
                        f'step_{update_step}'] = (
                            self._ct_optimizer_state_hash('b0', optimizer))
            self._ct_pending_auto_updates = set()
            self._ct_record_cuda_stage('step')
            for stage, values in self._ct_cuda_stage_current.items():
                for name, value in values.items():
                    self.log(
                        f'ct_cuda/{stage}_{name}', value,
                        on_step=True, on_epoch=False,
                        prog_bar=False, logger=True)
        if self.b2_v3_freeze_candidate_producers:
            verified = int(getattr(
                self, '_b2_v3_verified_optimizer_steps', 0))
            if verified < 2:
                self._verify_b2_v3_frozen_hashes()
                self._b2_v3_verified_optimizer_steps = verified + 1
        if self.use_decoder_token_consistency:
            self.decoder_token_consistency.update_teacher()

    def on_train_epoch_end(self):
        if (self.ct_separate_optimizers
                and self.config.optimizer.lower() != 'adamonecycle'):
            schedulers = self.lr_schedulers()
            if not isinstance(schedulers, (list, tuple)):
                schedulers = [schedulers]
            for name, scheduler in zip(
                    self._ct_optimizer_names, schedulers):
                if getattr(
                        self, f'_ct_{name}_updated_this_epoch', False):
                    scheduler.step()
        if (self.ct_separate_optimizers or self.ct_unified_auto) and 'b0' in getattr(
                self, '_ct_optimizer_names', ()):
            trace = getattr(self, '_ct_b0_prefix_hashes', {})
            trace[f'epoch_{int(self.current_epoch) + 1:03d}'] = (
                self._ct_module_parameter_hash('b0'))
            self._ct_b0_prefix_hashes = trace
        self._ct_epoch_boundary_complete = True
        acquisition = getattr(self, '_ct_epoch_acquisition_totals', {})
        for population, totals in acquisition.items():
            eligible = totals['eligible_rows']
            retained = totals['retained_rows']
            pool_targets = totals['pool_targets']
            sampled_targets = totals['sampled_targets']
            totals['row_recall'] = (
                retained / eligible if eligible > 0 else None)
            totals['point_recall'] = (
                sampled_targets / pool_targets
                if pool_targets > 0 else None)
            for metric in ('eligible_rows', 'retained_rows',
                           'pool_targets', 'sampled_targets',
                           'recovery_positive_rows',
                           'recovery_fallback_rows'):
                self.log(
                    f'ct_acquisition/{population}_{metric}',
                    float(totals[metric]), on_step=False, on_epoch=True)
            for metric in ('row_recall', 'point_recall'):
                if totals[metric] is not None:
                    self.log(
                        f'ct_acquisition/{population}_{metric}',
                        float(totals[metric]), on_step=False, on_epoch=True)
        if acquisition and int(getattr(self, 'global_rank', 0)) == 0:
            logger = getattr(self, 'logger', None)
            log_dir = getattr(logger, 'log_dir', None)
            if log_dir is None:
                log_dir = getattr(logger, 'save_dir', '.')
            output = Path(log_dir) / 'acquisition_supply' / (
                f'epoch_{int(self.current_epoch) + 1:02d}.json')
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps({
                'schema': 'ct_seqtrack.acquisition_training_supply.v1',
                'experiment_name': str(getattr(
                    self.config, 'experiment_name', 'unknown')),
                'seed': int(getattr(self.config, 'seed', 42) or 42),
                'epoch': int(self.current_epoch) + 1,
                'populations': acquisition,
                'targetness_balance': {
                    'source': self.ct_targetness_class_weight_source,
                    'positive_points': float(getattr(
                        self, 'ct_targetness_running_positive_points',
                        torch.zeros(())).detach().cpu()),
                    'negative_points': float(getattr(
                        self, 'ct_targetness_running_negative_points',
                        torch.zeros(())).detach().cpu()),
                    'positive_weight': float(getattr(
                        self, 'ct_targetness_running_positive_weight',
                        torch.ones(())).detach().cpu()),
                    'negative_weight': float(getattr(
                        self, 'ct_targetness_running_negative_weight',
                        torch.ones(())).detach().cpu()),
                },
            }, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        if self.b2_v3_freeze_candidate_producers:
            self._verify_b2_v3_frozen_hashes()
        rows = getattr(self, '_ct_epoch_binary_rows', {})
        if not (self.ct_joint_contract_version >= 2 and rows):
            return
        epoch_metrics = {}
        binary_names = ('presence', 'help', 'harm') if self.ct_enable_v27 else ('presence', 'alpha')
        for name in binary_names:
            entries = rows.get(name, [])
            if not entries:
                continue
            scores = np.concatenate([entry[0] for entry in entries])
            targets = np.concatenate([entry[1] for entry in entries])
            metrics = self._binary_curve_metrics_numpy(scores, targets)
            epoch_metrics.update({
                f'{name}_auroc': metrics['auroc'],
                f'{name}_auprc': metrics['auprc'],
                f'{name}_ap': metrics['auprc'],
                f'{name}_ece': metrics['ece'],
                f'{name}_positive_mean': metrics['positive_mean'],
                f'{name}_negative_mean': metrics['negative_mean'],
                f'{name}_positive_count': metrics['positive_count'],
                f'{name}_negative_count': metrics['negative_count'],
            })
            for bin_index, calibration in enumerate(metrics['calibration']):
                epoch_metrics.update({
                    f'{name}_calibration_bin{bin_index}_count':
                        calibration['count'],
                    f'{name}_calibration_bin{bin_index}_confidence':
                        calibration['confidence'],
                    f'{name}_calibration_bin{bin_index}_positive_rate':
                        calibration['positive_rate'],
                })
        alpha_uplift = rows.get('alpha_uplift', [])
        if alpha_uplift:
            epoch_metrics['alpha_counterfactual_uplift'] = float(np.mean(
                np.concatenate(alpha_uplift)))
        utility_rows = rows.get('bounded_utility', [])
        if utility_rows:
            utility = np.concatenate(utility_rows, axis=0)
            epoch_metrics['bounded_success_gain_mean'] = float(utility[:, 0].mean())
            epoch_metrics['bounded_precision_gain_mean'] = float(utility[:, 1].mean())
            epoch_metrics['bounded_utility_gain_mean'] = float(utility[:, :2].mean())
            if self.ct_enable_b3:
                epoch_metrics['utility_score_mse'] = float(np.mean(
                    (utility[:, 2] - utility[:, :2].mean(axis=1)) ** 2))
        if epoch_metrics:
            self.logger.experiment.add_scalars(
                'ct_epoch_calibration', epoch_metrics,
                global_step=self.global_step)

    @staticmethod
    def _binary_curve_metrics_numpy(scores, targets):
        """Exact epoch-level rank metrics plus five-bin calibration."""
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        targets = np.asarray(targets, dtype=np.float64).reshape(-1)
        finite = np.isfinite(scores) & np.isfinite(targets)
        scores = np.clip(scores[finite], 0.0, 1.0)
        targets = targets[finite] > 0.5
        positive_count = int(targets.sum())
        negative_count = int((~targets).sum())
        auroc = 0.5
        if positive_count and negative_count:
            order = np.argsort(scores, kind='mergesort')
            sorted_scores = scores[order]
            ranks = np.arange(1, len(scores) + 1, dtype=np.float64)
            starts = np.r_[0, np.flatnonzero(
                sorted_scores[1:] != sorted_scores[:-1]) + 1]
            ends = np.r_[starts[1:], len(scores)]
            for start, end in zip(starts, ends):
                ranks[start:end] = 0.5 * (start + 1 + end)
            inverse = np.empty_like(order)
            inverse[order] = np.arange(len(order))
            positive_rank_sum = ranks[inverse][targets].sum()
            auroc = float((
                positive_rank_sum
                - positive_count * (positive_count + 1) / 2.0
            ) / (positive_count * negative_count))
        auprc = 0.0
        if positive_count:
            order = np.argsort(-scores, kind='mergesort')
            sorted_scores = scores[order]
            sorted_targets = targets[order].astype(np.float64)
            true_positive = np.cumsum(sorted_targets)
            threshold_ends = np.r_[np.flatnonzero(
                sorted_scores[1:] != sorted_scores[:-1]),
                len(sorted_scores) - 1]
            true_positive = true_positive[threshold_ends]
            predicted_positive = threshold_ends.astype(np.float64) + 1.0
            precision = true_positive / predicted_positive
            recall = true_positive / positive_count
            auprc = float(np.sum(
                np.diff(np.r_[0.0, recall]) * precision))
        calibration = []
        for bin_index in range(5):
            lower = bin_index / 5.0
            upper = (bin_index + 1) / 5.0
            selected = (
                (scores >= lower)
                & (scores < upper if bin_index < 4 else scores <= upper))
            calibration.append({
                'count': int(selected.sum()),
                'confidence': float(scores[selected].mean())
                if selected.any() else 0.0,
                'positive_rate': float(targets[selected].mean())
                if selected.any() else 0.0,
            })
        sample_count = max(int(scores.size), 1)
        ece = float(sum(
            item['count'] / sample_count
            * abs(item['confidence'] - item['positive_rate'])
            for item in calibration))
        return {
            'auroc': auroc,
            'auprc': auprc,
            'ece': ece,
            'positive_mean': float(scores[targets].mean())
            if positive_count else 0.0,
            'negative_mean': float(scores[~targets].mean())
            if negative_count else 0.0,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'calibration': calibration,
        }

    def _accumulate_joint_binary_rows(self, data, output):
        if not (self.training and self.ct_joint_contract_version >= 2
                and self.use_ct_joint_full and self.ct_enable_b2):
            return
        if not hasattr(self, '_ct_epoch_binary_rows'):
            self._ct_epoch_binary_rows = {
                'presence': [], 'alpha': [], 'alpha_uplift': []}
        if self.ct_enable_v27:
            from utils.v27_training import accumulate_v27_binary_rows
            accumulate_v27_binary_rows(
                self._ct_epoch_binary_rows, data, output, self.ct_enable_b3)
            return
        if self.ct_joint_contract_version >= 3:
            action_probability = contract_v3_action_probability(
                output, self.ct_enable_b3)
            labels = data['ct_extension_labels'].detach()
            point_valid = data['ct_extension_valid_mask'].detach()
            support_valid = output['ct_b2_available'].detach().reshape(-1)
            presence_target = (
                (labels * point_valid).sum(dim=1) >= 1)
            presence_select = support_valid > 0
            if bool(presence_select.any()):
                self._ct_epoch_binary_rows['presence'].append((
                    output[
                        'ct_b2_extension_presence_probability'].detach()[
                            presence_select].float().cpu().numpy(),
                    presence_target[
                        presence_select].float().cpu().numpy(),
                ))
            target_xy = data['box_label'][:, :2].to(
                device=output['ct_b2_raw_box'].device,
                dtype=output['ct_b2_raw_box'].dtype)
            observation_xy = output[
                'observation_aux_estimation_boxes'][:, :2].detach()
            raw_xy = output['ct_b2_raw_box'][:, :2].detach()
            gain = (
                torch.linalg.norm(observation_xy - target_xy, dim=1)
                - torch.linalg.norm(raw_xy - target_xy, dim=1))
            helpful = gain > self.ct_router_help_margin
            harmful = (
                (gain < -self.ct_router_help_margin) | ~presence_target)
            utility_valid = (support_valid > 0) & (helpful | harmful)
            if bool(utility_valid.any()):
                self._ct_epoch_binary_rows['alpha_uplift'].append(
                    gain[utility_valid].float().cpu().numpy())
                if action_probability is not None:
                    if action_probability.shape != support_valid.shape:
                        raise RuntimeError(
                            "ct_b3_action_score must align with canonical "
                            "B2 rows")
                    self._ct_epoch_binary_rows['alpha'].append((
                        action_probability[
                            utility_valid].float().cpu().numpy(),
                        helpful[utility_valid].float().cpu().numpy(),
                    ))
            return
        endpoint_labels = data['search_v3_point_labels'].detach()
        tube_labels = data['trajectory_search_point_labels'].detach().reshape(
            -1, self.ct_tube_quota)
        labels = torch.cat((endpoint_labels, tube_labels), dim=1)
        endpoint_source = data['search_v3_point_source'].detach()
        tube_source = data[
            'trajectory_search_point_source'].detach().reshape(
                -1, self.ct_tube_quota)
        sources = torch.cat((endpoint_source, tube_source), dim=1)
        endpoint_valid = data['search_v3_point_valid_mask'].detach()
        tube_valid = data[
            'trajectory_search_point_valid_mask'].detach().reshape(
                -1, self.ct_tube_quota)
        point_valid = torch.cat((endpoint_valid, tube_valid), dim=1)
        support_valid = output['ct_search_support_valid'].detach().reshape(-1)
        extension_foreground = (
            labels * point_valid * (sources > 0).to(labels.dtype)).sum(dim=1)
        presence_target = extension_foreground >= 1
        presence_select = support_valid > 0
        if bool(presence_select.any()):
            self._ct_epoch_binary_rows['presence'].append((
                output['ct_search_presence_probability'].detach()[
                    presence_select].float().cpu().numpy(),
                presence_target[presence_select].float().cpu().numpy(),
            ))
        if not (self.ct_enable_b1
                and self.ct_enable_query_reliability_gate
                and self.ct_query_counterfactual_supervision):
            return
        target_xy = data['box_label'][:, :2].to(
            device=output['ct_search_raw_obs_xy'].device,
            dtype=output['ct_search_raw_obs_xy'].dtype)
        counterfactual = counterfactual_query_targets(
            output['ct_search_raw_obs_xy'],
            output['ct_search_raw_motion_xy'], target_xy,
            margin=self.ct_query_counterfactual_margin)
        obs_error = counterfactual['obs_error']
        motion_error = counterfactual['motion_error']
        helpful = counterfactual['helpful']
        harmful = counterfactual['harmful']
        alpha_valid = (
            (output['motion_prior_valid'].detach() > 0)
            & (support_valid > 0)
            & (output['ct_search_candidate_valid'].detach() > 0)
            & (counterfactual['valid'] > 0))
        if bool(alpha_valid.any()):
            self._ct_epoch_binary_rows['alpha'].append((
                torch.sigmoid(output['ct_query_gate_logit'].detach())[
                    alpha_valid].float().cpu().numpy(),
                helpful[alpha_valid].float().cpu().numpy(),
            ))
            self._ct_epoch_binary_rows['alpha_uplift'].append(
                counterfactual['uplift'][
                    alpha_valid].float().cpu().numpy())

    def encode_point_time(self, points):
        encoded_time = self.time_encoder(points[..., 3:4])
        return torch.cat((points[..., :3], encoded_time, points[..., 4:]), dim=-1)

    @staticmethod
    def is_paired_batch(batch):
        return isinstance(batch, dict) and "view_a" in batch and "view_b" in batch

    def build_observability_stats(self, input_dict, seg_logits, chunk_size):
        B = seg_logits.shape[0]
        device = seg_logits.device
        dtype = seg_logits.dtype

        if "num_points_in_search" in input_dict:
            num_points = input_dict["num_points_in_search"]
            if not torch.is_tensor(num_points):
                num_points = torch.as_tensor(num_points, device=device, dtype=dtype)
            num_points = num_points.to(device=device, dtype=dtype).reshape(B)
        else:
            num_points = seg_logits.new_full((B,), float(chunk_size))

        current_logits = seg_logits[:, :, -chunk_size:]
        fg_prob = torch.softmax(current_logits, dim=1)[:, 1, :]
        if getattr(self.config, "obs_stats_detach_seg", True):
            fg_prob = fg_prob.detach()
        soft_fg_count = fg_prob.sum(dim=1)
        mean_fg_score = fg_prob.mean(dim=1)
        bounded_fg_prob = torch.clamp(fg_prob, min=1e-6, max=1.0 - 1e-6)
        segmentation_entropy = -(
            bounded_fg_prob * torch.log(bounded_fg_prob)
            + (1.0 - bounded_fg_prob)
            * torch.log(1.0 - bounded_fg_prob)).mean(dim=1)
        estimated_fg_points = mean_fg_score * torch.clamp(num_points, min=0.0)

        valid_history_ratio = input_dict["valid_mask"].to(device=device, dtype=dtype).mean(dim=1)
        default_time_scale = getattr(self.config, 'default_time_step', getattr(self.config, 'time_step', 0.5))
        time_scale = max(float(getattr(self.config, "time_scale", default_time_scale)), 1e-6)
        observation_delta_t, real_delta_t, effective_delta_t = (
            resolve_observation_delta_t(
                input_dict,
                seg_logits,
                use_ct_v2=self.use_ct_v2,
                default_time_step=default_time_scale,
            ))
        current_delta_t_ratio = observation_delta_t / time_scale
        real_delta_t_ratio = real_delta_t / time_scale
        effective_delta_t_ratio = effective_delta_t / time_scale

        obs_stats = torch.stack((
            torch.log1p(torch.clamp(num_points, min=0.0)),
            torch.log1p(torch.clamp(estimated_fg_points, min=0.0)),
            mean_fg_score,
            valid_history_ratio,
            current_delta_t_ratio,
        ), dim=1)
        obs_stats = torch.nan_to_num(obs_stats, nan=0.0, posinf=0.0, neginf=0.0)

        obs_aux = {
            "obs_stats": obs_stats,
            "obs_num_points_search": num_points,
            "obs_soft_fg_count": soft_fg_count,
            "obs_estimated_fg_points": estimated_fg_points,
            "obs_mean_fg_score": mean_fg_score,
            "obs_segmentation_entropy": segmentation_entropy,
            "obs_valid_history_ratio": valid_history_ratio,
            "obs_current_delta_t_ratio": current_delta_t_ratio,
            "obs_current_delta_t_real_ratio": real_delta_t_ratio,
            "obs_current_delta_t_effective_ratio": effective_delta_t_ratio,
        }
        return obs_stats, obs_aux

    def apply_bounded_dynamics_residual(self, motion_obs_pred, dynamics_displacement_pred,
                                        dynamics_valid, obs_stats, obs_aux, input_dict):
        displacement_clamped, raw_norm, clamp_mask = clamp_vector_norm(
            dynamics_displacement_pred,
            self.dynamics_max_residual_norm,
            eps=getattr(self.config, 'dynamics_eps', 1e-3),
        )
        clamped_norm = torch.linalg.norm(displacement_clamped, dim=1, keepdim=True)
        obs_dyn_gap = torch.linalg.norm(
            motion_obs_pred[:, :3] - displacement_clamped, dim=1, keepdim=True)

        gate_stats = torch.cat((obs_stats, torch.log1p(obs_dyn_gap)), dim=1)
        gate_stats = torch.nan_to_num(gate_stats, nan=0.0, posinf=0.0, neginf=0.0)
        if self.dynamics_residual_detach_stats:
            gate_stats = gate_stats.detach()
        alpha_dyn = self.dynamics_residual_gate(gate_stats, dynamics_valid)

        condition_mask = torch.ones_like(alpha_dyn)
        current_delta_t = input_dict.get('current_delta_t')
        if current_delta_t is None:
            default_step = getattr(
                self.config, 'default_time_step', getattr(self.config, 'time_step', 0.5))
            current_delta_t = alpha_dyn.new_full((alpha_dyn.shape[0],), float(default_step))
        else:
            current_delta_t = current_delta_t.to(
                device=alpha_dyn.device, dtype=alpha_dyn.dtype).reshape(alpha_dyn.shape[0])
        if self.dynamics_long_gap_only:
            condition_mask = condition_mask * (
                current_delta_t.unsqueeze(1) >= self.dynamics_min_delta_t).to(alpha_dyn.dtype)
        if self.dynamics_sparse_only:
            num_points = obs_aux['obs_num_points_search'].to(
                device=alpha_dyn.device, dtype=alpha_dyn.dtype).reshape(alpha_dyn.shape[0], 1)
            condition_mask = condition_mask * (
                num_points <= self.dynamics_sparse_point_threshold).to(alpha_dyn.dtype)
        alpha_dyn = alpha_dyn * condition_mask

        effective_scale = self.dynamics_residual_scale
        if self.training and getattr(self, 'current_epoch', 0) < self.dynamics_warmup_epoch:
            effective_scale = 0.0
        effective_scale_tensor = motion_obs_pred.new_tensor(effective_scale)
        dynamics_residual = effective_scale_tensor * alpha_dyn * displacement_clamped
        motion_pred = torch.cat((
            motion_obs_pred[:, :3] + dynamics_residual,
            motion_obs_pred[:, 3:4],
        ), dim=1)

        applied_mask = torch.linalg.norm(
            dynamics_residual, dim=1, keepdim=True) > 1e-8
        aux = {
            'motion_obs_pred': motion_obs_pred,
            'motion_dynamics_residual': dynamics_residual,
            'dynamics_displacement_clamped': displacement_clamped,
            'dynamics_residual_alpha': alpha_dyn.squeeze(1),
            'dynamics_residual_scale_effective': effective_scale_tensor,
            'dynamics_residual_raw_norm': raw_norm.squeeze(1),
            'dynamics_residual_clamped_norm': clamped_norm.squeeze(1),
            'dynamics_residual_clamp_mask': clamp_mask.squeeze(1).to(motion_obs_pred.dtype),
            'dynamics_residual_applied_mask': applied_mask.squeeze(1).to(motion_obs_pred.dtype),
            'obs_dyn_center_gap': obs_dyn_gap.squeeze(1),
        }
        return motion_pred, aux

    @torch.no_grad()
    def predict_motion_from_history(
            self, ref_boxs, delta_t, valid_mask, current_delta_t,
            acquisition_features=None):
        """Public box-only B1 pre-pass with the paper-facing contract."""
        if not self.use_b1motion_v3:
            raise RuntimeError(
                "predict_motion_from_history requires use_b1motion_v3")
        device = next(self.physical_motion_encoder.parameters()).device
        dtype = next(self.physical_motion_encoder.parameters()).dtype

        def as_tensor(value, tensor_dtype=dtype):
            value = torch.as_tensor(
                value, device=device, dtype=tensor_dtype)
            return value

        ref_boxs = as_tensor(ref_boxs)
        delta_t = as_tensor(delta_t)
        valid_mask = as_tensor(valid_mask)
        current_delta_t = as_tensor(current_delta_t)
        if ref_boxs.dim() == 2:
            ref_boxs = ref_boxs.unsqueeze(0)
        if delta_t.dim() == 1:
            delta_t = delta_t.unsqueeze(0)
        if valid_mask.dim() == 1:
            valid_mask = valid_mask.unsqueeze(0)
        if current_delta_t.dim() == 0:
            current_delta_t = current_delta_t.unsqueeze(0)
        motion_ref_boxs = (
            torch.flip(ref_boxs, dims=(1,))
            if bool(getattr(self.config, 'shuffle_b1_signal', False))
            else ref_boxs)
        motion_kwargs = {}
        if getattr(self, 'ct_enable_v27', False):
            if acquisition_features is None:
                raise ValueError('v27 prepass requires causal acquisition features')
            features = as_tensor(acquisition_features).reshape(ref_boxs.shape[0], 17)
            motion_kwargs['acquisition_features'] = features
        prediction = self.physical_motion_encoder(
            motion_ref_boxs, delta_t, valid_mask, current_delta_t, **motion_kwargs)
        if bool(getattr(self.config, 'force_b1_invalid', False)):
            prediction = dict(prediction)
            prediction['valid'] = torch.zeros_like(prediction['valid'])
            prediction['source_id'] = torch.zeros_like(
                prediction['source_id'])
        return {
            key: prediction[key].detach()
            for key in (
                "mu_xy", "kinematic_prior_xy",
                "log_sigma_parallel_perp", "covariance_xy",
                "basis_velocity_xy", "direction_xy", "velocity_xy",
                "feature", "valid", "gap_ratio", "source_id",
                "acquisition_margin_parallel_perp")
        }

    def _build_motion_prepass_inputs_contract(
            self, history_boxes, history_ids, valid_mask,
            history_timestamps, current_timestamp,
            effective_history_timestamps, effective_current_timestamp,
            dynamics_time_mode_value, current_frame_id,
            history_quality=None, recursive_age=None, first_frame_wlh=None):
        """Build the shared causal box/time-only B1 tensor primitives."""
        if int(current_frame_id) <= 0:
            raise ValueError("motion pre-pass is only defined after frame 0")
        if len(history_boxes) != self.hist_num:
            return None
        default_step = float(getattr(
            self.config, "default_time_step",
            getattr(self.config, "time_step", 0.1)))
        pseudo_step = float(getattr(
            self.config, "pseudo_time_step", 0.1))
        real_fields = build_time_fields(
            history_timestamps,
            current_timestamp,
            frame_ids=history_ids,
            current_frame_id=current_frame_id,
            use_real_time=bool(getattr(
                self.config, "use_real_time", True)),
            default_step=default_step,
            pseudo_step=pseudo_step,
        )
        time_mode = normalize_dynamics_time_mode(dynamics_time_mode_value)
        effective_fields = build_effective_time_fields(
            time_mode,
            real_fields,
            effective_frame_timestamps=effective_history_timestamps,
            effective_current_timestamp=effective_current_timestamp,
            frame_ids=history_ids,
            current_frame_id=current_frame_id,
            default_step=float(getattr(
                self.config, "dynamics_fixed_delta_t", default_step)),
            pseudo_step=pseudo_step,
        )
        effective_delta_t = effective_fields[1]
        if getattr(self, 'ct_enable_v27', False):
            from utils.b1_acquisition import build_b1_input_arrays
            return build_b1_input_arrays(
                history_boxes, effective_delta_t, valid_mask,
                history_quality=history_quality,
                recursive_age=max(0, int(current_frame_id) - 1) if recursive_age is None else recursive_age,
                first_frame_wlh=first_frame_wlh, degrees=self.config.degrees,
                time_scale=float(getattr(self.config, 'time_scale', .5)))
        anchor = history_boxes[0]
        local_rows = []
        for box in history_boxes:
            local_box = points_utils.transform_box(box, anchor)
            yaw = (
                local_box.orientation.degrees
                * local_box.orientation.axis[-1]
                if bool(getattr(self.config, "degrees", True))
                else local_box.orientation.radians
                * local_box.orientation.axis[-1])
            local_rows.append(np.append(
                local_box.center, yaw).astype(np.float32))
        return {
            "ref_boxs": np.stack(local_rows, axis=0),
            "delta_t": np.asarray(effective_delta_t, dtype=np.float32),
            "valid_mask": np.asarray(valid_mask, dtype=np.float32),
            "current_delta_t": np.float32(effective_delta_t[0]),
        }

    def _empty_motion_prepass_prediction(self):
        feature_dim = int(getattr(
            self.config, "motion_v3_hidden_dim", 128))
        return {
            "mu_xy": np.zeros(2, dtype=np.float32),
            "kinematic_prior_xy": np.zeros(2, dtype=np.float32),
            "log_sigma_parallel_perp": np.zeros(2, dtype=np.float32),
            "acquisition_margin_parallel_perp": np.asarray(
                (2.0, 1.0), dtype=np.float32),
            "covariance_xy": np.eye(2, dtype=np.float32),
            "basis_velocity_xy": np.zeros(2, dtype=np.float32),
            "direction_xy": np.asarray((1.0, 0.0), dtype=np.float32),
            "velocity_xy": np.zeros(2, dtype=np.float32),
            "feature": np.zeros(feature_dim, dtype=np.float32),
            "valid": False,
            "gap_ratio": 1.0,
            "source_id": 0,
            "current_delta_t": float(getattr(
                self.config, "default_time_step", 0.5)),
        }

    @staticmethod
    def _unbatch_motion_prepass_predictions(
            tensor_prediction, current_delta_t):
        results = []
        for row in range(len(current_delta_t)):
            result = {}
            for key, value in tensor_prediction.items():
                item = value[row].detach().cpu()
                if key == "valid":
                    result[key] = bool(item.item() > 0)
                elif key == "source_id":
                    result[key] = int(item.item())
                elif key == "gap_ratio":
                    result[key] = float(item.item())
                else:
                    result[key] = item.numpy()
            result["current_delta_t"] = float(current_delta_t[row])
            finite_keys = (
                "mu_xy", "log_sigma_parallel_perp", "direction_xy",
                "velocity_xy", "acquisition_margin_parallel_perp")
            numerically_valid = all(
                key in result and bool(np.isfinite(result[key]).all())
                for key in finite_keys)
            if numerically_valid:
                result["log_sigma_parallel_perp"] = np.clip(
                    np.asarray(result["log_sigma_parallel_perp"],
                               dtype=np.float32), -4.0, 2.5)
            else:
                result["valid"] = False
                result["source_id"] = 0
            results.append(result)
        return results

    @torch.no_grad()
    def _predict_motion_prepass_contract(
            self, history_boxes, history_ids, valid_mask,
            history_timestamps, current_timestamp,
            effective_history_timestamps, effective_current_timestamp,
            dynamics_time_mode_value, current_frame_id,
            history_quality=None, recursive_age=None, first_frame_wlh=None):
        """Execute one row through the shared box/time-only B1 contract."""
        inputs = self._build_motion_prepass_inputs_contract(
            history_boxes, history_ids, valid_mask,
            history_timestamps, current_timestamp,
            effective_history_timestamps, effective_current_timestamp,
            dynamics_time_mode_value, current_frame_id,
            **(dict(history_quality=history_quality, recursive_age=recursive_age,
                    first_frame_wlh=first_frame_wlh)
               if getattr(self, 'ct_enable_v27', False) else {}))
        if inputs is None:
            return self._empty_motion_prepass_prediction()
        prediction = self.predict_motion_from_history(
            inputs["ref_boxs"], inputs["delta_t"], inputs["valid_mask"],
            inputs["current_delta_t"],
            **({'acquisition_features': inputs['acquisition_features']}
               if getattr(self, 'ct_enable_v27', False) else {}))
        result = self._unbatch_motion_prepass_predictions(
            prediction, [inputs["current_delta_t"]])[0]
        if getattr(self, 'ct_enable_v27', False):
            from utils.b1_acquisition import b1_input_digest
            result['input_digest'] = b1_input_digest(inputs)
            result['parameter_revision'] = int(getattr(self, 'global_step', 0))
        return result

    @torch.no_grad()
    def predict_motion_prepass(self, sequence, frame_id, results_bbs, recursive_state=None):
        """Build and execute B1 before the current point cloud is cropped.

        Only recursive boxes and physical timestamps are read.  In
        particular, the current ``3d_bbox`` annotation is outside this data
        path, preventing online ground-truth leakage.
        """
        if frame_id <= 0:
            raise ValueError("motion pre-pass is only defined after frame 0")
        history_ids, valid_mask = get_history_frame_ids_and_masks(
            frame_id, self.hist_num)
        history_boxes = get_last_n_bounding_boxes(results_bbs, valid_mask)
        previous_frames = [sequence[index] for index in history_ids]
        current_frame = sequence[frame_id]
        return self._predict_motion_prepass_contract(
            history_boxes,
            history_ids,
            valid_mask,
            [frame.get("timestamp") for frame in previous_frames],
            current_frame.get("timestamp"),
            [frame.get("_ct_effective_timestamp")
             for frame in previous_frames],
            current_frame.get("_ct_effective_timestamp"),
            current_frame.get(
                "_ct_dynamics_time_mode",
                getattr(self.config, "dynamics_time_mode", "true")),
            frame_id,
            **(dict(history_quality=np.stack([
                recursive_state.quality.get(index, np.zeros(4)) for index in history_ids])
                if recursive_state is not None else None,
                recursive_age=recursive_state.rollout_age(frame_id)
                if recursive_state is not None else frame_id - 1,
                first_frame_wlh=results_bbs[0].wlh)
               if getattr(self, 'ct_enable_v27', False) else {}),
        )


    def _forward_ct_contract_v3(
            self, input_dict, output_dict, observation_box,
            observation_stats, main_motion, history_valid_ratio,
            batch_size, frame_count, chunk_size, coarse_box=None):
        """Run the extension-query/full-base-memory contract-v3 path."""
        required = (
            'ct_base_evidence_points', 'ct_base_evidence_valid_mask',
            'ct_extension_points', 'ct_extension_valid_mask',
            'ct_extension_source', 'ct_search_support_valid',
            'search_v3_query_delta_t',
            'search_v3_gap_ratio', 'search_v3_support_anchor_xy',
            'motion_source_anchor', 'coordinate_anchor',
            'ref_boxs', 'bbox_size', 'valid_mask')
        missing = [key for key in required if key not in input_dict]
        if missing:
            raise KeyError(
                "CT contract-v3 input is missing: " + ", ".join(missing))
        aligned_features = output_dict['b0_point_aligned_features']
        if aligned_features.shape != (
                batch_size, frame_count, chunk_size, 64):
            raise ValueError(
                "B0 point-aligned feature contract must be [B,L,1024,64]")
        if chunk_size != 1024:
            raise ValueError("contract-v3 requires exactly 1024 B0 points")
        raw_frames = input_dict['points'].reshape(
            batch_size, frame_count, chunk_size, -1)
        observation_contract = self.b0_observation_contract(
            observation_box, observation_stats,
            current_features=aligned_features[:, -1],
            history_features=aligned_features[:, :-1])
        observation_box = observation_contract.box
        observation_stats = observation_contract.statistics
        history_features = observation_contract.history_features
        current_base_features = observation_contract.current_features
        timestamps_real = input_dict.get(
            'timestamps_effective' if self.ct_enable_v27 else 'timestamps_real')
        history_timestamps = None
        current_timestamp = None
        if timestamps_real is not None:
            timestamps_real = timestamps_real.to(
                device=current_base_features.device,
                dtype=current_base_features.dtype)
            history_timestamps = timestamps_real[:, :-1]
            current_timestamp = timestamps_real[:, -1]
        memory_result = build_box_memory_tokens(
            history_features,
            raw_frames[:, :-1],
            input_dict['ref_boxs'],
            input_dict['bbox_size'],
            input_dict['valid_mask'],
            foreground_tokens=8,
            context_tokens=4,
            history_timestamps=history_timestamps,
            current_timestamp=current_timestamp,
            current_box=observation_box.detach(),
            return_metadata=True,
            **({
                'v27_enabled': True,
                'history_point_ids': input_dict['b0_point_ids'][:, :-1],
                'history_point_valid_mask': input_dict['b0_valid_mask'][:, :-1],
                'history_unique_mask': input_dict['b0_unique_mask'][:, :-1],
                'return_point_ids': True,
            } if self.ct_enable_v27 else {}),
        )
        memory_tokens, memory_valid, memory_metadata = memory_result[:3]
        if self.ct_enable_v27:
            output_dict['ct_memory_point_ids'] = memory_result[3]
            output_dict['ct_memory_frame_uid'] = input_dict['b0_frame_uid'][:, :-1].repeat_interleave(12, dim=1)
            output_dict['ct_memory_valid_mask'] = memory_valid
        if memory_tokens.shape[1] != 36:
            raise RuntimeError("contract-v3 requires exactly 36 memory slots")
        memory_mode = str(getattr(
            self.config, 'ct_memory_mode', 'real')).strip().lower()
        memory_tokens, memory_valid, memory_metadata = apply_memory_control(
            memory_tokens, memory_valid, memory_metadata, memory_mode)
        base_evidence_mode = str(getattr(
            self.config, 'ct_base_evidence_mode', 'full')).strip().lower()
        base_valid_mask = input_dict[
            'ct_base_evidence_valid_mask'].to(current_base_features.device)
        if base_evidence_mode == 'empty':
            base_valid_mask = torch.zeros_like(base_valid_mask)
        elif base_evidence_mode != 'full':
            raise ValueError("ct_base_evidence_mode must be full or empty")
        extension_points = input_dict['ct_extension_points'].to(
            device=current_base_features.device,
            dtype=current_base_features.dtype)
        extension_points = self.encode_point_time(extension_points)
        expected_extension_count = (
            self.ct_expansion_point_count
            if self.ct_enable_v26_recovery else 256)
        if extension_points.shape[1:] != (expected_extension_count, 5):
            raise ValueError(
                "extension pre-pool points must have shape "
                f"[B,{expected_extension_count},5]")
        recursive_age = input_dict.get('ct_recursive_state_age')
        if recursive_age is not None:
            recursive_age = recursive_age.to(
                device=current_base_features.device,
                dtype=current_base_features.dtype)
        extension_valid_mask = input_dict[
            'ct_extension_valid_mask'].to(current_base_features.device)
        if self.ct_enable_v26_recovery:
            causal_support_valid = input_dict[
                'ct_search_support_valid'].to(
                    device=current_base_features.device,
                    dtype=extension_valid_mask.dtype).reshape(
                        batch_size, 1)
            extension_valid_mask = (
                extension_valid_mask * causal_support_valid)
        canonical_prior_contract = MotionPriorOutput(
            center_xy=main_motion['prior_xy'].detach(),
            direction_xy=main_motion['motion_direction_xy'].detach(),
            log_sigma=main_motion['log_sigma_parallel_perp'].detach(),
            valid=main_motion['valid'].detach(),
            source=input_dict['search_v3_prior_source_id'].to(
                current_base_features.device).detach(),
            acquisition_margin_parallel_perp=main_motion[
                'acquisition_margin_parallel_perp'].detach(),
        )
        if not self.ct_enable_v27:
            prior_contract = reexpress_motion_prior(
                canonical_prior_contract,
                input_dict['motion_source_anchor'],
                input_dict['coordinate_anchor'],
                degrees=bool(getattr(self.config, 'degrees', False)),
            )
        if self.ct_enable_v27:
            # B2 consumes the resolved acquisition, including CV fallback.
            # The differentiable learned prior above remains B1's loss input.
            actual = lambda name: input_dict[name].to(
                device=current_base_features.device,
                dtype=current_base_features.dtype).detach()
            from models.ct_v2.pipeline_contracts import AcquisitionRecord
            acquisition = AcquisitionRecord(
                endpoint_xy=actual('search_v3_support_anchor_xy'),
                direction_xy=actual('ct_acquisition_direction_xy'),
                acquisition_margin_parallel_perp=actual('ct_acquisition_margin'),
                valid=actual('ct_acquisition_resolved_valid'),
                source=actual('search_v3_prior_source_id'),
                source_anchor=actual('motion_source_anchor'),
                coordinate_anchor=actual('coordinate_anchor'),
                query_delta_t=actual('search_v3_query_delta_t'),
                statistical_direction_xy=actual('ct_acquisition_statistical_direction_xy'),
                statistical_log_sigma=actual('ct_acquisition_log_sigma'),
                learned_valid=actual('ct_acquisition_learned_valid'),
                gap_ratio=actual('search_v3_gap_ratio'),
                real_timestamp=input_dict.get('timestamps_real'),
                input_digest=input_dict.get('motion_input_digest', ''),
                parameter_revision=input_dict.get('ct_acquisition_parameter_revision', 0),
                fallback_reason=input_dict.get('ct_acquisition_fallback_reason', ''))
            # Project the statistical covariance onto the acquisition axes.
            cosine = (acquisition.direction_xy * acquisition.statistical_direction_xy).sum(-1).clamp(-1, 1)
            variance = (2 * acquisition.statistical_log_sigma).exp()
            sigma = torch.stack((
                cosine.square() * variance[:, 0] + (1-cosine.square()) * variance[:, 1],
                (1-cosine.square()) * variance[:, 0] + cosine.square() * variance[:, 1]), -1).clamp_min(1e-8)
            prior_contract = MotionPriorOutput(
                center_xy=acquisition.endpoint_xy,
                direction_xy=acquisition.direction_xy,
                log_sigma=.5 * sigma.log(), valid=acquisition.valid,
                source=acquisition.source,
                acquisition_margin_parallel_perp=acquisition.acquisition_margin_parallel_perp)
        support_alignment_error = validate_motion_prior_support_alignment(
            prior_contract,
            input_dict['search_v3_support_anchor_xy'],
            tolerance=1e-3,
        )
        with self.ct_plugin_rng.fork(current_base_features.device):
            joint_output = self.ct_joint_search_refiner(
                extension_points=extension_points,
                extension_valid_mask=extension_valid_mask,
                extension_source=input_dict[
                    'ct_extension_source'].to(current_base_features.device),
                current_base_features=current_base_features,
                current_base_valid_mask=base_valid_mask,
                memory_tokens=memory_tokens,
                memory_valid_mask=memory_valid,
                memory_metadata=memory_metadata,
                observation_box=observation_box,
                observation_stats=observation_stats,
                b1_center_xy=prior_contract.center_xy,
                b1_sigma_parallel_perp=torch.exp(prior_contract.log_sigma),
                b1_direction_xy=prior_contract.direction_xy,
                b1_valid=prior_contract.valid,
                query_delta_t=input_dict['search_v3_query_delta_t'].to(
                    current_base_features.device),
                gap_ratio=input_dict['search_v3_gap_ratio'].to(
                    current_base_features.device),
                recursive_age=recursive_age,
                **({
                    'first_box_size_wlh': input_dict['bbox_size'].detach(),
                    'extension_point_ids': input_dict['ct_extension_point_ids'],
                    'current_base_point_ids': input_dict['ct_base_point_ids'],
                    'current_base_unique_mask': input_dict['ct_base_unique_mask'],
                } if self.ct_enable_v27 else {}),
            )
        evidence_contract = EvidenceOutput(
            raw_box=joint_output['ct_b2_raw_box'],
            structural_available=joint_output['ct_b2_available'],
            presence_logit=joint_output[
                'ct_b2_extension_presence_logit'],
            targetness=joint_output['ct_search_targetness_logits'],
            evidence_summary=torch.cat((
                joint_output['ct_b2_base_evidence'],
                joint_output['ct_b2_extension_evidence']), dim=1),
            point_diagnostics={
                'entropy': joint_output['ct_search_targetness_entropy'],
                'ess': joint_output['ct_search_normalized_ess'],
                'count': joint_output[
                    'ct_search_extension_selected_count'],
                'vote_consistency': joint_output['ct_vote_consistency'],
                'covariance_xx': joint_output['ct_vote_covariance_xx'],
                'covariance_xy': joint_output['ct_vote_covariance_xy'],
                'covariance_yy': joint_output['ct_vote_covariance_yy'],
                'inlier_ratio': joint_output['ct_vote_inlier_ratio'],
                'effective_mass': joint_output['ct_vote_effective_mass'],
                'candidate_margin': joint_output[
                    'ct_vote_candidate_margin'],
                'compatible_hypothesis_count': joint_output[
                    'ct_vote_compatible_hypothesis_count'],
            },
        )
        raw_box = evidence_contract.raw_box
        candidate_available = joint_output[
            'ct_search_candidate_valid'].reshape(batch_size)
        router_availability = evidence_contract.structural_available
        candidate_source = torch.where(
            candidate_available > 0,
            torch.full_like(candidate_available, 2.0),
            torch.zeros_like(candidate_available))
        candidate_policy = str(getattr(
            self.config, 'ct_candidate_policy', 'b2_raw')).strip().lower()
        if candidate_policy == 'prior_backoff':
            dt = input_dict['search_v3_query_delta_t'].to(
                current_base_features.device).reshape(batch_size).clamp(
                    min=0.0)
            radius = torch.clamp(
                float(getattr(self.config, 'ct_router_radius_base', 0.5))
                + float(getattr(
                    self.config, 'ct_router_radius_per_second', 0.5)) * dt,
                max=float(getattr(
                    self.config, 'ct_router_radius_max', 2.0)))
            prior_residual = (
                prior_contract.center_xy - observation_box[:, :2])
            prior_norm = torch.linalg.norm(prior_residual, dim=1)
            prior_scale = torch.clamp(
                radius / prior_norm.clamp_min(1e-6), max=1.0)
            prior_box = observation_box.clone()
            prior_box[:, :2] = (
                observation_box[:, :2]
                + prior_residual * prior_scale.unsqueeze(1))
            use_b2 = candidate_available > 0
            use_b1 = (~use_b2) & (prior_contract.valid.reshape(-1) > 0)
            raw_box = torch.where(
                use_b2.unsqueeze(1), raw_box,
                torch.where(
                    use_b1.unsqueeze(1), prior_box, observation_box))
            candidate_available = (use_b2 | use_b1).to(
                observation_box.dtype)
            router_availability = candidate_available
            candidate_source = torch.where(
                use_b2, torch.full_like(candidate_available, 2.0),
                torch.where(
                    use_b1, torch.ones_like(candidate_available),
                    torch.zeros_like(candidate_available)))
        elif candidate_policy != 'b2_raw':
            raise ValueError(
                "ct_candidate_policy must be b2_raw or prior_backoff")
        joint_output.update({
            'ct_policy_candidate_box': raw_box,
            'ct_policy_candidate_valid': candidate_available,
            # 0=observation fallback, 1=B1 backoff, 2=B2 raw evidence.
            'ct_policy_candidate_source_id': candidate_source,
        })
        if self.ct_enable_b3:
            final_box, router_output = self.ct_joint_router(
                observation_box=observation_box,
                raw_box=raw_box,
                availability=router_availability,
                base_evidence=joint_output['ct_b2_base_evidence'],
                extension_evidence=joint_output['ct_b2_extension_evidence'],
                base_presence_probability=joint_output[
                    'ct_b2_base_presence_probability'],
                extension_presence_probability=joint_output[
                    'ct_b2_extension_presence_probability'],
                observation_stats=observation_stats,
                b1_sigma_parallel_perp=torch.exp(prior_contract.log_sigma),
                query_delta_t=input_dict['search_v3_query_delta_t'].to(
                    current_base_features.device),
                gap_ratio=input_dict['search_v3_gap_ratio'].to(
                    current_base_features.device),
                recursive_age=recursive_age,
                enabled=True,
                coarse_box=coarse_box,
                b1_center_xy=prior_contract.center_xy,
                targetness_entropy=evidence_contract.point_diagnostics[
                    'entropy'],
                normalized_ess=evidence_contract.point_diagnostics['ess'],
                extension_point_count=evidence_contract.point_diagnostics[
                    'count'],
                extension_voxel_count=input_dict.get(
                    'ct_search_extension_voxels'),
                targetness_mean=joint_output['ct_search_targetness_mean'],
                targetness_max=joint_output['ct_search_targetness_max'],
                vote_consistency=evidence_contract.point_diagnostics[
                    'vote_consistency'],
                vote_covariance_xx=evidence_contract.point_diagnostics[
                    'covariance_xx'],
                vote_covariance_xy=evidence_contract.point_diagnostics[
                    'covariance_xy'],
                vote_covariance_yy=evidence_contract.point_diagnostics[
                    'covariance_yy'],
                vote_inlier_ratio=evidence_contract.point_diagnostics[
                    'inlier_ratio'],
                vote_candidate_margin=evidence_contract.point_diagnostics[
                    'candidate_margin'],
                compatible_hypothesis_count=(
                    evidence_contract.point_diagnostics[
                        'compatible_hypothesis_count']),
                **({'mode_summary': torch.stack((
                    torch.log1p(joint_output['ct_vote_effective_mass']),
                    torch.log1p(joint_output['ct_vote_mode_unique_count']),
                    joint_output['ct_vote_mode_mean_targetness'],
                    .5 * joint_output['ct_vote_mode_mean_identity_margin']), dim=-1).detach()
                } if self.ct_enable_v27 else {}),
            )
        else:
            dt = input_dict['search_v3_query_delta_t'].to(
                current_base_features.device).reshape(batch_size).clamp(
                    min=0.0)
            residual = raw_box[:, :2] - observation_box[:, :2]
            residual_norm = torch.linalg.norm(residual, dim=1)
            radius = torch.clamp(
                float(getattr(self.config, 'ct_router_radius_base', 0.5))
                + float(getattr(
                    self.config, 'ct_router_radius_per_second', 0.5)) * dt,
                max=float(getattr(
                    self.config, 'ct_router_radius_max', 2.0)))
            scale = torch.clamp(
                radius / residual_norm.clamp_min(1e-6), max=1.0)
            bounded_residual = residual * scale.unsqueeze(1)
            if self.ct_enable_v27:
                from models.ct_v2.action_v27 import bounded_residual_xy
                bounded_residual, geometry = bounded_residual_xy(
                    observation_box, raw_box, input_dict['search_v3_query_delta_t'],
                    radius_base=float(getattr(self.config, 'ct_router_radius_base', .5)),
                    radius_per_second=float(getattr(self.config, 'ct_router_radius_per_second', .5)),
                    radius_max=float(getattr(self.config, 'ct_router_radius_max', 2.)))
                residual, radius, residual_norm = geometry['residual'], geometry['radius'], geometry['norm']
            zeros = observation_box.new_zeros((batch_size,))
            final_box = observation_box
            router_output = {
                'ct_b3_help_logit': zeros,
                'ct_b3_harm_logit': zeros,
                'ct_b3_help_probability': zeros,
                'ct_b3_harm_probability': zeros,
                'ct_b3_expected_center_gain': zeros,
                'ct_b3_expected_iou_gain': zeros,
                'ct_b3_action_score': zeros,
                'ct_b3_calibrated': zeros,
                'ct_b3_h3_residual': zeros,
                'ct_b3_h3_utility': zeros,
                'ct_b3_final_gate': zeros,
                'ct_router_logit': zeros,
                'ct_router_gate': zeros,
                'ct_router_applied_gate': zeros,
                'ct_router_evidence_valid': candidate_available,
                'ct_router_bounded_residual_xy': bounded_residual,
                'ct_router_residual_xy': residual,
                'ct_router_radius': radius,
                'ct_router_clip_rate': (
                    residual_norm > radius).to(observation_box.dtype),
                'ct_router_soft_box': observation_box,
            }
        if not self.training:
            mode = self.proposal_inference_mode
            if mode in ('obs', 'obs_only', 'observation'):
                final_box = observation_box
                if self.ct_enable_v27:
                    for gate_key in ('ct_b3_final_gate', 'ct_router_gate', 'ct_router_applied_gate'):
                        router_output[gate_key] = torch.zeros_like(candidate_available)
            elif mode == 'bounded_always':
                from models.ct_v2.action_v27 import bounded_residual_xy
                bounded, geometry = bounded_residual_xy(
                    observation_box, raw_box,
                    input_dict['search_v3_query_delta_t'],
                    radius_base=float(getattr(self.config, 'ct_router_radius_base', .5)),
                    radius_per_second=float(getattr(self.config, 'ct_router_radius_per_second', .5)),
                    radius_max=float(getattr(self.config, 'ct_router_radius_max', 2.)))
                applies = (candidate_available.reshape(-1) > 0) & geometry['finite']
                executed_box = torch.cat((observation_box[:, :2] + bounded, observation_box[:, 2:]), dim=1)
                final_box = torch.where(applies[:, None], executed_box, observation_box)
                for gate_key in ('ct_b3_final_gate', 'ct_router_gate', 'ct_router_applied_gate'):
                    router_output[gate_key] = applies.to(observation_box.dtype)
            elif mode == 'raw_search':
                final_box = torch.where(
                    candidate_available.reshape(
                        batch_size, 1).to(torch.bool),
                    raw_box, observation_box)
            elif (self.ct_enable_b3 and mode in (
                    'full', 'full_selective', 'selective')):
                # A missing/expired/mismatched artifact is a deployment
                # no-op, never an uncalibrated action and never a crash after
                # an expensive full-dataset evaluation has started.
                if not bool(self.ct_joint_router.calibrated):
                    final_box = observation_box
        if self.ct_enable_v27 and self.training:
            final_box = observation_box
        decision_contract = DecisionOutput(
            final_box=final_box,
            help_logit=router_output['ct_b3_help_logit'],
            harm_logit=router_output['ct_b3_harm_logit'],
            expected_center_gain=router_output[
                'ct_b3_expected_center_gain'],
            expected_iou_gain=router_output['ct_b3_expected_iou_gain'],
            applied=router_output['ct_b3_final_gate'],
            bounded_residual=router_output[
                'ct_router_bounded_residual_xy'],
        )
        final_box = decision_contract.final_box
        joint_output.update(router_output)
        joint_output.update({
            'ct_final_box': final_box,
            'candidate_valid': candidate_available,
            'ct_b1_candidate_center_xy': prior_contract.center_xy,
            'ct_b1_candidate_direction_xy': prior_contract.direction_xy,
            'ct_b1_support_alignment_error': support_alignment_error,
            'ct_search_geometry_valid': input_dict[
                'ct_search_geometry_valid'].to(
                    device=current_base_features.device,
                    dtype=current_base_features.dtype).reshape(batch_size),
            'ct_b1_geometry_source_id': input_dict[
                'search_v3_prior_source_id'].to(
                    device=current_base_features.device,
                    dtype=current_base_features.dtype).reshape(batch_size),
            'ct_memory_mode_id': current_base_features.new_full(
                (batch_size,), {
                    'real': 0, 'empty': 1, 'time_misaligned': 2, 'none': 3}[
                    memory_mode]),
            'ct_base_evidence_mode_id': current_base_features.new_full(
                (batch_size,), {'full': 0, 'empty': 1}[
                    base_evidence_mode]),
        })
        output_dict.update(joint_output)
        return final_box

    def _ct_plugin_parameter(self, name):
        return (
            (self.ct_enable_b1
             and name.startswith('physical_motion_encoder.'))
            or (self.ct_enable_b2
                and name.startswith('ct_joint_search_refiner.'))
            or (self.ct_enable_b3
                and name.startswith('ct_joint_router.'))
        )

    @staticmethod
    def _ct_any_plugin_parameter(name):
        return name.startswith((
            'physical_motion_encoder.',
            'ct_joint_search_refiner.',
            'ct_joint_router.',
        ))

    @staticmethod
    def _ct_plugin_group(name):
        if name.startswith('physical_motion_encoder.'):
            return 'b1'
        if name.startswith('ct_joint_search_refiner.'):
            return 'b2'
        if name.startswith('ct_joint_router.'):
            return 'b3'
        raise ValueError(f"not a CT plugin parameter: {name}")

    def _build_isolated_optimizer(self, parameters, learning_rate):
        optimizer_name = self.config.optimizer.lower()
        if optimizer_name == 'sgd':
            return torch.optim.SGD(
                parameters, lr=learning_rate, momentum=0.9,
                weight_decay=self.config.wd)
        if optimizer_name in ('adam', 'adamonecycle'):
            return torch.optim.Adam(
                parameters, lr=learning_rate,
                weight_decay=self.config.wd,
                betas=(0.5, 0.999), eps=1e-6)
        raise ValueError(
            "Invalid optimizer. Please choose from 'sgd', 'adam', or "
            "'adamonecycle'.")

    def configure_optimizers(self):
        if not (self.ct_separate_optimizers or self.ct_unified_auto):
            return super().configure_optimizers()
        if self.ct_initialization_policy == 'scratch_only':
            enabled_plugins = {
                'b1': self.ct_enable_b1,
                'b2': self.ct_enable_b2,
                'b3': self.ct_enable_b3,
            }
            frozen_active = []
            for name, parameter in self.named_parameters():
                if self._ct_any_plugin_parameter(name):
                    active = enabled_plugins[self._ct_plugin_group(name)]
                else:
                    active = True
                if active and not parameter.requires_grad:
                    frozen_active.append(name)
            if frozen_active:
                raise RuntimeError(
                    "scratch-only CT forbids frozen active parameters: "
                    + ", ".join(sorted(frozen_active)))
        named = [
            (name, parameter) for name, parameter in self.named_parameters()
            if parameter.requires_grad]
        enabled = {
            'b1': self.ct_enable_b1,
            'b2': self.ct_enable_b2,
            'b3': self.ct_enable_b3,
        }
        module_named = partition_named_parameter_groups(
            named, self._ct_any_plugin_parameter,
            self._ct_plugin_group, enabled)
        self._ct_named_parameters_by_module = module_named
        self._ct_optimizer_names = list(module_named)
        self._ct_b0_named_parameters = module_named['b0']
        self._ct_plugin_named_parameters = [
            item for name in ('b1', 'b2', 'b3')
            for item in module_named.get(name, [])]
        plugin_lr = float(getattr(
            self.config, 'ct_plugin_lr', self.config.lr))
        learning_rates = {
            name: float(getattr(
                self.config, f'ct_{name}_lr',
                self.config.lr if name == 'b0' else plugin_lr))
            for name in module_named}
        if self.ct_unified_auto:
            if self.config.optimizer.lower() != 'adam':
                raise ValueError(
                    "safe_seqtrack_auto_v1 requires Adam with StepLR")
            parameter_groups = [
                {
                    'params': [
                        parameter for _, parameter in module_named[name]],
                    'lr': learning_rates[name],
                    'name': name,
                }
                for name in self._ct_optimizer_names
            ]
            optimizer = torch.optim.Adam(
                parameter_groups,
                lr=float(self.config.lr),
                weight_decay=float(self.config.wd),
                betas=(0.5, 0.999), eps=1e-6)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=self.config.lr_decay_step,
                gamma=self.config.lr_decay_rate)
            self._ct_b0_prefix_hashes = {
                'initial': self._ct_module_parameter_hash('b0')}
            self._ct_b0_optimizer_state_hashes = {
                'initial': self._ct_optimizer_state_hash('b0', optimizer)}
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'name': 'scheduler_unified_auto',
                },
            }
        optimizers = [
            self._build_isolated_optimizer(
                [parameter for _, parameter in module_named[name]],
                learning_rates[name])
            for name in self._ct_optimizer_names]
        if self.config.optimizer.lower() == 'adamonecycle':
            if self.train_dataloader_length is None:
                raise ValueError(
                    "OneCycle isolated training needs train_dataloader_length")
            schedulers = [
                torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=float(getattr(
                        self.config, f'ct_{name}_max_lr',
                        learning_rates[name])),
                    epochs=self.config.epoch,
                    steps_per_epoch=self.train_dataloader_length)
                for name, optimizer in zip(
                    self._ct_optimizer_names, optimizers)]
            interval = 'step'
        else:
            schedulers = [
                torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=self.config.lr_decay_step,
                    gamma=self.config.lr_decay_rate)
                for optimizer in optimizers]
            interval = 'epoch'
        return optimizers, [
            {'scheduler': scheduler, 'interval': interval,
             'name': f'scheduler_{name}'}
            for name, scheduler in zip(self._ct_optimizer_names, schedulers)]

    @staticmethod
    def _b0_auxiliary_batch(input_dict):
        """Return whether a homogeneous microbatch is B0 auxiliary-only."""
        value = input_dict.get('ct_b0_auxiliary_only')
        if value is None:
            return False
        tensor = torch.as_tensor(value).reshape(-1).to(torch.bool)
        if not bool(torch.all(tensor == tensor[0])):
            raise RuntimeError(
                "canonical and auxiliary B0 rows must use separate forwards")
        return bool(tensor[0].item())

    @staticmethod
    def _finalize_observation_output(
            input_dict, output_dict, aux_box, observation_aux_box,
            updated_aux_box, seg_logits, motion_pred, updated_ref_boxs,
            obs_aux):
        """Populate the stable B0 output interface for full and aux paths."""
        output_dict["estimation_boxes"] = aux_box
        output_dict.update({
            "seg_logits": seg_logits,
            "motion_pred": motion_pred,
            'observation_aux_estimation_boxes': observation_aux_box,
            'aux_estimation_boxes': updated_aux_box,
            'ref_boxs': input_dict['ref_boxs'],
            'valid_mask': input_dict["valid_mask"],
            'updated_ref_boxs': updated_ref_boxs,
        })
        output_dict.update(obs_aux)
        if 'b0_unique_mask' in input_dict:
            from utils.v27_quality import observation_quality
            output_dict['ct_observation_quality'] = observation_quality(
                seg_logits, input_dict['b0_unique_mask'], input_dict['b0_raw_point_count'])
        for key in (
                "ct_search_used",
                "ct_search_expansion_ratio",
                "ct_search_baseline_points",
                "ct_search_expansion_points",
                "ct_search_query_delta_t",
                "ct_search_predicted_displacement",
                "trajectory_search_valid",
                "trajectory_search_gap_ratio",
                "trajectory_search_sigma_parallel",
                "trajectory_search_sigma_perpendicular",
                "search_has_usable_points"):
            if key in input_dict:
                output_dict[key] = input_dict[key]
        return output_dict

    def forward(self, input_dict):
        """
        Args:
            input_dict: {
            "points": (B,N,3+1+1)
            "candidate_bc": (B,N,9)
            ['points', #[2, 4096, 5] B*(num_hist*sample)*5
            'box_label', #B*4
            'ref_boxs', #B*(num_hist)*4
            'box_label_prev', #B*(num_hist)*4
            'motion_label', #B*(num_hist)*4
            'motion_state_label', #B*(num_hist), Subtract all previous histboxes from the current box
            'bbox_size', #B*3
            'seg_label', #B*(num_hist+1)*sample
            'valid_mask', #B*(num_hist)
            'prev_bc', #B*(num_hist)*sample*9
            'this_bc', #B*sample*9
            'candidate_bc'] #B*(num_hist*sample)*9

        }

        Returns: B,4

        """
        if self.is_paired_batch(input_dict):
            output_a = self.forward(input_dict["view_a"])
            output_b = self.forward(input_dict["view_b"])
            paired_output = {
                "view_a": output_a,
                "view_b": output_b,
            }
            return paired_output

        b0_auxiliary_only = self._b0_auxiliary_batch(input_dict)
        observation_only_forward = bool(getattr(
            self, '_ct_observation_only_forward', False))
        output_dict = {}
        points = self.encode_point_time(input_dict["points"])
        x = points.transpose(1, 2)

        if self.box_aware:
            candidate_bc = input_dict["candidate_bc"].transpose(1, 2) 
            x = torch.cat([x, candidate_bc], dim=1) 

        B, _, N = x.shape
        HL =  input_dict["valid_mask"].shape[1] # Number of historical frames, default 3
        L = HL + 1 # Total length of the point cloud sequence, 1 represents the current frame
        chunk_size = N // L

        seg_out = self.seg_pointnet(x) 
        seg_logits = seg_out[:, :2, :]  # B,2,N
        obs_stats, obs_aux = self.build_observability_stats(input_dict, seg_logits, chunk_size)
        pred_cls = torch.argmax(seg_logits, dim=1, keepdim=True)  # B,1,N
        mask_points = x[:, :4, :] * pred_cls 

        if self.box_aware:
            pred_bc = seg_out[:, 2:, :]
            mask_pred_bc = pred_bc * pred_cls
            mask_points = torch.cat([mask_points, mask_pred_bc], dim=1)
            output_dict['pred_bc'] = pred_bc.transpose(1, 2)

        # Coarse initial motion prediction
        point_feature = self.mini_pointnet(mask_points) #N*256
        motion_feature = point_feature
        if self.use_dynamics_encoder:
            dynamics_ref_boxs = input_dict["ref_boxs"]
            if (
                    self.use_ct_v2
                    and self.training
                    and (
                        "ct_motion_ref_boxs" in input_dict
                        or "canonical_ref_boxs" in input_dict)):
                # New CT-v2 batches provide a canonical-anchored history with
                # smooth relative trajectory errors. Old batches fall back to
                # the exact canonical history for checkpoint/YAML compatibility.
                dynamics_ref_boxs = input_dict.get(
                    "ct_motion_ref_boxs",
                    input_dict["canonical_ref_boxs"],
                )
            dynamics_delta_t = input_dict.get(
                "delta_t_effective", input_dict["delta_t"])
            dynamics_current_delta_t = input_dict.get(
                "current_delta_t_effective",
                input_dict.get("current_delta_t"))
            if self.use_ordered_trajectory_encoder:
                trajectory_output = self.dynamics_encoder.forward_trajectory(
                    dynamics_ref_boxs,
                    dynamics_delta_t,
                    input_dict["valid_mask"],
                    dynamics_current_delta_t,
                )
                z_dyn = trajectory_output["feature"]
                velocity_pred = trajectory_output["velocity"]
                dynamics_displacement_pred = trajectory_output["displacement"]
                dynamics_valid = trajectory_output["valid"]
                output_dict.update({
                    "trajectory_displacement_pred": trajectory_output[
                        "trajectory_displacement"],
                    "trajectory_yaw_displacement_pred": trajectory_output[
                        "yaw_displacement"],
                    "trajectory_yaw_rate_pred": trajectory_output["yaw_rate"],
                    "trajectory_log_sigma": trajectory_output["log_sigma"],
                    "trajectory_kinematic_displacement": trajectory_output[
                        "kinematic_displacement"],
                    "trajectory_gap_ratio": trajectory_output["gap_ratio"],
                })
            else:
                (z_dyn, velocity_pred, dynamics_displacement_pred,
                 dynamics_valid) = self.dynamics_encoder(
                    dynamics_ref_boxs,
                    dynamics_delta_t,
                    input_dict["valid_mask"],
                    dynamics_current_delta_t,
                )
            output_dict["velocity_pred"] = velocity_pred
            output_dict["dynamics_displacement_pred"] = dynamics_displacement_pred
            output_dict["dynamics_valid"] = dynamics_valid
            if self.dynamics_motion_mode == 'trajectory_adapter':
                motion_feature = point_feature
                trajectory_search_points = input_dict.get(
                    "trajectory_search_points")
                if trajectory_search_points is None:
                    point_count = int(getattr(
                        self.config, "trajectory_search_point_count", 128))
                    trajectory_search_points = point_feature.new_zeros(
                        (B, point_count, 5))
                trajectory_search_points = self.encode_point_time(
                    trajectory_search_points)
                trajectory_search_feature = self.trajectory_search_encoder(
                    trajectory_search_points.transpose(1, 2))
                trajectory_search_valid = input_dict.get(
                    "trajectory_search_valid")
                if trajectory_search_valid is None:
                    trajectory_search_valid = point_feature.new_zeros((B,))
                trajectory_search_feature = trajectory_search_feature * (
                    trajectory_search_valid.to(
                        device=point_feature.device,
                        dtype=point_feature.dtype,
                    ).reshape(B, 1)
                )
                output_dict["trajectory_search_feature"] = (
                    trajectory_search_feature)
                if self.use_trajectory_adapter:
                    adapter_scale = self.trajectory_adapter_scale
                    if (self.training
                            and getattr(self, 'current_epoch', 0)
                            < self.trajectory_adapter_warmup_epoch):
                        adapter_scale = 0.0
                    motion_feature, trajectory_adapter_aux = (
                        self.trajectory_adapter(
                            point_feature,
                            z_dyn,
                            trajectory_search_feature,
                            trajectory_output["log_sigma"],
                            trajectory_output["gap_ratio"],
                            dynamics_valid,
                            trajectory_search_valid,
                            enabled_scale=adapter_scale,
                        ))
                    output_dict.update(trajectory_adapter_aux)
            elif self.dynamics_motion_mode == 'residual':
                motion_feature = point_feature
            elif self.dynamics_motion_mode == 'proposal_innovation':
                motion_feature = point_feature
                if self.use_physical_time_adapter:
                    adapter_gap = input_dict.get(
                        "current_delta_t_effective",
                        input_dict.get("current_delta_t"),
                    )
                    adapter_scale = self.physical_time_adapter_scale
                    if (self.training
                            and getattr(self, 'current_epoch', 0)
                            < self.physical_time_adapter_warmup_epoch):
                        adapter_scale = 0.0
                    motion_feature, adapter_aux = self.physical_time_adapter(
                        point_feature,
                        z_dyn,
                        adapter_gap,
                        dynamics_valid,
                        enabled_scale=adapter_scale,
                    )
                    output_dict.update(adapter_aux)
            elif self.use_observability_gate:
                if self.obs_gate_fusion_mode == 'feature':
                    motion_feature, gate_aux = self.observability_gate(
                        point_feature, z_dyn, obs_stats, dynamics_valid)
                    output_dict.update(gate_aux)
                elif self.obs_gate_fusion_mode == 'confidence_residual':
                    obs_alpha, gate_aux = self.observability_gate.compute_alpha(
                        obs_stats,
                        dynamics_valid,
                        dtype=point_feature.dtype,
                        device=point_feature.device,
                    )
                    alpha_dyn_raw = obs_alpha[:, 1:2]
                    alpha_dyn_clamped = torch.clamp(
                        alpha_dyn_raw, max=self.obs_gate_max_dyn_alpha)
                    output_dict.update(gate_aux)
                    output_dict["obs_alpha_dyn_raw"] = alpha_dyn_raw.squeeze(1)
                    output_dict["obs_alpha_dyn_clamped"] = alpha_dyn_clamped.squeeze(1)
                    output_dict["obs_gate_residual_scale"] = point_feature.new_tensor(
                        self.obs_gate_residual_scale)
                    motion_feature = point_feature
                else:
                    raise ValueError(f"Unsupported obs_gate_fusion_mode: {self.obs_gate_fusion_mode}")
            else:
                motion_feature = torch.cat((point_feature, z_dyn), dim=1)
        motion_pred = self.motion_mlp(motion_feature)  # B,4
        if self.use_observability_gate and self.obs_gate_fusion_mode == 'confidence_residual':
            dyn_residual = (
                self.obs_gate_residual_scale
                * alpha_dyn_clamped
                * output_dict["dynamics_displacement_pred"]
            )
            output_dict["motion_obs_pred"] = motion_pred
            output_dict["motion_dyn_residual"] = dyn_residual
            motion_pred = torch.cat((
                motion_pred[:, :3] + dyn_residual,
                motion_pred[:, 3:4],
            ), dim=1)
        if self.use_dynamics_encoder and self.dynamics_motion_mode == 'residual':
            motion_pred, residual_aux = self.apply_bounded_dynamics_residual(
                motion_pred,
                output_dict['dynamics_displacement_pred'],
                output_dict['dynamics_valid'],
                obs_stats,
                obs_aux,
                input_dict,
            )
            output_dict.update(residual_aux)
        if (self.use_dynamics_encoder
                and self.dynamics_motion_mode == 'proposal_innovation'):
            innovation_scale = self.dynamics_innovation_scale
            if (self.training
                    and getattr(self, 'current_epoch', 0)
                    < self.dynamics_innovation_warmup_epoch):
                innovation_scale = 0.0
            innovation_valid = output_dict['dynamics_valid']
            if self.dynamics_innovation_disable_on_empty_search:
                innovation_valid = innovation_valid * build_search_usable_mask(
                    input_dict, obs_aux, innovation_valid)
            output_dict['dynamics_innovation_valid'] = innovation_valid
            innovation_alpha = self.dynamics_innovation_alpha
            if self.use_ct_v2 and self.ct_fusion_mode == 'adaptive':
                innovation_alpha, fusion_aux = self.ct_proposal_fusion(
                    point_feature,
                    z_dyn,
                    motion_pred[:, :3],
                    output_dict['dynamics_displacement_pred'],
                    obs_stats,
                    innovation_valid,
                    input_dict.get(
                        'current_delta_t_effective',
                        input_dict.get('current_delta_t')),
                    input_dict.get('ct_search_expansion_ratio'),
                )
                output_dict.update(fusion_aux)
            elif self.use_ct_v2:
                output_dict['ct_fusion_alpha'] = motion_pred.new_full(
                    (motion_pred.shape[0],),
                    float(self.dynamics_innovation_alpha),
                )
            final_center, innovation_aux = apply_proposal_innovation(
                motion_pred[:, :3],
                output_dict['dynamics_displacement_pred'],
                innovation_valid,
                input_dict.get(
                    'current_delta_t_effective',
                    input_dict.get('current_delta_t')),
                alpha=innovation_alpha,
                enabled_scale=innovation_scale,
                base_radius=self.dynamics_innovation_radius_base,
                radius_per_second=self.dynamics_innovation_radius_per_second,
                max_radius=self.dynamics_innovation_radius_max,
                eps=getattr(self.config, 'dynamics_eps', 1e-3),
            )
            output_dict['motion_obs_pred'] = motion_pred
            output_dict['dynamics_innovation_scale_effective'] = motion_pred.new_tensor(
                innovation_scale)
            if 'ct_fusion_alpha' in output_dict:
                # ``dynamics_innovation_alpha`` is the authoritative effective
                # coefficient after both the valid mask and warmup scale.
                output_dict['ct_fusion_alpha_applied'] = innovation_aux[
                    'dynamics_innovation_alpha']
            output_dict.update(innovation_aux)
            motion_pred = torch.cat((final_center, motion_pred[:, 3:4]), dim=1)

        if self.use_motion_cls:
            motion_state_feature = (
                motion_feature
                if self.dynamics_motion_mode == 'trajectory_adapter'
                else point_feature
            )
            motion_state_logits = self.motion_state_mlp(
                motion_state_feature)  # B,2
            motion_mask = torch.argmax(motion_state_logits, dim=1, keepdim=True)  # B,1
            motion_pred_masked = motion_pred * motion_mask
            output_dict['motion_cls'] = motion_state_logits # B*2
        else:
            motion_pred_masked = motion_pred


        prev_boxes = torch.zeros_like(motion_pred)

        # 1st stage prediction
        aux_box = points_utils.get_offset_box_tensor(prev_boxes, motion_pred_masked)

        # Get corners of the current and historical boxes
        bbox_size = input_dict["bbox_size"] 
        bbox_size_repeated = bbox_size.repeat_interleave(L, dim=0)

        ref_boxs = input_dict["ref_boxs"]
        box_seq = torch.cat((ref_boxs, aux_box.unsqueeze(1)), dim=1) 
        box_seq = box_seq.reshape(B*L,4) 
        box_seq_corner = get_tensor_corners_batch(box_seq[:,:3],bbox_size_repeated,box_seq[:,-1])
        box_seq_corners = box_seq_corner.reshape(B,L*8,-1) # B*(L*8)*3 represents a total of L*8 points, each with 3 features
        
        # Appending timestamp features to the box corners
        delta_T = input_dict["delta_T"]
        corner_stamps = create_corner_timestamps_from_deltas(
            delta_T, 8, current_time=getattr(self.config, 'main_time_current', 0.0)).to(
            device=box_seq_corners.device, dtype=box_seq_corners.dtype)
        corner_stamps = self.time_encoder(corner_stamps)
        box_seq_corners = torch.cat((box_seq_corners,corner_stamps),dim=-1) # B*(L*8)*4 where 4 represents features for x, y, z, and timestamp

        solo_x = x.reshape(B*L,-1,chunk_size) # Reshape into separate point clouds
        collect_pftc_features = self.use_point_feature_tc and self.training
        collect_b2_point_features = bool(
            self.use_ct_joint_full
            and self.ct_joint_contract_version >= 3
            and self.ct_enable_b2
            and not b0_auxiliary_only)
        collect_point_aligned_features = bool(
            collect_pftc_features or collect_b2_point_features)
        feature_result = self.feature_pointnet(
            solo_x, return_point_features=collect_point_aligned_features)
        if collect_point_aligned_features:
            feature, point_aligned_feature = feature_result
            point_aligned_feature = (
                point_aligned_feature.transpose(1, 2).reshape(
                    B, L, chunk_size, -1))
            if point_aligned_feature.shape[-1] != 64:
                raise RuntimeError(
                    "FeaturePointNet second-layer features must be 64d")
            if collect_pftc_features:
                output_dict["pftc_point_features"] = point_aligned_feature
            if collect_b2_point_features:
                output_dict["b0_point_aligned_features"] = (
                    point_aligned_feature)
                raw_frame_points = input_dict['points'].reshape(
                    B, L, chunk_size, -1)
                explicit_base = input_dict.get('ct_base_evidence_points')
                if explicit_base is None:
                    raise KeyError(
                        "contract-v3 B2 requires ct_base_evidence_points")
                explicit_base = explicit_base.to(
                    device=raw_frame_points.device,
                    dtype=raw_frame_points.dtype)
                if explicit_base.shape != raw_frame_points[:, -1].shape:
                    raise ValueError(
                        "base evidence must have shape [B,1024,5]")
                if not torch.equal(explicit_base, raw_frame_points[:, -1]):
                    raise RuntimeError(
                        "base evidence diverged from B0 current 1024 points")
                output_dict['ct_base_evidence_points'] = explicit_base
        else:
            feature = feature_result

        # (B*num) * C * N; N is the fixed Transformer token count per frame.
        feature = feature.transpose(1,2)
        NEW_N = feature.shape[1]
        points_feature = feature.reshape(B,L*NEW_N,-1)

        if (self.use_asymmetric_dual_query
                or self.use_ct_joint_full
                or self.use_decoder_token_consistency):
            delta_motion, decoder_state = self.Transformer(
                box_seq_corners,
                points_feature,
                input_dict["valid_mask"],
                return_decoder_state=True,
            )
            observation_query = (
                decoder_state[:, -1]
                if (self.use_asymmetric_dual_query
                    or self.use_ct_joint_full) else None)
            output_dict['decoder_state'] = decoder_state
            if observation_query is not None:
                output_dict['observation_query'] = observation_query
        else:
            delta_motion = self.Transformer(
                box_seq_corners, points_feature,
                input_dict["valid_mask"])  #B*4*4
            observation_query = None

        if self.training and self.ct_b0_rng_shift_control:
            # All arms consume the same post-observation B0 stream shift.
            # Plugin and auxiliary work use independent checkpointable RNGs.
            torch.rand((B,), device=feature.device)

        updated_ref_boxs = delta_motion[:,:HL,:]
        updated_aux_box =  delta_motion[:,-1,:]

        observation_aux_box = updated_aux_box
        if self.ct_enable_v27 and not self.training:
            current_valid = input_dict['ct_current_observation_valid'].reshape(-1, 1) > 0
            observation_aux_box = torch.where(
                current_valid, observation_aux_box, torch.zeros_like(observation_aux_box))
            updated_aux_box = observation_aux_box
        if b0_auxiliary_only or observation_only_forward:
            return self._finalize_observation_output(
                input_dict, output_dict, aux_box, observation_aux_box,
                observation_aux_box, seg_logits, motion_pred,
                updated_ref_boxs, obs_aux)
        if self.use_b1motion_v3:
            required_motion_keys = (
                'motion_main_ref_boxs',
                'motion_main_delta_t',
                'motion_main_current_delta_t',
                'motion_main_valid_mask',
            )
            missing = [
                key for key in required_motion_keys if key not in input_dict]
            if missing:
                raise KeyError(
                    "B1motion-v3 input is missing: " + ", ".join(missing))
            motion_main_ref_boxs = input_dict['motion_main_ref_boxs']
            if bool(getattr(self.config, 'shuffle_b1_signal', False)):
                motion_main_ref_boxs = torch.flip(
                    motion_main_ref_boxs, dims=(1,))
            if self.use_ct_joint_full and not self.ct_enable_b1:
                main_motion = self.physical_motion_encoder.kinematic_fallback(
                    motion_main_ref_boxs,
                    input_dict['motion_main_delta_t'],
                    input_dict['motion_main_valid_mask'],
                    input_dict['motion_main_current_delta_t'],
                )
            else:
                main_motion = self.physical_motion_encoder(
                    motion_main_ref_boxs,
                    input_dict['motion_main_delta_t'],
                    input_dict['motion_main_valid_mask'],
                    input_dict['motion_main_current_delta_t'],
                    **({'acquisition_features': input_dict['motion_acquisition_features']}
                       if self.ct_enable_v27 else {}),
                )
            if 'replay_b1_contract_present' in input_dict:
                present = input_dict['replay_b1_contract_present'].to(
                    device=main_motion['mu_xy'].device).reshape(-1) > 0.5
                comparisons = (
                    ('mu_xy', 'replay_b1_mu_xy'),
                    ('direction_xy', 'replay_b1_direction_xy'),
                    ('log_sigma_parallel_perp',
                     'replay_b1_log_sigma_parallel_perp'),
                    ('gap_ratio', 'replay_b1_gap_ratio'),
                    ('valid', 'replay_b1_valid'),
                )
                for motion_key, replay_key in comparisons:
                    if replay_key not in input_dict:
                        raise RuntimeError(
                            f"recursive replay is missing {replay_key}")
                    actual = main_motion[motion_key][present]
                    expected = input_dict[replay_key].to(
                        device=actual.device, dtype=actual.dtype)[present]
                    expected = expected.reshape_as(actual)
                    if not torch.allclose(
                            actual, expected, atol=1e-5, rtol=1e-5):
                        max_error = torch.max(torch.abs(
                            actual - expected)).detach().cpu().item()
                        raise RuntimeError(
                            "recursive replay B1 mismatch for "
                            f"{motion_key}: max_error={max_error:.3e}")
            if bool(getattr(self.config, 'force_b1_invalid', False)):
                main_motion = dict(main_motion)
                main_motion['valid'] = torch.zeros_like(main_motion['valid'])
                main_motion['source_id'] = torch.zeros_like(
                    main_motion['source_id'])
            # Both training and recursive inference define the newest history
            # box as the crop anchor.  Keep its (normally exact-zero) origin as
            # a diagnostic, but never add a GT-derived anchor correction to
            # the proposal: the prior is physical motion only.
            motion_prior_origin_xy = input_dict[
                'motion_main_ref_boxs'][:, 0, :2].to(
                    device=main_motion['prior_xy'].device,
                    dtype=main_motion['prior_xy'].dtype,
                )
            motion_prior_proposal_xy = main_motion['prior_xy']
            output_dict.update({
                'motion_prior_xy': main_motion['prior_xy'],
                'motion_prior_origin_xy': motion_prior_origin_xy,
                'motion_prior_proposal_xy': motion_prior_proposal_xy,
                'motion_prior_basis_velocity_xy': main_motion[
                    'basis_velocity_xy'],
                'motion_prior_velocity_xy': main_motion['velocity_xy'],
                'motion_prior_kinematic_xy':
                    main_motion['kinematic_prior_xy'],
                'motion_prior_residual_xy': main_motion[
                    'residual_xy'],
                'motion_prior_residual_unit_parallel_perp': main_motion[
                    'residual_unit_parallel_perp'],
                'motion_prior_envelope_parallel_perp': main_motion[
                    'envelope_parallel_perp'],
                'motion_prior_acquisition_margin_parallel_perp': main_motion[
                    'acquisition_margin_parallel_perp'],
                'motion_prior_valid': main_motion['valid'],
                'motion_prior_log_sigma_xy': main_motion['log_sigma_xy'],
                'motion_prior_log_sigma_parallel_perp': main_motion[
                    'log_sigma_parallel_perp'],
                'motion_prior_covariance_xy': main_motion['covariance_xy'],
                'motion_prior_direction_xy': main_motion[
                    'motion_direction_xy'],
                'motion_prior_source_id': main_motion['source_id'],
                'motion_prior_gap_ratio': main_motion['gap_ratio'],
            })
            history_valid_ratio = input_dict[
                'motion_main_valid_mask'].to(
                    device=point_feature.device,
                    dtype=point_feature.dtype,
                ).mean(dim=1)
            if self.use_motion_v3_legacy_fusion:
                trainer = getattr(self, '_trainer', None)
                trainer_state = str(getattr(
                    getattr(trainer, 'state', None), 'fn', '')).lower()
                in_fit_loop = self.training or 'fit' in trainer_state
                fusion_scale = self.motion_v3_fusion_scale
                if (in_fit_loop
                        and int(getattr(self, 'current_epoch', 0))
                        < self.motion_v3_warmup_epoch):
                    fusion_scale = 0.0
                updated_aux_box, fusion_diagnostics = self.motion_v3_fusion(
                    observation_aux_box,
                    point_feature,
                    obs_stats,
                    main_motion['feature'],
                    motion_prior_proposal_xy,
                    main_motion['velocity_xy'],
                    main_motion['valid'],
                    main_motion['gap_ratio'],
                    history_valid_ratio,
                    input_dict['motion_main_current_delta_t'],
                    enabled_scale=fusion_scale,
                )
                output_dict.update(fusion_diagnostics)
            else:
                updated_aux_box = observation_aux_box

            if self.use_search_evidence_v2:
                required_search_keys = (
                    'search_v2_points',
                    'search_v2_point_valid_mask',
                    'search_v2_geometry_valid',
                    'search_v2_endpoint_xy',
                    'search_v2_query_delta_t',
                    'search_v2_gap_ratio',
                    'search_v2_sigma_parallel',
                    'search_v2_sigma_perpendicular',
                    'search_v2_available_count',
                )
                missing_search = [
                    key for key in required_search_keys
                    if key not in input_dict]
                if missing_search:
                    raise KeyError(
                        "B2-v2 input is missing: "
                        + ", ".join(missing_search))
                raw_search_points = input_dict['search_v2_points'].to(
                    device=point_feature.device,
                    dtype=point_feature.dtype)
                encoded_search_points = self.encode_point_time(
                    raw_search_points)
                if encoded_search_points.shape[-1] != 5:
                    raise ValueError(
                        "B2-v2 requires a scalar point-time encoding")
                endpoint_xy = input_dict['search_v2_endpoint_xy'].to(
                    device=point_feature.device,
                    dtype=point_feature.dtype)
                point_xy = raw_search_points[..., :2]
                delta_to_endpoint = endpoint_xy.unsqueeze(1) - point_xy
                direction_norm = torch.linalg.norm(
                    endpoint_xy, dim=1, keepdim=True)
                default_direction = torch.zeros_like(endpoint_xy)
                default_direction[:, 0] = 1.0
                direction = torch.where(
                    (direction_norm > 1e-6).expand_as(endpoint_xy),
                    endpoint_xy / torch.clamp(direction_norm, min=1e-6),
                    default_direction)
                perpendicular = torch.stack((
                    -direction[:, 1], direction[:, 0]), dim=1)
                sigma_parallel = input_dict[
                    'search_v2_sigma_parallel'].to(
                        device=point_feature.device,
                        dtype=point_feature.dtype).reshape(B, 1)
                sigma_perpendicular = input_dict[
                    'search_v2_sigma_perpendicular'].to(
                        device=point_feature.device,
                        dtype=point_feature.dtype).reshape(B, 1)
                longitudinal = (
                    delta_to_endpoint
                    * direction.unsqueeze(1)).sum(dim=2) / torch.clamp(
                        sigma_parallel, min=0.25)
                lateral = (
                    delta_to_endpoint
                    * perpendicular.unsqueeze(1)).sum(dim=2) / torch.clamp(
                        sigma_perpendicular, min=0.20)
                search_point_inputs = torch.cat((
                    encoded_search_points,
                    delta_to_endpoint,
                    longitudinal.unsqueeze(2),
                    lateral.unsqueeze(2),
                ), dim=2)
                search_output = self.search_evidence_v2(
                    search_point_inputs,
                    point_xy,
                    input_dict['search_v2_point_valid_mask'],
                    input_dict['search_v2_geometry_valid'],
                    point_feature.detach(),
                    main_motion['feature'].detach(),
                    obs_stats.detach(),
                    input_dict['search_v2_query_delta_t'],
                    input_dict['search_v2_gap_ratio'],
                    sigma_parallel,
                    sigma_perpendicular,
                    input_dict['search_v2_available_count'],
                )
                output_dict.update(search_output)

                if self.use_joint_proposal_fusion:
                    trainer = getattr(self, '_trainer', None)
                    trainer_state = str(getattr(
                        getattr(trainer, 'state', None), 'fn', '')).lower()
                    in_fit_loop = self.training or 'fit' in trainer_state
                    joint_scale = 1.0
                    if in_fit_loop:
                        epoch = int(getattr(self, 'current_epoch', 0))
                        if epoch < self.joint_fusion_warmup_epochs:
                            joint_scale = 0.0
                        elif self.joint_fusion_ramp_epochs > 0:
                            joint_scale = min(
                                1.0,
                                (epoch - self.joint_fusion_warmup_epochs + 1)
                                / float(self.joint_fusion_ramp_epochs),
                            )
                    updated_aux_box, joint_diagnostics = (
                        self.joint_proposal_fusion(
                            observation_aux_box,
                            point_feature,
                            obs_stats,
                            main_motion['feature'],
                            motion_prior_proposal_xy,
                            main_motion['log_sigma_xy'],
                            main_motion['valid'],
                            history_valid_ratio,
                            search_output['search_evidence_token'],
                            search_output['search_proposal_xy'],
                            search_output['search_confidence'],
                            search_output['search_targetness_mass'],
                            search_output['search_targetness_entropy'],
                            search_output['search_candidate_valid'],
                            input_dict['search_v2_query_delta_t'],
                            input_dict['search_v2_gap_ratio'],
                            enabled_scale=joint_scale,
                        ))
                    output_dict.update(joint_diagnostics)
            if (self.training and 'motion_aux_ref_boxs' in input_dict
                    and not (self.use_ct_joint_full
                             and not self.ct_enable_b1)):
                aux_motion = self.physical_motion_encoder(
                    input_dict['motion_aux_ref_boxs'],
                    input_dict['motion_aux_delta_t'],
                    input_dict['motion_aux_valid_mask'],
                    input_dict['motion_aux_current_delta_t'],
                    **({'acquisition_features': input_dict['motion_aux_acquisition_features']}
                       if self.ct_enable_v27 else {}),
                )
                output_dict.update({
                    'motion_aux_prior_xy': aux_motion['prior_xy'],
                    'motion_aux_prior_velocity_xy':
                        aux_motion['velocity_xy'],
                    'motion_aux_prior_kinematic_xy':
                        aux_motion['kinematic_prior_xy'],
                    'motion_aux_prior_residual_unit_parallel_perp':
                        aux_motion['residual_unit_parallel_perp'],
                    'motion_aux_prior_envelope_parallel_perp':
                        aux_motion['envelope_parallel_perp'],
                    'motion_aux_prior_acquisition_margin_parallel_perp':
                        aux_motion['acquisition_margin_parallel_perp'],
                    'motion_aux_prior_valid': aux_motion['valid'],
                    'motion_aux_prior_gap_ratio': aux_motion['gap_ratio'],
                    'motion_aux_prior_log_sigma_xy':
                        aux_motion['log_sigma_xy'],
                    'motion_aux_prior_log_sigma_parallel_perp':
                        aux_motion['log_sigma_parallel_perp'],
                    'motion_aux_prior_direction_xy':
                        aux_motion['motion_direction_xy'],
                })

        if self.use_ct_joint_full:
            required_joint_keys = (
                'search_v3_points', 'search_v3_point_valid_mask',
                'search_v3_point_source', 'search_v3_branch_source',
                'trajectory_search_points',
                'trajectory_search_point_valid_mask',
                'trajectory_search_point_source',
                'trajectory_search_branch_source',
                'trajectory_search_valid',
                'search_v3_support_valid',
                'search_v3_query_delta_t', 'search_v3_gap_ratio',
            )
            missing_joint = [
                key for key in required_joint_keys if key not in input_dict]
            if missing_joint:
                raise KeyError(
                    "CT joint Full input is missing: "
                    + ", ".join(missing_joint))
            if observation_query is None:
                raise RuntimeError(
                    "CT joint Full requires the final observation decoder query")

            if (self.ct_enable_b2
                    and self.ct_joint_contract_version >= 3):
                updated_aux_box = self._forward_ct_contract_v3(
                    input_dict, output_dict, observation_aux_box,
                    obs_stats, main_motion, history_valid_ratio,
                    B, L, chunk_size, coarse_box=aux_box)

            if (self.ct_enable_b2
                    and self.ct_joint_contract_version < 3):
                endpoint_points = input_dict['search_v3_points'].to(
                    device=point_feature.device, dtype=point_feature.dtype)
                tube_points = input_dict['trajectory_search_points'].to(
                    device=point_feature.device, dtype=point_feature.dtype)
                if endpoint_points.shape[1] != self.ct_endpoint_quota:
                    raise ValueError("CT endpoint quota/data shape mismatch")
                if tube_points.shape[1] != self.ct_tube_quota:
                    raise ValueError("CT tube quota/data shape mismatch")
                raw_joint_points = torch.cat((
                    endpoint_points, tube_points), dim=1)
                encoded_joint_points = self.encode_point_time(raw_joint_points)
                if encoded_joint_points.shape[-1] != 5:
                    raise ValueError(
                        "CT joint Full requires scalar point-time encoding")
                endpoint_mask = input_dict[
                    'search_v3_point_valid_mask'].to(
                        device=point_feature.device,
                        dtype=point_feature.dtype)
                tube_mask = input_dict[
                    'trajectory_search_point_valid_mask'].to(
                    device=point_feature.device,
                    dtype=point_feature.dtype).reshape(
                        B, self.ct_tube_quota)
                joint_mask = torch.cat((endpoint_mask, tube_mask), dim=1)
                joint_source = torch.cat((
                    input_dict['search_v3_point_source'].to(
                        device=point_feature.device, dtype=torch.long),
                    input_dict['trajectory_search_point_source'].to(
                        device=point_feature.device, dtype=torch.long),
                ), dim=1)
                joint_branch_source = torch.cat((
                    input_dict['search_v3_branch_source'].to(
                        device=point_feature.device, dtype=torch.long),
                    input_dict['trajectory_search_branch_source'].to(
                        device=point_feature.device, dtype=torch.long),
                ), dim=1)
                learned_motion_valid = main_motion['valid'] * float(
                    self.ct_enable_b1)
                joint_output = self.ct_joint_search_refiner(
                    encoded_joint_points,
                    raw_joint_points[..., :2],
                    joint_mask,
                    joint_source,
                    observation_query,
                    obs_stats,
                    main_motion['feature'].detach(),
                    main_motion['kinematic_prior_xy'].detach(),
                    main_motion['prior_xy'].detach(),
                    main_motion['residual_unit_parallel_perp'].detach(),
                    torch.exp(main_motion[
                        'log_sigma_parallel_perp']).detach(),
                    main_motion['envelope_parallel_perp'].detach(),
                    main_motion['motion_direction_xy'].detach(),
                    learned_motion_valid,
                    input_dict['search_v3_query_delta_t'],
                    input_dict['search_v3_gap_ratio'],
                    history_valid_ratio,
                    point_branch_source=joint_branch_source,
                    search_support_valid=input_dict[
                        'search_v3_support_valid'],
                )
                support_mask = input_dict['search_v3_support_valid'].to(
                    device=point_feature.device).reshape(B) > 0
                observation_xy = observation_aux_box[:, :2].detach()
                unmasked_raw_search_xy = joint_output[
                    'ct_search_raw_alpha_xy'
                    if self.ct_joint_contract_version >= 2
                    else 'ct_search_raw_xy']
                joint_output['ct_search_unmasked_raw_xy'] = (
                    unmasked_raw_search_xy)
                effective_mask = joint_output[
                    'ct_search_effective'].reshape(B) > 0
                joint_output['ct_search_raw_xy'] = torch.where(
                    (support_mask & effective_mask).unsqueeze(1),
                    unmasked_raw_search_xy, observation_xy)
                output_dict.update(joint_output)
                output_dict['ct_search_geometry_valid'] = input_dict[
                    'ct_search_geometry_valid'].to(
                        device=point_feature.device,
                        dtype=point_feature.dtype).reshape(B)
                output_dict['ct_b1_geometry_source_id'] = input_dict[
                    'search_v3_prior_source_id'].to(
                        device=point_feature.device,
                        dtype=point_feature.dtype).reshape(B)
                output_dict['candidate_valid'] = joint_output[
                    'ct_search_support_valid']
                updated_aux_box, router_output = self.ct_joint_router(
                    observation_aux_box,
                    unmasked_raw_search_xy,
                    joint_output['ct_search_candidate_valid'],
                    obs_stats,
                    joint_output['ct_search_targetness_mean'],
                    joint_output['ct_search_targetness_max'],
                    joint_output['ct_search_targetness_entropy'],
                    joint_output['ct_search_normalized_ess'],
                    joint_output['ct_query_gate'],
                    input_dict['search_v3_query_delta_t'],
                    input_dict['search_v3_gap_ratio'],
                    enabled=self.ct_enable_b3,
                    extension_mass_ratio=joint_output[
                        'ct_search_extension_mass_ratio'],
                    extension_vote_rms=joint_output[
                        'ct_search_extension_vote_rms'],
                    presence_probability=joint_output[
                        'ct_search_presence_probability'],
                    total_point_count=input_dict.get(
                        'ct_search_total_point_count'),
                    extension_point_count=input_dict.get(
                        'ct_search_extension_count'),
                    extension_voxels=input_dict.get(
                        'ct_search_extension_voxels'),
                    coverage_need=input_dict.get(
                        'ct_search_coverage_need'),
                    quality_valid=input_dict.get(
                        'ct_search_quality_valid'),
                    proposal_valid=input_dict.get(
                        'ct_search_proposal_valid'),
                    predicted_displacement=input_dict.get(
                        'ct_search_predicted_displacement'),
                )
                output_dict.update(router_output)
                applied_mask = router_output[
                    'ct_router_applied_gate'].reshape(B) > 0
                deployed_raw = torch.where(
                    applied_mask.unsqueeze(1),
                    unmasked_raw_search_xy, observation_xy)
                deployed_query = torch.where(
                    applied_mask.unsqueeze(1),
                    joint_output['ct_query_search_internal'],
                    joint_output['ct_query_observation'])
                deployed_query_gate = (
                    joint_output['ct_query_gate_internal']
                    * applied_mask.to(point_feature.dtype))
                output_dict.update({
                    'ct_search_raw_xy': deployed_raw,
                    'ct_query_search': deployed_query,
                    'ct_query_gate': deployed_query_gate,
                    'ct_query_shift_norm': torch.linalg.norm(
                        deployed_query
                        - joint_output['ct_query_observation'], dim=1)
                    / math.sqrt(self.ct_joint_search_refiner.query_dim),
                })
                output_dict['ct_final_box'] = updated_aux_box
            elif not self.ct_enable_b2:
                updated_aux_box = observation_aux_box
                support_valid = input_dict[
                    'search_v3_support_valid'].to(
                        device=point_feature.device,
                        dtype=point_feature.dtype).reshape(B)
                output_dict.update({
                    # Compatibility alias is structural support, not a B2
                    # candidate-quality claim, even in the -B2 ablation.
                    'candidate_valid': support_valid,
                    'ct_search_support_valid': support_valid,
                    'ct_search_candidate_valid': point_feature.new_zeros((B,)),
                    'ct_query_gate': point_feature.new_zeros((B,)),
                    'ct_router_gate': point_feature.new_zeros((B,)),
                    'ct_router_applied_gate': point_feature.new_zeros((B,)),
                    'ct_final_box': observation_aux_box,
                })

        if self.use_search_evidence_v21:
            required_search_keys = (
                'search_v21_points',
                'search_v21_point_valid_mask',
                'search_v21_point_source',
                'search_v21_geometry_valid',
                'search_v21_endpoint_xy',
                'search_v21_query_delta_t',
                'search_v21_gap_ratio',
                'search_v21_sigma_parallel',
                'search_v21_sigma_perpendicular',
                'search_v21_available_count',
                'search_v21_extension_count',
                'search_v21_overlap_count',
            )
            missing_search = [
                key for key in required_search_keys if key not in input_dict]
            if missing_search:
                raise KeyError(
                    "B2-v2.1 input is missing: "
                    + ", ".join(missing_search))

            raw_search_points = input_dict['search_v21_points'].to(
                device=point_feature.device, dtype=point_feature.dtype)
            encoded_search_points = self.encode_point_time(raw_search_points)
            if encoded_search_points.shape[-1] != 5:
                raise ValueError(
                    "B2-v2.1 requires a scalar point-time encoding")
            endpoint_xy = input_dict['search_v21_endpoint_xy'].to(
                device=point_feature.device, dtype=point_feature.dtype)
            point_xy = raw_search_points[..., :2]
            delta_to_endpoint = endpoint_xy.unsqueeze(1) - point_xy
            direction_norm = torch.linalg.norm(
                endpoint_xy, dim=1, keepdim=True)
            default_direction = torch.zeros_like(endpoint_xy)
            default_direction[:, 0] = 1.0
            direction = torch.where(
                (direction_norm > 1e-6).expand_as(endpoint_xy),
                endpoint_xy / torch.clamp(direction_norm, min=1e-6),
                default_direction)
            perpendicular = torch.stack((
                -direction[:, 1], direction[:, 0]), dim=1)
            sigma_parallel = input_dict[
                'search_v21_sigma_parallel'].to(
                    device=point_feature.device,
                    dtype=point_feature.dtype).reshape(B, 1)
            sigma_perpendicular = input_dict[
                'search_v21_sigma_perpendicular'].to(
                    device=point_feature.device,
                    dtype=point_feature.dtype).reshape(B, 1)
            longitudinal = (
                delta_to_endpoint * direction.unsqueeze(1)
            ).sum(dim=2) / torch.clamp(sigma_parallel, min=0.25)
            lateral = (
                delta_to_endpoint * perpendicular.unsqueeze(1)
            ).sum(dim=2) / torch.clamp(sigma_perpendicular, min=0.20)
            search_point_inputs = torch.cat((
                encoded_search_points,
                delta_to_endpoint,
                longitudinal.unsqueeze(2),
                lateral.unsqueeze(2),
            ), dim=2)

            motion_dim = int(getattr(
                self.config, 'motion_v3_hidden_dim', 128))
            if self.use_b1motion_v3:
                v21_motion_feature = main_motion['feature']
                v21_motion_proposal_xy = motion_prior_proposal_xy
                v21_motion_log_sigma_xy = main_motion['log_sigma_xy']
                v21_motion_valid = main_motion['valid']
                v21_history_valid_ratio = history_valid_ratio
            else:
                v21_motion_feature = point_feature.new_zeros((B, motion_dim))
                v21_motion_proposal_xy = observation_aux_box[:, :2]
                v21_motion_log_sigma_xy = point_feature.new_zeros((B, 2))
                v21_motion_valid = point_feature.new_zeros((B,))
                v21_history_valid_ratio = input_dict['valid_mask'].to(
                    device=point_feature.device,
                    dtype=point_feature.dtype).mean(dim=1)

            search_output = self.search_evidence_v21(
                search_point_inputs,
                point_xy,
                input_dict['search_v21_point_valid_mask'],
                input_dict['search_v21_point_source'],
                input_dict['search_v21_geometry_valid'],
                point_feature.detach(),
                v21_motion_feature.detach(),
                v21_motion_valid,
                obs_stats.detach(),
                input_dict['search_v21_query_delta_t'],
                input_dict['search_v21_gap_ratio'],
                sigma_parallel,
                sigma_perpendicular,
                input_dict['search_v21_available_count'],
                input_dict['search_v21_extension_count'],
                input_dict['search_v21_overlap_count'],
            )
            output_dict.update(search_output)

            mode = self.proposal_inference_mode
            if self.training and mode != 'full':
                raise ValueError(
                    "proposal_inference_mode is evaluation-only; "
                    "training must use full")
            motion_mode_enabled = mode in ('obs_motion', 'full')
            search_mode_enabled = mode in ('obs_search', 'full')
            applied_motion_valid = v21_motion_valid * float(
                motion_mode_enabled)
            applied_search_valid = search_output[
                'search_v21_candidate_valid'] * float(search_mode_enabled)
            if self.use_advantage_proposal_fusion:
                trainer = getattr(self, '_trainer', None)
                trainer_state = str(getattr(
                    getattr(trainer, 'state', None), 'fn', '')).lower()
                in_fit_loop = self.training or 'fit' in trainer_state
                advantage_scale = 1.0
                if in_fit_loop:
                    epoch = int(getattr(self, 'current_epoch', 0))
                    if epoch < self.advantage_fusion_warmup_epochs:
                        advantage_scale = 0.0
                    elif self.advantage_fusion_ramp_epochs > 0:
                        advantage_scale = min(
                            1.0,
                            (epoch - self.advantage_fusion_warmup_epochs + 1)
                            / float(self.advantage_fusion_ramp_epochs))
                updated_aux_box, fusion_diagnostics = (
                    self.advantage_proposal_fusion(
                        observation_aux_box,
                        point_feature,
                        obs_stats,
                        v21_motion_feature,
                        v21_motion_proposal_xy,
                        v21_motion_log_sigma_xy,
                        applied_motion_valid,
                        v21_history_valid_ratio,
                        search_output['search_v21_evidence_token'],
                        search_output['search_v21_proposal_xy'],
                        applied_search_valid,
                        search_output['search_v21_targetness_mean'],
                        search_output['search_v21_targetness_max'],
                        search_output['search_v21_targetness_entropy'],
                        search_output['search_v21_effective_sample_size'],
                        search_output['search_v21_extension_weight_ratio'],
                        input_dict['search_v21_available_count'],
                        input_dict['search_v21_extension_count'],
                        input_dict['search_v21_overlap_count'],
                        input_dict['search_v21_query_delta_t'],
                        input_dict['search_v21_gap_ratio'],
                        enabled_scale=advantage_scale,
                    ))
            else:
                observation_refinement_xy = (
                    observation_aux_box[:, :2] - aux_box[:, :2])
                updated_aux_box, fusion_diagnostics = self.b3_risk_router(
                    observation_aux_box,
                    point_feature,
                    obs_stats,
                    obs_aux['obs_segmentation_entropy'],
                    observation_refinement_xy,
                    v21_motion_feature,
                    v21_motion_proposal_xy,
                    v21_motion_log_sigma_xy,
                    applied_motion_valid,
                    v21_history_valid_ratio,
                    search_output['search_v21_evidence_token'],
                    search_output['search_v21_proposal_xy'],
                    applied_search_valid,
                    search_output['search_v21_targetness_mean'],
                    search_output['search_v21_targetness_max'],
                    search_output['search_v21_targetness_entropy'],
                    search_output['search_v21_effective_sample_size'],
                    search_output['search_v21_extension_weight_ratio'],
                    input_dict['search_v21_available_count'],
                    input_dict['search_v21_extension_count'],
                    input_dict['search_v21_overlap_count'],
                    input_dict['search_v21_query_delta_t'],
                    input_dict['search_v21_gap_ratio'],
                    enabled_scale=self.b3_enabled_scale,
                )
            output_dict.update(fusion_diagnostics)
            output_dict.update({
                'search_v21_motion_proposal_xy': v21_motion_proposal_xy,
                'search_v21_motion_candidate_available': v21_motion_valid,
                'search_v21_search_candidate_available': search_output[
                    'search_v21_candidate_valid'],
                'search_v21_motion_candidate_valid': applied_motion_valid,
                'search_v21_search_candidate_valid': applied_search_valid,
                'proposal_inference_mode_id': point_feature.new_tensor({
                    'obs': 0, 'obs_motion': 1,
                    'obs_search': 2, 'full': 3,
                }[mode]),
            })

        if self.use_motion_conditioned_search_v22:
            required_search_keys = (
                'search_v22_points',
                'search_v22_point_valid_mask',
                'search_v22_point_source',
                'search_v22_geometry_valid',
                'search_v22_support_anchor_xy',
                'search_v22_query_delta_t',
                'search_v22_gap_ratio',
                'search_v22_sigma_parallel',
                'search_v22_sigma_perpendicular',
                'search_v22_available_count',
                'search_v22_extension_count',
                'search_v22_overlap_count',
            )
            missing_search = [
                key for key in required_search_keys if key not in input_dict]
            if missing_search:
                raise KeyError(
                    "B2-v2.2 input is missing: "
                    + ", ".join(missing_search))
            if not self.use_b1motion_v3:
                raise RuntimeError("B2-v2.2 requires a B1 motion state prior")

            raw_search_points = input_dict['search_v22_points'].to(
                device=point_feature.device, dtype=point_feature.dtype)
            encoded_search_points = self.encode_point_time(raw_search_points)
            if encoded_search_points.shape[-1] != 5:
                raise ValueError(
                    "B2-v2.2 requires a scalar point-time encoding")
            support_anchor_xy = input_dict[
                'search_v22_support_anchor_xy'].to(
                    device=point_feature.device, dtype=point_feature.dtype)
            point_xy = raw_search_points[..., :2]
            delta_to_support = support_anchor_xy.unsqueeze(1) - point_xy
            delta_to_motion = (
                motion_prior_proposal_xy.unsqueeze(1) - point_xy)
            direction_norm = torch.linalg.norm(
                support_anchor_xy, dim=1, keepdim=True)
            default_direction = torch.zeros_like(support_anchor_xy)
            default_direction[:, 0] = 1.0
            direction = torch.where(
                (direction_norm > 1e-6).expand_as(support_anchor_xy),
                support_anchor_xy / torch.clamp(direction_norm, min=1e-6),
                default_direction)
            perpendicular = torch.stack((
                -direction[:, 1], direction[:, 0]), dim=1)
            sigma_parallel = input_dict[
                'search_v22_sigma_parallel'].to(
                    device=point_feature.device,
                    dtype=point_feature.dtype).reshape(B, 1)
            sigma_perpendicular = input_dict[
                'search_v22_sigma_perpendicular'].to(
                    device=point_feature.device,
                    dtype=point_feature.dtype).reshape(B, 1)
            longitudinal = (
                delta_to_support * direction.unsqueeze(1)
            ).sum(dim=2) / torch.clamp(sigma_parallel, min=0.25)
            lateral = (
                delta_to_support * perpendicular.unsqueeze(1)
            ).sum(dim=2) / torch.clamp(sigma_perpendicular, min=0.20)
            search_point_inputs = torch.cat((
                encoded_search_points,
                delta_to_support,
                longitudinal.unsqueeze(2),
                lateral.unsqueeze(2),
            ), dim=2)

            search_output = self.motion_conditioned_search_refiner(
                search_point_inputs,
                point_xy,
                delta_to_motion,
                input_dict['search_v22_point_valid_mask'],
                input_dict['search_v22_point_source'],
                input_dict['search_v22_geometry_valid'],
                support_anchor_xy,
                point_feature,
                main_motion['feature'],
                motion_prior_proposal_xy,
                main_motion['valid'],
                obs_stats,
                input_dict['search_v22_query_delta_t'],
                input_dict['search_v22_gap_ratio'],
                sigma_parallel,
                sigma_perpendicular,
                input_dict['search_v22_available_count'],
                input_dict['search_v22_extension_count'],
                input_dict['search_v22_overlap_count'],
            )
            output_dict.update(search_output)

            mode = self.proposal_inference_mode
            if mode == 'obs_search':
                raise ValueError(
                    "B2-v2.2 has no independent obs_search candidate")
            motion_mode_enabled = mode in ('obs_motion', 'full',
                                           'full_selective')
            refined_mode_enabled = mode in (
                'obs_motion_search', 'full', 'full_selective')
            applied_motion_valid = (
                main_motion['valid'] * float(motion_mode_enabled))
            applied_refined_valid = (
                search_output['motion_search_candidate_valid']
                * float(refined_mode_enabled))
            observation_refinement_xy = (
                observation_aux_box[:, :2] - aux_box[:, :2])

            if self.use_signed_horizon_router:
                router_scale = (
                    0.0 if self.training else self.signed_router_enabled_scale)
                if mode == 'obs':
                    router_scale = 0.0
                updated_aux_box, router_diagnostics = (
                    self.signed_horizon_router(
                        observation_aux_box,
                        point_feature,
                        obs_stats,
                        obs_aux['obs_segmentation_entropy'],
                        observation_refinement_xy,
                        main_motion['feature'],
                        motion_prior_proposal_xy,
                        main_motion['log_sigma_xy'],
                        applied_motion_valid,
                        history_valid_ratio,
                        search_output['search_v22_evidence_token'],
                        search_output['motion_search_refined_xy'],
                        applied_refined_valid,
                        search_output['search_presence_probability'],
                        search_output['search_v22_targetness_mean'],
                        search_output['search_v22_targetness_max'],
                        search_output['search_v22_targetness_entropy'],
                        search_output['search_normalized_ess'],
                        search_output['search_v22_extension_weight_ratio'],
                        input_dict['search_v22_available_count'],
                        input_dict['search_v22_extension_count'],
                        input_dict['search_v22_overlap_count'],
                        support_anchor_xy,
                        search_output['search_raw_vote_xy'],
                        input_dict['search_v22_query_delta_t'],
                        input_dict['search_v22_gap_ratio'],
                        enabled_scale=router_scale,
                        forced_candidate=input_dict.get(
                            'selective_forced_candidate'),
                        forced_step_ratio=input_dict.get(
                            'selective_forced_step_ratio'),
                    ))
                output_dict.update(router_diagnostics)
            else:
                # Refiner-only training/evaluation never writes into history.
                updated_aux_box = observation_aux_box

            output_dict.update({
                'motion_prior_xy': motion_prior_proposal_xy,
                'motion_search_motion_candidate_valid':
                    applied_motion_valid,
                'motion_search_refined_candidate_valid':
                    applied_refined_valid,
                'proposal_inference_mode_id': point_feature.new_tensor({
                    'obs': 0,
                    'obs_motion': 1,
                    'obs_motion_search': 2,
                    'full': 3,
                    'full_selective': 3,
                }[mode]),
            })

        if self.use_motion_conditioned_search_v3:
            required_search_keys = (
                'search_v3_points', 'search_v3_point_valid_mask',
                'search_v3_point_source', 'search_v3_geometry_valid',
                'search_v3_support_anchor_xy', 'search_v3_query_delta_t',
                'search_v3_gap_ratio', 'search_v3_sigma_parallel',
                'search_v3_sigma_perpendicular',
                'search_v3_available_count', 'search_v3_extension_count',
                'search_v3_overlap_count', 'b2_v3_history_ref_boxs',
                'b2_v3_history_valid_mask', 'b2_v3_history_delta_t',
                'b2_v3_history_mode_id', 'b2_v3_history_anchor',
            )
            missing_search = [
                key for key in required_search_keys if key not in input_dict]
            if missing_search:
                raise KeyError(
                    "B2-v3 input is missing: " + ", ".join(missing_search))
            history_pairs = (
                ('b2_v3_history_ref_boxs', 'motion_main_ref_boxs'),
                ('b2_v3_history_valid_mask', 'motion_main_valid_mask'),
                ('b2_v3_history_delta_t', 'motion_main_delta_t'),
                ('b2_v3_history_anchor', 'motion_main_anchor'),
            )
            for search_key, motion_key in history_pairs:
                if motion_key not in input_dict:
                    raise KeyError(
                        f"B2-v3 history contract is missing {motion_key}")
                search_value = input_dict[search_key]
                motion_value = input_dict[motion_key]
                if (search_value.shape != motion_value.shape
                        or not torch.equal(search_value, motion_value)):
                    raise RuntimeError(
                        f"B1/B2-v3 state mismatch: {motion_key} != "
                        f"{search_key}")
            search_query_dt = input_dict['search_v3_query_delta_t'].reshape(-1)
            motion_query_dt = input_dict[
                'motion_main_current_delta_t'].reshape(-1)
            if (search_query_dt.shape != motion_query_dt.shape
                    or not torch.equal(search_query_dt, motion_query_dt)):
                raise RuntimeError(
                    "B1/B2-v3 query delta_t mismatch")

            raw_search_points = input_dict['search_v3_points'].to(
                device=point_feature.device, dtype=point_feature.dtype)
            encoded_search_points = self.encode_point_time(raw_search_points)
            if encoded_search_points.shape[-1] != 5:
                raise ValueError("B2-v3 requires scalar point-time encoding")
            support_anchor_xy = input_dict[
                'search_v3_support_anchor_xy'].to(
                    device=point_feature.device, dtype=point_feature.dtype)
            point_xy = raw_search_points[..., :2]
            delta_to_support = support_anchor_xy.unsqueeze(1) - point_xy
            delta_to_motion = (
                motion_prior_proposal_xy.unsqueeze(1) - point_xy)
            if self.use_uncertainty_geometry:
                motion_geometry_valid = (
                    main_motion['valid'] > 0).unsqueeze(1)
                direction_reference = torch.where(
                    motion_geometry_valid,
                    main_motion['motion_direction_xy'],
                    support_anchor_xy,
                )
            else:
                motion_geometry_valid = None
                direction_reference = support_anchor_xy
            direction_norm = torch.linalg.norm(
                direction_reference, dim=1, keepdim=True)
            default_direction = torch.zeros_like(support_anchor_xy)
            default_direction[:, 0] = 1.0
            direction = torch.where(
                (direction_norm > 1e-6).expand_as(support_anchor_xy),
                direction_reference / torch.clamp(
                    direction_norm, min=1e-6),
                default_direction)
            perpendicular = torch.stack((
                -direction[:, 1], direction[:, 0]), dim=1)
            sigma_parallel = input_dict['search_v3_sigma_parallel'].to(
                device=point_feature.device,
                dtype=point_feature.dtype).reshape(B, 1)
            sigma_perpendicular = input_dict[
                'search_v3_sigma_perpendicular'].to(
                    device=point_feature.device,
                    dtype=point_feature.dtype).reshape(B, 1)
            geometry_delta = delta_to_support
            if self.use_uncertainty_geometry:
                geometry_delta = torch.where(
                    motion_geometry_valid.unsqueeze(1),
                    delta_to_motion,
                    delta_to_support,
                )
            longitudinal = (
                geometry_delta * direction.unsqueeze(1)
            ).sum(dim=2) / torch.clamp(sigma_parallel, min=0.25)
            lateral = (
                geometry_delta * perpendicular.unsqueeze(1)
            ).sum(dim=2) / torch.clamp(sigma_perpendicular, min=0.20)
            search_geometry = (
                encoded_search_points,
                geometry_delta,
                longitudinal.unsqueeze(2),
                lateral.unsqueeze(2),
            )
            if self.use_uncertainty_geometry:
                mahalanobis = torch.clamp(
                    longitudinal.pow(2) + lateral.pow(2),
                    min=0.0, max=float(getattr(
                        self.config, 'search_v3_mahalanobis_clip', 25.0)))
                search_geometry = search_geometry + (
                    mahalanobis.unsqueeze(2),)
            search_point_inputs = torch.cat(search_geometry, dim=2)
            search_query_feature = None
            if self.use_asymmetric_dual_query:
                search_query_feature, dual_query_diagnostics = (
                    self.asymmetric_dual_query(
                        observation_query,
                        main_motion['feature'],
                        main_motion['log_sigma_parallel_perp'],
                        input_dict['search_v3_query_delta_t'],
                        input_dict['search_v3_gap_ratio'],
                        main_motion['valid'],
                    ))
                output_dict.update(dual_query_diagnostics)
                output_dict['search_query'] = search_query_feature
            search_output = self.state_aligned_search_refiner(
                search_point_inputs,
                point_xy,
                delta_to_motion,
                input_dict['search_v3_point_valid_mask'],
                input_dict['search_v3_point_source'],
                input_dict['search_v3_geometry_valid'],
                support_anchor_xy,
                point_feature,
                main_motion['feature'],
                motion_prior_proposal_xy,
                main_motion['valid'],
                obs_stats,
                input_dict['search_v3_query_delta_t'],
                input_dict['search_v3_gap_ratio'],
                sigma_parallel,
                sigma_perpendicular,
                input_dict['search_v3_available_count'],
                input_dict['search_v3_extension_count'],
                input_dict['search_v3_overlap_count'],
                query_feature=search_query_feature,
            )
            output_dict.update(search_output)

            raw_search_xy = search_output['search_v3_raw_vote_xy']
            legacy_refined_xy = search_output[
                'motion_search_v3_refined_xy']
            mode = self.proposal_inference_mode
            # Historical ``obs_vs_refined`` modes retain the clipped v3
            # candidate.  The formal ``selective`` mode is bound to the raw
            # Search candidate and cannot silently change candidate identity.
            official_search_xy = (
                raw_search_xy if mode == 'selective'
                else legacy_refined_xy)
            output_dict.update({
                'raw_search_xy': raw_search_xy,
                'legacy_clipped_search_xy': legacy_refined_xy,
                'official_search_xy': official_search_xy,
                'official_search_is_raw': point_feature.new_full(
                    (B,), float(mode == 'selective')),
            })
            motion_mode_enabled = mode in (
                'obs_vs_motion', 'obs_vs_all', 'selective')
            refined_mode_enabled = mode in (
                'obs_vs_refined', 'obs_vs_all', 'selective')
            intrinsic_motion_valid = main_motion['valid']
            intrinsic_refined_valid = search_output[
                'motion_search_v3_candidate_structural_valid']
            action_allowed_mask = point_feature.new_tensor((
                float(motion_mode_enabled), float(refined_mode_enabled),
            )).reshape(1, 2).expand(B, -1)
            applied_motion_valid = (
                intrinsic_motion_valid * action_allowed_mask[:, 0])
            applied_refined_valid = (
                intrinsic_refined_valid * action_allowed_mask[:, 1])
            observation_refinement_xy = (
                observation_aux_box[:, :2] - aux_box[:, :2])
            if self.use_action_consistent_router_v3:
                router_scale = (
                    0.0 if self.training else self.router_v3_enabled_scale)
                default_policy = (
                    ActionConsistentInnovationRouter.POLICY_OBSERVATION
                    if mode in ('obs_only', 'observation')
                    else ActionConsistentInnovationRouter.POLICY_AUTO)
                policy_override = input_dict.get(
                    'selective_v3_policy_override')
                if policy_override is None:
                    policy_override = point_feature.new_full(
                        (B,), default_policy, dtype=torch.long)
                updated_aux_box, router_diagnostics = (
                    self.action_consistent_router_v3(
                        observation_aux_box,
                        point_feature,
                        obs_stats,
                        obs_aux['obs_segmentation_entropy'],
                        observation_refinement_xy,
                        main_motion['feature'],
                        motion_prior_proposal_xy,
                        main_motion['log_sigma_xy'],
                        intrinsic_motion_valid,
                        history_valid_ratio,
                        search_output['search_v3_evidence_components'],
                        official_search_xy,
                        intrinsic_refined_valid,
                        search_output['search_v3_presence_probability'],
                        search_output['search_v3_targetness_mean'],
                        search_output['search_v3_targetness_max'],
                        search_output['search_v3_targetness_entropy'],
                        search_output['search_v3_normalized_ess'],
                        search_output['search_v3_extension_weight_ratio'],
                        input_dict['search_v3_available_count'],
                        input_dict['search_v3_extension_count'],
                        input_dict['search_v3_overlap_count'],
                        support_anchor_xy,
                        search_output['search_v3_raw_vote_xy'],
                        input_dict['search_v3_query_delta_t'],
                        input_dict['search_v3_gap_ratio'],
                        enabled_scale=router_scale,
                        policy_override=policy_override,
                        forced_step_ratio=input_dict.get(
                            'selective_v3_forced_step_ratio'),
                        action_allowed_mask=action_allowed_mask,
                        search_utility=search_output.get(
                            'search_v3_utility_probability'),
                        support_truncated=input_dict.get(
                            'search_v3_support_truncated'),
                    ))
                output_dict.update(router_diagnostics)
            else:
                updated_aux_box = observation_aux_box
            if mode in (
                    'observation', 'motion', 'raw_search',
                    'legacy_clipped'):
                if self.training and mode != 'observation':
                    raise RuntimeError(
                        "direct candidate attribution modes are evaluation-only")
                direct_xy = observation_aux_box[:, :2]
                direct_valid = torch.ones_like(intrinsic_motion_valid)
                if mode == 'motion':
                    direct_xy = motion_prior_proposal_xy
                    direct_valid = intrinsic_motion_valid
                elif mode == 'raw_search':
                    direct_xy = raw_search_xy
                    direct_valid = intrinsic_refined_valid
                elif mode == 'legacy_clipped':
                    direct_xy = legacy_refined_xy
                    direct_valid = intrinsic_refined_valid
                applied_xy = torch.where(
                    (direct_valid > 0).unsqueeze(1),
                    direct_xy,
                    observation_aux_box[:, :2],
                )
                updated_aux_box = torch.cat((
                    applied_xy, observation_aux_box[:, 2:]), dim=1)
            output_dict.update({
                'motion_prior_xy': motion_prior_proposal_xy,
                'motion_search_v3_motion_candidate_valid':
                    applied_motion_valid,
                'motion_search_v3_refined_candidate_valid':
                    applied_refined_valid,
                'motion_search_v3_intrinsic_motion_candidate_valid':
                    intrinsic_motion_valid,
                'motion_search_v3_intrinsic_refined_candidate_valid':
                    intrinsic_refined_valid,
                'proposal_inference_mode_id': point_feature.new_tensor({
                    'obs_only': 0,
                    'obs_vs_motion': 1,
                    'obs_vs_refined': 2,
                    'obs_vs_all': 3,
                    'observation': 4,
                    'motion': 5,
                    'raw_search': 6,
                    'legacy_clipped': 7,
                    'selective': 8,
                }[mode]),
            })


        return self._finalize_observation_output(
            input_dict, output_dict, aux_box, observation_aux_box,
            updated_aux_box, seg_logits, motion_pred, updated_ref_boxs,
            obs_aux)

    def _compute_pair_validity(self, output_a, output_b, data_a, data_b):
        box_a = output_a["aux_estimation_boxes"]
        box_b = output_b["aux_estimation_boxes"]
        valid = torch.ones(
            box_a.shape[0], device=box_a.device, dtype=torch.bool)

        def paired_setting(suffix, default):
            return getattr(self.config, f"b4_{suffix}", default)

        eps = paired_setting("timestamp_eps", 1e-6)
        anchor_eps = paired_setting("anchor_eps", 1e-4)
        delta_eps = paired_setting("delta_eps", 1e-5)

        if "current_timestamp" in data_a and "current_timestamp" in data_b:
            current_gap = torch.abs(
                data_a["current_timestamp"].to(box_a.device)
                - data_b["current_timestamp"].to(box_a.device))
            valid = valid & (current_gap.view(-1) <= eps)

        anchor_gap = box_a.new_zeros((box_a.shape[0],))
        if "coordinate_anchor" in data_a and "coordinate_anchor" in data_b:
            anchor_gap = torch.max(
                torch.abs(
                    data_a["coordinate_anchor"].to(
                        box_a.device, dtype=box_a.dtype)
                    - data_b["coordinate_anchor"].to(
                        box_a.device, dtype=box_a.dtype)),
                dim=1,
            ).values
            if (paired_setting("fail_on_anchor_mismatch", True)
                    and bool(torch.any(anchor_gap > anchor_eps).item())):
                raise RuntimeError(
                    "Paired history views do not share coordinate_anchor. "
                    f"max gap={float(anchor_gap.max().item()):.6g}, "
                    f"eps={anchor_eps:.6g}.")
            valid = valid & (anchor_gap <= anchor_eps)
        elif paired_setting("require_coordinate_anchor", True):
            raise KeyError(
                "Paired history training requires coordinate_anchor in both views.")
        elif "ref_boxs" in data_a and "ref_boxs" in data_b:
            anchor_gap = torch.max(
                torch.abs(
                    data_a["ref_boxs"][:, 0].to(
                        box_a.device, dtype=box_a.dtype)
                    - data_b["ref_boxs"][:, 0].to(
                        box_a.device, dtype=box_a.dtype)),
                dim=1,
            ).values
            valid = valid & (anchor_gap <= anchor_eps)

        current_point_gap = box_a.new_zeros((box_a.shape[0],))
        point_eps = paired_setting("point_eps", 1e-6)
        if "points" in data_a and "points" in data_b:
            point_sample_size = int(getattr(
                self.config, "point_sample_size", 0))
            if point_sample_size <= 0:
                raise ValueError(
                    "Shared-current-point check requires point_sample_size > 0.")
            current_xyz_a = data_a["points"][:, -point_sample_size:, :3].to(
                box_a.device, dtype=box_a.dtype)
            current_xyz_b = data_b["points"][:, -point_sample_size:, :3].to(
                box_a.device, dtype=box_a.dtype)
            current_point_gap = torch.amax(
                torch.abs(current_xyz_a - current_xyz_b), dim=(1, 2))
            if (paired_setting("fail_on_current_point_mismatch", True)
                    and bool(torch.any(current_point_gap > point_eps).item())):
                raise RuntimeError(
                    "Paired views do not share sampled current XYZ points. "
                    f"max gap={float(current_point_gap.max().item()):.6g}, "
                    f"eps={point_eps:.6g}.")
            valid = valid & (current_point_gap <= point_eps)
        elif paired_setting("require_shared_current_points", True):
            raise KeyError(
                "Paired history training requires points in both views.")

        history_gap = None
        if "history_offsets" in data_a and "history_offsets" in data_b:
            history_gap = torch.max(
                torch.abs(
                    data_a["history_offsets"].to(
                        box_a.device, dtype=box_a.dtype)
                    - data_b["history_offsets"].to(
                        box_a.device, dtype=box_a.dtype)),
                dim=1,
            ).values
        elif "delta_T_real" in data_a and "delta_T_real" in data_b:
            history_gap = torch.max(
                torch.abs(
                    data_a["delta_T_real"].to(
                        box_a.device, dtype=box_a.dtype)
                    - data_b["delta_T_real"].to(
                        box_a.device, dtype=box_a.dtype)),
                dim=1,
            ).values
        elif "timestamps_real" in data_a and "timestamps_real" in data_b:
            history_gap = torch.max(
                torch.abs(
                    data_a["timestamps_real"][:, :-1].to(
                        box_a.device, dtype=box_a.dtype)
                    - data_b["timestamps_real"][:, :-1].to(
                        box_a.device, dtype=box_a.dtype)),
                dim=1,
            ).values
        elif "delta_T" in data_a and "delta_T" in data_b:
            history_gap = torch.max(
                torch.abs(
                    data_a["delta_T"].to(
                        box_a.device, dtype=box_a.dtype)
                    - data_b["delta_T"].to(
                        box_a.device, dtype=box_a.dtype)),
                dim=1,
            ).values
        if history_gap is not None:
            valid = valid & (history_gap > delta_eps)

        if paired_setting("full_history_only", True):
            full_a = (
                data_a["valid_mask"].to(box_a.device).sum(dim=1)
                >= data_a["valid_mask"].shape[1])
            full_b = (
                data_b["valid_mask"].to(box_a.device).sum(dim=1)
                >= data_b["valid_mask"].shape[1])
            valid = valid & full_a & full_b
        return {
            "valid": valid,
            "anchor_gap": anchor_gap,
            "current_point_gap": current_point_gap,
            "history_gap": history_gap,
        }

    def _pftc_ramp(self):
        if self.pftc_ramp_epochs == 0:
            return 1.0
        current_epoch = int(getattr(self, "current_epoch", 0))
        return min(max(
            (current_epoch + 1) / float(self.pftc_ramp_epochs),
            0.0), 1.0)

    def compute_paired_loss(self, data, output):
        data_a, data_b = data["view_a"], data["view_b"]
        output_a, output_b = output["view_a"], output["view_b"]

        loss_a = self.compute_loss(data_a, output_a)
        loss_b = self.compute_loss(data_b, output_b)

        loss_total_sup = 0.5 * (loss_a["loss_total"] + loss_b["loss_total"])
        if self.use_decoder_token_consistency:
            if ("decoder_state" not in output_a
                    or "decoder_state" not in output_b):
                raise KeyError(
                    "decoder-token consistency requires decoder_state outputs")
            consistency = self.decoder_token_consistency(
                output_a["decoder_state"], output_b["decoder_state"])
            if (self.training
                    and not bool(self.decoder_tc_weight_selector.frozen)):
                box_gradient = torch.autograd.grad(
                    loss_total_sup,
                    output_b["decoder_state"],
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                consistency_gradient = torch.autograd.grad(
                    consistency["loss_decoder_token_consistency"],
                    output_b["decoder_state"],
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                if box_gradient is not None and consistency_gradient is not None:
                    unweighted_ratio = (
                        torch.linalg.vector_norm(consistency_gradient.detach())
                        / torch.linalg.vector_norm(
                            box_gradient.detach()).clamp_min(1e-12))
                    self.decoder_tc_weight_selector.observe(
                        float(unweighted_ratio.cpu()))
            weight = self.decoder_tc_weight_selector.value(loss_total_sup)
            loss_total = (
                loss_total_sup
                + weight * consistency["loss_decoder_token_consistency"])
            loss_dict = {
                "loss_total": loss_total,
                "loss_total_sup": loss_total_sup,
                "loss_total_a": loss_a["loss_total"],
                "loss_total_b": loss_b["loss_total"],
                "decoder_tc_weight": weight,
                "decoder_tc_weight_frozen": (
                    self.decoder_tc_weight_selector.frozen.to(loss_total)),
                "decoder_tc_gradient_audit_count": (
                    self.decoder_tc_weight_selector.ratio_count.to(
                        loss_total)),
                "decoder_tc_gradient_ratio": (
                    self.decoder_tc_weight_selector.selected_weighted_ratio.to(
                        loss_total)),
                "decoder_tc_gradient_guardrail_passed": (
                    self.decoder_tc_weight_selector.guardrail_passed.to(
                        loss_total)),
            }
            for key, value in loss_a.items():
                loss_dict[f"view_a_{key}"] = value
            for key, value in loss_b.items():
                loss_dict[f"view_b_{key}"] = value
            loss_dict.update(consistency)
            return loss_dict

        raise RuntimeError(
            "paired views are reserved for B4 decoder consistency; the "
            "legacy symmetric consistency objective was removed")

    def _ct_online_targetness_class_weights(
            self, extension_labels, extension_valid, population='targetness'):
        """Update the cumulative canonical preflight and return its weights.

        This is the same inverse-frequency calculation used by the standalone
        preflight artifact, but it is accumulated inside the actual scratch
        experiment.  Missing classes keep the neutral configured prior and
        never block B2/Full startup.
        """
        if self.training:
            positive_points = (
                extension_labels.detach().to(torch.float64)
                * extension_valid.detach().to(torch.float64)).sum()
            negative_points = (
                (1.0 - extension_labels.detach().to(torch.float64))
                * extension_valid.detach().to(torch.float64)).sum()
            update_cumulative_binary_class_balance(
                getattr(self, f'ct_{population}_running_positive_points'),
                getattr(self, f'ct_{population}_running_negative_points'),
                getattr(self, f'ct_{population}_running_positive_weight'),
                getattr(self, f'ct_{population}_running_negative_weight'),
                positive_points, negative_points)
        return (
            getattr(self, f'ct_{population}_running_positive_weight').to(
                device=extension_labels.device, dtype=extension_labels.dtype),
            getattr(self, f'ct_{population}_running_negative_weight').to(
                device=extension_labels.device, dtype=extension_labels.dtype),
        )

    def _compute_ct_contract_v3_loss(self, data, output, target_xy):
        """B2 acquisition and detached B3 action-risk objectives."""
        dtype = output['ct_search_targetness_logits'].dtype
        device = output['ct_search_targetness_logits'].device
        prepool_labels = data['ct_extension_labels'].to(
            device=device, dtype=dtype)
        prepool_valid = data['ct_extension_valid_mask'].to(
            device=device, dtype=dtype)
        if self.ct_enable_v26_recovery:
            prepool_valid = prepool_valid * data[
                'ct_search_support_valid'].to(
                    device=device, dtype=dtype).reshape(-1, 1)
        selected_indices = output.get('ct_extension_selected_indices')
        if selected_indices is not None:
            selected_indices = selected_indices.to(device=device).long()
            extension_labels = torch.gather(
                prepool_labels, 1, selected_indices)
            extension_valid = torch.gather(
                prepool_valid, 1, selected_indices)
            extension_valid = extension_valid * output.get(
                'ct_extension_selected_valid_mask',
                torch.ones_like(extension_valid)).to(dtype)
        else:
            extension_labels = prepool_labels
            extension_valid = prepool_valid
        base_labels = data['ct_base_evidence_labels'].to(
            device=device, dtype=dtype)
        base_valid = data['ct_base_evidence_valid_mask'].to(
            device=device, dtype=dtype)
        candidate_id = data.get('candidate_id')
        if candidate_id is None:
            candidate_id = torch.zeros(
                target_xy.shape[0], device=device, dtype=torch.long)
        else:
            candidate_id = candidate_id.to(device=device).reshape(-1)
        b0_view_id = data.get('b0_view_id', candidate_id)
        b0_view_id = b0_view_id.to(device=device).reshape(-1)
        if bool(torch.any(b0_view_id != 0)):
            raise RuntimeError(
                "B2/B3 loss accepts the canonical B0 view only")
        canonical_row = (candidate_id == 0).to(dtype)

        def weighted_mean(values, valid=None):
            """Candidate-stratified numerator/denominator aggregation."""
            return candidate_stratified_mean(
                values, valid, candidate_id)

        targetness_error = F.binary_cross_entropy_with_logits(
            output['ct_search_targetness_logits'],
            extension_labels,
            reduction='none')
        positive_weight, negative_weight = (
            self._ct_online_targetness_class_weights(
                extension_labels, extension_valid))
        targetness_error = targetness_error * (
            extension_labels * positive_weight
            + (1.0 - extension_labels) * negative_weight)
        loss_targetness = weighted_mean(
            targetness_error, extension_valid)
        relation_error = F.binary_cross_entropy_with_logits(
            output['ct_relation_logits_prepool'],
            prepool_labels, reduction='none')
        if self.ct_enable_v27:
            relation_positive_weight, relation_negative_weight = self._ct_online_targetness_class_weights(
                prepool_labels, prepool_valid, population='relation')
        else:
            relation_positive_weight, relation_negative_weight = positive_weight, negative_weight
        relation_error = relation_error * (
            prepool_labels * relation_positive_weight
            + (1.0 - prepool_labels) * relation_negative_weight)
        loss_relation = weighted_mean(relation_error, prepool_valid)
        relation_probability = torch.sigmoid(
            output['ct_relation_logits_prepool'])
        relation_select = prepool_valid > 0
        relation_scores_flat = relation_probability[relation_select]
        relation_targets_flat = prepool_labels[relation_select] > 0.5
        relation_auroc = target_xy.new_tensor(0.5)
        relation_auprc = target_xy.new_zeros(())
        relation_ece = target_xy.new_zeros(())
        positive_scores = relation_scores_flat[relation_targets_flat]
        negative_scores = relation_scores_flat[~relation_targets_flat]
        if positive_scores.numel() and negative_scores.numel():
            ascending = torch.argsort(relation_scores_flat, stable=True)
            ranks = torch.empty_like(ascending, dtype=dtype)
            ranks[ascending] = torch.arange(
                1, ascending.numel() + 1, device=device, dtype=dtype)
            positive_count = positive_scores.numel()
            negative_count = negative_scores.numel()
            relation_auroc = (
                ranks[relation_targets_flat].sum()
                - 0.5 * positive_count * (positive_count + 1)) / float(
                    positive_count * negative_count)
        if positive_scores.numel():
            order = torch.argsort(relation_scores_flat, descending=True)
            ordered_target = relation_targets_flat[order].to(dtype)
            precision_at_k = torch.cumsum(
                ordered_target, dim=0) / torch.arange(
                    1, ordered_target.numel() + 1,
                    device=device, dtype=dtype)
            relation_auprc = (
                precision_at_k * ordered_target).sum() / torch.clamp(
                    ordered_target.sum(), min=1.0)
        if relation_scores_flat.numel():
            ece_terms = []
            for bin_index in range(10):
                lower = bin_index / 10.0
                upper = (bin_index + 1) / 10.0
                in_bin = (
                    (relation_scores_flat >= lower)
                    & (relation_scores_flat < upper
                       if bin_index < 9 else relation_scores_flat <= upper))
                if bool(in_bin.any()):
                    ece_terms.append(
                        in_bin.to(dtype).mean()
                        * torch.abs(
                            relation_scores_flat[in_bin].mean()
                            - relation_targets_flat[in_bin].to(dtype).mean()))
            if ece_terms:
                relation_ece = torch.stack(ece_terms).sum()

        foreground = extension_valid * extension_labels
        vote_target = target_xy.unsqueeze(1).expand_as(
            output['ct_search_point_votes'])
        vote_error = F.smooth_l1_loss(
            output['ct_search_point_votes'], vote_target,
            reduction='none').mean(dim=2)
        loss_vote = weighted_mean(vote_error, foreground)

        availability = output['ct_b2_available'].detach()
        observation_xy = output[
            'observation_aux_estimation_boxes'][:, :2].detach()
        raw_xy = output['ct_b2_raw_box'][:, :2]
        raw_error_per_sample = F.smooth_l1_loss(
            raw_xy, target_xy, reduction='none').mean(dim=1)
        base_presence_target = (
            (base_labels * base_valid).sum(dim=1) > 0).to(dtype)
        extension_presence_target = (
            (prepool_labels * prepool_valid).sum(dim=1) > 0).to(dtype)
        if self.ct_enable_v27:
            extension_presence_target = (
                (extension_labels * extension_valid).sum(dim=1) > 0).to(dtype)
        target_bearing = extension_target_bearing_mask(
            availability, extension_labels, extension_valid)
        # A raw candidate is identifiable as extension evidence only on rows
        # where the extension actually contains target points.  Absence rows
        # still train presence below, but never receive a GT center gradient.
        loss_raw = weighted_mean(raw_error_per_sample, target_bearing)
        base_presence_error = F.binary_cross_entropy_with_logits(
            output['ct_b2_base_presence_logit'],
            base_presence_target, reduction='none')
        extension_presence_error = F.binary_cross_entropy_with_logits(
            output['ct_b2_extension_presence_logit'],
            extension_presence_target, reduction='none')
        # Presence is learned on the same canonical on-policy population used
        # by every other B2 term.  Spatial recovery candidates are forbidden.
        loss_base_presence = weighted_mean(
            base_presence_error, canonical_row)
        loss_extension_presence = weighted_mean(
            extension_presence_error, availability * canonical_row)
        loss_presence = (.5 if not self.ct_enable_v27 else 1.) * (
            loss_base_presence + loss_extension_presence)

        observation_error = torch.linalg.norm(observation_xy - target_xy, dim=1)
        bounded_xy = observation_xy + output['ct_router_bounded_residual_xy'].detach()
        bounded_distance_error = torch.linalg.norm(bounded_xy - target_xy, dim=1)
        if self.ct_enable_v27:
            from utils.v27_training import compute_b3_utility_loss
            utility = compute_b3_utility_loss(data, output, self.config)
            loss_b3 = utility['loss'] if self.ct_enable_b3 else target_xy.new_zeros(())
            loss_helpful, loss_harmful = utility['loss_help'], utility['loss_harm']
            loss_center_gain, loss_iou_gain = utility['loss_precision'], utility['loss_success']
            helpful, harmful, b3_valid = utility['help_label'].bool(), utility['harm_label'].bool(), utility['valid']
            center_gain_h1, iou_gain_h1 = utility['h1_precision_gain'], utility['h1_success_gain']
            output['ct_b3_h1_success_gain_label'] = iou_gain_h1.detach()
            output['ct_b3_h1_precision_gain_label'] = center_gain_h1.detach()
            output['ct_b3_h1_valid'] = b3_valid.detach()
            h3_center_gain = data.get('ct_h3_precision_gain', torch.zeros_like(center_gain_h1)).detach()
            h3_iou_gain = data.get('ct_h3_success_gain', torch.zeros_like(iou_gain_h1)).detach()
            h3_valid = data.get('ct_h3_valid', torch.zeros_like(b3_valid)).detach()
        else:
            observation_error = torch.linalg.norm(
                observation_xy - target_xy, dim=1)
            bounded_xy = (
                observation_xy
                + output['ct_router_bounded_residual_xy'].detach())
            bounded_distance_error = torch.linalg.norm(
                bounded_xy - target_xy, dim=1)
            center_gain_h1 = observation_error - bounded_distance_error

            # Same-size axis-aligned BEV IoU is a stable differentiable proxy for
            # the expected-IoU head.  Exact oriented IoU remains an evaluation and
            # calibration metric.
            size_xy = data['bbox_size'][:, :2].to(
                device=device, dtype=dtype).clamp_min(1e-3)

            def same_size_iou(center):
                overlap = torch.clamp(
                    size_xy - torch.abs(center - target_xy), min=0.0)
                intersection = overlap[:, 0] * overlap[:, 1]
                area = size_xy[:, 0] * size_xy[:, 1]
                return intersection / torch.clamp(
                    2.0 * area - intersection, min=1e-6)

            iou_gain_h1 = (
                same_size_iou(bounded_xy) - same_size_iou(observation_xy))
            h3_center_gain = center_gain_h1.new_zeros(center_gain_h1.shape)
            h3_iou_gain = iou_gain_h1.new_zeros(iou_gain_h1.shape)
            h3_valid = availability.new_zeros(availability.shape)
            if 'ct_h3_center_gain' in data:
                h3_center_gain = data['ct_h3_center_gain'].to(
                    device=device, dtype=dtype).reshape(-1).detach()
            elif 'ct_h3_gain' in data:
                h3_center_gain = data['ct_h3_gain'].to(
                    device=device, dtype=dtype).reshape(-1).detach()
            if 'ct_h3_iou_gain' in data:
                h3_iou_gain = data['ct_h3_iou_gain'].to(
                    device=device, dtype=dtype).reshape(-1).detach()
            if 'ct_h3_valid' in data:
                h3_valid = data['ct_h3_valid'].to(
                    device=device, dtype=dtype).reshape(-1).detach()

            combined_center_gain = torch.where(
                h3_valid > 0,
                0.5 * (center_gain_h1 + h3_center_gain),
                center_gain_h1)
            combined_iou_gain = torch.where(
                h3_valid > 0,
                0.5 * (iou_gain_h1 + h3_iou_gain),
                iou_gain_h1)
            helpful = (
                (center_gain_h1 > self.ct_router_help_margin)
                & (iou_gain_h1 >= 0.0)
                & ((h3_valid <= 0) | (
                    (h3_center_gain > self.ct_router_h3_margin)
                    & (h3_iou_gain >= 0.0))))
            harmful = (
                (center_gain_h1 < -self.ct_router_help_margin)
                | (iou_gain_h1 < 0.0)
                | ((h3_valid > 0) & (
                    (h3_center_gain < -self.ct_router_h3_margin)
                    | (h3_iou_gain < 0.0)))
                | ~extension_presence_target.to(torch.bool))
            # B3 risk labels are action labels on the canonical state
            # distribution.  Auxiliary acquisition views cannot train B3.
            b3_valid = availability * canonical_row
            helpful_error = F.binary_cross_entropy_with_logits(
                output['ct_b3_help_logit'], helpful.to(dtype), reduction='none')
            harmful_error = F.binary_cross_entropy_with_logits(
                output['ct_b3_harm_logit'], harmful.to(dtype), reduction='none')
            loss_helpful = weighted_mean(helpful_error, b3_valid)
            loss_harmful = weighted_mean(harmful_error, b3_valid)
            loss_center_gain = weighted_mean(F.smooth_l1_loss(
                output['ct_b3_expected_center_gain'],
                combined_center_gain.detach(), reduction='none'), b3_valid)
            loss_iou_gain = weighted_mean(F.smooth_l1_loss(
                output['ct_b3_expected_iou_gain'],
                combined_iou_gain.detach(), reduction='none'), b3_valid)
            loss_b3 = target_xy.new_zeros(())
            if self.ct_enable_b3:
                loss_b3 = (
                    loss_helpful + loss_harmful
                    + loss_center_gain + loss_iou_gain)

        def acquisition_metric(key):
            value = data.get(key)
            if value is None:
                return target_xy.new_zeros((target_xy.shape[0],))
            return value.to(device=device, dtype=dtype).reshape(-1)

        base_target_count = acquisition_metric(
            'ct_acquisition_base_target_count')
        expansion_target_count = acquisition_metric(
            'ct_acquisition_expansion_target_count')
        pool_target_count = acquisition_metric(
            'ct_acquisition_extension_pool_target_count')
        prepool_target_count = acquisition_metric(
            'ct_acquisition_sampled_target_count')
        selected_target_count = (
            extension_labels * extension_valid).sum(dim=1)
        selected_group = output.get('ct_extension_selected_group')
        if selected_group is None:
            selected_group = torch.ones_like(extension_labels, dtype=torch.long)
        else:
            selected_group = selected_group.to(device=device).long()

        def selected_group_target_rate(group_id):
            group_valid = extension_valid * (
                selected_group == int(group_id)).to(dtype)
            return (extension_labels * group_valid).sum() / torch.clamp(
                group_valid.sum(), min=1.0)

        relation_selected_target_rate = selected_group_target_rate(1)
        spatial_selected_target_rate = selected_group_target_rate(2)
        exploration_selected_target_rate = selected_group_target_rate(3)
        borrowed_selected_target_rate = selected_group_target_rate(4)
        recovery_positive = acquisition_metric('ct_recovery_positive')
        recovery_fallback = acquisition_metric('ct_recovery_fallback')
        eligible_rows = prepool_target_count > 0
        retained_rows = eligible_rows & (selected_target_count > 0)
        eligible_row_count = eligible_rows.to(dtype).sum()
        retained_row_count = retained_rows.to(dtype).sum()
        acquisition_row_recall = retained_row_count / torch.clamp(
            eligible_row_count, min=1.0)
        pool_target_sum = pool_target_count.sum()
        prepool_target_sum = prepool_target_count.sum()
        selected_target_sum = selected_target_count.sum()
        acquisition_point_recall = selected_target_sum / torch.clamp(
            prepool_target_sum, min=1.0)

        b2_total = (
            self.ct_targetness_weight * loss_targetness
            + self.ct_relation_weight * loss_relation
            + self.ct_vote_weight * loss_vote
            + self.ct_raw_search_weight * loss_raw
            + self.ct_presence_weight * loss_presence)
        b3_total = self.ct_router_weight * loss_b3
        plugin_total = b2_total + b3_total
        return {
            'loss_ct_plugin_total': plugin_total,
            'loss_ct_b2_total': b2_total,
            'loss_ct_b3_total': b3_total,
            'loss_ct_targetness': loss_targetness,
            'loss_ct_relation': loss_relation,
            'ct_relation_auroc': relation_auroc,
            'ct_relation_ap': relation_auprc,
            'ct_relation_auprc': relation_auprc,
            'ct_relation_ece': relation_ece,
            'loss_ct_vote': loss_vote,
            'loss_ct_raw_search': loss_raw,
            'loss_ct_presence': loss_presence,
            'loss_ct_base_presence': loss_base_presence,
            'loss_ct_extension_presence': loss_extension_presence,
            'loss_ct_utility': loss_b3,
            'loss_ct_utility_classification': (
                loss_helpful + loss_harmful),
            'loss_ct_expected_gain': (
                loss_center_gain + loss_iou_gain),
            **({
                'loss_ct_expected_success_gain': loss_iou_gain,
                'loss_ct_expected_precision_gain': loss_center_gain,
                'ct_h1_success_gain': weighted_mean(iou_gain_h1, b3_valid),
                'ct_h1_precision_gain': weighted_mean(center_gain_h1, b3_valid),
                'ct_h3_exposure_rate': (h3_valid > 0).to(dtype).mean(),
                'ct_relation_positive_points_cumulative': self.ct_relation_running_positive_points.to(dtype),
                'ct_relation_negative_points_cumulative': self.ct_relation_running_negative_points.to(dtype),
            } if self.ct_enable_v27 else {}),
            'loss_ct_helpful': loss_helpful,
            'loss_ct_harmful': loss_harmful,
            'loss_ct_expected_center_gain': loss_center_gain,
            'loss_ct_expected_iou_gain': loss_iou_gain,
            'loss_ct_h3': loss_b3,
            'ct_h1_signed_gain': weighted_mean(
                center_gain_h1, availability),
            'ct_h1_helpful_rate': weighted_mean(
                helpful.to(dtype), b3_valid),
            'ct_h1_harmful_rate': weighted_mean(
                harmful.to(dtype), b3_valid),
            'ct_b2_target_bearing_rate': target_bearing.mean(),
            'ct_relation_selected_target_enrichment': (
                (extension_labels * extension_valid).sum()
                / torch.clamp(extension_valid.sum(), min=1.0)) / torch.clamp(
                    (prepool_labels * prepool_valid).sum()
                    / torch.clamp(prepool_valid.sum(), min=1.0), min=1e-6),
            'ct_relation_group_target_rate': relation_selected_target_rate,
            'ct_spatial_group_target_rate': spatial_selected_target_rate,
            'ct_exploration_group_target_rate':
                exploration_selected_target_rate,
            'ct_borrowed_group_target_rate': borrowed_selected_target_rate,
            'ct_relation_vs_spatial_enrichment': (
                relation_selected_target_rate / torch.clamp(
                    spatial_selected_target_rate, min=1e-6)),
            'ct_relation_vs_exploration_enrichment': (
                relation_selected_target_rate / torch.clamp(
                    exploration_selected_target_rate, min=1e-6)),
            'ct_vote_consistency_mean': output[
                'ct_vote_consistency'].mean(),
            'ct_vote_covariance_trace_mean': (
                output['ct_vote_covariance_xx']
                + output['ct_vote_covariance_yy']).mean(),
            'ct_vote_covariance_anisotropy_mean': (
                torch.abs(output['ct_vote_covariance_xx']
                          - output['ct_vote_covariance_yy'])
                / torch.clamp(
                    output['ct_vote_covariance_xx']
                    + output['ct_vote_covariance_yy'], min=1e-6)).mean(),
            'ct_vote_inlier_ratio_mean': output[
                'ct_vote_inlier_ratio'].mean(),
            'ct_vote_effective_mass_mean': output[
                'ct_vote_effective_mass'].mean(),
            'ct_vote_candidate_margin_mean': output[
                'ct_vote_candidate_margin'].mean(),
            'ct_vote_compatible_hypothesis_count_mean': output[
                'ct_vote_compatible_hypothesis_count'].mean(),
            'ct_no_extension_counterfactual_gain': target_xy.new_zeros(()),
            'ct_b2_availability_rate': availability.float().mean(),
            'ct_b2_base_presence_target_rate':
                base_presence_target.mean(),
            'ct_b2_extension_presence_target_rate':
                extension_presence_target.mean(),
            'ct_targetness_positive_points_cumulative':
                self.ct_targetness_running_positive_points.to(
                    device=device, dtype=dtype),
            'ct_targetness_negative_points_cumulative':
                self.ct_targetness_running_negative_points.to(
                    device=device, dtype=dtype),
            'ct_targetness_positive_weight': positive_weight,
            'ct_targetness_negative_weight': negative_weight,
            'ct_acquisition_base_presence_rate': (
                base_target_count > 0).to(dtype).mean(),
            'ct_acquisition_expansion_coverage_rate': (
                expansion_target_count > 0).to(dtype).mean(),
            'ct_acquisition_pool_presence_rate': (
                pool_target_count > 0).to(dtype).mean(),
            'ct_acquisition_sampled_presence_rate': (
                selected_target_count > 0).to(dtype).mean(),
            'ct_acquisition_eligible_row_count': eligible_row_count,
            'ct_acquisition_retained_row_count': retained_row_count,
            'ct_acquisition_pool_target_sum': pool_target_sum,
            'ct_acquisition_prepool_target_sum': prepool_target_sum,
            'ct_acquisition_sampled_target_sum': selected_target_sum,
            'ct_acquisition_row_recall': acquisition_row_recall,
            'ct_acquisition_point_recall': acquisition_point_recall,
            'ct_recovery_positive_row_count': recovery_positive.sum(),
            'ct_recovery_fallback_row_count': recovery_fallback.sum(),
            # Compatibility alias.  From contract-v3 onward "recall" means
            # retained eligible rows, never a macro point ratio.
            'ct_acquisition_target_recall': acquisition_row_recall,
        }

    def _b0_point_regression_loss(self, prediction, target, data):
        if not self.ct_enable_v27:
            return F.smooth_l1_loss(prediction, target)
        valid = data['b0_valid_mask'].reshape(prediction.shape[:2]).to(prediction.dtype)
        error = F.smooth_l1_loss(prediction, target, reduction='none').mean(-1)
        return (error * valid).sum() / valid.sum().clamp_min(1)

    def compute_loss(self, data, output):
        if self.is_paired_batch(data):
            return self.compute_paired_loss(data, output)

        loss_total = 0.0
        loss_dict = {}

        # Shared by B1 diagnostics and Joint B2/B3 diagnostics.  This helper
        # must not live inside the B1-enabled branch: B2-only deliberately
        # disables B1 while still logging masked Search metrics.
        def masked_mean(per_sample, valid):
            valid = valid.to(
                device=per_sample.device,
                dtype=per_sample.dtype,
            ).reshape(-1)
            return (
                per_sample.reshape(-1) * valid
            ).sum() / torch.clamp(valid.sum(), min=1.0)

        def recursive_balanced_mean(
                per_sample, valid, recursive_age=None,
                recursive_age_valid=None, query_gap=None, query_gaps=None):
            """Reduce B1 losses by equal query-gap and age strata."""
            return recursive_gap_age_balanced_mean(
                per_sample, valid,
                recursive_age=recursive_age,
                recursive_age_valid=recursive_age_valid,
                query_gap=query_gap,
                query_gaps=(
                    query_gaps if query_gaps is not None
                    else self.motion_v3_aux_query_gaps),
            )

        def balanced_binary_loss(logits, targets, valid):
            """Average each observed class independently.

            Returning the mean of the present class means prevents abundant
            negative Search windows from collapsing Presence or alpha.
            """
            errors = F.binary_cross_entropy_with_logits(
                logits, targets, reduction='none')
            valid = valid.to(errors.dtype).reshape(-1)
            targets = targets.to(errors.dtype).reshape(-1)
            positive = valid * (targets > 0.5).to(errors.dtype)
            negative = valid * (targets <= 0.5).to(errors.dtype)
            positive_present = (positive.sum() > 0).to(errors.dtype)
            negative_present = (negative.sum() > 0).to(errors.dtype)
            positive_loss = masked_mean(errors, positive)
            negative_loss = masked_mean(errors, negative)
            return (
                positive_loss * positive_present
                + negative_loss * negative_present
            ) / torch.clamp(
                positive_present + negative_present, min=1.0)

        def binary_rank_metrics(scores, targets, valid):
            """Return exact mini-batch AUROC and average precision."""
            select = valid.detach().reshape(-1) > 0
            selected_scores = scores.detach().reshape(-1)[select]
            selected_targets = targets.detach().reshape(-1)[select] > 0.5
            default_auroc = scores.new_tensor(0.5)
            default_auprc = scores.new_tensor(0.0)
            positive_scores = selected_scores[selected_targets]
            negative_scores = selected_scores[~selected_targets]
            if positive_scores.numel() and negative_scores.numel():
                comparisons = (
                    positive_scores.unsqueeze(1)
                    - negative_scores.unsqueeze(0))
                auroc = (
                    (comparisons > 0).to(scores.dtype)
                    + 0.5 * (comparisons == 0).to(scores.dtype)
                ).mean()
            else:
                auroc = default_auroc
            if positive_scores.numel():
                order = torch.argsort(selected_scores, descending=True)
                sorted_targets = selected_targets[order].to(scores.dtype)
                precision = torch.cumsum(sorted_targets, dim=0) / torch.arange(
                    1, sorted_targets.numel() + 1,
                    device=scores.device, dtype=scores.dtype)
                auprc = (
                    precision * sorted_targets).sum() / torch.clamp(
                        sorted_targets.sum(), min=1.0)
            else:
                auprc = default_auprc
            return auroc, auprc

        final_estimation_boxes = output['aux_estimation_boxes']
        aux_estimation_boxes = output.get(
            'observation_aux_estimation_boxes', final_estimation_boxes)
        motion_pred = output['motion_pred']  
        seg_logits = output['seg_logits'] 
        updated_ref_boxs = output['updated_ref_boxs']
        with torch.no_grad():
            seg_label = data['seg_label'] 
            box_label = data['box_label'] 
            box_label_prev = data['box_label_prev'] 
            motion_label = data['motion_label'] 
            motion_state_label = data['motion_state_label'][:,0] 
            center_label = box_label[:, :3] 
            angle_label = torch.sin(box_label[:, 3]) 
            center_label_prev = box_label_prev[:, :3] 
            angle_label_prev = torch.sin(box_label_prev[:,0,3])
            center_label_motion = motion_label[:,0,:3] 
            angle_label_motion = torch.sin(motion_label[:,0,3]) 

        
            ref_label = data['box_label_prev']
            ref_center_label = ref_label[:, :, :3] #B*hist_num*3
            ref_angle_label = torch.sin(ref_label[:,:,3]) 

        loss_seg = F.cross_entropy(
            seg_logits,
            seg_label,
            weight=seg_logits.new_tensor([0.5, 2.0]),
        )
        if self.ct_enable_v27:
            point_valid = data['b0_valid_mask'].reshape_as(seg_label).to(seg_logits.dtype)
            weights = seg_logits.new_tensor([.5, 2.])
            errors = F.cross_entropy(seg_logits, seg_label, weight=weights, reduction='none')
            loss_seg = (errors * point_valid).sum() / (
                weights[seg_label] * point_valid).sum().clamp_min(1e-6)
        if self.use_motion_cls:
            motion_cls = output['motion_cls']  # B,2
            loss_motion_cls = F.cross_entropy(motion_cls, motion_state_label)
            loss_total += loss_motion_cls * self.config.motion_cls_seg_weight
            loss_dict['loss_motion_cls'] = loss_motion_cls

            loss_center_motion = F.smooth_l1_loss(motion_pred[:, :3], center_label_motion, reduction='none')
            loss_center_motion = (motion_state_label * loss_center_motion.mean(dim=1)).sum() / (
                    motion_state_label.sum() + 1e-6) # Balance within a batch
            loss_angle_motion = F.smooth_l1_loss(torch.sin(motion_pred[:, 3]), angle_label_motion, reduction='none')
            loss_angle_motion = (motion_state_label * loss_angle_motion).sum() / (motion_state_label.sum() + 1e-6)
        else:
            loss_center_motion = F.smooth_l1_loss(motion_pred[:, :3], center_label_motion)
            loss_angle_motion = F.smooth_l1_loss(torch.sin(motion_pred[:, 3]), angle_label_motion)



        # ----- Stage 1 loss ---------------------
        estimation_boxes = output['estimation_boxes']  
        loss_center = F.smooth_l1_loss(estimation_boxes[:, :3], center_label)
        loss_angle = F.smooth_l1_loss(torch.sin(estimation_boxes[:, 3]), angle_label)
        loss_total += 1 * (loss_center * self.config.center_weight + loss_angle * self.config.angle_weight)
        loss_dict["loss_center"] = loss_center
        loss_dict["loss_angle"] = loss_angle
        #-----------------------------------------

        loss_center_aux = F.smooth_l1_loss(aux_estimation_boxes[:, :3], center_label)

        loss_angle_aux = F.smooth_l1_loss(torch.sin(aux_estimation_boxes[:, 3]), angle_label)


        #---------------------refbox loss---------
        loss_center_ref = F.smooth_l1_loss(updated_ref_boxs[:,:,:3],ref_center_label)
        loss_angle_ref = F.smooth_l1_loss(torch.sin(updated_ref_boxs[:, :, 3]), ref_angle_label)
        #---------------------refbox loss---------


        loss_total += loss_seg * self.config.seg_weight \
                      + 1 * (loss_center_aux * self.config.center_weight + loss_angle_aux * self.config.angle_weight) \
                      + 1 * (loss_center_motion * self.config.center_weight + loss_angle_motion * self.config.angle_weight) \
                      + 1 * (loss_center_ref * self.config.ref_center_weight + loss_angle_ref * self.config.ref_angle_weight) 

        loss_dict.update({
            "loss_total": loss_total,
            "loss_seg": loss_seg,
            "loss_center_aux": loss_center_aux,
            "loss_center_motion": loss_center_motion,
            "loss_angle_aux": loss_angle_aux,
            "loss_angle_motion": loss_angle_motion,
            "loss_center_ref": loss_center_ref,
            "loss_angle_ref": loss_angle_ref,
        })
        # Snapshot the observation objective before any B1/B2/B3 term.  The
        # remaining B0-only heads below extend this value explicitly.
        b0_transaction_loss = loss_total
        b1_transaction_loss = loss_total.new_zeros(())
        b2_transaction_loss = loss_total.new_zeros(())
        b3_transaction_loss = loss_total.new_zeros(())
        if self._b0_auxiliary_batch(data):
            # Formal auxiliary views terminate at B0.  Keep the complete
            # observation objective, including the box-aware head, without
            # touching any B1/B2/B3 loss or diagnostic path.
            if self.use_dynamics_encoder or self.use_point_feature_tc:
                raise RuntimeError(
                    "B0 candidate decoupling requires formal CT B0 heads")
            if self.box_aware:
                prev_bc = torch.flatten(
                    data['prev_bc'], start_dim=1, end_dim=2)
                bc_label = torch.cat([prev_bc, data['this_bc']], dim=1)
                loss_bc = self._b0_point_regression_loss(output['pred_bc'], bc_label, data)
                weighted_bc = loss_bc * self.config.bc_weight
                loss_total = loss_total + weighted_bc
                b0_transaction_loss = b0_transaction_loss + weighted_bc
                loss_dict.update({
                    'loss_total': loss_total,
                    'loss_bc': loss_bc,
                })
            loss_dict.update({
                'loss_b0_transaction': b0_transaction_loss,
                'loss_b1_transaction': b1_transaction_loss,
                'loss_b2_transaction': b2_transaction_loss,
                'loss_b3_transaction': b3_transaction_loss,
                'loss_plugin_transaction': loss_total.new_zeros(()),
            })
            return loss_dict
        if self.use_dynamics_encoder and "velocity_pred" in output and "velocity_label" in data:
            velocity_label = (
                data.get("trajectory_velocity_label", data["velocity_label"])
                if self.use_ordered_trajectory_encoder
                else data["velocity_label"]
            )
            velocity_label = velocity_label.to(
                device=output["velocity_pred"].device,
                dtype=output["velocity_pred"].dtype)
            loss_velocity = F.smooth_l1_loss(output["velocity_pred"], velocity_label)
            loss_total += loss_velocity * getattr(self.config, "velocity_weight", 0.05)
            loss_dict.update({
                "loss_total": loss_total,
                "loss_velocity": loss_velocity,
            })

        if self.use_dynamics_encoder and "dynamics_displacement_pred" in output:
            if (self.use_ordered_trajectory_encoder
                    and "trajectory_displacement_label" in data):
                displacement_label = data[
                    "trajectory_displacement_label"][:, :3]
            else:
                displacement_label = data.get(
                    "dynamics_displacement_label", motion_label[:, 0, :3])
            displacement_label = displacement_label.to(
                device=output["dynamics_displacement_pred"].device,
                dtype=output["dynamics_displacement_pred"].dtype,
            )
            loss_dynamics_displacement = F.smooth_l1_loss(
                output["dynamics_displacement_pred"], displacement_label)
            displacement_weight = getattr(self.config, "dynamics_displacement_weight", 0.0)
            if displacement_weight != 0.0:
                loss_total += loss_dynamics_displacement * displacement_weight
            loss_dict.update({
                "loss_total": loss_total,
                "loss_dynamics_displacement": loss_dynamics_displacement,
            })

        if (self.use_ordered_trajectory_encoder
                and "trajectory_displacement_pred" in output
                and "trajectory_displacement_label" in data):
            trajectory_target = data["trajectory_displacement_label"].to(
                device=output["trajectory_displacement_pred"].device,
                dtype=output["trajectory_displacement_pred"].dtype,
            )
            trajectory_prediction = output["trajectory_displacement_pred"]
            trajectory_error = trajectory_prediction - trajectory_target
            trajectory_error = torch.cat((
                trajectory_error[:, :3],
                torch.atan2(
                    torch.sin(trajectory_error[:, 3:4]),
                    torch.cos(trajectory_error[:, 3:4]),
                ),
            ), dim=1)
            log_sigma = torch.clamp(
                output["trajectory_log_sigma"], min=-4.0, max=2.5)
            trajectory_nll_per_sample = (
                0.5 * trajectory_error.pow(2) * torch.exp(-2.0 * log_sigma)
                + log_sigma
            ).mean(dim=1)
            trajectory_valid = output["dynamics_valid"].reshape(-1).to(
                trajectory_nll_per_sample.dtype)
            loss_trajectory_nll = (
                trajectory_nll_per_sample * trajectory_valid
            ).sum() / torch.clamp(trajectory_valid.sum(), min=1.0)
            trajectory_nll_weight = float(getattr(
                self.config, "trajectory_nll_weight", 0.0))
            if trajectory_nll_weight != 0.0:
                loss_total += loss_trajectory_nll * trajectory_nll_weight
            loss_dict.update({
                "loss_total": loss_total,
                "loss_trajectory_nll": loss_trajectory_nll,
                "trajectory_sigma_mean": torch.exp(log_sigma).mean(),
            })

        if "trajectory_adapter_norm" in output:
            loss_trajectory_adapter_norm = (
                output["trajectory_adapter_norm"].pow(2).mean())
            adapter_l2_weight = float(getattr(
                self.config, "trajectory_adapter_l2_weight", 0.0))
            if adapter_l2_weight != 0.0:
                loss_total += (
                    loss_trajectory_adapter_norm * adapter_l2_weight)
            loss_dict.update({
                "loss_total": loss_total,
                "loss_trajectory_adapter_norm":
                    loss_trajectory_adapter_norm,
            })

        if (self.use_b1motion_v3
                and not (self.use_ct_joint_full
                         and not self.ct_enable_b1)):
            main_target = data['motion_main_target_xy'].to(
                device=output['motion_prior_xy'].device,
                dtype=output['motion_prior_xy'].dtype,
            )
            main_prior_per_sample = F.smooth_l1_loss(
                output['motion_prior_xy'],
                main_target,
                reduction='none',
            ).mean(dim=1)
            main_valid = output['motion_prior_valid']
            uncertainty_terms = None
            if (self.use_calibrated_motion_uncertainty
                    or self.use_ct_joint_full):
                uncertainty_terms = physical_motion_uncertainty_loss(
                    output['motion_prior_xy'],
                    main_target,
                    output['motion_prior_log_sigma_parallel_perp'],
                    output['motion_prior_direction_xy'],
                    main_valid,
                    kinematic_xy=output['motion_prior_kinematic_xy'],
                    envelope_parallel_perp=output[
                        'motion_prior_envelope_parallel_perp'],
                    residual_unit_parallel_perp=output[
                        'motion_prior_residual_unit_parallel_perp'],
                    beta=self.motion_v3_beta_nll_beta,
                    tail_direction_weight=(
                        self.motion_v3_tail_direction_weight),
                    tail_direction_margin=(
                        self.motion_v3_tail_direction_margin),
                    sigma_error_cap=self.motion_v3_sigma_error_cap,
                )
                main_prior_per_sample = uncertainty_terms[
                    'mean_per_sample']
                main_valid = uncertainty_terms['valid']
            recursive_age = data.get('ct_recursive_state_age')
            recursive_age_valid = data.get('ct_recursive_state_age_valid')
            if recursive_age is None:
                recursive_age = torch.full_like(main_prior_per_sample, -1.0)
            if recursive_age_valid is None:
                recursive_age_valid = torch.zeros_like(main_prior_per_sample)
            loss_motion_v3_prior = recursive_balanced_mean(
                main_prior_per_sample, main_valid,
                recursive_age=recursive_age,
                recursive_age_valid=recursive_age_valid)
            loss_total += (
                self.motion_v3_prior_weight * loss_motion_v3_prior)
            b1_transaction_loss = (
                b1_transaction_loss
                + self.motion_v3_prior_weight * loss_motion_v3_prior)
            margin_terms = acquisition_margin_pinball_loss(
                output['motion_prior_xy'],
                main_target,
                output['motion_prior_direction_xy'],
                output['motion_prior_acquisition_margin_parallel_perp'],
                main_valid,
                quantile=self.ct_acquisition_margin_quantile,
            )
            if self.ct_enable_v27:
                from models.ct_v2.motion import acquisition_margin_target_loss
                margin_terms = acquisition_margin_target_loss(
                    output['motion_prior_acquisition_margin_parallel_perp'],
                    data['motion_acquisition_target'],
                    main_valid * data['motion_acquisition_target_valid'].reshape_as(main_valid),
                    quantile=self.ct_acquisition_margin_quantile)
            loss_acquisition_margin = recursive_balanced_mean(
                margin_terms['loss_per_sample'], margin_terms['valid'],
                recursive_age=recursive_age,
                recursive_age_valid=recursive_age_valid)
            loss_total += (
                self.ct_acquisition_margin_weight
                * loss_acquisition_margin)
            b1_transaction_loss = (
                b1_transaction_loss
                + self.ct_acquisition_margin_weight
                * loss_acquisition_margin)
            if uncertainty_terms is not None:
                loss_motion_v3_nll = recursive_balanced_mean(
                    uncertainty_terms['nll_per_sample'], main_valid,
                    recursive_age=recursive_age,
                    recursive_age_valid=recursive_age_valid)
                motion_v3_gaussian_nll = masked_mean(
                    uncertainty_terms['gaussian_nll_per_sample'],
                    main_valid)
                loss_total += (
                    self.motion_v3_nll_weight * loss_motion_v3_nll)
                b1_transaction_loss = (
                    b1_transaction_loss
                    + self.motion_v3_nll_weight * loss_motion_v3_nll)

            main_prior_error = torch.linalg.norm(
                output['motion_prior_xy'].detach() - main_target,
                dim=1,
            )
            main_kinematic_error = torch.linalg.norm(
                output['motion_prior_kinematic_xy'].detach() - main_target,
                dim=1,
            )
            loss_dict.update({
                'loss_total': loss_total,
                'loss_motion_v3_prior': loss_motion_v3_prior,
                'loss_ct_acquisition_margin': loss_acquisition_margin,
                'ct_acquisition_margin_parallel_mean': masked_mean(
                    output[
                        'motion_prior_acquisition_margin_parallel_perp'][:, 0],
                    main_valid),
                'ct_acquisition_margin_perpendicular_mean': masked_mean(
                    output[
                        'motion_prior_acquisition_margin_parallel_perp'][:, 1],
                    main_valid),
                'motion_v3_prior_rmse': torch.sqrt(masked_mean(
                    main_prior_error.pow(2), main_valid)),
                'motion_v3_kinematic_rmse': torch.sqrt(masked_mean(
                    main_kinematic_error.pow(2), main_valid)),
                'motion_v3_prior_valid_rate': main_valid.float().mean(),
                'motion_v3_history_valid_ratio': data[
                    'motion_main_valid_mask'].float().mean(),
            })
            if uncertainty_terms is not None:
                normalized_error = (
                    uncertainty_terms['aligned_error']
                    * torch.exp(-output[
                        'motion_prior_log_sigma_parallel_perp']))
                mahalanobis_sq = normalized_error.pow(2).sum(dim=1)
                coverage_levels = (
                    ('50', 1.38629436112, 0.50),
                    ('80', 3.21887582487, 0.80),
                    ('95', 5.99146454711, 0.95),
                )
                coverage_errors = []
                uncertainty_metrics = {
                    'loss_motion_v3_nll': loss_motion_v3_nll,
                    'motion_v3_gaussian_nll': motion_v3_gaussian_nll,
                    'motion_v3_recoverable_saturation_rate': masked_mean(
                        uncertainty_terms[
                            'recoverable_saturation_per_sample'],
                        main_valid),
                    'motion_v3_tail_axis_fraction': masked_mean(
                        uncertainty_terms['tail_axis_fraction_per_sample'],
                        main_valid),
                    'motion_v3_sigma_parallel_mean': masked_mean(
                        torch.exp(output[
                            'motion_prior_log_sigma_parallel_perp'][:, 0]),
                        main_valid),
                    'motion_v3_sigma_perpendicular_mean': masked_mean(
                        torch.exp(output[
                            'motion_prior_log_sigma_parallel_perp'][:, 1]),
                        main_valid),
                }
                for label, threshold, nominal in coverage_levels:
                    empirical = masked_mean(
                        (mahalanobis_sq <= threshold).to(
                            mahalanobis_sq.dtype),
                        main_valid)
                    uncertainty_metrics[
                        f'motion_v3_coverage_{label}'] = empirical
                    coverage_errors.append(torch.abs(
                        empirical - empirical.new_tensor(nominal)))
                uncertainty_metrics['motion_v3_coverage_ece'] = (
                    torch.stack(coverage_errors).mean())
                loss_dict.update(uncertainty_metrics)

            if 'candidate_id' in data:
                candidate_id = data['candidate_id'].to(
                    device=main_valid.device).reshape(-1)
                candidate_masks = {
                    'candidate0': candidate_id == 0,
                    'candidate_nonzero': candidate_id != 0,
                }
                for bucket_name, bucket_mask in candidate_masks.items():
                    bucket_valid = main_valid * bucket_mask.to(
                        main_valid.dtype)
                    loss_dict.update({
                        f'motion_v3_prior_rmse_{bucket_name}': torch.sqrt(
                            masked_mean(
                                main_prior_error.pow(2), bucket_valid)),
                        f'motion_v3_kinematic_rmse_{bucket_name}': torch.sqrt(
                            masked_mean(
                                main_kinematic_error.pow(2), bucket_valid)),
                        f'motion_v3_count_{bucket_name}':
                            bucket_valid.sum(),
                    })

            if ('motion_aux_prior_xy' in output
                    and 'motion_aux_target_xy' in data):
                aux_target = data['motion_aux_target_xy'].to(
                    device=output['motion_aux_prior_xy'].device,
                    dtype=output['motion_aux_prior_xy'].dtype,
                )
                aux_prior_per_sample = F.smooth_l1_loss(
                    output['motion_aux_prior_xy'],
                    aux_target,
                    reduction='none',
                ).mean(dim=1)
                aux_valid = output['motion_aux_prior_valid']
                aux_uncertainty_terms = None
                if (self.use_calibrated_motion_uncertainty
                        or self.use_ct_joint_full):
                    aux_uncertainty_terms = physical_motion_uncertainty_loss(
                        output['motion_aux_prior_xy'],
                        aux_target,
                        output[
                            'motion_aux_prior_log_sigma_parallel_perp'],
                        output['motion_aux_prior_direction_xy'],
                        aux_valid,
                        kinematic_xy=output[
                            'motion_aux_prior_kinematic_xy'],
                        envelope_parallel_perp=output[
                            'motion_aux_prior_envelope_parallel_perp'],
                        residual_unit_parallel_perp=output[
                            'motion_aux_prior_residual_unit_parallel_perp'],
                        beta=self.motion_v3_beta_nll_beta,
                        tail_direction_weight=(
                            self.motion_v3_tail_direction_weight),
                        tail_direction_margin=(
                            self.motion_v3_tail_direction_margin),
                        sigma_error_cap=self.motion_v3_sigma_error_cap,
                    )
                    aux_prior_per_sample = aux_uncertainty_terms[
                        'mean_per_sample']
                    aux_valid = aux_uncertainty_terms['valid']
                aux_query_gap = data.get('motion_aux_query_gap_frames')
                loss_motion_v3_aux_prior = recursive_balanced_mean(
                    aux_prior_per_sample, aux_valid,
                    recursive_age=recursive_age,
                    recursive_age_valid=recursive_age_valid,
                    query_gap=aux_query_gap,
                    query_gaps=self.motion_v3_aux_query_gaps)
                loss_total += (
                    self.motion_v3_aux_prior_weight
                    * loss_motion_v3_aux_prior)
                b1_transaction_loss = (
                    b1_transaction_loss
                    + self.motion_v3_aux_prior_weight
                    * loss_motion_v3_aux_prior)
                if aux_uncertainty_terms is not None:
                    loss_motion_v3_aux_nll = recursive_balanced_mean(
                        aux_uncertainty_terms['nll_per_sample'], aux_valid,
                        recursive_age=recursive_age,
                        recursive_age_valid=recursive_age_valid,
                        query_gap=aux_query_gap,
                        query_gaps=self.motion_v3_aux_query_gaps)
                    motion_v3_aux_gaussian_nll = masked_mean(
                        aux_uncertainty_terms[
                            'gaussian_nll_per_sample'], aux_valid)
                    loss_total += (
                        self.motion_v3_aux_nll_weight
                        * loss_motion_v3_aux_nll)
                    b1_transaction_loss = (
                        b1_transaction_loss
                        + self.motion_v3_aux_nll_weight
                        * loss_motion_v3_aux_nll)
                aux_prior_error = torch.linalg.norm(
                    output['motion_aux_prior_xy'].detach() - aux_target,
                    dim=1,
                )
                aux_kinematic_error = torch.linalg.norm(
                    output['motion_aux_prior_kinematic_xy'].detach()
                    - aux_target,
                    dim=1,
                )
                loss_dict.update({
                    'loss_total': loss_total,
                    'loss_motion_v3_aux_prior':
                        loss_motion_v3_aux_prior,
                    'motion_v3_aux_prior_rmse': torch.sqrt(masked_mean(
                        aux_prior_error.pow(2), aux_valid)),
                    'motion_v3_aux_kinematic_rmse': torch.sqrt(masked_mean(
                        aux_kinematic_error.pow(2), aux_valid)),
                    'motion_v3_aux_gap_ratio': masked_mean(
                        output['motion_aux_prior_gap_ratio'], aux_valid),
                    'motion_v3_aux_history_valid_ratio': data[
                        'motion_aux_valid_mask'].float().mean(),
                })
                if aux_uncertainty_terms is not None:
                    loss_dict['loss_motion_v3_aux_nll'] = (
                        loss_motion_v3_aux_nll)
                    loss_dict['motion_v3_aux_gaussian_nll'] = (
                        motion_v3_aux_gaussian_nll)
                    loss_dict[
                        'motion_v3_aux_recoverable_saturation_rate'] = (
                            masked_mean(aux_uncertainty_terms[
                                'recoverable_saturation_per_sample'],
                                aux_valid))
                    loss_dict['motion_v3_aux_tail_axis_fraction'] = (
                        masked_mean(aux_uncertainty_terms[
                            'tail_axis_fraction_per_sample'], aux_valid))
                if 'motion_aux_query_gap_frames' in data:
                    aux_query_gap = data[
                        'motion_aux_query_gap_frames'].to(
                            device=aux_valid.device).reshape(-1)
                    for query_gap in self.motion_v3_aux_query_gaps:
                        gap_valid = aux_valid * (
                            aux_query_gap == query_gap).to(aux_valid.dtype)
                        loss_dict.update({
                            f'motion_v3_aux_prior_rmse_gap{query_gap}':
                                torch.sqrt(masked_mean(
                                    aux_prior_error.pow(2), gap_valid)),
                            f'motion_v3_aux_kinematic_rmse_gap{query_gap}':
                                torch.sqrt(masked_mean(
                                    aux_kinematic_error.pow(2), gap_valid)),
                            f'motion_v3_aux_count_gap{query_gap}':
                                gap_valid.sum(),
                        })

            if (self.use_motion_v3_legacy_fusion
                    and int(getattr(
                        self, 'current_epoch', 0))
                    >= self.motion_v3_warmup_epoch):
                final_xy = final_estimation_boxes[:, :2]
                target_xy = center_label[:, :2]
                loss_motion_v3_fused = F.smooth_l1_loss(
                    final_xy, target_xy)

                observation_error = torch.linalg.norm(
                    aux_estimation_boxes[:, :2].detach() - target_xy,
                    dim=1,
                )
                prior_error = torch.linalg.norm(
                    output['motion_prior_proposal_xy'].detach() - target_xy,
                    dim=1,
                )
                helpful = (
                    prior_error + self.motion_v3_help_margin
                    < observation_error)
                unhelpful = (
                    observation_error + self.motion_v3_help_margin
                    < prior_error)
                decisive = (helpful | unhelpful) & (main_valid > 0)
                helpful_target = helpful.to(final_xy.dtype)
                gate_probability = output['motion_gate_probability']
                gate_bce = F.binary_cross_entropy(
                    gate_probability,
                    helpful_target,
                    reduction='none',
                )
                positive = decisive & helpful
                negative = decisive & unhelpful
                positive_count = positive.to(final_xy.dtype).sum()
                negative_count = negative.to(final_xy.dtype).sum()
                positive_loss = (
                    gate_bce * positive.to(final_xy.dtype)
                ).sum() / torch.clamp(positive_count, min=1.0)
                negative_loss = (
                    gate_bce * negative.to(final_xy.dtype)
                ).sum() / torch.clamp(negative_count, min=1.0)
                present_classes = (
                    (positive_count > 0).to(final_xy.dtype)
                    + (negative_count > 0).to(final_xy.dtype))
                loss_motion_v3_gate = (
                    positive_loss * (positive_count > 0).to(final_xy.dtype)
                    + negative_loss * (negative_count > 0).to(final_xy.dtype)
                ) / torch.clamp(present_classes, min=1.0)
                loss_total += (
                    self.motion_v3_fused_weight * loss_motion_v3_fused
                    + self.motion_v3_gate_weight * loss_motion_v3_gate)

                predicted_helpful = (
                    gate_probability >= 0.5) & decisive
                true_positive = (
                    predicted_helpful & helpful).to(final_xy.dtype).sum()
                predicted_positive = predicted_helpful.to(
                    final_xy.dtype).sum()
                decisive_count = decisive.to(final_xy.dtype).sum()
                loss_dict.update({
                    'loss_total': loss_total,
                    'loss_motion_v3_fused': loss_motion_v3_fused,
                    'loss_motion_v3_gate': loss_motion_v3_gate,
                    'motion_v3_helpful_rate': (
                        positive.to(final_xy.dtype).sum()
                        / torch.clamp(decisive_count, min=1.0)),
                    'motion_v3_gate_precision': (
                        true_positive
                        / torch.clamp(predicted_positive, min=1.0)),
                    'motion_v3_gate_applied_rate': (
                        (gate_probability >= 0.5) & decisive
                    ).to(final_xy.dtype).sum() / torch.clamp(
                        decisive_count, min=1.0),
                    'motion_v3_gate_alpha_utilization': masked_mean(
                        output['motion_gate_alpha'] / max(
                            self.motion_v3_fusion.max_alpha, 1e-6),
                        main_valid),
                    'motion_v3_correction_norm': masked_mean(
                        torch.linalg.norm(
                            output['motion_correction_xy'], dim=1),
                        main_valid),
                    'motion_v3_clip_rate': masked_mean(
                        output['motion_fusion_clip_rate'], main_valid),
                    'motion_v3_observation_error': masked_mean(
                        observation_error, main_valid),
                    'motion_v3_prior_box_error': masked_mean(
                        prior_error, main_valid),
                    'motion_v3_final_error': masked_mean(
                        torch.linalg.norm(
                            final_xy.detach() - target_xy, dim=1),
                        main_valid),
                })

        # Shared B2 dispatch marker retained for source-contract audits used by
        # the CT21/CT22 regression suite.
        if self.use_ct_joint_full and self.ct_enable_b2:
            pass

        if (self.use_ct_joint_full and self.ct_enable_b2
                and self.ct_joint_contract_version >= 3):
            target_xy_v3 = center_label[:, :2].to(
                device=output['ct_b2_raw_box'].device,
                dtype=output['ct_b2_raw_box'].dtype)
            v3_loss = self._compute_ct_contract_v3_loss(
                data, output, target_xy_v3)
            loss_total = loss_total + v3_loss['loss_ct_plugin_total']
            b2_transaction_loss = v3_loss['loss_ct_b2_total']
            b3_transaction_loss = v3_loss['loss_ct_b3_total']
            loss_dict.update(v3_loss)
            loss_dict['loss_total'] = loss_total

        if (self.use_ct_joint_full and self.ct_enable_b2
                and self.ct_joint_contract_version < 3):
            target_xy = center_label[:, :2].to(
                device=output['ct_search_raw_xy'].device,
                dtype=output['ct_search_raw_xy'].dtype)
            observation_xy_all = aux_estimation_boxes[:, :2].detach()
            raw_xy_all = output['ct_search_unmasked_raw_xy'].detach()
            bounded_xy_all = (
                observation_xy_all
                + output['ct_router_bounded_residual_xy'].detach())
            final_xy_all = output['ct_final_box'][:, :2].detach()
            observation_error_all = torch.linalg.norm(
                observation_xy_all - target_xy, dim=1)
            raw_error_all = torch.linalg.norm(
                raw_xy_all - target_xy, dim=1)
            bounded_error_all = torch.linalg.norm(
                bounded_xy_all - target_xy, dim=1)
            final_error_all = torch.linalg.norm(
                final_xy_all - target_xy, dim=1)
            endpoint_labels = data['search_v3_point_labels'].to(
                device=target_xy.device, dtype=target_xy.dtype)
            tube_labels = data['trajectory_search_point_labels'].to(
                device=target_xy.device, dtype=target_xy.dtype)
            point_labels = torch.cat((endpoint_labels, tube_labels), dim=1)
            endpoint_valid = data['search_v3_point_valid_mask'].to(
                device=target_xy.device, dtype=target_xy.dtype)
            tube_valid = data['trajectory_search_point_valid_mask'].to(
                device=target_xy.device, dtype=target_xy.dtype).reshape(
                    -1, self.ct_tube_quota)
            point_valid = torch.cat((endpoint_valid, tube_valid), dim=1)
            endpoint_source = data['search_v3_point_source'].to(
                device=target_xy.device, dtype=target_xy.dtype)
            tube_source = data['trajectory_search_point_source'].to(
                device=target_xy.device, dtype=target_xy.dtype).reshape(
                    -1, self.ct_tube_quota)
            expansion_source = torch.cat((
                endpoint_source, tube_source), dim=1)
            search_support_valid = output[
                'ct_search_support_valid'].detach().to(target_xy.dtype)
            point_valid = point_valid * search_support_valid.unsqueeze(1)
            point_labels = point_labels * point_valid

            def targetness_focal_loss(logits):
                bce = F.binary_cross_entropy_with_logits(
                    logits, point_labels, reduction='none')
                probability = torch.sigmoid(logits)
                p_t = (
                    point_labels * probability
                    + (1.0 - point_labels) * (1.0 - probability))
                alpha_t = (
                    point_labels * self.ct_focal_alpha
                    + (1.0 - point_labels) * (1.0 - self.ct_focal_alpha))
                focal = (
                    alpha_t * (1.0 - p_t).pow(self.ct_focal_gamma) * bce)
                return (
                    focal * point_valid).sum() / torch.clamp(
                        point_valid.sum(), min=1.0)

            counterfactual_arms_enabled = bool(
                self.ct_joint_contract_version >= 2
                and self.ct_query_counterfactual_supervision
                and self.ct_enable_b1)
            if counterfactual_arms_enabled:
                # Equal auxiliary supervision prevents either counterfactual
                # arm from becoming a permanently untrained baseline.
                loss_ct_targetness_obs = targetness_focal_loss(
                    output['ct_search_targetness_logits_obs'])
                loss_ct_targetness_motion = targetness_focal_loss(
                    output['ct_search_targetness_logits_motion'])
                loss_ct_targetness = 0.5 * (
                    loss_ct_targetness_obs + loss_ct_targetness_motion)
            else:
                loss_ct_targetness_obs = targetness_focal_loss(
                    output['ct_search_targetness_logits'])
                loss_ct_targetness_motion = loss_ct_targetness_obs
                loss_ct_targetness = loss_ct_targetness_obs

            foreground = point_labels * point_valid
            foreground_count = foreground.sum(dim=1)
            extension_foreground_count = (
                foreground * (expansion_source > 0).to(
                    target_xy.dtype)).sum(dim=1)
            presence_target = (
                extension_foreground_count >= 1).to(target_xy.dtype)
            if (self.ct_joint_contract_version >= 2
                    and self.ct_presence_balanced_loss):
                loss_ct_presence = balanced_binary_loss(
                    output['ct_search_presence_logit'], presence_target,
                    search_support_valid)
            else:
                presence_error = F.binary_cross_entropy_with_logits(
                    output['ct_search_presence_logit'], presence_target,
                    reduction='none')
                loss_ct_presence = (
                    presence_error * search_support_valid).sum()
                loss_ct_presence = loss_ct_presence / torch.clamp(
                    search_support_valid.sum(), min=1.0)
            presence_auroc, presence_auprc = binary_rank_metrics(
                output['ct_search_presence_probability'], presence_target,
                search_support_valid)
            vote_error = F.smooth_l1_loss(
                output['ct_search_point_votes'],
                target_xy.unsqueeze(1).expand_as(
                    output['ct_search_point_votes']),
                reduction='none').mean(dim=2)
            loss_ct_vote = (
                vote_error * foreground).sum() / torch.clamp(
                    foreground.sum(), min=1.0)
            proposal_valid = (
                (foreground_count >= 1).to(target_xy.dtype)
                * output['ct_search_candidate_valid'].detach())
            if counterfactual_arms_enabled:
                raw_obs_per_sample = F.smooth_l1_loss(
                    output['ct_search_raw_obs_xy'], target_xy,
                    reduction='none').mean(dim=1)
                raw_motion_per_sample = F.smooth_l1_loss(
                    output['ct_search_raw_motion_xy'], target_xy,
                    reduction='none').mean(dim=1)
                raw_per_sample = 0.5 * (
                    raw_obs_per_sample + raw_motion_per_sample)
            else:
                raw_obs_per_sample = F.smooth_l1_loss(
                    output['ct_search_unmasked_raw_xy'], target_xy,
                    reduction='none').mean(dim=1)
                raw_motion_per_sample = raw_obs_per_sample
                raw_per_sample = raw_obs_per_sample
            loss_ct_raw_search = (
                raw_per_sample * proposal_valid).sum() / torch.clamp(
                    proposal_valid.sum(), min=1.0)

            loss_ct_query_gate = target_xy.new_zeros(())
            query_target = target_xy.new_zeros((target_xy.shape[0],))
            query_valid = target_xy.new_zeros((target_xy.shape[0],))
            query_auroc = target_xy.new_tensor(0.5)
            query_auprc = target_xy.new_tensor(0.0)
            counterfactual_obs_error = target_xy.new_zeros(
                (target_xy.shape[0],))
            counterfactual_motion_error = target_xy.new_zeros(
                (target_xy.shape[0],))
            counterfactual_helpful = torch.zeros(
                target_xy.shape[0], dtype=torch.bool,
                device=target_xy.device)
            counterfactual_harmful = torch.zeros_like(
                counterfactual_helpful)
            counterfactual_ambiguous = torch.zeros_like(
                counterfactual_helpful)
            counterfactual_base_valid = target_xy.new_zeros(
                (target_xy.shape[0],))
            if (self.ct_enable_b1
                    and self.ct_enable_query_reliability_gate):
                if (self.ct_joint_contract_version >= 2
                        and self.ct_query_counterfactual_supervision):
                    counterfactual = counterfactual_query_targets(
                        output['ct_search_raw_obs_xy'],
                        output['ct_search_raw_motion_xy'], target_xy,
                        margin=self.ct_query_counterfactual_margin)
                    counterfactual_obs_error = counterfactual['obs_error']
                    counterfactual_motion_error = counterfactual[
                        'motion_error']
                    counterfactual_helpful = counterfactual['helpful']
                    counterfactual_harmful = counterfactual['harmful']
                    counterfactual_ambiguous = counterfactual['ambiguous']
                    query_target = counterfactual['target']
                else:
                    observation_error = torch.linalg.norm(
                        aux_estimation_boxes[:, :2].detach() - target_xy,
                        dim=1)
                    raw_query_error = torch.linalg.norm(
                        output['ct_search_unmasked_raw_xy'].detach()
                        - target_xy, dim=1)
                    query_target = (
                        raw_query_error < observation_error).to(
                            target_xy.dtype)
                query_gate_error = F.binary_cross_entropy_with_logits(
                    output['ct_query_gate_logit'], query_target,
                    reduction='none')
                query_valid = (
                    output['motion_prior_valid'].detach()
                    * search_support_valid
                    * output['ct_search_candidate_valid'].detach())
                if (self.ct_joint_contract_version >= 2
                        and self.ct_query_counterfactual_supervision):
                    counterfactual_base_valid = query_valid
                    query_valid = query_valid * counterfactual['valid']
                    loss_ct_query_gate = balanced_binary_loss(
                        output['ct_query_gate_logit'], query_target,
                        query_valid)
                else:
                    loss_ct_query_gate = (
                        query_gate_error * query_valid).sum() / torch.clamp(
                            query_valid.sum(), min=1.0)
                query_auroc, query_auprc = binary_rank_metrics(
                    output['ct_query_gate_probability'], query_target,
                    query_valid)

            loss_ct_router = target_xy.new_zeros(())
            loss_ct_correction = target_xy.new_zeros(())
            h1_gain = target_xy.new_zeros((target_xy.shape[0],))
            h1_target = target_xy.new_zeros((target_xy.shape[0],))
            h3_gain = target_xy.new_zeros((target_xy.shape[0],))
            h3_target = target_xy.new_zeros((target_xy.shape[0],))
            h3_valid = target_xy.new_zeros((target_xy.shape[0],))
            if self.ct_enable_b3:
                observation_xy = aux_estimation_boxes[:, :2].detach()
                observation_error = torch.linalg.norm(
                    observation_xy - target_xy, dim=1)
                bounded_search_xy = (
                    observation_xy
                    + output['ct_router_bounded_residual_xy'].detach())
                bounded_search_error = torch.linalg.norm(
                    bounded_search_xy - target_xy, dim=1)
                h1_gain = observation_error - bounded_search_error
                h1_target = (
                    h1_gain > self.ct_router_help_margin).to(target_xy.dtype)
                h1_error = F.binary_cross_entropy_with_logits(
                    output['ct_router_logit'], h1_target,
                    reduction='none')
                router_valid = output[
                    'ct_search_candidate_valid'].detach()
                if 'ct_h3_gain' in data and 'ct_h3_valid' in data:
                    h3_gain = data['ct_h3_gain'].to(
                        device=target_xy.device,
                        dtype=target_xy.dtype).reshape(-1).detach()
                    h3_valid = data['ct_h3_valid'].to(
                        device=target_xy.device,
                        dtype=target_xy.dtype).reshape(-1).detach()
                    h3_valid = h3_valid * router_valid
                    h3_target = (
                        h3_gain > self.ct_router_h3_margin).to(
                            target_xy.dtype)
                    h3_error = F.binary_cross_entropy_with_logits(
                        output['ct_router_logit'], h3_target,
                        reduction='none')
                    router_error = torch.where(
                        h3_valid > 0,
                        0.25 * h1_error + 0.75 * h3_error,
                        h1_error)
                else:
                    router_error = h1_error
                loss_ct_router = (
                    router_error * router_valid).sum() / torch.clamp(
                        router_valid.sum(), min=1.0)
                correction_target = torch.where(
                    (h3_valid > 0).unsqueeze(1),
                    h3_target.unsqueeze(1), h1_target.unsqueeze(1))
                target_action_xy = (
                    observation_xy
                    + correction_target
                    * output['ct_router_bounded_residual_xy'].detach())
                correction_per_sample = F.smooth_l1_loss(
                    output['ct_router_soft_box'][:, :2],
                    target_action_xy.detach(),
                    reduction='none').mean(dim=1)
                loss_ct_correction = (
                    correction_per_sample * router_valid).sum(
                    ) / torch.clamp(router_valid.sum(), min=1.0)

            loss_total += (
                self.ct_targetness_weight * loss_ct_targetness
                + self.ct_vote_weight * loss_ct_vote
                + self.ct_raw_search_weight * loss_ct_raw_search
                + self.ct_presence_weight * loss_ct_presence
                + self.ct_query_gate_weight * loss_ct_query_gate
                + self.ct_router_weight * loss_ct_router
                + self.ct_correction_weight * loss_ct_correction)
            loss_dict.update({
                'loss_total': loss_total,
                'loss_ct_targetness': loss_ct_targetness,
                'loss_ct_targetness_obs': loss_ct_targetness_obs,
                'loss_ct_targetness_motion': loss_ct_targetness_motion,
                'loss_ct_vote': loss_ct_vote,
                'loss_ct_raw_search': loss_ct_raw_search,
                'loss_ct_presence': loss_ct_presence,
                'loss_ct_query_gate': loss_ct_query_gate,
                'loss_ct_router': loss_ct_router,
                'loss_ct_correction': loss_ct_correction,
                'ct_correction_ramp': target_xy.new_tensor(1.0),
                'ct_foreground_points': foreground_count.mean(),
                'ct_candidate_valid_rate': output[
                    'ct_search_candidate_valid'].float().mean(),
                'ct_structural_valid_rate': output[
                    'ct_search_structural_valid'].float().mean(),
                'ct_new_support_valid_rate': output[
                    'ct_search_new_support_valid'].float().mean(),
                'ct_search_effective_rate': output[
                    'ct_search_effective'].float().mean(),
                'ct_evidence_valid_rate': output[
                    'ct_router_evidence_valid'].float().mean(),
                'ct_extension_targetness_mass_ratio': output[
                    'ct_search_extension_mass_ratio'].mean(),
                'ct_extension_vote_rms': torch.nan_to_num(output[
                    'ct_search_extension_vote_rms'].detach(),
                    nan=0.0, posinf=0.0, neginf=0.0).mean(),
                'ct_search_presence_probability': output[
                    'ct_search_presence_probability'].mean(),
                'ct_presence_positive_count': (
                    search_support_valid * presence_target).sum(),
                'ct_presence_negative_count': (
                    search_support_valid * (1.0 - presence_target)).sum(),
                'ct_presence_positive_mean': masked_mean(
                    output['ct_search_presence_probability'],
                    search_support_valid * presence_target),
                'ct_presence_negative_mean': masked_mean(
                    output['ct_search_presence_probability'],
                    search_support_valid * (1.0 - presence_target)),
                'ct_presence_auroc': presence_auroc,
                'ct_presence_auprc': presence_auprc,
                'ct_search_support_valid_rate': search_support_valid.mean(),
                'ct_search_geometry_valid_rate': output[
                    'ct_search_geometry_valid'].float().mean(),
                'ct_b1_geometry_source_rate': masked_mean(
                    (output['ct_b1_geometry_source_id'] == 1).to(
                        target_xy.dtype),
                    data['ct_search_history_valid'].float()),
                'ct_fallback_geometry_source_rate': masked_mean(
                    (output['ct_b1_geometry_source_id'] == 2).to(
                        target_xy.dtype),
                    data['ct_search_history_valid'].float()),
                'ct_search_used_rate': output[
                    'ct_router_applied_gate'].mean(),
                'ct_search_inactive_reason/history': (
                    1.0 - data['ct_search_history_valid'].float()).mean(),
                'ct_search_inactive_reason/time': (
                    1.0 - data['ct_search_time_valid'].float()).mean(),
                'ct_search_inactive_reason/proposal': (
                    1.0 - data['ct_search_proposal_valid'].float()).mean(),
                'ct_search_inactive_reason/coverage': (
                    1.0 - data['ct_search_coverage_need'].float()).mean(),
                'ct_search_inactive_reason/point_support': (
                    1.0 - data[
                        'ct_search_point_support_valid'].float()).mean(),
                'ct_search_extension_count': data[
                    'ct_search_extension_count'].float().mean(),
                'ct_search_extension_voxels': data[
                    'ct_search_extension_voxels'].float().mean(),
                'ct_raw_vs_obs_error_gain': masked_mean(
                    observation_error_all - raw_error_all,
                    output['ct_search_effective'].detach()),
                'ct_raw_better_than_observation_rate': masked_mean(
                    (raw_error_all < observation_error_all).to(
                        target_xy.dtype),
                    output['ct_search_effective'].detach()),
                'ct_query_gate_mean': output[
                    'ct_query_gate_probability'].mean(),
                'ct_query_gate_std': output['ct_query_gate_probability'].std(
                    unbiased=False),
                'ct_query_gate_applied_mean': output[
                    'ct_query_gate_internal'].mean(),
                'ct_query_gate_auroc': query_auroc,
                'ct_query_gate_auprc': query_auprc,
                'ct_query_gate_positive_mean': masked_mean(
                    output['ct_query_gate_probability'],
                    query_valid * query_target),
                'ct_query_gate_negative_mean': masked_mean(
                    output['ct_query_gate_probability'],
                    query_valid * (1.0 - query_target)),
                'ct_alpha_positive_mean': masked_mean(
                    output['ct_query_gate_probability'],
                    query_valid * query_target),
                'ct_alpha_negative_mean': masked_mean(
                    output['ct_query_gate_probability'],
                    query_valid * (1.0 - query_target)),
                'ct_alpha_counterfactual_helpful_rate': masked_mean(
                    counterfactual_helpful.to(target_xy.dtype),
                    counterfactual_base_valid),
                'ct_alpha_counterfactual_harmful_rate': masked_mean(
                    counterfactual_harmful.to(target_xy.dtype),
                    counterfactual_base_valid),
                'ct_alpha_counterfactual_ambiguous_rate': masked_mean(
                    counterfactual_ambiguous.to(target_xy.dtype),
                    counterfactual_base_valid),
                'ct_alpha_counterfactual_valid_rate': query_valid.mean(),
                'ct_alpha_raw_obs_error': masked_mean(
                    counterfactual_obs_error,
                    counterfactual_base_valid),
                'ct_alpha_raw_motion_error': masked_mean(
                    counterfactual_motion_error,
                    counterfactual_base_valid),
                'ct_alpha_counterfactual_uplift': masked_mean(
                    counterfactual_obs_error - counterfactual_motion_error,
                    counterfactual_base_valid),
                'ct_query_shift_norm': output[
                    'ct_query_shift_norm'].mean(),
                'ct_motion_residual_saturation': output[
                    'ct_motion_residual_saturation'].mean(),
                'ct_router_gate_mean': output['ct_router_gate'].mean(),
                'ct_router_positive_rate': (
                    ((output['ct_router_gate'] >= 0.5).to(target_xy.dtype)
                     * output['ct_search_candidate_valid'].detach()).sum()
                    / torch.clamp(output[
                        'ct_search_candidate_valid'].detach().sum(), min=1.0)),
                'ct_search_applied_rate': output[
                    'ct_router_applied_gate'].mean(),
                'ct_h1_helpful_rate': (
                    (h1_gain > self.ct_router_help_margin).to(
                        target_xy.dtype)
                    * output['ct_search_candidate_valid'].detach()).sum()
                    / torch.clamp(output[
                        'ct_search_candidate_valid'].detach().sum(), min=1.0),
                'ct_h1_harmful_rate': (
                    (h1_gain < -self.ct_router_help_margin).to(
                        target_xy.dtype)
                    * output['ct_search_candidate_valid'].detach()).sum()
                    / torch.clamp(output[
                        'ct_search_candidate_valid'].detach().sum(), min=1.0),
                'ct_h3_helpful_rate': (
                    (h3_gain > self.ct_router_h3_margin).to(target_xy.dtype)
                    * h3_valid).sum() / torch.clamp(h3_valid.sum(), min=1.0),
                'ct_h3_harmful_rate': (
                    (h3_gain < -self.ct_router_h3_margin).to(target_xy.dtype)
                    * h3_valid).sum() / torch.clamp(h3_valid.sum(), min=1.0),
                'ct_h1_h3_conflict_rate': (
                    (h1_target != h3_target).to(target_xy.dtype)
                    * h3_valid).sum() / torch.clamp(h3_valid.sum(), min=1.0),
                'ct_h1_positive_h3_negative_rate': (
                    ((h1_target > 0) & (h3_target <= 0)).to(target_xy.dtype)
                    * h3_valid).sum() / torch.clamp(h3_valid.sum(), min=1.0),
                'ct_h1_negative_h3_positive_rate': (
                    ((h1_target <= 0) & (h3_target > 0)).to(target_xy.dtype)
                    * h3_valid).sum() / torch.clamp(h3_valid.sum(), min=1.0),
                'ct_h3_gain_when_applied': masked_mean(
                    h3_gain, h3_valid * output[
                        'ct_router_applied_gate'].detach()),
                'ct_observation_error': masked_mean(
                    observation_error_all,
                    output['ct_search_candidate_valid'].detach()),
                'ct_raw_error': masked_mean(
                    raw_error_all,
                    output['ct_search_candidate_valid'].detach()),
                'ct_bounded_error': masked_mean(
                    bounded_error_all,
                    output['ct_search_candidate_valid'].detach()),
                'ct_final_error': final_error_all.mean(),
                'ct_raw_search_rmse': raw_error_all.mean(),
                'ct_observation_rmse': observation_error_all.mean(),
                'ct_final_rmse': final_error_all.mean(),
            })
            for bin_index in range(5):
                lower = bin_index / 5.0
                upper = (bin_index + 1) / 5.0
                presence_in_bin = (
                    (search_support_valid > 0)
                    & (output[
                        'ct_search_presence_probability'].detach() >= lower)
                    & (output[
                        'ct_search_presence_probability'].detach() < upper
                       if bin_index < 4
                       else output[
                           'ct_search_presence_probability'].detach()
                           <= upper))
                presence_bin_mask = presence_in_bin.to(target_xy.dtype)
                loss_dict[
                    f'ct_presence_bin{bin_index}_count'] = (
                        presence_bin_mask.sum())
                loss_dict[
                    f'ct_presence_bin{bin_index}_confidence'] = masked_mean(
                        output['ct_search_presence_probability'],
                        presence_bin_mask)
                loss_dict[
                    f'ct_presence_bin{bin_index}_positive_rate'] = masked_mean(
                        presence_target, presence_bin_mask)

            for bin_index in range(5):
                lower = bin_index / 5.0
                upper = (bin_index + 1) / 5.0
                in_bin = (
                    (query_valid > 0)
                    & (output['ct_query_gate_probability'].detach() >= lower)
                    & (output['ct_query_gate_probability'].detach() < upper
                       if bin_index < 4
                       else output[
                           'ct_query_gate_probability'].detach() <= upper))
                bin_mask = in_bin.to(target_xy.dtype)
                loss_dict[
                    f'ct_query_reliability_bin{bin_index}_count'] = (
                        bin_mask.sum())
                loss_dict[
                    f'ct_query_reliability_bin{bin_index}_confidence'] = (
                        masked_mean(
                            output['ct_query_gate_probability'], bin_mask))
                loss_dict[
                    f'ct_query_reliability_bin{bin_index}_helpful_rate'] = (
                        masked_mean(query_target, bin_mask))
            for key in (
                    'ct_shadow_forward_count', 'ct_shadow_time_ms',
                    'ct_shadow_peak_memory_mb'):
                if key in data:
                    loss_dict[key] = data[key].float().mean()
            for key in (
                    'ct_recursive_state_age',
                    'ct_recursive_state_age_valid',
                    'ct_recursive_rollout_horizon',
                    'ct_recursive_reset_boundary',
                    'ct_recursive_state_source',
                    'ct_recursive_pre_reset_anchor_error',
                    'ct_recursive_anchor_error',
                    'ct_crop_target_points',
                    'ct_candidate_state_consistency'):
                if key in data:
                    loss_dict[key] = data[key].float().mean()

        if self.use_search_evidence_v2:
            target_xy = center_label[:, :2].to(
                device=output['search_proposal_xy'].device,
                dtype=output['search_proposal_xy'].dtype)
            point_labels = data['search_v2_point_labels'].to(
                device=output['search_targetness_logits'].device,
                dtype=output['search_targetness_logits'].dtype)
            point_valid = data['search_v2_point_valid_mask'].to(
                device=point_labels.device, dtype=point_labels.dtype)
            targetness_logits = output['search_targetness_logits']
            bce = F.binary_cross_entropy_with_logits(
                targetness_logits, point_labels, reduction='none')
            probability = torch.sigmoid(targetness_logits)
            p_t = (
                point_labels * probability
                + (1.0 - point_labels) * (1.0 - probability))
            alpha_t = (
                point_labels * self.search_v2_focal_alpha
                + (1.0 - point_labels)
                * (1.0 - self.search_v2_focal_alpha))
            focal = alpha_t * (1.0 - p_t).pow(
                self.search_v2_focal_gamma) * bce
            loss_search_targetness = (
                focal * point_valid).sum() / torch.clamp(
                    point_valid.sum(), min=1.0)

            foreground = point_labels * point_valid
            foreground_count = foreground.sum(dim=1)
            vote_error = F.smooth_l1_loss(
                output['search_point_center_votes'],
                target_xy.unsqueeze(1).expand_as(
                    output['search_point_center_votes']),
                reduction='none').mean(dim=2)
            loss_search_vote = (
                vote_error * foreground).sum() / torch.clamp(
                    foreground.sum(), min=1.0)

            proposal_valid = (foreground_count >= 1).to(point_labels.dtype)
            proposal_per_sample = F.smooth_l1_loss(
                output['search_proposal_xy'], target_xy,
                reduction='none').mean(dim=1)
            loss_search_proposal = (
                proposal_per_sample * proposal_valid).sum() / torch.clamp(
                    proposal_valid.sum(), min=1.0)

            confidence_target = (
                foreground_count >= 3).to(point_labels.dtype)
            loss_search_confidence = F.binary_cross_entropy_with_logits(
                output['search_confidence_logit'], confidence_target)
            loss_total += (
                self.search_v2_targetness_weight
                * loss_search_targetness
                + self.search_v2_vote_weight * loss_search_vote
                + self.search_v2_proposal_weight * loss_search_proposal
                + self.search_v2_confidence_weight
                * loss_search_confidence)
            loss_dict.update({
                'loss_total': loss_total,
                'loss_search_v2_targetness': loss_search_targetness,
                'loss_search_v2_vote': loss_search_vote,
                'loss_search_v2_proposal': loss_search_proposal,
                'loss_search_v2_confidence': loss_search_confidence,
                'search_v2_foreground_points': foreground_count.mean(),
                'search_v2_geometry_valid_rate': data[
                    'search_v2_geometry_valid'].float().mean(),
                'search_v2_candidate_valid_rate': output[
                    'search_candidate_valid'].float().mean(),
                'search_v2_confidence_mean': output[
                    'search_confidence'].mean(),
                'search_v2_targetness_mass': output[
                    'search_targetness_mass'].mean(),
                'search_v2_targetness_entropy': output[
                    'search_targetness_entropy'].mean(),
            })

            if self.use_joint_proposal_fusion:
                epoch = int(getattr(self, 'current_epoch', 0))
                if epoch < self.joint_fusion_warmup_epochs:
                    joint_ramp = 0.0
                elif self.joint_fusion_ramp_epochs <= 0:
                    joint_ramp = 1.0
                else:
                    joint_ramp = min(
                        1.0,
                        (epoch - self.joint_fusion_warmup_epochs + 1)
                        / float(self.joint_fusion_ramp_epochs))
                if joint_ramp > 0.0:
                    loss_joint_fused = F.smooth_l1_loss(
                        final_estimation_boxes[:, :2], target_xy)
                    observation_error = torch.linalg.norm(
                        aux_estimation_boxes[:, :2].detach() - target_xy,
                        dim=1)
                    motion_error = torch.linalg.norm(
                        output['motion_prior_proposal_xy'].detach()
                        - target_xy, dim=1)
                    search_error = torch.linalg.norm(
                        output['search_proposal_xy'].detach()
                        - target_xy, dim=1)
                    motion_valid = output['motion_prior_valid'] > 0
                    search_valid = output['search_candidate_valid'] > 0
                    motion_helpful = (
                        motion_valid
                        & (motion_error + self.joint_help_margin
                           < observation_error))
                    search_helpful = (
                        search_valid
                        & (search_error + self.joint_help_margin
                           < observation_error))
                    oracle = torch.zeros_like(
                        observation_error, dtype=torch.long)
                    oracle = torch.where(
                        motion_helpful, torch.ones_like(oracle), oracle)
                    choose_search = search_helpful & (
                        (~motion_helpful) | (search_error < motion_error))
                    oracle = torch.where(
                        choose_search,
                        torch.full_like(oracle, 2),
                        oracle)
                    loss_joint_gate = F.cross_entropy(
                        output['joint_gate_logits'], oracle)
                    loss_total += joint_ramp * (
                        self.joint_fused_weight * loss_joint_fused
                        + self.joint_gate_weight * loss_joint_gate)

                    selected = torch.argmax(
                        output['joint_gate_applied_probability'], dim=1)
                    selected_search = selected == 2
                    helpful_selected_search = (
                        selected_search & search_helpful)
                    loss_dict.update({
                        'loss_total': loss_total,
                        'loss_joint_fused': loss_joint_fused,
                        'loss_joint_gate': loss_joint_gate,
                        'joint_fusion_ramp': target_xy.new_tensor(joint_ramp),
                        'joint_observation_error': observation_error.mean(),
                        'joint_motion_error': motion_error.mean(),
                        'joint_search_error': search_error.mean(),
                        'joint_final_error': torch.linalg.norm(
                            final_estimation_boxes[:, :2].detach()
                            - target_xy, dim=1).mean(),
                        'joint_search_selected_rate':
                            selected_search.float().mean(),
                        'joint_search_helpful_precision':
                            helpful_selected_search.float().sum()
                            / torch.clamp(
                                selected_search.float().sum(), min=1.0),
                    })

        if self.use_search_evidence_v21:
            target_xy = center_label[:, :2].to(
                device=output['search_v21_proposal_xy'].device,
                dtype=output['search_v21_proposal_xy'].dtype)
            point_labels = data['search_v21_point_labels'].to(
                device=output['search_v21_targetness_logits'].device,
                dtype=output['search_v21_targetness_logits'].dtype)
            point_valid = data['search_v21_point_valid_mask'].to(
                device=point_labels.device, dtype=point_labels.dtype)

            def focal_bce(logits, target, alpha, gamma):
                bce = F.binary_cross_entropy_with_logits(
                    logits, target, reduction='none')
                probability = torch.sigmoid(logits)
                p_t = (
                    target * probability
                    + (1.0 - target) * (1.0 - probability))
                alpha_t = (
                    target * float(alpha)
                    + (1.0 - target) * (1.0 - float(alpha)))
                return alpha_t * (1.0 - p_t).pow(float(gamma)) * bce

            match_focal = focal_bce(
                output['search_v21_match_logits'],
                point_labels,
                self.search_v21_focal_alpha,
                self.search_v21_focal_gamma)
            targetness_focal = focal_bce(
                output['search_v21_targetness_logits'],
                point_labels,
                self.search_v21_focal_alpha,
                self.search_v21_focal_gamma)
            valid_point_count = torch.clamp(point_valid.sum(), min=1.0)
            loss_search_v21_match = (
                match_focal * point_valid).sum() / valid_point_count
            loss_search_v21_targetness = (
                targetness_focal * point_valid).sum() / valid_point_count

            foreground = point_labels * point_valid
            foreground_count = foreground.sum(dim=1)
            vote_error = F.smooth_l1_loss(
                output['search_v21_point_center_votes'],
                target_xy.unsqueeze(1).expand_as(
                    output['search_v21_point_center_votes']),
                reduction='none').mean(dim=2)
            loss_search_v21_vote = (
                vote_error * foreground).sum() / torch.clamp(
                    foreground.sum(), min=1.0)
            proposal_valid = (foreground_count >= 1).to(point_labels.dtype)
            proposal_per_sample = F.smooth_l1_loss(
                output['search_v21_proposal_xy'], target_xy,
                reduction='none').mean(dim=1)
            loss_search_v21_proposal = (
                proposal_per_sample * proposal_valid).sum() / torch.clamp(
                    proposal_valid.sum(), min=1.0)

            loss_total += (
                self.search_v21_match_weight * loss_search_v21_match
                + self.search_v21_targetness_weight
                * loss_search_v21_targetness
                + self.search_v21_vote_weight * loss_search_v21_vote
                + self.search_v21_proposal_weight
                * loss_search_v21_proposal)
            loss_dict.update({
                'loss_total': loss_total,
                'loss_search_v21_match': loss_search_v21_match,
                'loss_search_v21_targetness':
                    loss_search_v21_targetness,
                'loss_search_v21_vote': loss_search_v21_vote,
                'loss_search_v21_proposal': loss_search_v21_proposal,
                'search_v21_foreground_points': foreground_count.mean(),
                'search_v21_geometry_valid_rate': data[
                    'search_v21_geometry_valid'].float().mean(),
                'search_v21_candidate_valid_rate': output[
                    'search_v21_candidate_valid'].float().mean(),
                'search_v21_targetness_mean': output[
                    'search_v21_targetness_mean'].mean(),
                'search_v21_targetness_max': output[
                    'search_v21_targetness_max'].mean(),
                'search_v21_targetness_entropy': output[
                    'search_v21_targetness_entropy'].mean(),
                'search_v21_effective_sample_size': output[
                    'search_v21_effective_sample_size'].mean(),
                'search_v21_extension_weight_ratio': output[
                    'search_v21_extension_weight_ratio'].mean(),
                'search_v21_available_count': data[
                    'search_v21_available_count'].float().mean(),
                'search_v21_extension_count': data[
                    'search_v21_extension_count'].float().mean(),
                'search_v21_overlap_count': data[
                    'search_v21_overlap_count'].float().mean(),
            })

            advantage_ramp = 0.0
            if self.use_advantage_proposal_fusion:
                epoch = int(getattr(self, 'current_epoch', 0))
                if epoch < self.advantage_fusion_warmup_epochs:
                    advantage_ramp = 0.0
                elif self.advantage_fusion_ramp_epochs <= 0:
                    advantage_ramp = 1.0
                else:
                    advantage_ramp = min(
                        1.0,
                        (epoch - self.advantage_fusion_warmup_epochs + 1)
                        / float(self.advantage_fusion_ramp_epochs))

            if advantage_ramp > 0.0:
                observation_xy = aux_estimation_boxes[:, :2].detach()
                motion_xy = output[
                    'search_v21_motion_proposal_xy'].detach()
                search_xy = output['search_v21_proposal_xy'].detach()
                observation_error = torch.linalg.norm(
                    observation_xy - target_xy, dim=1)
                motion_error = torch.linalg.norm(
                    motion_xy - target_xy, dim=1)
                search_error = torch.linalg.norm(
                    search_xy - target_xy, dim=1)
                motion_valid = output[
                    'search_v21_motion_candidate_valid'] > 0
                search_valid = output[
                    'search_v21_search_candidate_valid'] > 0
                motion_helpful = (
                    motion_valid
                    & (motion_error + self.advantage_help_margin
                       <= observation_error))
                search_helpful = (
                    search_valid
                    & (search_error + self.advantage_help_margin
                       <= observation_error))

                help_target = torch.stack((
                    motion_helpful, search_helpful), dim=1).to(
                        target_xy.dtype)
                candidate_valid = torch.stack((
                    motion_valid, search_valid), dim=1).to(target_xy.dtype)
                help_alpha = target_xy.new_tensor((
                    self.advantage_help_alpha_motion,
                    self.advantage_help_alpha_search,
                )).reshape(1, 2)
                help_logits = output['advantage_help_logits']
                help_bce = F.binary_cross_entropy_with_logits(
                    help_logits, help_target, reduction='none')
                help_probability = torch.sigmoid(help_logits)
                help_p_t = (
                    help_target * help_probability
                    + (1.0 - help_target) * (1.0 - help_probability))
                help_alpha_t = (
                    help_target * help_alpha
                    + (1.0 - help_target) * (1.0 - help_alpha))
                help_focal = (
                    help_alpha_t
                    * (1.0 - help_p_t).pow(self.advantage_help_gamma)
                    * help_bce)
                per_candidate_help = (
                    help_focal * candidate_valid).sum(dim=0) / torch.clamp(
                        candidate_valid.sum(dim=0), min=1.0)
                present_candidate = (
                    candidate_valid.sum(dim=0) > 0).to(target_xy.dtype)
                loss_advantage_help = (
                    per_candidate_help * present_candidate).sum(
                    ) / torch.clamp(present_candidate.sum(), min=1.0)

                target_residual = target_xy - observation_xy
                candidate_residual = torch.stack((
                    output['advantage_motion_residual_xy'].detach(),
                    output['advantage_search_residual_xy'].detach(),
                ), dim=1)
                numerator = (
                    candidate_residual
                    * target_residual.unsqueeze(1)).sum(dim=2)
                denominator = candidate_residual.pow(2).sum(dim=2) + 1e-6
                oracle_step = torch.clamp(
                    numerator / denominator, min=0.0, max=1.0)
                helpful_mask = help_target * candidate_valid
                step_error = F.smooth_l1_loss(
                    torch.sigmoid(output['advantage_step_logits']),
                    oracle_step,
                    reduction='none')
                loss_advantage_step = (
                    step_error * helpful_mask).sum() / torch.clamp(
                        helpful_mask.sum(), min=1.0)
                loss_advantage_fused = F.smooth_l1_loss(
                    final_estimation_boxes[:, :2], target_xy)
                loss_total += advantage_ramp * (
                    self.advantage_fused_weight * loss_advantage_fused
                    + self.advantage_help_weight * loss_advantage_help
                    + self.advantage_step_weight * loss_advantage_step)

                applied_search = output[
                    'advantage_applied_weight'][:, 1] >= 0.1
                helpful_applied_search = applied_search & search_helpful

                def conditional_mean(value, mask):
                    mask = mask.to(value.dtype)
                    return (value * mask).sum() / torch.clamp(
                        mask.sum(), min=1.0)

                loss_dict.update({
                    'loss_total': loss_total,
                    'loss_advantage_fused': loss_advantage_fused,
                    'loss_advantage_help': loss_advantage_help,
                    'loss_advantage_step': loss_advantage_step,
                    'advantage_fusion_ramp': target_xy.new_tensor(
                        advantage_ramp),
                    'advantage_observation_error': observation_error.mean(),
                    'advantage_motion_error_valid': conditional_mean(
                        motion_error, motion_valid),
                    'advantage_search_error_valid': conditional_mean(
                        search_error, search_valid),
                    'advantage_final_error': torch.linalg.norm(
                        final_estimation_boxes[:, :2].detach()
                        - target_xy, dim=1).mean(),
                    'advantage_motion_helpful_rate': conditional_mean(
                        motion_helpful.to(target_xy.dtype), motion_valid),
                    'advantage_search_helpful_rate': conditional_mean(
                        search_helpful.to(target_xy.dtype), search_valid),
                    'advantage_motion_weight': output[
                        'advantage_applied_weight'][:, 0].mean(),
                    'advantage_search_weight': output[
                        'advantage_applied_weight'][:, 1].mean(),
                    'advantage_search_applied_rate':
                        applied_search.float().mean(),
                    'advantage_search_helpful_precision':
                        helpful_applied_search.float().sum()
                        / torch.clamp(
                            applied_search.float().sum(), min=1.0),
                })

        if self.use_motion_conditioned_search_v22:
            target_xy = center_label[:, :2].to(
                device=output['search_raw_vote_xy'].device,
                dtype=output['search_raw_vote_xy'].dtype)
            point_labels = data['search_v22_point_labels'].to(
                device=output['search_v22_targetness_logits'].device,
                dtype=output['search_v22_targetness_logits'].dtype)
            point_valid = data['search_v22_point_valid_mask'].to(
                device=point_labels.device, dtype=point_labels.dtype)

            def v22_focal_bce(logits, target, alpha, gamma):
                bce = F.binary_cross_entropy_with_logits(
                    logits, target, reduction='none')
                probability = torch.sigmoid(logits)
                p_t = (
                    target * probability
                    + (1.0 - target) * (1.0 - probability))
                alpha_t = (
                    target * float(alpha)
                    + (1.0 - target) * (1.0 - float(alpha)))
                return alpha_t * (1.0 - p_t).pow(float(gamma)) * bce

            match_focal = v22_focal_bce(
                output['search_v22_match_logits'],
                point_labels,
                self.search_v22_focal_alpha,
                self.search_v22_focal_gamma)
            targetness_focal = v22_focal_bce(
                output['search_v22_targetness_logits'],
                point_labels,
                self.search_v22_focal_alpha,
                self.search_v22_focal_gamma)
            valid_point_count = torch.clamp(point_valid.sum(), min=1.0)
            loss_search_v22_match = (
                match_focal * point_valid).sum() / valid_point_count
            loss_search_v22_targetness = (
                targetness_focal * point_valid).sum() / valid_point_count

            foreground = point_labels * point_valid
            foreground_count = foreground.sum(dim=1)
            vote_error = F.smooth_l1_loss(
                output['search_v22_point_center_votes'],
                target_xy.unsqueeze(1).expand_as(
                    output['search_v22_point_center_votes']),
                reduction='none').mean(dim=2)
            loss_search_v22_vote = (
                vote_error * foreground).sum() / torch.clamp(
                    foreground.sum(), min=1.0)
            proposal_valid = (foreground_count >= 1).to(point_labels.dtype)
            raw_proposal_error = F.smooth_l1_loss(
                output['search_raw_vote_xy'], target_xy,
                reduction='none').mean(dim=1)
            loss_search_v22_raw_proposal = (
                raw_proposal_error * proposal_valid).sum() / torch.clamp(
                    proposal_valid.sum(), min=1.0)
            refined_proposal_error = F.smooth_l1_loss(
                output['motion_search_refined_xy'], target_xy,
                reduction='none').mean(dim=1)
            refined_supervision_valid = (
                proposal_valid
                * output['motion_search_candidate_available'].detach())
            loss_search_v22_refined_proposal = (
                refined_proposal_error * refined_supervision_valid).sum(
                ) / torch.clamp(refined_supervision_valid.sum(), min=1.0)

            presence_target = (foreground_count >= 1).to(point_labels.dtype)
            presence_available = (
                output['motion_search_candidate_available'].detach())
            presence_error = F.binary_cross_entropy_with_logits(
                output['search_presence_logit'],
                presence_target,
                reduction='none')
            loss_search_v22_presence = (
                presence_error * presence_available).sum() / torch.clamp(
                    presence_available.sum(), min=1.0)

            loss_total += (
                self.search_v22_match_weight * loss_search_v22_match
                + self.search_v22_targetness_weight
                * loss_search_v22_targetness
                + self.search_v22_vote_weight * loss_search_v22_vote
                + self.search_v22_raw_proposal_weight
                * loss_search_v22_raw_proposal
                + self.search_v22_refined_proposal_weight
                * loss_search_v22_refined_proposal
                + self.search_v22_presence_weight
                * loss_search_v22_presence)
            loss_dict.update({
                'loss_total': loss_total,
                'loss_search_v22_match': loss_search_v22_match,
                'loss_search_v22_targetness':
                    loss_search_v22_targetness,
                'loss_search_v22_vote': loss_search_v22_vote,
                'loss_search_v22_raw_proposal':
                    loss_search_v22_raw_proposal,
                'loss_search_v22_refined_proposal':
                    loss_search_v22_refined_proposal,
                'loss_search_v22_presence': loss_search_v22_presence,
                'search_v22_foreground_points': foreground_count.mean(),
                'search_v22_geometry_valid_rate': data[
                    'search_v22_geometry_valid'].float().mean(),
                'search_v22_candidate_available_rate': output[
                    'motion_search_candidate_available'].float().mean(),
                'search_v22_candidate_valid_rate': output[
                    'motion_search_candidate_valid'].float().mean(),
                'search_v22_presence_probability': output[
                    'search_presence_probability'].mean(),
                'search_v22_normalized_ess': output[
                    'search_normalized_ess'].mean(),
                'search_v22_raw_ess': output['search_raw_ess'].mean(),
                'search_v22_raw_rmse': torch.linalg.norm(
                    output['search_raw_vote_xy'].detach() - target_xy,
                    dim=1).mean(),
                'search_v22_motion_rmse': torch.linalg.norm(
                    output['motion_prior_xy'].detach() - target_xy,
                    dim=1).mean(),
                'search_v22_refined_rmse': torch.linalg.norm(
                    output['motion_search_refined_xy'].detach() - target_xy,
                    dim=1).mean(),
            })
            if self.training and not torch.equal(
                    output['aux_estimation_boxes'],
                    output['observation_aux_estimation_boxes']):
                raise RuntimeError(
                    "B2-v2.2 refiner training must write exact observation")

        if self.use_motion_conditioned_search_v3:
            target_xy = center_label[:, :2].to(
                device=output['search_v3_raw_vote_xy'].device,
                dtype=output['search_v3_raw_vote_xy'].dtype)
            point_labels = data['search_v3_point_labels'].to(
                device=output['search_v3_targetness_logits'].device,
                dtype=output['search_v3_targetness_logits'].dtype)
            point_valid = data['search_v3_point_valid_mask'].to(
                device=point_labels.device, dtype=point_labels.dtype)

            def v3_focal_bce(logits, target, alpha, gamma):
                bce = F.binary_cross_entropy_with_logits(
                    logits, target, reduction='none')
                probability = torch.sigmoid(logits)
                p_t = (
                    target * probability
                    + (1.0 - target) * (1.0 - probability))
                alpha_t = (
                    target * float(alpha)
                    + (1.0 - target) * (1.0 - float(alpha)))
                return alpha_t * (1.0 - p_t).pow(float(gamma)) * bce

            match_focal = v3_focal_bce(
                output['search_v3_match_logits'], point_labels,
                self.search_v3_focal_alpha, self.search_v3_focal_gamma)
            targetness_focal = v3_focal_bce(
                output['search_v3_targetness_logits'], point_labels,
                self.search_v3_focal_alpha, self.search_v3_focal_gamma)
            valid_point_count = torch.clamp(point_valid.sum(), min=1.0)
            loss_search_v3_match = (
                match_focal * point_valid).sum() / valid_point_count
            loss_search_v3_targetness = (
                targetness_focal * point_valid).sum() / valid_point_count
            foreground = point_labels * point_valid
            foreground_count = foreground.sum(dim=1)
            vote_error = F.smooth_l1_loss(
                output['search_v3_point_center_votes'],
                target_xy.unsqueeze(1).expand_as(
                    output['search_v3_point_center_votes']),
                reduction='none').mean(dim=2)
            loss_search_v3_vote = (
                vote_error * foreground).sum() / torch.clamp(
                    foreground.sum(), min=1.0)
            proposal_valid = (foreground_count >= 1).to(point_labels.dtype)
            raw_proposal_error = F.smooth_l1_loss(
                output['search_v3_raw_vote_xy'], target_xy,
                reduction='none').mean(dim=1)
            loss_search_v3_raw_proposal = (
                raw_proposal_error * proposal_valid).sum() / torch.clamp(
                    proposal_valid.sum(), min=1.0)
            refined_proposal_error = F.smooth_l1_loss(
                output['motion_search_v3_refined_xy'], target_xy,
                reduction='none').mean(dim=1)
            refined_supervision_valid = (
                proposal_valid * output[
                    'motion_search_v3_candidate_structural_valid'].detach())
            loss_search_v3_refined_proposal = (
                refined_proposal_error * refined_supervision_valid).sum(
                ) / torch.clamp(refined_supervision_valid.sum(), min=1.0)
            presence_target = (foreground_count >= 1).to(point_labels.dtype)
            structural_available = output[
                'motion_search_v3_candidate_structural_valid'].detach()
            presence_error = F.binary_cross_entropy_with_logits(
                output['search_v3_presence_logit'],
                presence_target, reduction='none')
            loss_search_v3_presence = (
                presence_error * structural_available).sum() / torch.clamp(
                    structural_available.sum(), min=1.0)
            loss_search_v3_utility = target_xy.new_zeros(())
            utility_target = target_xy.new_zeros(
                (target_xy.shape[0],))
            if 'search_v3_utility_logit' in output:
                observation_error = torch.linalg.norm(
                    output['observation_aux_estimation_boxes'][:, :2].detach()
                    - target_xy, dim=1)
                raw_search_error = torch.linalg.norm(
                    output['search_v3_raw_vote_xy'].detach() - target_xy,
                    dim=1)
                utility_target = ((
                    raw_search_error + self.search_v3_utility_margin
                    < observation_error).to(target_xy.dtype)
                    * presence_target)
                utility_error = F.binary_cross_entropy_with_logits(
                    output['search_v3_utility_logit'],
                    utility_target,
                    reduction='none')
                loss_search_v3_utility = (
                    utility_error * structural_available).sum(
                    ) / torch.clamp(structural_available.sum(), min=1.0)
            loss_total += (
                self.search_v3_match_weight * loss_search_v3_match
                + self.search_v3_targetness_weight
                * loss_search_v3_targetness
                + self.search_v3_vote_weight * loss_search_v3_vote
                + self.search_v3_raw_proposal_weight
                * loss_search_v3_raw_proposal
                + self.search_v3_refined_proposal_weight
                * loss_search_v3_refined_proposal
                + self.search_v3_presence_weight
                * loss_search_v3_presence
                + self.search_v3_utility_weight
                * loss_search_v3_utility)
            loss_dict.update({
                'loss_total': loss_total,
                'loss_search_v3_match': loss_search_v3_match,
                'loss_search_v3_targetness': loss_search_v3_targetness,
                'loss_search_v3_vote': loss_search_v3_vote,
                'loss_search_v3_raw_proposal':
                    loss_search_v3_raw_proposal,
                'loss_search_v3_refined_proposal':
                    loss_search_v3_refined_proposal,
                'loss_search_v3_presence': loss_search_v3_presence,
                'loss_search_v3_utility': loss_search_v3_utility,
                'search_v3_utility_positive_rate': (
                    utility_target * structural_available).sum()
                    / torch.clamp(structural_available.sum(), min=1.0),
                'search_v3_foreground_points': foreground_count.mean(),
                'search_v3_structural_valid_rate':
                    structural_available.float().mean(),
                'search_v3_presence_probability': output[
                    'search_v3_presence_probability'].mean(),
                'search_v3_raw_rmse': torch.linalg.norm(
                    output['search_v3_raw_vote_xy'].detach() - target_xy,
                    dim=1).mean(),
                'search_v3_raw_rmse_foreground': (
                    torch.linalg.norm(
                        output['search_v3_raw_vote_xy'].detach() - target_xy,
                        dim=1) * proposal_valid).sum() / torch.clamp(
                            proposal_valid.sum(), min=1.0),
                'search_v3_motion_rmse': torch.linalg.norm(
                    output['motion_prior_xy'].detach() - target_xy,
                    dim=1).mean(),
                'search_v3_refined_rmse': torch.linalg.norm(
                    output['motion_search_v3_refined_xy'].detach()
                    - target_xy, dim=1).mean(),
            })
            if self.training and not torch.equal(
                    output['aux_estimation_boxes'],
                    output['observation_aux_estimation_boxes']):
                raise RuntimeError(
                    "B2-v3 refiner training must write exact observation")

        if self.box_aware:
            prev_bc = torch.flatten(data['prev_bc'], start_dim=1, end_dim=2)
            this_bc = data['this_bc'] #torch.Size([B, 1024, 9])
            bc_label = torch.cat([prev_bc, this_bc], dim=1) #torch.Size([B, 4096, 9])
            pred_bc = output['pred_bc'] #torch.Size([B, 4096, 9])
            loss_bc = self._b0_point_regression_loss(pred_bc, bc_label, data)
            loss_total += loss_bc * self.config.bc_weight
            b0_transaction_loss = (
                b0_transaction_loss + loss_bc * self.config.bc_weight)
            loss_dict.update({
                "loss_total": loss_total,
                "loss_bc": loss_bc
            })

        if self.use_point_feature_tc:
            if "pftc_point_features" not in output:
                raise KeyError(
                    "PFTC requires training-time point-aligned features.")
            time_key = f"timestamps_{self.pftc_time_field}"
            required = (
                "points", "seg_label", "box_label_prev", "box_label",
                "valid_mask", time_key)
            missing = [key for key in required if key not in data]
            if missing:
                raise KeyError(
                    "PFTC input is missing: " + ", ".join(missing))

            batch_size = output["pftc_point_features"].shape[0]
            num_frames = output["pftc_point_features"].shape[1]
            num_points = output["pftc_point_features"].shape[2]
            expected_points = num_frames * num_points
            if data["points"].shape[1] != expected_points:
                raise ValueError(
                    "PFTC point features no longer align with sampled XYZ: "
                    f"expected {expected_points}, got {data['points'].shape[1]}")
            boxes = torch.cat((
                data["box_label_prev"],
                data["box_label"].unsqueeze(1),
            ), dim=1)
            pftc_terms = self.point_feature_tc(
                point_features=output["pftc_point_features"],
                points=data["points"][..., :3].reshape(
                    batch_size, num_frames, num_points, 3),
                seg_mask=data["seg_label"].reshape(
                    batch_size, num_frames, num_points),
                boxes=boxes,
                history_valid_mask=data["valid_mask"],
                timestamps=data[time_key],
            )
            pftc_ramp = self._pftc_ramp()
            pftc_effective_weight = self.pftc_weight * pftc_ramp
            loss_total_sup = loss_total
            loss_total = (
                loss_total_sup
                + pftc_effective_weight * pftc_terms["loss"])
            b0_transaction_loss = (
                b0_transaction_loss
                + pftc_effective_weight * pftc_terms["loss"])
            loss_dict.update({
                "loss_total": loss_total,
                "loss_total_sup": loss_total_sup,
                "loss_pftc": pftc_terms["loss"],
                "pftc_ramp": loss_total.new_tensor(pftc_ramp),
                "pftc_lambda": loss_total.new_tensor(self.pftc_weight),
                "pftc_effective_weight": loss_total.new_tensor(
                    pftc_effective_weight),
            })
            loss_dict.update({
                key: value for key, value in pftc_terms.items()
                if key != "loss"
            })

        if getattr(self.config, "obs_gate_log_stats", False):
            obs_log_map = {
                "obs_num_points_search_mean": "obs_num_points_search",
                "obs_soft_fg_count_mean": "obs_soft_fg_count",
                "obs_estimated_fg_points_mean": "obs_estimated_fg_points",
                "obs_mean_fg_score": "obs_mean_fg_score",
                "obs_valid_history_ratio": "obs_valid_history_ratio",
                "obs_current_delta_t_ratio": "obs_current_delta_t_ratio",
                "obs_current_delta_t_real_ratio":
                    "obs_current_delta_t_real_ratio",
                "obs_current_delta_t_effective_ratio":
                    "obs_current_delta_t_effective_ratio",
            }
            for log_key, output_key in obs_log_map.items():
                if output_key in output:
                    loss_dict[log_key] = output[output_key].mean()

        if "obs_alpha" in output:
            obs_alpha = output["obs_alpha"]
            obs_gate_entropy = output["obs_gate_entropy"].mean()
            entropy_weight = getattr(self.config, "obs_gate_entropy_weight", 0.0)
            if entropy_weight != 0.0:
                loss_obs_gate_entropy = -obs_gate_entropy
                loss_total += entropy_weight * loss_obs_gate_entropy
                loss_dict["loss_total"] = loss_total
                loss_dict["loss_obs_gate_entropy"] = loss_obs_gate_entropy

            loss_dict.update({
                "obs_alpha_obs_mean": obs_alpha[:, 0].mean(),
                "obs_alpha_dyn_mean": obs_alpha[:, 1].mean(),
                "obs_alpha_dyn_min": obs_alpha[:, 1].min(),
                "obs_alpha_dyn_max": obs_alpha[:, 1].max(),
                "obs_gate_entropy": obs_gate_entropy,
            })

        if "obs_alpha_dyn_raw" in output and "obs_alpha_dyn_clamped" in output:
            loss_dict.update({
                "obs_alpha_dyn_raw_mean": output["obs_alpha_dyn_raw"].mean(),
                "obs_alpha_dyn_clamped_mean": output["obs_alpha_dyn_clamped"].mean(),
                "obs_alpha_dyn_clamped_max": output["obs_alpha_dyn_clamped"].max(),
            })
        if "motion_dyn_residual" in output:
            loss_dict["obs_dyn_residual_norm"] = torch.linalg.norm(
                output["motion_dyn_residual"], dim=1).mean()

        if "motion_dynamics_residual" in output:
            loss_dict.update({
                "dynamics_residual_norm": torch.linalg.norm(
                    output["motion_dynamics_residual"], dim=1).mean(),
                "dynamics_residual_alpha_mean": output["dynamics_residual_alpha"].mean(),
                "dynamics_residual_alpha_min": output["dynamics_residual_alpha"].min(),
                "dynamics_residual_alpha_max": output["dynamics_residual_alpha"].max(),
                "dynamics_residual_scale_effective": output[
                    "dynamics_residual_scale_effective"],
                "dynamics_residual_raw_norm": output[
                    "dynamics_residual_raw_norm"].mean(),
                "dynamics_residual_clamped_norm": output[
                    "dynamics_residual_clamped_norm"].mean(),
                "dynamics_residual_clamp_ratio": output[
                    "dynamics_residual_clamp_mask"].mean(),
                "dynamics_residual_applied_ratio": output[
                    "dynamics_residual_applied_mask"].mean(),
                "obs_dyn_center_gap": output["obs_dyn_center_gap"].mean(),
            })

        if "ct_fusion_alpha" in output:
            loss_dict.update({
                "ct_fusion_alpha_mean": output["ct_fusion_alpha"].mean(),
                "ct_fusion_alpha_min": output["ct_fusion_alpha"].min(),
                "ct_fusion_alpha_max": output["ct_fusion_alpha"].max(),
            })
        if "ct_fusion_alpha_applied" in output:
            loss_dict.update({
                "ct_fusion_alpha_applied_mean": output[
                    "ct_fusion_alpha_applied"].mean(),
                "ct_fusion_alpha_applied_max": output[
                    "ct_fusion_alpha_applied"].max(),
            })
        if "dynamics_innovation_applied_norm" in output:
            loss_dict.update({
                "ct_innovation_applied_norm": output[
                    "dynamics_innovation_applied_norm"].mean(),
                "ct_innovation_radius": output[
                    "dynamics_innovation_radius"].mean(),
                "ct_innovation_clamp_ratio": output[
                    "dynamics_innovation_clamp_mask"].mean(),
                "ct_innovation_applied_ratio": output[
                    "dynamics_innovation_applied_mask"].mean(),
            })
        for key in (
                "ct_search_used",
                "ct_search_expansion_ratio",
                "ct_search_baseline_points",
                "ct_search_expansion_points",
                "ct_search_query_delta_t",
                "ct_search_predicted_displacement",
                "trajectory_search_valid",
                "trajectory_search_gap_ratio",
                "trajectory_search_sigma_parallel",
                "trajectory_search_sigma_perpendicular",
                "search_has_usable_points"):
            if key in output:
                loss_dict[f"{key}_mean"] = output[key].float().mean()
        for key in (
                "trajectory_adapter_norm",
                "trajectory_adapter_scale",
                "trajectory_gap_activation",
                "trajectory_gap_ratio"):
            if key in output:
                loss_dict[f"{key}_mean"] = output[key].float().mean()

        # The unified automatic optimizer differentiates the sum of disjoint
        # transaction objectives.  B2 consumes detached B0 evidence, so plugin
        # losses have no path into the B0 transaction.
        recorded_b1_weighted_loss = loss_total.new_zeros(())
        for key, weight in (
                ('loss_motion_v3_prior', self.motion_v3_prior_weight),
                ('loss_ct_acquisition_margin',
                 self.ct_acquisition_margin_weight),
                ('loss_motion_v3_nll', self.motion_v3_nll_weight),
                ('loss_motion_v3_aux_prior',
                 self.motion_v3_aux_prior_weight),
                ('loss_motion_v3_aux_nll',
                 self.motion_v3_aux_nll_weight)):
            if key in loss_dict:
                recorded_b1_weighted_loss = (
                    recorded_b1_weighted_loss + weight * loss_dict[key])
        if not torch.equal(b1_transaction_loss, recorded_b1_weighted_loss):
            raise RuntimeError(
                "B1 transaction diverged from its recorded weighted losses")
        loss_dict['motion_v3_weighted_loss'] = recorded_b1_weighted_loss
        loss_dict['loss_b0_transaction'] = b0_transaction_loss
        loss_dict['loss_b1_transaction'] = b1_transaction_loss
        loss_dict['loss_b2_transaction'] = b2_transaction_loss
        loss_dict['loss_b3_transaction'] = b3_transaction_loss
        # Compatibility alias for v23 logs/checkpoints only.
        loss_dict['loss_plugin_transaction'] = (
            b1_transaction_loss + b2_transaction_loss
            + b3_transaction_loss)
        return loss_dict

    def on_train_epoch_start(self):
        pending_rng = getattr(self, '_ct_pending_global_rng_state', None)
        if pending_rng is not None:
            restore_global_rng_state(pending_rng)
            self._ct_pending_global_rng_state = None
        self._ct_epoch_boundary_complete = False
        for module_name in ('b0', 'b1', 'b2', 'b3', 'plugin'):
            setattr(self, f'_ct_{module_name}_updated_this_epoch', False)
        self._ct_epoch_binary_rows = {
            'presence': [], 'alpha': [], 'alpha_uplift': []}
        self._ct_epoch_acquisition_totals = {
            population: {
                'eligible_rows': 0.0,
                'retained_rows': 0.0,
                'pool_targets': 0.0,
                'sampled_targets': 0.0,
                'recovery_positive_rows': 0.0,
                'recovery_fallback_rows': 0.0,
            }
            for population in ('candidate0',)}
        if bool(getattr(
                self.config, 'ct_online_recursive_training', False)):
            self._ct_recursive_states = {}
            self._ct_online_batch_context = []
            train_loader = getattr(self.trainer, 'train_dataloader', None)
            if hasattr(train_loader, 'set_epoch'):
                train_loader.set_epoch(int(self.current_epoch))
            else:
                batch_sampler = getattr(train_loader, 'batch_sampler', None)
                if hasattr(batch_sampler, 'set_epoch'):
                    batch_sampler.set_epoch(int(self.current_epoch))

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        cuda_audit = (
            self.ct_cuda_stage_audit
            and torch.device(device).type == 'cuda')
        if cuda_audit:
            self._ct_cuda_stage_current = {}
            torch.cuda.reset_peak_memory_stats(device)
        if (isinstance(batch, dict)
                and batch.get('ct_stream_schema') in (
                    'ct_seqtrack.dual_stream.v1',
                    'ct_seqtrack.train.v2',
                    'ct_seqtrack.train.v3', 'ct_seqtrack.train.v4')):
            transferred = {
                'ct_stream_schema': batch['ct_stream_schema'],
                'observation': self._move_batch_to_device(
                    batch['observation'], device),
                # Raw nuScenes objects are deliberately processed only after
                # the current observation optimizer step has completed.
                'mechanism': batch.get('mechanism'),
            }
        elif (isinstance(batch, list) and batch
                and isinstance(batch[0], dict)
                and batch[0].get('online_recursive_raw', False)):
            # Raw nuScenes PointCloud/Box objects must remain on CPU until the
            # state-aware crop is built inside training_step.
            transferred = batch
        else:
            transferred = super().transfer_batch_to_device(
                batch, device, dataloader_idx)
        if cuda_audit:
            self._ct_record_cuda_stage('batch_transfer', device=device)
        return transferred

    def _ct_record_cuda_stage(self, stage, device=None):
        if not self.ct_cuda_stage_audit or not torch.cuda.is_available():
            return
        device = torch.device(self.device if device is None else device)
        if device.type != 'cuda':
            return
        values = {
            'allocated_mb': (
                torch.cuda.memory_allocated(device) / (1024.0 ** 2)),
            'reserved_mb': (
                torch.cuda.memory_reserved(device) / (1024.0 ** 2)),
            'peak_allocated_mb': (
                torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)),
            'peak_reserved_mb': (
                torch.cuda.max_memory_reserved(device) / (1024.0 ** 2)),
        }
        current = self._ct_cuda_stage_current.setdefault(stage, {})
        lifetime = self._ct_cuda_stage_max.setdefault(stage, {})
        for name, value in values.items():
            current[name] = max(float(value), float(current.get(name, 0.0)))
            lifetime[name] = max(float(value), float(lifetime.get(name, 0.0)))

    def _ct_record_observation_batch_fingerprint(self, batch):
        limit = int(self.ct_observation_fingerprint_steps)
        rows = self._ct_observation_batch_fingerprints
        if limit <= 0 or len(rows) >= limit:
            return
        digest = hashlib.sha256()

        def update(path, value):
            digest.update(str(path).encode('utf-8'))
            if torch.is_tensor(value):
                tensor = value.detach().cpu().contiguous()
                digest.update(str(tensor.dtype).encode('ascii'))
                digest.update(str(tuple(tensor.shape)).encode('ascii'))
                digest.update(tensor.numpy().tobytes())
            elif isinstance(value, np.ndarray):
                array = np.ascontiguousarray(value)
                digest.update(str(array.dtype).encode('ascii'))
                digest.update(str(tuple(array.shape)).encode('ascii'))
                digest.update(array.tobytes())
            elif isinstance(value, dict):
                for key in sorted(value):
                    update(f'{path}/{key}', value[key])
            elif isinstance(value, (tuple, list)):
                for index, item in enumerate(value):
                    update(f'{path}/{index}', item)
            else:
                digest.update(repr(value).encode('utf-8'))

        update('observation', batch)
        rows.append({
            'observation_step': len(rows) + 1,
            'sha256': digest.hexdigest(),
        })

    @staticmethod
    def _move_batch_to_device(value, device):
        if torch.is_tensor(value):
            return value.to(device=device, non_blocking=True)
        if isinstance(value, dict):
            return {
                key: SEQTRACK3D._move_batch_to_device(item, device)
                for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return type(value)(
                SEQTRACK3D._move_batch_to_device(item, device)
                for item in value)
        return value

    def _recursive_state_for_raw(self, raw):
        if not hasattr(self, '_ct_recursive_states'):
            self._ct_recursive_states = {}
        key = (int(raw['online_epoch']), str(raw['tracklet_key']))
        state = self._ct_recursive_states.get(key)
        if state is None:
            state = RecursiveTrackState(
                tracklet_id=int(raw['tracklet_id']),
                tracklet_key=str(raw['tracklet_key']),
                first_box=raw['first_frame']['3d_bbox'],
                timestamps={0: raw['first_frame'].get('timestamp')},
            )
            self._ct_recursive_states[key] = state
        return state

    @staticmethod
    def _ordered_online_history_frames(raw):
        return [
            raw['prev_frames'][key]
            for key in sorted(
                raw['prev_frames'], key=lambda value: abs(int(value)))
        ]

    def _online_rollout_horizon(self, raw):
        return rotating_rollout_horizon(
            getattr(
                self.config, 'ct_recursive_rollout_horizons', [1, 2, 4, 8]),
            raw['online_slot'], raw['online_epoch'],
            getattr(self.config, 'ct_recursive_tracklet_slots', 4))

    def _prepare_online_state_group(self, raw, state):
        """Apply one causal expert boundary before all candidate views."""
        horizon = self._online_rollout_horizon(raw)
        frame_id = int(raw['this_frame_id'])
        history_frames = self._ordered_online_history_frames(raw)
        history_ids = [int(value) for value in raw['prev_frame_ids']]
        pre_reset_anchor = state.history_boxes(
            [history_ids[0]], [1])[0]
        gt_anchor = history_frames[0]['3d_bbox']
        pre_reset_error = float(np.linalg.norm(
            np.asarray(pre_reset_anchor.center[:2], dtype=np.float64)
            - np.asarray(gt_anchor.center[:2], dtype=np.float64)))
        reseed_enabled = bool(getattr(
            self.config, 'ct_recursive_reseed_enabled', True))
        reset_boundary = bool(
            reseed_enabled and (frame_id - 1) % horizon == 0)
        if reset_boundary and frame_id > 1:
            state.reseed_history(
                history_ids,
                [frame['3d_bbox'] for frame in history_frames],
                [frame.get('timestamp') for frame in history_frames],
                before_frame_id=frame_id,
                rollout_horizon=horizon,
            )
        elif frame_id == 1:
            state.rollout_horizon = horizon
            state.last_reseed_before_frame = 1
        post_reset_anchor = state.history_boxes(
            [history_ids[0]], [1])[0]
        post_reset_error = float(np.linalg.norm(
            np.asarray(post_reset_anchor.center[:2], dtype=np.float64)
            - np.asarray(gt_anchor.center[:2], dtype=np.float64)))
        return {
            'rollout_horizon': horizon,
            'rollout_age': state.rollout_age(frame_id),
            'reset_boundary': reset_boundary,
            'pre_reset_anchor_error': pre_reset_error,
            'post_reset_anchor_error': post_reset_error,
        }

    def _online_motion_prepass(self, raw, state):
        return self._online_motion_prepass_batch([(raw, state)])[0]

    @torch.no_grad()
    def _online_motion_prepass_batch(self, raw_state_pairs):
        """Run one B1 forward for all causal slots, never per candidate."""
        if not (self.ct_joint_contract_version >= 2
                and self.use_b1_prepass_support and self.ct_enable_b1):
            return [None] * len(raw_state_pairs)
        prepass_inputs = []
        for raw, state in raw_state_pairs:
            contract = build_recursive_input_contract(
                state, raw['this_frame_id'], len(raw['prev_frame_ids']),
                self.config, candidate_id=0, offsets=raw['history_offsets'],
                epoch=raw.get('online_epoch', 0))
            history_boxes = state.history_boxes(
                contract['history_frame_ids'],
                contract['history_valid_mask'].tolist())
            history_frames = self._ordered_online_history_frames(raw)
            current_frame = raw['this_frame']
            inputs = self._build_motion_prepass_inputs_contract(
                history_boxes,
                contract['history_frame_ids'],
                contract['history_valid_mask'].tolist(),
                list(contract['history_timestamps']),
                current_frame.get('timestamp'),
                [frame.get('_ct_effective_timestamp')
                 for frame in history_frames],
                current_frame.get('_ct_effective_timestamp'),
                current_frame.get(
                    '_ct_dynamics_time_mode',
                    getattr(self.config, 'dynamics_time_mode', 'true')),
                int(raw['this_frame_id']),
                **(dict(history_quality=contract['history_quality'],
                        recursive_age=state.rollout_age(raw['this_frame_id']),
                        first_frame_wlh=state.target_size)
                   if self.ct_enable_v27 else {}),
            )
            if inputs is None:
                raise RuntimeError(
                    "online B1 prepass history length does not match hist_num")
            prepass_inputs.append(inputs)
        prediction = self.predict_motion_from_history(
            np.stack([item['ref_boxs'] for item in prepass_inputs]),
            np.stack([item['delta_t'] for item in prepass_inputs]),
            np.stack([item['valid_mask'] for item in prepass_inputs]),
            np.asarray([
                item['current_delta_t'] for item in prepass_inputs],
                dtype=np.float32),
            **({'acquisition_features': np.stack([
                item['acquisition_features'] for item in prepass_inputs])}
               if self.ct_enable_v27 else {}),
        )
        results = self._unbatch_motion_prepass_predictions(
            prediction,
            [item['current_delta_t'] for item in prepass_inputs])
        if self.ct_enable_v27:
            from utils.b1_acquisition import b1_input_digest
            for result, inputs in zip(results, prepass_inputs):
                result['input_digest'] = b1_input_digest(inputs)
                result['parameter_revision'] = int(self.global_step)
        return results

    def _process_online_raw(
            self, raw, state, motion_prediction=None, state_diagnostics=None):
        # The public/formal config describes the restored independent B0
        # observation stream.  Online recursive mechanism crops retain the
        # coherent shared-SE(2) contract and must not accidentally be parsed
        # with the observation candidate mode.
        processing_config = getattr(
            self, '_ct_mechanism_processing_config', None)
        if processing_config is None:
            processing_config = copy.deepcopy(self.config)
            processing_config.candidate_trajectory_mode = 'shared_se2'
            processing_config.num_candidates = 1
            processing_config.ct_recursive_candidate_views = 1
            processing_config.ct_b0_candidate_views = 1
            processing_config.ct_b0_candidate_weights = [1.0]
            self._ct_mechanism_processing_config = processing_config
        payload = {
            key: value for key, value in raw.items()
            if key not in (
                'online_recursive_raw', 'online_epoch',
                'online_batch_index', 'online_slot', 'shadow_future')
        }
        contract = build_recursive_input_contract(
            state, raw['this_frame_id'], len(raw['prev_frame_ids']),
            processing_config, candidate_id=raw['candidate_id'],
            offsets=raw['history_offsets'],
            epoch=raw.get('online_epoch', 0))
        if (contract['history_frame_ids'] != list(raw['prev_frame_ids'])
                or contract['history_valid_mask'].tolist()
                != list(raw['valid_mask'])):
            raise RuntimeError("raw/state recursive history contract mismatch")
        candidate_id = int(raw['candidate_id'])
        recovery_policy = str(getattr(
            self.config, 'ct_recovery_candidate_policy', 'off')).strip().lower()
        if recovery_policy != 'off':
            raise RuntimeError(
                "formal online training forbids GT-guided recovery candidates")
        b0_view_id = int(raw.get('b0_view_id', candidate_id))
        b0_auxiliary_only = bool(raw.get(
            'ct_b0_auxiliary_only', b0_view_id != 0))
        if b0_auxiliary_only != (b0_view_id != 0):
            raise RuntimeError("b0_view_id/auxiliary marker mismatch")
        payload['online_recursive_state'] = contract
        payload['candidate_shared_transform'] = contract[
            'candidate_shared_transform']
        payload['point_sampling_seeds'] = contract['point_sampling_seeds']
        payload['current_sampling_seed'] = contract[
            'current_sampling_seed']
        payload['ct_observation_only'] = b0_auxiliary_only
        if motion_prediction is not None:
            payload['motion_prediction'] = motion_prediction
        if 'motion_aux_frame_ids' in raw:
            aux_contract = build_recursive_input_contract(
                state, raw['this_frame_id'],
                len(raw['motion_aux_frame_ids']), processing_config,
                candidate_id=raw['candidate_id'],
                offsets=raw['motion_aux_offsets'],
                epoch=raw.get('online_epoch', 0))
            if (aux_contract['history_frame_ids']
                    != list(raw['motion_aux_frame_ids'])
                    or aux_contract['history_valid_mask'].tolist()
                    != list(raw['motion_aux_valid_mask'])):
                raise RuntimeError(
                    "raw/state auxiliary history contract mismatch")
            payload['online_motion_aux_state'] = aux_contract
        processed = motion_processing_mf(payload, processing_config)
        candidate_consistent = online_candidate_state_consistent(
            processed, contract['target_size'])
        if not candidate_consistent:
            raise RuntimeError(
                "candidate crop/history/Search state contract diverged")
        state_diagnostics = state_diagnostics or {}
        recursive_age = state_diagnostics.get('rollout_age')
        recursive_age_valid = recursive_age is not None
        current_labels = processed['seg_label'][
            -int(getattr(self.config, 'point_sample_size', 1024)):]
        processed.update({
            'ct_online_tracklet_id': np.int64(raw['tracklet_id']),
            'ct_online_frame_id': np.int64(raw['this_frame_id']),
            'ct_online_slot': np.int64(raw['online_slot']),
            'ct_online_epoch': np.int64(raw['online_epoch']),
            'ct_recursive_state_age': np.float32(
                recursive_age if recursive_age_valid else -1.0),
            'ct_recursive_state_age_valid': np.float32(
                recursive_age_valid),
            'ct_recursive_rollout_horizon': np.float32(
                state_diagnostics.get('rollout_horizon', 0)),
            'ct_recursive_reset_boundary': np.float32(
                bool(state_diagnostics.get('reset_boundary', False))),
            'ct_recursive_state_source': np.float32(
                1.0 if state_diagnostics.get('rollout_age', 0) > 0
                else 0.0),
            'ct_recursive_pre_reset_anchor_error': np.float32(
                state_diagnostics.get('pre_reset_anchor_error', 0.0)),
            'ct_recursive_anchor_error': np.float32(
                state_diagnostics.get('post_reset_anchor_error', 0.0)),
            'ct_crop_target_points': np.float32(np.sum(current_labels > 0)),
            'ct_candidate_state_consistency': np.float32(
                candidate_consistent),
            'b0_view_id': np.int64(b0_view_id),
            'ct_b0_auxiliary_only': np.float32(b0_auxiliary_only),
        })
        return processed

    def _prepare_online_recursive_batch(self, raw_items):
        processed = []
        context = []
        group_context = {}
        for raw in raw_items:
            if not raw.get('online_recursive_raw', False):
                raise ValueError("mixed online/non-online training batch")
            group_key = (
                int(raw['online_slot']), str(raw['tracklet_key']),
                int(raw['this_frame_id']))
            if group_key in group_context:
                continue
            state = self._recursive_state_for_raw(raw)
            if self.ct_joint_contract_version >= 2:
                diagnostics = self._prepare_online_state_group(raw, state)
            else:
                diagnostics = {}
            group_context[group_key] = {
                'raw': raw,
                'state': state,
                'diagnostics': diagnostics,
                'motion_prediction': None,
            }
        group_keys = list(group_context)
        prepass_predictions = self._online_motion_prepass_batch([
            (group_context[key]['raw'], group_context[key]['state'])
            for key in group_keys])
        for key, prediction in zip(group_keys, prepass_predictions):
            group_context[key]['motion_prediction'] = prediction
        for raw in raw_items:
            group_key = (
                int(raw['online_slot']), str(raw['tracklet_key']),
                int(raw['this_frame_id']))
            group = group_context[group_key]
            state = group['state']
            processed.append(self._process_online_raw(
                raw, state,
                motion_prediction=group['motion_prediction'],
                state_diagnostics=group['diagnostics']))
            context.append({'raw': raw, 'state': state})
        auxiliary_batch = None
        if self.ct_joint_contract_version >= 3:
            dual_stream = str(getattr(
                self.config, 'ct_training_topology', 'legacy')).strip().lower()
            if dual_stream == 'dual_stream':
                auxiliary_processed = []
                expected_slots = int(getattr(self.config, 'batch_size', 16))
                size_valid = (0 < len(processed) <= expected_slots) if self.ct_enable_v27 else len(processed) == expected_slots
                if (not size_valid
                        or any(int(np.asarray(
                            item['b0_view_id']).reshape(-1)[0]) != 0
                               for item in processed)):
                    raise RuntimeError(
                        'dual-stream mechanism transaction must contain one '
                        'canonical B0 row per tracklet slot')
            else:
                (
                    processed, auxiliary_processed,
                    context, _auxiliary_context,
                ) = partition_candidate_view_items(
                    processed, context,
                    canonical_batch_size=int(getattr(
                        self.config, 'batch_size', 16)),
                    candidate_views=int(getattr(
                        self.config, 'ct_b0_candidate_views', 1)))
            if auxiliary_processed:
                auxiliary_batch = default_collate(auxiliary_processed)
        batch = default_collate(processed)

        def attach_shadow_diagnostics(target, batch_size):
            target['ct_h3_gain'] = torch.zeros(
                batch_size, dtype=torch.float32)
            target['ct_h3_center_gain'] = torch.zeros(
                batch_size, dtype=torch.float32)
            target['ct_h3_iou_gain'] = torch.zeros(
                batch_size, dtype=torch.float32)
            target['ct_h3_valid'] = torch.zeros(
                batch_size, dtype=torch.float32)
            target['ct_shadow_forward_count'] = torch.tensor(0.0)
            target['ct_shadow_time_ms'] = torch.tensor(0.0)
            target['ct_shadow_peak_memory_mb'] = torch.tensor(0.0)

        attach_shadow_diagnostics(batch, len(processed))
        if auxiliary_batch is not None:
            attach_shadow_diagnostics(
                auxiliary_batch, len(auxiliary_processed))
            auxiliary_batch = self._move_batch_to_device(
                auxiliary_batch, self.device)
        self._ct_online_batch_context = context
        batch = self._move_batch_to_device(batch, self.device)
        return batch, auxiliary_batch

    def _local_prediction_to_world(self, local_box, anchor_box):
        values = local_box.detach().cpu().numpy().reshape(-1)[:4]
        return points_utils.getOffsetBB(
            anchor_box, values, degrees=self.config.degrees,
            use_z=self.config.use_z,
            limit_box=self.config.limit_box)

    def _shadow_forward(self, batch, seed):
        training_flags = {
            module: module.training for module in self.modules()}
        previous_joint_full = self.use_ct_joint_full
        previous_motion_v3 = self.use_b1motion_v3
        previous_b2 = self.ct_enable_b2
        previous_b3 = self.ct_enable_b3
        previous_b1 = self.ct_enable_b1
        cuda_devices = (
            [self.device.index
             if self.device.index is not None
             else torch.cuda.current_device()]
            if self.device.type == 'cuda' else [])
        try:
            for module in training_flags:
                module.training = False
            # H=3 future steps are deliberately observation-only.  Disabling
            # the B1/B2/B3 action gates is insufficient because the structural
            # Joint Full branch still validates and consumes Search tensors.
            # The shadow sampler intentionally omits those tensors, so bypass
            # both plugin entry points and execute the exact B0 forward.
            self.use_ct_joint_full = False
            self.use_b1motion_v3 = False
            self.ct_enable_b2 = False
            self.ct_enable_b3 = False
            self.ct_enable_b1 = False
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(int(seed))
                if self.device.type == 'cuda':
                    torch.cuda.manual_seed_all(int(seed))
                with torch.inference_mode():
                    return self(batch)
        finally:
            self.use_ct_joint_full = previous_joint_full
            self.use_b1motion_v3 = previous_motion_v3
            self.ct_enable_b2 = previous_b2
            self.ct_enable_b3 = previous_b3
            self.ct_enable_b1 = previous_b1
            for module, was_training in training_flags.items():
                module.training = was_training

    def _attach_h3_shadow_labels(self, batch, output):
        if self.ct_enable_v27:
            from utils.v27_training import attach_h3_shadow_labels_v27
            return attach_h3_shadow_labels_v27(self, batch, output)
        if (not bool(getattr(
                self.config, 'ct_online_recursive_training', False))
                or not self.ct_enable_b3):
            return
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        memory_before = (
            torch.cuda.memory_allocated(self.device)
            if self.device.type == 'cuda' else 0)
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        shadow_forward_count = 0
        for index, item in enumerate(self._ct_online_batch_context):
            raw = item['raw']
            if not raw.get('shadow_future'):
                continue
            if int(raw['candidate_id']) != 0:
                raise RuntimeError("H=3 shadow is candidate0-only")
            if float(output['ct_search_candidate_valid'][index].detach()) <= 0:
                continue
            state_before = item['state'].clone()
            anchor = state_before.history_boxes(
                [raw['prev_frame_ids'][0]], [1])[0]
            observation_local = output[
                'observation_aux_estimation_boxes'][index].detach().clone()
            search_local = observation_local.clone()
            search_local[:2] = (
                observation_local[:2]
                + output['ct_router_bounded_residual_xy'][index].detach())
            observation_box = self._local_prediction_to_world(
                observation_local, anchor)
            search_box = self._local_prediction_to_world(search_local, anchor)
            state_o = state_before.clone()
            state_s = state_before.clone()
            timestamp = raw['this_frame'].get('timestamp')
            state_o.append(raw['this_frame_id'], observation_box, timestamp)
            state_s.append(raw['this_frame_id'], search_box, timestamp)
            target = np.asarray(
                raw['this_frame']['3d_bbox'].center[:2], dtype=np.float64)
            cost_o = float(np.linalg.norm(observation_box.center[:2] - target))
            cost_s = float(np.linalg.norm(search_box.center[:2] - target))
            target_box = raw['this_frame']['3d_bbox']
            iou_gain = float(
                estimateOverlap(
                    search_box, target_box, dim=self.config.IoU_space)
                - estimateOverlap(
                    observation_box, target_box, dim=self.config.IoU_space))
            horizon_count = 1

            for future_raw in raw['shadow_future']:
                shadow_processed = [
                    self._process_online_raw(future_raw, state_o),
                    self._process_online_raw(future_raw, state_s),
                ]
                shadow_batch = self._move_batch_to_device(
                    default_collate(shadow_processed), self.device)
                shadow_seed = stable_uint32_seed(
                    int(getattr(self.config, 'seed', 42) or 42),
                    raw['tracklet_key'], raw['this_frame_id'],
                    future_raw['this_frame_id'], 'h3_shadow_observation')
                shadow_output = self._shadow_forward(
                    shadow_batch, shadow_seed)
                shadow_forward_count += 2
                future_boxes = []
                for branch_index, branch_state in enumerate((state_o, state_s)):
                    branch_anchor = branch_state.history_boxes(
                        [future_raw['prev_frame_ids'][0]], [1])[0]
                    local_observation = shadow_output[
                        'observation_aux_estimation_boxes'][branch_index]
                    world_box = self._local_prediction_to_world(
                        local_observation, branch_anchor)
                    branch_state.append(
                        future_raw['this_frame_id'], world_box,
                        future_raw['this_frame'].get('timestamp'))
                    future_boxes.append(world_box)
                future_target = np.asarray(
                    future_raw['this_frame']['3d_bbox'].center[:2],
                    dtype=np.float64)
                cost_o += float(np.linalg.norm(
                    future_boxes[0].center[:2] - future_target))
                cost_s += float(np.linalg.norm(
                    future_boxes[1].center[:2] - future_target))
                future_target_box = future_raw['this_frame']['3d_bbox']
                iou_gain += float(
                    estimateOverlap(
                        future_boxes[1], future_target_box,
                        dim=self.config.IoU_space)
                    - estimateOverlap(
                        future_boxes[0], future_target_box,
                        dim=self.config.IoU_space))
                horizon_count += 1
            center_gain = float(cost_o - cost_s) / float(horizon_count)
            iou_gain /= float(horizon_count)
            batch['ct_h3_gain'][index] = center_gain
            batch['ct_h3_center_gain'][index] = center_gain
            batch['ct_h3_iou_gain'][index] = iou_gain
            batch['ct_h3_valid'][index] = 1.0

        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        memory_after = (
            torch.cuda.max_memory_allocated(self.device)
            if self.device.type == 'cuda' else memory_before)
        batch['ct_shadow_forward_count'] = batch[
            'ct_shadow_forward_count'].new_tensor(float(shadow_forward_count))
        batch['ct_shadow_time_ms'] = batch[
            'ct_shadow_time_ms'].new_tensor(elapsed_ms)
        batch['ct_shadow_peak_memory_mb'] = batch[
            'ct_shadow_peak_memory_mb'].new_tensor(
                max(0, memory_after - memory_before) / (1024.0 ** 2))

    def _commit_online_recursive_predictions(self, output):
        if not bool(getattr(
                self.config, 'ct_online_recursive_training', False)):
            return
        seen_slots = set()
        for index, item in enumerate(self._ct_online_batch_context):
            raw = item['raw']
            slot = int(raw['online_slot'])
            if int(raw['candidate_id']) != 0:
                continue
            if slot in seen_slots:
                raise RuntimeError("online batch contains duplicate canonical slot")
            seen_slots.add(slot)
            state = item['state']
            anchor = state.history_boxes(
                [raw['prev_frame_ids'][0]], [1])[0]
            state_policy = str(getattr(
                self.config, 'ct_training_state_policy',
                'observation')).strip().lower()
            if state_policy != 'observation':
                raise RuntimeError(
                    "formal CT training permits only observation recursive "
                    "state; B2/B3 are shadow learners")
            final_box = self._local_prediction_to_world(
                output['observation_aux_estimation_boxes'][index], anchor)
            commit_canonical_prediction(
                state, raw['candidate_id'], raw['this_frame_id'], final_box,
                raw['this_frame'].get('timestamp'))
            if self.ct_enable_v27:
                state.quality[int(raw['this_frame_id'])] = output[
                    'ct_observation_quality'][index].detach().cpu().numpy().copy()

    def _ensure_ct_scalers(self):
        if self._ct_scalers:
            return
        names = list(getattr(self, '_ct_optimizer_names', ()))
        if not names:
            raise RuntimeError("isolated optimizer names are not configured")
        enabled = bool(
            self.ct_manual_amp_enabled and self.device.type == 'cuda')
        for name in names:
            try:
                scaler = torch.amp.GradScaler('cuda', enabled=enabled)
            except (AttributeError, TypeError):
                scaler = torch.cuda.amp.GradScaler(enabled=enabled)
            self._ct_scalers[name] = scaler
        self._ct_b0_scaler = self._ct_scalers.get('b0')
        self._ct_plugin_scaler = self._ct_scalers.get('b2')
        pending = self._ct_pending_scaler_state
        if isinstance(pending, dict):
            for name, scaler in self._ct_scalers.items():
                state = pending.get(name)
                # Accept the old two-scaler checkpoint for diagnostic resume.
                if state is None and name != 'b0':
                    state = pending.get('plugin')
                if state is not None:
                    scaler.load_state_dict(state)
        self._ct_pending_scaler_state = None

    @staticmethod
    def _slice_batch_rows(batch, row_mask):
        """Slice collated tensors while preserving scalar diagnostics."""
        batch_size = int(row_mask.numel())
        sliced = {}
        for key, value in batch.items():
            if (torch.is_tensor(value) and value.dim() > 0
                    and int(value.shape[0]) == batch_size):
                sliced[key] = value[row_mask]
            else:
                sliced[key] = value
        return sliced

    @staticmethod
    def _assign_parameter_gradients(parameters, gradients):
        for parameter, gradient in zip(parameters, gradients):
            parameter.grad = gradient

    def _ct_record_acquisition_supply(self, loss_dict, population):
        totals_by_population = getattr(
            self, '_ct_epoch_acquisition_totals', None)
        if not isinstance(totals_by_population, dict):
            return
        totals = totals_by_population[population]
        mapping = {
            'eligible_rows': 'ct_acquisition_eligible_row_count',
            'retained_rows': 'ct_acquisition_retained_row_count',
            'pool_targets': 'ct_acquisition_pool_target_sum',
            'sampled_targets': 'ct_acquisition_sampled_target_sum',
            'recovery_positive_rows': 'ct_recovery_positive_row_count',
            'recovery_fallback_rows': 'ct_recovery_fallback_row_count',
        }
        for target, source in mapping.items():
            value = loss_dict.get(source)
            if value is not None:
                totals[target] += float(value.detach().cpu().item())

    def _ct_isolated_optimizer_step(
            self, loss_dict, auxiliary_gradients=None,
            active_modules=None, advance_transaction=True):
        """Execute one disjoint transaction per active B0--B3 module."""
        auxiliary_gradients = auxiliary_gradients or {}
        self._ensure_ct_scalers()
        optimizers = self.optimizers(use_pl_optimizer=False)
        if not isinstance(optimizers, (list, tuple)):
            optimizers = [optimizers]
        names = list(self._ct_optimizer_names)
        if ('b0' in names
                and not hasattr(self, '_ct_b0_prefix_hashes')):
            self._ct_b0_prefix_hashes = {
                'initial': self._ct_module_parameter_hash('b0')}
        active_modules = (
            set(names) if active_modules is None else set(active_modules))
        unknown = active_modules - set(names)
        if unknown:
            raise RuntimeError(
                'optimizer transaction names are not configured: '
                + ', '.join(sorted(unknown)))
        if len(optimizers) != len(names):
            raise RuntimeError("optimizer/module cardinality mismatch")
        optimizer_map = dict(zip(names, optimizers))
        loss_key = {
            name: f'loss_{name}_transaction' for name in names}
        gradients_by_module = {}
        parameters_by_module = {}
        for name, optimizer in optimizer_map.items():
            if name not in active_modules:
                continue
            optimizer.zero_grad(set_to_none=True)
            parameters = [
                parameter for _, parameter
                in self._ct_named_parameters_by_module[name]]
            parameters_by_module[name] = parameters
            weight = float(loss_dict.get(
                f'ct_canonical_{name}_weight', 1.0))
            scaled_loss = self._ct_scalers[name].scale(
                weight * loss_dict[loss_key[name]])
            gradients = torch.autograd.grad(
                scaled_loss, parameters, retain_graph=False,
                allow_unused=True)
            module_auxiliary = auxiliary_gradients.get(name)
            if module_auxiliary is not None:
                if len(module_auxiliary) != len(gradients):
                    raise RuntimeError(
                        f"auxiliary/{name.upper()} gradient cardinality "
                        "mismatch")
                gradients = tuple(
                    auxiliary if canonical is None else canonical
                    if auxiliary is None else canonical + auxiliary
                    for canonical, auxiliary in zip(
                        gradients, module_auxiliary))
            gradients_by_module[name] = gradients
            self._assign_parameter_gradients(parameters, gradients)

        norms = {}
        stepped = {name: False for name in names}
        for name, optimizer in optimizer_map.items():
            if name not in active_modules:
                continue
            scaler = self._ct_scalers[name]
            scaler.unscale_(optimizer)
            clip = float(getattr(
                self.config, f'ct_{name}_gradient_clip_val',
                getattr(self.config, 'ct_plugin_gradient_clip_val',
                        getattr(self.config, 'gradient_clip_val', 0.0))))
            norms[name] = torch.nn.utils.clip_grad_norm_(
                parameters_by_module[name],
                max_norm=clip if clip > 0 else float('inf'))
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            stepped[name] = scaler.get_scale() >= scale_before
            if stepped[name]:
                getattr(self, f'ct_{name}_update_step').add_(1)
                setattr(self, f'_ct_{name}_updated_this_epoch', True)
                if name == 'b0':
                    update_step = int(self.ct_b0_update_step.item())
                    if update_step in (1, 100):
                        self._ct_b0_prefix_hashes[f'step_{update_step}'] = (
                            self._ct_module_parameter_hash('b0'))
        if stepped.get('b0'):
            self._ct_b0_updated_this_epoch = True
        plugin_stepped = any(
            stepped.get(name, False) for name in ('b1', 'b2', 'b3'))
        if plugin_stepped:
            self.ct_plugin_update_step.add_(1)
            self._ct_plugin_updated_this_epoch = True

        if self.config.optimizer.lower() == 'adamonecycle':
            schedulers = self.lr_schedulers()
            if not isinstance(schedulers, (list, tuple)):
                schedulers = [schedulers]
            for name, scheduler in zip(names, schedulers):
                if stepped[name]:
                    scheduler.step()
        trainer = getattr(self, '_trainer', None)
        if trainer is not None and advance_transaction:
            advance_lightning_manual_transaction(trainer)
        for name in names:
            norm = norms.get(name, torch.zeros((), device=self.device))
            loss_dict.update({
                f'ct_{name}_unscaled_grad_norm': torch.as_tensor(
                    norm, device=self.device),
                f'ct_{name}_amp_scale': torch.tensor(
                    float(self._ct_scalers[name].get_scale()),
                    device=self.device),
                f'ct_{name}_step_applied': torch.tensor(
                    float(stepped[name]), device=self.device),
                f'ct_{name}_update_step': getattr(
                    self, f'ct_{name}_update_step').detach().to(
                        device=self.device, dtype=torch.float32),
            })
        plugin_norms = [norms[name] for name in ('b1', 'b2', 'b3')
                        if name in norms]
        loss_dict['ct_plugin_unscaled_grad_norm'] = (
            torch.stack([torch.as_tensor(value, device=self.device)
                         for value in plugin_norms]).max()
            if plugin_norms else torch.zeros((), device=self.device))
        loss_dict['ct_plugin_step_applied'] = torch.tensor(
            float(plugin_stepped), device=self.device)
        loss_dict['ct_plugin_update_step'] = (
            self.ct_plugin_update_step.detach().to(
                device=self.device, dtype=torch.float32))
        self._ct_last_gradient_norm = {
            name: float(torch.as_tensor(value).detach().cpu())
            for name, value in norms.items()}

    def _ct_b0_auxiliary_microbatch_gradients(self, auxiliary_batch):
        """Accumulate weighted B0-only gradients from auxiliary views."""
        self._ensure_ct_scalers()
        microbatch_size = int(getattr(
            self.config, 'ct_auxiliary_microbatch_size', 16))
        view_ids = auxiliary_batch['b0_view_id'].reshape(-1)
        expected_ids = tuple(range(1, self.ct_b0_candidate_views))
        b0_named = self._ct_named_parameters_by_module.get('b0', [])
        if not b0_named:
            raise RuntimeError("B0 auxiliary views require an active B0")
        if self.ct_auxiliary_rng is None:
            raise RuntimeError("B0 auxiliary RNG is unavailable")
        b0_parameters = [parameter for _, parameter in b0_named]
        accumulated = [None for _ in b0_parameters]
        weighted_losses = []
        per_view_losses = {}
        with self.ct_auxiliary_rng.fork(self.device):
            with freeze_batchnorm_running_stats(self):
                for view_id in expected_ids:
                    row_mask = view_ids == view_id
                    row_count = int(row_mask.sum().item())
                    if row_count != microbatch_size:
                        raise RuntimeError(
                            "candidate-decoupled training requires one "
                            f"complete auxiliary B0 view; view{view_id} has "
                            f"{row_count} rows")
                    candidate_batch = self._slice_batch_rows(
                        auxiliary_batch, row_mask)
                    if not self._b0_auxiliary_batch(candidate_batch):
                        raise RuntimeError(
                            "auxiliary B0 microbatch is not marked B0-only")
                    candidate_output = self(candidate_batch)
                    candidate_loss = self.compute_loss(
                        candidate_batch, candidate_output)
                    view_loss = candidate_loss['loss_b0_transaction']
                    view_weight = self.ct_b0_candidate_weights[view_id]
                    weighted_loss = view_weight * view_loss
                    scaled_loss = self._ct_scalers['b0'].scale(weighted_loss)
                    gradients = torch.autograd.grad(
                        scaled_loss, b0_parameters,
                        retain_graph=False, allow_unused=True)
                    for index, gradient in enumerate(gradients):
                        if gradient is None:
                            continue
                        accumulated[index] = (
                            gradient if accumulated[index] is None
                            else accumulated[index] + gradient)
                    weighted_losses.append(weighted_loss.detach())
                    per_view_losses[f'loss_b0_view{view_id}'] = (
                        view_loss.detach())
        if len(weighted_losses) != len(expected_ids):
            raise RuntimeError(
                "missing one or more auxiliary B0 candidate views")
        aggregated = dict(per_view_losses)
        aggregated['loss_b0_weighted_auxiliary'] = torch.stack(
            weighted_losses).sum()
        return tuple(accumulated), aggregated

    def _ct_candidate_weighted_observation_loss(
            self, batch, output, loss_dict):
        """Apply the registered 1/2 + 3x1/6 B0 branch objective."""
        candidate_id = batch.get('candidate_id')
        if candidate_id is None:
            raise RuntimeError(
                "safe B0 observation batch lacks candidate_id")
        candidate_id = candidate_id.to(
            device=output['aux_estimation_boxes'].device).reshape(-1)
        expected = tuple(range(self.ct_b0_candidate_views))
        observed = tuple(sorted(
            int(value) for value in torch.unique(candidate_id).tolist()))
        if observed != expected:
            raise RuntimeError(
                "safe B0 batch must contain every candidate view exactly "
                f"once as a branch; observed {observed}")
        counts = [int((candidate_id == view_id).sum().item())
                  for view_id in expected]
        if len(set(counts)) != 1:
            raise RuntimeError(
                "safe B0 batch must be candidate-balanced; observed "
                + ",".join(str(value) for value in counts))
        branch_losses = []
        for view_id, weight in enumerate(self.ct_b0_candidate_weights):
            rows = candidate_id == view_id
            branch_batch = self._slice_batch_rows(batch, rows)
            branch_output = self._slice_batch_rows(output, rows)
            branch_loss = self.compute_loss(
                branch_batch, branch_output)['loss_b0_transaction']
            branch_losses.append(branch_loss)
            loss_dict[f'loss_b0_view{view_id}'] = branch_loss.detach()
        weighted = weighted_candidate_sum(
            branch_losses, self.ct_b0_candidate_weights)
        loss_dict['loss_total'] = weighted
        loss_dict['loss_b0_transaction'] = weighted
        loss_dict['ct_canonical_b0_weight'] = weighted.new_tensor(
            self.ct_b0_candidate_weights[0])
        return weighted

    def _forward_safe_mechanism(self, batch):
        """Build detached B0 contracts, then train the canonical plugins."""
        previous = bool(getattr(
            self, '_ct_observation_only_forward', False))
        self._ct_observation_only_forward = True
        try:
            original_modes = [(module, module.training) for module in self.modules()]
            if getattr(self, 'ct_enable_v27', False):
                self.eval()
            with torch.no_grad():
                observation_output = self(batch)
        finally:
            if 'original_modes' in locals():
                for module, training in original_modes:
                    module.training = training
            self._ct_observation_only_forward = previous

        required_motion = (
            'motion_main_ref_boxs', 'motion_main_delta_t',
            'motion_main_current_delta_t', 'motion_main_valid_mask')
        missing = [key for key in required_motion if key not in batch]
        if missing:
            raise KeyError(
                "safe mechanism input is missing: " + ", ".join(missing))
        motion_ref = batch['motion_main_ref_boxs']
        if bool(getattr(self.config, 'shuffle_b1_signal', False)):
            motion_ref = torch.flip(motion_ref, dims=(1,))
        if not self.ct_enable_b1:
            main_motion = self.physical_motion_encoder.kinematic_fallback(
                motion_ref, batch['motion_main_delta_t'],
                batch['motion_main_valid_mask'],
                batch['motion_main_current_delta_t'])
        else:
            main_motion = self.physical_motion_encoder(
                motion_ref, batch['motion_main_delta_t'],
                batch['motion_main_valid_mask'],
                batch['motion_main_current_delta_t'],
                **({'acquisition_features': batch['motion_acquisition_features']}
                   if getattr(self, 'ct_enable_v27', False) else {}))
        if 'replay_b1_contract_present' in batch:
            present = batch['replay_b1_contract_present'].to(
                device=main_motion['mu_xy'].device).reshape(-1) > 0.5
            for motion_key, replay_key in (
                    ('mu_xy', 'replay_b1_mu_xy'),
                    ('direction_xy', 'replay_b1_direction_xy'),
                    ('log_sigma_parallel_perp',
                     'replay_b1_log_sigma_parallel_perp'),
                    ('gap_ratio', 'replay_b1_gap_ratio'),
                    ('valid', 'replay_b1_valid')):
                if replay_key not in batch:
                    raise RuntimeError(
                        f"recursive replay is missing {replay_key}")
                actual = main_motion[motion_key][present]
                expected = batch[replay_key].to(
                    device=actual.device, dtype=actual.dtype)[present]
                expected = expected.reshape_as(actual)
                if not torch.allclose(
                        actual, expected, atol=1e-5, rtol=1e-5):
                    max_error = torch.max(torch.abs(
                        actual - expected)).detach().cpu().item()
                    raise RuntimeError(
                        "recursive replay B1 mismatch for "
                        f"{motion_key}: max_error={max_error:.3e}")
        if bool(getattr(self.config, 'force_b1_invalid', False)):
            main_motion = dict(main_motion)
            main_motion['valid'] = torch.zeros_like(main_motion['valid'])
            main_motion['source_id'] = torch.zeros_like(
                main_motion['source_id'])
        origin_xy = batch['motion_main_ref_boxs'][:, 0, :2].to(
            device=main_motion['prior_xy'].device,
            dtype=main_motion['prior_xy'].dtype)
        observation_output.update({
            'motion_prior_xy': main_motion['prior_xy'],
            'motion_prior_origin_xy': origin_xy,
            'motion_prior_proposal_xy': main_motion['prior_xy'],
            'motion_prior_basis_velocity_xy': main_motion[
                'basis_velocity_xy'],
            'motion_prior_velocity_xy': main_motion['velocity_xy'],
            'motion_prior_kinematic_xy': main_motion['kinematic_prior_xy'],
            'motion_prior_residual_xy': main_motion['residual_xy'],
            'motion_prior_residual_unit_parallel_perp': main_motion[
                'residual_unit_parallel_perp'],
            'motion_prior_envelope_parallel_perp': main_motion[
                'envelope_parallel_perp'],
            'motion_prior_acquisition_margin_parallel_perp': main_motion[
                'acquisition_margin_parallel_perp'],
            'motion_prior_valid': main_motion['valid'],
            'motion_prior_log_sigma_xy': main_motion['log_sigma_xy'],
            'motion_prior_log_sigma_parallel_perp': main_motion[
                'log_sigma_parallel_perp'],
            'motion_prior_covariance_xy': main_motion['covariance_xy'],
            'motion_prior_direction_xy': main_motion[
                'motion_direction_xy'],
            'motion_prior_source_id': main_motion['source_id'],
            'motion_prior_gap_ratio': main_motion['gap_ratio'],
        })
        if self.ct_enable_b1 and 'motion_aux_ref_boxs' in batch:
            aux_motion = self.physical_motion_encoder(
                batch['motion_aux_ref_boxs'], batch['motion_aux_delta_t'],
                batch['motion_aux_valid_mask'],
                batch['motion_aux_current_delta_t'],
                **({'acquisition_features': batch['motion_aux_acquisition_features']}
                   if getattr(self, 'ct_enable_v27', False) else {}))
            observation_output.update({
                'motion_aux_prior_xy': aux_motion['prior_xy'],
                'motion_aux_prior_velocity_xy': aux_motion['velocity_xy'],
                'motion_aux_prior_kinematic_xy': aux_motion[
                    'kinematic_prior_xy'],
                'motion_aux_prior_residual_unit_parallel_perp': aux_motion[
                    'residual_unit_parallel_perp'],
                'motion_aux_prior_envelope_parallel_perp': aux_motion[
                    'envelope_parallel_perp'],
                'motion_aux_prior_acquisition_margin_parallel_perp': aux_motion[
                    'acquisition_margin_parallel_perp'],
                'motion_aux_prior_valid': aux_motion['valid'],
                'motion_aux_prior_gap_ratio': aux_motion['gap_ratio'],
                'motion_aux_prior_log_sigma_xy': aux_motion['log_sigma_xy'],
                'motion_aux_prior_log_sigma_parallel_perp': aux_motion[
                    'log_sigma_parallel_perp'],
                'motion_aux_prior_direction_xy': aux_motion[
                    'motion_direction_xy'],
            })

        observation_box = observation_output[
            'observation_aux_estimation_boxes']
        if self.ct_enable_b2:
            batch_size = int(batch['points'].shape[0])
            frame_count = int(batch['valid_mask'].shape[1]) + 1
            chunk_size = int(batch['points'].shape[1]) // frame_count
            history_valid_ratio = batch['motion_main_valid_mask'].to(
                device=observation_box.device,
                dtype=observation_box.dtype).mean(dim=1)
            final_box = self._forward_ct_contract_v3(
                batch, observation_output, observation_box,
                observation_output['obs_stats'], main_motion,
                history_valid_ratio, batch_size, frame_count, chunk_size,
                coarse_box=observation_output['estimation_boxes'])
        else:
            final_box = observation_box
        observation_output['aux_estimation_boxes'] = final_box
        observation_output['ct_final_box'] = final_box
        return observation_output

    def training_step(self, batch, batch_idx):
        """
        Args:
            batch: {
            "points": stack_frames, (B,N,3+9+1)
            "seg_label": stack_label,
            "box_label": np.append(this_gt_bb_transform.center, theta),
            "box_size": this_gt_bb_transform.wlh
        }
        Returns:

        """
        if (isinstance(batch, dict)
                and batch.get('ct_stream_schema') in (
                    'ct_seqtrack.dual_stream.v1',
                    'ct_seqtrack.train.v2',
                    'ct_seqtrack.train.v3', 'ct_seqtrack.train.v4')):
            if getattr(self, '_ct_inside_dual_stream', False):
                raise RuntimeError('nested dual-stream transaction')
            self._ct_inside_dual_stream = True
            saved_routing = (
                self.use_ct_joint_full, self.use_b1motion_v3,
                self.ct_enable_b1, self.ct_enable_b2, self.ct_enable_b3)
            try:
                if self.ct_unified_auto:
                    self._ct_record_observation_batch_fingerprint(
                        batch['observation'])
                # The observation transaction is behavior-identical to the
                # clean SeqTrack B0: plugins are absent from its graph and B0
                # owns the only logical Lightning step.
                self.use_ct_joint_full = False
                self.use_b1motion_v3 = False
                self.ct_enable_b1 = False
                self.ct_enable_b2 = False
                self.ct_enable_b3 = False
                self._ct_transaction_modules = {'b0'}
                self._ct_auto_return_modules = {'b0'}
                self._ct_advance_transaction = True
                observation_loss = self.training_step(
                    batch['observation'], batch_idx)
            finally:
                (
                    self.use_ct_joint_full, self.use_b1motion_v3,
                    self.ct_enable_b1, self.ct_enable_b2, self.ct_enable_b3,
                ) = saved_routing

            mechanism_batch = batch.get('mechanism')
            if mechanism_batch is not None:
                rng_state = capture_global_rng_state()
                try:
                    self._ct_transaction_modules = {
                        name for name in ('b1', 'b2', 'b3')
                        if name in getattr(self, '_ct_optimizer_names', ())}
                    self._ct_auto_return_modules = set(
                        self._ct_transaction_modules)
                    self._ct_advance_transaction = False
                    self._ct_mechanism_transaction = True
                    self._ct_safe_mechanism_forward = self.ct_unified_auto
                    with freeze_batchnorm_running_stats(
                            self,
                            excluded_prefixes=(
                                'physical_motion_encoder',
                                'ct_joint_search_refiner',
                                'ct_joint_router')):
                        if (self.ct_enable_v27 and isinstance(mechanism_batch, dict)
                                and 'ct_mechanism_sequence' in mechanism_batch):
                            ticks = mechanism_batch['ct_mechanism_sequence']
                            row_counts = [len(tick) for tick in ticks]
                            total_rows = sum(row_counts)
                            mechanism_loss = sum(
                                self.training_step(tick, batch_idx) * (count / total_rows)
                                for tick, count in zip(ticks, row_counts))
                        else:
                            mechanism_loss = self.training_step(
                                mechanism_batch, batch_idx)
                finally:
                    restore_global_rng_state(rng_state)
                    self._ct_mechanism_transaction = False
                    self._ct_safe_mechanism_forward = False
            else:
                mechanism_loss = None
            self._ct_transaction_modules = None
            self._ct_auto_return_modules = None
            self._ct_advance_transaction = True
            self._ct_inside_dual_stream = False
            if self.ct_unified_auto and mechanism_loss is not None:
                return observation_loss + mechanism_loss
            return observation_loss

        online_batch = (
            isinstance(batch, list) and batch
            and isinstance(batch[0], dict)
            and batch[0].get('online_recursive_raw', False))
        if online_batch and self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        online_step_start = time.perf_counter() if online_batch else None
        auxiliary_batch = None
        auxiliary_gradients = {}
        if online_batch:
            batch, auxiliary_batch = self._prepare_online_recursive_batch(
                batch)
            if (self.ct_joint_contract_version >= 3
                    and not bool((batch['b0_view_id'] == 0).all())):
                raise RuntimeError(
                    "canonical transaction contains an auxiliary B0 view")
        amp_enabled = bool(
            self.ct_separate_optimizers
            and self.ct_manual_amp_enabled
            and self.device.type == 'cuda')
        autocast_context = (
            torch.autocast(device_type='cuda', dtype=torch.float16)
            if amp_enabled else contextlib.nullcontext())
        with autocast_context:
            output = (
                self._forward_safe_mechanism(batch)
                if getattr(self, '_ct_safe_mechanism_forward', False)
                else self(batch))
            self._ct_record_cuda_stage('forward')
            if online_batch:
                self._attach_h3_shadow_labels(batch, output)
            loss_dict = self.compute_loss(batch, output)
            if (self.ct_unified_auto and not online_batch
                    and getattr(self, '_ct_auto_return_modules', None)
                    == {'b0'}):
                self._ct_candidate_weighted_observation_loss(
                    batch, output, loss_dict)
            if online_batch:
                self._ct_record_acquisition_supply(
                    loss_dict, 'candidate0')
            if auxiliary_batch is not None:
                auxiliary_b0_gradients, auxiliary_loss = (
                    self._ct_b0_auxiliary_microbatch_gradients(
                        auxiliary_batch))
                auxiliary_gradients['b0'] = auxiliary_b0_gradients
                canonical_b0 = loss_dict['loss_b0_transaction']
                canonical_weight = self.ct_b0_candidate_weights[0]
                combined_b0 = (
                    canonical_weight * canonical_b0.detach()
                    + auxiliary_loss[
                        'loss_b0_weighted_auxiliary'].detach())
                loss_dict['loss_total'] = (
                    loss_dict['loss_total'] - canonical_b0
                    + combined_b0)
                loss_dict['loss_b0_candidate0'] = canonical_b0.detach()
                loss_dict['loss_b0_auxiliary'] = auxiliary_loss[
                    'loss_b0_weighted_auxiliary']
                for key, value in auxiliary_loss.items():
                    if key != 'loss_b0_weighted_auxiliary':
                        loss_dict[key] = value
                loss_dict['ct_canonical_b0_weight'] = canonical_b0.new_tensor(
                    canonical_weight)
            self._ct_record_cuda_stage('loss')
        self._accumulate_joint_binary_rows(batch, output)
        loss = loss_dict['loss_total']
        if self.ct_unified_auto:
            active_modules = getattr(
                self, '_ct_auto_return_modules', None)
            if active_modules:
                loss = sum(
                    (loss_dict[f'loss_{name}_transaction']
                     for name in ('b0', 'b1', 'b2', 'b3')
                     if name in active_modules),
                    output['aux_estimation_boxes'].new_zeros(()))
                loss_dict['loss_total'] = loss
        if online_batch:
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            online_step_ms = max(
                (time.perf_counter() - online_step_start) * 1000.0, 1e-6)
            shadow_ms = batch['ct_shadow_time_ms'].detach().to(
                device=loss.device, dtype=loss.dtype)
            loss_dict['ct_online_step_time_ms'] = loss.new_tensor(
                online_step_ms)
            loss_dict['ct_shadow_step_latency_ratio'] = (
                shadow_ms / online_step_ms)

        if self.is_paired_batch(batch):
            metric_batch = batch["view_a"]
            metric_output = output["view_a"]
        else:
            metric_batch = batch
            metric_output = output

        # log; mechanism-only forwards must not pollute B0 accuracy meters.
        log_batch_size = int(metric_batch['seg_label'].shape[0])
        if not getattr(self, '_ct_mechanism_transaction', False):
            seg_acc = self.seg_acc(torch.argmax(metric_output['seg_logits'], dim=1, keepdim=False),
                                   metric_batch['seg_label'])
            self.log('seg_acc_background/train', seg_acc[0], on_step=True,
                     on_epoch=True, prog_bar=False, logger=True,
                     batch_size=log_batch_size)
            self.log('seg_acc_foreground/train', seg_acc[1], on_step=True,
                     on_epoch=True, prog_bar=False, logger=True,
                     batch_size=log_batch_size)
            if self.use_motion_cls:
                motion_acc = self.motion_acc(torch.argmax(metric_output['motion_cls'], dim=1, keepdim=False),
                                             metric_batch['motion_state_label'][:,0]) # 0 represents motion relative to the first historical box
                self.log('motion_acc_static/train', motion_acc[0], on_step=True,
                         on_epoch=True, prog_bar=False, logger=True,
                         batch_size=log_batch_size)
                self.log('motion_acc_dynamic/train', motion_acc[1], on_step=True,
                         on_epoch=True, prog_bar=False, logger=True,
                         batch_size=log_batch_size)

        log_dict = {k: v.item() for k, v in loss_dict.items()}

        self.logger.experiment.add_scalars('loss', log_dict,
                                           global_step=self.global_step)
        if (online_batch and 'ct_router_gate' in output
                and hasattr(self.logger.experiment, 'add_histogram')):
            self.logger.experiment.add_histogram(
                'ct/router_probability_histogram',
                output['ct_router_gate'].detach(),
                global_step=self.global_step)

        if online_batch:
            self._commit_online_recursive_predictions(output)

        if self.ct_separate_optimizers:
            self._ct_isolated_optimizer_step(
                loss_dict, auxiliary_gradients,
                active_modules=getattr(
                    self, '_ct_transaction_modules', None),
                advance_transaction=getattr(
                    self, '_ct_advance_transaction', True))
            return loss.detach()
        return loss
