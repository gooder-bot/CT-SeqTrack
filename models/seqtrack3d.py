from datasets import points_utils
from models import base_model
from models.backbone.pointnet import MiniPointNet, SegPointNet, FeaturePointNet
from models.attn.Models import Seq2SeqFormer

import copy

import torch
from torch import nn
import torch.nn.functional as F

from torchmetrics import Accuracy

from datasets.misc_utils import get_tensor_corners_batch
from datasets.misc_utils import create_corner_timestamps_from_deltas
from models.dynamics import (
    DynamicsEncoder,
    DynamicsResidualGate,
    ZeroInitPhysicalTimeAdapter,
    apply_proposal_innovation,
    clamp_vector_norm,
)
from models.observability import ObservabilityGate
from models.time_encoding import TimeEncoding
from models.path_distillation import (
    endpoint_path_terms,
    freeze_batchnorm_running_stats,
    teacher_endpoint_confidence_terms,
    update_ema_module,
)
from models.ct_v2 import (
    AdvantageGatedProposalFusion,
    ContinuousTimeMotionEncoder,
    JointProposalFusion,
    OrderedPhysicalMotionEncoder,
    OrderedTrajectoryEncoder,
    PointFeatureTemporalConsistencyLoss,
    ProposalFusionGate,
    ReliabilityGatedProposalFusion,
    TrajectorySearchEvidence,
    TrajectorySearchEvidenceV21,
    TrajectoryPointEncoder,
    ZeroInitTrajectoryAdapter,
)
from models.ct_v2.contracts import (
    build_search_usable_mask,
    resolve_observation_delta_t,
)

# import vis_tool as vt

class SEQTRACK3D(base_model.MotionBaseModelMF):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.hist_num = getattr(config, 'hist_num', 1)
        self.seg_acc = Accuracy(task='multiclass',num_classes=2, average='none')

        self.box_aware = getattr(config, 'box_aware', False)
        self.use_motion_cls = getattr(config, 'use_motion_cls', True)
        self.use_dynamics_encoder = getattr(config, 'use_dynamics_encoder', False)
        if bool(getattr(config, 'dynamics_use_acceleration', False)):
            raise ValueError(
                "dynamics_use_acceleration is not implemented or consumed by "
                "DynamicsEncoder; keep it false to avoid a misleading ablation.")
        self.use_observability_gate = getattr(config, 'use_observability_gate', False)
        self.use_ct_v2 = bool(getattr(config, 'use_ct_v2', False))
        self.use_b1motion_v3 = bool(getattr(
            config, 'use_b1motion_v3', False))
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
        self.proposal_inference_mode = str(getattr(
            config, 'proposal_inference_mode', 'full')).strip().lower()
        if self.proposal_inference_mode not in (
                'obs', 'obs_motion', 'obs_search', 'full'):
            raise ValueError(
                "proposal_inference_mode must be obs, obs_motion, "
                "obs_search, or full")
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
        self.use_m3_path_distillation = bool(getattr(
            config, 'use_m3_path_distillation', False))
        self.use_point_feature_tc = bool(getattr(
            config, 'use_point_feature_tc', False))
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
        if self.use_point_feature_tc and (
                bool(getattr(config, 'use_twc', False))
                or self.use_m3_path_distillation):
            raise ValueError(
                "PFTC, legacy paired-view TWC, and M3 EMA path distillation "
                "must be evaluated as separate training objectives.")
        self.m3_irregular_supervision_weight = float(getattr(
            config, 'm3_irregular_supervision_weight', 0.0))
        self.m3_path_weight = float(getattr(config, 'm3_path_weight', 0.0))
        self.m3_theta_weight = float(getattr(config, 'm3_theta_weight', 0.5))
        self.m3_coarse_weight = float(getattr(
            config, 'm3_coarse_weight', 0.0))
        self.m3_teacher_momentum = float(getattr(
            config, 'm3_teacher_momentum', 0.996))
        self.m3_teacher_update_interval = int(getattr(
            config, 'm3_teacher_update_interval', 1))
        self.m3_teacher_confidence_mode = str(getattr(
            config, 'm3_teacher_confidence_mode', 'foreground_topk'))
        self.m3_teacher_confidence_topk = int(getattr(
            config, 'm3_teacher_confidence_topk', 32))
        self.m3_teacher_confidence_floor = float(getattr(
            config, 'm3_teacher_confidence_floor', 0.05))
        self.m3_teacher_agreement_center_scale = float(getattr(
            config, 'm3_teacher_agreement_center_scale', 1.0))
        self.m3_teacher_agreement_yaw_scale = float(getattr(
            config, 'm3_teacher_agreement_yaw_scale', 0.5))
        self.m3_freeze_irregular_bn_stats = bool(getattr(
            config, 'm3_freeze_irregular_bn_stats', True))
        self.m3_warmup_epoch = int(getattr(config, 'm3_warmup_epoch', 0))
        self.m3_ramp_epochs = int(getattr(config, 'm3_ramp_epochs', 5))
        if self.use_m3_path_distillation and bool(getattr(config, 'use_twc', False)):
            raise ValueError(
                "M3 asymmetric path distillation and legacy symmetric TWC "
                "must be evaluated as separate objectives.")
        if min(
                self.m3_irregular_supervision_weight,
                self.m3_path_weight,
                self.m3_theta_weight,
                self.m3_coarse_weight) < 0:
            raise ValueError("M3 loss weights must be non-negative")
        if not 0.0 <= self.m3_teacher_momentum < 1.0:
            raise ValueError("m3_teacher_momentum must be in [0, 1)")
        if self.m3_teacher_update_interval <= 0:
            raise ValueError("m3_teacher_update_interval must be positive")
        if self.m3_warmup_epoch < 0 or self.m3_ramp_epochs <= 0:
            raise ValueError("M3 warmup must be non-negative and ramp positive")
        if not 0.0 <= self.m3_teacher_confidence_floor <= 1.0:
            raise ValueError("m3_teacher_confidence_floor must be in [0, 1]")
        if self.m3_teacher_confidence_topk <= 0:
            raise ValueError("m3_teacher_confidence_topk must be positive")
        if min(
                self.m3_teacher_agreement_center_scale,
                self.m3_teacher_agreement_yaw_scale) <= 0:
            raise ValueError("M3 teacher agreement scales must be positive")
        if (self.m3_teacher_confidence_mode.strip().lower().replace("-", "_")
                not in (
                    "fixed", "uniform", "ones", "foreground",
                    "foreground_topk", "segmentation", "agreement",
                    "proposal_agreement", "hybrid")):
            raise ValueError("unsupported m3_teacher_confidence_mode")
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
                'use_twc': bool(getattr(config, 'use_twc', False)),
                'use_m3_path_distillation': self.use_m3_path_distillation,
                'use_m4_state_filter': bool(getattr(
                    config, 'use_m4_state_filter', False)),
                'use_m4_trajectory_tube': bool(getattr(
                    config, 'use_m4_trajectory_tube', False)),
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
                'use_twc': bool(getattr(config, 'use_twc', False)),
                'use_m3_path_distillation': self.use_m3_path_distillation,
                'use_point_feature_tc': self.use_point_feature_tc,
                'use_m4_state_filter': bool(getattr(
                    config, 'use_m4_state_filter', False)),
                'use_m4_trajectory_tube': bool(getattr(
                    config, 'use_m4_trajectory_tube', False)),
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
        if (self.use_advantage_proposal_fusion
                and not self.use_search_evidence_v21):
            raise ValueError(
                "advantage proposal fusion requires Search Evidence v2.1")
        if (self.use_search_evidence_v21
                and not self.use_advantage_proposal_fusion):
            raise ValueError(
                "Search Evidence v2.1 requires advantage proposal fusion")
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
                self.motion_v3_fused_weight,
                self.motion_v3_gate_weight,
                self.motion_v3_help_margin) < 0:
            raise ValueError("B1motion-v3 loss settings must be non-negative")
        if (not self.motion_v3_aux_query_gaps
                or any(value <= 0 for value in
                       self.motion_v3_aux_query_gaps)):
            raise ValueError("B1motion-v3 auxiliary query gaps must be positive")
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
            self.motion_acc = Accuracy(task='multiclass',num_classes=2, average='none')

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
            self.physical_motion_encoder = OrderedPhysicalMotionEncoder(
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
            )
            if self.use_motion_v3_legacy_fusion:
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

        self.m3_teacher = None
        if self.use_m3_path_distillation:
            teacher_config = copy.deepcopy(config)
            teacher_config.use_m3_path_distillation = False
            teacher_config.use_twc = False
            teacher_config.m3_path_weight = 0.0
            self.m3_teacher = SEQTRACK3D(
                teacher_config,
                train_dataloader_length=kwargs.get(
                    'train_dataloader_length', None),
            )
            for parameter in self.m3_teacher.parameters():
                parameter.requires_grad_(False)
            self.m3_teacher.eval()
            self.register_buffer(
                "m3_teacher_updates",
                torch.zeros((), dtype=torch.long),
            )
            self._m3_last_ema_global_step = -1

    @torch.no_grad()
    def _initialize_m3_teacher(self, force=False):
        if not self.use_m3_path_distillation:
            return
        if not force and int(self.m3_teacher_updates.item()) > 0:
            return
        student_state = {
            key: value
            for key, value in self.state_dict().items()
            if not key.startswith("m3_teacher.")
            and key != "m3_teacher_updates"
        }
        teacher_state = self.m3_teacher.state_dict()
        matched = {
            key: value
            for key, value in student_state.items()
            if key in teacher_state
            and hasattr(value, "shape")
            and teacher_state[key].shape == value.shape
        }
        teacher_state.update(matched)
        self.m3_teacher.load_state_dict(teacher_state, strict=True)
        self.m3_teacher.eval()

    def on_fit_start(self):
        if self.use_m3_path_distillation:
            self._initialize_m3_teacher()

    def on_load_checkpoint(self, checkpoint):
        if self.use_m3_path_distillation:
            return
        state_dict = checkpoint.get("state_dict", {})
        teacher_keys = [
            key for key in state_dict
            if key.startswith("m3_teacher.")
            or key == "m3_teacher_updates"
        ]
        for key in teacher_keys:
            state_dict.pop(key)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if not self.use_m3_path_distillation:
            return
        current_step = int(self.global_step)
        if current_step == self._m3_last_ema_global_step:
            return
        if current_step % self.m3_teacher_update_interval != 0:
            return
        self._initialize_m3_teacher()
        update_ema_module(
            self.m3_teacher,
            self,
            self.m3_teacher_momentum,
        )
        self.m3_teacher_updates.add_(1)
        self._m3_last_ema_global_step = current_step
        self.m3_teacher.eval()

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
            if (self.use_m3_path_distillation
                    and self.training
                    and self.m3_freeze_irregular_bn_stats):
                with freeze_batchnorm_running_stats(self):
                    output_b = self.forward(input_dict["view_b"])
            else:
                output_b = self.forward(input_dict["view_b"])
            paired_output = {
                "view_a": output_a,
                "view_b": output_b,
            }
            if self.use_m3_path_distillation and self.training:
                self._initialize_m3_teacher()
                self.m3_teacher.eval()
                with torch.no_grad():
                    paired_output["teacher_a"] = self.m3_teacher(
                        input_dict["view_a"])
            return paired_output

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
        feature_result = self.feature_pointnet(
            solo_x, return_point_features=collect_pftc_features)
        if collect_pftc_features:
            feature, point_aligned_feature = feature_result
            output_dict["pftc_point_features"] = (
                point_aligned_feature.transpose(1, 2).reshape(
                    B, L, chunk_size, -1))
        else:
            feature = feature_result
        # (B*num) * C * N; N is the fixed Transformer token count per frame.
        feature = feature.transpose(1,2)
        NEW_N = feature.shape[1]
        points_feature = feature.reshape(B,L*NEW_N,-1)

        delta_motion = self.Transformer(box_seq_corners,points_feature,input_dict["valid_mask"])  #B*4*4

        updated_ref_boxs = delta_motion[:,:HL,:]
        updated_aux_box =  delta_motion[:,-1,:]

        observation_aux_box = updated_aux_box
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
            main_motion = self.physical_motion_encoder(
                input_dict['motion_main_ref_boxs'],
                input_dict['motion_main_delta_t'],
                input_dict['motion_main_valid_mask'],
                input_dict['motion_main_current_delta_t'],
            )
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
                'motion_prior_velocity_xy': main_motion['velocity_xy'],
                'motion_prior_kinematic_xy':
                    main_motion['kinematic_prior_xy'],
                'motion_prior_valid': main_motion['valid'],
                'motion_prior_log_sigma_xy': main_motion['log_sigma_xy'],
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
            if self.training and 'motion_aux_ref_boxs' in input_dict:
                aux_motion = self.physical_motion_encoder(
                    input_dict['motion_aux_ref_boxs'],
                    input_dict['motion_aux_delta_t'],
                    input_dict['motion_aux_valid_mask'],
                    input_dict['motion_aux_current_delta_t'],
                )
                output_dict.update({
                    'motion_aux_prior_xy': aux_motion['prior_xy'],
                    'motion_aux_prior_velocity_xy':
                        aux_motion['velocity_xy'],
                    'motion_aux_prior_kinematic_xy':
                        aux_motion['kinematic_prior_xy'],
                    'motion_aux_prior_valid': aux_motion['valid'],
                    'motion_aux_prior_gap_ratio': aux_motion['gap_ratio'],
                    'motion_aux_prior_log_sigma_xy':
                        aux_motion['log_sigma_xy'],
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

            updated_aux_box, advantage_diagnostics = (
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
            output_dict.update(advantage_diagnostics)
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


        output_dict["estimation_boxes"] = aux_box
        output_dict.update({"seg_logits": seg_logits,
                            "motion_pred": motion_pred,
                            'observation_aux_estimation_boxes':
                                observation_aux_box,
                            'aux_estimation_boxes': updated_aux_box,
                            'ref_boxs': input_dict['ref_boxs'],
                            'valid_mask':input_dict["valid_mask"],
                            'updated_ref_boxs':updated_ref_boxs,
                            })
        output_dict.update(obs_aux)
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

    def _compute_pair_validity(self, output_a, output_b, data_a, data_b):
        box_a = output_a["aux_estimation_boxes"]
        box_b = output_b["aux_estimation_boxes"]
        valid = torch.ones(
            box_a.shape[0], device=box_a.device, dtype=torch.bool)

        def paired_setting(suffix, default):
            if self.use_m3_path_distillation:
                m3_key = f"m3_{suffix}"
                if hasattr(self.config, m3_key):
                    return getattr(self.config, m3_key)
            return getattr(self.config, f"twc_{suffix}", default)

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

    def compute_twc_loss(self, output_a, output_b, data_a, data_b):
        box_a = output_a["aux_estimation_boxes"]
        box_b = output_b["aux_estimation_boxes"]

        center_a, center_b = box_a[:, :3], box_b[:, :3]
        theta_a, theta_b = box_a[:, 3], box_b[:, 3]

        loss_center_per_sample = F.smooth_l1_loss(center_a, center_b, reduction="none").mean(dim=1)
        loss_theta_per_sample = (
            F.smooth_l1_loss(torch.sin(theta_a), torch.sin(theta_b), reduction="none")
            + F.smooth_l1_loss(torch.cos(theta_a), torch.cos(theta_b), reduction="none")
        )
        loss_twc_per_sample = (
            loss_center_per_sample
            + getattr(self.config, "twc_theta_weight", 0.5) * loss_theta_per_sample
        )

        pair_validity = self._compute_pair_validity(
            output_a, output_b, data_a, data_b)
        valid = pair_validity["valid"]
        anchor_gap = pair_validity["anchor_gap"]
        current_point_gap = pair_validity["current_point_gap"]

        valid_float = valid.to(dtype=box_a.dtype)
        valid_count = valid_float.sum().clamp_min(1.0)
        loss_twc = (loss_twc_per_sample * valid_float).sum() / valid_count

        center_gap = (torch.linalg.norm(center_a - center_b, dim=1) * valid_float).sum() / valid_count
        angle_gap = torch.abs(torch.atan2(torch.sin(theta_a - theta_b), torch.cos(theta_a - theta_b)))
        angle_gap = (angle_gap * valid_float).sum() / valid_count

        return {
            "loss_twc": loss_twc,
            "twc_valid_ratio": valid_float.mean(),
            "twc_center_gap": center_gap,
            "twc_angle_gap": angle_gap,
            "twc_anchor_gap": anchor_gap.mean(),
            "twc_anchor_gap_max": anchor_gap.max(),
            "twc_current_point_gap": current_point_gap.mean(),
            "twc_current_point_gap_max": current_point_gap.max(),
        }

    def compute_m3_path_loss(self, teacher_output, student_output,
                             data_a, data_b):
        """Distil the canonical-history endpoint into the irregular path.

        The teacher branch is detached inside ``endpoint_path_terms`` and the
        confidence is computed without labels, so M3 cannot leak GT quality
        into its sample weights.
        """
        refined_terms = endpoint_path_terms(
            teacher_output["aux_estimation_boxes"],
            student_output["aux_estimation_boxes"],
            theta_weight=self.m3_theta_weight,
        )
        coarse_terms = endpoint_path_terms(
            teacher_output["estimation_boxes"],
            student_output["estimation_boxes"],
            theta_weight=self.m3_theta_weight,
        )
        combined_path = (
            refined_terms["total"]
            + self.m3_coarse_weight * coarse_terms["total"]
        ) / (1.0 + self.m3_coarse_weight)
        pair_validity = self._compute_pair_validity(
            teacher_output, student_output, data_a, data_b)
        valid = pair_validity["valid"]
        confidence_terms = teacher_endpoint_confidence_terms(
            teacher_output,
            point_sample_size=int(getattr(
                self.config, "point_sample_size", 0)),
            mode=self.m3_teacher_confidence_mode,
            topk=self.m3_teacher_confidence_topk,
            floor=self.m3_teacher_confidence_floor,
            agreement_center_scale=self.m3_teacher_agreement_center_scale,
            agreement_yaw_scale=self.m3_teacher_agreement_yaw_scale,
        )
        confidence = confidence_terms["confidence"].to(
            dtype=combined_path.dtype)

        valid_float = valid.to(dtype=combined_path.dtype)
        sample_weight = valid_float * confidence
        weight_sum = sample_weight.sum().clamp_min(1e-6)

        def weighted_mean(value):
            return (value * sample_weight).sum() / weight_sum

        history_gap = pair_validity["history_gap"]
        if history_gap is None:
            history_gap_mean = combined_path.new_zeros(())
        else:
            history_gap_mean = weighted_mean(history_gap)

        return {
            "loss_m3_path": weighted_mean(combined_path),
            "m3_center_loss": weighted_mean(refined_terms["center_loss"]),
            "m3_yaw_loss": weighted_mean(refined_terms["yaw_loss"]),
            "m3_center_gap": weighted_mean(refined_terms["center_gap"]),
            "m3_yaw_gap": weighted_mean(refined_terms["yaw_gap"]),
            "m3_coarse_center_loss": weighted_mean(
                coarse_terms["center_loss"]),
            "m3_coarse_yaw_loss": weighted_mean(coarse_terms["yaw_loss"]),
            "m3_coarse_center_gap": weighted_mean(
                coarse_terms["center_gap"]),
            "m3_coarse_yaw_gap": weighted_mean(coarse_terms["yaw_gap"]),
            "m3_coarse_weight": combined_path.new_tensor(
                self.m3_coarse_weight),
            "m3_valid_ratio": valid_float.mean(),
            "m3_teacher_confidence": (
                confidence * valid_float).sum()
                / valid_float.sum().clamp_min(1.0),
            "m3_teacher_foreground_confidence": weighted_mean(
                confidence_terms["foreground"].to(
                    dtype=combined_path.dtype)),
            "m3_teacher_agreement_confidence": weighted_mean(
                confidence_terms["agreement"].to(
                    dtype=combined_path.dtype)),
            "m3_teacher_internal_center_gap": weighted_mean(
                confidence_terms["coarse_refined_center_gap"].to(
                    dtype=combined_path.dtype)),
            "m3_teacher_internal_yaw_gap": weighted_mean(
                confidence_terms["coarse_refined_yaw_gap"].to(
                    dtype=combined_path.dtype)),
            "m3_effective_sample_weight": sample_weight.mean(),
            "m3_anchor_gap": pair_validity["anchor_gap"].mean(),
            "m3_current_point_gap": (
                pair_validity["current_point_gap"].mean()),
            "m3_history_gap": history_gap_mean,
        }

    def _m3_effective_path_weight(self):
        current_epoch = int(getattr(self, "current_epoch", 0))
        if current_epoch < self.m3_warmup_epoch:
            return 0.0
        ramp_progress = (
            current_epoch - self.m3_warmup_epoch + 1
        ) / float(self.m3_ramp_epochs)
        return self.m3_path_weight * min(max(ramp_progress, 0.0), 1.0)

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

        if self.use_m3_path_distillation:
            if "teacher_a" not in output:
                raise KeyError(
                    "M3 training requires a canonical EMA teacher output.")
            beta = self.m3_irregular_supervision_weight
            loss_total_sup = (
                loss_a["loss_total"] + beta * loss_b["loss_total"]
            ) / (1.0 + beta)
            m3_loss_dict = self.compute_m3_path_loss(
                output["teacher_a"], output_b, data_a, data_b)
            effective_weight = self._m3_effective_path_weight()
            loss_total = (
                loss_total_sup
                + effective_weight * m3_loss_dict["loss_m3_path"]
            )
            loss_dict = {
                "loss_total": loss_total,
                "loss_total_sup": loss_total_sup,
                "loss_total_a": loss_a["loss_total"],
                "loss_total_b": loss_b["loss_total"],
                "m3_path_weight_effective": loss_total.new_tensor(
                    effective_weight),
                "m3_teacher_updates": self.m3_teacher_updates.to(
                    dtype=loss_total.dtype),
            }
            for key, value in loss_a.items():
                loss_dict[f"view_a_{key}"] = value
            for key, value in loss_b.items():
                loss_dict[f"view_b_{key}"] = value
            loss_dict.update(m3_loss_dict)
            return loss_dict

        loss_total_sup = 0.5 * (loss_a["loss_total"] + loss_b["loss_total"])
        twc_loss_dict = self.compute_twc_loss(output_a, output_b, data_a, data_b)

        twc_weight = getattr(self.config, "twc_weight", 0.05)
        warmup_epoch = getattr(self.config, "twc_warmup_epoch", 0)
        if getattr(self, "current_epoch", 0) < warmup_epoch:
            twc_weight = 0.0

        loss_total = loss_total_sup + twc_weight * twc_loss_dict["loss_twc"]
        loss_dict = {
            "loss_total": loss_total,
            "loss_total_sup": loss_total_sup,
            "loss_total_a": loss_a["loss_total"],
            "loss_total_b": loss_b["loss_total"],
        }
        for key, value in loss_a.items():
            loss_dict[f"view_a_{key}"] = value
        for key, value in loss_b.items():
            loss_dict[f"view_b_{key}"] = value
        loss_dict.update(twc_loss_dict)
        return loss_dict

    def compute_loss(self, data, output):
        if self.is_paired_batch(data):
            return self.compute_paired_loss(data, output)

        loss_total = 0.0
        loss_dict = {}
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

        if self.use_b1motion_v3:
            def masked_mean(per_sample, valid):
                valid = valid.to(
                    device=per_sample.device,
                    dtype=per_sample.dtype,
                ).reshape(-1)
                return (
                    per_sample.reshape(-1) * valid
                ).sum() / torch.clamp(valid.sum(), min=1.0)

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
            loss_motion_v3_prior = masked_mean(
                main_prior_per_sample, main_valid)
            loss_total += (
                self.motion_v3_prior_weight * loss_motion_v3_prior)

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
                'motion_v3_prior_rmse': torch.sqrt(masked_mean(
                    main_prior_error.pow(2), main_valid)),
                'motion_v3_kinematic_rmse': torch.sqrt(masked_mean(
                    main_kinematic_error.pow(2), main_valid)),
                'motion_v3_prior_valid_rate': main_valid.float().mean(),
                'motion_v3_history_valid_ratio': data[
                    'motion_main_valid_mask'].float().mean(),
            })

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
                loss_motion_v3_aux_prior = masked_mean(
                    aux_prior_per_sample, aux_valid)
                loss_total += (
                    self.motion_v3_aux_prior_weight
                    * loss_motion_v3_aux_prior)
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

        if self.box_aware:
            prev_bc = torch.flatten(data['prev_bc'], start_dim=1, end_dim=2)
            this_bc = data['this_bc'] #torch.Size([B, 1024, 9])
            bc_label = torch.cat([prev_bc, this_bc], dim=1) #torch.Size([B, 4096, 9])
            pred_bc = output['pred_bc'] #torch.Size([B, 4096, 9])
            loss_bc = F.smooth_l1_loss(pred_bc, bc_label)
            loss_total += loss_bc * self.config.bc_weight
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

        return loss_dict

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
        output = self(batch)
        loss_dict = self.compute_loss(batch, output)
        loss = loss_dict['loss_total']

        if self.is_paired_batch(batch):
            metric_batch = batch["view_a"]
            metric_output = output["view_a"]
        else:
            metric_batch = batch
            metric_output = output

        # log
        seg_acc = self.seg_acc(torch.argmax(metric_output['seg_logits'], dim=1, keepdim=False),
                               metric_batch['seg_label'])
        self.log('seg_acc_background/train', seg_acc[0], on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('seg_acc_foreground/train', seg_acc[1], on_step=True, on_epoch=True, prog_bar=False, logger=True)
        if self.use_motion_cls:
            motion_acc = self.motion_acc(torch.argmax(metric_output['motion_cls'], dim=1, keepdim=False),
                                         metric_batch['motion_state_label'][:,0]) # 0 represents motion relative to the first historical box
            self.log('motion_acc_static/train', motion_acc[0], on_step=True, on_epoch=True, prog_bar=False, logger=True)
            self.log('motion_acc_dynamic/train', motion_acc[1], on_step=True, on_epoch=True, prog_bar=False,
                     logger=True)

        log_dict = {k: v.item() for k, v in loss_dict.items()}

        self.logger.experiment.add_scalars('loss', log_dict,
                                           global_step=self.global_step)

        return loss
