"""Construction of the formal v25 B0--B3 model graph."""

import copy

import torch
from torch import nn

from models.attn.Models import Seq2SeqFormer
from models.backbone.pointnet import MiniPointNet, SegPointNet, FeaturePointNet
from models.time_encoding import TimeEncoding
from ctseqtrack.model.evidence import B2EvidenceAcquirer, B3SelectiveUpdater
from ctseqtrack.model.pipeline import B0Observation, B1PhysicalTimePrior
from ctseqtrack.runtime.optimization import (
    CheckpointableRNG,
    isolated_constructor_rng,
)


def _value(config, name, default):
    return getattr(config, name, default)


def _validate_formal_contract(model):
    if model.ct_joint_contract_version != 3:
        raise ValueError("formal CT-SeqTrack requires contract version 3")
    if model.ct_initialization_policy != "scratch_only":
        raise ValueError("formal CT-SeqTrack requires scratch initialization")
    if not model.ct_separate_optimizers:
        raise ValueError("formal CT-SeqTrack requires separate optimizers")
    if model.use_ct_joint_full and not model.use_b1motion_v3:
        raise ValueError("CT joint Full requires the B1 physical prior")
    if model.use_ct_joint_full and not (
        model.ct_enable_shared_motion_anchor and model.ct_enable_dynamic_residual_bound
    ):
        raise ValueError("CT joint Full requires shared anchor and dynamic bound")
    if model.ct_expansion_point_count != (
        model.ct_endpoint_quota + model.ct_tube_quota
    ):
        raise ValueError("CT endpoint+tube quotas must equal expansion point count")
    if min(model.ct_endpoint_quota, model.ct_tube_quota) <= 0:
        raise ValueError("CT endpoint/tube quotas must be positive")
    if model.proposal_inference_mode not in ("observation", "raw_search", "selective"):
        raise ValueError(
            "v25 proposal_inference_mode must be observation, raw_search "
            "or selective"
        )
    if (
        min(
            model.motion_v3_prior_weight,
            model.motion_v3_aux_prior_weight,
            model.motion_v3_nll_weight,
            model.motion_v3_aux_nll_weight,
            model.ct_targetness_weight,
            model.ct_vote_weight,
            model.ct_raw_search_weight,
            model.ct_presence_weight,
            model.ct_router_weight,
            model.ct_router_help_margin,
            model.ct_router_h3_margin,
        )
        < 0
    ):
        raise ValueError("formal v25 loss weights must be non-negative")
    if not model.motion_v3_aux_query_gaps or any(
        gap <= 0 for gap in model.motion_v3_aux_query_gaps
    ):
        raise ValueError("B1 auxiliary query gaps must be positive")


def _initialize_contract_state(model, config):
    model.hist_num = int(_value(config, "hist_num", 1))
    model.box_aware = bool(_value(config, "box_aware", False))
    model.use_motion_cls = bool(_value(config, "use_motion_cls", True))
    model.use_ct_joint_full = bool(_value(config, "use_ct_joint_full", False))
    model.use_b1motion_v3 = bool(_value(config, "use_b1motion_v3", False))
    model.ct_joint_contract_version = int(
        _value(config, "ct_joint_contract_version", 3)
    )
    model.ct_enable_b1 = bool(_value(config, "ct_enable_b1", False))
    model.ct_enable_b2 = bool(_value(config, "ct_enable_b2", False))
    model.ct_enable_b3 = bool(_value(config, "ct_enable_b3", False))
    model.ct_initialization_policy = str(
        _value(config, "ct_initialization_policy", "scratch_only")
    )
    model.ct_separate_optimizers = bool(_value(config, "ct_separate_optimizers", True))
    model.ct_manual_amp_enabled = bool(_value(config, "ct_manual_amp_enabled", False))
    model.ct_b0_rng_protocol = (
        str(_value(config, "ct_b0_rng_protocol", "off")).strip().lower()
    )
    if model.ct_b0_rng_protocol not in ("off", "post_observation_shift_v1"):
        raise ValueError("ct_b0_rng_protocol must be off or post_observation_shift_v1")
    model.ct_b0_rng_shift_control = (
        model.ct_b0_rng_protocol == "post_observation_shift_v1"
    )
    model.automatic_optimization = False

    seed = int(_value(config, "seed", 42) or 42)
    if model.use_ct_joint_full:
        model.ct_plugin_rng = CheckpointableRNG(seed + 24001)
        model.ct_auxiliary_rng = CheckpointableRNG(seed + 24002)
        model.ct_memory_control_rng = CheckpointableRNG(seed + 24003)
    else:
        model.ct_plugin_rng = None
        model.ct_auxiliary_rng = None
        model.ct_memory_control_rng = None
    model._ct_scalers = {}
    model._ct_b0_scaler = None
    model._ct_plugin_scaler = None
    model._ct_pending_scaler_state = None
    for module_name in ("b0", "b1", "b2", "b3"):
        model.register_buffer(
            f"ct_{module_name}_update_step", torch.zeros((), dtype=torch.long)
        )
    model.register_buffer("ct_plugin_update_step", torch.zeros((), dtype=torch.long))
    model._ct_epoch_boundary_complete = False
    method_promotion = _value(config, "ct_b2_method_promotion_manifest", None)
    if isinstance(method_promotion, dict):
        model._ct_b2_method_promotion = copy.deepcopy(method_promotion)
    preflight = _value(config, "ct_acquisition_preflight_manifest", None)
    if isinstance(preflight, dict):
        model._ct_acquisition_preflight = copy.deepcopy(preflight)


def _initialize_options(model, config):
    model.ct_enable_shared_motion_anchor = bool(
        _value(config, "ct_enable_shared_motion_anchor", True)
    )
    model.ct_enable_dynamic_residual_bound = bool(
        _value(config, "ct_enable_dynamic_residual_bound", True)
    )
    model.ct_enable_query_reliability_gate = bool(
        _value(config, "ct_enable_query_reliability_gate", True)
    )
    model.ct_expansion_point_count = int(
        _value(config, "ct_expansion_point_count", 256)
    )
    model.ct_endpoint_quota = int(_value(config, "ct_endpoint_quota", 128))
    model.ct_tube_quota = int(_value(config, "ct_tube_quota", 128))
    model.use_calibrated_motion_uncertainty = bool(
        _value(config, "use_calibrated_motion_uncertainty", False)
    )
    model.require_b1_calibration_passed = bool(
        _value(config, "require_b1_calibration_passed", False)
    )
    model.require_b1_calibration_artifact = bool(
        _value(config, "require_b1_calibration_artifact", False)
    )
    model.use_b1_prepass_support = bool(_value(config, "use_b1_prepass_support", False))
    model.proposal_inference_mode = (
        str(_value(config, "proposal_inference_mode", "observation")).strip().lower()
    )

    model.motion_v3_prior_weight = float(_value(config, "motion_v3_prior_weight", 0.1))
    model.motion_v3_aux_prior_weight = float(
        _value(config, "motion_v3_aux_prior_weight", 0.1)
    )
    model.motion_v3_nll_weight = float(_value(config, "motion_v3_nll_weight", 0.0))
    model.motion_v3_aux_nll_weight = float(
        _value(config, "motion_v3_aux_nll_weight", model.motion_v3_nll_weight)
    )
    model.motion_v3_aux_query_gaps = tuple(
        int(value) for value in _value(config, "motion_v3_aux_query_gaps", (2, 4))
    )
    model.ct_targetness_weight = float(_value(config, "ct_targetness_weight", 0.2))
    model.ct_vote_weight = float(_value(config, "ct_vote_weight", 1.0))
    model.ct_raw_search_weight = float(_value(config, "ct_raw_search_weight", 1.0))
    model.ct_presence_weight = float(_value(config, "ct_presence_weight", 0.2))
    model.ct_router_weight = float(_value(config, "ct_router_weight", 0.2))
    model.ct_router_help_margin = float(_value(config, "ct_router_help_margin", 0.05))
    model.ct_router_h3_margin = float(_value(config, "ct_router_h3_margin", 0.15))


def _initialize_b0(model, config, build_binary_accuracy):
    model.seg_acc = build_binary_accuracy()
    default_time_scale = _value(
        config, "default_time_step", _value(config, "time_step", 0.5)
    )
    model.time_encoder = TimeEncoding(
        mode=_value(config, "time_encoding", "raw"),
        scale=_value(config, "time_scale", default_time_scale),
        clip=_value(config, "time_clip", 4.0),
        fourier_bands=_value(config, "time_fourier_bands", 4),
        hidden_dim=_value(config, "time_hidden_dim", 16),
        output_scale=_value(
            config, "time_output_scale", _value(config, "pseudo_time_step", 0.1)
        ),
    )
    channel_count = 3 + 1 + 1 + (9 if model.box_aware else 0)
    model.seg_pointnet = SegPointNet(
        input_channel=channel_count,
        per_point_mlp1=[64, 64, 64, 128, 1024],
        per_point_mlp2=[512, 256, 128, 128],
        output_size=2 + (9 if model.box_aware else 0),
    )
    model.mini_pointnet = MiniPointNet(
        input_channel=3 + 1 + (9 if model.box_aware else 0),
        per_point_mlp=[64, 128, 256, 512],
        hidden_mlp=[512, 256],
        output_size=-1,
    )
    if model.use_motion_cls:
        model.motion_state_mlp = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )
        model.motion_acc = build_binary_accuracy()
    model.motion_mlp = nn.Sequential(
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(128, 4),
    )
    model.feature_pointnet = FeaturePointNet(
        input_channel=channel_count,
        per_point_mlp1=[64, 64, 64, 128, 1024],
        per_point_mlp2=[512, 256, 128, 128],
        output_size=128,
    )
    model.Transformer = Seq2SeqFormer(
        d_word_vec=64,
        d_model=64,
        d_inner=512,
        n_layers=3,
        n_head=4,
        d_k=64,
        d_v=64,
        n_position=1024 * 4,
    )


def _initialize_plugins(model, config):
    if not model.use_b1motion_v3:
        return
    seed = int(_value(config, "seed", 42) or 42)
    default_time_scale = _value(
        config, "default_time_step", _value(config, "time_step", 0.5)
    )
    with isolated_constructor_rng(seed, "b1.motion"):
        model.physical_motion_encoder = B1PhysicalTimePrior(
            hidden_dim=int(_value(config, "motion_v3_hidden_dim", 128)),
            step_dim=int(_value(config, "motion_v3_step_dim", 64)),
            eps=float(_value(config, "motion_v3_eps", 1e-3)),
            time_scale=float(_value(config, "time_scale", default_time_scale)),
            residual_velocity_scale=float(
                _value(config, "motion_v3_residual_velocity_scale", 4.0)
            ),
            initial_sigma=float(_value(config, "motion_v3_initial_sigma", 0.5)),
            motion_aligned_uncertainty=(model.use_calibrated_motion_uncertainty),
            min_direction_speed=float(
                _value(config, "motion_v3_min_direction_speed", 0.2)
            ),
            shared_kinematic_anchor=(
                model.use_ct_joint_full
                and model.ct_enable_shared_motion_anchor
                and model.ct_enable_dynamic_residual_bound
            ),
            max_acceleration=float(_value(config, "ct_motion_max_acceleration", 8.0)),
            max_displacement=float(_value(config, "ct_motion_max_displacement", 12.0)),
            acceleration_weight=float(
                _value(config, "ct_motion_acceleration_weight", 0.5)
            ),
        )
    if not model.use_ct_joint_full:
        return
    model.b0_observation_contract = B0Observation()
    if model.ct_enable_b2:
        with isolated_constructor_rng(seed, "b2.evidence_acquirer"):
            model.ct_joint_search_refiner = B2EvidenceAcquirer(
                feature_dim=64,
                num_heads=4,
                max_vote_offset=float(_value(config, "ct_search_max_vote_offset", 4.0)),
                attention_dropout=float(
                    _value(config, "ct_memory_attention_dropout", 0.0)
                ),
                presence_init_probability=float(
                    _value(config, "ct_search_presence_init_probability", 0.1)
                ),
                presence_threshold=float(
                    _value(config, "ct_search_presence_threshold", 0.5)
                ),
            )
    if model.ct_enable_b3:
        with isolated_constructor_rng(seed, "b3.selective_updater"):
            model.ct_joint_router = B3SelectiveUpdater(
                observation_stats_dim=5,
                hidden_dim=int(_value(config, "ct_router_hidden_dim", 64)),
                presence_threshold=float(
                    _value(config, "ct_search_presence_threshold", 0.5)
                ),
                decision_threshold=float(_value(config, "ct_router_threshold", 0.5)),
                radius_base=float(_value(config, "ct_router_radius_base", 0.5)),
                radius_per_second=float(
                    _value(config, "ct_router_radius_per_second", 0.5)
                ),
                radius_max=float(_value(config, "ct_router_radius_max", 2.0)),
                require_calibration=bool(
                    _value(config, "ct_require_action_calibration", False)
                ),
            )


def initialize_v25(model, config, build_binary_accuracy):
    """Initialize the exact formal graph without historical branches."""
    _initialize_contract_state(model, config)
    _initialize_options(model, config)
    _validate_formal_contract(model)
    _initialize_b0(model, config, build_binary_accuracy)
    _initialize_plugins(model, config)
    for module_name, enabled in (
        ("physical_motion_encoder", model.ct_enable_b1),
        ("ct_joint_search_refiner", model.ct_enable_b2),
        ("ct_joint_router", model.ct_enable_b3),
    ):
        module = getattr(model, module_name, None)
        if bool(enabled) != (module is not None):
            raise RuntimeError(f"formal module availability mismatch for {module_name}")
        if module is not None:
            for parameter in module.parameters():
                parameter.requires_grad_(True)


__all__ = ["initialize_v25"]
