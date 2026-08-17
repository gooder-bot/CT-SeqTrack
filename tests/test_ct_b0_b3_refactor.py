from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from models.ct_variant import configure_ct_variant
from models.ct_v2 import (
    B2EvidenceAcquirer,
    B3SelectiveUpdater,
    build_box_memory_tokens,
)
from utils.action_calibration import (
    SCORE_DEFINITION,
    calibrate_actions,
    require_selective_calibration,
    validate_action_calibration,
)
from utils.config import load_yaml_config
from utils.memory_promotion import evaluate_memory_promotion
from tools.report_ct_b1 import build_report


ROOT = Path(__file__).resolve().parents[1]


def test_four_formal_arms_are_single_variant_scratch_configs():
    expected = {
        "24_b0.yaml": "b0",
        "24_b1.yaml": "b1",
        "24_full_minus_b3.yaml": "full_minus_b3",
        "24_full.yaml": "full",
    }
    for name, variant in expected.items():
        config = load_yaml_config(ROOT / "cfgs" / "ct_seqtrack" / name)
        assert config["net_model"] == "ctseqtrack"
        assert config["ct_variant"] == variant
        assert config["ct_initialization_policy"] == "scratch_only"
        assert config["ct_training_state_policy"] == "observation"
        assert config["ct_module_isolation"] == "strict"
        assert config["save_top_k"] == 0


def test_b0_2x2_configs_are_matched_except_registered_factors():
    expected = {
        "24_b0_2x2_reseed0_rngshift0.yaml": (False, False),
        "24_b0_2x2_reseed0_rngshift1.yaml": (False, True),
        "24_b0_2x2_reseed1_rngshift0.yaml": (True, False),
        "24_b0_2x2_reseed1_rngshift1.yaml": (True, True),
    }
    reference = None
    ignored = {
        "experiment_name", "ct_recursive_reseed_enabled",
        "ct_b0_rng_shift_control",
    }
    for name, factors in expected.items():
        config = load_yaml_config(ROOT / "cfgs" / "ct_seqtrack" / name)
        configure_ct_variant(config)
        assert config["ct_variant"] == "b0"
        assert config["seed"] == 42
        assert config["batch_size"] == 16
        assert config["epoch"] == 60
        assert config["workers"] == 4
        assert config["check_val_every_n_epoch"] == 5
        assert (
            bool(config["ct_recursive_reseed_enabled"]),
            bool(config["ct_b0_rng_shift_control"]),
        ) == factors
        matched = {
            key: value for key, value in config.items()
            if key not in ignored
        }
        if reference is None:
            reference = matched
        else:
            assert matched == reference


def test_b0_normalization_removes_inherited_b1_b2_runtime_switches():
    config = load_yaml_config(
        ROOT / "cfgs" / "ct_seqtrack" /
        "24_b0_2x2_reseed0_rngshift0.yaml")
    configure_ct_variant(config)
    assert not config["use_b1motion_v3"]
    assert not config["ct_enable_b1"]
    assert not config["use_calibrated_motion_uncertainty"]
    assert not config["require_b1_calibration_artifact"]
    assert not config["require_b1_calibration_passed"]
    assert not config["search_v3_use_dynamic_sigma"]
    assert not config["use_b1_prepass_support"]
    assert not config["use_uncertainty_geometry"]


def test_every_formal_config_satisfies_normalized_scratch_contract():
    from utils.online_contract import validate_scratch_training_contract

    for path in sorted((ROOT / "cfgs" / "ct_seqtrack").glob("*.yaml")):
        config = load_yaml_config(path)
        configure_ct_variant(config)
        if config["ct_time_mode"] == "shuffled":
            config["dynamics_time_manifest"] = "frozen-manifest.json"
        validate_scratch_training_contract(config)


def test_memory_promotion_requires_both_paired_controls():
    rows = [{
        "tracklet_id": index,
        "real_success": 0.8,
        "empty_success": 0.7,
        "time_misaligned_success": 0.72,
        "real_precision": 0.9,
        "empty_precision": 0.8,
        "time_misaligned_precision": 0.82,
    } for index in range(30)]
    artifact = evaluate_memory_promotion(
        rows, {"real": "r", "empty": "e", "time_misaligned": "t"},
        resamples=50)
    assert artifact["passed"]
    rows[0]["real_success"] = -100.0
    failed = evaluate_memory_promotion(
        rows, {"real": "r", "empty": "e", "time_misaligned": "t"},
        resamples=50)
    assert not failed["passed"]


def test_b1_report_contains_registered_strata_and_mean_vs_cv():
    rows = [{
        "b1_valid": 1,
        "learned_motion_error": 0.2,
        "kinematic_error": 0.4,
        "b1_nll": 0.1,
        "target_in_support": 1,
        "support_volume": 8.0,
        "b1_coverage_50": 1,
        "b1_coverage_80": 1,
        "b1_coverage_95": 1,
        "query_delta_t": float(index + 1),
        "current_target_points": float(index),
        "recursive_age": float(index),
    } for index in range(6)]
    report = build_report(rows)
    assert report["overall"]["learned_minus_cv_rmse"] < 0
    assert set(report["strata"]) == {
        "time_gap", "sparsity", "recursive_age"}


def test_variant_normalization_rejects_cross_product_switches():
    config = SimpleNamespace(
        ct_variant="full", ct_prior_mode="learned_physical",
        ct_memory_mode="none", use_joint_proposal_fusion=True)
    with pytest.raises(ValueError, match="legacy runtime branches"):
        configure_ct_variant(config)


def _b2_inputs():
    torch.manual_seed(3)
    history_features = torch.randn(1, 3, 1024, 64)
    history_points = torch.randn(1, 3, 1024, 5)
    memory, memory_valid, metadata = build_box_memory_tokens(
        history_features, history_points, torch.zeros(1, 3, 4),
        torch.ones(1, 3) * 4.0, torch.ones(1, 3),
        return_metadata=True)
    base = torch.randn(1, 1024, 64, requires_grad=True)
    prior = torch.zeros(1, 2, requires_grad=True)
    return base, prior, dict(
        extension_points=torch.randn(1, 256, 5),
        extension_valid_mask=torch.ones(1, 256),
        extension_source=torch.ones(1, 256, dtype=torch.long),
        current_base_features=base,
        current_base_valid_mask=torch.ones(1, 1024),
        memory_tokens=memory,
        memory_valid_mask=memory_valid,
        memory_metadata=metadata,
        observation_box=torch.zeros(1, 4, requires_grad=True),
        observation_stats=torch.zeros(1, 5, requires_grad=True),
        b1_center_xy=prior,
        b1_sigma_parallel_perp=torch.ones(1, 2, requires_grad=True),
        b1_direction_xy=torch.tensor(
            [[1.0, 0.0]], requires_grad=True),
        b1_valid=torch.ones(1),
        query_delta_t=torch.ones(1),
        gap_ratio=torch.ones(1),
    )


def test_b2_acquisition_has_zero_gradient_to_b0_and_b1_inputs():
    base, prior, inputs = _b2_inputs()
    output = B2EvidenceAcquirer().train()(**inputs)
    output["ct_search_targetness_logits"].mean().backward()
    assert base.grad is None
    assert prior.grad is None
    assert inputs["observation_box"].grad is None
    assert inputs["b1_sigma_parallel_perp"].grad is None


def test_b3_has_zero_gradient_to_all_upstream_inputs():
    evidence = torch.randn(2, 64, requires_grad=True)
    raw = torch.randn(2, 4, requires_grad=True)
    prior = torch.randn(2, 2, requires_grad=True)
    _, output = B3SelectiveUpdater().train()(
        observation_box=torch.zeros(2, 4, requires_grad=True),
        raw_box=raw,
        availability=torch.ones(2),
        base_evidence=evidence,
        extension_evidence=evidence,
        base_presence_probability=torch.ones(2, requires_grad=True),
        extension_presence_probability=torch.ones(2, requires_grad=True),
        observation_stats=torch.zeros(2, 5, requires_grad=True),
        b1_sigma_parallel_perp=torch.ones(2, 2, requires_grad=True),
        b1_center_xy=prior,
        query_delta_t=torch.ones(2),
        gap_ratio=torch.ones(2),
    )
    output["ct_b3_help_logit"].sum().backward()
    assert evidence.grad is None
    assert raw.grad is None
    assert prior.grad is None


def test_action_calibration_is_tracklet_bootstrap_fail_closed():
    rows = []
    for tracklet in range(30):
        for frame in range(4):
            rows.append({
                "tracklet_id": f"t{tracklet}",
                "structural_available": 1,
                "presence_score": 0.9,
                "action_score": 0.8,
                "center_gain": 0.2 + frame * 0.01,
                "iou_gain": 0.03,
            })
    artifact = calibrate_actions(
        rows, "checkpoint", "config", "manifest",
        resamples=100)
    assert artifact["passed"]
    assert artifact["score_definition"] == SCORE_DEFINITION
    validate_action_calibration(
        artifact, "checkpoint", "config", "manifest")
    with pytest.raises(ValueError, match="checkpoint_sha256 mismatch"):
        validate_action_calibration(
            artifact, "different", "config", "manifest")


def test_selective_evaluation_without_artifact_is_an_error():
    with pytest.raises(RuntimeError, match="matching passed"):
        require_selective_calibration(False, "selective")
    require_selective_calibration(False, "observation")


def test_b4_paired_views_no_longer_depend_on_removed_loss_switch():
    config = load_yaml_config(
        ROOT / "cfgs" / "ct_v2" / "19_b4_decoder_alignment.yaml")
    assert config["use_b4_paired_views"]
    assert config["use_decoder_token_consistency"]
    sampler_source = (ROOT / "datasets" / "sampler.py").read_text(
        encoding="utf-8")
    assert "use_twc" not in sampler_source
