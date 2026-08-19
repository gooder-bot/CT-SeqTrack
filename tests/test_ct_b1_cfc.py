from pathlib import Path

import pytest
import torch

from ctseqtrack.config import configure_ct_variant
from ctseqtrack.model.cfc import FullGatedCfCCell
from ctseqtrack.model.prior import OrderedPhysicalMotionEncoder
from ctseqtrack.runtime.calibration import b1_calibration_config_sha256
from ctseqtrack.runtime.contracts import (
    build_online_resume_contract,
    resolved_training_config_sha256,
)
from tools.compare_b1_backbones import compare
from utils.config import load_yaml_config


ROOT = Path(__file__).resolve().parents[1]


def _parameter_count(module):
    return sum(parameter.numel() for parameter in module.parameters())


def _prior(backend):
    return OrderedPhysicalMotionEncoder(
        hidden_dim=128,
        step_dim=64,
        motion_aligned_uncertainty=True,
        shared_kinematic_anchor=True,
        temporal_backend=backend,
        cfc_backbone_units=105,
    )


def _inputs(batch=3):
    torch.manual_seed(19)
    boxes = torch.randn(batch, 3, 4)
    boxes[:, 0] = 0.0
    delta_t = torch.tensor([[0.5, 0.5, 1.0]]).repeat(batch, 1)
    valid = torch.ones(batch, 3)
    return boxes, delta_t, valid, delta_t[:, 0]


def test_cfc_cell_parameter_count_matches_gru_within_point_one_percent():
    cfc = FullGatedCfCCell(64, 128, backbone_units=105)
    gru = torch.nn.GRU(64, 128, batch_first=True)
    assert _parameter_count(cfc) == 74537
    assert _parameter_count(gru) == 74496
    difference = abs(_parameter_count(cfc) - _parameter_count(gru))
    assert difference / _parameter_count(gru) < 0.001


def test_cfc_cell_is_explicitly_sensitive_to_elapsed_time():
    cell = FullGatedCfCCell(2, 3, backbone_units=4)
    with torch.no_grad():
        for parameter in cell.parameters():
            parameter.zero_()
        cell.first_state.bias.fill_(-0.5)
        cell.second_state.bias.fill_(0.5)
        cell.time_a.bias.fill_(1.0)
    inputs = torch.zeros(2, 2)
    hidden = torch.zeros(2, 3)
    short = cell(inputs, hidden, torch.tensor([0.25, 0.25]))
    long = cell(inputs, hidden, torch.tensor([2.0, 2.0]))
    assert not torch.allclose(short, long)


def test_cfc_masked_transition_is_an_exact_hidden_state_noop():
    torch.manual_seed(7)
    prior = OrderedPhysicalMotionEncoder(
        hidden_dim=6, step_dim=4, temporal_backend="cfc", cfc_backbone_units=5
    )
    projected = torch.randn(2, 3, 4)
    valid = torch.tensor([[True, False, False], [True, True, False]])
    gaps = torch.tensor([[1.0, 500.0, 900.0], [1.0, 2.0, 700.0]])
    encoded = prior._encode_transitions(projected, valid, gaps)
    zero = torch.zeros(2, 6)
    first = prior.cfc(projected[:, 0], zero, gaps[:, 0] / prior.time_scale)
    second_row = prior.cfc(
        projected[1:2, 1], first[1:2], gaps[1:2, 1] / prior.time_scale
    )
    assert torch.equal(encoded[0], first[0])
    assert torch.allclose(encoded[1], second_row[0], atol=1e-6, rtol=1e-6)


def test_cfc_parameters_all_receive_finite_nonzero_gradients():
    torch.manual_seed(11)
    cell = FullGatedCfCCell(4, 5, backbone_units=6)
    output = cell(torch.randn(8, 4), torch.randn(8, 5), torch.rand(8) + 0.1)
    output.pow(2).mean().backward()
    for name, parameter in cell.named_parameters():
        assert parameter.requires_grad, name
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


def test_cfc_parameters_receive_gradients_on_second_cold_start_batch():
    torch.manual_seed(17)
    module = _prior("cfc").train()
    optimizer = torch.optim.SGD(module.parameters(), lr=1e-2)
    boxes, delta_t, valid, query = _inputs(batch=4)
    target = module.kinematic_fallback(boxes, delta_t, valid, query)["mu_xy"] + 0.5
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = module(boxes, delta_t, valid, query)
        loss = torch.nn.functional.smooth_l1_loss(output["mu_xy"], target)
        loss.backward()
        optimizer.step()
    for name, parameter in module.cfc.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


@pytest.mark.parametrize("backend", ["gru", "cfc"])
def test_b1_backend_vectorized_matches_individual_and_contract_is_finite(backend):
    torch.manual_seed(23)
    module = _prior(backend).eval()
    boxes, delta_t, valid, query = _inputs(batch=3)
    with torch.no_grad():
        vectorized = module(boxes, delta_t, valid, query)
        individual = [
            module(
                boxes[index : index + 1],
                delta_t[index : index + 1],
                valid[index : index + 1],
                query[index : index + 1],
            )
            for index in range(3)
        ]
    expected_keys = set(vectorized)
    for row in individual:
        assert set(row) == expected_keys
    for key, value in vectorized.items():
        if torch.is_floating_point(value):
            assert torch.isfinite(value).all(), key
        expected = torch.cat([row[key] for row in individual], dim=0)
        assert torch.allclose(value, expected, atol=1e-6, rtol=1e-6), key


@pytest.mark.parametrize("backend", ["gru", "cfc"])
def test_b1_backend_invalid_history_falls_back_without_nonfinite_values(backend):
    module = _prior(backend).eval()
    boxes, delta_t, valid, query = _inputs(batch=2)
    valid.zero_()
    boxes[0, 1, 0] = float("nan")
    with torch.no_grad():
        output = module(boxes, delta_t, valid, query)
    assert torch.equal(output["valid"], torch.zeros_like(output["valid"]))
    for key in ("mu_xy", "kinematic_prior_xy", "residual_xy", "velocity_xy"):
        assert torch.equal(output[key], torch.zeros_like(output[key])), key
    for key, value in output.items():
        if torch.is_floating_point(value):
            assert torch.isfinite(value).all(), key


def test_gru_and_cfc_zero_initialized_heads_share_kinematic_cold_start():
    torch.manual_seed(29)
    gru = _prior("gru").eval()
    torch.manual_seed(29)
    cfc = _prior("cfc").eval()
    boxes, delta_t, valid, query = _inputs(batch=2)
    with torch.no_grad():
        gru_output = gru(boxes, delta_t, valid, query)
        cfc_output = cfc(boxes, delta_t, valid, query)
    assert set(gru_output) == set(cfc_output)
    for output in (gru_output, cfc_output):
        assert torch.equal(output["mu_xy"], output["kinematic_prior_xy"])
        assert torch.equal(output["residual_xy"], torch.zeros_like(output["residual_xy"]))
    for key in (
        "mu_xy",
        "kinematic_prior_xy",
        "velocity_xy",
        "valid",
        "gap_ratio",
        "source_id",
    ):
        assert torch.equal(gru_output[key], cfc_output[key]), key


def test_backend_configuration_is_scratch_only_and_identity_bound():
    configs = {}
    for backend in ("gru", "cfc"):
        config = load_yaml_config(
            ROOT / "cfgs" / "ct_seqtrack" / f"b1_{backend}_mini_seed42.yaml"
        )
        configure_ct_variant(config)
        assert config["motion_v3_temporal_backend"] == backend
        assert config["motion_v3_cfc_backbone_units"] == 105
        assert config["ct_initialization_policy"] == "scratch_only"
        assert config["epoch"] == 60
        assert config["seed"] == 42
        assert "init_checkpoint" not in config
        assert config["ct_module_isolation"] == "strict"
        assert config["ct_separate_optimizers"] is True
        assert config["ct_enable_b1"] is True
        assert all(parameter.requires_grad for parameter in _prior(backend).parameters())
        configs[backend] = config
    gru_shared = dict(configs["gru"])
    cfc_shared = dict(configs["cfc"])
    for config in (gru_shared, cfc_shared):
        config.pop("experiment_name")
        config.pop("motion_v3_temporal_backend")
    assert gru_shared == cfc_shared
    assert resolved_training_config_sha256(configs["gru"]) != (
        resolved_training_config_sha256(configs["cfc"])
    )
    assert b1_calibration_config_sha256(configs["gru"]) != (
        b1_calibration_config_sha256(configs["cfc"])
    )
    gru_resume = build_online_resume_contract(configs["gru"])["fields"]
    cfc_resume = build_online_resume_contract(configs["cfc"])["fields"]
    assert gru_resume["motion_v3_temporal_backend"] == "gru"
    assert cfc_resume["motion_v3_temporal_backend"] == "cfc"
    assert gru_resume["motion_v3_cfc_backbone_units"] == 105
    assert cfc_resume["motion_v3_cfc_backbone_units"] == 105


def test_gru_and_cfc_checkpoints_are_not_strictly_interchangeable():
    gru = _prior("gru")
    cfc = _prior("cfc")
    assert any(key.startswith("gru.") for key in gru.state_dict())
    assert any(key.startswith("cfc.") for key in cfc.state_dict())
    with pytest.raises(RuntimeError):
        cfc.load_state_dict(gru.state_dict(), strict=True)
    with pytest.raises(RuntimeError):
        gru.load_state_dict(cfc.state_dict(), strict=True)


@pytest.mark.parametrize("backend", ["lstm", "", "CFC-MM"])
def test_unknown_temporal_backend_is_rejected(backend):
    with pytest.raises(ValueError, match="gru or cfc"):
        OrderedPhysicalMotionEncoder(temporal_backend=backend)
    with pytest.raises(ValueError, match="gru or cfc"):
        configure_ct_variant(
            {
                "ct_variant": "b1",
                "ct_prior_mode": "learned_physical",
                "motion_v3_temporal_backend": backend,
            }
        )


def _proposal_rows(error, nll, support, sampled):
    rows = []
    for index in range(6):
        rows.append(
            {
                "partition_group_key": f"scene-{index // 2}",
                "tracklet_key": f"track-{index // 2}",
                "frame_id": index,
                "candidate_id": 0,
                "b1_valid": 1,
                "learned_motion_error": error,
                "kinematic_error": 0.4,
                "b1_nll": nll,
                "target_in_support": support,
                "pool_target_count": sampled + 1,
                "sampled_target_count": sampled,
                "evidence_extension_unique_count": 32,
                "support_volume": 8.0,
                "b1_coverage_50": index < 3,
                "b1_coverage_80": index < 5,
                "b1_coverage_95": 1,
                "query_delta_t": float(index + 1),
                "current_target_points": float(index),
                "recursive_age": float(index),
            }
        )
    return rows


def _tracking_rows():
    return [
        {
            "partition_group_key": f"scene-{index // 2}",
            "tracklet_key": f"track-{index // 2}",
            "frame_id": index,
            "final_iou": 0.5,
            "final_distance": 0.5,
        }
        for index in range(6)
    ]


def test_backbone_comparison_applies_joint_promotion_gates():
    result = compare(
        _proposal_rows(0.2, 0.2, 0, 1),
        _proposal_rows(0.1, 0.1, 1, 2),
        _tracking_rows(),
        _tracking_rows(),
    )
    assert result["tracking_isolation_passed"]
    assert all(result["promotion_gates"].values())
    assert result["metric_gates_passed"]
    assert result["promoted_to_full_screen"] is None
    assert result["promotion_status"] == "pending_b0_hash_audit"


def test_backbone_comparison_identity_includes_tracklet_within_scene():
    rows = _proposal_rows(0.2, 0.2, 1, 2)
    tracking = _tracking_rows()
    for collection in (rows, tracking):
        collection[1]["frame_id"] = collection[0]["frame_id"]
        collection[1]["tracklet_key"] = "another-track"
    result = compare(rows, rows, tracking, tracking)
    assert result["aligned_proposal_rows"] == len(rows)
    assert result["aligned_tracking_rows"] == len(tracking)


def test_backbone_comparison_precision_rewards_lower_center_distance():
    proposals = _proposal_rows(0.2, 0.2, 1, 2)
    gru_tracking = _tracking_rows()
    cfc_tracking = _tracking_rows()
    for row in gru_tracking:
        row["final_distance"] = 1.5
    for row in cfc_tracking:
        row["final_distance"] = 0.25
    result = compare(proposals, proposals, gru_tracking, cfc_tracking)
    assert result["cfc"]["tracking"]["precision"] > (
        result["gru"]["tracking"]["precision"]
    )
