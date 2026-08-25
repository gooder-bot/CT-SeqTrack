import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from pyquaternion import Quaternion

from models.ct_v2.cfc import FullGatedCfCCell
from models.ct_v2.motion import (
    OrderedPhysicalMotionEncoder,
    physical_motion_uncertainty_loss,
    recursive_gap_age_balanced_mean,
)
from models.ct_variant import configure_ct_variant
from tools.report_ct_b1 import build_report
from utils.config import load_yaml_config
from utils.ct_search import build_b1_uncertainty_support
from utils.online_contract import (
    build_online_resume_contract,
    validate_online_resume_contract,
)
from utils.replay_cache import (
    b1_calibration_config_sha256,
    replay_config_sha256,
    validate_b1_calibration_state,
)


ROOT = Path(__file__).resolve().parents[1]


def _history(batch=3, length=4):
    boxes = torch.zeros(batch, length, 4)
    for index in range(length):
        boxes[:, index, 0] = -float(index)
        boxes[:, index, 1] = 0.1 * float(index * index)
    delta_t = torch.full((batch, length), 0.5)
    valid = torch.ones(batch, length)
    return boxes, delta_t, valid


def _nonzero_finite_gradient(parameters):
    gradients = [
        parameter.grad for parameter in parameters
        if parameter.grad is not None]
    return bool(gradients) and all(
        bool(torch.isfinite(gradient).all()) for gradient in gradients
    ) and any(bool(torch.count_nonzero(gradient)) for gradient in gradients)


def _loss_terms(module, query_gap):
    boxes, delta_t, valid = _history()
    output = module(
        boxes, delta_t, valid,
        torch.full((boxes.shape[0],), float(query_gap)))
    target = output["kinematic_prior_xy"].detach() + torch.tensor(
        [[0.4, -0.2], [1.2, 0.3], [70.0, -70.0]])
    terms = physical_motion_uncertainty_loss(
        output["prior_xy"], target,
        output["log_sigma_parallel_perp"],
        output["motion_direction_xy"], output["valid"],
        kinematic_xy=output["kinematic_prior_xy"],
        envelope_parallel_perp=output["envelope_parallel_perp"],
        residual_unit_parallel_perp=output[
            "residual_unit_parallel_perp"],
        beta=0.5, tail_direction_weight=0.25,
        tail_direction_margin=0.9, sigma_error_cap=12.0)
    return output, terms


def test_cfc_parameter_match_and_exclusive_instantiation():
    gru = OrderedPhysicalMotionEncoder(
        hidden_dim=128, step_dim=64, temporal_backend="gru")
    cfc = OrderedPhysicalMotionEncoder(
        hidden_dim=128, step_dim=64, temporal_backend="cfc",
        cfc_backbone_units=105)
    gru_count = sum(parameter.numel() for parameter in gru.gru.parameters())
    cfc_count = sum(parameter.numel() for parameter in cfc.cfc.parameters())
    assert gru_count == 74496
    assert cfc_count == 74537
    assert abs(cfc_count - gru_count) / gru_count < 0.001
    assert all(parameter.requires_grad for parameter in gru.parameters())
    assert all(parameter.requires_grad for parameter in cfc.parameters())
    assert not hasattr(gru, "cfc")
    assert not hasattr(cfc, "gru")


def test_cfc_is_time_sensitive_batched_and_padding_is_exact_noop():
    torch.manual_seed(7)
    cell = FullGatedCfCCell(64, 128, 105).eval()
    inputs = torch.randn(3, 64)
    hidden = torch.randn(3, 128)
    short = cell(inputs, hidden, torch.full((3,), 0.5))
    long = cell(inputs, hidden, torch.full((3,), 2.0))
    assert not torch.allclose(short, long)
    individual = torch.cat([
        cell(inputs[index:index + 1], hidden[index:index + 1], 0.5)
        for index in range(3)
    ], dim=0)
    assert torch.allclose(short, individual, atol=1e-6, rtol=1e-6)

    encoder = OrderedPhysicalMotionEncoder(
        hidden_dim=128, step_dim=64, temporal_backend="cfc",
        cfc_backbone_units=105, shared_kinematic_anchor=True,
        motion_aligned_uncertainty=True).eval()
    boxes, delta_t, valid = _history(batch=2, length=4)
    valid[:, 2:] = 0
    changed = boxes.clone()
    changed[:, 2:] = torch.randn_like(changed[:, 2:]) * 1000.0
    with torch.no_grad():
        baseline = encoder(boxes, delta_t, valid, torch.full((2,), 0.5))
        padded = encoder(changed, delta_t, valid, torch.full((2,), 0.5))
    assert torch.equal(baseline["feature"], padded["feature"])
    assert torch.equal(baseline["prior_xy"], padded["prior_xy"])


@pytest.mark.parametrize("backend", ["gru", "cfc"])
def test_b1_cold_start_and_sigma_are_decoupled_from_mean_envelope(backend):
    module = OrderedPhysicalMotionEncoder(
        hidden_dim=128, step_dim=64, temporal_backend=backend,
        cfc_backbone_units=105, shared_kinematic_anchor=True,
        motion_aligned_uncertainty=True, initial_sigma=0.5,
        log_sigma_min=np.log(0.1), log_sigma_max=2.5)
    boxes, delta_t, valid = _history()
    output = module(
        boxes, delta_t, valid, torch.tensor([0.5, 1.0, 2.0]))
    assert torch.equal(output["prior_xy"], output["kinematic_prior_xy"])
    assert torch.allclose(
        torch.exp(output["log_sigma_parallel_perp"]),
        torch.full((3, 2), 0.5), atol=1e-6, rtol=0.0)
    assert not torch.allclose(
        output["envelope_parallel_perp"][0],
        output["envelope_parallel_perp"][2])
    assert torch.allclose(
        output["log_sigma_parallel_perp"][0],
        output["log_sigma_parallel_perp"][2])


@pytest.mark.parametrize("backend", ["gru", "cfc"])
def test_mean_and_sigma_gradients_are_isolated_and_extreme_tail_is_finite(
        backend):
    torch.manual_seed(13)
    module = OrderedPhysicalMotionEncoder(
        hidden_dim=128, step_dim=64, temporal_backend=backend,
        cfc_backbone_units=105, shared_kinematic_anchor=True,
        motion_aligned_uncertainty=True)
    with torch.no_grad():
        module.velocity_residual_head.weight.normal_(0.0, 0.02)

    output, terms = _loss_terms(module, query_gap=2.0)
    assert torch.isfinite(output["prior_xy"]).all()
    assert torch.isfinite(terms["mean_per_sample"]).all()
    assert torch.isfinite(terms["nll_per_sample"]).all()
    assert torch.isfinite(terms["gaussian_nll_per_sample"]).all()
    assert float(terms["tail_axis_fraction_per_sample"][-1]) == 1.0

    terms["mean_per_sample"].mean().backward()
    temporal = module.gru if backend == "gru" else module.cfc
    assert _nonzero_finite_gradient(temporal.parameters())
    assert all(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        for parameter in temporal.parameters())
    assert _nonzero_finite_gradient(module.context.parameters())
    assert _nonzero_finite_gradient(
        module.velocity_residual_head.parameters())
    assert all(
        parameter.grad is None
        for parameter in module.log_sigma_head.parameters())

    module.zero_grad(set_to_none=True)
    _, terms = _loss_terms(module, query_gap=2.0)
    terms["nll_per_sample"].mean().backward()
    assert _nonzero_finite_gradient(module.log_sigma_head.parameters())
    for name, parameter in module.named_parameters():
        if not name.startswith("log_sigma_head."):
            assert parameter.grad is None or not bool(torch.count_nonzero(
                parameter.grad)), name


@pytest.mark.parametrize("backend", ["gru", "cfc"])
@pytest.mark.parametrize("query_gap", [1.0, 2.0])
def test_gap2_gap4_losses_reach_all_selected_b1_paths(backend, query_gap):
    torch.manual_seed(23)
    module = OrderedPhysicalMotionEncoder(
        hidden_dim=128, step_dim=64, temporal_backend=backend,
        cfc_backbone_units=105, shared_kinematic_anchor=True,
        motion_aligned_uncertainty=True)
    with torch.no_grad():
        module.velocity_residual_head.weight.normal_(0.0, 0.02)
    _, terms = _loss_terms(module, query_gap=query_gap)
    loss = terms["mean_per_sample"].mean() + 0.05 * terms[
        "nll_per_sample"].mean()
    loss.backward()
    temporal = module.gru if backend == "gru" else module.cfc
    assert _nonzero_finite_gradient(temporal.parameters())
    assert _nonzero_finite_gradient(
        module.velocity_residual_head.parameters())
    assert _nonzero_finite_gradient(module.log_sigma_head.parameters())


class _Box:
    def __init__(self):
        self.center = np.zeros(3, dtype=np.float64)
        self.wlh = np.asarray((2.0, 4.0, 1.5), dtype=np.float64)
        self.orientation = Quaternion(axis=[0, 0, 1], radians=0.0)


def test_fixed_b2_geometry_is_invariant_to_b1_sigma():
    base = {
        "mu_xy": [3.0, 0.5],
        "velocity_xy": [2.0, 0.0],
        "direction_xy": [1.0, 0.0],
        "valid": True,
        "source_id": 1,
    }
    narrow, narrow_diagnostics = build_b1_uncertainty_support(
        _Box(), dict(base, log_sigma_parallel_perp=[-2.0, -2.0]),
        use_dynamic_sigma=False, fixed_margins=(2.0, 1.0))
    broad, broad_diagnostics = build_b1_uncertainty_support(
        _Box(), dict(base, log_sigma_parallel_perp=[2.5, 2.5]),
        use_dynamic_sigma=False, fixed_margins=(2.0, 1.0))
    np.testing.assert_array_equal(narrow.center, broad.center)
    np.testing.assert_array_equal(narrow.wlh, broad.wlh)
    assert narrow_diagnostics["length"] == broad_diagnostics["length"]
    assert narrow_diagnostics["width"] == broad_diagnostics["width"]


def test_recursive_reduction_balances_age_inside_each_query_gap():
    values = torch.tensor([1.0, 3.0, 10.0])
    valid = torch.ones(3)
    ages = torch.tensor([0.0, 2.0, 0.0])
    age_valid = torch.ones(3)
    gaps = torch.tensor([2, 2, 4])
    reduced = recursive_gap_age_balanced_mean(
        values, valid, recursive_age=ages,
        recursive_age_valid=age_valid,
        query_gap=gaps, query_gaps=(2, 4))
    # gap2 averages its two non-empty age cells to 2; gap4 is 10.  The two
    # query gaps then receive equal weight: (2 + 10) / 2 = 6.
    assert reduced.item() == pytest.approx(6.0)
    missing_validity = recursive_gap_age_balanced_mean(
        values, valid, recursive_age=ages,
        recursive_age_valid=None,
        query_gap=gaps, query_gaps=(2, 4))
    assert missing_validity.item() == 0.0


def test_safe_v25_rejects_dynamic_or_changed_b2_geometry():
    config = load_yaml_config(
        ROOT / "cfgs" / "ct_seqtrack" / "25_b1.yaml")
    config["search_v3_use_dynamic_sigma"] = True
    with pytest.raises(ValueError, match="fixes B2 geometry"):
        configure_ct_variant(config)

    config = load_yaml_config(
        ROOT / "cfgs" / "ct_seqtrack" / "25_b1.yaml")
    config["search_v3_fixed_margin_parallel"] = 3.0
    with pytest.raises(ValueError, match="fixed B2 margins 2m/1m"):
        configure_ct_variant(config)


def test_backend_and_loss_identity_bind_resume_calibration_and_replay():
    config = load_yaml_config(
        ROOT / "cfgs" / "ct_seqtrack" / "25_b1.yaml")
    configure_ct_variant(config)
    contract = build_online_resume_contract(config)
    checkpoint = {
        "ct_online_resume_contract": contract,
        "ct_epoch_boundary_complete": True,
        "ct_global_rng_state": {"schema": "ct_seqtrack.global_rng.v1"},
    }
    changed_backend = copy.deepcopy(config)
    changed_backend["motion_v3_temporal_backend"] = "cfc"
    with pytest.raises(ValueError, match="b1_temporal_backend"):
        validate_online_resume_contract(checkpoint, changed_backend)
    assert b1_calibration_config_sha256(config) != (
        b1_calibration_config_sha256(changed_backend))
    assert replay_config_sha256(config) != replay_config_sha256(
        changed_backend)

    changed_loss = copy.deepcopy(config)
    changed_loss["motion_v3_beta_nll_beta"] = 0.25
    with pytest.raises(ValueError, match="b1_beta_nll_beta"):
        validate_online_resume_contract(checkpoint, changed_loss)
    assert b1_calibration_config_sha256(config) != (
        b1_calibration_config_sha256(changed_loss))
    assert replay_config_sha256(config) != replay_config_sha256(changed_loss)

    changed_aux_gap = copy.deepcopy(config)
    changed_aux_gap["motion_v3_aux_transition_gaps"] = [2, 3]
    with pytest.raises(ValueError, match="b1_aux_transition_gaps"):
        validate_online_resume_contract(checkpoint, changed_aux_gap)
    assert b1_calibration_config_sha256(config) != (
        b1_calibration_config_sha256(changed_aux_gap))
    assert replay_config_sha256(config) != replay_config_sha256(
        changed_aux_gap)

    changed_cap = copy.deepcopy(config)
    changed_cap["ct_motion_max_displacement"] = 10.0
    with pytest.raises(ValueError, match="b1_max_displacement"):
        validate_online_resume_contract(checkpoint, changed_cap)
    assert b1_calibration_config_sha256(config) != (
        b1_calibration_config_sha256(changed_cap))
    assert replay_config_sha256(config) != replay_config_sha256(changed_cap)

    with pytest.raises(RuntimeError, match="incompatible schema"):
        validate_b1_calibration_state({
            "schema": "ct_seqtrack.b1_uncertainty_calibration.v2"}, {})


def test_b1_transaction_source_keeps_auxiliary_terms_in_backward_ledger():
    source = (ROOT / "models" / "seqtrack3d.py").read_text(
        encoding="utf-8")
    assert "B1 transaction diverged from its recorded weighted losses" in source
    assert "('loss_motion_v3_aux_prior'" in source
    assert "('loss_motion_v3_aux_nll'" in source


def test_backend_report_uses_paired_tracklet_promotion():
    reference = []
    candidate = []
    for tracklet_id in range(4):
        for frame_id in range(1, 4):
            common = {
                "tracklet_id": tracklet_id,
                "frame_id": frame_id,
                "candidate_id": 0,
                "b1_valid": 1,
                "kinematic_error": 1.5,
                "b1_nll": 0.2,
                "target_in_support": 1,
                "support_volume": 8.0,
                "b1_coverage_50": 1,
                "b1_coverage_80": 1,
                "b1_coverage_95": 1,
                "query_delta_t": 0.5,
                "current_target_points": 10.0,
                "recursive_age": float(frame_id - 1),
                "recursive_age_valid": 1,
            }
            reference.append(dict(common, learned_motion_error=1.0))
            candidate.append(dict(common, learned_motion_error=0.8))
    report = build_report(candidate, reference)
    comparison = report["backend_comparison"]
    assert comparison["matched_endpoints"] == 12
    assert comparison["candidate_minus_reference_rmse"]["ci95"][1] < 0
    assert comparison["promotion"]["passed"]
