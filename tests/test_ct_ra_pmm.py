from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pyquaternion import Quaternion

from ctseqtrack.data.search import build_b1_uncertainty_support
from ctseqtrack.data.recursive import RecursiveTrackState
from ctseqtrack.model.base_losses import _auxiliary_prior_loss, _ra_pmm_view_loss
from ctseqtrack.model.prior import B1PhysicalTimePrior
from utils.candidate_utils import build_b1_physical_contract


class _Box:
    def __init__(self, center, yaw=0.0):
        self.center = np.asarray(center, dtype=np.float64)
        self.wlh = np.asarray((2.0, 4.0, 1.5), dtype=np.float64)
        self.orientation = Quaternion(axis=(0, 0, 1), radians=float(yaw))

    @property
    def rotation_matrix(self):
        return self.orientation.rotation_matrix


def _history(dt=0.5):
    # Newest velocity=2m/s, older velocity=1m/s, both along +x.
    boxes = torch.tensor(
        [[[0.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [-1.5, 0.0, 0.0, 0.0]]]
    )
    gaps = torch.full((1, 3), float(dt))
    return boxes, gaps, torch.ones(1, 3)


def _encoder():
    return B1PhysicalTimePrior(
        shared_kinematic_anchor=True,
        motion_aligned_uncertainty=True,
        ra_pmm=True,
        time_scale=0.5,
    )


def _flat_output(prior):
    return {
        "motion_prior_xy": prior["prior_xy"],
        "motion_prior_valid": prior["valid"],
        "motion_prior_kinematic_xy": prior["kinematic_prior_xy"],
        "motion_prior_direction_xy": prior["motion_direction_xy"],
        "motion_prior_mode_centers_xy": prior["mode_centers_xy"],
        "motion_prior_mode_probabilities": prior["mode_probabilities"],
        "motion_prior_expert_valid_mask": prior["expert_valid_mask"],
        "motion_prior_residual_acceleration_pp": prior["residual_acceleration_pp"],
        "motion_prior_residual_gate": prior["residual_gate"],
        "motion_prior_motion_quantiles_pp": prior["motion_quantiles_pp"],
        "motion_prior_support_quantiles_pp": prior["support_quantiles_pp"],
        "motion_prior_recoverability_probability": prior["recoverability_probability"],
    }


def _loss_model(encoder):
    return SimpleNamespace(
        physical_motion_encoder=encoder,
        motion_v3_hard_q50=0.0,
        motion_v3_hard_q90=10.0,
        motion_v3_dt_floor=0.05,
        motion_v3_mode_weight=0.2,
        motion_v3_acc_norm_weight=0.1,
        motion_v3_motion_quantile_weight=0.1,
        motion_v3_support_quantile_weight=0.1,
        motion_v3_recoverability_weight=0.1,
        motion_v3_censor_weight=0.05,
        motion_v3_acc_reg_weight=0.01,
        motion_v3_aux_query_gaps=(2, 4),
    )


def _data(target, dt):
    return {
        "motion_main_physical_target_xy": target,
        "motion_main_endpoint_target_xy": target.clone(),
        "motion_main_anchor_drift_xy": torch.zeros_like(target),
        "motion_main_gt_cv_difficulty": torch.zeros(target.shape[0]),
        "motion_main_support_cap_pp": torch.tensor([[4.0, 3.0]]),
        "motion_main_current_delta_t": torch.tensor([dt]),
    }


def test_ra_pmm_zero_init_reproduces_fixed_cv_ca_anchor_and_quantiles_are_ordered():
    encoder = _encoder()
    boxes, gaps, valid = _history()
    result = encoder(boxes, gaps, valid, torch.tensor([0.5]))
    # CV=1.0m, CA=1.25m, historical anchor=0.5*CV+0.5*CA.
    assert result["prior_xy"][0, 0].item() == pytest.approx(1.125, abs=5e-4)
    assert result["mode_probabilities"][0, :2].tolist() == pytest.approx(
        [0.5, 0.5], abs=3e-4
    )
    assert torch.all(
        result["motion_quantiles_pp"][:, 1:] > result["motion_quantiles_pp"][:, :-1]
    )
    assert torch.all(
        result["support_quantiles_pp"][:, 1:] > result["support_quantiles_pp"][:, :-1]
    )
    assert result["support_quantiles_pp"][0, 2].tolist() == pytest.approx([2.0, 1.0])


def test_quantile_heads_do_not_backpropagate_into_mean_context():
    encoder = _encoder()
    boxes, gaps, valid = _history()
    result = encoder(boxes, gaps, valid, torch.tensor([0.5]))
    (
        result["motion_quantiles_pp"].sum() + result["support_quantiles_pp"].sum()
    ).backward()
    assert encoder.motion_quantile_head.weight.grad is not None
    assert encoder.support_quantile_head.weight.grad is not None
    for parameter in encoder.context.parameters():
        assert parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
    for parameter in encoder.gru.parameters():
        assert parameter.grad is None or torch.count_nonzero(parameter.grad) == 0


@pytest.mark.parametrize("dt", (0.1, 0.2, 0.4))
def test_normalized_acceleration_supervision_updates_head_for_every_gap(dt):
    encoder = _encoder()
    boxes, gaps, valid = _history(dt=dt)
    result = encoder(boxes, gaps, valid, torch.tensor([dt]))
    target = result["kinematic_prior_xy"].detach() + torch.tensor([[0.3, 0.2]])
    transaction, _, _ = _ra_pmm_view_loss(
        _loss_model(encoder),
        _data(target, dt),
        _flat_output(result),
        "motion_main",
        "motion",
        1.0,
    )
    transaction.backward()
    gradient = encoder.residual_acceleration_head.weight.grad
    assert gradient is not None
    assert torch.linalg.norm(gradient).item() > 0.0


def test_ambiguous_experts_skip_mode_classification():
    encoder = _encoder()
    boxes, gaps, valid = _history()
    result = encoder(boxes, gaps, valid, torch.tensor([0.5]))
    # A lateral target makes the straight CV/CA/CTRV experts nearly tied.
    target = torch.tensor([[1.1, 10.0]])
    _, _, losses = _ra_pmm_view_loss(
        _loss_model(encoder),
        _data(target, 0.5),
        _flat_output(result),
        "motion_main",
        "motion",
        1.0,
    )
    assert losses["motion_v3_mode_skip_rate"].item() == pytest.approx(1.0)
    assert losses["loss_motion_v3_mode"].item() == pytest.approx(0.0)


@pytest.mark.parametrize("backend", ("gru", "cfc"))
def test_invalid_history_batch_cannot_poison_next_ra_pmm_step(backend):
    encoder = B1PhysicalTimePrior(
        shared_kinematic_anchor=True,
        motion_aligned_uncertainty=True,
        ra_pmm=True,
        temporal_backend=backend,
        cfc_backbone_units=105,
        time_scale=0.5,
    )
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    boxes, gaps, _ = _history()

    # A cold-start recursive row has no valid analytic expert.  Its sorted
    # expert errors are all +inf and must contribute an exact finite zero.
    invalid_result = encoder(boxes, gaps, torch.zeros(1, 3), torch.tensor([0.5]))
    invalid_loss, _, invalid_metrics = _ra_pmm_view_loss(
        _loss_model(encoder),
        _data(torch.zeros(1, 2), 0.5),
        _flat_output(invalid_result),
        "motion_main",
        "motion",
        1.0,
    )
    assert torch.isfinite(invalid_loss)
    assert all(torch.isfinite(value) for value in invalid_metrics.values())
    optimizer.zero_grad()
    invalid_loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in encoder.parameters()
    )
    optimizer.step()

    # The following valid row previously reached BCE with NaN parameters and
    # raised a CUDA device-side assert on the second batch.
    valid_result = encoder(boxes, gaps, torch.ones(1, 3), torch.tensor([0.5]))
    valid_target = valid_result["kinematic_prior_xy"].detach()
    valid_loss, _, valid_metrics = _ra_pmm_view_loss(
        _loss_model(encoder),
        _data(valid_target, 0.5),
        _flat_output(valid_result),
        "motion_main",
        "motion",
        1.0,
    )
    assert torch.isfinite(valid_loss)
    assert all(torch.isfinite(value) for value in valid_metrics.values())


def test_physical_target_is_translation_invariant_but_endpoint_tracks_anchor_drift():
    gt_history = [_Box((0.0, 0.0, 0.0)), _Box((-1.0, 0.0, 0.0))]
    current = _Box((1.0, 0.0, 0.0))
    first = build_b1_physical_contract(
        current,
        gt_history,
        [_Box((0.0, 0.0, 0.0)), _Box((-1.0, 0.0, 0.0))],
        0.5,
        history_delta_t=(0.5, 0.5),
        history_valid_mask=(1, 1),
    )
    translated = build_b1_physical_contract(
        current,
        gt_history,
        [_Box((5.0, 0.0, 0.0)), _Box((4.0, 0.0, 0.0))],
        0.5,
        history_delta_t=(0.5, 0.5),
        history_valid_mask=(1, 1),
    )
    np.testing.assert_allclose(
        first["physical_target_xy"], translated["physical_target_xy"]
    )
    np.testing.assert_allclose(
        translated["endpoint_target_xy"] - first["endpoint_target_xy"],
        np.asarray((-5.0, 0.0)),
    )


def test_targets_are_detached_and_unrecoverable_rows_train_risk_heads():
    encoder = _encoder()
    boxes, gaps, valid = _history()
    result = encoder(boxes, gaps, valid, torch.tensor([0.5]))
    physical_target = torch.tensor([[1.3, 0.2]], requires_grad=True)
    data = _data(physical_target, 0.5)
    data["motion_main_endpoint_target_xy"] = torch.tensor([[20.0, 10.0]])
    transaction, _, losses = _ra_pmm_view_loss(
        _loss_model(encoder),
        data,
        _flat_output(result),
        "motion_main",
        "motion",
        1.0,
    )
    transaction.backward()
    assert physical_target.grad is None
    assert losses["loss_motion_v3_censor"].item() > 0.0
    assert encoder.support_quantile_head.weight.grad is not None
    assert torch.linalg.norm(encoder.support_quantile_head.weight.grad).item() > 0.0
    assert encoder.recoverability_head.weight.grad is not None
    assert torch.linalg.norm(encoder.recoverability_head.weight.grad).item() > 0.0


def test_auxiliary_additions_are_owned_by_b1_transaction():
    model = SimpleNamespace(
        motion_v3_ra_pmm=False,
        use_calibrated_motion_uncertainty=False,
        use_ct_joint_full=False,
        motion_v3_aux_prior_weight=0.1,
        motion_v3_aux_nll_weight=0.0,
        motion_v3_aux_query_gaps=(2, 4),
    )
    output = {
        "motion_prior_xy": torch.zeros(1, 2),
        "motion_aux_prior_xy": torch.tensor([[0.5, 0.0]], requires_grad=True),
        "motion_aux_prior_kinematic_xy": torch.zeros(1, 2),
        "motion_aux_prior_valid": torch.ones(1),
        "motion_aux_prior_gap_ratio": torch.ones(1),
    }
    data = {
        "motion_aux_target_xy": torch.tensor([[1.0, 0.0]]),
        "motion_aux_valid_mask": torch.ones(1, 3),
        "motion_aux_query_gap_frames": torch.tensor([2]),
    }
    transaction, additions, _ = _auxiliary_prior_loss(model, data, output)
    assert transaction.item() == pytest.approx(sum(item.item() for item in additions))
    transaction.backward()
    assert output["motion_aux_prior_xy"].grad is not None


def test_dynamic_support_uses_q95_with_floor_cap_and_saturation():
    latest = _Box((0.0, 0.0, 0.0))
    base_prediction = {
        "mu_xy": np.asarray((1.0, 0.0)),
        "velocity_xy": np.asarray((2.0, 0.0)),
        "direction_xy": np.asarray((1.0, 0.0)),
        "log_sigma_parallel_perp": np.log(np.asarray((0.5, 0.5))),
        "valid": True,
        "source_id": 1,
    }
    prediction = dict(base_prediction)
    prediction["support_quantiles_pp"] = np.asarray(
        ((0.1, 0.1), (0.2, 0.2), (0.5, 0.4))
    )
    _, diagnostics = build_b1_uncertainty_support(
        latest,
        prediction,
        use_dynamic_sigma=True,
        coverage_scale=1.0,
        fixed_margins=(2.0, 1.0),
        max_margins=(4.0, 3.0),
    )
    assert diagnostics["sigma_parallel"] == pytest.approx(2.0)
    assert diagnostics["sigma_perpendicular"] == pytest.approx(1.0)
    assert diagnostics["support_saturated"] is False

    prediction["support_quantiles_pp"][2] = (8.0, 6.0)
    _, diagnostics = build_b1_uncertainty_support(
        latest,
        prediction,
        use_dynamic_sigma=True,
        coverage_scale=1.0,
        fixed_margins=(2.0, 1.0),
        max_margins=(4.0, 3.0),
    )
    assert diagnostics["sigma_parallel"] == pytest.approx(4.0)
    assert diagnostics["sigma_perpendicular"] == pytest.approx(3.0)
    assert diagnostics["support_saturated"] is True


def test_recursive_diagnostics_are_detached_causal_state_and_reanchors_invalidate_them():
    state = RecursiveTrackState(0, "scene/track", _Box((0.0, 0.0, 0.0)))
    diagnostic = np.arange(6, dtype=np.float32)
    state.append(
        1,
        _Box((1.0, 0.0, 0.0)),
        timestamp=0.5,
        observation_diagnostics=diagnostic,
        diagnostic_valid=True,
    )
    diagnostic[:] = -1.0
    contract = state.history_contract((1, 0), (1, 1))
    np.testing.assert_array_equal(
        contract["history_observation_diagnostics"][0], np.arange(6)
    )
    np.testing.assert_array_equal(contract["history_diagnostic_valid_mask"], (1, 0))
    state.reseed_history(
        (1,),
        (_Box((1.1, 0.0, 0.0)),),
        timestamps=(0.5,),
        before_frame_id=2,
        rollout_horizon=1,
    )
    reanchored = state.history_contract((1,), (1,))
    assert reanchored["history_diagnostic_valid_mask"].tolist() == [0]
    np.testing.assert_array_equal(
        reanchored["history_observation_diagnostics"], np.zeros((1, 6))
    )
