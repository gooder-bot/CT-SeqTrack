import math

import numpy as np
import pytest
import torch

from ctseqtrack.contracts import (
    MotionPriorOutput,
    motion_prior_covariance_xy,
    reexpress_motion_prior,
    validate_motion_prior_support_alignment,
)
from utils.candidate_utils import reexpress_motion_prediction


class Anchor:
    def __init__(self, x, y, yaw):
        self.center = np.asarray([x, y, 0.0], dtype=np.float64)
        cosine, sine = math.cos(yaw), math.sin(yaw)
        self.rotation_matrix = np.asarray([
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)


def make_prior(dtype=torch.float64, source=1):
    return MotionPriorOutput(
        center_xy=torch.tensor([[2.0, 1.0]], dtype=dtype),
        direction_xy=torch.tensor([[1.0, 0.0]], dtype=dtype),
        log_sigma=torch.log(torch.tensor([[2.0, 0.5]], dtype=dtype)),
        valid=torch.ones(1, dtype=dtype),
        source=torch.full((1,), source, dtype=torch.long),
    )


def test_motion_prior_identity_is_bitwise_exact():
    prior = make_prior(dtype=torch.float32)
    anchor = torch.tensor([[3.0, -4.0, 0.0, 0.3]])
    converted = reexpress_motion_prior(prior, anchor, anchor)
    assert torch.equal(converted.center_xy, prior.center_xy)
    assert torch.equal(converted.direction_xy, prior.direction_xy)
    assert converted.log_sigma is prior.log_sigma


def test_motion_prior_translation_and_rotation_reexpression():
    prior = make_prior()
    source = torch.tensor([[10.0, 0.0, 0.0, math.pi / 2]], dtype=torch.float64)
    target = torch.tensor([[8.0, 1.0, 0.0, 0.0]], dtype=torch.float64)
    converted = reexpress_motion_prior(prior, source, target)
    assert torch.allclose(
        converted.center_xy, torch.tensor([[1.0, 1.0]], dtype=torch.float64),
        atol=1e-12, rtol=0.0)
    assert torch.allclose(
        converted.direction_xy, torch.tensor([[0.0, 1.0]], dtype=torch.float64),
        atol=1e-12, rtol=0.0)


def test_motion_prior_degrees_matches_radians():
    prior = make_prior()
    source_radians = torch.tensor([[1.0, 2.0, 0.0, math.pi / 3]], dtype=torch.float64)
    target_radians = torch.tensor([[-2.0, 1.0, 0.0, -math.pi / 6]], dtype=torch.float64)
    source_degrees = source_radians.clone()
    target_degrees = target_radians.clone()
    source_degrees[:, 3] = torch.rad2deg(source_degrees[:, 3])
    target_degrees[:, 3] = torch.rad2deg(target_degrees[:, 3])
    radians = reexpress_motion_prior(prior, source_radians, target_radians)
    degrees = reexpress_motion_prior(
        prior, source_degrees, target_degrees, degrees=True)
    assert torch.allclose(degrees.center_xy, radians.center_xy, atol=1e-12)
    assert torch.allclose(degrees.direction_xy, radians.direction_xy, atol=1e-12)


def test_anisotropic_covariance_rotates_as_r_sigma_r_transpose():
    prior = make_prior()
    source = torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    target = torch.tensor([[0.0, 0.0, 0.0, -math.pi / 4]], dtype=torch.float64)
    converted = reexpress_motion_prior(prior, source, target)
    covariance_source = motion_prior_covariance_xy(prior)
    covariance_target = motion_prior_covariance_xy(converted)
    cosine = math.cos(math.pi / 4)
    sine = math.sin(math.pi / 4)
    rotation = torch.tensor(
        [[cosine, -sine], [sine, cosine]], dtype=torch.float64)
    expected = rotation @ covariance_source[0] @ rotation.T
    assert torch.allclose(covariance_target[0], expected, atol=1e-12)
    assert torch.equal(converted.log_sigma, prior.log_sigma)


def test_alignment_checks_only_live_learned_prior_rows():
    learned = make_prior(source=1)
    error = validate_motion_prior_support_alignment(
        learned, learned.center_xy + 5e-4, tolerance=1e-3)
    assert error.item() == pytest.approx(math.sqrt(2.0) * 5e-4)
    with pytest.raises(RuntimeError, match="different coordinate frames"):
        validate_motion_prior_support_alignment(
            learned, learned.center_xy + 2e-3, tolerance=1e-3)

    fallback = make_prior(source=2)
    validate_motion_prior_support_alignment(
        fallback, fallback.center_xy + 100.0, tolerance=1e-3)


@pytest.mark.parametrize("bad_anchor", [
    torch.tensor([[float("nan"), 0.0, 0.0, 0.0]]),
    torch.zeros(1, 3),
])
def test_motion_prior_reexpression_rejects_invalid_anchors(bad_anchor):
    prior = make_prior(dtype=torch.float32)
    with pytest.raises((ValueError, RuntimeError)):
        reexpress_motion_prior(prior, bad_anchor, torch.zeros(1, 4))


@pytest.mark.parametrize("field", ["center_xy", "direction_xy", "log_sigma"])
def test_motion_prior_reexpression_rejects_nonfinite_prior(field):
    prior = make_prior(dtype=torch.float32)
    values = getattr(prior, field).clone()
    values[0, 0] = float("nan")
    invalid = MotionPriorOutput(
        center_xy=values if field == "center_xy" else prior.center_xy,
        direction_xy=values if field == "direction_xy" else prior.direction_xy,
        log_sigma=values if field == "log_sigma" else prior.log_sigma,
        valid=prior.valid,
        source=prior.source,
    )
    with pytest.raises(ValueError, match="must be finite"):
        reexpress_motion_prior(invalid, torch.zeros(1, 4), torch.zeros(1, 4))


def test_support_prediction_uses_the_same_se2_reexpression():
    log_sigma = np.log(np.asarray([2.0, 0.5], dtype=np.float32))
    prediction = {
        "valid": True,
        "mu_xy": np.asarray([2.0, 1.0], dtype=np.float32),
        "direction_xy": np.asarray([1.0, 0.0], dtype=np.float32),
        "velocity_xy": np.asarray([3.0, 0.0], dtype=np.float32),
        "log_sigma_parallel_perp": log_sigma,
        "covariance_xy": np.diag(
            np.asarray([4.0, 0.25], dtype=np.float32)),
    }
    source = Anchor(10.0, 0.0, math.pi / 2)
    target = Anchor(8.0, 1.0, 0.0)
    converted = reexpress_motion_prediction(prediction, source, target)
    np.testing.assert_allclose(converted["mu_xy"], [1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(
        converted["direction_xy"], [0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(
        converted["velocity_xy"], [0.0, 3.0], atol=1e-6)
    np.testing.assert_allclose(
        converted["covariance_xy"], [[0.25, 0.0], [0.0, 4.0]],
        atol=1e-6)
    assert converted["log_sigma_parallel_perp"] is log_sigma


def test_support_prediction_identity_returns_original_dictionary():
    prediction = {
        "valid": True,
        "mu_xy": np.asarray([2.0, 1.0], dtype=np.float32),
    }
    anchor = Anchor(3.0, -2.0, 0.3)
    assert reexpress_motion_prediction(prediction, anchor, anchor) is prediction
