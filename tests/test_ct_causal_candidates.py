import pytest
import torch

from models.ct_v2.motion import OrderedPhysicalMotionEncoder
from utils.ct_history import (
    build_causal_temporal_history_offsets,
    normalize_causal_temporal_gaps,
    select_causal_temporal_candidates,
    select_uniform_temporal_candidates,
)
from utils.training_isolation import causal_candidate_weight


def test_complete_temporal_histories_isolate_query_gap():
    assert build_causal_temporal_history_offsets(3, 2) == [2, 3, 4]
    assert build_causal_temporal_history_offsets(3, 4) == [4, 5, 6]
    assert build_causal_temporal_history_offsets(3, 8) == [8, 9, 10]
    assert normalize_causal_temporal_gaps([2, 4, 8]) == [2, 4, 8]
    assert [causal_candidate_weight(index) for index in range(3)] == [
        0.5, 0.3, 0.2]
    assert sum(causal_candidate_weight(index) for index in range(3)) == 1.0


def test_selector_uses_closest_outside_then_remaining_boundary():
    selected = select_causal_temporal_candidates(
        {2: 0.9, 4: 1.05, 8: 1.8}, {2: True, 4: True, 8: True})
    assert selected[2] == {
        "gap": 4, "available": True,
        "boundary_ratio": 1.05, "role_satisfied": True}
    assert selected[1]["gap"] == 2
    assert selected[1]["role_satisfied"] is True


def test_selector_without_outside_uses_two_largest_ratios():
    selected = select_causal_temporal_candidates(
        {2: 0.4, 4: 0.8, 8: 0.7}, {2: True, 4: True, 8: True})
    assert selected[1]["gap"] == 4
    assert selected[2]["gap"] == 8
    assert selected[2]["role_satisfied"] is False


def test_selector_handles_short_history_without_gt_fallback():
    selected = select_causal_temporal_candidates(
        {2: 1.1, 4: 3.0, 8: 9.0}, {2: True, 4: False, 8: False})
    assert selected[2]["gap"] == 2
    assert selected[1]["available"] is False
    assert selected[1]["gap"] is None


def test_selector_ties_choose_smaller_gap():
    selected = select_causal_temporal_candidates(
        {2: 0.8, 4: 0.8, 8: 0.1}, {2: True, 4: True, 8: True})
    assert selected[1]["gap"] == 2
    assert selected[2]["gap"] == 4


def test_uniform_control_is_deterministic_available_and_not_ratio_ranked():
    ratios = {2: 100.0, 4: 0.01, 8: 1.0}
    available = {2: True, 4: False, 8: True}
    first = select_uniform_temporal_candidates(
        ratios, available, seed_parts=(42, "track", 9))
    second = select_uniform_temporal_candidates(
        ratios, available, seed_parts=(42, "track", 9))
    assert first == second
    assert {first[1]["gap"], first[2]["gap"]} == {2, 8}
    assert first[1]["role_satisfied"] and first[2]["role_satisfied"]


@pytest.mark.parametrize("gaps", [[2], [1, 2], [4, 2], [2, 2]])
def test_registered_gap_validation_is_fail_closed(gaps):
    with pytest.raises(ValueError):
        normalize_causal_temporal_gaps(gaps)


def test_vectorized_gap_b1_matches_one_gap_at_a_time():
    torch.manual_seed(17)
    module = OrderedPhysicalMotionEncoder(
        hidden_dim=16, step_dim=8,
        motion_aligned_uncertainty=True,
        shared_kinematic_anchor=True).eval()
    ref_boxs = torch.randn(4, 3, 4)
    ref_boxs[:, 0] = 0.0
    delta_t = torch.tensor([
        [1.0, 1.0, 1.0],
        [2.0, 1.0, 1.0],
        [4.0, 1.0, 1.0],
        [8.0, 1.0, 1.0],
    ])
    valid = torch.ones(4, 3)
    query = delta_t[:, 0]
    with torch.no_grad():
        vectorized = module(ref_boxs, delta_t, valid, query)
        individual = [
            module(
                ref_boxs[row:row + 1], delta_t[row:row + 1],
                valid[row:row + 1], query[row:row + 1])
            for row in range(4)
        ]
    for key in (
            "mu_xy", "direction_xy", "log_sigma_parallel_perp",
            "valid", "source_id"):
        expected = torch.cat([item[key] for item in individual], dim=0)
        assert torch.allclose(vectorized[key], expected, atol=1e-6, rtol=1e-6)
