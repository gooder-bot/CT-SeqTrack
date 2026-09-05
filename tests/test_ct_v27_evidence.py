"""v27 B2 的真实点身份、稀疏输入、局部几何与梯度所有权边界。"""

import copy

import pytest
import torch

from models.ct_v2.evidence_memory import (
    B2EvidenceAcquirer, _fps_indices, build_box_memory_tokens)
from models.ct_v2.evidence_v27 import (
    ExtensionLocalGeometry, evidence_box_geometry, unique_point_mask)


def _module():
    torch.manual_seed(127)
    return B2EvidenceAcquirer(
        v27_enabled=True, relation_aware_sampling=True,
        robust_consensus_voting=True).eval()


def _inputs(extension_count=40, base_count=12):
    generator = torch.Generator().manual_seed(271)
    points = torch.zeros(1, 768, 5)
    points[:, :extension_count] = torch.randn(
        1, extension_count, 5, generator=generator) * 0.4
    ids = torch.full((1, 768), -1, dtype=torch.long)
    ids[:, :extension_count] = torch.arange(extension_count) + 10000
    base_ids = torch.full((1, 1024), -1, dtype=torch.long)
    base_ids[:, :base_count] = torch.arange(base_count)
    base_features = torch.zeros(1, 1024, 64)
    base_features[:, :base_count] = torch.randn(
        1, base_count, 64, generator=generator)
    metadata = torch.zeros(1, 36, 8)
    metadata[:, [0, 1, 12, 13, 24, 25], 6] = 1.0
    memory_valid = torch.zeros(1, 36, dtype=torch.bool)
    memory_valid[:, [0, 1, 8, 12, 13, 20, 24, 25, 32]] = True
    return {
        "extension_points": points,
        "extension_valid_mask": ids >= 0,
        "extension_source": torch.where(ids >= 0, torch.ones_like(ids), torch.zeros_like(ids)),
        "extension_point_ids": ids,
        "current_base_features": base_features,
        "current_base_valid_mask": base_ids >= 0,
        "current_base_point_ids": base_ids,
        "memory_tokens": torch.randn(1, 36, 64, generator=generator),
        "memory_valid_mask": memory_valid,
        "memory_metadata": metadata,
        "observation_box": torch.tensor([[0.1, -0.2, 0.3, 0.4]]),
        "observation_stats": torch.zeros(1, 5),
        "b1_center_xy": torch.zeros(1, 2),
        "b1_sigma_parallel_perp": torch.ones(1, 2),
        "b1_direction_xy": torch.tensor([[1.0, 0.0]]),
        "b1_valid": torch.zeros(1),
        "query_delta_t": torch.tensor([0.5]),
        "gap_ratio": torch.ones(1),
        "first_box_size_wlh": torch.tensor([[1.8, 4.2, 1.6]]),
    }


def test_fps_never_repeats_coincident_xy_and_is_id_stable():
    xyz = torch.zeros(20, 2)
    ids = torch.arange(100, 120)
    first = _fps_indices(xyz, 30, ids)
    assert torch.unique(first).numel() == 20
    permutation = torch.randperm(20)
    second = _fps_indices(xyz[permutation], 30, ids[permutation])
    assert torch.equal(ids[first], ids[permutation][second])


def test_unique_mask_counts_original_measurements_and_checks_explicit_mask():
    ids = torch.tensor([[3, 3, 7, -1]])
    valid = ids >= 0
    assert unique_point_mask(ids, valid).tolist() == [[True, False, True, False]]
    with pytest.raises(ValueError, match="every valid ID once"):
        unique_point_mask(ids, valid, torch.tensor([[True, True, False, False]]))


def test_memory_uses_wlh_axes_unique_ids_and_ignores_invalid_zero_points():
    points = torch.zeros(1, 3, 5, 3)
    points[0, 0, :3] = torch.tensor([[2.5, 0., 0.], [2.5, 0., 0.], [0., 1.5, 0.]])
    ids = torch.full((1, 3, 5), -1, dtype=torch.long)
    ids[0, 0, :3] = torch.tensor([11, 11, 12])
    features = torch.randn(1, 3, 5, 64, requires_grad=True)
    tokens, valid, metadata, memory_ids = build_box_memory_tokens(
        features, points, torch.zeros(1, 3, 4), torch.tensor([[2., 6., 2.]]),
        torch.tensor([[1, 0, 0]]), v27_enabled=True,
        history_point_ids=ids, history_point_valid_mask=ids >= 0,
        return_metadata=True, return_point_ids=True)
    assert tokens.shape == (1, 36, 64)
    assert valid.sum().item() == 2
    assert memory_ids[0, 0].item() == 11
    assert memory_ids[0, 8].item() == 12
    assert metadata[0, 0, 6].item() == 1
    assert metadata[0, 8, 6].item() == 0
    assert torch.all(memory_ids[~valid] == -1)
    assert not tokens.requires_grad


@pytest.mark.parametrize("missing", ["extension_point_ids", "current_base_point_ids", "first_box_size_wlh"])
def test_v27_requires_explicit_point_identity_and_first_frame_size(missing):
    values = _inputs()
    del values[missing]
    with pytest.raises(ValueError):
        _module()(**values)


def test_v27_rejects_duplicate_extension_and_b0_overlap_ids():
    values = _inputs()
    values["extension_point_ids"][0, 1] = values["extension_point_ids"][0, 0]
    with pytest.raises(ValueError, match="must be unique"):
        _module()(**values)
    values = _inputs()
    values["extension_point_ids"][0, 0] = 0
    with pytest.raises(ValueError, match="intersect"):
        _module()(**values)


def test_v27_rejects_non_minus_one_padding_ids():
    values = _inputs()
    values["extension_point_ids"][0, -1] = 999
    with pytest.raises(ValueError, match="padding IDs"):
        _module()(**values)


def test_v27_has_fixed_dimensions_and_unique_relation_coverage_exploration_selection():
    values = _inputs(extension_count=768)
    module = _module()
    output = module(**values)
    assert module.local_geometry.edge_mlp[0].in_features == 131
    assert module.relation_context_encoder[0].in_features == 384
    assert module.relation_head[0].in_features == 132
    assert module.vote_head[0].in_features == 68
    assert module.extension_presence_head[0].in_features == 134
    assert output["ct_relation_logits_prepool"].shape == (1, 768)
    assert output["ct_search_point_votes"].shape == (1, 256, 2)
    assert torch.unique(output["ct_extension_selected_point_ids"]).numel() == 256
    groups = output["ct_extension_selected_group"]
    assert [(groups == i).sum().item() for i in (1, 2, 3)] == [128, 96, 32]
    assert output["ct_b2_available"].item() == 1


def test_v27_is_invariant_to_extension_permutation_and_masked_padding_values():
    module = _module()
    values = _inputs(extension_count=300)
    first = module(**values)
    permutation = torch.randperm(768)
    changed = copy.deepcopy(values)
    for key in ("extension_points", "extension_valid_mask", "extension_source", "extension_point_ids"):
        changed[key] = changed[key][:, permutation]
    changed["extension_points"][~changed["extension_valid_mask"]] = float("nan")
    changed["current_base_features"][~changed["current_base_valid_mask"]] = float("nan")
    changed["memory_tokens"][~changed["memory_valid_mask"]] = float("nan")
    changed["memory_metadata"][~changed["memory_valid_mask"]] = float("nan")
    second = module(**changed)
    assert torch.equal(first["ct_extension_selected_point_ids"], second["ct_extension_selected_point_ids"])
    for key in ("ct_b2_raw_box", "ct_search_targetness_logits", "ct_vote_effective_mass",
                "ct_vote_mode_unique_count", "ct_vote_mode_mean_identity_margin"):
        torch.testing.assert_close(first[key], second[key], atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("point_count", [0, 1, 2])
def test_v27_sparse_extension_keeps_true_count_and_empty_base_can_recover(point_count):
    values = _inputs(extension_count=point_count, base_count=0)
    values["memory_valid_mask"].zero_()
    output = _module()(**values)
    assert output["ct_search_extension_selected_count"].item() == point_count
    assert output["ct_base_unique_valid_mask"].sum().item() == 0
    assert torch.isfinite(output["ct_b2_raw_box"]).all()
    assert output["ct_b2_available"].item() == int(point_count > 0)
    if point_count == 0:
        assert torch.equal(output["ct_b2_raw_box"], values["observation_box"])
        assert output["ct_vote_mode_unique_count"].item() == 0
        assert output["ct_b2_extension_presence_probability"].item() == 0
        assert output["ct_search_normalized_ess"].item() == 0


def test_local_geometry_never_fills_missing_neighbors_with_far_points():
    module = ExtensionLocalGeometry()
    features = torch.randn(1, 3, 64, requires_grad=True)
    xyz = torch.tensor([[[0., 0., 0.], [.1, 0., 0.], [10., 0., 0.]]])
    output, counts = module(features, xyz, torch.ones(1, 3, dtype=torch.bool),
                            torch.tensor([[3, 2, 1]]), torch.tensor([.3]))
    assert counts.tolist() == [[1, 1, 0]]
    torch.testing.assert_close(output[:, 2], module.output_norm(features[:, 2]))


def test_long_vehicle_vote_radius_reaches_tail_center_without_changing_car_default():
    size = torch.tensor([[1.8, 4.2, 1.6], [.6, .8, 1.8], [2.5, 14., 3.]])
    lwh, local_radius, vote_radius = evidence_box_geometry(size, 3, size)
    assert local_radius.tolist() == pytest.approx([.9, .3, 1.0])
    assert vote_radius[:2].tolist() == [4.0, 4.0]
    assert vote_radius[2].item() > 7.0
    values = _inputs(extension_count=1)
    values["first_box_size_wlh"] = size[2:]
    values["extension_points"][0, 0, :2] = torch.tensor([7., 0.])
    module = _module()
    with torch.no_grad():
        for parameter in module.vote_head.parameters():
            parameter.zero_()
        module.vote_head[-1].bias[0] = torch.atanh(-7.0 / vote_radius[2])
    output = module(**values)
    torch.testing.assert_close(output["ct_b2_raw_box"][0, :2], torch.zeros(2), atol=1e-5, rtol=0)


def test_targetness_alone_weights_votes_and_raw_loss_does_not_train_relation_head():
    module = _module()
    output = module(**_inputs())
    mask = output["ct_extension_selected_valid_mask"]
    torch.testing.assert_close(output["ct_search_vote_weights"],
                               torch.sigmoid(output["ct_search_targetness_logits"]) * mask)
    gradient = torch.autograd.grad(output["ct_b2_raw_box"].sum(),
                                   module.relation_head[-1].weight, allow_unused=True)[0]
    assert gradient is None


def test_v27_all_upstream_inputs_are_detached_but_b2_geometry_and_heads_train():
    values = _inputs()
    upstream = ("extension_points", "current_base_features", "memory_tokens", "memory_metadata",
                "observation_box", "b1_center_xy", "b1_sigma_parallel_perp", "b1_direction_xy",
                "b1_valid", "query_delta_t", "gap_ratio", "first_box_size_wlh")
    for key in upstream:
        values[key].requires_grad_()
    module = _module()
    with torch.no_grad():
        module.attention_residual_gate.fill_(0.5)
    output = module(**values)
    (output["ct_b2_raw_box"].sum()
     + output["ct_relation_logits_prepool"].sum()
     + output["ct_search_targetness_logits"].sum()).backward()
    assert all(values[key].grad is None for key in upstream)
    for parameter in (module.local_geometry.edge_mlp[0].weight,
                      module.relation_head[-1].weight, module.vote_head[-1].weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum().item() > 0


def test_mode_absolute_mass_distinguishes_weak_votes_with_same_geometric_consensus():
    votes = torch.tensor([[[3., 0.], [3.1, 0.], [3., .1], [3.1, .1]]])
    valid = torch.ones(1, 4, dtype=torch.bool)
    kwargs = {"identity_margin": torch.ones(1, 4), "point_ids": torch.arange(4).unsqueeze(0)}
    strong = B2EvidenceAcquirer._consensus_vote(votes, torch.full((1, 4), .9), valid,
                                               torch.zeros(1, 2), **kwargs)
    weak = B2EvidenceAcquirer._consensus_vote(votes, torch.full((1, 4), .01), valid,
                                             torch.zeros(1, 2), **kwargs)
    torch.testing.assert_close(strong["consistency"], weak["consistency"])
    assert strong["effective_mass"].item() == pytest.approx(3.6)
    assert weak["effective_mass"].item() == pytest.approx(.04)
    assert weak["mode_unique_count"].item() == 4
    assert weak["mode_mean_targetness"].item() == pytest.approx(.01)


def test_exact_top_mode_inlier_mask_maps_back_to_selected_slots_after_id_sorting():
    votes = torch.tensor([[[8., 0.], [3., 0.], [3.1, 0.], [99., 99.], [3., .1]]])
    valid = torch.tensor([[True, True, True, False, True]])
    weights = torch.tensor([[.2, .9, .8, 0., .7]])
    ids = torch.tensor([[40, 30, 10, -1, 20]])
    summary = B2EvidenceAcquirer._consensus_vote(votes, weights, valid, torch.zeros(1, 2),
        identity_margin=torch.zeros_like(weights), point_ids=ids)
    assert summary['mode_inlier_mask'].tolist() == [[False, True, True, False, True]]
    assert summary['mode_inlier_mask'].sum() == summary['mode_unique_count'][0]
    permutation = torch.tensor([2, 4, 0, 3, 1])
    changed = B2EvidenceAcquirer._consensus_vote(votes[:, permutation], weights[:, permutation],
        valid[:, permutation], torch.zeros(1, 2), identity_margin=torch.zeros_like(weights),
        point_ids=ids[:, permutation])
    assert torch.equal(summary['mode_inlier_mask'][:, permutation], changed['mode_inlier_mask'])
    empty = _module()(**_inputs(extension_count=0))
    assert empty['ct_vote_mode_inlier_mask'].shape == (1, 256)
    assert not empty['ct_vote_mode_inlier_mask'].any()
