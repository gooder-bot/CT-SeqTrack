"""v31 支持域、身份记忆、唯一点预算及可微测量合同（CPU）。"""
import numpy as np
import pytest
import torch
from types import SimpleNamespace

from models.ct_v31.acquisition import (acquire_extension, band_grid_target,
    build_dual_support, support_membership)
from models.ct_v31.contracts import ObservationFeatures
from models.ct_v31.evidence import (B2IdentityEvidence, RawLocalGeometry,
                                    build_live_modes, select_evidence_points)
from models.ct_v31.memory import RawIdentityMemory


def _geometry(**kwargs):
    result = dict(b0_box=np.array([0., 0., 0., 0.]), box_size=np.array([4., 2., 2.]),
                  prior_center=np.array([1., 0., 0.]), recovery_center=np.array([30., 0., 10.]))
    result.update(kwargs)
    return result


def test_recovery_is_endpoint_only_with_own_vertical_support():
    support = build_dual_support(u=[0., 0.], **_geometry())
    cloud = np.array([[30., 0., 10.], [15., 0., 5.], [30., 0., 0.], [5., 0., 0.]])
    a, b = support_membership(cloud, support, [], np.arange(4))
    assert a.tolist() == [False, False, False, False]
    assert b.tolist() == [True, False, False, False]
    assert support["recovery"]["half"].tolist() == [6., 4., 3.25]
    maximum = build_dual_support(u=[1., 1.], **_geometry())
    assert maximum["recovery"]["half"].tolist() == [14., 9., 3.25]


def test_raw_id_exclusion_and_padding_keep_distinct_same_position_points():
    cloud = np.array([[2., 0., 0., .2, .3]] * 4 + [[30., 0., 10., .6, .8]])
    raw_ids = np.array([10, 11, 11, 12, 13])
    result = acquire_extension(cloud, raw_ids, b0_raw_ids=[10], anchor=[1., 0., 0.],
                               u=[0., 0.], **_geometry())
    assert result["extension_valid"].sum() == 3
    assert set(result["extension_ids"][:3]) == {11, 12, 13}
    assert set(result["extension_partition"][:3]) == {0, 1}
    assert (result["extension_ids"][3:] == -1).all()
    assert (result["extension_partition"][3:] == -1).all()
    row = np.flatnonzero(result["extension_ids"] == 13)[0]
    np.testing.assert_allclose(result["extension_points"][row], [29., 0., 10., .6, .8])


def test_pool_borrows_without_losing_true_partition_or_unique_ids():
    rng = np.random.default_rng(31)
    cloud = rng.uniform(-1., 1., (900, 5))
    result = acquire_extension(cloud, np.arange(900), b0_raw_ids=[], anchor=[0., 0., 0.],
                               u=[0., 0.], **_geometry())
    assert result["extension_valid"].sum() == 768
    assert len(np.unique(result["extension_ids"])) == 768
    assert (result["extension_partition"] == 0).all()


def test_band_label_uses_same_support_maximum_unique_target_set():
    cloud = np.array([[43., 0., 10.], [30., 8., 10.], [13., 0., 5.], [43., 0., 10.]])
    ids, fg = np.array([1, 2, 3, 1]), np.ones(4, dtype=bool)
    label = band_grid_target(cloud, ids, target_mask=fg, b0_raw_ids=[], **_geometry())
    assert label["acquisition_valid"] and label["acquisition_demand"]
    assert label["maximum_target_count"] == 2  # corridor point and duplicate are excluded
    support = build_dual_support(u=label["acquisition_target"], **_geometry())
    a, b = support_membership(cloud, support, [], ids)
    assert ((a | b) & fg).sum() == label["target_count"] == 2
    no_target = band_grid_target(cloud, ids, target_mask=~fg, b0_raw_ids=[], **_geometry())
    assert no_target["acquisition_valid"] and not no_target["acquisition_demand"]
    np.testing.assert_array_equal(no_target["acquisition_target"], [0., 0.])


def test_memory_keeps_persistent_template_and_two_accepted_raw_frames():
    size = [4., 2., 2.]
    memory = RawIdentityMemory(size)
    points = np.zeros((20, 5))
    points[:, 0] = np.linspace(-2., 2., 20)
    fg = np.arange(20) < 14
    memory.initialize(points, np.arange(20), fg, [0., 0., 0., 0.], 0.)
    initial = memory.export(0.)
    assert initial["memory_valid"].sum() == 12
    assert not memory.update(points, np.arange(20), fg.astype(float), [0., 0., 0., 0.], 1., .49)
    for time in (1., 2., 3.):
        assert memory.update(points + [time, 0., 0., 0., 0.], np.arange(20), fg.astype(float),
                             [time, 0., 0., 0.], time, .5)
    result = memory.export(4.)
    assert result["memory_valid"].sum() == 36
    np.testing.assert_array_equal(result["memory_points"][:12], initial["memory_points"][:12])
    np.testing.assert_array_equal(result["memory_metadata"][:12, 3], np.full(12, 4.))
    np.testing.assert_array_equal(result["memory_metadata"][12:24, 3], np.full(12, 2.))
    np.testing.assert_array_equal(result["memory_metadata"][24:, 3], np.full(12, 1.))
    assert result["memory_metadata"][12:24, 1].sum() == 8
    assert result["memory_metadata"][12:24, 2].sum() == 4


def test_memory_canonical_rotation_and_unique_fg_gate():
    memory = RawIdentityMemory([4., 2., 2.])
    points = np.array([[10., 22., 1.], [10., 20., 1.], [10., 18., 1.]])
    memory.initialize(points, [1, 2, 3], [True] * 3, [10., 20., 1., np.pi / 2], 0.)
    exported = memory.export(0.)
    row = np.flatnonzero(exported["memory_ids"] == 1)[0]
    np.testing.assert_allclose(exported["memory_points"][row, :3], [.5, 0., 0.], atol=1e-6)
    assert not memory.update(points, [1, 1, 2], [1., 1., 1.], [10., 20., 1., 0.], 1., .9)


def test_point_selection_guarantees_partition_quotas_and_borrows_unique_slots():
    torch.manual_seed(31)
    points = torch.randn(2, 768, 5)
    ids = torch.arange(768).repeat(2, 1)
    partition = torch.cat((torch.zeros(384, dtype=torch.long), torch.ones(384, dtype=torch.long)))[None].repeat(2, 1)
    valid = torch.ones(2, 768, dtype=torch.bool)
    valid[1, 386:] = False
    logits = torch.linspace(-2., 2., 768)[None].repeat(2, 1)
    selected = select_evidence_points(points, logits, valid, ids, partition)
    for row in selected:
        assert (row >= 0).sum() == 256
        assert len(row.unique()) == 256
    selected_partition = partition.gather(1, selected)
    assert (selected_partition[0] == 0).sum() == 128
    assert (selected_partition[0] == 1).sum() == 128
    assert (selected_partition[1] == 1).sum() == 2
    # 每区 identity 前64个均保留，不被空间采样覆盖。
    assert set(range(320, 384)).issubset(set(selected[0].tolist()))
    assert set(range(704, 768)).issubset(set(selected[0].tolist()))


def test_point_selection_permutation_and_empty_input():
    torch.manual_seed(32)
    points, logits = torch.randn(1, 768, 5), torch.randn(1, 768)
    ids = torch.arange(768)[None]
    valid, partition = torch.ones_like(ids, dtype=torch.bool), ids.remainder(2)
    first = select_evidence_points(points, logits, valid, ids, partition)
    order = torch.randperm(768)
    second = select_evidence_points(points[:, order], logits[:, order], valid[:, order], ids[:, order], partition[:, order])
    assert torch.equal(ids.gather(1, first), ids[:, order].gather(1, second))
    empty = select_evidence_points(points, logits, ~valid, ids, partition)
    assert (empty == -1).all()


def test_modes_keep_live_xyz_and_weights_but_fixed_members_and_covariance():
    votes = torch.tensor([[[0., 0., 4.], [.2, 0., 6.], [5., 0., 10.]]], requires_grad=True)
    weights = torch.tensor([[.9, .8, .4]], requires_grad=True)
    centers, valid, members, covariance = build_live_modes(votes, weights, torch.ones(1, 3, dtype=torch.bool),
                                                           torch.tensor([[1, 2, 3]]))
    assert valid.sum() == 2
    assert not members.requires_grad and not covariance.requires_grad
    assert centers[0, 0, 2].item() == pytest.approx((4 * .9 + 6 * .8) / 1.7)
    centers[0, 0].sum().backward()
    assert votes.grad[0, :2, 2].abs().sum() > 0
    assert weights.grad[0, :2].abs().sum() > 0
    assert votes.grad[0, 2].abs().sum() == 0


def _evidence_inputs(empty=False):
    torch.manual_seed(33)
    count = 16
    observation = ObservationFeatures(torch.zeros(1, 4), torch.randn(1, 4, count, 64, requires_grad=True),
        torch.randn(1, 4, 4, 128), torch.ones(1, 4, 4, dtype=torch.bool),
        torch.zeros(1, 4, count, 2), torch.zeros(1, 4, count, 9), torch.ones(1, 4, count),
        torch.ones(1, 4), torch.tensor([not empty]))
    batch = dict(extension_points=torch.randn(1, 768, 5), extension_ids=torch.arange(768)[None],
        extension_valid=torch.zeros(1, 768, dtype=torch.bool), extension_partition=torch.zeros(1, 768, dtype=torch.long),
        memory_points=torch.randn(1, 36, 5), memory_valid=torch.full((1, 36), not empty),
        memory_metadata=torch.zeros(1, 36, 8), memory_ids=torch.arange(36)[None],
        point_valid=torch.full((1, 4, count), not empty), box_size=torch.tensor([[4., 2., 2.]]))
    if not empty:
        batch["extension_valid"][:, :20] = True
    return observation, batch


def test_evidence_prevote_context_has_no_vote_head_bypass_and_identity_reads_memory():
    model = B2IdentityEvidence()
    observation, batch = _evidence_inputs()
    result = model(observation, batch)
    assert result.point_valid.sum() == 20
    assert result.mode_valid.any()
    result.point_features.square().sum().backward()
    assert model.vote_head[-1].weight.grad is None
    assert model.reliability_head[-1].weight.grad is None
    assert observation.point_features.grad.abs().sum() > 0
    model.zero_grad()
    result = model(observation, batch)
    result.identity_logits[:, :20].sum().backward()
    assert model.memory_attention.in_proj_weight.grad.abs().sum() > 0
    assert model.point_encoder[0].weight.grad.abs().sum() > 0


def test_empty_evidence_is_finite_and_does_not_create_fake_measurements():
    model = B2IdentityEvidence()
    observation, batch = _evidence_inputs(empty=True)
    result = model(observation, batch)
    assert not result.point_valid.any() and not result.mode_valid.any()
    assert (result.point_ids == -1).all() and (result.point_indices == -1).all()
    for value in vars(result).values():
        if value.is_floating_point():
            assert torch.isfinite(value).all()


def test_prior_feature_reaches_reliability_without_live_prior_geometry():
    model = B2IdentityEvidence()
    observation, batch = _evidence_inputs()
    feature = torch.randn(1, 128, requires_grad=True)
    geometry = torch.randn(1, 4, requires_grad=True)
    prior = SimpleNamespace(feature=feature, box=geometry)
    result = model(observation, batch, prior)
    result.reliability_logits[result.point_valid].sum().backward()
    assert feature.grad is not None and feature.grad.abs().sum() > 0
    assert geometry.grad is None
    assert model.vote_head[-1].weight.grad is None
    # GT 标注只能进入外部 loss，任何 GT 改变均不影响本 forward。
    model.eval()
    with torch.no_grad():
        before = model(observation, batch, prior)
        batch["target_box"] = torch.tensor([[1e6, -1e6, 50., 2.]])
        batch["extension_labels"] = torch.ones(1, 768)
        after = model(observation, batch, prior)
    assert torch.equal(before.vote_xyz, after.vote_xyz)
    assert torch.equal(before.identity_logits, after.identity_logits)


def test_raw_local_descriptor_is_permutation_invariant_and_cannot_move_raw_coordinates():
    geometry = RawLocalGeometry()
    raw = torch.tensor([[[0., 0., 0., .1, .2], [.1, 0., .1, .2, .4],
                         [10., 10., 10., 0., 0.]]], requires_grad=True)
    valid, ids = torch.ones(1, 3, dtype=torch.bool), torch.tensor([[12, 3, 8]])
    size = torch.tensor([[4., 2., 2.]])
    before = geometry(raw, valid, ids, size)
    permutation = torch.tensor([2, 0, 1])
    after = geometry(raw[:, permutation], valid[:, permutation], ids[:, permutation], size)
    assert torch.allclose(before[:, permutation], after, atol=1e-6, rtol=1e-6)
    assert torch.equal(before[:, 2], torch.zeros_like(before[:, 2]))
    before.sum().backward()
    assert raw.grad is None
    assert geometry.projection[0].weight.grad.abs().sum() > 0
