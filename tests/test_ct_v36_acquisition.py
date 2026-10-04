"""prior优先、重叠域与范围投影的确定性获取合同。"""
import numpy as np
import torch

from models.ct_v36.acquisition import build_dual_support, acquire_extension, support_membership
from models.ct_v31.evidence import select_evidence_points


def geometry(**kwargs):
    values = dict(b0_box=[0., 0., 0., 0.], box_size=[4., 2., 2.],
                  prior_box=[0., 0., 0., 0.], range_xy=[3., 3.], axes=np.eye(2))
    values.update(kwargs)
    return values


def test_overlap_remaining_points_are_available_to_second_domain():
    cloud = np.random.default_rng(36).uniform(-1., 1., (900, 3))
    result = acquire_extension(cloud, np.arange(900), b0_raw_ids=[], anchor=[0., 0., 0.], **geometry())
    assert result['extension_valid'].sum() == 768
    assert (result['extension_partition'][:384] == 0).all()
    assert (result['extension_partition'][384:] == 1).all()
    assert result['extension_membership'].all()
    assert np.unique(result['extension_ids']).size == 768


def test_prior_gets_first_384_and_deficient_local_borrows_with_actual_source():
    cloud = np.random.default_rng(4).uniform(-.5, .5, (900, 3)) + [30., 0., 10.]
    geom = geometry(prior_box=[30., 0., 10., 0.])
    result = acquire_extension(cloud, np.arange(900), b0_raw_ids=[], anchor=[0., 0., 0.], **geom)
    assert result['extension_valid'].sum() == 768
    assert (result['extension_partition'] == 0).all()
    assert result['extension_membership'][:, 0].all()
    assert not result['extension_membership'][:, 1].any()
    selected = select_evidence_points(torch.tensor(result['extension_points'])[None],
        torch.zeros(1, 768), torch.tensor(result['extension_valid'])[None],
        torch.tensor(result['extension_ids'])[None], torch.tensor(result['extension_partition'])[None])
    assert (selected >= 0).all() and selected.unique().numel() == 256


def test_empty_prior_borrows_local_pool_without_relabeling():
    cloud = np.random.default_rng(9).uniform(-1., 1., (900, 3))
    result = acquire_extension(cloud, np.arange(900), b0_raw_ids=[], anchor=[0., 0., 0.],
        **geometry(prior_box=[50., 0., 10., 0.]))
    assert result['extension_valid'].sum() == 768
    assert (result['extension_partition'] == 1).all()
    assert not result['extension_membership'][:, 0].any()
    assert result['extension_membership'][:, 1].all()


def test_raw_crop_exclusion_sparse_padding_and_input_order_invariance():
    cloud = np.array([[0., 0., 0.], [1., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
    ids = np.array([10, 11, 11, 12])
    a = acquire_extension(cloud, ids, b0_raw_ids=[10], anchor=[0., 0., 0.], **geometry())
    b = acquire_extension(cloud[::-1], ids[::-1], b0_raw_ids=[10], anchor=[0., 0., 0.], **geometry())
    assert a['extension_valid'].sum() == 2
    assert set(a['extension_ids'][:2]) == {11, 12}
    assert (a['extension_ids'][2:] == -1).all()
    np.testing.assert_array_equal(a['extension_ids'], b['extension_ids'])


def test_size_projection_range_caps_and_separate_vertical_centers():
    axes = np.array([[0., -1.], [1., 0.]])
    support = build_dual_support(**geometry(prior_box=[30., 0., 10., 0.], axes=axes, range_xy=[20., 20.]))
    np.testing.assert_allclose(support['prior']['half'], [1. + 12., 2. + 8., 3.25])
    np.testing.assert_allclose(support['local']['half'], [8.5, 6.25, 3.25])
    np.testing.assert_array_equal(support['range_xy'], [20., 20.])
    np.testing.assert_array_equal(support['range_clipped'], [True, True])
    points = np.array([[30., 0., 10.], [15., 0., 5.], [30., 0., 0.]])
    p, q = support_membership(points, support, [], np.arange(3))
    assert p.tolist() == [True, False, False] and not q.any()


def test_world_translation_occurs_before_float32_conversion():
    anchor = np.array([1e8, -1e8, 100.])
    offset = np.array([.125, -.25, .0625])
    result = acquire_extension((anchor + offset)[None], [1], b0_raw_ids=[], anchor=anchor,
        **geometry(b0_box=np.r_[anchor, 0.], prior_box=np.r_[anchor, 0.]))
    np.testing.assert_array_equal(result['extension_points'][0, :3], offset.astype(np.float32))
