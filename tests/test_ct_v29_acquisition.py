import copy
import math

import numpy as np
import pytest
from pyquaternion import Quaternion

from utils.acquisition_v29 import (
    acquisition_margin_grid_target_v29, apply_b0_vertical_hull,
    b0_vertical_interval, maximum_acquisition_supports_v29,
    support_membership, support_vertical_interval)
from utils.ct_search import build_causal_history_corridor, resolve_joint_search_geometry
from utils.v27_diagnostics import build_acquisition_diagnostics


class Box:
    def __init__(self, x=0., y=0., z=0., yaw=0., wlh=(2., 4., 2.)):
        self.center = np.asarray((x, y, z), dtype=np.float64)
        self.wlh = np.asarray(wlh, dtype=np.float64)
        self.orientation = Quaternion(axis=[0, 0, 1], radians=yaw)

    @property
    def rotation_matrix(self):
        return self.orientation.rotation_matrix


def geometry(boxes, *, margins=(2., 1.), learned=True, v29=True):
    mu = np.asarray((2., 0.))
    prediction = dict(mu_xy=mu, velocity_xy=2. * mu,
                      direction_xy=np.asarray((1., 0.)), valid=True, source_id=1,
                      acquisition_margin_parallel_perp=np.asarray(margins),
                      current_delta_t=.5, gap_ratio=1.)
    return resolve_joint_search_geometry(
        boxes, [.5, .5, .5], [1, 1, 1], prediction=prediction,
        use_b1_prepass=learned, use_acquisition_margin=True,
        fixed_margins=margins, enable_v27=True, first_frame_size=boxes[0].wlh,
        enable_v29=v29, b0_crop_box=boxes[0])


@pytest.mark.parametrize('yaw', [0., .7, math.pi / 2])
@pytest.mark.parametrize('learned', [False, True])
def test_v29_real_learned_cv_geometry_changes_only_z_and_preserves_inputs(yaw, learned):
    boxes = [Box(0., yaw=yaw), Box(-1., yaw=yaw), Box(-2., yaw=yaw)]
    before = copy.deepcopy(boxes)
    old = geometry(boxes, learned=learned, v29=False)
    new = geometry(boxes, learned=learned)
    for legacy, actual in zip(old[:2], new[:2]):
        np.testing.assert_array_equal(actual.center[:2], legacy.center[:2])
        np.testing.assert_array_equal(actual.wlh[:2], legacy.wlh[:2])
        np.testing.assert_array_equal(actual.rotation_matrix, legacy.rotation_matrix)
        np.testing.assert_allclose(support_vertical_interval(actual), [-3.25, 3.25])
    for source, saved in zip(boxes, before):
        np.testing.assert_array_equal(source.center, saved.center)
        np.testing.assert_array_equal(source.wlh, saved.wlh)
    assert new[2]['support_z_contract'] == 'b0_vertical_hull_v1'
    assert 'support_z_contract' not in old[2]
    # v27 最终 CV 返回值已有正确的一次 margin；不能将中间旧代码误称实际 double margin。
    np.testing.assert_allclose(old[0].wlh[:2],
        [old[2]['base_projected_width'] + 2., old[2]['base_projected_length'] + 4.])


def test_v29_corridor_shares_hull_and_legacy_stationary_behavior():
    boxes = [Box(2.), Box(1.), Box()]
    kwargs = dict(enabled=True, enable_v27=True, first_frame_size=boxes[0].wlh)
    old, _ = build_causal_history_corridor(boxes, [.5] * 3, [1] * 3, **kwargs)
    new, diagnostic = build_causal_history_corridor(boxes, [.5] * 3, [1] * 3,
        **kwargs, enable_v29=True, b0_crop_box=boxes[0])
    np.testing.assert_array_equal(old.center[:2], new.center[:2])
    np.testing.assert_array_equal(old.wlh[:2], new.wlh[:2])
    np.testing.assert_allclose(diagnostic['support_z_interval'], [-3.25, 3.25])
    empty, diagnostic = build_causal_history_corridor([Box()] * 3, [.5] * 3, [1] * 3,
        **kwargs, enable_v29=True, b0_crop_box=boxes[0])
    assert empty is None and diagnostic['reason'] == 'stationary'


def test_hull_preserves_old_vertical_extent_and_recovers_crop_height_only():
    baseline = Box(z=1.)
    support = Box(x=5., z=-4.)
    actual = apply_b0_vertical_hull(support, baseline)
    np.testing.assert_allclose(support_vertical_interval(actual), [-5., 4.25])
    np.testing.assert_array_equal(support.center, [5., 0., -4.])
    old, new = geometry([Box(), Box(-1.), Box(-2.)], v29=False), geometry([Box(), Box(-1.), Box(-2.)])
    points = np.asarray([[5., 0., 2.], [5., 0., 3.25], [20., 0., 2.]])
    np.testing.assert_array_equal(support_membership(points, old[0]), [False, False, False])
    np.testing.assert_array_equal(support_membership(points, new[0]), [True, False, False])
    with pytest.raises(ValueError, match='actual B0 crop anchor'):
        b0_vertical_interval(None)


@pytest.mark.parametrize('learned', [False, True])
def test_grid_matches_all_81_actual_supports_with_same_z_and_gt_free_maximum(learned):
    rng = np.random.default_rng(891)
    points = rng.uniform([-10., -8., -4.], [15., 8., 4.], (1100, 3))
    ids = np.arange(len(points), dtype=np.int64)
    labels = (points[:, 0] > 3.) & (points[:, 1] > 1.) & (points[:, 2] > 1.5)
    baseline_ids = ids[:43]
    boxes = [Box(yaw=.4), Box(-1., yaw=.4), Box(-2., yaw=.4)]
    margins = np.asarray((3.5, 2.))
    endpoint, tube, _ = geometry(boxes, margins=margins, learned=learned)
    kwargs = dict(endpoint_box=endpoint, tube_box=tube, actual_margins=margins,
                  b0_crop_box=boxes[0])
    before = np.random.get_state()
    result = acquisition_margin_grid_target_v29(points, ids, labels, baseline_ids, **kwargs)
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
    rows = []
    for i, par in enumerate(np.linspace(2., 6., 9)):
        for j, perp in enumerate(np.linspace(1., 3., 9)):
            ep, tb, _ = geometry(boxes, margins=(par, perp), learned=learned)
            member = (~np.isin(ids, baseline_ids)
                      & (support_membership(points, ep) | support_membership(points, tb)))
            rows.append((i, j, par, perp, int(np.sum(member & labels)), int(np.sum(member & ~labels))))
    required = math.ceil(.9 * rows[-1][4])
    chosen = min((r for r in rows if r[4] >= required),
                 key=lambda r: (r[5], -r[4], (r[2] - 2.) / 4. + (r[3] - 1.) / 2., r[0], r[1]))
    np.testing.assert_array_equal(result['grid_index'], chosen[:2])
    assert result['selected_target_count'] == chosen[4]
    assert result['selected_background_count'] == chosen[5]
    maximum = maximum_acquisition_supports_v29(endpoint, tube, None,
        actual_margins=margins, b0_crop_box=boxes[0])
    opposite = acquisition_margin_grid_target_v29(points, ids, ~labels, baseline_ids, **kwargs)
    assert opposite['global_novel_target_count'] != result['global_novel_target_count']
    repeated = maximum_acquisition_supports_v29(endpoint, tube, None,
        actual_margins=margins, b0_crop_box=boxes[0])
    for a, b in zip(maximum[:2], repeated[:2]):
        np.testing.assert_array_equal(a.center, b.center)
        np.testing.assert_array_equal(a.wlh, b.wlh)


def test_unreachable_and_no_geometry_are_missing_supervision_not_fake_points():
    boxes = [Box(), Box(-1.), Box(-2.)]
    endpoint, tube, _ = geometry(boxes)
    kwargs = dict(endpoint_box=endpoint, tube_box=tube, actual_margins=(2., 1.), b0_crop_box=boxes[0])
    out = acquisition_margin_grid_target_v29([[100., 0., 0.]], np.asarray([0]), [True], [], **kwargs)
    assert not out['valid'] and out['reason'] == 'outside_maximum_support'
    assert out['global_novel_target_count'] == 1 and out['max_reachable_target_count'] == 0
    out = acquisition_margin_grid_target_v29([[0., 0., 0.]], np.asarray([0]), [True], [0], **kwargs)
    assert out['valid'] and out['reason'] == 'no_novel_target'
    out = acquisition_margin_grid_target_v29([[0., 0., 0.]], np.asarray([0]), [True], [],
        **{**kwargs, 'endpoint_box': None, 'tube_box': None})
    assert not out['valid'] and out['reason'] == 'no_structural_geometry'


def test_novel_diagnostics_separate_z_exclusion_reachability_and_sample_retention():
    # ID10 是 B0 raw crop 未采样点，仍必须排除；ID14 在最大合法 XY 外。
    xyz = np.asarray([[0., 0., 2.], [1., 0., 2.], [2., 0., 3.5], [3., 0., 0.], [20., 0., 0.]])
    ids = np.arange(10, 15, dtype=np.int64)
    empty = (np.zeros((0, 3)), np.zeros(0, dtype=np.int64))
    baseline = (xyz[:1], ids[:1])
    ep = Box(x=1., wlh=(2., 6., 6.5))
    maximum = Box(x=1., wlh=(6., 16., 6.5))
    branch = (xyz[[0, 1, 3]], ids[[0, 1, 3]])
    kwargs = dict(global_pc=(xyz, ids), gt_box=Box(x=10., wlh=(8., 24., 10.)),
        anchor_box=Box(), baseline_pc=baseline, sampled_base_xyz=empty[0], sampled_base_ids=empty[1],
        endpoint_pc=branch, tube_pc=None, corridor_pc=None, pool_xyz=xyz[[1, 3]], pool_ids=ids[[1, 3]],
        prepool_xyz=xyz[[1]], prepool_ids=ids[[1]], prepool_valid=[1], source=[1], support_boxes=(ep,))
    out = build_acquisition_diagnostics(**kwargs, enable_v29=True, max_support_boxes=(maximum,))
    assert out['global_novel_target_count'] == 4
    assert out['support_novel_xy_target_count'] == 3
    assert out['support_novel_xyz_target_count'] == 2
    assert out['support_novel_z_excluded_target_count'] == 1
    assert out['max_legal_novel_target_count'] == 2
    assert out['max_legal_unreachable_target_count'] == 2
    assert out['support_novel_recall_of_reachable'] == 1.
    assert out['prepool_recall_of_reachable'] == .5
    assert out['prepool_novel_target_retention'] == .5
    assert out['actual_support_novel_outside_maximum_count'] == 0
    legacy = build_acquisition_diagnostics(**kwargs)
    assert legacy['acquisition_schema_version'] == 'ct_acquisition.v4'
    assert 'max_legal_novel_target_count' not in legacy
    with pytest.raises(ValueError, match='explicit maximum legal supports'):
        build_acquisition_diagnostics(**kwargs, enable_v29=True)
