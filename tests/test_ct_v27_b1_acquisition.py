import math

import numpy as np
import pytest
import torch
from pyquaternion import Quaternion

from models.ct_v2.motion import (
    OrderedPhysicalMotionEncoder, acquisition_margin_target_loss)
from models.ct_v2.pipeline_contracts import B1Input
from utils.b1_acquisition import (
    acquisition_margin_grid_target, b1_input_digest, build_b1_input_arrays,
    projected_box_half_extents)
from utils.ct_search import (
    bounded_novel_support_pool, build_causal_history_corridor,
    diagnostic_points_in_oriented_support, resolve_joint_search_geometry,
    sample_bounded_novel_prepool)
from utils.v27_diagnostics import build_acquisition_diagnostics


class Box:
    def __init__(self, x=0., y=0., yaw=0., wlh=(2., 4., 2.)):
        self.center = np.asarray((x, y, 0.))
        self.wlh = np.asarray(wlh)
        self.orientation = Quaternion(axis=[0, 0, 1], radians=yaw)

    @property
    def rotation_matrix(self):
        return self.orientation.rotation_matrix


def _prediction(mu, margins=(2., 1.)):
    return dict(mu_xy=np.asarray(mu), velocity_xy=np.asarray(mu) * 2.,
                direction_xy=np.asarray((1., 0.)), valid=True, source_id=1,
                acquisition_margin_parallel_perp=np.asarray(margins),
                current_delta_t=.5, gap_ratio=1.)


def test_raw_ids_keep_coincident_returns_and_exclude_float_drift():
    base = np.asarray([[0., 0., 0.]], dtype=np.float32)
    endpoint = np.asarray([[1e-5, 0., 0.], [1., 0., 0.], [1., 0., 0.]])
    tube = np.asarray([[1.00001, 0., 0.]])
    empty = np.zeros((0, 3))
    kwargs = dict(baseline_ids=np.asarray([10]), endpoint_ids=np.asarray([10, 11, 12]),
                  tube_ids=np.asarray([11]), corridor_ids=np.asarray([], dtype=np.int64),
                  enable_v27=True)
    points, source, ids = bounded_novel_support_pool(
        base, endpoint, tube, empty, return_ids=True, **kwargs)
    np.testing.assert_array_equal(ids, [11, 12])
    np.testing.assert_array_equal(source, [3, 1])
    assert len(points) == 2  # Coincident XYZ does not imply identical LiDAR return.
    _, mask, bits, audit = sample_bounded_novel_prepool(
        base, endpoint, tube, empty, local_quota=2, corridor_quota=1, **kwargs)
    assert mask.sum() == 2
    np.testing.assert_array_equal(audit['_selected_point_ids'], [11, 12, -1])
    np.testing.assert_array_equal(bits, [3, 1, 0])
    with pytest.raises(ValueError, match='original point IDs'):
        sample_bounded_novel_prepool(base, endpoint, tube, empty, enable_v27=True)


def test_projected_large_target_and_actual_axis_override_old_basis():
    box = Box(wlh=(3., 18., 4.))
    endpoint, tube, audit = resolve_joint_search_geometry(
        [box, Box(-1., wlh=box.wlh)], [.5, .5], [1, 1],
        prediction=_prediction((0., 8.), (6., 3.)), use_b1_prepass=True,
        use_acquisition_margin=True, enable_v27=True, first_frame_size=box.wlh)
    np.testing.assert_allclose(audit['acquisition_direction_world_xy'], [0., 1.], atol=1e-12)
    # Motion is perpendicular to the 18 m target: its length projects to width.
    np.testing.assert_allclose(endpoint.wlh, [24., 15., 4.])
    np.testing.assert_allclose(tube.wlh, [24., 23., 4.])
    np.testing.assert_allclose(tube.center, [0., 4., 0.])
    assert not audit['truncated']


def test_large_corridor_limits_motion_segment_not_object_size():
    boxes = [Box(2., wlh=(3., 18., 4.)), Box(1., wlh=(3., 18., 4.)),
             Box(0., wlh=(3., 18., 4.))]
    corridor, audit = build_causal_history_corridor(
        boxes, [.5, .5, .5], [1, 1, 1], enabled=True, enable_v27=True,
        first_frame_size=boxes[0].wlh)
    assert corridor is not None
    np.testing.assert_allclose(corridor.wlh, [5., 19., 4.])
    assert audit['valid']


def test_grid_label_matches_actual_supports_and_does_not_advance_rng():
    points = np.random.default_rng(9).uniform([-12., -9., -.9], [16., 9., .9], (2000, 3))
    ids = np.arange(len(points), dtype=np.int64)
    targets = ((points[:, 0] > 5.) & (points[:, 0] < 10.)
               & (points[:, 1] > 2.) & (points[:, 1] < 5.))
    base_ids = ids[:37]
    box = Box(yaw=.4, wlh=(2., 5., 2.))
    endpoint_world = np.asarray((6., 1., 0.))
    mu_local = (box.rotation_matrix.T @ endpoint_world)[:2]
    before = np.random.get_state()
    actual = acquisition_margin_grid_target(
        points, ids, targets, base_ids, anchor_center=box.center,
        endpoint_center=endpoint_world, object_wlh=box.wlh, object_yaw=.4)
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    novel = ~np.isin(ids, base_ids)
    rows = []
    for i, par in enumerate(np.linspace(2., 6., 9)):
        for j, perp in enumerate(np.linspace(1., 3., 9)):
            endpoint, tube, _ = resolve_joint_search_geometry(
                [box, Box(-1.)], [.5, .5], [1, 1],
                prediction=_prediction(mu_local, (par, perp)), use_b1_prepass=True,
                use_acquisition_margin=True, enable_v27=True, first_frame_size=box.wlh)
            member = novel & (diagnostic_points_in_oriented_support(points, endpoint)
                              | diagnostic_points_in_oriented_support(points, tube))
            fg, bg = int(np.sum(member & targets)), int(np.sum(member & ~targets))
            rows.append((i, j, par, perp, fg, bg))
    needed = math.ceil(.9 * rows[-1][4])
    best = min((r for r in rows if r[4] >= needed),
               key=lambda r: (r[5], -r[4], (r[2] - 2.) / 4. + (r[3] - 1.) / 2., r[0], r[1]))
    np.testing.assert_array_equal(actual['grid_index'], best[:2])
    assert actual['selected_target_count'] == best[4]
    assert actual['selected_background_count'] == best[5]


def test_grid_empty_unreachable_and_strict_boundary():
    kwargs = dict(anchor_center=[0., 0., 0.], endpoint_center=[0., 0., 0.],
                  object_wlh=[2., 4., 2.], object_yaw=0.)
    # Maximum x extent is 2 + 6 = 8; the exact face is excluded.
    out = acquisition_margin_grid_target([[8., 0., 0.]], np.asarray([0]),
                                         [True], [], **kwargs)
    assert not out['valid'] and out['reason'] == 'outside_maximum_support'
    out = acquisition_margin_grid_target([[0., 0., 0.]], np.asarray([0]),
                                         [True], [0], **kwargs)
    assert out['valid'] and out['reason'] == 'no_novel_target'
    np.testing.assert_array_equal(out['target_margin'], [2., 1.])


def test_shared_backend_initialization_and_margin_only_gradients():
    modules = [OrderedPhysicalMotionEncoder(
        temporal_backend=backend, shared_kinematic_anchor=True,
        adaptive_acquisition_margin=True, enable_v27=True, initialization_seed=42)
        for backend in ('gru', 'cfc')]
    shared = modules[0].state_dict()
    other = modules[1].state_dict()
    for key in shared.keys() & other.keys():
        assert torch.equal(shared[key], other[key]), key
    module = modules[0]
    boxes = torch.tensor([[[0., 0., 0., 0.], [-1., 0., 0., 0.], [-2., 0., 0., 0.]]])
    gaps = torch.full((1, 3), .5)
    quality = torch.ones(1, 17, requires_grad=True)
    contract = B1Input(boxes, gaps, torch.ones(1, 3), torch.tensor([.5]), quality)
    optimizer = torch.optim.SGD(module.parameters(), lr=.1)
    for _ in range(2):
        optimizer.zero_grad()
        output = module(**contract.encoder_kwargs())
        loss = acquisition_margin_target_loss(output['acquisition_margin_parallel_perp'],
                                              [[4., 2.]], [1.])['loss_per_sample'].mean()
        loss.backward()
        assert module.velocity_residual_head.weight.grad is None
        assert module.context[0].weight.grad is None
        assert quality.grad is None
        optimizer.step()
    assert module.acquisition_margin_head[0].weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match='acquisition_features'):
        module(boxes, gaps, torch.ones(1, 3), torch.tensor([.5]))


def test_causal_input_quality_masks_and_effective_clock_digest():
    boxes = [Box(3., yaw=.5), Box(2., yaw=.4), Box(1., yaw=.3)]
    quality = np.asarray([[128., .6, .2, 1.], [64., .4, .3, 1.], [999., .9, .1, 1.]])
    a = build_b1_input_arrays(boxes, [.5, .5, 1.], [1, 1, 0], history_quality=quality)
    b = build_b1_input_arrays(boxes, [.5, .5, .5], [1, 1, 0], history_quality=quality)
    assert a['acquisition_features'].shape == (17,)
    np.testing.assert_allclose(a['ref_boxs'][0], [0., 0., 0., 0.], atol=1e-7)
    np.testing.assert_array_equal(a['acquisition_features'][8:12], 0.)
    assert b1_input_digest(a) != b1_input_digest(b)
    half = projected_box_half_extents([2., 10., 2.], 0., math.pi / 2.)
    np.testing.assert_allclose(half, [1., 5., 1.], atol=1e-12)


def test_v27_diagnostics_join_global_labels_by_id_and_preserve_zero_denominators():
    world = np.asarray([[0., 0., 0.], [1., 0., 0.], [1., 0., 0.], [8., 0., 0.]])
    global_table = (world, np.asarray([10, 11, 12, 13]))
    baseline = (world[:1], np.asarray([10]))
    endpoint = (world[[1, 3]], np.asarray([11, 13]))
    tube = (world[[1, 2]], np.asarray([11, 12]))
    # Source 3 marks actual endpoint/tube overlap despite changed local XYZ.
    output = build_acquisition_diagnostics(
        global_table, Box(), Box(), baseline,
        np.zeros((3, 3)), np.asarray([10, 10, -1]), endpoint, tube, None,
        np.zeros((3, 3)), np.asarray([11, 12, 13]),
        np.ones((3, 3)) * 99., np.asarray([11, 12, -1]), [1, 1, 0], [3, 2, 0])
    assert output['acquisition_schema_version'] == 'ct_acquisition.v4'
    assert output['global_target_count_exact'] == 3
    assert output['global_novel_target_count'] == 2
    assert output['base_sampled_target_count'] == 1
    assert output['base_sampled_slot_count'] == 2
    assert output['pool_target_count'] == 2
    assert output['prepool_target_count'] == 2
    assert output['prepool_target_retention'] == 1.
    assert output['pool_background_count'] == 1
    empty = (np.zeros((0, 3)), np.zeros(0, dtype=np.int64))
    zero = build_acquisition_diagnostics(
        empty, Box(), Box(), empty, *empty, empty, empty, empty,
        *empty, *empty, np.zeros(0), np.zeros(0, dtype=np.int64))
    assert zero['global_target_denominator_valid'] == 0
    assert zero['prepool_target_recall'] == 0.
    assert zero['prepool_target_retention_denominator_valid'] == 0
