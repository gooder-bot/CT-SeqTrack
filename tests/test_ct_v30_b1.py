import copy
import ast
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from pyquaternion import Quaternion

from models.ct_v2.motion import OrderedPhysicalMotionEncoder, acquisition_margin_target_loss_v30
from utils.acquisition_v30 import (
    BAND_INITIAL, BAND_MAX, BAND_MIN, BAND_TIGHT_MAX, XY_CONTRACT,
    acquisition_margin_grid_target_v30, build_acquisition_supports_v30,
    maximum_acquisition_supports_v30, support_membership)
from utils.b1_acquisition import build_b0_crop_context, build_b1_input_arrays, b1_input_digest
from utils.ct_search import resolve_joint_search_geometry


class Box:
    def __init__(self, x=0., y=0., z=0., yaw=0., wlh=(2., 4., 2.)):
        self.center = np.asarray((x, y, z), dtype=np.float64)
        self.wlh = np.asarray(wlh, dtype=np.float64)
        self.orientation = Quaternion(axis=[0, 0, 1], radians=yaw)

    @property
    def rotation_matrix(self):
        return self.orientation.rotation_matrix


def supports(box, endpoint=None, margins=BAND_INITIAL, maximum=BAND_MAX):
    return build_acquisition_supports_v30(
        b0_crop_box=box, endpoint_center=box.center if endpoint is None else endpoint,
        object_wlh=box.wlh,
        object_yaw=float(box.orientation.radians * box.orientation.axis[-1]),
        band_margins=margins, band_margin_max=maximum)


@pytest.mark.parametrize('maximum,expected', [(BAND_MAX, (8.5, 6.25)), (BAND_TIGHT_MAX, (6.5, 4.75))])
def test_stationary_actual_b0_boundary_gets_external_band_and_hard_z(maximum, expected):
    box = Box()
    before = copy.deepcopy(box)
    endpoint, tube, diagnostic = supports(box, margins=maximum, maximum=maximum)
    np.testing.assert_allclose(tube.wlh[[1, 0]] * .5, expected)
    initial = supports(box)[1]
    # Initial band .75/.5 extends actual B0 half 4.5/3.25, even at zero motion.
    points = np.asarray([[5., 0., 0.], [0., 3.5, 2.], [0., 3.5, 3.25], [5.25, 0., 0.]])
    np.testing.assert_array_equal(support_membership(points, initial), [1, 1, 0, 0])
    assert diagnostic['support_xy_contract'] == XY_CONTRACT
    np.testing.assert_array_equal(box.center, before.center)
    np.testing.assert_array_equal(box.wlh, before.wlh)
    np.testing.assert_array_equal(box.rotation_matrix, before.rotation_matrix)


@pytest.mark.parametrize('yaw,travel', [(.7, (6., 2., 0.)), (math.pi / 2, (0., -7., 0.)), (0., (0., 8., 0.))])
def test_rotated_crop_and_size_project_correctly_without_lw_swap(yaw, travel):
    box = Box(11., -5., yaw=yaw, wlh=(3., 12., 4.))
    end = box.center + np.asarray(travel)
    endpoint, tube, diagnostic = supports(box, endpoint=end)
    # Every exact physical crop corner must be inside the positive-band tube.
    signs = np.asarray([[a, b, 0.] for a in (-1, 1) for b in (-1, 1)])
    half = .5 * box.wlh[[1, 0, 2]] * 1.25 + 2.
    crop_corners = box.center + (signs * half) @ box.rotation_matrix.T
    assert support_membership(crop_corners, tube).all()
    object_corners = end + (signs * .5 * box.wlh[[1, 0, 2]]) @ box.rotation_matrix.T
    assert support_membership(object_corners, endpoint).all()
    assert support_membership(object_corners, tube).all()
    np.testing.assert_allclose(diagnostic['acquisition_direction_world_xy'],
                               np.asarray(travel[:2]) / np.linalg.norm(travel[:2]), atol=1e-12)


def test_grid_label_matches_81_actual_crops_and_maximum_with_raw_id_exclusion():
    box = Box(yaw=.4)
    end = np.asarray((5., 1., 0.))
    points = np.random.default_rng(72).uniform([-8., -8., -4.], [12., 8., 4.], (1700, 3))
    ids = np.arange(len(points), dtype=np.int64)
    # Duplicate coordinates are distinct raw returns and both count.
    points[5] = points[4]
    truth = (points[:, 0] > 4.) & (points[:, 1] > 1.) & (points[:, 2] > 1.)
    base = ids[:35]
    ep, tb, _ = supports(box, end)
    kwargs = dict(endpoint_box=ep, tube_box=tb, actual_margins=BAND_INITIAL, b0_crop_box=box)
    state = np.random.get_state()
    result = acquisition_margin_grid_target_v30(points, ids, truth, base, **kwargs)
    after = np.random.get_state()
    assert state[0] == after[0] and state[2:] == after[2:]
    np.testing.assert_array_equal(state[1], after[1])
    table = []
    novel = ~np.isin(ids, base)
    for i, x in enumerate(np.linspace(BAND_MIN[0], BAND_MAX[0], 9)):
        for j, y in enumerate(np.linspace(BAND_MIN[1], BAND_MAX[1], 9)):
            a, b, _ = supports(box, end, margins=(x, y))
            membership = novel & (support_membership(points, a) | support_membership(points, b))
            table.append((i, j, x, y, int((membership & truth).sum()), int((membership & ~truth).sum())))
    reachable = table[-1][4]
    feasible = [row for row in table if row[4] >= math.ceil(.9 * reachable)]
    chosen = min(feasible, key=lambda row: (row[5], -row[4],
        (row[2] - .25) / 3.75 + (row[3] - .25) / 2.75, row[0], row[1]))
    np.testing.assert_array_equal(result['grid_index'], chosen[:2])
    assert result['demand'] and result['max_reachable_target_count'] == reachable
    assert result['selected_target_count'] == chosen[4]
    assert result['selected_background_count'] == chosen[5]
    maximum = maximum_acquisition_supports_v30(ep, tb, actual_margins=BAND_INITIAL, b0_crop_box=box)
    direct = supports(box, end, margins=BAND_MAX)
    for actual, expected in zip(maximum[:2], direct[:2]):
        np.testing.assert_allclose(actual.center, expected.center)
        np.testing.assert_allclose(actual.wlh, expected.wlh)
        np.testing.assert_array_equal(support_membership(points, actual), support_membership(points, expected))


def test_absent_unreachable_and_duplicate_ids_are_distinct_label_cases():
    box = Box()
    ep, tb, _ = supports(box)
    kwargs = dict(endpoint_box=ep, tube_box=tb, actual_margins=BAND_INITIAL, b0_crop_box=box)
    absent = acquisition_margin_grid_target_v30([[0., 0., 0.]], np.array([8]), [True], [8], **kwargs)
    assert absent['valid'] and not absent['demand'] and absent['reason'] == 'no_novel_target'
    outside = acquisition_margin_grid_target_v30([[8.5, 0., 0.]], np.array([8]), [True], [], **kwargs)
    assert not outside['valid'] and not outside['demand']
    assert outside['reason'] == 'outside_maximum_support'
    with pytest.raises(ValueError, match='unique raw'):
        acquisition_margin_grid_target_v30([[5., 0., 0.]] * 2, np.array([8, 8]), [True, True], [], **kwargs)


@pytest.mark.parametrize('learned', [True, False])
def test_dispatch_stationary_band_bypasses_legacy_minimum(learned):
    boxes = [Box()] * 3
    pred = dict(valid=True, mu_xy=np.zeros(2), current_delta_t=.5,
                acquisition_margin_parallel_perp=np.asarray(BAND_MIN))
    ep, tb, diagnostic = resolve_joint_search_geometry(
        boxes, [.5] * 3, [1] * 3, prediction=pred, use_b1_prepass=learned,
        use_acquisition_margin=True, fixed_margins=BAND_MIN,
        enable_v27=True, enable_v29=True, enable_v30=True, b0_crop_box=boxes[0])
    np.testing.assert_allclose(tb.wlh[[1, 0]] * .5, [4.75, 3.5])
    assert diagnostic['prior_source'] == ('b1' if learned else 'fallback_cv')
    assert ep is not None and diagnostic['valid']
    np.testing.assert_allclose(diagnostic['acquisition_margin_parallel_perp'], BAND_MIN)


def test_context_api_keeps_legacy17_and_appends_actual_current_crop_once():
    boxes = [Box(1.), Box(), Box(-1.)]
    context = build_b0_crop_context(17, boxes[0])
    np.testing.assert_allclose(context, [17, 0, 4.5, 3.25])
    old = build_b1_input_arrays(boxes, [.5] * 3, [1] * 3)
    new = build_b1_input_arrays(boxes, [.5] * 3, [1] * 3, enable_v30=True, base_crop_context=context)
    assert old['acquisition_features'].shape == (17,)
    assert new['acquisition_features'].shape == (21,)
    np.testing.assert_array_equal(new['acquisition_features'][:17], old['acquisition_features'])
    np.testing.assert_allclose(new['acquisition_features'][17:],
        [np.log(18) / np.log(1025), 0., np.log(4.5), np.log(3.25)])
    assert b1_input_digest(new) != b1_input_digest(old)
    empty = build_b0_crop_context(0, boxes[0])
    assert empty[1] == 1.
    with pytest.raises(ValueError, match='base_crop_context'):
        build_b1_input_arrays(boxes, [.5] * 3, [1] * 3, enable_v30=True)
    with pytest.raises(ValueError, match='non-negative integer'):
        build_b0_crop_context(1.2, boxes[0])


def test_initial_prediction_has_anchor_band_but_no_anchor_is_base_only():
    boxes = [Box()] * 3
    kwargs = dict(enable_v30=True, enable_v27=True, b0_crop_box=boxes[0])
    ep, tube, diagnostic = resolve_joint_search_geometry(boxes, [.5] * 3, [1, 0, 0], **kwargs)
    np.testing.assert_allclose(tube.wlh[[1, 0]] * .5, [4.75, 3.5])
    assert not diagnostic['motion_prior_valid']
    assert diagnostic['fallback_kind'] == 'anchor_no_transition'
    ep, tube, diagnostic = resolve_joint_search_geometry(boxes, [.5] * 3, [0, 0, 0], **kwargs)
    assert ep is None and tube is None and diagnostic['prior_source'] == 'base_only'


def test_motion_axis_is_not_rotated_by_a_different_actual_crop_center():
    history = [Box(), Box(-1.), Box(-2.)]
    prediction = dict(valid=True, mu_xy=np.asarray([2., 0.]),
                      acquisition_margin_parallel_perp=np.asarray(BAND_INITIAL))
    crop = Box(y=2.)
    _, _, diagnostic = resolve_joint_search_geometry(
        history, [.5] * 3, [1] * 3, prediction=prediction, use_b1_prepass=True,
        use_acquisition_margin=True, enable_v27=True, enable_v30=True, b0_crop_box=crop)
    np.testing.assert_allclose(diagnostic['acquisition_direction_world_xy'], [1., 0.], atol=1e-12)
    np.testing.assert_allclose(diagnostic['endpoint_center'], [2., 0., 0.])


def test_context_half_extents_match_actual_b0_crop_scalar_offset_semantics():
    # Load the actual CPU crop function without importing the nuScenes/Lightning
    # dataset package; no reimplementation of membership is used by this check.
    from utils.point_identity import raw_point_ids
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('v30_b1_test_data_classes', root / 'datasets/data_classes.py')
    classes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(classes)
    tree = ast.parse((root / 'datasets/points_utils.py').read_text(encoding='utf-8-sig'))
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'crop_pc_oriented')
    namespace = dict(copy=copy, np=np, Quaternion=Quaternion, PointCloud=classes.PointCloud, raw_point_ids=raw_point_ids)
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'datasets/points_utils.py', 'exec'), namespace)
    RealBox, PointCloud = classes.Box, classes.PointCloud
    crop_pc_oriented = namespace['crop_pc_oriented']
    box = RealBox(center=[10., -3., 1.], size=[2., 4., 2.],
                  orientation=Quaternion(axis=[0, 0, 1], radians=.73))
    # B0 scales dimensions first, then adds offset to each face (not full size).
    local = np.asarray([[4.49, 0., 0.], [4.51, 0., 0.], [0., 3.24, 0.], [0., 3.26, 0.],
                        [0., 0., 3.24], [0., 0., 3.26]])
    world = local @ box.rotation_matrix.T + box.center
    raw = PointCloud(world.T, point_ids=np.arange(len(world), dtype=np.int64))
    crop, mask = crop_pc_oriented(raw, box, scale=1.25, offset=2., return_mask=True)
    np.testing.assert_array_equal(mask, [1, 0, 1, 0, 1, 0])
    context = build_b0_crop_context(crop.nbr_points(), box)
    np.testing.assert_allclose(context, [3, 0, 4.5, 3.25])


@pytest.mark.parametrize('backend', ['gru', 'cfc'])
def test_v30_head_exact_initial_band_and_isolated_acquisition_gradients(backend):
    module = OrderedPhysicalMotionEncoder(enable_v30=True, adaptive_acquisition_margin=True,
                                         temporal_backend=backend)
    context = torch.randn(2, 128, requires_grad=True)
    quality = torch.randn(2, 21, requires_grad=True)
    optimizer = torch.optim.SGD(module.parameters(), lr=.1)
    initial = module._acquisition_margin(context, quality)
    torch.testing.assert_close(initial, torch.tensor([BAND_INITIAL] * 2))
    for _ in range(2):
        optimizer.zero_grad()
        prediction = module._acquisition_margin(context, quality)
        loss = acquisition_margin_target_loss_v30(prediction, [[3., 2.], [.25, .25]], [1, 1], [1, 0])['loss']
        loss.backward()
        assert context.grad is None and quality.grad is None
        assert module.velocity_residual_head.weight.grad is None
        assert module.log_sigma_head.weight.grad is None
        optimizer.step()
    assert module.acquisition_margin_head[0].weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match='21'):
        module._acquisition_margin(context, torch.zeros(2, 17))


def test_demand_balancing_resists_negative_replication_and_preserves_missing_cases():
    prediction = torch.tensor([[.75, .5], [.75, .5]], requires_grad=True)
    target = torch.tensor([[3., 2.], [.25, .25]], requires_grad=True)
    original = acquisition_margin_target_loss_v30(prediction, target, [1, 1], [1, 0])
    expanded_pred = torch.cat((prediction[:1], prediction[1:].expand(91, -1)))
    expanded_target = torch.cat((target[:1], target[1:].expand(91, -1)))
    expanded = acquisition_margin_target_loss_v30(expanded_pred, expanded_target, [1] * 92, [1] + [0] * 91)
    torch.testing.assert_close(original['loss'], expanded['loss'])
    original['loss'].backward()
    assert target.grad is None
    assert (prediction.grad[0] < 0).all() and (prediction.grad[1] > 0).all()
    empty_pred = torch.zeros(2, 2, requires_grad=True)
    empty = acquisition_margin_target_loss_v30(empty_pred, torch.ones(2, 2), [0, 0], [1, 0])
    empty['loss'].backward()
    assert empty['loss'] == 0 and torch.equal(empty_pred.grad, torch.zeros(2, 2))
    one = acquisition_margin_target_loss_v30(prediction[:1], target[:1], [1], [1])
    torch.testing.assert_close(one['loss'], one['demand_loss'])


def test_age_balance_operates_inside_each_demand_group_and_masks_invalid_age():
    prediction = torch.zeros(4, 2, requires_grad=True)
    target = torch.tensor([[1., 1.], [3., 3.], [9., 9.], [2., 2.]])
    out = acquisition_margin_target_loss_v30(prediction, target, [1] * 4, [1, 1, 1, 0],
        recursive_age=[0, 0, 3, 0], recursive_age_valid=[1] * 4)
    torch.testing.assert_close(out['demand_loss'], torch.tensor(.9 * ((1 + 3) / 2 + 9) / 2))
    torch.testing.assert_close(out['loss'], torch.tensor((.9 * 5.5 + .9 * 2) / 2))
    masked = acquisition_margin_target_loss_v30(prediction, target, [1] * 4, [1, 1, 1, 0], recursive_age=[0] * 4)
    assert masked['loss'] == 0 and masked['valid'].sum() == 0
