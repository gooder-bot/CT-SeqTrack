"""真实 host prepass → sampler → acquisition 标签的 v30 整合检查。"""
import copy
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.test_ct_v27_input_flow import sampler_runtime, _case  # noqa: F401
from tests.test_ct_v27_full_model import full_model_runtime  # noqa: F401
from utils.acquisition_v29 import support_membership
from utils.b1_acquisition import build_b0_crop_context
from utils.config import load_yaml_config
from utils.point_identity import raw_point_ids
from utils.recursive_state import RecursiveTrackState
from utils.v27_input import build_v27_eval_input
from models.ct_v2.motion import acquisition_margin_target_loss_v30


ROOT = Path(__file__).resolve().parents[1]


def _model_case(runtime, name='30_full_cfc_mini.yaml'):
    cfg = runtime[0][3](load_yaml_config(ROOT / 'cfgs/ct_seqtrack' / name))
    model = runtime[2](cfg).eval()
    sampler, _, sequence, state, payload, _, _ = _case(runtime[0], 'full')
    payload.update(is_initial_query=False, history_reference_reliable=False)
    cfg = copy.deepcopy(model.config)
    cfg.candidate_trajectory_mode = 'shared_se2'
    return model, sampler, cfg, sequence, state, payload


def _sample(model, sampler, cfg, sequence, state, payload):
    prediction = model.predict_motion_prepass(sequence, 8, state.results_bbs, recursive_state=state)
    payload['motion_prediction'] = prediction
    return sampler.motion_processing_mf(payload, cfg), prediction


@pytest.mark.parametrize('prepass_route', ['evaluation', 'online_batch'])
def test_real_host_and_sampler_crop_once_and_share_main_aux_current_context(full_model_runtime, monkeypatch, prepass_route):
    model, sampler, cfg, sequence, state, payload = _model_case(full_model_runtime)
    from datasets import points_utils
    import utils.b1_acquisition as acquisition
    original_crop = points_utils.generate_subwindow_with_aroundboxs
    original_arrays = acquisition.build_b1_input_arrays
    crop_calls, input_calls = [], []

    def crop(pc, sample_bb, ref_o, *args, **kwargs):
        result = original_crop(pc, sample_bb, ref_o, *args, **kwargs)
        if pc is sequence[8]['pc'] and kwargs.get('scale') == cfg.bb_scale and kwargs.get('offset') == cfg.bb_offset:
            crop_calls.append(result)
        return result

    def arrays(*args, **kwargs):
        result = original_arrays(*args, **kwargs)
        input_calls.append((kwargs['base_crop_context'], result))
        return result

    monkeypatch.setattr(points_utils, 'generate_subwindow_with_aroundboxs', crop)
    monkeypatch.setattr(acquisition, 'build_b1_input_arrays', arrays)
    if prepass_route == 'online_batch':
        prediction = model._online_motion_prepass_batch([(payload, state)])[0]
        payload['motion_prediction'] = prediction
        row = sampler.motion_processing_mf(payload, cfg)
    else:
        row, prediction = _sample(model, sampler, cfg, sequence, state, payload)
    assert len(crop_calls) == 1
    assert prediction['_b0_raw_crop'] is crop_calls[0]
    assert len(input_calls) == 3  # host prepass、sampler main、sampler auxiliary。
    assert all(context is prediction['_b0_crop_context'] for context, _ in input_calls)
    expected = build_b0_crop_context(len(np.unique(raw_point_ids(crop_calls[0]))),
        state.results_bbs[-1], scale=cfg.bb_scale, offset=cfg.bb_offset)
    np.testing.assert_array_equal(prediction['_b0_crop_context'], expected)
    for _, inputs in input_calls:
        assert inputs['acquisition_features'].shape == (21,)
        np.testing.assert_array_equal(inputs['acquisition_features'][-4:], row['motion_acquisition_features'][-4:])
    np.testing.assert_array_equal(row['motion_aux_acquisition_features'][-4:], row['motion_acquisition_features'][-4:])
    assert row['b0_raw_point_count'][-1] == len(raw_point_ids(crop_calls[0]))
    observed_ids = row['b0_point_ids'][-1][row['b0_point_unique_mask'][-1].astype(bool)]
    assert set(observed_ids).issubset(set(raw_point_ids(crop_calls[0])))


def test_first_query_anchor_without_transition_retains_real_minimum_band(full_model_runtime):
    model, _, _, sequence, _, _ = _model_case(full_model_runtime)
    anchor = sequence[0]['3d_bbox']
    local = np.asarray([[0., 0., 0.], [4.6, 0., 0.], [0., 3.4, 0.]])
    sequence[1]['pc'] = full_model_runtime[0][1].PointCloud(
        (local @ anchor.rotation_matrix.T + anchor.center).T)
    state = RecursiveTrackState(0, 'test/track', anchor, timestamps={0: sequence[0]['timestamp']})
    prediction = model.predict_motion_prepass(sequence, 1, state.results_bbs, recursive_state=state)
    assert not prediction['valid']
    batch, _ = build_v27_eval_input(model, sequence, 1, state.results_bbs,
                                    recursive_state=state, motion_prediction=prediction)
    torch.testing.assert_close(batch['ct_acquisition_margin'], torch.tensor([[.25, .25]]))
    torch.testing.assert_close(batch['ct_acquisition_support_half_size'], torch.tensor([[4.75, 3.5]]))
    assert batch['ct_search_history_valid'].item() == 1
    assert batch['ct_search_support_valid'].item() == 1
    assert batch['ct_acquisition_learned_valid'].item() == 0
    assert batch['ct_acquisition_resolved_valid'].item() == 1
    ids = batch['ct_extension_point_ids'][batch['ct_extension_valid_mask'].bool()].tolist()
    assert set(ids) == {1, 2}
    assert batch['b0_raw_point_count'][0, -1].item() == 1


@pytest.mark.parametrize('name', [
    '30_full_cfc_mini.yaml', '30_ablate_tight_band_mini.yaml',
    '30_ablate_legacy_acquisition_mini.yaml'])
def test_host_actual_maximum_and_grid_use_the_same_registered_geometry(full_model_runtime, monkeypatch, name):
    model, sampler, cfg, sequence, state, payload = _model_case(full_model_runtime, name)
    import utils.acquisition_v30 as acquisition
    original_resolve = sampler.resolve_joint_search_geometry
    original_maximum = acquisition.maximum_acquisition_supports_v30
    original_grid = acquisition.acquisition_margin_grid_target_v30
    calls = {}

    def resolve(*args, **kwargs):
        result = original_resolve(*args, **kwargs)
        calls['resolve'] = (copy.deepcopy(args), copy.deepcopy(kwargs), copy.deepcopy(result))
        return result

    def maximum(*args, **kwargs):
        result = original_maximum(*args, **kwargs)
        calls['maximum'] = (copy.deepcopy(args), copy.deepcopy(kwargs), copy.deepcopy(result))
        return result

    def grid(*args, **kwargs):
        result = original_grid(*args, **kwargs)
        calls['grid'] = (copy.deepcopy(args), copy.deepcopy(kwargs), copy.deepcopy(result))
        return result

    monkeypatch.setattr(sampler, 'resolve_joint_search_geometry', resolve)
    monkeypatch.setattr(acquisition, 'maximum_acquisition_supports_v30', maximum)
    monkeypatch.setattr(acquisition, 'acquisition_margin_grid_target_v30', grid)
    anchor = state.results_bbs[-1]
    sequence[8]['3d_bbox'].center = anchor.center + anchor.rotation_matrix @ np.asarray([6., 0., 0.])
    row, prediction = _sample(model, sampler, cfg, sequence, state, payload)
    args, kwargs, actual = calls['resolve']
    _, maximum_kwargs, maximum_result = calls['maximum']
    grid_args, grid_kwargs, label = calls['grid']
    np.testing.assert_allclose(row['ct_acquisition_margin'], prediction['acquisition_margin_parallel_perp'])
    np.testing.assert_allclose(maximum_kwargs['actual_margins'], row['ct_acquisition_margin'])
    np.testing.assert_allclose(grid_kwargs['actual_margins'], row['ct_acquisition_margin'])
    np.testing.assert_array_equal(maximum_kwargs['margin_max'], cfg.ct_acquisition_margin_max)
    np.testing.assert_array_equal(grid_kwargs['margin_min'], cfg.ct_acquisition_margin_min)
    np.testing.assert_array_equal(grid_kwargs['margin_max'], cfg.ct_acquisition_margin_max)
    for given, expected in zip((grid_kwargs['endpoint_box'], grid_kwargs['tube_box']), actual[:2]):
        np.testing.assert_allclose(given.center, expected.center)
        np.testing.assert_allclose(given.wlh, expected.wlh)
        np.testing.assert_allclose(given.rotation_matrix, expected.rotation_matrix)
    points, ids, target, base_ids = grid_args
    novel = ~np.isin(ids, base_ids)
    target = np.asarray(target, dtype=bool)
    table = []
    minimum, maximum_band = np.asarray(cfg.ct_acquisition_margin_min), np.asarray(cfg.ct_acquisition_margin_max)
    for i, x in enumerate(np.linspace(minimum[0], maximum_band[0], 9)):
        for j, y in enumerate(np.linspace(minimum[1], maximum_band[1], 9)):
            direct_kwargs = dict(kwargs, prediction=dict(kwargs['prediction'],
                acquisition_margin_parallel_perp=np.asarray([x, y])))
            direct = original_resolve(*args, **direct_kwargs)
            member = novel & (support_membership(points, direct[0]) | support_membership(points, direct[1])
                             | support_membership(points, grid_kwargs.get('corridor_box')))
            table.append((i, j, x, y, int((member & target).sum()), int((member & ~target).sum())))
            if i == j == 8:
                for observed, expected in zip(maximum_result[:2], direct[:2]):
                    np.testing.assert_allclose(observed.center, expected.center)
                    np.testing.assert_allclose(observed.wlh, expected.wlh)
                    np.testing.assert_array_equal(support_membership(points, observed), support_membership(points, expected))
    reachable = table[-1][4]
    assert reachable > 0 and label['valid'] and label['demand']
    feasible = [entry for entry in table if entry[4] >= math.ceil(.9 * reachable)]
    chosen = min(feasible, key=lambda entry: (entry[5], -entry[4],
        float(((np.asarray(entry[2:4]) - minimum) / (maximum_band - minimum)).sum()), entry[0], entry[1]))
    np.testing.assert_array_equal(label['grid_index'], chosen[:2])
    assert label['max_reachable_target_count'] == reachable
    assert label['selected_target_count'] == chosen[4]
    assert label['selected_background_count'] == chosen[5]


def test_real_sampler_unreachable_gt_is_excluded_from_demand_groups(full_model_runtime):
    model, sampler, cfg, sequence, state, payload = _model_case(full_model_runtime)
    anchor = state.results_bbs[-1]
    local = np.asarray([[0., 0., 0.], [5., 0., 0.], [30., 0., 0.]])
    sequence[8]['pc'] = full_model_runtime[0][1].PointCloud(
        (local @ anchor.rotation_matrix.T + anchor.center).T)
    rows = []
    for center in (0., 5., 30.):
        sequence[8]['3d_bbox'].center = anchor.center + anchor.rotation_matrix @ np.asarray([center, 0., 0.])
        row, _ = _sample(model, sampler, cfg, sequence, state, payload)
        rows.append(row)
    np.testing.assert_array_equal([row['motion_acquisition_target_valid'] for row in rows], [1, 1, 0])
    np.testing.assert_array_equal([row['motion_acquisition_demand'] for row in rows], [0, 1, 0])
    assert rows[-1]['motion_margin_global_novel_target_count'] == 1
    assert rows[-1]['motion_margin_max_reachable_target_count'] == 0
    prediction = torch.tensor([[.75, .5]] * 3, requires_grad=True)
    result = acquisition_margin_target_loss_v30(prediction,
        np.stack([row['motion_acquisition_target'] for row in rows]),
        [row['motion_acquisition_target_valid'] for row in rows],
        [row['motion_acquisition_demand'] for row in rows])
    result['loss'].backward()
    assert result['demand_count'] == result['no_demand_count'] == 1
    torch.testing.assert_close(prediction.grad[-1], torch.zeros(2))
    assert (prediction.grad[0] > 0).all()
