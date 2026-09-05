"""v27真实采样函数的稀疏输入、GT边界与预加载共享对象复核。"""
import ast
import copy
import os
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from tests.test_ct_v27_input_flow import sampler_runtime, _case
from utils.config import load_yaml_config
from utils.v27_input import build_v27_eval_input


ROOT = Path(__file__).resolve().parents[1]


def _source_definition(path, name):
    tree = ast.parse((ROOT / path).read_text(encoding='utf-8-sig'))
    for item in tree.body:
        if isinstance(item, ast.ClassDef):
            if item.name == name:
                return item
            for child in item.body:
                if isinstance(child, ast.FunctionDef) and child.name == name:
                    return child
    raise KeyError(name)


def _executable(path, name):
    namespace = dict(globals())
    definition = copy.deepcopy(_source_definition(path, name))
    exec(compile(ast.Module(body=[definition], type_ignores=[]), path, 'exec'), namespace)
    return namespace[name]


@pytest.mark.parametrize('reference', [False, True])
@pytest.mark.parametrize('count', [0, 1, 2])
def test_train_observation_and_reference_keep_sparse_rows_without_gt_history_retry(sampler_runtime, reference, count):
    sampler, config, sequence, state, payload, host, prediction = _case(sampler_runtime, 'b0')
    _, classes, _, EasyDict = sampler_runtime
    if reference:
        config = EasyDict(load_yaml_config(ROOT / 'cfgs/27_seqtrack_reference.yaml'))
        host.config = config
    config.candidate_trajectory_mode = 'shared_se2'
    config.ct_observation_payload_mode = 'legacy'
    for index, frame in enumerate(sequence):
        frame['pc'] = classes.PointCloud(np.repeat(frame['3d_bbox'].center[:, None], count, axis=1),
                                        point_ids=np.arange(count, dtype=np.int64) + 500)
    payload.pop('online_recursive_state')
    payload.pop('online_motion_aux_state')
    payload['motion_prediction'] = None
    training = sampler.motion_processing_mf(payload, config)
    assert training['points'].shape == (4096, 5)
    assert np.isfinite(training['points']).all()
    assert training['b0_unique_mask'].sum() == 4 * count
    assert training['b0_valid_mask'].sum() == (4096 if count else 0)
    assert np.all(training['b0_point_ids'] >= 0) if count else np.all(training['b0_point_ids'] == -1)
    evaluation, _ = build_v27_eval_input(host, sequence, 8, state.results_bbs, recursive_state=state)
    assert evaluation['points'].shape == (1, 4096, 5)
    assert evaluation['b0_unique_mask'].sum() == 4 * count
    feature_class = _executable('models/backbone/pointnet.py', 'FeaturePointNet')
    feature = feature_class(5, [32, 64, 128], [128, 64], output_size=16).eval()
    with torch.no_grad():
        feature_output, aligned = feature(torch.from_numpy(training['points'].T).float().unsqueeze(0),
                                           return_point_features=True)
    assert feature_output.shape == (1, 16, 16) and aligned.shape == (1, 64, 4096)
    assert torch.isfinite(feature_output).all() and torch.isfinite(aligned).all()


def test_legacy_observation_still_rejects_all_empty_gt_histories(sampler_runtime):
    sampler, config, sequence, _, payload, _, _ = _case(sampler_runtime, 'b0')
    _, classes, _, _ = sampler_runtime
    for frame in sequence:
        frame['pc'] = classes.PointCloud(np.empty((3, 0)))
    config.ct_enable_v27 = False
    payload.pop('online_recursive_state')
    with pytest.raises(AssertionError, match='not enough valid box'):
        sampler.motion_processing_mf(payload, config)


def test_current_and_history_gt_changes_leave_deployed_mechanism_inputs_unchanged(sampler_runtime):
    sampler, config, _, _, payload, _, _ = _case(sampler_runtime, 'full')
    original = sampler.motion_processing_mf(payload, config)
    changed_payload = copy.deepcopy(payload)
    changed_payload['this_frame']['3d_bbox'].center += np.array([18., -11., 2.])
    changed_payload['this_frame']['3d_bbox'].wlh *= 1.5
    for frame in changed_payload['prev_frames'].values():
        frame['3d_bbox'].center += np.array([-8., 17., 1.])
        frame['3d_bbox'].wlh *= 1.2
    changed = sampler.motion_processing_mf(changed_payload, config)
    for key in ('points', 'ref_boxs', 'bbox_size', 'b0_point_ids', 'b0_valid_mask', 'b0_unique_mask',
                'motion_main_ref_boxs', 'motion_acquisition_features', 'ct_extension_points',
                'ct_extension_point_ids', 'ct_extension_source', 'ct_acquisition_margin',
                'ct_acquisition_direction_xy', 'ct_search_support_valid'):
        np.testing.assert_array_equal(original[key], changed[key], err_msg=key)
    assert not np.array_equal(original['box_label'], changed['box_label'])
    assert not np.array_equal(original['target_bbox_size'], changed['target_bbox_size'])


def _raw_snapshot(sequence):
    return [(frame['pc'].points.copy(), frame['pc'].point_ids.copy(),
             frame['3d_bbox'].center.copy(), frame['3d_bbox'].wlh.copy(),
             frame['3d_bbox'].rotation_matrix.copy()) for frame in sequence]


def test_preloaded_shared_frame_objects_remain_unchanged_after_train_and_eval(sampler_runtime):
    sampler, config, sequence, state, payload, host, _ = _case(sampler_runtime, 'full')
    before = _raw_snapshot(sequence)
    dataset = SimpleNamespace(preloading=True, training_samples=[sequence], dynamics_time_mode='true',
                              get_endpoint_key=lambda seq, frame: f'track/{seq}/{frame}')
    fetch = _executable('datasets/nuscenes_lidar_mf.py', 'get_frames')
    first = fetch(dataset, 0, [0, 1, 1, 8])
    assert first[1]['pc'] is first[2]['pc'] is sequence[1]['pc']
    sampler.motion_processing_mf(payload, config)
    build_v27_eval_input(host, sequence, 8, state.results_bbs, recursive_state=state)
    after = _raw_snapshot(sequence)
    for old, new in zip(before, after):
        for old_value, new_value in zip(old, new):
            np.testing.assert_array_equal(old_value, new_value)
    dataset.dynamics_time_mode = 'fixed'
    dataset.dynamics_fixed_delta_t = .25
    fixed = fetch(dataset, 0, [1, 8])
    assert [frame['_ct_effective_timestamp'] for frame in fixed] == [.25, 2.]
    assert '_ct_effective_timestamp' not in sequence[1]


def test_preload_cache_reuses_raw_ids_and_separates_scene_roles(tmp_path):
    # Pickle-compatible raw source shell; this exercises the real cache I/O.
    source = {'pc': SimpleNamespace(points=np.array([[1., 2.], [0., 0.], [0., 0.]]),
                                   point_ids=np.array([12, 19])), 'timestamp': .5}
    calls = []
    dataset = SimpleNamespace(path=str(tmp_path), category_name='Trailer', split='train_track',
        version='v1.0-trainval', preload_offset=10, min_points=-1,
        ct_scene_manifest={'content_sha256': 'a' * 64}, ct_scene_role='train',
        tracklet_anno_list=[[{'frame': 0}]], _virtual_rate_cache_tag=lambda: '')
    dataset._get_frame_from_anno_data = lambda anno: (calls.append(anno) or copy.deepcopy(source))
    load = _executable('datasets/nuscenes_lidar_mf.py', '_load_data')
    first = load(dataset)
    second = load(dataset)
    assert len(calls) == 1
    np.testing.assert_array_equal(first[0][0]['pc'].point_ids, second[0][0]['pc'].point_ids)
    dataset.ct_scene_role = 'calibration'
    load(dataset)
    assert len(calls) == 2
    assert len(list(tmp_path.glob('*_v27_ids_*.dat'))) == 2
