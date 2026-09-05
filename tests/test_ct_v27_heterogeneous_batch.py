"""Real online assembly must collate mixed support/history/visibility states."""

import copy
import types

import numpy as np
import pytest
import torch

from tests.test_ct_v27_full_model import full_model_runtime, _construct
from tests.test_ct_v27_input_flow import sampler_runtime, _case
from utils.recursive_state import RecursiveTrackState


def _mixed_cases(runtime, model):
    sampler, _, original, _, _, _, _ = _case(runtime[0], 'full')
    classes = runtime[0][1]
    descriptions = [('mature', 8), ('initial', 1), ('two_history', 2),
                    ('three_history', 3), ('empty_current', 8),
                    ('no_global_target', 8), ('unreachable_target', 8),
                    ('reachable_novel_target', 8), ('empty_history', 8)]
    sequences = []
    for description, frame_id in descriptions:
        sequence = copy.deepcopy(original)
        if description == 'empty_current':
            sequence[frame_id]['pc'] = classes.PointCloud(np.empty((3, 0)))
        elif description in ('no_global_target', 'unreachable_target'):
            sequence[frame_id]['3d_bbox'].center += np.asarray((100., 100., 0.))
            if description == 'unreachable_target':
                sequence[frame_id]['pc'] = classes.PointCloud(np.concatenate((
                    sequence[frame_id]['pc'].points,
                    sequence[frame_id]['3d_bbox'].center[:, None]), axis=1))
        elif description == 'reachable_novel_target':
            sequence[frame_id]['3d_bbox'].center = sequence[frame_id - 1]['3d_bbox'].center + np.asarray((7., 0., 0.))
            sequence[frame_id]['pc'] = classes.PointCloud(np.concatenate((
                sequence[frame_id]['pc'].points,
                sequence[frame_id]['3d_bbox'].center[:, None]), axis=1))
        elif description == 'empty_history':
            for frame in sequence[:frame_id]:
                frame['pc'] = classes.PointCloud(np.empty((3, 0)))
        sequences.append(sequence)

    dataset = types.SimpleNamespace(
        hist_num=3, get_num_tracklets=lambda: len(sequences),
        get_num_frames_total=lambda: sum(map(len, sequences)),
        get_num_frames_tracklet=lambda index: len(sequences[index]),
        get_frames=lambda index, frame_ids: [sequences[index][frame] for frame in frame_ids],
        get_tracklet_key=lambda index: 'mixed/' + descriptions[index][0])
    config = copy.deepcopy(model.config)
    config.candidate_trajectory_mode = 'shared_se2'
    config.num_candidates = config.ct_recursive_candidate_views = config.ct_b0_candidate_views = 1
    config.ct_b0_candidate_weights = [1.]
    raw_sampler = sampler.MotionTrackingSamplerMF(dataset, config=config)
    raw_rows = []
    model._ct_recursive_states = {}
    for slot, (description, frame_id) in enumerate(descriptions):
        sequence = sequences[slot]
        key = dataset.get_tracklet_key(slot)
        state = RecursiveTrackState(slot, key, sequence[0]['3d_bbox'],
                                    timestamps={0: sequence[0]['timestamp']})
        for index in range(1, frame_id):
            deployed = copy.deepcopy(sequence[index]['3d_bbox'])
            deployed.center[0] += .2
            state.append(index, deployed, sequence[index]['timestamp'], quality=[100., .4, .2, 1.])
        model._ct_recursive_states[(0, key)] = state
        raw_rows.append(raw_sampler._online_raw_view(0, 0, slot, slot, frame_id, 0))
    return descriptions, raw_rows


@pytest.mark.parametrize('arm,backend', [('b1_gru', 'gru'), ('b1_cfc', 'cfc'),
                                      ('full', 'gru'), ('full', 'cfc')])
@pytest.mark.parametrize('reverse', [False, True])
def test_real_online_collates_heterogeneous_history_and_support(full_model_runtime, arm, backend, reverse):
    if arm == 'full' and backend == 'cfc':
        model = _construct(full_model_runtime, 'full')
        config = copy.deepcopy(model.config)
        config.motion_v3_temporal_backend = 'cfc'
        model = full_model_runtime[2](config)
    else:
        model = _construct(full_model_runtime, arm)
    model.train()
    descriptions, raws = _mixed_cases(full_model_runtime, model)
    rows = []
    for raw in raws:
        state = model._recursive_state_for_raw(raw)
        prediction = model._online_motion_prepass(raw, state)
        diagnostics = model._prepare_online_state_group(raw, state)
        rows.append(model._process_online_raw(raw, state, prediction, diagnostics))
    expected = set(rows[0])
    expected_shapes = {key: np.shape(value) for key, value in rows[0].items()}
    expected_dtypes = {key: np.asarray(value).dtype for key, value in rows[0].items()
                       if np.asarray(value).dtype.kind in 'fciub'}
    for (description, _), row in zip(descriptions, rows):
        assert set(row) == expected, (description, sorted(set(row) ^ expected))
        assert {key: np.shape(value) for key, value in row.items()} == expected_shapes, description
        for key, value in row.items():
            if np.asarray(value).dtype.kind in 'fciub':
                assert np.asarray(value).dtype == expected_dtypes[key], (description, key)
                assert np.isfinite(value).all(), (description, key)
    count_keys = ('global_novel_target_count', 'max_reachable_target_count',
                  'selected_target_count', 'selected_background_count')
    assert rows[1]['motion_acquisition_target_valid'] == 0
    assert all(rows[1]['motion_margin_' + key] == 0 for key in count_keys)
    assert rows[5]['motion_margin_global_novel_target_count'] == 0
    assert rows[6]['motion_margin_global_novel_target_count'] > 0
    assert rows[6]['motion_margin_max_reachable_target_count'] == 0
    assert rows[6]['motion_acquisition_target_valid'] == 0
    if reverse:
        # default_collate takes keys from row0: cold-start-first can silently
        # drop keys that mature-first would instead report as a KeyError.
        raws = [raws[1], raws[0], *reversed(raws[2:])]
    batch, auxiliary = model._prepare_online_recursive_batch(raws)
    assert auxiliary is None
    assert batch['points'].shape == (len(raws), 4096, 5)
    assert batch['motion_main_ref_boxs'].shape == (len(raws), 3, 4)
    assert batch['motion_aux_ref_boxs'].shape == (len(raws), 3, 4)
    assert batch['motion_acquisition_target'].shape == (len(raws), 2)
    assert batch['motion_margin_global_novel_target_count'].shape == (len(raws),)
    assert all(torch.isfinite(value).all() for value in batch.values()
               if torch.is_tensor(value))
