"""v28 观测兼容、原始测量边界与完整场景协议。"""
import copy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.test_ct_v27_input_flow import sampler_runtime, _case
from tests.test_ct_v27_protocol_identity import _splits
from utils.b0_sampling import B0EmptyHistoryError, regularize_b0_seqtrack_compat
from utils.point_identity import sampled_identity
from utils.sampling_utils import StatelessObservationBatchSampler
from utils.v27_input import build_v27_eval_input
from utils.v27_protocol import build_scene_manifest as build_v27_manifest
from utils.v28_protocol import build_scene_manifest, select_scene_protocol


@pytest.mark.parametrize('mini', [True, False])
def test_all_training_scenes_and_internal_roles_never_include_official_eval(mini):
    source = _splits(mini)
    version = 'v1.0-mini' if mini else 'v1.0-trainval'
    old = build_v27_manifest(source, version)
    new = build_scene_manifest(source, version)
    assert len(new['scenes']['train']) == (8 if mini else 350)
    assert new['parameter_training_overlap'] is True
    assert new['scenes']['calibration'] == old['scenes']['calibration']
    assert new['scenes']['dev'] == old['scenes']['dev']
    assert set(new['scenes']['calibration'] + new['scenes']['dev']) <= set(new['scenes']['train'])
    assert not set(new['scenes']['test']) & set(new['scenes']['train'])
    assert old['schema'] == 'ct_seqtrack.scene_protocol.v27'
    assert new == build_v27_manifest(source, version, enable_v28=True)
    assert new == build_scene_manifest({k: list(reversed(v)) for k, v in source.items()}, version)
    config = SimpleNamespace(version=version)
    _, role, scenes = select_scene_protocol(config, 'val', source)
    assert role == 'test' and scenes == new['scenes']['test']


def test_whole_population_shuffle_is_repeatable_unbalanced_and_keeps_original_step_budget():
    dataset = list(range(5051 * 4))
    sampler = StatelessObservationBatchSampler(dataset, batch_size=16, seed=42)
    state = torch.get_rng_state().clone()
    batches = list(sampler)
    assert len(batches) == len(sampler) == 1262
    flat = [index for batch in batches for index in batch]
    assert len(flat) == len(set(flat)) == 1262 * 16
    assert any(len({sum(index % 4 == view for index in batch) for view in range(4)}) > 1
               for batch in batches)
    assert list(sampler) == batches
    assert torch.equal(state, torch.get_rng_state())
    sampler.set_epoch(1)
    assert list(sampler) != batches
    sampler.set_epoch(0)
    assert list(sampler) == batches
    # 不限制 batch_size 为四的倍数，且 drop_last=False 精确覆盖全部总体。
    complete = StatelessObservationBatchSampler(list(range(19)), 3, 42, drop_last=False)
    assert sorted(index for batch in complete for index in batch) == list(range(19))


@pytest.mark.parametrize('count', [0, 1, 2, 3, 16, 23])
def test_b0_slots_match_original_sampling_without_reverting_public_sparse_evidence(sampler_runtime, count):
    sampler, classes, _, _ = sampler_runtime
    xyz = np.arange(count * 3, dtype=np.float32).reshape(count, 3) + 1
    pc = classes.PointCloud(xyz.T, point_ids=np.arange(count) + 101)
    sampled, indices = regularize_b0_seqtrack_compat(xyz, 16, seed=7)
    ids, valid, unique = sampled_identity(pc, indices, 16)
    if count <= 2:
        assert indices is None and not sampled.any()
        assert np.all(ids == -1) and not valid.any() and not unique.any()
        ordinary, ordinary_indices = sampler.points_utils.regularize_pc(xyz, 16, seed=7)
        assert (ordinary_indices is not None) == (count > 0)
        if count:
            assert ordinary.any()
    else:
        rng = np.random.default_rng(7)
        expected = (rng.choice(count, 16, replace=16 > count)
                    if count != 16 else np.arange(count))
        np.testing.assert_array_equal(indices, expected)
        np.testing.assert_array_equal(sampled, xyz[expected])
        assert valid.all()


def test_observation_restores_gt_empty_rejection_but_recursive_eval_retains_empty_endpoint(sampler_runtime):
    sampler, config, sequence, state, payload, host, _ = _case(sampler_runtime, 'b0')
    config.ct_enable_v28 = True
    _, classes, _, _ = sampler_runtime
    for frame in sequence:
        frame['pc'] = classes.PointCloud(np.empty((3, 0)))
    training_payload = copy.deepcopy(payload)
    training_payload.pop('online_recursive_state')
    with pytest.raises(B0EmptyHistoryError, match='not enough valid box'):
        sampler.motion_processing_mf(training_payload, config)
    # 评测使用预测历史；不能根据当前/历史 GT 可见性跳过 endpoint。
    evaluated, _ = build_v27_eval_input(host, sequence, 8, state.results_bbs, recursive_state=state)
    assert evaluated['points'].shape == (1, 4096, 5)
    assert not evaluated['b0_valid_mask'].any()


def test_crop_and_point_slots_match_legacy_b0_with_same_seed(sampler_runtime):
    sampler, config, sequence, state, payload, _, _ = _case(sampler_runtime, 'b0')
    config.ct_enable_v28 = True
    result = sampler.motion_processing_mf(payload, config)
    refs = state.history_boxes(payload['prev_frame_ids'], payload['valid_mask'])
    actual_ids = []
    frames = [sequence[index] for index in payload['prev_frame_ids']] + [sequence[8]]
    crop_boxes = refs + [refs[0]]
    seeds = list(payload['point_sampling_seeds']) + [payload['current_sampling_seed']]
    for slot, (frame, crop_box, seed) in enumerate(zip(frames, crop_boxes, seeds)):
        # 原路径在 crop box 系内裁剪，再转到最近历史 anchor；不启用 canonicalize。
        crop = sampler.points_utils.generate_subwindow_with_aroundboxs(
            frame['pc'], crop_box, refs[0], config.bb_scale, config.bb_offset)
        points, indices = regularize_b0_seqtrack_compat(crop.points.T, 1024, seed=int(seed))
        np.testing.assert_array_equal(result['points'][slot * 1024:(slot + 1) * 1024, :3],
                                      points.astype(np.float32))
        actual_ids.append(sampled_identity(crop, indices, 1024)[0])
    np.testing.assert_array_equal(result['b0_point_ids'], np.stack(actual_ids))


def test_sparse_base_is_never_reclassified_as_novel_and_extension_keeps_two_real_ids(sampler_runtime):
    sampler, config, sequence, state, payload, _, _ = _case(sampler_runtime, 'full')
    config.ct_enable_v28 = True
    _, classes, _, _ = sampler_runtime
    anchor = state.results_bbs[-1]
    local = np.array([[.1, 0., 0.], [.2, 0., 0.], [4.9, 0., 0.], [5., 0., 0.]])
    world = (anchor.rotation_matrix @ local.T) + anchor.center[:, None]
    sequence[8]['pc'] = classes.PointCloud(world, point_ids=np.array([11, 12, 21, 22]))
    result = sampler.motion_processing_mf(payload, config)
    assert result['b0_raw_point_count'][-1] == 2
    assert np.all(result['b0_point_ids'][-1] == -1)
    assert not result['b0_valid_mask'][-1].any()
    novel = result['ct_extension_point_ids'][result['ct_extension_valid_mask'] > 0]
    assert set(novel) == {21, 22}
    assert len(novel) == 2
    np.testing.assert_array_equal(result['ct_base_evidence_points'], result['points'][-1024:])
    np.testing.assert_array_equal(sequence[8]['pc'].point_ids, [11, 12, 21, 22])


@pytest.mark.parametrize('query', [1, 8])
def test_deployed_prior_uses_prediction_role_and_first_query_exception(sampler_runtime, query):
    _, config, sequence, state, _, host, _ = _case(sampler_runtime, 'b0')
    config.ct_enable_v28 = True
    deployed = (state if query == 8 else None)
    results = state.results_bbs[:query]
    evaluated, _ = build_v27_eval_input(host, sequence, query, results, recursive_state=deployed)
    prior = evaluated['points'][0, :3072, 4].numpy()
    assert set(np.unique(prior)) <= ({0., 1.} if query == 1 else {np.float32(.2), np.float32(.8)})
    # 兼容修复没有让当前 GT 接管预测 crop、先验或输入尺寸。
    changed = copy.deepcopy(sequence)
    changed[query]['3d_bbox'].center += 50
    changed[query]['3d_bbox'].wlh *= 3
    other, _ = build_v27_eval_input(host, changed, query, results, recursive_state=deployed)
    for key in ('points', 'ref_boxs', 'bbox_size', 'candidate_bc', 'b0_point_ids'):
        assert torch.equal(evaluated[key], other[key]), key


def _retry_sampler(sampler_runtime):
    sampler_module, _, _, EasyDict = sampler_runtime
    sampler = object.__new__(sampler_module.MotionTrackingSamplerMF)
    sampler.config = EasyDict(seed=42, ct_enable_v28=True, ct_observation_retry_max_attempts=64)
    sampler.epoch = 0
    sampler.online_recursive_training = False
    sampler.use_paired_history = False
    sampler.candidate_trajectory_mode = 'independent'
    sampler.random_sample = False
    sampler.num_candidates = 4
    sampler.dataset = SimpleNamespace(hist_num=3, get_num_frames_total=lambda: 5,
                                      get_frames=lambda *args, **kwargs: (None, None))
    sampler._locate_tracklet = lambda anno: (0, anno)
    return sampler


def test_retry_can_change_candidate_without_moving_global_rng_and_records_identity(sampler_runtime):
    sampler = _retry_sampler(sampler_runtime)
    seen = []
    def build(tracklet, frame, first, current, candidate, offsets, **kwargs):
        # 第0候选全部无有效GT历史，重抽须允许进入其他候选；且异常路径恢复 RNG。
        seen.append(frame * 4 + candidate)
        np.random.rand(3)
        random.random()
        torch.rand(2)
        if candidate == 0:
            raise B0EmptyHistoryError('not enough valid box')
        return {'candidate_id': np.int64(candidate)}
    sampler._build_view = build
    python_before, numpy_before, torch_before = random.getstate(), np.random.get_state(), torch.get_rng_state()
    result = sampler[0]
    assert result['candidate_id'] != 0
    assert result['ct_observation_original_index'] == 0
    assert result['ct_observation_actual_index'] == seen[-1]
    assert result['ct_observation_retry_count'] == len(seen) - 1
    assert result['ct_observation_retry_reason'] == 1
    assert random.getstate() == python_before
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_before[1])
    assert np.random.get_state()[2:] == numpy_before[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)
    assert sampler[0] == result


def test_retry_cap_does_not_hide_unrelated_assertions(sampler_runtime):
    sampler = _retry_sampler(sampler_runtime)
    attempts = []
    def empty(*args, **kwargs):
        attempts.append(1)
        raise B0EmptyHistoryError('not enough valid box')
    sampler._build_view = empty
    with pytest.raises(RuntimeError, match='64 retries'):
        sampler[0]
    assert len(attempts) == 65  # 原始请求一次，加最多64次替换。
    def broken(*args, **kwargs):
        raise AssertionError('unrelated geometry failure')
    sampler._build_view = broken
    with pytest.raises(AssertionError, match='unrelated geometry failure'):
        sampler[0]
