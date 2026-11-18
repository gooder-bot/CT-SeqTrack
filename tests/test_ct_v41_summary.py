"""summary/full 诊断边界与最终指标复算；静态写入，服务器执行。"""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.ct_v41.config import normalize_config
from models.ct_v41.data import BatchBuilder, EndpointRequest, RawEndpointDataset
from models.ct_v41.runtime import (CompactTrackingEvaluation, TrackingEvaluation,
                                  _full_stage_summary, loss_episode_summary, make_evaluation, move_tensors)
from models.ct_v41.verification import current_q0_support
from tests.helpers.raw_tracking import TinySource, fake_output
from utils.batch_materialization import attach_materialization_cache, clear_materialization_cache
from utils.raw_frame_memo import RawFrameMemo


def cfg():
    return normalize_config(dict(ct_engineering_check=True, v41_attention_backend='math',
        batch_size=2, workers=0, epoch=2, v32_reserve_windows=0))


def test_evaluation_transfers_only_fields_consumed_on_device():
    fields = {key: torch.ones(1) for key in (
        'points', 'box_size', 'target_box', 'target_box_size', 'segmentation_labels', 'diagnostic_age')}
    evaluation = move_tensors(fields, 'meta', evaluation=True)
    training = move_tensors(fields, 'meta')
    for key, value in fields.items():
        assert (evaluation[key] is value) == (key not in ('points', 'box_size'))
        assert (training[key] is value) == (key == 'diagnostic_age')


def metric_batch(track, frame, *, x, source=0, last=False):
    first = dict(timestamp=0., scene_id='scene', raw_frame_id=10, meta=dict(track_id=track))
    current = dict(timestamp=frame * .25, scene_id='scene', raw_frame_id=10 + frame,
                   meta=dict(track_id=track))
    raw = dict(tracklet_key='track/' + track, first_frame=first, frames={frame: current},
        request=SimpleNamespace(branch=4, frame=frame, ends_window=last))
    predicted = torch.tensor([[x, 0., 0., 0.]], dtype=torch.float32)
    batch = dict(target_box=torch.zeros(1, 4), box_size=torch.tensor([[4., 2., 2.]]),
        target_box_size=torch.tensor([[4., 2., 2.]]), anchor_box=torch.zeros(1, 4))
    compact = SimpleNamespace(accepted_box=predicted, selected_index=torch.tensor([source]))
    full = SimpleNamespace(accepted_box=predicted, selected_index=torch.tensor([source]),
        selected_quality=torch.ones(1), decoder=SimpleNamespace(q0_box=predicted),
        observation=SimpleNamespace(coarse_box=predicted, sequence_valid=torch.ones(1, dtype=torch.bool)))
    return [raw], batch, compact, full


def test_compact_matches_existing_final_metrics_and_stream_recovery_without_candidates(monkeypatch):
    import models.ct_v41.candidate_records as records

    def forbidden(*args, **kwargs):
        raise AssertionError('summary must not construct candidate geometry or support purity')

    monkeypatch.setattr(records, 'candidate_records', forbidden)
    emitted = []
    compact = make_evaluation(cfg(), row_sink=emitted.append)
    legacy = TrackingEvaluation(cfg())
    singleton = RawEndpointDataset(TinySource((1,)))
    compact.add_singleton_tracks(singleton)
    legacy.add_singleton_tracks(singleton)
    # 先失跟、中间IoU区间、恢复；另一轨末帧仍失跟。初始化全计入官方分母。
    specifications = [('a', 1, 10., 1, False), ('b', 1, 0., 0, False),
                      ('a', 2, 2., 0, False), ('b', 2, 12., 2, True),
                      ('a', 3, 0., 3, True)]
    for track, frame, x, source, last in specifications:
        rows, batch, output, full_output = metric_batch(track, frame, x=x, source=source, last=last)
        compact.add_batch(rows, batch, output)
        legacy._add_batch_metrics(rows, batch, full_output)
    result, reference = compact.summary(), legacy._metrics_summary()
    assert not hasattr(compact, 'rows')
    assert result['success'] == pytest.approx(reference['success'], abs=1e-10, rel=0)
    assert result['precision'] == pytest.approx(reference['precision'], abs=1e-10, rel=0)
    assert result['prediction_frames'] == 5 and result['initialization_frames'] == 3
    assert result['total_frames'] == len(emitted) == 8
    replay = loss_episode_summary(emitted)
    for name in ('all', 'recovered', 'unrecovered'):
        assert result['loss_episodes'][name]['episodes'] == replay[name]['episodes']
        assert result['loss_episodes'][name]['frames']['total'] == replay[name]['frames']['total']
        assert result['loss_episodes'][name]['seconds']['total'] == replay[name]['seconds']['total']
    assert result['diagnostics']['over5m_frames'] == 2
    assert result['diagnostics']['over10m_frames'] == 1
    assert result['diagnostics']['max_loss_frames'] == 2
    assert result['selection']['q0'] == dict(count=2, rate=2 / 5)
    assert all(not any('box' in key or 'candidate' in key for key in row) for row in emitted)
    for row in emitted:
        assert {'scene_id', 'track_id', 'raw_frame_id', 'tracklet', 'frame', 'timestamp',
                'initialization', 'iou', 'distance'} <= row.keys()


def test_compact_partial_summary_is_idempotent_and_later_recovery_is_not_double_counted():
    evaluator = CompactTrackingEvaluation(cfg())
    rows, batch, output, _ = metric_batch('a', 1, x=10.)
    evaluator.add_batch(rows, batch, output)
    partial = evaluator.summary()
    assert partial == evaluator.summary()
    assert partial['loss_episodes']['unrecovered']['episodes'] == 1
    rows, batch, output, _ = metric_batch('a', 2, x=0., last=True)
    evaluator.add_batch(rows, batch, output)
    result = evaluator.summary()
    assert result['loss_episodes']['all']['episodes'] == 1
    assert result['loss_episodes']['recovered']['episodes'] == 1
    assert result['loss_episodes']['unrecovered']['episodes'] == 0
    with pytest.raises(ValueError, match='complete and causal'):
        evaluator.add_batch(rows, batch, output)


def test_full_stage_denominators_use_source_validity_not_deduplication():
    good = dict(iou=1., center_error_m=0., axis_yaw_error_rad=0.)
    bad = dict(iou=0., center_error_m=8., axis_yaw_error_rad=.2)
    row = dict(coarse_valid=False, coarse_geometry=bad, success=1., precision=1.,
        diagnostic_candidates=dict(prior_motion_valid=False,
            raw_valid=[True, True, True, False], valid=[True, False, False, False],
            selected_index=0, geometry=[good, good, good, bad], selected_geometry=good))
    result = _full_stage_summary([row])
    assert result['prior_motion_valid_frames'] == 0
    assert result['values']['prior']['valid_frames'] == 1
    assert result['values']['prior']['selectable_frames'] == 0
    assert result['values']['prior']['success'] == 100.
    assert result['values']['recovery_measurement']['success'] is None
    assert result['values']['coarse']['valid_frames'] == 0


def measurement_output(batch):
    output = fake_output(batch, quality=.8)
    boxes = batch['fallback_box'][:, None].repeat(1, 4, 1)
    output.candidates = SimpleNamespace(boxes=boxes,
        valid=torch.ones(len(boxes), 4, dtype=torch.bool),
        measurement_candidate=torch.tensor([[True, False, False, True]]).expand(len(boxes), -1))
    output.q0_is_fallback = torch.zeros(len(boxes), dtype=torch.bool)
    output.accepted_box = boxes[:, 0]
    output.legacy_q0_quality = output.selected_quality.clone()
    support = current_q0_support(output.observation, batch, output.accepted_box)['mask']
    trusted = torch.full((len(boxes),), .9)
    output.commit_decision = SimpleNamespace(trusted_score=trusted, support_mask=support,
        eligible=support.sum(-1) >= 3, source=output.selected_index.clone())
    return output


def test_detail_gate_preserves_inputs_supervision_and_full_state_but_skips_raw_commit(monkeypatch):
    raw = RawEndpointDataset(TinySource((4,)))
    detailed, compact = BatchBuilder(cfg()), BatchBuilder(cfg())
    compact.detailed_diagnostics = False
    request = EndpointRequest(0, 0, 5, 1, 4, 1)
    full_batch = detailed.prepare([raw[request]])

    def no_full_cloud_target_mask(*args, **kwargs):
        raise AssertionError('summary prepare must not count full-cloud GT coverage')

    with monkeypatch.context() as patch:
        patch.setattr(RawFrameMemo, 'target_mask', no_full_cloud_target_mask)
        summary_batch = compact.prepare([raw[request]])
    assert any(key.startswith('diagnostic_') for key in full_batch)
    assert not any(key.startswith('diagnostic_') for key in summary_batch)
    for key in summary_batch:
        assert torch.equal(summary_batch[key], full_batch[key]), key
    assert all(not key.startswith('acquisition_') for batch in (full_batch, summary_batch) for key in batch)
    assert 'diagnostic_search_trusted_age' in full_batch
    assert 'diagnostic_raw_point_count' not in full_batch  # 辅助轨迹无需整云覆盖
    assert 'bc_targets' not in full_batch and 'bc_targets' not in summary_batch
    full_output, summary_output = measurement_output(full_batch), measurement_output(summary_batch)
    detailed.commit(full_output, full_batch, diagnostics=True)

    def forbidden(*args, **kwargs):
        raise AssertionError('Full summary commit must not reconstruct raw support')

    monkeypatch.setattr(compact, '_points', forbidden)
    assert compact.commit(summary_output, summary_batch, diagnostics=False) == []
    left, right = detailed.states[request.state_key], compact.states[request.state_key]
    for key in ('next_frame', 'trusted_velocity_valid', 'previous_measurement_eligible',
                'last_supported_time', 'last_strong_time'):
        assert getattr(left, key) == getattr(right, key), key
    np.testing.assert_array_equal(left.trusted_velocity, right.trusted_velocity)
    for frame in left.boxes:
        np.testing.assert_array_equal(left.boxes[frame], right.boxes[frame])


@pytest.mark.parametrize('auxiliary', [False, True], ids=['main', 'auxiliary'])
def test_real_math_tracker_summary_full_preserve_losses_gradients_and_commit(auxiliary):
    """真实主/辅前向对照；控制同权重/dropout，不用假输出代替梯度验收。"""
    from models.ct_v41.model import JointTracker
    from models.ct_v41.training_diagnostics import TrainingDiagnostics

    def same_state(left, right):
        if isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                same_state(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            assert len(left) == len(right)
            for first, second in zip(left, right):
                same_state(first, second)
        else:
            assert left == right

    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(714)
            full_config = normalize_config(dict(cfg(), diagnostics_level='full'))
            summary_config = normalize_config(dict(full_config, diagnostics_level='summary'))
            full_model = JointTracker(full_config).train()
            summary_model = deepcopy(full_model)
            summary_model.config = summary_config
            raw = RawEndpointDataset(TinySource((8, 8)), training=not auxiliary, enable_m=True)
            branch, start = (5, 1) if auxiliary else (3, 3)
            rows = [raw[EndpointRequest(0, track, branch, start, 8, start)] for track in (0, 1)]
            # KITTI正常训练视图接受PointCloud协议；复用TinySource的真实点/ID。
            for row in rows:
                for frame in row['frames'].values():
                    frame['pc'] = SimpleNamespace(points=frame['pc'].T, point_ids=frame['point_ids'])
            rng = torch.get_rng_state()
            paths = []
            for detailed, model, config in ((True, full_model, full_config),
                                             (False, summary_model, summary_config)):
                torch.set_rng_state(rng)
                builder = BatchBuilder(config)
                builder.detailed_diagnostics = detailed
                batch = attach_materialization_cache(builder.prepare(rows), enabled=True)
                if auxiliary:
                    output = model.forward_aux(batch, include_local_diagnostics=detailed)
                    losses = model.compute_aux_losses(batch, output)
                else:
                    # 用原先独立准备 prior 的调用方式对照 host 现在的单次前向。
                    prior = model.plan_prior(batch) if detailed else None
                    output = model(batch, prior=prior, include_local_diagnostics=detailed)
                    if detailed:
                        output.loss_statistics = {}
                    losses = model.compute_losses(batch, output)
                    assert losses['m_valid_endpoints'] > 0
                    if detailed:
                        TrainingDiagnostics(config.seed, optimized=True).add_batch(rows, batch, output, losses)
                losses['loss_total'].backward()
                gradients = {name: None if parameter.grad is None else parameter.grad.detach().clone()
                             for name, parameter in model.named_parameters()}
                builder.commit(output, batch, diagnostics=False)
                clear_materialization_cache(batch)
                paths.append((output, losses, gradients, builder, torch.get_rng_state()))
            first, second = paths
            for field in ('accepted_box', 'selected_index', 'selected_quality'):
                torch.testing.assert_close(getattr(first[0], field), getattr(second[0], field), rtol=1e-6, atol=1e-7)
            for field in ('boxes', 'valid', 'raw_valid', 'measurement_candidate'):
                torch.testing.assert_close(getattr(first[0].candidates, field), getattr(second[0].candidates, field),
                                           rtol=1e-6, atol=1e-7)
            assert first[1].keys() == second[1].keys()
            for name in first[1]:
                torch.testing.assert_close(first[1][name], second[1][name], rtol=1e-6, atol=1e-7, msg=name)
            assert first[2].keys() == second[2].keys()
            for name, gradient in first[2].items():
                other = second[2][name]
                assert (gradient is None) == (other is None), name
                if gradient is not None:
                    assert torch.isfinite(gradient).all(), name
                    torch.testing.assert_close(gradient, other, rtol=1e-6, atol=1e-7, msg=name)
            active = ('verifier.', 'recovery_head.') if auxiliary else (
                'observation.', 'decoder.', 'prior.', 'multistep.', 'verifier.', 'recovery_head.')
            for prefix in active:
                assert any(value is not None and bool(value.abs().sum() > 0)
                           for name, value in first[2].items() if name.startswith(prefix)), prefix
            if auxiliary:
                assert all(value is None for name, value in first[2].items()
                           if name.startswith(('observation.', 'decoder.', 'prior.', 'multistep.')))
            assert torch.equal(first[4], second[4])
            assert first[3].states.keys() == second[3].states.keys()
            for key in first[3].states:
                same_state(vars(first[3].states[key]), vars(second[3].states[key]))
            # 详细统计也不能额外更新BN或其他buffer。
            for name, buffer in full_model.named_buffers():
                torch.testing.assert_close(buffer, dict(summary_model.named_buffers())[name], rtol=0, atol=0)
    finally:
        torch.set_num_threads(threads)
