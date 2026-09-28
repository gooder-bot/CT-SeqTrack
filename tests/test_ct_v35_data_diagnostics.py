"""v35 合法首测、历史来源及被动诊断的无侵入 CPU 合同。"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import tempfile

import numpy as np
import pytest
import torch

from models.ct_v31.config import normalize_config
from models.ct_v31.data import (BatchBuilder, RawEndpointDataset, ReadyQueueBatchSampler,
    KNOWN_GT_BOX, PERTURBED_BOX, stable_seed)
from models.ct_v31.losses import observation_loss_statistics
from models.ct_v31.model import JointTracker
from models.ct_v31.runtime import capture_rng_state, resume_payload, validate_resume_payload, TrackingEvaluation
from models.ct_v31.training_diagnostics import TrainingDiagnostics
from tests.test_ct_v31_joint import make_batch, model_config
from tests.test_ct_v31_runtime import TinySource, cfg, request, fake_prior, fake_output
from tests.test_ct_v32_training import MINI_TRAIN_LENGTHS
from tests.test_ct_v34_resume import CPUContractHost, _assert_identical


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def v35_config(**updates):
    return normalize_config(dict(net_model='ctseqtrackv35', experiment_family='ct_seqtrack_v35',
        ct_engineering_check=True, workers=0, point_sample_size=8, batch_size=2, epoch=2,
        v31_short_window=4, **updates))


@pytest.mark.parametrize('branch', [0, 1, 2, 3, 4])
def test_true_first_prediction_uses_exact_initial_box_and_hint(branch):
    raw = RawEndpointDataset(TinySource())
    row = raw[request(1, branch=branch)]
    builder = BatchBuilder(v35_config())
    batch = builder.prepare([row])
    state = next(iter(builder.states.values()))
    np.testing.assert_array_equal(state.seed_offset, np.zeros(4))
    np.testing.assert_array_equal(state.boxes[0], row['first_frame']['3d_bbox'])
    assert batch['history_is_initial'].tolist() == [[False, False, True]]
    assert batch['history_box_provenance'].tolist() == [[0, 0, KNOWN_GT_BOX]]
    # TinySource 的实际四个历史输入点均在初始框内，合法已知 hint 为 1。
    assert torch.all(batch['points'][0, 2, :, 4][batch['point_valid'][0, 2]] == 1.)
    if branch in (1, 2, 3):
        legacy = BatchBuilder(cfg(net_model='ctseqtrackv34'))
        old = legacy.prepare([row])
        assert 'history_is_initial' not in old
        assert old['history_box_provenance'][0, -1] == PERTURBED_BOX
        assert np.any(next(iter(legacy.states.values())).seed_offset != 0)


@pytest.mark.parametrize('branch', [0, 2])
def test_midwindow_gt_or_perturbed_seed_is_never_initial_frame(branch):
    raw = RawEndpointDataset(TinySource())
    row = raw[request(4, branch=branch, start=4, end=7)]
    new = BatchBuilder(v35_config()).prepare([row])
    old = BatchBuilder(cfg(net_model='ctseqtrackv34')).prepare([row])
    assert not new['history_is_initial'].any()
    for key in ('points', 'point_valid', 'history_boxes', 'window_seed_offset', 'history_box_provenance'):
        torch.testing.assert_close(new[key], old[key], rtol=0, atol=0)
    assert new['history_box_provenance'][0, -1] == (KNOWN_GT_BOX if branch == 0 else PERTURBED_BOX)


def test_initial_flag_follows_real_frame_zero_not_padding_or_accepted_boxes():
    raw, builder = RawEndpointDataset(TinySource()), BatchBuilder(v35_config())
    first = builder.prepare([raw[request(1)]])
    builder.acquire(first, fake_prior(first))
    builder.commit(fake_output(first), first)
    second = builder.prepare([raw[request(2)]])
    assert second['history_is_initial'].tolist() == [[False, True, False]]
    assert second['history_box_provenance'].tolist() == [[0, KNOWN_GT_BOX, 3]]


def test_v35_seed_identity_preserves_all_sixty_epoch_budgets_and_old_manifests():
    old = ReadyQueueBatchSampler(MINI_TRAIN_LENGTHS, 16, short_window=4)
    new = ReadyQueueBatchSampler(MINI_TRAIN_LENGTHS, 16, short_window=4,
                                 initial_seed_policy=ReadyQueueBatchSampler.INITIAL_SEED_POLICY)
    total_steps = total_rows = 0
    for epoch in range(60):
        new.set_epoch(epoch)
        assert len(new) == 1195 and new.row_count == 19108
        total_steps += len(new)
        total_rows += new.row_count
        if epoch in (0, 9, 59):
            old.set_epoch(epoch)
            assert list(new) == list(old)
            legacy, upgraded = old.state_dict(), new.state_dict()
            assert 'initial_seed_policy' not in legacy
            assert upgraded['initial_seed_policy'] == ReadyQueueBatchSampler.INITIAL_SEED_POLICY
            assert upgraded['seed_policy'] == legacy['seed_policy']
            assert upgraded['plan_sha256'] == legacy['plan_sha256']
            assert upgraded['manifest_sha256'] != legacy['manifest_sha256']
    assert total_steps == 71700 and total_rows == 1146480


def test_resume_rejects_missing_or_changed_initial_policy_even_with_valid_manifest_hash():
    config = v35_config()
    sampler = ReadyQueueBatchSampler((3, 3), 2,
                                     initial_seed_policy=ReadyQueueBatchSampler.INITIAL_SEED_POLICY).state_dict()
    payload = resume_payload(config, completed_epoch=1, complete=True, rows=sampler['rows'], steps=4, sampler=sampler)
    assert validate_resume_payload(payload, config) is payload
    for value in (None, 'wrong'):
        changed = deepcopy(payload)
        if value is None:
            changed['sampler'].pop('initial_seed_policy')
        else:
            changed['sampler']['initial_seed_policy'] = value
        content = {k: v for k, v in changed['sampler'].items() if k != 'manifest_sha256'}
        changed['sampler']['manifest_sha256'] = hashlib.sha256(json.dumps(
            content, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        with pytest.raises(ValueError, match='initial seed policy'):
            validate_resume_payload(changed, config)


def test_detached_endpoint_loss_numerators_match_actual_batch_reductions():
    torch.manual_seed(35)
    model = JointTracker(model_config('b0')).eval()
    batch = make_batch(3, 8)
    batch['history_valid'][0, :2] = False
    batch['point_valid'][1, -1] = False
    batch['point_valid'][2] = False
    output = model(batch)
    losses = model.compute_losses(batch, output)
    stats = observation_loss_statistics(batch, output)
    for name, terms in stats.items():
        coefficient = .1 if name == 'loss_seg' else 1.
        torch.testing.assert_close(terms['batch_contribution'].sum(), losses[name] * coefficient)
        assert all(not value.requires_grad for value in terms.values())
    assert stats['loss_main']['denominator'].tolist() == [1., 1., 0.]
    assert losses['loss_total'].requires_grad


def _train_two_steps(diagnostics):
    random.seed(350)
    np.random.seed(350)
    torch.manual_seed(350)
    host = CPUContractHost(v35_config(v35_train_diagnostics=diagnostics)).train()
    host.on_train_epoch_start()
    raw = RawEndpointDataset(TinySource((3, 3)))
    optimizer = host.configure_optimizers()['optimizer']
    snapshots = []
    # 令第一个端点必被固定散列采到，验证 IoU 路径也不消耗 RNG/梯度。
    chosen = next('sample/' + str(i) for i in range(1000)
        if stable_seed(42, 0, 'sample/' + str(i), 3, 1, 1, 'v35_training_iou') % 16 == 0)
    for frame in (1, 2):
        rows = [raw[replace(request(frame, track=i, end=3), epoch=0)] for i in range(2)]
        rows[0]['tracklet_key'] = chosen
        optimizer.zero_grad(set_to_none=True)
        loss = host.training_step(rows, frame - 1)
        output, batch, _ = host._pending_train
        loss.backward()
        snapshots.append(dict(loss=loss.detach().clone(), prediction=output.accepted_box.detach().clone(),
            gradients={name: None if p.grad is None else p.grad.clone() for name, p in host.named_parameters()}))
        optimizer.step()
        host.on_train_batch_end(loss, rows, frame - 1)
    return host, dict(steps=snapshots, state=deepcopy(host.state_dict()),
                     optimizer=deepcopy(optimizer.state_dict()), rng=capture_rng_state())


def test_training_diagnostics_on_off_preserve_predictions_gradients_rng_and_adam_bitwise():
    plain, expected = _train_two_steps(False)
    observed, actual = _train_two_steps(True)
    _assert_identical(expected, actual)
    assert plain.training_diagnostics is None
    report = observed.training_diagnostics.summary()
    assert report['totals']['endpoints'] == 4 and report['batches'] == 2
    assert report['sampled_geometry']
    assert report['iou_sample_endpoints'] == len(report['sampled_geometry'])
    assert report['iou_sample_fraction'] == len(report['sampled_geometry']) / 4
    assert report['batch_loss_valid_counts']['main_quality'] == 4
    assert all(group['depth'] in (0, 1) for group in report['groups'].values())
    assert 'diagnostic_history_raw_target_count' in report['totals']['statistics']
    assert report['totals']['statistics']['history_is_initial']['count'] == 4
    for flag in ('sequence_empty', 'current_raw_missing', 'raw_present_but_crop_miss',
                 'all_sampled_four_frames_background', 'current_sampled_missing_history_has_target'):
        assert report['totals']['statistics'][flag] == dict(sum=0., count=4, mean=0.)
    # 真实有效端点归约的贡献总和应与真实 batch 标量总和一致。
    for name, terms in report['totals']['loss_statistics'].items():
        coefficient = .1 if name == 'loss_seg' else .5 if name == 'loss_main_quality' else 1.
        original = 'loss_quality' if name == 'loss_main_quality' else name
        assert terms['actual_batch_contribution_sum'] == pytest.approx(
            report['batch_losses'][original]['batch_sum'] * coefficient, rel=1e-6, abs=1e-6)
    assert all('loss_main_quality' in group['loss_statistics'] for group in report['groups'].values())


def test_diagnostic_epoch_reports_are_exclusive_and_idempotent():
    directory = Path(tempfile.mkdtemp(prefix='v35_diagnostics_test_',
        dir=Path(__file__).resolve().parents[1] / 'artifacts' / 'ct_checks'))
    stats = TrainingDiagnostics()
    path = stats.write_epoch(directory)
    assert path.name == 'epoch001.json'
    assert stats.write_epoch(directory) == path
    content = path.read_bytes()
    stats.batches += 1
    with pytest.raises(FileExistsError, match='diagnostic differs'):
        stats.write_epoch(directory)
    assert path.read_bytes() == content


def test_v35_evaluation_exports_signed_world_geometry_and_local_evidence_only_for_v35():
    raw = RawEndpointDataset(TinySource((2, 2)))
    for version in ('v34', 'v35'):
        config = (v35_config() if version == 'v35' else normalize_config(dict(
            net_model='ctseqtrackv34', experiment_family='ct_seqtrack_v34',
            ct_engineering_check=True, workers=0, point_sample_size=8, batch_size=2)))
        builder = BatchBuilder(config)
        rows = [raw[request(1, branch=4, end=2, track=i)] for i in range(2)]
        batch = builder.prepare(rows)
        model = JointTracker(config).eval()
        prior = model.plan_prior(batch)
        builder.acquire(batch, prior, training=False)
        with torch.no_grad():
            output = model(batch, prior=prior)
        evaluation = TrackingEvaluation(config)
        evaluation.add_batch(rows, batch, output)
        record = [r for r in evaluation.rows if not r['initialization']][-1]
        if version == 'v34':
            assert 'diagnostic_fine_box_world' not in record
        else:
            world = output.hypothesis_boxes[-1, 0].detach().double().numpy().copy()
            world[:3] += batch['anchor_box'][-1, :3].double().numpy()
            np.testing.assert_array_equal(record['diagnostic_fine_box_world'], world)
            assert record['diagnostic_first_frame_size_lwh'] == [4., 2., 2.]
            for field in ('local_neighbor_count', 'local_selected_count', 'local_mean_support', 'local_delta_norm'):
                assert np.asarray(record['diagnostic_' + field]).shape == (4, 8)
