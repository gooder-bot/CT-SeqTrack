"""V41 两流覆盖/一次Adam/独立RNG/测量提交合同；仅写入，服务器执行。"""
from collections import Counter, defaultdict
from copy import deepcopy
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from models.ct_v41.auxiliary import (AuxiliaryDiagnostics, FullTrackAuxiliarySampler, insertion_indices,
                                    insertion_identity, isolated_auxiliary_rng)
from models.ct_v41.config import normalize_config
from models.ct_v41.data import (BatchBuilder, EndpointRequest, RawEndpointDataset,
                               ReadyQueueBatchSampler, point_arrays)
from models.ct_v41.verification import current_q0_support
from models.ctseqtrackv41 import CTSEQTRACKV41
from tests.helpers.raw_tracking import TinySource, fake_output, fake_prior


def cfg(**kwargs):
    values = dict(ct_engineering_check=True, v41_attention_backend='math',
        batch_size=2, workers=0, epoch=11, v32_reserve_windows=0,
        v41_diagnostics=False, v41_memory_logging=False)
    values.update(kwargs)
    return normalize_config(values)


def test_auxiliary_uses_only_complete_tracks_without_changing_main_four_branches():
    lengths = (100,) + (9,) * 10
    main = ReadyQueueBatchSampler(lengths, 4, short_window=4, long_window=8,
        curriculum_epochs=10, reserve_windows=0)
    main.set_epoch(9)
    counts = Counter(r.branch for batch in main for r in batch)
    n = sum(x - 1 for x in lengths)
    assert counts == dict.fromkeys(range(4), n)
    auxiliary = FullTrackAuxiliarySampler(lengths, 3, seed=42, source_sha256='fixed')
    auxiliary.set_epoch(9)
    tracks = defaultdict(list)
    for batch in auxiliary:
        assert len({r.state_key for r in batch}) == len(batch)
        for request in batch:
            assert request.branch == 5 and request.window_start == 1
            assert request.window_end == lengths[request.tracklet]
            tracks[request.tracklet].append(request.frame)
    assert sum(map(len, tracks.values())) == auxiliary.row_count <= n // 4
    assert 0 not in tracks  # 单条超过预算，不能为了填满预算截断。
    for tracklet, frames in tracks.items():
        assert frames == list(range(1, lengths[tracklet]))
    same = FullTrackAuxiliarySampler(lengths, 3, seed=42, source_sha256='fixed')
    same.set_epoch(9)
    assert same.state_dict() == auxiliary.state_dict()
    assert main.row_count == 4 * n


@pytest.mark.parametrize('m,a,expected', [(10, 3, (2, 5, 9)), (4, 4, (0, 1, 2, 3)),
                                        (7, 0, ()), (0, 0, ())])
def test_auxiliary_insertion_is_uniform_and_never_adds_an_adam_step(m, a, expected):
    assert insertion_indices(m, a) == expected
    assert len(set(expected)) == a
    assert insertion_identity(expected) == insertion_identity(list(expected))


def test_tiny_budget_skips_oversized_tracks_and_invalid_insertion_fails():
    sampler = FullTrackAuxiliarySampler((2,), batch_size=1)
    assert sampler.budget == sampler.row_count == len(sampler) == 0
    with pytest.raises(ValueError, match='A <= M'):
        insertion_indices(2, 3)


def test_auxiliary_rng_restores_python_numpy_and_torch_even_on_failure():
    random.seed(7); np.random.seed(8); torch.manual_seed(9)
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    with pytest.raises(RuntimeError, match='auxiliary failed'):
        with isolated_auxiliary_rng(42, 3, 2):
            random.random(); np.random.uniform(); torch.rand(4)
            raise RuntimeError('auxiliary failed')
    assert random.getstate() == before[0]
    now = np.random.get_state()
    assert now[0] == before[1][0] and np.array_equal(now[1], before[1][1])
    assert now[2:] == before[1][2:]
    assert torch.equal(torch.get_rng_state(), before[2])


class ToyTracker(nn.Module):
    def __init__(self):
        super().__init__()
        self.main_parameter = nn.Parameter(torch.tensor(1.))
        self.auxiliary_parameter = nn.Parameter(torch.tensor(1.))

    def compute_losses(self, batch, output):
        return {'loss_total': output.value.square()}

    def compute_aux_losses(self, batch, output):
        return {'loss_total': output.value.square()}


class ToyBuilder:
    def __init__(self, tracker, events, name):
        self.tracker, self.events, self.name = tracker, events, name
        self._pending, self.states = object(), {}

    def commit(self, output, prepared):
        assert self.tracker.main_parameter.item() < 1
        assert self.tracker.auxiliary_parameter.item() < 1
        self.events.append(self.name)
        self._pending = None

    def clear_materialization_cache(self):
        pass


def make_toy_host(monkeypatch):
    tracker, events = ToyTracker(), []
    host = CTSEQTRACKV41(cfg(), tracker=tracker)
    host.train_builder = ToyBuilder(tracker, events, 'main_commit')
    host.auxiliary_builder = ToyBuilder(tracker, events, 'auxiliary_commit')
    host._auxiliary_by_main_step = {0: 0}
    host._auxiliary_iterator = iter([[object(), object()]])
    monkeypatch.setattr(host, 'log', lambda *args, **kwargs: None)
    monkeypatch.setattr(host, '_forward_raw', lambda *args, **kwargs:
        ({}, SimpleNamespace(value=2 * tracker.main_parameter)))
    monkeypatch.setattr(host, '_forward_auxiliary_raw', lambda *args, **kwargs:
        ({}, SimpleNamespace(value=3 * tracker.auxiliary_parameter)))
    monkeypatch.setattr(host, '_auxiliary_diagnostic_records', lambda *args: ([], 0))
    return host, tracker, events


def test_main_plus_quarter_auxiliary_has_one_optimizer_step_then_two_commits(monkeypatch):
    host, tracker, events = make_toy_host(monkeypatch)
    optimizer = host.configure_optimizers()['optimizer']
    loss = host.training_step([object(), object()], 0)
    assert loss.item() == pytest.approx(4 + .25 * 9)
    loss.backward()
    assert tracker.main_parameter.grad.item() == pytest.approx(8.)
    assert tracker.auxiliary_parameter.grad.item() == pytest.approx(4.5)
    assert not events
    optimizer.step()
    host.on_train_batch_end(None, None, 0)
    assert events == ['main_commit', 'auxiliary_commit']
    assert host._epoch_steps == host._auxiliary_batches == 1
    assert host._epoch_rows == host._auxiliary_rows == 2
    assert all(state['step'].item() == 1 for state in optimizer.state.values())
    assert host._pending_train is host._pending_auxiliary is None


def test_auxiliary_loss_failure_drops_both_uncommitted_transactions(monkeypatch):
    host, tracker, events = make_toy_host(monkeypatch)
    def fail(*args):
        torch.rand(3)
        raise RuntimeError('bad auxiliary loss')
    monkeypatch.setattr(tracker, 'compute_aux_losses', fail)
    before = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match='bad auxiliary loss'):
        host.training_step([object()], 0)
    assert torch.equal(before, torch.get_rng_state())
    assert not events and host._epoch_steps == 0
    assert host._pending_train is host._pending_auxiliary is None
    assert host.train_builder._pending is host.auxiliary_builder._pending is None


def output_for(batch, selected=0, score=.9, delta=0.):
    out = fake_output(batch, quality=.8)
    boxes = batch['fallback_box'][:, None].repeat(1, 4, 1)
    boxes[:, 0, 0] += delta
    boxes[:, 3] = batch['target_box']
    out.candidates = SimpleNamespace(boxes=boxes, valid=torch.ones(len(boxes), 4, dtype=torch.bool),
        measurement_candidate=torch.tensor([[True, False, False, True]]).expand(len(boxes), -1))
    out.q0_is_fallback = torch.zeros(len(boxes), dtype=torch.bool)
    out.selected_index.fill_(selected)
    out.accepted_box = boxes[:, selected]
    out.legacy_q0_quality = torch.full((len(boxes),), .8)
    out.selected_quality = out.legacy_q0_quality.clone() if selected == 0 else torch.zeros(len(boxes))
    support = current_q0_support(out.observation, batch, boxes[:, selected])['mask']
    measurement = selected in (0, 3)
    support = support & measurement
    trusted = torch.full((len(boxes),), score if measurement else 0.)
    out.commit_decision = SimpleNamespace(trusted_score=trusted, support_mask=support,
        eligible=(support.sum(-1) >= 3) & (trusted >= .5), source=out.selected_index.clone())
    return out


def test_recovery_is_measurement_but_does_not_inherit_legacy_quality_or_restore_old_velocity():
    raw, builder = RawEndpointDataset(TinySource((6,))), BatchBuilder(cfg())
    request = lambda f: EndpointRequest(0, 0, 5, 1, 6, f)
    first = builder.prepare([raw[request(1)]])
    assert 'future_displacement' not in first
    builder.commit(output_for(first, score=.1, delta=20.), first)
    second = builder.prepare([raw[request(2)]])
    record = builder.commit(output_for(second, selected=3), second, diagnostics=True)[0]
    state = builder.states[request(2).state_key]
    assert record['commit_source'] == 'recovery_measurement' and not record['propagation_only']
    assert record['measurement_eligible'] and state.previous_quality == 0.
    assert state.trusted[-1][1] == pytest.approx(101.)
    assert not state.trusted_velocity_valid  # frame1未可信写入，不能跨缺测恢复速度。
    third = builder.prepare([raw[request(3)]])
    builder.commit(output_for(third, selected=3), third)
    assert state.trusted_velocity_valid  # 后续相邻真实支持才更新速度。


def test_recovery_seed_is_capped_even_when_dual_search_is_inactive():
    raw, builder = RawEndpointDataset(TinySource((6,))), BatchBuilder(cfg())
    request = lambda f: EndpointRequest(0, 0, 5, 1, 6, f)
    first = builder.prepare([raw[request(1)]])
    assert first['recovery_seed_valid'].item() and not first['diagnostic_search_backup_active'].item()
    builder.commit(output_for(first, score=.1, delta=20.), first)
    state = builder.states[request(1).state_key]
    state.trusted[-1] = (np.zeros(4), 98.)
    state.trusted_velocity = np.asarray([2., 0., 0.])
    state.trusted_velocity_valid = True
    state.previous_measurement_eligible = True
    second = builder.prepare([raw[request(2)]])
    assert not second['diagnostic_search_backup_active'].item()
    assert second['recovery_seed_box'][0, 0].item() + second['anchor_box'][0, 0].item() == pytest.approx(3.)
    assert second['fallback_box'][0, 0].item() + second['anchor_box'][0, 0].item() == pytest.approx(6.)


def test_auxiliary_cannot_reseed_from_midtrajectory_gt():
    raw, builder = RawEndpointDataset(TinySource((6,))), BatchBuilder(cfg())
    illegal = EndpointRequest(0, 0, 5, 3, 6, 3)
    with pytest.raises(ValueError, match='legal first frame'):
        builder.prepare([raw[illegal]])


def test_normal_training_cached_view_maps_back_to_raw_ids_and_raw_acquisition(monkeypatch):
    import datasets.kitti_v41 as dataset
    class ScatteredSource(TinySource):
        def get_frames(self, index, frame_ids):
            frames = super().get_frames(index, frame_ids)
            for frame in frames:
                frame['pc'] = frame['pc'][[4, 0, 5, 1, 2, 3]]
            return frames
    source = ScatteredSource((6,))
    raw = RawEndpointDataset(source, training=True, enable_m=True)
    row = raw[EndpointRequest(0, 0, 0, 1, 2, 1)]
    enabled = [False]
    def view(frame, roi_boxes, box_size, **kwargs):
        xyz, ids = point_arrays(frame)
        indices = np.asarray([1, 3, 4, 5]) if enabled[0] else np.arange(len(ids))
        return dict(points=xyz[indices], ids=ids[indices], raw_indices=indices, cache_used=enabled[0])
    monkeypatch.setattr(dataset, 'normal_train_point_view', view)
    full_builder, cached_builder = BatchBuilder(cfg()), BatchBuilder(cfg())
    complete = full_builder.prepare([row])
    full_builder.supervise_prior(complete, fake_prior(complete), training=True)
    enabled[0] = True
    cached = cached_builder.prepare([row])
    cached_builder.supervise_prior(cached, fake_prior(cached), training=True)
    for field in ('points', 'point_ids', 'point_valid', 'segmentation_labels', 'bc_targets',
                  'acquisition_target', 'acquisition_valid', 'diagnostic_raw_point_count',
                  'diagnostic_search_raw_target_count', 'diagnostic_search_union_target_count'):
        assert torch.equal(complete[field], cached[field]), field
    assert cached['diagnostic_train_cache_used'].all()
    assert not complete['diagnostic_train_cache_used'].any()
    assert complete['diagnostic_raw_point_count'].item() == 6


def test_input_metadata_fingerprint_changes_raw_source_identity():
    left, right = TinySource((3,)), TinySource((3,))
    left.input_metadata_sha256, right.input_metadata_sha256 = 'a' * 64, 'b' * 64
    assert RawEndpointDataset(left).source_sha256 != RawEndpointDataset(right).source_sha256


def test_passive_auxiliary_diagnostics_count_long_loss_reappearance_and_censoring():
    diagnostics = AuxiliaryDiagnostics()
    records = []
    for frame in range(1, 13):
        records.append(dict(tracklet='long', frame=frame, depth=frame - 1, timestamp=frame * .1,
            ends_trajectory=frame == 12, selected_exact_iou=.05 if frame == 1 else .2 if frame <= 10 else .8,
            sampled_target_count=0 if frame < 10 else 4,
            trusted_age_seconds=frame * .1, supported_age_seconds=frame * .1))
    diagnostics.add(records, xyz_supervised_count=3)
    diagnostics.add([dict(tracklet='censored', frame=f, depth=f - 1, timestamp=f * .1,
        ends_trajectory=f == 4, selected_exact_iou=.05, sampled_target_count=0,
        trusted_age_seconds=f * .1, supported_age_seconds=f * .1) for f in range(1, 5)])
    report = diagnostics.state_dict()
    assert report['rows'] == 16 and report['depth_ge8_endpoints'] == 4 and report['max_depth'] == 11
    assert report['loss_episodes'] == 2 and report['recoveries'] == 1
    assert report['max_failure_frames'] == 10
    assert report['reappearance_training_endpoints'] == 2
    assert report['target_visibility_return_events'] == 1
    assert report['unrecovered_completed_trajectories'] == 1
    assert report['xyz_supervised_endpoints'] == 3 and report['unfinished_diagnostic_trajectories'] == 0
    assert sum(report['ages']['trusted_age_seconds']['bins']) == 16
    diagnostics.reset()
    assert diagnostics.state_dict()['rows'] == 0


class EmptyLoader:
    def __init__(self, sampler):
        self.batch_sampler = sampler
        self.generator = torch.Generator()

    def __iter__(self):
        return iter(())


def test_resume_reconstructs_saved_auxiliary_selection_instead_of_trusting_a_signed_manifest():
    lengths = (9,) * 16
    main = ReadyQueueBatchSampler(lengths, 2, short_window=4, reserve_windows=0, source_sha256='fixed')
    auxiliary = FullTrackAuxiliarySampler(lengths, 2, source_sha256='fixed')
    host = CTSEQTRACKV41(cfg(), tracker=ToyTracker(), loaders={
        'train': EmptyLoader(main), 'auxiliary': EmptyLoader(auxiliary)})
    saved = deepcopy(auxiliary.state_dict())
    saved['selected_tracklets'].reverse()
    host._resume_auxiliary = dict(sampler=saved, main_batches=len(main),
        insertion_indices=list(insertion_indices(len(main), len(auxiliary))),
        insertion_plan_sha256=insertion_identity(insertion_indices(len(main), len(auxiliary))))
    with pytest.raises(ValueError, match='selection/plan mismatch'):
        host.on_train_epoch_start()


def test_real_lightning_complete_dual_stream_checkpoint_and_resume(tmp_path, monkeypatch):
    """CPU toy只替代网络/点云；真实PL hooks、sampler、Adam、scheduler与恢复合同。

    服务器执行；捕获普通Callback早于module epoch-end而导致的半轮checkpoint，
    并将续训两轮的请求、随机前向及Adam状态与连续三轮逐项比较。
    """
    pl = pytest.importorskip('pytorch_lightning')
    from torch.utils.data import DataLoader, Dataset
    import models.ct_v41.entry as entry
    import models.ct_v41.runtime as runtime
    from models.ct_v41.data import raw_collate
    from models.ct_v41.identity import runtime_key
    from utils.lightning_runtime import FinalWindowCheckpoint

    # 不读取KITTI/登记，保留真实payload与aux计划验证；测试身份不会用于正式run。
    monkeypatch.setattr(entry, 'source_identity', lambda: {'sha256': 'toy-source', 'files': {}})
    monkeypatch.setattr(runtime, 'dataset_identity', lambda _: {'toy_input': 'fixed'})
    config = cfg(epoch=3, lr=.01, lr_milestones=[1, 2], v31_curriculum_epochs=2)
    lengths = (5, 5, 5, 5)

    class Requests(Dataset):
        def __len__(self):
            return sum(n - 1 for n in lengths)

        def __getitem__(self, request):
            return {'request': request}

    class TransactionBuilder:
        def __init__(self):
            self.reset()

        def reset(self):
            self.states, self._pending = {}, None

        def prepare(self, rows):
            assert self._pending is None
            for row in rows:
                request = row['request']
                if request.starts_window:
                    assert request.state_key not in self.states
                    self.states[request.state_key] = request.frame
                assert self.states[request.state_key] == request.frame
            self._pending = rows
            return {}

        def commit(self, output, prepared):
            # 实际Adam必须先完成，不能仅靠测试手动调用hook的顺序通过。
            assert not torch.equal(output.parameter.detach(), output.before)
            for row in self._pending:
                request = row['request']
                if request.ends_window:
                    self.states.pop(request.state_key)
                else:
                    self.states[request.state_key] += 1
            self._pending = None

        def clear_materialization_cache(self):
            pass

    class LifecycleHost(CTSEQTRACKV41):
        def __init__(self):
            main = ReadyQueueBatchSampler(lengths, config.batch_size, short_window=4,
                long_window=8, curriculum_epochs=2, reserve_windows=0,
                source_sha256='toy-data', initial_seed_policy=config.v35_seed_policy)
            auxiliary = FullTrackAuxiliarySampler(lengths, config.batch_size,
                source_sha256='toy-data')
            loaders = {role: DataLoader(Requests(), batch_sampler=sampler,
                collate_fn=raw_collate, num_workers=0, generator=torch.Generator())
                for role, sampler in (('train', main), ('auxiliary', auxiliary))}
            super().__init__(config, tracker=ToyTracker(), loaders=loaders)
            self.train_builder, self.auxiliary_builder = TransactionBuilder(), TransactionBuilder()
            self.events, self.requests, self.forward_values = [], [], []
            self.epoch_audits = []

        def val_dataloader(self):
            return []

        def _toy_forward(self, rows, builder, parameter, scale, role):
            batch = builder.prepare(rows)
            noise = torch.rand(()) + random.random() + float(np.random.uniform())
            output = SimpleNamespace(value=scale * parameter + .01 * noise,
                                     parameter=parameter, before=parameter.detach().clone())
            self.requests.append((int(self.current_epoch), role, tuple(row['request'] for row in rows)))
            self.forward_values.append(float(output.value.detach()))
            return batch, output

        def _forward_raw(self, rows, builder, *, training):
            return self._toy_forward(rows, builder, self.tracker.main_parameter, 2., 'main')

        def _forward_auxiliary_raw(self, rows):
            return self._toy_forward(rows, self.auxiliary_builder,
                                     self.tracker.auxiliary_parameter, 3., 'auxiliary')

        @staticmethod
        def _auxiliary_diagnostic_records(*args):
            return [], 0

        def on_train_epoch_end(self):
            super().on_train_epoch_end()
            assert self._epoch_complete and self._auxiliary_complete
            self.events.append(('complete', self._completed_epoch))
            self.epoch_audits.append(deepcopy(self._last_auxiliary_audit))

        def on_save_checkpoint(self, checkpoint):
            assert self.events[-1] == ('complete', int(self.current_epoch) + 1)
            super().on_save_checkpoint(checkpoint)
            payload = checkpoint[runtime_key(self.config)]
            assert payload['epoch_complete'] and payload['auxiliary']['complete']
            assert payload['completed_epoch'] == payload['sampler']['epoch'] + 1
            self.events.append(('save', payload['completed_epoch']))

    def fit(directory, epochs, checkpoint=None, seed=42):
        pl.seed_everything(seed, workers=True)
        host = LifecycleHost()
        trainer = pl.Trainer(default_root_dir=str(directory), accelerator='cpu', devices=1,
            precision=32, max_epochs=epochs, logger=False, enable_progress_bar=False,
            enable_model_summary=False, num_sanity_val_steps=0, limit_val_batches=0,
            reload_dataloaders_every_n_epochs=1, accumulate_grad_batches=1,
            callbacks=[FinalWindowCheckpoint(keep=3, every_n_epochs=1)])
        trainer.fit(host, ckpt_path=None if checkpoint is None else str(checkpoint))
        return host, trainer

    continuous, continuous_trainer = fit(tmp_path / 'continuous', 3)
    first, _ = fit(tmp_path / 'split', 1)
    saved = tmp_path / 'split' / 'formal_checkpoints' / 'epoch=001.ckpt'
    assert saved.is_file()
    # 故意换全局RNG种子；on_train_epoch_start必须恢复checkpoint的主流RNG。
    resumed, resumed_trainer = fit(tmp_path / 'split', 3, saved, seed=999)
    assert first.events == [('complete', 1), ('save', 1)]
    assert resumed.events == [('complete', 2), ('save', 2), ('complete', 3), ('save', 3)]
    assert first.requests + resumed.requests == continuous.requests
    assert first.forward_values + resumed.forward_values == continuous.forward_values
    assert first.epoch_audits + resumed.epoch_audits == continuous.epoch_audits
    assert resumed.run_id == first.run_id
    assert resumed._completed_epoch == 3 and resumed._epoch_rows == 64
    assert resumed._last_auxiliary_audit['rows'] == 4
    assert resumed_trainer.global_step == continuous_trainer.global_step
    for name, value in continuous.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[name]), name

    def equal_tree(left, right):
        if torch.is_tensor(left):
            assert torch.equal(left, right)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                equal_tree(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            assert len(left) == len(right)
            for a, b in zip(left, right):
                equal_tree(a, b)
        else:
            assert left == right

    equal_tree(continuous_trainer.optimizers[0].state_dict(), resumed_trainer.optimizers[0].state_dict())
    equal_tree(continuous_trainer.lr_scheduler_configs[0].scheduler.state_dict(),
               resumed_trainer.lr_scheduler_configs[0].scheduler.state_dict())
