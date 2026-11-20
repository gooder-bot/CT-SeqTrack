from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools.benchmark_v41_training import (
    align_initialization, gpu_activity, paired_result, PAIR_FIELDS, point_payload, workload_digest,
    Window, WindowComplete, checkout_run_directory,
)


def test_endpoint_digest_preserves_batch_order_and_cached_plan():
    def request(frame):
        return SimpleNamespace(epoch=0, tracklet=0, branch=1, window_start=1,
                               window_end=4, frame=frame, drain=False)
    one, two = request(1), request(2)
    digest, rows = workload_digest([[one], [one, two]], 1, 1)
    assert rows == 2
    assert digest == workload_digest([[one, two]], 0, 1)[0]
    assert digest != workload_digest([[two, one]], 0, 1)[0]
    with pytest.raises(RuntimeError):
        workload_digest([[one]], 0, 2)


def test_external_reference_logs_meet_checkout_contract_and_are_archived(tmp_path):
    reference = tmp_path / 'v2'
    destination = tmp_path / 'v3/artifacts/ct_checks/pair/reference'
    destination.mkdir(parents=True)
    with checkout_run_directory(reference, destination) as run:
        assert run.parent == reference / 'artifacts/ct_checks'
        (run / 'run_manifest.json').write_text('metadata')
    assert not run.exists()
    assert (destination / 'run/run_manifest.json').read_text() == 'metadata'
    with checkout_run_directory(tmp_path / 'v3', destination) as local:
        assert local == destination / 'run'


def test_payload_shared_views_and_optional_first_frame():
    xyz = np.ones((3, 10), dtype=np.float32)
    indices = np.arange(3, dtype=np.int64)
    frame = dict(pc=SimpleNamespace(points=xyz), _v41_normal_train_cache=dict(ids=indices, raw_indices=indices))
    rows = [dict(frames={1: frame}, first_frame=frame), dict(frames={2: frame})]
    assert point_payload(rows) == dict(raw_array_bytes=xyz.nbytes, gt10_array_bytes=indices.nbytes)


def test_cuda_union_and_pair_reject_different_workload():
    def event(name, start, end):
        return SimpleNamespace(name=name, time_range=SimpleNamespace(start=start, end=end))
    activity = gpu_activity([event('kernel', 0, 1000), event('Memcpy DtoH', 500, 1300)])
    assert activity == dict(events=2, union_cuda_ms=1.3, h2d_copies=0, d2h_copies=1)
    assert gpu_activity([]) is None
    baseline = dict.fromkeys(PAIR_FIELDS, 1)
    baseline['main_endpoints_per_second'] = 20
    assert paired_result(baseline, dict(baseline, main_endpoints_per_second=30)) == 1.5
    with pytest.raises(ValueError, match='auxiliary_window_sha256'):
        paired_result(baseline, dict(baseline, auxiliary_window_sha256='changed'))


def test_shared_weights_buffers_and_rng_before_optimizer(tmp_path, monkeypatch):
    from models.ct_v41 import runtime
    monkeypatch.setattr(runtime, 'capture_rng_state', lambda: torch.get_rng_state())
    monkeypatch.setattr(runtime, 'restore_rng_state', torch.set_rng_state)
    old = torch.nn.Module()
    old.prior = torch.nn.Module()
    old.prior.acquisition_head = torch.nn.Linear(2, 2)
    old.tail = torch.nn.BatchNorm1d(3)
    with torch.no_grad():
        old.tail.weight.fill_(2)
        old.tail.running_mean.fill_(4)
    path = tmp_path / 'scratch.pt'
    reference = align_initialization(old, path, reference=True, revision_pair='v2-v3')
    random_draw = torch.rand(3)
    new = torch.nn.Module()
    new.tail = torch.nn.BatchNorm1d(3)
    with pytest.raises(ValueError, match='inventory'):
        align_initialization(new, path, reference=False)
    candidate = align_initialization(new, path, reference=False, revision_pair='v2-v3')
    assert torch.equal(new.tail.weight, old.tail.weight)
    assert torch.equal(new.tail.running_mean, old.tail.running_mean)
    assert torch.equal(torch.rand(3), random_draw)
    assert candidate['shared_initialization_sha256'] == reference['shared_initialization_sha256']
    new.tail = torch.nn.BatchNorm1d(4)
    with pytest.raises(ValueError, match='changed owner'):
        align_initialization(new, path, reference=False, revision_pair='v2-v3')


def test_v4_rejects_initial_value_drift_before_loading_weights(tmp_path, monkeypatch):
    from models.ct_v41 import runtime
    monkeypatch.setattr(runtime, 'capture_rng_state', torch.get_rng_state)
    monkeypatch.setattr(runtime, 'restore_rng_state', torch.set_rng_state)
    net = torch.nn.Linear(2, 2)
    path = tmp_path / 'initial.pt'
    reference = align_initialization(net, path, reference=True)
    assert align_initialization(net, path, reference=False) == reference
    with torch.no_grad():
        net.bias.add_(1)
    with pytest.raises(ValueError, match='different scratch initialization'):
        align_initialization(net, path, reference=False)


def test_payload_counts_prepared_fields_and_cached_history_once():
    from dataclasses import dataclass
    @dataclass
    class Pool:
        ids: np.ndarray
    xyz = np.ones((5, 3), dtype=np.float32)
    pool = Pool(np.array([0, 2]))
    packet = ({'points': xyz}, [pool], {0: xyz})
    seen = point_payload([{'frames': {}, '_prepared_start': packet}] * 2)
    assert seen['prepared_start_array_bytes'] == xyz.nbytes + pool.ids.nbytes


def test_v4_reports_drift_but_enforces_workload_and_fresh_inputs():
    from copy import deepcopy
    from tools.v41_equivalence_probe import EquivalenceProbe
    probe = EquivalenceProbe(SimpleNamespace(revision_pair='v3-v4'), WindowComplete)
    expected = dict(main_requests=[dict(frame=3, window_start=3), dict(frame=4, window_start=3)],
        main=dict(point_ids=torch.tensor([[1, 2], [2, 3]]),
                  losses=dict(loss_total=torch.tensor(1.))), optimization_loss=torch.tensor(1.),
        common_gradients_before_adam={'weight': torch.ones(2)}, BN_after_step={}, state_after_commit={},
        committed=(2, 0, 0, 1))
    current = deepcopy(expected)
    current['main']['point_ids'][1, 0] += 1  # 递归裁点允许随模型变化
    current['main']['losses']['loss_total'] += 1e-6
    current['common_gradients_before_adam']['weight'] += .25
    probe.compare_step(10, current, expected)
    assert probe.first_difference['path'] == 'step_10/main/point_ids'
    assert probe.differences['main/losses']['outside_tolerance'] == 0
    assert probe.differences['common_gradients_before_adam']['outside_tolerance'] == 1
    assert 'committed' in probe.compared_groups
    current['main']['point_ids'][0, 1] += 1
    with pytest.raises(AssertionError, match='fresh_inputs/point_ids'):
        probe.compare_step(10, current, expected)
    current['main_requests'][0]['frame'] += 1
    with pytest.raises(AssertionError, match='main_requests'):
        probe.compare_step(10, current, expected)


def test_v4_pair_rejects_even_one_ulp_in_sampled_gradients():
    from copy import deepcopy
    from tools.v41_equivalence_probe import EquivalenceProbe
    probe = EquivalenceProbe(SimpleNamespace(revision_pair='v4-v4'), WindowComplete)
    expected = dict(main_requests=[dict(frame=4, window_start=3)],
        main=dict(point_ids=torch.tensor([[5]])), optimization_loss=torch.tensor(1.),
        common_gradients_before_adam={'weight': torch.ones(2)}, BN_after_step={},
        state_after_commit={}, committed=(1, 0, 0, 1))
    current = deepcopy(expected)
    probe.compare_step(1, current, expected)
    assert not probe.differences and 'common_gradients_before_adam' in probe.compared_groups
    grad = current['common_gradients_before_adam']['weight']
    grad[0] = torch.nextafter(grad[0], torch.tensor(2.))
    with pytest.raises(AssertionError, match='step_1/common_gradients_before_adam/weight'):
        probe.compare_step(1, current, expected)


@pytest.mark.parametrize('pair,reference,corrected', [
    ('v2-v3', True, True), ('v2-v3', False, False), ('v3-v4', True, False), ('v3-v4', False, False),
    ('v4-v4', True, False), ('v4-v4', False, False)])
def test_probe_changes_only_historical_reference_main_loss(pair, reference, corrected):
    from contextlib import ExitStack
    from tools.v41_equivalence_probe import EquivalenceProbe
    value = torch.tensor(2., requires_grad=True)
    def main(batch, output):
        losses = {'loss_total': 3 * value}
        if corrected:
            losses['loss_acquisition'] = value
        return losses
    model = SimpleNamespace(_auxiliary_insertions=(1, 2), _epoch_steps=3,
        _forward_auxiliary_raw=lambda rows: rows, training_step=lambda *a: None,
        on_before_optimizer_step=lambda *a: None,
        tracker=SimpleNamespace(compute_losses=main, compute_aux_losses=lambda *a: {'loss_total': value}))
    probe = EquivalenceProbe(SimpleNamespace(revision_pair=pair, reference=reference), WindowComplete)
    with ExitStack() as hooks:
        probe.install(hooks, model)
        loss = model.tracker.compute_losses({}, None)['loss_total']
        assert model.tracker.compute_aux_losses({}, None)['loss_total'] is value
        (gradient,) = torch.autograd.grad(loss, value)
        assert gradient.item() == pytest.approx(2.95 if corrected else 3.)


def test_equivalence_accepts_float_tolerance_and_rejects_one_raw_id():
    from collections import Counter
    from tools.v41_equivalence_probe import compare
    stats = Counter()
    compare(torch.tensor([1. + 1e-6]), torch.ones(1), 'loss', stats)
    with pytest.raises(AssertionError, match='raw_ids'):
        compare(torch.tensor([100001]), torch.tensor([100000]), 'raw_ids', stats)
    assert stats['float_tensors'] == 1


def test_equivalence_samples_real_auxiliary_successor_and_reports_short_plans(tmp_path):
    from contextlib import ExitStack
    import json
    from models.ct_v41.data import EndpointRequest
    from tools.v41_equivalence_probe import EquivalenceProbe
    probe = EquivalenceProbe(SimpleNamespace(destination=tmp_path), WindowComplete)
    model = SimpleNamespace(_auxiliary_insertions=(10, 21), _forward_auxiliary_raw=lambda rows: rows,
        _epoch_steps=21, tracker=SimpleNamespace(compute_losses=lambda *a: {}, compute_aux_losses=lambda *a: {}),
        training_step=lambda *a: None, on_before_optimizer_step=lambda *a: None,
        auxiliary_builder=SimpleNamespace(states={}))
    assert [index for index in range(22) if probe.selected(model, index)] == [0, 10, 21]
    continued = EndpointRequest(0, 3, 5, 1, 4, 2)
    first = EndpointRequest(0, 2, 5, 1, 4, 1)
    model.auxiliary_builder.states[continued.state_key] = SimpleNamespace(next_frame=2)
    with ExitStack() as hooks:
        probe.install(hooks, model)
        model._forward_auxiliary_raw([{'request': continued}, {'request': first}])
    assert probe.continuations == 1
    assert [item['reused'] for item in probe.observations['auxiliary_requests']] == [True, False]
    model._auxiliary_insertions = (10,)
    with pytest.raises(RuntimeError, match='two scheduled'):
        probe.selected(model, 0)
    assert json.loads((tmp_path / 'equivalence-failure.json').read_text())['status'] == 'not_verified'


def test_window_matches_completed_commits_with_scheduled_auxiliary_steps(monkeypatch):
    class Plan(list):
        source_sha256 = 'same-data'

    def plan(count, branch):
        return Plan([[SimpleNamespace(epoch=0, tracklet=0, branch=branch, window_start=1,
            window_end=count + 1, frame=i + 1, drain=False)] for i in range(count)])

    cuda = SimpleNamespace(synchronize=lambda: None, reset_peak_memory_stats=lambda: None,
        max_memory_allocated=lambda: 0, max_memory_reserved=lambda: 0)
    model = SimpleNamespace(_main_batches=225, _auxiliary_insertions=list(range(4, 225, 5)),
        _loaders={role: SimpleNamespace(batch_sampler=plan(count, branch)) for role, count, branch in
                  [('train', 225, 0), ('auxiliary', 45, 5)]})
    window = Window(SimpleNamespace(trace_mode=None), SimpleNamespace(cuda=cuda))
    monkeypatch.setattr('tools.benchmark_v41_training.child_cpu_seconds', lambda: {})
    trainer = SimpleNamespace(lightning_module=model)
    with pytest.raises(WindowComplete):
        for completed in range(1, model._main_batches + 1):
            trainer.global_step = model._epoch_rows = model._epoch_steps = completed
            model._auxiliary_batches = model._auxiliary_rows = completed // 5
            window.after_advance(trainer)
    report = window.report
    assert (report['warmup_steps'], report['main_endpoints'], report['auxiliary_endpoints']) == (20, 200, 40)
    by_role = report['host_step_intervals_by_role']
    assert by_role['main_only']['steps'] == 160 and by_role['main_with_auxiliary']['steps'] == 40
