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
    reference = align_initialization(old, path, reference=True)
    random_draw = torch.rand(3)
    new = torch.nn.Module()
    new.tail = torch.nn.BatchNorm1d(3)
    candidate = align_initialization(new, path, reference=False)
    assert torch.equal(new.tail.weight, old.tail.weight)
    assert torch.equal(new.tail.running_mean, old.tail.running_mean)
    assert torch.equal(torch.rand(3), random_draw)
    assert candidate['shared_initialization_sha256'] == reference['shared_initialization_sha256']
    new.tail = torch.nn.BatchNorm1d(4)
    with pytest.raises(ValueError, match='changed owner'):
        align_initialization(new, path, reference=False)


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
