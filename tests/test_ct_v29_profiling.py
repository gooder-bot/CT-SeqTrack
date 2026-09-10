"""短测速入口与严格数值对照的轻量回归；不运行真实数据或CUDA长训练。"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools import profile_ct_v29_training as tool
from utils import v29_profiling as profiling


def test_three_arm_plan_uses_production_entry_same_card_scratch_and_no_hash_gate():
    plan = tool.build_plan('/data/full', tool.ROOT / 'artifacts/ct_checks/performance_test', gpu='2')
    assert [phase['name'] for phase in plan['phases']] == [
        f'{arm}_{variant}' for arm in tool.ARMS
        for variant in ('legacy1', 'optimized1', 'optimized2', 'legacy2')]
    assert plan['schedule'] == 'ABBA' and len(plan['waves']) == 12
    assert all(len(wave) == 1 for wave in plan['waves'])
    for phase in plan['phases']:
        argv, env = phase['argv'], phase['environment']
        assert argv[argv.index('--workers') + 1] == '4'
        assert argv[argv.index('--limit_train_batches') + 1] == '100'
        assert argv[argv.index('--check_val_every_n_epoch') + 1] == '5'
        assert '--ct_engineering_check' in argv
        assert all(key not in argv for key in ('--preloading', '--checkpoint', '--init_checkpoint', '--assert-passed'))
        assert env['CUDA_VISIBLE_DEVICES'] == '2'
        assert env['CT_V29_PROFILE_WARMUP'] == '20'
        assert 'CT_V28_AUDIT_DIR' not in env
        assert env['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'


@pytest.mark.parametrize('kwargs', [dict(gpu='1,2'), dict(warmup=100), dict(steps=101),
                                   dict(mode='equivalence', steps=8), dict(arms=('bad',)),
                                   dict(arms=('b0', 'b0')), dict(gpus=['1', '2', '3']),
                                   dict(parallel=True), dict(parallel=True, gpus=['1', '2']),
                                   dict(parallel=True, gpus=['1', '1', '3']),
                                   dict(parallel=True, gpus=['1', 'x', '3']),
                                   dict(parallel=True, gpus=['1', '2', '3'], mode='profile'),
                                   dict(parallel=True, gpus=['1', '2', '3'], mode='equivalence')])
def test_invalid_short_plan_rejected(kwargs):
    with pytest.raises(ValueError):
        tool.build_plan('/data', tool.ROOT / 'artifacts/ct_checks/test_invalid', **kwargs)


def test_formal_output_not_used_for_disposable_checks():
    with pytest.raises(ValueError, match='artifacts'):
        tool.build_plan('/data', tool.ROOT / 'output' / 'bad')


@pytest.mark.parametrize('mode', ['profile', 'equivalence'])
def test_profile_and_equivalence_keep_sequential_ab(mode):
    plan = tool.build_plan('/data', tool.ROOT / 'artifacts/ct_checks/perf_ab', mode=mode)
    assert plan['schedule'] == 'AB' and len(plan['waves']) == 6
    assert [phase['name'] for phase in plan['phases']] == [
        f'{arm}_{variant}' for arm in tool.ARMS for variant in ('legacy', 'optimized')]


def test_parallel_benchmark_has_four_abba_waves_on_fixed_three_cards():
    plan = tool.build_plan('/data', tool.ROOT / 'artifacts/ct_checks/perf_parallel',
                           parallel=True, gpus=['1', '2', '3'])
    assert plan['parallel'] and len(plan['waves']) == 4
    for wave, variant in zip(plan['waves'], ('legacy1', 'optimized1', 'optimized2', 'legacy2')):
        assert wave == [f'{arm}_{variant}' for arm in tool.ARMS]
    for phase in plan['phases']:
        assert phase['gpu'] == phase['environment']['CUDA_VISIBLE_DEVICES']
        assert phase['gpu'] == str(tool.ARMS.index(phase['arm']) + 1)


def test_disabled_stage_has_no_cuda_or_clock_calls(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('disabled profiling performed work')
    monkeypatch.setattr(profiling.time, 'perf_counter', forbidden)
    monkeypatch.setattr(torch.cuda, 'synchronize', forbidden)
    with profiling.profile_stage(SimpleNamespace(), 'forward'):
        pass
    monkeypatch.delenv('CT_V29_PROFILE_DIR', raising=False)
    assert profiling.make_runtime_callback() is None


def test_benchmark_does_not_enable_stage_instrumentation(tmp_path, monkeypatch):
    recorder = profiling.RuntimeRecorder(tmp_path, 'benchmark', warmup=0, steps=2)
    recorder.step = 1
    monkeypatch.setattr(recorder, 'synchronize', lambda: pytest.fail('stage synchronization in benchmark'))
    with recorder.stage('forward'):
        pass
    assert not recorder.stage_values


def test_profile_stats_exclude_warmup_and_preserve_nested_names(tmp_path, monkeypatch):
    ticks = iter([0., .010, .020, .040, .050, .090])
    monkeypatch.setattr(profiling.time, 'perf_counter', lambda: next(ticks))
    recorder = profiling.RuntimeRecorder(tmp_path, 'benchmark', warmup=1, steps=3)
    host = SimpleNamespace(device=torch.device('cpu'))
    recorder.host = host
    for index in range(3):
        recorder.batch_start(host, None, index)
        recorder.batch_end()
    result = recorder.report()
    assert result['measured_steps'] == 2
    assert result['batch']['mean_ms'] == pytest.approx(30.)
    assert result['cycle']['mean_ms'] == pytest.approx(40.)
    assert result['cycle']['p90_ms'] == pytest.approx(48.)
    assert '前100' in result['measurement_scope']
    assert json.loads((tmp_path / 'runtime.json').read_text(encoding='utf-8'))['mode'] == 'benchmark'


def test_strict_difference_detects_signed_zero_and_dtype():
    assert profiling.first_difference(torch.tensor([0.]), torch.tensor([-0.]))
    assert profiling.first_difference(torch.ones(1), torch.ones(1, dtype=torch.float64))
    source = {'parameter': torch.ones(2), 'state': [1, None, 'a']}
    assert profiling.first_difference(source, profiling.snapshot(source)) is None
    assert profiling.first_difference(source, dict(source, extra=1))


def test_equivalence_records_training_inputs_losses_and_actions_but_not_h3_time(tmp_path):
    recorder = profiling.RuntimeRecorder(tmp_path, 'equivalence')
    recorder.step = 1
    host = SimpleNamespace(_ct_runtime_profiler=recorder, _ct_mechanism_transaction=True)
    profiling.record_equivalence_transaction(host,
        dict(points=torch.ones(1), ct_targetness=torch.ones(1),
             ct_h3_valid=torch.ones(1), ct_shadow_time_ms=torch.tensor(999.)),
        dict(ct_final_box=torch.ones(4), ct_router_applied_gate=torch.ones(1), irrelevant=3),
        dict(loss_b0_transaction=torch.tensor(2.), ct_diagnostic=torch.tensor(7.)))
    row = recorder.transactions[0]
    assert set(row['input']) == {'points', 'ct_targetness'}
    assert set(row['output']) == {'ct_final_box', 'ct_router_applied_gate'}
    assert set(row['losses']) == {'loss_b0_transaction'}
    assert row['mechanism']


def test_compact_comparison_fails_first_change_and_missing_snapshot(tmp_path):
    a, b = tmp_path / 'a', tmp_path / 'b'
    for root in (a, b):
        (root / 'snapshots').mkdir(parents=True)
        for step in (0, *profiling.AUDIT_STEPS):
            torch.save(dict(state={'weight': torch.ones(2)}, step=step, transactions=[{'input': torch.ones(1)}]),
                       root / 'snapshots' / f'step_{step:03d}.pt')
    assert tool.compare_compact(a, b)['passed']
    torch.save(dict(state={'weight': torch.zeros(2)}, step=3, transactions=[{'input': torch.ones(1)}]),
               b / 'snapshots' / 'step_003.pt')
    result = tool.compare_compact(a, b)
    assert not result['passed'] and result['step'] == 3
    (b / 'snapshots' / 'step_003.pt').unlink()
    assert tool.compare_compact(a, b)['difference'] == 'required snapshot missing'


def test_failed_phase_stops_and_inherited_full_audit_is_removed(tmp_path, monkeypatch):
    import tools.run_ct_v29_checks as checks
    monkeypatch.setattr(checks, 'require_idle_gpu', lambda gpu: None)
    monkeypatch.setenv('CT_V28_AUDIT_DIR', '/old/full_audit')
    monkeypatch.setenv('CT_V28_AUDIT_ACTIVATIONS', '1')
    monkeypatch.setenv('CT_V29_H3_BENCH_DIR', '/old/microbenchmark')
    monkeypatch.setenv('CT_V29_PROFILE_MODE', 'old_mode')
    calls = []
    def popen(argv, **kwargs):
        calls.append(argv)
        assert 'CT_V28_AUDIT_DIR' not in kwargs['env']
        assert 'CT_V28_AUDIT_ACTIVATIONS' not in kwargs['env']
        assert 'CT_V29_H3_BENCH_DIR' not in kwargs['env']
        assert 'CT_V29_PROFILE_MODE' not in kwargs['env']
        return SimpleNamespace(pid=10001, poll=lambda: 9, wait=lambda **_: 9)
    monkeypatch.setattr(tool.subprocess, 'Popen', popen)
    monkeypatch.setattr(tool.os, 'killpg', lambda *args: None, raising=False)
    phases = [dict(name=name, config=__file__, output=str(tmp_path / 'run' / name),
                   log=str(tmp_path / 'run' / name / 'train.log'), argv=[name], environment={})
              for name in ('bad', 'must_not_run')]
    result = tool.execute_plan(dict(output=str(tmp_path / 'run'), gpu='1',
                                    mode='benchmark', phases=phases))
    assert result['status'] == 'failed' and calls == [['bad']]


def _runtime(values):
    return dict(status='complete', measured_steps=len(values), cycle=profiling.summarize(values),
                rows=[dict(cycle_ms=value) for value in values])


def test_abba_reports_distributions_throughput_and_repeat_variation():
    runs = [dict(name=f'{variant}{repeat}', variant=variant, runtime=_runtime([value - 1, value + 1]))
            for variant, repeat, value in [('legacy', 1, 100.), ('optimized', 1, 80.),
                                           ('optimized', 2, 82.), ('legacy', 2, 102.)]]
    result = tool.benchmark_comparison(runs)
    assert result['consistent_measured_reduction']
    assert result['legacy']['repeat_difference_ms'] == 2.
    assert result['optimized']['pooled']['mean_ms'] == 81.
    assert result['optimized']['pooled']['p50_ms'] == 81.
    assert result['optimized']['pooled']['p90_ms'] == pytest.approx(82.4)
    assert result['optimized']['pooled']['iterations_per_second'] == pytest.approx(1000 / 81)
    assert result['optimized']['pooled']['samples_per_second'] == pytest.approx(16000 / 81)
    assert result['observed_speedup'] == pytest.approx(101 / 81)
    assert not result['formal_initialization_allowed']
    # 两次优化均比 A 快，但差值小于 A 自身重复差异；不能把漂移当改善。
    runs[-1]['runtime'] = _runtime([120.])
    runs[1]['runtime'] = _runtime([98.])
    runs[2]['runtime'] = _runtime([99.])
    assert tool.benchmark_comparison(runs)['observed_speedup'] > 1.
    assert not tool.benchmark_comparison(runs)['consistent_measured_reduction']


def _parallel_execution_fixture(tmp_path, monkeypatch, *, failure=None, busy=None):
    import tools.run_ct_v29_checks as checks
    plan = tool.build_plan('/data', tool.ROOT / 'artifacts/ct_checks/planned_parallel',
                           parallel=True, gpus=['1', '2', '3'])
    plan['output'] = str(tmp_path / 'run')
    for phase in plan['phases']:
        phase['config'] = __file__
        phase['output'] = str(tmp_path / 'run' / phase['name'])
        phase['log'] = str(Path(phase['output']) / 'train.log')
        phase['argv'] = [phase['name']]
        phase['environment']['CT_V29_PROFILE_DIR'] = phase['output']
    events, processes, stopped = [], [], []
    interrupted = [False]

    def idle(gpu):
        events.append(('idle', gpu))
        if gpu == busy:
            raise RuntimeError('external GPU task exists')

    class Process:
        def __init__(self, argv, **kwargs):
            self.pid = 10000 + len(processes)
            self.name, self.returncode = argv[0], None
            events.append(('launch', self.name))
            if failure == 'spawn' and len(processes) == 1:
                raise OSError('spawn failure')
            processes.append(self)
            assert kwargs['start_new_session'] == (tool.os.name == 'posix')
            target = Path(kwargs['env']['CT_V29_PROFILE_DIR'])
            variant = 'optimized' if 'optimized' in self.name else 'legacy'
            (target / 'runtime.json').write_text(json.dumps(_runtime([80. if variant == 'optimized' else 100.])),
                                                encoding='utf-8')

        def poll(self):
            if self.returncode is not None:
                return self.returncode
            if failure == 'interrupt':
                if not interrupted[0]:
                    interrupted[0] = True
                    raise KeyboardInterrupt
                return None
            if failure == 'phase':
                self.returncode = 9 if self is processes[1] else None
                return self.returncode
            if failure == 'spawn':
                return None
            self.returncode = 0
            return 0

        def terminate(self):
            stopped.append(self.pid)
            self.returncode = -15

        def kill(self):
            stopped.append(self.pid)
            self.returncode = -9

        def wait(self, timeout=None):
            return self.returncode

    def kill_group(pid, sig):
        assert pid in [process.pid for process in processes]
        stopped.append(pid)
        next(process for process in processes if process.pid == pid).returncode = -int(sig)

    monkeypatch.setattr(checks, 'require_idle_gpu', idle)
    monkeypatch.setattr(tool.subprocess, 'Popen', Process)
    monkeypatch.setattr(tool.os, 'killpg', kill_group, raising=False)
    monkeypatch.setattr(tool.time, 'sleep', lambda _: None)
    return plan, events, processes, stopped


def test_parallel_checks_all_gpus_before_launching_each_wave(tmp_path, monkeypatch):
    plan, events, processes, stopped = _parallel_execution_fixture(tmp_path, monkeypatch)
    result = tool.execute_plan(plan)
    assert result['status'] == 'complete' and len(processes) == 12 and not stopped
    for wave in range(4):
        assert events[wave * 6:wave * 6 + 3] == [('idle', '1'), ('idle', '2'), ('idle', '3')]
        assert [event[0] for event in events[wave * 6 + 3:wave * 6 + 6]] == ['launch'] * 3
    assert all(result['comparisons'][arm]['consistent_measured_reduction'] for arm in tool.ARMS)
    assert all('samples_per_second' in run['metrics'] for run in result['runs'])


def test_busy_target_gpu_starts_nothing_and_never_stops_external_task(tmp_path, monkeypatch):
    plan, _, processes, stopped = _parallel_execution_fixture(tmp_path, monkeypatch, busy='3')
    result = tool.execute_plan(plan)
    assert result['status'] == 'failed' and not processes and not stopped


@pytest.mark.parametrize('failure,expected_children', [('phase', 3), ('spawn', 1), ('interrupt', 3)])
def test_wave_failure_stops_only_owned_children(tmp_path, monkeypatch, failure, expected_children):
    plan, _, processes, stopped = _parallel_execution_fixture(tmp_path, monkeypatch, failure=failure)
    if failure == 'interrupt':
        with pytest.raises(KeyboardInterrupt):
            tool.execute_plan(plan)
        result = json.loads((Path(plan['output']) / 'report.json').read_text(encoding='utf-8'))
        assert result['status'] == 'interrupted'
    else:
        result = tool.execute_plan(plan)
        assert result['status'] == 'failed'
    assert len(processes) == expected_children
    assert set(stopped) <= {process.pid for process in processes}
    assert all(process.returncode is not None for process in processes)


def test_callback_lifecycle_and_actual_optimizer_hooks_with_lightning_marker_stub(tmp_path, monkeypatch):
    class Checkpoint:
        pass
    monkeypatch.setitem(sys.modules, 'pytorch_lightning.callbacks', SimpleNamespace(Checkpoint=Checkpoint))
    monkeypatch.setenv('CT_V29_PROFILE_DIR', str(tool.ROOT / 'artifacts/ct_checks/mock_callback'))
    monkeypatch.setenv('CT_V29_PROFILE_MODE', 'profile')
    monkeypatch.setenv('CT_V29_PROFILE_WARMUP', '0')
    monkeypatch.setenv('CT_V29_PROFILE_STEPS', '2')
    monkeypatch.delenv('CT_V28_AUDIT_DIR', raising=False)
    callback = profiling.make_runtime_callback()
    assert isinstance(callback, Checkpoint)
    host = torch.nn.Linear(2, 1)
    host.device = torch.device('cpu')
    host.config = SimpleNamespace(ct_enable_v29=True, ct_engineering_check=True)
    optimizer = torch.optim.Adam(host.parameters(), foreach=False, fused=False)
    trainer = SimpleNamespace(max_epochs=1, optimizers=[optimizer])
    callback.setup(trainer, host, 'fit')
    host._ct_runtime_profiler.directory = tmp_path
    callback.on_fit_start(trainer, host)
    for index in range(2):
        callback.on_train_batch_start(trainer, host, {}, index)
        optimizer.zero_grad(set_to_none=True)
        with profiling.profile_stage(host, 'observation_forward'):
            loss = host(torch.ones(2)).sum()
        callback.on_before_backward(trainer, host, loss)
        loss.backward()
        callback.on_after_backward(trainer, host)
        optimizer.step()
        callback.on_train_batch_end(trainer, host, loss, {}, index)
    callback.on_train_epoch_end(trainer, host)
    report = json.loads((tmp_path / 'runtime.json').read_text(encoding='utf-8'))
    assert report['status'] == 'complete'
    assert set(report['stages']) == {'observation_forward', 'backward', 'optimizer'}
    callback.teardown(trainer, host, 'fit')
    assert host._ct_runtime_profiler is None and profiling._CURRENT is None


def test_capture_state_includes_actual_counters_bn_and_all_optimizer_state():
    host = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2))
    host.register_buffer('ct_b0_update_step', torch.tensor(3))
    host.register_buffer('ct_targetness_running_positive_points', torch.tensor(8.))
    host._ct_any_plugin_parameter = lambda name: False
    optimizer = torch.optim.Adam(host.parameters(), foreach=False, fused=False)
    host(torch.tensor([[1., 2.], [3., 4.]])).sum().backward()
    optimizer.step()
    result = profiling.capture_training_state(host, SimpleNamespace(optimizers=[optimizer]))
    assert result['model']['ct_b0_update_step'].item() == 3
    assert result['model']['ct_targetness_running_positive_points'].item() == 8
    assert '1.running_mean' in result['shared_b0']['buffers']
    assert result['shared_b0']['adam']['0.weight']['step'].item() == 1
