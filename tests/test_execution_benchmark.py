"""测速工具的窗口、配对顺序、原日志核验与依赖失败合同。"""
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace

import pytest
import torch

from tools import execution_benchmark as benchmark


def config():
    from models.ct_v31.config import load_config
    return load_config('cfgs/ct_seqtrack/35_full_w_scaled_lr_mini.yaml',
        dict(ct_engineering_check=True, workers=0, batch_size=2, point_sample_size=8,
             v32_reserve_windows=0))


@dataclass
class Request:
    frame: int
    branch: int = 0


def test_timer_measured_window_starts_after_warmup_commit(monkeypatch):
    clock = iter(range(20))
    monkeypatch.setattr(benchmark.time, 'perf_counter', lambda: float(next(clock)))
    timer = benchmark._WindowTimer(torch, torch.device('cpu'), warmup=1, steps=2)
    for frame in range(3):
        timer.before_batch([dict(tracklet_key='tiny', request=Request(frame))])
        timer.after_commit()
    report = timer.report()
    assert report['completed_steps'] == 3 and report['measured_rows'] == 2
    assert report['window_seconds'] == 5.
    assert report['cpu_enqueue_step_seconds'] == [1., 1.]
    assert report['cuda_event_step_ms'] == []
    assert 'not GPU latency' in report['cpu_enqueue_note']


def test_dependency_failure_is_clear_and_does_not_create_output(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'pytorch_lightning', None)
    with pytest.raises(RuntimeError, match='requires pytorch-lightning'):
        benchmark.run_benchmark(config(), 'cpu', tmp_path / 'no-output')
    assert not (tmp_path / 'no-output').exists()


@pytest.mark.parametrize('updates', [dict(steps=0), dict(warmup=-1), dict(repeats=True), dict(trace_steps=-1)])
def test_invalid_budget_is_rejected_before_running(tmp_path, updates):
    with pytest.raises(ValueError, match='integer'):
        benchmark.run_benchmark(config(), 'cpu', tmp_path / 'no-output', **updates)


def fake_runtime(monkeypatch, tmp_path, *, mismatch=False, console_mismatch=False, durations=None):
    monkeypatch.setitem(sys.modules, 'pytorch_lightning', SimpleNamespace(__version__='test'))
    monkeypatch.setattr(benchmark, '__file__', str(tmp_path / 'tools/execution_benchmark.py'))
    from models.ct_v31 import entry
    monkeypatch.setattr(entry, 'configure_numerics', lambda: None)
    calls = []
    def run_one(base, profile, device, directory, **options):
        calls.append((profile, directory.name, options))
        elapsed = (durations or {}).get(directory.name, 1.)
        return dict(profile=profile, endpoint_order_sha256='rows', initial_state_sha256='initial',
            completed_steps=options['warmup'] + options['steps'], measured_rows=2 * options['steps'],
            window_seconds=elapsed, endpoints_per_second=2. * options['steps'] / elapsed,
            cpu_enqueue_step_seconds=[.1] * options['steps'], cuda_event_step_ms=[],
            logging=dict(csv_field_counts={'train/loss_total': 3}, tensorboard_field_counts={'train/loss_total': 3},
                diagnostic_files=['epoch001.json'], values_sha256=profile if mismatch else 'same',
                console_message_count=1, console_messages_sha256=profile if console_mismatch else 'same-console'))
    monkeypatch.setattr(benchmark, '_run_one', run_one)
    return calls, tmp_path / 'artifacts/ct_checks/benchmark'


def test_paired_runs_rotate_order_verify_logs_and_do_not_promote_profile(tmp_path, monkeypatch):
    calls, destination = fake_runtime(monkeypatch, tmp_path)
    report = benchmark.run_benchmark(config(), 'cpu', destination, warmup=1, steps=2, repeats=3, trace_steps=1)
    assert [row[0] for row in calls] == ['legacy', 'optimized_fp32', 'optimized_fp32', 'legacy',
                                      'legacy', 'optimized_fp32', 'legacy', 'optimized_fp32']
    assert report['paired_data_and_initialization_match']
    assert report['logging_equivalence']['original_metric_and_epoch_diagnostic_values_match']
    assert report['logging_equivalence']['console_messages_match']
    assert report['benchmark_epoch_index'] == 0 and 'not measured' in report['workload_note']
    assert report['summary']['legacy']['cpu_enqueue_step_median_seconds'] == .1
    assert report['summary']['legacy']['median_window_average_step_seconds'] == .5
    assert report['speedup']['ratio_of_median_window_times'] == 1.
    assert report['speedup']['median_ratio'] == 1. and report['speedup']['ratio_population_stddev'] == 0.
    assert report['speedup']['performance_pass_threshold'] is None
    assert not report['speedup']['automatically_promoted']
    assert report['status'] == 'passed' and not report['correctness_verified']
    assert not report['checkpoint_saved'] and not report['formal_run']
    assert json.loads((destination / 'report.json').read_text(encoding='utf-8'))['status'] == 'passed'


def test_changed_fp32_logger_values_fail_and_leave_failure_report(tmp_path, monkeypatch):
    _, destination = fake_runtime(monkeypatch, tmp_path, mismatch=True)
    with pytest.raises(RuntimeError, match='changed original Trainer logs'):
        benchmark.run_benchmark(config(), 'cpu', destination, warmup=1, steps=2, repeats=1, trace_steps=0)
    report = json.loads((destination / 'report.json').read_text(encoding='utf-8'))
    assert report['status'] == 'failed'
    assert not report['logging_equivalence']['original_metric_and_epoch_diagnostic_values_match']


def test_changed_console_values_fail_with_preserved_field_frequency(tmp_path, monkeypatch):
    _, destination = fake_runtime(monkeypatch, tmp_path, console_mismatch=True)
    with pytest.raises(RuntimeError, match='ConsoleProgress messages'):
        benchmark.run_benchmark(config(), 'cpu', destination, warmup=1, steps=2, repeats=1, trace_steps=0)
    report = json.loads((destination / 'report.json').read_text(encoding='utf-8'))
    assert report['logging_equivalence']['fields_and_frequencies_match']
    assert not report['logging_equivalence']['console_messages_match']


def test_speedup_reports_measured_repeat_variation_without_a_threshold(tmp_path, monkeypatch):
    _, destination = fake_runtime(monkeypatch, tmp_path, durations={
        'repeat1-legacy': 2., 'repeat1-optimized_fp32': 1.,
        'repeat2-legacy': 3., 'repeat2-optimized_fp32': 2.,
        'repeat3-legacy': 1., 'repeat3-optimized_fp32': 2.})
    report = benchmark.run_benchmark(config(), 'cpu', destination, warmup=1, steps=2, repeats=3, trace_steps=0)
    speedup = report['speedup']
    assert [pair['speedup_ratio'] for pair in speedup['paired_repeats']] == [2., 1.5, .5]
    assert speedup['median_ratio'] == 1.5
    assert speedup['ratio_of_median_window_times'] == 1.
    assert (speedup['min_ratio'], speedup['max_ratio'], speedup['repeats_faster_than_legacy']) == (.5, 2., 2)
    assert speedup['ratio_coefficient_of_variation'] > 0
    assert report['status'] == 'passed' and speedup['performance_pass_threshold'] is None


@pytest.mark.skipif(importlib.util.find_spec('pytorch_lightning') is None, reason='requires actual Lightning')
def test_actual_trainer_benchmark_keeps_original_logs_and_saves_no_checkpoint(monkeypatch):
    from models.ct_v31.data import build_loaders
    from models import ctseqtrackv31
    from tests.test_ct_v31_runtime import TinySource
    original = ctseqtrackv31.CTSEQTRACKV31
    def build(cfg):
        loaders = build_loaders(cfg, roles=('train',), sources={'train': TinySource((4, 3))})
        return original(cfg, loaders=loaders)
    monkeypatch.setattr(ctseqtrackv31, 'CTSEQTRACKV31', build)
    root = Path('artifacts/ct_checks').resolve()
    temporary = Path(tempfile.mkdtemp(prefix='execution_benchmark_test_', dir=root))
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        report = benchmark.run_benchmark(config(), 'cpu', temporary / 'run',
            warmup=1, steps=2, repeats=1, trace_steps=1)
        assert report['status'] == 'passed'
        assert report['logging_equivalence']['original_metric_and_epoch_diagnostic_values_match']
        assert report['logging_equivalence']['console_messages_match']
        for run in report['runs']:
            assert 'lr-Adam' in run['logging']['csv_field_counts']
            assert 'lr-Adam' in run['logging']['tensorboard_field_counts']
            assert run['console_messages'][0].startswith('[v35 train] epoch=1/1 batch=1/3 step=1 loss=')
        assert not list(temporary.rglob('*.ckpt'))
        assert len(list(temporary.rglob('metrics.csv'))) == 4
    finally:
        torch.set_num_threads(previous)
        assert root in temporary.resolve().parents
        shutil.rmtree(temporary)
