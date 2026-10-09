"""v37 两精度比较必须使用明确基线，并包含真实闭环推理工作量。"""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import tempfile

import pytest
import torch

from tools.check_execution_performance import paired_baseline, model_version, parse_args, _numerical_delta
from tools import execution_benchmark as benchmark


def test_default_baseline_preserves_old_checks_and_pairs_v37_precisions():
    for name in ('ctseqtrackv35', 'ctseqtrackv36'):
        assert paired_baseline({'net_model': name}, 'optimized_bf16') == 'legacy'
    assert paired_baseline({'net_model': 'ctseqtrackv37'}, 'optimized_bf16') == 'optimized_fp32'
    assert paired_baseline({'net_model': 'ctseqtrackv37'}, 'optimized_fp32') == 'legacy'
    assert model_version({'net_model': 'ctseqtrackv37'}) == '37'
    with pytest.raises(ValueError, match='distinct'):
        paired_baseline({'net_model': 'ctseqtrackv37'}, 'optimized_fp32', 'optimized_fp32')
    args = parse_args(['--cfg', 'v37.yaml', '--candidate_profile', 'optimized_bf16',
                       '--baseline_profile', 'optimized_fp32', '--inference'])
    assert args.inference and args.baseline_profile == 'optimized_fp32'


def test_paired_inference_rotates_preserves_schedule_and_reports_latency(tmp_path, monkeypatch):
    calls = []
    def one(config, profile, device, directory, **options):
        calls.append(profile)
        return dict(profile=profile, endpoint_order_sha256='raw-order', initial_state_sha256='initial',
            completed_steps=120, measured_rows=100, window_seconds=10., endpoints_per_second=10.,
            cpu_enqueue_step_seconds=[.1, .3], cuda_event_step_ms=[100., 300.])
    monkeypatch.setattr(benchmark, '_run_inference_one', one)
    report = benchmark._paired_inference({}, torch.device('cpu'), tmp_path,
        profiles=('optimized_fp32', 'optimized_bf16'), warmup=20, steps=100, repeats=3)
    assert calls == ['optimized_fp32', 'optimized_bf16', 'optimized_bf16', 'optimized_fp32',
                     'optimized_fp32', 'optimized_bf16']
    assert report['paired_data_and_initialization_match']
    assert report['summary']['optimized_fp32']['cuda_event_step_p90_ms'] == 300.
    assert report['speedup']['paired_ratios'] == [1., 1., 1.]


def test_paired_inference_rejects_changed_input_schedule(tmp_path, monkeypatch):
    def one(config, profile, device, directory, **options):
        return dict(profile=profile, endpoint_order_sha256=profile, initial_state_sha256='initial',
                    completed_steps=120, measured_rows=100)
    monkeypatch.setattr(benchmark, '_run_inference_one', one)
    with pytest.raises(RuntimeError, match='endpoint_order'):
        benchmark._paired_inference({}, torch.device('cpu'), tmp_path,
            profiles=('optimized_fp32', 'optimized_bf16'), warmup=20, steps=100, repeats=1)


def test_v37_precision_delta_distinguishes_utility_commit_and_state():
    def snapshot(quality, choice):
        return dict(output=dict(accepted_box=torch.zeros(1, 4), selected_index=torch.tensor([choice]),
            commit_quality=torch.tensor([[quality, 0., .1, .1]]), q0_is_fallback=torch.tensor([False]),
            candidates={'valid': torch.tensor([[True, True, False, False]])},
            verification=dict(logits=torch.zeros(1, 4, 3), support_mask=torch.ones(1, 4, 3, dtype=torch.bool),
                commit_support_mask=torch.tensor([[[True] * 3, [False] * 3, [False] * 3, [False] * 3]])),
            evidence={key: torch.zeros(1, 2, dtype=torch.long)
                      for key in ('point_indices', 'point_valid', 'members')}),
            losses={}, rng={}, host_steps=1, optimizer={'state': {}}, state={})
    report = _numerical_delta(snapshot(.49, 0), snapshot(.51, 1))
    assert report['commit_quality_threshold_flips'] == 1
    assert report['output_choice_flips'] == 1 and report['max_abs_output_utility'] == 0.
    assert report['recursive_state_equal']
    assert report['commit_support_flips'] == 0
    threshold = _numerical_delta(snapshot(.49, 0), snapshot(.51, 0))
    assert threshold['selected_update_eligibility_flips'] == 1


def test_actual_inference_loop_includes_diagnostics_without_checkpoint(tmp_path, monkeypatch):
    from models.ct_v31.config import load_config
    from models.ct_v31 import data
    from tests.test_ct_v31_runtime import TinySource
    cfg = load_config('cfgs/ct_seqtrack/35_full_w_scaled_lr_mini.yaml', dict(
        ct_engineering_check=True, workers=0, batch_size=2, point_sample_size=8, v32_reserve_windows=0))
    original = data.build_loaders
    monkeypatch.setattr(data, 'build_loaders', lambda config, roles:
        original(config, roles=roles, sources={'val': TinySource((4, 3))}))
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        report = benchmark._run_inference_one(cfg, 'optimized_fp32', torch.device('cpu'),
                                              tmp_path / 'inference', warmup=1, steps=2)
    finally:
        torch.set_num_threads(previous)
    assert report['status'] == 'passed' and report['completed_steps'] == 3
    assert report['diagnostics_retained'] and report['measured_rows'] > 0
    assert (tmp_path / 'inference/frames.jsonl').stat().st_size > 0
    assert not list(tmp_path.rglob('*.ckpt'))


@pytest.mark.skipif(importlib.util.find_spec('pytorch_lightning') is None, reason='requires actual Lightning')
def test_v37_actual_trainer_pair_keeps_original_loggers_and_diagnostics(monkeypatch):
    from models.ct_v37.config import normalize_config
    from models.ct_v37.data import build_loaders
    from models import ctseqtrackv37
    from tests.test_ct_v31_runtime import TinySource
    cfg = normalize_config(dict(ct_engineering_check=True, point_sample_size=8, batch_size=2,
        workers=0, v32_reserve_windows=0))
    original = ctseqtrackv37.CTSEQTRACKV37
    def build(config):
        return original(config, loaders=build_loaders(config, roles=('train',),
            sources={'train': TinySource((4, 3))}))
    monkeypatch.setattr(ctseqtrackv37, 'CTSEQTRACKV37', build)
    destination = Path(tempfile.mkdtemp(prefix='v37_execution_trainer_', dir='artifacts/ct_checks')) / 'paired'
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        report = benchmark.run_benchmark(cfg, 'cpu', destination,
            warmup=1, steps=2, repeats=1, trace_steps=0)
    finally:
        torch.set_num_threads(previous)
    assert report['status'] == 'passed'
    assert report['logging_equivalence']['original_metric_and_epoch_diagnostic_values_match']
    assert report['logging_equivalence']['console_messages_match']
    assert report['paired_data_and_initialization_match']
    assert not list(destination.rglob('*.ckpt'))
