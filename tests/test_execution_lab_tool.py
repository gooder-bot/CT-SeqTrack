"""独立工程入口：已知失败排除、过期检查拒绝、配对速度与日志合同。"""
from types import SimpleNamespace

import pytest

from tools import check_ct_execution_lab as lab
from utils.execution_lab import STAGE_A, active_options, enabled, execution_scope


def test_scope_is_private_nested_and_exception_safe():
    assert not active_options()
    with pytest.raises(ValueError, match='engineering'):
        with execution_scope(('labels',)):
            pass
    with pytest.raises(ValueError, match='unknown'):
        with execution_scope(('typo',), engineering=True):
            pass
    with execution_scope(('labels',), engineering=True):
        assert enabled('labels')
        with pytest.raises(RuntimeError):
            with execution_scope(('pool',), engineering=True):
                assert not enabled('labels') and enabled('pool')
                raise RuntimeError('test')
        assert enabled('labels') and not enabled('pool')
    assert not active_options()


def test_cli_requires_original_and_current_checks():
    for argv in ([], ['--mode', 'benchmark'], ['--baseline-root', 'x', '--epochs', '10', '10'],
                 ['--baseline-root', 'x', '--synthetic'], ['--baseline-root', 'x', '--check-steps', '0']):
        with pytest.raises(SystemExit):
            lab.parse_args(argv)
    args = lab.parse_args(['--baseline-root', 'x'])
    assert (args.check_steps, args.warmup, args.steps, args.repeats, args.epochs) == (8, 20, 100, 3, [0, 10])


def test_stage_a_excludes_known_prior_even_if_short_window_happens_to_match(tmp_path, monkeypatch):
    monkeypatch.setattr(lab, 'verify_disabled', lambda *a: {'status': 'passed'})
    calls = []
    def check(args, cfg, options, baseline=(), numerical=False):
        calls.append((tuple(options), numerical))
        if tuple(options) == ('unique_mask',):
            raise AssertionError('simulated strict failure')
        return {'full': {'status': 'passed'}}
    monkeypatch.setattr(lab, 'check_matrix', check)
    report = lab.run_checks(SimpleNamespace(), tmp_path, {})
    assert report['stage_a']['accepted_options'] == ['labels', 'pool', 'geometry_gather', 'commit_indices']
    assert report['stage_a']['rejected_options'] == ['prior_read', 'unique_mask']
    assert all('prior_read' not in options for options, _ in calls if len(options) > 1)
    assert calls[-1][1] and calls[-1][0][-1] == 'factorized_projection'
    assert (tmp_path / 'check-prior_read.json').exists()


def passing_report():
    return dict(schema='ct_seqtrack.execution_lab.v1', synthetic=False, code_bundle={'sha256': 'code'},
        environment={'torch': 'registered'}, configuration_sha256={'full': 'resolved'},
        arguments=dict(arms=list(lab.ARMS), epochs=[0, 10], check_steps=8, path='data'),
        checks=dict(disabled_baseline={'status': 'passed'},
                    stage_a=dict(status='passed', accepted_options=['labels']),
                    stage_b={'status': 'contract_passed_numerics_changed_score_unverified'}))


@pytest.mark.parametrize('case', ['synthetic', 'code', 'environment', 'arms', 'steps', 'path', 'baseline', 'stage_b', 'config'])
def test_benchmark_rejects_stale_incomplete_or_synthetic_check(case):
    report = passing_report()
    if case == 'synthetic': report['synthetic'] = True
    elif case == 'code': report['code_bundle']['sha256'] = 'stale'
    elif case == 'environment': report['environment'] = {}
    elif case == 'arms': report['arguments']['arms'] = ['full']
    elif case == 'steps': report['arguments']['check_steps'] = 1
    elif case == 'path': report['arguments']['path'] = 'other'
    elif case == 'baseline': report['checks']['disabled_baseline']['status'] = 'failed'
    elif case == 'config': report['configuration_sha256']['full'] = 'old-config'
    else: report['checks']['stage_b']['status'] = 'contract_failure'
    with pytest.raises(ValueError):
        lab.validate_check(report, SimpleNamespace(path='data'), {'sha256': 'code'}, {'torch': 'registered'}, {'full': 'resolved'})


def test_valid_check_returns_only_accepted_options():
    assert lab.validate_check(passing_report(), SimpleNamespace(path='data'),
                              {'sha256': 'code'}, {'torch': 'registered'}, {'full': 'resolved'}) == ('labels',)


def test_resolved_config_change_invalidates_engineering_identity():
    from tools.check_execution_performance import load_engineering_config
    cfg = load_engineering_config('cfgs/ct_seqtrack/37_full_mini.yaml')
    before = lab.configuration_identities({'full': cfg})
    changed = dict(cfg, lr=cfg.lr * .5)
    assert lab.configuration_identities({'full': changed}) != before


def fake_runs(stage='A', different_values=False):
    names = ('baseline', 'stage_a') if stage == 'A' else ('stage_a', 'stage_b')
    runs = []
    for repeat in range(1, 4):
        for index, name in enumerate(names if repeat % 2 else tuple(reversed(names))):
            candidate = name == names[1]
            runs.append(dict(implementation_id=name, repeat=repeat, endpoint_order_sha256='raw',
                initial_state_sha256='weights', completed_steps=120, measured_rows=1600,
                final_state_sha256=name if different_values else 'same',
                window_seconds=1. if candidate else 2., endpoints_per_second=1600. if candidate else 800.,
                logging=dict(csv_field_counts={'loss': 10}, tensorboard_field_counts={'loss': 10},
                    diagnostic_files=['epoch011.json'], console_message_count=3,
                    values_sha256=name if different_values else 'same', console_messages_sha256='same')))
    return names, runs


def test_stage_a_trainer_log_or_final_parameter_changes_are_not_speed_success():
    names, runs = fake_runs(different_values=True)
    with pytest.raises(AssertionError, match='逐位'):
        lab.summarize_pair(runs, names, 10, 'A', 3)
    names, runs = fake_runs(stage='B', different_values=True)
    report = lab.summarize_pair(runs, names, 10, 'B', 3)
    assert report['all_pairs_faster'] and report['paired_speedup_ratios'] == [2., 2., 2.]
    assert not report['logging_values_equal']
    runs[-1]['endpoint_order_sha256'] = 'changed'
    with pytest.raises(AssertionError, match='配对数据'):
        lab.summarize_pair(runs, names, 10, 'B', 3)


def test_mixed_timing_does_not_claim_stable_speedup():
    names, runs = fake_runs()
    next(r for r in runs if r['implementation_id'] == 'stage_a' and r['repeat'] == 3)['window_seconds'] = 3.
    report = lab.summarize_pair(runs, names, 0, 'A', 3)
    assert report['median_ratio'] == 2. and not report['all_pairs_faster']


def test_numeric_report_keeps_nonfinite_sentinels_and_nested_output_schema():
    import numpy as np
    import torch
    from tools.execution_lab_runtime import numerical_report
    for array in (np.array, torch.tensor):
        report = numerical_report({'x': array([float('inf')])}, {'x': array([-float('inf')])})
        assert report['tolerance_failures'] == 1
    with pytest.raises(AssertionError, match='interface'):
        numerical_report({'output': {'candidates': {'valid': True}}}, {'output': {'candidates': {}}})
