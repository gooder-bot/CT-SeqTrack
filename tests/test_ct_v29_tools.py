"""v29 工具控制面：版本分派、原始 observation 包及失败停止。"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools import ct_action_v27_runtime as runtime
from tools import run_ct_v29_checks as checks
from tools.preflight_ct_v28 import inspect_v29_observation_batch


def test_v29_runtime_dispatch_and_rows_schema():
    assert runtime.use_v27_runtime(['--v29'])
    assert runtime.use_v27_runtime(['--config', 'cfgs/ct_seqtrack/29_full_cfc_nuscenes_full.yaml'])
    assert runtime.rows_schema({'ct_enable_v29': True, 'ct_enable_v28': True}).endswith('.v29')
    assert runtime.rows_schema({'ct_enable_v28': True}).endswith('.v28')
    assert runtime.rows_schema({}).endswith('.v27')


def test_v29_calibration_cli_enables_internal_policy_fitting(monkeypatch, tmp_path):
    captured = {}
    class Runner:
        def __init__(self, *args, **kwargs):
            self.config = {'ct_enable_v29': True}
            self.checkpoint_sha256 = 'checkpoint'
            self.config_sha256 = 'config'
            self.scene_manifest = {'parameter_training_overlap': True}
        def __call__(self, role, policy):
            assert role == 'calibration' and policy == {'kind': 'never'}
            return []
    def fit(rows, runner, **kwargs):
        captured.update(kwargs)
        return dict(action_policy={'kind': 'never'}, dev_locked_metrics={},
                    parameter_training_overlap=True)
    monkeypatch.setattr(runtime, 'TrackerClosedLoopRunner', Runner)
    monkeypatch.setattr(runtime, 'calibrate_actions_v27', fit)
    runtime.calibrate_main(['--v29', '--config', 'config.yaml', '--checkpoint', 'model.ckpt',
                            '--output', str(tmp_path / 'policy.json')])
    assert captured['enable_v29'] is True


@pytest.mark.parametrize('bad_value', (False, True))
def test_preflight_uses_v29_raw_collate_and_host_prepare_without_optimizer(monkeypatch, bad_value):
    import utils.v29_rollin as rollin
    items = [{'mode': 'teacher', 'sample': {'points': torch.tensor([1.])}},
             {'mode': 'rollin', 'frames': ['raw-frame']}]
    prepared = []
    def prepare(host, received):
        assert received == items
        assert not torch.is_grad_enabled()
        prepared.append(True)
        return {'points': torch.tensor([float('nan') if bad_value else 2.])}
    monkeypatch.setattr(rollin, 'prepare_observation_batch', prepare)
    class Host:
        def __call__(self, batch):
            assert not torch.is_grad_enabled()
            return {'observation': batch['points'] * 2}
        def compute_loss(self, batch, output):
            return {'loss': output['observation'].mean()}
    if bad_value:
        with pytest.raises(RuntimeError, match='non-finite'):
            inspect_v29_observation_batch(items, [[0, 1]], None, host=Host())
    else:
        report = inspect_v29_observation_batch(items, [[0, 1]], None, host=Host())
        assert report['teacher_rows'] == report['rollin_rows'] == 1
        assert report['optimizer_steps'] == 0 and report['losses'] == {'loss': 4.}
    assert prepared == [True]


def test_engineering_plan_uses_one_gpu_four_scratch_runs_and_three_resume_checks():
    plan = checks.build_plan('/data/nuscenes', checks.ROOT / 'artifacts/ct_checks/v29_tool_test',
                             gpu='2', workers=12, python='python')
    phases = plan['phases']
    assert [item['name'] for item in phases] == [
        'preflight', 'b0_a', 'b0_b', 'full_cfc', 'full_gru',
        'compare_b0_b', 'compare_full_cfc', 'compare_full_gru',
        'resume_b0', 'resume_full_cfc', 'resume_full_gru']
    for phase in phases:
        assert '--preloading' not in phase['argv']
        assert '--init_checkpoint' not in phase['argv']
    for phase in phases[1:5]:
        argv = phase['argv']
        assert argv[argv.index('--steps') + 1] == '100'
        assert argv[argv.index('--seed') + 1] == '42'
        assert '--numerical-audit' in argv
    assert plan['gpu'] == '2' and not plan['formal_initialization_allowed']


def test_busy_gpu_rejected_without_killing_processes(monkeypatch):
    answers = iter(['GPU-test\n', 'GPU-test, 12345\n'])
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(stdout=next(answers))
    monkeypatch.setattr(checks.subprocess, 'run', run)
    with pytest.raises(RuntimeError, match='not exclusive'):
        checks.require_idle_gpu('1')
    assert len(calls) == 2 and all(argv[0] == 'nvidia-smi' for argv in calls)


def test_failed_engineering_phase_stops_all_later_work(monkeypatch, tmp_path):
    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=7)
    monkeypatch.setattr(checks.subprocess, 'run', run)
    plan = dict(output=str(tmp_path / 'run'), source_identity={}, gpu='1',
        phases=[dict(name=name, argv=[name], gpu_work=False,
                     log=str(tmp_path / 'run' / (name + '.log'))) for name in ('fails', 'never_run')])
    report = checks.execute_plan(plan)
    assert calls == [['fails']]
    assert report['status'] == 'failed' and not report['passed']


def test_pass_gate_rejects_changed_source_and_plan_only(monkeypatch, tmp_path):
    monkeypatch.setattr(checks, 'source_identity', lambda: {'current': 1})
    path = tmp_path / 'report.json'
    report = dict(schema='ct_seqtrack.engineering_report.v29', passed=True,
                  status='passed', source_identity={'current': 1})
    path.write_text(json.dumps(report), encoding='utf-8')
    assert checks.assert_passed(path)['passed']
    report['source_identity'] = {'old': 1}
    path.write_text(json.dumps(report), encoding='utf-8')
    with pytest.raises(RuntimeError, match='changed'):
        checks.assert_passed(path)
    report.update(status='planned', passed=False)
    path.write_text(json.dumps(report), encoding='utf-8')
    with pytest.raises(RuntimeError, match='have not passed'):
        checks.assert_passed(path)
