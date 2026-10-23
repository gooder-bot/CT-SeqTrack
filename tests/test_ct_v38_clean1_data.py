"""两条固定短轨迹与冻结 v38 的数据、标签、选择和 accepted 递推对照。"""
from pathlib import Path
import os
import subprocess
import sys
import tempfile

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'artifacts/ct_checks/20261023_v38_clean1'
REMOVED_BATCH = {'memory_points', 'memory_valid', 'memory_ids', 'memory_metadata',
    'extension_points', 'extension_ids', 'extension_valid', 'extension_partition',
    'extension_labels', 'diagnostic_reachable_count', 'diagnostic_acquired_count'}
REMOVED_STATE = {'prediction_source', 'last_selected_commit_quality'}
REMOVED_COMMIT = {'memory_write', 'memory_fg_count', 'memory_true_fg_count', 'memory_wrong_write'}
REMOVED_LOSSES = {'loss_identity', 'loss_modes', 'loss_modes_angle', 'loss_modes_center',
    'loss_reliability', 'loss_vote', 'mode_positive_fraction'}
REMOVED_DIAGNOSTICS = {'reachable_novel_target_points', 'reachable_novel_target_points_recorded_frames',
    'acquired_novel_target_points', 'acquired_novel_target_points_recorded_frames',
    'acquisition_reachable_ratio', 'acquisition_realized_ratio', 'mode_formation_rate',
    'any_mode_formation_on_acquired_target_rate', 'target_mode_formation_on_acquired_target_rate',
    'memory_writes', 'memory_wrong_write_rate', 'memory_foreground_purity', 'wrong_write_fg_purity_threshold'}


def compare(old, new, *, path='', removed=frozenset()):
    if isinstance(old, dict):
        assert set(new) == set(old) - set(removed), (path, set(old) ^ set(new))
        for key in new:
            compare(old[key], new[key], path=path + '/' + str(key))
    elif torch.is_tensor(old):
        if old.dtype.is_floating_point:
            torch.testing.assert_close(new, old, rtol=1e-5, atol=1e-6, msg=path)
        else:
            assert torch.equal(new, old), path
    elif isinstance(old, (list, tuple)):
        assert len(new) == len(old), path
        for index, (left, right) in enumerate(zip(old, new)):
            compare(left, right, path=f'{path}/{index}')
    elif isinstance(old, float):
        assert new == pytest.approx(old, abs=1e-6, rel=1e-5), path
    else:
        assert new == old, path


@pytest.mark.parametrize('arm', ['b1', 'full'])
def test_frozen_short_rollouts_and_supervision_preserve_effective_data(arm, comparison_root):
    root, frozen = comparison_root
    directory = Path(tempfile.mkdtemp(prefix='rollout_', dir=root))
    results = []
    for name, source in [('old', frozen), ('new', ROOT)]:
        destination = directory / f'{name}_{arm}.pt'
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        process = subprocess.run([sys.executable, str(ROOT / 'tests/helpers/clean1_data_worker.py'),
            '--root', str(source), '--out', str(destination), '--arm', arm],
            cwd=source, env=environment, capture_output=True, text=True, encoding='utf-8')
        assert process.returncode == 0, process.stdout + process.stderr
        results.append(torch.load(destination, map_location='cpu', weights_only=False))
    old, new = results
    compare(old['sampler'], new['sampler'], path='sampler')
    assert old['final_states'] == new['final_states'] == {}
    assert len(old['steps']) == len(new['steps']) == 5
    demands = torch.cat([step['batch']['acquisition_demand'] for step in new['steps']])
    assert bool(demands.any()) and bool((~demands).any())
    for index, (before, after) in enumerate(zip(old['steps'], new['steps'])):
        compare(before['batch'], after['batch'], path=f'step{index}/batch', removed=REMOVED_BATCH)
        assert not any(key.startswith(('memory_', 'extension_')) for key in after['batch'])
        assert before['states'].keys() == after['states'].keys()
        for key in after['states']:
            compare(before['states'][key], after['states'][key], path=f'step{index}/state', removed=REMOVED_STATE)
        for left, right in zip(before['commit'], after['commit']):
            compare(left, right, path=f'step{index}/commit', removed=REMOVED_COMMIT)
        for key in ('accepted', 'prior', 'selected', 'valid', 'raw_valid', 'alias', 'quality', 'losses'):
            if key == 'losses':
                assert all(float(before[key][name]) == 0 for name in REMOVED_LOSSES)
            compare(before[key], after[key], path=f'step{index}/{key}',
                removed=REMOVED_LOSSES if key == 'losses' else frozenset())
    # 旧版只有空B2统计和schema有计划差异；有效评估指标需相同。
    old_report, new_report = old['evaluation'], new['evaluation']
    old_report.pop('schema'); new_report.pop('schema')
    compare(old_report.pop('diagnostics'), new_report.pop('diagnostics'),
        path='evaluation/diagnostics', removed=REMOVED_DIAGNOSTICS)
    compare(old_report, new_report, path='evaluation')
