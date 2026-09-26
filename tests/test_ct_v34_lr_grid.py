"""八组 LR 网格共享原比较证据；验证固定硬门及注册差异。"""
import json
import shutil

import pytest
import yaml

from tools import compare_v34_b0 as base
from tools import compare_v34_lr_grid as grid
from tests.test_ct_v34_comparison import read, replace_evaluation, runs, write


def revise_config(root, **changes):
    path = root / 'resolved_config.yaml'
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    config.update(changes)
    path.write_text(yaml.safe_dump(config), encoding='utf-8')
    manifest = read(root / 'run_manifest.json')
    manifest['config_sha256'] = base.config_digest(config, manifest)
    write(root / 'run_manifest.json', manifest)
    for path in (root / 'training_audits').glob('*.json'):
        audit = read(path)
        if 'config_sha256' in audit:
            audit['config_sha256'] = manifest['config_sha256']
            write(path, audit)


@pytest.fixture
def grid_runs(runs, tmp_path):
    result = dict(registered_reference=runs['reference'], baseline=runs['baseline'], old_b0=runs['old_b0'])
    for name, role, lr in [('reference', 'reference', 1e-4), ('reference_half', 'reference', 5e-5),
        *((label.lower(), label.split('_')[0], grid.RECIPES[label.split('_')[1]]) for label in grid.CANDIDATES)]:
        root = tmp_path / ('grid_' + name)
        shutil.copytree(runs[role], root)
        revise_config(root, experiment_name=name, lr=lr)
        result[name] = root
    return result


def test_grid_reuses_registered_gates_and_same_lr_pairs(grid_runs):
    # 重跑 R 比旧门槛更好，仍不改变预注册门槛，也不会隐藏对新 R 的负差。
    replace_evaluation(grid_runs['reference'], 'reference', dynamic_iou=.99, static_iou=.99,
                       dynamic_distance=.01, static_distance=.01)
    report = grid.compare_grid(**grid_runs)
    assert report['status'] == 'passed'
    assert set(report['passing_rank']) == set(grid.CANDIDATES)
    assert set(report['all_candidate_rank']) == set(grid.CANDIDATES)
    for candidate in report['candidates'].values():
        assert candidate['gates']['overall']['target'] == 'registered_reference'
        assert candidate['gates']['ever_moving']['target'] == 'baseline'
    for recipe in grid.RECIPES:
        pair = report['raw_contributions']['W_minus_S_' + recipe]
        assert pair['left'] == 'W_' + recipe and pair['right'] == 'S_' + recipe
        assert pair['final60']['overall_delta_pp']['success'] == pytest.approx(0.)
    assert report['raw_contributions']['S_half_minus_R1']['final60']['overall_delta_pp']['success'] < 0
    assert report['arms']['S_half']['diagnostics']['60']['context']['recorded_frames'] == 8


def test_old_comparison_defaults_remain_half_lr(grid_runs):
    base.read_run(grid_runs['s_half'], 'S')
    with pytest.raises(base.EvidenceError, match='config.lr'):
        base.read_run(grid_runs['s_normal'], 'S')
    normal = base.compare_runs(grid_runs['registered_reference'], grid_runs['baseline'],
        grid_runs['old_b0'], grid_runs['s_normal'], grid_runs['w_normal'], expected_lr=1e-4)
    assert normal['status'] == 'passed'
    with pytest.raises(base.EvidenceError, match='未登记的学习率'):
        base.read_run(grid_runs['s_half'], 'S', expected_lr=7.5e-5)


def test_grid_cannot_mislabel_lr(grid_runs):
    revise_config(grid_runs['w_quarter'], lr=5e-5)
    with pytest.raises(base.EvidenceError, match='config.lr'):
        grid.compare_grid(**grid_runs)


def test_grid_requires_one_structure_across_all_learning_rates(grid_runs):
    revise_config(grid_runs['s_normal'], observation_loss_weight=2.)
    revise_config(grid_runs['w_normal'], observation_loss_weight=2.)
    with pytest.raises(base.EvidenceError, match='除登记差异外配置不同'):
        grid.compare_grid(**grid_runs)


def test_grid_requires_one_source_across_all_learning_rates(grid_runs):
    for key in ('s_quarter', 'w_quarter'):
        path = grid_runs[key] / 'run_manifest.json'
        manifest = read(path)
        manifest['source']['files']['synthetic.py'] = 'c' * 64
        manifest['source']['sha256'] = base.digest(manifest['source']['files'])
        write(path, manifest)
    with pytest.raises(base.EvidenceError, match='执行源码不同'):
        grid.compare_grid(**grid_runs)


def test_grid_ranking_uses_final60_and_does_not_write_evidence(grid_runs):
    replace_evaluation(grid_runs['w_quarter'], 'W', dynamic_iou=.9, static_iou=.9)
    before = {str(p): (p.stat().st_size, p.stat().st_mtime_ns)
              for root in grid_runs.values() for p in root.rglob('*') if p.is_file()}
    report = grid.compare_grid(**grid_runs)
    after = {str(p): (p.stat().st_size, p.stat().st_mtime_ns)
             for root in grid_runs.values() for p in root.rglob('*') if p.is_file()}
    assert report['selected'] == report['all_candidate_rank'][0] == 'W_quarter'
    assert before == after


def test_grid_cli_reports_invalid_evidence(grid_runs, capsys):
    revise_config(grid_runs['reference_half'], lr=1e-4)
    argv = [part for name in grid.ARGUMENTS
            for part in ('--' + name, str(grid_runs[name.replace('-', '_')]))]
    assert grid.main(argv) == 2
    report = json.loads(capsys.readouterr().out)
    assert report['schema'] == grid.REPORT_SCHEMA and report['status'] == 'invalid_evidence'
