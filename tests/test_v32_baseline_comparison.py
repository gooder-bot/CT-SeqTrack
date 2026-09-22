"""结果验收仅使用临时合成记录；不读取或修改既有正式 output。"""
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from tools.compare_v32_baselines import (BUDGET, COMMON_CONFIG, COUNTS, EPOCHS,
    MODEL_SCHEMA, compare_runs, main)
from utils.tracking_metrics import metric_contributions


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False) + '\n', encoding='utf-8')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_epoch(root, epoch, *, iou=.6, distance=.7):
    folder = root / 'evaluation' / f'epoch={epoch:03d}'
    folder.mkdir(parents=True, exist_ok=True)
    success, precision = map(float, metric_contributions(iou, distance))
    rows = []
    for track in range(106):
        # 59*21+47*20=2179 个预测端点，另有106个初始化端点。
        length = 22 if track < 59 else 21
        for frame in range(length):
            initial = frame == 0
            rows.append(dict(tracklet=f'synthetic/{track:03d}', frame=frame,
                initialization=initial, iou=1. if initial else iou,
                distance=0. if initial else distance, success=1. if initial else success,
                precision=1. if initial else precision))
    (folder / 'frames.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    metrics = dict(schema=MODEL_SCHEMA, metric_mode='benchmark_compat', checkpoint_epoch=epoch,
                   complete_coverage=True, **COUNTS,
                   success=100 * sum(row['success'] for row in rows) / len(rows),
                   precision=100 * sum(row['precision'] for row in rows) / len(rows))
    write_json(folder / 'metrics.json', metrics)


def write_summary(root):
    metrics = [read_json(root / 'evaluation' / f'epoch={epoch:03d}' / 'metrics.json') for epoch in EPOCHS]
    write_json(root / 'results.json', dict(final=metrics[-1], checkpoint_epochs=EPOCHS,
        late3={key: sum(row[key] for row in metrics) / 3 for key in ('success', 'precision')}))


@pytest.fixture
def runs(tmp_path):
    paths = []
    for seed in (42, 52):
        for label, model in (('b0', 'ctseqtrackv32'), ('ref', 'seqtrack_reference')):
            root = tmp_path / f'{label}-{seed}'
            root.mkdir()
            write_json(root / 'run_manifest.json', dict(schema=MODEL_SCHEMA, model=model, seed=seed,
                arm='b0', enabled=dict(B1=False, B2=False, B3=False)))
            config = dict(COMMON_CONFIG, net_model=model, seed=seed, ct_engineering_check=False,
                          test=False, init_checkpoint=None, v31_evaluate_late3=True)
            (root / 'resolved_config.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
            write_json(root / 'training_budget.json', dict(schema=MODEL_SCHEMA, **BUDGET,
                epoch_complete=True, sampler=dict(source_sha256='a' * 64)))
            for epoch in EPOCHS:
                write_epoch(root, epoch)
            write_summary(root)
            paths.append(root)
    return paths


def arguments(runs):
    return [part for label, path in zip(('b0-42', 'ref-42', 'b0-52', 'ref-52'), runs)
            for part in ('--' + label, str(path))]


def test_complete_four_runs_pass_and_comparison_is_read_only(runs, capsys):
    before = {str(path): (path.stat().st_size, path.stat().st_mtime_ns)
              for root in runs for path in root.rglob('*') if path.is_file()}
    assert main(arguments(runs)) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['status'] == 'passed'
    assert report['verified']['optimizer_steps'] == 71700
    assert report['criterion']['checkpoint_epoch'] == 60
    assert report['pairs']['42']['delta_pp'] == dict(success=0., precision=0.)
    assert report['pairs']['52']['passed']
    after = {str(path): (path.stat().st_size, path.stat().st_mtime_ns)
             for root in runs for path in root.rglob('*') if path.is_file()}
    assert before == after


@pytest.mark.parametrize('seed_index,metric', [(0, 'success'), (2, 'precision')])
def test_each_seed_and_each_final_metric_must_pass(runs, capsys, seed_index, metric):
    kwargs = dict(iou=.4) if metric == 'success' else dict(distance=1.2)
    write_epoch(runs[seed_index], 60, **kwargs)
    write_summary(runs[seed_index])
    assert main(arguments(runs)) == 1
    report = json.loads(capsys.readouterr().out)
    assert report['status'] == 'failed'
    assert len(report['failed_checks']) == 1 and metric in report['failed_checks'][0]


def test_better_late3_cannot_replace_failing_final60(runs, capsys):
    for epoch in (58, 59):
        write_epoch(runs[0], epoch, iou=.99, distance=.01)
    write_epoch(runs[0], 60, iou=.5)
    write_summary(runs[0])
    assert main(arguments(runs)) == 1
    report = json.loads(capsys.readouterr().out)
    pair = report['pairs']['42']
    assert pair['b0_late3']['success'] > pair['reference_late3']['success']
    assert pair['b0_final60']['success'] < pair['reference_final60']['success'] - 2


@pytest.mark.parametrize('file,field,value', [
    ('run_manifest.json', 'schema', 'ct_seqtrack.joint_identity.v31'),
    ('run_manifest.json', 'seed', 52),
    ('run_manifest.json', 'model', 'seqtrack_reference'),
    ('training_budget.json', 'completed_epoch', 59),
    ('training_budget.json', 'epoch_complete', False),
    ('training_budget.json', 'last_epoch_rows', 19107),
    ('training_budget.json', 'last_epoch_steps', 1196),
    ('training_budget.json', 'optimizer_steps', 71911),
    ('results.json', 'checkpoint_epochs', [57, 58, 59]),
])
def test_wrong_identity_or_budget_is_invalid_evidence(runs, capsys, file, field, value):
    path = runs[0] / file
    record = read_json(path)
    record[field] = value
    write_json(path, record)
    assert main(arguments(runs)) == 2
    report = json.loads(capsys.readouterr().out)
    assert report['status'] == 'invalid_evidence' and field in report['message']


@pytest.mark.parametrize('field,value', [('ct_engineering_check', True), ('lr', .001),
                                      ('version', 'v1.0-trainval'), ('category_name', 'Pedestrian')])
def test_engineering_and_nonmatching_config_are_rejected(runs, capsys, field, value):
    path = runs[0] / 'resolved_config.yaml'
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    config[field] = value
    path.write_text(yaml.safe_dump(config), encoding='utf-8')
    assert main(arguments(runs)) == 2
    assert field in json.loads(capsys.readouterr().out)['message']


@pytest.mark.parametrize('section,field,value', [
    ('final', 'checkpoint_epoch', 59), ('final', 'complete_coverage', False),
    ('final', 'metric_mode', 'geometry_exact'), ('final', 'frames', 2284),
    ('final', 'success', 99.), ('late3', 'precision', 99.),
])
def test_false_final_or_late3_summary_is_rejected(runs, capsys, section, field, value):
    path = runs[0] / 'results.json'
    results = read_json(path)
    results[section][field] = value
    write_json(path, results)
    assert main(arguments(runs)) == 2
    assert field in json.loads(capsys.readouterr().out)['message']


def test_missing_evidence_never_creates_a_pass(runs, capsys):
    (runs[2] / 'training_budget.json').unlink()
    assert main(arguments(runs)) == 2
    report = json.loads(capsys.readouterr().out)
    assert report['status'] == 'invalid_evidence' and 'training_budget.json' in report['message']


def test_same_seed_training_source_must_match(runs, capsys):
    path = runs[1] / 'training_budget.json'
    budget = read_json(path)
    budget['sampler']['source_sha256'] = 'b' * 64
    write_json(path, budget)
    assert main(arguments(runs)) == 2
    assert 'source_sha256' in json.loads(capsys.readouterr().out)['message']


@pytest.mark.parametrize('change', ['duplicate', 'wrong_contribution', 'wrong_pair_keys', 'wrong_epoch_keys'])
def test_frame_evidence_is_checked_in_every_epoch_and_pair(runs, capsys, change):
    epochs = EPOCHS if change == 'wrong_pair_keys' else [58]
    for epoch in epochs:
        path = runs[1] / 'evaluation' / f'epoch={epoch:03d}' / 'frames.jsonl'
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        if change == 'duplicate':
            rows[-1] = rows[-2]
        elif change == 'wrong_contribution':
            rows[1]['success'] += .1
        else:
            for row in rows:
                if row['tracklet'] == 'synthetic/000':
                    row['tracklet'] = 'synthetic/different'
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    assert main(arguments(runs)) == 2
    message = json.loads(capsys.readouterr().out)['message']
    assert {'duplicate': '重复帧键', 'wrong_contribution': 'success',
            'wrong_pair_keys': '键集合', 'wrong_epoch_keys': '键集合'}[change] in message


def test_script_works_as_standalone_and_missing_runs_exit_two(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'tools' / 'compare_v32_baselines.py'
    completed = subprocess.run([sys.executable, str(script), *arguments([tmp_path / str(i) for i in range(4)])],
                               capture_output=True, text=True, encoding='utf-8')
    assert completed.returncode == 2
    assert json.loads(completed.stdout)['status'] == 'invalid_evidence'
    assert not list(tmp_path.iterdir())
