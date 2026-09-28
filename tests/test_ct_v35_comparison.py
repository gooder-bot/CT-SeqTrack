"""构造小型完整证据，验证四组锁定、第二种子与风险门分离。"""
from copy import deepcopy
import json
import hashlib
import shutil

import pytest
import yaml

from tools import compare_v35_b0 as compare
from tests import test_ct_v34_comparison as old
from tests.test_ct_v35_identity import config35


def seal_run(root, config, *, v35):
    manifest = old.read(root / 'run_manifest.json')
    schema = 'ct_seqtrack.joint_identity.v35' if v35 else 'ct_seqtrack.joint_identity.v33'
    manifest.update(model=config['net_model'], schema=schema, seed=config['seed'])
    manifest['config_sha256'] = compare.config_digest(config, manifest)
    old.write(root / 'run_manifest.json', manifest)
    (root / 'resolved_config.yaml').write_text(yaml.safe_dump(dict(config)), encoding='utf-8')
    for epoch in range(1, 61):
        path = root / f'training_audits/epoch={epoch:03d}.json'
        audit = old.read(path)
        sampler = audit['sampler']
        sampler['seed'] = config['seed']
        if v35:
            sampler.update(short_window=4, initial_seed_policy=config['v35_seed_policy'])
            audit.update(schema='ct_seqtrack.v35.training_audit.v1', model_schema=schema,
                         config_sha256=manifest['config_sha256'])
        old.seal_sampler(sampler)
        old.write(path, audit)
    old.write(root / 'training_budget.json', dict(schema=schema, epoch_complete=True,
                                                **compare.BUDGET, sampler=sampler))
    old.replace_evaluation(root, 'W' if v35 else 'reference')
    if v35:
        add_local_rows(root)


def add_local_rows(root):
    for epoch in compare.EPOCHS:
        path = root / f'evaluation/epoch={epoch:03d}/frames.jsonl'
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        for row in rows:
            if not row['initialization']:
                row.update({'diagnostic_' + field: [[0.] * 8 for _ in range(4)] for field in
                    ('local_neighbor_count', 'local_selected_count', 'local_mean_support', 'local_delta_norm')})
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')


@pytest.fixture
def runs(tmp_path, monkeypatch):
    legacy = old.runs.__wrapped__(tmp_path, monkeypatch)
    monkeypatch.setattr(compare, 'COUNTS', old.compare.COUNTS)
    # 冻结旧 W-quarter 对照身份，与v34既有工具一致。
    root = legacy['W']
    cfg = yaml.safe_load((root / 'resolved_config.yaml').read_text(encoding='utf-8'))
    cfg['lr'] = 2.5e-5
    manifest = old.read(root / 'run_manifest.json')
    manifest['config_sha256'] = compare.config_digest(cfg, manifest)
    old.write(root / 'run_manifest.json', manifest)
    (root / 'resolved_config.yaml').write_text(yaml.safe_dump(cfg), encoding='utf-8')
    for p in (root / 'training_audits').glob('*.json'):
        row = old.read(p)
        row['config_sha256'] = manifest['config_sha256']
        old.write(p, row)
    result = dict(registered_reference=legacy['reference'], baseline=legacy['baseline'],
                  old_b0=legacy['old_b0'], old_w_quarter=root)
    for recipe in compare.RECIPES:
        target = tmp_path / recipe
        shutil.copytree(legacy['S'], target)
        seal_run(target, config35(recipe), v35=True)
        result[recipe] = target
    monkeypatch.setattr(compare, 'REGISTERED_FRAME_SHA256', {label: tuple(
        hashlib.sha256((path / f'evaluation/epoch={e:03d}/frames.jsonl').read_bytes()).hexdigest()
        for e in compare.EPOCHS) for label, path in (
        ('reference', result['registered_reference']), ('baseline', result['baseline']),
        ('old_b0', result['old_b0']), ('old_w_quarter', result['old_w_quarter']))})
    return result


def add_stage2(runs, tmp_path, recipe='quarter'):
    chosen, ref = tmp_path / 'chosen52', tmp_path / 'reference52'
    shutil.copytree(runs[recipe], chosen)
    seal_run(chosen, config35(recipe, 52), v35=True)
    shutil.copytree(runs['registered_reference'], ref)
    config = yaml.safe_load((ref / 'resolved_config.yaml').read_text(encoding='utf-8'))
    config['seed'] = 52
    seal_run(ref, config, v35=False)
    return dict(runs, selected_seed52=chosen, reference_seed52=ref)


def test_four_complete_candidates_tie_prefers_lower_lr_and_still_waits_seed52(runs):
    report = compare.compare_runs(**runs)
    assert report['status'] == 'passed' and report['selected'] == 'quarter'
    assert report['all_candidate_rank'] == ['quarter', 'half', 'normal', 'scaled']
    assert report['full_planning_paused'] and report['full_pause_reasons'] == ['seed52_not_verified']
    slices = report['explanatory_slices']['quarter']
    original = report['arms']['quarter']['groups']
    assert slices['raw0_2']['n'] == original['raw0']['n'] + original['raw1_2']['n'] == 4
    assert slices['raw_visible']['n'] == 6
    assert slices['raw_visible']['final60']['coarse_fine']['valid_pairs'] == 6
    assert slices['raw0_2']['late3']['checkpoint_rows'] == 12
    assert slices['raw0_2']['late3']['coarse_fine']['valid_pairs'] == 12
    assert set(slices['raw0_2']['epochs']) == {'58', '59', '60'}


def test_incomplete_or_reused_candidate_is_rejected(runs):
    with pytest.raises(compare.EvidenceError, match='四个不同'):
        compare.compare_runs(**dict(runs, half=runs['quarter']))
    (runs['scaled'] / 'training_audits/epoch=059.json').unlink()
    with pytest.raises(compare.EvidenceError, match='无法读取'):
        compare.compare_runs(**runs)


def test_scaled_candidate_can_win_and_lock_its_seed52_recipe(runs, tmp_path):
    old.replace_evaluation(runs['scaled'], 'W', dynamic_iou=.95, static_iou=.95,
                           dynamic_distance=.1, static_distance=.1)
    add_local_rows(runs['scaled'])
    report = compare.compare_runs(**add_stage2(runs, tmp_path, recipe='scaled'))
    assert report['selected'] == 'scaled' and report['selected_lr'] == 1.5e-4
    assert report['all_candidate_rank'][0] == 'scaled'
    assert report['stage2']['locked_recipe'] == 'scaled'
    assert report['stage2']['locked_lr'] == 1.5e-4
    assert report['reproducibility_confirmed']


def test_cli_requires_fourth_candidate(runs, capsys):
    argv = [s for name, path in runs.items() if name != 'scaled'
            for s in ('--' + name.replace('_', '-'), str(path))]
    with pytest.raises(SystemExit) as caught:
        compare.main(argv)
    assert caught.value.code == 2
    assert '--scaled' in capsys.readouterr().err


def test_fixed_seed42_and_initialization_policy_are_checked(runs):
    path = runs['quarter'] / 'training_audits/epoch=003.json'
    value = old.read(path)
    value['sampler']['initial_seed_policy'] = 'wrong'
    old.seal_sampler(value['sampler'])
    old.write(path, value)
    with pytest.raises(compare.EvidenceError, match='initial_seed_policy'):
        compare.compare_runs(**runs)


def test_stage2_locks_lr_seed_and_same_source(runs, tmp_path):
    both = add_stage2(runs, tmp_path)
    report = compare.compare_runs(**both)
    assert report['stage2']['locked_lr'] == 2.5e-5
    assert report['stage2']['explanatory_slices']['selected_seed52']['raw0_2']['n'] == 4
    assert report['stage2']['fixed_registration']['reference_seed'] == 42
    assert report['stage2']['fixed_registration']['passed']
    assert report['reproducibility_confirmed']
    assert not report['full_planning_paused']
    seal_run(both['selected_seed52'], config35('half', 52), v35=True)
    with pytest.raises(compare.EvidenceError, match='lr'):
        compare.compare_runs(**both)


def test_seed52_reverse_result_pauses_full_without_automatic_retraining(runs, tmp_path):
    both = add_stage2(runs, tmp_path)
    old.replace_evaluation(both['selected_seed52'], 'W', dynamic_iou=.1, static_iou=.1,
                           dynamic_distance=3., static_distance=3.)
    add_local_rows(both['selected_seed52'])
    report = compare.compare_runs(**both)
    assert report['status'] == 'passed'  # seed42原评分门不被改写。
    assert not report['reproducibility_confirmed'] and report['full_planning_paused']
    assert report['automatic_retraining'] is False


def test_tail_counts_are_separate_from_scores():
    run = dict(rows={e: {('t', 0): dict(distance=0), ('t', 1): dict(distance=d)}
                    for e, d in zip(compare.EPOCHS, (11, 2, 15))})
    assert compare.tail_counts(run) == dict(epochs={'58': 1, '59': 0, '60': 1}, final60=1, late3=2 / 3)
    diagnostics = compare.tail_diagnostics(run)
    assert diagnostics['gt10m']['final60'] == dict(frames=1, tracklets=1)
    assert diagnostics['gt5m']['late3'] == dict(frames=2 / 3, tracklets=2 / 3)


def test_tail_worsening_pauses_full_without_changing_score_gates(runs, tmp_path):
    # 一个>10m尾部可与较高整体S/P并存；它暂停Full但不得替换得分赢家。
    from utils.tracking_metrics import metric_contributions
    root = runs['quarter']
    old.replace_evaluation(root, 'W', dynamic_iou=.99, static_iou=.99,
                           dynamic_distance=.01, static_distance=.01)
    add_local_rows(root)
    for epoch in compare.EPOCHS:
        path = root / f'evaluation/epoch={epoch:03d}/frames.jsonl'
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        rows[6]['distance'] = 11.
        s, p = metric_contributions(rows[6]['iou'], 11.)
        rows[6].update(success=float(s), precision=float(p))
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
        metrics_path = path.parent / 'metrics.json'
        metrics = old.read(metrics_path)
        metrics.update(compare.score(rows))
        old.write(metrics_path, metrics)
    old.write(root / 'results.json', dict(final=metrics, late3=compare.score(rows), checkpoint_epochs=compare.EPOCHS))
    report = compare.compare_runs(**runs)
    assert report['selected'] == 'quarter' and report['status'] == 'passed'
    assert report['tail_risks']['quarter']['worsened']
    assert report['full_planning_paused'] and report['full_pause_reasons'] == ['tail_gt10m_worsened']


@pytest.mark.parametrize('mutation', ['missing_local', 'config_checksum', 'source', 'budget', 'frame_score'])
def test_tampered_evidence_is_rejected(runs, mutation):
    root = runs['half']
    if mutation in ('missing_local', 'frame_score'):
        old.mutate_frame(root, lambda rows: rows[1].pop('diagnostic_local_mean_support')
                         if mutation == 'missing_local' else rows[1].update(success=.123))
    else:
        path = root / ('training_budget.json' if mutation == 'budget' else 'run_manifest.json')
        value = old.read(path)
        if mutation == 'config_checksum':
            value['config_sha256'] = '0' * 64
        elif mutation == 'source':
            value['source']['files']['synthetic.py'] = 'a' * 64
            value['source']['sha256'] = compare.digest(value['source']['files'])
        else:
            value['optimizer_steps'] -= 1
        old.write(path, value)
    with pytest.raises(compare.EvidenceError):
        compare.compare_runs(**runs)


def test_output_policy_and_stdout(runs, tmp_path, capsys):
    argv = [s for name, path in runs.items() for s in ('--' + name.replace('_', '-'), str(path))]
    assert compare.main(argv) == 0
    assert json.loads(capsys.readouterr().out)['selected'] == 'quarter'
    # pytest临时目录也可位于ct_checks内；显式选择禁写范围，避免依赖--basetemp。
    outside = compare.base.ROOT / 'v35_disallowed_report_test.json'
    assert compare.main(argv + ['--output', str(outside)]) == 2
    assert json.loads(capsys.readouterr().out)['status'] == 'invalid_evidence'


def test_weaker_reference_cannot_replace_registered_evidence(runs):
    old.replace_evaluation(runs['registered_reference'], 'reference', dynamic_iou=.1, static_iou=.1,
                           dynamic_distance=3., static_distance=3.)
    with pytest.raises(compare.EvidenceError, match='fingerprint'):
        compare.compare_runs(**runs)


def test_seed52_matched_success_does_not_hide_fixed_reference_failure(runs, tmp_path):
    both = add_stage2(runs, tmp_path)
    # 新参考更弱；候选胜过同seed参考，却未达到原登记R。
    old.replace_evaluation(both['reference_seed52'], 'reference', dynamic_iou=.2, static_iou=.2,
                           dynamic_distance=1.5, static_distance=1.5)
    old.replace_evaluation(both['selected_seed52'], 'W', dynamic_iou=.62, static_iou=.62,
                           dynamic_distance=.49, static_distance=.49)
    add_local_rows(both['selected_seed52'])
    report = compare.compare_runs(**both)
    assert report['stage2']['candidates']['selected_seed52']['passed']
    assert not report['stage2']['fixed_registration']['passed']
    assert not report['reproducibility_confirmed'] and report['full_planning_paused']


def test_self_consistent_seed52_reference_protocol_change_is_rejected(runs, tmp_path):
    both = add_stage2(runs, tmp_path)
    root = both['reference_seed52']
    path = root / 'run_manifest.json'
    manifest = old.read(path)
    manifest['reference_protocol']['implementation_sha'] = 'a' * 64
    config = yaml.safe_load((root / 'resolved_config.yaml').read_text(encoding='utf-8'))
    manifest['config_sha256'] = compare.config_digest(config, manifest)
    old.write(path, manifest)
    with pytest.raises(compare.EvidenceError, match='reference_protocol'):
        compare.compare_runs(**both)
