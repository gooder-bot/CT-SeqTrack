"""临时目录中三组完整登记/60轮审计/原帧正向验收；不读正式结果。"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from models.ct_v38.data import ReadyQueueBatchSampler
from models.ct_v38.config import load_config, config_identity, recipe_identity
from models.ct_v38.runtime import reliability_summary
from tools import compare_v38 as compare
from utils.execution_profile import execution_identity
from tests.helpers import formal_run_factory as old
from tests.test_compare_v38 import record


def new_evaluation(root, arm, *, failing=False):
    rows = old.make_rows('W', **(dict(dynamic_iou=.01, static_iou=.01,
        dynamic_distance=3., static_distance=3.) if failing else {}))
    for row in rows:
        if row['initialization']:
            continue
        template = record(arm=arm)
        candidate = template.pop('diagnostic_candidates')
        count = min(row['diagnostic_target_count'], 3)
        candidate['support'] = dict(raw_ids=list(range(count)), count=count,
            target_count=count, target_purity=1. if count else 0.)
        candidate['geometry'] = [dict(exact_iou=row['iou'],
            success=row['success'], precision=row['precision']) for _ in range(3)]
        candidate['trusted_target'] = row['iou'] >= .5 and count >= 3
        candidate['trusted_eligible'] = count >= 3
        template.update(success=row['success'], precision=row['precision'],
            geometric_fg_count=count, supported=count >= 3, trusted_write=count >= 3)
        row.pop('memory_write', None)
        row.pop('memory_wrong_write', None)
        row.update(template, strong=count >= 3, diagnostic_candidates=candidate)
    reliability = reliability_summary([row['diagnostic_candidates'] for row in rows if not row['initialization']])
    for epoch in compare.original.EPOCHS:
        directory = root / f'evaluation/epoch={epoch:03d}'
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'frames.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
        metrics = dict(schema=compare.SCHEMA, checkpoint_epoch=epoch, metric_mode='benchmark_compat',
            complete_coverage=True, **compare.original.COUNTS, **compare.original.score(rows),
            trusted_reliability=reliability)
        old.write(directory / 'metrics.json', metrics)
    old.write(root / 'results.json', dict(final=metrics, late3=compare.original.score(rows),
        checkpoint_epochs=compare.original.EPOCHS, policy_activation='active_from_epoch1'))


@pytest.fixture
def complete_runs(tmp_path, monkeypatch):
    legacy = old.runs(tmp_path, monkeypatch)
    actual_root = compare.ROOT
    monkeypatch.setattr(compare, 'ROOT', tmp_path)
    monkeypatch.setattr(compare.original, 'COUNTS', old.compare.COUNTS)
    hashes = {label: tuple(hashlib.sha256((legacy[label] / f'evaluation/epoch={epoch:03d}/frames.jsonl').read_bytes()).hexdigest()
        for epoch in compare.original.EPOCHS) for label in ('reference', 'baseline')}
    monkeypatch.setattr(compare.original, 'REGISTERED_FRAME_SHA256', hashes)
    files = {}
    for relative in ('models/ct_v38/data.py', 'utils/tracking_metrics.py'):
        data = (actual_root / relative).read_bytes()
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        files[relative] = hashlib.sha256(data).hexdigest()
    source = dict(files=files, sha256=compare.original.digest(files))
    specs, configs = [], {}
    for label in compare.LABELS:
        cfg = load_config(actual_root / f'cfgs/ct_seqtrack/38_{label}_mini.yaml')
        configs[label] = cfg
        specs.append(dict(label=label, arm=cfg.v38_arm, config_sha256=config_identity(cfg),
            execution_profile='optimized_fp32', training_execution_recipe=recipe_identity(cfg)))
    registration = dict(schema=compare.REGISTRATION_SCHEMA, source=source, runs=specs,
        registered_main_model='full_standard', policy_activation='active_from_epoch1',
        reference_frames_sha256={name: list(values) for name, values in hashes.items()})
    registration_path = tmp_path / 'cfgs/ct_seqtrack/38_clean1_registration.json'
    old.write(registration_path, registration)
    registration_hash = hashlib.sha256(registration_path.read_bytes()).hexdigest()
    sampler = ReadyQueueBatchSampler([4778], batch_size=16, short_window=4,
        source_sha256='a' * 64, initial_seed_policy='initial_exact_other_perturbed_v1')
    plans = {}
    for epoch in range(1, 61):
        sampler.set_epoch(epoch - 1)
        plans[epoch] = sampler.state_dict()
    paths = dict(registered_reference=legacy['reference'], baseline=legacy['baseline'])
    for index, spec in enumerate(specs):
        label, cfg = spec['label'], configs[spec['label']]
        root = tmp_path / label
        paths[label] = root
        run_id = str(index + 1) * 32
        manifest = dict(schema=compare.SCHEMA, model='ctseqtrackv38', seed=42, arm=cfg.v38_arm,
            temporal_backend='cfc', evaluation_rng_policy='per_checkpoint_seed_v1',
            policy_activation='active_from_epoch1', enabled=dict(B1=True, B2=False, B3=cfg.v38_arm == 'full'),
            source=source, calibration_required=False, run_id=run_id, config_sha256=config_identity(cfg),
            registration=dict(schema=compare.REGISTRATION_SCHEMA, label=label, sha256=registration_hash),
            training_execution_recipe=recipe_identity(cfg), execution_identity=execution_identity(cfg, source),
            common_parameter_hash='a' * 64, prior_parameter_hash='b' * 64,
            base_initial_parameter_hash='c' * 64, initial_parameter_hash=('d' if index < 2 else 'e') * 64)
        old.write(root / 'run_manifest.json', manifest)
        (root / 'resolved_config.yaml').write_text(yaml.safe_dump(dict(cfg)), encoding='utf-8')
        for epoch in range(1, 61):
            old.write(root / f'training_audits/epoch={epoch:03d}.json', dict(
                schema='ct_seqtrack.v38_clean1.training_audit.v1', model_schema=compare.SCHEMA, run_id=run_id,
                config_sha256=config_identity(cfg), completed_epoch=epoch, rows=19108, optimizer_steps=1195,
                epoch_complete=True, sampler=plans[epoch]))
        old.write(root / 'training_budget.json', dict(schema=compare.SCHEMA, **compare.original.BUDGET,
            epoch_complete=True, sampler=plans[60]))
        new_evaluation(root, cfg.v38_arm)
    return paths


def test_complete_comparison_reconstructs_three_budgets_and_keeps_fixed_full(complete_runs):
    before = {str(path): (path.stat().st_size, path.stat().st_mtime_ns)
        for root in complete_runs.values() for path in root.rglob('*') if path.is_file()}
    report = compare.compare_runs(**complete_runs)
    assert report['status'] == 'passed' and report['selected'] == 'full_standard'
    assert set(report['candidates']) == set(compare.LABELS) and 'passing_rank' not in report
    assert set(report['raw_contributions']) == {'full_minus_b1', 'hold_minus_standard'}
    assert report['mechanism_diagnostics']['full_standard']['60']['trusted_reliability']['count'] == 8
    assert before == {str(path): (path.stat().st_size, path.stat().st_mtime_ns)
        for root in complete_runs.values() for path in root.rglob('*') if path.is_file()}
    # 完整证据但 Full 未过门，不能用通过的 B1/hold 取代它。
    new_evaluation(complete_runs['full_standard'], 'full', failing=True)
    failed = compare.compare_runs(**complete_runs)
    assert failed['status'] == 'failed' and failed['selected'] is None
    assert failed['candidates']['b1_hold']['passed']
    # 当前源码/数据登记完整也不能豁免少一轮审计。
    audit_path = complete_runs['b1_hold'] / 'training_audits/epoch=059.json'
    value = old.read(audit_path)
    value['optimizer_steps'] = 1194
    old.write(audit_path, value)
    with pytest.raises(compare.EvidenceError, match='optimizer_steps'):
        compare.compare_runs(**complete_runs)
