"""完整联合链比较器：原帧门槛、旧 B0 冻结、候选证据及六组配置身份。"""
from copy import deepcopy
import hashlib
import json
import math
import shutil

import pytest

from models.ct_v31.config import load_config
from tools import compare_v35_full as compare
from tests import test_ct_v35_comparison as v35
from tests import test_ct_v34_comparison as old


def add_candidates(root, arm):
    for epoch in compare.original.EPOCHS:
        path = root / f'evaluation/epoch={epoch:03d}/frames.jsonl'
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        for row in rows:
            if row['initialization']:
                continue
            valid = [True] + [arm == 'full'] * 3
            quality = [.8, .7, .6, .5]
            row['diagnostic_candidates'] = dict(schema=compare.CANDIDATE_SCHEMA,
                boxes_anchor_relative=[[0.] * 4 for _ in range(4)],
                boxes_world=[[0.] * 4 for _ in range(4)],
                quality_logits=[math.log(q / (1 - q)) for q in quality], quality=quality,
                valid=valid, geometry=[deepcopy(row['fine_geometry']) if yes else None for yes in valid],
                selected_index=0, selected_quality=.8,
                extension=dict(capacity=256, slots=[], raw_ids=[], acquisition_indices=[],
                    xyz_anchor_relative=[], identity_logits=[], identity_support=[],
                    vote_xyz_anchor_relative=[], reliability_logits=[], target_labels=[], partition=[]),
                mode_formed=valid[1:], prior=dict(box_anchor_relative=[0.] * 4, box_world=[0.] * 4),
                acquisition=dict(valid_count=0, target_count=0),
                modes=[dict(candidate_valid=yes, formed=yes, member_slots=[], member_raw_ids=[],
                    member_count=0, target_count=0, target_purity=None) for yes in valid[1:]])
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')


@pytest.fixture
def runs(tmp_path, monkeypatch):
    existing = v35.runs.__wrapped__(tmp_path, monkeypatch)
    kwargs = {key: existing[key] for key in ('registered_reference', 'baseline', 'old_b0', 'old_w_quarter')}
    source = dict(files={'synthetic.py': 'd' * 64})
    source['sha256'] = compare.original.digest(source['files'])
    registration = dict(schema=compare.REGISTRATION_SCHEMA, source=source, runs=[], reused_b0={})
    for recipe in compare.LRS:
        root = existing[recipe]
        kwargs[recipe + '_b0'] = root
        manifest = old.read(root / 'run_manifest.json')
        registration['reused_b0'][recipe] = dict(config_sha256=manifest['config_sha256'],
            manifest_source=manifest['source'], effective_source=manifest['source'],
            resume_sha256={},
            frames_sha256={str(e): hashlib.sha256((root / f'evaluation/epoch={e:03d}/frames.jsonl').read_bytes()).hexdigest()
                           for e in compare.original.EPOCHS})
        for arm in compare.ARMS:
            label = recipe + '_' + arm
            target = tmp_path / label
            shutil.copytree(root, target)
            config = load_config(f'cfgs/ct_seqtrack/35_{arm}_w_{recipe}_lr_mini.yaml')
            v35.seal_run(target, config, v35=True)
            manifest = old.read(target / 'run_manifest.json')
            manifest.update(source=source, arm=arm, temporal_backend='cfc', calibration_required=False,
                enabled=dict(B1=True, B2=arm in ('b1_b2', 'full'), B3=arm == 'full'))
            old.write(target / 'run_manifest.json', manifest)
            add_candidates(target, arm)
            registration['runs'].append(dict(label=label, arm=arm, lr=compare.LRS[recipe],
                config_sha256=manifest['config_sha256']))
            kwargs[label] = target
    monkeypatch.setattr(compare, 'load_registration', lambda: deepcopy(registration))
    historical = registration['reused_b0']['normal']['manifest_source']
    monkeypatch.setattr(compare.piecewise, 'load_source_registration', lambda: dict(before=historical, after=historical))
    return kwargs


def test_six_runs_preserve_old_gates_same_lr_chain_and_compare_strong_b0(runs):
    before = {str(p): (p.stat().st_size, p.stat().st_mtime_ns)
              for root in runs.values() for p in root.rglob('*') if p.is_file()}
    report = compare.compare_runs(**runs)
    assert report['status'] == 'passed'
    assert report['selected_full'] == 'normal_full'  # 相同分数选较低 LR。
    assert set(report['candidates']) == set(compare.LABELS)
    assert report['full_vs_strong_b0']['scaled_full']['all_four_nonnegative']
    assert report['candidate_diagnostics']['scaled_full']['epochs']['60']['eligible_mode_frames'] == 8
    assert report['candidate_diagnostics']['scaled_b1_b2']['epochs']['60']['eligible_mode_frames'] == 0
    assert report['candidate_diagnostics']['normal_full']['epochs']['60']['candidate_record_coverage'] == 1.
    assert not report['automatic_retraining']
    after = {str(p): (p.stat().st_size, p.stat().st_mtime_ns)
             for root in runs.values() for p in root.rglob('*') if p.is_file()}
    assert before == after


def test_point_level_candidate_objects_released_only_after_validation(runs):
    registration = compare.load_registration()
    spec = next(value for value in registration['runs'] if value['label'] == 'scaled_full')
    run = compare.read_run(runs['scaled_full'], spec, registration['source'])
    for rows in run['rows'].values():
        for key, row in rows.items():
            if key[1]:
                assert set(row['diagnostic_candidates']) == {'schema', 'valid', 'geometry', 'selected_index'}
    # 原始台账保留完整，不能通过删去未保留在内存中的点级字段躲过验证。
    def remove_field(rows):
        rows[1]['diagnostic_candidates']['extension'].pop('target_labels')
    old.mutate_frame(runs['scaled_full'], remove_field)
    with pytest.raises(compare.EvidenceError, match='扩展字段缺失'):
        compare.read_run(runs['scaled_full'], spec, registration['source'])


@pytest.mark.parametrize('mutation', ['source', 'budget', 'missing_candidates', 'selected_invalid', 'wrong_geometry', 'old_b0'])
def test_tampering_cannot_silently_change_registered_question(runs, mutation):
    root = runs['scaled_full']
    if mutation == 'budget':
        (root / 'training_audits/epoch=059.json').unlink()
    elif mutation == 'source':
        path = root / 'run_manifest.json'
        value = old.read(path)
        value['source']['files']['synthetic.py'] = 'f' * 64
        value['source']['sha256'] = compare.original.digest(value['source']['files'])
        old.write(path, value)
    elif mutation == 'old_b0':
        old.replace_evaluation(runs['scaled_b0'], 'W', dynamic_iou=.6)
        v35.add_local_rows(runs['scaled_b0'])
    else:
        def alter(rows):
            if mutation == 'missing_candidates':
                rows[1].pop('diagnostic_candidates')
            elif mutation == 'selected_invalid':
                rows[1]['diagnostic_candidates']['valid'][0] = False
            else:
                rows[1]['diagnostic_candidates']['geometry'][0]['iou'] = .01
        old.mutate_frame(root, alter)
    with pytest.raises(compare.EvidenceError):
        compare.compare_runs(**runs)


def test_missing_or_duplicate_group_rejected(runs):
    with pytest.raises(compare.EvidenceError, match='六组'):
        compare.compare_runs(**dict(runs, normal_full=runs['scaled_full']))
    incomplete = dict(runs)
    incomplete.pop('normal_full')
    with pytest.raises(compare.EvidenceError, match='六组'):
        compare.compare_runs(**incomplete)


def test_full_gate_failure_not_hidden_by_passing_b1(runs):
    for label in ('normal_full', 'scaled_full'):
        old.replace_evaluation(runs[label], 'W', dynamic_iou=.1, static_iou=.1,
            dynamic_distance=3., static_distance=3.)
        v35.add_local_rows(runs[label])
        add_candidates(runs[label], 'full')
    report = compare.compare_runs(**runs)
    assert report['candidates']['scaled_b1']['passed']
    assert report['status'] == 'failed' and report['selected_full'] is None
    assert not report['full_vs_strong_b0']['normal_full']['all_four_nonnegative']


def test_full_can_select_mode_while_b1_b2_cannot(runs):
    def select_mode(rows):
        row = rows[1]
        row.update(selected_index=1, selected_quality=.9)
        candidate = row['diagnostic_candidates']
        candidate.update(selected_index=1, selected_quality=.9)
        candidate['quality'][1] = .9
        candidate['quality_logits'][1] = math.log(9.)
    old.mutate_frame(runs['scaled_full'], select_mode)
    report = compare.compare_runs(**runs)
    assert report['status'] == 'passed'
    old.mutate_frame(runs['scaled_b1_b2'], select_mode)
    with pytest.raises(compare.EvidenceError, match='未开启 B3'):
        compare.compare_runs(**runs)


def test_complete_candidate_permission_and_oracle_not_deployable_score():
    row = old.make_rows('W')[1]
    # 已有 fixture 的序列化合同另覆盖全流程；这里特别检查 mode 原始 ID 权限。
    record = dict(schema=compare.CANDIDATE_SCHEMA, valid=[True, False, False, False],
        boxes_anchor_relative=[[0.] * 4 for _ in range(4)], boxes_world=[[0.] * 4 for _ in range(4)],
        quality=[.8, .7, .6, .5], quality_logits=[math.log(q / (1 - q)) for q in (.8, .7, .6, .5)],
        selected_index=0, selected_quality=.8, geometry=[row['fine_geometry'], None, None, None],
        extension=dict(capacity=256, slots=[3], raw_ids=[97], acquisition_indices=[2],
            xyz_anchor_relative=[[0., 0., 0.]], identity_logits=[0.], identity_support=[.5],
            vote_xyz_anchor_relative=[[0., 0., 0.]], reliability_logits=[0.], target_labels=[0.], partition=[0]),
        mode_formed=[False] * 3,
        modes=[dict(candidate_valid=False, formed=False, member_slots=[3], member_raw_ids=[98],
                    member_count=1, target_count=0, target_purity=0.)] * 3)
    row['diagnostic_candidates'] = record
    with pytest.raises(compare.EvidenceError, match='成员映射'):
        compare.validate_candidates(row, 'example')


@pytest.mark.parametrize('separator', ['/', '\\'])
def test_legacy_resume_fingerprint_paths_portable(separator):
    provenance = {'run_manifest.json': 'a' * 64,
        'resume_manifests' + separator + 'resume.json': 'b' * 64}
    assert compare.resume_fingerprints(provenance) == {'resume_manifests/resume.json': 'b' * 64}
