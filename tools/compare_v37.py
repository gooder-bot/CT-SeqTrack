"""v37 五组只读原帧验收：固定 Full FP32 主模型，final60/late-3，严禁挑 best。"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ''):
    sys.path.insert(0, str(ROOT))

from tools import compare_v35_b0 as original
from tools import compare_v35_full as previous
from models.ct_v37.config import config_identity, normalize_config
from models.ct_v37.identity import SCHEMA
from models.ct_v31.data import ReadyQueueBatchSampler
from utils.execution_profile import validate_execution_identity

base = original.base
EvidenceError, require = original.EvidenceError, original.require
LABELS = ('b0', 'b1', 'b1_b2', 'full', 'full_bf16')
SOURCES = ['q0', 'prior', 'mode0', 'mode1']
REPORT_SCHEMA = 'ct_seqtrack.v37.comparison.v1'
REGISTRATION_SCHEMA = 'ct_seqtrack.v37.registration.v1'
CANDIDATE_SCHEMA = 'ct_seqtrack.v37.candidate_records.v1'


def load_registration():
    value = original.read_object(ROOT / 'cfgs/ct_seqtrack/37_registration.json')
    require(value.get('schema') == REGISTRATION_SCHEMA, 'v37 registration schema mismatch')
    source = value.get('source', {})
    require(isinstance(source.get('files'), dict) and source['files'], 'missing registered source')
    require(source.get('sha256') == original.digest(source['files']), 'source checksum mismatch')
    for path, checksum in source['files'].items():
        original._sha256(checksum, 'source.' + path)
    # 解释旧记录时，排程重建代码必须与登记相同。
    path = 'models/ct_v31/data.py'
    require(source['files'].get(path) == hashlib.sha256((ROOT / path).read_bytes()).hexdigest(),
            'registered sampler reconstruction source differs')
    runs = value.get('runs', [])
    require(len(runs) == 5 and {r.get('label') for r in runs} == set(LABELS), 'five registered runs required')
    for run in runs:
        label = run['label']
        require(run.get('arm') == ('full' if label == 'full_bf16' else label), 'registered arm mismatch')
        require(run.get('execution_profile') == ('optimized_bf16' if label == 'full_bf16' else 'optimized_fp32'),
                'registered precision mismatch')
        original._sha256(run.get('config_sha256'), label + '.config_sha256')
    require(set(value.get('reused', {})) == {'strong_b0', 'strong_b1'}, 'fixed strong B0/B1 evidence required')
    return value


@lru_cache(maxsize=64)
def _rebuilt_sampler(payload):
    value = json.loads(payload)
    sampler = ReadyQueueBatchSampler(**{key: value[key] for key in (
        'lengths', 'batch_size', 'seed', 'training', 'short_window', 'long_window',
        'curriculum_epochs', 'source_sha256', 'reserve_windows', 'seed_translation',
        'seed_yaw_degrees', 'initial_seed_policy')})
    sampler.set_epoch(value['epoch'])
    return sampler.state_dict()


def validate_sampler(value, config, epoch, label):
    source = original._sampler(value, config, epoch, label)
    require(value.get('initial_seed_policy') == 'initial_exact_other_perturbed_v1', label + ': initial seed mismatch')
    lengths = value.get('lengths')
    require(isinstance(lengths, list) and lengths and all(type(n) is int and n >= 0 for n in lengths),
            label + ': invalid track lengths')
    require(value == _rebuilt_sampler(json.dumps(value, sort_keys=True, separators=(',', ':'))),
            label + ': sampler does not reconstruct')
    return source


def _array(value, shape, label):
    require(isinstance(value, list), label + ': list required')
    result = np.asarray(value, dtype=np.float64)
    require(result.shape == shape and np.isfinite(result).all(), label + ': nonfinite or invalid shape')
    return result


def _ids(value, label):
    require(isinstance(value, list) and all(type(x) is int and x >= 0 for x in value)
            and len(set(value)) == len(value), label + ': unique nonnegative raw IDs required')
    return set(value)


def validate_candidates(row, label, arm):
    """只依据最终框和当前授权点复算选择/支持；GT 只核对被动审计。"""
    record = row.get('diagnostic_candidates')
    require(isinstance(record, dict) and record.get('schema') == CANDIDATE_SCHEMA, label + ': missing candidates')
    require(record.get('candidate_order') == SOURCES, label + ': candidate order mismatch')
    valid = record.get('valid')
    require(isinstance(valid, list) and len(valid) == 4 and all(type(x) is bool for x in valid), label + ': invalid validity')
    boxes = _array(record.get('boxes_anchor_relative'), (4, 4), label + '.boxes')
    world = _array(record.get('boxes_world'), (4, 4), label + '.world')
    anchor = _array(record.get('anchor_world'), (4,), label + '.anchor')
    size = _array(record.get('first_frame_size_lwh'), (3,), label + '.size')
    require(bool((size > 0).all()), label + ': nonpositive size')
    require(np.allclose(world[:, :3], boxes[:, :3] + anchor[:3], rtol=0, atol=1e-6)
            and np.allclose(world[:, 3], boxes[:, 3], rtol=0, atol=1e-6), label + ': coordinates mismatch')
    geometry = record.get('geometry')
    require(isinstance(geometry, list) and len(geometry) == 4 and any(valid), label + ': geometry shape')
    for i, good in enumerate(valid):
        require(isinstance(geometry[i], dict) if good else geometry[i] is None, label + ': geometry/valid mismatch')
        if good:
            require(0 <= base.number(geometry[i].get('exact_iou'), label + '.iou') <= 1, label + ': invalid IoU')
    selected = record.get('selected_index')
    shadow_fallback = arm == 'b1_b2' and selected == 0 and record.get('q0_is_fallback') is True
    require(type(selected) is int and 0 <= selected < 4 and (valid[selected] or shadow_fallback), label + ': invalid selected index')
    require(row.get('selected_index') == selected, label + ': selected index differs from row')
    selected_geometry = record.get('selected_geometry')
    require(isinstance(selected_geometry, dict), label + ': missing selected geometry')
    original.close(row.get('distance'), selected_geometry['center_error_m'], label + '.selected distance', 2e-4)
    original.close(row.get('selected_quality'), record.get('selected_commit_quality'), label + '.selected quality', 2e-6)
    oracle = max((i for i in range(4) if valid[i]), key=lambda i: (geometry[i]['success'], geometry[i]['precision']))
    require(record.get('oracle', {}).get('index') == oracle, label + ': oracle mismatch')
    ext = record.get('extension', {})
    ext_ids = _ids(ext.get('raw_ids'), label + '.extension')
    n = len(ext_ids)
    slots = ext.get('slots')
    _ids(slots, label + '.slots')
    require(len(slots) == n, label + ': selected extension shape')
    for key, width in (('xyz_anchor_relative', 3), ('vote_xyz_anchor_relative', 3)):
        # Empty point sets are serialized as [], with no invented vectors.
        if n:
            _array(ext.get(key), (n, width), label + '.' + key)
        else:
            require(ext.get(key) == [], label + ': empty extension shape')
    modes = record.get('modes')
    require(isinstance(modes, list) and len(modes) == 2, label + ': modes missing')
    memberships = []
    for mode in modes:
        members = _ids(mode.get('member_raw_ids'), label + '.members')
        require(members <= ext_ids and mode.get('member_count') == len(members), label + ': members outside extension')
        memberships.append(members)
    require(not memberships[0] & memberships[1], label + ': duplicate mode raw IDs')
    if valid[2] and valid[3]:
        require(float(np.linalg.norm(boxes[2, :2] - boxes[3, :2])) > .75 - 2e-6,
                label + ': final mode centers not deduplicated')
    if arm in ('b0', 'b1'):
        require(selected == 0 and not any(valid[2:]) and record.get('utility') is None,
                label + ': baseline must not execute candidate selection')
        return record
    utility = _array(record.get('utility'), (4,), label + '.utility')
    logits = _array(record.get('verification_logits'), (4, 3), label + '.logits')
    expected_utility = .5 * (1 / (1 + np.exp(-np.clip(logits[:, 0], -80, 80)))
                             + 1 / (1 + np.exp(-np.clip(logits[:, 1], -80, 80))))
    require(np.allclose(utility, expected_utility, atol=2e-6, rtol=0), label + ': S/P utility mismatch')
    expected_selected = max((i for i in range(4) if valid[i]), key=lambda i: utility[i]) if arm == 'full' else 0
    require(selected == expected_selected, label + ': selection/shadow policy mismatch')
    qualities = _array(record.get('commit_quality'), (4,), label + '.quality')
    require(bool(((qualities >= 0) & (qualities <= 1)).all()), label + ': quality out of range')
    original.close(record.get('selected_commit_quality'), qualities[selected], label + '.commit quality', 2e-6)
    require(qualities[1] == 0, label + ': prior cannot carry commit quality')
    expected_quality = np.where(np.asarray(valid[2:]), 1 / (1 + np.exp(-np.clip(logits[2:, 2], -80, 80))), 0.)
    require(np.allclose(qualities[2:], expected_quality, rtol=0, atol=2e-6),
            label + ': mode commit quality must use Qmode')
    support = record.get('final_support')
    require(isinstance(support, list) and len(support) == 4, label + ': final support missing')
    pool = record.get('support_pool', {})
    pool_ids = pool.get('raw_ids')
    require(isinstance(pool_ids, list) and all(type(x) is int and x >= -1 for x in pool_ids), label + ': support IDs')
    xyz = _array(pool.get('xyz_anchor_relative'), (len(pool_ids), 3), label + '.support xyz')
    probabilities = _array(pool.get('probabilities'), (len(pool_ids),), label + '.support probability')
    targets = _array(pool.get('target_labels'), (len(pool_ids),), label + '.support targets')
    sources = _array(pool.get('source'), (len(pool_ids),), label + '.support source')
    require(bool(((sources == 0) | (sources == 1)).all()), label + ': invalid support source')
    require(bool(((probabilities >= 0) & (probabilities <= 1)).all()), label + ': probability range')
    lookup = {}
    for i, raw in enumerate(pool_ids):
        if raw >= 0 and (raw not in lookup or probabilities[i] > probabilities[lookup[raw]]):
            lookup[raw] = i
    for k, value in enumerate(support):
        ids = _ids(value.get('raw_ids'), label + '.support')
        authorized = _ids(value.get('authorized_raw_ids'), label + '.authorized')
        require(ids <= authorized <= set(lookup) and value.get('count') == len(ids), label + ': unauthorized support')
        true_count = 0
        for raw in authorized:
            slot = lookup[raw]
            delta = xyz[slot] - boxes[k, :3]
            c, s = math.cos(boxes[k, 3]), math.sin(boxes[k, 3])
            local = np.array([c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1], delta[2]])
            margin = size / 2 - np.abs(local)
            if raw in ids:
                require(probabilities[slot] >= .5 and bool((margin >= -2e-5).all()), label + ': support outside own final box')
                true_count += int(targets[slot] > .5)
            elif probabilities[slot] >= .5 and bool((margin > 2e-5).all()):
                raise EvidenceError(label + ': omitted interior foreground support')
        require(value.get('target_count') == true_count, label + ': support truth mismatch')
        if ids:
            original.close(value.get('target_purity'), true_count / len(ids), label + '.purity', 1e-6)
        else:
            require(value.get('target_purity') is None, label + ': empty support purity must be null')
    # q0评分可读全部selected256；提交仍只允许B0。mode允许B0+自身成员。
    commit_support = record.get('commit_support')
    require(isinstance(commit_support, list) and len(commit_support) == 4, label + ': missing commit support')
    commit_sets = []
    for k, value in enumerate(commit_support):
        expected = set(support[k]['raw_ids'])
        if k == 0:
            expected = {raw for raw in expected if sources[lookup[raw]] == 0}
        if k == 1 or (k == 0 and record.get('q0_is_fallback')):
            expected = set()
        actual = _ids(value.get('raw_ids'), label + '.commit support')
        require(actual == expected and value.get('count') == len(expected), label + ': commit support permission mismatch')
        truth_count = sum(int(targets[lookup[raw]] > .5) for raw in expected)
        require(value.get('target_count') == truth_count, label + ': commit truth mismatch')
        commit_sets.append(actual)
    own = commit_sets[selected]
    forbidden = selected == 1 or (selected == 0 and record.get('q0_is_fallback'))
    if forbidden:
        own = set()
    strong = len(own) >= 3
    eligible = strong and qualities[selected] >= .5 and not forbidden
    require(row.get('geometric_fg_count') == len(own) and row.get('strong') is strong,
            label + ': commit own-box foreground mismatch')
    require(row.get('supported') is eligible and row.get('trusted_write') is eligible,
            label + ': trusted support mismatch')
    require(row.get('memory_write') is eligible, label + ': memory eligibility mismatch')
    written = row.get('memory_written_raw_ids')
    flags = row.get('memory_written_foreground')
    written_ids = _ids(written, label + '.written IDs')
    require(isinstance(flags, list) and len(flags) == len(written) and all(type(x) is bool for x in flags),
            label + ': memory flags mismatch')
    foreground_ids = {raw for raw, flag in zip(written, flags) if flag}
    require(foreground_ids <= own and row.get('memory_fg_count') == len(foreground_ids), label + ': memory foreign FG')
    if eligible:
        require(row.get('memory_source_index') == selected and 3 <= len(foreground_ids) <= 8,
                label + ': memory source/count mismatch')
    else:
        require(not written_ids and row.get('memory_source_index') is None, label + ': ineligible write')
    true_written = sum(int(targets[lookup[raw]] > .5) for raw in foreground_ids)
    require(row.get('memory_true_fg_count') == true_written and row.get('memory_wrong_write') is
            (eligible and true_written / max(len(foreground_ids), 1) < .5), label + ': memory truth mismatch')
    return record


def mechanism_summary(rows):
    records = [row['diagnostic_candidates'] for row in rows if row.get('frame_index', row.get('frame', 0)) != 0
               and isinstance(row.get('diagnostic_candidates'), dict)]
    # Collector presence determines coverage; no absent field is counted as zero evidence.
    if not records:
        records = [row['diagnostic_candidates'] for row in rows if isinstance(row.get('diagnostic_candidates'), dict)]
    count = len(records)
    result = dict(prediction_records=count, interpretation='same_state_candidate_chain; not closed_loop_oracle')
    for key in ('good_mode_exists', 'bad_q0_good_mode', 'good_mode_selected', 'recovery_opportunity',
                'qualifying_mode_count', 'good_seed_count', 'seed_good_refinement_good',
                'seed_good_refinement_bad', 'seed_bad_refinement_good'):
        result[key] = sum(record['mechanism'][key] for record in records)
    result['selected_counts'] = {source: sum(record['selected_index'] == i for record in records)
                                 for i, source in enumerate(SOURCES)}
    result['formed_modes'] = sum(mode['formed'] for record in records for mode in record['modes'])
    result['valid_final_modes'] = sum(sum(record['valid'][2:]) for record in records)
    result['recovery_candidate_selection_rate'] = (result['good_mode_selected'] / result['good_mode_exists']
        if result['good_mode_exists'] else None)
    result['good_seed_preservation_rate'] = (result['seed_good_refinement_good'] / result['good_seed_count']
        if result['good_seed_count'] else None)
    result['identity_selected_target_points'] = sum(sum(x > .5 for x in record['extension']['target_labels'])
        for record in records)
    result['acquired_target_points'] = sum(record['acquisition']['target_count'] for record in records)
    result['selected_final_support_target_count'] = sum(record['final_support'][record['selected_index']]['target_count']
        for record in records if record['final_support'])
    result['oracle_success_mean'] = math.fsum(r['oracle']['success'] for r in records) / count if count else None
    for candidate in ('q0', 'prior'):
        result[candidate + '_center_error_mean_m'] = math.fsum(r[candidate + '_geometry']['center_error_m']
            for r in records) / count if count else None
    return result


def _metrics_identity(metrics, epoch, label):
    original.equal_fields(metrics, dict(schema=SCHEMA, checkpoint_epoch=epoch,
        metric_mode='benchmark_compat', **original.COUNTS), label + '.metrics')
    require(metrics.get('complete_coverage') is True, label + ': incomplete coverage')


def read_run(directory, specification, source):
    root, arm, label = Path(directory).resolve(), specification['arm'], specification['label']
    provenance = {}
    def read(relative, yaml_file=False):
        path = root / relative
        value = original.read_object(path, yaml_file=yaml_file)
        provenance[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        return value
    manifest = read('run_manifest.json')
    original.equal_fields(manifest, dict(schema=SCHEMA, model='ctseqtrackv37', seed=42, arm=arm,
        temporal_backend='cfc', evaluation_rng_policy='per_checkpoint_seed_v1'), label + '.manifest')
    require(manifest.get('enabled') == dict(B1=arm != 'b0', B2=arm in ('b1_b2', 'full'), B3=arm == 'full')
            and manifest.get('calibration_required') is False, label + ': enabled/calibration mismatch')
    run_id = manifest.get('run_id')
    require(isinstance(run_id, str) and len(run_id) == 32 and all(c in '0123456789abcdef' for c in run_id), label + ': run ID')
    require(manifest.get('source') == source, label + ': unregistered source')
    registration_fingerprint = hashlib.sha256((ROOT / 'cfgs/ct_seqtrack/37_registration.json').read_bytes()).hexdigest()
    require(manifest.get('registration') == dict(schema=REGISTRATION_SCHEMA, label=label,
            sha256=registration_fingerprint), label + ': run registration fingerprint mismatch')
    for field in ('common_parameter_hash', 'initial_parameter_hash'):
        original._sha256(manifest.get(field), label + '.' + field)
    if arm == 'b0':
        require(manifest.get('prior_parameter_hash') is None, label + ': B0 has no prior parameters')
    else:
        original._sha256(manifest.get('prior_parameter_hash'), label + '.prior_parameter_hash')
    config = read('resolved_config.yaml', True)
    require(dict(normalize_config(config)) == config, label + ': incomplete normalized config')
    require(config['v37_arm'] == arm and config['execution_profile'] == specification['execution_profile']
            and config['test'] is False and config['ct_engineering_check'] is False, label + ': formal identity mismatch')
    require(config_identity(config) == manifest.get('config_sha256') == specification['config_sha256'], label + ': config mismatch')
    validate_execution_identity(manifest.get('execution_identity'), config, source)
    for path in sorted((root / 'resume_manifests').glob('*.json')):
        original.equal_fields(read(path.relative_to(root).as_posix()), dict(schema='ct_seqtrack.v37.resume.v1',
            run_id=run_id, config_sha256=manifest['config_sha256'], source=source), label + '.resume')
    budget = read('training_budget.json')
    original.equal_fields(budget, dict(schema=SCHEMA, **original.BUDGET), label + '.budget')
    require(budget.get('epoch_complete') is True, label + ': epoch60 incomplete')
    training_source = validate_sampler(budget.get('sampler'), config, 60, label + '.sampler')
    require(len(list((root / 'training_audits').glob('epoch=*.json'))) == 60, label + ': sixty audits required')
    plans = {}
    for epoch in range(1, 61):
        audit = read(f'training_audits/epoch={epoch:03d}.json')
        original.equal_fields(audit, dict(schema='ct_seqtrack.v37.training_audit.v1', model_schema=SCHEMA, run_id=run_id,
            config_sha256=manifest['config_sha256'], completed_epoch=epoch, rows=19108, optimizer_steps=1195), label + '.audit')
        require(audit.get('epoch_complete') is True and validate_sampler(audit.get('sampler'), config, epoch, label) == training_source,
                label + ': incomplete audit or changed data')
        plans[epoch] = audit['sampler']['plan_sha256']
    require(audit['sampler'] == budget['sampler'], label + ': budget/audit sampler mismatch')
    results = read('results.json')
    require(results.get('checkpoint_epochs') == original.EPOCHS, label + ': final window must be 58/59/60')
    rows_by_epoch, scores, mechanisms = {}, {}, {}
    for epoch in original.EPOCHS:
        relative = f'evaluation/epoch={epoch:03d}'
        path = root / relative / 'frames.jsonl'
        rows = original.read_frames(path)
        provenance[relative + '/frames.jsonl'] = hashlib.sha256(path.read_bytes()).hexdigest()
        metrics = read(relative + '/metrics.json')
        _metrics_identity(metrics, epoch, label)
        for key, row in rows.items():
            if key[1]:
                where = label + f'.{epoch}.{key}'
                base._check_diagnostics(row, where)
                original._context_diagnostics(row, where, required=True)
                original._local_diagnostics(row, where)
                validate_candidates(row, where, arm)
        mechanisms[str(epoch)] = mechanism_summary(list(rows.values()))
        scores[epoch] = original.score(rows.values())
        for metric in original.METRICS:
            original.close(metrics.get(metric), scores[epoch][metric], label + '.' + metric, 2e-6)
        for row in rows.values():
            row.pop('diagnostic_candidates', None)
        rows_by_epoch[epoch] = rows
    late3 = {m: math.fsum(scores[e][m] for e in original.EPOCHS) / 3 for m in original.METRICS}
    _metrics_identity(results.get('final', {}), 60, label)
    for metric in original.METRICS:
        original.close(results['final'].get(metric), scores[60][metric], label + '.final.' + metric, 2e-6)
        original.close(results.get('late3', {}).get(metric), late3[metric], label + '.late3.' + metric, 2e-6)
    return dict(directory=str(root), config=config, manifest=manifest, source=training_source,
        sampler_plans=plans, rows=rows_by_epoch, scores=scores, late3=late3, provenance=provenance,
        mechanism_diagnostics=mechanisms)


def _pin_reused(run, pin, label):
    require(run['manifest']['config_sha256'] == pin['config_sha256'] and
            run['manifest']['source'] == pin['manifest_source'], label + ': historical identity mismatch')
    actual = {str(e): run['provenance'][f'evaluation/epoch={e:03d}/frames.jsonl'] for e in original.EPOCHS}
    require(actual == pin['frames_sha256'], label + ': original frames mismatch')


def summarize_loaded(runs, registration):
    require(len({runs[x]['manifest']['run_id'] for x in LABELS}) == 5, 'new runs must have distinct run IDs')
    require(all(runs[x]['sampler_plans'] == runs['b0']['sampler_plans'] for x in LABELS), 'new run sampler plans differ')
    require(len({runs[x]['manifest']['common_parameter_hash'] for x in LABELS}) == 1,
            'common B0/q0 initialization differs')
    require(len({runs[x]['manifest']['prior_parameter_hash'] for x in LABELS if x != 'b0'}) == 1,
            'common B1 initialization differs')
    require(runs['full']['manifest']['initial_parameter_hash'] == runs['full_bf16']['manifest']['initial_parameter_hash'],
            'Full FP32/BF16 initial parameters differ')
    comparisons = [(f'{arm}_minus_{ref}', arm, ref) for arm in LABELS
                   for ref in ('reference', 'baseline', 'strong_b0', 'strong_b1')]
    comparisons += [('b1_minus_b0', 'b1', 'b0'), ('b1_b2_minus_b1', 'b1_b2', 'b1'),
                    ('full_minus_b1', 'full', 'b1'), ('full_minus_b1_b2', 'full', 'b1_b2'),
                    ('full_bf16_minus_full', 'full_bf16', 'full')]
    report = base.compare_loaded(runs, candidate_labels=LABELS,
        config_differences=('v37_arm', 'v31_arm', 'execution_profile'), comparisons=comparisons)
    passed = report['candidates']['full']['passed']
    report.update(schema=REPORT_SCHEMA, status='passed' if passed else 'failed',
        registered_main_model='full', selected='full' if passed else None,
        source_sha256=registration['source']['sha256'], reused_registration=registration['reused'],
        registered_frame_sha256=original.REGISTERED_FRAME_SHA256,
        center_tail_diagnostics={label: original.tail_diagnostics(run) for label, run in runs.items()},
        explanatory_slices=original.explanatory_slices(runs, runs['baseline']['rows'][60]),
        mechanism_diagnostics={label: runs[label]['mechanism_diagnostics'] for label in LABELS})
    report.pop('passing_rank', None)
    report['criterion']['ranking'] = 'Full FP32 fixed before training; BF16/control/best epochs never replace it'
    old_tail = original.tail_counts(runs['old_w_quarter'])
    report['tail_risks'] = {}
    for label in LABELS:
        counts = original.tail_counts(runs[label])
        delta = {stage: counts[stage] - old_tail[stage] for stage in ('final60', 'late3')}
        report['tail_risks'][label] = dict(tail_gt10m=counts, old_w_quarter=old_tail,
            delta_frames=delta, worsened=any(x > 0 for x in delta.values()))
    report['risk_review_required'] = report['tail_risks']['full']['worsened']
    report['limitations'].extend(['五组单 seed42；相邻 checkpoint 不是独立重复。',
        'B1+B2 包含完整候选及评分监督，输出 q0；Full 差值包含闭环状态耦合。',
        'BF16 是执行配置对照，不从其分数选择论文主模型。',
        '固定报告 >5m、>10m 及未恢复失跟；漂移风险不改写原 R/C 数值门。',
        '候选 oracle 只表示同状态一步上界，不能代替候选闭环训练和评测。'])
    return report


def compare_runs(*, registered_reference, baseline, old_b0, old_w_quarter, strong_b0, strong_b1, **paths):
    registration = load_registration()
    require(set(paths) == set(LABELS) and len({str(Path(p).resolve()) for p in paths.values()}) == 5,
            'five distinct complete v37 run directories required')
    runs = dict(reference=base.read_run(registered_reference, 'reference'), baseline=base.read_run(baseline, 'baseline'),
        old_b0=base.read_run(old_b0, 'old_b0'), old_w_quarter=base.read_run(old_w_quarter, 'W', expected_lr=2.5e-5))
    original.verify_registered_evidence(runs)
    runs['strong_b0'] = original.read_run(strong_b0, 'v35', expected_lr=1.5e-4)
    old_registration = previous.load_registration()
    b1_spec = next(run for run in old_registration['runs'] if run['label'] == 'scaled_b1')
    runs['strong_b1'] = previous.read_run(strong_b1, b1_spec, old_registration['source'])
    for label in ('strong_b0', 'strong_b1'):
        _pin_reused(runs[label], registration['reused'][label], label)
    specs = {spec['label']: spec for spec in registration['runs']}
    for label in LABELS:
        runs[label] = read_run(paths[label], specs[label], registration['source'])
    return summarize_loaded(runs, registration)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('registered-reference', 'baseline', 'old-b0', 'old-w-quarter', 'strong-b0', 'strong-b1'):
        parser.add_argument('--' + name, required=True, type=Path)
    for label in LABELS:
        parser.add_argument('--' + label.replace('_', '-'), dest=label, required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = vars(parser.parse_args(argv))
    output = args.pop('output')
    try:
        report = compare_runs(**args)
        if output:
            base.write_report(output, report)
        code = 0 if report['status'] == 'passed' else 1
    except (EvidenceError, OSError, ValueError, TypeError, KeyError, StopIteration) as error:
        report, code = dict(schema=REPORT_SCHEMA, status='invalid_evidence', message=str(error)), 2
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return code


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    raise SystemExit(main())
