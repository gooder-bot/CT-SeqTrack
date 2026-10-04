"""v36 五组只读原帧验收；A 是预登记主模型，C/D/E/F 只作消融。

0=A 通过固定 R/C 门，1=A 未通过，2=任一必需证据无效或缺失。
验证最高分仅作描述；正式门只使用 final60 与 58/59/60 均值。
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import compare_v35_b0 as original
from tools import compare_v35_full as previous
from models.ct_v36.config import config_identity, normalize_config
from models.ct_v36.contracts import SCHEMA
from models.ct_v36.sampler import ReadyQueueBatchSampler
from models.ct_v36.diagnostics import mechanism_summary, measurement_followups

base = original.base
EvidenceError, require = original.EvidenceError, original.require
ROOT = Path(__file__).resolve().parents[1]
LABELS = ('A', 'C', 'D', 'E', 'F')
REPORT_SCHEMA = 'ct_seqtrack.v36.comparison.v1'
REGISTRATION_SCHEMA = 'ct_seqtrack.v36.registration.v1'
CANDIDATE_SCHEMA = 'ct_seqtrack.v36.candidate_records.v1'
SOURCES = ['q0', 'prior', 'mode0', 'mode1']


def load_registration():
    value = original.read_object(ROOT / 'cfgs/ct_seqtrack/36_registration.json')
    require(value.get('schema') == REGISTRATION_SCHEMA, 'v36 registration schema mismatch')
    source = value.get('source', {})
    require(isinstance(source.get('files'), dict) and source['files'], 'missing registered source files')
    require(source.get('sha256') == original.digest(source['files']), 'source checksum mismatch')
    for path, checksum in source['files'].items():
        original._sha256(checksum, 'registration.source.' + path)
    # 排程重建必须使用登记的算法，不能由后来更改的排程代码解释旧证据。
    sampler_path = 'models/ct_v36/sampler.py'
    require(source['files'].get(sampler_path) == hashlib.sha256((ROOT / sampler_path).read_bytes()).hexdigest(),
            'registered sampler differs from local reconstruction code')
    runs = value.get('runs', [])
    require(len(runs) == 5 and {r.get('arm') for r in runs} == set(LABELS), 'five registered arms required')
    for run in runs:
        original._sha256(run.get('config_sha256'), run['arm'] + '.config_sha256')
        require(isinstance(run.get('config'), str) and run['config'], 'missing registered config path')
    reused = value.get('reused', {})
    require(set(reused) == {'strong_b0', 'strong_b1'}, 'fixed strong B0/B1 evidence required')
    for label, pin in reused.items():
        original._sha256(pin.get('config_sha256'), label + '.config_sha256')
        require(isinstance(pin.get('manifest_source'), dict), label + ': missing source')
        require(set(pin.get('frames_sha256', {})) == {'58', '59', '60'}, label + ': missing fixed frames')
        for checksum in pin['frames_sha256'].values():
            original._sha256(checksum, label + '.frames')
    return value


@lru_cache(maxsize=240)
def _rebuilt_sampler(payload):
    value = json.loads(payload)
    sampler = ReadyQueueBatchSampler(**{key: value[key] for key in (
        'lengths', 'batch_size', 'seed', 'training', 'short_window', 'long_window',
        'curriculum_epochs', 'source_sha256', 'annotation_sha256', 'reserve_windows',
        'seed_translation', 'seed_yaw_degrees', 'continuous')})
    sampler.set_epoch(value['epoch'])
    return sampler.state_dict()


def validate_sampler(value, config, epoch, label):
    require(isinstance(value, dict), label + ': missing sampler')
    content = {k: v for k, v in value.items() if k != 'manifest_sha256'}
    require(value.get('manifest_sha256') == original.digest(content), label + ': sampler checksum mismatch')
    expected = dict(schema=ReadyQueueBatchSampler.SCHEMA, epoch=epoch - 1, rows=19108,
        batches=1195, seed=42, batch_size=16, short_window=4, long_window=8,
        curriculum_epochs=10, reserve_windows=112, seed_translation=.3, seed_yaw_degrees=1.5)
    original.equal_fields(value, expected, label)
    require(value.get('training') is True and value.get('continuous') is (config['v36_arm'] != 'F'),
            label + ': training/continuous identity mismatch')
    lengths = value.get('lengths')
    require(isinstance(lengths, list) and lengths and all(type(n) is int and n >= 0 for n in lengths),
            label + ': invalid track lengths')
    for field in ('source_sha256', 'annotation_sha256', 'plan_sha256', 'endpoint_order_sha256'):
        original._sha256(value.get(field), label + '.' + field)
    rebuilt = _rebuilt_sampler(json.dumps(content, sort_keys=True, separators=(',', ':')))
    require(value == rebuilt, label + ': sampler plan does not reconstruct from registered algorithm')
    return value['source_sha256']


def _array(value, shape, label):
    require(isinstance(value, list), label + ': list required')
    result = np.asarray(value, dtype=np.float64)
    require(result.shape == shape and np.isfinite(result).all(), label + ': nonfinite or invalid shape')
    return result


def _ids(value, label):
    require(isinstance(value, list) and all(type(x) is int and x >= 0 for x in value)
            and len(set(value)) == len(value), label + ': raw IDs must be unique nonnegative integers')
    return set(value)


def _sigmoid(value):
    return 1 / (1 + math.exp(-value)) if value >= 0 else math.exp(value) / (1 + math.exp(value))


def validate_candidates(row, label, arm):
    """独立复算选择、测量接纳、自身框支持与 memory 写入；GT 不参与排名。"""
    record = row.get('diagnostic_candidates')
    require(isinstance(record, dict) and record.get('schema') == CANDIDATE_SCHEMA, label + ': missing candidate records')
    require(record.get('candidate_order') == SOURCES, label + ': invalid candidate order')
    candidates, action = record.get('candidates'), record.get('action')
    require(isinstance(candidates, list) and len(candidates) == 4 and isinstance(action, dict), label + ': candidate/action shape')
    anchor = _array(record.get('anchor_world'), (4,), label + '.anchor')
    size = _array(record.get('first_frame_size_lwh'), (3,), label + '.size')
    require(bool((size > 0).all()), label + ': nonpositive size')
    boxes = _array(record.get('boxes_anchor_relative'), (4, 4), label + '.boxes')
    world = _array(record.get('boxes_world'), (4, 4), label + '.world')
    require(np.allclose(world[:, :3], boxes[:, :3] + anchor[:3], rtol=0, atol=1e-6)
            and np.allclose(world[:, 3], boxes[:, 3], rtol=0, atol=1e-6), label + ': coordinate contract mismatch')
    pool = record.get('support_pool', {})
    raw_ids = pool.get('raw_ids')
    require(isinstance(raw_ids, list) and all(type(x) is int and x >= -1 for x in raw_ids), label + ': invalid support pool IDs')
    n = len(raw_ids)
    xyz = _array(pool.get('xyz_anchor_relative'), (n, 3), label + '.support_xyz')
    probs = _array(pool.get('probabilities'), (n,), label + '.probabilities')
    targets = _array(pool.get('target_labels'), (n,), label + '.target_labels')
    require(bool(((probs >= 0) & (probs <= 1)).all()), label + ': probability out of range')
    # 无效重复槽概率为零；每个真实 ID 只保留实际有效概率对应槽。
    lookup = {}
    for index, raw in enumerate(raw_ids):
        if raw >= 0 and (raw not in lookup or probs[index] > probs[lookup[raw]]):
            lookup[raw] = index
    supports, eligible, valid = [], [], []
    for index, candidate in enumerate(candidates):
        where = label + '.' + SOURCES[index]
        require(candidate.get('source') == SOURCES[index] and type(candidate.get('valid')) is bool, where + ': source/valid')
        valid.append(candidate['valid'])
        require(np.allclose(_array(candidate.get('box_world'), (4,), where), world[index], rtol=0, atol=1e-6), where + ': world mismatch')
        logits = _array(candidate.get('logits'), (3,), where + '.logits')
        probabilities = [_sigmoid(float(x)) for x in logits]
        expected = dict(success_score=probabilities[0], precision_score=probabilities[1],
            utility=(probabilities[0] + probabilities[1]) / 2,
            measurement_score=probabilities[2] if index != 1 else 0.)
        for key, value in expected.items():
            original.close(candidate.get(key), value, where + '.' + key, 2e-6)
        support = _ids(candidate.get('support_raw_ids'), where + '.support')
        allowed = _ids(candidate.get('authorized_raw_ids'), where + '.authorized')
        require(support <= allowed <= set(lookup), where + ': support exceeds own authorized evidence')
        require(candidate.get('support_count') == len(support), where + ': duplicate support count')
        foreground = 0
        for raw in allowed:
            slot = lookup[raw]
            delta = xyz[slot] - boxes[index, :3]
            c, s = math.cos(boxes[index, 3]), math.sin(boxes[index, 3])
            local = np.array([c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1], delta[2]])
            margin = size / 2 - np.abs(local)
            if raw in support:
                require(probs[slot] >= .5 and bool((margin >= -2e-5).all()), where + ': support is not own-box predicted FG')
                foreground += int(targets[slot] > .5)
            elif probs[slot] >= .5 and bool((margin > 2e-5).all()):
                raise EvidenceError(where + ': omitted interior foreground support')
        require(candidate.get('support_target_count') == foreground, where + ': foreground truth count mismatch')
        if support:
            original.close(candidate.get('support_target_purity'), foreground / len(support), where + '.purity', 1e-6)
        else:
            require(candidate.get('support_target_purity') is None, where + ': empty support purity must be null')
        geometry = candidate.get('geometry')
        require(isinstance(geometry, dict) if valid[-1] else geometry is None, where + ': invalid geometry presence')
        truth = False
        if valid[-1]:
            iou = base.number(geometry.get('exact_iou'), where + '.exact_iou')
            require(0 <= iou <= 1, where + ': exact IoU out of range')
            truth = index != 1 and foreground >= 1 and foreground / max(len(support), 1) >= .5 and iou >= .5
        require(candidate.get('measurement_target') is truth, where + ': measurement truth mismatch')
        good = valid[-1] and index != 1 and len(support) >= 1 and candidate['measurement_score'] >= .5
        require(candidate.get('eligible') is good, where + ': eligibility mismatch')
        supports.append(support)
        eligible.append(good)
    require(any(valid) and record.get('valid') == valid, label + ': no valid candidate or mismatched validity')
    output_index = max((i for i in range(4) if valid[i]), key=lambda i: candidates[i]['utility'])
    measurement_index = max((i for i in range(4) if eligible[i]), key=lambda i: candidates[i]['measurement_score'], default=-1)
    for where, payload in (('action', action), ('row', row)):
        require(payload.get('output_index') == output_index and payload.get('measurement_index') == measurement_index,
                label + '.' + where + ': output utility / independent highest measurement score mismatch')
        require(payload.get('measurement_accepted') is (measurement_index >= 0)
                and payload.get('output_measurement_disagreement') is (measurement_index >= 0 and measurement_index != output_index),
                label + '.' + where + ': action flags mismatch')
    require(row.get('selected_index') == output_index, label + ': accepted output index mismatch')
    good_available = any(c['valid'] and c['measurement_target'] for c in candidates)
    correct = candidates[measurement_index]['measurement_target'] if measurement_index >= 0 else None
    require(action.get('measurement_correct') is correct
            and action.get('good_measurement_available') is good_available
            and action.get('missed_good_measurement') is (good_available and measurement_index < 0)
            and action.get('wrong_measurement_over_good_candidate') is (good_available and measurement_index >= 0 and not correct),
            label + ': passive measurement truth flags mismatch')
    original.close(row.get('selected_quality'), candidates[output_index]['utility'], label + '.selected utility', 2e-6)
    original.close(row.get('distance'), candidates[output_index]['geometry']['center_error_m'], label + '.accepted distance', 2e-4)
    own = supports[measurement_index] if measurement_index >= 0 else set()
    require(_ids(row.get('measurement_fg_ids'), label + '.measurement IDs') == own, label + ': measurement support mismatch')
    memory_eligible = arm != 'C' and measurement_index >= 0 and len(own) >= 3
    require(action.get('memory_eligible') is memory_eligible and row.get('memory_write_eligible') is memory_eligible,
            label + ': memory eligibility must use measurement own three points')
    require(row.get('memory_write') is memory_eligible, label + ': eligible memory was not committed')
    written = _ids(row.get('memory_written_fg_ids'), label + '.written IDs')
    require(written <= own and row.get('memory_fg_count') == len(written), label + ': memory not measurement subset')
    if measurement_index >= 0:
        require(np.allclose(_array(row.get('measurement_world'), (4,), label + '.measurement world'),
                            world[measurement_index], rtol=0, atol=1e-6), label + ': wrong measurement box')
    else:
        require(row.get('measurement_world') is None, label + ': absent measurement must be null')
    if memory_eligible:
        require(row.get('memory_source_index') == measurement_index and 3 <= len(written) <= 8,
                label + ': memory source/count mismatch')
        require(np.allclose(_array(row.get('memory_world'), (4,), label + '.memory world'),
                            world[measurement_index], rtol=0, atol=1e-6), label + ': memory is not measurement own box')
    else:
        require(not written and row.get('memory_source_index') == -1 and row.get('memory_world') is None,
                label + ': ineligible memory write evidence')
    true_written = sum(int(targets[lookup[raw]] > .5) for raw in written)
    require(row.get('memory_true_fg_count') == true_written and row.get('memory_wrong_write') is
            (memory_eligible and true_written / max(len(written), 1) < .5), label + ': memory truth mismatch')
    if arm == 'C':
        require(not any(valid[2:]) and record.get('acquisition', {}).get('enabled') is False,
                label + ': C must have no extension modes or acquisition')
    return record


def _metrics_identity(metrics, epoch, label):
    original.equal_fields(metrics, dict(schema=SCHEMA, checkpoint_epoch=epoch,
        metric_mode='benchmark_compat', **original.COUNTS), label + '.metrics')
    require(metrics.get('complete_coverage') is True, label + ': incomplete evaluation coverage')
    for metric in original.METRICS:
        require(0 <= base.number(metrics.get(metric), label + '.' + metric) <= 100, label + ': score out of range')


def validate_context(row, label):
    # v36 CandidateSet 不再携带 v35 decoder 的 query/local 诊断字段。
    # observation 保留的支持摘要仍验证；缺旧字段不能伪造零值补齐。
    _array(row.get('diagnostic_history_support'), (3, 3), label + '.history_support')
    _array(row.get('diagnostic_current_support'), (3,), label + '.current_support')
    if row.get('diagnostic_query_context_norm') is not None:
        original._context_diagnostics(row, label, required=True)
    if any('diagnostic_local_' + field in row for field in
           ('neighbor_count', 'selected_count', 'mean_support', 'delta_norm')):
        original._local_diagnostics(row, label)


def validation_peaks(root, read, config):
    epochs = list(range(config['check_val_every_n_epoch'], 61, config['check_val_every_n_epoch']))
    paths = sorted((root / 'validation').glob('epoch=*.json'))
    require([p.name for p in paths] == [f'epoch={e:03d}.json' for e in epochs], 'validation epoch coverage mismatch')
    scores = []
    for epoch in epochs:
        metrics = read(f'validation/epoch={epoch:03d}.json')
        _metrics_identity(metrics, epoch, 'validation')
        scores.append(dict(epoch=epoch, **{m: metrics[m] for m in original.METRICS}))
    return dict(best_success=max(scores, key=lambda x: (x['success'], -x['epoch'])),
        best_precision=max(scores, key=lambda x: (x['precision'], -x['epoch'])),
        recorded_epochs=epochs, formal_gate=False,
        interpretation='each peak retains paired other metric and epoch; ties use earliest epoch')


def read_run(directory, specification, source):
    root, arm = Path(directory).resolve(), specification['arm']
    provenance = {}
    def read(relative, yaml_file=False):
        path = root / relative
        value = original.read_object(path, yaml_file=yaml_file)
        provenance[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        return value
    manifest = read('run_manifest.json')
    original.equal_fields(manifest, dict(schema=SCHEMA, model='ctseqtrackv36', seed=42, arm=arm,
        temporal_backend='cfc', evaluation_rng_policy='endpoint_eval0_v1'), arm + '.manifest')
    require(manifest.get('enabled') == dict(B1=True, B2=arm != 'C', B3=True)
            and manifest.get('calibration_required') is False, arm + ': enabled/calibration mismatch')
    run_id = manifest.get('run_id')
    require(isinstance(run_id, str) and len(run_id) == 32 and all(c in '0123456789abcdef' for c in run_id), arm + ': invalid run ID')
    require(manifest.get('source') == source, arm + ': unregistered execution source')
    config = read('resolved_config.yaml', True)
    require(dict(normalize_config(config)) == config, arm + ': incomplete normalized config')
    require(config['v36_arm'] == arm and config['test'] is False and config['ct_engineering_check'] is False,
            arm + ': formal scratch training required')
    require(config_identity(config) == manifest.get('config_sha256') == specification['config_sha256'], arm + ': config identity mismatch')
    for path in sorted((root / 'resume_manifests').glob('*.json')):
        resume = read(path.relative_to(root).as_posix())
        original.equal_fields(resume, dict(schema='ct_seqtrack.v36.resume.v1', run_id=run_id,
            config_sha256=manifest['config_sha256'], source=source), arm + '.resume')
    budget = read('training_budget.json')
    original.equal_fields(budget, dict(schema=SCHEMA, **original.BUDGET), arm + '.budget')
    require(budget.get('epoch_complete') is True, arm + ': epoch60 incomplete')
    training_source = validate_sampler(budget.get('sampler'), config, 60, arm + '.budget.sampler')
    audit_paths = sorted((root / 'training_audits').glob('epoch=*.json'))
    require(len(audit_paths) == 60, arm + ': exactly sixty audits required')
    orders = {}
    for epoch in range(1, 61):
        audit = read(f'training_audits/epoch={epoch:03d}.json')
        original.equal_fields(audit, dict(schema='ct_seqtrack.v36.training_audit.v1', model_schema=SCHEMA,
            config_sha256=manifest['config_sha256'], completed_epoch=epoch, rows=19108, optimizer_steps=1195), arm + '.audit')
        require(audit.get('epoch_complete') is True, arm + ': incomplete audit')
        require(validate_sampler(audit.get('sampler'), config, epoch, arm + '.audit') == training_source
                and audit['sampler']['annotation_sha256'] == budget['sampler']['annotation_sha256'], arm + ': changing training data')
        orders[epoch] = audit['sampler']['endpoint_order_sha256']
    require(audit['sampler'] == budget['sampler'], arm + ': epoch60 sampler differs from budget')
    results = read('results.json')
    require(results.get('checkpoint_epochs') == original.EPOCHS, arm + ': final window must be 58/59/60')
    rows_by_epoch, scores, mechanisms, followups = {}, {}, {}, {}
    for epoch in original.EPOCHS:
        relative = f'evaluation/epoch={epoch:03d}'
        path = root / relative / 'frames.jsonl'
        rows = original.read_frames(path)
        provenance[relative + '/frames.jsonl'] = hashlib.sha256(path.read_bytes()).hexdigest()
        metrics = read(relative + '/metrics.json')
        _metrics_identity(metrics, epoch, arm)
        for key, row in rows.items():
            if key[1]:
                where = arm + f'.{epoch}.{key}'
                base._check_diagnostics(row, where)
                validate_context(row, where)
                validate_candidates(row, where, arm)
        mechanisms[str(epoch)] = mechanism_summary(list(rows.values()))
        followups[str(epoch)] = measurement_followups(list(rows.values()))
        scores[epoch] = original.score(rows.values())
        for metric in original.METRICS:
            original.close(metrics.get(metric), scores[epoch][metric], arm + '.' + metric, 2e-6)
        # 点级记录已完整检验；原始文件和 SHA 保留，避免五组长期持有数 GB 对象。
        for row in rows.values():
            row.pop('diagnostic_candidates', None)
        rows_by_epoch[epoch] = rows
    late3 = {m: math.fsum(scores[e][m] for e in original.EPOCHS) / 3 for m in original.METRICS}
    _metrics_identity(results.get('final', {}), 60, arm)
    for metric in original.METRICS:
        original.close(results['final'].get(metric), scores[60][metric], arm + '.final.' + metric, 2e-6)
        original.close(results.get('late3', {}).get(metric), late3[metric], arm + '.late3.' + metric, 2e-6)
    peaks = validation_peaks(root, read, config)
    return dict(directory=str(root), config=config, manifest=manifest, source=training_source,
        annotation_sha256=budget['sampler']['annotation_sha256'], endpoint_order=orders,
        rows=rows_by_epoch, scores=scores, late3=late3, provenance=provenance,
        mechanism_diagnostics=mechanisms, measurement_followups=followups, validation_peaks=peaks)


def _pin_reused(run, pin, label):
    require(run['manifest']['config_sha256'] == pin['config_sha256'] and
            run['manifest']['source'] == pin['manifest_source'], label + ': historical identity mismatch')
    actual = {str(e): run['provenance'][f'evaluation/epoch={e:03d}/frames.jsonl'] for e in original.EPOCHS}
    require(actual == pin['frames_sha256'], label + ': original frame fingerprints mismatch')


def summarize_loaded(runs, registration):
    require(len({runs[x]['manifest']['run_id'] for x in LABELS}) == 5, 'new arms must have distinct run IDs')
    require(len({runs[x]['annotation_sha256'] for x in LABELS}) == 1, 'new arms annotation/effective-time identities differ')
    require(all(runs[x]['endpoint_order'] == runs['A']['endpoint_order'] for x in LABELS),
            'A/C/D/E/F endpoint order differs; F changes state reset only')
    comparisons = [(f'{arm}_minus_{ref}', arm, ref) for arm in LABELS
                   for ref in ('reference', 'baseline', 'strong_b0', 'strong_b1')]
    comparisons += [('A_minus_' + control, 'A', control) for control in LABELS[1:]]
    report = base.compare_loaded(runs, candidate_labels=LABELS,
        config_differences=('v36_arm',), comparisons=comparisons)
    passed = report['candidates']['A']['passed']
    report.update(schema=REPORT_SCHEMA, status='passed' if passed else 'failed',
        registered_main_model='A', selected='A' if passed else None,
        source_sha256=registration['source']['sha256'],
        registered_frame_sha256=original.REGISTERED_FRAME_SHA256,
        reused_registration=registration['reused'],
        center_tail_diagnostics={label: original.tail_diagnostics(run) for label, run in runs.items()},
        explanatory_slices=original.explanatory_slices(runs, runs['baseline']['rows'][60]),
        mechanism_diagnostics={label: runs[label]['mechanism_diagnostics'] for label in LABELS},
        measurement_followups={label: runs[label]['measurement_followups'] for label in LABELS},
        validation_peaks={label: runs[label]['validation_peaks'] for label in LABELS})
    report.pop('passing_rank', None)
    report['criterion']['ranking'] = 'A is fixed before training; C/D/E/F and validation peaks never replace A'
    old_tail = original.tail_counts(runs['old_w_quarter'])
    report['tail_risks'] = {}
    for label in LABELS:
        counts = original.tail_counts(runs[label])
        delta = {stage: counts[stage] - old_tail[stage] for stage in ('final60', 'late3')}
        report['tail_risks'][label] = dict(tail_gt10m=counts, old_w_quarter=old_tail,
            delta_frames=delta, worsened=any(x > 0 for x in delta.values()))
    report['risk_review_required'] = report['tail_risks']['A']['worsened']
    report['limitations'].extend(['五组固定单 seed42；消融差值不是跨种子因果结论。',
        '严重漂移与未恢复失跟单独复核；风险不改写原 R/C 数值门。',
        '验证最高分仅描述训练过程，不替代固定 final60/late-3。'])
    return report


def compare_runs(*, registered_reference, baseline, old_b0, old_w_quarter, strong_b0, strong_b1, **paths):
    registration = load_registration()
    require(set(paths) == set(LABELS) and len({str(Path(p).resolve()) for p in paths.values()}) == 5,
            'five distinct complete A/C/D/E/F run directories required')
    runs = dict(reference=base.read_run(registered_reference, 'reference'),
        baseline=base.read_run(baseline, 'baseline'), old_b0=base.read_run(old_b0, 'old_b0'),
        old_w_quarter=base.read_run(old_w_quarter, 'W', expected_lr=2.5e-5))
    original.verify_registered_evidence(runs)
    runs['strong_b0'] = original.read_run(strong_b0, 'v35', expected_lr=1.5e-4)
    old_registration = previous.load_registration()
    b1_spec = next(run for run in old_registration['runs'] if run['label'] == 'scaled_b1')
    runs['strong_b1'] = previous.read_run(strong_b1, b1_spec, old_registration['source'])
    for label in ('strong_b0', 'strong_b1'):
        _pin_reused(runs[label], registration['reused'][label], label)
    specs = {spec['arm']: spec for spec in registration['runs']}
    for arm in LABELS:
        runs[arm] = read_run(paths[arm], specs[arm], registration['source'])
    return summarize_loaded(runs, registration)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('registered-reference', 'baseline', 'old-b0', 'old-w-quarter', 'strong-b0', 'strong-b1'):
        parser.add_argument('--' + name, required=True, type=Path)
    for arm in LABELS:
        parser.add_argument('--' + arm.lower(), '--' + arm, dest=arm, required=True, type=Path)
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
    sys.stdout.reconfigure(encoding='utf-8')
    raise SystemExit(main())
