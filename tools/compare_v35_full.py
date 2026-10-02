"""v35 seed42 六组联合模块实验：只读原帧重算，固定 R/C 门与同 LR 四臂比较。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import compare_v35_b0 as original
from tools import compare_v35_piecewise_b0 as piecewise
from models.ct_v31.config import normalize_config

base = original.base
EvidenceError, require = original.EvidenceError, original.require
ROOT = Path(__file__).resolve().parents[1]
REPORT_SCHEMA = 'ct_seqtrack.v35.full_seed42_comparison.v1'
REGISTRATION_SCHEMA = 'ct_seqtrack.v35.full_seed42_registration.v1'
CANDIDATE_SCHEMA = 'ct_seqtrack.v35.candidate_records.v1'
LRS = dict(scaled=1.5e-4, normal=1e-4)
ARMS = ('b1', 'b1_b2', 'full')
LABELS = tuple(f'{recipe}_{arm}' for recipe in LRS for arm in ARMS)


def load_registration():
    value = original.read_object(ROOT / 'cfgs/ct_seqtrack/35_full_seed42_registration.json')
    require(value.get('schema') == REGISTRATION_SCHEMA, 'Full 登记 schema 不匹配')
    source = value.get('source', {})
    require(isinstance(source.get('files'), dict) and bool(source['files']), '登记缺执行源码')
    require(source.get('sha256') == original.digest(source['files']), '登记源码 checksum 不匹配')
    for path, checksum in source['files'].items():
        original._sha256(checksum, 'registration.source.' + path)
    runs = value.get('runs', [])
    require(len(runs) == 6 and {r.get('label') for r in runs} == set(LABELS), '需要固定六组登记')
    for run in runs:
        recipe, arm = run['label'].split('_', 1)
        require(run.get('arm') == arm and run.get('lr') == LRS[recipe], '登记 arm/LR 不匹配')
        original._sha256(run.get('config_sha256'), run['label'] + '.config_sha256')
    return value


def _finite_vector(value, length, label):
    require(isinstance(value, list) and len(value) == length, label + ': shape 不匹配')
    return [base.number(v, label) for v in value]


def _arm_diagnostics(row, label, arm):
    # 旧 B0 wrapper 强制 selected_index==0；Full 必须允许真实 mode 被选择。
    base._check_diagnostics(row, label)
    selected = row.get('selected_index')
    require(type(selected) is int and 0 <= selected < 4, label + ': selected_index 不合法')
    require(arm == 'full' or selected == 0, label + ': 未开启 B3 不得选择 mode')
    require(0 <= base.number(row.get('selected_quality'), label) <= 1, label + ': quality 越界')
    for field in ('strong', 'supported', 'memory_write', 'memory_wrong_write'):
        require(type(row.get(field)) is bool, label + ': 缺布尔诊断 ' + field)
    require(row['supported'] is (row['strong'] and row['selected_quality'] >= .5),
            label + ': supported 与 strong/quality 合同不匹配')


def validate_candidates(row, label):
    """候选日志只检验证据，不影响指标或模型选择；invalid 几何必须为 null。"""
    record = row.get('diagnostic_candidates')
    require(isinstance(record, dict) and record.get('schema') == CANDIDATE_SCHEMA,
            label + ': 缺完整候选记录')
    valid = record.get('valid')
    require(isinstance(valid, list) and len(valid) == 4 and all(type(v) is bool for v in valid)
            and valid[0], label + ': 候选 valid 不合法')
    for key in ('boxes_anchor_relative', 'boxes_world'):
        boxes = record.get(key)
        require(isinstance(boxes, list) and len(boxes) == 4, label + ': ' + key)
        for box in boxes:
            _finite_vector(box, 4, label + '.' + key)
    quality = _finite_vector(record.get('quality'), 4, label + '.quality')
    logits = _finite_vector(record.get('quality_logits'), 4, label + '.quality_logits')
    for probability, logit in zip(quality, logits):
        expected = 1 / (1 + math.exp(-logit)) if logit >= 0 else math.exp(logit) / (1 + math.exp(logit))
        original.close(probability, expected, label + '.quality sigmoid', 2e-6)
    selected = record.get('selected_index')
    require(type(selected) is int and 0 <= selected < 4 and valid[selected], label + ': 选择无效候选')
    require(selected == row.get('selected_index'), label + ': 候选/正式 selected_index 不一致')
    original.close(record.get('selected_quality'), quality[selected], label + '.selected_quality', 2e-6)
    original.close(row.get('selected_quality'), quality[selected], label + '.row_quality', 2e-6)
    expected_selected = max((i for i in range(4) if valid[i]), key=lambda i: quality[i])
    require(selected == expected_selected, label + ': 选择不符合 quality 排名/同分 q0 优先')
    geometry = record.get('geometry')
    require(isinstance(geometry, list) and len(geometry) == 4, label + ': 几何数不匹配')
    for i, geom in enumerate(geometry):
        if not valid[i]:
            require(geom is None, label + ': 无效候选几何应为 null')
            continue
        require(isinstance(geom, dict), label + ': 有效候选缺几何')
        for field in ('iou', 'center_error_m', 'center_xy_error_m', 'yaw_error_rad', 'axis_yaw_error_rad'):
            require(base.number(geom.get(field), label + '.' + field) >= 0, label + ': 负几何值')
        require(geom['iou'] <= 1, label + ': IoU 超界')
    original.close(geometry[selected]['iou'], row['iou'], label + '.accepted IoU', 2e-6)
    original.close(geometry[selected]['center_error_m'], row['distance'], label + '.accepted distance', 2e-6)
    modes = record.get('modes')
    require(isinstance(modes, list) and len(modes) == 3, label + ': modes 数不匹配')
    ext = record.get('extension')
    require(isinstance(ext, dict) and ext.get('capacity') == 256, label + ': 扩展证据 capacity')
    slots, ids = ext.get('slots'), ext.get('raw_ids')
    require(isinstance(slots, list) and all(type(s) is int and 0 <= s < 256 for s in slots)
            and len(set(slots)) == len(slots), label + ': 扩展槽重复/越界')
    require(isinstance(ids, list) and len(ids) == len(slots)
            and all(type(i) is int and i >= 0 for i in ids) and len(set(ids)) == len(ids),
            label + ': 扩展 raw ID 无效/重复')
    for field in ('acquisition_indices', 'xyz_anchor_relative', 'identity_logits', 'identity_support',
                  'vote_xyz_anchor_relative', 'reliability_logits', 'target_labels', 'partition'):
        values = ext.get(field)
        require(isinstance(values, list) and len(values) == len(slots), label + ': 扩展字段缺失/长度 ' + field)
        for value in values:
            if field in ('xyz_anchor_relative', 'vote_xyz_anchor_relative'):
                _finite_vector(value, 3, label + '.' + field)
            else:
                base.number(value, label + '.' + field)
    require(all(type(i) is int and 0 <= i < 768 for i in ext['acquisition_indices'])
            and len(set(ext['acquisition_indices'])) == len(slots), label + ': 扩展池索引不合法')
    require(all(value in (0, 1) for value in ext['target_labels'])
            and all(value in (0, 1) for value in ext['partition'])
            and all(0 <= value <= 1 for value in ext['identity_support']), label + ': 扩展标签/来源/支持越界')
    slot_to_id = dict(zip(slots, ids))
    slot_to_target = dict(zip(slots, ext['target_labels']))
    formed = record.get('mode_formed')
    require(isinstance(formed, list) and len(formed) == 3 and all(type(v) is bool for v in formed),
            label + ': mode_formed 不合法')
    for i, mode in enumerate(modes):
        require(mode.get('formed') is formed[i] and (not valid[i + 1] or formed[i]), label + ': mode formed/valid 不一致')
        require(mode.get('candidate_valid') is valid[i + 1], label + ': mode 权限与 decoder 不一致')
        members = mode.get('member_slots')
        require(isinstance(members, list) and len(set(members)) == len(members)
                and set(members) <= set(slots), label + ': mode 使用了无效扩展成员')
        require(mode.get('member_count') == len(members)
                and mode.get('member_raw_ids') == [slot_to_id[s] for s in members],
                label + ': mode 成员映射不一致')
        target_count = mode.get('target_count')
        require(type(target_count) is int and 0 <= target_count <= len(members), label + ': mode GT 点数')
        require(target_count == sum(slot_to_target[s] > 0 for s in members), label + ': mode GT 点数与成员不一致')
        if members:
            original.close(mode.get('target_purity'), target_count / len(members), label + '.purity', 1e-6)
        else:
            require(mode.get('target_purity') is None, label + ': 空 mode 纯度应为 null')
    prior = record.get('prior')
    require(isinstance(prior, dict), label + ': 缺 prior 记录')
    for field in ('box_anchor_relative', 'box_world'):
        _finite_vector(prior.get(field), 4, label + '.prior.' + field)
    acquisition = record.get('acquisition')
    require(isinstance(acquisition, dict), label + ': 缺获取阶段记录')
    count, foreground = acquisition.get('valid_count'), acquisition.get('target_count')
    require(type(count) is int and type(foreground) is int and 0 <= foreground <= count <= 768,
            label + ': 获取点数不合法')
    require(len(slots) <= count, label + ': 256 读取点数超过 768 获取点数')
    return record


def read_run(directory, specification, source):
    """固定登记配置摘要封住所有超参数；预算与原帧仍独立重算。"""
    root, label = Path(directory).resolve(), specification['label']
    provenance = {}
    def read(relative, yaml_file=False):
        path = root / relative
        value = original.read_object(path, yaml_file=yaml_file)
        provenance[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        return value
    manifest = read('run_manifest.json')
    schema, arm = 'ct_seqtrack.joint_identity.v35', specification['arm']
    original.equal_fields(manifest, dict(schema=schema, model='ctseqtrackv35', seed=42,
        arm=arm, temporal_backend='cfc',
        evaluation_rng_policy='per_checkpoint_seed_v1'), label + '.manifest')
    require(manifest.get('calibration_required') is False, label + ': 当前 B3 不使用外部 calibration')
    require(manifest.get('enabled') == dict(B1=True, B2=arm in ('b1_b2', 'full'), B3=arm == 'full'),
            label + ': enabled 不一致')
    require(manifest.get('source') == source, label + ': 未登记的执行源码')
    config = read('resolved_config.yaml', True)
    require(dict(normalize_config(config)) == config, label + ': 配置未完整规范化')
    original.equal_fields(config, dict(v31_arm=arm, lr=specification['lr'], seed=42,
        init_checkpoint=None), label + '.config')
    require(config.get('ct_engineering_check') is False and config.get('test') is False
            and config.get('v31_evaluate_late3') is True, label + ': 需要完整正式训练与 late3')
    require(original.config_digest(config, manifest) == manifest.get('config_sha256') ==
            specification['config_sha256'], label + ': 配置身份不匹配登记')
    for path in sorted((root / 'resume_manifests').glob('*.json')):
        event = read(path.relative_to(root).as_posix())
        original.equal_fields(event, dict(schema='ct_seqtrack.v35.resume_source.v1',
            config_sha256=manifest['config_sha256'], original_source_sha256=source['sha256'], source=source),
            label + '.resume')
    budget = read('training_budget.json')
    original.equal_fields(budget, dict(schema=schema, **original.BUDGET), label + '.budget')
    require(budget.get('epoch_complete') is True, label + ': epoch60 未完整')
    training_source = original._sampler(budget.get('sampler'), config, 60, label + '.budget.sampler')
    for epoch in range(1, 61):
        audit = read(f'training_audits/epoch={epoch:03d}.json')
        original.equal_fields(audit, dict(schema='ct_seqtrack.v35.training_audit.v1', model_schema=schema,
            config_sha256=manifest['config_sha256'], completed_epoch=epoch,
            rows=original.BUDGET['last_epoch_rows'], optimizer_steps=original.BUDGET['last_epoch_steps']), label + '.audit')
        require(audit.get('epoch_complete') is True, label + ': 未完整训练轮次')
        require(original._sampler(audit.get('sampler'), config, epoch, label + '.audit') == training_source,
                label + ': 跨轮次训练数据不一致')
    require(audit['sampler'] == budget['sampler'], label + ': 最后一轮 sampler 不匹配')
    results = read('results.json')
    require(results.get('checkpoint_epochs') == original.EPOCHS, label + ': 必须评测 58/59/60')
    rows_by_epoch, scores = {}, {}
    def metrics_identity(metrics, epoch):
        original.equal_fields(metrics, dict(schema=schema, checkpoint_epoch=epoch,
            metric_mode='benchmark_compat', **original.COUNTS), label + '.metrics')
        require(metrics.get('complete_coverage') is True, label + ': 评测覆盖不完整')
    for epoch in original.EPOCHS:
        relative = f'evaluation/epoch={epoch:03d}'
        path = root / relative / 'frames.jsonl'
        rows = original.read_frames(path)
        provenance[relative + '/frames.jsonl'] = hashlib.sha256(path.read_bytes()).hexdigest()
        metrics = read(relative + '/metrics.json')
        metrics_identity(metrics, epoch)
        for key, row in rows.items():
            if not key[1]:
                continue
            where = label + f'.{epoch}.{key}'
            _arm_diagnostics(row, where, arm)
            original._context_diagnostics(row, where, required=True)
            original._local_diagnostics(row, where)
            record = validate_candidates(row, where)
            # 先完整检验每个候选/真实点台账，再只保留本报告需要的摘要。
            # frames.jsonl 与其 SHA 原样保留，避免六组 × 三轮长期持有数 GB 点级 Python 对象。
            row['diagnostic_candidates'] = {field: record[field] for field in
                ('schema', 'valid', 'geometry', 'selected_index')}
        scores[epoch] = original.score(rows.values())
        for metric in original.METRICS:
            original.close(metrics.get(metric), scores[epoch][metric], label + '.' + metric, 2e-6)
        rows_by_epoch[epoch] = rows
    late3 = {m: math.fsum(scores[e][m] for e in original.EPOCHS) / 3 for m in original.METRICS}
    metrics_identity(results.get('final', {}), 60)
    for metric in original.METRICS:
        original.close(results['final'].get(metric), scores[60][metric], label + '.final.' + metric, 2e-6)
        original.close(results.get('late3', {}).get(metric), late3[metric], label + '.late3.' + metric, 2e-6)
    return dict(directory=str(root), config=config, manifest=manifest, source=training_source,
        rows=rows_by_epoch, scores=scores, late3=late3, provenance=provenance)


def candidate_summary(run):
    result = {}
    for epoch in original.EPOCHS:
        records = [row['diagnostic_candidates'] for key, row in run['rows'][epoch].items() if key[1]]
        oracle, chosen, q0 = [], [], []
        misses, escapes, mode_frames = 0, 0, 0
        for record in records:
            geometry = record['geometry']
            best = max(g['iou'] for g in geometry if g is not None)
            selected = geometry[record['selected_index']]['iou']
            oracle.append(best)
            chosen.append(selected)
            q0.append(geometry[0]['iou'])
            misses += int(best >= .5 and selected < .1)
            escapes += int(geometry[0]['iou'] < .1 and selected >= .5)
            mode_frames += int(any(record['valid'][1:]))
        count = len(records)
        result[str(epoch)] = dict(prediction_frames=count, candidate_record_coverage=1.,
            mean_best_eligible_iou=math.fsum(oracle) / count,
            mean_selected_iou=math.fsum(chosen) / count,
            mean_q0_iou=math.fsum(q0) / count,
            mean_oracle_selection_gap=math.fsum(a - b for a, b in zip(oracle, chosen)) / count,
            good_candidate_bad_selection_frames=misses, q0_bad_selected_good_frames=escapes,
            eligible_mode_frames=mode_frames)
    return dict(epochs=result, interpretation='同模型自身递推状态内的候选诊断；oracle 不是真实可部署策略，q0 不是独立 B1+B2 递推结果')


def resume_fingerprints(provenance):
    """旧恢复读取器使用本机分隔符；登记与跨平台比较统一为 POSIX 相对路径。"""
    return {relative.replace('\\', '/'): checksum for relative, checksum in provenance.items()
            if relative.replace('\\', '/').startswith('resume_manifests/')}


def compare_runs(*, registered_reference, baseline, old_b0, old_w_quarter, normal_b0, scaled_b0, **paths):
    registration = load_registration()
    require(set(paths) == set(LABELS), '需要六组完整新实验')
    require(len({str(Path(p).resolve()) for p in paths.values()}) == 6, '六组不能复用同一目录')
    runs = dict(reference=base.read_run(registered_reference, 'reference'),
        baseline=base.read_run(baseline, 'baseline'), old_b0=base.read_run(old_b0, 'old_b0'),
        old_w_quarter=base.read_run(old_w_quarter, 'W', expected_lr=2.5e-5))
    original.verify_registered_evidence(runs)
    historical_registration = piecewise.load_source_registration()
    for recipe, path in (('normal', normal_b0), ('scaled', scaled_b0)):
        label = recipe + '_b0'
        run = original.read_run(path, 'v35', expected_lr=LRS[recipe])
        registered = registration['reused_b0'][recipe]
        require(run['manifest']['config_sha256'] == registered['config_sha256']
                and run['manifest']['source'] == registered['manifest_source'], label + ': 旧 B0 身份不匹配')
        require(piecewise.validate_resume_sources(run, historical_registration) == registered['effective_source'],
                label + ': 旧 B0 恢复来源不匹配')
        resume_hashes = resume_fingerprints(run['provenance'])
        require(resume_hashes == registered['resume_sha256'], label + ': 旧 B0 恢复记录 fingerprint 不匹配')
        hashes = {str(e): run['provenance'][f'evaluation/epoch={e:03d}/frames.jsonl'] for e in original.EPOCHS}
        require(hashes == registered['frames_sha256'], label + ': 旧 B0 原帧 fingerprint 不匹配')
        runs[label] = run
    specs = {run['label']: run for run in registration['runs']}
    for label in LABELS:
        runs[label] = read_run(paths[label], specs[label], registration['source'])
        recipe = label.split('_', 1)[0]
        original._same_config(runs[label], runs[recipe + '_b0'], allowed={'v31_arm'}, label=label + ' vs 同 LR B0')
    comparisons = []
    for recipe in LRS:
        chain = [recipe + '_' + arm for arm in ('b0', *ARMS)]
        comparisons.extend((f'{right}_minus_{left}', right, left) for left, right in zip(chain, chain[1:]))
    comparisons += [(label + '_minus_scaled_b0', label, 'scaled_b0') for label in LABELS]
    comparisons.append(('scaled_full_minus_normal_full', 'scaled_full', 'normal_full'))
    report = base.compare_loaded(runs, candidate_labels=LABELS, config_differences=('lr', 'v31_arm'), comparisons=comparisons)
    report.update(schema=REPORT_SCHEMA, source_sha256=registration['source']['sha256'],
        registered_frame_sha256=original.REGISTERED_FRAME_SHA256,
        center_tail_diagnostics={label: original.tail_diagnostics(run) for label, run in runs.items()},
        explanatory_slices=original.explanatory_slices(runs, runs['baseline']['rows'][60]),
        candidate_diagnostics={label: candidate_summary(runs[label]) for label in LABELS},
        new_training_runs=6, training_seed=42, automatic_retraining=False,
        candidate_ledger_storage='逐帧完整校验后仅在内存保留候选评分摘要；磁盘原帧、点级记录及其 SHA 不变',
        source_note='新臂与旧 B0 为登记的不同执行源码；公共 B0 不变证据见实施报告，不声称源码完全相同')
    report['criterion']['ranking'] = '只在 Full 两档中按原门通过、final60 S/P、较低 LR 排序；模块作用按同 LR 逐臂比较'
    ranking = sorted(('scaled_full', 'normal_full'), key=lambda label:
        (-runs[label]['scores'][60]['success'], -runs[label]['scores'][60]['precision'], specs[label]['lr']))
    passing = [label for label in ranking if report['candidates'][label]['passed']]
    report.update(all_full_rank=ranking, passing_full_rank=passing, selected_full=passing[0] if passing else None)
    old_tail = original.tail_counts(runs['old_w_quarter'])
    report['tail_risks'] = {}
    report['full_vs_strong_b0'] = {}
    for label in LABELS:
        tail = original.tail_counts(runs[label])
        delta = {stage: tail[stage] - old_tail[stage] for stage in ('final60', 'late3')}
        report['tail_risks'][label] = dict(tail_gt10m=tail, old_w_quarter=old_tail,
            delta_frames=delta, worsened=any(d > 0 for d in delta.values()))
        if label.endswith('_full'):
            current, reference = report['arms'][label]['groups']['overall'], report['arms']['scaled_b0']['groups']['overall']
            differences = {stage + '_' + metric: current[stage][metric] - reference[stage][metric]
                for stage in ('final60', 'late3') for metric in original.METRICS}
            report['full_vs_strong_b0'][label] = dict(delta_pp=differences, all_four_nonnegative=all(d >= 0 for d in differences.values()))
    report['status'] = 'passed' if passing else 'failed'
    report['selected'] = report['selected_full']
    report['limitations'].extend(['仅 seed42；不能据此宣称跨 seed 稳定增益。',
        '新增各臂从头联合训练；Full 相对 B1+B2 同时改变候选监督与选择，非纯推理选择消融。',
        '原 R/C 硬门、超过现有最强 B0 与漂移风险是三个分别报告的问题。'])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for label in ('registered_reference', 'baseline', 'old_b0', 'old_w_quarter', 'normal_b0', 'scaled_b0', *LABELS):
        parser.add_argument('--' + label.replace('_', '-'), required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = vars(parser.parse_args(argv))
    output = args.pop('output')
    try:
        report = compare_runs(**args)
        if output:
            base.write_report(output, report)
        code = 0 if report['status'] == 'passed' else 1
    except (EvidenceError, OSError, ValueError, TypeError, KeyError) as error:
        report, code = dict(schema=REPORT_SCHEMA, status='invalid_evidence', message=str(error)), 2
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return code


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    raise SystemExit(main())
