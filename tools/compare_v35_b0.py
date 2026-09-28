"""v35 四配方与可选第二种子的只读正式验收；评分数学复用 v34。"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import compare_v34_b0 as base
from tools.compare_v34_b0 import (EvidenceError, require, equal_fields, read_object,
    BUDGET, COMMON_CONFIG, COUNTS, EPOCHS, EXCLUDED_IDENTITY, METRICS, CONTEXT_LRS,
    config_digest, digest, read_frames, score, close)
from models.ct_v31.config import V35_DEFAULTS, normalize_config
_sha256 = base._sha256
_diagnostics = base._diagnostics
_context_diagnostics = base._context_diagnostics
LABELS = ('reference', 'baseline', 'old_b0', 'S', 'W', 'v35')
REPORT_SCHEMA = 'ct_seqtrack.v35.b0_comparison.v1'
RECIPES = dict(quarter=2.5e-5, half=5e-5, normal=1e-4, scaled=1.5e-4)
# 固定原始逐帧证据（58/59/60），来自2026-09-28八组正式复核。
# 不接受命名相同但分数更弱的新参考来降低登记门槛；路径可迁移。
REGISTERED_FRAME_SHA256 = {
    'reference': ('b5a9b0a571f4b4b7b97f5d8196937f5f66c1ecf3544d24e7e9e890a0a28092c9',
        'dc57cab49715eb5e25b2d4053ac94bdff2fff314dbfaa1d62f7049bf0ef21a19',
        '0115681e71460ff05652c886a229e52489464eeda8190bad4e24f76bd09576bd'),
    'baseline': ('257a98cd560838852492a674154d8418b2cd5a52d9d3a77a64d4fe8edc1ef081',
        '2a4454f03f5bade87ca0e3aae324604f9e075f2e5e8db9360a8d11f626979023',
        '7524d6b032f0bc1815d291ca49f66cc2913179ac7d3f982f9f2e3bb3d42a748c'),
    'old_b0': ('2c564197784ebf42ec6a19ce2444b133bdb7ed8fec44e0014e2ea11f845cef7a',
        '1bd598ad5109680aebddfad64795d3451ee0e98147b3d7d600e08d30ec30aa53',
        '824a437a691c4258972da50df7a941f5c3b0526120e332eb1ad3f2c985680f31'),
    'old_w_quarter': ('44888f0eabe2cdd921dfb7add86c980afeb97b9b7aeab0912c06bb3dcb7f97a4',
        '5c41d2f03edae4ba77f121aef628f5aaa2c6372bd4ae88a6739b5a98c41dc1f0',
        '6a0e647e6415cc538bdb59b8c30b5d82d3c25e7f38e12bb8111aa82a07d2625c')}


def verify_registered_evidence(runs):
    for label, hashes in REGISTERED_FRAME_SHA256.items():
        actual = tuple(runs[label]['provenance'].get(f'evaluation/epoch={e:03d}/frames.jsonl') for e in EPOCHS)
        require(actual == hashes, label + ': 固定登记逐帧fingerprint不匹配，拒绝替换原证据')


def _local_diagnostics(row, label):
    for field in ('local_neighbor_count', 'local_selected_count', 'local_mean_support', 'local_delta_norm'):
        name = 'diagnostic_' + field
        value = row.get(name)
        require(isinstance(value, list) and len(value) == 4, label + ': missing/invalid ' + name)
        for values in value:
            require(isinstance(values, list) and len(values) == 8, label + ': invalid shape ' + name)
            for item in values:
                number = base.number(item, label + '.' + name)
                require(number >= 0, label + ': negative ' + name)
                if field.endswith('count'):
                    require(number == int(number), label + ': count must be integral')
                if field == 'local_selected_count':
                    require(number <= 16, label + ': local selected count >16')
                if field == 'local_mean_support':
                    require(number <= 1, label + ': local support >1')
    for total, selected in zip(row['diagnostic_local_neighbor_count'], row['diagnostic_local_selected_count']):
        require(all(s <= n for n, s in zip(total, selected)), label + ': selected count exceeds neighborhood')

def _sampler(value, config, epoch, label):
    require(isinstance(value, dict), label + ': 缺 sampler')
    require(value.get('manifest_sha256') == digest({k: v for k, v in value.items()
            if k != 'manifest_sha256'}), label + ': sampler checksum 不匹配')
    expected = dict(epoch=epoch - 1, rows=BUDGET['last_epoch_rows'], batch_size=16, seed=config['seed'])
    is_reference = config['net_model'] == 'seqtrack_reference'
    expected['schema'] = ('seqtrack_reference.v32.teacher.v1' if is_reference
                          else 'ct_seqtrack.v32.ready_queue.v2')
    if not is_reference:
        expected.update(short_window=config['v31_short_window'], long_window=8, curriculum_epochs=10,
            reserve_windows=112, seed_translation=.3, seed_yaw_degrees=1.5,
            batches=BUDGET['last_epoch_steps'],
            reserve_policy='stratified_singleton_windows_then_fill_v1',
            drain_policy='running_b0_bn_from_first_reserve_or_partial_v1',
            seed_policy='shared_anchor_local_translation_world_yaw_v1')
        _sha256(value.get('plan_sha256'), label + '.plan_sha256')
    equal_fields(value, expected, label)
    if config['net_model'] == 'ctseqtrackv35':
        require(value.get('initial_seed_policy') == V35_DEFAULTS['v35_seed_policy'], label + ': initial_seed_policy mismatch')
    require(value.get('training') is True, label + ': 非训练排程')
    return _sha256(value.get('source_sha256'), label + '.source_sha256')


def read_run(directory, label, *, expected_lr=None, expected_seed=42):
    require(label in LABELS, '未登记的比较角色: ' + str(label))
    allowed_lrs = (tuple(RECIPES.values()) if label == 'v35' else
                  (CONTEXT_LRS if label in ('S', 'W') else
                   ((1e-4, 5e-5) if label == 'reference' else
                    ((5e-5,) if label == 'baseline' else (1e-4,)))))
    default_lr = 5e-5 if label in ('baseline', 'S', 'W', 'v35') else 1e-4
    expected_lr = default_lr if expected_lr is None else expected_lr
    require(expected_lr in allowed_lrs, label + ': 未登记的学习率')
    root = Path(directory).resolve()
    provenance = {}
    def read(relative, yaml_file=False):
        path = root / relative
        value = read_object(path, yaml_file=yaml_file)
        provenance[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        return value
    manifest = read('run_manifest.json')
    version = 35 if label == 'v35' else (34 if label in ('S', 'W') else (32 if label == 'old_b0' else 33))
    schema = f'ct_seqtrack.joint_identity.v{version}'
    model = 'seqtrack_reference' if label == 'reference' else f'ctseqtrackv{version}'
    equal_fields(manifest, dict(schema=schema, model=model, seed=expected_seed, arm='b0'), label + '.manifest')
    require(manifest.get('enabled') == dict(B1=False, B2=False, B3=False), label + ': 只能比较 B0')
    require(manifest.get('evaluation_rng_policy') == 'per_checkpoint_seed_v1', label + ': 评测 RNG 身份错误')
    config = read('resolved_config.yaml', True)
    expected = dict(COMMON_CONFIG, experiment_family=f'ct_seqtrack_v{version}', net_model=model, seed=expected_seed,
                    v31_short_window=4 if label in ('W', 'v35') else 3)
    expected['lr'] = expected_lr
    if label in ('baseline', 'S', 'W', 'v35'):
        expected.update(lr_schedule='multistep', lr_milestones=[20, 50])
    equal_fields(config, expected, label + '.config')
    if label == 'v35':
        for key, value in V35_DEFAULTS.items():
            require(type(config.get(key)) is type(value) and config[key] == value, label + ': invalid ' + key)
        try:
            normalized = normalize_config(config)
        except (ValueError, TypeError) as error:
            raise EvidenceError(str(error)) from error
        require(dict(normalized) == config, label + ': incomplete normalized configuration')
    if label in ('reference', 'old_b0'):
        require(config.get('lr_schedule', 'step') == 'step' and config.get('lr_milestones', []) == [],
                label + ': 历史参考必须使用原 StepLR 配方')
    require(config.get('lr_warmup_steps', 0) == 0, label + ': 不允许 warmup')
    require(config.get('ct_engineering_check') is False and config.get('test') is False
            and config.get('init_checkpoint') is None and config.get('v31_evaluate_late3') is True,
            label + ': 需要完整 scratch 正式训练/同身份恢复')
    require(config_digest(config, manifest) == manifest.get('config_sha256'), label + ': config 身份 checksum 不匹配')
    source = manifest.get('source')
    require(isinstance(source, dict) and isinstance(source.get('files'), dict) and source['files'], label + ': 缺源码身份')
    for path, checksum in source['files'].items():
        _sha256(checksum, label + '.source.' + path)
    require(source.get('sha256') == digest(source['files']), label + ': 源码 checksum 不匹配')
    budget = read('training_budget.json')
    equal_fields(budget, dict(schema=schema, **BUDGET), label + '.budget')
    require(budget.get('epoch_complete') is True, label + ': 未完成 epoch60')
    training_source = _sampler(budget.get('sampler'), config, 60, label + '.budget.sampler')
    audits = {}
    for epoch in range(1, 61):
        audit = read(f'training_audits/epoch={epoch:03d}.json')
        equal_fields(audit, dict(completed_epoch=epoch, rows=BUDGET['last_epoch_rows'],
            optimizer_steps=BUDGET['last_epoch_steps']), label + '.audit')
        require(audit.get('epoch_complete') is True, label + ': 存在未完成轮次')
        if label in ('S', 'W', 'v35'):
            equal_fields(audit, dict(schema=f'ct_seqtrack.v{version}.training_audit.v1', model_schema=schema,
                config_sha256=manifest['config_sha256']), label + '.audit.identity')
        require(_sampler(audit.get('sampler'), config, epoch, label + f'.audit{epoch}') == training_source,
                label + ': 训练数据身份跨轮次变化')
        audits[epoch] = audit
    require(audits[60]['sampler'] == budget['sampler'], label + ': budget 与 epoch60 sampler 不一致')
    results = read('results.json')
    require(results.get('checkpoint_epochs') == EPOCHS, label + ': 需要58/59/60正式评测')
    require(isinstance(results.get('late3'), dict), label + ': 缺 late3 汇总')
    rows_by_epoch, scores = {}, {}
    def check_metrics(metrics, epoch, where):
        require(isinstance(metrics, dict), where + ': 缺正式 metrics 对象')
        equal_fields(metrics, dict(schema=schema, checkpoint_epoch=epoch,
            metric_mode='benchmark_compat', **COUNTS), where)
        require(metrics.get('complete_coverage') is True, where + ': 评测不完整')
    for epoch in EPOCHS:
        relative = f'evaluation/epoch={epoch:03d}'
        rows = read_frames(root / relative / 'frames.jsonl')
        provenance[relative + '/frames.jsonl'] = hashlib.sha256((root / relative / 'frames.jsonl').read_bytes()).hexdigest()
        metrics = read(relative + '/metrics.json')
        check_metrics(metrics, epoch, label + '.metrics')
        if label in ('baseline', 'S', 'W', 'v35'):
            for key, row in rows.items():
                if key[1]:
                    _diagnostics(row, f'{label}.{epoch}.{key}')
        for key, row in rows.items():
            if key[1]:
                _context_diagnostics(row, f'{label}.{epoch}.{key}', required=label in ('S', 'W', 'v35'))
                if label == 'v35':
                    _local_diagnostics(row, f'{label}.{epoch}.{key}')
        scores[epoch] = score(rows.values())
        for m in METRICS:
            close(metrics.get(m), scores[epoch][m], label + f'.epoch{epoch}.' + m, 2e-6)
        rows_by_epoch[epoch] = rows
    late3 = {m: math.fsum(scores[e][m] for e in EPOCHS) / 3 for m in METRICS}
    check_metrics(results.get('final', {}), 60, label + '.final')
    for m in METRICS:
        close(results['final'].get(m), scores[60][m], label + '.final.' + m, 2e-6)
        close(results.get('late3', {}).get(m), late3[m], label + '.late3.' + m, 2e-6)
    return dict(directory=str(root), config=config, manifest=manifest, source=training_source,
                rows=rows_by_epoch, scores=scores, late3=late3, provenance=provenance)


def tail_counts(run):
    """超过10m只作风险项，分母为预测帧，不更改S/P硬门。"""
    counts = {str(e): sum(r['distance'] > 10 for k, r in run['rows'][e].items() if k[1])
              for e in EPOCHS}
    return dict(epochs=counts, final60=counts['60'], late3=math.fsum(counts.values()) / 3)


def tail_diagnostics(run):
    """中心长尾帧数及涉及轨迹数；5m项仅描述，不新增硬门。"""
    output = {}
    for threshold in (5, 10):
        epochs = {}
        for epoch in EPOCHS:
            keys = [k for k, row in run['rows'][epoch].items() if k[1] and row['distance'] > threshold]
            epochs[str(epoch)] = dict(frames=len(keys), tracklets=len({key[0] for key in keys}))
        output[f'gt{threshold}m'] = dict(epochs=epochs, final60=epochs['60'],
            late3={key: math.fsum(epoch[key] for epoch in epochs.values()) / 3
                   for key in ('frames', 'tracklets')})
    return output


def explanatory_slices(runs, canonical):
    """固定C的GT可见性切片；只解释S/P与同状态细化，不增加晋级门。"""
    predictions = sorted(key for key in canonical if key[1] > 0)
    groups = dict(raw0_2=[key for key in predictions if canonical[key]['diagnostic_target_count'] <= 2],
                  raw_visible=[key for key in predictions if canonical[key]['diagnostic_target_count'] > 0])
    result = {}
    for label, run in runs.items():
        result[label] = {}
        for group, keys in groups.items():
            epochs = {}
            pooled = []
            for epoch in EPOCHS:
                rows = [run['rows'][epoch][key] for key in keys]
                pooled.extend(rows)
                epochs[str(epoch)] = dict(n=len(rows), **score(rows), coarse_fine=base.refinement(rows))
            late3 = dict(n=len(keys), checkpoint_rows=len(pooled),
                **{metric: math.fsum(epochs[str(epoch)][metric] for epoch in EPOCHS) / 3
                   if keys else None for metric in METRICS}, coarse_fine=base.refinement(pooled))
            result[label][group] = dict(n=len(keys), tracks=len({key[0] for key in keys}),
                grouping_source='固定C的GT原始目标点数；排除初始化帧', epochs=epochs,
                final60=epochs['60'], late3=late3,
                late3_aggregation='S/P为三轮算术均值；coarse_fine合并三轮有效配对，valid_pairs为累计数')
    return result


def _same_config(left, right, *, allowed, label):
    excluded = EXCLUDED_IDENTITY | {'experiment_name'} | set(allowed)
    require({k: v for k, v in left['config'].items() if k not in excluded} ==
            {k: v for k, v in right['config'].items() if k not in excluded}, label + ': configuration mismatch')


def compare_runs(*, registered_reference, baseline, old_b0, old_w_quarter,
                 quarter, half, normal, scaled, selected_seed52=None, reference_seed52=None):
    paths = dict(quarter=quarter, half=half, normal=normal, scaled=scaled)
    require(len({str(Path(path).resolve()) for path in paths.values()}) == 4,
            '需要四个不同的完整seed42候选')
    runs = dict(reference=base.read_run(registered_reference, 'reference'),
        baseline=base.read_run(baseline, 'baseline'), old_b0=base.read_run(old_b0, 'old_b0'),
        old_w_quarter=base.read_run(old_w_quarter, 'W', expected_lr=RECIPES['quarter']))
    verify_registered_evidence(runs)
    runs.update({name: read_run(path, 'v35', expected_lr=lr)
                 for name, lr in RECIPES.items() for path in (paths[name],)})
    comparisons = [(name + '_minus_R', name, 'reference') for name in RECIPES]
    comparisons += [(name + '_minus_C', name, 'baseline') for name in RECIPES]
    report = base.compare_loaded(runs, candidate_labels=tuple(RECIPES), config_differences=('lr',),
                                 comparisons=comparisons)
    report['schema'] = REPORT_SCHEMA
    report['registered_frame_sha256'] = REGISTERED_FRAME_SHA256
    report['center_tail_diagnostics'] = {label: tail_diagnostics(run) for label, run in runs.items()}
    report['explanatory_slices'] = explanatory_slices(runs, runs['baseline']['rows'][60])
    ranking = sorted(RECIPES, key=lambda name: (-runs[name]['scores'][60]['success'],
        -runs[name]['scores'][60]['precision'], RECIPES[name]))
    passing = [name for name in ranking if report['candidates'][name]['passed']]
    # 只有完整四组中通过原双门的候选可锁定第二种子，完全同分优先低LR。
    selected = passing[0] if passing else None
    report.update(all_candidate_rank=ranking, passing_rank=passing, selected=selected,
                  selected_lr=None if selected is None else RECIPES[selected], stage2=None)
    report['criterion']['ranking'] = '通过双门者按final60 Success、Precision降序，完全同分取较低LR'
    old_tail = tail_counts(runs['old_w_quarter'])
    risks = {}
    for name in RECIPES:
        current = tail_counts(runs[name])
        delta = {stage: current[stage] - old_tail[stage] for stage in ('final60', 'late3')}
        risks[name] = dict(tail_gt10m=current, old_w_quarter=old_tail, delta_frames=delta,
                           worsened=any(value > 0 for value in delta.values()))
    report['tail_risks'] = risks
    report['full_planning_paused'] = selected is None or risks[selected]['worsened']
    report['full_pause_reasons'] = (['no_score_gate_winner'] if selected is None else
        ['tail_gt10m_worsened'] if risks[selected]['worsened'] else ['seed52_not_verified'])
    # 首阶段通过不代表跨seed稳定；Full仍待第二阶段。
    if selected is not None:
        report['full_planning_paused'] = True
    require((selected_seed52 is None) == (reference_seed52 is None), 'seed52模型与参考必须同时提供')
    if selected_seed52 is not None:
        require(selected is not None, 'seed42无合格胜出者，不能自动进入seed52')
        chosen = read_run(selected_seed52, 'v35', expected_lr=RECIPES[selected], expected_seed=52)
        reference = read_run(reference_seed52, 'reference', expected_lr=1e-4, expected_seed=52)
        require(reference['manifest'].get('reference_protocol') ==
                runs['reference']['manifest'].get('reference_protocol'),
                'seed52 reference_protocol与固定R不一致')
        _same_config(chosen, runs[selected], allowed={'seed'}, label='seed52 locked recipe')
        _same_config(reference, runs['reference'], allowed={'seed'}, label='seed52 reference')
        require(chosen['manifest']['source'] == runs[selected]['manifest']['source'] ==
                reference['manifest']['source'], '第二阶段必须与锁定候选使用同一执行源码')
        second_runs = dict(reference=reference, baseline=runs['baseline'], old_b0=runs['old_b0'],
                           selected_seed52=chosen)
        second = base.compare_loaded(second_runs, candidate_labels=('selected_seed52',),
            comparisons=(('selected_seed52_minus_reference52', 'selected_seed52', 'reference'),))
        second['locked_recipe'] = selected
        second['locked_lr'] = RECIPES[selected]
        second['same_seed_reference'] = 52
        second['explanatory_slices'] = explanatory_slices(second_runs, runs['baseline']['rows'][60])
        second['center_tail_diagnostics'] = dict(selected_seed52=tail_diagnostics(chosen),
                                                 reference_seed52=tail_diagnostics(reference))
        fixed = base.compare_loaded(dict(reference=runs['reference'], baseline=runs['baseline'],
            old_b0=runs['old_b0'], selected_seed52=chosen), candidate_labels=('selected_seed52',),
            comparisons=(('selected_seed52_minus_registered_R', 'selected_seed52', 'reference'),))
        second['fixed_registration'] = dict(fixed['candidates']['selected_seed52'],
            reference_seed=42, baseline_seed=42, criterion=fixed['criterion'],
            reference_overall=fixed['arms']['reference']['groups']['overall'],
            baseline_ever_moving=fixed['arms']['baseline']['groups']['ever_moving'])
        report['stage2'] = second
        matched_pass = second['candidates']['selected_seed52']['passed']
        fixed_pass = second['fixed_registration']['passed']
        stable = matched_pass and fixed_pass
        tail52 = tail_counts(chosen)
        tail52_worse = any(tail52[stage] > old_tail[stage] for stage in ('final60', 'late3'))
        report['stage2_tail_risk'] = dict(tail_gt10m=tail52, old_w_quarter=old_tail, worsened=tail52_worse)
        report['full_pause_reasons'] = ([reason for reason, condition in (
            ('tail_gt10m_worsened', risks[selected]['worsened'] or tail52_worse),
            ('seed52_score_not_confirmed', not matched_pass),
            ('seed52_fixed_registration_not_confirmed', not fixed_pass)) if condition])
        report['full_planning_paused'] = bool(report['full_pause_reasons'])
        report['reproducibility_confirmed'] = stable
        report['automatic_retraining'] = False
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('registered-reference', 'baseline', 'old-b0', 'old-w-quarter', 'quarter', 'half', 'normal', 'scaled'):
        parser.add_argument('--' + name, required=True, type=Path)
    for name in ('selected-seed52', 'reference-seed52', 'output'):
        parser.add_argument('--' + name, type=Path)
    args = vars(parser.parse_args(argv))
    output = args.pop('output')
    try:
        report = compare_runs(**args)
        if output:
            base.write_report(output, report)
        code = 0 if report['status'] == 'passed' else 1
    except (EvidenceError, OSError, ValueError, TypeError, KeyError) as error:
        report = dict(schema=REPORT_SCHEMA, status='invalid_evidence', message=str(error))
        code = 2
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return code


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    raise SystemExit(main())

