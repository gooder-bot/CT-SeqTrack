"""v35 原四组与追加分段调度的五组正式比较；固定源代码增量，原评分门不变。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import compare_v35_b0 as original

base = original.base
EvidenceError, require = original.EvidenceError, original.require
REPORT_SCHEMA = 'ct_seqtrack.v35.piecewise_comparison.v1'
BASELINE_SOURCE_SHA256 = '80eda2f59c5a510578536bbc238a024145eb5e7e575f89b272975702fed1948e'
CHANGED_FILES = ('models/ct_v31/config.py', 'models/ct_v31/entry.py',
                 'models/ct_v31/lr_schedule.py', 'models/ctseqtrackv31.py')
RECIPES = dict(original.RECIPES, piecewise=5e-5)
STAGES = [5e-5, 1e-5, 5e-6]


def load_source_registration():
    return original.read_object(Path(__file__).resolve().parents[1] /
        'cfgs/ct_seqtrack/35_piecewise_source_registration.json')


def verify_schedule_sources(runs):
    registration = load_source_registration()
    before, after = registration['before'], registration['after']
    require(before['sha256'] == BASELINE_SOURCE_SHA256, '原四组源码基准未匹配')
    for source in (before, after):
        require(source['sha256'] == original.digest(source['files']), '登记源码checksum不匹配')
    require(set(before['files']) == set(after['files']), '调度增量不得增加或移除模型源码文件')
    changed = {p for p in before['files'] if before['files'][p] != after['files'][p]}
    require(changed == set(CHANGED_FILES) == set(registration['changed_files']),
            '调度增量包含未登记源码改动')
    for name in original.RECIPES:
        require(runs[name]['manifest']['source'] == before, name + ': 原四组源码不匹配')
    require(runs['piecewise']['manifest']['source'] == after, 'piecewise: 新调度源码不匹配')
    for name in RECIPES:
        validate_resume_sources(runs[name], registration)
    return registration


def validate_resume_sources(run, registration):
    """新增记录只追加；不将旧manifest重写成新版本。"""
    initial = run['manifest']['source']
    effective = initial
    for path in sorted((Path(run['directory']) / 'resume_manifests').glob('*.json')):
        event = original.read_object(path)
        require(event.get('schema') == 'ct_seqtrack.v35.resume_source.v1', 'resume源码记录schema错误')
        require(event.get('config_sha256') == run['manifest']['config_sha256'], 'resume配置身份错误')
        require(event.get('original_source_sha256') == initial['sha256'], 'resume初始源码错误')
        next_source = event.get('source')
        require(next_source in (registration['before'], registration['after']), 'resume存在未登记源码')
        require(not (effective == registration['after'] and next_source != effective), '不允许恢复回旧调度源码')
        effective = next_source
        run['provenance'][str(path.relative_to(run['directory']))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return effective


def compare_runs(*, registered_reference, baseline, old_b0, old_w_quarter,
                 quarter, half, normal, scaled, piecewise, selected_seed52=None, reference_seed52=None):
    paths = dict(quarter=quarter, half=half, normal=normal, scaled=scaled, piecewise=piecewise)
    require(len({str(Path(p).resolve()) for p in paths.values()}) == 5, '需要五个不同的完整seed42候选')
    runs = dict(reference=base.read_run(registered_reference, 'reference'),
        baseline=base.read_run(baseline, 'baseline'), old_b0=base.read_run(old_b0, 'old_b0'),
        old_w_quarter=base.read_run(old_w_quarter, 'W', expected_lr=2.5e-5))
    original.verify_registered_evidence(runs)
    for name, lr in RECIPES.items():
        runs[name] = original.read_run(paths[name], 'v35', expected_lr=lr,
            expected_schedule='piecewise' if name == 'piecewise' else 'multistep')
    registration = verify_schedule_sources(runs)
    original._same_config(runs['piecewise'], runs['half'],
        allowed={'lr_schedule', 'lr_stage_values'}, label='追加调度相对原half')
    comparisons = [(name + '_minus_' + ref, name, target)
                   for name in RECIPES for ref, target in (('R', 'reference'), ('C', 'baseline'))]
    comparisons.append(('piecewise_minus_half', 'piecewise', 'half'))
    # 先独立验证固定源码增量，再复用原评分器；不修改/伪造任何run的source。
    report = base.compare_loaded(runs, candidate_labels=tuple(original.RECIPES),
        config_differences=('lr',), comparisons=comparisons)
    extra = base.compare_loaded({k: runs[k] for k in ('reference', 'baseline', 'old_b0', 'piecewise')},
        candidate_labels=('piecewise',), comparisons=())
    report['candidates']['piecewise'] = extra['candidates']['piecewise']
    report['schema'] = REPORT_SCHEMA
    report['source_revision'] = dict(before=registration['before']['sha256'],
        after=registration['after']['sha256'], changed_files=list(CHANGED_FILES),
        scope='仅分段调度、配方登记、宿主选择与恢复来源记录；原四组配置及调度保持')
    report['registered_frame_sha256'] = original.REGISTERED_FRAME_SHA256
    report['center_tail_diagnostics'] = {k: original.tail_diagnostics(v) for k, v in runs.items()}
    report['explanatory_slices'] = original.explanatory_slices(runs, runs['baseline']['rows'][60])
    rank = sorted(RECIPES, key=lambda n: (-runs[n]['scores'][60]['success'],
        -runs[n]['scores'][60]['precision'], RECIPES[n], n == 'piecewise'))
    passing = [n for n in rank if report['candidates'][n]['passed']]
    selected = passing[0] if passing else None
    report.update(status='passed' if passing else 'failed', all_candidate_rank=rank,
        passing_rank=passing, selected=selected, selected_lr=RECIPES.get(selected),
        selected_schedule=('piecewise' if selected == 'piecewise' else 'multistep') if selected else None,
        stage2=None, automatic_retraining=False)
    report['criterion']['ranking'] = '通过原双门者按final60 S、P、较低初始LR；完全同分同初始LR时原half优先piecewise'
    report['criterion']['piecewise'] = dict(completed_epoch_boundaries=[20, 50], lr_stage_values=STAGES,
        interpretation='第1–20/21–50/51–60轮；与half比较整段调度，不能归因为单个初始LR')
    old_tail = original.tail_counts(runs['old_w_quarter'])
    risks = {}
    for name in RECIPES:
        tail = original.tail_counts(runs[name])
        delta = {s: tail[s] - old_tail[s] for s in ('final60', 'late3')}
        risks[name] = dict(tail_gt10m=tail, old_w_quarter=old_tail,
            delta_frames=delta, worsened=any(v > 0 for v in delta.values()))
    report['tail_risks'] = risks
    report['full_planning_paused'] = True
    report['full_pause_reasons'] = (['no_score_gate_winner'] if selected is None else
        ['tail_gt10m_worsened'] if risks[selected]['worsened'] else ['seed52_not_verified'])
    require((selected_seed52 is None) == (reference_seed52 is None), 'seed52模型与参考必须同时提供')
    if selected_seed52 is not None:
        require(selected is not None, 'seed42无合格胜出者，不能自动进入seed52')
        chosen = original.read_run(selected_seed52, 'v35', expected_lr=RECIPES[selected], expected_seed=52,
            expected_schedule='piecewise' if selected == 'piecewise' else 'multistep')
        reference = original.read_run(reference_seed52, 'reference', expected_lr=1e-4, expected_seed=52)
        require(reference['manifest'].get('reference_protocol') == runs['reference']['manifest'].get('reference_protocol'),
                'seed52 reference_protocol与固定R不一致')
        original._same_config(chosen, runs[selected], allowed={'seed'}, label='seed52 locked recipe')
        original._same_config(reference, runs['reference'], allowed={'seed'}, label='seed52 reference')
        selected_source = validate_resume_sources(runs[selected], registration)
        require(chosen['manifest']['source'] == reference['manifest']['source'] == selected_source,
                '第二阶段必须与锁定候选实际执行源码一致')
        require(validate_resume_sources(chosen, registration) ==
                validate_resume_sources(reference, registration) == selected_source,
                '第二阶段恢复源码必须与锁定候选一致')
        second_runs = dict(reference=reference, baseline=runs['baseline'], old_b0=runs['old_b0'], selected_seed52=chosen)
        second = base.compare_loaded(second_runs, candidate_labels=('selected_seed52',),
            comparisons=(('selected_seed52_minus_reference52', 'selected_seed52', 'reference'),))
        second.update(locked_recipe=selected, locked_lr=RECIPES[selected], same_seed_reference=52,
            locked_schedule=report['selected_schedule'])
        fixed = base.compare_loaded(dict(second_runs, reference=runs['reference']),
            candidate_labels=('selected_seed52',), comparisons=())
        second['fixed_registration'] = dict(fixed['candidates']['selected_seed52'], reference_seed=42,
            baseline_seed=42, criterion=fixed['criterion'],
            reference_overall=fixed['arms']['reference']['groups']['overall'],
            baseline_ever_moving=fixed['arms']['baseline']['groups']['ever_moving'])
        second['explanatory_slices'] = original.explanatory_slices(second_runs, runs['baseline']['rows'][60])
        second['center_tail_diagnostics'] = {k: original.tail_diagnostics(second_runs[k]) for k in ('selected_seed52', 'reference')}
        report['stage2'] = second
        matched, fixed_pass = second['candidates']['selected_seed52']['passed'], second['fixed_registration']['passed']
        tail52 = original.tail_counts(chosen)
        tail_worse = any(tail52[s] > old_tail[s] for s in ('final60', 'late3'))
        report['stage2_tail_risk'] = dict(tail_gt10m=tail52, old_w_quarter=old_tail, worsened=tail_worse)
        report['reproducibility_confirmed'] = matched and fixed_pass
        report['full_pause_reasons'] = [reason for reason, yes in (
            ('tail_gt10m_worsened', risks[selected]['worsened'] or tail_worse),
            ('seed52_score_not_confirmed', not matched),
            ('seed52_fixed_registration_not_confirmed', not fixed_pass)) if yes]
        report['full_planning_paused'] = bool(report['full_pause_reasons'])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('registered-reference', 'baseline', 'old-b0', 'old-w-quarter', *RECIPES):
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
