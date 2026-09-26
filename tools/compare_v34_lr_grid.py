"""只读比较八组 mini seed42：两组 SeqTrack 与 S/W 的三档学习率。

晋级门始终来自 --registered-reference（既有 R）与 --baseline（既有 C）；
本轮 reference/reference-half 只增加对照，不随重跑分数改动登记门槛。
0=至少一组通过，1=完整证据均未通过，2=证据无效或不完整。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import compare_v34_b0 as base


REPORT_SCHEMA = 'ct_seqtrack.v34.b0_lr_grid.v1'
RECIPES = dict(normal=1e-4, half=5e-5, quarter=2.5e-5)
CANDIDATES = tuple(f'{arm}_{recipe}' for arm in ('S', 'W') for recipe in RECIPES)
ARGUMENTS = ('registered-reference', 'reference', 'reference-half', 'baseline', 'old-b0',
             's-normal', 's-half', 's-quarter', 'w-normal', 'w-half', 'w-quarter')


def compare_grid(*, registered_reference, reference, reference_half, baseline, old_b0,
                 s_normal, s_half, s_quarter, w_normal, w_half, w_quarter):
    inputs = dict(registered_reference=(registered_reference, 'reference', 1e-4),
                  reference=(reference, 'reference', 1e-4),
                  reference_half=(reference_half, 'reference', 5e-5),
                  baseline=(baseline, 'baseline', 5e-5), old_b0=(old_b0, 'old_b0', 1e-4))
    paths = dict(S_normal=s_normal, S_half=s_half, S_quarter=s_quarter,
                 W_normal=w_normal, W_half=w_half, W_quarter=w_quarter)
    for label, path in paths.items():
        arm, recipe = label.split('_')
        inputs[label] = (path, arm, RECIPES[recipe])
    runs = {label: base.read_run(path, role, expected_lr=lr)
            for label, (path, role, lr) in inputs.items()}
    # R1/R05 的框架、日程与数据保持相同；它们不要求与历史 R 源码相同。
    allowed = base.EXCLUDED_IDENTITY | {'experiment_name', 'lr'}
    reference_configs = [{k: v for k, v in runs[label]['config'].items() if k not in allowed}
                         for label in ('reference', 'reference_half')]
    base.require(reference_configs[0] == reference_configs[1], '本轮 R1/R05 除学习率与标签外配置不同')
    base.require(runs['reference']['manifest']['source'] == runs['reference_half']['manifest']['source'],
                 '本轮 R1/R05 执行源码不同')
    comparisons = [('R05_minus_R1', 'reference_half', 'reference'),
                   ('R1_minus_registered_R', 'reference', 'registered_reference')]
    for label in CANDIDATES:
        comparisons.extend([(label + '_minus_C', label, 'baseline'),
                            (label + '_minus_registered_R', label, 'registered_reference'),
                            (label + '_minus_R1', label, 'reference'),
                            (label + '_minus_R05', label, 'reference_half')])
    for recipe in RECIPES:
        comparisons.append((f'W_minus_S_{recipe}', 'W_' + recipe, 'S_' + recipe))
    report = base.compare_loaded(runs, candidate_labels=CANDIDATES,
        reference_label='registered_reference', config_differences=('lr',), comparisons=comparisons)
    report['schema'] = REPORT_SCHEMA
    report['recipes'] = dict(reference=dict(lr=1e-4, scheduler='StepLR', step_size=20),
                            reference_half=dict(lr=5e-5, scheduler='StepLR', step_size=20),
                            context=dict(learning_rates=RECIPES, scheduler='MultiStepLR',
                                         milestones=[20, 50], warmup_steps=0))
    report['criterion'].update(
        overall='final60/late3 S/P >= registered_reference（既有 R）',
        rerun_references='本轮 R1/R05 为额外对照，不能替换登记 R 的四项硬门',
        ranking='六组均列出固定 final60 Success、Precision 降序；晋级仅取双门通过者')
    arms = report['arms']
    report['all_candidate_rank'] = sorted(CANDIDATES, key=lambda label: (
        -arms[label]['groups']['overall']['final60']['success'],
        -arms[label]['groups']['overall']['final60']['precision'], label))
    report['limitations'].append('三档 LR 共用一个 seed；网格选出的分数不能表述为跨 seed 的稳定提升。')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ARGUMENTS:
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--output', type=Path)
    args = vars(parser.parse_args(argv))
    output = args.pop('output')
    try:
        report = compare_grid(**args)
        if output:
            base.write_report(output, report)
        code = 0 if report['status'] == 'passed' else 1
    except (base.EvidenceError, OSError) as error:
        report = dict(schema=REPORT_SCHEMA, status='invalid_evidence', message=str(error))
        code = 2
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return code


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    raise SystemExit(main())
