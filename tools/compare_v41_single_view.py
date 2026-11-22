"""读取已有训练账本，对照四独立候选与单视角 CGP 的训练成本和固定轮次 test 得分。"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.compare_v41_training_arms import EPOCHS, METRICS, epoch_evaluation, read, require, sha
from models.ct_v41.config import config_identity


MODES = ('four_candidates', 'single_view_cgp')
# 两版源码和运行目录可以不同。输入、随机初始化与 E57 库另按内容指纹核对。
RUN_FIELDS = {'cfg', 'log_dir', 'path', 'devices', 'checkpoint', 'tag', 'experiment_name',
              'v41_history_training', 'v41_history_bank', 'v41_fixed_replay_bank'}


def training_run(root, mode):
    manifest = read(root / 'run_manifest.json')
    config = yaml.safe_load((root / 'resolved_config.yaml').read_text(encoding='utf-8-sig'))
    recipe = manifest['actual_recipe']
    require(manifest['run_type'] == 'formal' and config['test'] is False
            and config_identity(config) == manifest['config_sha256']
            and config['v41_history_training'] == recipe['v41_history_training'] == mode
            and all(config.get(key) == value for key, value in recipe.items()), f'{root}: 训练配方不匹配')
    require(config['eval_split'] == 'test' and config['check_val_every_n_epoch'] == 1
            and config['epoch'] == 60 and config['report_checkpoint'] == 'last'
            and config['v41_joint_search'] and config['v41_joint_replay'] == 'fixed_e2',
            f'{root}: 需要六候选联合模型、逐轮 test 和固定 E60')
    available = sorted(int(path.stem.removeprefix('epoch=')) for path in
                       (root / 'epochs').glob('epoch=[0-9][0-9][0-9].json'))
    require(available == list(range(1, len(available) + 1)) and len(available) <= 60,
            f'{root}: 已保存的轮次不连续')
    return manifest, {k: v for k, v in config.items() if k not in RUN_FIELDS}, len(available)


def epoch_row(root, mode, manifest, epoch):
    report = read(root / 'epochs' / f'epoch={epoch:03d}.json')
    evaluation = epoch_evaluation(root, 'test', epoch)
    audit = read(root / 'training_audits' / f'epoch={epoch:03d}.json')
    history, sampler = audit['history_training'], audit['sampler']
    require(audit['run_id'] == manifest['run_id'] and audit['config_sha256'] == manifest['config_sha256']
            and audit['completed_epoch'] == epoch and audit['epoch_complete'] and audit['auxiliary']['complete']
            and history['mode'] == mode and history['bank_sha256'] == manifest['history_bank_sha256'],
            f'{root}: E{epoch} 训练记录不完整或不属于此运行')
    endpoints = sum(max(0, length - 1) for length in sampler['lengths'])
    views = 4 if mode == 'four_candidates' else 1
    require(audit['rows'] == sampler['rows'] == views * endpoints and audit['rows'] > 0
            and audit['optimizer_steps'] == sampler['batches'] > 0, f'{root}: E{epoch} 主训练预算异常')
    require(all(0 <= evaluation[k] <= 100 for k in METRICS), f'{root}: E{epoch} 得分无效')
    losses = report['losses']
    # 旧四候选账本只记录每次受扰重置的供体匹配结果；包括退回原末框扰动的样本。
    perturbed = sum(history['augmentation'].values())
    reset = history['endpoints'].get('reset', 0)
    replay = sum(audit['joint_replay']['used'].values())
    require(0 <= perturbed <= reset and reset + replay == audit['rows']
            and replay == history['endpoints'].get('natural_replay', 0),
            f'{root}: E{epoch} 重置和回放计数不符')
    return dict(main_rows=audit['rows'], optimizer_steps=audit['optimizer_steps'],
                aux_rows=audit['auxiliary']['rows'], replay_rows=replay,
                weights_sha256=history.get('weights_sha256'), e2_donor=audit['joint_replay']['donor_weights_sha256'],
                clean_resets=reset - perturbed, perturbed_resets=perturbed,
                ordinary_reset_seeds=history.get('ordinary_reset_seeds'),
                augmentation=history['augmentation'], train_seconds=report['timing']['train_seconds'],
                eval_seconds=report['timing']['eval_seconds'],
                main_loss=losses.get('train/loss_total_epoch', losses.get('train/loss_total')),
                optimization_loss=losses.get('train/optimization_loss_epoch', losses.get('train/optimization_loss')),
                test_frames=evaluation['frames'], success=evaluation['success'], precision=evaluation['precision'],
                over10m_frames=evaluation.get('diagnostics', {}).get('over10m_frames'))


def compare(candidate_run, single_run):
    roots = dict(zip(MODES, (candidate_run, single_run)))
    runs = {mode: training_run(root, mode) for mode, root in roots.items()}
    baseline, single = (runs[m][0] for m in MODES)
    configs = [runs[m][1] for m in MODES]
    differences = sorted(k for k in configs[0].keys() | configs[1].keys()
                         if k not in configs[0] or k not in configs[1] or configs[0][k] != configs[1][k])
    require(not differences, f'除训练视角外，配置还有差异: {differences}')
    for key in ('history_bank_sha256', 'initial_parameter_hash', 'dataset_manifest', 'input_metadata'):
        require(baseline[key] == single[key], f'两组 {key} 不一致')
    require(sha(single['history_bank_sha256']) and sha(single['initial_parameter_hash'])
            and baseline['run_id'] != single['run_id'], 'E57、初始化或运行身份无效')
    completed = {m: runs[m][2] for m in MODES}
    common = min(completed.values())
    require(common > 0, '两组至少要各有一轮完整训练和 test 记录')
    curve, e2_weights = [], {}
    for epoch in range(1, common + 1):
        rows = {m: epoch_row(roots[m], m, runs[m][0], epoch) for m in MODES}
        for mode, row in rows.items():
            if epoch == 2:
                e2_weights[mode] = row['weights_sha256']
                require(sha(e2_weights[mode]), f'{mode}: 缺少 E2 权重指纹')
            if epoch <= 2:
                require(row['e2_donor'] is None and row['replay_rows'] == 0, f'{mode}: E{epoch} 提前回放')
            else:
                require(row['e2_donor'] == e2_weights[mode], f'{mode}: E{epoch} 回放供体与本组 E2 不符')
        left, right = (rows[m] for m in MODES)
        require(left['test_frames'] == right['test_frames'] and left['main_rows'] == 4 * right['main_rows']
                and left['aux_rows'] == right['aux_rows'], f'E{epoch} 数据或端点覆盖不一致')
        curve.append(dict(epoch=epoch, **rows, single_minus_four_pp={k: right[k] - left[k] for k in METRICS}))

    totals = {m: {k: sum(row[m][k] for row in curve)
                  for k in ('main_rows', 'optimizer_steps', 'train_seconds', 'eval_seconds', 'aux_rows',
                            'replay_rows', 'clean_resets', 'perturbed_resets')}
              for m in MODES}
    require(all(math.isfinite(v['train_seconds']) and v['train_seconds'] >= 0 for v in totals.values()),
            '训练耗时无效')
    late = None
    if common == 60:
        late = {m: {k: fmean(curve[e - 1][m][k] for e in EPOCHS) for k in METRICS} for m in MODES}
        late['single_minus_four_pp'] = {k: late[MODES[1]][k] - late[MODES[0]][k] for k in METRICS}
    return dict(split='test', completed_epochs=completed, common_epochs=common,
                source_sha256={m: runs[m][0]['source']['sha256'] for m in MODES}, totals=totals, epochs=curve,
                e60=(curve[-1] if common == 60 else None), late3=late)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-run', type=Path, required=True, help='已有 four_candidates 目录')
    parser.add_argument('--single-run', type=Path, required=True, help='新 single_view_cgp 目录')
    args = parser.parse_args()
    print(json.dumps(compare(args.candidate_run, args.single_run), indent=2, allow_nan=False))
