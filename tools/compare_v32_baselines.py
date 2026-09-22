"""只读核验四次 mini 基线实验；0=达标，1=未达标，2=证据不完整或不匹配。"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

import yaml

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.tracking_metrics import metric_contributions


MODEL_SCHEMA = 'ct_seqtrack.joint_identity.v32'
REPORT_SCHEMA = 'ct_seqtrack.v32.baseline_comparison.v1'
EPOCHS = [58, 59, 60]
COUNTS = dict(tracklets=106, frames=2285, prediction_frames=2179)
COMMON_CONFIG = dict(
    experiment_family='ct_seqtrack_v32', dataset='nuscenes_mf', version='v1.0-mini',
    category_name='Car', v31_arm='b0', ct_coordinate_mode='global', ct_frame_stride=1,
    ct_partition_seed=42, hist_num=3, point_sample_size=1024, bb_scale=1.25, bb_offset=2.,
    batch_size=16, workers=4, epoch=60, precision=32, trainer_devices=1,
    lr=.0001, wd=0., lr_decay_step=20, lr_decay_rate=.1, check_val_every_n_epoch=5,
    dynamics_time_mode='true', limit_train_batches=1., limit_val_batches=1.,
    time_scale=.5, v31_short_window=3, v31_long_window=8, v31_curriculum_epochs=10,
    v32_reserve_windows=112, v32_seed_translation=.3, v32_seed_yaw_degrees=1.5,
)
BUDGET = dict(completed_epoch=60, last_epoch_rows=19108,
              last_epoch_steps=1195, optimizer_steps=71700)


class EvidenceError(ValueError):
    """输入不能支持这次预注册比较，而不是模型已经被判定为未达标。"""


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def read_object(path, *, yaml_file=False):
    try:
        text = path.read_text(encoding='utf-8')
        value = yaml.safe_load(text) if yaml_file else json.loads(text)
    except (OSError, UnicodeError, ValueError, yaml.YAMLError) as error:
        raise EvidenceError(f'{path}: 无法读取有效证据: {error}') from error
    require(isinstance(value, dict), f'{path}: 应为对象')
    return value


def number(value, label):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value), f'{label}: 必须是有限数值')
    return float(value)


def equal_fields(value, expected, label):
    for key, target in expected.items():
        actual = value.get(key)
        require(not isinstance(actual, bool) and actual == target,
                f'{label}.{key}: 需要 {target!r}，得到 {actual!r}')


def close(actual, expected, label, tolerance=1e-6):
    actual = number(actual, label)
    require(abs(actual - expected) <= tolerance,
            f'{label}: 保存值 {actual} 与重算值 {expected} 不一致')


def check_metrics(metrics, epoch, label):
    equal_fields(metrics, dict(schema=MODEL_SCHEMA, checkpoint_epoch=epoch,
                               metric_mode='benchmark_compat', **COUNTS), label)
    require(metrics.get('complete_coverage') is True, f'{label}.complete_coverage: 必须为 true')
    for key in ('success', 'precision'):
        require(0 <= number(metrics.get(key), f'{label}.{key}') <= 100,
                f'{label}.{key}: 必须在 0..100 内')


def frame_scores(path):
    keys, frames_by_track = set(), defaultdict(list)
    observed = []
    try:
        with path.open(encoding='utf-8') as stream:
            for line_number, line in enumerate(stream, 1):
                label = f'{path}:{line_number}'
                require(bool(line.strip()), f'{label}: 不允许空记录')
                row = json.loads(line)
                require(isinstance(row, dict), f'{label}: 每行应为对象')
                track, frame = row.get('tracklet'), row.get('frame')
                require(isinstance(track, str) and bool(track), f'{label}: 缺少 tracklet')
                require(isinstance(frame, int) and not isinstance(frame, bool) and frame >= 0,
                        f'{label}: frame 必须为非负整数')
                key = (track, frame)
                require(key not in keys, f'{label}: 重复帧键 {key}')
                keys.add(key)
                frames_by_track[track].append(frame)
                require(row.get('initialization') is (frame == 0),
                        f'{label}: initialization 必须且只能对应 frame=0')
                iou = number(row.get('iou'), f'{label}.iou')
                distance = number(row.get('distance'), f'{label}.distance')
                require(0 <= iou <= 1 and distance >= 0, f'{label}: IoU/距离范围错误')
                if frame == 0:
                    require(iou == 1. and distance == 0., f'{label}: 初始化帧必须为 GT 框')
                observed.append((iou, distance, row.get('success'), row.get('precision')))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f'{path}: 无法读取有效逐帧记录: {error}') from error
    require(len(keys) == COUNTS['frames'] and len(frames_by_track) == COUNTS['tracklets'],
            f'{path}: 需要 106 轨迹/2285 帧，得到 {len(frames_by_track)}/{len(keys)}')
    for track, frames in frames_by_track.items():
        require(sorted(frames) == list(range(len(frames))),
                f'{path}: 轨迹 {track} 存在缺失的初始化或预测端点')
    successes, precisions = metric_contributions([row[0] for row in observed],
                                                [row[1] for row in observed])
    for index, (row, success, precision) in enumerate(zip(observed, successes, precisions), 1):
        close(row[2], float(success), f'{path}:{index}.success', 1e-8)
        close(row[3], float(precision), f'{path}:{index}.precision', 1e-8)
    return keys, dict(success=100 * math.fsum(successes) / len(keys),
                      precision=100 * math.fsum(precisions) / len(keys))


def read_run(directory, model, seed):
    root = Path(directory).resolve()
    label = str(root)
    manifest = read_object(root / 'run_manifest.json')
    equal_fields(manifest, dict(schema=MODEL_SCHEMA, model=model, seed=seed, arm='b0'),
                 label + '/run_manifest')
    require(manifest.get('enabled') == dict(B1=False, B2=False, B3=False),
            f'{label}/run_manifest.enabled: 基线不能启用 B1/B2/B3')
    config = read_object(root / 'resolved_config.yaml', yaml_file=True)
    equal_fields(config, dict(COMMON_CONFIG, net_model=model, seed=seed), label + '/resolved_config')
    require(config.get('ct_engineering_check') is False,
            f'{label}/resolved_config.ct_engineering_check: 必须为 false，工程结果不能验收')
    require(config.get('test') is False and config.get('init_checkpoint') is None,
            f'{label}/resolved_config: 必须为 scratch 正式训练或同身份恢复目录')
    require(config.get('v31_evaluate_late3') is True,
            f'{label}/resolved_config.v31_evaluate_late3: 必须完成 58/59/60 评测')
    budget = read_object(root / 'training_budget.json')
    equal_fields(budget, dict(schema=MODEL_SCHEMA, **BUDGET), label + '/training_budget')
    require(budget.get('epoch_complete') is True, f'{label}/training_budget.epoch_complete: 必须为 true')
    sampler = budget.get('sampler')
    require(isinstance(sampler, dict), f'{label}/training_budget.sampler: 缺少训练数据身份')
    source = sampler.get('source_sha256')
    require(isinstance(source, str) and len(source) == 64
            and all(character in '0123456789abcdef' for character in source),
            f'{label}/training_budget.sampler.source_sha256: 必须为 SHA256')
    results = read_object(root / 'results.json')
    require(results.get('checkpoint_epochs') == EPOCHS,
            f'{label}/results.checkpoint_epochs: 必须恰好为 [58, 59, 60]')
    require(isinstance(results.get('final'), dict) and isinstance(results.get('late3'), dict),
            f'{label}/results: 缺少 final 或 late3')
    check_metrics(results['final'], 60, label + '/results.final')
    metrics_by_epoch, canonical_keys = {}, None
    for epoch in EPOCHS:
        evaluation = root / 'evaluation' / f'epoch={epoch:03d}'
        metrics = read_object(evaluation / 'metrics.json')
        check_metrics(metrics, epoch, str(evaluation / 'metrics.json'))
        keys, recomputed = frame_scores(evaluation / 'frames.jsonl')
        if canonical_keys is None:
            canonical_keys = keys
        require(keys == canonical_keys, f'{label}: 58/59/60 的逐帧键集合不一致')
        for key, score in recomputed.items():
            close(metrics.get(key), score, f'{evaluation}/metrics.{key}')
        metrics_by_epoch[epoch] = recomputed
    for key in ('success', 'precision'):
        close(results['final'].get(key), metrics_by_epoch[60][key], f'{label}/results.final.{key}')
        close(results['late3'].get(key), math.fsum(metrics_by_epoch[e][key] for e in EPOCHS) / 3,
              f'{label}/results.late3.{key}')
    return dict(directory=label, source_sha256=source, keys=canonical_keys,
                final=metrics_by_epoch[60], late3=results['late3'])


def compare_runs(b0_42, ref_42, b0_52, ref_52):
    runs = {}
    for seed, b0, reference in ((42, b0_42, ref_42), (52, b0_52, ref_52)):
        runs[seed] = (read_run(b0, 'ctseqtrackv32', seed),
                      read_run(reference, 'seqtrack_reference', seed))
    all_keys = runs[42][0]['keys']
    pairs, failures = {}, []
    for seed, (b0, reference) in runs.items():
        require(b0['source_sha256'] == reference['source_sha256'],
                f'seed{seed}: B0/reference 的训练 source_sha256 不一致')
        require(b0['keys'] == reference['keys'] == all_keys,
                f'seed{seed}: B0/reference/双种子的评测逐帧键集合不一致')
        deltas = {key: b0['final'][key] - reference['final'][key] for key in ('success', 'precision')}
        checks = {key: difference >= -2. - 1e-9 for key, difference in deltas.items()}
        failures.extend(f'seed{seed}.{key}: B0-reference={deltas[key]:.6f}pp < -2pp'
                        for key, passed in checks.items() if not passed)
        pairs[str(seed)] = dict(
            b0_directory=b0['directory'], reference_directory=reference['directory'],
            source_sha256=b0['source_sha256'], b0_final60=b0['final'],
            reference_final60=reference['final'], delta_pp=deltas, passed=all(checks.values()),
            b0_late3=b0['late3'], reference_late3=reference['late3'])
    return dict(schema=REPORT_SCHEMA, status='failed' if failures else 'passed',
        criterion=dict(checkpoint_epoch=60, metrics=['success', 'precision'], seeds=[42, 52],
                       maximum_drop_pp=2., every_seed_and_metric_required=True),
        verified=dict(checkpoint_epochs=EPOCHS, **COUNTS, **BUDGET),
        pairs=pairs, failed_checks=failures)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('b0-42', 'ref-42', 'b0-52', 'ref-52'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = compare_runs(args.b0_42, args.ref_42, args.b0_52, args.ref_52)
        code = 0 if report['status'] == 'passed' else 1
    except EvidenceError as error:
        report = dict(schema=REPORT_SCHEMA, status='invalid_evidence', message=str(error))
        code = 2
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return code


if __name__ == '__main__':
    # Windows 重定向默认编码可能是 GBK；JSON stdout 统一为 UTF-8。
    sys.stdout.reconfigure(encoding='utf-8')
    raise SystemExit(main())
