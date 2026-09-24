"""只读比较 reference/旧B0/v33 A/B/C；打印 JSON，不训练、不推理、不写证据。

0=至少一组 final60/late-3 四项均不低于 reference，1=均未达到，2=证据无效。
相邻 checkpoint 和同一 seed 的三个 LR 配方不构成独立复验。
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.ct_v31.runtime import loss_episode_summary
from tools.compare_v32_baselines import (BUDGET, COMMON_CONFIG, COUNTS, EPOCHS,
    EvidenceError, close, equal_fields, number, read_object, require)
from utils.tracking_metrics import metric_contributions

MODEL_SCHEMA = 'ct_seqtrack.joint_identity.v33'
OLD_SCHEMA = 'ct_seqtrack.joint_identity.v32'
REPORT_SCHEMA = 'ct_seqtrack.v33.b0_comparison.v1'
# 已保存GPU FP32积分与CPU阈值重算可有约3e-8归约舍入差。
# 2e-7仅容纳FP32数值差，远小于逐帧贡献的最小0.025档；总分仍按原记录核验。
PER_FRAME_CONTRIBUTION_TOLERANCE = 2e-7
RECIPES = {'A': (1e-4, 'step', []), 'B': (1e-4, 'multistep', [20, 50]),
           'C': (5e-5, 'multistep', [20, 50])}
DIAGNOSTIC_FIELDS = ('diagnostic_target_count', 'diagnostic_crop_target_count',
    'diagnostic_sampled_target_count', 'diagnostic_crop_point_count',
    'diagnostic_sampled_point_count', 'diagnostic_gt_xy_displacement',
    'diagnostic_empty_crop', 'diagnostic_background_only_crop')


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_frames(path):
    rows, tracks = {}, defaultdict(list)
    try:
        with path.open(encoding='utf-8') as stream:
            for line_number, line in enumerate(stream, 1):
                row = json.loads(line)
                label = f'{path}:{line_number}'
                require(isinstance(row, dict), f'{label}: 应为逐帧对象')
                track, frame = row.get('tracklet'), row.get('frame')
                require(isinstance(track, str) and bool(track), f'{label}: 缺 tracklet')
                require(type(frame) is int and frame >= 0, f'{label}: frame 非法')
                key = (track, frame)
                require(key not in rows, f'{label}: 重复帧键')
                require(row.get('initialization') is (frame == 0), f'{label}: initialization 不一致')
                iou, distance = number(row.get('iou'), label), number(row.get('distance'), label)
                require(0 <= iou <= 1 and distance >= 0, f'{label}: IoU/距离越界')
                if frame == 0:
                    require(iou == 1 and distance == 0, f'{label}: 非法初始化')
                success, precision = metric_contributions(iou, distance)
                close(row.get('success'), float(success), label + '.success', PER_FRAME_CONTRIBUTION_TOLERANCE)
                close(row.get('precision'), float(precision), label + '.precision', PER_FRAME_CONTRIBUTION_TOLERANCE)
                rows[key], tracks[track] = row, tracks[track] + [frame]
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f'{path}: 无法读取逐帧记录: {error}') from error
    require(len(rows) == COUNTS['frames'] and len(tracks) == COUNTS['tracklets'],
            f'{path}: 帧/轨迹分母不匹配')
    for track, frames in tracks.items():
        require(sorted(frames) == list(range(len(frames))), f'{path}: {track} 缺失端点')
    require(sum(key[1] > 0 for key in rows) == COUNTS['prediction_frames'], f'{path}: 预测分母不匹配')
    return rows


def score(rows):
    values = list(rows)
    return {name: 100 * math.fsum(row[name] for row in values) / len(values) if values else None
            for name in ('success', 'precision')}


def _check_metrics(value, schema, epoch, label):
    equal_fields(value, dict(schema=schema, checkpoint_epoch=epoch,
        metric_mode='benchmark_compat', **COUNTS), label)
    require(value.get('complete_coverage') is True, label + ': 评测不完整')


def _check_diagnostics(row, label):
    require(type(row.get('coarse_valid')) is bool, f'{label}: 缺少 coarse_valid')
    for name in DIAGNOSTIC_FIELDS:
        require(name in row, f'{label}: 缺少 v33 被动字段 {name}')
    for name in DIAGNOSTIC_FIELDS[:5]:
        require(type(row[name]) is int and row[name] >= 0, f'{label}: {name} 必须为非负整数')
    raw, crop, sampled = [row[name] for name in DIAGNOSTIC_FIELDS[:3]]
    require(raw >= crop >= sampled, f'{label}: raw/crop/sampled FG 计数不守恒')
    crop_points, sampled_points = [row[name] for name in DIAGNOSTIC_FIELDS[3:5]]
    require(crop_points >= sampled_points >= sampled and crop_points >= crop,
            f'{label}: 有效点分母错误')
    require(row['diagnostic_empty_crop'] is (crop_points == 0), f'{label}: 空 crop 标签错误')
    require(row['diagnostic_background_only_crop'] is (crop_points > 0 and crop == 0),
            f'{label}: 纯背景 crop 标签错误')
    require(number(row['diagnostic_gt_xy_displacement'], label) >= 0, f'{label}: 位移为负')
    number(row.get('timestamp'), label + '.timestamp')
    for prefix in ('coarse_geometry', 'fine_geometry'):
        geometry = row.get(prefix)
        require(isinstance(geometry, dict), f'{label}: 缺少 {prefix}')
        for name in ('iou', 'center_error_m', 'center_xy_error_m', 'yaw_error_rad', 'axis_yaw_error_rad'):
            require(number(geometry.get(name), label + '.' + prefix + '.' + name) >= 0,
                    f'{label}: 几何误差为负')
        require(geometry['iou'] <= 1, f'{label}: 几何 IoU 越界')


def read_run(directory, label, seed):
    root = Path(directory).resolve()
    manifest = read_object(root / 'run_manifest.json')
    schema = manifest.get('schema')
    model = 'seqtrack_reference' if label == 'reference' else ('ctseqtrackv32' if label == 'old_b0' else 'ctseqtrackv33')
    allowed_schema = (OLD_SCHEMA, MODEL_SCHEMA) if label == 'reference' else ((OLD_SCHEMA,) if label == 'old_b0' else (MODEL_SCHEMA,))
    require(schema in allowed_schema, f'{label}: schema 不符')
    equal_fields(manifest, dict(model=model, seed=seed, arm='b0'), label + '.manifest')
    require(manifest.get('enabled') == dict(B1=False, B2=False, B3=False), label + ': 只能为 B0/reference')
    config = read_object(root / 'resolved_config.yaml', yaml_file=True)
    common = {key: value for key, value in COMMON_CONFIG.items()
              if key not in ('experiment_family', 'lr')}
    family = 'ct_seqtrack_v33' if schema == MODEL_SCHEMA else 'ct_seqtrack_v32'
    equal_fields(config, dict(common, experiment_family=family, seed=seed, net_model=model), label + '.config')
    require(config.get('ct_engineering_check') is False and config.get('test') is False
            and config.get('init_checkpoint') is None and config.get('v31_evaluate_late3') is True,
            label + ': 工程/单独评测/初始化权重不能作为正式训练')
    recipe = RECIPES.get(label, RECIPES['A'])
    actual = (config.get('lr'), config.get('lr_schedule', 'step'), config.get('lr_milestones', []))
    require(actual == recipe, f'{label}: LR 配方不匹配 {actual!r} != {recipe!r}')
    budget = read_object(root / 'training_budget.json')
    equal_fields(budget, dict(schema=schema, **BUDGET), label + '.budget')
    require(budget.get('epoch_complete') is True, label + ': epoch 未完整结束')
    source = budget.get('sampler', {}).get('source_sha256')
    require(isinstance(source, str) and len(source) == 64
            and all(c in '0123456789abcdef' for c in source), label + ': 缺训练数据 SHA')
    provenance = {}
    for path in (root / 'run_manifest.json', root / 'resolved_config.yaml', root / 'training_budget.json', root / 'results.json'):
        try:
            provenance[str(path)] = _sha(path)
        except OSError as error:
            raise EvidenceError(str(error)) from error
    for epoch in range(1, 61):
        path = root / 'training_audits' / f'epoch={epoch:03d}.json'
        audit = read_object(path)
        equal_fields(audit, dict(completed_epoch=epoch, rows=19108, optimizer_steps=1195), str(path))
        require(audit.get('epoch_complete') is True, f'{path}: epoch 未完成')
        require(audit.get('sampler', {}).get('source_sha256') == source, f'{path}: source 不一致')
        provenance[str(path)] = _sha(path)
    results = read_object(root / 'results.json')
    require(results.get('checkpoint_epochs') == EPOCHS, label + ': 必须有58/59/60')
    _check_metrics(results.get('final', {}), schema, 60, label + '.final')
    rows_by_epoch, scores, canonical = {}, {}, None
    for epoch in EPOCHS:
        directory = root / 'evaluation' / f'epoch={epoch:03d}'
        rows = read_frames(directory / 'frames.jsonl')
        metrics = read_object(directory / 'metrics.json')
        _check_metrics(metrics, schema, epoch, str(directory))
        if canonical is None:
            canonical = set(rows)
        require(set(rows) == canonical, label + ': checkpoint帧键不一致')
        if label in RECIPES:
            for key, row in rows.items():
                if key[1]:
                    _check_diagnostics(row, f'{label}.{epoch}.{key}')
        scores[epoch] = score(rows.values())
        for name, value in scores[epoch].items():
            close(metrics.get(name), value, f'{label}.{epoch}.{name}')
        rows_by_epoch[epoch] = rows
        for name in ('frames.jsonl', 'metrics.json'):
            provenance[str(directory / name)] = _sha(directory / name)
    late3 = {name: math.fsum(scores[e][name] for e in EPOCHS) / 3 for name in ('success', 'precision')}
    for name in late3:
        close(results['final'].get(name), scores[60][name], label + '.final.' + name)
        close(results.get('late3', {}).get(name), late3[name], label + '.late3.' + name)
    return dict(directory=str(root), schema=schema, manifest=manifest, config=config,
                source=source, rows=rows_by_epoch, scores=scores, late3=late3, provenance=provenance)


def _supplement(path, canonical):
    """显式读取既有逐帧诊断；也接收9/23 original首预测probe，不扩充覆盖。"""
    if path is None:
        return {}
    observed = {}
    try:
        for line in Path(path).read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            if row.get('event') not in (None, 'frame') or row.get('variant') not in (None, 'original'):
                continue
            key = (row.get('tracklet'), row.get('frame'))
            require(key in canonical and key[1] > 0, f'{path}: 补充帧键不属于正式预测集合')
            require(key not in observed, f'{path}: 重复补充帧键')
            aliases = dict(raw_target_count='diagnostic_target_count',
                           crop_target_count='diagnostic_crop_target_count',
                           sampled_target_count='diagnostic_sampled_target_count',
                           current_point_count='diagnostic_sampled_point_count',
                           physical_displacement='diagnostic_gt_xy_displacement')
            observed[key] = {aliases.get(name, name): value for name, value in row.items()}
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f'{path}: 无法读取补充诊断: {error}') from error
    return observed


def fixed_groups(old_rows, supplement, motion_rows):
    groups = defaultdict(set)
    coverage = defaultdict(int)
    first_raw = {key[0]: row.get('diagnostic_target_count',
                 supplement.get(key, {}).get('diagnostic_target_count'))
                 for key, row in old_rows.items() if key[1] == 1}
    for key, row in old_rows.items():
        if not key[1]:
            continue
        groups['all_predictions'].add(key)
        first_count = first_raw.get(key[0])
        if first_count is not None:
            groups['first_raw_zero_tracks' if first_count == 0 else 'first_raw_positive_tracks'].add(key)
        if key[1] == 1:
            groups['first_predictions'].add(key)
        raw = row.get('diagnostic_target_count')
        crop = row.get('diagnostic_crop_target_count')
        if crop is None and raw is not None and 'diagnostic_novel_target_count' in row:
            crop = raw - row['diagnostic_novel_target_count']
        extra = supplement.get(key, {})
        for field, expected in (('diagnostic_target_count', raw), ('diagnostic_crop_target_count', crop)):
            if field in extra and expected is not None:
                require(extra[field] == expected, f'{key}: 补充 {field} 与旧B0不一致')
        raw = extra.get('diagnostic_target_count', raw)
        crop = extra.get('diagnostic_crop_target_count', crop)
        require(type(raw) is int and type(crop) is int and raw >= crop >= 0,
                f'{key}: 旧B0需 raw/crop FG 计数，可通过 --old-b0-diagnostics 补充')
        groups['raw_zero' if raw == 0 else 'raw_positive'].add(key)
        groups['old_b0_crop_zero' if crop == 0 else 'old_b0_crop_positive'].add(key)
        if raw > 0 and crop == 0:
            groups['old_b0_missed_crop'].add(key)
        if key[1] == 1:
            groups['first_raw_zero' if raw == 0 else 'first_raw_positive'].add(key)
        # sampled 点数为0与 raw crop为空等价；不把截断后点数称为 raw数量。
        points = extra.get('diagnostic_crop_point_count', row.get('diagnostic_crop_point_count'))
        if points is None:
            points = extra.get('diagnostic_sampled_point_count', row.get('diagnostic_sampled_point_count'))
        if points is not None:
            coverage['old_crop_empty_background'] += 1
            if points == 0:
                groups['old_b0_empty_crop'].add(key)
            elif crop == 0:
                groups['old_b0_background_only_crop'].add(key)
        displacement = motion_rows[key]['diagnostic_gt_xy_displacement']
        motion_group = 'moving' if displacement >= .15 else 'near_static'
        groups[motion_group].add(key)
        groups[motion_group + ('_raw_zero' if raw == 0 else '_raw_positive')].add(key)
        if raw > 0 and crop == 0:
            groups[motion_group + '_old_b0_missed_crop'].add(key)
    return groups, dict(coverage)


def _geometry_means(rows):
    values = list(rows)
    result = {}
    for prefix in ('coarse_geometry', 'fine_geometry'):
        available = [row[prefix] for row in values if isinstance(row.get(prefix), dict)
                     and (prefix != 'coarse_geometry' or row.get('coarse_valid') is True)]
        result[prefix] = dict(count=len(available), mean={name: float(np.mean([row[name] for row in available]))
            for name in ('iou', 'center_error_m', 'center_xy_error_m', 'yaw_error_rad')}) if available else None
    paired = [(row['coarse_geometry'], row['fine_geometry']) for row in values
              if row.get('coarse_valid') is True and isinstance(row.get('coarse_geometry'), dict)
              and isinstance(row.get('fine_geometry'), dict)]
    delta_iou = [fine['iou'] - coarse['iou'] for coarse, fine in paired]
    reductions = [coarse['center_error_m'] - fine['center_error_m'] for coarse, fine in paired]
    def change_counts(changes):
        # 仅将1e-7以内几何浮点差作为不变，不把明显负值算改善。
        improved = sum(value > 1e-7 for value in changes)
        worsened = sum(value < -1e-7 for value in changes)
        return dict(improved_frames=improved, worsened_frames=worsened,
                    unchanged_frames=len(changes) - improved - worsened,
                    improved_rate=improved / len(changes) if changes else None,
                    worsened_rate=worsened / len(changes) if changes else None)
    result['paired_refinement'] = dict(valid_pairs=len(paired),
        mean_delta_iou=float(np.mean(delta_iou)) if paired else None,
        mean_center_error_reduction_m=float(np.mean(reductions)) if paired else None,
        iou=change_counts(delta_iou), center_error=change_counts(reductions),
        unchanged_tolerance=1e-7)
    return result


def _observation_summary(rows):
    values = [row for row in rows if not row.get('initialization', False)]
    normalized = []
    for row in values:
        item = dict(row)
        if ('diagnostic_crop_target_count' not in item
                and 'diagnostic_target_count' in item and 'diagnostic_novel_target_count' in item):
            item['diagnostic_crop_target_count'] = item['diagnostic_target_count'] - item['diagnostic_novel_target_count']
        normalized.append(item)
    counts = {}
    for name in DIAGNOSTIC_FIELDS[:5]:
        available = [row[name] for row in normalized if name in row]
        counts[name] = dict(recorded_frames=len(available), total=sum(available) if available else None)
    flags = {}
    for name in ('diagnostic_empty_crop', 'diagnostic_background_only_crop', 'diagnostic_moving',
                 'strong', 'supported', 'trusted_velocity_valid'):
        available = [row[name] for row in values if name in row]
        flags[name] = dict(recorded_frames=len(available),
                           true_frames=sum(bool(value) for value in available) if available else None)
    reasons = defaultdict(int)
    for row in values:
        if 'trusted_velocity_reset_reason' in row:
            reasons[str(row['trusted_velocity_reset_reason'])] += 1
    def occurrence(fields, eligible, event):
        recorded = [row for row in normalized if all(field in row for field in fields)]
        candidates = [row for row in recorded if eligible(row)]
        events = sum(event(row) for row in candidates) if recorded else None
        return dict(total_prediction_frames=len(values), recorded_frames=len(recorded),
                    eligible_frames=len(candidates), event_frames=events,
                    rate_among_eligible=events / len(candidates) if candidates else None,
                    frequency_among_recorded_predictions=events / len(recorded) if recorded else None)
    raw, crop, sampled = DIAGNOSTIC_FIELDS[:3]
    conditions = dict(
        raw_target_absent=occurrence((raw,), lambda row: True, lambda row: row[raw] == 0),
        raw_positive_crop_missed=occurrence((raw, crop), lambda row: row[raw] > 0,
                                           lambda row: row[crop] == 0),
        crop_positive_sampled_target_lost=occurrence((crop, sampled), lambda row: row[crop] > 0,
                                                     lambda row: row[sampled] == 0))
    return dict(point_counts=counts, flags=flags, own_observation_conditions=conditions,
                crop_count_source='saved diagnostic_crop_target_count, otherwise saved raw minus novel FG',
                trusted_velocity_reset_reasons=dict(reasons))


def compare_runs(reference, old_b0, a, b, c, *, seed=42, old_b0_diagnostics=None):
    roots = dict(reference=reference, old_b0=old_b0, A=a, B=b, C=c)
    runs = {label: read_run(root, label, seed) for label, root in roots.items()}
    canonical = set(runs['old_b0']['rows'][60])
    source = runs['old_b0']['source']
    for label, run in runs.items():
        require(run['source'] == source, label + ': 训练数据 source 不匹配')
        require(all(set(rows) == canonical for rows in run['rows'].values()), label + ': 跨臂帧键不匹配')
    ignored = {'lr', 'lr_schedule', 'lr_milestones', 'cfg', 'path', 'log_dir', 'tag',
               'checkpoint', 'init_checkpoint', 'test', 'eval_checkpoint_epoch', 'accelerator', 'experiment_name'}
    base = {key: value for key, value in runs['A']['config'].items() if key not in ignored}
    code = runs['A']['manifest'].get('source', {}).get('sha256')
    require(isinstance(code, str) and len(code) == 64, 'A: 缺源码摘要')
    for label in ('B', 'C'):
        other = {key: value for key, value in runs[label]['config'].items() if key not in ignored}
        require(other == base, f'{label}: A/B/C 除登记LR及路径标记外配置必须相同')
        require(runs[label]['manifest'].get('source', {}).get('sha256') == code,
                f'{label}: A/B/C 源码必须相同')
    old_rows = runs['old_b0']['rows'][60]
    for epoch in EPOCHS:
        for key in canonical:
            if not key[1]:
                continue
            raw_count = old_rows[key].get('diagnostic_target_count')
            motion = runs['A']['rows'][60][key]['diagnostic_gt_xy_displacement']
            timestamp = runs['A']['rows'][60][key]['timestamp']
            for label in RECIPES:
                row = runs[label]['rows'][epoch][key]
                if raw_count is not None:
                    require(row['diagnostic_target_count'] == raw_count, f'{label}.{key}: 原始目标计数不一致')
                close(row['diagnostic_gt_xy_displacement'], motion, f'{label}.{key}.physical_displacement')
                close(row['timestamp'], timestamp, f'{label}.{key}.timestamp')
            for label in ('reference', 'old_b0'):
                saved = runs[label]['rows'][epoch][key].get('timestamp')
                if saved is not None:
                    close(saved, timestamp, f'{label}.{key}.timestamp')
    by_track = defaultdict(list)
    for key, row in runs['A']['rows'][60].items():
        if key[1]:
            by_track[key[0]].append((key[1], row['timestamp']))
    for track, values in by_track.items():
        ordered = sorted(values)
        require(all(right[1] > left[1] for left, right in zip(ordered, ordered[1:])),
                f'{track}: 物理时间戳必须严格递增')
    supplement = _supplement(old_b0_diagnostics, canonical)
    groups, coverage = fixed_groups(old_rows, supplement, runs['A']['rows'][60])
    groups = {name: keys for name, keys in sorted(groups.items())}
    arms = {}
    for label, run in runs.items():
        frames = run['rows'][60]
        # 时间戳是共同原始帧属性；只在报告副本补旧记录，不重评模型或改写JSONL。
        episode_rows = [dict(row, timestamp=runs['A']['rows'][60][key]['timestamp'])
                        if key[1] else dict(row) for key, row in frames.items()]
        arms[label] = dict(directory=run['directory'], final60=run['scores'][60], late3=run['late3'],
            epochs=run['scores'], loss_episodes=loss_episode_summary(episode_rows),
            loss_episode_timestamp_source='A final60 same-frame physical timestamps; report copy only',
            observation_diagnostics=_observation_summary(frames.values()),
            groups={name: dict(n=len(keys), final60=score(frames[key] for key in keys),
                late3={metric: float(np.mean([score(run['rows'][e][key] for key in keys)[metric]
                    for e in EPOCHS])) for metric in ('success', 'precision')},
                **_geometry_means(frames[key] for key in keys)) for name, keys in groups.items() if keys})
    comparisons = {}
    for left, right in [('A', 'old_b0'), ('B', 'A'), ('C', 'B'),
                        ('A', 'reference'), ('B', 'reference'), ('C', 'reference')]:
        item = {stage: {metric: arms[left][stage][metric] - arms[right][stage][metric]
                for metric in ('success', 'precision')} for stage in ('final60', 'late3')}
        item['groups'] = {name: dict(n=len(keys),
            final60_delta_pp={metric: arms[left]['groups'][name]['final60'][metric]
                - arms[right]['groups'][name]['final60'][metric] for metric in ('success', 'precision')},
            final60_overall_contribution_pp={metric: 100 * math.fsum(
                runs[left]['rows'][60][key][metric] - runs[right]['rows'][60][key][metric]
                for key in keys) / COUNTS['frames'] for metric in ('success', 'precision')})
            for name, keys in groups.items() if keys}
        comparisons[f'{left}_minus_{right}'] = item
    target_checks = {label: {stage + '.' + name:
        comparisons[f'{label}_minus_reference'][stage][name] >= -1e-9
        for stage in ('final60', 'late3') for name in ('success', 'precision')} for label in RECIPES}
    target = {label: all(checks.values()) for label, checks in target_checks.items()}
    eligible = [label for label in RECIPES if target[label]]
    selected = max(eligible, key=lambda label: (arms[label]['final60']['success'],
                   arms[label]['final60']['precision'])) if eligible else None
    provenance = {path: sha for run in runs.values() for path, sha in run['provenance'].items()}
    if old_b0_diagnostics is not None:
        provenance[str(Path(old_b0_diagnostics).resolve())] = _sha(Path(old_b0_diagnostics))
    return dict(schema=REPORT_SCHEMA, status='target_met' if any(target.values()) else 'target_not_met',
        criterion=dict(final60_and_late3_success_and_precision_at_least_reference=True,
            maximum_drop_pp=0., seed=seed, selection_order=['final60.success', 'final60.precision']),
        target_met=target, target_checks=target_checks, selected_recipe=selected,
        verified=dict(**COUNTS, **BUDGET, checkpoint_epochs=EPOCHS),
        group_definition=dict(fixed_on='old_b0 final60 frame keys; raw/motion are model-independent GT diagnostics',
            moving_threshold_m=.15, old_crop_is_endogenous=True,
            supplemental_coverage=coverage, first_prediction_probe_is_not_full_coverage=True),
        arms=arms, comparisons=comparisons, sources_sha256=provenance,
        limitations=['single seed and two mini validation scenes',
            'three LR recipes and adjacent checkpoints are not independent replications',
            'A-minus-oldB0 evaluates the whole repair package, not an individual change',
            'episode seconds use validated A final60 physical timestamps by identical frame keys; no nominal dt substitution'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'old-b0', 'a', 'b', 'c'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--old-b0-diagnostics', type=Path,
                        help='显式既有JSONL；允许9/23 probe original子集，保留覆盖分母')
    args = parser.parse_args(argv)
    try:
        report = compare_runs(args.reference, args.old_b0, args.a, args.b, args.c,
                              seed=args.seed, old_b0_diagnostics=args.old_b0_diagnostics)
        code = 0 if report['status'] == 'target_met' else 1
    except (EvidenceError, ValueError, TypeError, KeyError) as error:
        report = dict(schema=REPORT_SCHEMA, status='invalid_evidence', message=str(error))
        code = 2
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return code


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    raise SystemExit(main())
