"""当前比较工具的固定 R/C 证据、数值和分组公共接口。

历史模型只读取原始 JSON/YAML，不导入或重建其训练代码。
"""
from __future__ import annotations
from collections import defaultdict
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import yaml
from utils.tracking_metrics import metric_contributions

ROOT = Path(__file__).resolve().parents[1]
REPORT_SCHEMA = 'ct_seqtrack.tracking_audit.v1'
LABELS = ('reference', 'baseline')

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


DIAGNOSTIC_FIELDS = ('diagnostic_target_count', 'diagnostic_crop_target_count',
    'diagnostic_sampled_target_count', 'diagnostic_crop_point_count',
    'diagnostic_sampled_point_count', 'diagnostic_gt_xy_displacement',
    'diagnostic_empty_crop', 'diagnostic_background_only_crop')


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


EVER_MOVING_COUNTS = (31, 657)


FIRST_RAW0_COUNTS = (36, 36)  # 首测当帧数、轨迹数；不是这些轨迹全程的781帧。


RAW_GROUPS = ('raw0', 'raw1_2', 'raw3_9', 'raw10plus')


CONTEXT_FIELDS = ('diagnostic_history_support', 'diagnostic_current_support',
                  'diagnostic_query_context_norm')


EXCLUDED_IDENTITY = {'cfg', 'path', 'log_dir', 'tag', 'checkpoint', 'init_checkpoint',
    'test', 'eval_checkpoint_epoch', 'accelerator', 'v31_evaluate_late3'}


METRICS = ('success', 'precision')


CONTEXT_LRS = (1e-4, 5e-5, 2.5e-5)


DEFAULT_COMPARISONS = (('S_minus_C', 'S', 'baseline'), ('W_minus_S', 'W', 'S'),
                       ('S_minus_R', 'S', 'reference'), ('W_minus_R', 'W', 'reference'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode('utf-8')).hexdigest()


def config_digest(config, manifest):
    """按保存版本重建身份；不让当前 normalize 向 v32 历史注入新默认值。"""
    payload = {k: v for k, v in config.items() if k not in EXCLUDED_IDENTITY}
    require(not payload.pop('dynamics_time_manifest', None), '正式比较只接受 true time')
    if payload.get('lr_warmup_steps') == 0:
        payload.pop('lr_warmup_steps')
    payload.update(schema=manifest['schema'], evaluation_rng_policy='per_checkpoint_seed_v1')
    if manifest['model'] == 'seqtrack_reference':
        require(isinstance(manifest.get('reference_protocol'), dict), 'reference 缺原协议身份')
        payload['reference_protocol'] = manifest['reference_protocol']
    return digest(payload)


def _sha256(value, label):
    require(isinstance(value, str) and len(value) == 64
            and all(c in '0123456789abcdef' for c in value), label + ': 需要 SHA256')
    return value


def score(rows):
    values = list(rows)
    return {m: 100 * math.fsum(r[m] for r in values) / len(values) if values else None
            for m in METRICS}


def read_frames(path):
    rows, tracks = {}, defaultdict(list)
    try:
        for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            r = json.loads(line)
            label = f'{path}:{line_number}'
            require(isinstance(r, dict), label + ': 需要逐帧对象')
            t, f = r.get('tracklet'), r.get('frame')
            require(isinstance(t, str) and t and type(f) is int and f >= 0, label + ': 非法帧键')
            require((t, f) not in rows, label + ': 重复帧键')
            require(r.get('initialization') is (f == 0), label + ': initialization 不匹配')
            iou, distance = number(r.get('iou'), label), number(r.get('distance'), label)
            require(0 <= iou <= 1 and distance >= 0, label + ': 几何越界')
            require(f != 0 or (iou == 1 and distance == 0), label + ': 非法初始化')
            require(isinstance(r.get('scene_id'), str) and r['scene_id'], label + ': 缺 scene_id')
            for flag in ('strong', 'supported', 'memory_write', 'memory_wrong_write', 'coarse_valid'):
                if flag in r:
                    require(type(r[flag]) is bool, label + ': 非法布尔字段 ' + flag)
            if r.get('selected_quality') is not None:
                require(0 <= number(r['selected_quality'], label + '.selected_quality') <= 1,
                        label + ': quality 越界')
            for name in ('coarse_geometry', 'fine_geometry'):
                if r.get(name) is not None:
                    require(isinstance(r[name], dict), label + ': 非法 ' + name)
                    require(0 <= number(r[name].get('iou'), label + '.' + name) <= 1
                            and number(r[name].get('center_error_m'), label + '.' + name) >= 0,
                            label + ': 几何诊断越界')
            rows[t, f] = r
            tracks[t].append(f)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f'{path}: 无法读取逐帧证据: {error}') from error
    require(len(rows) == COUNTS['frames'] and len(tracks) == COUNTS['tracklets']
            and sum(k[1] > 0 for k in rows) == COUNTS['prediction_frames'], str(path) + ': 帧/轨迹分母错误')
    for track, frames in tracks.items():
        require(sorted(frames) == list(range(len(frames))), f'{path}: {track} 帧不连续')
    values = list(rows.values())
    success, precision = metric_contributions([r['iou'] for r in values], [r['distance'] for r in values])
    for r, s, p in zip(values, success, precision):
        for m, v in zip(METRICS, (s, p)):
            # GPU/CPU FP32积分舍入不同；校验容差不用于放宽晋级门。
            close(r.get(m), float(v), f'{path}.{r["frame"]}.{m}', 2e-7)
            r[m] = float(v)
    return rows


def _sampler(value, config, epoch, label):
    require(isinstance(value, dict), label + ': 缺 sampler')
    require(value.get('manifest_sha256') == digest({k: v for k, v in value.items()
            if k != 'manifest_sha256'}), label + ': sampler checksum 不匹配')
    expected = dict(epoch=epoch - 1, rows=BUDGET['last_epoch_rows'], batch_size=16, seed=42)
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
    require(value.get('training') is True, label + ': 非训练排程')
    return _sha256(value.get('source_sha256'), label + '.source_sha256')


def _diagnostics(row, label):
    _check_diagnostics(row, label)
    require(type(row.get('selected_index')) is int and row['selected_index'] == 0, label + ': B0 必须选择 q0')
    require(0 <= number(row.get('selected_quality'), label) <= 1, label + ': quality 越界')
    for field in ('strong', 'supported', 'memory_write', 'memory_wrong_write'):
        require(type(row.get(field)) is bool, label + ': 缺布尔诊断 ' + field)
    require(row['supported'] is (row['strong'] and row['selected_quality'] >= .5),
            label + ': supported 与 strong/quality 合同不匹配')


def _context_diagnostics(row, label, *, required):
    present = [row.get(field) is not None for field in CONTEXT_FIELDS]
    if not required and not any(present):
        return
    require(all(present), label + ': 缺 v34 context 诊断 ' + ','.join(
        field for field, exists in zip(CONTEXT_FIELDS, present) if not exists))
    def check_array(value, shape, where):
        if not shape:
            number(value, where)
            return
        require(isinstance(value, list) and len(value) == shape[0], where + ': shape 必须为 ' + str(shape))
        for index, item in enumerate(value):
            check_array(item, shape[1:], f'{where}[{index}]')
    check_array(row[CONTEXT_FIELDS[0]], (3, 3), label + '.' + CONTEXT_FIELDS[0])
    check_array(row[CONTEXT_FIELDS[1]], (3,), label + '.' + CONTEXT_FIELDS[1])
    require(number(row[CONTEXT_FIELDS[2]], label + '.' + CONTEXT_FIELDS[2]) >= 0,
            label + ': context norm 不能为负')


def read_run(directory, label, *, expected_lr=None):
    require(label in LABELS, '未登记的比较角色: ' + str(label))
    allowed_lrs = (CONTEXT_LRS if label in ('S', 'W') else
                   ((1e-4, 5e-5) if label == 'reference' else
                    ((5e-5,) if label == 'baseline' else (1e-4,))))
    default_lr = 5e-5 if label in ('baseline', 'S', 'W') else 1e-4
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
    version = 34 if label in ('S', 'W') else (32 if label == 'old_b0' else 33)
    schema = f'ct_seqtrack.joint_identity.v{version}'
    model = 'seqtrack_reference' if label == 'reference' else f'ctseqtrackv{version}'
    equal_fields(manifest, dict(schema=schema, model=model, seed=42, arm='b0'), label + '.manifest')
    require(manifest.get('enabled') == dict(B1=False, B2=False, B3=False), label + ': 只能比较 B0')
    require(manifest.get('evaluation_rng_policy') == 'per_checkpoint_seed_v1', label + ': 评测 RNG 身份错误')
    config = read('resolved_config.yaml', True)
    expected = dict(COMMON_CONFIG, experiment_family=f'ct_seqtrack_v{version}', net_model=model, seed=42,
                    v31_short_window=4 if label == 'W' else 3)
    expected['lr'] = expected_lr
    if label in ('baseline', 'S', 'W'):
        expected.update(lr_schedule='multistep', lr_milestones=[20, 50])
    equal_fields(config, expected, label + '.config')
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
        if label in ('S', 'W'):
            equal_fields(audit, dict(schema='ct_seqtrack.v34.training_audit.v1', model_schema=schema,
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
        if label in ('baseline', 'S', 'W'):
            for key, row in rows.items():
                if key[1]:
                    _diagnostics(row, f'{label}.{epoch}.{key}')
        for key, row in rows.items():
            if key[1]:
                _context_diagnostics(row, f'{label}.{epoch}.{key}', required=label in ('S', 'W'))
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


def fixed_groups(canonical):
    pred = sorted(k for k in canonical if k[1] > 0)
    raw = lambda k: canonical[k]['diagnostic_target_count']
    moving = lambda k: canonical[k]['diagnostic_gt_xy_displacement'] >= .15
    ever = {k[0] for k in pred if moving(k)}
    first_zero = {k[0] for k in pred if k[1] == 1 and raw(k) == 0}
    groups = dict(overall=sorted(canonical), predictions=pred,
        first_raw0=[k for k in pred if k[1] == 1 and raw(k) == 0],
        first_raw0_tracks=[k for k in pred if k[0] in first_zero],
        ever_moving=[k for k in pred if k[0] in ever],
        ever_moving_quiet=[k for k in pred if k[0] in ever and not moving(k)],
        moving=[k for k in pred if moving(k)])
    for label, lower, upper in (('raw0', 0, 0), ('raw1_2', 1, 2), ('raw3_9', 3, 9), ('raw10plus', 10, math.inf)):
        groups[label] = [k for k in pred if lower <= raw(k) <= upper]
        groups['ever_moving_quiet_' + label] = sorted(set(groups[label]) & set(groups['ever_moving_quiet']))
    for before, after in ((False, True), (True, False)):
        groups[f'motion_cross_{int(before)}{int(after)}'] = [k for k in pred if k[1] >= 2
            and moving((k[0], k[1] - 1)) == before and moving(k) == after]
    require((len(ever), len(groups['ever_moving'])) == EVER_MOVING_COUNTS,
            '共同 GT 分组不是登记的31条曾移动轨迹/657预测帧')
    require((len(groups['first_raw0']), len(first_zero)) == FIRST_RAW0_COUNTS,
            '共同 GT first_raw0 分组不是登记的36个首测帧/36条轨迹')
    return groups


def refinement(rows):
    pairs = [(r['coarse_geometry'], r['fine_geometry']) for r in rows
             if r.get('coarse_valid') is True and isinstance(r.get('coarse_geometry'), dict)
             and isinstance(r.get('fine_geometry'), dict)]
    def stats(values, lower_better=False):
        return dict(mean=float(np.mean(values)) if values else None,
                    median=float(np.median(values)) if values else None,
                    worsened_rate=sum(v > 1e-7 if lower_better else v < -1e-7 for v in values) / len(values) if values else None)
    return dict(valid_pairs=len(pairs), delta_iou=stats([f['iou'] - c['iou'] for c, f in pairs]),
        delta_center_error_m=stats([f['center_error_m'] - c['center_error_m'] for c, f in pairs], True))


def quality_summary(rows):
    rows = list(rows)
    required = ('selected_quality', 'strong', 'supported')
    recorded = [r for r in rows if all(r.get(k) is not None for k in required)]
    strong = [r for r in recorded if r['strong']]
    supported = [r for r in recorded if r['supported']]
    bad = [r for r in recorded if r['iou'] < .1]
    supported_bad = sum(r['iou'] < .1 for r in supported)
    qualities = [r['selected_quality'] for r in rows if r.get('selected_quality') is not None]
    return dict(prediction_frames=len(rows), recorded_frames=len(recorded),
        quality_recorded_frames=len(qualities),
        quality_mean=float(np.mean(qualities)) if qualities else None,
        quality_median=float(np.median(qualities)) if qualities else None,
        strong=len(strong) if recorded else None,
        supported=len(supported) if recorded else None,
        supported_bad_iou=supported_bad if recorded else None,
        # 分别为错误框占 supported 的比例，及错误框被标为 supported 的比例。
        supported_bad_iou_rate=supported_bad / len(supported) if supported else None,
        bad_iou_supported_rate=supported_bad / len(bad) if bad else None,
        strong_bad_iou=sum(r['strong'] and r['iou'] < .1 for r in recorded) if recorded else None,
        strong_quality_pass_rate=sum(r['selected_quality'] >= .5 for r in strong) / len(strong) if strong else None,
        memory_wrong_write_recorded_frames=sum('memory_wrong_write' in r for r in rows),
        memory_wrong_writes=sum(r['memory_wrong_write'] for r in rows if 'memory_wrong_write' in r)
            if any('memory_wrong_write' in r for r in rows) else None)


def context_summary(rows):
    rows = list(rows)
    recorded = [r for r in rows if all(r.get(field) is not None for field in CONTEXT_FIELDS)]
    return dict(prediction_frames=len(rows), recorded_frames=len(recorded),
        history_support_mean=np.mean([r[CONTEXT_FIELDS[0]] for r in recorded], axis=0).tolist() if recorded else None,
        current_support_mean=np.mean([r[CONTEXT_FIELDS[1]] for r in recorded], axis=0).tolist() if recorded else None,
        query_context_norm_mean=float(np.mean([r[CONTEXT_FIELDS[2]] for r in recorded])) if recorded else None)


def raw_contributions(runs, arms, groups, comparisons=DEFAULT_COMPARISONS):
    """固定GT raw四组分解总体分差；分母含初始化，其分差恒为零。"""
    denominator = len(groups['overall'])
    output = {}
    for name, left, right in comparisons:
        def stage(epochs, score_stage):
            contributions = {group: {metric: 100 * math.fsum(
                runs[left]['rows'][epoch][key][metric] - runs[right]['rows'][epoch][key][metric]
                for epoch in epochs for key in groups[group]) / (denominator * len(epochs))
                for metric in METRICS} for group in RAW_GROUPS}
            def arm_score(label, metric):
                overall = arms[label]['groups']['overall']
                return overall['epochs'][score_stage][metric] if score_stage in overall['epochs'] else overall[score_stage][metric]
            overall = {metric: arm_score(left, metric) - arm_score(right, metric) for metric in METRICS}
            summed = {metric: math.fsum(contributions[group][metric] for group in RAW_GROUPS) for metric in METRICS}
            for metric in METRICS:
                close(summed[metric], overall[metric], name + '.raw_additivity.' + metric, 1e-10)
            return dict(overall_delta_pp=overall, raw_contribution_pp=contributions, raw_sum_pp=summed)
        epochs = {str(epoch): stage([epoch], str(epoch)) for epoch in EPOCHS}
        output[name] = dict(left=left, right=right, units='percentage_points',
            denominator_frames_including_initialization=denominator,
            epochs=epochs, final60=epochs['60'], late3=stage(EPOCHS, 'late3'))
    return output


def compare_loaded(runs, *, candidate_labels=('S', 'W'), reference_label='reference',
                   config_differences=(), comparisons=DEFAULT_COMPARISONS):
    """共用固定帧组和硬门；多学习率入口仅显式扩大已登记的配置差异。"""
    canonical = runs['baseline']['rows'][60]
    keys = set(canonical)
    for label, run in runs.items():
        require(run['source'] == runs['baseline']['source'], label + ': 训练数据身份不一致')
        for epoch, rows in run['rows'].items():
            require(set(rows) == keys, label + ': 跨轮次/跨模型帧键不一致')
            for key, row in rows.items():
                require(row['scene_id'] == canonical[key]['scene_id'], label + ': scene 不一致')
                if not key[1]:
                    continue
                for field in ('diagnostic_target_count', 'diagnostic_gt_xy_displacement', 'timestamp'):
                    if row.get(field) is not None:
                        close(row[field], canonical[key][field], label + '.' + field, 1e-6)
            by_track = defaultdict(list)
            for key in sorted(rows):
                if rows[key].get('timestamp') is not None:
                    by_track[key[0]].append(number(rows[key]['timestamp'], label + '.timestamp'))
            require(all(all(b > a for a, b in zip(ts, ts[1:])) for ts in by_track.values()), label + ': 时间戳不递增')
    allowed = EXCLUDED_IDENTITY | {'experiment_name', 'v31_short_window'} | set(config_differences)
    configs = [{k: v for k, v in runs[label]['config'].items() if k not in allowed} for label in candidate_labels]
    require(all(config == configs[0] for config in configs), 'S/W 除登记差异外配置不同')
    source = runs[candidate_labels[0]]['manifest']['source']
    require(all(runs[label]['manifest']['source'] == source for label in candidate_labels), 'S/W 执行源码不同')
    groups = fixed_groups(canonical)
    arms = {}
    for label, run in runs.items():
        group_scores = {}
        for name, subset in groups.items():
            epochs = {str(e): score(run['rows'][e][k] for k in subset) for e in EPOCHS}
            group_scores[name] = dict(n=len(subset), tracks=len({k[0] for k in subset}), epochs=epochs,
                final60=epochs['60'], late3={m: math.fsum(epochs[str(e)][m] for e in EPOCHS) / 3
                    if subset else None for m in METRICS})
        diagnostics = {}
        for e in EPOCHS:
            rows = run['rows'][e]
            predictions = [rows[k] for k in groups['predictions']]
            diagnostics[str(e)] = dict(quality=quality_summary(predictions), context=context_summary(predictions),
                loss_episodes=loss_episode_summary(predictions),
                coarse_fine={g: refinement(rows[k] for k in groups[g]) for g in
                    ('predictions', 'raw0', 'raw1_2', 'raw3_9', 'raw10plus', 'first_raw0', 'ever_moving')})
        arms[label] = dict(directory=run['directory'], groups=group_scores, diagnostics=diagnostics,
            config_sha256=run['manifest']['config_sha256'], source_sha256=run['manifest']['source']['sha256'],
            input_sha256=run['provenance'])
    candidates = {}
    for label in candidate_labels:
        gates = {}
        for group, target in (('overall', reference_label), ('ever_moving', 'baseline')):
            deltas = {f'{stage}_{m}': arms[label]['groups'][group][stage][m] - arms[target]['groups'][group][stage][m]
                      for stage in ('final60', 'late3') for m in METRICS}
            gates[group] = dict(target=target, delta_pp=deltas, passed=all(v >= 0 for v in deltas.values()))
        candidates[label] = dict(gates=gates, passed=all(g['passed'] for g in gates.values()))
    ranking = sorted((label for label in candidate_labels if candidates[label]['passed']),
        key=lambda label: (-arms[label]['groups']['overall']['final60']['success'],
                           -arms[label]['groups']['overall']['final60']['precision'], label))
    return dict(schema=REPORT_SCHEMA, status='passed' if ranking else 'failed', candidates=candidates,
        selected=ranking[0] if ranking else None, passing_rank=ranking,
        raw_contributions=raw_contributions(runs, arms, groups, comparisons),
        criterion=dict(overall='final60/late3 S/P >= reference', ever_moving='final60/late3 S/P >= baseline C',
            old_b0='恢复目标与描述性对照，不是硬门', moving_threshold_m=.15,
            grouping_source='baseline C 的共同 GT raw 点数/相邻 XY 位移；跨模型核对已有字段',
            ranking='通过者按 final60 Success、Precision 依次降序；完全相同按 S/W 名称'),
        verified=dict(**COUNTS, **BUDGET, checkpoint_epochs=EPOCHS,
            ever_moving_tracklets=EVER_MOVING_COUNTS[0], ever_moving_prediction_frames=EVER_MOVING_COUNTS[1],
            first_raw0_frames=FIRST_RAW0_COUNTS[0], first_raw0_tracklets=FIRST_RAW0_COUNTS[1]),
        limitations=['共同帧组是被动诊断；模型各自递推状态不同，不是同状态干预。',
            'coarse/fine 为各模型同状态配对，不能推断 coarse 闭环分数。',
            '旧证据未记录的时间/几何/支持字段输出 None 与覆盖数，不填零。',
            '相邻 checkpoint 不等于独立 seed；起停仅指0.15m位移阈值跨越。'], arms=arms)


def write_report(path, report):
    destination = Path(path).resolve()
    allowed = (ROOT / 'artifacts' / 'ct_checks').resolve()
    require(allowed in destination.parents, '--output 只允许 artifacts/ct_checks 下的新文件')
    require(not destination.exists(), '--output 拒绝覆盖既有文件')
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
    except OSError as error:
        raise EvidenceError(f'无法独占写入报告: {error}') from error


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


def loss_episode_summary(rows):
    """IoU<.1 开始、IoU>=.5 结束；恢复帧不计入失跟长度。

    未恢复段右删失，单独报告；缺 timestamp 的旧记录不伪造秒数。
    所有段包含 [.1,.5) 中间帧，不能与低 IoU 帧数混用。
    """
    tracks = {}
    for row in rows:
        if not row.get('initialization', False):
            tracks.setdefault(row['tracklet'], []).append(row)
    episodes = []
    for track, values in tracks.items():
        values = sorted(values, key=lambda row: row['frame'])
        active = None
        for row in values:
            if active is None and row['iou'] < .1:
                active = dict(tracklet=track, start_frame=row['frame'],
                              start_time=row.get('timestamp'), frames=0)
            if active is None:
                continue
            recovered = row['iou'] >= .5
            if not recovered:
                active['frames'] += 1
            last = row is values[-1]
            if recovered or last:
                start, end = active.pop('start_time'), row.get('timestamp')
                seconds = float(end - start) if start is not None and end is not None else None
                if seconds is not None and seconds < 0:
                    raise ValueError('evaluation timestamps must be causal')
                episodes.append(dict(active, end_frame=row['frame'], recovered=recovered,
                                     seconds=seconds))
                active = None
    def distribution(values):
        return dict(count=len(values), median=float(np.median(values)) if values else None,
                    p90=float(np.percentile(values, 90)) if values else None,
                    total=float(sum(values)) if values else 0.)
    def summarize(values):
        timed = [row['seconds'] for row in values if row['seconds'] is not None]
        return dict(episodes=len(values), frames=distribution([row['frames'] for row in values]),
                    seconds=distribution(timed), seconds_complete=len(timed) == len(values))
    recovered = [row for row in episodes if row['recovered']]
    censored = [row for row in episodes if not row['recovered']]
    return dict(definition='start_iou_lt_0.1_end_iou_ge_0.5_recovery_frame_excluded',
                all=summarize(episodes), recovered=summarize(recovered),
                unrecovered=summarize(censored),
                unrecovered_rate=len(censored) / len(episodes) if episodes else None,
                total_lost_frames=sum(row['frames'] for row in episodes),
                low_iou_frames=sum(row['iou'] < .1 for values in tracks.values() for row in values),
                episodes=episodes)


@lru_cache(maxsize=64)
def _rebuilt_sampler(payload):
    from models.ct_v38.data import ReadyQueueBatchSampler
    value = json.loads(payload)
    sampler = ReadyQueueBatchSampler(**{key: value[key] for key in (
        'lengths', 'batch_size', 'seed', 'training', 'short_window', 'long_window',
        'curriculum_epochs', 'source_sha256', 'reserve_windows', 'seed_translation',
        'seed_yaw_degrees', 'initial_seed_policy')})
    sampler.set_epoch(value['epoch'])
    return sampler.state_dict()


def validate_sampler(value, config, epoch, label):
    source = _sampler(value, config, epoch, label)
    require(value.get('initial_seed_policy') == 'initial_exact_other_perturbed_v1', label + ': initial seed mismatch')
    lengths = value.get('lengths')
    require(isinstance(lengths, list) and lengths and all(type(n) is int and n >= 0 for n in lengths),
            label + ': invalid track lengths')
    require(value == _rebuilt_sampler(json.dumps(value, sort_keys=True, separators=(',', ':'))),
            label + ': sampler does not reconstruct')
    return source
