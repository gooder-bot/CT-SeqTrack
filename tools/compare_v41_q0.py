"""按固定 dev 58/59/60 轮比较同一 V41 执行版本的两种 q0 中心损失。"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path, PurePosixPath
from statistics import fmean

import yaml


EPOCHS = (58, 59, 60)
AXES = ('longitudinal', 'lateral', 'vertical')
MODES = ('uniform', 'inverse_lwh')
# 登记的配对训练预算。离线复核须能在活动 YAML 更新后重读两份已结束的 run。
REFERENCE_TRAINING = dict(epoch=60, batch_size=16, seed=42, workers=4, lr=.00015,
                          lr_schedule='multistep', lr_milestones=[20, 40], lr_decay_rate=.2,
                          lr_warmup_steps=0, wd=0)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def numeric_environment(record):
    # GPU 序号是分配信息。同型号的不同卡可用于复现。
    fa = record.get('flash_attention')
    fa = fa if isinstance(fa, dict) else {}
    values = {k: record.get(k) for k in ('torch', 'lightning', 'precision_policy')}
    values.update({k: fa.get(k) for k in ('backend', 'flash_attn', 'device_name', 'kernel_dtype', 'projection_dtype')})
    return {k: v if isinstance(v, str) and v else None for k, v in values.items()}


def environment_audit(roots, manifests):
    sessions = {}
    for arm, root, manifest in zip(('A', 'B'), roots, manifests):
        sessions[arm + '/initial'] = numeric_environment(manifest)
        for path in sorted((root / 'resume_manifests').glob('*.json')):
            resumed = read_json(path)
            require(resumed['run_id'] == manifest['run_id'] and resumed['config_sha256'] == manifest['config_sha256']
                    and resumed['source']['sha256'] == manifest['source']['sha256'],
                    f'resume identity differs in {path.name}')
            sessions[arm + '/' + path.name] = numeric_environment(resumed)
    unknown = {name: [k for k, v in env.items() if v is None] for name, env in sessions.items() if None in env.values()}
    differences = [k for k in sessions['A/initial'] if len({env[k] for env in sessions.values() if env[k]}) > 1]
    status = 'mismatch' if differences else 'unknown' if unknown else 'matched'
    return dict(status=status, differing_fields=differences, missing_fields=unknown, sessions=sessions)


def identity(root, mode):
    manifest = read_json(root / 'run_manifest.json')
    config = yaml.safe_load((root / 'resolved_config.yaml').read_text(encoding='utf-8-sig'))
    recipe = dict(manifest['training_execution_recipe'])
    require(manifest['model'] == 'ctseqtrackv41' and manifest['run_type'] == 'formal'
            and recipe['execution_revision'] in ('v3', 'v4') and manifest['arm'] == 'full'
            and recipe['execution_revision'] == config['v41_execution_revision']
            and 'source_run_id' not in manifest, 'expected a formal V41 v3/v4 Full run with consistent execution revision')
    require(recipe.pop('q0_center_weighting') == mode and config['v41_q0_center_weighting'] == mode,
            f'expected q0 center weighting {mode}')
    require(not config.get('init_checkpoint') and not config.get('checkpoint')
            and not config.get('test') and config['epoch'] == 60
            and manifest['evaluation_protocol']['split'] == 'dev', 'expected scratch60 dev training')
    # 各运行目录、命令行设备编号与显示选项可以不同。训练、采样和评估语义必须一致。
    incidental = {'cfg', 'tag', 'experiment_name', 'path', 'log_dir', 'devices', 'accelerator', 'v41_q0_center_weighting',
                  'console_mode', 'log_every_n_steps', 'progress_refresh_seconds',
                  'plain_log_interval_seconds', 'tensorboard', 'v41_memory_logging'}
    comparable = {k: v for k, v in config.items() if k not in incidental}
    evidence = {name: manifest[name] for name in ('initial_parameter_hash', 'dataset_manifest',
                'input_metadata', 'evaluation_rng_policy', 'evaluation_protocol')}
    actual = dict(manifest['actual_recipe'])
    require(actual.pop('v41_q0_center_weighting') == mode and actual
            and all(config.get(key) == value for key, value in actual.items()),
            'actual training recipe disagrees with resolved config')
    evidence.update(source_sha256=manifest['source']['sha256'], recipe=recipe, actual_recipe=actual)
    require(all(evidence.values()) and manifest['run_id'] and manifest['config_sha256'],
            'missing run identity')
    return manifest, comparable, evidence


def budget(root, manifest):
    final = read_json(root / 'training_budget.json')
    require(final['completed_epoch'] == 60 and final['epoch_complete'] is True
            and final['auxiliary']['complete'] is True, 'incomplete scratch60 training budget')
    schedules, totals = [], dict(main_endpoints=0, auxiliary_endpoints=0, optimizer_steps=0)
    for number in range(1, 61):
        record = read_json(root / 'training_audits' / f'epoch={number:03d}.json')
        aux = record['auxiliary']
        require(record['completed_epoch'] == number and record['epoch_complete'] is True
                and record['run_id'] == manifest['run_id'] and record['config_sha256'] == manifest['config_sha256']
                and aux['complete'] is True, f'incomplete or mismatched training audit at epoch {number}')
        main, auxiliary = record['sampler'], aux['sampler']
        require(record['rows'] == main['rows'] > 0 and record['optimizer_steps'] == main['batches'] > 0
                and aux['rows'] == auxiliary['rows'] and aux['batches'] == auxiliary['batches']
                and aux['main_batches'] == record['optimizer_steps'], f'inconsistent training budget at {number}')
        schedules.append((main['plan_sha256'], auxiliary['plan_sha256'], aux['insertion_plan_sha256'],
                          record['rows'], record['optimizer_steps'], aux['rows'], aux['batches']))
        totals['main_endpoints'] += record['rows']
        totals['optimizer_steps'] += record['optimizer_steps']
        totals['auxiliary_endpoints'] += aux['rows']
    require(final['optimizer_steps'] == totals['optimizer_steps'] and final['last_epoch_rows'] == record['rows']
            and final['last_epoch_steps'] == record['optimizer_steps']
            and final['sampler'] == main and final['auxiliary'] == aux, 'final budget disagrees with epoch audits')
    return schedules, totals


def finite(x):
    return type(x) in (float, int) and math.isfinite(x)


def frame_stream(root, epoch):
    name = f'epoch={epoch:03d}'
    report = read_json(root / 'epochs' / (name + '.json'))
    result = report['evaluation']['dev']
    require(result['epoch'] == result['checkpoint_epoch'] == report['epoch'] == epoch
            and result['split'] == 'dev' and result['diagnostics_level'] == 'full'
            and result['complete_coverage'] is True, f'expected full, complete dev evaluation at {epoch}')
    # 以 epoch 报告指向的会话为准。兼容从服务器拷回的绝对路径和中断留下的旧流。
    parts = PurePosixPath(result['frame_file'].replace('\\', '/')).parts[-4:]
    require(len(parts) == 4 and parts[:2] == ('frame_streams', 'dev') and parts[2] != '..'
            and parts[-1] == name + '.jsonl', 'invalid dev frame_file')
    rows = {}
    track_frames = defaultdict(list)
    with root.joinpath(*parts).open(encoding='utf-8-sig') as stream:
        for line in stream:
            row = json.loads(line)
            key = (str(row['scene_id']).zfill(4), int(row['track_id']), int(row['raw_frame_id']))
            require(key not in rows and key[0] in ('0017', '0018'), f'duplicate or non-dev frame {key}')
            require(all(finite(row[k]) and 0 <= row[k] <= 1 for k in ('iou', 'success', 'precision'))
                    and finite(row['distance']) and row['distance'] >= 0, f'invalid frame metrics {key}')
            require(type(row['frame']) is int and type(row['initialization']) is bool
                    and row['initialization'] == (row['frame'] == 0), f'invalid initialization {key}')
            if row['initialization']:
                require((row['iou'], row['distance'], row['success'], row['precision']) == (1, 0, 1, 1),
                        f'invalid first-frame scores {key}')
            else:
                for field in ('diagnostic_fine_box_world', 'diagnostic_target_box_world'):
                    require(len(row[field]) == 4 and all(map(finite, row[field])), f'invalid {field} at {key}')
                count = row['diagnostic_sampled_target_count']
                require(type(count) is int and count >= 0 and type(row['q0_is_fallback']) is bool,
                        f'invalid full q0 diagnostics {key}')
            keep = ('frame', 'initialization', 'distance', 'iou', 'success', 'precision')
            if not row['initialization']:
                keep += ('diagnostic_fine_box_world', 'diagnostic_target_box_world',
                         'diagnostic_sampled_target_count', 'q0_is_fallback')
            rows[key] = {field: row[field] for field in keep}
            track_frames[key[:2]].append((key[2], row['frame']))
    for track, frames in track_frames.items():
        require([idx for _, idx in sorted(frames)] == list(range(len(frames))), f'incomplete track {track}')
    require(rows and len(rows) == result['total_frames'] and len(track_frames) == result['initialization_frames']
            and len(rows) - len(track_frames) == result['prediction_frames'], 'frame stream coverage mismatch')
    for metric in ('success', 'precision'):
        require(math.isclose(100 * fmean(r[metric] for r in rows.values()), result[metric], abs_tol=.0002),
                f'frame stream disagrees with dev {metric}')
    return rows


def eligible(row):
    return not row['initialization'] and not row['q0_is_fallback'] and row['diagnostic_sampled_target_count'] >= 10


def q0_errors(row):
    prediction, target = row['diagnostic_fine_box_world'], row['diagnostic_target_box_world']
    dx, dy, dz = (prediction[i] - target[i] for i in range(3))
    c, s = math.cos(target[3]), math.sin(target[3])
    return tuple(100 * abs(v) for v in (c * dx + s * dy, -s * dx + c * dy, dz))


def sp(rows):
    return {field: 100 * fmean(r[field] for r in rows) for field in ('success', 'precision')}


def arm_metrics(rows, common):
    groups = defaultdict(list)
    for key, row in sorted(rows.items()):
        groups[key[:2]].append(row)
    tracks = {}
    for (scene, track), values in groups.items():
        if len(values) < 100:
            continue
        lost, events = False, 0
        for row in values[1:]:
            if not lost and row['iou'] < .1:
                events += 1
                lost = True
            elif row['iou'] >= .5:
                lost = False
        tracks[f'{scene}/{track}'] = dict(**sp(values), frames=len(values), loss_events=events,
                                         over10m_frames=sum(r['distance'] > 10 for r in values[1:]))
    require(tracks, 'missing long dev tracks')
    # 两组闭环可能没有共同可见帧；仍需报告覆盖数和最终跟踪成绩。
    axis_mae = dict.fromkeys(AXES)
    if common:
        errors = zip(*(q0_errors(rows[key]) for key in common))
        axis_mae = dict(zip(AXES, map(fmean, errors)))
    long = {field: fmean(t[field] for t in tracks.values()) for field in ('success', 'precision')}
    long.update({field: sum(t[field] for t in tracks.values()) for field in ('loss_events', 'over10m_frames')})
    return dict(overall=sp(rows.values()), q0_mae_cm=axis_mae, long_tracks=tracks,
                long_track_summary=long, eligible_frames=sum(map(eligible, rows.values())),
                common_visible_frames=len(common))


def average_tree(items):
    first = items[0]
    if isinstance(first, dict):
        return {k: average_tree([item[k] for item in items]) for k in first}
    if None in items:
        return None
    return fmean(items)


def difference(a, b):
    if isinstance(a, dict):
        return {key: difference(a[key], b[key]) for key in a}
    if a is None or b is None:
        return None
    return b - a


def compare(run_a, run_b):
    roots = [Path(run_a), Path(run_b)]
    manifests, configs, evidence = zip(*(identity(root, mode) for root, mode in zip(roots, MODES)))
    revision = configs[0]['v41_execution_revision']
    require(revision == configs[1]['v41_execution_revision'], 'q0 arms differ in execution revision')
    require(manifests[0]['run_id'] != manifests[1]['run_id'] and configs[0] == configs[1]
            and evidence[0] == evidence[1], 'arms differ in training settings, source, initialization or data identity')
    require(all(configs[0].get(k) == v for k, v in REFERENCE_TRAINING.items()),
            'expected the reference Full scratch60 training recipe for this q0 experiment')
    environment = environment_audit(roots, manifests)
    budgets = [budget(root, manifest) for root, manifest in zip(roots, manifests)]
    require(budgets[0] == budgets[1], 'actual training schedules or budgets differ')
    epochs, expected = {}, None
    for epoch in EPOCHS:
        a, b = (frame_stream(root, epoch) for root in roots)
        inventory = {key: row['frame'] for key, row in a.items()}
        require(inventory == {key: row['frame'] for key, row in b.items()}
                and (expected is None or inventory == expected), 'physical frame inventories differ')
        expected = inventory
        common = []
        for key, first in a.items():
            second = b[key]
            if not first['initialization']:
                require(all(math.isclose(x, y, abs_tol=1e-4) for x, y in zip(
                    first['diagnostic_target_box_world'], second['diagnostic_target_box_world'])),
                    f'different GT at {key}')
            if eligible(first) and eligible(second):
                common.append(key)
        metrics_a, metrics_b = arm_metrics(a, common), arm_metrics(b, common)
        epochs[str(epoch)] = dict(A=metrics_a, B=metrics_b, B_minus_A=difference(metrics_a, metrics_b))
    late = average_tree(list(epochs.values()))
    return dict(epochs=epochs, final60=epochs['60'], late3=late, retention=retention(late, environment['status']),
                budget=budgets[0][1], execution_revision=revision,
                arms={arm: dict(run_id=m['run_id'], weighting=mode, config_sha256=m['config_sha256'])
                      for arm, m, mode in zip(('A', 'B'), manifests, MODES)},
                source_sha256=evidence[0]['source_sha256'], initial_parameter_hash=evidence[0]['initial_parameter_hash'],
                training_recipe=evidence[0]['actual_recipe'], numeric_environment=environment,
                split='dev', fixed_epochs=list(EPOCHS), frames_per_epoch=len(expected))


def retention(late, environment_status='matched'):
    a, b = late['A'], late['B']
    def lower_mae(axis):
        before, after = a['q0_mae_cm'][axis], b['q0_mae_cm'][axis]
        return None if before is None or after is None else after < before

    checks = {
        'lateral_mae_lower': lower_mae('lateral'),
        'vertical_mae_lower': lower_mae('vertical'),
        'overall_success_higher': b['overall']['success'] > a['overall']['success'],
        'overall_precision_not_lower': b['overall']['precision'] >= a['overall']['precision'],
        'long_track_equal_weight_success_not_lower': b['long_track_summary']['success'] >= a['long_track_summary']['success'],
        'long_track_over10m_frames_not_higher': b['long_track_summary']['over10m_frames'] <= a['long_track_summary']['over10m_frames'],
    }
    return dict(warrants_independent_seed_check=all(checks.values()) and environment_status == 'matched',
                checks=checks, numeric_environment_status=environment_status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_a', help='uniform 训练目录')
    parser.add_argument('run_b', help='inverse_lwh 训练目录')
    args = parser.parse_args()
    try:
        report = compare(args.run_a, args.run_b)
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
        parser.error(f'无法比较 A/B 运行记录：{error}')
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
