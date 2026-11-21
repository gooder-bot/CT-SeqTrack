"""校验两份 V41 训练账本，在相同物理帧评估 1/4/4/8 与四独立候选。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from statistics import fmean
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


EPOCHS = (58, 59, 60)
MODES = ('window_1448', 'four_candidates')
METRICS = ('success', 'precision')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(value):
    return isinstance(value, str) and len(value) == 64 and all(x in '0123456789abcdef' for x in value)


def training_pair(roots, *, require_q0_lateral=False, require_epoch_test=False):
    from models.ct_v41.auxiliary import insertion_identity, insertion_indices
    from models.ct_v41.config import config_identity, normalize_config, recipe_identity
    from models.ct_v41.history_augmentation import TRAIN_DATA
    from models.ct_v41.joint_replay import CAPACITY_PER_CATEGORY, CATEGORIES

    manifests = [read(root / 'run_manifest.json') for root in roots]
    configs = []
    for root, mode, m in zip(roots, MODES, manifests):
        recipe = m['actual_recipe']
        require(m['run_type'] == 'formal' and recipe['v41_history_training'] == mode and
                recipe['v41_joint_search'] and recipe['dataset'] == 'kitti_mf' and
                (recipe['seed'], recipe['batch_size'], recipe['epoch']) == (42, 16, 60),
                f'incorrect formal arm: {mode}')
        cfg = normalize_config(yaml.safe_load((root/'resolved_config.yaml').read_text(encoding='utf-8-sig')))
        policy = m.get('evaluation_protocol', {})
        if require_epoch_test or policy:
            require((policy.get('split'), policy.get('every_n_epochs'), policy.get('selection')) ==
                    (cfg.eval_split, cfg.check_val_every_n_epoch, cfg.report_checkpoint),
                    f'{mode} evaluation protocol differs from resolved configuration')
        if require_epoch_test:
            require((cfg.eval_split, cfg.check_val_every_n_epoch, cfg.report_checkpoint) == ('test', 1, 'last'),
                    f'{mode} requires test evaluation every epoch and a fixed final checkpoint')
        require(cfg.v41_joint_replay == 'fixed_e2' and not cfg.v41_q0_lateral_replay,
                f'{mode} requires fixed E2 joint replay')
        require(not require_q0_lateral or cfg.v41_q0_lateral,
                f'{mode} requires the independent q0 lateral head')
        if cfg.v41_q0_lateral:
            expected = recipe_identity(cfg)
            recorded = m.get('training_execution_recipe', {})
            # 两组历史六槽均执行 R。早期回执尚无按训练/推理拆分的标记。
            if (cfg.eval_split == 'dev' and
                    not {'training_enabled', 'inference_enabled'}.intersection(recorded.get('recovery', {}))):
                expected['recovery'].pop('training_enabled')
                expected['recovery'].pop('inference_enabled')
            require(recipe.get('v41_q0_lateral') is True
                    and recipe.get('v41_q0_lateral_warmup_epochs') == cfg.v41_q0_lateral_warmup_epochs
                    and recipe.get('v41_joint_replay') == 'fixed_e2'
                    and recorded == expected,
                    f'{mode} lateral/replay execution recipe disagrees with resolved configuration')
        require(cfg.test is False and not cfg.ct_engineering_check and config_identity(cfg) == m['config_sha256']
                and all(cfg.get(k) == v for k, v in recipe.items()),
                f'resolved configuration disagrees with the formal manifest for {mode}')
        configs.append(cfg)
    a, b = manifests
    cfg = configs[0]
    require(all((c.eval_split, c.check_val_every_n_epoch, c.report_checkpoint) ==
                (cfg.eval_split, cfg.check_val_every_n_epoch, cfg.report_checkpoint) for c in configs),
            'arms have different training evaluation protocols')
    require(cfg.v41_q0_lateral == configs[1].v41_q0_lateral, 'arms differ in q0 lateral activation')
    swapped = dict(configs[0], v41_history_training=configs[1].v41_history_training,
                   experiment_name=configs[1].experiment_name)
    require(config_identity(swapped) == b['config_sha256'],
            'full resolved training configuration differs beyond history mode and experiment name')
    require(a['run_id'] != b['run_id'], 'arms share a run identity')
    for k in ('source', 'history_bank_sha256', 'initial_parameter_hash', 'dataset_manifest', 'input_metadata'):
        require(a[k] == b[k], f'arm identity differs: {k}')
    source = a['source']['sha256'] if isinstance(a['source'], dict) else a['source']
    require(sha(source) and sha(a['initial_parameter_hash']) and sha(a['history_bank_sha256']),
            'invalid source, initial weights or history bank hash')
    require({k: v for k, v in a['actual_recipe'].items() if k != 'v41_history_training'} ==
            {k: v for k, v in b['actual_recipe'].items() if k != 'v41_history_training'},
            'joint training recipe differs in more than the ordinary window policy')
    exposure, prefixes, catalog, donor, catalog_counts = defaultdict(list), [], None, None, None
    aux_endpoints = 0
    for epoch in range(1, 61):
        rows = [read(root / 'training_audits' / f'epoch={epoch:03d}.json') for root in roots]
        left, right = rows
        for root, m, mode, row in zip(roots, manifests, MODES, rows):
            if require_epoch_test:
                epoch_evaluation(root, 'test', epoch)
            require(row['run_id'] == m['run_id'] and row['config_sha256'] == m['config_sha256']
                    and row['epoch_complete'] and row['completed_epoch'] == epoch
                    and (row['rows'], row['optimizer_steps']) == (76324, 4771),
                    f'incomplete/mismatched E{epoch} training budget')
            history = row['history_training']
            require(history['mode'] == mode and history['bank_sha256'] == m['history_bank_sha256'],
                    f'E{epoch} augmentation bank mismatch')
            exposure[mode].append({k: v for k, v in history.items()
                                   if k not in ('mode', 'bank_sha256', 'batch_sha256', 'loss_sha256',
                                                'weights_sha256')})
            require(row.get('q0_lateral_epoch') == (epoch if cfg.v41_q0_lateral else None),
                    f'E{epoch} incorrect q0 lateral epoch')
            main, aux = row['sampler'], row['auxiliary']
            sampler = aux['sampler']
            require(main['rows'] == row['rows'] and main['batches'] == row['optimizer_steps']
                    and main['source_sha256'] == sampler['source_sha256'] == TRAIN_DATA
                    and sha(main['plan_sha256']) and sha(sampler['plan_sha256']),
                    f'E{epoch} physical queue source or main budget mismatch')
            require(aux['complete'] is True and aux['main_batches'] == main['batches']
                    and 0 < aux['rows'] == sampler['rows'] <= sampler['budget'] == int(19081*cfg.v41_aux_fraction)
                    and 0 < aux['batches'] == sampler['batches'] <= aux['rows']
                    and aux['loss_weight'] == cfg.v41_aux_loss_weight
                    and aux['insertion_indices'] == list(insertion_indices(main['batches'], aux['batches']))
                    and aux['insertion_plan_sha256'] == insertion_identity(aux['insertion_indices']),
                    f'E{epoch} inconsistent auxiliary budget or insertion schedule')
            require(row['joint_replay'].get('policy') == 'ct_seqtrack.fixed_joint_e2.v1'
                    and row['joint_replay'].get('branch0_slots') == 19081, f'wrong E{epoch} replay policy/slots')
        aux_endpoints += left['auxiliary']['rows']
        for k in ('plan_sha256', 'rows', 'batches', 'source_sha256'):
            require(left['sampler'][k] == right['sampler'][k], f'E{epoch} physical queue differs: {k}')
        for k in ('insertion_plan_sha256', 'insertion_indices', 'rows', 'batches', 'loss_weight', 'sampler'):
            require(left['auxiliary'][k] == right['auxiliary'][k], f'E{epoch} auxiliary plan differs: {k}')
        r, s = left['joint_replay'], right['joint_replay']
        for k in ('plan_sha256', 'manifest_sha256', 'donor_weights_sha256', 'used', 'catalog'):
            require(r[k] == s[k], f'E{epoch} fixed replay differs: {k}')
        counts = r['catalog']
        require(sha(r['plan_sha256']) and set(counts) == set(CATEGORIES)
                and all(type(n) is int and n >= 0 for n in counts.values())
                and set(r['used']) <= set(CATEGORIES)
                and all(type(n) is int and n >= 0 for n in r['used'].values()),
                f'E{epoch} invalid fixed replay plan or categories')
        if epoch <= 2:
            h, j = left['history_training'], right['history_training']
            fields = ('batch_sha256', 'loss_sha256', 'weights_sha256')
            require(all(sha(h[k]) and h[k] == j[k] for k in fields), f'E{epoch} warmup mismatch')
            require(sum(counts.values()) == sum(r['used'].values()) == 0 and r['manifest_sha256'] is None
                    and r['donor_weights_sha256'] is None, f'E{epoch} premature replay')
            prefixes.append({k: h[k] for k in fields})
        else:
            if epoch == 3:
                catalog, donor = r['manifest_sha256'], r['donor_weights_sha256']
                catalog_counts = counts
                require(donor == prefixes[1]['weights_sha256'], 'fixed replay donor differs from common E2 weights')
            require(sha(catalog) and sha(donor) and r['manifest_sha256'] == catalog
                    and r['donor_weights_sha256'] == donor and counts == catalog_counts
                    and all(r['used'].get(k, 0) == min(n, CAPACITY_PER_CATEGORY) for k, n in counts.items()),
                    f'E{epoch} invalid fixed catalog or replay quota')
    for root, row in zip(roots, rows):
        final = read(root / 'training_budget.json')
        require(final['completed_epoch'] == 60 and final['epoch_complete'] is True
                and final['last_epoch_rows'] == row['rows'] and final['last_epoch_steps'] == row['optimizer_steps']
                and final['optimizer_steps'] == 60*4771 and final['sampler'] == row['sampler']
                and final['auxiliary'] == row['auxiliary'], 'final budget disagrees with completed epoch audits')
    shared = {k: v for k, v in a['actual_recipe'].items() if k != 'v41_history_training'}
    shared.update(v41_joint_replay=cfg.v41_joint_replay, v41_q0_lateral=cfg.v41_q0_lateral,
                  v41_q0_lateral_warmup_epochs=cfg.v41_q0_lateral_warmup_epochs if cfg.v41_q0_lateral else None)
    return dict(runs=[m['run_id'] for m in manifests], training_eval_split=cfg.eval_split,
                history_bank_sha256=a['history_bank_sha256'],
                shared_recipe=shared, source_sha256=source, initial_parameter_hash=a['initial_parameter_hash'],
                train_dataset_sha256=TRAIN_DATA, auxiliary_endpoints_per_arm=aux_endpoints,
                common_warmup=prefixes, replay_manifest_sha256=catalog, donor_weights_sha256=donor,
                main_endpoints_per_epoch=76324, main_updates_per_arm=60*4771, history_exposure=exposure)


def q0_measurement(row, *, feedback):
    from tools.compare_v41_q0 import eligible, q0_errors

    before = row.get('diagnostic_original_q0_box_world')
    after = row.get('diagnostic_fine_box_world')
    target = row.get('diagnostic_target_box_world')
    points = row.get('diagnostic_q0_lateral_unique_points')
    applied = row.get('diagnostic_q0_lateral_applied_m')
    require(all(isinstance(box, list) and len(box) == 4 and all(math.isfinite(x) for x in box)
                for box in (before, after, target)) and type(points) is int and points >= 0
            and isinstance(applied, (int, float)) and math.isfinite(applied), 'invalid q0 lateral diagnostics')
    require((feedback and points >= 3) or applied == 0., 'q0 moved during warmup or without point support')
    dx, dy = -math.sin(before[3])*applied, math.cos(before[3])*applied
    require(math.isclose(after[0] - before[0], dx, abs_tol=1e-4)
            and math.isclose(after[1] - before[1], dy, abs_tol=1e-4) and after[2:] == before[2:],
            'q0 box movement differs from the recorded lateral displacement')
    original = dict(row, diagnostic_fine_box_world=before)
    return dict(eligible=eligible(row), target=target, original_width_mae_cm=q0_errors(original)[1],
                corrected_width_mae_cm=q0_errors(row)[1], mean_abs_applied_cm=100*abs(applied))


def epoch_evaluation(root, split, epoch):
    report = read(root / 'epochs' / f'epoch={epoch:03d}.json')
    result = report['evaluation'].get(split, {})
    require(report['epoch'] == result.get('epoch') == result.get('checkpoint_epoch') == epoch
            and result.get('split') == split and result.get('complete_coverage') is True,
            f'expected a complete E{epoch} {split} evaluation in the training run')
    return result


def read_split(root, split, epoch, *, run=None):
    recipe = run['actual_recipe'] if run else {}
    epoch_split = (run or {}).get('evaluation_protocol', {}).get('split', 'dev')
    if split == epoch_split:
        result = epoch_evaluation(root, split, epoch)
    elif split == 'test':
        # 旧 dev 训练另外保存的固定 checkpoint 独立 test。
        root = root / 'heldout_test' / f'epoch={epoch:03d}'
        evaluated = read(root / 'run_manifest.json')
        require(run is not None and evaluated['source_run_id'] == run['run_id'] and
                evaluated['checkpoint_completed_epoch'] == epoch and evaluated['run_type'] == 'formal' and
                evaluated['source'] == run['source'] and evaluated['actual_recipe'] == run['actual_recipe']
                and evaluated['config_sha256'] == run['config_sha256']
                and evaluated['history_bank_sha256'] == run['history_bank_sha256']
                and evaluated['dataset_manifest'] == run['dataset_manifest']
                and evaluated['input_metadata'] == run['input_metadata'],
                f'E{epoch} test ledger belongs to another checkpoint/run')
        result = read(root / 'results.json')
    else:
        raise ValueError(f'{split} was not evaluated by the {epoch_split} training protocol')
    require(result['epoch'] == result['checkpoint_epoch'] == epoch
            and result['split'] == split and result['complete_coverage'] is True
            and result['diagnostics_level'] == 'full', f'expected a complete E{epoch} {split} run')
    parts = PurePosixPath(result['frame_file'].replace(chr(92), '/')).parts[-4:]
    require(len(parts) == 4 and parts[:2] == ('frame_streams', split) and parts[2] not in ('.', '..')
            and parts[-1] == f'epoch={epoch:03d}.jsonl', f'invalid {split} frame ledger')
    tracks, digest = defaultdict(dict), hashlib.sha256()
    with root.joinpath(*parts).open('rb') as stream:
        for line in stream:
            digest.update(line)
            row = json.loads(line)
            scene, track, frame = str(row['scene_id']).zfill(4), row['track_id'], row['frame']
            scans = tracks[scene, track]
            require(scene in (('0017', '0018') if split == 'dev' else ('0019', '0020'))
                    and frame not in scans and row['initialization'] is (frame == 0),
                    f'invalid physical track/frame: {scene}, {track}, {frame}')
            item = {k: row[k] for k in ('raw_frame_id', 'timestamp', 'iou', 'distance', *METRICS)}
            item['raw'] = None if frame == 0 else int(row['diagnostic_search_raw_target_count'])
            require(all(0 <= item[k] <= 1 for k in ('iou', *METRICS)) and math.isfinite(item['timestamp'])
                    and math.isfinite(item['distance']) and item['distance'] >= 0
                    and (frame == 0 or item['raw'] >= 0), 'invalid ledger measurements')
            if frame == 0:
                require([item[k] for k in METRICS] == [1, 1] and item['distance'] == 0,
                        'invalid first-frame metrics')
            elif recipe.get('v41_q0_lateral'):
                item['q0'] = q0_measurement(row, feedback=epoch > recipe['v41_q0_lateral_warmup_epochs'])
            scans[frame] = item
    if result.get('frame_file_sha256'):
        require(digest.hexdigest() == result['frame_file_sha256'], 'frame ledger checksum mismatch')
    ordered = {}
    for key, frames in tracks.items():
        require(sorted(frames) == list(range(len(frames))), f'incomplete track: {key}')
        ordered[key] = [frames[i] for i in range(len(frames))]
        require(all(b['raw_frame_id'] > a['raw_frame_id'] and b['timestamp'] > a['timestamp']
                    for a,b in zip(ordered[key], ordered[key][1:])), f'unordered track: {key}')
    all_frames = [row for track in ordered.values() for row in track]
    require(len(all_frames) == result['frames'], f'{split} frame count disagrees with result')
    for k in METRICS:
        require(math.isclose(100*fmean(r[k] for r in all_frames), result[k], abs_tol=.0002),
                f'{split} {k} disagrees with result')
    return ordered


def follow_reappearance(track, start):
    origin = track[start]
    streak, last = 0, origin
    for j in range(start, len(track)):
        row = track[j]
        if (row['timestamp'] - origin['timestamp'] > 1.0 + 1e-6 or
                (j > start and row['raw_frame_id'] != last['raw_frame_id'] + 1)):
            break
        streak = streak + 1 if row['iou'] >= .5 else 0
        if streak == 3:
            return 'stable', row['timestamp'] - origin['timestamp']
        last = row
    status = 'observed_no' if last['timestamp'] - origin['timestamp'] >= 1.0 - 1e-6 else 'censored'
    return status, None


def outcome(track, start):
    return follow_reappearance(track, start)[0]


def return_summary(events, mode):
    counts = Counter(event[mode] for event in events)
    scored = counts['stable'] + counts['observed_no']
    confirmed = [event['confirmation_seconds'][mode] for event in events if event[mode] == 'stable']
    return dict(events=len(events), stable=counts['stable'], observed_no=counts['observed_no'],
                censored=counts['censored'], assessable=scored,
                stable_fraction_all=counts['stable']/len(events) if events else None,
                mean_confirmation_seconds=fmean(confirmed) if confirmed else None,
                stable_fraction_assessable=counts['stable']/scored if scored else None)


def assess_pair(a, b, *, q0_lateral=False):
    require(a.keys() == b.keys(), 'evaluation track IDs differ')
    paired, returns, severe = [], [], [0., 0.]
    sums, n = [[0., 0.], [0., 0.]], 0
    common_q0 = []
    for key, track in sorted(a.items()):
        other = b[key]
        require(len(track) == len(other) and all(
            (r['timestamp'], r['raw_frame_id'], r['raw']) == (s['timestamp'], s['raw_frame_id'], s['raw'])
            for r, s in zip(track, other)), f'different physical frames/raw returns: {key}')
        n += len(track)
        if q0_lateral:
            for left, right in zip(track[1:], other[1:]):
                measured = (left['q0'], right['q0'])
                require(all(math.isclose(x, y, abs_tol=2e-4) for x, y in
                            zip(measured[0]['target'], measured[1]['target'])), 'paired q0 target boxes differ')
                if all(item['eligible'] for item in measured):
                    common_q0.append(measured)
        pair = {}
        for arm, values in enumerate((track, other)):
            seconds = sum(row['timestamp'] - prev['timestamp'] for prev, row in zip(values, values[1:])
                          if row['raw_frame_id'] == prev['raw_frame_id'] + 1 and row['distance'] > 10.)
            severe[arm] += seconds
            pair[MODES[arm]] = dict(severe_seconds=seconds)
            for j, k in enumerate(METRICS):
                score_sum = sum(r[k] for r in values)
                sums[arm][j] += score_sum
                pair[MODES[arm]][k] = 100 * score_sum / len(values)
        paired.append(dict(scene=key[0], track=key[1], frames=len(track), **pair,
                           window_minus_four_pp={k:pair[MODES[0]][k]-pair[MODES[1]][k] for k in METRICS}))
        for i in range(2, len(track)):
            if track[i-1]['raw'] == 0 and track[i]['raw'] > 0 and (
                    track[i]['raw_frame_id'] == track[i-1]['raw_frame_id'] + 1):
                observed = [follow_reappearance(values, i) for values in (track, other)]
                returns.append(dict(scene=key[0], track=key[1], scan=track[i]['raw_frame_id'], raw=track[i]['raw'],
                                    confirmation_seconds={m: delay for m, (_, delay) in zip(MODES, observed)},
                                    **{m: status for m, (status, _) in zip(MODES, observed)}))
    totals = {mode: {**{k:100*sums[arm][i]/n for i,k in enumerate(METRICS)},
                     'severe_seconds': severe[arm]} for arm,mode in enumerate(MODES)}
    result = dict(frames=n, total=totals, paired_tracks=paired, return_events=returns,
                return_counts={m:return_summary(returns, m) for m in MODES},
                window_minus_four_pp={k:totals[MODES[0]][k]-totals[MODES[1]][k] for k in METRICS})
    if q0_lateral:
        fields = ('original_width_mae_cm', 'corrected_width_mae_cm', 'mean_abs_applied_cm')
        result['q0_common_visible'] = dict(frames=len(common_q0),
            arms={mode: {k: fmean(pair[i][k] for pair in common_q0) if common_q0 else None for k in fields}
                  for i, mode in enumerate(MODES)})
    return result


def compare(roots, split=None, *, require_q0_lateral=False, require_epoch_test=False):
    audit = training_pair(roots, require_q0_lateral=require_q0_lateral, require_epoch_test=require_epoch_test)
    split = split or audit['training_eval_split']
    require(not require_epoch_test or split == 'test', 'per-epoch test comparison requires the test split')
    runs = [read(root / 'run_manifest.json') for root in roots]
    epochs = {e:assess_pair(*(read_split(root, split, e, run=run)
                              for root,run in zip(roots, runs)),
                            q0_lateral=audit['shared_recipe']['v41_q0_lateral']) for e in EPOCHS}
    def events(rows):
        return [(r['scene'], r['track'], r['scan'], r['raw']) for r in rows]
    require(all(events(epochs[e]['return_events']) == events(epochs[60]['return_events']) for e in EPOCHS),
            'physical return events vary across epochs')
    by_track = defaultdict(list)
    for e in EPOCHS:
        for row in epochs[e]['paired_tracks']:
            by_track[row['scene'], row['track']].append(row)
    track_deltas = []
    for (scene, track), rows in sorted(by_track.items()):
        require(len(rows) == len(EPOCHS) and all(row['frames'] == rows[0]['frames'] for row in rows),
                'physical track coverage varies across epochs')
        track_deltas.append(dict(scene=scene, track=track, frames_per_epoch=rows[0]['frames'],
                                 window_minus_four_pp={k:fmean(r['window_minus_four_pp'][k] for r in rows)
                                                       for k in METRICS}))
    late = {m:{k:fmean(epochs[e]['total'][m][k] for e in EPOCHS) for k in (*METRICS, 'severe_seconds')}
            for m in MODES}
    delta = {k:late[MODES[0]][k] - late[MODES[1]][k] for k in METRICS}
    diffs = list(delta.values()) + list(epochs[60]['window_minus_four_pp'].values())
    choice = (MODES[0] if min(diffs) > 0 else MODES[1] if max(diffs) < 0 else 'inconclusive')
    return dict(schema='ct_seqtrack.v41.window_ab.v1', split=split, training=audit, epochs=epochs,
                evaluation_source='training_epochs' if split == audit['training_eval_split'] else 'standalone_test',
                late3=late, late3_window_minus_four_pp=delta, late3_tracks=track_deltas,
                physical_return_events=len(epochs[60]['return_events']),
                fixed_epoch_preference=choice if split == audit['training_eval_split'] else None,
                single_seed_dev_choice=choice if split == 'dev' else None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--window-run', type=Path, required=True)
    parser.add_argument('--candidate-run', type=Path, required=True)
    parser.add_argument('--split', choices=('dev', 'test'), help='默认读取该运行训练期间使用的 split')
    parser.add_argument('--require-epoch-test', action='store_true', help='核验两组 60 轮训练均完整评估 test')
    parser.add_argument('--require-q0-lateral', action='store_true', help='要求两组正式训练均启用 q0 横向头')
    args = parser.parse_args()
    print(json.dumps(compare((args.window_run, args.candidate_run), args.split,
                             require_q0_lateral=args.require_q0_lateral,
                             require_epoch_test=args.require_epoch_test), indent=2, allow_nan=False))
