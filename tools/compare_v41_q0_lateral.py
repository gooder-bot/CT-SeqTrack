"""将一次 q0 横向训练的固定 E60、E58–60 与历史 uniform KITTI test 闭环逐帧配对。"""
import argparse
import json
import math
from pathlib import Path, PurePosixPath
from statistics import fmean
import sys

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.build_v41_q0_replay import THRESHOLDS, difficulty
from tools.compare_v41_joint_search import EPOCHS, RECEIPTS, load_rows, ordered, read_json, require
from tools.compare_v41_q0 import AXES, REFERENCE_TRAINING, average_tree, budget, q0_errors, sp


def geometry(row):
    if row['initialization']:
        return {}
    fields = ('diagnostic_target_box_world', 'diagnostic_fine_box_world', 'diagnostic_coarse_box_world',
              'diagnostic_target_size_lwh', 'diagnostic_first_frame_size_lwh')
    for name in fields:
        size = 3 if 'size' in name else 4
        require(len(row[name]) == size and all(map(math.isfinite, row[name])), f'invalid {name}')
    require(type(row['q0_is_fallback']) is bool and type(row['coarse_valid']) is bool, 'invalid q0 validity')
    genuine = row['coarse_valid'] and not row['q0_is_fallback']
    visible = genuine and row['diagnostic_sampled_target_count'] >= THRESHOLDS['min_target_points']
    hard = False
    if visible:
        iou = row['coarse_geometry']['iou']
        require(0 <= iou <= 1, 'invalid coarse IoU')
        _, reasons = difficulty(np.asarray(row[fields[2]]), np.asarray(row[fields[0]]), iou)
        hard = bool(reasons)
    fine, target = row[fields[1]], row[fields[0]]
    return dict(target=target, target_size=row[fields[3]], first_size=row[fields[4]], visible=visible,
                hard=hard, errors=q0_errors(row),
                axis_yaw_deg=abs(math.degrees(math.remainder(fine[3] - target[3], math.pi))))


def lateral_geometry(row):
    if not row['initialization']:
        require('diagnostic_original_q0_box_world' in row and 'diagnostic_q0_lateral_applied_m' in row,
                'expected full q0 lateral diagnostics from the new run')
    return geometry(row)


def arm_summary(rows, tracks, common, hard):
    return dict(**sp(rows.values()), track_equal_success=fmean(sp(values)['success'] for values in tracks.values()),
                over10m_frames=sum(row['distance'] > 10 for row in rows.values()),
                eligible_frames=sum(row.get('visible', False) for row in rows.values()),
                common_visible_frames=len(common), common_visible_hard_frames=len(hard),
                q0_mae_cm={axis: fmean(rows[key]['errors'][i] for key in common) if common else None
                           for i, axis in enumerate(AXES)},
                q0_axis_yaw_p95_deg=float(np.percentile([rows[k]['axis_yaw_deg'] for k in common], 95))
                    if common else None,
                difficult_lateral_mae_cm=fmean(rows[key]['errors'][1] for key in hard) if hard else None)


def paired_epoch(old, new):
    require(old.keys() == new.keys(), 'physical test inventory differs')
    common, hard = [], []
    for key, a in old.items():
        b = new[key]
        require(a['frame'] == b['frame'] and a['timestamp'] == b['timestamp'] and a.get('raw') == b.get('raw'),
                f'unmatched scan or annotation at {key}')
        if a['initialization']:
            continue
        for name in ('target', 'target_size', 'first_size'):
            require(np.allclose(a[name], b[name], rtol=0, atol=1e-4), f'different {name} at {key}')
        if a['visible'] and b['visible']:
            common.append(key)
            if a['hard']:
                hard.append(key)
    old_tracks, new_tracks = ordered(old), ordered(new)
    tracks = {}
    for key, first in old_tracks.items():
        second = new_tracks[key]
        gain = math.fsum(b['success'] - a['success'] for a, b in zip(first, second))
        streak, last_scan, newly_lost = 0, None, False
        for a, b in zip(first[1:], second[1:]):
            if last_scan is not None and b['scan'] != last_scan + 1:
                streak = 0
            streak = streak + 1 if b['iou'] < .1 and a['iou'] >= .1 else 0
            newly_lost |= streak >= 3
            last_scan = b['scan']
        tracks[f'{key[0]}/{key[1]}'] = dict(frames=len(first), success_gain_sum=gain,
            uniform=sp(first), lateral=sp(second), success_contribution_pp=100 * gain / len(old),
            uniform_over10m_frames=sum(a['distance'] > 10 for a in first),
            lateral_over10m_frames=sum(b['distance'] > 10 for b in second),
            new_sustained_loss=newly_lost, new_severe_drift=any(b['distance'] > 10 for b in second)
                and not any(a['distance'] > 10 for a in first))
    return (dict(uniform=arm_summary(old, old_tracks, common, hard),
                 lateral=arm_summary(new, new_tracks, common, hard)), tracks)


def tail_audit(per_epoch):
    names = sorted(per_epoch[0])
    require(all(sorted(table) == names for table in per_epoch), 'track inventory differs across epochs')
    gains = {key: math.fsum(table[key]['success_gain_sum'] for table in per_epoch) for key in names}
    best = min(names, key=lambda key: (-gains[key], key))
    remaining = sum(t[key]['frames'] for t in per_epoch for key in names if key != best)
    total_gain = math.fsum(gains.values())
    rest_gain = math.fsum(gains[key] for key in names if key != best)
    concentrated = total_gain > 0 and rest_gain <= 0
    sustained = sorted(k for k in names if any(t[k]['new_sustained_loss'] for t in per_epoch))
    return dict(top_contributor=best, success_delta_without_top_pp=100 * rest_gain / remaining if remaining else None,
                gain_concentrated_in_one_track=concentrated, new_sustained_loss_tracks=sustained,
                new_over10m_tracks=sorted(k for k in names if any(t[k]['new_severe_drift'] for t in per_epoch)),
                concentrated_gain_and_other_track_loss=concentrated and any(k != best for k in sustained))


def adoption(result, tail):
    a, b = result['uniform'], result['lateral']
    before, after = a['difficult_lateral_mae_cm'], b['difficult_lateral_mae_cm']
    checks = dict(success_not_lower=b['success'] >= a['success'], precision_not_lower=b['precision'] >= a['precision'],
                  difficult_lateral_mae_lower=None if before is None or after is None else after < before,
                  over10m_not_higher=b['over10m_frames'] <= a['over10m_frames'],
                  concentration_and_loss_veto_clear=not tail['concentrated_gain_and_other_track_loss'])
    return dict(checks=checks, meets_adoption_criteria=all(checks.values()))


def comparison(root, receipts):
    root, receipts = Path(root), Path(receipts)
    manifest = read_json(root / 'run_manifest.json')
    cfg = yaml.safe_load((root / 'resolved_config.yaml').read_text(encoding='utf-8-sig'))
    history = read_json(receipts.parent / 'dev_fixed_comparison.json')
    frozen_rows = [r for r in read_json(receipts)['receipts'] if r['arm'] == 'uniform']
    frozen = {r['epoch']: r for r in frozen_rows}
    require(len(frozen_rows) == len(EPOCHS) and set(frozen) == set(EPOCHS)
            and all(r['source_sha256'] == history['source_sha256'] for r in frozen_rows),
            'missing or mismatched frozen uniform test receipts')
    recipe = manifest['actual_recipe']
    execution = manifest['training_execution_recipe']
    require(manifest['model'] == 'ctseqtrackv41' and manifest['run_type'] == 'formal' and
            cfg['v41_q0_lateral_replay'] and not cfg.get('v41_joint_search') and not cfg.get('test')
            and not cfg.get('checkpoint') and not cfg.get('init_checkpoint') and
            execution['implementation_id'] == 'v41_q0_lateral_replay_v1'
            and execution['execution_revision'] == cfg['v41_execution_revision']
            and all(manifest[k] for k in ('run_id','config_sha256','dataset_manifest','input_metadata'))
            and manifest['source']['sha256']
            and cfg['eval_split'] == 'test'
            and manifest['evaluation_rng_policy'] == 'isolated_split_seed_v2'
            and manifest['evaluation_protocol']['split'] == 'test' and not manifest.get('source_run_id')
            and manifest['run_id'] != history['arms']['A']['run_id'] and manifest['q0_replay_manifest_sha256']
            and recipe.get('v41_q0_lateral_replay') is True and recipe['v41_q0_center_weighting'] == 'uniform'
            and all(recipe.get(k) == cfg.get(k) == v for k, v in {**REFERENCE_TRAINING,
                'dataset':'kitti_mf','v41_arm':'full','category_name':'Car'}.items()),
            'expected a single formal q0 lateral scratch60 KITTI Full test training run')
    _, totals = budget(root, manifest)
    require(totals == history['budget'], 'actual training budget differs from historical uniform')
    for epoch in range(1, 61):
        audit = read_json(root / 'training_audits' / f'epoch={epoch:03d}.json')
        replay = audit['q0_replay']
        require(replay['manifest_sha256'] == manifest['q0_replay_manifest_sha256']
                and 4 * replay['branch0_slots'] == audit['rows']
                and replay['used']['hard'] == replay['used']['normal'] == replay['planned_pairs']
                == min(replay['branch0_slots'] // 4, replay['available_events']),
                f'q0 replay identity or paired budget differs at E{epoch}')
        donors = {item['epoch']: item for item in replay['donors']}
        require(set(donors) == {10, 60} and all(d['run_id'] == history['arms']['A']['run_id']
                and d['source_sha256'] == history['source_sha256'] for d in donors.values())
                and donors[60]['checkpoint_sha256'] == frozen[60]['checkpoint_sha256'],
                'q0 replay donors differ from frozen uniform source or E60 checkpoint')
    epochs, tables, inventory = {}, [], None
    for e in EPOCHS:
        report = read_json(root / 'epochs' / f'epoch={e:03d}.json')
        current = report['evaluation']['test']
        require(report['epoch'] == current['epoch'] == current['checkpoint_epoch'] == e
                and current['diagnostics_level'] == 'full' and current['split'] == 'test'
                and current['complete_coverage'] is True, f'expected complete fixed E{e} test')
        parts = PurePosixPath(current['frame_file'].replace('\\', '/')).parts[-4:]
        require(len(parts) == 4 and parts[:2] == ('frame_streams','test') and parts[2] != '..'
                and parts[-1] == f'epoch={e:03d}.jsonl', 'invalid test ledger path')
        ref = frozen[e]
        old, old_hash = load_rows(receipts.parent / ref['archive'], ref, joint=False, extra_fields=geometry)
        new, new_hash = load_rows(root.joinpath(*parts), current, joint=False, extra_fields=lateral_geometry)
        require(inventory is None or inventory == set(old), 'fixed-epoch physical inventory differs')
        inventory = set(old)
        result, tracks = paired_epoch(old, new)
        tables.append(tracks)
        epochs[e] = dict(metrics=result, tracks=tracks, tail=tail_audit([tracks]), uniform_source_sha256=ref['source_sha256'],
                         uniform_checkpoint_sha256=ref['checkpoint_sha256'], uniform_frame_sha256=old_hash,
                         lateral_frame_sha256=new_hash)
    late = average_tree([epochs[e]['metrics'] for e in EPOCHS])
    tail = tail_audit(tables)
    late_tracks = {key: dict(uniform=average_tree([t[key]['uniform'] for t in tables]),
        lateral=average_tree([t[key]['lateral'] for t in tables]),
        mean_success_contribution_pp=fmean(t[key]['success_contribution_pp'] for t in tables),
        new_sustained_loss_epochs=[e for e, t in zip(EPOCHS, tables) if t[key]['new_sustained_loss']],
        new_over10m_epochs=[e for e, t in zip(EPOCHS, tables) if t[key]['new_severe_drift']])
        for key in sorted(tables[0])}
    return dict(schema='ct_seqtrack.q0_lateral.paired_test.v1', fixed_epochs=list(EPOCHS), split='test',
        history=dict(run_id=history['arms']['A']['run_id'], execution=history['execution_revision'],
                     inference_recovery_enabled=True,
                     source_sha256=history['source_sha256'], test_receipts=str(receipts),
                     training_budget_receipt=str(receipts.parent / 'dev_fixed_comparison.json')),
        lateral_run=dict(run_id=manifest['run_id'], execution=cfg['v41_execution_revision'],
                         inference_recovery_enabled=execution['recovery'].get('inference_enabled', True),
                         source_sha256=manifest['source']['sha256'], config_sha256=manifest['config_sha256'],
                         replay_manifest_sha256=manifest['q0_replay_manifest_sha256']), budget=totals,
        convention='Each epoch uses both real q0 and >=10 sampled target points for common visibility. '
            'Difficulty is fixed by the historical coarse box and training thresholds. Each arm runs on its own history. '
            'Late3 equally averages epoch summaries. New sustained loss means >=3 consecutive physical scans '
            'with new IoU <0.1 and baseline IoU >=0.1 on every scan. Remove the largest frame-weighted '
            'Success contributor to test gain concentration.',
        epochs=epochs, final60=epochs[60], late3=dict(metrics=late, tracks=late_tracks, tail=tail, **adoption(late, tail)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='新训练目录，含 E58–60 的 full test 明细')
    parser.add_argument('--baseline-receipts', type=Path, default=RECEIPTS,
                        help='历史 test_verification.json，旁边须有 test_frames/ 和 dev_fixed_comparison.json')
    args = parser.parse_args()
    print(json.dumps(comparison(args.run, args.baseline_receipts), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
