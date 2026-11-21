"""Compare one joint KITTI run against the frozen uniform test ledgers at E58–60."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from statistics import fmean, median
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


EPOCHS = (58, 59, 60)
RECEIPTS = (Path(__file__).resolve().parents[1] / 'artifacts' / 'ct_checks' /
            '20261121_q0_final_review' / 'formal' / 'test_verification.json')


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def load_rows(path, expected, *, joint, extra_fields=None):
    """Only retain the signed measurements needed by this audit, verify raw JSONL SHA."""
    rows, digest = {}, hashlib.sha256()
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rb') as stream:
        for line in stream:
            digest.update(line)
            row = json.loads(line)
            scene = str(row['scene_id']).zfill(4)
            key = scene, int(row['track_id']), int(row['raw_frame_id'])
            require(key not in rows and scene in ('0019', '0020'), f'invalid test frame {key}')
            item = {k: row[k] for k in ('frame', 'initialization', 'timestamp', 'iou', 'distance',
                                        'success', 'precision')}
            item['scan'] = key[2]
            require(type(item['initialization']) is bool and item['initialization'] == (item['frame'] == 0)
                    and math.isfinite(item['timestamp']) and math.isfinite(item['distance'])
                    and item['distance'] >= 0 and all(0 <= item[k] <= 1 for k in ('iou','success','precision')),
                    f'invalid metrics at {key}')
            if item['initialization']:
                require((item['iou'], item['distance'], item['success'], item['precision']) == (1, 0, 1, 1),
                        f'invalid initialization at {key}')
            else:
                record = row['diagnostic_candidates']
                item.update(raw=row['diagnostic_search_raw_target_count'],
                            union=row['diagnostic_search_union_target_count'],
                            sampled=row['diagnostic_sampled_target_count'],
                            write=row['trusted_write'], written_good=record['selected_trusted_target'],
                            written_iou=record['selected_geometry']['exact_iou'],
                            selected=int(record['selected_index']))
                require(0 <= item['sampled'] <= item['union'] <= item['raw'], f'invalid coverage at {key}')
                if item['write']:
                    chosen = next(v for v in record['measurement_trust_records'] if v['index'] == item['selected'])
                    item['purity'] = chosen['support']['target_purity']
                if joint:
                    good = [valid and box['exact_iou'] >= .5
                            for valid, box in zip(record['raw_valid'], record['geometry'])]
                    require(record['candidate_order'][4:] == ['local_vote', 'remote_vote']
                            and len(good) == 6, f'invalid joint candidate ledger at {key}')
                    item.update(coarse_hit=any(record['coarse_hits_target_bev']),
                                fine_target=record['remote_sampled_target_count'],
                                old_good=any(good[:4]), local_good=good[4], remote_good=good[5])
                    for k in ('joint_pending_started', 'joint_remote_takeover', 'joint_scan_called',
                              'joint_extra_points', 'joint_scan_points', 'joint_target_support_count',
                              'joint_max_other_support_count'):
                        item[k] = row[k]
                    require(item['joint_remote_takeover'] == (item['selected'] == 5)
                            and (not item['joint_remote_takeover'] or item['write']),
                            f'invalid remote takeover at {key}')
            if extra_fields is not None:
                item.update(extra_fields(row))
            rows[key] = item
    if 'frame_file_sha256' in expected:
        require(digest.hexdigest() == expected['frame_file_sha256'], f'ledger SHA mismatch: {path}')
    require(len(rows) == expected['frames'], f'frame count differs: {path}')
    for k in ('success', 'precision'):
        require(math.isclose(100 * fmean(row[k] for row in rows.values()), expected[k], abs_tol=.0002),
                f'{k} disagrees with official report: {path}')
    return rows, digest.hexdigest()


def ordered(rows):
    tracks = defaultdict(list)
    for (scene, track, scan), row in sorted(rows.items()):
        tracks[scene, track].append(row)
    for key, track in tracks.items():
        require([r['frame'] for r in track] == list(range(len(track)))
                and all(a['timestamp'] < b['timestamp'] for a,b in zip(track, track[1:])),
                f'incomplete track: {key}')
    return tracks


def aggregate(tracks, joint):
    all_rows = [row for track in tracks.values() for row in track]
    predictions = [row for row in all_rows if not row['initialization']]
    writes = [r for r in predictions if r['write']]
    bad = [r for r in writes if not r['written_good']]
    events = recovered = lost_frames = low_iou_starts_empty = 0
    for track in tracks.values():
        lost = False
        for r in track[1:]:
            if not lost and r['iou'] < .1:
                events += 1
                low_iou_starts_empty += int(r['raw'] == 0)
                lost = True
            if lost and r['iou'] >= .5:
                recovered += 1
                lost = False
            lost_frames += int(lost)
    result = dict(frames=len(all_rows), predictions=len(predictions), tracks=len(tracks),
                  success=100*fmean(r['success'] for r in all_rows),
                  precision=100*fmean(r['precision'] for r in all_rows),
                  loss_events=events, loss_starts_raw_empty=low_iou_starts_empty,
                  recovered_events=recovered, unrecovered_events=events-recovered, lost_frames=lost_frames,
                  low_iou_frames=sum(r['iou'] < .1 for r in predictions),
                  over5m=sum(r['distance'] > 5 for r in predictions),
                  over10m=sum(r['distance'] > 10 for r in predictions),
                  raw_empty=sum(r['raw'] == 0 for r in predictions),
                  union_missed_present=sum(r['raw'] > 0 and r['union'] == 0 for r in predictions),
                  sampling_missed_present=sum(r['union'] > 0 and r['sampled'] == 0 for r in predictions),
                  writes=len(writes), wrong_writes=len(bad),
                  wrong_write_fraction=len(bad)/len(writes) if writes else None,
                  wrong_write_low_purity=sum(r['purity'] < .5 for r in bad),
                  wrong_write_geometry_with_pure_support=sum(r['purity'] >= .5 and r['written_iou'] < .5 for r in bad))
    if joint:
        calls = [r for r in predictions if r['joint_scan_called']]
        remote = [r for r in predictions if r['joint_remote_takeover']]
        misses = [r for r in predictions if r['raw'] > 0 and r['union'] == 0]
        def other_car_evidence(row):
            return row['joint_target_support_count'] == 0 and row['joint_max_other_support_count'] >= 3
        result.update(local_selected=sum(r['selected'] == 4 for r in predictions), remote_takeovers=len(remote),
                      remote_verified_other_car=sum(other_car_evidence(r) for r in remote),
                      remote_failed_without_other_car_evidence=sum(
                          not r['written_good'] and not other_car_evidence(r) for r in remote),
                      pending_started=sum(r['joint_pending_started'] for r in predictions), scan_calls=len(calls),
                      calls_per_prediction=len(calls)/len(predictions),
                      sampled_remote_after_local_miss=sum(r['fine_target'] > 0 for r in misses),
                      coarse_hits_after_local_miss=sum(r['coarse_hit'] for r in misses),
                      new_good_local_frames=sum(r['local_good'] and not r['old_good'] for r in predictions),
                      new_good_remote_frames=sum(r['remote_good'] and not r['old_good'] for r in predictions),
                      new_good_frames=sum((r['local_good'] or r['remote_good']) and not r['old_good']
                                          for r in predictions),
                      inspected_points=sum(r['joint_scan_points'] for r in predictions),
                      extra_refinement_points=sum(r['joint_extra_points'] for r in predictions),
                      extra_points_per_call=fmean(r['joint_extra_points'] for r in calls) if calls else None)
        result['fine_target_frames'] = {
            f'at_least_{threshold}': sum(r['fine_target'] >= threshold for r in calls)
            for threshold in (1, 3, 10)}
    return result


def paired_events(baseline, joint):
    # Fixed events: raw return following an actual raw-empty frame and an open historical loss.
    # Censor at the next empty scan or track end. Recovery requires three consecutive IoU >= .5.
    delays = dict(uniform=[], joint=[])
    events = healthy = healthy_rows = 0
    harmed_tracks = []
    healthy_delta = Counter()
    for key, track in baseline.items():
        other = joint[key]
        if all(r['iou'] >= .1 for r in track):
            healthy += 1
            if any(r['iou'] < .1 for r in other):
                harmed_tracks.append(dict(scene=key[0], track_id=key[1]))
            healthy_rows += len(track)
            for k in ('success', 'precision'):
                healthy_delta[k] += sum(b[k] - a[k] for a,b in zip(track, other))
        lost = False
        for i in range(1, len(track)):
            if i > 1 and track[i-1]['raw'] == 0 and track[i]['raw'] > 0 and lost:
                events += 1
                for arm, trajectory in (('uniform', track), ('joint', other)):
                    streak = 0
                    last_scan = None
                    for r in trajectory[i:]:
                        if r['raw'] == 0:
                            break
                        if last_scan is not None and r['scan'] != last_scan + 1:
                            streak = 0
                        streak = streak + 1 if r['iou'] >= .5 else 0
                        last_scan = r['scan']
                        if streak == 3:
                            delays[arm].append(r['timestamp'] - track[i]['timestamp'])
                            break
            if track[i]['iou'] < .1:
                lost = True
            elif track[i]['iou'] >= .5:
                lost = False
    return dict(baseline_healthy_tracks=healthy, healthy_tracks_with_new_loss=len(harmed_tracks),
                newly_lost_healthy_tracks=harmed_tracks,
                healthy_frames=healthy_rows,
                healthy_success_delta_pp=100*healthy_delta['success']/healthy_rows if healthy_rows else None,
                healthy_precision_delta_pp=100*healthy_delta['precision']/healthy_rows if healthy_rows else None,
                baseline_lost_raw_returns=events,
                sustained_recovery={arm: dict(recovered=len(v), censored=events-len(v),
                    median_seconds_to_third_good_frame=median(v) if v else None) for arm,v in delays.items()})


def joint_result(root, epoch, *, standalone):
    manifest = read_json(root/'run_manifest.json')
    if standalone:
        result = read_json(root/'results.json')
        require(manifest['checkpoint_completed_epoch'] == epoch, 'test checkpoint epoch mismatch')
        run_id = manifest['source_run_id']
    else:
        report = read_json(root/'epochs'/f'epoch={epoch:03d}.json')
        require(report['epoch'] == epoch, 'invalid training epoch report')
        result = report['evaluation']['test']
        run_id = manifest['run_id']
    recipe = manifest['actual_recipe']
    require(manifest['run_type'] == 'formal' and recipe.get('v41_joint_search') is True
            and (recipe['v41_arm'], recipe['dataset'], recipe['category_name']) == ('full', 'kitti_mf', 'Car')
            and recipe['v41_q0_center_weighting'] == 'uniform' and recipe['epoch'] == 60
            and result['split'] == 'test' and result['complete_coverage'] is True and 'joint_latency' not in result
            and result['diagnostics_level'] == 'full' and result['epoch'] == result['checkpoint_epoch']
            == epoch, f'invalid joint test run: {root}')
    parts = PurePosixPath(result['frame_file'].replace('\\','/')).parts[-4:]
    require(len(parts) == 4 and parts[:2] == ('frame_streams','test') and parts[2] not in ('.','..')
            and parts[-1] == f'epoch={epoch:03d}.jsonl', f'invalid test ledger path: {parts}')
    extra = None
    if recipe.get('v41_q0_lateral'):
        from tools.compare_v41_q0_lateral import lateral_geometry
        extra = lateral_geometry
    rows, digest = load_rows(root.joinpath(*parts), result, joint=True, extra_fields=extra)
    return run_id, manifest, result, rows, digest


def comparison(roots, receipts, *, standalone, latency_root=None):
    receipts_data = read_json(receipts)
    frozen = {r['epoch']: r for r in receipts_data['receipts'] if r['arm'] == 'uniform'}
    require(set(frozen) == set(EPOCHS), 'missing frozen uniform test ledgers')
    reports = []
    lateral_epochs, lateral_tracks = [], []
    identity = None
    for epoch, root in zip(EPOCHS, roots):
        reference = frozen[epoch]
        run_id, manifest, result, current, digest = joint_result(root, epoch, standalone=standalone)
        recipe = manifest['actual_recipe']
        execution = manifest['training_execution_recipe']
        extra = None
        if recipe.get('v41_q0_lateral'):
            from tools.compare_v41_q0_lateral import geometry, paired_epoch, tail_audit
            from tools.compare_v41_q0 import average_tree
            extra = geometry
        baseline, old_digest = load_rows(receipts.parent/reference['archive'], reference, joint=False,
                                         extra_fields=extra)
        observed = (run_id, manifest['source']['sha256'], manifest['config_sha256'],
                    manifest['input_metadata'], manifest['dataset_manifest'], recipe, execution)
        require(identity is None or identity == observed, 'expected all three checkpoints from a single run')
        identity = observed
        require(baseline.keys() == current.keys(), 'test inventory differs between runs')
        for key, a in baseline.items():
            b = current[key]
            require(a['frame'] == b['frame'] and a['timestamp'] == b['timestamp']
                    and a.get('raw') == b.get('raw'), f'unmatched physical frame: {key}')
        a, b = ordered(baseline), ordered(current)
        old, new = aggregate(a, False), aggregate(b, True)
        delta = {k: new[k]-value for k,value in old.items() if value is not None and new[k] is not None}
        reports.append(dict(epoch=epoch, uniform=old, joint=new, joint_minus_uniform=delta,
                            paired=paired_events(a,b), uniform_sha256=old_digest, joint_sha256=digest))
        if extra is not None:
            metrics, tracks = paired_epoch(baseline, current)
            lateral_epochs.append(dict(uniform=metrics['uniform'], joint=metrics['lateral']))
            lateral_tracks.append(tracks)
            reports[-1]['q0_geometry'] = lateral_epochs[-1]
    late = {}
    for arm in ('uniform','joint','joint_minus_uniform'):
        keys = reports[-1][arm]
        late[arm] = {k: fmean(r[arm][k] for r in reports) for k in keys if
                     all(isinstance(r[arm][k], (float,int)) for r in reports)}
    latency = None
    if latency_root is not None:
        result = read_json(latency_root/'results.json')
        manifest = read_json(latency_root/'run_manifest.json')
        latency = result['joint_latency']
        require(manifest['source_run_id'] == identity[0] and manifest['source']['sha256'] == identity[1]
                and manifest['config_sha256'] == identity[2] and result['split'] == 'test'
                and result['checkpoint_epoch'] == 60 and result['complete_coverage'] is True
                and latency['batch_size'] == 1 and latency['measured_frames'] > 0,
                'latency requires same-run E60 standalone test, batch size one')
    summary = dict(schema='ct_seqtrack.joint_search.paired_test.v1', source_run_id=identity[0],
                compared_recipe=recipe, training_execution_recipe=execution,
                recovery_cohort='Uniform loss open before a raw zero-to-positive return among prediction frames. The frozen initialization lacks raw counts. Confirm at the third consecutive physical scan with IoU >= .5, censored at next raw zero or track end.',
                false_takeover='Conservative point evidence: zero own support and at least three support IDs in one other annotated car.',
                latency_ms=dict(joint=latency, uniform=None, uniform_source='historical ledgers contain no matched B=1 timing'),
                late3_mean=late, final_epoch=60, epochs=reports)
    if lateral_epochs:
        summary['q0_late3'] = dict(metrics=average_tree(lateral_epochs), tail=tail_audit(lateral_tracks))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--joint-run', type=Path,
                       help='Formal joint training directory with built-in full TEST evaluations at E58–60')
    group.add_argument('--joint-tests', nargs=3, type=Path, metavar=('E58','E59','E60'),
                        help='Three standalone --test run directories with full results.json and frame_streams/')
    parser.add_argument('--baseline-receipts', type=Path, default=RECEIPTS)
    parser.add_argument('--latency-test', type=Path, help='Optional E60 --test --v41_joint_latency output directory')
    args = parser.parse_args()
    roots = args.joint_tests if args.joint_tests else [args.joint_run] * len(EPOCHS)
    print(json.dumps(comparison(roots, args.baseline_receipts, standalone=bool(args.joint_tests),
                                latency_root=args.latency_test),
                     indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
