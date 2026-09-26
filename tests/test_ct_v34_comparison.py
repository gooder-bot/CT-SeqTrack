"""v34 正式比较的合成证据测试；不训练、不读取/修改正式 output。"""
from copy import deepcopy
import json

import pytest
import yaml

from tools import compare_v34_b0 as compare
from utils.tracking_metrics import metric_contributions


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def seal_sampler(sampler):
    sampler['manifest_sha256'] = compare.digest({k: v for k, v in sampler.items() if k != 'manifest_sha256'})
    return sampler


def make_rows(label, *, dynamic_iou=.8, static_iou=.8, dynamic_distance=.2, static_distance=.2):
    rows = []
    for track in ('dynamic', 'quiet'):
        for frame in range(5):
            initial = frame == 0
            iou = 1. if initial else (dynamic_iou if track == 'dynamic' else static_iou)
            distance = 0. if initial else (dynamic_distance if track == 'dynamic' else static_distance)
            s, p = metric_contributions(iou, distance)
            row = dict(tracklet=track, frame=frame, initialization=initial, iou=iou, distance=distance,
                success=float(s), precision=float(p), scene_id='test_scene', timestamp=frame * .5)
            if not initial:
                raw = (0, 2, 5, 12)[frame - 1]
                row.update(diagnostic_target_count=raw, diagnostic_gt_xy_displacement=.2 if track == 'dynamic' and frame == 2 else .01)
                if label != 'old_b0':
                    geometry = dict(iou=iou, center_error_m=distance, center_xy_error_m=distance,
                                    yaw_error_rad=0., axis_yaw_error_rad=0.)
                    row.update(diagnostic_crop_target_count=raw, diagnostic_sampled_target_count=raw,
                        diagnostic_crop_point_count=raw, diagnostic_sampled_point_count=raw,
                        diagnostic_empty_crop=raw == 0, diagnostic_background_only_crop=False,
                        coarse_valid=True, coarse_geometry=geometry, fine_geometry=deepcopy(geometry),
                        selected_index=0, selected_quality=.8, strong=raw >= 3, supported=raw >= 3,
                        memory_write=raw >= 3, memory_wrong_write=False)
                if label in ('S', 'W'):
                    row.update(diagnostic_history_support=[[.2, .3, .4]] * 3,
                               diagnostic_current_support=[.1, .2, .3], diagnostic_query_context_norm=.75)
            rows.append(row)
    if label in ('reference', 'old_b0'):
        for row in rows:
            # 历史版本确实可能没有这些字段，不伪装成已记录的零值。
            row.pop('diagnostic_gt_xy_displacement', None)
            if label == 'old_b0':
                row.pop('timestamp', None)
    return rows


def replace_evaluation(root, label, **kwargs):
    manifest = read(root / 'run_manifest.json')
    rows = make_rows(label, **kwargs)
    for epoch in compare.EPOCHS:
        directory = root / 'evaluation' / f'epoch={epoch:03d}'
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'frames.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
        metrics = dict(schema=manifest['schema'], checkpoint_epoch=epoch, metric_mode='benchmark_compat',
                       complete_coverage=True, **compare.COUNTS, **compare.score(rows))
        write(directory / 'metrics.json', metrics)
    write(root / 'results.json', dict(final=metrics, late3=compare.score(rows), checkpoint_epochs=compare.EPOCHS))


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setattr(compare, 'COUNTS', dict(tracklets=2, frames=10, prediction_frames=8))
    monkeypatch.setattr(compare, 'EVER_MOVING_COUNTS', (1, 4))
    monkeypatch.setattr(compare, 'FIRST_RAW0_COUNTS', (2, 2))
    roots = {}
    for label in compare.LABELS:
        root = tmp_path / label
        root.mkdir()
        roots[label] = root
        version = 34 if label in ('S', 'W') else (32 if label == 'old_b0' else 33)
        schema = f'ct_seqtrack.joint_identity.v{version}'
        model = 'seqtrack_reference' if label == 'reference' else f'ctseqtrackv{version}'
        config = dict(compare.COMMON_CONFIG, net_model=model, seed=42,
            experiment_family=f'ct_seqtrack_v{version}', experiment_name=label,
            v31_short_window=4 if label == 'W' else 3, ct_engineering_check=False,
            test=False, init_checkpoint=None, v31_evaluate_late3=True, lr_warmup_steps=0)
        if label in ('baseline', 'S', 'W'):
            config.update(lr=5e-5, lr_schedule='multistep', lr_milestones=[20, 50])
        manifest = dict(schema=schema, model=model, seed=42, arm='b0',
            enabled=dict(B1=False, B2=False, B3=False), evaluation_rng_policy='per_checkpoint_seed_v1',
            source=dict(files={'synthetic.py': 'b' * 64}))
        manifest['source']['sha256'] = compare.digest(manifest['source']['files'])
        if label == 'reference':
            manifest['reference_protocol'] = dict(schema='synthetic-reference')
        manifest['config_sha256'] = compare.config_digest(config, manifest)
        write(root / 'run_manifest.json', manifest)
        (root / 'resolved_config.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
        for epoch in range(1, 61):
            sampler = dict(schema='seqtrack_reference.v32.teacher.v1' if label == 'reference'
                           else 'ct_seqtrack.v32.ready_queue.v2', epoch=epoch - 1, seed=42,
                training=True, rows=19108, batch_size=16, source_sha256='a' * 64)
            if label != 'reference':
                sampler.update(short_window=config['v31_short_window'], long_window=8, curriculum_epochs=10,
                    reserve_windows=112, seed_translation=.3, seed_yaw_degrees=1.5, batches=1195,
                    reserve_policy='stratified_singleton_windows_then_fill_v1',
                    drain_policy='running_b0_bn_from_first_reserve_or_partial_v1',
                    seed_policy='shared_anchor_local_translation_world_yaw_v1', plan_sha256='c' * 64)
            seal_sampler(sampler)
            audit = dict(completed_epoch=epoch, epoch_complete=True, rows=19108, optimizer_steps=1195, sampler=sampler)
            if label in ('S', 'W'):
                audit.update(schema='ct_seqtrack.v34.training_audit.v1', model_schema=schema,
                             config_sha256=manifest['config_sha256'])
            write(root / 'training_audits' / f'epoch={epoch:03d}.json', audit)
        write(root / 'training_budget.json', dict(schema=schema, epoch_complete=True, **compare.BUDGET, sampler=sampler))
        values = dict(dynamic_iou=.6, static_iou=.6, dynamic_distance=.5, static_distance=.5)
        if label == 'reference':
            values.update(dynamic_iou=.65, static_iou=.65, dynamic_distance=.45, static_distance=.45)
        if label in ('S', 'W'):
            values.update(dynamic_iou=.8, static_iou=.8, dynamic_distance=.2, static_distance=.2)
        replace_evaluation(root, label, **values)
    return roots


def compare_all(runs):
    return compare.compare_runs(*(runs[k] for k in compare.LABELS))


def test_both_gates_pass_and_short_window_difference_is_registered(runs):
    before = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for root in runs.values() for p in root.rglob('*') if p.is_file()}
    report = compare_all(runs)
    after = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for root in runs.values() for p in root.rglob('*') if p.is_file()}
    assert before == after
    assert report['status'] == 'passed' and report['passing_rank'] == ['S', 'W']
    assert report['arms']['baseline']['groups']['ever_moving']['n'] == 4
    assert report['arms']['S']['groups']['motion_cross_01']['n'] == 1
    assert report['arms']['S']['groups']['motion_cross_10']['n'] == 1
    old = report['arms']['old_b0']['diagnostics']['60']
    assert old['quality']['strong'] is None and old['quality']['recorded_frames'] == 0
    assert old['quality']['supported_bad_iou_rate'] is None
    quality = report['arms']['S']['diagnostics']['60']['quality']
    assert quality['supported_bad_iou_rate'] == 0 and quality['bad_iou_supported_rate'] is None
    assert old['coarse_fine']['predictions']['delta_iou']['mean'] is None
    assert old['context']['recorded_frames'] == 0 and old['context']['query_context_norm_mean'] is None
    context = report['arms']['S']['diagnostics']['60']['context']
    assert context['recorded_frames'] == 8 and context['query_context_norm_mean'] == .75
    assert context['current_support_mean'] == pytest.approx([.1, .2, .3])


def test_raw_contributions_use_all_frames_and_sum_to_overall_delta(runs):
    report = compare_all(runs)
    assert set(report['raw_contributions']) == {'S_minus_C', 'W_minus_S', 'S_minus_R', 'W_minus_R'}
    for comparison in report['raw_contributions'].values():
        assert comparison['denominator_frames_including_initialization'] == 10
        for stage in [*comparison['epochs'].values(), comparison['final60'], comparison['late3']]:
            for metric in compare.METRICS:
                contributions = stage['raw_contribution_pp']
                assert sum(contributions[g][metric] for g in compare.RAW_GROUPS) == pytest.approx(stage['overall_delta_pp'][metric])
    delta = report['arms']['S']['groups']['raw0']['final60']['success'] - report['arms']['baseline']['groups']['raw0']['final60']['success']
    assert report['raw_contributions']['S_minus_C']['final60']['raw_contribution_pp']['raw0']['success'] == pytest.approx(delta * 2 / 10)


@pytest.mark.parametrize('field,value', [
    ('diagnostic_history_support', [[.1, .2, .3]] * 2),
    ('diagnostic_history_support', [[.1, .2, float('inf')]] * 3),
    ('diagnostic_current_support', [.1, .2]),
    ('diagnostic_current_support', [.1, float('nan'), .3]),
    ('diagnostic_query_context_norm', None),
    ('diagnostic_query_context_norm', float('inf'))])
def test_v34_context_requires_exact_shape_and_finite_values(runs, field, value):
    mutate_frame(runs['S'], lambda rows: rows[1].update({field: value}))
    with pytest.raises(compare.EvidenceError, match=field):
        compare_all(runs)


def test_first_raw0_fixed_count_is_checked(runs, monkeypatch):
    monkeypatch.setattr(compare, 'FIRST_RAW0_COUNTS', (36, 36))
    with pytest.raises(compare.EvidenceError, match='first_raw0'):
        compare_all(runs)


def test_overall_pass_does_not_excuse_dynamic_regression(runs):
    for label in ('S', 'W'):
        replace_evaluation(runs[label], label, dynamic_iou=.55, static_iou=.99,
                           dynamic_distance=.6, static_distance=.01)
    report = compare_all(runs)
    assert report['status'] == 'failed' and report['selected'] is None
    for result in report['candidates'].values():
        assert result['gates']['overall']['passed']
        assert not result['gates']['ever_moving']['passed']


def test_passing_groups_rank_by_final_success_then_precision(runs):
    replace_evaluation(runs['W'], 'W', dynamic_iou=.9, static_iou=.9)
    assert compare_all(runs)['selected'] == 'W'


def test_late3_is_a_gate_even_when_final_passes(runs):
    for label in ('S', 'W'):
        root = runs[label]
        final = read(root / 'results.json')['final']
        replace_evaluation(root, label, dynamic_iou=.0, static_iou=.0,
                           dynamic_distance=3., static_distance=3.)
        poor = read(root / 'evaluation/epoch=058/metrics.json')
        poor_rows = (root / 'evaluation/epoch=058/frames.jsonl').read_text(encoding='utf-8')
        replace_evaluation(root, label)
        write(root / 'evaluation/epoch=058/metrics.json', poor)
        (root / 'evaluation/epoch=058/frames.jsonl').write_text(poor_rows, encoding='utf-8')
        write(root / 'results.json', dict(final=final, checkpoint_epochs=compare.EPOCHS,
            late3={m: (poor[m] + 2 * final[m]) / 3 for m in compare.METRICS}))
    report = compare_all(runs)
    assert report['status'] == 'failed'
    for candidate in report['candidates'].values():
        gate = candidate['gates']['overall']
        assert gate['delta_pp']['final60_success'] > 0
        assert gate['delta_pp']['late3_success'] < 0


def test_metrics_tampering_cannot_override_frame_recomputation(runs):
    path = runs['W'] / 'evaluation/epoch=060/metrics.json'
    value = read(path)
    value['precision'] += .1
    write(path, value)
    with pytest.raises(compare.EvidenceError, match='precision'):
        compare_all(runs)


def mutate_frame(root, change):
    path = root / 'evaluation/epoch=060/frames.jsonl'
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    change(rows)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')


def test_frame_key_misalignment_is_invalid(runs):
    mutate_frame(runs['S'], lambda rows: [r.update(tracklet='different') for r in rows if r['tracklet'] == 'dynamic'])
    with pytest.raises(compare.EvidenceError, match='帧键'):
        compare_all(runs)


def test_saved_frame_score_tampering_is_invalid(runs):
    mutate_frame(runs['W'], lambda rows: rows[1].update(success=.123))
    with pytest.raises(compare.EvidenceError, match='success'):
        compare_all(runs)


@pytest.mark.parametrize('field', ['diagnostic_gt_xy_displacement', 'diagnostic_target_count', 'supported'])
def test_missing_group_or_support_fields_never_become_zero(runs, field):
    mutate_frame(runs['S'], lambda rows: rows[1].pop(field))
    with pytest.raises(compare.EvidenceError, match=field):
        compare_all(runs)


def test_identity_error_is_invalid(runs):
    path = runs['S'] / 'run_manifest.json'
    value = read(path)
    value['config_sha256'] = '0' * 64
    write(path, value)
    with pytest.raises(compare.EvidenceError, match='config 身份'):
        compare_all(runs)


@pytest.mark.parametrize('field,value', [('short_window', 3), ('curriculum_epochs', 11)])
def test_recorded_curriculum_must_match_group_and_ten_epoch_contract(runs, field, value):
    path = runs['W'] / 'training_audits/epoch=010.json'
    audit = read(path)
    audit['sampler'][field] = value
    seal_sampler(audit['sampler'])
    write(path, audit)
    with pytest.raises(compare.EvidenceError, match=field):
        compare_all(runs)


def test_displacement_group_cannot_depend_on_new_predictions(runs):
    mutate_frame(runs['W'], lambda rows: rows[2].update(diagnostic_gt_xy_displacement=.01))
    with pytest.raises(compare.EvidenceError, match='diagnostic_gt_xy_displacement'):
        compare_all(runs)


def test_report_output_rejects_external_paths_and_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(compare, 'ROOT', tmp_path)
    with pytest.raises(compare.EvidenceError, match='artifacts/ct_checks'):
        compare.write_report(tmp_path / 'outside.json', {})
    target = tmp_path / 'artifacts/ct_checks/new/report.json'
    compare.write_report(target, {'status': 'passed'})
    with pytest.raises(compare.EvidenceError, match='覆盖'):
        compare.write_report(target, {})
    assert read(target)['status'] == 'passed'


def test_main_exit_codes_and_stdout_json(runs, capsys):
    args = [item for name, label in [('reference', 'reference'), ('baseline', 'baseline'),
        ('old-b0', 'old_b0'), ('s', 'S'), ('w', 'W')] for item in ('--' + name, str(runs[label]))]
    assert compare.main(args) == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'passed'
    for label in ('S', 'W'):
        replace_evaluation(runs[label], label, dynamic_iou=.1, static_iou=.1,
                           dynamic_distance=1.5, static_distance=1.5)
    assert compare.main(args) == 1
    assert json.loads(capsys.readouterr().out)['status'] == 'failed'
    mutate_frame(runs['W'], lambda rows: rows[1].pop('diagnostic_target_count'))
    assert compare.main(args) == 2
    assert json.loads(capsys.readouterr().out)['status'] == 'invalid_evidence'
