"""被动评测与固定组比较：仅临时合成证据，不改正式 output。"""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch
import yaml

from models.ct_v31.runtime import TrackingEvaluation, loss_episode_summary
from tools import compare_v33_b0 as compare
from utils.tracking_metrics import metric_contributions


def test_loss_episodes_include_intermediate_overlap_and_separate_censoring():
    rows = [dict(tracklet='a', frame=i, initialization=i == 0, iou=iou, timestamp=time)
            for i, (iou, time) in enumerate([(1., 0.), (.05, .4), (.2, 1.1),
                                             (.5, 1.4), (.05, 2.2), (.3, 3.)])]
    report = loss_episode_summary(rows)
    assert report['total_lost_frames'] == 4
    assert report['low_iou_frames'] == 2
    assert report['all']['frames'] == dict(count=2, median=2., p90=2., total=4.)
    assert report['recovered']['seconds']['median'] == pytest.approx(1.)
    assert report['unrecovered']['seconds']['median'] == pytest.approx(.8)
    assert report['unrecovered_rate'] == .5
    assert report['episodes'][0]['end_frame'] == 3
    assert report['episodes'][0]['frames'] == 2  # 恢复帧排除


def test_old_rows_never_invent_seconds_and_zero_events_are_not_zero_recovery_time():
    rows = [dict(tracklet='a', frame=1, initialization=False, iou=.05)]
    report = loss_episode_summary(rows)
    assert report['unrecovered']['frames']['total'] == 1
    assert report['all']['seconds']['count'] == 0
    assert report['all']['seconds']['median'] is None
    assert not report['all']['seconds_complete']
    assert report['unrecovered_rate'] == 1.
    empty = loss_episode_summary([])
    assert empty['unrecovered_rate'] is None
    assert empty['recovered']['seconds']['median'] is None


@pytest.mark.parametrize('sequence_valid', [True, False])
def test_evaluation_geometry_is_passive_and_q0_differs_from_selected_mode(sequence_valid):
    evaluation = TrackingEvaluation()
    request = SimpleNamespace(branch=4, frame=1)
    first = dict(timestamp=10., scene_id='test')
    raw = dict(request=request, tracklet_key='track', first_frame=first,
               frames={1: dict(timestamp=10.6, scene_id='test')})
    target = torch.tensor([[0., 0., 0., 0.]])
    coarse = torch.tensor([[2., 0., 0., 3.14159265]], requires_grad=True)
    fine = torch.tensor([[[1., 0., 0., 0.], [0., 0., 0., 0.]]], requires_grad=True)
    batch = dict(target_box=target, box_size=torch.tensor([[4., 2., 2.]]),
                 target_box_size=torch.tensor([[4., 2., 2.]]),
                 physical_displacement=torch.tensor([[.15, 0.]]),
                 diagnostic_target_count=torch.tensor([0]),
                 diagnostic_crop_target_count=torch.tensor([0]),
                 diagnostic_crop_point_count=torch.tensor([3]),
                 diagnostic_sampled_target_count=torch.tensor([0]),
                 diagnostic_sampled_point_count=torch.tensor([3]))
    output = SimpleNamespace(accepted_box=target, selected_quality=torch.tensor([.9]),
        selected_index=torch.tensor([1]), hypothesis_boxes=fine,
        observation=SimpleNamespace(coarse_box=coarse, sequence_valid=torch.tensor([sequence_valid])),
        evidence=SimpleNamespace(mode_valid=None))
    before = {key: value.clone() for key, value in batch.items()}
    rng = torch.get_rng_state().clone()
    evaluation.add_batch([raw], batch, output, [dict(strong=False, trusted_velocity_reset_reason='initial')])
    row = evaluation.rows[-1]
    assert row['iou'] == pytest.approx(1.)
    assert row['coarse_geometry']['center_error_m'] == pytest.approx(2.)
    assert row['coarse_valid'] is sequence_valid
    assert row['fine_geometry']['center_error_m'] == pytest.approx(1.)
    assert row['coarse_geometry']['axis_yaw_error_rad'] < 1e-6
    assert row['diagnostic_background_only_crop'] and not row['diagnostic_empty_crop']
    assert row['diagnostic_moving']
    assert row['trusted_velocity_reset_reason'] == 'initial'
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(value, before[key]) for key, value in batch.items())
    assert coarse.grad is None and fine.grad is None
    summary = evaluation.summary()
    assert summary['success'] == 100.
    assert 'loss_episodes' in summary
    assert all(value is None or isinstance(value, (int, float)) for value in summary['diagnostics'].values())


def test_paired_refinement_excludes_empty_sequence_dummy_and_reports_both_directions():
    def geometry(iou, error):
        return dict(iou=iou, center_error_m=error, center_xy_error_m=error, yaw_error_rad=0.)
    rows = [dict(coarse_valid=True, coarse_geometry=geometry(.2, 2.), fine_geometry=geometry(.6, 1.)),
            dict(coarse_valid=True, coarse_geometry=geometry(.5, 1.), fine_geometry=geometry(.3, 1.5)),
            dict(coarse_valid=False, coarse_geometry=geometry(1., 0.), fine_geometry=geometry(0., 20.))]
    report = compare._geometry_means(rows)
    assert report['coarse_geometry']['count'] == 2
    paired = report['paired_refinement']
    assert paired['valid_pairs'] == 2
    assert paired['mean_delta_iou'] == pytest.approx(.1)
    assert paired['mean_center_error_reduction_m'] == pytest.approx(.25)
    assert paired['iou']['improved_frames'] == paired['iou']['worsened_frames'] == 1
    assert paired['center_error']['improved_rate'] == paired['center_error']['worsened_rate'] == .5
    assert compare._geometry_means(rows[-1:])['paired_refinement']['mean_delta_iou'] is None


def test_summary_distinguishes_unrecorded_reference_points_from_observed_zero():
    evaluation = TrackingEvaluation()
    target = torch.zeros(1, 4)
    batch = dict(target_box=target, box_size=torch.tensor([[4., 2., 2.]]),
                 target_box_size=torch.tensor([[4., 2., 2.]]))
    output = SimpleNamespace(accepted_box=target, selected_quality=torch.ones(1),
        selected_index=torch.zeros(1, dtype=torch.long), evidence=SimpleNamespace(mode_valid=None))
    fields = dict(raw_target_points='diagnostic_target_count',
                  novel_target_points='diagnostic_novel_target_count',
                  reachable_novel_target_points='diagnostic_reachable_count',
                  acquired_novel_target_points='diagnostic_acquired_count')
    for frame, values in enumerate((None, (0, 0, 0, 0), (7, 3, 2, 1)), start=1):
        if values is not None:
            batch.update({field: torch.tensor([value]) for field, value in zip(fields.values(), values)})
        raw = dict(request=SimpleNamespace(branch=4, frame=frame), tracklet_key='reference',
                   first_frame=dict(timestamp=0.), frames={frame: dict(timestamp=float(frame))})
        evaluation.add_batch([raw], batch, output)
        summary = evaluation.summary()
        diagnostic = summary['diagnostics']
        for index, name in enumerate(fields):
            expected = None if frame == 1 else (0 if frame == 2 else values[index])
            assert diagnostic[name] == expected
            assert diagnostic[name + '_recorded_frames'] == frame - 1
        assert summary['prediction_frames'] == frame and summary['success'] == summary['precision'] == 100.
        assert all(value is None or isinstance(value, (int, float)) for value in diagnostic.values())
        if frame < 3:
            assert diagnostic['acquisition_reachable_ratio'] is None
            assert diagnostic['acquisition_realized_ratio'] is None
    assert diagnostic['acquisition_reachable_ratio'] == pytest.approx(2 / 3)
    assert diagnostic['acquisition_realized_ratio'] == pytest.approx(1 / 2)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def _rows(new, iou=.6):
    rows = []
    for track in ('a', 'b'):
        for frame in range(3):
            initial = frame == 0
            overlap, distance = (1., 0.) if initial else (iou, .3)
            s, p = metric_contributions(overlap, distance)
            row = dict(tracklet=track, frame=frame, initialization=initial,
                       iou=overlap, distance=distance, success=float(s), precision=float(p))
            if not initial:
                row.update(diagnostic_target_count=0 if frame == 1 else 2,
                           diagnostic_novel_target_count=0 if frame == 1 else 1)
                if new:
                    row.update(timestamp=float(frame), diagnostic_crop_target_count=0 if frame == 1 else 1,
                        coarse_valid=True,
                        diagnostic_sampled_target_count=0 if frame == 1 else 1,
                        diagnostic_crop_point_count=0 if frame == 1 else 3,
                        diagnostic_sampled_point_count=0 if frame == 1 else 3,
                        diagnostic_gt_xy_displacement=.1 if frame == 1 else .3,
                        diagnostic_empty_crop=frame == 1, diagnostic_background_only_crop=False)
                    for key in ('coarse_geometry', 'fine_geometry'):
                        row[key] = dict(iou=overlap, center_error_m=distance,
                            center_xy_error_m=distance, yaw_error_rad=0., axis_yaw_error_rad=0.)
            rows.append(row)
    return rows


def test_recorded_fp32_contribution_rounding_is_allowed_but_metric_step_is_not(tmp_path, monkeypatch):
    monkeypatch.setattr(compare, 'COUNTS', dict(tracklets=1, frames=2, prediction_frames=1))
    # 正式reference epoch58第30行的真实数值：CPU P=0.42500001192092896。
    rows = [dict(tracklet='rounding', frame=0, initialization=True,
                 iou=1., distance=0., success=1., precision=1.),
            dict(tracklet='rounding', frame=1, initialization=False,
                 iou=0.6396202525264773, distance=1.1808601765756999,
                 success=0.625, precision=0.42499998211860657)]
    path = tmp_path / 'frames.jsonl'
    def write():
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    write()
    accepted = compare.read_frames(path)
    assert accepted[('rounding', 1)]['precision'] == 0.42499998211860657
    assert compare.score(accepted.values())['precision'] == 100 * (1 + 0.42499998211860657) / 2
    rows[1]['precision'] += .025
    write()
    with pytest.raises(compare.EvidenceError, match='precision'):
        compare.read_frames(path)


@pytest.fixture
def runs(tmp_path, monkeypatch):
    monkeypatch.setattr(compare, 'COUNTS', dict(tracklets=2, frames=6, prediction_frames=4))
    result = {}
    for name in ('reference', 'old_b0', 'A', 'B', 'C'):
        root = tmp_path / name
        root.mkdir()
        new = name in compare.RECIPES
        schema = compare.MODEL_SCHEMA if new else compare.OLD_SCHEMA
        model = 'ctseqtrackv33' if new else ('seqtrack_reference' if name == 'reference' else 'ctseqtrackv32')
        _write(root / 'run_manifest.json', dict(schema=schema, model=model, seed=42, arm='b0',
            enabled=dict(B1=False, B2=False, B3=False), source=dict(sha256='b' * 64)))
        recipe = compare.RECIPES.get(name, compare.RECIPES['A'])
        config = dict(compare.COMMON_CONFIG, experiment_family='ct_seqtrack_v33' if new else 'ct_seqtrack_v32',
            net_model=model, seed=42, ct_engineering_check=False, test=False, init_checkpoint=None,
            v31_evaluate_late3=True, lr=recipe[0], lr_schedule=recipe[1], lr_milestones=recipe[2])
        (root / 'resolved_config.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
        sampler = dict(source_sha256='a' * 64)
        _write(root / 'training_budget.json', dict(schema=schema, **compare.BUDGET,
                                                  epoch_complete=True, sampler=sampler))
        for epoch in range(1, 61):
            _write(root / 'training_audits' / f'epoch={epoch:03d}.json',
                dict(completed_epoch=epoch, epoch_complete=True, rows=19108,
                     optimizer_steps=1195, sampler=sampler))
        for epoch in compare.EPOCHS:
            directory = root / 'evaluation' / f'epoch={epoch:03d}'
            directory.mkdir(parents=True)
            rows = _rows(new)
            (directory / 'frames.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
            metrics = dict(schema=schema, checkpoint_epoch=epoch, metric_mode='benchmark_compat',
                           complete_coverage=True, **compare.COUNTS, **compare.score(rows))
            _write(directory / 'metrics.json', metrics)
        _write(root / 'results.json', dict(final=metrics, late3=compare.score(rows),
                                          checkpoint_epochs=compare.EPOCHS))
        result[name] = root
    return result


def _compare(runs, **kwargs):
    return compare.compare_runs(*(runs[key] for key in ('reference', 'old_b0', 'A', 'B', 'C')), **kwargs)


def test_registered_comparison_uses_fixed_old_groups_and_never_writes(runs):
    def inventory():
        return {str(path): (path.stat().st_size, path.stat().st_mtime_ns)
                for root in runs.values() for path in root.rglob('*') if path.is_file()}
    before = inventory()
    report = _compare(runs)
    assert report['status'] == 'target_met'
    assert report['criterion']['maximum_drop_pp'] == 0.
    assert report['selected_recipe'] == 'A'
    assert set(report['comparisons']) >= {'A_minus_old_b0', 'B_minus_A', 'C_minus_B'}
    assert report['arms']['A']['groups']['raw_zero']['n'] == 2
    assert report['arms']['A']['groups']['old_b0_crop_positive']['n'] == 2
    assert report['arms']['A']['groups']['moving']['n'] == 2
    assert report['arms']['A']['groups']['moving_raw_positive']['n'] == 2
    assert report['arms']['A']['groups']['near_static_raw_zero']['n'] == 2
    assert 'old_b0_empty_crop' not in report['arms']['A']['groups']  # 不由新版裁剪代替旧版分组
    assert inventory() == before


def test_own_crop_and_sampling_conditions_keep_denominators_and_unknowns():
    rows = [dict(initialization=False, diagnostic_target_count=0, diagnostic_novel_target_count=0),
            dict(initialization=False, diagnostic_target_count=2, diagnostic_novel_target_count=2),
            dict(initialization=False, diagnostic_target_count=4, diagnostic_novel_target_count=2,
                 diagnostic_sampled_target_count=0),
            dict(initialization=False, diagnostic_target_count=5, diagnostic_novel_target_count=2,
                 diagnostic_sampled_target_count=1)]
    result = compare._observation_summary(rows)['own_observation_conditions']
    assert result['raw_target_absent']['rate_among_eligible'] == .25
    assert result['raw_positive_crop_missed']['eligible_frames'] == 3
    assert result['raw_positive_crop_missed']['rate_among_eligible'] == pytest.approx(1 / 3)
    sampled = result['crop_positive_sampled_target_lost']
    assert sampled['recorded_frames'] == 2 and sampled['eligible_frames'] == 2
    assert sampled['rate_among_eligible'] == .5
    unknown = compare._observation_summary([dict(initialization=False)])
    assert unknown['own_observation_conditions']['raw_target_absent']['event_frames'] is None
    assert unknown['flags']['diagnostic_empty_crop']['true_frames'] is None


def test_old_episode_seconds_use_shared_physical_time_without_rewriting_old_rows(runs):
    root = runs['old_b0']
    for epoch in compare.EPOCHS:
        directory = root / 'evaluation' / f'epoch={epoch:03d}'
        rows = _rows(False, iou=.05)
        (directory / 'frames.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
        metrics = json.loads((directory / 'metrics.json').read_text(encoding='utf-8'))
        metrics.update(compare.score(rows))
        _write(directory / 'metrics.json', metrics)
    _write(root / 'results.json', dict(final=metrics, checkpoint_epochs=compare.EPOCHS, late3=compare.score(rows)))
    report = _compare(runs)
    episodes = report['arms']['old_b0']['loss_episodes']
    assert episodes['all']['seconds_complete']
    assert episodes['unrecovered']['seconds']['median'] == 1.
    saved = [json.loads(line) for line in (root / 'evaluation/epoch=060/frames.jsonl').read_text(encoding='utf-8').splitlines()]
    assert all('timestamp' not in row for row in saved)


def test_stronger_final_cannot_replace_failing_late3(runs):
    for label in compare.RECIPES:
        root = runs[label]
        scores = []
        for epoch in compare.EPOCHS:
            directory = root / 'evaluation' / f'epoch={epoch:03d}'
            rows = _rows(True, iou=.8 if epoch == 60 else .2)
            (directory / 'frames.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
            metrics = json.loads((directory / 'metrics.json').read_text(encoding='utf-8'))
            metrics.update(compare.score(rows))
            _write(directory / 'metrics.json', metrics)
            scores.append(compare.score(rows))
        _write(root / 'results.json', dict(final=metrics, checkpoint_epochs=compare.EPOCHS,
            late3={name: sum(row[name] for row in scores) / 3 for name in ('success', 'precision')}))
    report = _compare(runs)
    assert report['status'] == 'target_not_met' and report['selected_recipe'] is None
    assert all(checks['final60.success'] and not checks['late3.success']
               for checks in report['target_checks'].values())


@pytest.mark.parametrize('problem', ['recipe', 'algorithm', 'source', 'missing_epoch', 'missing_diagnostic', 'duplicate'])
def test_invalid_evidence_cannot_be_reported_as_pass(runs, problem):
    root = runs['B']
    if problem in ('recipe', 'algorithm'):
        path = root / 'resolved_config.yaml'
        value = yaml.safe_load(path.read_text(encoding='utf-8'))
        value['lr' if problem == 'recipe' else 'v31_temporal_backend'] = .001 if problem == 'recipe' else 'gru'
        path.write_text(yaml.safe_dump(value), encoding='utf-8')
    elif problem == 'source':
        path = root / 'run_manifest.json'
        value = json.loads(path.read_text(encoding='utf-8'))
        value['source']['sha256'] = 'c' * 64
        _write(path, value)
    elif problem == 'missing_epoch':
        (root / 'training_audits/epoch=050.json').unlink()
    else:
        path = root / 'evaluation/epoch=058/frames.jsonl'
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        if problem == 'duplicate':
            rows[-1] = rows[-2]
        else:
            rows[1].pop('coarse_geometry')
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    with pytest.raises(compare.EvidenceError):
        _compare(runs)


def test_explicit_partial_probe_preserves_coverage_and_rejects_wrong_counts(runs, tmp_path):
    path = tmp_path / 'probe.jsonl'
    value = dict(event='frame', variant='original', tracklet='a', frame=1,
                 raw_target_count=0, crop_target_count=0, current_point_count=0)
    path.write_text(json.dumps(value) + '\n', encoding='utf-8')
    report = _compare(runs, old_b0_diagnostics=path)
    assert report['group_definition']['supplemental_coverage']['old_crop_empty_background'] == 1
    assert report['arms']['A']['groups']['old_b0_empty_crop']['n'] == 1
    value['raw_target_count'] = 9
    path.write_text(json.dumps(value) + '\n', encoding='utf-8')
    with pytest.raises(compare.EvidenceError, match='不一致'):
        _compare(runs, old_b0_diagnostics=path)
