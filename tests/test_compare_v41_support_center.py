import json
from copy import deepcopy
import pytest

from tools import compare_v41_support_center as comparator
from tools.compare_v41_support_center import BASELINE_RUN_ID, paired_counts, summarize, training_evidence, training_run


def prediction(scan, success=1., distance=0.):
    return dict(raw_frame_id=scan, timestamp=scan * .1, raw=5 if scan else None,
                distance=distance, success=success, precision=success)


def test_pairing_checks_physical_scans_and_total_kitti_coverage():
    tracks = {('0019', i): [prediction(0)] for i in range(118)}
    tracks['0020', 122] = [prediction(0)]
    tracks['0020', 123] = [prediction(i) for i in range(6305)]
    trial = dict(tracks)
    trial['0020', 123] = list(tracks['0020', 123])
    trial['0020', 123][1] = prediction(1, 0., 10.)
    trial['0020', 123][2] = prediction(2, .5, 10.01)
    _, counts = paired_counts(tracks, trial)
    assert counts['0020', 123]['trial_over10m'] == 1
    assert counts['0020', 123]['trial_success'] == 6303.5
    for period in ({60: counts}, {58: counts, 59: counts, 60: counts}):
        summary = summarize(period)
        tracks_out = [r['epoch_mean'] for r in summary['per_track'].values()]
        all_rows = summary['scopes']['all']['epoch_mean']
        assert len(tracks_out) == all_rows['tracks'] == 120
        for field in ('frames', 'predictions', 'baseline_over10m', 'trial_over10m', 'delta_over10m'):
            assert sum(r[field] for r in tracks_out) == all_rows[field]
        for metric in ('success', 'precision'):
            for side in ('baseline', 'trial', 'delta'):
                field = f'{side}_{metric}'
                value = sum(r['frames'] * r[field] for r in tracks_out) / all_rows['frames']
                assert value == pytest.approx(all_rows[field])
    trial['0020', 123][1] = prediction(3)
    with pytest.raises(ValueError, match='physical'):
        paired_counts(tracks, trial)
    with pytest.raises(ValueError, match='coverage'):
        paired_counts(dict(list(tracks.items())[1:]), dict(list(tracks.items())[1:]))


def counts(frames, gain):
    return dict(frames=frames, baseline_success=1., trial_success=1.+gain,
                baseline_precision=1., trial_precision=1.,
                baseline_over10m=0, trial_over10m=0)


def test_largest_success_contribution_per_period_and_fixed_sensitivities():
    short, long, extra = ('0019', 1), ('0020', 122), ('0020', 123)
    early = {short: counts(4, 3), long: counts(10, 2), extra: counts(2, 0)}
    late = {short: counts(4, 0), long: counts(10, 1), extra: counts(2, 0)}
    last = summarize({60: late})
    late3 = summarize({58: early, 59: early, 60: late})
    assert last['max_track'] == '0020/122'
    assert late3['max_track'] == '0019/1'
    assert late3['scopes']['without_max']['epoch_mean']['delta_success'] == pytest.approx(100 * 5 / 36)
    assert late3['scopes']['all']['epochs'][58]['delta_success'] == pytest.approx(100 * 5 / 16)
    assert last['scopes']['without_0020_122'] == last['scopes']['without_max']
    assert last['scopes']['without_0020_123']['epoch_mean']['frames'] == 14
    assert last['scopes']['scene_0019']['epoch_mean']['tracks'] == 1
    assert summarize({60: {key: counts(2, 0) for key in late}})['max_track'] == '0019/1'
    assert late3['per_track']['0019/1']['epochs'][60]['delta_success'] == 0.
    assert late3['per_track']['0019/1']['epoch_mean']['delta_success'] == pytest.approx(50.)


def test_training_evidence_requires_equal_content_and_achieved_budget_across_mounts(monkeypatch):
    manifest = dict(input_metadata=dict(files={'label_02/0000.txt': 'a'}, content_sha256='aggregate',
                                        root='/data-a/training'), dataset_manifest='split',
                    history_bank_sha256='history', q0_replay_manifest_sha256='replay', local_prefix_sha256='prefix',
                    actual_recipe=dict(lr=.0001))
    trial = deepcopy(manifest)
    trial['input_metadata']['root'] = '/mounted/kitti/training'
    trial['actual_recipe']['v41_support_center'] = True
    plan = [('main', 'aux', 'insertion', 4, 2, 2, 1)] * 60
    totals = dict(main_endpoints=240, optimizer_steps=120, auxiliary_endpoints=120)
    runs = {'original': (plan, totals), 'updated': (list(plan), dict(totals))}
    monkeypatch.setattr(comparator, 'budget', lambda root, run: runs[root])
    def compare_training():
        return training_evidence('original', manifest, 'updated', trial)
    checked = compare_training()
    assert checked['matched_training_epochs'] == 60
    assert checked['baseline'] == checked['trial'] == dict(totals, auxiliary_batches=60)
    trial['input_metadata']['files']['label_02/0000.txt'] = 'different'
    with pytest.raises(ValueError, match='input_metadata'):
        compare_training()
    trial['input_metadata']['files'] = manifest['input_metadata']['files']
    runs['updated'][1]['auxiliary_endpoints'] -= 1
    with pytest.raises(ValueError, match='exposures'):
        compare_training()
    runs['updated'][1]['auxiliary_endpoints'] += 1
    runs['updated'][0][59] = ('different-main-plan', *plan[59][1:])
    with pytest.raises(ValueError, match='schedules'):
        compare_training()


def test_comparator_rejects_wrong_baseline_identity_and_unfinished_run(tmp_path):
    manifest = dict(run_id=BASELINE_RUN_ID, run_type='formal', model='ctseqtrackv41',
        actual_recipe=dict(v41_arm='full', dataset='kitti_mf', category_name='Car', v41_local_vote=True, epoch=60),
        training_execution_recipe=dict(implementation_id='v41_uniform_local_vote_v1'))
    def write(name, data):
        (tmp_path / name).write_text(json.dumps(data), encoding='utf-8')
    write('run_manifest.json', manifest)
    write('training_budget.json', dict(completed_epoch=60, epoch_complete=True, auxiliary=dict(complete=True)))
    assert training_run(tmp_path, upgraded=False) == manifest
    manifest['run_id'] = 'different-seed42'
    write('run_manifest.json', manifest)
    with pytest.raises(ValueError, match='11/27'):
        training_run(tmp_path, upgraded=False)
    manifest['actual_recipe']['v41_support_center'] = True
    manifest['training_execution_recipe']['implementation_id'] = 'v41_uniform_local_vote_support_center_v1'
    write('run_manifest.json', manifest)
    write('training_budget.json', dict(completed_epoch=60, epoch_complete=True, auxiliary=dict(complete=False)))
    with pytest.raises(ValueError, match='incomplete'):
        training_run(tmp_path, upgraded=True)
