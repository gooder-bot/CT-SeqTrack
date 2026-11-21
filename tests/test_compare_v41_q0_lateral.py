import gzip
import hashlib
import json
import math

import pytest

from tools.compare_v41_q0_lateral import adoption, comparison, geometry, lateral_geometry, paired_epoch, tail_audit


def row(i, *, coarse_y=0., q_y=0., visible=True, yaw=0.):
    record = dict(initialization=False, q0_is_fallback=not visible, coarse_valid=visible,
        diagnostic_target_box_world=[0., 0., 0., 0.], diagnostic_fine_box_world=[0., q_y, 0., yaw],
        diagnostic_coarse_box_world=[0., coarse_y, 0., 0.], diagnostic_target_size_lwh=[4., 2., 2.],
        diagnostic_first_frame_size_lwh=[4., 2., 2.], diagnostic_sampled_target_count=10,
        coarse_geometry={'iou': .8})
    return dict(geometry(record), frame=i, scan=i, initialization=False, timestamp=.1*i, distance=q_y,
                iou=.8, success=.8, precision=1., raw=10)


def test_hard_cohort_is_anchored_to_uniform_with_paired_visibility():
    old, new = {}, {}
    for data in (old, new):
        data['0020', 1, 0] = dict(frame=0, scan=0, timestamp=0., initialization=True, distance=0.,
                                  iou=1., success=1., precision=1.)
    for i, (old_coarse, new_coarse, eligible) in enumerate(((.4, 0., True), (0., .4, True),
                                                             (.4, .4, False)), 1):
        old['0020', 1, i] = row(i, coarse_y=old_coarse, q_y=.1*i, yaw=math.radians(179))
        new['0020', 1, i] = row(i, coarse_y=new_coarse, q_y=.05*i, visible=eligible)
    scores, _ = paired_epoch(old, new)
    assert scores['uniform']['common_visible_frames'] == 2
    assert scores['uniform']['common_visible_hard_frames'] == 1
    assert scores['uniform']['difficult_lateral_mae_cm'] == pytest.approx(10.)
    assert scores['lateral']['difficult_lateral_mae_cm'] == pytest.approx(5.)
    assert scores['uniform']['q0_axis_yaw_p95_deg'] == pytest.approx(1.)
    assert adoption(scores, {'concentrated_gain_and_other_track_loss':False})['meets_adoption_criteria']
    new['0020', 1, 2]['target'][0] = 1.
    with pytest.raises(ValueError, match='different target'):
        paired_epoch(old, new)


def test_concentrated_gain_requires_new_sustained_loss_on_other_track_and_adjacent_scans():
    records = {'0020/1':dict(frames=50, success_gain_sum=10., new_sustained_loss=False, new_severe_drift=False),
               '0020/2':dict(frames=50, success_gain_sum=-2., new_sustained_loss=True, new_severe_drift=True)}
    report = tail_audit([records]*3)
    assert report['concentrated_gain_and_other_track_loss']
    assert report['new_over10m_tracks'] == ['0020/2']
    assert report['success_delta_without_top_pp'] == -4.
    old, new = {}, {}
    for i, scan in enumerate([0, 1, 2, 4]):
        original = dict(row(i), scan=scan, initialization=i==0)
        old['0020',1,scan] = original
        new['0020',1,scan] = dict(original, iou=0. if i else 1.)
    assert not paired_epoch(old,new)[1]['0020/1']['new_sustained_loss']
    old['0020',1,3] = dict(old.pop(('0020',1,4)),scan=3)
    new['0020',1,3] = dict(new.pop(('0020',1,4)),scan=3)
    assert paired_epoch(old,new)[1]['0020/1']['new_sustained_loss']
    with pytest.raises(ValueError, match='lateral diagnostics'):
        lateral_geometry({'initialization':False})


def test_comparison_uses_historical_budget_and_requires_shared_replay_identity(tmp_path):
    import yaml
    from tests.test_compare_v41_q0 import make_run, write_json
    from tools.compare_v41_q0 import EPOCHS, read_json

    root = make_run(tmp_path / 'q0', 'uniform')
    manifest_path = root / 'run_manifest.json'
    manifest = read_json(manifest_path)
    manifest['evaluation_protocol']['split'] = 'test'
    manifest['actual_recipe']['v41_q0_lateral_replay'] = True
    manifest['training_execution_recipe']['implementation_id'] = 'v41_q0_lateral_replay_v1'
    manifest['q0_replay_manifest_sha256'] = 'q0-fixture'
    write_json(manifest_path, manifest)
    cfg_path = root / 'resolved_config.yaml'
    cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
    cfg.update(eval_split='test', v41_q0_lateral_replay=True)
    cfg_path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
    fixed = tmp_path / 'formal'
    donors = [dict(epoch=e, checkpoint_sha256=str(e), source_sha256='frozen',run_id='A') for e in (10,60)]
    for path in (root/'training_audits').glob('*.json'):
        audit = read_json(path)
        audit['q0_replay'] = dict(manifest_sha256='q0-fixture', donors=donors, branch0_slots=3,
                                  available_events=1, planned_pairs=0, used=dict(hard=0,normal=0))
        write_json(path, audit)
    write_json(fixed / 'dev_fixed_comparison.json', dict(source_sha256='frozen', execution_revision='v3',
               arms={'A':{'run_id':'A'}}, budget=dict(main_endpoints=720,auxiliary_endpoints=120,optimizer_steps=180)))
    init = dict(scene_id='0019',track_id=1,raw_frame_id=0,frame=0,initialization=True,timestamp=0.,
                success=1.,precision=1.,distance=0.,iou=1.)
    current = dict(init, raw_frame_id=1,frame=1,initialization=False,timestamp=.1, q0_is_fallback=False,
        coarse_valid=True, diagnostic_target_box_world=[0.,0.,0.,0.], diagnostic_fine_box_world=[0.,.2,0.,0.],
        diagnostic_coarse_box_world=[0.,.4,0.,0.], diagnostic_sampled_target_count=10,
        diagnostic_target_size_lwh=[4.,2.,2.], diagnostic_first_frame_size_lwh=[4.,2.,2.],
        diagnostic_search_raw_target_count=12,diagnostic_search_union_target_count=12,trusted_write=False,
        coarse_geometry=dict(iou=.75), diagnostic_candidates=dict(selected_index=0,selected_trusted_target=False,
        selected_geometry=dict(exact_iou=.9)))
    receipts = []
    result = dict(frames=2,success=100.,precision=100.,diagnostics_level='full',split='test',complete_coverage=True)
    for e in EPOCHS:
        name = f'uniform_E{e}.jsonl.gz'
        blob = ('\n'.join(map(json.dumps,[init,current])) + '\n').encode()
        (fixed/name).write_bytes(gzip.compress(blob))
        receipts.append(dict(result,arm='uniform',epoch=e,archive=name,source_sha256='frozen',
                             checkpoint_sha256=str(e),frame_file_sha256=hashlib.sha256(blob).hexdigest()))
        test = root/'frame_streams'/'test'/'session'/f'epoch={e:03d}.jsonl'
        test.parent.mkdir(exist_ok=True, parents=True)
        improved = dict(current,diagnostic_original_q0_box_world=current['diagnostic_fine_box_world'],
                        diagnostic_fine_box_world=[0.,.1,0.,0.],diagnostic_q0_lateral_applied_m=-.1)
        test.write_text('\n'.join(map(json.dumps,[init,improved])),encoding='utf-8')
        write_json(root/'epochs'/f'epoch={e:03d}.json',
                   dict(epoch=e,evaluation={'test':dict(result,epoch=e,checkpoint_epoch=e,frame_file=str(test))}))
    receipt_path = fixed/'test_verification.json'
    write_json(receipt_path,dict(receipts=receipts))
    report = comparison(root,receipt_path)
    assert report['history']['inference_recovery_enabled'] is True
    assert report['lateral_run']['inference_recovery_enabled'] is False
    assert report['late3']['meets_adoption_criteria']
    assert report['late3']['metrics']['lateral']['difficult_lateral_mae_cm'] == 10.
    assert report['late3']['tracks']['0019/1']['uniform']['success'] == 100.
    changed_path = root/'training_audits'/'epoch=040.json'
    changed = read_json(changed_path)
    changed['q0_replay']['manifest_sha256'] = 'other'
    write_json(changed_path, changed)
    with pytest.raises(ValueError, match='replay identity'):
        comparison(root,receipt_path)
