from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from models.ct_v41.config import config_identity, load_config, recipe_identity
from models.ct_v41.joint_state import JointSearchState, Measurement
from tools.audit_v41_training_gt import collect_track, repair_past, sampled_coverage, summarize
from tools.compare_v41_training_arms import MODES, ROOT, assess_pair, outcome, q0_measurement, read_split, training_pair


def pose(x):
    return np.array([float(x), 0., 0., 0.])


def test_joint_teacher_is_causal_and_only_updates_existing_sources():
    joint = JointSearchState([Measurement(pose(i), float(i), i, i != 0) for i in (0, 2, 3)],
                             pose(-20), 3., np.ones((4, 3)), 0., pending_box=pose(10),
                             pending_scan=3, weak_history_boxes={1: pose(-9), 3: pose(-20)})
    base = SimpleNamespace(joint=joint, next_frame=4, boxes={i: pose(-i) for i in (1, 2, 3)},
                           times={i: float(i) for i in (1, 2, 3)}, transitions={1: True, 2: True},
                           trusted=[(pose(2), 2.), (pose(3), 3.)], trusted_velocity_valid=True,
                           trusted_velocity=np.zeros(3), last_velocity_update_time=3.,
                           previous_measurement_eligible=False)
    corrected = deepcopy(base)
    truth = {i: (float(i), pose(i*4)) for i in (1, 2, 3)}
    repair_past(corrected, truth)
    assert base.joint.motion_box[0] == -20 and base.joint.pending_scan == 3
    assert corrected.joint.pending_scan is None and corrected.joint.pending_box is None
    np.testing.assert_array_equal(corrected.joint.crop_center(corrected, 4.), pose(16))
    assert not corrected.joint.measurements[1].adjacent and corrected.joint.measurements[2].adjacent
    assert list(corrected.joint.weak_history_boxes) == [1, 3]
    assert [m.time for m in corrected.joint.measurements] == [0., 2., 3.]
    np.testing.assert_array_equal(corrected.joint.template_xyz, joint.template_xyz)
    bad = deepcopy(base)
    bad.trusted = [(pose(0), 0.), (pose(2), 2.)]
    bad.last_velocity_update_time = 2.
    repair_past(bad, truth)
    assert not bad.trusted_velocity_valid and bad.last_velocity_update_time is None
    with pytest.raises(ValueError, match='last committed'):
        repair_past(deepcopy(base), {**truth, 4:(4., pose(16))})


def test_count_unique_raw_ids_across_overlapping_local_and_remote_pools():
    xyz = np.array([[0., 0., 0.], [.5, 0., 0.], [1., 0., 0.], [2., 0., 0.]])
    frame = {'3d_bbox':pose(0), 'box_size': np.array([4., 2., 2.])}
    builder = SimpleNamespace(_points=lambda frame:(xyz, np.array([4, 5, 6, 7])),
                              config=SimpleNamespace(bb_scale=1., bb_offset=0.))
    ctx = dict(row=dict(frames={1: frame}, request=SimpleNamespace(frame=1)), sampled_ids=[4, 5],
               state=SimpleNamespace(size=np.array([2., 1., 1.])),
               search_plan=SimpleNamespace(main_box=pose(-.5), backup_box=None))
    output = SimpleNamespace(prior=SimpleNamespace(box=torch.zeros(1, 4)), joint=SimpleNamespace(
        observations=SimpleNamespace(coarse=[SimpleNamespace(proposals_sensor=[[1.,0.,0.]])],
                                     remote=SimpleNamespace(ids=torch.tensor([[5, 6, -1]]),
                                                            point_valid=torch.tensor([[True,True,False]])))))
    batch = dict(joint_rows=[dict(pending_valid=False)], box_size=torch.tensor([[2.,1.,1.]]))
    assert sampled_coverage(builder, batch, output, ctx) == dict(raw=4, roi=4, sampled=3)


def test_followup_distinguishes_incomplete_from_failure_and_preserves_frame_pairing():
    def frame(i, raw, iou, distance=0.):
        return dict(raw=raw, iou=iou, timestamp=i/10, raw_frame_id=i, success=iou,
                    precision=1., distance=distance)
    rows = [frame(0, None, 1.), frame(1, 0, 0.), frame(2, 2, .7),
            frame(3, 2, .5), frame(4, 1, .51)]
    b = deepcopy(rows)
    b[3].update(distance=12., iou=.3, success=.3)
    akey = ('0018', 12)
    result = assess_pair({akey:rows}, {akey:b})
    assert result['return_counts']['window_1448']['stable_fraction_assessable'] == 1.
    assert result['return_counts']['four_candidates']['censored'] == 1
    assert result['return_counts']['four_candidates']['stable_fraction_assessable'] is None
    assert result['return_counts']['four_candidates']['stable_fraction_all'] == 0.
    assert result['return_counts']['window_1448']['mean_confirmation_seconds'] == pytest.approx(.2)
    assert result['total']['four_candidates']['severe_seconds'] == pytest.approx(.1)
    assert result['window_minus_four_pp']['success'] == pytest.approx(4.)
    assert outcome([frame(i, 1, .49) for i in range(11)], 0) == 'observed_no'
    b[3]['raw'] = 20
    with pytest.raises(ValueError, match='physical frames'):
        assess_pair({akey:rows}, {akey:b})
    short = [{'scan':1,'sampled':3,'iou':.9}, {'scan':2,'sampled':0,'iou':.1}]
    summary = summarize([dict(reason='crop', natural=short, clone=short, corrected=short)])
    assert summary['outcomes']['natural']['censored_before_stable'] == 1


def test_frozen_joint_forward_calibrates_clone_on_a_real_data_path():
    from models.ct_v41.data import RawEndpointDataset
    from models.ctseqtrackv41 import CTSEQTRACKV41
    from tests.helpers.raw_tracking import TinySource

    class MovingSource(TinySource):
        def get_frames(self, index, frame_ids):
            rows = super().get_frames(index, frame_ids)
            for i, row in zip(frame_ids, rows):
                row['raw_frame_id'] = i
                if i >= 2:
                    row['pc'][:, 0] += 30.
                    row['3d_bbox'][0] += 30.
            return rows

    dataset = RawEndpointDataset(MovingSource((6,), dt=.1), joint_search=True)
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(42)
        host = CTSEQTRACKV41(dict(v41_joint_search=True, workers=0, batch_size=1,
                             ct_engineering_check=True, v41_attention_backend='math')).eval()
        events = collect_track(host, dataset, host.config, 0, 2)
    first = next(r for r in events if r['frame'] == 2)
    assert first['reason'] == 'crop' and first['natural'] == first['clone']
    assert len(first['corrected']) == 3 and first['corrected'][0]['sampled'] >= 3
    assert first['corrected'][0]['anchor'] != first['natural'][0]['anchor']


@pytest.mark.parametrize('lateral', [False, True], ids=('history_pair', 'joint_pair'))
def test_warmup_audit_requires_two_matching_training_runs(tmp_path, lateral):
    from models.ct_v41.auxiliary import insertion_identity, insertion_indices
    from models.ct_v41.history_augmentation import TRAIN_DATA
    from models.ct_v41.joint_replay import CATEGORIES

    digest = 'b'*64
    insertions = list(insertion_indices(4771, 2))
    roots = [tmp_path / mode for mode in MODES]
    for root, mode in zip(roots, MODES):
        (root/'training_audits').mkdir(parents=True)
        family = 'joint' if lateral else 'history'
        config = load_config(ROOT/'cfgs/ct_seqtrack'/f'41_{family}_{mode}.yaml')
        config_hash = config_identity(config)
        (root/'resolved_config.yaml').write_text(yaml.safe_dump(dict(config)))
        recipe = dict(v41_history_training=mode, v41_joint_search=True, dataset='kitti_mf', seed=42,
                      epoch=60, batch_size=16)
        manifest = dict(actual_recipe=recipe, run_type='formal', run_id=mode, source=digest,
                        history_bank_sha256=digest, initial_parameter_hash=digest, config_sha256=config_hash,
                        dataset_manifest={'train': '0000-0016'}, input_metadata={'content_sha256': digest})
        if lateral:
            recipe.update(v41_q0_lateral=True, v41_q0_lateral_warmup_epochs=config.v41_q0_lateral_warmup_epochs,
                          v41_joint_replay=config.v41_joint_replay)
            manifest['training_execution_recipe'] = recipe_identity(config)
        (root/'run_manifest.json').write_text(json.dumps(manifest))
        for e in range(1, 61):
            audit = dict(completed_epoch=e, epoch_complete=True, rows=76324, optimizer_steps=4771,
                         run_id=mode, config_sha256=config_hash, history_training=dict(mode=mode, bank_sha256=digest,
                         batch_sha256=digest, loss_sha256=digest, weights_sha256=digest),
                         auxiliary=dict(complete=True, main_batches=4771, insertion_plan_sha256=insertion_identity(insertions),
                         insertion_indices=insertions, rows=5, batches=2, loss_weight=config.v41_aux_loss_weight,
                         sampler=dict(rows=5, batches=2, budget=4770, source_sha256=TRAIN_DATA, plan_sha256=digest)),
                         sampler=dict(plan_sha256=digest, rows=76324, batches=4771, source_sha256=TRAIN_DATA),
                         joint_replay=dict(policy='ct_seqtrack.fixed_joint_e2.v1', branch0_slots=19081,
                         catalog={k: (12 if e >= 3 and k == 'crop_miss' else 0) for k in CATEGORIES},
                         used={} if e < 3 else {'crop_miss':12}, plan_sha256=digest,
                         manifest_sha256=None if e < 3 else digest, donor_weights_sha256=None if e < 3 else digest))
            if lateral:
                audit['q0_lateral_epoch'] = e
            (root/'training_audits'/f'epoch={e:03}.json').write_text(json.dumps(audit))
        (root/'training_budget.json').write_text(json.dumps(dict(completed_epoch=60, epoch_complete=True,
            last_epoch_rows=audit['rows'], last_epoch_steps=audit['optimizer_steps'], optimizer_steps=60*4771,
            sampler=audit['sampler'], auxiliary=audit['auxiliary'])))
    validated = training_pair(roots, require_q0_lateral=lateral)
    assert validated['main_updates_per_arm'] == 286260
    assert validated['shared_recipe']['v41_q0_lateral'] is lateral
    if not lateral:
        with pytest.raises(ValueError, match='requires the independent q0 lateral head'):
            training_pair(roots, require_q0_lateral=True)
    path = roots[1]/'training_audits/epoch=002.json'
    changed = json.loads(path.read_text())
    changed['history_training']['loss_sha256'] = 'c'*64
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='warmup mismatch'):
        training_pair(roots)
    changed['history_training']['loss_sha256'] = digest
    path.write_text(json.dumps(changed))
    for root in roots:
        path = root/'training_audits/epoch=003.json'
        changed = json.loads(path.read_text())
        changed['joint_replay']['donor_weights_sha256'] = 'd'*64
        path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='differs from common E2'):
        training_pair(roots)
    # 值未列入 actual_recipe，且篡改后的单臂文件 SHA 与清单内部仍一致。
    path = roots[1]/'resolved_config.yaml'
    config = yaml.safe_load(path.read_text())
    config.update(ct_pointcloud_cache_bytes=33554432)
    path.write_text(yaml.safe_dump(config))
    path = roots[1]/'run_manifest.json'
    m = json.loads(path.read_text())
    m.update(config_sha256=config_identity(config))
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match='full resolved training configuration'):
        training_pair(roots)


def test_paired_q0_diagnostics_measure_the_applied_lateral_component():
    row = dict(initialization=False, q0_is_fallback=False, diagnostic_sampled_target_count=10,
               diagnostic_q0_lateral_unique_points=3, diagnostic_q0_lateral_applied_m=-0.15,
               diagnostic_original_q0_box_world=[0., 0.25, 0., 0.],
               diagnostic_fine_box_world=[0., 0.10, 0., 0.], diagnostic_target_box_world=[0.] * 4)
    measured = q0_measurement(row, feedback=True)
    assert measured['eligible']
    assert (measured['original_width_mae_cm'], measured['corrected_width_mae_cm']) == pytest.approx((25, 10))
    with pytest.raises(ValueError, match='moved during warmup'):
        q0_measurement(row, feedback=False)


def test_standalone_heldout_ledgers_check_their_training_origin(tmp_path):
    base = dict(run_id='train-A', source={'sha256':'a'}, actual_recipe={'mode':'A'},
                config_sha256='a'*64, history_bank_sha256='b'*64,
                input_metadata={}, dataset_manifest={})
    test_root = tmp_path/'heldout_test/epoch=060'
    part = 'frame_streams/test/session/epoch=060.jsonl'
    ledger = test_root/part
    ledger.parent.mkdir(parents=True)
    evaluated = dict(base, source_run_id='train-A', checkpoint_completed_epoch=60, run_type='formal')
    (test_root/'run_manifest.json').write_text(json.dumps(evaluated))
    (test_root/'results.json').write_text(json.dumps(dict(epoch=60, checkpoint_epoch=60, split='test',
        complete_coverage=True, diagnostics_level='full', frame_file='/other_root/'+part,
        frames=1, success=100., precision=100.)))
    ledger.write_text(json.dumps(dict(scene_id='0019', track_id=1, frame=0, raw_frame_id=1, timestamp=.1,
        iou=1., distance=0., initialization=True, success=1., precision=1.)))
    assert len(read_split(tmp_path, 'test', 60, run=base)) == 1
    with pytest.raises(ValueError, match='another checkpoint'):
        read_split(tmp_path, 'test', 60, run=dict(base, run_id='train-B'))

