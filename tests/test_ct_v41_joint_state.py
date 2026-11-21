"""联合搜索状态的物理时间、冷启动与两帧接管合同。"""
from types import SimpleNamespace

import numpy as np
import torch

from models.ct_v41.joint_state import JointSearchState, physical_frame
from models.ct_v41.prior import PhysicalTimePrior


def box(x):
    return np.array([x, 2., 3., 0.])


def points(x):
    return box(x)[:3] + np.array([[.1, 0, 0], [.2, .1, 0], [-.2, 0, 0]])


def initial(xyz):
    state = SimpleNamespace(boxes={0: box(0)}, times={0: .0}, size=np.array([4., 2., 1.5]),
        trusted=[(box(0), 0.)], previous_measurement_eligible=None, consecutive_untrusted_outputs=0)
    first = {'timestamp': .0, 'raw_frame_id': 20}
    return state, JointSearchState.from_seed(state, {0: first}, first, xyz, box(0), {0: len(xyz) >= 3})


def advance(state, joint, i, *, own, forecast, source=0, eligible=True, scanned=False, proposal=None):
    updated = joint.advance(state, frame=i, now=.1*i, scan=20+i, box=box(own), prior=box(forecast),
        source=source, supported=eligible, support_xyz=points(own), searched=scanned, proposal=proposal)
    state.previous_measurement_eligible = eligible
    state.consecutive_untrusted_outputs = 0 if eligible else state.consecutive_untrusted_outputs + 1
    return updated


def test_sparse_first_box_waits_for_two_adjacent_local_measurements_and_freezes_template():
    state, joint = initial(points(0)[:2])
    assert not len(joint.template_xyz)
    assert not advance(state, joint, 1, own=.5, forecast=.5)
    assert not len(joint.template_xyz) and not state.trusted_velocity_valid
    assert advance(state, joint, 2, own=1., forecast=1.)
    np.testing.assert_allclose(joint.template_xyz, points(1.) - box(1.)[:3])
    first = joint.template_xyz
    advance(state, joint, 3, own=1.8, forecast=1.5)
    assert first is joint.template_xyz


def test_wrong_returned_boxes_do_not_advance_crop_and_prior_uses_exactly_one_physical_step():
    state, joint = initial(points(0))
    assert advance(state, joint, 1, own=.5, forecast=0.)
    prior = PhysicalTimePrior().eval()
    for i in range(2, 8):
        now = .1*i
        anchor = joint.crop_center(state, now)
        if i > 2:
            np.testing.assert_allclose(anchor, box(.5*i), atol=1e-6)
        fields = {k: torch.tensor(v)[None] for k, v in joint.motion_fields(anchor, now).items()}
        fields.update(current_dt=torch.tensor([.1]), trusted_velocity=torch.tensor([[5., 0., 0.]]),
                      fallback_box=torch.zeros(1, 4))
        with torch.no_grad():
            predicted = prior(fields)
        torch.testing.assert_close(predicted.mean_xy, torch.tensor([[.5, 0.]]))
        sensor_x = float(predicted.box[0, 0]) + anchor[0]
        assert abs(sensor_x - .5*i) < 1e-5
        advance(state, joint, i, own=-100., forecast=sensor_x, eligible=False)
        assert joint.measurements[-1].scan == 21 and state.trusted[-1][1] == .1


def test_remote_proposal_lives_one_scan_and_confirmed_jump_does_not_create_speed():
    state, joint = initial(points(0))
    assert not joint.search_due(state, .1, 21)
    advance(state, joint, 1, own=.1, forecast=.1, scanned=True, proposal=box(30))
    assert joint.pending_at(22) and not joint.pending_at(23)
    assert joint.search_due(state, .2, 22)
    # 本次复核失败，即使又报出同一候选也不在本帧续约。
    advance(state, joint, 2, own=.2, forecast=.2, scanned=True, proposal=box(30))
    assert not joint.pending_at(23)
    advance(state, joint, 3, own=.3, forecast=.3, scanned=True, proposal=box(30))
    assert not advance(state, joint, 4, own=30., forecast=.4, source=5, scanned=True)
    assert joint.history_start == 4 and len(joint.measurements) == 1 and not state.trusted_velocity_valid
    assert advance(state, joint, 5, own=30.5, forecast=30.)
    np.testing.assert_allclose(state.trusted_velocity, [5., 0., 0.])


def test_sparse_search_budget_uses_elapsed_time_and_numeric_raw_frame():
    state, joint = initial(points(0))
    assert physical_frame({'raw_frame_id': None, 'frame_id': '000023'}, 0) == 23
    assert physical_frame({'frame_id': 'sample-token'}, 14) == 14
    assert not joint.search_due(state, .99, 30) and joint.search_due(state, 1., 30)
    state.consecutive_untrusted_outputs = 2
    assert not joint.search_due(state, .19, 22) and joint.search_due(state, .2, 22)


def test_builder_routes_configured_patrol_intervals_and_prioritizes_pending_scan():
    from models.ct_v41.data import BatchBuilder, EndpointRequest, RawEndpointDataset
    from tests.helpers.raw_tracking import TinySource

    builder = BatchBuilder(dict(v41_joint_search=True, ct_engineering_check=True,
        v41_attention_backend='math', v41_patrol_seconds=.25, v41_weak_patrol_seconds=.12))
    builder.detailed_diagnostics = False
    dataset = RawEndpointDataset(TinySource(dt=.1), training=True, enable_m=True, joint_search=True)

    def window(frame):
        row = dataset[EndpointRequest(0, 0, 0, frame, frame+1, frame)]
        return row, builder._new_state(row)

    row, state = window(4)
    assert state.joint.last_scan_time == 100.25  # 首帧 100.0，中途 seed 在 100.3
    assert not builder._prepare_row(row, state)[1]['joint_scheduled']
    state.consecutive_untrusted_outputs = 2
    assert builder._prepare_row(row, state)[1]['joint_scheduled']  # 0.15 秒已过弱巡查间隔
    state.consecutive_untrusted_outputs = 0
    state.joint.pending_scan, state.joint.pending_box = 3, state.joint.motion_box.copy()
    assert builder._prepare_row(row, state)[1]['joint_scheduled']  # 下一物理扫描立即复查
    state.joint.pending_scan = 2
    assert not builder._prepare_row(row, state)[1]['joint_scheduled']
    row, state = window(5)
    assert builder._prepare_row(row, state)[1]['joint_scheduled']  # 恰好到下一健康时点


def test_builder_recrops_weak_history_on_motion_and_retains_reported_accepted_box():
    from models.ct_v41.data import BatchBuilder, EndpointRequest, PREDICTED_BOX, RawEndpointDataset
    from tests.helpers.raw_tracking import TinySource

    builder = BatchBuilder(dict(v41_arm='full', v41_joint_search=True, dataset='kitti_mf',
                                ct_engineering_check=True, v41_attention_backend='math'))
    builder.detailed_diagnostics = False
    raw = RawEndpointDataset(TinySource(dt=.1), joint_search=True)
    first, second = [raw[EndpointRequest(0, 0, 4, 1, 4, i)] for i in (1, 2)]
    assert 'joint_other_cars' not in first and raw._scan_annotations is None
    state = builder._new_state(first)
    pose = state.boxes[0].copy()
    wrong = pose.copy()
    wrong[0] += 50.
    now = first['frames'][1]['timestamp']
    state.joint.advance(state, frame=1, scan=1, now=now, box=wrong,
        prior=pose, source=0, supported=False, support_xyz=np.empty((0, 3)), searched=False, proposal=None)
    state.boxes[1], state.times[1], state.box_provenance[1] = wrong, now, PREDICTED_BOX
    state.previous_measurement_eligible = False
    builder._start_materialization_cache()
    data, _ = builder._prepare_row(second, state)
    assert state.boxes[1][0] == wrong[0]
    np.testing.assert_allclose(data['anchor_box'], pose, atol=1e-6)
    np.testing.assert_allclose(data['history_boxes'][2, :3], np.zeros(3), atol=1e-6)
    valid = data['point_valid'][2]
    assert valid.any() and (data['points'][2, valid, 4] == .5).all()


def test_duplicate_physical_ids_in_first_crop_do_not_open_template_or_velocity(monkeypatch):
    from models.ct_v41.data import BatchBuilder, EndpointRequest, RawEndpointDataset
    from tests.helpers.raw_tracking import TinySource

    builder = BatchBuilder(dict(v41_arm='full', v41_joint_search=True, dataset='kitti_mf',
                                ct_engineering_check=True, v41_attention_backend='math'))
    raw = RawEndpointDataset(TinySource(dt=.1), joint_search=True)
    row = raw[EndpointRequest(0, 0, 4, 1, 4, 1)]
    pts = np.zeros((4, 3))
    monkeypatch.setattr(builder, '_view', lambda *args, **kwargs: (pts, np.array([1, 1, 2, 2]), False))
    state = builder._new_state(row)
    assert not len(state.joint.template_xyz) and state.last_strong_time is None
    assert not state.joint.measurements[-1].can_start_pair and not state.trusted_velocity_valid


def test_empty_teacher_seed_does_not_supply_motion_or_break_prior_measured_velocity():
    state = SimpleNamespace(boxes={2: box(0.), 3: box(.5), 4: box(40.)},
                            times={2: .2, 3: .3, 4: .4}, previous_measurement_eligible=None)
    frames = {i: dict(raw_frame_id=i) for i in state.boxes}
    joint = JointSearchState.from_seed(state, frames, {'timestamp': .0}, points(0), box(0),
                                      {2: True, 3: True, 4: False})
    assert [s.scan for s in joint.measurements] == [2, 3]
    assert state.trusted[-1][1] == .3 and state.previous_measurement_eligible is False
    np.testing.assert_allclose(joint.motion_box, box(1.))
    np.testing.assert_allclose(joint.crop_center(state, .5), box(1.5))
