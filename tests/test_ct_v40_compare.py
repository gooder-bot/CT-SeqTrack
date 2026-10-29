"""V40 逐帧搜索与候选几何验收的拒绝测试；仅编写，待服务器执行。"""
from copy import deepcopy

import numpy as np
import pytest

from models.ct_v40.candidate_records import _geometry_both
from models.ct_v40.config import normalize_config
from models.ct_v40.identity import CANDIDATE_SCHEMA
from tools.compare_ct_v40 import validate_candidates, validate_search_rows, validate_search_summary, SOURCES
from tools._tracking_audit import EvidenceError


def row_fixture(arm='full', *, frame=1, anchor_x=100., q0_x=0., score=.9,
                support_count=3, target_count=3):
    anchor = np.asarray([anchor_x, 200., 0., 0.], np.float64)
    target_world = np.asarray([100., 200., 0., 0.], np.float64)
    target = target_world.copy()
    target[:3] -= anchor[:3]
    size = np.asarray([4., 2., 2.], np.float32)
    boxes = np.asarray([[q0_x, 0., 0., 0.], [q0_x + 2., 0., 0., 0.],
                        [q0_x - 2., 0., 0., 0.]], np.float32)
    geometry = [_geometry_both(box, target.astype(np.float32), size, size) for box in boxes]
    valid = [True, arm != 'b0', True]
    utility = [(g['success'] + g['precision']) / 2 for g in geometry]
    oracle = max((k for k in range(3) if valid[k]), key=lambda k: utility[k])
    purity = target_count / support_count if support_count else 0.
    trusted_target = geometry[0]['exact_iou'] >= .5 and support_count >= 3 and purity >= .5
    eligible = support_count >= 3 and score >= .5
    world = boxes[0].astype(np.float64)
    world[:3] += anchor[:3]
    record = dict(schema=CANDIDATE_SCHEMA, names=list(SOURCES), valid=valid, raw_valid=valid.copy(),
        boxes_anchor_relative=boxes.tolist(), alias_of=[-1, -1, -1], q0_is_fallback=False,
        selected_index=0, selected_source='q0', utility=[.8, .2, .1] if arm == 'full' else None,
        selected_utility=.8 if arm == 'full' else None,
        support=dict(raw_ids=list(range(support_count)), count=support_count,
                     target_count=target_count, target_purity=purity),
        geometry=geometry, q0_geometry=geometry[0], prior_geometry=geometry[1],
        trusted_propagation_geometry=geometry[2], selected_geometry=geometry[0],
        trusted_valid=True, trusted_target=trusted_target, trusted_score=score if arm == 'full' else None,
        legacy_q0_quality=score, trusted_eligible=eligible,
        oracle=dict(index=oracle, success=geometry[oracle]['success'], precision=geometry[oracle]['precision'],
                    utility=utility[oracle]), mechanism=dict(replaced_real_q0=False,
            erroneous_takeover=False, beneficial_takeover=False, recovered_bad_q0=False))
    reason = 'initialization' if arm == 'full' else 'policy_disabled'
    return dict(tracklet='unit', frame=frame, initialization=False, timestamp=100. + frame * .5,
        diagnostic_candidates=record, selected_index=0, selected_source='q0',
        success=geometry[0]['success'], precision=geometry[0]['precision'],
        iou=geometry[0]['iou'], distance=geometry[0]['center_error_m'],
        geometric_fg_count=support_count, supported=eligible, trusted_write=eligible,
        measurement_eligible=eligible, trusted_velocity_valid=False,
        diagnostic_anchor_box_world=anchor.tolist(), diagnostic_target_box_world=target_world.tolist(),
        diagnostic_first_frame_size_lwh=size.tolist(), diagnostic_target_size_lwh=size.tolist(),
        diagnostic_fine_box_world=world.tolist(), diagnostic_accepted_box_world=world.tolist(),
        diagnostic_search_main_box_world=anchor.tolist(), diagnostic_search_backup_box_world=anchor.tolist(),
        diagnostic_previous_measurement_eligible=-1, diagnostic_search_triggered=False,
        diagnostic_search_backup_active=False, search_reason=reason,
        diagnostic_search_reason_id=1 if arm == 'full' else 0,
        diagnostic_search_main_quota=768, diagnostic_search_backup_quota=256,
        diagnostic_search_trusted_age=frame * .5, diagnostic_search_backup_elapsed=0.,
        diagnostic_search_main_count=4, diagnostic_search_backup_only_count=0,
        diagnostic_search_sampled_main=4, diagnostic_search_sampled_backup=0,
        diagnostic_search_raw_target_count=4, diagnostic_search_main_target_count=4,
        diagnostic_search_union_target_count=4, diagnostic_search_sampled_main_target_count=4,
        diagnostic_search_sampled_backup_target_count=0, diagnostic_target_count=4,
        diagnostic_crop_target_count=4, diagnostic_sampled_target_count=4,
        diagnostic_crop_point_count=4, diagnostic_sampled_point_count=4)


def rows_for(row):
    return {('unit', 0): dict(frame=0, timestamp=100., initialization=True), ('unit', 1): row}


@pytest.mark.parametrize('arm', ['b0', 'b1', 'b1_mtp', 'full'])
def test_candidate_geometry_and_initial_single_roi_are_accepted(arm):
    row = row_fixture(arm)
    validate_candidates(row, 'unit', arm)
    summary = validate_search_rows(rows_for(row), normalize_config(dict(v40_arm=arm)), 'unit')
    assert summary['triggered_frames'] == 0
    assert summary['raw_target_present_frames'] == 1


@pytest.mark.parametrize('value', [float('nan'), -1., 1.1])
def test_unselected_candidate_exact_iou_cannot_be_nonfinite_or_forged(value):
    row = row_fixture()
    row['diagnostic_candidates']['geometry'][1]['exact_iou'] = value
    with pytest.raises(EvidenceError, match='geometry'):
        validate_candidates(row, 'unit', 'full')


def test_unselected_candidate_boxes_are_checked_against_geometry():
    row = row_fixture()
    row['diagnostic_candidates']['boxes_anchor_relative'][1][0] = 8.
    with pytest.raises(EvidenceError, match='geometry'):
        validate_candidates(row, 'unit', 'full')


@pytest.mark.parametrize('field,value', [('trusted_target', False), ('trusted_eligible', False)])
def test_trusted_target_and_gate_are_checked_after_geometry(field, value):
    row = row_fixture()
    row['diagnostic_candidates'][field] = value
    with pytest.raises(EvidenceError, match='trusted'):
        validate_candidates(row, 'unit', 'full')


def test_oracle_is_recomputed_with_original_valid_order():
    row = row_fixture()
    row['diagnostic_candidates']['oracle']['index'] = 2
    with pytest.raises(EvidenceError, match='oracle'):
        validate_candidates(row, 'unit', 'full')


def test_missing_or_forged_search_summary_cannot_override_frame_evidence():
    rows = rows_for(row_fixture())
    cfg = normalize_config(dict(v40_arm='full'))
    summary = validate_search_rows(rows, cfg, 'unit')
    assert validate_search_summary(summary, rows, cfg, 'unit') == summary
    for changed in (None, dict(summary, backup_added_target_frames=1)):
        with pytest.raises(EvidenceError, match='search summary'):
            validate_search_summary(changed, rows, cfg, 'unit')


@pytest.mark.parametrize('field,value', [('diagnostic_search_main_quota', 1024),
    ('diagnostic_search_triggered', True), ('diagnostic_search_sampled_main', 1025),
    ('diagnostic_search_union_target_count', 5), ('diagnostic_search_main_count', None),
    ('diagnostic_previous_measurement_eligible', 0)])
def test_missing_or_inconsistent_search_evidence_is_rejected(field, value):
    row = row_fixture()
    row[field] = value
    with pytest.raises(EvidenceError):
        validate_search_rows(rows_for(row), normalize_config(dict(v40_arm='full')), 'unit')


def test_full_backup_summary_is_recomputed_from_actual_previous_commit():
    first = row_fixture(q0_x=20., score=.1, support_count=0, target_count=0)
    second = row_fixture(frame=2, anchor_x=120., q0_x=-20.)
    second.update(diagnostic_previous_measurement_eligible=0, diagnostic_search_triggered=True,
        diagnostic_search_backup_active=True, search_reason='accepted_untrusted', diagnostic_search_reason_id=3,
        diagnostic_search_backup_box_world=[100., 200., 0., 0.],
        diagnostic_search_main_count=0, diagnostic_search_backup_only_count=4,
        diagnostic_search_sampled_main=0, diagnostic_search_sampled_backup=4,
        diagnostic_search_main_target_count=0, diagnostic_search_sampled_main_target_count=0,
        diagnostic_search_sampled_backup_target_count=4)
    rows = rows_for(first)
    rows['unit', 2] = second
    for row in (first, second):
        validate_candidates(row, 'unit', 'full')
    cfg = normalize_config(dict(v40_arm='full'))
    summary = validate_search_rows(rows, cfg, 'unit')
    assert summary['triggered_frames'] == summary['backup_added_target_frames'] == 1
    assert summary['correct_trusted_writes_with_backup_points'] == 1
    assert summary['main_crop_lost_frames'] == 1 and summary['union_crop_lost_frames'] == 0
    altered = deepcopy(rows)
    altered['unit', 2]['diagnostic_previous_measurement_eligible'] = 1
    with pytest.raises(EvidenceError, match='previous eligibility'):
        validate_search_rows(altered, cfg, 'unit')
    with pytest.raises(EvidenceError, match='trigger'):
        validate_search_rows(rows, normalize_config(dict(v40_arm='b1')), 'unit')
