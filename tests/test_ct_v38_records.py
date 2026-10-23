"""v38 预测→提交→被动台账、未选 q0 可信标签与报告切片。"""
from dataclasses import replace
import math

import pytest
import torch

from models.ct_v38.candidate_records import candidate_records
from models.ct_v38.data import BatchBuilder, RawEndpointDataset
from models.ct_v38.losses import supervision_targets
from models.ct_v38.model import JointTracker, build_candidates, select_candidates
from models.ct_v38.runtime import TrackingEvaluation, reliability_summary, motion_slices
from tests.helpers.raw_tracking import TinySource, request
from tests.test_ct_v38_data import cfg


@pytest.fixture(autouse=True)
def cpu_state():
    threads, rng = torch.get_num_threads(), torch.get_rng_state()
    torch.set_num_threads(1)
    torch.manual_seed(738)
    yield
    torch.set_num_threads(threads)
    torch.set_rng_state(rng)


def case(*, selected=1, trusted_probability=.9, same_boxes=False, q0_real=True):
    """真实数据准备/模型特征，合成可解释的候选几何用于独立台账核验。"""
    config = cfg()
    builder = BatchBuilder(config)
    row = RawEndpointDataset(TinySource())[request(1, branch=4)]
    batch = builder.prepare([row])
    model = JointTracker(config).eval()
    with torch.no_grad():
        prior = model.plan_prior(batch)
        builder.supervise_prior(batch, prior, training=False)
        base = model(batch, prior)
        decoder_boxes = base.decoder.q0_box.clone()
        # 仅测试合成：设已形成的 q0 等于标签，明确应有正 trusted 事件。
        decoder_boxes[:] = batch['target_box']
        decoder = replace(base.decoder, q0_box=decoder_boxes)
        observation = replace(base.observation,
            foreground_probability=torch.ones_like(base.observation.foreground_probability),
            sequence_valid=torch.tensor([q0_real]))
        prior_box = decoder_boxes.clone()
        fallback_box = decoder_boxes.clone()
        if not same_boxes:
            prior_box[:, 0] += 3.
            fallback_box[:, 0] -= 3.
        prior = replace(prior, box=prior_box)
        batch['fallback_box'] = fallback_box
        candidates = build_candidates(decoder, observation, batch, prior)
        verification = model.verifier(candidates, observation, batch, prior)
        logits = torch.zeros_like(verification.logits)
        logits[:, selected, :2] = 2.
        logits[:, 0, 2] = math.log(trusted_probability / (1. - trusted_probability))
        verification = replace(verification, logits=logits)
        selection, commit, quality, table = select_candidates(candidates, verification, observation, batch)
        output = replace(base, accepted_box=selection.accepted_box, selected_index=selection.selected_index,
            selected_quality=quality, prior=prior, observation=observation, decoder=decoder,
            candidates=candidates, verification=verification, selection_decision=selection,
            commit_decision=commit, legacy_q0_quality=commit.legacy_q0_quality,
            output_utility=selection.output_utility, commit_quality=table,
            q0_is_fallback=candidates.q0_is_fallback)
    return config, builder, [row], batch, output


def test_unselected_real_q0_keeps_full_T_event_support_and_calibration():
    _, builder, _, batch, output = case(selected=1)
    targets = supervision_targets(batch, output)
    assert output.selected_index.item() == 1
    assert output.commit_decision.support_mask.sum() == 0
    assert output.commit_decision.trusted_score.item() == 0.
    committed = builder.commit(output, batch, diagnostics=True)[0]
    record = candidate_records(batch, output)[0]
    assert not committed['supported'] and committed['propagation_only']
    assert record['trusted_valid'] and record['trusted_target']
    assert not record['trusted_eligible']
    assert record['support']['count'] == targets['support_count'].item() == 4
    assert record['support']['target_purity'] == targets['support_purity'].item() == 1.
    assert record['trusted_score'] == pytest.approx(.9)
    reliability = reliability_summary([record])
    assert reliability['count'] == reliability['positive'] == 1
    assert reliability['negative'] == 0
    assert reliability['brier'] == pytest.approx(.01)
    assert sum(bucket['count'] for bucket in reliability['bins']) == 1


@pytest.mark.parametrize('foreground_count,expected', [(0, False), (1, False), (2, True), (4, True)])
def test_record_T_target_matches_loss_target_at_purity_boundary(foreground_count, expected):
    _, builder, _, batch, output = case(selected=1)
    batch['segmentation_labels'][:, -1].zero_()
    batch['segmentation_labels'][:, -1, :foreground_count] = 1
    targets = supervision_targets(batch, output)
    builder.commit(output, batch)
    record = candidate_records(batch, output)[0]
    assert record['trusted_target'] is expected
    assert targets['trusted'].item() == float(expected)
    assert record['support']['target_count'] == foreground_count
    assert record['support']['target_purity'] == pytest.approx(foreground_count / 4.)


@pytest.mark.parametrize('q0_real,valid,alias,selected', [
    (True, [True, False, False], [-1, 0, 0], 0),
    (False, [False, True, False], [1, -1, 1], 1),
])
def test_same_box_dedup_preserves_source_and_cannot_create_trusted_alias(q0_real, valid, alias, selected):
    _, builder, _, batch, output = case(same_boxes=True, q0_real=q0_real)
    assert output.selected_index.item() == selected
    committed = builder.commit(output, batch, diagnostics=True)[0]
    record = candidate_records(batch, output)[0]
    assert record['valid'] == valid and record['alias_of'] == alias
    assert record['selected_source'] == ('q0' if q0_real else 'prior')
    assert record['trusted_valid'] is q0_real
    assert record['support']['count'] == (4 if q0_real else 0)
    assert committed['supported'] is q0_real
    assert reliability_summary([record])['count'] == int(q0_real)


def test_full_evaluation_reports_unselected_q0_T_and_paired_oracle_without_extra_writes():
    config, builder, rows, batch, output = case(selected=1)
    evaluation = TrackingEvaluation(config)
    committed = builder.commit(output, batch, diagnostics=True)
    evaluation.add_batch(rows, batch, output, committed)
    report = evaluation.summary()
    assert report['prediction_frames'] == 1
    assert report['selection']['prior']['count'] == 1
    assert report['trusted']['writes'] == 0
    assert report['trusted_reliability']['count'] == 1
    assert report['trusted_reliability']['positive'] == 1
    assert report['diagnostics']['erroneous_takeover'] == 1
    assert 'memory_writes' not in report['diagnostics']
    assert report['same_state_oracle']['interpretation'] == 'passive_one_step_not_closed_loop'
    assert report['same_state_oracle']['success'] >= report['success']
    prediction = [row for row in evaluation.rows if not row['initialization']][0]
    assert prediction['selected_source'] == 'prior'
    assert prediction['selected_quality'] == 0.
    assert prediction['diagnostic_candidates']['trusted_score'] == pytest.approx(.9)
    assert prediction['q0_trusted_score'] == pytest.approx(.9)
    assert prediction['trusted_score'] == 0.
    assert prediction['commit_quality'] == 0.


def test_invalid_velocity_age_is_masked_in_metadata_and_never_freshened_by_propagation():
    _, builder, rows, batch, output = case(selected=2)
    state = builder.states[rows[0]['request'].state_key]
    assert not state.trusted_velocity_valid and state.last_velocity_update_time is None
    batch['trusted_velocity_age'][:] = 999.
    verifier = JointTracker(cfg()).verifier
    result = verifier(output.candidates, output.observation, batch, output.prior)
    assert result.metadata[0, :, 43:].eq(0).all()
    committed = builder.commit(output, batch, diagnostics=True)[0]
    assert committed['trusted_velocity_age'] == 0.
    assert committed['last_velocity_update_time'] is None
    assert committed['commit_source'] == 'trusted_propagation'
    assert not committed['strong'] and not committed['trusted_position_updated']


def test_motion_slices_use_endpoint_motion_and_ever_moving_track_differently():
    rows = [dict(initialization=True, tracklet='moving', frame=0)]
    for frame, moving in enumerate((False, True, True, False), 1):
        rows.append(dict(initialization=False, tracklet='moving', frame=frame,
            diagnostic_moving=moving, success=frame / 10., precision=frame / 5., distance=frame * 3.))
    rows.append(dict(initialization=False, tracklet='static', frame=1,
        diagnostic_moving=False, success=.9, precision=.8, distance=.1))
    result = motion_slices(rows)
    assert not result['initialization_included']
    assert result['ever_moving_tracks']['frames'] == 4
    assert result['moving_endpoints']['frames'] == 2
    assert result['low_motion_endpoints']['frames'] == 3
    assert result['start_boundary']['frames'] == result['stop_boundary']['frames'] == 1
    assert result['start_boundary']['success'] == pytest.approx(20.)
    assert result['stop_boundary']['success'] == pytest.approx(40.)
    assert result['ever_moving_tracks']['over5m'] == 3
    assert result['ever_moving_tracks']['over10m'] == 1


def test_reliability_bins_include_boundary_one_and_exclude_nonreal_q0():
    report = reliability_summary([
        dict(trusted_score=1., trusted_valid=True, trusted_target=True),
        dict(trusted_score=.5, trusted_valid=True, trusted_target=False),
        dict(trusted_score=0., trusted_valid=True, trusted_target=False),
        dict(trusted_score=.9, trusted_valid=False, trusted_target=None),
        dict(trusted_score=None, trusted_valid=True, trusted_target=True),
    ])
    assert report['count'] == 3 and report['positive'] == 1 and report['negative'] == 2
    assert report['bins'][9]['count'] == report['bins'][5]['count'] == report['bins'][0]['count'] == 1
    assert report['brier'] == pytest.approx(.25 / 3.)
