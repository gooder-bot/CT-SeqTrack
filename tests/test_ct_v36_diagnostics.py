"""独立测量台账的标签权限、实际证据与严格提交后观察窗口。"""
from dataclasses import replace
import copy
import json

import numpy as np
import pytest
import torch

from models.ct_v36.candidate_records import candidate_records
from models.ct_v36.contracts import CandidateSet, TrackOutput
from models.ct_v36.diagnostics import mechanism_summary, measurement_followups
from models.ct_v36.model import select_candidates
from models.ct_v36.runtime import TrackingEvaluation, loss_episode_summary
from models.ct_v36.verification import FinalBoxVerifier
from tests.test_ct_v36_network import inputs, evidence_fixture


@pytest.fixture(autouse=True)
def cpu_threads():
    previous=torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(36)
    yield
    torch.set_num_threads(previous)


def logged_output(*, wrong=False):
    batch,observation,prior=inputs()
    batch['target_box']=torch.zeros(1,4)
    batch['anchor_box']=torch.tensor([[1.e8,-2.e8,5.,0.]],dtype=torch.double)
    batch['current_target_world']=batch['anchor_box'].clone()
    batch['prior_origin_target_world']=batch['anchor_box'].clone()
    batch['prior_origin_target_world'][:,0]+=1.
    batch['event_target_box']=batch['prior_origin_target_world'].clone()
    batch['event_boxes']=batch['anchor_box'][:,None].expand(-1,3,-1).clone()
    batch['event_boxes'][0,1,0]+=1.
    batch['event_times']=torch.tensor([[1.,2.,0.]],dtype=torch.double)
    batch['event_valid']=torch.tensor([[True,True,False]])
    batch['event_segment']=torch.tensor([[0,0,0]])
    batch['event_edge_valid']=torch.tensor([[False,True,False]])
    batch['current_time']=torch.tensor([3.],dtype=torch.double)
    batch['diagnostic_search_enabled']=torch.tensor([True])
    batch['diagnostic_search_target_count']=torch.tensor([5])
    batch['diagnostic_novel_target_count']=torch.tensor([9])
    batch['diagnostic_reachable_count']=torch.tensor([7])
    observation=replace(observation,foreground_probability=torch.ones_like(observation.foreground_probability))
    evidence=evidence_fixture(batch)
    evidence.point_indices[0,:8]=torch.arange(8)
    boxes=torch.tensor([[[0.,0.,0.,0.],[6.,0.,0.,0.],[2.5,0.,0.,0.],[10.,0.,0.,0.]]])
    candidates=CandidateSet(boxes,torch.ones(1,4,dtype=torch.bool),torch.zeros(1,3,4),
        torch.zeros(1,4,64),boxes[:,2:].clone())
    prior.box=boxes[:,1].clone()
    prior.world_box=batch['anchor_box']+prior.box.double()
    prior.origin_world=batch['prior_origin_target_world'].clone()
    prior.origin_frame=torch.tensor([1])
    prior.mean_xy=torch.tensor([[5.,0.]],requires_grad=True)
    prior.range_anchor=torch.tensor([[.5,.25]])
    prior.range_rate=torch.tensor([[.5,.25]])
    prior.horizon=torch.tensor([1.])
    verification=FinalBoxVerifier().eval()(candidates,observation,evidence,batch,prior)
    logits=torch.tensor([[[0.,0.,2.],[3.,3.,-20.],[-1.,-1.,3. if wrong else -2.],[-1.,-1.,4.]]],requires_grad=True)
    verification=replace(verification,logits=logits)
    accepted,quality,decision=select_candidates(candidates,verification)
    output=TrackOutput(accepted,decision.output_index,quality,prior,observation,evidence,candidates,verification,decision)
    return batch,output


@pytest.mark.parametrize('wrong',[False,True])
def test_records_independent_measurement_correctness_actual_support_and_ranges(wrong):
    batch,output=logged_output(wrong=wrong)
    record=candidate_records(batch,output)[0]
    action=record['action']
    assert action['output_index']==1 and action['measurement_index']==(2 if wrong else 0)
    assert action['output_measurement_disagreement']
    assert action['measurement_correct'] is (not wrong)
    assert action['wrong_measurement_over_good_candidate'] is wrong
    assert action['memory_eligible']
    assert record['candidates'][3]['measurement_score']>.9
    assert not record['candidates'][3]['eligible']  # 分数不能替代当前框内实证。
    assert record['candidates'][1]['measurement_score']==0
    assert record['candidates'][0]['support_count']==32
    assert len(set(record['candidates'][0]['support_raw_ids']))==32
    assert record['candidates'][0]['measurement_target']
    assert record['acquisition']['search_target_count']==5
    assert record['acquisition']['selected_target_count']==8
    assert record['acquisition']['pool_target_count']==24
    assert record['propagation']['axis_covered']==[False,True]
    assert record['propagation']['uncertainty_rectangle_area_m2']==2.
    np.testing.assert_allclose(record['propagation']['decomposition_residual_xy'],[0.,0.])
    assert record['propagation']['absolute_xy_error_m']==6.
    assert record['event_cv']['absolute_xy_error_m']==2.
    assert record['event_cv']['motion_error_xy_m']==2.
    assert 'not_covariance' in record['propagation']['range_semantics']
    json.dumps(record,allow_nan=False)


def test_gt_diagnostics_are_passive_without_rng_forward_or_autograd_mutation():
    batch,output=logged_output()
    state=torch.get_rng_state().clone()
    before=copy.deepcopy({name:value.detach() for name,value in vars(output.verification).items()})
    accepted=output.accepted_box.detach().clone()
    selected=output.decision.measurement_index.clone()
    first=candidate_records(batch,output)[0]
    changed=dict(batch,target_box=batch['target_box']+100,
        current_target_world=batch['current_target_world']+100,
        extension_labels=torch.zeros_like(batch['extension_labels']),
        segmentation_labels=torch.zeros_like(batch['segmentation_labels']))
    second=candidate_records(changed,output)[0]
    assert first['action']['measurement_correct'] and not second['action']['measurement_correct']
    assert second['candidates'][0]['support_target_count']==0
    torch.testing.assert_close(torch.get_rng_state(),state,rtol=0,atol=0)
    torch.testing.assert_close(output.accepted_box,accepted,rtol=0,atol=0)
    torch.testing.assert_close(output.decision.measurement_index,selected,rtol=0,atol=0)
    for name,value in before.items():
        torch.testing.assert_close(getattr(output.verification,name),value,rtol=0,atol=0)
    assert output.verification.logits.grad is None
    output.selected_quality.sum().backward()
    assert output.verification.logits.grad[0,1,:2].abs().sum()>0
    assert output.verification.logits.grad[0,[0,2,3]].count_nonzero()==0


def test_mechanism_summary_keeps_disabled_search_missing_and_actual_memory_labels():
    batch,output=logged_output()
    correct=candidate_records(batch,output)[0]
    batch,output=logged_output(wrong=True)
    incorrect=candidate_records(batch,output)[0]
    disabled=copy.deepcopy(correct);disabled['acquisition']['enabled']=False
    rows=[dict(diagnostic_candidates=correct,memory_write=True,memory_wrong_write=True),
          dict(diagnostic_candidates=incorrect,memory_write=True,memory_wrong_write=False),
          dict(diagnostic_candidates=disabled),dict(),dict(initialization=True)]
    result=mechanism_summary(rows)
    assert result['recorded_frames']==3 and result['missing_frames']==1
    assert result['measurement']['correct_disagreements']==2
    assert result['measurement']['incorrect_disagreements']==1
    assert result['memory']['wrong_writes']==1  # 不能继承事件正确性标签。
    assert result['acquisition']['enabled_frames']==2
    assert result['acquisition']['disabled_frames']==1
    assert result['acquisition']['search_target_count']==dict(recorded_frames=2,total=10)
    assert result['position_ranges']['by_measurement_age']['[1,2)']['recorded_frames']==3
    assert result['event_cv']['actual_minus_cv_error_m']['mean']==4.


def test_event_age_is_independent_from_accepted_core_prior_origin_age():
    batch,output=logged_output()
    batch['current_time']=torch.tensor([5.],dtype=torch.double)
    record=candidate_records(batch,output)[0]
    assert record['propagation']['prior_origin_age']==1.
    assert record['propagation']['measurement_age']==3.
    summary=mechanism_summary([dict(diagnostic_candidates=record)])
    assert summary['position_ranges']['by_measurement_age']['[2,inf)']['recorded_frames']==1
    del batch['event_times']
    record=candidate_records(batch,output)[0]
    assert record['propagation']['measurement_age'] is None
    assert mechanism_summary([dict(diagnostic_candidates=record)])['position_ranges']['by_measurement_age']['unknown']['recorded_frames']==1


def followup_row(frame,iou,correct=None,*,event=False):
    action=dict(measurement_accepted=event,measurement_index=0 if event else -1,
        measurement_correct=correct,output_measurement_disagreement=event)
    return dict(tracklet='a',frame=frame,timestamp=frame*.5,iou=iou,
        diagnostic_candidates=dict(action=action))


def test_followups_strict_future_superseded_same_frame_and_right_censor():
    rows=[followup_row(1,.05,True,event=True),followup_row(2,.2,False,event=True),
        followup_row(3,.6,True,event=True),followup_row(4,.7),followup_row(5,.05,False,event=True)]
    original=copy.deepcopy(rows)
    result=measurement_followups(rows)
    events=result['events']
    assert [e['outcome'] for e in events]==['superseded_before_recovery','recovered_before_next_event',
        'subsequent_loss_before_next_event','track_end_unrecovered']
    assert [e['right_censored'] for e in events]==[False,False,False,True]
    assert events[1]['future_prediction_steps']==1 and events[1]['elapsed_sensor_seconds']==.5
    assert events[2]['same_frame_recovery'] and not events[2]['ongoing_loss_at_commit']
    assert result['by_correctness']['incorrect']['recovery_steps']['count']==1
    assert result['by_correctness']['correct']['recovery_steps']['count']==0
    assert rows==original
    episodes=loss_episode_summary(rows)
    assert episodes['all']['episodes']==2 and episodes['unrecovered']['episodes']==1
    assert episodes['episodes'][0]['frames']==2  # 原benchmark恢复帧仍不计失跟长度。


def test_followups_missing_timestamp_no_fabricated_seconds_and_unique_frames():
    rows=[followup_row(1,.05,True,event=True),followup_row(2,.6)]
    rows[0].pop('timestamp')
    result=measurement_followups(rows)
    assert result['events'][0]['elapsed_sensor_seconds'] is None
    assert result['all']['recovery_steps']['mean']==1.
    assert result['all']['recovery_sensor_seconds']['count']==0
    with pytest.raises(ValueError,match='unique and causal'):
        measurement_followups(rows+[rows[1]])


def test_runtime_summary_has_both_passive_extensions_without_changing_metrics():
    tracker=TrackingEvaluation()
    tracker.add_initialization('a',timestamp=0.)
    tracker.rows.append(dict(tracklet='a',frame=1,timestamp=.5,initialization=False,
        iou=.6,distance=.3,success=.6,precision=.8,mode_count=0,target_mode_count=0))
    summary=tracker.summary()
    assert summary['success']==80. and summary['precision']==90.
    assert summary['mechanism_diagnostics']['missing_frames']==1
    assert summary['measurement_followups']['all']['events']==0
