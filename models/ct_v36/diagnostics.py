"""v36 只读机制汇总与事件后的观察窗口；不反馈模型或状态。"""
from __future__ import annotations

from collections import Counter
import numpy as np


def _mean(values):
    values=[value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def _ratio(numerator,denominator):
    return numerator/denominator if denominator else None


def _distribution(values):
    values=[value for value in values if value is not None]
    return dict(count=len(values),mean=_mean(values),median=float(np.median(values)) if values else None,
                p90=float(np.percentile(values,90)) if values else None)


def _age_bin(age):
    if age is None:return 'unknown'
    if age<.5:return '[0,0.5)'
    if age<1:return '[0.5,1)'
    if age<2:return '[1,2)'
    return '[2,inf)'


def _range_summary(records):
    p=[record['propagation'] for record in records if record['propagation']['prior_valid']]
    return dict(recorded_frames=len(p),parallel_coverage=_mean([x['axis_covered'][0] for x in p]),
        perpendicular_coverage=_mean([x['axis_covered'][1] for x in p]),
        joint_coverage=_mean([x['joint_covered'] for x in p]),
        area_m2=_distribution([x['uncertainty_rectangle_area_m2'] for x in p]),
        absolute_xy_error_m=_distribution([x['absolute_xy_error_m'] for x in p]),
        anchor_xy_error_m=_distribution([x.get('anchor_xy_error_m') for x in p]),
        motion_xy_error_m=_distribution([x.get('motion_xy_error_m') for x in p]),
        valid_motion_xy_error_m=_distribution([x.get('motion_xy_error_m') for x in p if x['motion_valid']]))


def mechanism_summary(rows):
    predictions=[row for row in rows if not row.get('initialization',False)]
    records=[row['diagnostic_candidates'] for row in predictions if row.get('diagnostic_candidates') is not None]
    actions=[record['action'] for record in records]
    admitted=[action for action in actions if action['measurement_accepted']]
    correct=sum(action['measurement_correct'] is True for action in admitted)
    wrong=sum(action['measurement_correct'] is False for action in admitted)
    disagreements=[action for action in admitted if action['output_measurement_disagreement']]
    reasons=Counter(reason for action in admitted for reason in action['wrong_measurement_reasons'])
    outputs=Counter(record['candidates'][action['output_index']]['source'] for record,action in zip(records,actions))
    measurements=Counter(record['candidates'][action['measurement_index']]['source']
        for record,action in zip(records,actions) if action['measurement_index']>=0)
    ranges=_range_summary(records)
    ranges['by_measurement_age']={name:_range_summary([record for record in records
        if _age_bin(record['propagation']['measurement_age'])==name])
        for name in ('[0,0.5)','[0.5,1)','[1,2)','[2,inf)','unknown')}
    ranges['interpretation']='unclipped_marginal_position_ranges; joint_coverage_measured_not_guaranteed; not_covariance'
    cv=[record for record in records if record.get('event_cv') is not None]
    search=[record for record in records if record['acquisition']['enabled'] is True]
    def search_field(field):
        values=[r['acquisition'][field] for r in search if r['acquisition'].get(field) is not None]
        return dict(recorded_frames=len(values),total=int(sum(values)) if values else None)
    opportunity=[record for record in search if record['candidates'][0]['valid']
        and record['candidates'][0]['geometry']['exact_iou']<.1
        and record['acquisition']['selected_target_count']>0]
    usable=sum(any(mode['candidate_valid'] and mode['decoded_exact_iou']>=.5 for mode in record['modes'])
               for record in opportunity)
    writes=[row for row in predictions if row.get('memory_write') is True]
    memory_labels=[row for row in writes if row.get('memory_wrong_write') is not None]
    return dict(schema='ct_seqtrack.v36.mechanism_summary.v1',prediction_frames=len(predictions),
        recorded_frames=len(records),missing_frames=len(predictions)-len(records),
        measurement=dict(accepted=len(admitted),correct=correct,incorrect=wrong,
            correctness_rate=_ratio(correct,correct+wrong),output_disagreements=len(disagreements),
            correct_disagreements=sum(x['measurement_correct'] is True for x in disagreements),
            incorrect_disagreements=sum(x['measurement_correct'] is False for x in disagreements),
            missed_good_measurement=sum(x['missed_good_measurement'] for x in actions),
            wrong_over_available_good=sum(x['wrong_measurement_over_good_candidate'] for x in actions),
            wrong_reasons=dict(reasons),output_sources=dict(outputs),measurement_sources=dict(measurements),
            output_measurement_distance_m=_distribution([x['output_measurement_center_distance_m'] for x in admitted]),
            interpretation='labels_on_recorded_candidates; ordinary_BCE_and_threshold_do_not_prove_calibration'),
        position_ranges=ranges,
        event_cv=dict(recorded_frames=len(cv),motion_valid_frames=sum(r['event_cv']['motion_valid'] for r in cv),
            actual_prior_xy_error_m=_distribution([r['propagation']['absolute_xy_error_m'] for r in cv]),
            cv_xy_error_m=_distribution([r['event_cv']['absolute_xy_error_m'] for r in cv]),
            actual_minus_cv_error_m=_distribution([r['propagation']['absolute_xy_error_m']-r['event_cv']['absolute_xy_error_m'] for r in cv]),
            interpretation='offline_geometry_in_same_logged_states_not_intervened_tracking_score'),
        acquisition=dict(enabled_frames=len(search),disabled_frames=sum(r['acquisition']['enabled'] is False for r in records),
            unknown_enabled_frames=sum(r['acquisition']['enabled'] is None for r in records),
            **{field:search_field(field) for field in ('novel_target_count','reachable_count','search_target_count',
                'pool_target_count','selected_target_count','pool_count','selected_count')},
            q0_lost_with_selected_target=len(opportunity),with_usable_mode=usable,
            usable_mode_rate=_ratio(usable,len(opportunity)),
            q0_lost_definition='valid_q0_exact_iou_lt_0.1_and_selected_target_count_gt_0',
            usable_mode_definition='valid_final_mode_exact_iou_ge_0.5'),
        memory=dict(writes=len(writes),correctness_recorded_writes=len(memory_labels),
            wrong_writes=sum(bool(row['memory_wrong_write']) for row in memory_labels),
            wrong_write_rate=_ratio(sum(bool(row['memory_wrong_write']) for row in memory_labels),len(memory_labels)),
            interpretation='actual_written_foreground_subset; not_inherited_from_event_correctness'))


def _action(row):
    return (row.get('diagnostic_candidates') or {}).get('action',{})


def _accepted_event(row):
    return bool(row.get('measurement_accepted',_action(row).get('measurement_accepted',False)))


def measurement_followups(rows):
    """t提交只能解释t之后的输出；下一事件是替代事件，不算已经恢复。"""
    tracks={}
    for row in rows:
        if not row.get('initialization',False):
            tracks.setdefault(row['tracklet'],[]).append(row)
    events=[]
    for track,values in tracks.items():
        values=sorted(values,key=lambda row:row['frame'])
        if any(a['frame']>=b['frame'] for a,b in zip(values,values[1:])):
            raise ValueError('followup frame keys must be unique and causal')
        status=[];lost=False
        for row in values:
            before=lost
            if row['iou']<.1:lost=True
            recovered=lost and row['iou']>=.5
            if recovered:lost=False
            status.append((lost,bool(before and recovered)))
        for index,row in enumerate(values):
            if not _accepted_event(row):continue
            ongoing,same_frame_recovery=status[index]
            action=_action(row)
            record=dict(tracklet=track,event_frame=int(row['frame']),
                measurement_index=int(row.get('measurement_index',action.get('measurement_index',-1))),
                measurement_correct=action.get('measurement_correct'),
                output_measurement_disagreement=bool(row.get('output_measurement_disagreement',action.get('output_measurement_disagreement',False))),
                ongoing_loss_at_commit=ongoing,same_frame_recovery=same_frame_recovery,
                outcome='track_end_unrecovered' if ongoing else 'track_end_without_new_loss',
                right_censored=True,end_frame=int(values[-1]['frame']))
            end=values[-1]
            for following in values[index+1:]:
                # 当前帧输出先于该帧的新事件提交，因此恢复/再失跟优先。
                if ongoing and following['iou']>=.5:
                    record.update(outcome='recovered_before_next_event',right_censored=False);end=following;break
                if not ongoing and following['iou']<.1:
                    record.update(outcome='subsequent_loss_before_next_event',right_censored=False);end=following;break
                if _accepted_event(following):
                    record.update(outcome='superseded_before_recovery' if ongoing else 'superseded_without_new_loss',
                                  right_censored=False);end=following;break
            record['end_frame']=int(end['frame'])
            record['future_prediction_steps']=int(end['frame']-row['frame'])
            start_time,end_time=row.get('timestamp'),end.get('timestamp')
            seconds=float(end_time-start_time) if start_time is not None and end_time is not None else None
            if seconds is not None and seconds<0:
                raise ValueError('followup sensor timestamps must be causal')
            record['elapsed_sensor_seconds']=seconds
            events.append(record)
    def summarize(selected):
        recovered=[r for r in selected if r['outcome']=='recovered_before_next_event']
        return dict(events=len(selected),outcomes=dict(Counter(r['outcome'] for r in selected)),
            same_frame_recoveries=sum(r['same_frame_recovery'] for r in selected),
            ongoing_loss_events=sum(r['ongoing_loss_at_commit'] for r in selected),
            recovery_steps=_distribution([r['future_prediction_steps'] for r in recovered]),
            recovery_sensor_seconds=_distribution([r['elapsed_sensor_seconds'] for r in recovered]))
    return dict(schema='ct_seqtrack.v36.measurement_followups.v1',
        definition='strictly_after_commit; benchmark_iou_lt0.1_to_ge0.5; recovery_precedes_same_frame_next_commit',
        interpretation='observational_associations_not_causal_effects; superseded_is_a_competing_event_not_recovery',
        all=summarize(events),by_correctness={name:summarize([r for r in events if r['measurement_correct'] is value])
            for name,value in (('correct',True),('incorrect',False),('unknown',None))},events=events)
