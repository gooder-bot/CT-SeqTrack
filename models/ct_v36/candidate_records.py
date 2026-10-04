"""v36 已完成前向的被动台账；监督标签只用于提交后的解释。"""
from __future__ import annotations

import numpy as np
import torch

from .losses import supervision_targets

CANDIDATE_RECORD_SCHEMA = 'ct_seqtrack.v36.candidate_records.v1'
SOURCES = ['q0', 'prior', 'mode0', 'mode1']


def _array(value):
    return value.detach().cpu().numpy()


def _geometry(box, target, exact_iou, utility_target):
    delta = np.asarray(box[:3], np.float64)-np.asarray(target[:3], np.float64)
    yaw = float(abs(np.arctan2(np.sin(box[3]-target[3]),np.cos(box[3]-target[3]))))
    return dict(exact_iou=float(exact_iou),center_error_m=float(np.linalg.norm(delta)),
        center_xy_error_m=float(np.linalg.norm(delta[:2])),center_error_xyz=delta.tolist(),
        yaw_error_rad=yaw,axis_yaw_error_rad=min(yaw,np.pi-yaw),
        success_target=float(utility_target[0]),precision_target=float(utility_target[1]))


def _event_cv(batch, index, target):
    """事件CV纯算术反事实，不调用事件编码网络，也不使用GT决定运动边。"""
    required=('event_boxes','event_times','event_valid','event_segment','event_edge_valid','current_time')
    if not all(name in batch for name in required):
        return None
    boxes,times,valid,segments,incoming,now = [_array(batch[name])[index] for name in required]
    slots=np.flatnonzero(valid)
    if not len(slots):
        return None
    latest=int(slots[-1]); velocity=np.zeros(2,np.float64); motion_valid=False; midpoint=None
    for slot in range(1,len(valid)):
        if (valid[slot-1] and valid[slot] and incoming[slot]
                and segments[slot-1]==segments[slot]==segments[latest]):
            gap=float(times[slot]-times[slot-1])
            if gap<=0:
                raise ValueError('passive event CV requires causal event timestamps')
            velocity=(boxes[slot,:2]-boxes[slot-1,:2])/gap
            midpoint=float((times[slot]+times[slot-1])*.5); motion_valid=True
    horizon=float(now-times[latest])
    if horizon<0:
        raise ValueError('passive event CV cannot query before its origin')
    world=boxes[latest].astype(np.float64,copy=True)
    world[:2]+=velocity*horizon
    result=dict(world_box=world.tolist(),motion_valid=motion_valid,horizon=horizon,
        velocity_xy=velocity.tolist(),velocity_age=None if midpoint is None else float(now-midpoint),
        absolute_xy_error_m=float(np.linalg.norm(world[:2]-target[:2])),
        scope='same_logged_event_state_offline_geometry_not_a_new_rollout')
    if 'event_target_box' in batch:
        origin_gt=_array(batch['event_target_box'])[index]
        result['motion_error_xy']=(velocity*horizon-(target[:2]-origin_gt[:2])).tolist()
        result['motion_error_xy_m']=float(np.linalg.norm(result['motion_error_xy']))
    return result


@torch.no_grad()
def candidate_records(batch, output):
    """每个eval端点一个记录；无网络/RNG/张量原位修改。"""
    targets=supervision_targets(batch,output)
    target_arrays={name:_array(value) for name,value in targets.items()}
    candidates,verification,evidence,prior=output.candidates,output.verification,output.evidence,output.prior
    boxes,valid,logits=_array(candidates.boxes),_array(candidates.valid).astype(bool),_array(verification.logits)
    utilities,scores=_array(verification.utility),_array(verification.measurement_score)
    success_precision=_array(verification.logits[...,:2].sigmoid())
    seeds=_array(candidates.seed_boxes)
    sizes=_array(batch['box_size'])
    target_sizes=_array(batch.get('target_box_size',batch['box_size']))
    anchors=_array(batch['anchor_box']).astype(np.float64)
    truths=_array(batch['target_box']).astype(np.float64)
    current_world=(_array(batch['current_target_world']).astype(np.float64)
                   if 'current_target_world' in batch else None)
    output_indices=_array(output.decision.output_index)
    measurements=_array(output.decision.measurement_index)
    memory_eligible=_array(output.decision.memory_eligible)
    support=_array(verification.support_mask).astype(bool)
    authorized=_array(verification.authorized).astype(bool)
    ids=_array(verification.support_ids)
    support_xyz=_array(verification.support_xyz)
    probabilities=_array(verification.support_probabilities)
    pool_labels=_array(batch['extension_labels'])
    pool_valid=_array(batch['extension_valid']).astype(bool)
    fields={name:_array(getattr(evidence,name)) for name in ('mode_valid','members','point_valid',
        'point_ids','point_indices','point_xyz','vote_xyz','centers_xyz','selected_identity_logits','reliability_logits')}
    prior_fields={name:_array(getattr(prior,name)) for name in ('world_box','origin_world','origin_frame',
        'mean_xy','axes','range_xy','range_anchor','range_rate','horizon','velocity_age','prior_valid','motion_valid')}
    origins_gt=_array(batch['prior_origin_target_world']).astype(np.float64) if 'prior_origin_target_world' in batch else None
    passive={name:_array(value) for name,value in batch.items() if name.startswith('diagnostic_') and torch.is_tensor(value)}
    records=[]
    for i in range(len(boxes)):
        world=boxes[i].astype(np.float64,copy=True);world[:,:3]+=anchors[i,:3]
        target=truths[i].copy();target[:3]+=anchors[i,:3]
        if current_world is not None:
            target=current_world[i]
        event_cv=_event_cv(batch,i,target)
        oi,mi=int(output_indices[i]),int(measurements[i])
        if oi not in range(4) or not valid[i,oi] or mi not in (-1,0,2,3):
            raise ValueError('invalid committed candidate indices in passive record')
        rows=[]
        for j in range(4):
            count=int(target_arrays['support_count'][i,j])
            foreground=int(target_arrays['foreground_count'][i,j])
            truth=bool(target_arrays['measurement'][i,j]) and bool(valid[i,j])
            reasons=[]
            if j==1: reasons.append('propagation_source')
            if count<1: reasons.append('no_predicted_foreground_support')
            if foreground<1: reasons.append('no_true_target_support')
            if count and target_arrays['support_purity'][i,j]<.5: reasons.append('support_purity_below_half')
            if target_arrays['exact_iou'][i,j]<.5: reasons.append('geometry_iou_below_half')
            rows.append(dict(source=SOURCES[j],valid=bool(valid[i,j]),box_world=world[j].tolist(),
                logits=logits[i,j].tolist(),success_score=float(success_precision[i,j,0]),
                precision_score=float(success_precision[i,j,1]),utility=float(utilities[i,j]),
                measurement_score=float(scores[i,j]),measurement_target=truth,
                support_raw_ids=ids[i,support[i,j]].tolist(),support_count=count,
                support_target_count=foreground,support_target_purity=float(target_arrays['support_purity'][i,j]) if count else None,
                authorized_raw_ids=ids[i,authorized[i,j]].tolist(),
                eligible=bool(valid[i,j] and j!=1 and count>=1 and scores[i,j]>=.5),
                rejection_truth_reasons=reasons if valid[i,j] else ['invalid_candidate'],
                geometry=_geometry(world[j],target,target_arrays['exact_iou'][i,j],target_arrays['utility'][i,j]) if valid[i,j] else None))
        good_available=any(row['valid'] and row['measurement_target'] for row in rows)
        correct=bool(rows[mi]['measurement_target']) if mi>=0 else None
        disagreement=mi>=0 and oi!=mi
        action=dict(output_index=oi,measurement_index=mi,measurement_accepted=mi>=0,
            memory_eligible=bool(memory_eligible[i]),
            output_measurement_disagreement=disagreement,measurement_correct=correct,
            good_measurement_available=good_available,
            missed_good_measurement=bool(good_available and mi<0),
            wrong_measurement_over_good_candidate=bool(good_available and mi>=0 and not correct),
            accepted_exact_iou=float(target_arrays['exact_iou'][i,oi]),
            output_measurement_center_distance_m=float(np.linalg.norm(world[oi,:3]-world[mi,:3])) if mi>=0 else None,
            wrong_measurement_reasons=rows[mi]['rejection_truth_reasons'] if mi>=0 and not correct else [])
        pworld=prior_fields['world_box'][i].astype(np.float64)
        origin=prior_fields['origin_world'][i].astype(np.float64)
        axes=prior_fields['axes'][i].astype(np.float64)
        range_xy=prior_fields['range_xy'][i].astype(np.float64)
        actual_error=pworld[:2]-target[:2]
        axis_error=np.abs(actual_error@axes)
        age=float(prior_fields['horizon'][i])
        propagation=dict(prior_world=pworld.tolist(),origin_world=origin.tolist(),
            origin_frame=int(prior_fields['origin_frame'][i]),prior_valid=bool(prior_fields['prior_valid'][i]),
            motion_valid=bool(prior_fields['motion_valid'][i]),prior_origin_age=age,
            measurement_age=None if event_cv is None else event_cv['horizon'],
            velocity_age=float(prior_fields['velocity_age'][i]),axes_columns=axes.tolist(),
            range_xy=range_xy.tolist(),range_anchor=prior_fields['range_anchor'][i].tolist(),
            range_rate=prior_fields['range_rate'][i].tolist(),absolute_error_xy=actual_error.tolist(),
            absolute_xy_error_m=float(np.linalg.norm(actual_error)),axis_absolute_error=axis_error.tolist(),
            axis_covered=(axis_error<=range_xy).tolist(),joint_covered=bool((axis_error<=range_xy).all()),
            uncertainty_rectangle_area_m2=float(4*np.prod(range_xy)),
            range_semantics='empirical_marginal_0.9_position_error_bounds_not_covariance_or_joint_0.9_guarantee')
        if origins_gt is not None:
            anchor_error=origin[:2]-origins_gt[i,:2]
            true_motion=target[:2]-origins_gt[i,:2]
            predicted_motion=prior_fields['mean_xy'][i].astype(np.float64)
            motion_error=predicted_motion-true_motion
            propagation.update(anchor_error_xy=anchor_error.tolist(),anchor_xy_error_m=float(np.linalg.norm(anchor_error)),
                true_motion_xy=true_motion.tolist(),motion_error_xy=motion_error.tolist(),
                motion_xy_error_m=float(np.linalg.norm(motion_error)),
                decomposition_residual_xy=(actual_error-anchor_error-motion_error).tolist())
        selected=fields['point_valid'][i].astype(bool);slots=np.flatnonzero(selected)
        indices=fields['point_indices'][i,slots]
        if np.any(indices<0) or np.any(indices>=pool_labels.shape[1]):
            raise ValueError('valid selected evidence must reference the original extension pool')
        selected_labels=pool_labels[i,indices]
        search=dict(enabled=bool(passive['diagnostic_search_enabled'][i]) if 'diagnostic_search_enabled' in passive else None,
            pool_count=int(pool_valid[i].sum()),pool_target_count=int(((pool_labels[i]>.5)&pool_valid[i]).sum()),
            selected_count=int(selected.sum()),selected_target_count=int((selected_labels>.5).sum()))
        for field in ('target_count','novel_target_count','reachable_count','search_point_count','search_target_count',
                      'search_point_count_by_partition','search_target_count_by_partition','acquired_count','range_clipped'):
            value=passive.get('diagnostic_'+field)
            search[field]=value[i].tolist() if value is not None else None
        modes=[]
        for mode in range(2):
            member=fields['members'][i,mode].astype(bool)&selected
            member_slots=np.flatnonzero(member)
            member_indices=fields['point_indices'][i,member_slots]
            foreground=int((pool_labels[i,member_indices]>.5).sum())
            modes.append(dict(formed=bool(fields['mode_valid'][i,mode]),candidate_valid=bool(valid[i,mode+2]),
                member_raw_ids=fields['point_ids'][i,member_slots].tolist(),member_count=len(member_slots),
                target_count=foreground,target_purity=foreground/len(member_slots) if len(member_slots) else None,
                seed_box_anchor_relative=seeds[i,mode].tolist(),
                decoded_exact_iou=float(target_arrays['exact_iou'][i,mode+2]) if valid[i,mode+2] else None))
        records.append(dict(schema=CANDIDATE_RECORD_SCHEMA,candidate_order=SOURCES,
            coordinate_system='world_xyz_absolute_yaw; pool_geometry_anchor_translation_relative',
            boxes_anchor_relative=boxes[i].tolist(),boxes_world=world.tolist(),valid=valid[i].tolist(),
            anchor_world=anchors[i].tolist(),target_world=target.tolist(),
            first_frame_size_lwh=sizes[i].tolist(),target_size_lwh=target_sizes[i].tolist(),
            candidates=rows,action=action,propagation=propagation,event_cv=event_cv,
            acquisition=search,modes=modes,extension=dict(raw_ids=fields['point_ids'][i,slots].tolist(),
                original_indices=indices.tolist(),point_xyz=fields['point_xyz'][i,slots].tolist(),
                vote_xyz=fields['vote_xyz'][i,slots].tolist(),target_labels=selected_labels.tolist()),
            support_pool=dict(raw_ids=ids[i].tolist(),xyz_anchor_relative=support_xyz[i].tolist(),
                probabilities=probabilities[i].tolist(),target_labels=target_arrays['support_labels'][i].tolist())))
    return records
