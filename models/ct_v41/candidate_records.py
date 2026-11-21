"""四候选被动台账；q0 历史口径保留，另记录选中测量与 R 资格。"""
import numpy as np
import torch

from utils.diagnostic_transfer import tensors_to_numpy
from utils.tracking_metrics import BoxMetricsTarget, LocalYawBox, metric_contributions
from .identity import (CANDIDATE_SCHEMA, JOINT_CANDIDATE_SCHEMA, JOINT_CANDIDATES,
                       JOINT_MEASUREMENTS, JOINT_PRIORITY, LEGACY_CANDIDATES)

CANDIDATE_RECORD_SCHEMA = CANDIDATE_SCHEMA
NAMES = LEGACY_CANDIDATES


def _candidate_geometry(boxes, target, size, target_size):
    truth = BoxMetricsTarget(LocalYawBox(target, target_size[[1, 0, 2]]), up_axis=(0, 0, 1))
    wlh = size[[1, 0, 2]]
    metrics = truth.local_yaw_metrics(boxes, wlh)
    exact, compat, distance = np.asarray(metrics).T
    success, precision = metric_contributions(compat, distance)
    records = []
    for k, prediction in enumerate(boxes):
        yaw = float(abs(np.arctan2(np.sin(prediction[3] - target[3]),
                                  np.cos(prediction[3] - target[3]))))
        records.append(dict(iou=float(compat[k]), exact_iou=float(exact[k]), center_error_m=float(distance[k]),
            center_xy_error_m=float(np.linalg.norm(prediction[:2] - target[:2])),
            yaw_error_rad=yaw, axis_yaw_error_rad=min(yaw, np.pi - yaw),
            success=success[k], precision=precision[k]))
    return records


def _geometry_both(prediction, target, size, target_size):
    return _candidate_geometry([prediction], target, size, target_size)[0]


@torch.no_grad()
def candidate_records(batch, output, *, optimized=True):
    candidates, verification, commit = output.candidates, output.verification, output.commit_decision
    from .verification import final_candidate_support
    joint = getattr(output, 'joint', None)
    names = JOINT_CANDIDATES if joint is not None else NAMES
    all_support = (joint.support_mask if joint is not None else
                   final_candidate_support(output.observation, batch, candidates.boxes)['mask'])
    all_support = all_support & candidates.measurement_candidate[..., None]
    values = dict(boxes=candidates.boxes, valid=candidates.valid, raw_valid=candidates.raw_valid,
        alias_of=candidates.alias_of, target=batch['target_box'], size=batch['box_size'],
        target_size=batch['target_box_size'], selected=output.selected_index,
        support=all_support, measurement=candidates.measurement_candidate,
        ids=joint.support_ids if joint is not None else batch['point_ids'][:, -1, None],
        labels=joint.target_members(batch) if joint is not None else batch['segmentation_labels'][:, -1, None],
        legacy_quality=commit.legacy_q0_quality,
        eligible=commit.eligible, prior_valid=output.prior.valid)
    if joint is not None:
        obs = joint.observations
        proposals = {grid.row: grid.proposals_sensor for grid in obs.coarse}
        values.update(identity_ok=joint.identity_ok, geometry_ok=joint.geometry_ok,
                      remote_proposed=joint.remote_proposed, remote_valid=joint.remote_valid,
                      template_valid=obs.template_valid, anchor=batch['anchor_box'],
                      identity_logits=torch.cat((obs.legacy_match_logits,
                          obs.local.match_logit[:, None], obs.remote.match_logit[:, None]), 1))
    if output.recovery is not None:
        values.update(recovery_seed=output.recovery.seed_box,
                      recovery_delta=output.recovery.normalized_delta,
                      recovery_read_mask=output.recovery.point_mask)
    if verification is not None:
        values.update(utility=verification.utility, trust_logits=verification.logits[..., 2])
    arrays = tensors_to_numpy(values, optimized=optimized)
    if verification is not None:
        arrays['trust'] = torch.as_tensor(arrays.pop('trust_logits'), dtype=torch.get_default_dtype()).sigmoid().numpy()
    result = []
    for i in range(len(output.accepted_box)):
        geometry = _candidate_geometry(arrays['boxes'][i], arrays['target'][i],
                                       arrays['size'][i], arrays['target_size'][i])
        valid = arrays['valid'][i].astype(bool)
        support = arrays['support'][i, 0].astype(bool)
        support_count = int(support.sum())
        target_count = int((support & (arrays['labels'][i, 0] > .5)).sum())
        purity = target_count / support_count if support_count else 0.
        real_q0 = bool(arrays['raw_valid'][i, 0])
        trusted_target = bool(real_q0 and geometry[0]['exact_iou'] >= .5
                              and support_count >= 3 and purity >= .5)
        selected = int(arrays['selected'][i])
        utility = arrays['utility'][i].tolist() if verification is not None else None
        trusted_score = (float(arrays['trust'][i, 0])
                         if verification is not None and real_q0 else None)
        true_utility = [(item['success'] + item['precision']) / 2 for item in geometry]
        priority = JOINT_PRIORITY if joint is not None else (0, 3, 1, 2)
        oracle = max((k for k in priority if valid[k]), key=lambda k: true_utility[k])
        measurement_records = []
        for k in (JOINT_MEASUREMENTS if joint is not None else (0, 3)):
            evidence_index = k if joint is not None else 0
            mask = arrays['support'][i, k].astype(bool)
            count = int(mask.sum())
            foreground = int((mask & (arrays['labels'][i, evidence_index] > .5)).sum())
            target_purity = foreground/count if count else 0.
            measurement_valid = bool(arrays['measurement'][i, k] and arrays['raw_valid'][i, k])
            score = (float(arrays['trust'][i, k])
                     if verification is not None and measurement_valid else None)
            target_trusted = bool(measurement_valid and geometry[k]['exact_iou'] >= .5
                                  and count >= 3 and target_purity >= .5)
            measurement_records.append(dict(index=k, name=names[k], valid=measurement_valid,
                trusted_score=score, trusted_target=target_trusted if measurement_valid else None,
                support=dict(raw_ids=arrays['ids'][i, evidence_index, mask].astype(int).tolist(), count=count,
                             target_count=foreground, target_purity=target_purity)))
            if joint is not None:
                measurement_records[-1].update(identity_ok=bool(arrays['identity_ok'][i, k]),
                    geometry_score=score, geometry_target=geometry[k]['exact_iou'] >= .5 if measurement_valid else None)
        selected_record = next((item for item in measurement_records if item['index'] == selected), None)
        selected_measurement = bool(arrays['measurement'][i, selected])
        result.append(dict(schema=JOINT_CANDIDATE_SCHEMA if joint is not None else CANDIDATE_RECORD_SCHEMA,
            names=list(names), candidate_order=list(names),
            boxes_anchor_relative=arrays['boxes'][i].tolist(), valid=valid.tolist(),
            raw_valid=arrays['raw_valid'][i].astype(bool).tolist(), alias_of=arrays['alias_of'][i].tolist(),
            measurement_candidate=arrays['measurement'][i].astype(bool).tolist(),
            geometry=geometry, q0_geometry=geometry[0], prior_geometry=geometry[1],
            trusted_propagation_geometry=geometry[2], recovery_geometry=geometry[3] if output.recovery is not None else None,
            selected_geometry=geometry[selected],
            selected_index=selected, selected_source=names[selected],
            selected_utility=utility[selected] if utility is not None else None, utility=utility,
            legacy_q0_quality=float(arrays['legacy_quality'][i]),
            trusted_score=trusted_score, trusted_target=trusted_target if real_q0 else None,
            trusted_valid=real_q0, trusted_eligible=bool(arrays['eligible'][i]),
            selected_trusted_valid=bool(selected_record is not None and selected_record['valid']),
            selected_trusted_target=(None if selected_record is None else selected_record['trusted_target']),
            selected_trusted_score=(None if selected_record is None else selected_record['trusted_score']),
            selected_is_measurement=selected_measurement, propagation_only=not selected_measurement,
            measurement_trust_records=measurement_records,
            recovery_trusted_valid=measurement_records[1]['valid'],
            recovery_trusted_target=measurement_records[1]['trusted_target'],
            recovery_trusted_score=measurement_records[1]['trusted_score'],
            recovery_seed_box_anchor_relative=(arrays['recovery_seed'][i].tolist()
                                               if 'recovery_seed' in arrays else None),
            recovery_normalized_delta=(arrays['recovery_delta'][i].tolist()
                                       if 'recovery_delta' in arrays else None),
            recovery_read_point_count=(int(arrays['recovery_read_mask'][i].sum())
                                       if 'recovery_read_mask' in arrays else 0),
            support=dict(raw_ids=arrays['ids'][i, 0, support].astype(int).tolist(), count=support_count,
                         target_count=target_count, target_purity=purity),
            q0_is_fallback=not real_q0, prior_motion_valid=bool(arrays['prior_valid'][i]),
            oracle=dict(index=oracle, success=geometry[oracle]['success'],
                        precision=geometry[oracle]['precision'], utility=true_utility[oracle]),
            mechanism=dict(replaced_real_q0=bool(real_q0 and selected != 0),
                erroneous_takeover=bool(real_q0 and selected != 0 and true_utility[selected] < true_utility[0]),
                beneficial_takeover=bool(real_q0 and selected != 0 and true_utility[selected] > true_utility[0]),
                recovered_bad_q0=bool(real_q0 and geometry[0]['exact_iou'] < .1
                                     and geometry[selected]['exact_iou'] >= .5))))
        if joint is not None:
            from .joint_grid import inside
            centers = proposals.get(i, np.empty((0, 3), np.float32))
            target_sensor = arrays['target'][i].copy()
            target_sensor[:3] += arrays['anchor'][i, :3]
            half_size = arrays['target_size'][i].copy() * .5
            half_size[2] = np.inf
            covers = inside(centers, target_sensor, half_size)
            result[-1].update(identity_ok=arrays['identity_ok'][i].tolist(),
                geometry_ok=arrays['geometry_ok'][i].tolist(), remote_proposed=bool(arrays['remote_proposed'][i]),
                remote_selectable=bool(arrays['remote_valid'][i]), template_valid=bool(arrays['template_valid'][i]),
                identity_logits={names[k]: float(v) for k, v in zip(JOINT_MEASUREMENTS, arrays['identity_logits'][i])},
                coarse_centers_sensor=centers.tolist(), coarse_hits_target_bev=covers.tolist(),
                remote_sampled_target_count=int(arrays['labels'][i, 5].sum()),
                measurement_score_target='exact_iou_at_least_half')
    return result
