"""四候选被动台账；q0 历史口径保留，另记录选中测量与 R 资格。"""
import numpy as np
import torch

from utils.diagnostic_transfer import tensors_to_numpy
from utils.tracking_metrics import LocalYawBox, box_metrics, metric_contributions
from .identity import CANDIDATE_SCHEMA

CANDIDATE_RECORD_SCHEMA = CANDIDATE_SCHEMA
NAMES = ('q0', 'prior', 'trusted_propagation', 'recovery_measurement')


def _geometry(prediction, target, size, target_size):
    overlap, distance = box_metrics(LocalYawBox(prediction, size[[1, 0, 2]]),
        LocalYawBox(target, target_size[[1, 0, 2]]), up_axis=(0, 0, 1),
        mode='benchmark_compat', dim=3)
    yaw = float(abs(np.arctan2(np.sin(prediction[3] - target[3]),
                              np.cos(prediction[3] - target[3]))))
    return dict(iou=overlap, center_error_m=distance,
                center_xy_error_m=float(np.linalg.norm(prediction[:2] - target[:2])),
                yaw_error_rad=yaw, axis_yaw_error_rad=min(yaw, np.pi - yaw))

def _geometry_both(box, target, size, target_size):
    result = _geometry(box, target, size, target_size)
    result['exact_iou'] = box_metrics(LocalYawBox(box, size[[1, 0, 2]]),
        LocalYawBox(target, target_size[[1, 0, 2]]), up_axis=(0, 0, 1), mode='geometry_exact', dim=3)[0]
    result['success'], result['precision'] = metric_contributions(result['iou'], result['center_error_m'])
    return result


@torch.no_grad()
def candidate_records(batch, output, *, optimized=True):
    candidates, verification, commit = output.candidates, output.verification, output.commit_decision
    from .verification import current_candidate_support
    all_support = current_candidate_support(output.observation, batch, candidates.boxes)['mask']
    all_support = all_support & candidates.measurement_candidate[..., None]
    values = dict(boxes=candidates.boxes, valid=candidates.valid, raw_valid=candidates.raw_valid,
        alias_of=candidates.alias_of, target=batch['target_box'], size=batch['box_size'],
        target_size=batch['target_box_size'], selected=output.selected_index,
        support=all_support, measurement=candidates.measurement_candidate,
        ids=batch['point_ids'][:, -1],
        labels=batch['segmentation_labels'][:, -1], legacy_quality=commit.legacy_q0_quality,
        eligible=commit.eligible, prior_valid=output.prior.valid)
    if output.recovery is not None:
        values.update(recovery_seed=output.recovery.seed_box,
                      recovery_delta=output.recovery.normalized_delta,
                      recovery_read_mask=output.recovery.point_mask)
    if verification is not None:
        values.update(utility=verification.utility, logits=verification.logits)
    arrays = tensors_to_numpy(values, optimized=optimized)
    result = []
    for i in range(len(output.accepted_box)):
        geometry = [_geometry_both(arrays['boxes'][i, k], arrays['target'][i],
            arrays['size'][i], arrays['target_size'][i]) for k in range(len(NAMES))]
        valid = arrays['valid'][i].astype(bool)
        support = arrays['support'][i, 0].astype(bool)
        support_count = int(support.sum())
        target_count = int((support & (arrays['labels'][i] > .5)).sum())
        purity = target_count / support_count if support_count else 0.
        real_q0 = bool(arrays['raw_valid'][i, 0])
        trusted_target = bool(real_q0 and geometry[0]['exact_iou'] >= .5
                              and support_count >= 3 and purity >= .5)
        selected = int(arrays['selected'][i])
        utility = [float(arrays['utility'][i, k]) for k in range(len(NAMES))] if verification is not None else None
        trusted_score = (float(torch.sigmoid(torch.tensor(float(arrays['logits'][i, 0, 2]))))
                         if verification is not None and real_q0 else None)
        true_utility = [(item['success'] + item['precision']) / 2 for item in geometry]
        oracle = max((k for k in (0, 3, 1, 2) if valid[k]), key=lambda k: true_utility[k])
        measurement_records = []
        for k in (0, 3):
            mask = arrays['support'][i, k].astype(bool)
            count = int(mask.sum())
            foreground = int((mask & (arrays['labels'][i] > .5)).sum())
            target_purity = foreground/count if count else 0.
            measurement_valid = bool(arrays['measurement'][i, k] and arrays['raw_valid'][i, k])
            score = (float(torch.sigmoid(torch.tensor(float(arrays['logits'][i, k, 2]))))
                     if verification is not None and measurement_valid else None)
            target_trusted = bool(measurement_valid and geometry[k]['exact_iou'] >= .5
                                  and count >= 3 and target_purity >= .5)
            measurement_records.append(dict(index=k, name=NAMES[k], valid=measurement_valid,
                trusted_score=score, trusted_target=target_trusted if measurement_valid else None,
                support=dict(raw_ids=arrays['ids'][i, mask].astype(int).tolist(), count=count,
                             target_count=foreground, target_purity=target_purity)))
        selected_record = next((item for item in measurement_records if item['index'] == selected), None)
        selected_measurement = bool(arrays['measurement'][i, selected])
        result.append(dict(schema=CANDIDATE_RECORD_SCHEMA, names=list(NAMES), candidate_order=list(NAMES),
            boxes_anchor_relative=arrays['boxes'][i].tolist(), valid=valid.tolist(),
            raw_valid=arrays['raw_valid'][i].astype(bool).tolist(), alias_of=arrays['alias_of'][i].tolist(),
            measurement_candidate=arrays['measurement'][i].astype(bool).tolist(),
            geometry=geometry, q0_geometry=geometry[0], prior_geometry=geometry[1],
            trusted_propagation_geometry=geometry[2], recovery_geometry=geometry[3], selected_geometry=geometry[selected],
            selected_index=selected, selected_source=NAMES[selected],
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
            support=dict(raw_ids=arrays['ids'][i, support].astype(int).tolist(), count=support_count,
                         target_count=target_count, target_purity=purity),
            q0_is_fallback=not real_q0, prior_motion_valid=bool(arrays['prior_valid'][i]),
            oracle=dict(index=oracle, success=geometry[oracle]['success'],
                        precision=geometry[oracle]['precision'], utility=true_utility[oracle]),
            mechanism=dict(replaced_real_q0=bool(real_q0 and selected != 0),
                erroneous_takeover=bool(real_q0 and selected != 0 and true_utility[selected] < true_utility[0]),
                beneficial_takeover=bool(real_q0 and selected != 0 and true_utility[selected] > true_utility[0]),
                recovered_bad_q0=bool(real_q0 and geometry[0]['exact_iou'] < .1
                                     and geometry[selected]['exact_iou'] >= .5))))
    return result
