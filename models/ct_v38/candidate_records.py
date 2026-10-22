"""三候选被动台账：只有已完成的 forward/commit 才读取 GT。"""
import numpy as np
import torch

from utils.diagnostic_transfer import tensors_to_numpy
from models.ct_v37.candidate_records import _geometry_both

CANDIDATE_RECORD_SCHEMA = 'ct_seqtrack.v38.candidate_records.v1'
NAMES = ('q0', 'prior', 'trusted_propagation')


@torch.no_grad()
def candidate_records(batch, output, *, optimized=True):
    candidates, verification, commit = output.candidates, output.verification, output.commit_decision
    from .verification import current_q0_support
    q0_support = current_q0_support(output.observation, batch, candidates.boxes[:, 0])['mask']
    q0_support = q0_support & candidates.raw_valid[:, :1]
    values = dict(boxes=candidates.boxes, valid=candidates.valid, raw_valid=candidates.raw_valid,
        alias_of=candidates.alias_of, target=batch['target_box'], size=batch['box_size'],
        target_size=batch['target_box_size'], selected=output.selected_index,
        support=q0_support, ids=batch['point_ids'][:, -1],
        labels=batch['segmentation_labels'][:, -1], legacy_quality=commit.legacy_q0_quality,
        eligible=commit.eligible, prior_valid=output.prior.valid)
    if verification is not None:
        values.update(utility=verification.utility, logits=verification.logits)
    arrays = tensors_to_numpy(values, optimized=optimized)
    result = []
    for i in range(len(output.accepted_box)):
        geometry = [_geometry_both(arrays['boxes'][i, k], arrays['target'][i],
            arrays['size'][i], arrays['target_size'][i]) for k in range(3)]
        valid = arrays['valid'][i].astype(bool)
        support = arrays['support'][i].astype(bool)
        support_count = int(support.sum())
        target_count = int((support & (arrays['labels'][i] > .5)).sum())
        purity = target_count / support_count if support_count else 0.
        real_q0 = bool(arrays['raw_valid'][i, 0])
        trusted_target = bool(real_q0 and geometry[0]['exact_iou'] >= .5
                              and support_count >= 3 and purity >= .5)
        selected = int(arrays['selected'][i])
        utility = [float(arrays['utility'][i, k]) for k in range(3)] if verification is not None else None
        trusted_score = (float(torch.sigmoid(torch.tensor(float(arrays['logits'][i, 0, 2]))))
                         if verification is not None and real_q0 else None)
        true_utility = [(item['success'] + item['precision']) / 2 for item in geometry]
        oracle = max((k for k in range(3) if valid[k]), key=lambda k: true_utility[k])
        result.append(dict(schema=CANDIDATE_RECORD_SCHEMA, names=list(NAMES), candidate_order=list(NAMES),
            boxes_anchor_relative=arrays['boxes'][i].tolist(), valid=valid.tolist(),
            raw_valid=arrays['raw_valid'][i].astype(bool).tolist(), alias_of=arrays['alias_of'][i].tolist(),
            geometry=geometry, q0_geometry=geometry[0], prior_geometry=geometry[1],
            trusted_propagation_geometry=geometry[2], selected_geometry=geometry[selected],
            selected_index=selected, selected_source=NAMES[selected],
            selected_utility=utility[selected] if utility is not None else None, utility=utility,
            legacy_q0_quality=float(arrays['legacy_quality'][i]),
            trusted_score=trusted_score, trusted_target=trusted_target if real_q0 else None,
            trusted_valid=real_q0, trusted_eligible=bool(arrays['eligible'][i]),
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
