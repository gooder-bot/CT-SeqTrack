"""v37 完整候选与权限的被动台账；GT 只能在损失或提交后审计中使用。"""
import numpy as np
import torch

from models.ct_v31.candidate_records import _geometry
from utils.diagnostic_transfer import tensors_to_numpy
from utils.tracking_metrics import LocalYawBox, box_metrics, metric_contributions

CANDIDATE_RECORD_SCHEMA = 'ct_seqtrack.v37.candidate_records.v1'


def _array(value):
    return value.detach().cpu().numpy()


def _json(value):
    if torch.is_tensor(value):
        value = _array(value)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    if isinstance(value, dict):
        return {key: _json(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        value = value.item()
    return None if isinstance(value, float) and not np.isfinite(value) else value


def _geometry_both(box, target, size, target_size):
    result = _geometry(box, target, size, target_size)
    result['exact_iou'] = box_metrics(LocalYawBox(box, size[[1, 0, 2]]),
        LocalYawBox(target, target_size[[1, 0, 2]]), up_axis=(0, 0, 1), mode='geometry_exact', dim=3)[0]
    result['success'], result['precision'] = metric_contributions(result['iou'], result['center_error_m'])
    return result


@torch.no_grad()
def candidate_records(batch, output, *, optimized=True):
    candidates = getattr(output, 'candidates', None)
    verification = getattr(output, 'verification', None)
    evidence, prior = output.evidence, output.prior
    count = len(output.accepted_box)
    if candidates is None:
        # Baseline 未执行候选竞争；prior 几何仍被动导出，不伪造分数。
        boxes = torch.cat((output.hypothesis_boxes[:, :1], prior.box[:, None],
                           output.hypothesis_boxes[:, 1:3]), 1)
        valid = torch.zeros((count, 4), dtype=torch.bool, device=boxes.device)
        valid[:, :2] = True
    else:
        boxes, valid = candidates.boxes, candidates.valid
    # 每个 dtype/device 合并一次传输；下方每候选循环只处理 CPU 数组。
    tensors = dict(boxes=boxes, valid=valid, anchors=batch['anchor_box'], targets=batch['target_box'],
        sizes=batch['box_size'], target_sizes=batch['target_box_size'], selected=output.selected_index,
        quality=output.selected_quality, accepted=output.accepted_box,
        fallback=getattr(output, 'q0_is_fallback', ~output.observation.sequence_valid),
        pool_labels=batch['extension_labels'], pool_valid=batch['extension_valid'],
        pool_parts=batch['extension_partition'], segmentation_labels=batch['segmentation_labels'][:, -1],
        coarse=output.observation.coarse_box)
    evidence_names = (
        'point_ids', 'point_indices', 'point_valid', 'point_xyz', 'vote_xyz',
        'selected_identity_logits', 'reliability_logits', 'members', 'mode_valid', 'centers_xyz', 'covariance_xy')
    tensors.update({'evidence_' + name: getattr(evidence, name) for name in evidence_names})
    support_names = []
    if verification is not None:
        for name in ('logits', 'utility', 'support_mask', 'authorized', 'support_ids',
                     'support_xyz', 'support_probabilities', 'commit_support_mask', 'support_source',
                     'metadata', 'local_neighbor_count', 'local_selected_count', 'local_mean_support', 'local_delta_norm'):
            value = getattr(verification, name, None)
            if value is not None:
                support_names.append(name)
                tensors['support_' + name] = value
    for name, value in (('commit_quality', getattr(output, 'commit_quality', None)),
                        ('seed_boxes', getattr(candidates, 'seed_boxes', None))):
        if value is not None:
            tensors[name] = value
    prior_names = []
    for name in ('box', 'valid', 'context_valid', 'mean_xy', 'kinematic_xy', 'envelope',
                 'log_sigma', 'acquisition_fraction', 'direction_xy', 'feature'):
        value = getattr(prior, name, None)
        if value is not None:
            prior_names.append(name)
            tensors['prior_' + name] = value
    passive_names = []
    for owner_name, owner in (('decoder', output.decoder), ('candidates', candidates)):
        if owner is None:
            continue
        for name in ('raw_pose', 'pose_raw', 'pose_output', 'mode_residual', 'q0_pose_raw',
                     'physical_context', 'physical_context_norm', 'query_context_norm',
                     'local_neighbor_count', 'local_selected_count', 'local_mean_support', 'local_delta_norm'):
            value = getattr(owner, name, None)
            if value is not None:
                key = owner_name + '_' + name
                passive_names.append(key)
                tensors['passive_' + key] = value
        physical = getattr(owner, 'physical_context', None)
        if physical is not None and getattr(owner, 'physical_context_norm', None) is None:
            key = owner_name + '_physical_context_norm'
            passive_names.append(key)
            tensors['passive_' + key] = physical.detach().norm(dim=-1)
    acquisition_names = ('search_point_count', 'search_target_count', 'search_point_count_by_partition',
                         'search_target_count_by_partition')
    for name in acquisition_names:
        if batch.get('diagnostic_' + name) is not None:
            tensors['acquisition_' + name] = batch['diagnostic_' + name]
    arrays = tensors_to_numpy(tensors, optimized=optimized)
    boxes, valid = arrays['boxes'], arrays['valid'].astype(bool)
    anchors, targets, sizes, target_sizes = (arrays[name] for name in ('anchors', 'targets', 'sizes', 'target_sizes'))
    selected, quality, accepted, fallback = (arrays[name] for name in ('selected', 'quality', 'accepted', 'fallback'))
    selected, fallback = selected.astype(int), fallback.astype(bool)
    fields = {name: arrays['evidence_' + name] for name in evidence_names}
    pool_labels, pool_valid, pool_parts = (arrays[name] for name in ('pool_labels', 'pool_valid', 'pool_parts'))
    pool_valid = pool_valid.astype(bool)
    support_fields = {name: arrays['support_' + name] for name in support_names}
    commit_quality, seed_boxes = arrays.get('commit_quality'), arrays.get('seed_boxes')
    records = []
    for i in range(count):
        geometry = [_geometry_both(boxes[i, k], targets[i], sizes[i], target_sizes[i])
                    if valid[i, k] else None for k in range(4)]
        q0_geometry = _geometry_both(boxes[i, 0], targets[i], sizes[i], target_sizes[i])
        prior_geometry = _geometry_both(boxes[i, 1], targets[i], sizes[i], target_sizes[i])
        legal = [k for k in range(4) if geometry[k] is not None]
        oracle = max(legal, key=lambda k: (geometry[k]['success'], geometry[k]['precision']))
        ext_keep = fields['point_valid'][i].astype(bool)
        slots = np.flatnonzero(ext_keep)
        indices = fields['point_indices'][i, slots].astype(int)
        if len(indices) and ((indices < 0).any() or (indices >= pool_labels.shape[1]).any()):
            raise ValueError('candidate ledger contains invalid acquisition indices')
        labels = pool_labels[i, indices]
        all_labels = np.zeros(len(ext_keep), dtype=np.int64)
        all_labels[slots] = labels
        modes = []
        for mode in range(2):
            members = fields['members'][i, mode].astype(bool) & ext_keep
            member_slots = np.flatnonzero(members)
            foreground = int((all_labels[member_slots] > .5).sum())
            seed = seed_boxes[i, mode] if seed_boxes is not None else np.r_[fields['centers_xyz'][i, mode], boxes[i, 0, 3]]
            modes.append(dict(formed=bool(fields['mode_valid'][i, mode]),
                candidate_valid=bool(valid[i, mode + 2]), member_slots=member_slots.tolist(),
                member_raw_ids=fields['point_ids'][i, member_slots].tolist(),
                member_count=len(member_slots), target_count=foreground,
                target_purity=foreground / len(member_slots) if len(member_slots) else None,
                qualifying_target_mode=bool(fields['mode_valid'][i, mode] and foreground >= 1
                                           and foreground >= .5 * len(member_slots)),
                seed_box_anchor_relative=seed.tolist(),
                seed_geometry=_geometry_both(seed, targets[i], sizes[i], target_sizes[i])
                    if fields['mode_valid'][i, mode] else None,
                covariance_xy=fields['covariance_xy'][i, mode].tolist()))
        support, commit_support = [], []
        support_pool = None
        if verification is not None:
            labels_for_support = np.r_[arrays['segmentation_labels'][i], all_labels]
            support_pool = dict(raw_ids=support_fields['support_ids'][i].tolist(),
                xyz_anchor_relative=support_fields['support_xyz'][i].tolist(),
                probabilities=support_fields['support_probabilities'][i].tolist(),
                target_labels=labels_for_support.tolist(),
                current_b0_count=int(batch['point_valid'].shape[-1]),
                source=(support_fields['support_source'][i].tolist() if 'support_source' in support_fields else
                        [0] * int(batch['point_valid'].shape[-1]) + [1] * len(ext_keep)))
            for k in range(4):
                mask = support_fields['support_mask'][i, k].astype(bool)
                fg = int((mask & (labels_for_support > .5)).sum())
                n = int(mask.sum())
                support.append(dict(raw_ids=support_fields['support_ids'][i, mask].tolist(),
                    count=n, target_count=fg, target_purity=fg / n if n else None,
                    authorized_raw_ids=support_fields['support_ids'][i,
                        support_fields['authorized'][i, k].astype(bool)].tolist(),
                    measurement_correct=bool(k != 1 and not (k == 0 and fallback[i])
                        and geometry[k] is not None and geometry[k]['exact_iou'] >= .5
                        and fg >= 1 and fg >= .5 * n)))
                commit_mask = support_fields.get('commit_support_mask')
                if commit_mask is not None:
                    mask = commit_mask[i, k].astype(bool)
                    n, fg = int(mask.sum()), int((mask & (labels_for_support > .5)).sum())
                    commit_support.append(dict(raw_ids=support_fields['support_ids'][i, mask].tolist(),
                        count=n, target_count=fg, target_purity=fg / n if n else None,
                        measurement_correct=bool(k != 1 and not (k == 0 and fallback[i])
                            and geometry[k] is not None and geometry[k]['exact_iou'] >= .5
                            and fg >= 1 and fg >= .5 * n)))
        good_mode = [k for k in (2, 3) if geometry[k] is not None and geometry[k]['exact_iou'] >= .5]
        bad_q0 = q0_geometry['exact_iou'] < .1
        damaged = improved = preserved = good_seed = 0
        for k, mode in enumerate(modes, start=2):
            seed_geometry, final = mode['seed_geometry'], geometry[k]
            if seed_geometry is not None and final is not None:
                damaged += seed_geometry['exact_iou'] >= .5 and final['exact_iou'] < .5
                improved += seed_geometry['exact_iou'] < .5 and final['exact_iou'] >= .5
                preserved += seed_geometry['exact_iou'] >= .5 and final['exact_iou'] >= .5
            good_seed += bool(seed_geometry is not None and seed_geometry['exact_iou'] >= .5)
        world = boxes[i].astype(np.float64).copy()
        world[:, :3] += anchors[i, :3]
        prior_fields = {name: _json(arrays['prior_' + name][i]) for name in prior_names}
        passive = {name: _json(arrays['passive_' + name][i]) for name in passive_names}
        passive['coarse_box_anchor_relative'] = arrays['coarse'][i].tolist()
        passive['coarse_geometry'] = _geometry_both(arrays['coarse'][i], targets[i], sizes[i], target_sizes[i])
        passive['local_reader'] = dict(coordinate_scale='first_frame_lwh', size_lwh=sizes[i].tolist(),
            radius_normalized=1., max_neighbors_per_corner=16, corners=8,
            decoder_candidate_order=['q0', 'mode0', 'mode1', 'invalid'])
        for name in ('metadata', 'local_neighbor_count', 'local_selected_count', 'local_mean_support', 'local_delta_norm'):
            if name in support_fields:
                passive['verification_' + name] = _json(support_fields[name][i])
        acquisition = dict(valid_count=int(pool_valid[i].sum()),
            target_count=int((pool_valid[i] & (pool_labels[i] > .5)).sum()))
        for name in acquisition_names:
            value = arrays.get('acquisition_' + name)
            acquisition[name] = _json(value[i]) if value is not None else None
        record = dict(schema=CANDIDATE_RECORD_SCHEMA,
            coordinate_system='anchor_translation_relative_world_xyz_absolute_yaw_rad',
            candidate_order=['q0', 'prior', 'mode0', 'mode1'],
            anchor_world=anchors[i].tolist(), first_frame_size_lwh=sizes[i].tolist(),
            target_size_lwh=target_sizes[i].tolist(),
            boxes_anchor_relative=boxes[i].tolist(), boxes_world=world.tolist(), valid=valid[i].tolist(),
            geometry=geometry, q0_geometry=q0_geometry, prior_geometry=prior_geometry,
            q0_semantics='propagation_alias' if fallback[i] else 'observation',
            prior=prior_fields, modes=modes, acquisition=acquisition,
            extension=dict(capacity=len(ext_keep), slots=slots.tolist(), raw_ids=fields['point_ids'][i, slots].tolist(),
                acquisition_indices=indices.tolist(), xyz_anchor_relative=fields['point_xyz'][i, slots].tolist(),
                vote_xyz_anchor_relative=fields['vote_xyz'][i, slots].tolist(),
                identity_logits=fields['selected_identity_logits'][i, slots].tolist(),
                reliability_logits=fields['reliability_logits'][i, slots].tolist(),
                target_labels=labels.tolist(), partition=pool_parts[i, indices].tolist()),
            final_support=support, commit_support=commit_support, support_pool=support_pool,
            support_semantics=dict(final_support='scorer_own_box_foreground',
                                   commit_support='commit_policy_own_box_foreground_before_quality_gate'),
            selected_index=int(selected[i]), selected_commit_quality=float(quality[i]),
            selected_geometry=_geometry_both(accepted[i], targets[i], sizes[i], target_sizes[i]),
            selected_utility=(float(support_fields['utility'][i, selected[i]]) if 'utility' in support_fields else None),
            utility=support_fields['utility'][i].tolist() if 'utility' in support_fields else None,
            verification_logits=support_fields['logits'][i].tolist() if 'logits' in support_fields else None,
            commit_quality=commit_quality[i].tolist() if commit_quality is not None else None,
            q0_is_fallback=bool(fallback[i]), passive=passive,
            oracle=dict(index=oracle, success=geometry[oracle]['success'],
                precision_paired=geometry[oracle]['precision'], interpretation='same_state_one_step_only'),
            mechanism=dict(good_mode_exists=bool(good_mode), bad_q0_good_mode=bool(bad_q0 and good_mode),
                recovery_opportunity=bool(bad_q0 and acquisition['target_count'] > 0),
                qualifying_mode_count=sum(mode['qualifying_target_mode'] for mode in modes),
                good_seed_count=int(good_seed), seed_good_refinement_good=int(preserved),
                good_mode_selected=int(selected[i]) in good_mode,
                seed_good_refinement_bad=int(damaged), seed_bad_refinement_good=int(improved)))
        records.append(_json(record))
    return records
