"""v35 完整链的被动候选台账；只消费已经完成的前向，不参与任何决策。

公开 XYZ 均沿世界轴、仅减去 anchor 平移，yaw 已是绝对朝向。
256 个 extension 槽只导出 valid 槽及其原槽号，缺席槽仍是 padding，不能
解释为新测量。GT 标签和几何只用于离线归因，调用位于 accepted 提交之后。
"""
from __future__ import annotations

import numpy as np
import torch

from utils.tracking_metrics import LocalYawBox, box_metrics


CANDIDATE_RECORD_SCHEMA = 'ct_seqtrack.v35.candidate_records.v1'


def _array(value):
    return value.detach().cpu().numpy()


def _geometry(prediction, target, size, target_size):
    overlap, distance = box_metrics(LocalYawBox(prediction, size[[1, 0, 2]]),
        LocalYawBox(target, target_size[[1, 0, 2]]), up_axis=(0, 0, 1),
        mode='benchmark_compat', dim=3)
    yaw = float(abs(np.arctan2(np.sin(prediction[3] - target[3]),
                              np.cos(prediction[3] - target[3]))))
    return dict(iou=overlap, center_error_m=distance,
                center_xy_error_m=float(np.linalg.norm(prediction[:2] - target[:2])),
                yaw_error_rad=yaw, axis_yaw_error_rad=min(yaw, np.pi - yaw))


@torch.no_grad()
def candidate_records(batch, output):
    """每个端点一个 JSON 可序列化台账；无网络调用、RNG 或 tensor 原位操作。"""
    evidence, prior = output.evidence, output.prior
    boxes = _array(output.hypothesis_boxes)
    logits = _array(output.quality_logits)
    qualities = _array(output.quality_logits.sigmoid())
    valid = _array(output.hypothesis_valid).astype(bool)
    anchors = _array(batch['anchor_box']).astype(np.float64)
    targets, sizes = _array(batch['target_box']), _array(batch['box_size'])
    target_sizes = _array(batch['target_box_size'])
    selected = _array(output.selected_index)
    selected_quality = _array(output.selected_quality)
    sequence_valid = _array(output.observation.sequence_valid).astype(bool)
    current_valid = _array(output.observation.current_valid).astype(bool)
    prior_fields = {field: _array(getattr(prior, field)) for field in
                    ('box', 'valid', 'acquisition_fraction', 'direction_xy', 'mean_xy',
                     'log_sigma', 'kinematic_xy', 'envelope')}
    context_valid = prior.context_valid if prior.context_valid is not None else prior.valid
    prior_fields['context_valid'] = _array(context_valid)
    fields = {field: _array(getattr(evidence, field)) for field in
              ('mode_valid', 'centers_xyz', 'covariance_xy', 'members', 'point_valid',
               'point_ids', 'point_indices', 'point_xyz', 'selected_identity_logits',
               'vote_xyz', 'reliability_logits')}
    identity_support = _array(evidence.selected_identity_logits.sigmoid())
    pool_valid = _array(batch['extension_valid']).astype(bool)
    pool_partition = _array(batch['extension_partition'])
    # 所有正式 Full 批次都有监督标签；只在本被动台账中读，不馈回模型或状态。
    pool_labels = _array(batch['extension_labels'])
    search_fields = {field: _array(batch['diagnostic_' + field])
                     for field in ('search_point_count', 'search_target_count',
                                   'search_point_count_by_partition', 'search_target_count_by_partition')
                     if 'diagnostic_' + field in batch}
    records = []
    for index in range(len(boxes)):
        anchor = anchors[index]
        world_boxes = boxes[index].astype(np.float64, copy=True)
        world_boxes[:, :3] += anchor[:3]
        prior_box = prior_fields['box'][index]
        prior_world = prior_box.astype(np.float64, copy=True)
        prior_world[:3] += anchor[:3]
        point_valid = fields['point_valid'][index].astype(bool)
        slots = np.flatnonzero(point_valid)
        indices = fields['point_indices'][index, slots]
        labels = pool_labels[index, indices]
        members = fields['members'][index].astype(bool) & point_valid[None]
        modes = []
        for mode in range(3):
            member_slots = np.flatnonzero(members[mode])
            member_indices = fields['point_indices'][index, member_slots]
            target_count = int((pool_labels[index, member_indices] > 0).sum())
            count = len(member_slots)
            modes.append(dict(formed=bool(fields['mode_valid'][index, mode]),
                candidate_valid=bool(valid[index, mode + 1]),
                center_anchor_relative=fields['centers_xyz'][index, mode].tolist(),
                covariance_xy=fields['covariance_xy'][index, mode].tolist(),
                member_slots=member_slots.tolist(),
                member_raw_ids=fields['point_ids'][index, member_slots].tolist(),
                member_count=count, target_count=target_count,
                target_purity=target_count / count if count else None))
        pool_keep = pool_valid[index]
        partition = pool_partition[index]
        pool_fg = pool_labels[index] > 0
        acquisition = dict(valid_count=int(pool_keep.sum()),
            target_count=int((pool_keep & pool_fg).sum()),
            count_by_partition=[int((pool_keep & (partition == p)).sum()) for p in (0, 1)],
            target_count_by_partition=[int((pool_keep & pool_fg & (partition == p)).sum())
                                       for p in (0, 1)])
        for field in ('search_point_count', 'search_target_count',
                      'search_point_count_by_partition', 'search_target_count_by_partition'):
            value = search_fields.get(field)
            acquisition[field] = value[index].tolist() if value is not None else None
        extension = dict(capacity=int(len(point_valid)), slots=slots.tolist(),
            raw_ids=fields['point_ids'][index, slots].tolist(),
            acquisition_indices=indices.tolist(),
            xyz_anchor_relative=fields['point_xyz'][index, slots].tolist(),
            identity_logits=fields['selected_identity_logits'][index, slots].tolist(),
            identity_support=identity_support[index, slots].tolist(),
            vote_xyz_anchor_relative=fields['vote_xyz'][index, slots].tolist(),
            reliability_logits=fields['reliability_logits'][index, slots].tolist(),
            target_labels=labels.tolist(), partition=pool_partition[index, indices].tolist())
        records.append(dict(schema=CANDIDATE_RECORD_SCHEMA,
            coordinate_system='anchor_translation_relative_world_xyz_absolute_yaw_rad',
            candidate_order=['q0', 'mode0', 'mode1', 'mode2'],
            boxes_anchor_relative=boxes[index].tolist(), boxes_world=world_boxes.tolist(),
            quality_logits=logits[index].tolist(), quality=qualities[index].tolist(),
            valid=valid[index].tolist(),
            geometry=[_geometry(boxes[index, k], targets[index], sizes[index], target_sizes[index])
                      if valid[index, k] else None for k in range(4)],
            selected_index=int(selected[index]), selected_quality=float(selected_quality[index]),
            current_valid=bool(current_valid[index]), sequence_valid=bool(sequence_valid[index]),
            q0_prior_fallback=bool(not sequence_valid[index]),
            mode_formed=fields['mode_valid'][index].astype(bool).tolist(),
            prior=dict(box_anchor_relative=prior_box.tolist(), box_world=prior_world.tolist(),
                physical_valid=bool(prior_fields['valid'][index]),
                context_valid=bool(prior_fields['context_valid'][index]),
                acquisition_fraction=prior_fields['acquisition_fraction'][index].tolist(),
                direction_xy=prior_fields['direction_xy'][index].tolist(),
                mean_xy=prior_fields['mean_xy'][index].tolist(),
                log_sigma=prior_fields['log_sigma'][index].tolist(),
                kinematic_xy=prior_fields['kinematic_xy'][index].tolist(),
                envelope=prior_fields['envelope'][index].tolist()),
            acquisition=acquisition, extension=extension, modes=modes))
    return records
