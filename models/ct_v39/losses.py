"""v39 监督专用入口：GT 不参与 forward、候选、路由或提交。"""
import torch
from torch.nn import functional as F

from models.ct_v38.base_losses import (local_observation_box_terms, observation_segmentation_loss,
    observation_bc_loss, masked_mean, quality_targets)
from models.ct_v38.motion import physical_motion_uncertainty_loss
from models.ct_v38.geometry import anchor_yaw
from .tensor_geometry import pair_box_metrics
from .multistep import compute_multistep_loss
from .g_losses import compute_g_losses


def base_losses(batch, output):
    """保留强 B0/B1 目标和权重；不构造或计算已删除的 acquisition 任务。"""
    target = batch['target_box']; size = batch['box_size'].to(target).clamp_min(1e-3)
    observation, prior, decoder = output.observation, output.prior, output.decoder
    result = {}; yaw = anchor_yaw(batch, target)
    for name, predicted, truth, valid, cw, aw, endpoint in (
        ('coarse', observation.coarse_box, target, observation.sequence_valid, 2., 10., False),
        ('main', decoder.q0_box, target, observation.sequence_valid, 2., 10., False),
        ('history', decoder.history_boxes, batch['history_target_boxes'], batch['history_valid'], .2, 1., True)):
        center, angle = local_observation_box_terms(predicted, truth, valid, yaw, cw, aw, endpoint_mean=endpoint)
        result['loss_'+name] = center+angle
        result['loss_'+name+'_center'], result['loss_'+name+'_angle'] = center.detach(), angle.detach()
    result['loss_seg'], result['loss_seg_current'], result['loss_seg_history'] = observation_segmentation_loss(
        observation.segmentation_logits.float(), batch['segmentation_labels'], batch['point_valid'])
    result['loss_bc'], result['loss_bc_current'], result['loss_bc_history'] = observation_bc_loss(
        observation.bc_prediction.float(), batch['bc_targets'], batch['point_valid'])
    physical = physical_motion_uncertainty_loss(prior.mean_xy, batch['physical_displacement'], prior.log_sigma,
        prior.direction_xy.detach(), batch['physical_valid'].bool() & prior.valid,
        kinematic_xy=prior.kinematic_xy.detach(), envelope_parallel_perp=prior.envelope.detach(),
        residual_unit_parallel_perp=prior.unit_residual)
    result['loss_physical'] = masked_mean(physical['mean_per_sample'], physical['valid'])
    result['loss_sigma'] = masked_mean(physical['nll_per_sample'], physical['valid'])
    with torch.no_grad():
        boxes = decoder.q0_box.detach()[:, None]
        iou = pair_box_metrics(boxes, target, size, batch.get('target_box_size'))['exact_iou']
        qt = quality_targets(boxes, target, size, target.new_ones((len(target), 1)), iou)[:, 0]
    result['loss_quality'] = F.binary_cross_entropy_with_logits(decoder.q0_quality_logits.float(), qt)
    result['loss_total'] = (result['loss_coarse']+result['loss_main']+result['loss_history']
        +.1*result['loss_seg']+result['loss_bc']+.1*result['loss_physical']+.05*result['loss_sigma']+.5*result['loss_quality'])
    result['quality_target_mean'] = qt.mean().detach()
    return result


def compute_losses(tx, output, model, config):
    if not tx.training or output.loss_inputs is None:
        raise ValueError('v39 training losses require the complete supervised B0 transaction')
    result = base_losses(output.loss_inputs, output)
    motion = compute_multistep_loss(dict(tx.inputs, **tx.targets), output.prior, model.multistep,
        weight=config.v39_m_loss_weight, horizon_weights=tuple(config.v39_m_weights))
    result.update(motion)
    weights = dict(success=config.v39_loss_success, precision=config.v39_loss_precision,
        trusted=config.v39_loss_trusted, ranking=config.v39_loss_ranking,
        matching=config.v39_loss_match, correction=config.v39_loss_correction,
        targetness=config.v39_loss_targetness, region_nomatch=config.v39_loss_nomatch,
        quality=config.v39_loss_qg)
    geometry_targets = dict(tx.targets, region_scale=config.bb_scale, region_offset=config.bb_offset)
    targets = dict(geometry_targets)
    targets['quality_candidate_mask'] = output.candidates.measurement_eligible.clone()
    targets['quality_candidate_mask'][:, 0] = False  # q0 保留原 B0 quality 监督。
    geometry = compute_g_losses(output.pool, tx.templates, output.candidates.boxes,
        tx.inputs['box_size'], output.verification, targets, refinement=output.refinements, weights=weights)
    result.update(geometry)
    auxiliary = output.auxiliary
    aux = compute_g_losses(output.pool, tx.templates, auxiliary.boxes, tx.inputs['box_size'],
        auxiliary.verification, geometry_targets, refinement=auxiliary.refinement, weights=weights)
    # 正式候选和扰动候选均按有效 endpoint 归一化；只对实际选中辅助端点计入附加监督。
    fraction = auxiliary.active.float().mean()
    result.update({'loss_aux_'+k.removeprefix('loss_g_'): value for k,value in aux.items() if k != 'loss_g'})
    result['loss_g_aux'] = aux['loss_g']*fraction
    result['loss_total'] = result['loss_total'] + motion['loss_contribution_m'] + geometry['loss_g'] + result['loss_g_aux']
    if not bool(torch.isfinite(result['loss_total'])):
        raise FloatingPointError('nonfinite v39 joint loss')
    return result
