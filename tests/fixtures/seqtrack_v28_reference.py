"""Frozen SeqTrack3D oracle for v28 CPU regression.

Source: local read-only seqtrack/models/seqtrack3d.py (original SeqTrack3D).
SHA256: d0bfa7805681fda53a7fb6354719c9cd911c1e29ce61051034d63f05c478f163
Only adaptation: CE class-weight .cuda() -> .to(seg_logits.device).
Forward/loss arithmetic and inherited layouts are unchanged.
"""
import torch
import torch.nn.functional as F
from datasets import points_utils
from datasets.misc_utils import get_tensor_corners_batch, create_corner_timestamps


def reference_forward(self, input_dict):
    """
        Args:
            input_dict: {
            "points": (B,N,3+1+1)
            "candidate_bc": (B,N,9)
            ['points', #[2, 4096, 5] B*(num_hist*sample)*5
            'box_label', #B*4
            'ref_boxs', #B*(num_hist)*4
            'box_label_prev', #B*(num_hist)*4
            'motion_label', #B*(num_hist)*4
            'motion_state_label', #B*(num_hist), Subtract all previous histboxes from the current box
            'bbox_size', #B*3
            'seg_label', #B*(num_hist+1)*sample
            'valid_mask', #B*(num_hist)
            'prev_bc', #B*(num_hist)*sample*9
            'this_bc', #B*sample*9
            'candidate_bc'] #B*(num_hist*sample)*9

        }

        Returns: B,4

        """
    output_dict = {}
    x = input_dict['points'].transpose(1, 2)
    if self.box_aware:
        candidate_bc = input_dict['candidate_bc'].transpose(1, 2)
        x = torch.cat([x, candidate_bc], dim=1)
    B, _, N = x.shape
    HL = input_dict['valid_mask'].shape[1]
    L = HL + 1
    chunk_size = N // L
    seg_out = self.seg_pointnet(x)
    seg_logits = seg_out[:, :2, :]
    pred_cls = torch.argmax(seg_logits, dim=1, keepdim=True)
    mask_points = x[:, :4, :] * pred_cls
    if self.box_aware:
        pred_bc = seg_out[:, 2:, :]
        mask_pred_bc = pred_bc * pred_cls
        mask_points = torch.cat([mask_points, mask_pred_bc], dim=1)
        output_dict['pred_bc'] = pred_bc.transpose(1, 2)
    point_feature = self.mini_pointnet(mask_points)
    motion_pred = self.motion_mlp(point_feature)
    if self.use_motion_cls:
        motion_state_logits = self.motion_state_mlp(point_feature)
        motion_mask = torch.argmax(motion_state_logits, dim=1, keepdim=True)
        motion_pred_masked = motion_pred * motion_mask
        output_dict['motion_cls'] = motion_state_logits
    else:
        motion_pred_masked = motion_pred
    prev_boxes = torch.zeros_like(motion_pred)
    aux_box = points_utils.get_offset_box_tensor(prev_boxes, motion_pred_masked)
    bbox_size = input_dict['bbox_size']
    bbox_size_repeated = bbox_size.repeat_interleave(L, dim=0)
    ref_boxs = input_dict['ref_boxs']
    box_seq = torch.cat((ref_boxs, aux_box.unsqueeze(1)), dim=1)
    box_seq = box_seq.reshape(B * L, 4)
    box_seq_corner = get_tensor_corners_batch(box_seq[:, :3], bbox_size_repeated, box_seq[:, -1])
    box_seq_corners = box_seq_corner.reshape(B, L * 8, -1)
    corner_stamps = create_corner_timestamps(B, HL, 8).to(self.device)
    box_seq_corners = torch.cat((box_seq_corners, corner_stamps), dim=-1)
    solo_x = x.reshape(B * L, -1, chunk_size)
    feature = self.feature_pointnet(solo_x)
    feature = feature.transpose(1, 2)
    NEW_N = feature.shape[1]
    points_feature = feature.reshape(B, L * NEW_N, -1)
    delta_motion = self.Transformer(box_seq_corners, points_feature, input_dict['valid_mask'])
    updated_ref_boxs = delta_motion[:, :HL, :]
    updated_aux_box = delta_motion[:, -1, :]
    output_dict['estimation_boxes'] = aux_box
    output_dict.update({'seg_logits': seg_logits, 'motion_pred': motion_pred, 'aux_estimation_boxes': updated_aux_box, 'ref_boxs': input_dict['ref_boxs'], 'valid_mask': input_dict['valid_mask'], 'updated_ref_boxs': updated_ref_boxs})
    return output_dict


def reference_compute_loss(self, data, output):
    loss_total = 0.0
    loss_dict = {}
    aux_estimation_boxes = output['aux_estimation_boxes']
    motion_pred = output['motion_pred']
    seg_logits = output['seg_logits']
    updated_ref_boxs = output['updated_ref_boxs']
    with torch.no_grad():
        seg_label = data['seg_label']
        box_label = data['box_label']
        box_label_prev = data['box_label_prev']
        motion_label = data['motion_label']
        motion_state_label = data['motion_state_label'][:, 0]
        center_label = box_label[:, :3]
        angle_label = torch.sin(box_label[:, 3])
        center_label_prev = box_label_prev[:, :3]
        angle_label_prev = torch.sin(box_label_prev[:, 0, 3])
        center_label_motion = motion_label[:, 0, :3]
        angle_label_motion = torch.sin(motion_label[:, 0, 3])
        ref_label = data['box_label_prev']
        ref_center_label = ref_label[:, :, :3]
        ref_angle_label = torch.sin(ref_label[:, :, 3])
    loss_seg = F.cross_entropy(seg_logits, seg_label, weight=torch.tensor([0.5, 2.0]).to(seg_logits.device))
    if self.use_motion_cls:
        motion_cls = output['motion_cls']
        loss_motion_cls = F.cross_entropy(motion_cls, motion_state_label)
        loss_total += loss_motion_cls * self.config.motion_cls_seg_weight
        loss_dict['loss_motion_cls'] = loss_motion_cls
        loss_center_motion = F.smooth_l1_loss(motion_pred[:, :3], center_label_motion, reduction='none')
        loss_center_motion = (motion_state_label * loss_center_motion.mean(dim=1)).sum() / (motion_state_label.sum() + 1e-06)
        loss_angle_motion = F.smooth_l1_loss(torch.sin(motion_pred[:, 3]), angle_label_motion, reduction='none')
        loss_angle_motion = (motion_state_label * loss_angle_motion).sum() / (motion_state_label.sum() + 1e-06)
    else:
        loss_center_motion = F.smooth_l1_loss(motion_pred[:, :3], center_label_motion)
        loss_angle_motion = F.smooth_l1_loss(torch.sin(motion_pred[:, 3]), angle_label_motion)
    estimation_boxes = output['estimation_boxes']
    loss_center = F.smooth_l1_loss(estimation_boxes[:, :3], center_label)
    loss_angle = F.smooth_l1_loss(torch.sin(estimation_boxes[:, 3]), angle_label)
    loss_total += 1 * (loss_center * self.config.center_weight + loss_angle * self.config.angle_weight)
    loss_dict['loss_center'] = loss_center
    loss_dict['loss_angle'] = loss_angle
    loss_center_aux = F.smooth_l1_loss(aux_estimation_boxes[:, :3], center_label)
    loss_angle_aux = F.smooth_l1_loss(torch.sin(aux_estimation_boxes[:, 3]), angle_label)
    loss_center_ref = F.smooth_l1_loss(updated_ref_boxs[:, :, :3], ref_center_label)
    loss_angle_ref = F.smooth_l1_loss(torch.sin(updated_ref_boxs[:, :, 3]), ref_angle_label)
    loss_total += loss_seg * self.config.seg_weight + 1 * (loss_center_aux * self.config.center_weight + loss_angle_aux * self.config.angle_weight) + 1 * (loss_center_motion * self.config.center_weight + loss_angle_motion * self.config.angle_weight) + 1 * (loss_center_ref * self.config.ref_center_weight + loss_angle_ref * self.config.ref_angle_weight)
    loss_dict.update({'loss_total': loss_total, 'loss_seg': loss_seg, 'loss_center_aux': loss_center_aux, 'loss_center_motion': loss_center_motion, 'loss_angle_aux': loss_angle_aux, 'loss_angle_motion': loss_angle_motion, 'loss_center_ref': loss_center_ref, 'loss_angle_ref': loss_angle_ref})
    if self.box_aware:
        prev_bc = torch.flatten(data['prev_bc'], start_dim=1, end_dim=2)
        this_bc = data['this_bc']
        bc_label = torch.cat([prev_bc, this_bc], dim=1)
        pred_bc = output['pred_bc']
        loss_bc = F.smooth_l1_loss(pred_bc, bc_label)
        loss_total += loss_bc * self.config.bc_weight
        loss_dict.update({'loss_total': loss_total, 'loss_bc': loss_bc})
    return loss_dict
