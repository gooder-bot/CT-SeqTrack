"""独立原 SeqTrack 网络与原 loss；运行适配见 README.md。"""
from types import SimpleNamespace
import torch
from torch import nn
import torch.nn.functional as F
from ._vendor import points as points_utils
from ._vendor.pointnet import MiniPointNet, SegPointNet, FeaturePointNet
from ._vendor.attn.Models import Seq2SeqFormer
from ._vendor.misc import get_tensor_corners_batch, create_corner_timestamps
from .protocol import reference_config
from .numerics import replace_adaptive_max_pool, original_weighted_segmentation_loss

class _Base(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = reference_config(config)
    @property
    def device(self):
        return next(self.parameters()).device

class ReferenceTracker(_Base):
    enable_b1 = enable_b2 = enable_b3 = False
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        config = self.config
        self.hist_num = getattr(config, 'hist_num', 1)

        self.box_aware = getattr(config, 'box_aware', False)
        self.use_motion_cls = getattr(config, 'use_motion_cls', True)
        self.seg_pointnet = SegPointNet(input_channel=3 + 1 + 1 + (9 if self.box_aware else 0),
                                        per_point_mlp1=[64, 64, 64, 128, 1024],
                                        per_point_mlp2=[512, 256, 128, 128],
                                        output_size=2 + (9 if self.box_aware else 0))
        self.mini_pointnet = MiniPointNet(input_channel=3 + 1 + (9 if self.box_aware else 0),
                                          per_point_mlp=[64, 128, 256, 512],
                                          hidden_mlp=[512, 256],
                                          output_size=-1)

        if self.use_motion_cls:
            self.motion_state_mlp = nn.Sequential(nn.Linear(256, 128),
                                                  nn.BatchNorm1d(128),
                                                  nn.ReLU(),
                                                  nn.Linear(128, 128),
                                                  nn.BatchNorm1d(128),
                                                  nn.ReLU(),
                                                  nn.Linear(128, 2))

        self.motion_mlp = nn.Sequential(nn.Linear(256, 128),
                                        nn.BatchNorm1d(128),
                                        nn.ReLU(),
                                        nn.Linear(128, 128),
                                        nn.BatchNorm1d(128),
                                        nn.ReLU(),
                                        nn.Linear(128, 4))

        self.feature_pointnet = FeaturePointNet(
            input_channel=3 + 1 + 1 + (9 if self.box_aware else 0),
            per_point_mlp1=[64, 64, 64, 128, 1024],
            per_point_mlp2=[512, 256, 128, 128],
            output_size=128)

        self.Transformer = Seq2SeqFormer(d_word_vec=64, d_model=64, d_inner=512,
            n_layers=3, n_head=4, d_k=64, d_v=64, n_position = 1024*4)
        replace_adaptive_max_pool(self)


    def forward_original(self, input_dict):
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
        x = input_dict["points"].transpose(1, 2) 

        if self.box_aware:
            candidate_bc = input_dict["candidate_bc"].transpose(1, 2) 
            x = torch.cat([x, candidate_bc], dim=1) 

        B, _, N = x.shape
        HL =  input_dict["valid_mask"].shape[1] # Number of historical frames, default 3
        L = HL + 1 # Total length of the point cloud sequence, 1 represents the current frame
        chunk_size = N // L

        seg_out = self.seg_pointnet(x) 
        seg_logits = seg_out[:, :2, :]  # B,2,N
        pred_cls = torch.argmax(seg_logits, dim=1, keepdim=True)  # B,1,N
        mask_points = x[:, :4, :] * pred_cls 

        if self.box_aware:
            pred_bc = seg_out[:, 2:, :]
            mask_pred_bc = pred_bc * pred_cls
            mask_points = torch.cat([mask_points, mask_pred_bc], dim=1)
            output_dict['pred_bc'] = pred_bc.transpose(1, 2)

        # Coarse initial motion prediction
        point_feature = self.mini_pointnet(mask_points) #N*256
        motion_pred = self.motion_mlp(point_feature)  # B,4

        if self.use_motion_cls:
            motion_state_logits = self.motion_state_mlp(point_feature)  # B,2
            motion_mask = torch.argmax(motion_state_logits, dim=1, keepdim=True)  # B,1
            motion_pred_masked = motion_pred * motion_mask
            output_dict['motion_cls'] = motion_state_logits # B*2
        else:
            motion_pred_masked = motion_pred


        prev_boxes = torch.zeros_like(motion_pred)

        # 1st stage prediction
        aux_box = points_utils.get_offset_box_tensor(prev_boxes, motion_pred_masked)

        # Get corners of the current and historical boxes
        bbox_size = input_dict["bbox_size"] 
        bbox_size_repeated = bbox_size.repeat_interleave(L, dim=0)

        ref_boxs = input_dict["ref_boxs"]
        box_seq = torch.cat((ref_boxs, aux_box.unsqueeze(1)), dim=1) 
        box_seq = box_seq.reshape(B*L,4) 
        box_seq_corner = get_tensor_corners_batch(box_seq[:,:3],bbox_size_repeated,box_seq[:,-1])
        box_seq_corners = box_seq_corner.reshape(B,L*8,-1) # B*(L*8)*3 represents a total of L*8 points, each with 3 features
        
        # Appending timestamp features to the box corners
        corner_stamps = create_corner_timestamps(B,HL,8).to(self.device)
        box_seq_corners = torch.cat((box_seq_corners,corner_stamps),dim=-1) # B*(L*8)*4 where 4 represents features for x, y, z, and timestamp

        solo_x = x.reshape(B*L,-1,chunk_size) # Reshape into separate point clouds
        feature = self.feature_pointnet(solo_x) #(B*num) * C * N Note: N is the number of points per frame
        feature = feature.transpose(1,2) 
        NEW_N = feature.shape[1]
        points_feature = feature.reshape(B,L*NEW_N,-1)

        delta_motion = self.Transformer(box_seq_corners,points_feature,input_dict["valid_mask"])  #B*4*4

        updated_ref_boxs = delta_motion[:,:HL,:]
        updated_aux_box =  delta_motion[:,-1,:]

        
        output_dict["estimation_boxes"] = aux_box
        output_dict.update({"seg_logits": seg_logits,
                            "motion_pred": motion_pred,
                            'aux_estimation_boxes': updated_aux_box,
                            'ref_boxs': input_dict['ref_boxs'],
                            'valid_mask':input_dict["valid_mask"],
                            'updated_ref_boxs':updated_ref_boxs,
                            })

        return output_dict


    def compute_loss(self, data, output):
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
            motion_state_label = data['motion_state_label'][:,0] 
            center_label = box_label[:, :3] 
            angle_label = torch.sin(box_label[:, 3]) 
            center_label_prev = box_label_prev[:, :3] 
            angle_label_prev = torch.sin(box_label_prev[:,0,3])
            center_label_motion = motion_label[:,0,:3] 
            angle_label_motion = torch.sin(motion_label[:,0,3]) 

        
            ref_label = data['box_label_prev']
            ref_center_label = ref_label[:, :, :3] #B*hist_num*3
            ref_angle_label = torch.sin(ref_label[:,:,3]) 

        loss_seg = original_weighted_segmentation_loss(seg_logits, seg_label)
        if self.use_motion_cls:
            motion_cls = output['motion_cls']  # B,2
            loss_motion_cls = F.cross_entropy(motion_cls, motion_state_label)
            loss_total += loss_motion_cls * self.config.motion_cls_seg_weight
            loss_dict['loss_motion_cls'] = loss_motion_cls

            loss_center_motion = F.smooth_l1_loss(motion_pred[:, :3], center_label_motion, reduction='none')
            loss_center_motion = (motion_state_label * loss_center_motion.mean(dim=1)).sum() / (
                    motion_state_label.sum() + 1e-6) # Balance within a batch
            loss_angle_motion = F.smooth_l1_loss(torch.sin(motion_pred[:, 3]), angle_label_motion, reduction='none')
            loss_angle_motion = (motion_state_label * loss_angle_motion).sum() / (motion_state_label.sum() + 1e-6)
        else:
            loss_center_motion = F.smooth_l1_loss(motion_pred[:, :3], center_label_motion)
            loss_angle_motion = F.smooth_l1_loss(torch.sin(motion_pred[:, 3]), angle_label_motion)



        # ----- Stage 1 loss ---------------------
        estimation_boxes = output['estimation_boxes']  
        loss_center = F.smooth_l1_loss(estimation_boxes[:, :3], center_label)
        loss_angle = F.smooth_l1_loss(torch.sin(estimation_boxes[:, 3]), angle_label)
        loss_total += 1 * (loss_center * self.config.center_weight + loss_angle * self.config.angle_weight)
        loss_dict["loss_center"] = loss_center
        loss_dict["loss_angle"] = loss_angle
        #-----------------------------------------

        loss_center_aux = F.smooth_l1_loss(aux_estimation_boxes[:, :3], center_label)

        loss_angle_aux = F.smooth_l1_loss(torch.sin(aux_estimation_boxes[:, 3]), angle_label)


        #---------------------refbox loss---------
        loss_center_ref = F.smooth_l1_loss(updated_ref_boxs[:,:,:3],ref_center_label)
        loss_angle_ref = F.smooth_l1_loss(torch.sin(updated_ref_boxs[:, :, 3]), ref_angle_label)
        #---------------------refbox loss---------


        loss_total += loss_seg * self.config.seg_weight \
                      + 1 * (loss_center_aux * self.config.center_weight + loss_angle_aux * self.config.angle_weight) \
                      + 1 * (loss_center_motion * self.config.center_weight + loss_angle_motion * self.config.angle_weight) \
                      + 1 * (loss_center_ref * self.config.ref_center_weight + loss_angle_ref * self.config.ref_angle_weight) 

        loss_dict.update({
            "loss_total": loss_total,
            "loss_seg": loss_seg,
            "loss_center_aux": loss_center_aux,
            "loss_center_motion": loss_center_motion,
            "loss_angle_aux": loss_angle_aux,
            "loss_angle_motion": loss_angle_motion,
            "loss_center_ref": loss_center_ref,
            "loss_angle_ref": loss_angle_ref,
        })
        if self.box_aware:
            prev_bc = torch.flatten(data['prev_bc'], start_dim=1, end_dim=2)
            this_bc = data['this_bc'] #torch.Size([B, 1024, 9])
            bc_label = torch.cat([prev_bc, this_bc], dim=1) #torch.Size([B, 4096, 9])
            pred_bc = output['pred_bc'] #torch.Size([B, 4096, 9])
            loss_bc = F.smooth_l1_loss(pred_bc, bc_label)
            loss_total += loss_bc * self.config.bc_weight
            loss_dict.update({
                "loss_total": loss_total,
                "loss_bc": loss_bc
            })

        return loss_dict

    def plan_prior(self, batch):
        return None

    def compute_losses(self, batch, output):
        return self.compute_loss(batch, output)

    def forward(self, batch, prior=None):
        output = ReferenceOutput(self.forward_original(batch))
        local = output['aux_estimation_boxes']
        anchor = batch.get('reference_anchor')
        if anchor is None:
            accepted = local
        else:
            # 原 getOffsetBB 的 use_z=True / limit_box=False 数学；首帧尺寸外置。
            anchor = anchor.to(device=local.device)
            displacement = local.to(anchor)
            cosine, sine = anchor[:, 3].cos(), anchor[:, 3].sin()
            xyz = torch.stack((cosine * displacement[:, 0] - sine * displacement[:, 1],
                               sine * displacement[:, 0] + cosine * displacement[:, 1],
                               displacement[:, 2]), dim=-1) + anchor[:, :3]
            accepted = torch.cat((xyz, (anchor[:, 3] + displacement[:, 3])[:, None]), -1)
            empty = batch.get('reference_empty')
            if empty is not None:
                accepted = torch.where(empty.bool()[:, None], anchor, accepted)
        output.accepted_box = accepted
        output.selected_index = torch.zeros(len(local), dtype=torch.long, device=local.device)
        # 原模型无质量头：该占位仅供统一评测接口，不进入状态写入或 quality 指标。
        output.selected_quality = local.new_ones(len(local))
        output.evidence = SimpleNamespace(mode_valid=None)
        return output


class ReferenceOutput(dict):
    """保留原 loss 字典和统一评测需要的坐标输出。"""
