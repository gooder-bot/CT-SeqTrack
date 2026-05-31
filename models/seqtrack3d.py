from datasets import points_utils
from models import base_model
from models.backbone.pointnet import MiniPointNet, SegPointNet, FeaturePointNet
from models.attn.Models import Seq2SeqFormer

import torch
from torch import nn
import torch.nn.functional as F

from torchmetrics import Accuracy

from datasets.misc_utils import get_tensor_corners_batch
from datasets.misc_utils import create_corner_timestamps_from_deltas
from models.dynamics import DynamicsEncoder
from models.observability import ObservabilityGate
from models.time_encoding import TimeEncoding

# import vis_tool as vt

class SEQTRACK3D(base_model.MotionBaseModelMF):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.hist_num = getattr(config, 'hist_num', 1)
        self.seg_acc = Accuracy(task='multiclass',num_classes=2, average='none')

        self.box_aware = getattr(config, 'box_aware', False)
        self.use_motion_cls = getattr(config, 'use_motion_cls', True)
        self.use_dynamics_encoder = getattr(config, 'use_dynamics_encoder', False)
        self.use_observability_gate = getattr(config, 'use_observability_gate', False)
        self.dynamics_motion_mode = str(getattr(config, 'dynamics_motion_mode', 'feature')).lower()
        if self.dynamics_motion_mode not in ('feature', 'residual'):
            raise ValueError("dynamics_motion_mode must be 'feature' or 'residual'.")
        if self.use_observability_gate and not self.use_dynamics_encoder:
            raise ValueError("use_observability_gate=True requires use_dynamics_encoder=True.")
        self.dynamics_hidden_dim = int(getattr(config, 'dynamics_hidden_dim', 128))
        default_time_scale = getattr(config, 'default_time_step', getattr(config, 'time_step', 0.5))
        self.time_encoder = TimeEncoding(
            mode=getattr(config, 'time_encoding', 'raw'),
            scale=getattr(config, 'time_scale', default_time_scale),
            clip=getattr(config, 'time_clip', 4.0),
            fourier_bands=getattr(config, 'time_fourier_bands', 4),
            hidden_dim=getattr(config, 'time_hidden_dim', 16),
            output_scale=getattr(config, 'time_output_scale', getattr(config, 'pseudo_time_step', 0.1)),
        )
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
            self.motion_acc = Accuracy(task='multiclass',num_classes=2, average='none')

        motion_feature_dim = 256
        if self.use_dynamics_encoder:
            self.dynamics_encoder = DynamicsEncoder(
                hidden_dim=self.dynamics_hidden_dim,
                eps=getattr(config, 'dynamics_eps', 1e-3),
                use_query_gap=getattr(config, 'dynamics_use_query_gap', True),
            )
            if self.use_observability_gate:
                self.observability_gate = ObservabilityGate(
                    feature_dim=256,
                    dynamics_dim=self.dynamics_hidden_dim,
                    stats_dim=getattr(config, 'obs_gate_num_stats', 5),
                    hidden_dim=getattr(config, 'obs_gate_hidden_dim', 64),
                    init_obs_bias=getattr(config, 'obs_gate_init_obs_bias', 1.0),
                    min_dyn_valid=getattr(config, 'obs_gate_min_dyn_valid', 0.5),
                )
            else:
                motion_feature_dim += self.dynamics_hidden_dim

        self.motion_mlp = nn.Sequential(nn.Linear(motion_feature_dim, 128),
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

    def encode_point_time(self, points):
        encoded_time = self.time_encoder(points[..., 3:4])
        return torch.cat((points[..., :3], encoded_time, points[..., 4:]), dim=-1)

    @staticmethod
    def is_paired_batch(batch):
        return isinstance(batch, dict) and "view_a" in batch and "view_b" in batch

    def build_observability_stats(self, input_dict, seg_logits, chunk_size):
        B = seg_logits.shape[0]
        device = seg_logits.device
        dtype = seg_logits.dtype

        if "num_points_in_search" in input_dict:
            num_points = input_dict["num_points_in_search"]
            if not torch.is_tensor(num_points):
                num_points = torch.as_tensor(num_points, device=device, dtype=dtype)
            num_points = num_points.to(device=device, dtype=dtype).reshape(B)
        else:
            num_points = seg_logits.new_full((B,), float(chunk_size))

        current_logits = seg_logits[:, :, -chunk_size:]
        fg_prob = torch.softmax(current_logits, dim=1)[:, 1, :]
        if getattr(self.config, "obs_stats_detach_seg", True):
            fg_prob = fg_prob.detach()
        soft_fg_count = fg_prob.sum(dim=1)
        mean_fg_score = fg_prob.mean(dim=1)
        estimated_fg_points = mean_fg_score * torch.clamp(num_points, min=0.0)

        valid_history_ratio = input_dict["valid_mask"].to(device=device, dtype=dtype).mean(dim=1)
        default_time_scale = getattr(self.config, 'default_time_step', getattr(self.config, 'time_step', 0.5))
        time_scale = max(float(getattr(self.config, "time_scale", default_time_scale)), 1e-6)
        if "current_delta_t" in input_dict:
            current_delta_t = input_dict["current_delta_t"].to(device=device, dtype=dtype).reshape(B)
        else:
            current_delta_t = seg_logits.new_full((B,), float(default_time_scale))
        current_delta_t_ratio = current_delta_t / time_scale

        obs_stats = torch.stack((
            torch.log1p(torch.clamp(num_points, min=0.0)),
            torch.log1p(torch.clamp(estimated_fg_points, min=0.0)),
            mean_fg_score,
            valid_history_ratio,
            current_delta_t_ratio,
        ), dim=1)
        obs_stats = torch.nan_to_num(obs_stats, nan=0.0, posinf=0.0, neginf=0.0)

        obs_aux = {
            "obs_stats": obs_stats,
            "obs_num_points_search": num_points,
            "obs_soft_fg_count": soft_fg_count,
            "obs_estimated_fg_points": estimated_fg_points,
            "obs_mean_fg_score": mean_fg_score,
            "obs_valid_history_ratio": valid_history_ratio,
            "obs_current_delta_t_ratio": current_delta_t_ratio,
        }
        return obs_stats, obs_aux


    def forward(self, input_dict):
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
        if self.is_paired_batch(input_dict):
            return {
                "view_a": self.forward(input_dict["view_a"]),
                "view_b": self.forward(input_dict["view_b"]),
            }

        output_dict = {}
        points = self.encode_point_time(input_dict["points"])
        x = points.transpose(1, 2)

        if self.box_aware:
            candidate_bc = input_dict["candidate_bc"].transpose(1, 2) 
            x = torch.cat([x, candidate_bc], dim=1) 

        B, _, N = x.shape
        HL =  input_dict["valid_mask"].shape[1] # Number of historical frames, default 3
        L = HL + 1 # Total length of the point cloud sequence, 1 represents the current frame
        chunk_size = N // L

        seg_out = self.seg_pointnet(x) 
        seg_logits = seg_out[:, :2, :]  # B,2,N
        obs_stats, obs_aux = self.build_observability_stats(input_dict, seg_logits, chunk_size)
        pred_cls = torch.argmax(seg_logits, dim=1, keepdim=True)  # B,1,N
        mask_points = x[:, :4, :] * pred_cls 

        if self.box_aware:
            pred_bc = seg_out[:, 2:, :]
            mask_pred_bc = pred_bc * pred_cls
            mask_points = torch.cat([mask_points, mask_pred_bc], dim=1)
            output_dict['pred_bc'] = pred_bc.transpose(1, 2)

        # Coarse initial motion prediction
        point_feature = self.mini_pointnet(mask_points) #N*256
        motion_feature = point_feature
        if self.use_dynamics_encoder:
            z_dyn, velocity_pred, dynamics_displacement_pred, dynamics_valid = self.dynamics_encoder(
                input_dict["ref_boxs"],
                input_dict["delta_t"],
                input_dict["valid_mask"],
                input_dict.get("current_delta_t"),
            )
            output_dict["velocity_pred"] = velocity_pred
            output_dict["dynamics_displacement_pred"] = dynamics_displacement_pred
            output_dict["dynamics_valid"] = dynamics_valid
            if self.use_observability_gate:
                motion_feature, gate_aux = self.observability_gate(
                    point_feature, z_dyn, obs_stats, dynamics_valid)
                output_dict.update(gate_aux)
            else:
                motion_feature = torch.cat((point_feature, z_dyn), dim=1)
        motion_pred = self.motion_mlp(motion_feature)  # B,4
        if self.use_dynamics_encoder and self.dynamics_motion_mode == 'residual':
            output_dict["motion_residual_pred"] = motion_pred
            motion_pred = torch.cat((
                motion_pred[:, :3] + output_dict["dynamics_displacement_pred"],
                motion_pred[:, 3:4],
            ), dim=1)

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
        delta_T = input_dict["delta_T"]
        corner_stamps = create_corner_timestamps_from_deltas(delta_T, 8).to(
            device=box_seq_corners.device, dtype=box_seq_corners.dtype)
        corner_stamps = self.time_encoder(corner_stamps)
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
        output_dict.update(obs_aux)

        return output_dict

    def compute_twc_loss(self, output_a, output_b, data_a, data_b):
        box_a = output_a["aux_estimation_boxes"]
        box_b = output_b["aux_estimation_boxes"]

        center_a, center_b = box_a[:, :3], box_b[:, :3]
        theta_a, theta_b = box_a[:, 3], box_b[:, 3]

        loss_center_per_sample = F.smooth_l1_loss(center_a, center_b, reduction="none").mean(dim=1)
        loss_theta_per_sample = (
            F.smooth_l1_loss(torch.sin(theta_a), torch.sin(theta_b), reduction="none")
            + F.smooth_l1_loss(torch.cos(theta_a), torch.cos(theta_b), reduction="none")
        )
        loss_twc_per_sample = (
            loss_center_per_sample
            + getattr(self.config, "twc_theta_weight", 0.5) * loss_theta_per_sample
        )

        valid = torch.ones_like(loss_twc_per_sample, dtype=torch.bool)
        eps = getattr(self.config, "twc_timestamp_eps", 1e-6)
        anchor_eps = getattr(self.config, "twc_anchor_eps", 1e-4)
        delta_eps = getattr(self.config, "twc_delta_eps", 1e-5)

        if "current_timestamp" in data_a and "current_timestamp" in data_b:
            current_gap = torch.abs(data_a["current_timestamp"].to(box_a.device)
                                    - data_b["current_timestamp"].to(box_a.device))
            valid = valid & (current_gap.view(-1) <= eps)

        if "ref_boxs" in data_a and "ref_boxs" in data_b:
            anchor_gap = torch.max(
                torch.abs(data_a["ref_boxs"][:, 0].to(box_a.device, dtype=box_a.dtype)
                          - data_b["ref_boxs"][:, 0].to(box_a.device, dtype=box_a.dtype)),
                dim=1,
            ).values
            valid = valid & (anchor_gap <= anchor_eps)

        if "delta_T" in data_a and "delta_T" in data_b:
            delta_gap = torch.max(
                torch.abs(data_a["delta_T"].to(box_a.device, dtype=box_a.dtype)
                          - data_b["delta_T"].to(box_a.device, dtype=box_a.dtype)),
                dim=1,
            ).values
            valid = valid & (delta_gap > delta_eps)

        if getattr(self.config, "twc_full_history_only", True):
            full_a = data_a["valid_mask"].to(box_a.device).sum(dim=1) >= data_a["valid_mask"].shape[1]
            full_b = data_b["valid_mask"].to(box_a.device).sum(dim=1) >= data_b["valid_mask"].shape[1]
            valid = valid & full_a & full_b

        valid_float = valid.to(dtype=box_a.dtype)
        valid_count = valid_float.sum().clamp_min(1.0)
        loss_twc = (loss_twc_per_sample * valid_float).sum() / valid_count

        center_gap = (torch.linalg.norm(center_a - center_b, dim=1) * valid_float).sum() / valid_count
        angle_gap = torch.abs(torch.atan2(torch.sin(theta_a - theta_b), torch.cos(theta_a - theta_b)))
        angle_gap = (angle_gap * valid_float).sum() / valid_count

        return {
            "loss_twc": loss_twc,
            "twc_valid_ratio": valid_float.mean(),
            "twc_center_gap": center_gap,
            "twc_angle_gap": angle_gap,
        }

    def compute_paired_loss(self, data, output):
        data_a, data_b = data["view_a"], data["view_b"]
        output_a, output_b = output["view_a"], output["view_b"]

        loss_a = self.compute_loss(data_a, output_a)
        loss_b = self.compute_loss(data_b, output_b)

        loss_total_sup = 0.5 * (loss_a["loss_total"] + loss_b["loss_total"])
        twc_loss_dict = self.compute_twc_loss(output_a, output_b, data_a, data_b)

        twc_weight = getattr(self.config, "twc_weight", 0.05)
        warmup_epoch = getattr(self.config, "twc_warmup_epoch", 0)
        if getattr(self, "current_epoch", 0) < warmup_epoch:
            twc_weight = 0.0

        loss_total = loss_total_sup + twc_weight * twc_loss_dict["loss_twc"]
        loss_dict = {
            "loss_total": loss_total,
            "loss_total_sup": loss_total_sup,
            "loss_total_a": loss_a["loss_total"],
            "loss_total_b": loss_b["loss_total"],
        }
        for key, value in loss_a.items():
            loss_dict[f"view_a_{key}"] = value
        for key, value in loss_b.items():
            loss_dict[f"view_b_{key}"] = value
        loss_dict.update(twc_loss_dict)
        return loss_dict

    def compute_loss(self, data, output):
        if self.is_paired_batch(data):
            return self.compute_paired_loss(data, output)

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

        loss_seg = F.cross_entropy(seg_logits, seg_label, weight=torch.tensor([0.5, 2.0]).cuda())
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
        if self.use_dynamics_encoder and "velocity_pred" in output and "velocity_label" in data:
            velocity_label = data["velocity_label"].to(device=output["velocity_pred"].device,
                                                       dtype=output["velocity_pred"].dtype)
            loss_velocity = F.smooth_l1_loss(output["velocity_pred"], velocity_label)
            loss_total += loss_velocity * getattr(self.config, "velocity_weight", 0.05)
            loss_dict.update({
                "loss_total": loss_total,
                "loss_velocity": loss_velocity,
            })

        if self.use_dynamics_encoder and "dynamics_displacement_pred" in output:
            displacement_label = motion_label[:, 0, :3].to(
                device=output["dynamics_displacement_pred"].device,
                dtype=output["dynamics_displacement_pred"].dtype,
            )
            loss_dynamics_displacement = F.smooth_l1_loss(
                output["dynamics_displacement_pred"], displacement_label)
            displacement_weight = getattr(self.config, "dynamics_displacement_weight", 0.0)
            if displacement_weight != 0.0:
                loss_total += loss_dynamics_displacement * displacement_weight
            loss_dict.update({
                "loss_total": loss_total,
                "loss_dynamics_displacement": loss_dynamics_displacement,
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

        if getattr(self.config, "obs_gate_log_stats", False):
            obs_log_map = {
                "obs_num_points_search_mean": "obs_num_points_search",
                "obs_soft_fg_count_mean": "obs_soft_fg_count",
                "obs_estimated_fg_points_mean": "obs_estimated_fg_points",
                "obs_mean_fg_score": "obs_mean_fg_score",
                "obs_valid_history_ratio": "obs_valid_history_ratio",
                "obs_current_delta_t_ratio": "obs_current_delta_t_ratio",
            }
            for log_key, output_key in obs_log_map.items():
                if output_key in output:
                    loss_dict[log_key] = output[output_key].mean()

        if "obs_alpha" in output:
            obs_alpha = output["obs_alpha"]
            obs_gate_entropy = output["obs_gate_entropy"].mean()
            entropy_weight = getattr(self.config, "obs_gate_entropy_weight", 0.0)
            if entropy_weight != 0.0:
                loss_obs_gate_entropy = -obs_gate_entropy
                loss_total += entropy_weight * loss_obs_gate_entropy
                loss_dict["loss_total"] = loss_total
                loss_dict["loss_obs_gate_entropy"] = loss_obs_gate_entropy

            loss_dict.update({
                "obs_alpha_obs_mean": obs_alpha[:, 0].mean(),
                "obs_alpha_dyn_mean": obs_alpha[:, 1].mean(),
                "obs_alpha_dyn_min": obs_alpha[:, 1].min(),
                "obs_alpha_dyn_max": obs_alpha[:, 1].max(),
                "obs_gate_entropy": obs_gate_entropy,
            })

        return loss_dict

    def training_step(self, batch, batch_idx):
        """
        Args:
            batch: {
            "points": stack_frames, (B,N,3+9+1)
            "seg_label": stack_label,
            "box_label": np.append(this_gt_bb_transform.center, theta),
            "box_size": this_gt_bb_transform.wlh
        }
        Returns:

        """
        output = self(batch)
        loss_dict = self.compute_loss(batch, output)
        loss = loss_dict['loss_total']

        if self.is_paired_batch(batch):
            metric_batch = batch["view_a"]
            metric_output = output["view_a"]
        else:
            metric_batch = batch
            metric_output = output

        # log
        seg_acc = self.seg_acc(torch.argmax(metric_output['seg_logits'], dim=1, keepdim=False),
                               metric_batch['seg_label'])
        self.log('seg_acc_background/train', seg_acc[0], on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log('seg_acc_foreground/train', seg_acc[1], on_step=True, on_epoch=True, prog_bar=False, logger=True)
        if self.use_motion_cls:
            motion_acc = self.motion_acc(torch.argmax(metric_output['motion_cls'], dim=1, keepdim=False),
                                         metric_batch['motion_state_label'][:,0]) # 0 represents motion relative to the first historical box
            self.log('motion_acc_static/train', motion_acc[0], on_step=True, on_epoch=True, prog_bar=False, logger=True)
            self.log('motion_acc_dynamic/train', motion_acc[1], on_step=True, on_epoch=True, prog_bar=False,
                     logger=True)

        log_dict = {k: v.item() for k, v in loss_dict.items()}

        self.logger.experiment.add_scalars('loss', log_dict,
                                           global_step=self.global_step)

        return loss
