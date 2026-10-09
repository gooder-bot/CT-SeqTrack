"""v36 q0保护与独立mode残差；prior是直接动作，不经pose头修改。"""
from __future__ import annotations

import torch
from torch import nn

from models.ct_v31.decoder import SharedHypothesisDecoder as LegacyDecoder, _masked_source
from models.ct_v31.geometry import anchor_yaw, transform_boxes
from models.ct_v31.observation import box_corners_xyz
from .contracts import CandidateSet
from .verification import LocalBoxReader, build_current_pool, candidate_authorization, projected_xy_size
from utils.feature_precision import feature_autocast, feature_fp32


class SharedHypothesisDecoder(LegacyDecoder):
    def __init__(self, prior_dim=128, dropout=.2, time_scale=.5, enable_modes=True, **unused):
        # 沿用已验证B0 encoder/query/context/pose结构；没有旧quality或memory参数。
        super().__init__(prior_dim=prior_dim,dropout=dropout,query_context=True,
            time_scale=time_scale,initial_context=True,local_observation=False)
        del self.quality_head
        del self.memory_projection
        self.local_reader = LocalBoxReader(zero_output=True)
        self.enable_modes = bool(enable_modes)
        if self.enable_modes:
            self.mode_pose_head = nn.Linear(64,5)
            nn.init.zeros_(self.mode_pose_head.weight)
            with torch.no_grad():
                self.mode_pose_head.bias.copy_(torch.tensor([0.,0.,0.,0.,1.]))
        else:
            del self.extension_projection
            self.mode_pose_head = None

    def _physical_context(self, observation, prior, size):
        reference = observation.coarse_box
        delta = prior.box.detach().to(reference)[...,:3]-reference.detach()[...,:3]
        c,s = reference.detach()[:,3].cos(),reference.detach()[:,3].sin()
        x,y,z = delta.unbind(-1)
        delta = torch.stack((c*x+s*y,-s*x+c*y,z),-1)/size
        range_size = projected_xy_size(size,prior.origin_world.detach()[:,3].to(reference),
                                      prior.axes.detach().to(reference))
        summary = torch.cat((delta,torch.log1p(prior.range_xy.detach().to(reference)/range_size),
            torch.log1p(prior.horizon.detach().to(reference).clamp_min(0)/self.context_time_scale)[:,None],
            torch.log1p(prior.velocity_age.detach().to(reference).clamp_min(0)/self.context_time_scale)[:,None],
            prior.prior_valid.detach().to(reference)[:,None],prior.motion_valid.detach().to(reference)[:,None],
            prior.initial.detach().to(reference)[:,None],observation.quality.detach()[:,:2]),-1)
        if summary.shape[1] != 12 or not bool(torch.isfinite(summary).all()):
            raise ValueError('v36 B1 decoder summary must have 12 finite values')
        feature = _masked_source(prior.feature,prior.prior_valid.bool(),'v36 prior')
        return self.prior_adapter(torch.cat((feature,summary.detach()),-1))

    @staticmethod
    def _pose_boxes(residual, reference, seed_yaw):
        sine,cosine = residual[...,3],residual[...,4]
        nonzero = sine.square()+cosine.square()>1e-12
        angle = torch.atan2(torch.where(nonzero,sine,0.),torch.where(nonzero,cosine,1.))
        yaw = seed_yaw+angle
        return torch.cat((reference+residual[...,:3],torch.atan2(yaw.sin(),yaw.cos())[...,None]),-1)

    def forward(self, observation, evidence, batch, prior):
        reference = observation.coarse_box
        b = len(reference)
        history_valid = batch['history_valid'].bool()
        history = torch.where(history_valid[...,None],batch['history_boxes'].to(reference),0.).detach()
        size = batch['box_size'].detach().to(reference)
        times = self._frame_times(batch,reference)
        prior_box = prior.box.detach().to(reference)
        q0_valid = observation.sequence_valid.bool()
        coarse = torch.where(q0_valid[:,None],reference,prior_box)
        members = evidence.members.detach().bool() & evidence.point_valid[:,None].bool()
        mode_valid = evidence.mode_valid.detach().bool() & members.any(-1)
        center = torch.where(mode_valid[...,None],evidence.centers_xyz,0.)
        mode_seed = torch.cat((center,coarse[:,None,3:4].detach().expand(-1,2,-1)),-1)
        seeds = torch.cat((history,coarse[:,None],mode_seed),1)
        corner_seeds = torch.cat((history,coarse[:,None],mode_seed.detach()),1)
        query_valid = torch.cat((history_valid,q0_valid[:,None],mode_valid),1)
        candidate_valid = torch.cat((q0_valid[:,None],prior.prior_valid[:,None].bool(),mode_valid),1)
        if not bool(candidate_valid.any(-1).all()):
            raise ValueError('every endpoint requires a legal observation or prior')
        yaw = anchor_yaw(batch,reference)
        local_seeds = transform_boxes(seeds,yaw)
        local_corners = transform_boxes(corner_seeds,yaw)
        output_reference = torch.cat((local_seeds[:,:3,:3],torch.zeros_like(local_seeds[:,3:4,:3]),
                                      local_seeds[:,4:,:3]),1)
        corners = box_corners_xyz(local_corners,size[:,None])
        query_times = torch.cat((times[:,:3],times[:,-1:].expand(-1,3)),1)
        corner_input = torch.cat((corners,query_times[:,:,None,None].expand(-1,-1,8,-1),
            output_reference.detach()[:,:,None].expand(-1,-1,8,-1)),-1)
        queries = self.corner_projection(corner_input.reshape(b,48,7))
        context = self._query_context(observation,batch,local_seeds[:,:3],history_valid,size)
        context = context+self._physical_context(observation,prior,size)
        by_query = torch.cat((context.new_zeros(b,3,64),context[:,None].expand(-1,3,-1)),1)
        queries = queries+by_query.repeat_interleave(8,dim=1)
        pool = build_current_pool(observation,evidence,batch)
        permissions = candidate_authorization(pool,members,candidate_valid,q0_extension=False)
        candidate_seeds = torch.cat((coarse[:,None],prior_box[:,None],mode_seed.detach()),1)
        increment,_ = self.local_reader(pool,candidate_seeds,permissions,size)
        increment = increment[:,[0,2,3]]
        queries = queries+torch.cat((increment.new_zeros(b,3,8,64),increment),1).reshape(b,48,64)
        source,source_valid = self.encode_sources(observation,history_valid)
        extension_valid = evidence.point_valid.bool()
        extension = (self.extension_projection(_masked_source(evidence.point_features,extension_valid,'v36 extension'))
                     if self.enable_modes else evidence.point_features.new_zeros(evidence.point_features.shape))
        extension = torch.where(extension_valid[...,None],extension,0.)
        kv = torch.cat((source,extension),1)
        # history和q0完全看不到extension；memory没有进入定位KV。
        ext_allowed = torch.cat((torch.zeros(b,4,extension_valid.shape[1],device=reference.device,dtype=torch.bool),
                                  members),1)
        allowed = torch.cat((source_valid[:,None].expand(-1,6,-1),ext_allowed),-1)
        allowed = (allowed & query_valid[...,None]).repeat_interleave(8,dim=1)
        bf16 = self.execution_profile == 'optimized_bf16'
        with feature_autocast(queries, enabled=bf16):
            decoded,_ = self.decoder(queries,None,kv,allowed,
                query_valid=query_valid.repeat_interleave(8,dim=1)[...,None],zero_invalid_rows=True)
        decoded = feature_fp32(decoded, enabled=bf16)
        features = self.box_projection(decoded.reshape(b,6,512))
        features = torch.where(query_valid[...,None],features,0.)
        mode_residual = (self.mode_pose_head(features[:,4:]) if self.enable_modes
                         else features.new_zeros(b, 2, 5))
        residual = torch.cat((self.pose_head(features[:,:4]),mode_residual),1)
        boxes = self._pose_boxes(residual,output_reference,local_seeds[...,3])
        boxes = transform_boxes(boxes,yaw,to_world=True)
        boxes = torch.where(query_valid[...,None],boxes,0.)
        candidates = torch.cat((boxes[:,3:4],prior_box[:,None],boxes[:,4:]),1)
        # 最终decoded中心再次去重：保留首个模式，不补候选、不重跑网络。
        final_duplicate = (mode_valid.all(-1) &
            (torch.linalg.vector_norm(candidates[:,2,:2].detach()-candidates[:,3,:2].detach(),dim=-1)<=.75))
        candidate_valid[:,3] &= ~final_duplicate
        current_features = torch.cat((features[:,3:4],features.new_zeros(b,1,64),features[:,4:]),1)
        return CandidateSet(boxes=candidates,valid=candidate_valid,history_boxes=boxes[:,:3],
                            features=current_features,seed_boxes=mode_seed)
