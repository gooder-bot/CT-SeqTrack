"""v36 当前几何与身份读取分工；两个顺序模式只消费缓存。"""
from __future__ import annotations

import torch
from torch import nn

from models.ct_v31.evidence import RawLocalGeometry, select_evidence_points, _gather
from models.ct_v31.observation import B0Observation
from .contracts import EvidenceHypotheses


@torch.no_grad()
def sequential_members(votes, weights, valid, point_ids):
    """第一簇最终成员移除后再建第二簇，refine不能回读已解释点。"""
    b, n, _ = votes.shape
    good = (valid.bool() & (point_ids >= 0) & torch.isfinite(votes).all(-1)
            & torch.isfinite(weights) & (weights > 0))
    order = torch.argsort(point_ids.masked_fill(~good, torch.iinfo(torch.long).max), stable=True)
    inverse = torch.argsort(order, stable=True)
    ids = point_ids.gather(1, order)
    unique = torch.cat((torch.ones(b, 1, dtype=torch.bool, device=votes.device),
                        ids[:, 1:] != ids[:, :-1]), 1)
    good = good.gather(1, order) & unique
    xy = _gather(torch.nan_to_num(votes.detach()), order)[..., :2]
    weight = weights.detach().gather(1, order).masked_fill(~good, 0.)
    distance_matrix = torch.linalg.vector_norm(xy[:, :, None] - xy[:, None], dim=-1)
    remaining = good.clone()
    members, hubers, valids, previous = [], [], [], None
    for _ in range(2):
        mass = ((distance_matrix <= 1.) * (weight * remaining)[:, None]).sum(-1)
        seed = mass.masked_fill(~remaining, -torch.inf).argmax(1)
        active = remaining.any(1)
        center = _gather(xy, seed[:, None]).squeeze(1)
        for _ in range(3):
            distance = torch.linalg.vector_norm(xy - center[:, None], dim=-1)
            member = remaining & active[:, None] & (distance <= 1.)
            robust = weight * member * (.5 / distance.clamp_min(.5))
            center = (robust[..., None] * xy).sum(1) / robust.sum(1).clamp_min(1e-8)[:, None]
        distance = torch.linalg.vector_norm(xy - center[:, None], dim=-1)
        member = remaining & active[:, None] & (distance <= 1.)
        okay = active & member.any(1)
        if previous is not None:
            okay &= torch.linalg.vector_norm(center - previous, dim=-1) > .75
        member &= okay[:, None]
        members.append(member.gather(1, inverse))
        hubers.append((.5 / distance.clamp_min(.5)).gather(1, inverse))
        valids.append(okay)
        # 同时排除最终成员和最终中心邻域，不只抑制最初seed。
        remaining &= ~(member | ((distance <= .75) & okay[:, None]))
        previous = center
    return torch.stack(members, 1), torch.stack(hubers, 1), torch.stack(valids, 1)


def build_live_modes(votes, weights, valid, point_ids, *, detach_mass_weights=False):
    if type(detach_mass_weights) is not bool:
        raise TypeError('detach_mass_weights must be bool')
    valid = (valid.bool() & torch.isfinite(votes).all(-1) & torch.isfinite(weights)
             & (weights>0) & (point_ids>=0))
    members, huber, mode_valid = sequential_members(votes, weights, valid, point_ids)
    clean = torch.where(valid[..., None], votes, 0.)
    # 只切重心权重的直接梯度；成员/前向数值不变，当前 votes 仍保持 live。
    mass_weights = weights.detach() if detach_mass_weights else weights
    mass = torch.where(valid, mass_weights, 0.)[:, None] * members * huber
    denominator = mass.sum(-1).clamp_min(1e-8)
    centers = (mass[..., None] * clean[:, None]).sum(2) / denominator[..., None]
    centers = torch.where(mode_valid[..., None], centers, 0.)
    with torch.no_grad():
        delta = clean[:, None, :, :2].detach() - centers[:, :, None, :2].detach()
        covariance = (mass.detach()[..., None, None] * delta[..., :, None] * delta[..., None, :]).sum(2)
        covariance /= denominator.detach()[..., None, None]
    return centers, mode_valid, members, covariance


class B2IdentityEvidence(nn.Module):
    def __init__(self, config=None, feature_dim=64, *, detach_mass_weights=False):
        super().__init__()
        if feature_dim != 64:
            raise ValueError('v36 evidence features are fixed to 64')
        if type(detach_mass_weights) is not bool:
            raise TypeError('detach_mass_weights must be bool')
        self.detach_mass_weights = detach_mass_weights
        self.point_encoder = nn.Sequential(nn.Linear(5,64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64,64), nn.LayerNorm(64), nn.GELU())
        self.local_geometry = RawLocalGeometry()
        self.memory_metadata = nn.Linear(8,64)
        self.memory_attention = nn.MultiheadAttention(64,4,dropout=0.,batch_first=True)
        self.b0_summary = nn.Linear(64,64)
        self.identity_norm = nn.LayerNorm(64)
        self.identity_head = nn.Sequential(nn.Linear(64,64), nn.GELU(), nn.Linear(64,1))
        self.observation_attention = nn.MultiheadAttention(64,4,dropout=0.,batch_first=True)
        self.geometry_norm = nn.LayerNorm(64)
        self.vote_head = nn.Sequential(nn.Linear(64,64), nn.GELU(), nn.Linear(64,3))
        self.reliability_head = nn.Sequential(nn.Linear(64,64), nn.GELU(), nn.Linear(64,1))

    @staticmethod
    def _attention(module, query, keys, valid):
        zero = keys.new_zeros((len(keys),1,64))
        output = module(query, torch.cat((keys,zero),1), torch.cat((keys,zero),1),
            key_padding_mask=~torch.cat((valid,torch.ones(len(keys),1,dtype=torch.bool,device=keys.device)),1),
            need_weights=True, average_attn_weights=False)[0]
        return torch.where(valid.any(-1)[:,None,None],output,0.)

    def encode_current(self, observation, batch):
        """每端点只编码一次当前raw池，不读memory/identity/prior context。"""
        points = batch['extension_points']
        if points.shape[1:] != (768,5):
            raise ValueError('v36 requires a [B,768,5] extension pool')
        ids = batch['extension_ids'].long()
        valid = batch['extension_valid'].bool() & (ids>=0) & torch.isfinite(points).all(-1)
        clean = torch.where(valid[...,None],points,0.)
        size = batch['box_size'].detach().to(points)
        raw = self.point_encoder(torch.cat((clean[...,:3]/size[:,None],clean[...,3:]),-1))
        local = self.local_geometry(clean,valid,ids,size)
        current_valid = B0Observation.measurement_mask(batch)[:,-1]
        current = torch.where(current_valid[...,None],observation.point_features[:,-1],0.)
        return dict(points=clean, ids=ids, valid=valid, size=size, raw=raw, local=local,
                    current=current, current_valid=current_valid)

    def identify(self, cache, observation, batch):
        memory_points = batch['memory_points']
        valid = batch['memory_valid'].bool() & torch.isfinite(memory_points).all(-1)
        metadata = torch.nan_to_num(batch['memory_metadata']).detach()
        metadata = torch.cat((metadata[...,:3],torch.log1p(metadata[...,3:4].clamp_min(0)),metadata[...,4:]),-1)
        memory = self.point_encoder(torch.where(valid[...,None],memory_points,0.))
        memory = torch.where(valid[...,None],memory+self.memory_metadata(metadata),0.)
        observed = B0Observation.measurement_mask(batch)
        feature = torch.where(observed[...,None],observation.point_features,0.)
        summary = feature.sum((1,2))/observed.sum((1,2)).clamp_min(1)[:,None]
        read = self._attention(self.memory_attention,cache['raw']+cache['local'],memory,valid)
        identity_feature = self.identity_norm(cache['raw']+cache['local']+read+self.b0_summary(summary)[:,None])
        logits = self.identity_head(identity_feature).squeeze(-1).masked_fill(~cache['valid'],0.)
        return logits,memory,valid

    def geometry_votes(self, cache, selected):
        """固定raw IDs时memory改变不影响本函数；可直接用于权限测试。"""
        valid = selected>=0
        raw,local = _gather(cache['raw'],selected),_gather(cache['local'],selected)
        read = self._attention(self.observation_attention,raw+local,cache['current'],cache['current_valid'])
        geometry = torch.where(valid[...,None],self.geometry_norm(raw+local+read),0.)
        xyz = torch.where(valid[...,None],_gather(cache['points'][...,:3],selected),0.)
        votes = torch.where(valid[...,None],xyz+self.vote_head(geometry)*cache['size'][:,None],0.)
        reliability = self.reliability_head(geometry).squeeze(-1).masked_fill(~valid,0.)
        return geometry,xyz,votes,reliability

    def forward(self, observation, batch, prior=None):
        cache = self.encode_current(observation,batch)
        logits,memory,memory_valid = self.identify(cache,observation,batch)
        selected = select_evidence_points(cache['points'],logits,cache['valid'],cache['ids'],
                                          batch['extension_partition'].long())
        valid = selected>=0
        geometry,xyz,votes,reliability = self.geometry_votes(cache,selected)
        selected_logits = _gather(logits,selected).masked_fill(~valid,0.)
        ids = _gather(cache['ids'],selected).masked_fill(~valid,-1)
        weights = selected_logits.sigmoid()*reliability.sigmoid()*valid
        centers,mode_valid,members,covariance = build_live_modes(votes,weights,valid,ids,
            detach_mass_weights=self.detach_mass_weights)
        return EvidenceHypotheses(centers_xyz=centers,mode_valid=mode_valid,members=members,
            covariance_xy=covariance,point_features=geometry,point_xyz=xyz,point_valid=valid,
            point_ids=ids,point_indices=selected,identity_logits=logits,selected_identity_logits=selected_logits,
            vote_xyz=votes,reliability_logits=reliability,memory_features=memory,memory_valid=memory_valid)
