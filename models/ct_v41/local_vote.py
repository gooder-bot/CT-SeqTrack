"""四槽局部测量。旧 B3 核验最终框，岭回归仅决定是否切换本帧位置。"""
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .geometry import rotate_xyz
from .joint_observation import (FEATURES, ShapeMatcher, PointVoteGeometry, _mlp,
                                _mean, _balanced, _identity_label)


GATE_FEATURES = (
    'rival_q0', 'rival_s', 'rival_p', 'local_s', 'local_p', 'rival_t',
    'local_t', 'rival_support_log', 'local_support_log', 'local_fg_peak',
    'vote_cluster_log', 'vote_fg_cluster_log', 'separation_m', 'refinement_m',
    'shape_local', 'shape_rival', 'template_ready', 'local_cluster_scatter',
)


@dataclass
class LocalVoteOutput:
    box: Tensor
    valid: Tensor
    vote: Tensor
    foreground_logit: Tensor
    point_valid: Tensor
    pre_center: Tensor
    refine_input_mask: Tensor
    template: Tensor
    template_ready: Tensor
    peak: Tensor
    cluster: Tensor
    rival_index: Optional[Tensor] = None
    features: Optional[Tensor] = None
    relative_gain: Optional[Tensor] = None
    shape_logits: Optional[Tensor] = None
    shape_members: Optional[Tensor] = None


class LocalVoteHead(PointVoteGeometry):
    """只实例化局部几何和模板比较参数，最终 S/P/T 由四槽共享核验器给出。"""
    def __init__(self):
        super().__init__()
        self.shape = ShapeMatcher()
        self.point = _mlp(7, FEATURES)
        self.local_features = nn.Linear(64, FEATURES, bias=False)
        self.condition = _mlp(3 * FEATURES + 1, FEATURES)
        self.foreground = nn.Linear(FEATURES, 1)
        self.offset = nn.Linear(FEATURES, 3)
        self.refine = _mlp(2 * FEATURES + 3, 3)
        for layer in (self.offset, self.refine[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def load_donor(self, state_dict):
        """仅供冻结诊断导入六槽供体。正式 scratch 初始化不调用。"""
        source = state_dict.get('state_dict', state_dict)
        for prefix in ('tracker.joint_search.', 'joint_search.', ''):
            if all(prefix + key in source for key in self.state_dict()):
                self.load_state_dict({key: source[prefix + key] for key in self.state_dict()}, strict=True)
                return
        raise ValueError('incomplete donor local measurement / template weights')

    def forward(self, pool, batch, seed):
        size = batch['box_size']
        mask = batch['local_template_valid']
        template_xyz = batch['local_template_xyz'].to(size)
        ready = mask.sum(-1) >= 3
        template = self.shape.describe(template_xyz / size[:, None], mask)
        template = torch.where(ready[:, None], template, 0.)
        measured = self._point_votes(
            pool['xyz'], pool['valid'], size, seed.detach(), batch['anchor_box'],
            template, ready, pool['features'])
        return LocalVoteOutput(measured.box, measured.valid, measured.votes, measured.foreground_logit,
            pool['valid'], measured.center, measured.refine_input, template, ready, measured.peak, measured.cluster)

    def gate_features(self, result, candidates, verifier, pool, size):
        # 先固定前三槽原赢家。第四槽的加入不能改变其余位置的核验或平局顺序。
        rival = verifier.utility[:, :3].detach().masked_fill(~candidates.valid[:, :3], -torch.inf).argmax(-1)
        row = torch.arange(len(rival), device=rival.device)
        rival_box = candidates.boxes[row, rival].detach()
        shape_scores, shape_members = [], []
        for box in (result.box.detach(), rival_box):
            _, members, score = self._shape_at(pool['xyz'], pool['valid'], box, size,
                                               result.template, result.template_ready)
            shape_scores.append(score)
            shape_members.append(members)
        result.shape_logits = torch.stack(shape_scores, 1)
        result.shape_members = torch.stack(shape_members, 1)
        result.rival_index = rival
        with torch.no_grad():
            logits = result.foreground_logit
            point_index, cluster = result.peak, result.cluster
            count = cluster.sum(-1)
            distance = (result.vote - result.pre_center[:, None]).norm(dim=-1)
            probability = logits.sigmoid()
            p = verifier.logits.sigmoid()
            support = verifier.support_mask.sum(-1).to(p).log1p()
            result.features = torch.stack((
                (rival == 0).to(p), p[row, rival, 0], p[row, rival, 1], p[:, 3, 0], p[:, 3, 1],
                p[row, rival, 2], p[:, 3, 2], support[row, rival], support[:, 3],
                probability[row, point_index], count.to(p).log1p(),
                (cluster & (probability >= .5)).sum(-1).to(p).log1p(),
                (result.box[:, :3] - rival_box[:, :3]).norm(dim=-1),
                (result.pre_center - result.box[:, :3]).norm(dim=-1),
                *shape_scores, result.template_ready.to(p),
                distance.masked_fill(~cluster, 0.).sum(-1) / count.clamp_min(1)), -1)
            result.features = torch.where(result.valid[:, None], result.features, 0.)
        return result.features


def local_vote_losses(result, batch):
    valid = result.point_valid
    own = valid & (batch['segmentation_labels'][:, -1] > 0)
    truth = batch['target_box'][:, :3].detach().to(result.box)
    yaw = result.box[:, 3].detach()
    size = batch['box_size'].detach().to(result.box)
    vote_error = rotate_xyz(result.vote - truth[:, None], yaw) / size[:, None]
    center_error = rotate_xyz(result.box[:, :3] - truth, yaw) / size
    inside = result.shape_members
    identity_target = _identity_label(inside, own[:, None])
    # 精修资格由本次实际读取的输入邻域定义，写入与身份标签使用最终框。
    refine_valid = (result.refine_input_mask & own).any(-1)
    terms = dict(
        loss_local_foreground=_balanced(result.foreground_logit, own, valid),
        loss_local_vote=_mean(F.smooth_l1_loss(vote_error, torch.zeros_like(vote_error),
                                               reduction='none').mean(-1), own),
        loss_local_refine=_mean(F.smooth_l1_loss(center_error, torch.zeros_like(center_error),
                                                 reduction='none').mean(-1),
                                refine_valid),
        loss_local_match=_balanced(result.shape_logits, identity_target, inside.any(-1)),
    )
    terms['loss_local_total'] = (.25 * terms['loss_local_foreground'] + .25 * terms['loss_local_vote']
                                 + .1 * terms['loss_local_refine'] + .25 * terms['loss_local_match'])
    terms['local_refine_supervised_count'] = refine_valid.sum().detach()
    return terms


class LocalGainGate(nn.Module):
    """用自然训练轨迹的充分统计量拟合标准化岭回归。持续参数随模型存档。"""
    def __init__(self):
        super().__init__()
        dim = len(GATE_FEATURES)
        self.register_buffer('mu', torch.zeros(dim, dtype=torch.float64))
        self.register_buffer('sd', torch.ones_like(self.mu))
        self.register_buffer('w', torch.zeros_like(self.mu))
        self.register_buffer('mean_y', self.mu.new_zeros(()))
        self.register_buffer('fitted', torch.tensor(False))
        # 扩展行 [1, X, y] 的 Gram 矩阵 ，只占固定空间；只支持完整轮次恢复。
        self.register_buffer('moments', self.mu.new_zeros(dim+2, dim+2), persistent=False)

    @torch.no_grad()
    def load_archive(self, record):
        if (tuple(record['features']) != GATE_FEATURES or record['penalty'] != 20.
                or record['threshold'] != 0.):
            raise ValueError('ridge feature schema, penalty or decision threshold mismatch')
        fields = {}
        for key in ('mu', 'sd', 'w', 'mean_y'):
            dest = getattr(self, key)
            value = torch.as_tensor(record[key], dtype=dest.dtype, device=dest.device)
            if value.shape != dest.shape or not torch.isfinite(value).all():
                raise ValueError(f'invalid ridge field {key}')
            fields[key] = value
        if (fields['sd'] <= 0.).any():
            raise ValueError('ridge scales must be positive')
        for key, value in fields.items():
            getattr(self, key).copy_(value)
        self.fitted.fill_(True)

    def forward(self, x):
        return self.mean_y + ((x.double() - self.mu) / self.sd) @ self.w

    @torch.no_grad()
    def observe(self, result, targets):
        x = result.features.double()
        utility = targets['utility']
        y = utility[:, 3] - utility.gather(1, result.rival_index[:, None]).squeeze(1)
        values = torch.cat((torch.ones_like(x[:, :1]), x, y[:, None].double()), 1)
        active = torch.where(result.valid[:, None], values, 0.)
        self.moments.add_(active.T @ active)

    def reset(self):
        self.moments.zero_()

    @torch.no_grad()
    def fit(self):
        values = self.moments
        count = int(values[0, 0].item())
        if count:
            mean = values[0, 1:-1] / count
            target_mean = values[0, -1] / count
            gram = values[1:-1, 1:-1] - count * torch.outer(mean, mean)
            sd = (gram.diag() / count).clamp_min(0.).sqrt().clamp_min(.001)
            rhs = (values[1:-1, -1] - count * mean * target_mean) / sd
            penalized = gram / torch.outer(sd, sd)
            penalized.diagonal().add_(20.)
            weight = torch.linalg.solve(penalized, rhs)
            if not torch.isfinite(weight).all():
                raise FloatingPointError('nonfinite local gain ridge coefficients')
            self.mu.copy_(mean)
            self.sd.copy_(sd)
            self.w.copy_(weight)
            self.mean_y.copy_(target_mean)
            self.fitted.fill_(True)
        return dict(samples=count, fitted=bool(self.fitted.item()))
