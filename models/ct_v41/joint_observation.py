"""V41 联合观测。候选自己的当前点同时用于定位、身份、几何和提交。"""
from dataclasses import dataclass, fields
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .geometry import inside_final_boxes, rotate_xyz
from . import joint_candidates
from .joint_grid import FINE_POINTS, GRID_FEATURES, fine_crop, grid_labels, inside, make_sparse_scan


FEATURES = 32
TEMPLATE_POINTS = 128
# 首轮固定的辅助监督权重；保留原 B0 / MTP / B3 主损失的尺度。
JOINT_LOSS_WEIGHTS = {
    'loss_joint_foreground': .25, 'loss_joint_vote': .25, 'loss_joint_refine': .1,
    'loss_joint_match': .25, 'loss_joint_swap': .1, 'loss_joint_grid_object': .1,
    'loss_joint_grid_match': .1, 'loss_joint_grid_offset': .1,
}


def _mlp(inputs, outputs):
    return nn.Sequential(nn.Linear(inputs, FEATURES), nn.LayerNorm(FEATURES),
                         nn.GELU(), nn.Linear(FEATURES, outputs))


def _tensor(array, like):
    # 输入可以是只读 KITTI numpy 视图；不修改数组，也不返回可写共享别名。
    return torch.tensor(np.asarray(array), dtype=like.dtype, device=like.device)


def _mean(values, mask):
    return torch.where(mask, values, 0.).sum() / mask.sum().clamp_min(1)


def _balanced(logits, labels, valid):
    loss = F.binary_cross_entropy_with_logits(logits, labels.to(logits), reduction='none')
    pos, neg = valid & labels, valid & ~labels
    terms = _mean(loss, pos) + _mean(loss, neg)
    return terms / (pos.any().to(loss) + neg.any().to(loss)).clamp_min(1)


class ShapeMatcher(nn.Module):
    """首框和所有精修候选共用形状编码；零 logit 是无匹配选项。"""
    def __init__(self):
        super().__init__()
        self.point = _mlp(3, FEATURES)
        self.combine = _mlp(2 * FEATURES + 1, FEATURES)
        self.compare = _mlp(4 * FEATURES + 1, 1)

    def describe(self, local_points, valid):
        features = self.point(local_points.clamp(-2., 2.))
        count = valid.sum(-1)
        mean = torch.where(valid[..., None], features, 0.).sum(-2) / count.clamp_min(1)[..., None]
        largest = features.masked_fill(~valid[..., None], -torch.inf).amax(-2)
        largest = torch.where((count > 0)[..., None], largest, 0.)
        return self.combine(torch.cat((mean, largest, torch.log1p(count.to(mean))[..., None]), -1))

    def forward(self, descriptor, query, has_query):
        descriptor, query = torch.broadcast_tensors(descriptor, query)
        valid = has_query.expand(descriptor.shape[:-1])
        query = torch.where(valid[..., None], query, 0.)
        return self.compare(torch.cat((descriptor, query, descriptor*query,
                                       (descriptor-query).abs(), valid[..., None].to(query)), -1)).squeeze(-1)


@dataclass
class JointMeasurement:
    box: torch.Tensor
    valid: torch.Tensor
    logits: torch.Tensor
    match_logit: torch.Tensor
    match_wins: torch.Tensor
    xyz: torch.Tensor
    ids: torch.Tensor
    probability: torch.Tensor
    point_valid: torch.Tensor
    support_mask: torch.Tensor
    confirmed: torch.Tensor
    vote: torch.Tensor
    foreground_logit: torch.Tensor
    descriptor: torch.Tensor
    members: torch.Tensor

    @classmethod
    def empty_remote(cls, fallback):
        b = len(fallback)
        xyz = fallback.new_zeros((b, FINE_POINTS, 3))
        ids = torch.full((b, FINE_POINTS), -1, dtype=torch.long, device=fallback.device)
        mask = ids >= 0
        valid = mask.any(-1)
        values = fallback.new_zeros((b, FINE_POINTS))
        return cls(box=fallback.detach(), valid=valid, logits=fallback.new_zeros((b, 3)),
                   match_logit=values[:, 0], match_wins=valid, xyz=xyz, ids=ids,
                   probability=values, point_valid=mask, support_mask=mask,
                   confirmed=valid, vote=xyz, foreground_logit=values,
                   descriptor=fallback.new_zeros((b, FEATURES)), members=mask)

    def with_rows(self, index, value):
        return JointMeasurement(**{
            f.name: getattr(self, f.name).index_copy(0, index, getattr(value, f.name))
            for f in fields(self)
        })


@dataclass
class CoarseGrid:
    row: int
    scan: object
    class_logit: torch.Tensor
    instance_logit: torch.Tensor
    offsets: torch.Tensor
    proposals_sensor: np.ndarray


@dataclass
class JointSearchResult:
    local: JointMeasurement
    remote: JointMeasurement
    local_position_valid: torch.Tensor
    legacy_match_wins: torch.Tensor
    legacy_match_logits: torch.Tensor
    template_valid: torch.Tensor
    coarse: list
    legacy_members: torch.Tensor
    swapped_logits: object
    remote_position_valid: Optional[torch.Tensor] = None


def vote_neighborhood(votes, logits, valid, size, yaw):
    index = logits.masked_fill(~valid, -torch.inf).argmax(-1)
    row = torch.arange(len(index), device=index.device)
    seed = votes[row, index].detach()
    distance = rotate_xyz(votes.detach()-seed[:, None], yaw) / size[:, None]
    return index, valid & (distance.abs() <= .75).all(-1)


@dataclass
class PointVotes:
    box: torch.Tensor
    valid: torch.Tensor
    votes: torch.Tensor
    foreground_logit: torch.Tensor
    center: torch.Tensor
    refine_input: torch.Tensor
    summary: torch.Tensor
    peak: torch.Tensor
    cluster: torch.Tensor


class PointVoteGeometry(nn.Module):
    """局部与远区共用的逐点前景、中心投票和中心精修。"""
    def _shape_at(self, xyz, point_valid, box, size, template, has_template):
        relative = rotate_xyz(xyz-box[:, None, :3], box[:, 3]) / size[:, None]
        members = point_valid & (relative.abs() <= .5).all(-1)
        descriptor = self.shape.describe(relative, members)
        score = self.shape(descriptor, template, has_template)
        return descriptor, members, score

    def _point_votes(self, xyz, valid, size, seed, anchor, template, has_template,
                 local_features=None):
        yaw = seed[:, 3]
        relative = rotate_xyz(xyz-seed[:, None, :3], yaw) / size[:, None]
        sensor = xyz + anchor[:, None, :3]
        ray = F.normalize(rotate_xyz(sensor, yaw), dim=-1)
        point = self.point(torch.cat((relative.clamp(-3., 3.), ray,
                                      sensor[..., 2:3].clamp(-5., 5.)), -1))
        if local_features is not None:
            # 借用 B0 的点特征，搜索辅助监督不改写 q0 及其共享编码器梯度。
            point = point + self.local_features(local_features.detach())
        query = template[:, None].expand_as(point)
        available = has_template[:, None, None].expand(*point.shape[:-1], 1).to(point)
        feature = self.condition(torch.cat((point, query, point*query, available), -1))
        logit = self.foreground(feature).squeeze(-1)
        offset = rotate_xyz(self.offset(feature).tanh() * size[:, None], yaw, to_world=True)
        votes = xyz + offset
        admissible = valid.any(-1)
        peak, cluster = vote_neighborhood(votes, logit, valid, size, yaw)
        cluster_weight = torch.softmax(logit.masked_fill(~cluster, -1e4), dim=-1) * cluster
        center = (cluster_weight[..., None] * votes).sum(1)
        center = torch.where(admissible[:, None], center, seed[:, :3])
        delta = rotate_xyz(xyz-center[:, None], yaw) / size[:, None]
        near = valid & (delta.abs() <= .75).all(-1)
        weight = torch.softmax(logit.masked_fill(~near, -1e4), -1) * near
        summary = (weight[..., None]*feature).sum(1)
        mean_position = (delta * weight[..., None]).sum(1)
        refine = self.refine(torch.cat((summary, template, mean_position), -1)).tanh() * size * .5
        corrected = center + rotate_xyz(refine, yaw, to_world=True)
        box = torch.cat((torch.where(admissible[:, None], corrected, seed[:, :3]), yaw[:, None]), -1)
        return PointVotes(box, admissible, votes, logit, center, near, summary, peak, cluster)


class JointSearchObservation(PointVoteGeometry):
    def __init__(self):
        super().__init__()
        self.shape = ShapeMatcher()
        self.cell = _mlp(GRID_FEATURES, FEATURES)
        self.cell_class = nn.Linear(FEATURES, 1)
        self.cell_offset = nn.Linear(FEATURES, 3)
        self.cell_match = _mlp(3 * FEATURES + 1, 1)
        self.point = _mlp(7, FEATURES)
        self.local_features = nn.Linear(64, FEATURES, bias=False)
        self.condition = _mlp(3 * FEATURES + 1, FEATURES)
        self.foreground = nn.Linear(FEATURES, 1)
        self.offset = nn.Linear(FEATURES, 3)
        self.refine = _mlp(2 * FEATURES + 3, 3)
        self.quality = _mlp(2 * FEATURES + 4, 3)
        for layer in (self.cell_offset, self.offset, self.refine[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.quality[-1].weight)
        nn.init.constant_(self.quality[-1].bias, -1.)

    def _templates(self, rows, size):
        pts = np.zeros((len(rows), TEMPLATE_POINTS, 3), dtype=np.float32)
        valid = np.zeros(pts.shape[:2], dtype=bool)
        for i, row in enumerate(rows):
            template = np.asarray(row['template_xyz'], dtype=np.float32)
            if template.ndim != 2 or template.shape[1] != 3:
                raise ValueError("joint template must contain unique frame-local XYZ points")
            if len(template) > TEMPLATE_POINTS:
                template = template[np.linspace(0, len(template)-1, TEMPLATE_POINTS, dtype=np.int64)]
            pts[i, :len(template)], valid[i, :len(template)] = template, True
        mask = torch.as_tensor(valid, device=size.device)
        ready = mask.sum(-1) >= 3
        template = self.shape.describe(_tensor(pts, size) / size[:, None], mask)
        return torch.where(ready[:, None], template, 0.), ready

    def _measure(self, xyz, ids, valid, size, seed, anchor, template, has_template,
                 local_features=None):
        vote = self._point_votes(
            xyz, valid, size, seed, anchor, template, has_template, local_features)
        box, logit = vote.box, vote.foreground_logit
        descriptor, members, identity = self._shape_at(xyz, valid, box.detach(), size, template, has_template)
        probability = torch.where(valid, logit.sigmoid(), 0.)
        support = members & (probability >= .5)
        stats = torch.stack((torch.log1p(members.sum(-1).to(size)), torch.log1p(support.sum(-1).to(size)),
                             (probability*members).sum(-1)/members.sum(-1).clamp_min(1),
                             identity.detach().sigmoid()), -1)
        quality = self.quality(torch.cat((vote.summary.detach(), descriptor.detach(), stats.detach()), -1))
        return JointMeasurement(box, vote.valid, quality, identity,
                                members.any(-1) & (identity > 0.), xyz, ids,
                                probability, valid, support, torch.zeros_like(vote.valid),
                                vote.votes, logit, descriptor, members)

    def _coarse(self, row_index, scan, query, has_query, anchor, size, prior):
        f = self.cell(_tensor(scan.features, query))
        class_logit = self.cell_class(f).squeeze(-1)
        offset = self.cell_offset(f).tanh() * size.norm()
        centers = _tensor(scan.centers, query) - anchor[:3] + offset
        q = query.expand_as(f)
        ready = has_query.expand(len(f), 1).to(q)
        instance = self.cell_match(torch.cat((f, q, f*q, ready), -1)).squeeze(-1)
        record = CoarseGrid(row_index, scan, class_logit, instance, offset,
                            np.empty((0, 3), dtype=np.float32))
        # 有界软先验：两条旧轨都漂移时，远处位置仍可赢得排名。
        motion_cost = torch.tanh((centers[:, :2] - prior[:2]).norm(dim=-1) / size[0].clamp_min(1.))
        ranking = class_logit + torch.where(has_query, instance, 0.) - .5 * motion_cost
        return record, centers, ranking

    def _outside_candidates(self, row_indices, rows, size, anchor, prior, template, has_template):
        b = len(row_indices)
        xyz = np.zeros((b, FINE_POINTS, 3), np.float32)
        ids = np.full((b, FINE_POINTS), -1, np.int64)
        seeds = prior.detach().cpu().numpy().copy()
        origins = anchor.detach().cpu().numpy()
        sizes = size.detach().cpu().numpy()
        pending = np.zeros(b, bool)
        records, rivals = [], []
        for i, row_index in enumerate(row_indices):
            row = rows[row_index]
            scan = (row['sparse_scan'] if 'sparse_scan' in row
                    else make_sparse_scan(row['scan_xyz'], row['scan_ids']))
            if scan is None:
                rivals.append(size.new_tensor(-torch.inf))
                continue
            record, centers, ranking = self._coarse(row_index, scan, template[i], has_template[i],
                                                     anchor[i], size[i], prior[i])
            records.append(record)
            # 一次同步取本行的无梯度路由；原始扫描不会上传或进入 B0。
            routing = torch.cat((centers, ranking[:, None], record.instance_logit[:, None]),
                                -1).detach().cpu().numpy()
            locations, scores = routing[:, :3], routing[:, 3].copy()
            local_half = .625 * sizes[i] + 2.
            scores[inside(locations, np.asarray(row['main_box']), local_half)] = -np.inf
            if row['backup_valid']:
                scores[inside(locations, np.asarray(row['backup_box']), local_half)] = -np.inf
            chosen = []
            distinct_half = .75 * sizes[i]
            for _ in range(3):
                best = int(np.argmax(scores))
                if not np.isfinite(scores[best]):
                    break
                chosen.append(best)
                pose = np.r_[locations[best], seeds[i, 3]]
                scores[inside(locations, pose, distinct_half)] = -np.inf
            record.proposals_sensor = np.asarray(locations[chosen] + origins[i, :3], dtype=np.float32)
            pending[i] = bool(row['pending_valid'])
            if pending[i]:
                seeds[i] = row['pending_box']
            elif chosen:
                seeds[i, :3] = locations[chosen[0]]
            else:
                rivals.append(size.new_tensor(-torch.inf))
                continue
            # 只精裁一处；其余两处只保留同/异实例的粗匹配 logit。
            same = inside(locations, seeds[i], distinct_half)
            rivals_i = [idx for idx in chosen if not same[idx]]
            rivals.append(record.instance_logit[rivals_i].amax().detach() if rivals_i
                          else size.new_tensor(-torch.inf))
            sensor_box = seeds[i].copy()
            sensor_box[:3] += origins[i, :3]
            cloud, raw_ids = fine_crop(scan, sensor_box, sizes[i])
            xyz[i, :len(cloud)] = cloud - origins[i, :3]
            ids[i, :len(cloud)] = raw_ids
        return (_tensor(xyz, size), torch.as_tensor(ids, device=size.device), _tensor(seeds, size),
                torch.as_tensor(pending, device=size.device), records, torch.stack(rivals))

    def _remote(self, rows, size, anchor, prior, template, ready):
        active = [i for i, row in enumerate(rows)
                  if row.get('sparse_scan') is not None or len(row['scan_xyz'])]
        rival = size.new_full((len(rows),), -torch.inf)
        if not active:
            return JointMeasurement.empty_remote(prior), [], rival
        index = torch.as_tensor(active, device=size.device)
        active_size, active_anchor = size[index], anchor[index]
        query, has_query = template[index], ready[index]
        xyz, ids, seed, pending, coarse, active_rival = self._outside_candidates(
            active, rows, active_size, active_anchor, prior[index], query, has_query)
        result = self._measure(xyz, ids, ids >= 0, active_size, seed, active_anchor, query, has_query)
        offset = rotate_xyz(result.box[:, :3]-seed[:, :3], seed[:, 3]) / active_size
        result.confirmed = pending & has_query & result.valid & (offset.abs() <= .75).all(-1)
        if len(active) == len(rows):
            return result, coarse, active_rival
        full = JointMeasurement.empty_remote(prior).with_rows(index, result)
        return full, coarse, rival.index_copy(0, index, active_rival)

    @staticmethod
    def _competing_local_mode(local, boxes, size, cold_rows=None):
        # 对冷启动行检查每一个框外前景投票，三点簇旁的高分孤立点不会遮住它。
        votes = local.vote.detach()
        rivals = torch.zeros(boxes.shape[:2], dtype=torch.bool, device=boxes.device)
        if cold_rows is None:
            cold_rows = range(len(boxes))
        for i in cold_rows:
            supported = local.point_valid[i] & (local.probability[i].detach() >= .5)
            points = votes[i, supported]
            if len(points) < 3:
                continue
            poses = boxes[i]
            local_xyz = rotate_xyz(points[None] - poses[:, None, :3], poses[:, 3]) / size[i]
            for j, xyz in enumerate(local_xyz):
                outside = xyz[(xyz.abs() > .75).any(-1)]
                if len(outside) < 3:
                    continue
                # 车体系归一化后的无穷范数 <= .5 即落入以该票为中心的同尺寸框。
                # 限制距离矩阵为 64 × 外部真实票数，暖模板行不构造点对。
                clusters = [(torch.cdist(chunk, outside, p=float('inf')) <= .5).sum(-1).ge(3).any()
                            for chunk in outside.split(64)]
                rivals[i, j] = torch.stack(clusters).any()
        return rivals

    def forward(self, batch, observation, decoder, prior, pool, legacy_boxes, legacy_valid=None):
        rows = batch['joint_rows']
        size = batch['box_size']
        anchor = batch['anchor_box']
        template, ready = self._templates(rows, size)
        local = self._measure(pool['xyz'], pool['ids'], pool['valid'], size, decoder.q0_box.detach(),
                              anchor, template, ready, pool['features'])
        remote, coarse, coarse_rival = self._remote(rows, size, anchor, prior.box, template, ready)
        descriptors, members, identity = [], [], []
        for index in (0, 3):
            d, m, score = self._shape_at(pool['xyz'], pool['valid'], legacy_boxes[:, index].detach(),
                                         size, template, ready)
            if legacy_valid is not None:
                m = m & legacy_valid[:, index, None]
            descriptors.append(d)
            members.append(m)
            identity.append(score)
        boxes = torch.stack((legacy_boxes[:, 0], legacy_boxes[:, 3], local.box, remote.box), 1).detach()
        scores = torch.stack((*identity, local.match_logit, remote.match_logit), -1)
        region = torch.stack((members[0].any(-1), members[1].any(-1), local.members.any(-1),
                              remote.members.any(-1) & ready), -1)
        separated = ~inside_final_boxes(boxes[..., :3], boxes, size * 1.5)
        rivals = scores[:, None].expand(-1, 4, -1).masked_fill(~(separated & region[:, None]), -torch.inf)
        # 首扫即可竞争下一扫描的复核名额。只有通过接管复核的外区框才能反过来否决本帧局部位置。
        remote.match_wins = (region[:, 3] & (remote.match_logit > rivals[:, 3, :3].amax(-1).clamp_min(0))
                             & (remote.match_logit > coarse_rival))
        remote_position_valid = joint_candidates.remote_takeover_ready(remote, ready)
        remote_rival = rivals[:, :3, 3].masked_fill(~remote_position_valid[:, None], -torch.inf)
        competition = torch.maximum(rivals[:, :3, :3].amax(-1), remote_rival)
        wins = region[:, :3] & (scores[:, :3] > competition.clamp_min(0))
        cold_rows = [i for i, row in enumerate(rows) if len(row['template_xyz']) < 3]
        contested = self._competing_local_mode(local, boxes[:, :3], size, cold_rows)
        local_wins = wins & ~contested
        local.match_wins = local_wins[:, 2]
        # 单点弱测量可参与本帧位置比较。另一区域已有正身份分或冷启动存在第二簇时，
        # 仍需赢得身份竞争；未达到无匹配分的局部候选不能据此取得可信写入资格。
        local_weak = region[:, 2] & ~contested[:, 2] & (competition[:, 2] <= 0.)
        local_position_valid = local.match_wins | local_weak
        swap = None
        if self.training:
            own = torch.stack((*descriptors, local.descriptor, remote.descriptor), 1)
            swap = self.shape(own[:, :, None], template[None, None], ready[None, None])
        return JointSearchResult(local, remote, local_position_valid, local_wins[:, :2],
                                 scores[:, :2], ready, coarse, torch.stack(members, 1), swap,
                                 remote_position_valid=remote_position_valid)


def _identity_label(members, target_points):
    count = members.sum(-1)
    own = (members & target_points).sum(-1)
    return (own > 0) & (own * 2 >= count)


def joint_search_losses(result, batch):
    target = batch['target_box'].detach().to(result.local.xyz)
    size = batch['box_size'].detach().to(target)
    target_size = batch.get('target_box_size', size).detach().to(target)
    zero = result.local.box.sum() * 0.
    point_losses, vote_losses, refine_losses = [], [], []
    candidate_labels, candidate_masks, candidate_scores = [], [], []
    for measurement in (result.local, result.remote):
        if measurement is result.local and 'segmentation_labels' in batch:
            labels = batch['segmentation_labels'][:, -1].detach().to(measurement.xyz.device) > 0
        else:
            labels = inside_final_boxes(measurement.xyz, target[:, None], target_size)[:, 0]
        labels = labels & measurement.point_valid
        point_losses.append((_balanced(measurement.foreground_logit, labels, measurement.point_valid),
                             measurement.point_valid.any()))
        error = rotate_xyz(measurement.vote-target[:, None, :3], measurement.box[:, 3].detach()) / size[:, None]
        term = F.smooth_l1_loss(error, torch.zeros_like(error), reduction='none').mean(-1)
        vote_losses.append((_mean(term, labels), labels.any()))
        # 稀疏重现可以只有一个本车点。直接监督其点到中心投票，即使粗框不可达。
        # 精修只在实际选到的中心模式含本车回波时监督，避免把邻车模式硬拉向本车。
        region = inside_final_boxes(measurement.xyz, measurement.box[:, None], 1.5 * size)[:, 0]
        refine_valid = (region & labels).any(-1)
        center_error = rotate_xyz(measurement.box[:, :3] - target[:, :3], measurement.box[:, 3].detach()) / size
        center_loss = F.smooth_l1_loss(center_error, torch.zeros_like(center_error),
                                       reduction='none').mean(-1)
        refine_losses.append((_mean(center_loss, refine_valid), refine_valid.any()))
        if measurement is result.local:
            for i in range(2):
                candidate_labels.append(_identity_label(result.legacy_members[:, i], labels))
                candidate_masks.append(result.legacy_members[:, i].any(-1))
                candidate_scores.append(result.legacy_match_logits[:, i])
        candidate_labels.append(_identity_label(measurement.members, labels))
        candidate_masks.append(measurement.members.any(-1))
        candidate_scores.append(measurement.match_logit)
    labels = torch.stack(candidate_labels, -1)
    masks = torch.stack(candidate_masks, -1)
    match_loss = _balanced(torch.stack(candidate_scores, -1), labels, masks)
    swapped_loss = zero
    if result.swapped_logits is not None:
        track_id = batch['joint_track_id'].to(target.device).reshape(-1)
        different = track_id[:, None] != track_id[None, :]
        # 只交换已知是本车的候选：同帧背景/邻车可能恰好属于被换入的查询。
        pairs = ((labels & masks)[:, :, None] & different[:, None, :]
                 & result.template_valid[None, None, :])
        swapped_loss = _mean(F.softplus(result.swapped_logits), pairs)

    grid_object, grid_instance, grid_offset = [], [], []
    if result.coarse:
        sensor_target = target.clone()
        sensor_target[:, :3] += batch['anchor_box'][:, :3].to(target)
        target_values = sensor_target.cpu().numpy()
        sizes = target_size.cpu().numpy()
        neighbors = batch['joint_other_cars']
        for record in result.coarse:
            i = record.row
            vehicle, own, offset = grid_labels(record.scan, target_values[i], sizes[i], neighbors[i])
            vehicle = torch.as_tensor(vehicle, device=target.device)
            own = torch.as_tensor(own, device=target.device)
            grid_object.append((_balanced(record.class_logit, vehicle, torch.ones_like(vehicle)),
                                torch.ones((), dtype=torch.bool, device=target.device)))
            # 直接训练全部有车回波的体素与最像目标的背景，不依赖离散前三命中。
            background = (~vehicle).nonzero(as_tuple=True)[0]
            hard = record.instance_logit[background].detach().topk(min(32, len(background))).indices
            candidates = vehicle.clone()
            candidates[background[hard]] = True
            eligible = candidates & result.template_valid[i]
            grid_instance.append((_balanced(record.instance_logit, own, eligible), eligible.any()))
            error = (record.offsets - _tensor(offset, target)) / size[i]
            loss = F.smooth_l1_loss(error, torch.zeros_like(error), reduction='none').mean(-1)
            grid_offset.append((_mean(loss, vehicle), vehicle.any()))
    def average(items):
        if not items:
            return zero
        values, available = zip(*items)
        return _mean(torch.stack(values), torch.stack(available))
    terms = dict(loss_joint_foreground=average(point_losses), loss_joint_vote=average(vote_losses),
                 loss_joint_refine=average(refine_losses), loss_joint_match=match_loss,
                 loss_joint_swap=swapped_loss, loss_joint_grid_object=average(grid_object),
                 loss_joint_grid_match=average(grid_instance), loss_joint_grid_offset=average(grid_offset))
    terms['loss_joint_total'] = sum(JOINT_LOSS_WEIGHTS[name] * term for name, term in terms.items())
    return terms
