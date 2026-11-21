"""联合搜索的候选证据。每个框携带自己读取的点及原始扫描行号。"""
from dataclasses import dataclass, replace

import torch
from torch.nn import functional as F

from .geometry import inside_final_boxes


@dataclass
class JointCandidateEvidence:
    observations: object
    support_ids: torch.Tensor
    support_xyz: torch.Tensor
    support_mask: torch.Tensor
    identity_ok: torch.Tensor
    geometry_ok: torch.Tensor
    remote_proposed: torch.Tensor
    remote_valid: torch.Tensor

    @property
    def remote_box(self):
        return self.observations.remote.box.detach()

    def target_members(self, batch):
        """仅在损失或诊断中读取当前真值。坐标为相对同一锚点的传感器轴 XYZ。"""
        xyz = self.support_xyz
        size = batch.get('target_box_size', batch['box_size']).to(xyz)
        target = batch['target_box'].to(xyz)[:, None]
        if 'segmentation_labels' in batch:
            # 前五槽读相同有序局部池；沿用原始传感器精度下生成的局部标记。
            local = batch['segmentation_labels'][:, -1].to(xyz.device) > 0
            local = F.pad(local, (0, xyz.shape[2] - local.shape[1]))
            remote = inside_final_boxes(xyz[:, 5], target, size)
            return torch.cat((local[:, None].expand(-1, 5, -1), remote), 1) & (self.support_ids >= 0)
        inside = inside_final_boxes(xyz.flatten(1, 2), target, size)
        return inside.reshape_as(self.support_mask) & (self.support_ids >= 0)


def _pack(base, local, remote, *, xyz=False, fill=0):
    axis = 2 if xyz else -1
    width = max(value.shape[axis] for value in (base, local, remote))
    padded = []
    for value in (base, local, remote):
        pad = (0, width - value.shape[axis])
        padded.append(F.pad(value, (0, 0) + pad if xyz else pad, value=fill))
    return torch.cat(padded, 1).detach()


def remote_takeover_ready(remote, template_valid):
    support = (remote.support_mask & (remote.ids >= 0)).sum(-1)
    return (remote.valid & template_valid & remote.match_wins & remote.confirmed
            & (remote.logits[:, 2].detach() >= 0) & (support >= 3))


def combine_joint_candidates(candidates, verification, observations, pool, *, remote_position_valid=None):
    local, remote = observations.local, observations.remote
    ids = _pack(pool['ids'][:, None].expand(-1, 4, -1), local.ids[:, None],
                remote.ids[:, None], fill=-1)
    xyz = _pack(pool['xyz'][:, None].expand(-1, 4, -1, -1), local.xyz[:, None],
                remote.xyz[:, None], xyz=True)
    support = _pack(verification.support_mask, local.support_mask[:, None],
                    remote.support_mask[:, None]) & (ids >= 0)
    off = torch.zeros_like(local.valid)
    old_match = observations.legacy_match_wins
    identity = torch.stack((old_match[:, 0], off, off, old_match[:, 1],
                            local.match_wins, remote.match_wins), 1).detach()
    logits = torch.cat((verification.logits, local.logits[:, None], remote.logits[:, None]), 1)
    raw = torch.stack((local.valid, remote.valid), 1).detach()
    measurement = torch.cat((candidates.measurement_candidate, raw), 1)
    geometry = (logits[..., 2].detach() >= 0) & measurement
    remote_support = support[:, 5].sum(-1)
    identified = remote.valid & observations.template_valid & remote.match_wins
    # 首框稀疏时，框内一个实际前景点即可暂存到下一次扫描复查。
    proposed = identified | (remote.valid & ~observations.template_valid & (remote_support > 0))
    # 原生前向传入本次身份竞争使用的资格；独立重组时按调用者当前证据复算。
    if remote_position_valid is None:
        remote_position_valid = remote_takeover_ready(remote, observations.template_valid)
    evidence = JointCandidateEvidence(observations, ids, xyz, support, identity, geometry,
                                      proposed.detach(), remote_position_valid.detach())
    valid = torch.stack((local.valid & observations.local_position_valid, remote_position_valid), 1).detach()
    boxes = torch.stack((local.box.detach(), remote.box.detach()), 1)
    combined = replace(candidates, boxes=torch.cat((candidates.boxes, boxes), 1),
        valid=torch.cat((candidates.valid, valid), 1), raw_valid=torch.cat((candidates.raw_valid, raw), 1),
        alias_of=torch.cat((candidates.alias_of, torch.full_like(raw, -1, dtype=torch.long)), 1),
        measurement_candidate=measurement)
    # metadata 和局部邻居读数来自四槽 verifier，新两槽的读数归 observations 所有。
    return combined, replace(verification, logits=logits, support_mask=support), evidence
