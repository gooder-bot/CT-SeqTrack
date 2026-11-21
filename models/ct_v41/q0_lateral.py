"""V41 当前实采点的米制横向投票；使用原 q0 预测轴定义左右。"""
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .base_losses import frame_endpoint_mean, masked_mean
from .geometry import rotate_xyz


@dataclass
class LateralEstimate:
    base_box: Tensor        # 原 decoder 的可微 q0，供原中心与 yaw 监督
    box: Tensor             # 唯一向 B3 提出的 q0
    suggestion: Tensor      # d，米，已按首框宽限幅
    gate: Tensor            # 有效观测的采用比例，预热时同样接受监督
    applied: Tensor         # 本帧实际施加的有符号位移，预热时为零
    support_count: Tensor   # 参与投票的唯一 raw ID 数
    support: Tensor
    valid: Tensor           # 去重后的当前实采点，点监督无需通过预测前景阈值
    point_displacement: Tensor
    gate_logits: Tensor


class Q0LateralHead(nn.Module):
    def __init__(self):
        super().__init__()
        # 车体系 XYZ、实采特征、BoxCloud、预测前景概率、当前实例上下文。
        self.point_head = nn.Sequential(nn.Linear(3 + 64 + 9 + 1 + 64, 64),
                                        nn.GELU(), nn.Linear(64, 1))
        self.gate_head = nn.Sequential(nn.Linear(5, 16), nn.GELU(), nn.Linear(16, 1))
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.constant_(self.gate_head[-1].bias, -3.)

    def forward(self, decoder, observation, pool, size, *, apply):
        base = decoder.q0_box
        # 读出及监督共用冻结的原 q0 轴，避免新损失通过旋转监督坐标本身获益。
        local = rotate_xyz(pool['xyz'] - base.detach()[:, None, :3], base.detach()[:, 3])
        valid = pool['valid'] & observation.sequence_valid[:, None]
        probability = pool['probability']
        inputs = torch.cat((local, pool['features'], observation.bc_prediction[:, -1],
                            probability[..., None],
                            decoder.q0_context[:, None].expand(-1, local.shape[1], -1)), -1)
        point_displacement = self.point_head(torch.where(valid[..., None], inputs, 0.)).squeeze(-1)
        # valid 已按 raw ID 去重；每个当前点最多投一票。
        support = valid & (probability >= .5)
        count = support.sum(-1)
        enough = count >= 3
        weights = torch.where(support, probability, 0.)
        votes = local[..., 1] + point_displacement
        ordered_votes, order = votes.sort(dim=-1, stable=True)
        ordered_weight = weights.gather(-1, order)
        # FP32 的 [.5, 1] 都是 2^-24 的整数倍。整数前缀和能精确取半权重，
        # 同时满足 CUDA deterministic_algorithms 对 cumsum 的要求。
        cumulative = (ordered_weight * (1 << 24)).long().cumsum(-1)
        index = (2 * cumulative < cumulative[:, -1:]).sum(-1, keepdim=True)
        median = ordered_votes.gather(-1, index).squeeze(-1)
        width = size[:, 1]
        d = torch.where(enough, median.clamp(-width, width), 0.)

        # 采用比例仅评价已经提出的估计。点头从逐点 GT 学习，不随门的损失移动。
        with torch.no_grad():
            xy = local[..., :2]
            maximum = xy.masked_fill(~support[..., None], -torch.inf).amax(1)
            minimum = xy.masked_fill(~support[..., None], torch.inf).amin(1)
            span = torch.where(enough[:, None], maximum - minimum, 0.) / size[:, :2]
            disagreement = (weights * (votes - median[:, None]).abs()).sum(-1)
            disagreement = disagreement / weights.sum(-1).clamp_min(1.)
            gate_input = torch.cat((count.to(base).log1p()[:, None], span,
                                    (disagreement / width)[:, None], (d.abs() / width)[:, None]), -1)
        logits = self.gate_head(gate_input).squeeze(-1)
        gate = torch.where(enough, logits.sigmoid(), 0.)
        applied = gate * d if apply else torch.zeros_like(d)
        axis = torch.stack((-base[:, 3].detach().sin(), base[:, 3].detach().cos()), -1)
        shifted = torch.cat((base[:, :2] + applied[:, None] * axis, base[:, 2:]), -1)
        box = torch.where(enough[:, None], shifted, base) if apply else base
        return LateralEstimate(base, box, d, gate, applied, count, support, valid,
                               point_displacement, logits)


def lateral_losses(estimate, batch):
    base = estimate.base_box.detach()
    axis = torch.stack((-base[:, 3].sin(), base[:, 3].cos()), -1)
    target = batch['target_box'].detach()[:, :2]
    positive = batch['segmentation_labels'][:, -1] > .5
    supervised = estimate.valid & positive
    points = torch.where(estimate.valid[..., None], batch['points'][:, -1, :, :2].detach(), 0.)
    point_target = ((target[:, None] - points) * axis[:, None]).sum(-1)
    terms = F.smooth_l1_loss(estimate.point_displacement, point_target, reduction='none')
    point_loss = frame_endpoint_mean(terms, supervised)

    needed = ((target - base[:, :2]) * axis).sum(-1)
    enough = estimate.support_count >= 3
    has_target_vote = (estimate.support & positive).any(-1)
    motion = estimate.gate * estimate.suggestion.detach()
    fit = F.smooth_l1_loss(motion, needed, reduction='none')
    # 即使 d 恰为 0，纯背景支持仍明确监督拒绝。无建议的帧不更新采用比例。
    reject = F.softplus(estimate.gate_logits)
    gate_loss = masked_mean(torch.where(has_target_vote, fit, reject), enough)
    return dict(loss_q0_lateral_point=point_loss, loss_q0_lateral_gate=gate_loss)
