"""按当前 B0 支持修正 q0。监督只用于自然训练前态的轮末拟合。"""
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from .geometry import rotate_xyz


FEATURE_COUNT = 42
SHIFT_CAP = (.65, .38, .30)  # q0 车体系中的米


@dataclass
class SupportCenterEstimate:
    raw_box: Tensor
    box: Tensor
    features: Tensor
    count: Tensor
    high_count: Tensor
    branch: Tensor          # 0 原 q0、1 密集中心回归、2 稀疏局部融合
    local_distance: Tensor   # 局部票到其 q0 种子的距离，无有效票时为零


@torch.no_grad()
def center_features(batch, prior, observation, box):
    """以原 q0 轴和首框 L/W/H 表示点分布，顺序与冻结原型相同。"""
    size, yaw = batch['box_size'], box[:, 3]
    valid = batch['point_valid'][:, -1]
    probability = observation.foreground_probability[:, -1]
    origin = box[:, :3]
    sine, cosine = -yaw.sin()[:, None], yaw.cos()[:, None]

    def to_body(xyz):
        x, y, z = xyz.unbind(-1)
        return torch.stack((cosine * x - sine * y, sine * x + cosine * y, z), dim=-1)

    body = to_body(batch['points'][:, -1, :, :3] - origin[:, None]) / size[:, None]
    local = valid & (body.abs() <= .8).all(-1)
    foreground = local & (probability >= .5)
    high = local & (probability >= .85)
    counts = torch.stack((foreground, high, local, valid), dim=1).sum(-1)
    count, high_count = counts[:, 0], counts[:, 1]
    soft = probability.square() * local
    denominator = soft.sum(-1, keepdim=True).clamp_min(.01)
    avg = (body * soft[..., None]).sum(1) / denominator
    variance = ((body - avg[:, None]).square() * soft[..., None]).sum(1)
    sd = (variance / denominator).sqrt()
    high_mean = (body * high[..., None]).sum(1) / high_count.float()[:, None].clamp_min(.01)
    absent = ~foreground[..., None]
    lower = body.masked_fill(absent, 3.).amin(1)
    upper = body.masked_fill(absent, -3.).amax(1)
    has_foreground = count[:, None] > 0

    # 四个参考位移和当前视线共享一个车体系，按原特征顺序一起换轴。
    world = origin + batch['anchor_box'][:, :3]
    references = torch.stack((prior.box[:, :3], batch['fallback_box'][:, :3],
                              observation.coarse_box[:, :3], batch['history_boxes'][:, -1, :3]), dim=1)
    vectors = to_body(torch.cat((references - origin[:, None], -world[:, None]), dim=1))
    relative = (vectors[:, :4] / size[:, None]).clamp(-2., 2.).flatten(1)
    distance = world.norm(dim=-1, keepdim=True).clamp_min(.01)
    pieces = (avg, sd, high_mean, torch.where(has_foreground, lower, 0.), torch.where(has_foreground, upper, 0.),
              relative, vectors[:, 4] / distance, (distance / 50.).clamp_max(3.), size / 4., prior.log_sigma,
              prior.valid.float()[:, None], torch.log1p(counts) / 7.,
              batch['consecutive_untrusted_outputs'].float().clamp_max(8)[:, None] / 8.)
    return torch.cat(pieces, dim=-1), count, high_count


def _fit_center(x, target, weight):
    """CPU float64，三个轴分别做八轮 Huber IRLS，截距不受岭惩罚。"""
    mu = x.mean(0)
    scale = x.std(0).clip(.04)
    standardized = ((x - mu) / scale).clip(-4., 4.)
    design = np.column_stack((standardized, np.ones(len(x))))
    penalty = np.eye(design.shape[1]) * len(x) * .018
    penalty[-1, -1] = 0.
    cap = np.asarray(SHIFT_CAP)
    target = target.clip(-2 * cap, 2 * cap)
    betas = np.zeros((design.shape[1], 3))
    for axis in range(3):
        for _ in range(8):
            error = target[:, axis] - design @ betas[:, axis]
            weights = weight * np.minimum(1., .18 / np.maximum(np.abs(error), 1e-8))
            betas[:, axis] = np.linalg.solve(design.T @ (design * weights[:, None]) + penalty,
                                             design.T @ (target[:, axis] * weights))
    return dict(mu=mu, scale=scale, coefs=betas[:-1], bias=betas[-1], cap=cap)


class _CenterWeights(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('mu', torch.zeros(FEATURE_COUNT))
        self.register_buffer('scale', torch.ones(FEATURE_COUNT))
        self.register_buffer('coefs', torch.zeros(FEATURE_COUNT, 3))
        self.register_buffer('bias', torch.zeros(3))
        self.register_buffer('cap', torch.tensor(SHIFT_CAP))
        self.register_buffer('fitted', torch.tensor(False))

    @torch.no_grad()
    def store(self, fields):
        if set(fields) != {'mu', 'scale', 'coefs', 'bias', 'cap'}:
            raise ValueError('invalid support-center calibration fields')
        converted = {}
        for key in fields:
            dest = getattr(self, key)
            value = torch.as_tensor(fields[key], dtype=dest.dtype, device=dest.device)
            if value.shape != dest.shape or not torch.isfinite(value).all():
                raise ValueError('invalid support-center calibration field ' + key)
            converted[key] = value
        if (converted['scale'] < .04).any() or not torch.equal(converted['cap'], self.cap.new_tensor(SHIFT_CAP)):
            raise ValueError('invalid support-center scale or displacement cap')
        for key, value in converted.items():
            getattr(self, key).copy_(value)
        self.fitted.fill_(True)


class SupportCenter(nn.Module):
    def __init__(self):
        super().__init__()
        self.active = _CenterWeights()
        self.pending = _CenterWeights()
        self._rows = []  # 仅同一轮的自然辅助步；每轮约 4,770 行，不保存训练图。

    @torch.no_grad()
    def before_vote(self, batch, prior, observation, raw):
        features, count, high = center_features(batch, prior, observation, raw)
        active = self.active
        dense = active.fitted & (high >= 10)
        scaled = ((features - active.mu) / active.scale).clamp(-4., 4.)
        delta = (scaled @ active.coefs + active.bias).clamp(-active.cap, active.cap)
        amount = .5 * (count / (count + 6.)) * (count >= 3) * dense
        world_delta = rotate_xyz(delta * amount[:, None], raw[:, 3], to_world=True)
        box = torch.cat((raw[:, :3] + world_delta, raw[:, 3:]), dim=-1)
        return SupportCenterEstimate(raw, box, features, count, high, dense.long(), torch.zeros_like(amount))

    @torch.no_grad()
    def after_vote(self, estimate, observation, local):
        seed = estimate.box
        delta = local.box[:, :3] - seed[:, :3]
        distance = delta.norm(dim=-1)
        sparse = (self.active.fitted & (estimate.high_count < 10) & local.valid
                  & observation.current_valid & (distance <= 1.))
        estimate.box = torch.cat((seed[:, :3] + .3 * delta * sparse[:, None], seed[:, 3:]), dim=-1)
        estimate.branch = torch.where(sparse, 2, estimate.branch)
        estimate.local_distance = torch.where(local.valid, distance, 0.)

    @torch.no_grad()
    def observe(self, estimate, batch):
        raw = estimate.raw_box
        target = rotate_xyz(batch['target_box'][:, :3] - raw[:, :3], raw[:, 3])
        visible = ((batch['segmentation_labels'][:, -1] > 0) & batch['point_valid'][:, -1]).any(-1)
        weight = torch.where(visible, 1., .25)
        # 等到轮末合并再筛选，避免每一辅助批在 CUDA 上做动态索引的同步。
        rows = torch.cat((estimate.features, target, weight[:, None], (estimate.count >= 3)[:, None]), dim=-1)
        self._rows.append(rows.detach())

    @torch.no_grad()
    def start_epoch(self):
        if self.pending.fitted.item():
            self.active.load_state_dict(self.pending.state_dict())
        self._rows.clear()

    @torch.no_grad()
    def fit_pending(self):
        count = 0
        if self._rows:
            rows = torch.cat(self._rows).cpu().double().numpy()
            eligible = rows[rows[:, -1] > 0]
            count = len(eligible)
            if count:
                fields = _fit_center(eligible[:, :FEATURE_COUNT], eligible[:, FEATURE_COUNT:-2], eligible[:, -2])
                self.pending.store(fields)
        if not count:
            self.pending.load_state_dict(self.active.state_dict())
        self._rows.clear()
        return dict(samples=count, fitted=bool(self.pending.fitted.item()), active_fitted=bool(self.active.fitted.item()))

    def load_archive(self, path):
        """冻结验证直接读原型系数，常规训练只在轮末拟合。"""
        if self.training:
            raise ValueError('loading frozen support-center coefficients requires eval mode')
        with np.load(path, allow_pickle=False) as data:
            self.active.store(data)
