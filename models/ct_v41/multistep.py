"""M 仅用于训练：从 t−1 的因果 context 监督第 2/4 个采样端点。"""
import torch
from torch import nn
from torch.nn import functional as F


def box_scale(box_size):
    if box_size.ndim != 2 or box_size.shape[-1] != 3:
        raise ValueError('box_size must be [B,3]')
    return (box_size.float()[:, 0].clamp_min(0) * box_size.float()[:, 1].clamp_min(0)).sqrt().clamp_min(1.)


class FutureMotionHead(nn.Module):
    def __init__(self, time_scale=.5):
        super().__init__()
        if time_scale <= 0:
            raise ValueError('M time_scale must be positive')
        self.time_scale = float(time_scale)
        self.head = nn.Sequential(nn.Linear(130, 64), nn.LayerNorm(64), nn.ReLU(), nn.Linear(64, 2))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, context, current_dt, query_dt, box_size):
        if context.ndim != 2 or context.shape[-1] != 128 or query_dt.shape != (len(context), 2):
            raise ValueError('M requires context[B,128] and query_dt[B,2] for horizons 2/4')
        if current_dt.numel() != len(context) or box_size.shape != (len(context), 3):
            raise ValueError('M time/size batch shape mismatch')
        with torch.autocast(device_type=context.device.type, enabled=False):
            context, query_dt = context.float(), query_dt.to(context.device, torch.float32)
            current_dt = current_dt.to(context.device, torch.float32).reshape(-1)
            if (not bool(torch.isfinite(query_dt).all()) or bool((query_dt < 0).any())
                    or not bool(torch.isfinite(current_dt).all()) or bool((current_dt <= 0).any())):
                raise ValueError('M requires finite nonnegative query_dt and positive current_dt')
            q = torch.stack((torch.log1p(query_dt / self.time_scale),
                             torch.log1p(query_dt / current_dt[:, None])), -1)
            features = torch.cat((context[:, None].expand(-1, 2, -1), q), -1)
            return self.head(features) * box_scale(box_size.to(context))[:, None, None]


def multistep_loss(predicted, target, valid, box_size, prior_valid=None, weights=(1., .5)):
    if predicted.ndim != 3 or predicted.shape[1:] != (2, 2) or target.shape != predicted.shape:
        raise ValueError('M displacement must be [B,2,2]')
    if valid.shape != predicted.shape[:2]:
        raise ValueError('M validity must be [B,2]')
    mask = valid.bool()
    if prior_valid is not None:
        if prior_valid.shape != (len(predicted),):
            raise ValueError('M prior validity must be [B]')
        mask = mask & prior_valid[:, None].bool()
    weights = predicted.new_tensor(weights, dtype=torch.float32)
    if weights.shape != (2,) or not bool(torch.isfinite(weights).all()) or bool((weights <= 0).any()):
        raise ValueError('M requires two finite positive horizon weights')
    # 无效尾部允许未填充/NaN 标签，但它们不能污染损失及梯度。
    truth = torch.where(mask[..., None], target.detach().to(predicted), 0.)
    prediction = torch.where(mask[..., None], predicted, 0.)
    if not bool(torch.isfinite(truth).all()):
        raise ValueError('valid M targets must be finite')
    scale = box_scale(box_size.to(predicted))[:, None, None]
    terms = F.smooth_l1_loss(prediction.float() / scale, truth.float() / scale,
                             reduction='none', beta=1.).mean(-1)
    weighted_valid = mask.to(terms) * weights[None]
    per_endpoint = (terms * weighted_valid).sum(-1) / weighted_valid.sum(-1).clamp_min(1e-12)
    active = mask.any(-1)
    return (per_endpoint * active).sum() / active.sum().clamp_min(1)


def compute_multistep_loss(batch, prior, head, *, weight=.05, horizon_weights=(1., .5)):
    """仅训练 loss 调用；主先验从不读取此处的未来标签或查询时间。"""
    valid = batch['future_valid'].bool() & prior.valid[:, None].bool()
    query_dt = torch.where(valid, batch['future_dt'], 0.)
    if bool((query_dt[valid] <= 0).any()):
        raise ValueError('valid M future_dt must be positive')
    predicted = head(prior.feature, batch['current_dt'], query_dt, batch['box_size'])
    loss = multistep_loss(predicted, batch['future_displacement'], valid, batch['box_size'],
                          weights=horizon_weights)
    target = torch.where(valid[..., None], batch['future_displacement'].to(predicted), 0.)
    error = torch.where(valid[..., None], predicted.detach() - target, 0.).norm(dim=-1)
    result = dict(loss_m=loss, loss_contribution_m=float(weight)*loss,
                  m_valid_endpoints=valid.any(-1).sum().detach())
    for index, horizon in enumerate((2, 4)):
        result[f'm_h{horizon}_error_m'] = (error[:, index].sum() / valid[:, index].sum().clamp_min(1)).detach()
        result[f'm_h{horizon}_valid'] = valid[:, index].sum().detach()
    return result
