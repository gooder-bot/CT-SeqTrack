"""v30 六动作共用转移：不写递归状态、不读取 GT、也不消费全局 RNG。"""
from __future__ import annotations

import math
import torch

from utils.sampling_utils import stable_uint32_seed


def mechanism_behavior_policy(seed, epoch, tracklet_key, frame_index=None):
    """轨迹级 2 never / 1 evidence-top full / 1 uniform / 4 max-q>0。"""
    if (not isinstance(tracklet_key, str) or not tracklet_key.strip()
            or tracklet_key in ('unknown', 'eval')):
        raise ValueError('v30 behavior requires a real tracklet key')
    if int(epoch) < 0:
        raise ValueError('v30 behavior epoch must be nonnegative')
    bucket = stable_uint32_seed(int(seed), int(epoch), tracklet_key, 'mechanism_behavior_v30') % 8
    if bucket < 2:
        return {'kind': 'never'}
    if bucket == 2:
        return {'kind': 'always'}
    if bucket == 3:
        return {'kind': 'explore', 'action_seed': stable_uint32_seed(
            int(seed), int(epoch), tracklet_key, frame_index, 'mechanism_action_v30')}
    return {'kind': 'threshold', 'threshold': 0.}


def normalize_mode_policy(policy):
    kind = policy.get('kind')
    if kind in ('never', 'always'):
        return {'kind': kind}
    if kind == 'explore':
        return {'kind': kind, 'action_seed': int(policy['action_seed'])}
    if kind == 'threshold' and math.isfinite(float(policy.get('threshold', float('nan')))):
        return {'kind': kind, 'threshold': float(policy['threshold'])}
    raise ValueError('v30 policy requires never, always, finite threshold, or seeded explore')


@torch.no_grad()
def choose_mode_action(observation_box, action_boxes, action_valid, action_scores,
                       policy, *, evidence_top_index=None, mode_ids=None):
    """返回 final_box/applied/chosen_action_index/best_action_index。

    always 固定选择 B2 纯加权支持×紧致度排名第一的全幅（默认槽1），
    完全独立于质量头与 B3 q。threshold 仅按 q，精确并列先小位移再模式。
    chosen=-1 表示 observation；best 保留最大 q 的候选，便于同状态诊断。
    """
    policy = normalize_mode_policy(policy)
    observation, actions, scores = observation_box.detach(), action_boxes.detach(), action_scores.detach()
    if (observation.ndim != 2 or observation.shape[1] < 4
            or actions.shape != (len(observation), 6, observation.shape[1])
            or scores.shape != (len(observation), 6)
            or action_valid.shape != scores.shape):
        raise ValueError('v30 transition requires observation[B,D], actions[B,6,D], validity/scores[B,6]')
    batch = len(observation)
    valid = (action_valid.detach().bool() & torch.isfinite(actions).all(-1)
             & torch.isfinite(observation).all(-1)[:, None])
    norm = torch.linalg.norm(actions[..., :2] - observation[:, None, :2], dim=-1)
    valid &= norm > 1e-6
    scored_valid = valid & torch.isfinite(scores)
    # 初始槽序为模式0半/全、模式1半/全、模式2半/全。可显式提供稳定模式 ID。
    if mode_ids is None:
        order = torch.arange(6, device=scores.device)[None].expand(batch, -1)
    else:
        if mode_ids.shape != (batch, 3):
            raise ValueError('mode_ids must have shape [B,3]')
        order = torch.argsort(mode_ids.repeat_interleave(2, 1), stable=True)
    order = order.gather(1, torch.argsort(norm.gather(1, order), stable=True))
    ranked = scores.masked_fill(~scored_valid, -torch.inf)
    order = order.gather(1, torch.argsort(ranked.gather(1, order), descending=True, stable=True))
    best = order[:, 0].masked_fill(~scored_valid.any(1), -1)
    rows = torch.arange(batch, device=scores.device)
    if policy['kind'] == 'always':
        top = (torch.zeros(batch, dtype=torch.long, device=scores.device)
               if evidence_top_index is None else evidence_top_index.detach().long())
        chosen = top * 2 + 1
        legal_top = (top >= 0) & (top < 3)
        chosen = chosen.masked_fill(~legal_top | ~valid[rows, chosen.clamp(0, 5)], -1)
    elif policy['kind'] == 'never':
        chosen = torch.full_like(best, -1)
    elif policy['kind'] == 'explore':
        # host 每条真实轨迹/帧传入独立 seed；固定六槽，不产生动态 nonzero。
        # CUDA 不支持 bool 排序；0/1 整数键保持合法槽在前及原槽稳定顺序。
        legal_order = torch.argsort((~valid).to(torch.int64), stable=True)
        count = valid.sum(1)
        offsets = (policy['action_seed'] + rows * 2654435761) % count.clamp_min(1)
        chosen = legal_order.gather(1, offsets[:, None]).squeeze(1).masked_fill(count == 0, -1)
    else:
        chosen = best.clone()
        chosen = chosen.masked_fill((best < 0) | ~(scores[rows, best.clamp_min(0)] > policy['threshold']), -1)
    applied = chosen >= 0
    candidate = actions[rows, chosen.clamp_min(0)]
    final = torch.cat((torch.where(applied[:, None], candidate[:, :2], observation[:, :2]), observation[:, 2:]), 1)
    return dict(final_box=final, applied=applied, chosen_action_index=chosen,
                best_action_index=best, action_valid=valid,
                max_action_score=ranked[rows, best.clamp_min(0)].masked_fill(best < 0, 0.))
