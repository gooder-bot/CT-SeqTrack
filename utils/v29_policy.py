"""v29 机制行为分配与训练/部署共用的纯动作转移，不维护递归状态。"""
import torch

from utils.action_calibration_v27 import normalize_policy
from utils.sampling_utils import stable_uint32_seed
from utils.v29_contracts import V29_CONTRACTS


BEHAVIOR_NAMESPACE = 'mechanism_behavior_v1'
BEHAVIOR_CONTRACT = V29_CONTRACTS['ct_mechanism_behavior_contract']
TRANSITION_CONTRACT = V29_CONTRACTS['ct_state_transition_contract']
UTILITY_TARGET_CONTRACT = V29_CONTRACTS['ct_b3_target_contract']
POLICY_FITTING_CONTRACT = V29_CONTRACTS['ct_policy_fitting_contract']


def mechanism_behavior_policy(seed, epoch, tracklet_key):
    """epoch使用从0开始的训练轮次；真实轨迹key决定整条轨迹的行为类型。"""
    if not isinstance(tracklet_key, str) or not tracklet_key.strip():
        raise ValueError('v29 mechanism behavior requires a real tracklet key')
    if tracklet_key in ('eval', 'unknown'):
        raise ValueError('v29 mechanism behavior forbids placeholder tracklet keys')
    if int(epoch) < 0:
        raise ValueError('v29 behavior epoch must be nonnegative')
    bucket = stable_uint32_seed(int(seed), int(epoch), tracklet_key,
                                BEHAVIOR_NAMESPACE) % 4
    if bucket == 0:
        return {'kind': 'never'}
    if bucket == 1:
        return {'kind': 'always'}
    return {'kind': 'threshold', 'threshold': 0.}


def policy_transition(observation_box, bounded_candidate_box,
                      structural_valid, action_score, policy):
    """同状态完整框[B,D] -> (最终框, accepted[B])；不消费GT/presence/RNG。

    bounded_candidate_box已完成动作半径约束；这里不再缩放动作。
    训练和部署都只替换XY，保留B0的Z/yaw及其他框字段。
    """
    policy = normalize_policy(policy)
    if (observation_box.ndim != 2 or observation_box.shape[1] < 4
            or bounded_candidate_box.shape != observation_box.shape):
        raise ValueError('v29 policy transition requires matching boxes [B,D>=4]')
    if (bounded_candidate_box.device != observation_box.device
            or bounded_candidate_box.dtype != observation_box.dtype):
        raise ValueError('v29 transition boxes must share device and dtype')
    batch = len(observation_box)
    if structural_valid.numel() != batch or action_score.numel() != batch:
        raise ValueError('v29 transition requires one validity and score per row')
    observation = observation_box.detach()
    candidate = bounded_candidate_box.detach()
    score = action_score.detach().reshape(-1).to(observation)
    available = structural_valid.detach().reshape(-1).to(observation.device)
    valid = ((available > 0) & torch.isfinite(available)
             & torch.isfinite(observation).all(1)
             & torch.isfinite(candidate).all(1) & torch.isfinite(score))
    if policy['kind'] == 'never':
        accepted = torch.zeros_like(valid)
    elif policy['kind'] == 'always':
        accepted = valid
    else:
        accepted = valid & (score > policy['threshold'])
    final = torch.cat((torch.where(accepted[:, None], candidate[:, :2],
                                   observation[:, :2]), observation[:, 2:]), dim=1)
    return final, accepted
