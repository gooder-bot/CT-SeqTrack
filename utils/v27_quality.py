"""仅从过去观测预测产生质量摘要，不读取 GT。"""
import torch


def observation_quality(seg_logits, unique_mask, raw_point_count):
    batch, frames, points = unique_mask.shape
    probability = seg_logits.detach().softmax(1).reshape(batch, 2, frames, points)[:, :, -1]
    unique = unique_mask[:, -1].to(probability.dtype)
    count = unique.sum(-1)
    foreground = (probability[:, 1] * unique).sum(-1) / count.clamp_min(1)
    entropy = -(probability * probability.clamp_min(1e-8).log()).sum(1)
    entropy = (entropy * unique).sum(-1) / count.clamp_min(1)
    valid = (count > 0).to(probability.dtype)
    return torch.stack((raw_point_count[:, -1].to(probability.dtype), foreground, entropy, valid), -1).detach()
