"""Training-only contracts for closed-loop risk-aware proposal arbitration."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

import torch
import torch.nn.functional as F


CRPA_ROLLOUT_SCHEMA = "ct_seqtrack.b3_crpa_rollout.v1"
CRPA_ROUTER_SCHEMA = "ct_seqtrack.b3_crpa_router.v1"


def counterfactual_gain_targets(
        observation_xy: torch.Tensor,
        target_xy: torch.Tensor,
        candidate_residual_xy: torch.Tensor,
        candidate_valid: torch.Tensor,
        step_cap: torch.Tensor,
        eps: float = 1e-6) -> dict[str, torch.Tensor]:
    """Return the best bounded step and gain along each candidate direction.

    Shapes are ``[B,2]`` for candidate-valued outputs and ``[B,2,2]`` for
    residuals.  The residual is assumed to have already passed the same radius
    clipping used by online inference.
    """
    if observation_xy.dim() != 2 or observation_xy.shape[1] != 2:
        raise ValueError("observation_xy must have shape [B,2]")
    if target_xy.shape != observation_xy.shape:
        raise ValueError("target_xy must match observation_xy")
    if (candidate_residual_xy.dim() != 3
            or candidate_residual_xy.shape[1:] != (2, 2)):
        raise ValueError("candidate_residual_xy must have shape [B,2,2]")
    candidate_valid = candidate_valid.to(
        device=observation_xy.device, dtype=observation_xy.dtype)
    if candidate_valid.shape != observation_xy.shape:
        raise ValueError("candidate_valid must have shape [B,2]")
    step_cap = torch.as_tensor(
        step_cap, device=observation_xy.device,
        dtype=observation_xy.dtype).reshape(-1, 1)
    if step_cap.shape[0] not in (1, observation_xy.shape[0]):
        raise ValueError("step_cap must have one value per sample")
    if step_cap.shape[0] == 1 and observation_xy.shape[0] > 1:
        step_cap = step_cap.repeat(observation_xy.shape[0], 1)
    step_cap = torch.clamp(step_cap, min=0.0, max=1.0)

    target_offset = target_xy - observation_xy
    numerator = (
        target_offset.unsqueeze(1) * candidate_residual_xy).sum(dim=2)
    denominator = candidate_residual_xy.pow(2).sum(dim=2) + float(eps)
    alpha = torch.clamp(numerator / denominator, min=0.0)
    alpha = torch.minimum(alpha, step_cap.expand_as(alpha))
    alpha = alpha * candidate_valid
    candidate_xy = (
        observation_xy.unsqueeze(1)
        + alpha.unsqueeze(2) * candidate_residual_xy)
    observation_error = torch.linalg.norm(
        observation_xy - target_xy, dim=1, keepdim=True)
    candidate_error = torch.linalg.norm(
        candidate_xy - target_xy.unsqueeze(1), dim=2)
    gain = torch.clamp(observation_error - candidate_error, min=0.0)
    gain = gain * candidate_valid
    normalized_step = alpha / torch.clamp(
        step_cap.expand_as(alpha), min=float(eps))
    normalized_step = normalized_step * candidate_valid
    return {
        "oracle_gain": gain,
        "oracle_alpha": alpha,
        "oracle_step_ratio": normalized_step,
        "oracle_candidate_xy": candidate_xy,
        "observation_error": observation_error.squeeze(1),
        "oracle_candidate_error": candidate_error,
    }


def pinball_loss(
        prediction: torch.Tensor,
        target: torch.Tensor,
        quantile: float) -> torch.Tensor:
    if not 0.0 < float(quantile) < 1.0:
        raise ValueError("quantile must be in (0,1)")
    residual = target - prediction
    return torch.maximum(
        float(quantile) * residual,
        (float(quantile) - 1.0) * residual)


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=value.device, dtype=value.dtype)
    return (value * mask).sum() / torch.clamp(mask.sum(), min=1.0)


def crpa_router_loss(
        prediction: dict[str, torch.Tensor],
        oracle_gain: torch.Tensor,
        oracle_step_ratio: torch.Tensor,
        candidate_valid: torch.Tensor,
        step_gain_margin: float = 0.05,
        ranking_margin: float = 0.02) -> dict[str, torch.Tensor]:
    """Compute the fixed CRPA offline objective from rollout labels."""
    q10 = prediction["q10"]
    q50 = prediction["q50"]
    step_ratio = prediction["step_ratio"]
    for value in (oracle_gain, oracle_step_ratio, candidate_valid):
        if value.shape != q10.shape:
            raise ValueError("CRPA labels must match the [B,2] router heads")
    valid = candidate_valid.to(q10.dtype)
    loss_q10 = _masked_mean(
        pinball_loss(q10, oracle_gain, 0.10), valid)
    loss_q50 = _masked_mean(
        pinball_loss(q50, oracle_gain, 0.50), valid)
    step_mask = valid * (oracle_gain >= float(step_gain_margin)).to(q10.dtype)
    loss_step = _masked_mean(
        F.smooth_l1_loss(
            step_ratio, oracle_step_ratio, reduction="none"),
        step_mask,
    )

    both_valid = candidate_valid.to(torch.bool).all(dim=1)
    gain_delta = oracle_gain[:, 0] - oracle_gain[:, 1]
    decisive = both_valid & (gain_delta.abs() > float(ranking_margin))
    direction = torch.sign(gain_delta)
    predicted_delta = q50[:, 0] - q50[:, 1]
    ranking_error = torch.relu(
        float(ranking_margin) - direction * predicted_delta)
    loss_rank = _masked_mean(ranking_error, decisive)
    loss_total = (
        loss_q10 + 0.5 * loss_q50
        + 0.5 * loss_step + 0.2 * loss_rank)
    return {
        "loss": loss_total,
        "loss_q10": loss_q10,
        "loss_q50": loss_q50,
        "loss_step": loss_step,
        "loss_rank": loss_rank,
        "step_supervision_rate": step_mask.mean(),
        "ranking_supervision_rate": decisive.to(q10.dtype).mean(),
    }


def stable_tracklet_partition(
        tracklet_keys: Iterable[str], seed: int = 42) -> dict[str, str]:
    """Deterministically assign whole tracklets to 70/15/15 partitions."""
    assignments: dict[str, str] = {}
    for raw_key in tracklet_keys:
        key = str(raw_key)
        digest = hashlib.sha256(f"{int(seed)}:{key}".encode("utf-8")).digest()
        value = int.from_bytes(digest[:8], byteorder="big") / float(2 ** 64)
        if value < 0.70:
            partition = "train"
        elif value < 0.85:
            partition = "dev"
        else:
            partition = "calibration"
        prior = assignments.setdefault(key, partition)
        if prior != partition:
            raise RuntimeError("tracklet partitioning is not deterministic")
    return assignments

