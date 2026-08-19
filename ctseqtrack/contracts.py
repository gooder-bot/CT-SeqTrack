"""Typed contracts for the paper-facing CT-SeqTrack B0--B3 pipeline.

The tracker still exposes a flat dictionary to the legacy evaluator.  These
containers define ownership inside the model and make it explicit which
values may cross a module boundary.  Tensor validation is intentionally
lightweight so the contracts can also be used by unit tests and exporters.
"""

from dataclasses import dataclass
import math
from typing import Dict, Optional

import torch


Tensor = torch.Tensor


@dataclass(frozen=True)
class ObservationOutput:
    box: Tensor
    statistics: Tensor
    current_features: Optional[Tensor] = None
    history_features: Optional[Tensor] = None

    def detached(self):
        return ObservationOutput(
            box=self.box.detach(),
            statistics=self.statistics.detach(),
            current_features=(
                None
                if self.current_features is None
                else self.current_features.detach()
            ),
            history_features=(
                None
                if self.history_features is None
                else self.history_features.detach()
            ),
        )


@dataclass(frozen=True)
class MotionPriorOutput:
    center_xy: Tensor
    direction_xy: Tensor
    log_sigma: Tensor
    valid: Tensor
    source: Tensor
    mode_centers_xy: Optional[Tensor] = None
    mode_probabilities: Optional[Tensor] = None
    motion_quantiles_pp: Optional[Tensor] = None
    support_quantiles_pp: Optional[Tensor] = None
    recoverability_probability: Optional[Tensor] = None
    expert_disagreement: Optional[Tensor] = None
    residual_acceleration_pp: Optional[Tensor] = None
    residual_gate: Optional[Tensor] = None

    def detached(self):
        def detach(value):
            return None if value is None else value.detach()

        return MotionPriorOutput(
            center_xy=self.center_xy.detach(),
            direction_xy=self.direction_xy.detach(),
            log_sigma=self.log_sigma.detach(),
            valid=self.valid.detach(),
            source=self.source.detach(),
            mode_centers_xy=detach(self.mode_centers_xy),
            mode_probabilities=detach(self.mode_probabilities),
            motion_quantiles_pp=detach(self.motion_quantiles_pp),
            support_quantiles_pp=detach(self.support_quantiles_pp),
            recoverability_probability=detach(self.recoverability_probability),
            expert_disagreement=detach(self.expert_disagreement),
            residual_acceleration_pp=detach(self.residual_acceleration_pp),
            residual_gate=detach(self.residual_gate),
        )


def _rotate_xy(vectors, yaw):
    """Rotate batched xy row vectors by ``yaw`` radians."""
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)
    x, y = vectors.unbind(dim=1)
    return torch.stack(
        (
            cosine * x - sine * y,
            sine * x + cosine * y,
        ),
        dim=1,
    )


def reexpress_motion_prior(prior, source_anchor, target_anchor, *, degrees=False):
    """Express a physical B1 prior in another local SE(2) frame.

    B1 predicts a center displacement and motion direction in the newest
    recursive-box frame.  Auxiliary recovery candidates change the B0/B2 crop
    anchor without changing that physical B1 prediction.  Before B2 consumes
    the prior, its center and direction must therefore be re-expressed in the
    candidate crop frame.  Parallel/perpendicular uncertainty magnitudes are
    invariant under this rigid frame change because their direction is rotated
    with them.

    Anchors are ``[world_x, world_y, world_z, yaw]``.  Exact identity rows are
    copied from ``prior`` so candidate0 and online inference remain bitwise
    unchanged.
    """
    if not isinstance(prior, MotionPriorOutput):
        raise TypeError("prior must be a MotionPriorOutput")
    center = prior.center_xy
    direction = prior.direction_xy
    if center.ndim != 2 or center.shape[1] != 2:
        raise ValueError("motion prior center must have shape [B,2]")
    if direction.shape != center.shape:
        raise ValueError("motion prior direction must match center shape")
    if prior.log_sigma.shape != center.shape:
        raise ValueError("motion prior log sigma must match center shape")
    if not bool(
        torch.isfinite(center).all()
        and torch.isfinite(direction).all()
        and torch.isfinite(prior.log_sigma).all()
    ):
        raise ValueError("motion prior center/direction/log sigma must be finite")

    source = torch.as_tensor(source_anchor, device=center.device, dtype=center.dtype)
    target = torch.as_tensor(target_anchor, device=center.device, dtype=center.dtype)
    expected_anchor_shape = (center.shape[0], 4)
    if source.shape != expected_anchor_shape or target.shape != expected_anchor_shape:
        raise ValueError("motion/coordinate anchors must both have shape [B,4]")
    if not bool(torch.isfinite(source).all() and torch.isfinite(target).all()):
        raise ValueError("motion/coordinate anchors must be finite")

    # Float32 trigonometry avoids avoidable AMP error while preserving float64
    # tests and callers.  Results are cast back to the B1 output dtype below.
    compute_dtype = (
        torch.float32
        if center.dtype in (torch.float16, torch.bfloat16)
        else center.dtype
    )
    source_work = source.to(compute_dtype)
    target_work = target.to(compute_dtype)
    center_work = center.to(compute_dtype)
    direction_work = direction.to(compute_dtype)
    angle_scale = math.pi / 180.0 if degrees else 1.0
    source_yaw = source_work[:, 3] * angle_scale
    target_yaw = target_work[:, 3] * angle_scale

    world_center = source_work[:, :2] + _rotate_xy(center_work, source_yaw)
    converted_center = _rotate_xy(world_center - target_work[:, :2], -target_yaw).to(
        center.dtype
    )
    world_direction = _rotate_xy(direction_work, source_yaw)
    converted_direction = _rotate_xy(world_direction, -target_yaw).to(direction.dtype)

    identity = torch.all(source == target, dim=1, keepdim=True)
    converted_center = torch.where(identity, center, converted_center)
    converted_direction = torch.where(identity, direction, converted_direction)
    converted_mode_centers = prior.mode_centers_xy
    if converted_mode_centers is not None:
        if converted_mode_centers.shape != (center.shape[0], 3, 2):
            raise ValueError("motion mode centers must have shape [B,3,2]")
        flat_modes = converted_mode_centers.reshape(-1, 2).to(compute_dtype)
        repeated_source_center = (
            source_work[:, None, :2].expand(-1, 3, -1).reshape(-1, 2)
        )
        repeated_target_center = (
            target_work[:, None, :2].expand(-1, 3, -1).reshape(-1, 2)
        )
        repeated_source_yaw = source_yaw[:, None].expand(-1, 3).reshape(-1)
        repeated_target_yaw = target_yaw[:, None].expand(-1, 3).reshape(-1)
        world_modes = repeated_source_center + _rotate_xy(
            flat_modes, repeated_source_yaw
        )
        converted_mode_centers = (
            _rotate_xy(world_modes - repeated_target_center, -repeated_target_yaw)
            .reshape(-1, 3, 2)
            .to(prior.mode_centers_xy.dtype)
        )
        converted_mode_centers = torch.where(
            identity.unsqueeze(1), prior.mode_centers_xy, converted_mode_centers
        )
    return MotionPriorOutput(
        center_xy=converted_center,
        direction_xy=converted_direction,
        log_sigma=prior.log_sigma,
        valid=prior.valid,
        source=prior.source,
        mode_centers_xy=converted_mode_centers,
        mode_probabilities=prior.mode_probabilities,
        motion_quantiles_pp=prior.motion_quantiles_pp,
        support_quantiles_pp=prior.support_quantiles_pp,
        recoverability_probability=prior.recoverability_probability,
        expert_disagreement=prior.expert_disagreement,
        residual_acceleration_pp=prior.residual_acceleration_pp,
        residual_gate=prior.residual_gate,
    )


def motion_prior_covariance_xy(prior, *, eps=1e-6):
    """Reconstruct the xy covariance represented by a motion prior.

    ``log_sigma`` stores standard deviations along the learned motion
    direction and its perpendicular axis.  A rigid coordinate-frame change
    therefore rotates the basis but leaves both log standard deviations
    unchanged.  Keeping this reconstruction next to the SE(2) conversion
    makes that contract executable in tests and diagnostics.
    """
    if not isinstance(prior, MotionPriorOutput):
        raise TypeError("prior must be a MotionPriorOutput")
    if prior.direction_xy.ndim != 2 or prior.direction_xy.shape[1] != 2:
        raise ValueError("motion prior direction must have shape [B,2]")
    if prior.log_sigma.shape != prior.direction_xy.shape:
        raise ValueError("motion prior log sigma must match direction shape")
    if float(eps) <= 0:
        raise ValueError("covariance epsilon must be positive")

    direction = prior.direction_xy
    norm = torch.linalg.norm(direction, dim=1, keepdim=True)
    fallback = torch.zeros_like(direction)
    fallback[:, 0] = 1.0
    parallel = torch.where(
        (norm > float(eps)).expand_as(direction),
        direction / norm.clamp_min(float(eps)),
        fallback,
    )
    perpendicular = torch.stack((-parallel[:, 1], parallel[:, 0]), dim=1)
    basis = torch.stack((parallel, perpendicular), dim=2)
    variance = torch.exp(2.0 * prior.log_sigma)
    return basis @ torch.diag_embed(variance) @ basis.transpose(1, 2)


def validate_motion_prior_support_alignment(
    prior, support_center_xy, *, tolerance=1e-3, learned_source_id=1
):
    """Fail fast when B1 and its sampled support use different coordinates.

    Only learned-B1 support rows are checked.  Kinematic fallback and base-only
    rows have different source semantics and are intentionally excluded.
    Returns the per-row Euclidean error for logging and tests.
    """
    support = torch.as_tensor(
        support_center_xy,
        device=prior.center_xy.device,
        dtype=prior.center_xy.dtype,
    )
    if support.shape != prior.center_xy.shape:
        raise ValueError("support center must match motion prior shape [B,2]")
    if float(tolerance) < 0:
        raise ValueError("motion prior support tolerance must be non-negative")
    error = torch.linalg.norm(prior.center_xy - support, dim=1)
    check = (prior.valid.reshape(-1) > 0) & (
        prior.source.reshape(-1).to(torch.long) == int(learned_source_id)
    )
    if bool(torch.any(check & (~torch.isfinite(error)))):
        raise RuntimeError("B1/support alignment error is non-finite")
    failing = check & (error > float(tolerance))
    if bool(torch.any(failing)):
        maximum = float(error[failing].max().detach().cpu())
        raise RuntimeError(
            "B1 prior and search support use different coordinate frames: "
            f"max center error={maximum:.6g} m, "
            f"tolerance={float(tolerance):.6g} m"
        )
    return error


@dataclass(frozen=True)
class EvidenceOutput:
    raw_box: Tensor
    structural_available: Tensor
    presence_logit: Tensor
    targetness: Tensor
    evidence_summary: Tensor
    point_diagnostics: Dict[str, Tensor]


@dataclass(frozen=True)
class DecisionOutput:
    final_box: Tensor
    help_logit: Tensor
    harm_logit: Tensor
    expected_center_gain: Tensor
    expected_iou_gain: Tensor
    applied: Tensor
    bounded_residual: Tensor
