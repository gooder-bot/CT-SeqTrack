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
class B1Input:
    """One causal tensor contract reused by prepass and trainable B1."""

    ref_boxs: Tensor
    delta_t: Tensor
    valid_mask: Tensor
    current_delta_t: Tensor
    acquisition_features: Optional[Tensor] = None

    def encoder_kwargs(self):
        return dict(ref_boxs=self.ref_boxs, delta_t=self.delta_t,
                    valid_mask=self.valid_mask,
                    current_delta_t=self.current_delta_t,
                    acquisition_features=self.acquisition_features)

    def detached(self):
        return B1Input(**{
            key: None if value is None else value.detach()
            for key, value in self.encoder_kwargs().items()})


@dataclass(frozen=True)
class AcquisitionRecord:
    """Actual resolved acquisition geometry, distinct from learned B1 output.

    Both directions are explicit: the acquisition frame follows the resolved
    displacement, while statistical uncertainty retains its original basis.
    No current annotation or training target belongs to this record.
    """

    endpoint_xy: Tensor
    direction_xy: Tensor
    acquisition_margin_parallel_perp: Tensor
    valid: Tensor
    source: Tensor
    source_anchor: Tensor
    coordinate_anchor: Tensor
    query_delta_t: Tensor
    statistical_direction_xy: Optional[Tensor] = None
    statistical_log_sigma: Optional[Tensor] = None
    learned_valid: Optional[Tensor] = None
    gap_ratio: Optional[Tensor] = None
    real_timestamp: Optional[Tensor] = None
    input_digest: str = ''
    parameter_revision: int = 0
    fallback_reason: str = ''

    def detached(self):
        return AcquisitionRecord(**{
            key: value.detach() if torch.is_tensor(value) else value
            for key, value in self.__dict__.items()})

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
            current_features=(None if self.current_features is None else
                              self.current_features.detach()),
            history_features=(None if self.history_features is None else
                              self.history_features.detach()),
        )


@dataclass(frozen=True)
class MotionPriorOutput:
    center_xy: Tensor
    direction_xy: Tensor
    log_sigma: Tensor
    valid: Tensor
    source: Tensor
    # Search acquisition geometry is deliberately independent from the
    # statistical covariance used for B1 calibration.  ``None`` preserves
    # strict compatibility with v25 callers/checkpoints.
    acquisition_margin_parallel_perp: Optional[Tensor] = None

    def detached(self):
        return MotionPriorOutput(
            center_xy=self.center_xy.detach(),
            direction_xy=self.direction_xy.detach(),
            log_sigma=self.log_sigma.detach(),
            valid=self.valid.detach(),
            source=self.source.detach(),
            acquisition_margin_parallel_perp=(
                None if self.acquisition_margin_parallel_perp is None else
                self.acquisition_margin_parallel_perp.detach()),
        )


def _rotate_xy(vectors, yaw):
    """Rotate batched xy row vectors by ``yaw`` radians."""
    cosine = torch.cos(yaw)
    sine = torch.sin(yaw)
    x, y = vectors.unbind(dim=1)
    return torch.stack((
        cosine * x - sine * y,
        sine * x + cosine * y,
    ), dim=1)


def reexpress_motion_prior(
        prior, source_anchor, target_anchor, *, degrees=False):
    """Express a physical B1 prior in another local SE(2) frame.

    Exact identity rows select the original values, preserving the canonical
    online path bitwise.  Parallel/perpendicular uncertainty magnitudes are
    invariant because their direction vector is transformed with them.
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
            and torch.isfinite(prior.log_sigma).all()):
        raise ValueError(
            "motion prior center/direction/log sigma must be finite")

    source = torch.as_tensor(
        source_anchor, device=center.device, dtype=center.dtype)
    target = torch.as_tensor(
        target_anchor, device=center.device, dtype=center.dtype)
    expected_anchor_shape = (center.shape[0], 4)
    if (source.shape != expected_anchor_shape
            or target.shape != expected_anchor_shape):
        raise ValueError(
            "motion/coordinate anchors must both have shape [B,4]")
    if not bool(torch.isfinite(source).all() and torch.isfinite(target).all()):
        raise ValueError("motion/coordinate anchors must be finite")

    compute_dtype = (
        torch.float32 if center.dtype in (torch.float16, torch.bfloat16)
        else center.dtype)
    source_work = source.to(compute_dtype)
    target_work = target.to(compute_dtype)
    center_work = center.to(compute_dtype)
    direction_work = direction.to(compute_dtype)
    angle_scale = math.pi / 180.0 if degrees else 1.0
    source_yaw = source_work[:, 3] * angle_scale
    target_yaw = target_work[:, 3] * angle_scale

    world_center = (
        source_work[:, :2] + _rotate_xy(center_work, source_yaw))
    converted_center = _rotate_xy(
        world_center - target_work[:, :2], -target_yaw).to(center.dtype)
    world_direction = _rotate_xy(direction_work, source_yaw)
    converted_direction = _rotate_xy(
        world_direction, -target_yaw).to(direction.dtype)

    identity = torch.all(source == target, dim=1, keepdim=True)
    converted_center = torch.where(identity, center, converted_center)
    converted_direction = torch.where(
        identity, direction, converted_direction)
    return MotionPriorOutput(
        center_xy=converted_center,
        direction_xy=converted_direction,
        log_sigma=prior.log_sigma,
        valid=prior.valid,
        source=prior.source,
        acquisition_margin_parallel_perp=(
            prior.acquisition_margin_parallel_perp),
    )


def motion_prior_covariance_xy(prior, *, eps=1e-6):
    """Reconstruct xy covariance from direction and log-sigma axes."""
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
        direction / norm.clamp_min(float(eps)), fallback)
    perpendicular = torch.stack((-parallel[:, 1], parallel[:, 0]), dim=1)
    basis = torch.stack((parallel, perpendicular), dim=2)
    variance = torch.exp(2.0 * prior.log_sigma)
    return basis @ torch.diag_embed(variance) @ basis.transpose(1, 2)


def validate_motion_prior_support_alignment(
        prior, support_center_xy, *, tolerance=1e-3,
        learned_source_id=1):
    """Fail fast if learned B1 and B2 support use different frames."""
    support = torch.as_tensor(
        support_center_xy,
        device=prior.center_xy.device,
        dtype=prior.center_xy.dtype)
    if support.shape != prior.center_xy.shape:
        raise ValueError("support center must match motion prior shape [B,2]")
    if float(tolerance) < 0:
        raise ValueError("motion prior support tolerance must be non-negative")
    error = torch.linalg.norm(prior.center_xy - support, dim=1)
    check = (
        prior.valid.reshape(-1) > 0
    ) & (
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
            f"tolerance={float(tolerance):.6g} m")
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

