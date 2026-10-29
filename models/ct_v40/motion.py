"""物理运动坐标系与独立不确定性监督。"""

import torch


def motion_aligned_axes(velocity_xy, min_speed=0.2, eps=1e-6):
    """Return stable parallel/perpendicular axes for planar motion."""
    if velocity_xy.dim() != 2 or velocity_xy.shape[1] != 2:
        raise ValueError("velocity_xy must have shape [B,2]")
    speed = torch.linalg.norm(velocity_xy, dim=1, keepdim=True)
    fallback = torch.zeros_like(velocity_xy)
    fallback[:, 0] = 1.0
    direction = torch.where(
        (speed >= float(min_speed)).expand_as(velocity_xy),
        velocity_xy / torch.clamp(speed, min=float(eps)),
        fallback,
    )
    perpendicular = torch.stack(
        (-direction[:, 1], direction[:, 0]), dim=1)
    return direction, perpendicular, speed.squeeze(1)


def motion_aligned_covariance(
        velocity_xy, log_sigma_parallel_perp, min_speed=0.2, eps=1e-6,
        direction_xy=None):
    """Convert motion-frame standard deviations to an XY covariance."""
    if log_sigma_parallel_perp.shape != velocity_xy.shape:
        raise ValueError(
            "log_sigma_parallel_perp must match velocity_xy")
    default_direction, _, speed = motion_aligned_axes(
        velocity_xy, min_speed=min_speed, eps=eps)
    if direction_xy is None:
        direction = default_direction
    else:
        if direction_xy.shape != velocity_xy.shape:
            raise ValueError("direction_xy must match velocity_xy")
        direction_norm = torch.linalg.norm(
            direction_xy, dim=1, keepdim=True)
        direction = torch.where(
            (direction_norm > float(eps)).expand_as(direction_xy),
            direction_xy / torch.clamp(direction_norm, min=float(eps)),
            default_direction,
        )
    perpendicular = torch.stack(
        (-direction[:, 1], direction[:, 0]), dim=1)
    log_sigma = log_sigma_parallel_perp
    low_speed = speed < float(min_speed)
    isotropic = log_sigma.mean(dim=1, keepdim=True).expand_as(log_sigma)
    log_sigma = torch.where(low_speed.unsqueeze(1), isotropic, log_sigma)
    variance = torch.exp(2.0 * log_sigma)
    covariance = (
        variance[:, 0, None, None]
        * direction.unsqueeze(2) * direction.unsqueeze(1)
        + variance[:, 1, None, None]
        * perpendicular.unsqueeze(2) * perpendicular.unsqueeze(1)
    )
    marginal_log_sigma = 0.5 * torch.log(torch.clamp(
        torch.diagonal(covariance, dim1=1, dim2=2), min=float(eps)))
    return {
        "log_sigma_parallel_perp": log_sigma,
        "covariance_xy": covariance,
        "motion_direction_xy": direction,
        "log_sigma_xy": marginal_log_sigma,
        "motion_speed": speed,
    }


def physical_motion_uncertainty_loss(
        mean_xy, target_xy, log_sigma_parallel_perp,
        motion_direction_xy, valid, eps=1e-6, *,
        kinematic_xy=None, envelope_parallel_perp=None,
        residual_unit_parallel_perp=None, beta=0.5,
        tail_direction_weight=0.25, tail_direction_margin=0.9,
        sigma_error_cap=12.0, mean_huber_beta=0.25):
    """Masked robust residual learning and isolated beta-NLL sigma terms.

    Returned losses are per sample.  Callers own weighting and masked
    reduction so this helper can also be used by calibration diagnostics.
    Sigma targets are detached by construction: uncertainty must not steer
    the temporal backbone or the learned mean.
    """
    if mean_xy.shape != target_xy.shape or mean_xy.shape[-1] != 2:
        raise ValueError("mean_xy and target_xy must both have shape [B,2]")
    if log_sigma_parallel_perp.shape != mean_xy.shape:
        raise ValueError("motion log sigma must have shape [B,2]")
    if motion_direction_xy.shape != mean_xy.shape:
        raise ValueError("motion direction must have shape [B,2]")
    perpendicular = torch.stack((
        -motion_direction_xy[:, 1], motion_direction_xy[:, 0]), dim=1)
    if not 0.0 <= float(beta) <= 1.0:
        raise ValueError("motion beta-NLL beta must be in [0,1]")
    if tail_direction_weight < 0 or not 0.0 <= tail_direction_margin <= 1.0:
        raise ValueError("motion tail loss settings are invalid")
    if sigma_error_cap <= 0 or mean_huber_beta <= 0:
        raise ValueError("motion robust-loss scales must be positive")

    error = target_xy - mean_xy
    aligned_error = torch.stack((
        (error * motion_direction_xy).sum(dim=1),
        (error * perpendicular).sum(dim=1),
    ), dim=1)
    safe_log_sigma = torch.clamp(
        torch.nan_to_num(log_sigma_parallel_perp), min=-4.0, max=2.5)
    safe_aligned_error = torch.nan_to_num(
        aligned_error.detach(), nan=0.0,
        posinf=float(sigma_error_cap), neginf=-float(sigma_error_cap))
    variance = torch.exp(2.0 * safe_log_sigma)
    raw_nll_per_axis = 0.5 * (
        safe_aligned_error.pow(2) / variance + torch.log(variance))
    capped_error = torch.clamp(
        safe_aligned_error,
        min=-float(sigma_error_cap), max=float(sigma_error_cap))
    beta_nll_per_axis = 0.5 * (
        capped_error.pow(2) / variance + torch.log(variance))
    if beta > 0:
        beta_nll_per_axis = (
            beta_nll_per_axis * variance.detach().pow(float(beta)))

    use_normalized_residual = all(value is not None for value in (
        kinematic_xy, envelope_parallel_perp,
        residual_unit_parallel_perp))
    if use_normalized_residual:
        for name, value in (
                ("kinematic_xy", kinematic_xy),
                ("envelope_parallel_perp", envelope_parallel_perp),
                ("residual_unit_parallel_perp",
                 residual_unit_parallel_perp)):
            if value.shape != mean_xy.shape:
                raise ValueError(f"{name} must have shape [B,2]")
        target_residual = target_xy - kinematic_xy
        aligned_target_residual = torch.stack((
            (target_residual * motion_direction_xy).sum(dim=1),
            (target_residual * perpendicular).sum(dim=1),
        ), dim=1)
        safe_envelope = torch.clamp(
            envelope_parallel_perp, min=float(eps))
        target_unit = aligned_target_residual / safe_envelope
        predicted_unit = residual_unit_parallel_perp
        recoverable_axis = torch.abs(target_unit) <= 1.0
        mean_per_axis = torch.nn.functional.smooth_l1_loss(
            predicted_unit, torch.clamp(target_unit, min=-1.0, max=1.0),
            reduction="none", beta=float(mean_huber_beta))
        recoverable_count = recoverable_axis.to(mean_xy.dtype).sum(dim=1)
        tail_axis = ~recoverable_axis
        tail_count = tail_axis.to(mean_xy.dtype).sum(dim=1)
        tail_hinge = torch.relu(
            float(tail_direction_margin)
            - torch.sign(target_unit) * predicted_unit)
        robust_mean = torch.where(
            recoverable_axis,
            mean_per_axis,
            float(tail_direction_weight) * tail_hinge,
        ).mean(dim=1)
        recoverable_saturation = (
            ((torch.abs(predicted_unit) >= 0.95) & recoverable_axis)
            .to(mean_xy.dtype).sum(dim=1)
            / torch.clamp(recoverable_count, min=1.0))
        tail_axis_fraction = tail_count / float(mean_xy.shape[1])
    else:
        robust_mean = torch.nn.functional.smooth_l1_loss(
            mean_xy, target_xy, reduction="none").mean(dim=1)
        recoverable_saturation = torch.zeros_like(robust_mean)
        tail_axis_fraction = torch.zeros_like(robust_mean)
    valid = valid.to(device=mean_xy.device, dtype=mean_xy.dtype).reshape(-1)
    finite = (
        torch.isfinite(aligned_error).all(dim=1)
        & torch.isfinite(raw_nll_per_axis).all(dim=1)
        & torch.isfinite(beta_nll_per_axis).all(dim=1)
        & torch.isfinite(robust_mean)
    ).to(mean_xy.dtype)
    return {
        "mean_per_sample": robust_mean,
        "nll_per_sample": beta_nll_per_axis.sum(dim=1),
        "gaussian_nll_per_sample": raw_nll_per_axis.sum(dim=1),
        "aligned_error": aligned_error,
        "recoverable_saturation_per_sample": recoverable_saturation,
        "tail_axis_fraction_per_sample": tail_axis_fraction,
        "valid": valid * finite,
    }
