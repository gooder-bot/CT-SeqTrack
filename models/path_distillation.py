"""Small, testable building blocks for endpoint path distillation.

The M3 objective is deliberately independent of physical timestamps.  It
matches a canonical-path teacher prediction to an irregular-path student
prediction at the same endpoint, after the sampler has verified that the
current observation and coordinate frame are shared.
"""

from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F


def endpoint_path_terms(reference_boxes, student_boxes, theta_weight=0.5):
    """Return per-sample center/yaw distillation terms.

    Args:
        reference_boxes: ``[B, 4]`` canonical teacher boxes.
        student_boxes: ``[B, 4]`` irregular-path student boxes.
        theta_weight: Weight of the periodic yaw term.
    """
    if reference_boxes.shape != student_boxes.shape:
        raise ValueError("reference_boxes and student_boxes must have identical shapes")
    if reference_boxes.dim() != 2 or reference_boxes.shape[1] < 4:
        raise ValueError("endpoint boxes must have shape [B, >=4]")

    reference_boxes = reference_boxes.detach()
    reference_center = reference_boxes[:, :3]
    student_center = student_boxes[:, :3]
    reference_yaw = reference_boxes[:, 3]
    student_yaw = student_boxes[:, 3]

    center_loss = F.smooth_l1_loss(
        student_center, reference_center, reduction="none").mean(dim=1)
    yaw_loss = (
        F.smooth_l1_loss(
            torch.sin(student_yaw), torch.sin(reference_yaw), reduction="none")
        + F.smooth_l1_loss(
            torch.cos(student_yaw), torch.cos(reference_yaw), reduction="none")
    )
    total_loss = center_loss + float(theta_weight) * yaw_loss

    center_gap = torch.linalg.norm(student_center - reference_center, dim=1)
    yaw_gap = torch.abs(torch.atan2(
        torch.sin(student_yaw - reference_yaw),
        torch.cos(student_yaw - reference_yaw),
    ))
    return {
        "total": total_loss,
        "center_loss": center_loss,
        "yaw_loss": yaw_loss,
        "center_gap": center_gap,
        "yaw_gap": yaw_gap,
    }


def teacher_endpoint_confidence_terms(
        output,
        point_sample_size,
        mode="foreground_topk",
        topk=32,
        floor=0.05,
        agreement_center_scale=1.0,
        agreement_yaw_scale=0.5):
    """Build GT-free confidence terms from a teacher forward pass.

    The foreground term measures whether the current search contains a
    confident target-like region.  The agreement term follows the
    local/global-proposal agreement idea used by trajectory-prior trackers:
    the teacher is trusted more when its coarse observation proposal and
    sequence-refined endpoint agree.  ``hybrid`` uses the geometric mean so
    neither signal can dominate by scale alone.
    """
    boxes = output["aux_estimation_boxes"]
    batch_size = boxes.shape[0]
    mode = str(mode).strip().lower().replace("-", "_")
    floor = float(floor)
    if not 0.0 <= floor <= 1.0:
        raise ValueError("teacher confidence floor must be in [0, 1]")

    if mode in ("fixed", "uniform", "ones"):
        ones = boxes.new_ones((batch_size,))
        return {
            "confidence": ones,
            "foreground": ones,
            "agreement": ones,
            "coarse_refined_center_gap": boxes.new_zeros((batch_size,)),
            "coarse_refined_yaw_gap": boxes.new_zeros((batch_size,)),
        }
    if mode not in (
            "foreground", "foreground_topk", "segmentation",
            "agreement", "proposal_agreement", "hybrid"):
        raise ValueError(
            "m3_teacher_confidence_mode must be fixed, foreground_topk, "
            "agreement, or hybrid")

    foreground_confidence = boxes.new_ones((batch_size,))
    if mode in ("foreground", "foreground_topk", "segmentation", "hybrid"):
        if "seg_logits" not in output:
            raise KeyError("foreground teacher confidence requires seg_logits")
        point_sample_size = int(point_sample_size)
        if point_sample_size <= 0:
            raise ValueError("point_sample_size must be positive")
        logits = output["seg_logits"]
        current_logits = logits[:, :, -point_sample_size:]
        foreground = torch.softmax(current_logits, dim=1)[:, 1, :]
        selected = min(max(int(topk), 1), foreground.shape[1])
        foreground_confidence = torch.topk(
            foreground, k=selected, dim=1).values.mean(dim=1)

    agreement_confidence = boxes.new_ones((batch_size,))
    center_gap = boxes.new_zeros((batch_size,))
    yaw_gap = boxes.new_zeros((batch_size,))
    if mode in ("agreement", "proposal_agreement", "hybrid"):
        if "estimation_boxes" not in output:
            raise KeyError("proposal agreement requires estimation_boxes")
        coarse_boxes = output["estimation_boxes"]
        if coarse_boxes.shape != boxes.shape:
            raise ValueError(
                "teacher coarse/refined endpoint boxes must have identical shapes")
        center_scale = float(agreement_center_scale)
        yaw_scale = float(agreement_yaw_scale)
        if center_scale <= 0 or yaw_scale <= 0:
            raise ValueError("teacher agreement scales must be positive")
        center_gap = torch.linalg.norm(
            coarse_boxes[:, :3] - boxes[:, :3], dim=1)
        yaw_gap = torch.abs(torch.atan2(
            torch.sin(coarse_boxes[:, 3] - boxes[:, 3]),
            torch.cos(coarse_boxes[:, 3] - boxes[:, 3]),
        ))
        agreement_confidence = torch.exp(
            -center_gap / center_scale - yaw_gap / yaw_scale)

    if mode in ("foreground", "foreground_topk", "segmentation"):
        confidence = foreground_confidence
    elif mode in ("agreement", "proposal_agreement"):
        confidence = agreement_confidence
    else:
        confidence = torch.sqrt(
            foreground_confidence.clamp_min(0.0)
            * agreement_confidence.clamp_min(0.0))

    return {
        "confidence": confidence.detach().clamp(min=floor, max=1.0),
        "foreground": foreground_confidence.detach().clamp(min=0.0, max=1.0),
        "agreement": agreement_confidence.detach().clamp(min=0.0, max=1.0),
        "coarse_refined_center_gap": center_gap.detach(),
        "coarse_refined_yaw_gap": yaw_gap.detach(),
    }


def teacher_endpoint_confidence(output, point_sample_size, mode="foreground_topk",
                                topk=32, floor=0.05,
                                agreement_center_scale=1.0,
                                agreement_yaw_scale=0.5):
    """Return only the final GT-free teacher confidence weight."""
    return teacher_endpoint_confidence_terms(
        output,
        point_sample_size=point_sample_size,
        mode=mode,
        topk=topk,
        floor=floor,
        agreement_center_scale=agreement_center_scale,
        agreement_yaw_scale=agreement_yaw_scale,
    )["confidence"]


@contextmanager
def freeze_batchnorm_running_stats(module):
    """Use stored BN statistics without disabling affine gradients."""
    batchnorm_states = []
    for child in module.modules():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            batchnorm_states.append((child, child.training))
            child.eval()
    try:
        yield
    finally:
        for child, was_training in batchnorm_states:
            child.train(was_training)


@torch.no_grad()
def update_ema_module(teacher, student, momentum):
    """EMA-update a frozen teacher from matching student tensors."""
    momentum = float(momentum)
    if not 0.0 <= momentum < 1.0:
        raise ValueError("EMA momentum must be in [0, 1)")

    student_parameters = dict(student.named_parameters())
    for name, teacher_parameter in teacher.named_parameters():
        student_parameter = student_parameters.get(name)
        if student_parameter is None:
            raise KeyError(f"EMA student is missing parameter {name}")
        teacher_parameter.mul_(momentum).add_(
            student_parameter.detach(), alpha=1.0 - momentum)

    student_buffers = dict(student.named_buffers())
    for name, teacher_buffer in teacher.named_buffers():
        student_buffer = student_buffers.get(name)
        if student_buffer is None or teacher_buffer.shape != student_buffer.shape:
            continue
        if torch.is_floating_point(teacher_buffer):
            teacher_buffer.mul_(momentum).add_(
                student_buffer.detach(), alpha=1.0 - momentum)
        else:
            teacher_buffer.copy_(student_buffer.detach())
