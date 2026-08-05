"""Lightweight EMA consistency on final decoder tokens."""

import copy
import math

import torch
from torch import nn
import torch.nn.functional as F


def _off_diagonal(matrix):
    size = matrix.shape[0]
    return matrix.flatten()[:-1].view(size - 1, size + 1)[:, 1:].flatten()


class DecoderTokenConsistencyLoss(nn.Module):
    """EMA-projector alignment on a stop-gradient online representation."""

    def __init__(
            self, input_dim=64, projection_dim=64, hidden_dim=128,
            teacher_momentum=0.996, invariance_weight=1.0,
            variance_weight=1.0, covariance_weight=0.04,
            variance_target=1.0, eps=1e-4):
        super().__init__()
        self.input_dim = int(input_dim)
        self.projection_dim = int(projection_dim)
        self.teacher_momentum = float(teacher_momentum)
        self.invariance_weight = float(invariance_weight)
        self.variance_weight = float(variance_weight)
        self.covariance_weight = float(covariance_weight)
        self.variance_target = float(variance_target)
        self.eps = float(eps)
        if not 0.0 <= self.teacher_momentum < 1.0:
            raise ValueError("decoder teacher momentum must be in [0,1)")
        self.student_projector = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), self.projection_dim),
        )
        self.teacher_projector = copy.deepcopy(self.student_projector)
        for parameter in self.teacher_projector.parameters():
            parameter.requires_grad_(False)
        self.teacher_projector.eval()
        self.register_buffer("teacher_updates", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def update_teacher(self):
        momentum = self.teacher_momentum
        for teacher, student in zip(
                self.teacher_projector.parameters(),
                self.student_projector.parameters()):
            teacher.mul_(momentum).add_(student.detach(), alpha=1.0 - momentum)
        for teacher, student in zip(
                self.teacher_projector.buffers(),
                self.student_projector.buffers()):
            if teacher.is_floating_point():
                teacher.mul_(momentum).add_(
                    student.detach(), alpha=1.0 - momentum)
            else:
                teacher.copy_(student)
        self.teacher_updates.add_(1)
        self.teacher_projector.eval()

    def forward(self, clean_decoder_state, irregular_decoder_state):
        if (clean_decoder_state.dim() != 3
                or irregular_decoder_state.shape != clean_decoder_state.shape
                or clean_decoder_state.shape[-1] != self.input_dim):
            raise ValueError(
                "decoder states must share shape [B,L,input_dim]")
        batch_tokens = clean_decoder_state.shape[0] * clean_decoder_state.shape[1]
        clean = clean_decoder_state.detach().reshape(batch_tokens, -1)
        irregular = irregular_decoder_state.reshape(batch_tokens, -1)
        with torch.no_grad():
            teacher = self.teacher_projector(clean)
        student = self.student_projector(irregular)
        invariance = (1.0 - F.cosine_similarity(
            student, teacher, dim=1, eps=self.eps)).mean()
        student_std = torch.sqrt(
            student.var(dim=0, unbiased=False) + self.eps)
        variance = F.relu(self.variance_target - student_std).mean()
        centered = student - student.mean(dim=0, keepdim=True)
        denominator = max(centered.shape[0] - 1, 1)
        covariance = centered.T @ centered / float(denominator)
        covariance_guard = _off_diagonal(covariance).pow(2).sum(
            ) / float(self.projection_dim)
        loss = (
            self.invariance_weight * invariance
            + self.variance_weight * variance
            + self.covariance_weight * covariance_guard)

        with torch.no_grad():
            eigenvalues = torch.linalg.eigvalsh(covariance.float()).clamp_min(0)
            probability = eigenvalues / eigenvalues.sum().clamp_min(self.eps)
            effective_rank = torch.exp(-(
                probability * torch.log(probability.clamp_min(self.eps))).sum())
            teacher_std = teacher.std(dim=0, unbiased=False).mean()
        return {
            "loss_decoder_token_consistency": loss,
            "decoder_tc_invariance": invariance,
            "decoder_tc_variance": variance,
            "decoder_tc_covariance": covariance_guard,
            "decoder_tc_feature_std": student_std.mean(),
            "decoder_tc_teacher_feature_std": teacher_std,
            "decoder_tc_effective_rank": effective_rank.to(student),
            "decoder_tc_effective_rank_ratio": (
                effective_rank.to(student) / float(self.projection_dim)),
            "decoder_tc_teacher_updates": self.teacher_updates.to(student),
        }


class GradientRatioWeightSelector(nn.Module):
    """Freeze one candidate weight after a bounded gradient-ratio audit."""

    def __init__(
            self, candidates=(0.001, 0.003, 0.01), audit_batches=200,
            target_ratio=0.075):
        super().__init__()
        candidates = torch.as_tensor(candidates, dtype=torch.float32)
        if candidates.numel() == 0 or bool(torch.any(candidates <= 0)):
            raise ValueError("decoder consistency weights must be positive")
        self.audit_batches = int(audit_batches)
        self.target_ratio = float(target_ratio)
        if self.audit_batches <= 0:
            raise ValueError("gradient audit_batches must be positive")
        self.register_buffer("candidates", candidates)
        self.register_buffer("ratio_sum", torch.zeros(()))
        self.register_buffer("ratio_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("selected_index", torch.zeros((), dtype=torch.long))
        self.register_buffer("frozen", torch.zeros((), dtype=torch.bool))
        self.register_buffer("guardrail_passed", torch.zeros((), dtype=torch.bool))
        self.register_buffer("selected_weighted_ratio", torch.zeros(()))

    @torch.no_grad()
    def observe(self, unweighted_ratio):
        if bool(self.frozen):
            return
        ratio = float(unweighted_ratio)
        if not math.isfinite(ratio) or ratio < 0:
            return
        self.ratio_sum.add_(ratio)
        self.ratio_count.add_(1)
        mean_ratio = self.ratio_sum / self.ratio_count.clamp_min(1)
        weighted = self.candidates * mean_ratio
        self.selected_index.copy_(torch.argmin(torch.abs(
            weighted - self.target_ratio)))
        self.selected_weighted_ratio.copy_(weighted[self.selected_index])
        if int(self.ratio_count) >= self.audit_batches:
            self.guardrail_passed.copy_((
                self.selected_weighted_ratio >= 0.05)
                & (self.selected_weighted_ratio <= 0.10))
            self.frozen.fill_(True)

    def value(self, reference):
        return self.candidates[self.selected_index].to(reference)
