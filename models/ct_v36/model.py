"""v36 单次前向组合；输出效用和测量接纳各自排序。"""
import torch
from torch import nn
from models.ct_v31.model import canonicalize_batch
from models.ct_v31.observation import B0Observation
from .config import normalize_config
from .contracts import CommitDecision, EvidenceHypotheses, TrackOutput
from .randomness import component_rng


def select_candidates(candidates, verification, *, memory_enabled=True):
    valid = candidates.valid.bool()
    if not bool(valid.any(-1).all()):
        raise ValueError('every endpoint must have a valid position candidate')
    if (not bool(torch.isfinite(candidates.boxes[valid]).all())
            or not bool(torch.isfinite(verification.logits[valid]).all())):
        raise FloatingPointError('nonfinite valid candidate or verification')
    utility = verification.utility
    selected = utility.masked_fill(~valid, -torch.inf).argmax(-1)
    scores = verification.measurement_score
    counts = verification.support_mask.sum(-1)
    eligible = valid & (counts >= 1) & (scores >= .5)
    eligible[:, 1] = False
    measurement = scores.masked_fill(~eligible, -torch.inf).argmax(-1)
    measurement = torch.where(eligible.any(-1), measurement, -1)
    row = torch.arange(len(selected), device=selected.device)
    memory = (measurement >= 0) & (counts[row, measurement.clamp_min(0)] >= 3) & bool(memory_enabled)
    decision = CommitDecision(selected.detach(), measurement.detach(), memory.detach())
    return candidates.boxes[row, selected], utility[row, selected], decision


class JointTracker(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        from .prior import EventPrior
        from .evidence import B2IdentityEvidence
        from .decoder import SharedHypothesisDecoder
        from .verification import FinalBoxVerifier
        self.config = normalize_config(config)
        self.enable_b1 = self.enable_b3 = True
        self.enable_b2 = self.config.v36_arm != 'C'
        seed = self.config.seed
        with component_rng(seed, 'observation'):
            self.observation = B0Observation(128, query_context=True, coarse_condition=True,
                                             time_scale=self.config.time_scale)
        with component_rng(seed, 'decoder'):
            self.decoder = SharedHypothesisDecoder(prior_dim=128, dropout=.2,
                time_scale=self.config.time_scale, enable_modes=self.enable_b2)
        backend = {'D': 'cv', 'E': 'legacy'}.get(self.config.v36_arm, 'learned')
        self.prior = EventPrior(time_scale=self.config.time_scale, backend=backend,
                               seed=seed, temporal_backend=self.config.v31_temporal_backend)
        with component_rng(seed, 'evidence'):
            self.evidence = B2IdentityEvidence() if self.enable_b2 else None
        with component_rng(seed, 'verification'):
            self.verifier = FinalBoxVerifier(time_scale=self.config.time_scale)

    def plan_prior(self, batch):
        return self.prior(canonicalize_batch(batch))

    def forward(self, batch, prior=None):
        prepared = canonicalize_batch(batch)
        prior = self.plan_prior(prepared) if prior is None else prior
        observation = self.observation(prepared)
        evidence = (self.evidence(observation, prepared, prior) if self.evidence is not None
                    else EvidenceHypotheses.empty(observation.coarse_box))
        candidates = self.decoder(observation, evidence, prepared, prior)
        verification = self.verifier(candidates, observation, evidence, prepared, prior)
        accepted, quality, decision = select_candidates(candidates, verification, memory_enabled=self.enable_b2)
        return TrackOutput(accepted, decision.output_index, quality, prior, observation,
                           evidence, candidates, verification, decision)

    def compute_losses(self, batch, output):
        from .losses import compute_losses
        return compute_losses(canonicalize_batch(batch), output, arm=self.config.v36_arm,
                              time_scale=self.config.time_scale)
