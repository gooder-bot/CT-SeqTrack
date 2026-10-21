"""v37共享训练图；B1+B2与Full只在选择/提交政策上不同。"""
import torch
from torch import nn
from utils.execution_lab import enabled
from utils.b2_centroid_recipes import detach_mass_weights, recipe_scope

from models.ct_v31.contracts import EvidenceHypotheses as V35Evidence, TrackOutput as V35Output
from models.ct_v31.model import canonicalize_batch, select_hypothesis
from models.ct_v31.observation import B0Observation
from models.ct_v31.decoder import SharedHypothesisDecoder as V35Decoder
from models.ct_v31.prior import PhysicalTimePrior, empty_prior
from models.ct_v31.losses import compute_losses as v35_losses
from .config import normalize_config
from .contracts import TrackOutput


def select_candidates(candidates, verification, *, apply_policy):
    """S/P效用仅排序；提交质量从旧q0头与独立Qmode合成。"""
    valid = candidates.valid.bool()
    if not bool(valid.any(-1).all()):
        raise ValueError('v37 every endpoint needs a position candidate')
    if not bool(torch.isfinite(candidates.boxes[valid]).all()) or not bool(torch.isfinite(verification.logits[valid]).all()):
        raise FloatingPointError('nonfinite v37 candidate or verification')
    utility = verification.utility
    selected = (utility.masked_fill(~valid, -torch.inf).argmax(-1) if apply_policy
                else torch.zeros(len(valid), dtype=torch.long, device=valid.device))
    q0_quality = candidates.q0_quality_logits.sigmoid()
    q0_quality = torch.where(candidates.q0_is_fallback, 0., q0_quality)
    commit_quality = torch.cat((q0_quality[:, None], q0_quality.new_zeros(len(valid), 1),
                               verification.logits[:, 2:, 2].sigmoid()), 1)
    commit_quality = torch.where(valid, commit_quality, 0.)
    row = torch.arange(len(selected), device=selected.device)
    return (candidates.boxes[row, selected], selected.detach(), commit_quality[row, selected],
            utility[row, selected], commit_quality)


class JointTracker(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        from .decoder import SharedHypothesisDecoder
        from .evidence import B2IdentityEvidence
        from .verification import FinalBoxVerifier
        self.config = normalize_config(config)
        self.enable_b1 = self.config.v37_arm != 'b0'
        self.enable_b2 = self.config.v37_arm in ('b1_b2', 'full')
        self.enable_b3 = self.config.v37_arm == 'full'
        # 顺序与 v35 JointTracker 完全相同；B0/B1不实例化任何新增参数。
        self.observation = B0Observation(token_count=128, query_context=True,
            coarse_condition=True, time_scale=self.config.time_scale)
        if self.enable_b2:
            self.decoder = SharedHypothesisDecoder(prior_dim=128, dropout=.2, time_scale=self.config.time_scale)
        else:
            self.decoder = V35Decoder(prior_dim=128, dropout=.2, query_context=True,
                time_scale=self.config.time_scale, initial_context=True, local_observation=True)
        profile = self.config.get('execution_profile', 'optimized_fp32')
        self.observation.execution_profile = self.decoder.execution_profile = profile
        self.execution_profile = profile
        self.prior = (PhysicalTimePrior(self.config.time_scale, self.config.v31_temporal_backend)
                      if self.enable_b1 else None)
        with torch.random.fork_rng(devices=[]):
            self.evidence = (B2IdentityEvidence(detach_mass_weights=detach_mass_weights(self.config))
                             if self.enable_b2 else None)
            self.verifier = FinalBoxVerifier(self.config.time_scale) if self.enable_b2 else None

    def plan_prior(self, batch):
        prepared = canonicalize_batch(batch)
        return self.prior(prepared) if self.prior is not None else empty_prior(prepared)

    def forward(self, batch, prior=None):
        # 直接调用也遵守配方：B1保留执行算法，新B2配方显式隔离工程开关。
        with recipe_scope(self.config):
            return self._forward(batch, prior)

    def _forward(self, batch, prior=None):
        prepared = canonicalize_batch(batch)
        prior = self.plan_prior(prepared) if prior is None else prior
        observation = self.observation(prepared)
        if not self.enable_b2:
            evidence = V35Evidence.empty(observation.coarse_box)
            decoder = self.decoder(observation, evidence, prepared, prior)
            accepted, selected, quality = select_hypothesis(decoder)
            return V35Output(accepted, selected, quality, prior, observation, evidence, decoder)
        evidence = self.evidence(observation, prepared, prior)
        if enabled('pool'):
            from .verification import build_current_pool
            # 只传递当前调用的 live 点池；verifier 在自己的边界 detach。
            pool = build_current_pool(observation, evidence, prepared)
            candidates = self.decoder(observation, evidence, prepared, prior, _current_pool=pool)
            verification = self.verifier(candidates, observation, evidence, prepared, prior, _current_pool=pool)
        else:
            candidates = self.decoder(observation, evidence, prepared, prior)
            verification = self.verifier(candidates, observation, evidence, prepared, prior)
        # 旧诊断布局保留q0原logit；新增两个mode记录自身提交头，不伪造第三mode。
        candidates.decoder.quality_logits = torch.cat((candidates.q0_quality_logits[:, None],
            verification.logits[:, 2:, 2], verification.logits.new_full((len(prepared['points']), 1), -20.)), 1)
        accepted, selected, quality, utility, commit_quality = select_candidates(
            candidates, verification, apply_policy=self.enable_b3)
        return TrackOutput(accepted, selected, quality, prior, observation, evidence, candidates.decoder,
            candidates, verification, utility, commit_quality, candidates.q0_is_fallback)

    def compute_losses(self, batch, output):
        with recipe_scope(self.config):
            return self._compute_losses(batch, output)

    def _compute_losses(self, batch, output):
        prepared = canonicalize_batch(batch)
        if not self.enable_b2:
            return v35_losses(prepared, output, enable_b1=self.enable_b1, enable_b2=False, enable_b3=False)
        from .losses import compute_losses
        return compute_losses(prepared, output)
