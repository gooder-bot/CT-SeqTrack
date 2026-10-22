"""v38 保持强 concat B1 主路；新增 B3 只读 detached 三候选。"""
import torch
from torch import nn

from models.ct_v31.contracts import EvidenceHypotheses
from models.ct_v31.model import canonicalize_batch
from models.ct_v31.observation import B0Observation
from models.ct_v31.decoder import SharedHypothesisDecoder
from models.ct_v31.prior import PhysicalTimePrior
from models.ct_v31.losses import compute_losses as base_losses
from .config import normalize_config, recipe_scope
from .contracts import CandidateSet, SelectionDecision, CommitDecision, TrackOutput
from .verification import FinalBoxVerifier, current_q0_support


def build_candidates(decoder, observation, batch, prior):
    """去重仅用于 B3；保留原 decoder 数学及完整来源。"""
    boxes = torch.stack((decoder.hypothesis_boxes[:, 0], prior.box.to(decoder.hypothesis_boxes),
                         batch['fallback_box'].detach().to(decoder.hypothesis_boxes)), 1)
    if not bool(torch.isfinite(boxes).all()):
        raise FloatingPointError('v38 position candidates must be finite')
    real = observation.sequence_valid.detach().bool()
    raw_valid = torch.stack((real, torch.ones_like(real), torch.ones_like(real)), -1)
    valid = raw_valid.clone()
    alias = torch.full_like(valid, -1, dtype=torch.long)
    detached = boxes.detach()
    for later in range(1, 3):
        for earlier in range(later):
            delta = detached[:, later]-detached[:, earlier]
            angle = torch.atan2(delta[:, 3].sin(), delta[:, 3].cos()).abs()
            same = (delta[:, :3].abs().amax(-1) <= 1e-4) & (angle <= 1e-5)
            duplicate = valid[:, later] & valid[:, earlier] & same
            alias[:, later] = torch.where(duplicate, earlier, alias[:, later])
            valid[:, later] &= ~duplicate
    # 全空 q0 是 prior 的显示别名，从不参与竞争或取得测量权限。
    alias[:, 0] = torch.where(~real, 1, alias[:, 0])
    if not bool(valid.any(-1).all()):
        raise RuntimeError('v38 requires a finite position candidate')
    return CandidateSet(boxes, valid, raw_valid, alias, ~real,
        decoder.quality_logits[:, 0], decoder)


def select_candidates(candidates, verification, observation, batch, *, apply_policy=True):
    """零 margin：仅严格更高效用替换真实 q0；prop 永不获得可信资格。"""
    valid = candidates.valid
    b, device = len(valid), valid.device
    row = torch.arange(b, device=device)
    legacy = candidates.q0_quality_logits.sigmoid()
    if not bool(torch.isfinite(candidates.boxes).all()) or not bool(torch.isfinite(legacy).all()):
        raise FloatingPointError('nonfinite v38 candidate or legacy q0 quality')
    if apply_policy:
        if verification is None:
            raise ValueError('v38 Full requires the independent verifier')
        if not bool(torch.isfinite(verification.logits).all()):
            raise FloatingPointError('nonfinite v38 verifier logits')
        utility = verification.utility
        selected = utility.masked_fill(~valid, -torch.inf).argmax(-1)
        # argmax 的固定顺序实现 q0/prior/fallback 平局优先，无容差门槛。
        selected_utility = utility[row, selected]
        candidate_trust = torch.zeros_like(utility)
        candidate_trust[:, 0] = torch.where(candidates.raw_valid[:, 0],
            verification.logits[:, 0, 2].sigmoid(), 0.)
    else:
        # B1 不增加任何神经前向；其全空输出原本就是 prior。
        selected = torch.where(candidates.raw_valid[:, 0], torch.zeros(b, device=device, dtype=torch.long),
                               torch.ones(b, device=device, dtype=torch.long))
        selected_utility = legacy.new_zeros(b)
        candidate_trust = legacy.new_zeros(b, 3)
        candidate_trust[:, 0] = torch.where(candidates.raw_valid[:, 0], legacy, 0.)
    selected_real = (selected == 0) & candidates.raw_valid[:, 0]
    support = current_q0_support(observation, batch, candidates.boxes[:, 0])
    support_mask = support['mask'] & selected_real[:, None]
    trusted_score = candidate_trust[row, selected]
    eligible = selected_real & (support['count'] >= 3) & (trusted_score >= .5)
    accepted = candidates.boxes[row, selected]
    selection = SelectionDecision(selected.detach(), accepted, selected_utility, selected.detach())
    commit = CommitDecision(trusted_score.detach(), legacy.detach(), support_mask.detach(),
                            eligible.detach(), selected.detach())
    # 原 B1 的全空 quality 也保留；Full 才显式将 propagation 的旧接口量归零。
    compatibility_quality = torch.where(selected_real, legacy, 0.) if apply_policy else legacy
    return selection, commit, compatibility_quality, candidate_trust


class JointTracker(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = normalize_config(config)
        self.enable_b1, self.enable_b2 = True, False
        self.enable_b3 = self.config.v38_arm == 'full'
        # 与强 v37 B1 完全同序实例化；新 B3 不消耗公共初始化随机数。
        self.observation = B0Observation(token_count=128, query_context=True,
            coarse_condition=True, time_scale=self.config.time_scale)
        self.decoder = SharedHypothesisDecoder(prior_dim=128, dropout=.2, query_context=True,
            time_scale=self.config.time_scale, initial_context=True, local_observation=True)
        self.prior = PhysicalTimePrior(self.config.time_scale, self.config.v31_temporal_backend)
        self.execution_profile = self.config.execution_profile
        self.observation.execution_profile = self.decoder.execution_profile = self.execution_profile
        self.evidence = None
        with torch.random.fork_rng(devices=[]):
            self.verifier = FinalBoxVerifier(self.config.time_scale) if self.enable_b3 else None

    def plan_prior(self, batch):
        with recipe_scope(self.config):
            return self.prior(canonicalize_batch(batch))

    def forward(self, batch, prior=None):
        with recipe_scope(self.config):
            prepared = canonicalize_batch(batch)
            prior = self.prior(prepared) if prior is None else prior
            observation = self.observation(prepared)
            evidence = EvidenceHypotheses.empty(observation.coarse_box)
            decoder = self.decoder(observation, evidence, prepared, prior)
            candidates = build_candidates(decoder, observation, prepared, prior)
            verification = self.verifier(candidates, observation, prepared, prior) if self.enable_b3 else None
            decision, commit, quality, candidate_trust = select_candidates(
                candidates, verification, observation, prepared, apply_policy=self.enable_b3)
            return TrackOutput(decision.accepted_box, decision.selected_index, quality, prior, observation,
                evidence, decoder, candidates, verification, decision, commit,
                candidates.q0_quality_logits.sigmoid(), decision.output_utility,
                candidate_trust, candidates.q0_is_fallback)

    def compute_losses(self, batch, output):
        with recipe_scope(self.config):
            prepared = canonicalize_batch(batch)
            if not self.enable_b3:
                return base_losses(prepared, output, enable_b1=True, enable_b2=False, enable_b3=False)
            from .losses import compute_losses
            return compute_losses(prepared, output)
