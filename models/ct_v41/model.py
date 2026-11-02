"""v41 Full 恢复链；工程消融仍保留 B0/B1/M 接口。"""
import hashlib
import torch
from torch import nn

from .input import online_batch, loss_batch
from .observation import B0Observation
from .decoder import SharedHypothesisDecoder
from .prior import PhysicalTimePrior, empty_prior
from .attention import configure_attention_modules
from .multistep import FutureMotionHead, compute_multistep_loss
from .base_losses import compute_losses as base_losses
from .config import normalize_config, recipe_scope
from .contracts import CandidateSet, SelectionDecision, CommitDecision, TrackOutput
from .verification import FinalBoxVerifier, current_candidate_support, build_current_pool
from .recovery import RecoveryResidualHead


def build_candidates(decoder, observation, batch, prior, *, enable_b1=True, recovery=None):
    """仅旧三框内部去重；R 即使同几何也保留当前测量资格。"""
    recovery_box = batch['fallback_box'].detach() if recovery is None else recovery.box.detach()
    boxes = torch.stack((decoder.q0_box, prior.box.to(decoder.q0_box),
                         batch['fallback_box'].detach().to(decoder.q0_box),
                         recovery_box.to(decoder.q0_box)), 1)
    if not bool(torch.isfinite(boxes).all()):
        raise FloatingPointError('v41 position candidates must be finite')
    real = observation.sequence_valid.detach().bool()
    prior_available = torch.full_like(real, bool(enable_b1))
    recovery_valid = torch.zeros_like(real) if recovery is None else recovery.valid.detach().bool()
    raw_valid = torch.stack((real, prior_available, torch.ones_like(real), recovery_valid), -1)
    measurement = torch.stack((real, torch.zeros_like(real), torch.zeros_like(real), recovery_valid), -1)
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
    # 全空 q0 显示真正来源：B1 用 prior 槽，B0 直接用 trusted-CV 槽。
    alias[:, 0] = torch.where(~real, 1 if enable_b1 else 2, alias[:, 0])
    if not bool(valid.any(-1).all()):
        raise RuntimeError('v41 requires a finite position candidate')
    return CandidateSet(boxes, valid, raw_valid, alias, ~real,
        decoder.q0_quality_logits, measurement)


def select_candidates(candidates, verification, observation, batch, *, apply_policy=True):
    """零 margin，精确同分 q0→R→prior→fallback；传播永不获得可信资格。"""
    valid = candidates.valid
    b, device = len(valid), valid.device
    row = torch.arange(b, device=device)
    legacy = candidates.q0_quality_logits.sigmoid()
    if not bool(torch.isfinite(candidates.boxes).all()) or not bool(torch.isfinite(legacy).all()):
        raise FloatingPointError('nonfinite v41 candidate or legacy q0 quality')
    if apply_policy:
        if verification is None:
            raise ValueError('v41 Full requires the independent verifier')
        if not bool(torch.isfinite(verification.logits).all()):
            raise FloatingPointError('nonfinite v41 verifier logits')
        utility = verification.utility
        priority = torch.tensor([0, 3, 1, 2], device=device, dtype=torch.long)
        ordered = utility.masked_fill(~valid, -torch.inf)[:, priority]
        selected = priority[ordered.argmax(-1)]
        selected_utility = utility[row, selected]
        candidate_trust = torch.where(candidates.measurement_candidate,
            verification.logits[..., 2].sigmoid(), 0.)
    else:
        # B0 不伪造 B1 候选；全空时按真实的可用传播来源输出。
        propagation = torch.where(candidates.raw_valid[:, 1],
            torch.ones(b, device=device, dtype=torch.long),
            torch.full((b,), 2, device=device, dtype=torch.long))
        selected = torch.where(candidates.raw_valid[:, 0], torch.zeros(b, device=device, dtype=torch.long),
                               propagation)
        selected_utility = legacy.new_zeros(b)
        candidate_trust = legacy.new_zeros(b, 4)
        candidate_trust[:, 0] = torch.where(candidates.raw_valid[:, 0], legacy, 0.)
    selected_real = (selected == 0) & candidates.raw_valid[:, 0]
    selected_measurement = candidates.measurement_candidate[row, selected]
    support = current_candidate_support(observation, batch, candidates.boxes)
    support_mask = support['mask'][row, selected] & selected_measurement[:, None]
    trusted_score = candidate_trust[row, selected]
    eligible = selected_measurement & (support['count'][row, selected] >= 3) & (trusted_score >= .5)
    accepted = candidates.boxes[row, selected]
    selection = SelectionDecision(selected.detach(), accepted, selected_utility, selected.detach())
    commit = CommitDecision(trusted_score.detach(), legacy.detach(), support_mask.detach(),
                            eligible.detach(), selected.detach())
    # 此旧统计永远只表示实际选中的 q0 quality；不能把 R 的 T 冒充它。
    compatibility_quality = torch.where(selected_real, legacy, 0.)
    return selection, commit, compatibility_quality, candidate_trust


class JointTracker(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = normalize_config(config)
        self.enable_b1 = self.config.enable_b1
        self.enable_m = self.config.enable_m
        self.enable_b3 = self.config.enable_b3
        if (self.enable_m or self.enable_b3) and not self.enable_b1:
            raise ValueError('v41 M/B3 requires B1')
        # 与强 v37 B1 完全同序实例化；新 B3 不消耗公共初始化随机数。
        self.observation = B0Observation(token_count=128, query_context=True,
            coarse_condition=True, time_scale=self.config.time_scale)
        self.decoder = SharedHypothesisDecoder(prior_dim=128, dropout=.2, query_context=True,
            time_scale=self.config.time_scale, initial_context=True, local_observation=True)
        # B0 先完成相同 decoder 初始化再移除 adapter，公共有效参数抽样不偏移。
        if not self.enable_b1:
            self.decoder.prior_adapter = None
        # B0 初始化时消费与 B1 相同的随机流，但不注册/保留/执行 prior。
        # 工程消融的公共初始化结束 RNG 一致，B0 optimizer 中也无死参数。
        initialized_prior = PhysicalTimePrior(self.config.time_scale, self.config.v31_temporal_backend)
        self.prior = initialized_prior if self.enable_b1 else None
        del initialized_prior
        self.execution_profile = self.config.execution_profile
        if self.execution_profile != 'optimized_fp32':
            raise ValueError('v41 dense backbone/projections require optimized_fp32; BF16 is attention-core only')
        self.observation.execution_profile = self.decoder.execution_profile = self.execution_profile
        configure_attention_modules(self.decoder, backend=self.config.v41_attention_backend)
        with torch.random.fork_rng(devices=[]):
            self.verifier = FinalBoxVerifier(self.config.time_scale) if self.enable_b3 else None
        with torch.random.fork_rng(devices=[]):
            self.recovery_head = RecoveryResidualHead() if self.enable_b3 else None
        # 私有、稳定命名的 CPU 初始化 RNG；既不扰动公共参数，也不初始化 CUDA。
        with torch.random.fork_rng(devices=[]):
            seed = int.from_bytes(hashlib.sha256(
                f'{self.config.seed}:ct_v40:multistep'.encode('utf-8')).digest()[:8], 'little') % (2**63)
            torch.random.default_generator.manual_seed(seed)
            self.multistep = FutureMotionHead(self.config.time_scale) if self.enable_m else None

    def plan_prior(self, batch):
        with recipe_scope(self.config):
            prepared = online_batch(batch)
            with torch.autocast(device_type=prepared['points'].device.type, enabled=False):
                return self.prior(prepared) if self.enable_b1 else empty_prior(prepared)

    def _finish_forward(self, prepared, prior, observation, decoder):
        recovery, pool = None, None
        if self.enable_b3:
            seed = prepared['recovery_seed_box']
            seed_valid = prepared['recovery_seed_valid']
            pool = build_current_pool(observation, prepared)
            feature, statistics = self.verifier.read_local(
                pool, seed[:, None], seed_valid[:, None], prepared['box_size'])
            recovery = self.recovery_head(feature[:, 0], seed, prepared['box_size'],
                statistics['local_point_mask'][:, 0], seed_valid)
        candidates = build_candidates(decoder, observation, prepared, prior,
            enable_b1=self.enable_b1, recovery=recovery)
        # 修正后的 R 重新读取最终几何；不把 seed 支持错当成最终框支持。
        verification = self.verifier(candidates, observation, prepared, prior, pool=pool) if self.enable_b3 else None
        decision, commit, quality, candidate_trust = select_candidates(
            candidates, verification, observation, prepared, apply_policy=self.enable_b3)
        return TrackOutput(decision.accepted_box, decision.selected_index, quality, prior, observation,
            decoder, candidates, verification, decision, commit,
            candidates.q0_quality_logits.sigmoid(), decision.output_utility,
            candidate_trust, candidates.q0_is_fallback, recovery)

    def forward(self, batch, prior=None):
        with recipe_scope(self.config), torch.autocast(device_type=batch['points'].device.type, enabled=False):
            prepared = online_batch(batch)
            if not self.enable_b1:
                # 外部传入的 learned prior 也不能绕过 B0 消融。
                prior = empty_prior(prepared)
            elif prior is None:
                prior = self.prior(prepared)
            observation = self.observation(prepared)
            decoder = self.decoder(observation, prepared, prior)
            return self._finish_forward(prepared, prior, observation, decoder)

    def forward_aux(self, batch):
        """全轨迹辅助流：上游用推理统计且无图，只有 B3/R 可微。

        host 隔离 RNG、管理独立状态事务；此处精确恢复每个子模块原模式，
        不用整树 train() 破坏主训练既有的 BN 局部隔离。
        """
        if not self.enable_b3:
            raise ValueError('v41 recovery auxiliary stream requires Full')
        with recipe_scope(self.config), torch.autocast(device_type=batch['points'].device.type, enabled=False):
            prepared = online_batch(batch)
            upstream = (self.observation, self.decoder, self.prior)
            modes = [(module, module.training) for root in upstream if root is not None for module in root.modules()]
            try:
                for root in upstream:
                    if root is not None:
                        root.eval()
                with torch.no_grad():
                    prior = self.prior(prepared)
                    observation = self.observation(prepared)
                    decoder = self.decoder(observation, prepared, prior)
            finally:
                for module, training in modes:
                    module.training = training
            return self._finish_forward(prepared, prior, observation, decoder)

    def compute_aux_losses(self, batch, output):
        if not self.enable_b3:
            raise ValueError('v41 recovery auxiliary losses require Full')
        with recipe_scope(self.config):
            from .losses import compute_recovery_losses
            return compute_recovery_losses(loss_batch(batch), output,
                recovery_weight=float(self.config.get('v41_recovery_weight', .1)))

    def compute_losses(self, batch, output):
        with recipe_scope(self.config):
            prepared = loss_batch(batch)
            if not self.enable_b3:
                losses = base_losses(prepared, output, enable_b1=self.enable_b1)
            else:
                from .losses import compute_losses
                losses = compute_losses(prepared, output, enable_b1=self.enable_b1,
                    recovery_weight=float(self.config.get('v41_recovery_weight', .1)))
            # M 只属于训练损失。评测即使主动请求基础 loss，也不读取未来标签。
            if self.enable_m and self.training:
                motion = compute_multistep_loss(prepared, output.prior, self.multistep,
                    weight=self.config.v41_m_weight, horizon_weights=(1., .5))
                losses.update(motion)
                losses['loss_total'] = losses['loss_total'] + motion['loss_contribution_m']
            return losses
