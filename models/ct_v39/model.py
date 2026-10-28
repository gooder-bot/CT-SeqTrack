"""v39 因果事务：先 G 路由，按需 B0，统一七槽选择与唯一提交证据。"""
from dataclasses import fields
from types import SimpleNamespace
import time

import torch
from torch import nn

from models.ct_v38.observation import B0Observation
from models.ct_v38.decoder import SharedHypothesisDecoder
from models.ct_v38.input import canonicalize_batch
from models.ct_v38.data import stable_seed
from .config import normalize_config, recipe_scope
from .prior import PhysicalTimePrior, PriorContext
from .multistep import FutureMotionHead
from .efficient_observation import configure_efficient_modules
from .g_observation import GObservation
from .gpu_prepare import merge_pools, pad_pool


SOURCES = ('q0', 'prior', 'trusted_cv', 'g_q0', 'g_prior', 'g_trusted_cv', 'g_reacquire')


def subset_prior(prior, index):
    return PriorContext(**{f.name: (getattr(prior, f.name)[index]
        if torch.is_tensor(getattr(prior, f.name)) else getattr(prior, f.name)) for f in fields(prior)})


def geometry_aliases(boxes, valid):
    """仅记录几何别名；不删除观测来源和相同框的写入资格。"""
    alias = torch.full_like(valid, -1, dtype=torch.long)
    for later in range(boxes.shape[1]):
        for earlier in range(later):
            delta = boxes[:, later].detach() - boxes[:, earlier].detach()
            same = (delta[:, :3].abs().amax(-1) <= 1e-4)
            same &= torch.atan2(delta[:, 3].sin(), delta[:, 3].cos()).abs() <= 1e-5
            duplicate = same & valid[:, earlier] & valid[:, later] & (alias[:, later] < 0)
            alias[:, later] = torch.where(duplicate, earlier, alias[:, later])
    return alias


def choose(verification, valid, measured=None):
    score = verification['utility'].detach().masked_fill(~valid, -torch.inf)
    # 几何完全相同且效用相同的情况下保留有当前观测证据的来源。
    best = score.amax(-1, keepdim=True)
    tied = valid & (score == best)
    if measured is not None:
        observed_tie = tied & measured
        tied = torch.where(observed_tie.any(-1, keepdim=True), observed_tie, tied)
    return tied.long().argmax(-1)


def route_decision(inputs, boxes, valid, measured, verification, config):
    b = len(boxes); row = torch.arange(b, device=boxes.device)
    selected = choose(verification, valid, measured)
    delta = boxes.detach() - boxes.detach()[row, selected, None]
    same = (delta[..., :3].abs().amax(-1) <= 1e-4)
    same &= torch.atan2(delta[..., 3].sin(), delta[..., 3].cos()).abs() <= 1e-5
    competitor = verification['utility'].detach().masked_fill(~valid | same, -torch.inf).amax(-1)
    gap = verification['utility'].detach()[row, selected] - competitor
    trust = verification['logits'][row, selected, 2].detach().sigmoid()
    cheap = measured[row, selected] & valid[row, selected]
    cheap &= trust >= config.v39_route_threshold
    cheap &= verification['support_count'][row, selected] >= config.v39_route_min_support
    cheap &= verification['region_no_match_logits'][row, selected].detach().sigmoid() <= config.v39_route_nomatch_max
    cheap &= gap >= config.v39_route_margin
    cheap &= inputs['consecutive_cheap'] < config.v39_max_cheap_frames
    cheap &= inputs['full_age'] <= config.v39_max_full_gap
    return selected, cheap, gap


def row_select_pool(first, second, use_first):
    values = []
    for i in range(len(use_first)):
        pool = first if bool(use_first[i]) else second
        valid = pool['valid'][i]
        values.append((pool['xyz'][i, valid], pool['point_ids'][i, valid]))
    return pad_pool(values)


class V39Model(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = normalize_config(config)
        cfg = self.config
        self.observation = B0Observation(token_count=128, query_context=True,
            coarse_condition=True, time_scale=cfg.time_scale)
        self.decoder = SharedHypothesisDecoder(prior_dim=128, dropout=.2, query_context=True,
            time_scale=cfg.time_scale, initial_context=True, local_observation=True)
        self.prior = PhysicalTimePrior(cfg.time_scale, cfg.v31_temporal_backend)
        with torch.random.fork_rng(devices=[]):
            self.multistep = FutureMotionHead(cfg.time_scale)
            self.g = GObservation(feature_dim=cfg.v39_g_feature_dim, k=cfg.v39_local_neighbors,
                sinkhorn_iterations=cfg.v39_sinkhorn_iterations,
                match_probability_threshold=cfg.v39_g_match_probability_threshold,
                match_radius_min=cfg.v39_g_match_radius_min, match_radius_max=cfg.v39_g_match_radius_max,
                match_radius_fraction=cfg.v39_g_match_radius_fraction)
        self.g.feature_bf16 = cfg.v39_feature_bf16
        configure_efficient_modules(self.observation, self.decoder, cfg)

    def plan_prior(self, inputs):
        return self.prior(inputs)

    def _verify(self, pool, templates, boxes, size, valid, measured, et=None, ep=None):
        return self.g.verify(pool, templates, boxes.detach(), size,
            candidate_valid=valid, measurement_eligible=measured,
            encoded_pool=ep, encoded_templates=et)

    def forward_transaction(self, tx, builder):
        with recipe_scope(self.config):
            return self._forward_transaction(tx, builder)

    @torch.no_grad()
    def _prefix(self, tx, builder, prior):
        cfg = self.config; inputs = tx.inputs; b = len(tx.rows)
        row = torch.arange(b, device=tx.device); size = inputs['box_size']
        # G 边界：B0/B1 几何全部 detach，G 不读取任何 supervision 字段。
        centers = torch.stack((prior.box.detach(), inputs['fallback_box'].detach()), 1)
        pool = builder.prepare_current_G(tx, centers)
        normal_pool = pool
        et = self.g.encode_templates(tx.templates, size)
        ep = self.g.encode(pool, size)
        prefix_refinement = self.g.refine(pool, tx.templates, centers, size,
                                        encoded_pool=ep, encoded_templates=et)
        boxes = prior.box.detach()[:, None].expand(-1, 7, -1).clone()
        boxes[:, 2] = centers[:, 1]
        boxes[:, 4:6] = prefix_refinement['boxes'].detach()
        valid = torch.zeros((b, 7), device=tx.device, dtype=torch.bool)
        valid[:, 1:3] = True
        valid[:, 4:6] = prefix_refinement['valid']
        measured = valid.clone(); measured[:, :3] = False
        prefix = self._verify(pool, tx.templates, boxes, size, valid, measured, et, ep)
        # presence 与 correspondence 分开：存在目标但对应困难时仍交给完整 B0。
        local_absent = prefix['region_no_match_logits'][:, 1:3].detach().sigmoid().amin(-1) >= cfg.v39_search_nomatch_min
        stale = (inputs['consecutive_untrusted_outputs'] >= 2) & (inputs['supported_age'] >= cfg.v39_search_untrusted_seconds)
        local_absent |= pool['valid'].sum(-1) < cfg.v39_g_min_support
        search = local_absent | stale
        recovery_refinement = None
        recovery_inputs = None
        if bool(search.any()):
            coarse_pool = builder.prepare_search(tx, centers, prior, search)
            proposals = self.g.coarse_proposals(coarse_pool, tx.templates, size, centers[:, 1, 3],
                encoded_templates=et, voxel_size=cfg.v39_coarse_voxel_size, peaks=cfg.v39_coarse_peaks)
            allowed = proposals['valid'] & search[:, None]
            fine_pool = builder.prepare_current_G(tx, proposals['boxes'], allowed=allowed,
                                                  maximum=cfg.v39_refine_crop_points)
            fine_ep = self.g.encode(fine_pool, size)
            recovery_refinement = self.g.refine(fine_pool, tx.templates, proposals['boxes'], size,
                encoded_pool=fine_ep, encoded_templates=et, reacquire=True)
            recovery_refinement['supervision_valid'] = allowed
            recovery_inputs = (fine_pool, proposals['boxes'].detach(), allowed)
            recovery_valid = allowed & recovery_refinement['valid']
            rv = self._verify(fine_pool, tx.templates, recovery_refinement['boxes'], size,
                              recovery_valid, recovery_valid, et, fine_ep)
            chosen = choose(rv, recovery_valid, recovery_valid)
            boxes[:, 6] = recovery_refinement['boxes'][row, chosen].detach()
            valid[:, 6] = recovery_valid[row, chosen]
            measured[:, 6] = valid[:, 6]
            pool = merge_pools(pool, fine_pool)
            ep = self.g.encode(pool, size)
            prefix = self._verify(pool, tx.templates, boxes, size, valid, measured, et, ep)
        return pool, normal_pool, centers, boxes, valid, measured, prefix, search, local_absent, recovery_inputs

    def _forward_transaction(self, tx, builder):
        cfg = self.config; inputs = tx.inputs; b = len(tx.rows)
        row = torch.arange(b, device=tx.device); size = inputs['box_size']
        start = time.perf_counter()
        prior = self.plan_prior(inputs)
        tx.timings['state_prior_host_seconds'] = time.perf_counter()-start
        (pool, normal_pool, centers, boxes, valid, measured, prefix, search,
         local_absent, recovery_inputs) = self._prefix(tx, builder, prior)
        prefix_selected, eligible_cheap, margin = route_decision(inputs, boxes, valid, measured, prefix, cfg)
        cheap = eligible_cheap & (inputs['route_window'] if tx.training else
                                  torch.full_like(eligible_cheap, cfg.v39_inference_mode == 'adaptive'))
        policy_full = ~cheap
        # 选 crop 的规则只读取 prefix；当前 teacher B0 不得反过来决定轻路径。
        trust = prefix['logits'][..., 2].detach().sigmoid() * measured
        good_crop = valid & measured & (trust >= cfg.v39_g_trust_threshold)
        good_crop &= prefix['support_count'] >= cfg.v39_g_min_support
        good_crop &= prefix['region_no_match_logits'].detach().sigmoid() < .5
        recovery_probe = local_absent & valid[:, 6] & (trust[:, 6] >= cfg.v39_recovery_crop_trust_threshold)
        recovery_probe &= prefix['support_count'][:, 6] >= cfg.v39_g_min_support
        good_crop[:, 6] |= recovery_probe
        crop_choice = choose(prefix, good_crop, measured)
        default_crop = torch.where(inputs['trusted_velocity_valid'][:, None], centers[:, 1], centers[:, 0])
        crop_boxes = torch.where(good_crop.any(-1)[:, None], boxes[row, crop_choice], default_crop)
        full_executed = torch.ones_like(cheap) if tx.training else policy_full
        if not tx.training and bool(cheap.all()):
            # 已通过当前观测核验的轻路径直接使用 prefix 证据提交。
            # 不物化 B0，也不重复模板编码、无效 q0 拟合及七槽匹配。
            return self._decision_output(tx, prior, boxes, valid, measured, prefix,
                prefix_selected, size.new_zeros(b), pool, None, None, full_executed,
                policy_full, search, margin, prefix_selected, eligible_cheap, crop_boxes, [])
        indices = full_executed.nonzero(as_tuple=False).flatten()
        observation = decoder = None
        q0_quality = size.new_zeros(b)
        q0_boxes = prior.box.detach().clone()
        q0_valid = torch.zeros_like(cheap); q0_measured = q0_valid.clone()
        if len(indices):
            heavy, targets = builder.materialize_B0(tx, indices, crop_boxes)
            prepared = canonicalize_batch(heavy)
            observation = self.observation(prepared)
            decoder = self.decoder(observation, prepared, subset_prior(prior, indices))
            q0_boxes[indices] = decoder.q0_box.detach()
            q0_quality[indices] = decoder.q0_quality_logits.detach().sigmoid()
            q0_valid[indices] = observation.sequence_valid.detach()
            q0_measured[indices] = prepared['point_valid'][:, -1].any(-1) & q0_valid[indices]
            tx.b0_inputs = prepared
        boxes[:, 0] = q0_boxes
        valid[:, 0] = q0_valid; measured[:, 0] = q0_measured
        # 所有候选统一核验已有当前证据，不按 teacher q0 增补轻路径观测。
        # 正常双中心最多512；恢复触发时额外两个精化区各256，原云按ID去重。
        final_pool = pool
        et = self.g.encode_templates(tx.templates, size)
        final_ep = self.g.encode(final_pool, size)
        q0_refinement = self.g.refine(final_pool, tx.templates, q0_boxes[:, None], size,
                                     encoded_pool=final_ep, encoded_templates=et)
        q0_refinement['supervision_valid'] = q0_valid[:, None]
        boxes[:, 3] = q0_refinement['boxes'][:, 0].detach()
        valid[:, 3] = q0_refinement['valid'][:, 0] & q0_valid
        measured[:, 3] = valid[:, 3]
        verification = self._verify(final_pool, tx.templates, boxes, size, valid, measured, et, final_ep)
        selected_full = choose(verification, valid, measured)
        selected = torch.where(cheap, prefix_selected, selected_full)
        refinements = [q0_refinement]
        if tx.training:
            # 路由不保留计算图；训练在相同原始池上重算可微纠偏，模板随当前权重编码。
            prefix_refinement = self.g.refine(normal_pool, tx.templates, centers, size,
                                               encoded_templates=et)
            refinements.append(prefix_refinement)
            if recovery_inputs is not None:
                recovery_pool, recovery_boxes, recovery_allowed = recovery_inputs
                recovery_refinement = self.g.refine(recovery_pool, tx.templates, recovery_boxes, size,
                    encoded_templates=et, reacquire=True)
                recovery_refinement['supervision_valid'] = recovery_allowed
                refinements.append(recovery_refinement)
        output = self._decision_output(tx, prior, boxes, valid, measured, verification,
            selected, q0_quality, final_pool, observation, decoder, full_executed,
            policy_full, search, margin, prefix_selected, eligible_cheap, crop_boxes, refinements)
        if tx.training:
            output.loss_inputs = dict(tx.b0_inputs, **tx.b0_targets)
            output.auxiliary = self._auxiliary(tx, final_pool, final_ep, et, q0_boxes)
        return output

    def _decision_output(self, tx, prior, boxes, valid, measured, verification, selected,
                         q0_quality, pool, observation, decoder, full_executed, policy_full,
                         search, margin, prefix_selected, eligible_cheap, crop_boxes, refinements):
        cfg = self.config
        row = torch.arange(len(boxes), device=boxes.device)
        accepted = boxes[row, selected]
        measured_selected = measured[row, selected]
        support = verification['support_mask'][row, selected] & measured_selected[:, None]
        selected_trust = verification['logits'][row, selected, 2].detach().sigmoid() * measured_selected
        trusted = measured_selected & (selected_trust >= cfg.v39_g_trust_threshold)
        trusted &= support.sum(-1) >= cfg.v39_g_min_support
        trusted &= verification['region_no_match_logits'][row, selected].detach().sigmoid() < .5
        quality = torch.where(selected == 0, q0_quality, verification['q_logits'][row, selected].detach().sigmoid())
        quality = quality * measured_selected
        candidates = SimpleNamespace(boxes=boxes, valid=valid, raw_valid=valid.clone(),
            measurement_eligible=measured, alias_of=geometry_aliases(boxes, valid),
            q0_quality_logits=None if decoder is None else decoder.q0_quality_logits,
            q0_is_fallback=~valid[:, 0])
        output = SimpleNamespace(prior=prior, observation=observation, decoder=decoder,
            candidates=candidates, verification=verification, selected_index=selected.detach(),
            accepted_box=accepted, selected_quality=quality.detach(),
            commit_decision=SimpleNamespace(eligible=trusted.detach(), support_mask=support.detach(),
                trusted_score=selected_trust.detach(), legacy_quality=quality.detach()),
            pool=pool, full_executed=full_executed, policy_full=policy_full,
            search_triggered=search, refinements=refinements, route_margin=margin,
            prefix_selected=prefix_selected, prefix_cheap=eligible_cheap,
            q0_quality=q0_quality, loss_inputs=None,
            crop_boxes=crop_boxes, loss_statistics={})
        return output

    def _auxiliary(self, tx, pool, ep, et, q0):
        """固定 hash 25%端点的近/远扰动仅监督 G，不加入七槽和状态。"""
        perturb = q0.detach()[:, None].expand(-1, 2, -1).clone()
        active = torch.zeros(len(tx.rows), dtype=torch.bool, device=tx.device)
        for i, row in enumerate(tx.rows):
            r = row['request']
            seed = stable_seed(self.config.seed, r.epoch, row['tracklet_key'], r.branch, r.frame, 'g_aux')
            active[i] = seed / 2**32 < .25
            generator = torch.Generator(device='cpu').manual_seed(seed)
            draw = torch.rand((2, 3), generator=generator).to(tx.device)
            radius = torch.stack((.5*draw[0, 0], 1.+draw[1, 0])) * tx.inputs['box_size'][i, :2].norm()
            angle = draw[:, 1] * (2.*torch.pi)
            perturb[i, :, :2] += radius[:, None] * torch.stack((angle.cos(), angle.sin()), -1)
            perturb[i, :, 3] += (draw[:, 2]*2.-1.) * (torch.pi/6.)
        valid = active[:, None].expand(-1, 2)
        verification = self._verify(pool, tx.templates, perturb, tx.inputs['box_size'], valid, valid, et, ep)
        refinement = self.g.refine(pool, tx.templates, perturb, tx.inputs['box_size'], encoded_pool=ep, encoded_templates=et)
        refinement['supervision_valid'] = valid
        return SimpleNamespace(boxes=perturb, verification=verification, refinement=refinement, active=active)

    def compute_losses(self, tx, output):
        from .losses import compute_losses
        return compute_losses(tx, output, self, self.config)


JointTracker = V39Model
