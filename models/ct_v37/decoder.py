"""v35 共享 decoder；q0/history 无新增 KV，两个 mode 用独立残差头。"""
from dataclasses import replace

import torch
from torch import nn

from models.ct_v31.decoder import SharedHypothesisDecoder as V35Decoder, _masked_source
from models.ct_v31.contracts import DecoderOutput, EvidenceHypotheses as V35Evidence
from models.ct_v31.geometry import anchor_yaw, rotate_xy, transform_boxes
from models.ct_v31.observation import box_corners_xyz
from utils.feature_precision import feature_autocast, feature_fp32
from .contracts import CandidateSet
from .verification import build_current_pool, candidate_authorization, read_lwh_local


def _padded_evidence(evidence):
    """旧局部算子固定四候选；第三mode仅为无效槽，不重复真实点。"""
    z = lambda value: torch.zeros_like(value[:, :1])
    return V35Evidence(torch.cat((evidence.centers_xyz, z(evidence.centers_xyz)), 1),
        torch.cat((evidence.mode_valid, z(evidence.mode_valid)), 1),
        torch.cat((evidence.members, z(evidence.members)), 1),
        torch.cat((evidence.covariance_xy, z(evidence.covariance_xy)), 1),
        evidence.point_features, evidence.point_xyz, evidence.point_valid, evidence.point_ids,
        evidence.point_indices, evidence.identity_logits, evidence.selected_identity_logits,
        evidence.vote_xyz, evidence.reliability_logits, evidence.memory_features, evidence.memory_valid)


class SharedHypothesisDecoder(V35Decoder):
    def __init__(self, prior_dim=128, dropout=.2, time_scale=.5):
        super().__init__(prior_dim=prior_dim, dropout=dropout, query_context=True,
                         time_scale=time_scale, initial_context=True, local_observation=True)
        del self.memory_projection
        # 新头不消费原 B1 的初始化随机数；旧共享头仍只服务 q0/history。
        with torch.random.fork_rng(devices=[]):
            self.mode_pose_head = nn.Linear(64, 5)
            nn.init.zeros_(self.mode_pose_head.weight)
            with torch.no_grad():
                self.mode_pose_head.bias.copy_(torch.tensor([0., 0., 0., 0., 1.]))

    def forward(self, observation, evidence, batch, prior):
        reference = observation.coarse_box
        b = len(reference)
        history_valid = batch['history_valid'].bool()
        history = torch.where(history_valid[..., None], batch['history_boxes'].to(reference), 0.).detach()
        size = batch['box_size'].detach().to(reference)
        times = self._frame_times(batch, reference)
        padded = _padded_evidence(evidence)
        seeds, corner_seeds, query_valid, members, prior_box = self._seeds(
            observation, padded, history, history_valid, prior)
        yaw = anchor_yaw(batch, reference)
        local_seeds, local_corners = transform_boxes(seeds, yaw), transform_boxes(corner_seeds, yaw)
        output_reference = torch.cat((local_seeds[:, :3, :3],
            torch.zeros_like(local_seeds[:, 3:4, :3]), local_seeds[:, 4:, :3]), 1)
        local_prior = replace(prior, box=transform_boxes(prior.box.detach().to(reference), yaw),
            direction_xy=rotate_xy(prior.direction_xy.detach().to(reference), yaw))
        corners = box_corners_xyz(local_corners, size[:, None])
        query_times = torch.cat((times[:, :3], times[:, -1:].expand(-1, 4)), 1)
        corner_input = torch.cat((corners, query_times[:, :, None, None].expand(-1, -1, 8, -1),
            output_reference.detach()[:, :, None].expand(-1, -1, 8, -1)), -1)
        queries = self.corner_projection(corner_input.reshape(b, 56, 7))
        physical = self._prior_embedding(observation, local_prior, local_seeds[:, 3], size, times)
        by_query = torch.cat((torch.zeros_like(physical[:, None]).expand(-1, 3, -1),
                              physical[:, None].expand(-1, 4, -1)), 1)
        queries = queries + by_query.repeat_interleave(8, dim=1)
        context = self._query_context(observation, batch, local_seeds[:, :3], history_valid, size)
        by_query = torch.cat((torch.zeros_like(context[:, None]).expand(-1, 3, -1),
                              context[:, None].expand(-1, 4, -1)), 1)
        queries = queries + by_query.repeat_interleave(8, dim=1)
        # q0 严格沿用 v35 的 LWH 邻域与参数；一次读取中只禁止它看 extension。
        pool = build_current_pool(observation, evidence, batch)
        local_valid = torch.cat((query_valid[:, 3:4], torch.ones_like(query_valid[:, 3:4]), query_valid[:, 4:6]), 1)
        permission = candidate_authorization(pool, evidence.members, local_valid, q0_extension=False)
        local_seeds_public = torch.cat((corner_seeds[:, 3:4], prior_box[:, None], corner_seeds[:, 4:6]), 1)
        increment, local_diag = read_lwh_local(self.local_reader, pool, local_seeds_public, permission, size)
        local = torch.cat((increment[:, [0, 2, 3]], increment.new_zeros(b, 1, 8, 64)), 1)
        queries = queries + torch.cat((local.new_zeros(b, 3, 8, 64), local), 1).reshape(b, 56, 64)
        diagnostics = {key: torch.cat((value[:, [0, 2, 3]], torch.zeros_like(value[:, :1])), 1)
                       for key, value in local_diag.items()}
        source, source_valid = self.encode_sources(observation, history_valid)
        ext_valid = evidence.point_valid.bool()
        extension = self.extension_projection(_masked_source(evidence.point_features, ext_valid, 'extension'))
        extension = torch.where(ext_valid[..., None], extension, 0.)
        # 不保留任何 memory K/V；q0 与三个历史槽都看不到 extension。
        kv = torch.cat((source, extension), 1)
        ext_allowed = torch.cat((torch.zeros(b, 4, ext_valid.shape[1], dtype=torch.bool,
                                             device=reference.device), members), 1)
        allowed = torch.cat((source_valid[:, None].expand(-1, 7, -1), ext_allowed), -1)
        allowed = (allowed & query_valid[..., None]).repeat_interleave(8, dim=1)
        bf16 = self.execution_profile == 'optimized_bf16'
        with feature_autocast(queries, enabled=bf16):
            decoded, _ = self.decoder(queries, None, kv, allowed,
                query_valid=query_valid.repeat_interleave(8, dim=1)[..., None], zero_invalid_rows=True)
        decoded = feature_fp32(decoded, enabled=bf16)
        features = self.box_projection(decoded.reshape(b, 7, 512))
        features = torch.where(query_valid[..., None], features, 0.)
        residual = torch.cat((self.pose_head(features[:, :4]), self.mode_pose_head(features[:, 4:])), 1)
        sine, cosine = residual[..., 3], residual[..., 4]
        nonzero = sine.square() + cosine.square() > 1e-12
        angle = torch.atan2(torch.where(nonzero, sine, 0.), torch.where(nonzero, cosine, 1.))
        result_yaw = local_seeds[..., 3] + angle
        boxes = torch.cat((output_reference + residual[..., :3],
                           torch.atan2(result_yaw.sin(), result_yaw.cos())[..., None]), -1)
        boxes = torch.where(query_valid[..., None], transform_boxes(boxes, yaw, to_world=True), 0.)
        q0 = torch.where(observation.sequence_valid[:, None], boxes[:, 3], prior_box)
        mode_valid = query_valid[:, 4:6].clone()
        duplicate = mode_valid.all(-1) & ((boxes[:, 4, :2].detach() - boxes[:, 5, :2].detach()).norm(dim=-1) <= .75)
        mode_valid[:, 1] &= ~duplicate
        q0_logits = self.quality_head(features[:, 3]).squeeze(-1)
        legacy_boxes = torch.cat((q0[:, None], boxes[:, 4:]), 1)
        legacy_valid = torch.cat((torch.ones_like(mode_valid[:, :1]), mode_valid,
                                  torch.zeros_like(mode_valid[:, :1])), 1)
        legacy_logits = torch.cat((q0_logits[:, None], q0_logits.new_full((b, 3), -20.)), 1)
        legacy = DecoderOutput(legacy_boxes, legacy_logits, legacy_valid, boxes[:, :3],
                               features[:, 3:], query_context_norm=context.detach().norm(dim=-1), **diagnostics)
        candidates = torch.cat((q0[:, None], prior_box[:, None], boxes[:, 4:6]), 1)
        valid = torch.cat((observation.sequence_valid[:, None], torch.ones_like(mode_valid[:, :1]), mode_valid), 1)
        candidate_features = torch.cat((features[:, 3:4], features.new_zeros(b, 1, 64), features[:, 4:6]), 1)
        return CandidateSet(candidates, valid, boxes[:, :3], candidate_features, seeds[:, 4:6],
            q0_logits, ~observation.sequence_valid.bool(), legacy,
            pose_residual=residual[:, 4:6], physical_context=physical.detach(), q0_pose_raw=residual[:, 3])
