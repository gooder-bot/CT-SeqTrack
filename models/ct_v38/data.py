"""v38 无身份点记忆的原 W 数据流与唯一输出/可信几何提交。

B1 保留原 prepare、acquisition target、运动边与 q0 quality 状态数学。
Full 仅改变唯一输出和可信位置资格；来源风险不回写 CfC 的运动边。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from utils.diagnostic_transfer import tensors_to_numpy
from utils.execution_profile import optimized_execution
from utils.raw_frame_memo import raw_transaction
from models.ct_v31.acquisition import _unique_mask
from models.ct_v31.data import (
    AcceptedState, BatchBuilder as LegacyBatchBuilder, EndpointRequest,
    KNOWN_GT_BOX, PERTURBED_BOX, PREDICTED_BOX, RawEndpointDataset,
    ReadyQueueBatchSampler, box_array, box_size, build_loaders as legacy_loaders,
    inside_box, metadata, option, point_arrays, raw_collate, stable_seed,
)
from .config import as_v35_config, normalize_config, recipe_scope


class EmptyIdentityMemory:
    """仅保留旧 B1 张量接口；不读首帧点、不保存或更新身份记录。"""
    def export(self, timestamp):
        return dict(memory_points=np.zeros((36, 5), np.float32),
                    memory_valid=np.zeros(36, bool),
                    memory_ids=np.full(36, -1, np.int64),
                    memory_metadata=np.zeros((36, 8), np.float32))


@dataclass
class V38AcceptedState(AcceptedState):
    prediction_source: str = 'seed'
    previous_is_propagation: bool = False
    consecutive_propagations: int = 0
    consecutive_untrusted_outputs: int = 0
    last_supported_quality: float = 1.
    last_selected_commit_quality: float = 1.
    last_velocity_update_time: float | None = None


def build_loaders(config, roles=('train', 'val'), sources=None):
    config = normalize_config(config)
    with recipe_scope(config):
        return legacy_loaders(as_v35_config(config), roles=roles, sources=sources)


build_dataloaders = build_loaders


class BatchBuilder(LegacyBatchBuilder):
    def __init__(self, config):
        self.v38_config = normalize_config(config)
        super().__init__(as_v35_config(self.v38_config))

    def _new_state(self, row):
        # 与 v35 相同的合法 seed 几何和局部 RNG；仅删除身份点记忆初始化。
        r, frames = row['request'], row['frames']
        seed_ids = range(max(0, r.frame - 3), r.frame)
        size = box_size(row['first_frame']['3d_bbox'], row['first_frame'])
        boxes = {i: box_array(frames[i]['3d_bbox']) for i in seed_ids}
        times = {i: float(frames[i]['timestamp']) for i in seed_ids}
        seed_offset = np.zeros(4, dtype=np.float64)
        provenance = KNOWN_GT_BOX
        if r.branch in (1, 2, 3) and not (self.is_v35 and r.frame == 1):
            rng = np.random.default_rng(stable_seed(option(self.config, 'seed', 42),
                row['tracklet_key'], r.state_key, 'window_seed'))
            local_xy = rng.uniform(-float(option(self.config, 'v32_seed_translation', .3)),
                                   float(option(self.config, 'v32_seed_translation', .3)), 2)
            yaw = boxes[r.frame - 1][3]
            c, s = np.cos(yaw), np.sin(yaw)
            seed_offset[:2] = np.asarray([[c, -s], [s, c]]) @ local_xy
            limit = np.deg2rad(float(option(self.config, 'v32_seed_yaw_degrees', 1.5)))
            seed_offset[3] = rng.uniform(-limit, limit)
            for box in boxes.values():
                box[:3] += seed_offset[:3]
                box[3] = np.arctan2(np.sin(box[3] + seed_offset[3]),
                                    np.cos(box[3] + seed_offset[3]))
            provenance = PERTURBED_BOX
        trusted = [(boxes[i].copy(), times[i]) for i in seed_ids][-2:]
        velocity = np.zeros(3, dtype=np.float64)
        if len(trusted) == 2:
            gap = trusted[-1][1] - trusted[-2][1]
            if gap <= 0:
                raise ValueError('seed timestamps must strictly increase')
            velocity = (trusted[-1][0][:3] - trusted[-2][0][:3]) / gap
        last_time = times[r.frame - 1]
        strong_times = [times[i] for i in seed_ids
                       if int(inside_box(self._points(frames[i])[0], boxes[i], size).sum()) >= 3]
        return V38AcceptedState(boxes=boxes, times=times, next_frame=r.frame, size=size,
            memory=EmptyIdentityMemory(), trusted=trusted, transitions={i: True for i in seed_ids},
            trusted_velocity=velocity, trusted_velocity_valid=len(trusted) == 2,
            previous_quality=1., last_strong_time=max(strong_times) if strong_times else None,
            last_supported_time=last_time, initialization_time=last_time,
            previous_innovation=0., box_provenance={i: provenance for i in seed_ids},
            seed_offset=seed_offset,
            last_velocity_update_time=last_time if len(trusted) == 2 else None)

    @staticmethod
    def _velocity_age(state, timestamp):
        # 无效速度不用年龄伪造新鲜度；台账仍保留最后一次合法更新时间。
        if not state.trusted_velocity_valid or state.last_velocity_update_time is None:
            return 0.
        return max(0., float(timestamp) - state.last_velocity_update_time)

    def _prepare_row(self, row, state):
        result, context = super()._prepare_row(row, state)
        result.update(previous_is_propagation=np.bool_(state.previous_is_propagation),
            consecutive_propagations=np.int64(state.consecutive_propagations),
            consecutive_untrusted_outputs=np.int64(state.consecutive_untrusted_outputs),
            trusted_velocity_age=np.float32(self._velocity_age(state, context['timestamp'])))
        return result, context

    @raw_transaction
    def prepare(self, rows):
        with recipe_scope(self.v38_config):
            return super().prepare(rows)

    @raw_transaction
    def acquire(self, batch, prior, *, training=True):
        with recipe_scope(self.v38_config):
            # 底层 arm 固定 b1：无扩展点获取，原 band/acquisition 监督仍保留。
            return super().acquire(batch, prior, training=training)

    @raw_transaction
    def commit(self, output, batch, *, diagnostics=False):
        with recipe_scope(self.v38_config):
            return self._commit(output, batch, diagnostics=diagnostics)

    def _commit(self, output, batch, *, diagnostics=False):
        if self._pending is None:
            raise RuntimeError('no pending v38 transaction (duplicate commit)')
        full = self.v38_config.v38_arm == 'full'
        selected = output.selected_index.detach().long()
        quality = output.selected_quality.detach()
        b = len(self._pending)
        if selected.shape != (b,) or quality.shape != (b,):
            raise ValueError('v38 selection/quality must match pending rows')
        if bool(((selected < 0) | (selected > (2 if full else 1))).any()):
            raise ValueError('v38 selection must index q0/prior/trusted fallback')
        if not bool(torch.isfinite(quality).all()):
            raise ValueError('v38 previous quality must be finite')
        fallback = getattr(output, 'q0_is_fallback', None)
        if fallback is None:
            sequence = getattr(output.observation, 'sequence_valid', None)
            fallback = (~sequence.bool() if sequence is not None else torch.zeros_like(selected, dtype=torch.bool))
        fallback = fallback.detach().bool()
        if fallback.shape != (b,):
            raise ValueError('v38 fallback mask must match pending rows')
        propagation = (selected != 0) | fallback
        trusted_score = quality
        expected_eligible = None
        canonical_support = None
        if full:
            candidates = output.candidates
            row = torch.arange(b, device=selected.device)
            if candidates.boxes.shape != (b, 3, 4) or candidates.valid.shape != (b, 3):
                raise ValueError('v38 Full requires exactly three candidate slots')
            if not bool(candidates.valid[row, selected].all()) or bool(((selected == 0) & fallback).any()):
                raise ValueError('v38 cannot commit invalid or propagation-alias q0')
            if not torch.equal(output.accepted_box.detach(), candidates.boxes.detach()[row, selected]):
                raise ValueError('accepted box differs from unique selected candidate')
            legacy = output.legacy_q0_quality.detach()
            if legacy.shape != (b,):
                raise ValueError('v38 requires separate legacy q0 quality')
            expected_quality = torch.where(propagation, torch.zeros_like(legacy), legacy)
            if not torch.equal(quality, expected_quality):
                raise ValueError('previous quality must be legacy q0 quality or propagation zero')
            trusted_score = output.commit_decision.trusted_score.detach()
            expected_eligible = output.commit_decision.eligible.detach().bool()
            if trusted_score.shape != (b,) or expected_eligible.shape != (b,):
                raise ValueError('v38 commit decision must match pending rows')
            if (not bool(torch.isfinite(trusted_score).all())
                    or bool(((trusted_score < 0) | (trusted_score > 1)).any())
                    or bool((trusted_score[propagation] != 0).any())):
                raise ValueError('propagation cannot carry trusted quality')
            from .verification import current_q0_support
            canonical_support = current_q0_support(output.observation, batch, candidates.boxes[:, 0])['mask']
            canonical_support = canonical_support & ~propagation[:, None]
            if not torch.equal(output.commit_decision.support_mask.detach().bool(), canonical_support):
                raise ValueError('v38 commit support differs from actual unique current-B0 set')
            if not torch.equal(output.commit_decision.source.detach().long(), selected):
                raise ValueError('v38 commit source differs from selected candidate')
        transferred = tensors_to_numpy(dict(boxes=output.accepted_box, quality=quality,
            trusted_score=trusted_score, selected=selected, propagation=propagation,
            probabilities=output.observation.foreground_probability[:, -1],
            point_ids=batch['point_ids'][:, -1], point_valid=batch['point_valid'][:, -1],
            prior_boxes=output.prior.box), optimized=optimized_execution(self.config))
        eligible_cpu = None if expected_eligible is None else expected_eligible.cpu().numpy()
        support_cpu = None if canonical_support is None else canonical_support.cpu().numpy()
        staged = []
        # 先独立复算全部授权支持与提交资格，任何不一致都不留下半个 batch 的状态。
        for i, context in enumerate(self._pending):
            state, row = context['state'], context['row']
            box = transferred['boxes'][i].astype(np.float64, copy=True)
            box[:3] += context['anchor'][:3]
            if box.shape != (4,) or not np.isfinite(box).all():
                raise FloatingPointError('non-finite v38 accepted box')
            ids = transferred['point_ids'][i, transferred['point_valid'][i]]
            probs = transferred['probabilities'][i, transferred['point_valid'][i]]
            cloud, _ = self._points(row['frames'][row['request'].frame])
            id_to_index = self._raw_index(row['frames'][row['request'].frame])
            xyz = cloud[[id_to_index[int(value)] for value in ids]]
            support = _unique_mask(ids) & np.isfinite(probs) & (probs >= .5) & inside_box(xyz, box, state.size)
            raw_count = int(support.sum())
            # Full 的 T 标签、硬门与提交共用同一 float32 公开点集合，避免
            # raw-world double 与 anchor-local float 的边界差异偷偷改提交资格。
            count = int(support_cpu[i].sum()) if full else raw_count
            prop = bool(transferred['propagation'][i])
            # B1 原状态数学不读新 T；Full 的传播输出没有当前测量资格。
            strong = count >= self.v38_config.v38_min_support and (not prop if full else True)
            score = float(transferred['trusted_score'][i])
            supported = strong and bool(np.isfinite(score) and score >= self.v38_config.v38_trust_threshold)
            if full and bool(eligible_cpu[i]) != supported:
                raise ValueError('v38 commit eligibility disagrees with unique current-B0 support')
            staged.append((context, box, count, raw_count, strong, supported, prop, score))
        records = []
        for i, (context, box, count, raw_count, strong, supported, prop, score) in enumerate(staged):
            row, state, now = context['row'], context['state'], context['timestamp']
            r = row['request']
            state.boxes[r.frame] = box
            state.box_provenance[r.frame] = PREDICTED_BOX
            state.times[r.frame] = now
            state.next_frame += 1
            innovation = float(np.linalg.norm(transferred['boxes'][i, :2] - transferred['prior_boxes'][i, :2]))
            pair_valid = innovation <= max(.5 * np.hypot(*state.size[:2]), 1e-3)
            state.transitions[r.frame] = pair_valid
            state.previous_innovation = innovation
            state.previous_quality = float(transferred['quality'][i])
            if strong:
                state.last_strong_time = now
            velocity_updated = False
            velocity_reason = 'unsupported_retained'
            if supported:
                previous, previous_time = state.trusted[-1]
                if pair_valid and previous_time == state.times.get(r.frame - 1):
                    gap = now - previous_time
                    state.trusted_velocity = (box[:3] - previous[:3]) / gap
                    state.trusted_velocity_valid = True
                    state.last_velocity_update_time = now
                    velocity_updated = True
                    velocity_reason = 'supported_adjacent'
                else:
                    state.trusted_velocity = np.zeros(3, dtype=np.float64)
                    state.trusted_velocity_valid = False
                    velocity_reason = 'invalid_transition' if not pair_valid else 'nonadjacent_supported'
                state.trusted.append((box.copy(), now))
                state.trusted[:] = state.trusted[-2:]
                state.last_supported_time = now
                state.last_supported_quality = score
            source = ('q0', 'prior', 'trusted_propagation')[int(transferred['selected'][i])]
            if prop and source == 'q0':
                source = 'prior'
            state.prediction_source = source
            state.previous_is_propagation = prop
            state.consecutive_propagations = state.consecutive_propagations + 1 if prop else 0
            state.consecutive_untrusted_outputs = 0 if supported else state.consecutive_untrusted_outputs + 1
            state.last_selected_commit_quality = score
            if diagnostics:
                records.append(dict(memory_write=False, memory_fg_count=0, memory_true_fg_count=0,
                    memory_wrong_write=False, strong=bool(strong), supported=bool(supported),
                    geometric_fg_count=count if not (full and prop) else 0,
                    raw_geometry_fg_count=raw_count,
                    trusted_velocity_valid=bool(state.trusted_velocity_valid),
                    trusted_velocity_reset_reason=velocity_reason,
                    weak_age=self._strong_age(state, now), supported_age=max(0., now - state.last_supported_time),
                    trusted_velocity_age=self._velocity_age(state, now),
                    trusted_position_updated=bool(supported), trusted_velocity_updated=velocity_updated,
                    trusted_write=bool(supported), propagation_only=prop, commit_source=source,
                    commit_quality=score, last_supported_quality=state.last_supported_quality,
                    last_velocity_update_time=state.last_velocity_update_time,
                    consecutive_propagations=state.consecutive_propagations,
                    consecutive_untrusted_outputs=state.consecutive_untrusted_outputs,
                    support_policy='none' if prop else 'current_b0_only'))
            for old in list(state.boxes):
                if old < r.frame - 2:
                    state.boxes.pop(old)
                    state.times.pop(old)
                    state.transitions.pop(old, None)
                    state.box_provenance.pop(old, None)
            if r.ends_window:
                self.states.pop(r.state_key)
        self._pending = None
        return records
