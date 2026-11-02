"""独立整轨辅助流：只决定端点和插入位置，不读取 GT 或模型输出。"""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
import hashlib
import json
import math
import random


def _seed(*parts):
    return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:8], 'little') % (2**32)


def insertion_indices(main_batches, auxiliary_batches):
    """A 个辅助 batch 均匀附着于 M 个主 Adam 步；不增加优化器步。"""
    main_batches, auxiliary_batches = int(main_batches), int(auxiliary_batches)
    if main_batches < 0 or auxiliary_batches < 0 or auxiliary_batches > main_batches:
        raise ValueError('auxiliary batches must satisfy 0 <= A <= M')
    if not auxiliary_batches:
        return ()
    result = tuple(((j + 1) * main_batches) // auxiliary_batches - 1
                   for j in range(auxiliary_batches))
    if len(set(result)) != auxiliary_batches or result[0] < 0 or result[-1] >= main_batches:
        raise RuntimeError('invalid one-auxiliary-batch-per-main-step insertion plan')
    return result


def insertion_identity(indices):
    return hashlib.sha256(json.dumps(list(indices), separators=(',', ':')).encode()).hexdigest()


class AuxiliaryDiagnostics:
    """只消费已提交输出的被动台账；不提供采样、候选或状态更新输入。"""
    AGE_UPPER_BOUNDS = (.5, 1., 2., 4., 8., 16.)

    def __init__(self):
        self.reset()

    def reset(self):
        self.rows = self.depth_ge8 = self.max_depth = self.xyz_supervised = 0
        self.loss_episodes = self.recoveries = self.low_iou_frames = 0
        self.reappearance_endpoints = self.visibility_return_events = 0
        self.completed_tracks = self.unrecovered_tracks = self.max_failure_frames = 0
        self.max_failure_seconds = 0.
        self.tracks = {}
        self.ages = {name: dict(count=0, sum=0., max=0., bins=[0] * 7)
                     for name in ('trusted_age_seconds', 'supported_age_seconds')}

    def add(self, records, xyz_supervised_count=0):
        self.xyz_supervised += int(xyz_supervised_count)
        for row in records:
            frame, depth, timestamp = int(row['frame']), int(row['depth']), float(row['timestamp'])
            key = row['tracklet']
            state = self.tracks.setdefault(key, dict(last_frame=0, lost=None, sampled_target_count=None))
            if frame != state['last_frame'] + 1:
                raise ValueError('auxiliary passive diagnostics require causal complete trajectories')
            self.rows += 1
            self.depth_ge8 += depth >= 8
            self.max_depth = max(self.max_depth, depth)
            for name in self.ages:
                value = float(row[name])
                if not math.isfinite(value) or value < 0:
                    raise ValueError('auxiliary passive age must be finite and nonnegative')
                stats = self.ages[name]
                stats['count'] += 1; stats['sum'] += value; stats['max'] = max(stats['max'], value)
                index = next((i for i, edge in enumerate(self.AGE_UPPER_BOUNDS) if value < edge), 6)
                stats['bins'][index] += 1
            sampled = int(row['sampled_target_count'])
            previously_lost = state['lost'] is not None
            if previously_lost and sampled >= 3:
                self.reappearance_endpoints += 1
                if state['sampled_target_count'] is not None and state['sampled_target_count'] < 3:
                    self.visibility_return_events += 1
            iou = float(row['selected_exact_iou'])
            if not math.isfinite(iou) or not 0 <= iou <= 1:
                raise ValueError('auxiliary passive selected exact IoU must be in [0,1]')
            self.low_iou_frames += iou < .1
            if state['lost'] is None and iou < .1:
                self.loss_episodes += 1
                state['lost'] = dict(timestamp=timestamp, frames=0)
            if state['lost'] is not None:
                failure = state['lost']
                if iou >= .5:
                    self.recoveries += 1
                    self.max_failure_seconds = max(self.max_failure_seconds, timestamp - failure['timestamp'])
                    state['lost'] = None
                else:
                    failure['frames'] += 1
                    self.max_failure_frames = max(self.max_failure_frames, failure['frames'])
                    self.max_failure_seconds = max(self.max_failure_seconds, timestamp - failure['timestamp'])
            state['last_frame'], state['sampled_target_count'] = frame, sampled
            if row['ends_trajectory']:
                self.completed_tracks += 1
                self.unrecovered_tracks += state['lost'] is not None
                del self.tracks[key]

    def state_dict(self):
        return dict(schema='ct_seqtrack.v41.auxiliary_diagnostics.v1', passive_only=True,
            rows=self.rows, depth_ge8_endpoints=self.depth_ge8, max_depth=self.max_depth,
            xyz_supervised_endpoints=self.xyz_supervised,
            age_bin_intervals=['[0,.5)', '[.5,1)', '[1,2)', '[2,4)', '[4,8)', '[8,16)', '[16,infinity)'],
            ages={name: dict(value, bins=list(value['bins'])) for name, value in self.ages.items()},
            lost_definition='selected_exact_iou_lt_0.1_until_ge_0.5_recovery_frame_excluded',
            low_iou_frames=self.low_iou_frames, loss_episodes=self.loss_episodes,
            recoveries=self.recoveries, completed_trajectories=self.completed_tracks,
            unrecovered_completed_trajectories=self.unrecovered_tracks,
            max_failure_frames=self.max_failure_frames, max_failure_seconds=self.max_failure_seconds,
            reappearance_definition='current_sampled_GT_points_ge_3_and_previous_accepted_episode_lost',
            reappearance_training_endpoints=self.reappearance_endpoints,
            target_visibility_return_events=self.visibility_return_events,
            unfinished_diagnostic_trajectories=len(self.tracks))


class FullTrackAuxiliarySampler:
    """seed+epoch 打乱整轨后贪心装入 N/4；过长轨迹跳过，绝不截断/重置。"""
    SCHEMA = 'ct_seqtrack.v41.full_track_aux.v1'
    SELECTION_POLICY = 'seed_epoch_shuffled_greedy_whole_tracks_v1'
    INITIAL_SEED_POLICY = 'frame0_exact_no_reseed_v1'

    def __init__(self, lengths, batch_size=16, *, seed=42, fraction=.25, source_sha256=''):
        self.lengths = tuple(int(n) for n in lengths)
        self.batch_size, self.seed, self.epoch = int(batch_size), int(seed), 0
        self.fraction, self.source_sha256 = float(fraction), str(source_sha256)
        if self.batch_size < 1 or any(n < 0 for n in self.lengths):
            raise ValueError('auxiliary lengths and batch size must be nonnegative/positive')
        if not math.isfinite(self.fraction) or not 0 <= self.fraction <= .25:
            raise ValueError('auxiliary endpoint fraction must be between zero and one quarter')
        self._plan = self._selected = None

    @property
    def endpoint_count(self):
        return sum(max(n - 1, 0) for n in self.lengths)

    @property
    def budget(self):
        return int(math.floor(self.endpoint_count * self.fraction))

    def set_epoch(self, epoch):
        epoch = int(epoch)
        if epoch < 0:
            raise ValueError('epoch must be nonnegative')
        if self.epoch != epoch:
            self.epoch, self._plan, self._selected = epoch, None, None

    @property
    def selected_tracklets(self):
        if self._selected is None:
            candidates = [i for i, n in enumerate(self.lengths) if n >= 2]
            random.Random(_seed(self.seed, self.epoch, 'v41_auxiliary_tracks')).shuffle(candidates)
            selected, remaining = [], self.budget
            for tracklet in candidates:
                cost = self.lengths[tracklet] - 1
                if cost <= remaining:
                    selected.append(tracklet)
                    remaining -= cost
            self._selected = tuple(selected)
        return self._selected

    @property
    def row_count(self):
        return sum(self.lengths[i] - 1 for i in self.selected_tracklets)

    def _materialized(self):
        if self._plan is None:
            # 延迟导入避免 data -> auxiliary -> data 循环；请求沿用同一 typed 契约。
            from .data import EndpointRequest
            queue, active, batches = deque(self.selected_tracklets), [], []
            while queue or active:
                while queue and len(active) < self.batch_size:
                    tracklet = queue.popleft()
                    active.append(EndpointRequest(self.epoch, tracklet, 5, 1,
                                                   self.lengths[tracklet], 1))
                batches.append(tuple(active))
                active = [EndpointRequest(r.epoch, r.tracklet, 5, 1, r.window_end, r.frame + 1)
                          for r in active if not r.ends_window]
            self._plan = tuple(batches)
        return self._plan

    def __iter__(self):
        yield from self._materialized()

    def __len__(self):
        return len(self._materialized())

    def state_dict(self):
        digest = hashlib.sha256()
        for batch in self._materialized():
            digest.update(repr(tuple((r.tracklet, r.branch, r.window_start,
                                      r.window_end, r.frame, r.drain) for r in batch)).encode())
        payload = dict(schema=self.SCHEMA, epoch=self.epoch, lengths=list(self.lengths),
            source_sha256=self.source_sha256, seed=self.seed, batch_size=self.batch_size,
            endpoint_count=self.endpoint_count, fraction=self.fraction, budget=self.budget,
            selected_tracklets=list(self.selected_tracklets), rows=self.row_count,
            batches=len(self), selection_policy=self.SELECTION_POLICY,
            initial_seed_policy=self.INITIAL_SEED_POLICY, plan_sha256=digest.hexdigest())
        payload['manifest_sha256'] = hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        return payload


@contextmanager
def isolated_auxiliary_rng(seed, epoch, batch_index):
    """辅助 raw/forward/loss 使用独立 RNG，退出（含异常）恢复主流全部 RNG。"""
    import numpy as np
    import torch
    from .runtime import capture_rng_state, restore_rng_state
    state = capture_rng_state()
    try:
        auxiliary_seed = _seed(int(seed), int(epoch), int(batch_index), 'v41_auxiliary_rng')
        random.seed(auxiliary_seed)
        np.random.seed(auxiliary_seed)
        torch.manual_seed(auxiliary_seed)
        yield
    finally:
        restore_rng_state(state)
