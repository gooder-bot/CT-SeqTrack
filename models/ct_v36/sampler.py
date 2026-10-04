"""四曝光、独立随机流及因果 lane 排程；调度链与状态重置分开。"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math

import numpy as np
from torch.utils.data import Sampler


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:8], 'little') % (2**32)


def endpoint_seed(seed, request, tracklet_key, source_frame, stream):
    """绝不读取 window/state/reset/drain/worker/arm；评测各 checkpoint 共用规则。"""
    epoch = request.epoch if request.branch != 4 else 0
    return stable_seed(int(seed), 'v36_sample', epoch, str(tracklet_key), request.branch,
                       request.frame, int(source_frame), str(stream))


@dataclass(frozen=True)
class EndpointRequest:
    epoch: int
    tracklet: int
    branch: int
    window_start: int
    window_end: int
    frame: int
    drain: bool = False

    @property
    def state_key(self):
        return self.epoch, self.tracklet, self.branch, self.window_start

    @property
    def starts_window(self):
        return self.frame == self.window_start

    @property
    def ends_window(self):
        return self.frame + 1 == self.window_end


class ReadyQueueBatchSampler(Sampler):
    SCHEMA = 'ct_seqtrack.v36.continuous_lanes.v1'
    RESERVE_POLICY = 'independent_short_singletons_112_v1'
    DRAIN_POLICY = 'running_b0_bn_from_first_reserve_or_partial_v1'
    SEED_POLICY = 'endpoint_streams_initial_exact_short_shared_perturb_v1'

    def __init__(self, lengths, batch_size=16, *, seed=42, training=True,
                 short_window=4, long_window=8, curriculum_epochs=10,
                 source_sha256='', annotation_sha256='', reserve_windows=112, seed_translation=.3,
                 seed_yaw_degrees=1.5, continuous=True, **unused):
        if unused:
            raise TypeError('unknown v36 sampler options: ' + ','.join(unused))
        self.lengths = tuple(int(n) for n in lengths)
        self.batch_size, self.seed, self.training = int(batch_size), int(seed), bool(training)
        self.short_window, self.long_window = int(short_window), int(long_window)
        self.curriculum_epochs, self.reserve_windows = int(curriculum_epochs), int(reserve_windows)
        self.source_sha256 = str(source_sha256)
        self.annotation_sha256 = str(annotation_sha256)
        self.seed_translation, self.seed_yaw_degrees = float(seed_translation), float(seed_yaw_degrees)
        self.continuous, self.epoch, self._plan = bool(continuous), 0, None
        if (self.batch_size < 1 or min(self.short_window, self.long_window, self.curriculum_epochs) < 1
                or any(n < 0 for n in self.lengths) or self.reserve_windows < 0):
            raise ValueError('invalid v36 sampler sizes')

    @property
    def endpoint_count(self):
        return sum(max(n - 1, 0) for n in self.lengths)

    @property
    def row_count(self):
        return self.endpoint_count * (4 if self.training else 1)

    def set_epoch(self, epoch):
        if int(epoch) < 0:
            raise ValueError('negative epoch')
        if int(epoch) != self.epoch:
            self.epoch, self._plan = int(epoch), None

    def window_length(self, maximum):
        fraction = 1. if self.curriculum_epochs == 1 else min(self.epoch / (self.curriculum_epochs - 1), 1.)
        return 1 + int(math.floor((maximum - 1) * fraction + 1e-8))

    def _jobs(self):
        jobs = []
        for tracklet, length in enumerate(self.lengths):
            for branch in (range(4) if self.training else (4,)):
                whole = not self.training or (branch == 3 and self.epoch >= self.curriculum_epochs)
                width = max(length - 1, 1) if whole else self.window_length(
                    1 if branch == 0 else self.short_window if branch in (1, 2) else self.long_window)
                start, span = 1, max(1, width // 2) if branch == 2 else width
                while start < length:
                    end = min(start + span, length)
                    jobs.append((tracklet, branch, start, end))
                    start, span = end, width
        rng = np.random.default_rng(stable_seed(self.seed, self.epoch if self.training else 0, 'v36_lanes'))
        rng.shuffle(jobs)
        return jobs, rng

    def _materialized(self):
        if self._plan is not None:
            return self._plan
        jobs, rng = self._jobs()
        singleton = [j for j in jobs if j[3] - j[2] == 1]
        reserve = []
        if self.training:
            eligible = [j for j in singleton if j[1] in (0, 1, 2)]
            # 优先短分支分层；不足只用仍然独立的 singleton 补齐。
            quotas = [self.reserve_windows // 3 + int(i < self.reserve_windows % 3) for i in range(3)]
            for branch, quota in enumerate(quotas):
                reserve.extend([j for j in eligible if j[1] == branch][:quota])
            selected = set(reserve)
            reserve += [j for j in eligible if j not in selected][:max(0, self.reserve_windows-len(reserve))]
        selected = set(reserve)
        singles = [j for j in singleton if j not in selected]
        chains = sorted((j for j in jobs if j[3] - j[2] > 1), key=lambda j: j[3]-j[2], reverse=True)
        n = self.batch_size
        reserved = [[] for _ in range(n)]
        for index, job in enumerate(reserve):
            reserved[index % n].append(job)
        capacities = [self.row_count // n + int(i < self.row_count % n) - len(reserved[i]) for i in range(n)]
        lanes, loads = [[] for _ in range(n)], [0] * n
        exact = min(capacities, default=0) >= 0
        for job in chains:
            width = job[3] - job[2]
            choices = [i for i in range(n) if loads[i] + width <= capacities[i]] if exact else []
            if not choices:
                exact = False  # 小型工程数据可能只有一条超长链；不截断或重复端点。
                choices = list(range(n))
            lane = min(choices, key=lambda i: (loads[i], i))
            lanes[lane].append(job); loads[lane] += width
        for job in singles:
            choices = [i for i in range(n) if loads[i] < capacities[i]] if exact else list(range(n))
            if not choices:
                raise RuntimeError('v36 singleton capacity mismatch')
            lane = min(choices, key=lambda i: (loads[i], i))
            lanes[lane].append(job); loads[lane] += 1
        rows = []
        for lane, ordinary in enumerate(lanes):
            rng.shuffle(ordinary)
            lane_rows = []
            for job, is_reserve in [(j, False) for j in ordinary] + [(j, True) for j in reserved[lane]]:
                tracklet, branch, start, end = job
                for frame in range(start, end):
                    state_start, state_end = start, end
                    if self.training and branch == 3 and self.epoch >= self.curriculum_epochs and not self.continuous:
                        state_start = 1 + ((frame - 1) // self.long_window) * self.long_window
                        state_end = min(state_start + self.long_window, self.lengths[tracklet])
                    lane_rows.append(EndpointRequest(self.epoch, tracklet, branch, state_start, state_end, frame, is_reserve))
            rows.append(lane_rows)
        batches, drain = [], False
        for index in range(max((len(row) for row in rows), default=0)):
            batch = [row[index] for row in rows if index < len(row)]
            drain |= len(batch) < n or any(r.drain for r in batch)
            batches.append(tuple(replace(r, drain=drain) for r in batch))
        self._plan = tuple(batches)
        return self._plan

    def __iter__(self):
        yield from self._materialized()

    def __len__(self):
        return len(self._materialized())

    def state_dict(self):
        plan = self._materialized()
        payload = dict(schema=self.SCHEMA, epoch=self.epoch, lengths=list(self.lengths),
            seed=self.seed, batch_size=self.batch_size, training=self.training, rows=self.row_count,
            short_window=self.short_window, long_window=self.long_window,
            curriculum_epochs=self.curriculum_epochs, continuous=self.continuous,
            source_sha256=self.source_sha256, reserve_windows=self.reserve_windows,
            annotation_sha256=self.annotation_sha256,
            reserve_policy=self.RESERVE_POLICY, drain_policy=self.DRAIN_POLICY, seed_policy=self.SEED_POLICY,
            seed_translation=self.seed_translation, seed_yaw_degrees=self.seed_yaw_degrees,
            batches=len(plan), plan_sha256=hashlib.sha256(repr(plan).encode()).hexdigest(),
            endpoint_order_sha256=hashlib.sha256(repr(tuple(tuple((r.tracklet,r.branch,r.frame,r.drain)
                for r in batch) for batch in plan)).encode()).hexdigest())
        payload['manifest_sha256'] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        return payload
