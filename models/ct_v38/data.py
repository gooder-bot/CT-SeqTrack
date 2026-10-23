"""v38 原始端点、W 课程与 accepted 状态；worker 不拥有预测状态。"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import hashlib
import json
import math

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from utils.diagnostic_transfer import tensors_to_numpy
from utils.execution_profile import optimized_execution
from utils.raw_frame_memo import RawFrameMemo, raw_transaction
from .config import normalize_config, recipe_scope
from .acquisition_supervision import _unique_mask, band_grid_target


def option(config, name, default=None):
    return config.get(name, default) if isinstance(config, dict) else getattr(config, name, default)


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:8], 'little') % (2**32)


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
    """预取只排原始帧；每个 batch 的每条独立窗口最多一个端点。

固定 slot 完成一个窗口后从 ready queue 取下一个窗口。DataLoader 可预取
未来 raw batch，主进程按序消费时前一帧已经提交，故无需 worker IPC 状态。
"""
    SCHEMA = 'ct_seqtrack.v32.ready_queue.v2'
    RESERVE_POLICY = 'stratified_singleton_windows_then_fill_v1'
    DRAIN_POLICY = 'running_b0_bn_from_first_reserve_or_partial_v1'
    SEED_POLICY = 'shared_anchor_local_translation_world_yaw_v1'
    INITIAL_SEED_POLICY = 'initial_exact_other_perturbed_v1'

    def __init__(self, lengths, batch_size=16, *, seed=42, training=True,
                 short_window=3, long_window=8, curriculum_epochs=10, source_sha256='',
                 reserve_windows=112, seed_translation=.3, seed_yaw_degrees=1.5,
                 initial_seed_policy=None):
        self.lengths = tuple(int(n) for n in lengths)
        self.batch_size = int(batch_size)
        self.seed, self.training, self.epoch = int(seed), bool(training), 0
        self.short_window, self.long_window = int(short_window), int(long_window)
        self.curriculum_epochs = int(curriculum_epochs)
        self.source_sha256 = str(source_sha256)
        self.reserve_windows = int(reserve_windows)
        self.seed_translation = float(seed_translation)
        self.seed_yaw_degrees = float(seed_yaw_degrees)
        if initial_seed_policy not in (None, self.INITIAL_SEED_POLICY):
            raise ValueError('unknown initial seed policy')
        self.initial_seed_policy = initial_seed_policy
        if self.batch_size < 1 or min(self.short_window, self.long_window, self.curriculum_epochs) < 1:
            raise ValueError('v31 batch/window/curriculum sizes must be positive')
        if any(n < 0 for n in self.lengths):
            raise ValueError('negative tracklet length')
        if self.reserve_windows < 0 or min(self.seed_translation, self.seed_yaw_degrees) < 0:
            raise ValueError('v32 reserve and seed perturbation must be nonnegative')
        self._plan = None

    @property
    def endpoint_count(self):
        return sum(max(n - 1, 0) for n in self.lengths)

    @property
    def row_count(self):
        return self.endpoint_count * (4 if self.training else 1)

    def set_epoch(self, epoch):
        epoch = int(epoch)
        if epoch < 0:
            raise ValueError('epoch must be nonnegative')
        if epoch != self.epoch:
            self.epoch, self._plan = epoch, None

    def window_length(self, maximum):
        if self.curriculum_epochs == 1:
            return int(maximum)
        fraction = min(self.epoch / (self.curriculum_epochs - 1), 1.)
        return 1 + int(math.floor((maximum - 1) * fraction + 1e-8))

    def _windows(self):
        windows = []
        for tracklet, length in enumerate(self.lengths):
            if length < 2:
                continue
            for branch in (range(4) if self.training else (4,)):
                maximum = (1 if branch == 0 else self.short_window if branch in (1, 2)
                           else self.long_window)
                width = self.window_length(maximum) if self.training else length
                # branch 2 的首个短窗口改变后续边界；不改变端点集合。
                first_width = max(1, width // 2) if branch == 2 else width
                start, span = 1, first_width
                while start < length:
                    end = min(start + span, length)
                    windows.append((tracklet, branch, start, end))
                    start, span = end, width
        if self.training:
            np.random.default_rng(stable_seed(self.seed, self.epoch, 'windows')).shuffle(windows)
        return windows

    def _queues(self):
        windows = self._windows()
        if not self.training or not self.reserve_windows:
            return deque(windows), deque()
        # 优先各 branch 28 个真正独立的单步窗口；小数据不足时从其他
        # branch 补齐。只重排窗口，不截断递归、不重复端点、不重放前缀。
        quotas = [self.reserve_windows // 4 + (branch < self.reserve_windows % 4)
                  for branch in range(4)]
        selected, counts = set(), [0] * 4
        for index, (_, branch, start, end) in enumerate(windows):
            if end - start == 1 and counts[branch] < quotas[branch]:
                selected.add(index)
                counts[branch] += 1
        for index, (_, _, start, end) in enumerate(windows):
            if len(selected) >= self.reserve_windows:
                break
            if end - start == 1:
                selected.add(index)
        return (deque(window for i, window in enumerate(windows) if i not in selected),
                deque(window for i, window in enumerate(windows) if i in selected))

    def _batches(self):
        pending, reserve = self._queues()
        active = []
        drain = False
        while pending or reserve or active:
            while (pending or reserve) and len(active) < self.batch_size:
                if pending:
                    tracklet, branch, start, end = pending.popleft()
                else:
                    tracklet, branch, start, end = reserve.popleft()
                    drain = True
                active.append(EndpointRequest(self.epoch, tracklet, branch, start, end, start))
            # drain 属于整批统计策略，而不是某一条轨迹的模型状态。
            yield tuple(replace(r, drain=drain) for r in active)
            active = [EndpointRequest(r.epoch, r.tracklet, r.branch, r.window_start,
                                      r.window_end, r.frame + 1)
                      for r in active if not r.ends_window]

    def _materialized(self):
        if self._plan is None:
            self._plan = tuple(self._batches())
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
                       batch_size=self.batch_size, seed=self.seed, training=self.training,
                       short_window=self.short_window, long_window=self.long_window,
                       curriculum_epochs=self.curriculum_epochs, rows=self.row_count,
                       source_sha256=self.source_sha256, reserve_windows=self.reserve_windows,
                       reserve_policy=self.RESERVE_POLICY, drain_policy=self.DRAIN_POLICY,
                       seed_policy=self.SEED_POLICY, seed_translation=self.seed_translation,
                       seed_yaw_degrees=self.seed_yaw_degrees, batches=len(self._materialized()),
                       plan_sha256=digest.hexdigest())
        # 旧版本不增加键，保持已有 manifest SHA；v35 只增加首测语义身份。
        if self.initial_seed_policy is not None:
            payload['initial_seed_policy'] = self.initial_seed_policy
        payload['manifest_sha256'] = hashlib.sha256(json.dumps(
            payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        return payload


def box_array(box):
    if isinstance(box, (list, tuple, np.ndarray)):
        result = np.asarray(box, dtype=np.float64)
        if result.shape != (4,):
            raise ValueError('box arrays must be world XYZ,yaw')
        return result.copy()
    if hasattr(box, 'orientation'):
        yaw = float(box.orientation.yaw_pitch_roll[0])
    else:
        yaw = float(box.yaw)
    return np.r_[np.asarray(box.center, dtype=np.float64), yaw]


def box_size(box, frame=None):
    if hasattr(box, 'wlh'):
        return np.asarray(box.wlh, dtype=np.float64)[[1, 0, 2]].copy()
    if frame is None or 'box_size' not in frame:
        raise ValueError('array boxes require frame box_size in LWH order')
    return np.asarray(frame['box_size'], dtype=np.float64).copy()


def point_arrays(frame):
    pc = frame['pc']
    if hasattr(pc, 'points'):
        xyz = np.asarray(pc.points, dtype=np.float64)[:3].T
        ids = np.asarray(pc.point_ids, dtype=np.int64)
    else:
        xyz = np.asarray(pc, dtype=np.float64)[:, :3]
        ids = np.asarray(frame.get('point_ids', np.arange(len(xyz))), dtype=np.int64)
    if xyz.shape != (len(ids), 3) or not np.isfinite(xyz).all():
        raise ValueError('raw point geometry/identity must be finite and aligned')
    if len(np.unique(ids)) != len(ids) or (ids < 0).any():
        raise ValueError('raw frame IDs must be unique and nonnegative')
    return xyz, ids


def inside_box(points, box, size, *, scale=1., offset=0.):
    """只在 membership 中旋转；提供给模型的 XYZ 仍为世界轴。"""
    delta = np.asarray(points)[..., :3] - np.asarray(box)[:3]
    c, s = np.cos(box[3]), np.sin(box[3])
    local = np.stack((c * delta[..., 0] + s * delta[..., 1],
                      -s * delta[..., 0] + c * delta[..., 1], delta[..., 2]), -1)
    return (np.abs(local) <= np.asarray(size) * (float(scale) / 2) + float(offset)).all(-1)


def box_corners(box, size):
    # 与 B0 observation.box_corners_xyz 的 BC channel 顺序严格一致。
    signs = np.asarray([[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
                        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]])
    local = signs * np.asarray(size)[None] / 2
    c, s = np.cos(box[3]), np.sin(box[3])
    rotation = np.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return local @ rotation.T + box[:3]


def metadata(source, tracklet, frame):
    if hasattr(source, 'get_frame_metadata'):
        return source.get_frame_metadata(tracklet, frame)
    return source.get_frames(tracklet, [frame])[0]


class RawEndpointDataset(Dataset):
    """无状态 raw 工厂。标签与 raw 帧一并传输，但不参与 worker 裁剪。"""
    def __init__(self, source):
        self.source = source
        self.lengths = tuple(source.get_num_frames_tracklet(i)
                             for i in range(source.get_num_tracklets()))
        self.source_sha256 = self._source_digest()

    def _source_digest(self):
        """只读 annotation 元数据；迁移路径不变，但换 track/token/time 必须拒绝续训。"""
        digest = hashlib.sha256()
        for index, length in enumerate(self.lengths):
            key = self.source.get_tracklet_key(index) if hasattr(self.source, 'get_tracklet_key') else str(index)
            digest.update(json.dumps((key, length), separators=(',', ':')).encode())
            annotations = getattr(self.source, 'tracklet_anno_list', None)
            for frame in range(length):
                endpoint = (self.source.get_endpoint_key(index, frame)
                            if hasattr(self.source, 'get_endpoint_key') else str(frame))
                if annotations is not None and hasattr(self.source, '_anno_timestamp'):
                    timestamp = float(self.source._anno_timestamp(annotations[index][frame]))
                else:
                    timestamp = float(metadata(self.source, index, frame)['timestamp'])
                digest.update(json.dumps((endpoint, timestamp), separators=(',', ':')).encode())
        return digest.hexdigest()

    def __len__(self):
        return sum(max(n - 1, 0) for n in self.lengths)

    def __getitem__(self, request):
        if not isinstance(request, EndpointRequest):
            raise TypeError('v31 raw dataset requires EndpointRequest')
        r = request
        history_ids = tuple(max(0, r.frame - 3 + i) for i in range(3))
        ids = sorted(set(history_ids + (r.frame,)))
        if r.starts_window:
            ids = sorted(set(ids + [0]))
        frames = dict(zip(ids, self.source.get_frames(r.tracklet, ids)))
        first = frames[0] if 0 in frames else metadata(self.source, r.tracklet, 0)
        key = (self.source.get_tracklet_key(r.tracklet)
               if hasattr(self.source, 'get_tracklet_key') else str(r.tracklet))
        return dict(request=r, frames=frames, first_frame=first,
                    history_ids=history_ids, tracklet_key=str(key))


def raw_collate(rows):
    return rows


def raw_dataset_factory(config, role):
    """只加载原始数据适配器，不构造历史 sampler。"""
    from datasets import get_raw_dataset
    return get_raw_dataset(config, role)


def _build_loaders(config, roles=('train', 'val'), sources=None):
    """返回 role -> DataLoader；sources 支持不依赖 nuScenes 的真实流程测试。"""
    loaders = {}
    for role in roles:
        dataset_role = ('dev' if str(option(config, 'dataset', '')).startswith('kitti')
                        else 'test') if role == 'val' else role
        source = sources[role] if sources is not None else raw_dataset_factory(config, dataset_role)
        raw = RawEndpointDataset(source)
        sampler = ReadyQueueBatchSampler(raw.lengths, int(option(config, 'batch_size', 16)),
            seed=int(option(config, 'seed', 42)), training=role == 'train',
            short_window=int(option(config, 'v31_short_window', 3)),
            long_window=int(option(config, 'v31_long_window', 8)),
            curriculum_epochs=int(option(config, 'v31_curriculum_epochs', 10)),
            source_sha256=raw.source_sha256,
            reserve_windows=int(option(config, 'v32_reserve_windows', 112)),
            seed_translation=float(option(config, 'v32_seed_translation', .3)),
            seed_yaw_degrees=float(option(config, 'v32_seed_yaw_degrees', 1.5)),
            initial_seed_policy=option(config, 'v35_seed_policy', ReadyQueueBatchSampler.INITIAL_SEED_POLICY))
        workers = int(option(config, 'workers', 4))
        kwargs = dict(num_workers=workers, collate_fn=raw_collate, pin_memory=False,
                      generator=torch.Generator().manual_seed(stable_seed(option(config, 'seed', 42), role)))
        if workers:
            kwargs.update(prefetch_factor=2, persistent_workers=False)
        loaders[role] = DataLoader(raw, batch_sampler=sampler, **kwargs)
    return loaders


def build_loaders(config, roles=('train', 'val'), sources=None):
    config = normalize_config(config)
    with recipe_scope(config):
        return _build_loaders(config, roles=roles, sources=sources)


build_dataloaders = build_loaders


@dataclass
class V38AcceptedState:
    boxes: dict
    times: dict
    next_frame: int
    size: np.ndarray
    trusted: list
    transitions: dict
    trusted_velocity: np.ndarray
    trusted_velocity_valid: bool
    previous_quality: float
    last_strong_time: float | None
    last_supported_time: float
    initialization_time: float
    previous_innovation: float
    box_provenance: dict
    seed_offset: np.ndarray
    previous_is_propagation: bool = False
    consecutive_propagations: int = 0
    consecutive_untrusted_outputs: int = 0
    last_supported_quality: float = 1.
    last_velocity_update_time: float | None = None


# GT 仅用于合法初框、监督与被动诊断。
MISSING_BOX, KNOWN_GT_BOX, PERTURBED_BOX, PREDICTED_BOX = range(4)


class BatchBuilder:
    """唯一状态所有者；prepare -> supervise_prior -> commit 是一次事务。"""
    def __init__(self, config):
        self.config = self.v38_config = normalize_config(config)
        self.train_diagnostics = bool(option(self.config, 'v35_train_diagnostics', True))
        self.states = {}
        self._pending = None
        self._raw_memo = None

    def _start_materialization_cache(self):
        self._raw_memo = (RawFrameMemo(point_arrays, box_array, box_size, box_corners, inside_box)
                          if optimized_execution(self.config) else None)

    def clear_materialization_cache(self):
        self._raw_memo = None

    def _points(self, frame):
        return self._raw_memo.points(frame) if self._raw_memo is not None else point_arrays(frame)

    def _raw_index(self, frame):
        if self._raw_memo is not None:
            return self._raw_memo.index(frame)
        return {int(value): j for j, value in enumerate(point_arrays(frame)[1])}

    def reset(self):
        self.states.clear()
        self._pending = None
        self.clear_materialization_cache()

    def _new_state(self, row):
        # 与 v35 相同的合法 seed 几何和局部 RNG；仅删除身份点记忆初始化。
        r, frames = row['request'], row['frames']
        seed_ids = range(max(0, r.frame - 3), r.frame)
        size = box_size(row['first_frame']['3d_bbox'], row['first_frame'])
        boxes = {i: box_array(frames[i]['3d_bbox']) for i in seed_ids}
        times = {i: float(frames[i]['timestamp']) for i in seed_ids}
        seed_offset = np.zeros(4, dtype=np.float64)
        provenance = KNOWN_GT_BOX
        if r.branch in (1, 2, 3) and not (r.frame == 1):
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
            trusted=trusted, transitions={i: True for i in seed_ids},
            trusted_velocity=velocity, trusted_velocity_valid=len(trusted) == 2,
            previous_quality=1., last_strong_time=max(strong_times) if strong_times else None,
            last_supported_time=last_time, initialization_time=last_time,
            previous_innovation=0., box_provenance={i: provenance for i in seed_ids},
            seed_offset=seed_offset,
            last_velocity_update_time=last_time if len(trusted) == 2 else None)

    @staticmethod
    def _strong_age(state, timestamp):
        since = state.initialization_time if state.last_strong_time is None else state.last_strong_time
        return max(0., float(timestamp) - since)

    @staticmethod
    def _velocity_age(state, timestamp):
        # 无效速度不用年龄伪造新鲜度；台账仍保留最后一次合法更新时间。
        if not state.trusted_velocity_valid or state.last_velocity_update_time is None:
            return 0.
        return max(0., float(timestamp) - state.last_velocity_update_time)

    @raw_transaction
    def prepare(self, rows):
        with recipe_scope(self.v38_config):
            return self._prepare(rows)

    def _prepare(self, rows):
        if self._pending is not None:
            raise RuntimeError('previous v38 transaction has not been committed')
        if not rows:
            raise ValueError('empty raw batch')
        keys = [row['request'].state_key for row in rows]
        if len(keys) != len(set(keys)):
            raise ValueError('adjacent endpoints of one window cannot share a batch')
        prepared, contexts = [], []
        for row in rows:
            r, frames = row['request'], row['frames']
            if r.starts_window:
                if r.state_key in self.states:
                    raise RuntimeError('duplicate window initialization')
                self.states[r.state_key] = self._new_state(row)
            if r.state_key not in self.states:
                raise RuntimeError('missing accepted predecessor; no GT reseeding is allowed')
            state = self.states[r.state_key]
            if state.next_frame != r.frame:
                raise RuntimeError('out-of-order accepted state')
            data, context = self._prepare_row(row, state)
            prepared.append(data)
            contexts.append(context)
        batch = {key: torch.from_numpy(np.stack([x[key] for x in prepared])) for key in prepared[0]}
        self._pending = contexts
        return batch

    def _prepare_row(self, row, state):
        r, frames = row['request'], row['frames']
        current = frames[r.frame]
        now = float(current['timestamp'])
        effective_now = float(current.get('_ct_effective_timestamp', now))
        anchor = state.boxes[r.frame - 1].copy()
        history_valid = np.asarray([r.frame - 3 + i >= 0 for i in range(3)], dtype=bool)
        history = np.stack([state.boxes[i] for i in row['history_ids']])
        history_provenance = np.asarray([state.box_provenance[i] if valid else MISSING_BOX
            for i, valid in zip(row['history_ids'], history_valid)], dtype=np.int64)
        history_local = history.copy()
        history_local[:, :3] -= anchor[:3]
        physical_times = np.asarray([float(frames[i]['timestamp']) - now for i in row['history_ids']])
        effective_times = np.asarray([float(frames[i].get('_ct_effective_timestamp',
                                                frames[i]['timestamp'])) - effective_now
                                      for i in row['history_ids']])
        if not (physical_times < 0).all() or not (effective_times < 0).all():
            raise ValueError('B1 history must be strictly in the past')
        n = int(option(self.config, 'point_sample_size', 1024))
        scale, offset = float(option(self.config, 'bb_scale', 1.25)), float(option(self.config, 'bb_offset', 2.))
        points, valid, raw_ids, seg, bc, truth_boxes = [], [], [], [], [], []
        record_training = self.train_diagnostics and r.branch != 4
        history_raw_counts, history_crop_counts = [], []
        b0_ids = None
        for slot, frame_id in enumerate(row['history_ids'] + (r.frame,)):
            frame = frames[frame_id]
            xyz, ids = self._points(frame)
            crop_box = history[slot] if slot < 3 else anchor
            mask = inside_box(xyz, crop_box, state.size, scale=scale, offset=offset)
            selected = np.flatnonzero(mask)
            if slot < 3 and not history_valid[slot]:
                selected = selected[:0]
            if slot == 3:
                b0_ids = ids[selected].copy()
            if len(selected) > n:
                rng = np.random.default_rng(stable_seed(option(self.config, 'seed', 42),
                    r.epoch, row['tracklet_key'], r.branch, r.frame, slot))
                selected = np.sort(rng.choice(selected, n, replace=False))
            count = len(selected)
            point = np.zeros((n, 5), dtype=np.float32)
            point[:count, :3] = xyz[selected] - anchor[:3]
            point[:count, 3] = float(frame['timestamp']) - now
            if slot < 3:
                hint = inside_box(xyz[selected], crop_box, state.size).astype(np.float32)
                point[:count, 4] = hint if history_provenance[slot] == KNOWN_GT_BOX else .2 + .6 * hint
            else:
                point[:count, 4] = .5
            point_valid = np.arange(n) < count
            point_ids = np.full(n, -1, dtype=np.int64)
            point_ids[:count] = ids[selected]
            if self._raw_memo is None:
                target = box_array(frame['3d_bbox'])
                target_size = box_size(frame['3d_bbox'], frame)
                centers_corners = np.vstack((target[:3], box_corners(target, target_size)))
                full_target = None
            else:
                target, target_size, centers_corners = self._raw_memo.geometry(frame)
                full_target = None
            if record_training and slot < 3:
                # 仅被动日志；不存在的历史不借用复制槽的 GT 计数。
                if full_target is None:
                    full_target = (inside_box(xyz, target, target_size) if self._raw_memo is None
                                   else self._raw_memo.target_mask(frame))
                history_raw_counts.append(int(full_target.sum()) if history_valid[slot] else 0)
                history_crop_counts.append(int((full_target & mask).sum()) if history_valid[slot] else 0)
            label = np.full(n, -1, dtype=np.int64)
            label[:count] = inside_box(xyz[selected], target, target_size).astype(np.int64)
            distances = np.zeros((n, 9), dtype=np.float32)
            distances[:count] = np.linalg.norm(xyz[selected, None] - centers_corners[None], axis=-1)
            target[:3] -= anchor[:3]
            points.append(point); valid.append(point_valid); raw_ids.append(point_ids)
            seg.append(label); bc.append(distances); truth_boxes.append(target)
        current_gt = box_array(current['3d_bbox'])
        previous_gt = box_array(frames[r.frame - 1]['3d_bbox'])
        recovery = state.trusted[-1][0].copy()
        recovery[:3] += state.trusted_velocity * (now - state.trusted[-1][1])
        fallback = recovery.copy()
        fallback[:3] -= anchor[:3]
        pair_valid = history_valid[:-1] & history_valid[1:]
        pair_valid &= np.asarray([state.transitions.get(i, False) for i in row['history_ids'][1:]])
        result = dict(points=np.stack(points), point_valid=np.stack(valid), point_ids=np.stack(raw_ids),
            history_boxes=history_local.astype(np.float32), history_valid=history_valid,
            history_box_provenance=history_provenance,
            window_seed_offset=state.seed_offset.astype(np.float32),
            history_pair_valid=pair_valid,
            history_times=effective_times.astype(np.float32), current_dt=np.float32(-effective_times[-1]),
            frame_times=np.r_[physical_times, 0.].astype(np.float32),
            box_size=state.size.astype(np.float32), anchor_box=anchor.astype(np.float32),
            fallback_box=fallback.astype(np.float32),
            target_box=np.asarray(truth_boxes[-1], dtype=np.float32),
            target_box_size=box_size(current['3d_bbox'], current).astype(np.float32),
            history_target_boxes=np.asarray(truth_boxes[:3], dtype=np.float32),
            segmentation_labels=np.stack(seg), bc_targets=np.stack(bc),
            physical_displacement=(current_gt[:2] - previous_gt[:2]).astype(np.float32),
            physical_valid=np.bool_(True), branch_id=np.int64(r.branch),
            trusted_velocity=state.trusted_velocity.astype(np.float32),
            trusted_velocity_valid=np.bool_(state.trusted_velocity_valid),
            previous_quality=np.float32(state.previous_quality),
            weak_age=np.float32(self._strong_age(state, now)),
            supported_age=np.float32(max(0., now - state.last_supported_time)),
            previous_innovation=np.float32(state.previous_innovation))
        result['history_is_initial'] = (history_valid
            & (np.asarray(row['history_ids']) == 0) & (history_provenance == KNOWN_GT_BOX))
        if record_training:
            result.update(diagnostic_history_raw_target_count=np.asarray(history_raw_counts, dtype=np.int64),
                          diagnostic_history_crop_target_count=np.asarray(history_crop_counts, dtype=np.int64))
        context = dict(row=row, state=state, anchor=anchor, timestamp=now, recovery=recovery,
                       b0_ids=b0_ids, current_gt=current_gt)
        result.update(previous_is_propagation=np.bool_(state.previous_is_propagation),
            consecutive_propagations=np.int64(state.consecutive_propagations),
            consecutive_untrusted_outputs=np.int64(state.consecutive_untrusted_outputs),
            trusted_velocity_age=np.float32(self._velocity_age(state, now)))
        return result, context

    @raw_transaction
    def supervise_prior(self, batch, prior, *, training=True):
        """仅生成原 B1 acquisition 监督与被动诊断，不获取扩展点。"""
        with recipe_scope(self.config):
            return self._supervise_prior(batch, prior, training=training)

    def _supervise_prior(self, batch, prior, *, training=True):
        if self._pending is None:
            raise RuntimeError('prior supervision requires prepared raw rows')
        optimized = optimized_execution(self.config)
        transferred = tensors_to_numpy(dict(boxes=prior.box, directions=prior.direction_xy), optimized=optimized)
        boxes, directions = (transferred[key] for key in ('boxes', 'directions'))
        row_diagnostics = {}
        if optimized and (not training or self.train_diagnostics):
            for i in range(len(self._pending)):
                row_diagnostics[i, 'targets'] = (batch['segmentation_labels'][i, -1] == 1).sum()
                row_diagnostics[i, 'points'] = batch['point_valid'][i, -1].sum()
                row_diagnostics[i, 'displacement'] = torch.linalg.vector_norm(batch['physical_displacement'][i])
            row_diagnostics = tensors_to_numpy(row_diagnostics, optimized=True)
        values = []
        for i, context in enumerate(self._pending):
            row, state, anchor = context['row'], context['state'], context['anchor']
            current = row['frames'][row['request'].frame]
            xyz, ids = self._points(current)
            target_mask = (inside_box(xyz, context['current_gt'], box_size(current['3d_bbox'], current))
                           if self._raw_memo is None else self._raw_memo.target_mask(current))
            geometry = dict(b0_raw_ids=context['b0_ids'], b0_box=anchor, box_size=state.size,
                prior_center=boxes[i, :3] + anchor[:3], recovery_center=context['recovery'][:3],
                support_yaw=float(np.arctan2(directions[i, 1], directions[i, 0])),
                recovery_yaw=float(context['recovery'][3]),
                crop_scale=float(option(self.config, 'bb_scale', 1.25)),
                crop_offset=float(option(self.config, 'bb_offset', 2.)))
            targets = (band_grid_target(xyz, ids, target_mask=target_mask, **geometry)
                       if training
                       else dict(acquisition_target=np.zeros(2, np.float32), acquisition_valid=False,
                                 acquisition_demand=False))
            extra = dict(acquisition_target=np.asarray(targets['acquisition_target'], dtype=np.float32),
                         acquisition_valid=np.bool_(targets['acquisition_valid']),
                         acquisition_demand=np.bool_(targets['acquisition_demand']))
            if not training or self.train_diagnostics:
                # 只用于离线诊断的 GT 计数；不改变实际 support、采样、候选或状态。
                b0_target_count = int((np.isin(ids, context['b0_ids']) & target_mask).sum())
                extra.update(diagnostic_target_count=np.int64(target_mask.sum()),
                             diagnostic_raw_point_count=np.int64(len(ids)),
                             diagnostic_crop_target_count=np.int64(b0_target_count),
                             diagnostic_sampled_target_count=np.int64(
                                 row_diagnostics[i, 'targets'] if optimized else
                                 (batch['segmentation_labels'][i, -1] == 1).sum().item()),
                             diagnostic_crop_point_count=np.int64(len(context['b0_ids'])),
                             diagnostic_sampled_point_count=np.int64(
                                 row_diagnostics[i, 'points'] if optimized else
                                 batch['point_valid'][i, -1].sum().item()),
                             diagnostic_gt_xy_displacement=np.float32(
                                 row_diagnostics[i, 'displacement'] if optimized else
                                 torch.linalg.vector_norm(batch['physical_displacement'][i]).item()),
                             diagnostic_novel_target_count=np.int64(target_mask.sum() - b0_target_count))
            values.append(extra)
        device = batch['points'].device
        for key in values[0]:
            batch[key] = torch.from_numpy(np.stack([x[key] for x in values])).to(device)
        return batch

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
        if output.accepted_box.shape != (b, 4) or output.prior.box.shape != (b, 4):
            raise ValueError('v38 accepted/prior boxes must match pending rows as [B,4]')
        probability = output.observation.foreground_probability
        point_ids, point_valid = batch['point_ids'], batch['point_valid']
        if (point_ids.ndim != 3 or point_ids.shape[0] != b or point_ids.shape[1] < 1
                or probability.shape != point_ids.shape or point_valid.shape != point_ids.shape):
            raise ValueError('v38 foreground probability/point IDs/validity must align as [B,T,N]')
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
        if not np.isfinite(transferred['prior_boxes']).all():
            raise FloatingPointError('non-finite v38 prior box')
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
            innovation = float(np.linalg.norm(transferred['boxes'][i, :2] - transferred['prior_boxes'][i, :2]))
            staged.append((context, box, count, raw_count, strong, supported, prop, score, innovation))
        records = []
        for i, (context, box, count, raw_count, strong, supported, prop, score, innovation) in enumerate(staged):
            row, state, now = context['row'], context['state'], context['timestamp']
            r = row['request']
            state.boxes[r.frame] = box
            state.box_provenance[r.frame] = PREDICTED_BOX
            state.times[r.frame] = now
            state.next_frame += 1
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
            state.previous_is_propagation = prop
            state.consecutive_propagations = state.consecutive_propagations + 1 if prop else 0
            state.consecutive_untrusted_outputs = 0 if supported else state.consecutive_untrusted_outputs + 1
            if diagnostics:
                records.append(dict(strong=bool(strong), supported=bool(supported),
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
