"""v31 原始帧数据流：worker 不拥有预测状态，也不执行模型。

每个训练端点恰好出现四次；窗口长度只决定 accepted 历史寿命，不增加
前缀 forward。所有裁剪、状态读取和获取均在主进程完成。
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


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
    SCHEMA = 'ct_seqtrack.v31.ready_queue.v1'

    def __init__(self, lengths, batch_size=16, *, seed=42, training=True,
                 short_window=3, long_window=8, curriculum_epochs=10, source_sha256=''):
        self.lengths = tuple(int(n) for n in lengths)
        self.batch_size = int(batch_size)
        self.seed, self.training, self.epoch = int(seed), bool(training), 0
        self.short_window, self.long_window = int(short_window), int(long_window)
        self.curriculum_epochs = int(curriculum_epochs)
        self.source_sha256 = str(source_sha256)
        if self.batch_size < 1 or min(self.short_window, self.long_window, self.curriculum_epochs) < 1:
            raise ValueError('v31 batch/window/curriculum sizes must be positive')
        if any(n < 0 for n in self.lengths):
            raise ValueError('negative tracklet length')
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

    def _batches(self):
        pending = deque(self._windows())
        active = []
        while pending or active:
            while pending and len(active) < self.batch_size:
                tracklet, branch, start, end = pending.popleft()
                active.append(EndpointRequest(self.epoch, tracklet, branch, start, end, start))
            yield tuple(active)
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
        payload = dict(schema=self.SCHEMA, epoch=self.epoch, lengths=list(self.lengths),
                       batch_size=self.batch_size, seed=self.seed, training=self.training,
                       short_window=self.short_window, long_window=self.long_window,
                       curriculum_epochs=self.curriculum_epochs, rows=self.row_count,
                       source_sha256=self.source_sha256)
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


def build_loaders(config, roles=('train', 'val'), sources=None):
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
            source_sha256=raw.source_sha256)
        workers = int(option(config, 'workers', 4))
        kwargs = dict(num_workers=workers, collate_fn=raw_collate, pin_memory=False,
                      generator=torch.Generator().manual_seed(stable_seed(option(config, 'seed', 42), role)))
        if workers:
            kwargs.update(prefetch_factor=2, persistent_workers=False)
        loaders[role] = DataLoader(raw, batch_sampler=sampler, **kwargs)
    return loaders


@dataclass
class AcceptedState:
    boxes: dict
    times: dict
    next_frame: int
    size: np.ndarray
    memory: object
    trusted: list
    transitions: dict
    trusted_velocity: np.ndarray
    previous_quality: float
    last_strong_time: float
    last_supported_time: float
    previous_innovation: float


class BatchBuilder:
    """主进程唯一状态所有者；prepare -> acquire -> commit 是一次事务。"""
    def __init__(self, config):
        self.config = config
        self.states = {}
        self._pending = None

    def reset(self):
        self.states.clear()
        self._pending = None

    def _new_state(self, row):
        from .memory import RawIdentityMemory
        r, frames = row['request'], row['frames']
        seed_ids = range(max(0, r.frame - 3), r.frame)
        size = box_size(row['first_frame']['3d_bbox'], row['first_frame'])
        boxes = {i: box_array(frames[i]['3d_bbox']) for i in seed_ids}
        times = {i: float(frames[i]['timestamp']) for i in seed_ids}
        memory = RawIdentityMemory(size)
        first = row['first_frame']
        xyz, ids = point_arrays(first)
        first_box = box_array(first['3d_bbox'])
        memory.initialize(xyz, ids, inside_box(xyz, first_box, size), first_box,
                          float(first['timestamp']))
        trusted = [(boxes[i].copy(), times[i]) for i in seed_ids][-2:]
        velocity = np.zeros(3, dtype=np.float64)
        if len(trusted) == 2:
            gap = trusted[-1][1] - trusted[-2][1]
            if gap <= 0:
                raise ValueError('seed timestamps must strictly increase')
            velocity = (trusted[-1][0][:3] - trusted[-2][0][:3]) / gap
        last_time = times[r.frame - 1]
        return AcceptedState(boxes, times, r.frame, size, memory, trusted,
                             {i: True for i in seed_ids}, velocity, 1.,
                             last_time, last_time, 0.)

    def prepare(self, rows):
        if self._pending is not None:
            raise RuntimeError('previous v31 transaction has not been committed')
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
        b0_ids, b0_xyz = None, None
        for slot, frame_id in enumerate(row['history_ids'] + (r.frame,)):
            frame = frames[frame_id]
            xyz, ids = point_arrays(frame)
            crop_box = history[slot] if slot < 3 else anchor
            mask = inside_box(xyz, crop_box, state.size, scale=scale, offset=offset)
            selected = np.flatnonzero(mask)
            if slot < 3 and not history_valid[slot]:
                selected = selected[:0]
            if slot == 3:
                b0_ids, b0_xyz = ids[selected].copy(), xyz[selected].copy()
            if len(selected) > n:
                rng = np.random.default_rng(stable_seed(option(self.config, 'seed', 42),
                    r.epoch, row['tracklet_key'], r.branch, r.frame, slot))
                selected = np.sort(rng.choice(selected, n, replace=False))
            count = len(selected)
            point = np.zeros((n, 5), dtype=np.float32)
            point[:count, :3] = xyz[selected] - anchor[:3]
            point[:count, 3] = float(frame['timestamp']) - now
            point[:count, 4] = (inside_box(xyz[selected], crop_box, state.size).astype(np.float32)
                                if slot < 3 else .5)
            point_valid = np.arange(n) < count
            point_ids = np.full(n, -1, dtype=np.int64)
            point_ids[:count] = ids[selected]
            target = box_array(frame['3d_bbox'])
            target_size = box_size(frame['3d_bbox'], frame)
            label = np.full(n, -1, dtype=np.int64)
            label[:count] = inside_box(xyz[selected], target, target_size).astype(np.int64)
            distances = np.zeros((n, 9), dtype=np.float32)
            centers_corners = np.vstack((target[:3], box_corners(target, target_size)))
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
        memory = state.memory.export(now)
        result = dict(points=np.stack(points), point_valid=np.stack(valid), point_ids=np.stack(raw_ids),
            history_boxes=history_local.astype(np.float32), history_valid=history_valid,
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
            previous_quality=np.float32(state.previous_quality),
            weak_age=np.float32(max(0., now - state.last_strong_time)),
            supported_age=np.float32(max(0., now - state.last_supported_time)),
            previous_innovation=np.float32(state.previous_innovation),
            **memory)
        context = dict(row=row, state=state, anchor=anchor, timestamp=now, recovery=recovery,
                       b0_ids=b0_ids, b0_xyz=b0_xyz, current_gt=current_gt)
        return result, context

    def acquire(self, batch, prior, *, training=True):
        from .acquisition import (acquire_extension, band_grid_target, build_dual_support,
                                  support_membership)
        if self._pending is None:
            raise RuntimeError('acquisition requires prepared raw rows')
        boxes = prior.box.detach().cpu().numpy()
        fractions = prior.acquisition_fraction.detach().cpu().numpy()
        directions = prior.direction_xy.detach().cpu().numpy()
        values = []
        for i, context in enumerate(self._pending):
            row, state, anchor = context['row'], context['state'], context['anchor']
            current = row['frames'][row['request'].frame]
            xyz, ids = point_arrays(current)
            target_mask = inside_box(xyz, context['current_gt'], box_size(current['3d_bbox'], current))
            geometry = dict(b0_raw_ids=context['b0_ids'], b0_box=anchor, box_size=state.size,
                prior_center=boxes[i, :3] + anchor[:3], recovery_center=context['recovery'][:3],
                support_yaw=float(np.arctan2(directions[i, 1], directions[i, 0])),
                recovery_yaw=float(context['recovery'][3]),
                crop_scale=float(option(self.config, 'bb_scale', 1.25)),
                crop_offset=float(option(self.config, 'bb_offset', 2.)))
            enable_extension = option(self.config, 'v31_arm', 'full') in ('b1_b2', 'full')
            if enable_extension:
                extra = acquire_extension(xyz, ids, anchor=anchor[:3], u=fractions[i],
                    seed=stable_seed(option(self.config, 'seed', 42), row['request']), **geometry)
            else:
                extra = dict(extension_points=np.zeros((768, 5), np.float32),
                             extension_ids=np.full(768, -1, np.int64),
                             extension_valid=np.zeros(768, bool),
                             extension_partition=np.full(768, -1, np.int64))
            lookup = {int(raw_id): bool(label) for raw_id, label in zip(ids, target_mask)}
            labels = np.full(768, -1, dtype=np.int64)
            keep = extra['extension_valid']
            labels[keep] = [lookup[int(raw_id)] for raw_id in extra['extension_ids'][keep]]
            targets = (band_grid_target(xyz, ids, target_mask=target_mask, **geometry)
                       if training and option(self.config, 'v31_arm', 'full') != 'b0'
                       else dict(acquisition_target=np.zeros(2, np.float32), acquisition_valid=False,
                                 acquisition_demand=False))
            extra.update(extension_labels=labels,
                         acquisition_target=np.asarray(targets['acquisition_target'], dtype=np.float32),
                         acquisition_valid=np.bool_(targets['acquisition_valid']),
                         acquisition_demand=np.bool_(targets['acquisition_demand']))
            if not training:
                # 只用于离线诊断的 GT 计数；不改变实际 support、采样、候选或状态。
                if enable_extension:
                    maximum = build_dual_support(u=np.ones(2), **{
                        k: v for k, v in geometry.items() if k != 'b0_raw_ids'})
                    members = support_membership(xyz, maximum, context['b0_ids'], ids)
                    reachable = int(((members[0] | members[1]) & target_mask).sum())
                else:
                    reachable = 0
                b0_target_count = int((np.isin(ids, context['b0_ids']) & target_mask).sum())
                extra.update(diagnostic_target_count=np.int64(target_mask.sum()),
                             diagnostic_novel_target_count=np.int64(target_mask.sum() - b0_target_count),
                             diagnostic_reachable_count=np.int64(reachable),
                             diagnostic_acquired_count=np.int64((labels[keep] == 1).sum()))
            values.append(extra)
        device = batch['points'].device
        for key in values[0]:
            batch[key] = torch.from_numpy(np.stack([x[key] for x in values])).to(device)
        return batch

    def commit(self, output, batch, *, diagnostics=False):
        if self._pending is None:
            raise RuntimeError('no pending v31 transaction (duplicate commit)')
        boxes = output.accepted_box.detach().cpu().numpy()
        quality = output.selected_quality.detach().cpu().numpy()
        probabilities = output.observation.foreground_probability[:, -1].detach().cpu().numpy()
        point_ids = batch['point_ids'][:, -1].detach().cpu().numpy()
        point_valid = batch['point_valid'][:, -1].detach().cpu().numpy()
        prior_boxes = output.prior.box.detach().cpu().numpy()
        extension_ids = batch['extension_ids'].detach().cpu().numpy()
        extension_valid = batch['extension_valid'].detach().cpu().numpy()
        extension_probabilities = output.evidence.identity_logits.detach().sigmoid().cpu().numpy()
        records = []
        for index, context in enumerate(self._pending):
            row, state = context['row'], context['state']
            r = row['request']
            box = boxes[index].astype(np.float64, copy=True)
            box[:3] += context['anchor'][:3]
            if not np.isfinite(box).all():
                raise FloatingPointError('non-finite accepted box')
            state.boxes[r.frame] = box
            state.times[r.frame] = context['timestamp']
            state.next_frame += 1
            ids = point_ids[index, point_valid[index]]
            probs = probabilities[index, point_valid[index]]
            if option(self.config, 'v31_arm', 'full') in ('b1_b2', 'full'):
                ids = np.r_[ids, extension_ids[index, extension_valid[index]]]
                probs = np.r_[probs, extension_probabilities[index, extension_valid[index]]]
            cloud, cloud_ids = point_arrays(row['frames'][r.frame])
            id_to_index = {int(value): j for j, value in enumerate(cloud_ids)}
            xyz = cloud[[id_to_index[int(value)] for value in ids]]
            updated = state.memory.update(xyz, ids, probs, box, context['timestamp'], float(quality[index]))
            innovation = float(np.linalg.norm(boxes[index, :2] - prior_boxes[index, :2]))
            pair_valid = innovation <= max(.5 * np.hypot(*state.size[:2]), 1e-3)
            state.transitions[r.frame] = pair_valid
            state.previous_innovation, state.previous_quality = innovation, float(quality[index])
            if int(point_valid[index].sum()) >= 3:
                state.last_strong_time = context['timestamp']
            if updated:
                previous, previous_time = state.trusted[-1]
                # 只接受相邻可信帧之间、未跨重定位的速度。旧可信速度可继续传播。
                if pair_valid and previous_time == state.times.get(r.frame - 1):
                    gap = context['timestamp'] - previous_time
                    state.trusted_velocity = (box[:3] - previous[:3]) / gap
                state.trusted.append((box.copy(), context['timestamp']))
                state.trusted[:] = state.trusted[-2:]
                state.last_supported_time = context['timestamp']
            if diagnostics:
                foreground_ids = (state.memory.recent[-1]['ids'][state.memory.recent[-1]['fg']]
                                  if updated else np.empty(0, dtype=np.int64))
                foreground_xyz = cloud[[id_to_index[int(value)] for value in foreground_ids]]
                current = row['frames'][r.frame]
                correct = inside_box(foreground_xyz, context['current_gt'],
                                     box_size(current['3d_bbox'], current))
                records.append(dict(memory_write=bool(updated), memory_fg_count=int(len(correct)),
                                    memory_true_fg_count=int(correct.sum()),
                                    memory_wrong_write=bool(updated and correct.mean() < .5)))
            for old in list(state.boxes):
                if old < r.frame - 2:
                    state.boxes.pop(old)
                    state.times.pop(old)
                    state.transitions.pop(old, None)
            if r.ends_window:
                self.states.pop(r.state_key)
        self._pending = None
        return records
