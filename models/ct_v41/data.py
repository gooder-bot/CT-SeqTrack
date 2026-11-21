"""v41 原始端点、W 课程与 accepted 状态；worker 不拥有预测状态。"""
from __future__ import annotations

from collections import Counter, OrderedDict, deque
from copy import deepcopy
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
from .joint_state import EMPTY_IDS, EMPTY_XYZ, JointSearchState, physical_frame, relative
from .search import _unique_mask, inside_box, make_search_plan, sample_search_points


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
        self._plan_sha256 = None

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
            self._plan_sha256 = None

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
        if self._plan_sha256 is None:
            digest = hashlib.sha256()
            for batch in self._materialized():
                digest.update(repr(tuple((r.tracklet, r.branch, r.window_start,
                                          r.window_end, r.frame, r.drain) for r in batch)).encode())
            self._plan_sha256 = digest.hexdigest()
        payload = dict(schema=self.SCHEMA, epoch=self.epoch, lengths=list(self.lengths),
                       batch_size=self.batch_size, seed=self.seed, training=self.training,
                       short_window=self.short_window, long_window=self.long_window,
                       curriculum_epochs=self.curriculum_epochs, rows=self.row_count,
                       source_sha256=self.source_sha256, reserve_windows=self.reserve_windows,
                       reserve_policy=self.RESERVE_POLICY, drain_policy=self.DRAIN_POLICY,
                       seed_policy=self.SEED_POLICY, seed_translation=self.seed_translation,
                       seed_yaw_degrees=self.seed_yaw_degrees, batches=len(self._materialized()),
                       plan_sha256=self._plan_sha256)
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
    if '_v41_raw_ref' in frame:
        # KITTI 内部帧只来自已检查的 bin；ID 在读取时按行号构造。
        if 'pc' in frame:
            return frame['pc'].points.T, frame['pc'].point_ids
        points, ids = frame['_v41_raw_ref'].arrays()
        return points.T, ids
    pc = frame['pc']
    if hasattr(pc, 'points'):
        xyz = np.asarray(pc.points, dtype=np.float64)[:3].T
        ids = np.asarray(pc.point_ids, dtype=np.int64)
    else:
        xyz = np.asarray(pc, dtype=np.float64)[:, :3]
        ids = (np.asarray(frame['point_ids'], dtype=np.int64) if 'point_ids' in frame
               else np.arange(len(xyz), dtype=np.int64))
    if xyz.shape != (len(ids), 3) or not np.isfinite(xyz).all():
        raise ValueError('raw point geometry/identity must be finite and aligned')
    if len(np.unique(ids)) != len(ids) or (ids < 0).any():
        raise ValueError('raw frame IDs must be unique and nonnegative')
    return xyz, ids


def box_corners(box, size):
    # 与 B0 observation.box_corners_xyz 的 BC channel 顺序严格一致。
    signs = np.asarray([[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
                        [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]])
    local = signs * np.asarray(size)[None] / 2
    c, s = np.cos(box[3]), np.sin(box[3])
    rotation = np.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return local @ rotation.T + box[:3]


def metadata(source, tracklet, frame):
    if hasattr(source, 'endpoint_metadata'):
        return source.endpoint_metadata(tracklet, frame)
    if hasattr(source, 'get_frame_metadata'):
        return source.get_frame_metadata(tracklet, frame)
    return source.get_frames(tracklet, [frame])[0]


def future_motion_targets(source, request, frames, track_length):
    """worker 仅读未来 annotation；第 h 个端点从 s=t−1 计，绝不读取未来点云。"""
    getter = getattr(source, 'endpoint_metadata', None) or getattr(source, 'get_frame_metadata', None)
    if getter is None:
        raise TypeError('v41 M requires get_frame_metadata; future point-cloud fallback is forbidden')
    r = request
    previous = frames[r.frame - 1]
    origin = box_array(previous['3d_bbox'])[:2]
    start_time = float(previous['timestamp'])
    future = np.zeros((2, 2), dtype=np.float32)
    elapsed = np.zeros(2, dtype=np.float32)
    valid = np.zeros(2, dtype=bool)
    for j, horizon in enumerate((2, 4)):
        index = r.frame - 1 + horizon
        if index >= track_length:
            continue
        try:
            meta = getter(r.tracklet, index)
        except KeyError:
            # 明确缺 annotation 只屏蔽该辅助目标；I/O/解码异常继续抛出。
            continue
        if not isinstance(meta, dict) or meta.get('3d_bbox') is None:
            continue
        if 'scene_id' in previous and meta.get('scene_id') != previous['scene_id']:
            continue
        try:
            dt = float(meta.get('timestamp')) - start_time
            displacement = box_array(meta['3d_bbox'])[:2] - origin
        except (TypeError, ValueError, IndexError, AttributeError):
            continue
        if dt > 0 and np.isfinite(dt) and np.isfinite(displacement).all():
            # float32 溢出也属于无效辅助标签，不得污染主任务。
            if abs(dt) > np.finfo(np.float32).max or np.abs(displacement).max() > np.finfo(np.float32).max:
                continue
            future[j], elapsed[j], valid[j] = displacement, dt, True
    return dict(future_displacement=future, future_dt=elapsed, future_valid=valid)


class RawEndpointDataset(Dataset):
    """按端点交付观测与标签。KITTI 主流可携带 GT10 视图，预测 ROI 由主进程决定。"""
    def __init__(self, source, *, training=False, enable_m=False, worker_transport=False,
                 joint_search=False, four_candidates=False):
        self.source = source
        self.training, self.enable_m = bool(training), bool(enable_m)
        self.worker_transport = worker_transport
        self.joint_search = bool(joint_search)
        self.four_candidates = bool(four_candidates and training)
        self._scan_annotations = None
        self._scan_car_boxes = {}
        self.lengths = tuple(source.get_num_frames_tracklet(i)
                             for i in range(source.get_num_tracklets()))
        self.source_sha256 = self._source_digest()

    def _source_digest(self):
        """只读 annotation 元数据；迁移路径不变，但换 track/token/time 必须拒绝续训。"""
        digest = hashlib.sha256()
        metadata_sha = getattr(self.source, 'input_metadata_sha256', None)
        if metadata_sha is not None:
            digest.update(json.dumps(('input_metadata_sha256', metadata_sha),
                                     separators=(',', ':')).encode())
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

    def other_cars_for(self, row):
        """只由训练标签或 full 评估写账调用。普通在线行不读取邻车标注。"""
        if self._scan_annotations is None:
            self._scan_annotations = {}
            for tracklet, rows in enumerate(getattr(self.source, 'tracklet_anno_list', ())):
                for step, anno in enumerate(rows):
                    if 'scene' in anno and 'frame' in anno and anno.get('type', 'Car') == 'Car':
                        key = str(anno['scene']), int(anno['frame'])
                        self._scan_annotations.setdefault(key, []).append((tracklet, step))
        r = row['request']
        current = row['frames'][r.frame]
        key = str(current.get('scene_id', '')), physical_frame(current, r.frame)
        if key not in self._scan_car_boxes:
            tracks, boxes = [], []
            for tracklet, step in self._scan_annotations.get(key, ()):
                geo = metadata(self.source, tracklet, step)
                tracks.append(tracklet)
                boxes.append(np.r_[box_array(geo['3d_bbox']), box_size(geo['3d_bbox'], geo)])
            self._scan_car_boxes[key] = (np.asarray(tracks, dtype=np.int64),
                                         np.asarray(boxes, dtype=np.float32).reshape(-1, 7))
        tracks, boxes = self._scan_car_boxes[key]
        return boxes[tracks != r.tracklet]

    def __getitem__(self, request):
        return self.__getitems__([request])[0]

    def __getitems__(self, requests):
        # 调度仍沿用 A 的物理队列。只改变 B 普通行的状态寿命，先于原始帧运输。
        if self.four_candidates:
            requests = [replace(r, window_start=r.frame, window_end=r.frame + 1)
                        if 0 <= r.branch < 4 and not (r.starts_window and r.ends_window)
                        else r for r in requests]
        # 相同标注帧在一个 IPC 包内共享；不同 tracklet 的框绝不共用。
        frames = {}
        required = None
        if self.worker_transport:
            # 先取全批并集。别的分支可能需要同一标注帧的完整 GT10。
            required = {(r.tracklet, index) for r in requests if 0 <= r.branch < 4
                        for index in range(max(0, r.frame - (3 if r.starts_window else 1)), r.frame + 1)}
        return [self._get_row(request, frames, required) for request in requests]

    def _get_row(self, request, frame_cache, required):
        if not isinstance(request, EndpointRequest):
            raise TypeError('v31 raw dataset requires EndpointRequest')
        r = request
        history_ids = tuple(max(0, r.frame - 3 + i) for i in range(3))
        ids = sorted(set(history_ids + (r.frame,)))
        read = getattr(self.source, 'endpoint_frames', None) or self.source.get_frames
        missing = [i for i in ids if (r.tracklet, i) not in frame_cache]
        if missing:
            values = (read(r.tracklet, missing) if required is None
                      else read(r.tracklet, missing, worker_required=required))
            for index, frame in zip(missing, values):
                frame_cache[r.tracklet, index] = frame
        frames = {i: frame_cache[r.tracklet, i] for i in ids}
        key = (self.source.get_tracklet_key(r.tracklet)
               if hasattr(self.source, 'get_tracklet_key') else str(r.tracklet))
        result = dict(request=r, frames=frames, history_ids=history_ids, tracklet_key=str(key))
        if self.joint_search and r.branch != 4:
            result['joint_other_cars'] = self.other_cars_for(result)
        if r.starts_window:
            if 0 in frames:
                result['first_frame'] = frames[0]
            else:
                first = (r.tracklet, 'first_metadata')
                if first not in frame_cache:
                    if self.joint_search:
                        frame_cache[first] = (read(r.tracklet, [0]) if required is None else
                                              read(r.tracklet, [0], worker_required=set()))[0]
                    else:
                        frame_cache[first] = metadata(self.source, r.tracklet, 0)
                result['first_frame'] = frame_cache[first]
        if self.training and self.enable_m and 0 <= r.branch < 4:
            result['future_supervision'] = future_motion_targets(
                self.source, r, frames, self.lengths[r.tracklet])
        return result


def raw_collate(rows):
    return rows


def full_epoch_diagnostics(config, epoch):
    return config.diagnostics_level == 'full' or int(epoch) in config.full_diagnostic_epochs


def _worker_transport_enabled(config, source):
    if not option(config, 'workers', 4) or option(config, 'dataset') != 'kitti_mf':
        return False
    from datasets.kitti_v41 import KITTIV41Dataset
    return isinstance(source, KITTIV41Dataset)


# _prepare_row 将有效点按原顺序放在每帧前缀；以下是对应字段的尾部填充值。
_POINT_PADDING = dict(points=0., point_ids=-1, point_valid=False, segmentation_labels=-1, bc_targets=0.)


class _WindowStartCollator:
    """只物化无预测前驱的训练首行；可被 spawn 序列化，常驻 worker 仅读请求的 epoch。"""
    def __init__(self, config):
        self.config = config
        self.builder = None

    def __getstate__(self):
        return dict(config=self.config, builder=None)

    def __call__(self, rows):
        starts = [row for row in rows if row['request'].starts_window
                  and 0 <= row['request'].branch < 4
                  and not full_epoch_diagnostics(self.config, row['request'].epoch + 1)]
        if not starts:
            return rows
        if self.builder is None:
            self.builder = BatchBuilder(self.config)
            self.builder.detailed_diagnostics = False
        builder = self.builder
        packets = {}
        # 保留轻量帧定位信息。独立工具切换诊断、重排采样 ID 时可原样回读。
        light_frames = {}
        builder._start_materialization_cache()
        try:
            with recipe_scope(builder.v41_config):
                for row in starts:
                    r = row['request']
                    fields, context = builder._prepare_row(row, builder._new_state(row))
                    valid = fields['point_valid']
                    width = int(valid.sum(axis=1).max())
                    if width < valid.shape[1]:
                        for key in _POINT_PADDING:
                            fields[key] = fields[key][:, :width]
                    del context['row']
                    frame_map = {}
                    for index, frame in row['frames'].items():
                        if id(frame) not in light_frames:
                            cache = frame['_v41_normal_train_cache']
                            light_frames[id(frame)] = dict(frame, _v41_normal_train_cache={
                                'lower': cache['lower'], 'upper': cache['upper']})
                        frame_map[index] = light_frames[id(frame)]
                    packet = (builder._preparation_id, r, fields, context,
                              builder._next_history_pools.pop(r.state_key, None))
                    first = frame_map[0] if 0 in frame_map else row['first_frame']
                    packets[id(row)] = dict(row, frames=frame_map, first_frame=first, _prepared_start=packet)
            # 已经由首行消费、且仅作为其他行较老历史的 GT10 也不必随队列搬运。
            current = {id(frame) for row in rows if id(row) not in packets
                       for i, frame in row['frames'].items()
                       if row['request'].starts_window or i >= row['request'].frame - 1}
            result = []
            for row in rows:
                if id(row) in packets:
                    result.append(packets[id(row)])
                else:
                    frames = {i: light_frames.get(id(f), f) if id(f) not in current else f
                              for i, f in row['frames'].items()}
                    result.append(dict(row, frames=frames))
            return result
        finally:
            builder.clear_materialization_cache()


def raw_dataset_factory(config, role):
    """只加载原始数据适配器，不构造历史 sampler。"""
    if str(option(config, 'dataset', '')) == 'kitti_mf':
        from datasets.kitti_v41 import get_v41_raw_dataset
        return get_v41_raw_dataset(config, role)
    from datasets import get_raw_dataset
    return get_raw_dataset(config, role)


def _build_loaders(config, roles=('train', 'val'), sources=None):
    """返回 role -> DataLoader；sources 支持不依赖 nuScenes 的真实流程测试。"""
    loaders = {}
    for role in roles:
        kitti = str(option(config, 'dataset', '')).startswith('kitti')
        dataset_role = (('auxiliary' if kitti else 'train') if role == 'auxiliary' else
                        ('dev' if kitti else 'test') if role == 'val' else role)
        source = sources[role] if sources is not None else raw_dataset_factory(config, dataset_role)
        transport = _worker_transport_enabled(config, source)
        joint = bool(option(config, 'v41_joint_search', False))
        raw = RawEndpointDataset(source, training=role == 'train', joint_search=joint,
                                 four_candidates=option(config, 'v41_history_training') == 'four_candidates',
                                 enable_m=option(config, 'enable_m', False), worker_transport=transport)
        if role == 'auxiliary':
            from .auxiliary import FullTrackAuxiliarySampler
            sampler = FullTrackAuxiliarySampler(raw.lengths, int(option(config, 'batch_size', 16)),
                seed=int(option(config, 'seed', 42)), fraction=float(option(config, 'v41_aux_fraction', .25)),
                source_sha256=raw.source_sha256)
        else:
            size = 1 if role in ('val', 'test') and option(config, 'v41_joint_latency', False) else int(
                option(config, 'batch_size', 16))
            sampler = ReadyQueueBatchSampler(raw.lengths, size,
                seed=int(option(config, 'seed', 42)), training=role == 'train',
                short_window=int(option(config, 'v31_short_window', 4)),
                long_window=int(option(config, 'v31_long_window', 8)),
                curriculum_epochs=int(option(config, 'v31_curriculum_epochs', 10)),
                source_sha256=raw.source_sha256,
                reserve_windows=int(option(config, 'v32_reserve_windows', 112)),
                seed_translation=float(option(config, 'v32_seed_translation', .3)),
                seed_yaw_degrees=float(option(config, 'v32_seed_yaw_degrees', 1.5)),
                initial_seed_policy=option(config, 'v35_seed_policy', ReadyQueueBatchSampler.INITIAL_SEED_POLICY))
        workers = int(option(config, 'workers', 4))
        prepare_starts = transport and role == 'train' and source.v41_data_role == 'train' and not joint
        collate = _WindowStartCollator(config) if prepare_starts else raw_collate
        kwargs = dict(num_workers=workers, collate_fn=collate, pin_memory=False,
                      generator=torch.Generator().manual_seed(stable_seed(option(config, 'seed', 42), role)))
        if workers:
            kwargs.update(prefetch_factor=2, persistent_workers=True)
        loaders[role] = DataLoader(raw, batch_sampler=sampler, **kwargs)
    return loaders


def build_loaders(config, roles=('train', 'val'), sources=None):
    config = normalize_config(config)
    with recipe_scope(config):
        return _build_loaders(config, roles=roles, sources=sources)


build_dataloaders = build_loaders


@dataclass
class V41AcceptedState:
    boxes: dict
    times: dict
    next_frame: int
    size: np.ndarray
    trusted: list
    transitions: dict
    trusted_velocity: np.ndarray
    trusted_velocity_valid: bool
    last_strong_time: float | None
    last_supported_time: float
    initialization_time: float
    box_provenance: dict
    seed_offset: np.ndarray
    previous_is_propagation: bool = False
    consecutive_propagations: int = 0
    consecutive_untrusted_outputs: int = 0
    last_supported_quality: float = 1.
    last_velocity_update_time: float | None = None
    previous_measurement_eligible: bool | None = None
    joint: JointSearchState | None = None


# GT 仅用于合法初框、监督与被动诊断。
MISSING_BOX, KNOWN_GT_BOX, PERTURBED_BOX, PREDICTED_BOX = range(4)


@dataclass(frozen=True)
class _HistoryCrop:
    signature: tuple
    xyz: np.ndarray
    ids: np.ndarray
    gt10: bool


class BatchBuilder:
    """唯一状态所有者；prepare -> commit 是一次事务。"""
    def __init__(self, config):
        self.config = self.v41_config = normalize_config(config)
        self.joint_search = bool(option(self.config, 'v41_joint_search', False))
        self.history_bank = None
        self.history_counts = Counter()
        self._preparation_id = hashlib.sha256(repr(sorted(self.config.items())).encode()).digest()
        self.train_diagnostics = bool(option(self.config, 'v35_train_diagnostics', True))
        # host 按轮选择 summary/full；默认保留独立工具的历史详细行为。
        # 只控制被动统计，不控制监督、搜索、候选或状态提交。
        self.detailed_diagnostics = True
        self.states = {}
        self._pending = None
        self._raw_memo = None
        self._history_pools = {}
        self._next_history_pools = {}
        self._sparse_scans = OrderedDict()

    def __getstate__(self):
        # 原始世界坐标点池只在本进程中复用，既不是跟踪状态也不是恢复数据。
        state = vars(self).copy()
        state.pop('_history_pools', None)
        state.pop('_next_history_pools', None)
        state.pop('_sparse_scans', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._history_pools = {}
        self._next_history_pools = {}
        self._sparse_scans = OrderedDict()

    def _start_materialization_cache(self):
        self._next_history_pools.clear()
        self._coverage_cache = {}
        self._raw_memo = (RawFrameMemo(point_arrays, box_array, box_size, box_corners, inside_box)
                          if optimized_execution(self.config) else None)

    def clear_materialization_cache(self):
        self._raw_memo = None
        self._coverage_cache = {}
        self._next_history_pools.clear()

    def _points(self, frame):
        return self._raw_memo.points(frame) if self._raw_memo is not None else point_arrays(frame)

    def _joint_scan(self, frame):
        from .joint_grid import make_sparse_scan
        ref = frame.get('_v41_raw_ref')
        key = (ref, frame.get('input_metadata_sha256'), frame.get('v41_data_role'))
        if ref is not None and key in self._sparse_scans:
            self._sparse_scans.move_to_end(key)
            return self._sparse_scans[key]
        xyz, ids = self._points(frame)
        scan = make_sparse_scan(xyz, ids)
        if ref is not None:
            self._sparse_scans[key] = scan
            if len(self._sparse_scans) > 4:
                self._sparse_scans.popitem(last=False)
        return scan

    def _view(self, frame, rois, size, *, scale, offset, normal_train):
        if normal_train and option(self.config, 'dataset', '') == 'kitti_mf':
            from datasets.kitti_v41 import normal_train_point_view
            view = normal_train_point_view(frame, rois, size, scale=scale, offset=offset)
            if view['cache_used'] or '_v41_raw_ref' in frame:
                return view['points'], view['ids'], bool(view['cache_used'])
        xyz, ids = self._points(frame)
        return xyz, ids, False

    def _history_crop(self, frame, index, box, size, scale, offset, normal_train, previous, retain):
        # 只有带物理文件身份的不可变 KITTI endpoint 可跨步复用。
        # 其他数据源可能逐次增强或替换点，即使 tracklet/index 相同也重新裁剪。
        ref = frame.get('_v41_raw_ref')
        signature = None
        if ref is not None and (index in previous or retain is not None):
            gt10 = frame.get('_v41_normal_train_cache') if normal_train else None
            bounds = None if gt10 is None else (gt10['lower'].tobytes(), gt10['upper'].tobytes())
            signature = (ref, frame.get('input_metadata_sha256'), frame.get('v41_data_role'),
                         bounds, box.tobytes(), size.tobytes(), scale, offset)
            cached = previous.get(index)
            if cached is not None and cached.signature == signature:
                if retain is not None:
                    retain[index] = cached
                return cached.xyz, cached.ids, None, cached.gt10
        xyz, ids, used = self._view(frame, [box], size, scale=scale, offset=offset,
                                    normal_train=normal_train)
        indices = np.flatnonzero(inside_box(xyz, box, size, scale=scale, offset=offset))
        if signature is not None and retain is not None:
            # 复制有序未抽样子集；不保留整云、GT10 数组或其视图的引用。
            pool = _HistoryCrop(signature, xyz[indices], ids[indices], used)
            retain[index] = pool
            return pool.xyz, pool.ids, None, used
        return xyz, ids, indices, used

    def _target_coverage(self, frame, target, size):
        """被动总数。若真实 GT 也严格在 GT10 内，该视图包含整帧所有正点。"""
        key = id(frame), id(frame['3d_bbox']), id(frame.get('box_size'))
        if key not in self._coverage_cache:
            cache = frame.get('_v41_normal_train_cache')
            if cache is not None and '_v41_raw_ref' in frame and self.config.dataset == 'kitti_mf':
                from datasets.kitti_v41 import normal_train_point_view
                view = normal_train_point_view(frame, [target], size, scale=1., offset=0.)
                xyz, used = view['points'], view['cache_used']
                count = view['raw_count'] if used else len(xyz)
            else:
                xyz, _ = self._points(frame)
                used = False
                count = len(xyz)
            mask = inside_box(xyz, target, size)
            self._coverage_cache[key] = (frame, count, mask, used)
        return self._coverage_cache[key][1:]

    def reset(self):
        self.history_counts.clear()
        self.states.clear()
        self._history_pools.clear()
        self._sparse_scans.clear()
        self._pending = None
        self.clear_materialization_cache()

    def _new_state(self, row):
        # 与 v35 相同的合法 seed 几何和局部 RNG；仅删除身份点记忆初始化。
        r, frames = row['request'], row['frames']
        if r.branch == 5 and (r.window_start != 1 or r.frame != 1):
            raise ValueError('auxiliary trajectories may initialize only from the legal first frame')
        seed_ids = range(max(0, r.frame - 3), r.frame)
        size = box_size(row['first_frame']['3d_bbox'], row['first_frame'])
        boxes = {i: box_array(frames[i]['3d_bbox']) for i in seed_ids}
        times = {i: float(frames[i]['timestamp']) for i in seed_ids}
        seed_offset = np.zeros(4, dtype=np.float64)
        provenance = KNOWN_GT_BOX
        if r.branch in (1, 2, 3) and not (r.frame == 1):
            seed = option(self.config, 'seed', 42)
            if self.config.v41_history_training == 'legacy':
                rng = np.random.default_rng(stable_seed(seed, row['tracklet_key'], r.state_key, 'window_seed'))
            else:
                if self.history_bank is None:
                    raise ValueError('paired history training needs its validated E57 history bank')
                key = (seed, r.epoch, row['tracklet_key'], r.frame, r.branch)
                rng = np.random.default_rng(stable_seed(*key, 'last_seed'))
                donor_rng = np.random.default_rng(stable_seed(*key, 'history_donor'))
                deltas, match = self.history_bank.sample(row['tracklet_key'], seed_ids, times, boxes, donor_rng)
                self.history_counts[match] += 1
                for index, delta in zip(seed_ids, deltas):
                    boxes[index][:3] += delta
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
        last_strong_time = None
        supported_seeds = {}
        for i in sorted(seed_ids, key=times.__getitem__, reverse=True):
            xyz, ids, _ = self._view(frames[i], [boxes[i]], size, scale=1., offset=0.,
                                     normal_train=0 <= r.branch < 4)
            members = inside_box(xyz, boxes[i], size)
            count = (np.count_nonzero(_unique_mask(ids[members])) if self.joint_search else
                     np.count_nonzero(members))
            if self.joint_search:
                supported_seeds[i] = count >= 3
            if count >= 3 and last_strong_time is None:
                last_strong_time = times[i]
                if not self.joint_search:
                    break
        state = V41AcceptedState(boxes=boxes, times=times, next_frame=r.frame, size=size,
            trusted=trusted, transitions={i: True for i in seed_ids},
            trusted_velocity=velocity, trusted_velocity_valid=len(trusted) == 2,
            last_strong_time=last_strong_time,
            last_supported_time=last_time, initialization_time=last_time,
            box_provenance={i: provenance for i in seed_ids},
            seed_offset=seed_offset,
            last_velocity_update_time=last_time if len(trusted) == 2 else None)
        if self.joint_search:
            first = row['first_frame']
            first_box = box_array(first['3d_bbox'])
            xyz, ids, _ = self._view(first, [first_box], size, scale=1., offset=0.,
                                     normal_train=0 <= r.branch < 4)
            members = inside_box(xyz, first_box, size)
            unique_first_points = xyz[members][_unique_mask(ids[members])]
            state.joint = JointSearchState.from_seed(state, frames, first, unique_first_points, first_box,
                supported_seeds, patrol_seconds=self.config.v41_patrol_seconds)
        return state

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
        with recipe_scope(self.v41_config):
            return self._prepare(rows)

    def _prepare(self, rows):
        if self._pending is not None:
            raise RuntimeError('previous v41 transaction has not been committed')
        if not rows:
            raise ValueError('empty raw batch')
        keys = [row['request'].state_key for row in rows]
        if len(keys) != len(set(keys)):
            raise ValueError('adjacent endpoints of one window cannot share a batch')
        prepared, contexts = [], []
        for row in rows:
            r = row['request']
            packet = row.get('_prepared_start')
            if (packet is not None and (packet[0] != self._preparation_id or packet[1] != r
                                        or self.detailed_diagnostics or not r.starts_window)):
                packet = None
            replay = row.get('_joint_replay_state')
            q0_replay = row.get('_q0_replay_state')
            if q0_replay is not None:
                if not (option(self.config, 'v41_q0_lateral_replay', False) and not self.joint_search
                        and replay is None and q0_replay.joint is None and r.branch == 0
                        and r.starts_window and r.ends_window and r.state_key not in self.states):
                    raise ValueError('invalid q0 replay prefix for one-step training slot')
                packet, state = None, deepcopy(q0_replay)
            elif replay is not None:
                if not (self.joint_search and r.branch == 0 and r.starts_window and r.ends_window
                        and r.state_key not in self.states and replay.joint is not None):
                    raise ValueError('invalid joint replay prefix for one-step training slot')
                packet, state = None, deepcopy(replay)
            elif r.starts_window:
                if r.state_key in self.states:
                    raise RuntimeError('duplicate window initialization')
                state = self._new_state(row) if packet is None else deepcopy(packet[3]['state'])
            elif r.state_key not in self.states:
                raise RuntimeError('missing accepted predecessor; no GT reseeding is allowed')
            else:
                state = self.states[r.state_key]
            if state.next_frame != r.frame:
                raise RuntimeError('out-of-order accepted state')
            if packet is None:
                data, context = self._prepare_row(row, state)
            else:
                _, _, data, seed_context, pools = packet
                context = dict(seed_context, row=row, state=state)
                if pools:
                    self._next_history_pools[r.state_key] = pools
            prepared.append(data)
            contexts.append(context)
        slots = int(option(self.config, 'point_sample_size', 1024))
        compact = any(row['point_valid'].shape[1] < slots for row in prepared)
        batch = {}
        for key in prepared[0]:
            arrays = [row[key] for row in prepared]
            if key in ('joint_rows', 'joint_other_cars'):
                batch[key] = tuple(arrays)
                continue
            if compact and key in _POINT_PADDING:
                # 直接写最终批，混排的续行、稠密首行与零点首行都保留同一输入形状。
                sample = arrays[0]
                values = np.full((len(arrays), sample.shape[0], slots, *sample.shape[2:]),
                                 _POINT_PADDING[key], dtype=sample.dtype)
                for dest, src in zip(values, arrays):
                    dest[:, :src.shape[1]] = src
            else:
                values = np.stack(arrays)
            batch[key] = torch.from_numpy(values)
        self._pending = contexts
        return batch

    def _prepare_row(self, row, state):
        r, frames = row['request'], row['frames']
        current = frames[r.frame]
        now = float(current['timestamp'])
        effective_now = float(current.get('_ct_effective_timestamp', now))
        anchor = state.boxes[r.frame - 1].copy()
        joint = state.joint
        if joint is not None:
            anchor = joint.crop_center(state, now)
        history_valid = np.asarray([r.frame - 3 + i >= 0 for i in range(3)], dtype=bool)
        if joint is not None:
            history_valid &= np.asarray(row['history_ids']) >= joint.history_start
        history = np.stack([joint.weak_history_boxes.get(i, state.boxes[i])
                            if joint is not None else state.boxes[i] for i in row['history_ids']])
        history_provenance = np.asarray([state.box_provenance[i] if valid else MISSING_BOX
            for i, valid in zip(row['history_ids'], history_valid)], dtype=np.int64)
        history_local = history.copy()
        history_local[:, :3] -= anchor[:3]
        physical_times = np.asarray([float(frames[i]['timestamp']) - now for i in row['history_ids']])
        effective_times = (physical_times if joint is not None else
            np.asarray([float(frames[i].get('_ct_effective_timestamp', frames[i]['timestamp'])) - effective_now
                        for i in row['history_ids']]))
        if not (physical_times < 0).all() or not (effective_times < 0).all():
            raise ValueError('B1 history must be strictly in the past')
        n = int(option(self.config, 'point_sample_size', 1024))
        scale, offset = float(option(self.config, 'bb_scale', 1.25)), float(option(self.config, 'bb_offset', 2.))
        search_plan = make_search_plan(anchor=anchor, trusted_box=state.trusted[-1][0],
            trusted_time=state.trusted[-1][1], velocity=state.trusted_velocity,
            velocity_valid=state.trusted_velocity_valid, timestamp=now,
            previous_measurement_eligible=state.previous_measurement_eligible,
            policy=option(self.config, 'v41_search_policy', 'accepted_only'),
            max_age=float(option(self.config, 'v41_search_max_age', 1.5)),
            main_quota=int(option(self.config, 'v41_search_main_quota', 768)),
            backup_quota=int(option(self.config, 'v41_search_backup_quota', 256)))
        # R 的公开seed始终存在；搜索是否触发不影响该几何，也不读取GT。
        recovery_seed = state.trusted[-1][0].copy()
        recovery_seed_elapsed = (min(now - state.trusted[-1][1],
            float(option(self.config, 'v41_search_max_age', 1.5))) if state.trusted_velocity_valid else 0.)
        recovery_seed[:3] += state.trusted_velocity * recovery_seed_elapsed
        recovery_seed_local = recovery_seed.copy()
        recovery_seed_local[:3] -= anchor[:3]
        points, valid, raw_ids, seg, bc, truth_boxes = [], [], [], [], [], []
        cache_used = []
        main_training = 0 <= r.branch < 4
        previous_pools = {} if r.starts_window else self._history_pools.get(r.state_key, {})
        next_pools = {}
        record_training = self.detailed_diagnostics and self.train_diagnostics and main_training
        record_current = record_training or (self.detailed_diagnostics and r.branch == 4)
        history_raw_counts, history_crop_counts = [], []
        search_target_counts = {}
        for slot, frame_id in enumerate(row['history_ids'] + (r.frame,)):
            frame = frames[frame_id]
            is_current = slot == 3
            exists = is_current or history_valid[slot]
            crop_box = anchor if is_current else history[slot]
            if is_current:
                rois = [crop_box] if search_plan.backup_box is None else [crop_box, search_plan.backup_box]
                xyz, ids, used = self._view(frame, rois, state.size, scale=scale, offset=offset,
                                            normal_train=main_training)
                search_sample = sample_search_points(xyz, ids, search_plan, state.size,
                    count=n, main_quota=search_plan.main_quota, backup_quota=search_plan.backup_quota,
                    seed=stable_seed(option(self.config, 'seed', 42), r.epoch,
                                     row['tracklet_key'], r.branch, r.frame, slot),
                    scale=scale, offset=offset, known_unique_ids='_v41_raw_ref' in frame)
                selected = search_sample.indices
            elif exists:
                # slot 0 在下一步已过期。只为同窗仍将读取的历史帧准备点池。
                retain = next_pools if slot > 0 and not r.ends_window else None
                xyz, ids, crop_indices, used = self._history_crop(frame, frame_id, crop_box, state.size,
                    scale, offset, main_training, previous_pools, retain)
                selected = crop_indices
                pool_size = len(ids) if crop_indices is None else len(crop_indices)
                if pool_size > n:
                    rng = np.random.default_rng(stable_seed(option(self.config, 'seed', 42),
                        r.epoch, row['tracklet_key'], r.branch, r.frame, slot))
                    selected = np.sort(rng.choice(pool_size if crop_indices is None else crop_indices,
                                                  n, replace=False))
            else:
                used = False
                if self.detailed_diagnostics:
                    _, _, used = self._view(frame, [crop_box], state.size, scale=scale, offset=offset,
                                             normal_train=main_training)
                selected = np.empty(0, dtype=np.int64)
            cache_used.append(used)
            if selected is None:
                sampled_xyz, sampled_ids = xyz, ids
            elif len(selected):
                sampled_xyz, sampled_ids = xyz[selected], ids[selected]
            else:
                sampled_xyz, sampled_ids = np.empty((0, 3), dtype=np.float64), np.empty(0, dtype=np.int64)
            count = len(sampled_ids)
            point = np.zeros((n, 5), dtype=np.float32)
            point[:count, :3] = sampled_xyz - anchor[:3]
            point[:count, 3] = float(frame['timestamp']) - now
            if is_current:
                point[:count, 4] = .5
            elif count and joint is not None and frame_id in joint.weak_history_boxes:
                point[:count, 4] = .5
            elif count:
                hint = inside_box(sampled_xyz, crop_box, state.size).astype(np.float32)
                point[:count, 4] = hint if history_provenance[slot] == KNOWN_GT_BOX else .2 + .6 * hint
            point_ids = np.full(n, -1, dtype=np.int64)
            point_ids[:count] = sampled_ids
            points.append(point)
            valid.append(np.arange(n) < count)
            raw_ids.append(point_ids)
            if not main_training and not is_current:
                continue
            if main_training and self._raw_memo is not None:
                target, target_size, centers_corners = self._raw_memo.geometry(frame)
            else:
                target = box_array(frame['3d_bbox'])
                target_size = box_size(frame['3d_bbox'], frame)
                if main_training:
                    centers_corners = np.vstack((target[:3], box_corners(target, target_size)))
            label_needed = main_training or r.branch == 5 or record_current
            if label_needed:
                label = np.full(n, -1, dtype=np.int64)
                label[:count] = inside_box(sampled_xyz, target, target_size)
                seg.append(label)
            if main_training:
                distances = np.zeros((n, 9), dtype=np.float32)
                distances[:count] = np.linalg.norm(sampled_xyz[:, None] - centers_corners[None], axis=-1)
                bc.append(distances)
            if is_current and record_current:
                raw_count, target_mask, gt_view = self._target_coverage(frame, target, target_size)
                target_count = int(target_mask.sum())
                # 所有下标属于同一个搜索视图，不需要 raw-ID 哈希表或再次扫描裁剪区。
                view_target = target_mask if gt_view == used else inside_box(xyz, target, target_size)
                union = search_sample.union_indices
                main = search_sample.main_indices
                union_count = int(view_target[union].sum())
                sampled_target = label[:count] == 1
                sampled_main = np.isin(selected[sampled_target], main, assume_unique=True).sum()
                search_target_counts = dict(
                    diagnostic_search_raw_target_count=np.int64(target_count),
                    diagnostic_search_main_target_count=np.int64(view_target[main].sum()),
                    diagnostic_search_union_target_count=np.int64(union_count),
                    diagnostic_search_sampled_main_target_count=np.int64(sampled_main),
                    diagnostic_search_sampled_backup_target_count=np.int64(sampled_target.sum() - sampled_main),
                    diagnostic_target_count=np.int64(target_count), diagnostic_raw_point_count=np.int64(raw_count),
                    diagnostic_crop_target_count=np.int64(union_count), diagnostic_crop_point_count=np.int64(len(union)),
                    diagnostic_sampled_target_count=np.int64(sampled_target.sum()), diagnostic_sampled_point_count=np.int64(count),
                    diagnostic_novel_target_count=np.int64(target_count - union_count))
            if not is_current and record_training:
                if exists:
                    _, target_mask, gt_view = self._target_coverage(frame, target, target_size)
                    target_count = int(target_mask.sum())
                    if crop_indices is None:
                        crop_count = inside_box(xyz, target, target_size).sum()
                    elif gt_view == used:
                        crop_count = target_mask[crop_indices].sum()
                    else:
                        crop_count = inside_box(xyz[crop_indices], target, target_size).sum()
                history_raw_counts.append(target_count if exists else 0)
                history_crop_counts.append(int(crop_count) if exists else 0)
            target[:3] -= anchor[:3]
            truth_boxes.append(target)
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
            recovery_seed_box=recovery_seed_local.astype(np.float32),
            recovery_seed_valid=np.bool_(True),
            target_box=np.asarray(truth_boxes[-1], dtype=np.float32),
            target_box_size=target_size.astype(np.float32),
            branch_id=np.int64(r.branch),
            trusted_velocity=state.trusted_velocity.astype(np.float32),
            trusted_velocity_valid=np.bool_(state.trusted_velocity_valid),
            weak_age=np.float32(self._strong_age(state, now)),
            supported_age=np.float32(max(0., now - state.last_supported_time)))
        if seg:
            result['segmentation_labels'] = np.stack(seg)
        if main_training:
            result.update(history_target_boxes=np.asarray(truth_boxes[:3], dtype=np.float32),
                          bc_targets=np.stack(bc), physical_valid=np.bool_(True))
        if main_training or record_current:
            previous_gt = box_array(frames[r.frame - 1]['3d_bbox'])
            current_gt = box_array(current['3d_bbox'])
            displacement = (current_gt[:2] - previous_gt[:2]).astype(np.float32)
            result['physical_displacement'] = displacement
        result['history_is_initial'] = (history_valid
            & (np.asarray(row['history_ids']) == 0) & (history_provenance == KNOWN_GT_BOX))
        if option(self.config, 'enable_m', False) and 0 <= r.branch < 4:
            if 'future_supervision' not in row:
                raise ValueError('M training requires worker-produced future metadata labels')
            future = row['future_supervision']
            if set(future) != {'future_displacement', 'future_dt', 'future_valid'}:
                raise ValueError('future supervision may contain only the three M label fields')
            result.update(future)
        if self.detailed_diagnostics:
            result.update(diagnostic_search_triggered=np.bool_(search_plan.triggered),
                diagnostic_train_cache_used=np.asarray(cache_used, dtype=bool),
                diagnostic_recovery_seed_box_world=recovery_seed.astype(np.float32),
                diagnostic_recovery_seed_elapsed=np.float32(recovery_seed_elapsed),
                diagnostic_search_reason_id=np.int64({'policy_disabled': 0, 'initialization': 1,
                    'accepted_eligible': 2, 'accepted_untrusted': 3}[search_plan.reason]),
                diagnostic_search_main_box_world=search_plan.main_box.astype(np.float32),
                diagnostic_search_backup_box_world=(search_plan.main_box if search_plan.backup_box is None
                                                   else search_plan.backup_box).astype(np.float32),
                diagnostic_search_backup_active=np.bool_(search_plan.backup_box is not None),
                diagnostic_search_main_quota=np.int64(search_plan.main_quota),
                diagnostic_search_backup_quota=np.int64(search_plan.backup_quota),
                diagnostic_search_backup_elapsed=np.float32(search_plan.backup_elapsed),
                diagnostic_search_trusted_age=np.float32(search_plan.trusted_age),
                diagnostic_search_main_count=np.int64(search_sample.main_count),
                diagnostic_search_backup_only_count=np.int64(search_sample.backup_only_count),
                diagnostic_search_sampled_main=np.int64(search_sample.sampled_main),
                diagnostic_search_sampled_backup=np.int64(search_sample.sampled_backup),
                diagnostic_previous_measurement_eligible=np.int64(-1 if state.previous_measurement_eligible is None
                                                                 else int(state.previous_measurement_eligible)))
            result.update(search_target_counts)
        if record_training:
            result.update(diagnostic_history_raw_target_count=np.asarray(history_raw_counts, dtype=np.int64),
                          diagnostic_history_crop_target_count=np.asarray(history_crop_counts, dtype=np.int64))
        context = dict(row=row, state=state, anchor=anchor, timestamp=now, search_plan=search_plan,
                       sampled_xyz=sampled_xyz, sampled_ids=sampled_ids)
        if joint is not None:
            scan = physical_frame(current, r.frame)
            scheduled = joint.search_due(state, now, scan, patrol_seconds=self.config.v41_patrol_seconds,
                                         weak_patrol_seconds=self.config.v41_weak_patrol_seconds)
            sparse_scan = self._joint_scan(current) if scheduled else None
            scan_xyz, scan_ids = (sparse_scan.xyz, sparse_scan.ids) if sparse_scan is not None else (EMPTY_XYZ, EMPTY_IDS)
            pending = joint.pending_at(scan)
            backup = search_plan.backup_box
            result.update(joint.motion_fields(anchor, now), joint_track_id=np.int64(r.tracklet),
                          joint_other_cars=row.get('joint_other_cars', np.empty((0, 7), dtype=np.float32)))
            result['joint_rows'] = dict(template_xyz=joint.template_xyz, scan_xyz=scan_xyz, scan_ids=scan_ids,
                sparse_scan=sparse_scan,
                pending_valid=pending, pending_box=relative(joint.pending_box if pending else anchor, anchor),
                main_box=relative(search_plan.main_box, anchor), backup_valid=backup is not None,
                backup_box=relative(backup if backup is not None else anchor, anchor))
            context.update(joint_scheduled=scheduled, joint_scan=scan)
            if r.branch == 5 and not (seg[-1] > 0).any():
                _, gt_mask, gt_view = self._target_coverage(current, box_array(current['3d_bbox']), target_size)
                in_view = gt_mask if gt_view == used else inside_box(xyz, box_array(current['3d_bbox']), target_size)
                context.update(joint_raw_target_count=int(gt_mask.sum()),
                               joint_union_target_count=int(in_view[search_sample.union_indices].sum()))
        if next_pools:
            self._next_history_pools[r.state_key] = next_pools
        result.update(previous_is_propagation=np.bool_(state.previous_is_propagation),
            consecutive_propagations=np.int64(state.consecutive_propagations),
            consecutive_untrusted_outputs=np.int64(state.consecutive_untrusted_outputs),
            trusted_velocity_age=np.float32(self._velocity_age(state, now)))
        return result, context

    @raw_transaction
    def commit(self, output, batch, *, diagnostics=False, defer_joint_diagnostics=False):
        with recipe_scope(self.v41_config):
            return self._commit(output, batch, diagnostics=diagnostics, defer_joint_diagnostics=defer_joint_diagnostics)

    @staticmethod
    def _support_attribution(row, observed, other_cars):
        own = row['frames'][row['request'].frame]
        return dict(joint_target_support_count=int(inside_box(observed, box_array(own['3d_bbox']),
                                                              box_size(own['3d_bbox'], own)).sum()),
                    joint_max_other_support_count=max((int(inside_box(observed, car[:4], car[4:]).sum())
                                                       for car in other_cars), default=0))

    def complete_joint_diagnostics(self, rows, committed, raw_dataset):
        """真实提交已结束；只在 full 日志中归因这些独立物理点。"""
        for row, record in zip(rows, committed):
            observed = record.pop('_joint_support_xyz')
            other_cars = raw_dataset.other_cars_for(row) if len(observed) else ()
            record.update(self._support_attribution(row, observed, other_cars))

    def _commit(self, output, batch, *, diagnostics=False, defer_joint_diagnostics=False):
        if self._pending is None:
            raise RuntimeError('no pending v41 transaction (duplicate commit)')
        full = self.v41_config.v41_arm == 'full'
        b0_only = not bool(option(self.config, 'enable_b1', True))
        selected = output.selected_index.detach().long()
        quality = output.selected_quality.detach()
        b = len(self._pending)
        if output.accepted_box.shape != (b, 4) or output.prior.box.shape != (b, 4):
            raise ValueError('v41 accepted/prior boxes must match pending rows as [B,4]')
        probability = output.observation.foreground_probability
        point_ids, point_valid = batch['point_ids'], batch['point_valid']
        if (point_ids.ndim != 3 or point_ids.shape[0] != b or point_ids.shape[1] < 1
                or probability.shape != point_ids.shape or point_valid.shape != point_ids.shape):
            raise ValueError('v41 foreground probability/point IDs/validity must align as [B,T,N]')
        if selected.shape != (b,) or quality.shape != (b,):
            raise ValueError('v41 selection/quality must match pending rows')
        slots = 6 if self.joint_search else 4
        invalid_selection = ((selected != 0) & (selected != 2) if b0_only
                             else (selected < 0) | (selected >= (slots if full else 2)))
        checks = {'v41 selection must index q0/prior/trusted fallback/recovery': ~invalid_selection.any(),
                  'v41 selected quality must be finite': torch.isfinite(quality).all()}
        fallback = getattr(output, 'q0_is_fallback', None)
        if fallback is None:
            sequence = getattr(output.observation, 'sequence_valid', None)
            fallback = (~sequence.bool() if sequence is not None else torch.zeros_like(selected, dtype=torch.bool))
        fallback = fallback.detach().bool()
        if fallback.shape != (b,):
            raise ValueError('v41 fallback mask must match pending rows')
        propagation = (selected != 0) | fallback
        trusted_score = quality
        if full:
            candidates = output.candidates
            row = torch.arange(b, device=selected.device)
            if candidates.boxes.shape != (b, slots, 4) or candidates.valid.shape != (b, slots):
                raise ValueError('v41 Full candidate slots differ from its configured observation path')
            measurement = candidates.measurement_candidate.detach().bool()
            if measurement.shape != (b, slots):
                raise ValueError('v41 measurement candidates do not align with the configured slots')
            # 越界选择也会被拒绝；clamp 仅让异步检查完成前的 gather 安全。
            safe_selected = selected.clamp(0, slots - 1)
            propagation = ~measurement[row, safe_selected]
            checks['only real q0/recovery may be measurement candidates'] = ~measurement[:, 1:3].any()
            checks['propagation-alias q0 cannot be a measurement candidate'] = ~(measurement[:, 0] & fallback).any()
            checks['v41 cannot commit invalid or propagation-alias q0'] = (
                candidates.valid[row, safe_selected].all() & ~((selected == 0) & fallback).any())
            checks['accepted box differs from unique selected candidate'] = (
                output.accepted_box.detach() == candidates.boxes.detach()[row, safe_selected]).all()
            legacy = output.legacy_q0_quality.detach()
            if legacy.shape != (b,):
                raise ValueError('v41 requires separate legacy q0 quality')
            expected_quality = torch.where((selected == 0) & ~fallback, legacy, torch.zeros_like(legacy))
            checks['only real q0 can carry legacy quality'] = (quality == expected_quality).all()
            trusted_score = output.commit_decision.trusted_score.detach()
            expected_eligible = output.commit_decision.eligible.detach().bool()
            if trusted_score.shape != (b,) or expected_eligible.shape != (b,):
                raise ValueError('v41 commit decision must match pending rows')
            checks['propagation cannot carry trusted quality'] = (
                ((trusted_score >= 0) & (trusted_score <= 1) & ((trusted_score == 0) | ~propagation)).all())
            if self.joint_search:
                evidence = output.joint
                canonical_support = evidence.support_mask[row, safe_selected] & ~propagation[:, None]
                identity_ok = evidence.identity_ok[row, safe_selected]
                geometry_ok = evidence.geometry_ok[row, safe_selected]
                checks['unconfirmed remote candidate selected'] = ((selected != 5) | evidence.remote_valid).all()
            else:
                from .verification import selected_commit_support
                canonical_support = selected_commit_support(output, batch, safe_selected) & ~propagation[:, None]
            support = output.commit_decision.support_mask.detach().bool()
            source = output.commit_decision.source.detach().long()
            if support.shape != canonical_support.shape or source.shape != selected.shape:
                raise ValueError('v41 commit source/support has invalid shape')
            checks['v41 commit support differs from actual unique current-B0 set'] = (support == canonical_support).all()
            checks['v41 commit source differs from selected candidate'] = (source == selected).all()
        need_raw_support = not self.joint_search and (not full or diagnostics)
        transfer = dict(boxes=output.accepted_box, checks=torch.stack(tuple(checks.values())),
            trusted_score=trusted_score, propagation=propagation, prior_boxes=output.prior.box)
        if diagnostics or self.joint_search:
            transfer['selected'] = selected
        if self.joint_search:
            transfer.update(identity_ok=identity_ok, geometry_ok=geometry_ok,
                joint_mask=canonical_support, joint_ids=evidence.support_ids[row, safe_selected],
                joint_xyz=evidence.support_xyz[row, safe_selected], remote_box=evidence.remote_box,
                remote_proposed=evidence.remote_proposed)
            if diagnostics:
                remote = evidence.observations.remote
                transfer['joint_fine_ids'] = torch.where(remote.point_valid, remote.ids, -1)
        if full:
            transfer.update(eligible=expected_eligible, support_count=canonical_support.sum(-1))
        if need_raw_support:
            transfer.update(probabilities=output.observation.foreground_probability[:, -1],
                point_ids=batch['point_ids'][:, -1], point_valid=batch['point_valid'][:, -1])
        transferred = tensors_to_numpy(transfer, optimized=optimized_execution(self.config))
        for message, passed in zip(checks, transferred['checks']):
            if not passed:
                raise ValueError(message)
        if not np.isfinite(transferred['prior_boxes']).all():
            raise FloatingPointError('non-finite v41 prior box')
        if self.joint_search and not np.isfinite(transferred['remote_box']).all():
            raise FloatingPointError('non-finite joint remote box')
        staged = []
        # 先独立复算全部授权支持与提交资格，任何不一致都不留下半个 batch 的状态。
        for i, context in enumerate(self._pending):
            state, row = context['state'], context['row']
            box = transferred['boxes'][i].astype(np.float64, copy=True)
            box[:3] += context['anchor'][:3]
            if box.shape != (4,) or not np.isfinite(box).all():
                raise FloatingPointError('non-finite v41 accepted box')
            raw_count = None
            if need_raw_support:
                ids = transferred['point_ids'][i, transferred['point_valid'][i]]
                probs = transferred['probabilities'][i, transferred['point_valid'][i]]
                xyz = context['sampled_xyz']
                foreground = np.isfinite(probs) & (probs >= .5)
                if not np.array_equal(ids, context['sampled_ids']):
                    # 外部调用者提供子集或重排时仍按原帧物理 ID 取对应坐标。
                    cloud, raw_ids = self._points(row['frames'][row['request'].frame])
                    positions = {int(value): j for j, value in enumerate(raw_ids)}
                    xyz = cloud[[positions[int(value)] for value in ids]]
                    foreground &= _unique_mask(ids)
                raw_count = int(np.count_nonzero(inside_box(xyz[foreground], box, state.size)))
            # Full 的 T 标签、硬门与提交共用同一 float32 公开点集合，避免
            # raw-world double 与 anchor-local float 的边界差异偷偷改提交资格。
            count = int(transferred['support_count'][i]) if full else raw_count
            prop = bool(transferred['propagation'][i])
            # B1 原状态数学不读新 T；Full 的传播输出没有当前测量资格。
            strong = count >= self.v41_config.v41_min_support and (not prop if full else True)
            score = float(transferred['trusted_score'][i])
            supported = strong and bool(np.isfinite(score) and score >= self.v41_config.v41_trust_threshold)
            if self.joint_search:
                supported = bool(strong and transferred['identity_ok'][i] and transferred['geometry_ok'][i])
                source = int(transferred['selected'][i])
                if source == 5 and not (supported and state.joint.pending_at(context['joint_scan'])
                                        and len(state.joint.template_xyz) >= 3):
                    raise ValueError('joint remote takeover requires a confirmed pending scan and identity template')
                mask = transferred['joint_mask'][i]
                ids = transferred['joint_ids'][i, mask]
                if len(np.unique(ids)) != count or (ids < 0).any():
                    raise ValueError('joint support must contain distinct physical point IDs')
                observed = transferred['joint_xyz'][i, mask].astype(np.float64) + context['anchor'][:3]
                if not np.isfinite(observed).all():
                    raise ValueError('nonfinite joint support')
                # 各候选自身的支持几何与标签共享同一扫描。独立模板只需要在建锚时复制。
                if source == 5 or not len(state.joint.template_xyz) or diagnostics:
                    if source == 5:
                        raw = batch['joint_rows'][i]
                        source_ids, cloud = raw['scan_ids'], raw['scan_xyz']
                    else:
                        source_ids, cloud = context['sampled_ids'], context['sampled_xyz']
                    if len(ids):
                        indices = np.argsort(source_ids) if not np.all(source_ids[:-1] <= source_ids[1:]) else None
                        offsets = np.searchsorted(source_ids, ids, sorter=indices)
                        if (offsets >= len(source_ids)).any():
                            raise ValueError('joint support ID outside candidate scan')
                        indices = offsets if indices is None else indices[offsets]
                        if not np.array_equal(source_ids[indices], ids) or not np.allclose(
                                cloud[indices], observed, rtol=0., atol=1e-4):
                            raise ValueError('joint support coordinates differ from the candidate scan IDs')
                context['joint_support_xyz'] = observed
                raw_count = count
            if full and bool(transferred['eligible'][i]) != supported:
                raise ValueError('v41 commit eligibility disagrees with unique current-B0 support')
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
            observation_refresh = supported if self.joint_search else strong
            if observation_refresh:
                state.last_strong_time = now
            velocity_updated = False
            velocity_reason = 'unsupported_retained'
            if self.joint_search:
                source = int(transferred['selected'][i])
                prior = transferred['prior_boxes'][i].astype(np.float64, copy=True)
                prior[:3] += context['anchor'][:3]
                proposal = None
                if transferred['remote_proposed'][i]:
                    proposal = transferred['remote_box'][i].astype(np.float64, copy=True)
                    proposal[:3] += context['anchor'][:3]
                velocity_updated = state.joint.advance(state, frame=r.frame, scan=context['joint_scan'],
                    now=now, box=box, prior=prior, source=source, supported=supported,
                    support_xyz=context['joint_support_xyz'], searched=context['joint_scheduled'], proposal=proposal)
                state.transitions[r.frame] = velocity_updated
                if supported:
                    state.last_supported_quality = score
                    velocity_reason = 'supported_adjacent' if velocity_updated else 'measurement_reanchored'
            elif supported:
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
            state.previous_is_propagation = prop
            # 记录真实提交资格，不由 T 分数或搜索位置推断；初始化仍为 None。
            state.previous_measurement_eligible = bool(supported and not prop)
            state.consecutive_propagations = state.consecutive_propagations + 1 if prop else 0
            state.consecutive_untrusted_outputs = 0 if supported else state.consecutive_untrusted_outputs + 1
            if diagnostics:
                source = ('q0', 'prior', 'trusted_propagation', 'recovery_measurement',
                          'local_vote', 'remote_vote')[int(transferred['selected'][i])]
                if prop and source == 'q0':
                    source = 'prior'
                records.append(dict(strong=bool(strong), supported=bool(supported),
                    geometric_fg_count=count if not (full and prop) else 0,
                    raw_geometry_fg_count=raw_count,
                    trusted_velocity_valid=bool(state.trusted_velocity_valid),
                    trusted_velocity_reset_reason=velocity_reason,
                    weak_age=self._strong_age(state, now), supported_age=max(0., now - state.last_supported_time),
                    trusted_velocity_age=self._velocity_age(state, now),
                    trusted_position_updated=bool(supported), trusted_velocity_updated=velocity_updated,
                    trusted_write=bool(supported), propagation_only=prop, commit_source=source,
                    measurement_eligible=state.previous_measurement_eligible,
                    search_reason=context['search_plan'].reason,
                    commit_quality=score, last_supported_quality=state.last_supported_quality,
                    last_velocity_update_time=state.last_velocity_update_time,
                    consecutive_propagations=state.consecutive_propagations,
                    consecutive_untrusted_outputs=state.consecutive_untrusted_outputs,
                    support_policy='none' if prop else ('candidate_point_ids' if self.joint_search else 'current_b0_only')))
                if self.joint_search:
                    observed = context['joint_support_xyz']
                    fine_ids = transferred['joint_fine_ids'][i]
                    fine_ids = np.unique(fine_ids[fine_ids >= 0])
                    scheduled = bool(context['joint_scheduled'])
                    pending = state.joint.pending_scan == context['joint_scan']
                    records[-1].update(joint_coarse_scan=scheduled, joint_scan_called=scheduled,
                        joint_scan_points=len(batch['joint_rows'][i]['scan_ids']),
                        joint_extra_points=int((~np.isin(fine_ids, context['sampled_ids'])).sum()),
                        joint_fine_points=len(fine_ids),
                        joint_template_points=len(state.joint.template_xyz),
                        joint_pending=pending, joint_pending_started=pending, joint_remote_takeover=source == 'remote_vote',
                        joint_identity_ok=bool(transferred['identity_ok'][i]), joint_geometry_ok=bool(transferred['geometry_ok'][i]))
                    if defer_joint_diagnostics:
                        records[-1]['_joint_support_xyz'] = observed
                    else:
                        records[-1].update(self._support_attribution(row, observed, row.get('joint_other_cars', ())))
            for old in list(state.boxes):
                if old < r.frame - 2:
                    state.boxes.pop(old)
                    state.times.pop(old)
                    state.transitions.pop(old, None)
                    state.box_provenance.pop(old, None)
            if r.ends_window:
                self.states.pop(r.state_key, None)
            elif r.starts_window:
                self.states[r.state_key] = state
        # 全批状态成功提交才发布。异常清理仅丢弃本步新建的池，保留前次提交。
        for context in self._pending:
            key = context['row']['request'].state_key
            if key in self._next_history_pools:
                self._history_pools[key] = self._next_history_pools[key]
            else:
                self._history_pools.pop(key, None)
        self._pending = None
        return records
