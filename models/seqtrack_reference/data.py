"""原 SeqTrack teacher4 数据和独立递归评测适配；不复用 B0 预处理。"""
from collections import Counter, deque
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from pyquaternion import Quaternion

from datasets.data_classes import Box, PointCloud
from ._vendor import points as geometry
from ._vendor.spatial import points_in_box
from ._vendor.misc import get_history_frame_ids_and_masks, generate_timestamp_prev_list
from ._vendor.processing import motion_processing_mf
from .protocol import reference_config


def option(config, key, default=None):
    return config.get(key, default) if isinstance(config, dict) else getattr(config, key, default)


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256(repr(parts).encode()).digest()[:8], 'little') % (2**32)


def metadata(source, tracklet, frame):
    if hasattr(source, 'get_frame_metadata'):
        return source.get_frame_metadata(tracklet, frame)
    return source.get_frames(tracklet, [frame])[0]


def box_array(box):
    return np.r_[np.asarray(box.center, dtype=np.float64),
                 box.orientation.radians * box.orientation.axis[-1]]


def normalize_frame(frame):
    """仅适配 raw source 的对象/数组表示；保留真实框尺寸和原始点。"""
    result = dict(frame)
    box = frame['3d_bbox']
    if not hasattr(box, 'wlh'):
        value = np.asarray(box, dtype=np.float64)
        size = np.asarray(frame['box_size'], dtype=np.float64)[[1, 0, 2]]
        box = Box(value[:3].copy(), size.copy(), Quaternion(axis=[0, 0, 1], radians=value[3]))
    result['3d_bbox'] = deepcopy(box)
    if 'pc' in frame:
        pc = frame['pc']
        xyz = np.asarray(pc.points[:3] if hasattr(pc, 'points') else np.asarray(pc)[:, :3].T)
        result['pc'] = PointCloud(xyz.copy())
    return result


@dataclass(frozen=True)
class ReferenceRequest:
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


class ReferenceBatchSampler(Sampler):
    """训练仅合法非首帧×4，评测每轨迹只递归一次；末批保留。"""
    SCHEMA = 'seqtrack_reference.v32.teacher.v1'

    def __init__(self, lengths, batch_size=16, *, seed=42, training=True, source_sha256=''):
        self.lengths = tuple(map(int, lengths))
        self.batch_size, self.seed = int(batch_size), int(seed)
        self.training, self.source_sha256 = bool(training), str(source_sha256)
        self.epoch = 0
        if self.batch_size < 1 or any(value < 0 for value in self.lengths):
            raise ValueError('invalid reference sampler dimensions')

    @property
    def endpoint_count(self):
        return sum(max(0, n - 1) for n in self.lengths)

    @property
    def row_count(self):
        return self.endpoint_count * (4 if self.training else 1)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        if self.epoch < 0:
            raise ValueError('negative epoch')

    def __iter__(self):
        if self.training:
            requests = [ReferenceRequest(self.epoch, track, candidate, frame, frame + 1, frame)
                        for track, length in enumerate(self.lengths)
                        for frame in range(1, length) for candidate in range(4)]
            order = np.random.default_rng(stable_seed(self.seed, self.epoch, 'reference_teacher4')).permutation(len(requests))
            for start in range(0, len(order), self.batch_size):
                yield tuple(requests[int(i)] for i in order[start:start + self.batch_size])
        else:
            ready = deque((track, n) for track, n in enumerate(self.lengths) if n > 1)
            active = []
            while ready or active:
                while ready and len(active) < self.batch_size:
                    track, length = ready.popleft()
                    active.append(ReferenceRequest(self.epoch, track, 4, 1, length, 1))
                yield tuple(active)
                active = [ReferenceRequest(r.epoch, r.tracklet, 4, 1, r.window_end, r.frame + 1)
                          for r in active if not r.ends_window]

    def __len__(self):
        return math.ceil(self.row_count / self.batch_size) if self.training else sum(1 for _ in self)

    def state_dict(self):
        result = dict(schema=self.SCHEMA, epoch=self.epoch, lengths=list(self.lengths),
            batch_size=self.batch_size, seed=self.seed, training=self.training,
            source_sha256=self.source_sha256, rows=self.row_count,
            resampling='uniform_legal_nonfirst_candidate_on_invalid_history',
            nominal_endpoint_exposures=4 if self.training else 1)
        result['manifest_sha256'] = hashlib.sha256(json.dumps(result, sort_keys=True,
            separators=(',', ':')).encode()).hexdigest()
        return result


class ReferenceDataset(Dataset):
    """worker 只处理原 teacher 输入；没有模型/预测历史或 production 裁剪。"""
    def __init__(self, source, *, training, config=None):
        self.source, self.training = source, bool(training)
        self.config = reference_config(config)
        self.lengths = tuple(source.get_num_frames_tracklet(i) for i in range(source.get_num_tracklets()))
        self.endpoints = tuple((i, t) for i, n in enumerate(self.lengths) for t in range(1, n))
        digest = hashlib.sha256()
        for i, n in enumerate(self.lengths):
            key = source.get_tracklet_key(i) if hasattr(source, 'get_tracklet_key') else str(i)
            digest.update(json.dumps((key, n), separators=(',', ':')).encode())
            annotations = getattr(source, 'tracklet_anno_list', None)
            for t in range(n):
                endpoint = source.get_endpoint_key(i, t) if hasattr(source, 'get_endpoint_key') else str(t)
                if annotations is not None and hasattr(source, '_anno_timestamp'):
                    timestamp = float(source._anno_timestamp(annotations[i][t]))
                else:
                    timestamp = float(metadata(source, i, t)['timestamp'])
                digest.update(json.dumps((endpoint, timestamp), separators=(',', ':')).encode())
        self.source_sha256 = digest.hexdigest()

    def __len__(self):
        return len(self.endpoints) * (4 if self.training else 1)

    def _raw(self, request):
        history, valid = get_history_frame_ids_and_masks(request.frame, 3)
        ids = sorted(set(history + [request.frame]))
        frames = {i: normalize_frame(frame) for i, frame in zip(ids, self.source.get_frames(request.tracklet, ids))}
        first = frames.get(0)
        if first is None:
            first = normalize_frame(metadata(self.source, request.tracklet, 0))
        key = (self.source.get_tracklet_key(request.tracklet)
               if hasattr(self.source, 'get_tracklet_key') else str(request.tracklet))
        return dict(request=request, frames=frames, first_frame=first,
                    history_ids=tuple(history), valid_mask=valid, tracklet_key=str(key))

    def __getitem__(self, request):
        if not isinstance(request, ReferenceRequest):
            raise TypeError('ReferenceDataset requires ReferenceRequest')
        nominal = request
        rejected = []
        while True:
            row = self._raw(request)
            if not self.training:
                return row
            frames = row['frames']
            data = dict(first_frame=row['first_frame'], this_frame=frames[request.frame],
                prev_frames={str(-i - 1): frames[t] for i, t in enumerate(row['history_ids'])},
                candidate_id=request.branch, valid_mask=row['valid_mask'])
            # 原断言条件：三个历史槽（含重复首帧）中有至少一个槽含真实框内点。
            empty = sum(points_in_box(frames[t]['3d_bbox'], frames[t]['pc'].points[:3]).sum()
                        < self.config.limit_num_points_in_prev_box for t in row['history_ids'])
            if empty < self.config.empty_box_limit:
                row['prepared_reference'] = motion_processing_mf(data, self.config)
                row['nominal_request'] = nominal
                row['rejected_requests'] = rejected
                return row
            rejected.append(asdict(request))
            # 保留原 torch.randint 随机替换语义；仅分母按登记协议排除首帧。
            if len(rejected) >= 10000 or not self.endpoints:
                raise RuntimeError('reference invalid-history resampling exhausted 10000 attempts')
            index = int(torch.randint(0, len(self.endpoints) * 4, (1,)).item())
            track, frame = self.endpoints[index // 4]
            request = ReferenceRequest(nominal.epoch, track, index % 4, frame, frame + 1, frame)


def raw_collate(rows):
    return rows


def build_loaders(config, roles=('train', 'val'), sources=None):
    loaders = {}
    for role in roles:
        if sources is None:
            from datasets import get_raw_dataset
            dataset_role = ('dev' if str(option(config, 'dataset', '')).startswith('kitti') else 'test') if role == 'val' else role
            source = get_raw_dataset(config, dataset_role)
        else:
            source = sources[role]
        dataset = ReferenceDataset(source, training=role == 'train', config=config)
        sampler = ReferenceBatchSampler(dataset.lengths, option(config, 'batch_size', 16),
            seed=option(config, 'seed', 42), training=role == 'train', source_sha256=dataset.source_sha256)
        workers = int(option(config, 'workers', 4))
        kwargs = dict(num_workers=workers, collate_fn=raw_collate, pin_memory=False,
            generator=torch.Generator().manual_seed(stable_seed(option(config, 'seed', 42), role)))
        if workers:
            kwargs.update(prefetch_factor=2, persistent_workers=False)
        loaders[role] = DataLoader(dataset, batch_sampler=sampler, **kwargs)
    return loaders


def evaluation_input(row, predicted_boxes, config=None):
    """原 newest-first / 伪时间 / 重复采样推理；框尺寸仅从首帧读取。"""
    cfg = reference_config(config)
    request, frames = row['request'], row['frames']
    valid = row['valid_mask']
    refs = [deepcopy(predicted_boxes[t]) for t in row['history_ids']]
    anchor = deepcopy(refs[0])
    clouds = [geometry.generate_subwindow_with_aroundboxs(frames[t]['pc'], refs[i], anchor,
              scale=cfg.bb_scale, offset=cfg.bb_offset) for i, t in enumerate(row['history_ids'])]
    current = geometry.generate_subwindow_with_aroundboxs(frames[request.frame]['pc'], anchor,
              anchor, scale=cfg.bb_scale, offset=cfg.bb_offset)
    refs = [geometry.transform_box(box, anchor) for box in refs]
    sampled = [geometry.regularize_pc(pc.points.T, cfg.point_sample_size)[0] for pc in clouds]
    this = geometry.regularize_pc(current.points.T, cfg.point_sample_size, seed=1)[0]
    hint = [points_in_box(box, points.T, cfg.bb_scale).astype(float) for box, points in zip(refs, sampled)]
    if request.frame != 1:
        hint = [np.where(mask == 1, .8, .2) for mask in hint]
    stamps = generate_timestamp_prev_list(valid, cfg.point_sample_size)
    points = [np.concatenate((xyz, time, mask[:, None]), -1) for xyz, time, mask in zip(sampled, stamps, hint)]
    points.append(np.concatenate((this, np.full((cfg.point_sample_size, 1), .1),
                                 np.full((cfg.point_sample_size, 1), .5)), -1))
    bc = [geometry.get_point_to_box_distance(xyz, box) for xyz, box in zip(sampled, refs)]
    return dict(points=np.concatenate(points).astype(np.float32),
        ref_boxs=np.stack([box_array(box) for box in refs]).astype(np.float32),
        valid_mask=np.asarray(valid, dtype=np.int64),
        bbox_size=np.asarray(row['first_frame']['3d_bbox'].wlh, dtype=np.float32),
        candidate_bc=np.concatenate(bc + [np.zeros_like(bc[0])]).astype(np.float32),
        reference_anchor=box_array(anchor))


class BatchBuilder:
    """teacher 训练无递归状态；评测只提交预测框，不回写历史或 GT。"""
    def __init__(self, config=None):
        self.config = config
        self.states, self._pending, self.exposure_records = {}, None, []

    def reset(self):
        self.states.clear()
        self._pending = None
        self.exposure_records.clear()

    def prepare(self, rows):
        if self._pending is not None or not rows:
            raise RuntimeError('reference requires a nonempty single pending transaction')
        prepared = []
        for row in rows:
            r = row['request']
            if r.branch != 4:
                sample = dict(row['prepared_reference'])
            else:
                if r.starts_window:
                    if r.state_key in self.states:
                        raise RuntimeError('duplicate reference rollout initialization')
                    self.states[r.state_key] = {0: deepcopy(row['first_frame']['3d_bbox'])}
                state = self.states.get(r.state_key)
                if state is None or max(state) != r.frame - 1:
                    raise RuntimeError('reference evaluation must consume accepted predecessor')
                sample = evaluation_input(row, state, self.config)
            first_box = row['first_frame']['3d_bbox']
            current = row['frames'][r.frame]['3d_bbox']
            # 登记的共同输入合同：train/eval 网络尺寸均来自合法首帧。
            # 原 teacher 标签/几何处理与 loss 目标不改，保留原连续 dtype。
            sample['bbox_size'] = np.asarray(first_box.wlh,
                dtype=np.asarray(sample['bbox_size']).dtype).copy()
            sample.update(target_box=box_array(current),
                target_box_size=np.asarray(current.wlh, dtype=np.float64)[[1, 0, 2]],
                box_size=np.asarray(first_box.wlh, dtype=np.float64)[[1, 0, 2]])
            prepared.append(sample)
        batch = {key: torch.from_numpy(np.stack([item[key] for item in prepared])) for key in prepared[0]}
        # 原 np.astype('int') 在 Windows 为 int32；CE 的离散标签统一为 int64。
        # 不改连续输入/标签 dtype，Linux 原 int64 路径数值不变。
        for key in ('seg_label', 'motion_state_label', 'valid_mask'):
            if key in batch:
                batch[key] = batch[key].long()
        batch['reference_empty'] = batch['points'][..., :3].sum((1, 2)) == 0
        self._pending = rows
        return batch

    def acquire(self, batch, prior=None, *, training=True):
        if self._pending is None:
            raise RuntimeError('prepare reference batch first')
        return batch

    def commit(self, output, batch, *, diagnostics=False):
        if self._pending is None:
            raise RuntimeError('duplicate reference commit')
        predictions = output.accepted_box.detach().cpu().numpy()
        records = []
        for index, row in enumerate(self._pending):
            r = row['request']
            if r.branch != 4:
                self.exposure_records.append(dict(nominal=asdict(row['nominal_request']),
                    actual=asdict(r), rejected=row['rejected_requests']))
            else:
                first = row['first_frame']['3d_bbox']
                value = predictions[index]
                if not np.isfinite(value).all():
                    raise FloatingPointError('nonfinite SeqTrack reference prediction')
                self.states[r.state_key][r.frame] = Box(value[:3].copy(), first.wlh.copy(),
                    Quaternion(axis=[0, 0, 1], radians=float(value[3])))
                # 只需要最新三框；缺失历史重复首帧只发生前三帧。
                for old in list(self.states[r.state_key]):
                    if old < r.frame - 2:
                        del self.states[r.state_key][old]
                if r.ends_window:
                    del self.states[r.state_key]
            records.append({})
        self._pending = None
        return records if diagnostics else None

    def exposure_summary(self):
        actual = Counter((r['actual']['tracklet'], r['actual']['frame'], r['actual']['branch'])
                         for r in self.exposure_records)
        nominal = Counter((r['nominal']['tracklet'], r['nominal']['frame'], r['nominal']['branch'])
                          for r in self.exposure_records)
        encoded = json.dumps(self.exposure_records, sort_keys=True, separators=(',', ':')).encode()
        return dict(schema='seqtrack_reference.v32.exposure.v1', nominal_rows=len(self.exposure_records),
            actual_rows=sum(actual.values()), nominal_unique_rows=len(nominal), actual_unique_rows=len(actual),
            resampled_rows=sum(bool(r['rejected']) for r in self.exposure_records),
            rejected_attempts=sum(len(r['rejected']) for r in self.exposure_records),
            nominal_candidate_counts=dict(Counter(str(r['nominal']['branch']) for r in self.exposure_records)),
            actual_candidate_counts=dict(Counter(str(r['actual']['branch']) for r in self.exposure_records)),
            actual_exposure_counts=[dict(tracklet=k[0], frame=k[1], candidate=k[2], count=v)
                                    for k, v in sorted(actual.items())],
            records_sha256=hashlib.sha256(encoded).hexdigest())
