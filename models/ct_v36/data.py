"""v36 主进程事务：输出框、测量事件和身份记忆各自提交，worker 只读原帧。"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from models.ct_v31.data import (box_array, box_size, box_corners, inside_box,
                               point_arrays, option, metadata, raw_dataset_factory)
from .contracts import MeasurementEvent
from .memory import RawIdentityMemory
from .sampler import EndpointRequest, ReadyQueueBatchSampler, stable_seed, endpoint_seed


def geometry_summary(xyz, ids, box, size):
    """唯一授权点数/框内点数及框自身坐标 mean/std；无 GT、score 或 teacher 标记。"""
    xyz, ids = np.asarray(xyz, np.float64), np.asarray(ids, np.int64)
    _, index = np.unique(ids, return_index=True)
    xyz = xyz[np.sort(index)]
    chosen = xyz[inside_box(xyz, box, size)]
    result = np.zeros(8, np.float32)
    result[:2] = np.log1p([len(xyz), len(chosen)]) / np.log1p(1280.)
    if len(chosen):
        local = chosen - box[:3]
        c, s = np.cos(box[3]), np.sin(box[3])
        local[:, :2] = local[:, :2] @ np.asarray([[c, -s], [s, c]])
        local /= np.maximum(size, 1e-6)
        result[2:5], result[5:8] = local.mean(0), local.std(0)
    return result


def _empty_memory():
    return dict(memory_points=np.zeros((36, 5), np.float32), memory_valid=np.zeros(36, bool),
                memory_ids=np.full(36, -1, np.int64), memory_metadata=np.zeros((36, 8), np.float32))


class RawEndpointDataset(Dataset):
    """注释索引只用于 loss 标签；历史事件不触发点云重载或额外网络 forward。"""
    def __init__(self, source, *, memory_enabled=True):
        self.source, self.memory_enabled = source, bool(memory_enabled)
        self.lengths = tuple(source.get_num_frames_tracklet(i) for i in range(source.get_num_tracklets()))
        self.annotations = []
        digest = hashlib.sha256()
        annotation_digest = hashlib.sha256(b'ct_seqtrack.v36.annotation_clock.v1')
        for tracklet, length in enumerate(self.lengths):
            key = source.get_tracklet_key(tracklet) if hasattr(source, 'get_tracklet_key') else str(tracklet)
            records = [metadata(source, tracklet, i) for i in range(length)]
            boxes = np.stack([box_array(x['3d_bbox']) for x in records]) if records else np.empty((0, 4))
            sizes = np.stack([box_size(x['3d_bbox'], x) for x in records]) if records else np.empty((0, 3))
            times = np.asarray([float(x['timestamp']) for x in records], np.float64)
            effective = np.asarray([float(x.get('_ct_effective_timestamp', x['timestamp'])) for x in records], np.float64)
            if length > 1 and ((np.diff(times) <= 0).any() or (np.diff(effective) <= 0).any()):
                raise ValueError('v36 timestamps must strictly increase')
            self.annotations.append(dict(boxes=boxes, times=times, effective_times=effective))
            # 与 v31/v35 保持完全相同的 physical dataset 身份，支持严格跨版本对照。
            digest.update(json.dumps((key, length), separators=(',', ':')).encode())
            annotation_digest.update(json.dumps((key, length), separators=(',', ':')).encode())
            source_annotations = getattr(source, 'tracklet_anno_list', None)
            for frame in range(length):
                endpoint = source.get_endpoint_key(tracklet, frame) if hasattr(source, 'get_endpoint_key') else str(frame)
                timestamp = (float(source._anno_timestamp(source_annotations[tracklet][frame]))
                    if source_annotations is not None and hasattr(source, '_anno_timestamp') else float(times[frame]))
                digest.update(json.dumps((endpoint, timestamp), separators=(',', ':')).encode())
                annotation_digest.update(json.dumps((endpoint, boxes[frame].tolist(), sizes[frame].tolist(),
                    float(times[frame]), float(effective[frame])), separators=(',', ':')).encode())
        self.source_sha256 = digest.hexdigest()
        self.annotation_sha256 = annotation_digest.hexdigest()

    def __len__(self):
        return sum(max(n-1, 0) for n in self.lengths)

    def __getitem__(self, request):
        if not isinstance(request, EndpointRequest):
            raise TypeError('v36 requires an EndpointRequest')
        r = request
        history_ids = tuple(max(0, r.frame - 3 + i) for i in range(3))
        ids = sorted(set(history_ids + (r.frame,)))
        if r.starts_window and self.memory_enabled:
            ids = sorted(set(ids + [0]))
        frames = dict(zip(ids, self.source.get_frames(r.tracklet, ids)))
        first = frames[0] if 0 in frames else metadata(self.source, r.tracklet, 0)
        key = self.source.get_tracklet_key(r.tracklet) if hasattr(self.source, 'get_tracklet_key') else str(r.tracklet)
        return dict(request=r, frames=frames, first_frame=first, history_ids=history_ids,
                    tracklet_key=str(key), annotations=self.annotations[r.tracklet])


def raw_collate(rows):
    return rows


def build_loaders(config, roles=('train', 'val'), sources=None):
    loaders = {}
    for role in roles:
        dataset_role = ('dev' if str(option(config, 'dataset', '')).startswith('kitti') else 'test') if role == 'val' else role
        source = sources[role] if sources is not None else raw_dataset_factory(config, dataset_role)
        raw = RawEndpointDataset(source, memory_enabled=option(config, 'v36_arm', 'A') != 'C')
        sampler = ReadyQueueBatchSampler(raw.lengths, int(option(config, 'batch_size', 16)),
            seed=int(option(config, 'seed', 42)), training=role == 'train',
            short_window=int(option(config, 'v31_short_window', 4)),
            long_window=int(option(config, 'v31_long_window', 8)),
            curriculum_epochs=int(option(config, 'v31_curriculum_epochs', 10)),
            source_sha256=raw.source_sha256, reserve_windows=int(option(config, 'v32_reserve_windows', 112)),
            annotation_sha256=raw.annotation_sha256,
            seed_translation=float(option(config, 'v32_seed_translation', .3)),
            seed_yaw_degrees=float(option(config, 'v32_seed_yaw_degrees', 1.5)),
            continuous=option(config, 'v36_arm', 'A') != 'F')
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
    effective_times: dict
    next_frame: int
    size: np.ndarray
    memory: object
    events: list
    transitions: dict
    seed_offset: np.ndarray
    initial_frames: set = field(default_factory=set)
    last_memory_time: float | None = None
    previous_quality: float = 0.
    previous_innovation: float = 0.
    segment: int = 0
    last_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float64))
    velocity_valid: bool = False
    velocity_time: float = 0.


class BatchBuilder:
    """唯一状态所有者；GT 标签从 annotations 单独生成，不保存进事件。"""
    def __init__(self, config):
        self.config = config
        self.memory_enabled = option(config, 'v36_arm', 'A') != 'C'
        self.states, self._pending = {}, None

    def reset(self):
        self.states.clear()
        self._pending = None

    def _crop_indices(self, frame, box, size, row):
        xyz, ids = point_arrays(frame)
        selected = np.flatnonzero(inside_box(xyz, box, size,
            scale=float(option(self.config, 'bb_scale', 1.25)), offset=float(option(self.config, 'bb_offset', 2.))))
        n = int(option(self.config, 'point_sample_size', 1024))
        return xyz, ids, selected, n

    def _sample(self, selected, n, row, source_frame):
        if len(selected) > n:
            rng = np.random.default_rng(endpoint_seed(option(self.config, 'seed', 42), row['request'],
                row['tracklet_key'], source_frame, 'b0_points'))
            return np.sort(rng.choice(selected, n, replace=False))
        return selected

    def _new_state(self, row):
        r, frames, annotations = row['request'], row['frames'], row['annotations']
        seed_ids = list(range(max(0, r.frame-3), r.frame))
        boxes = {i: box_array(frames[i]['3d_bbox']) for i in seed_ids}
        size = box_size(row['first_frame']['3d_bbox'], row['first_frame'])
        offset = np.zeros(4, np.float64)
        perturbed = r.branch in (1, 2, 3) and r.frame != 1
        if perturbed:
            rng = np.random.default_rng(endpoint_seed(option(self.config, 'seed', 42), r,
                row['tracklet_key'], r.frame-1, 'seed_pose'))
            xy = rng.uniform(-float(option(self.config, 'v32_seed_translation', .3)),
                             float(option(self.config, 'v32_seed_translation', .3)), 2)
            yaw = boxes[r.frame-1][3]
            c, s = np.cos(yaw), np.sin(yaw)
            offset[:2] = np.asarray([[c, -s], [s, c]]) @ xy
            limit = np.deg2rad(float(option(self.config, 'v32_seed_yaw_degrees', 1.5)))
            offset[3] = rng.uniform(-limit, limit)
            for box in boxes.values():
                box[:3] += offset[:3]
                box[3] = np.arctan2(np.sin(box[3]+offset[3]), np.cos(box[3]+offset[3]))
        events = []
        for j, i in enumerate(seed_ids):
            xyz, ids, selected, n = self._crop_indices(frames[i], boxes[i], size, row)
            selected = self._sample(selected, n, row, i)
            events.append(MeasurementEvent(boxes[i].copy(), i, float(annotations['times'][i]),
                float(annotations['effective_times'][i]), geometry_summary(xyz[selected], ids[selected], boxes[i], size),
                initial=i == 0 and not perturbed, edge_valid=j > 0))
        memory = None
        if self.memory_enabled:
            memory = RawIdentityMemory(size)
            first = row['first_frame']
            xyz, ids = point_arrays(first)
            first_box = box_array(first['3d_bbox'])
            memory.initialize(xyz, ids, inside_box(xyz, first_box, size), first_box,
                              float(annotations['effective_times'][0]))
        state = AcceptedState(boxes, {i:float(annotations['times'][i]) for i in seed_ids},
            {i:float(annotations['effective_times'][i]) for i in seed_ids}, r.frame, size, memory,
            events, {i:True for i in seed_ids}, offset)
        state.initial_frames={i for i in seed_ids if i==0 and not perturbed}
        state.last_memory_time=float(annotations['effective_times'][0]) if memory is not None else None
        if len(events) >= 2:
            state.last_velocity = (events[-1].box[:3]-events[-2].box[:3]) / (events[-1].effective_time-events[-2].effective_time)
            state.velocity_valid, state.velocity_time = True, events[-1].effective_time
        return state

    def prepare(self, rows):
        if self._pending is not None:
            raise RuntimeError('previous v36 transaction has not been committed')
        if not rows:
            raise ValueError('empty raw batch')
        keys = [x['request'].state_key for x in rows]
        if len(set(keys)) != len(keys):
            raise ValueError('one state may appear only once per batch')
        prepared, contexts = [], []
        for row in rows:
            r = row['request']
            if r.starts_window:
                if r.state_key in self.states:
                    raise RuntimeError('duplicate state initialization')
                self.states[r.state_key] = self._new_state(row)
            if r.state_key not in self.states:
                raise RuntimeError('missing predecessor; no implicit GT reseeding')
            state = self.states[r.state_key]
            if state.next_frame != r.frame:
                raise RuntimeError('out-of-order accepted state')
            batch, context = self._prepare_row(row, state)
            prepared.append(batch); contexts.append(context)
        self._pending = contexts
        return {key:torch.from_numpy(np.stack([x[key] for x in prepared])) for key in prepared[0]}

    def _prepare_row(self, row, state):
        r, frames, annotation = row['request'], row['frames'], row['annotations']
        now = float(annotation['times'][r.frame]); effective_now = float(annotation['effective_times'][r.frame])
        anchor = state.boxes[r.frame-1].copy()
        history_ids = row['history_ids']
        history_valid = np.asarray([r.frame-3+i >= 0 for i in range(3)], bool)
        history = np.stack([state.boxes[i] for i in history_ids])
        history_local = history.copy(); history_local[:, :3] -= anchor[:3]
        physical_times = annotation['times'][list(history_ids)]-now
        effective_times = annotation['effective_times'][list(history_ids)]-effective_now
        if not (physical_times < 0).all() or not (effective_times < 0).all():
            raise ValueError('history must be strictly earlier than current endpoint')
        points, valid, raw_ids, seg, bc, truth_boxes, history_geometry = [], [], [], [], [], [], []
        for slot, frame_id in enumerate(history_ids + (r.frame,)):
            frame = frames[frame_id]
            crop_box = history[slot] if slot < 3 else anchor
            xyz, ids, selected, n = self._crop_indices(frame, crop_box, state.size, row)
            if slot < 3 and not history_valid[slot]:
                selected = selected[:0]
            if slot == 3:
                b0_ids, b0_xyz = ids[selected].copy(), xyz[selected].copy()
            selected = self._sample(selected, n, row, frame_id)
            count = len(selected)
            point = np.zeros((n, 5), np.float32)
            point[:count, :3] = xyz[selected]-anchor[:3]
            point[:count, 3] = annotation['times'][frame_id]-now
            # 所有非首框历史使用同一种可观测 hint，不泄露教师/预测来源。
            if slot < 3:
                hint = inside_box(xyz[selected], crop_box, state.size).astype(np.float32)
                point[:count, 4] = hint if frame_id in state.initial_frames else .2+.6*hint
                history_geometry.append(geometry_summary(xyz[selected], ids[selected], crop_box, state.size))
            else:
                point[:count, 4] = .5
            mask = np.arange(n) < count
            point_ids = np.full(n, -1, np.int64); point_ids[:count] = ids[selected]
            target = box_array(frame['3d_bbox']); target_size = box_size(frame['3d_bbox'], frame)
            label = np.full(n, -1, np.int64); label[:count] = inside_box(xyz[selected], target, target_size)
            distances = np.zeros((n, 9), np.float32)
            centers_corners = np.vstack((target[:3], box_corners(target, target_size)))
            if count:
                distances[:count] = np.linalg.norm(xyz[selected,None,:]-centers_corners[None,:,:], axis=-1)
            target[:3] -= anchor[:3]
            points.append(point); valid.append(mask); raw_ids.append(point_ids)
            seg.append(label); bc.append(distances); truth_boxes.append(target.astype(np.float32))
        latest = state.events[-1]
        fallback = latest.box.copy()
        if state.velocity_valid:
            fallback[:3] += state.last_velocity*(effective_now-latest.effective_time)
        fallback_local = fallback.copy(); fallback_local[:3] -= anchor[:3]
        event_boxes = np.zeros((3,4), np.float64); event_times = np.zeros(3, np.float64)
        event_valid = np.zeros(3, bool); event_edge_valid = np.zeros(3, bool)
        event_geometry = np.zeros((3,8), np.float32); event_frame_ids = np.full(3, -1, np.int64)
        event_source = np.zeros(3, np.int64); event_segment = np.zeros(3, np.int64); event_initial=np.zeros(3,bool)
        for slot, event in zip(range(3-len(state.events), 3), state.events):
            event_boxes[slot],event_times[slot],event_valid[slot] = event.box,event.effective_time,True
            event_edge_valid[slot] = event.edge_valid and slot > 3-len(state.events)
            event_geometry[slot],event_frame_ids[slot] = event.geometry,event.frame
            event_source[slot],event_segment[slot],event_initial[slot] = event.source,event.segment,event.initial
        memory = state.memory.export(effective_now) if state.memory is not None else _empty_memory()
        current_gt = box_array(frames[r.frame]['3d_bbox'])
        previous_gt = annotation['boxes'][r.frame-1].copy()
        result = dict(points=np.stack(points), point_valid=np.stack(valid), point_ids=np.stack(raw_ids),
            history_boxes=history_local.astype(np.float32), history_valid=history_valid,
            history_pair_valid=history_valid[:-1]&history_valid[1:]&np.asarray([state.transitions.get(i,False) for i in history_ids[1:]]),
            history_times=effective_times.astype(np.float32), current_dt=np.float32(-effective_times[-1]),
            frame_times=np.r_[physical_times,0.].astype(np.float32), box_size=state.size.astype(np.float32),
            anchor_box=anchor, fallback_box=fallback_local.astype(np.float32),
            target_box=np.asarray(truth_boxes[-1],np.float32), target_box_size=box_size(frames[r.frame]['3d_bbox'],frames[r.frame]).astype(np.float32),
            history_target_boxes=np.asarray(truth_boxes[:3],np.float32), segmentation_labels=np.stack(seg),bc_targets=np.stack(bc),
            physical_displacement=(current_gt[:2]-previous_gt[:2]).astype(np.float32), physical_valid=np.bool_(True),
            branch_id=np.int64(r.branch), current_frame=np.int64(r.frame), current_time=np.float64(effective_now),
            current_sensor_time=np.float64(now), history_geometry=np.stack(history_geometry),
            history_is_initial=history_valid&np.asarray([i in state.initial_frames for i in history_ids]),
            trusted_velocity=(state.last_velocity if state.velocity_valid else np.zeros(3)).astype(np.float32),
            trusted_velocity_valid=np.bool_(state.velocity_valid),
            previous_quality=np.float32(state.previous_quality),previous_innovation=np.float32(state.previous_innovation),
            weak_age=np.float32(max(0.,effective_now-latest.effective_time)),supported_age=np.float32(max(0.,effective_now-latest.effective_time)),
            diagnostic_measurement_age=np.float64(max(0.,effective_now-latest.effective_time)),
            diagnostic_memory_age=np.float64(0. if state.last_memory_time is None else max(0.,effective_now-state.last_memory_time)),
            diagnostic_memory_age_valid=np.bool_(state.last_memory_time is not None),
            event_boxes=event_boxes,event_times=event_times,event_valid=event_valid,event_edge_valid=event_edge_valid,
            event_segment=event_segment,event_source=event_source,event_initial=event_initial,event_frame_ids=event_frame_ids,
            event_geometry=event_geometry,event_target_box=annotation['boxes'][latest.frame].copy(),
            current_target_world=current_gt,previous_target_world=previous_gt,**memory)
        context = dict(row=row,state=state,anchor=anchor,timestamp=now,effective_time=effective_now,
                       b0_ids=b0_ids,b0_xyz=b0_xyz,current_gt=current_gt,recovery=fallback)
        return result, context

    def acquire(self, batch, prior, *, training=True):
        from .acquisition import acquire_extension, build_dual_support, support_membership
        if self._pending is None:
            raise RuntimeError('acquisition requires a pending prepare')
        boxes=prior.world_box.detach().cpu().numpy(); ranges=prior.range_xy.detach().cpu().numpy()
        axes=prior.axes.detach().cpu().numpy(); origins=prior.origin_frame.detach().cpu().numpy()
        values=[]
        for index,context in enumerate(self._pending):
            row,state=context['row'],context['state']; r=row['request']; current=row['frames'][r.frame]
            xyz,ids=point_arrays(current)
            geometry=dict(b0_box=context['anchor'],box_size=state.size,prior_box=boxes[index],
                range_xy=ranges[index],axes=axes[index],crop_scale=float(option(self.config,'bb_scale',1.25)),
                crop_offset=float(option(self.config,'bb_offset',2.)))
            if self.memory_enabled:
                extra=acquire_extension(xyz,ids,b0_raw_ids=context['b0_ids'],anchor=context['anchor'][:3],
                    seed=endpoint_seed(option(self.config,'seed',42),r,row['tracklet_key'],r.frame,'extension_pool'),**geometry)
            else:
                extra=dict(extension_points=np.zeros((768,5),np.float32),extension_ids=np.full(768,-1,np.int64),
                    extension_valid=np.zeros(768,bool),extension_partition=np.full(768,-1,np.int64),
                    extension_membership=np.zeros((768,2),bool))
            # 原帧标签和覆盖在完成真实获取之后生成，不改变采样。
            truth=inside_box(xyz,context['current_gt'],box_size(current['3d_bbox'],current))
            if self.memory_enabled:
                actual_support=build_dual_support(**geometry)
                actual_members=support_membership(xyz,actual_support,context['b0_ids'],ids)
                maximum_support=build_dual_support(**dict(geometry,range_xy=np.asarray([12.,8.])))
                maximum_members=support_membership(xyz,maximum_support,context['b0_ids'],ids)
                actual_union=actual_members[0]|actual_members[1]
                maximum_union=maximum_members[0]|maximum_members[1]
            else:
                actual_members=(np.zeros(len(ids),bool),np.zeros(len(ids),bool))
                actual_union=maximum_union=np.zeros(len(ids),bool)
            lookup={int(raw_id):bool(label) for raw_id,label in zip(ids,truth)}
            labels=np.full(768,-1,np.int64); mask=extra['extension_valid']
            labels[mask]=[lookup[int(i)] for i in extra['extension_ids'][mask]]
            origin=int(origins[index])
            if origin < 0 or origin >= r.frame:
                raise ValueError('range origin must reference an earlier annotation')
            extra.update(extension_labels=labels,prior_origin_target_world=row['annotations']['boxes'][origin].copy(),
                diagnostic_search_enabled=np.bool_(self.memory_enabled),
                diagnostic_raw_point_count=np.int64(len(ids)),diagnostic_target_count=np.int64(truth.sum()),
                diagnostic_crop_target_count=np.int64((truth&np.isin(ids,context['b0_ids'])).sum()),
                diagnostic_novel_target_count=np.int64((truth&~np.isin(ids,context['b0_ids'])).sum()),
                diagnostic_reachable_count=np.int64((truth&maximum_union).sum()),
                diagnostic_search_point_count=np.int64(actual_union.sum()),
                diagnostic_search_target_count=np.int64((truth&actual_union).sum()),
                diagnostic_search_point_count_by_partition=np.asarray([m.sum() for m in actual_members],np.int64),
                diagnostic_search_target_count_by_partition=np.asarray([(truth&m).sum() for m in actual_members],np.int64),
                diagnostic_range_clipped=np.asarray(ranges[index]>[12.,8.],bool),
                diagnostic_acquired_count=np.int64((labels[mask]==1).sum()),
                diagnostic_sampled_target_count=np.int64((batch['segmentation_labels'][index,-1]==1).sum().item()),
                diagnostic_crop_point_count=np.int64(len(context['b0_ids'])),
                diagnostic_sampled_point_count=np.int64(batch['point_valid'][index,-1].sum().item()),
                diagnostic_gt_xy_displacement=np.float32(np.linalg.norm(context['current_gt'][:2]-row['annotations']['boxes'][r.frame-1,:2])))
            values.append(extra)
        device=batch['points'].device
        for key in values[0]:
            batch[key]=torch.from_numpy(np.stack([value[key] for value in values])).to(device)
        return batch

    def _append_event(self,state,event,prior_world,range_xy,axes):
        previous=state.events[-1]
        gap=event.effective_time-previous.effective_time
        if gap <= 0 or event.frame <= previous.frame:
            raise ValueError('measurement events must strictly advance time/frame')
        if state.velocity_valid:
            aligned=(event.box[:2]-prior_world[:2])@axes
            c,s=np.cos(prior_world[3]),np.sin(prior_world[3])
            rotation=np.asarray([[c,-s],[s,c]])
            half_extent=np.abs(axes.T@rotation)@(state.size[:2]*.5)
            # 运动入边使用未截断的位置范围加实际旋转物体半尺寸；不是搜索上限。
            tolerance=range_xy+half_extent
            edge=bool(np.max(np.abs(aligned)/np.maximum(tolerance,1e-6))<=1.)
        else:
            # 没有已有运动时，同一新 segment 的任意正事件间隔都能启动。
            # 不用尚未建立的静止 prior 拒绝首次运动，也不要求相邻 sensor 帧。
            edge=True
        if not edge:
            state.segment+=1
        event.segment,event.edge_valid=state.segment,edge
        if edge:
            state.last_velocity=(event.box[:3]-previous.box[:3])/gap
            state.velocity_valid,state.velocity_time=True,event.effective_time
        else:
            state.velocity_valid=False
        state.events.append(event);state.events[:]=state.events[-3:]
        return edge

    def commit(self,output,batch,*,diagnostics=False):
        if self._pending is None:
            raise RuntimeError('no pending v36 transaction (duplicate commit)')
        def cpu(value):
            return value.detach().cpu().numpy()
        boxes=cpu(output.accepted_box);candidate_boxes=cpu(output.candidates.boxes)
        selected=cpu(output.decision.output_index); measured=cpu(output.decision.measurement_index)
        utilities=cpu(output.verification.utility);scores=cpu(output.verification.measurement_score)
        authorized=cpu(output.verification.authorized);support=cpu(output.verification.support_mask)
        ids=cpu(output.verification.support_ids);local_xyz=cpu(output.verification.support_xyz)
        probabilities=cpu(output.verification.support_probabilities)
        candidate_valid=cpu(output.candidates.valid); prior_world=cpu(output.prior.world_box)
        range_xy=cpu(output.prior.range_xy);axes=cpu(output.prior.axes)
        declared_memory=cpu(output.decision.memory_eligible)
        actions=[]
        for index,context in enumerate(self._pending):
            oi,mi=int(selected[index]),int(measured[index])
            if oi<0 or oi>=4 or not candidate_valid[index,oi] or not np.isfinite(boxes[index]).all():
                raise ValueError('invalid output commit')
            if not np.allclose(boxes[index],candidate_boxes[index,oi],atol=1e-5,rtol=0):
                raise ValueError('accepted box does not match selected candidate')
            if mi not in (-1,0,2,3):
                raise ValueError('propagation cannot be a measurement')
            if mi>=0 and (not candidate_valid[index,mi] or scores[index,mi]<.5 or not support[index,mi].any()
                    or not np.isfinite(candidate_boxes[index,mi]).all()):
                raise ValueError('ineligible independent measurement')
            count=(len(np.unique(ids[index,support[index,mi]&authorized[index,mi]&(ids[index]>=0)]))
                   if mi>=0 else 0)
            if mi>=0 and count<1:
                raise ValueError('measurement requires unique authorized foreground support')
            if bool(declared_memory[index]) != (self.memory_enabled and mi>=0 and count>=3):
                raise ValueError('memory eligibility disagrees with unique measurement support')
            actions.append((oi,mi))
        records=[]
        for index,(context,(oi,mi)) in enumerate(zip(self._pending,actions)):
            row,state=context['row'],context['state'];r=row['request'];anchor=context['anchor']
            world=boxes[index].astype(np.float64,copy=True);world[:3]+=anchor[:3]
            innovation=float(np.linalg.norm(world[:2]-prior_world[index,:2]))
            wrote=False;fg_ids=np.empty(0,np.int64);edge=False;measurement_world=None
            if mi>=0:
                measurement_world=candidate_boxes[index,mi].astype(np.float64,copy=True)
                measurement_world[:3]+=anchor[:3]
                auth=authorized[index,mi]& (ids[index]>=0)
                auth_ids=ids[index,auth];auth_xyz=local_xyz[index,auth].astype(np.float64)+anchor[:3]
                geometry=geometry_summary(auth_xyz,auth_ids,measurement_world,state.size)
                event=MeasurementEvent(measurement_world,r.frame,context['timestamp'],context['effective_time'],geometry,
                    source=int(mi>=2))
                edge=self._append_event(state,event,prior_world[index],range_xy[index],axes[index])
                fg_mask=support[index,mi]&auth
                fg_ids=np.unique(ids[index,fg_mask])
                if state.memory is not None and len(fg_ids)>=3:
                    prob=probabilities[index,auth].copy()
                    # 只允许最终框内授权 FG；授权 BG 可按原容量保留。
                    keep=(prob<.5)|inside_box(auth_xyz,measurement_world,state.size)
                    wrote=state.memory.commit_measurement(auth_xyz[keep],auth_ids[keep],prob[keep],measurement_world,
                        context['effective_time'],float(scores[index,mi]))
                    if wrote:
                        state.last_memory_time=context['effective_time']
            # 先以旧快照完成独立测量入边/记忆提交，再提交本帧输出历史。
            state.boxes[r.frame]=world;state.times[r.frame]=context['timestamp']
            state.effective_times[r.frame]=context['effective_time'];state.next_frame+=1
            state.transitions[r.frame]=innovation<=max(.5*np.hypot(*state.size[:2]),1e-3)
            state.previous_innovation=innovation;state.previous_quality=float(utilities[index,oi])
            if diagnostics:
                raw,raw_ids=point_arrays(row['frames'][r.frame]);lookup={int(x):j for j,x in enumerate(raw_ids)}
                written=(state.memory.recent[-1]['ids'][state.memory.recent[-1]['fg']]
                         if wrote else np.empty(0,np.int64))
                truth=inside_box(raw[[lookup[int(x)] for x in written]],context['current_gt'],
                    box_size(row['frames'][r.frame]['3d_bbox'],row['frames'][r.frame]))
                records.append(dict(measurement_accepted=mi>=0,measurement_index=mi,output_index=oi,
                    output_measurement_disagreement=mi>=0 and mi!=oi,
                    measurement_world=None if measurement_world is None else measurement_world.tolist(),
                    memory_write=bool(wrote),memory_write_eligible=bool(self.memory_enabled and mi>=0 and len(fg_ids)>=3),
                    memory_world=state.memory.recent[-1]['box'].tolist() if wrote else None,
                    memory_source_index=mi if wrote else -1,memory_written_fg_ids=written.tolist(),
                    memory_fg_count=int(len(written)),memory_true_fg_count=int(truth.sum()),
                    memory_wrong_write=bool(wrote and len(truth)>0 and truth.mean()<.5),
                    measurement_fg_ids=fg_ids.tolist(),measurement_edge_valid=bool(edge),
                    measurement_age=float(context['effective_time']-state.events[-1].effective_time),
                    memory_age=(None if state.last_memory_time is None else float(context['effective_time']-state.last_memory_time)),
                    memory_age_valid=state.last_memory_time is not None,
                    event_frame_ids=[int(e.frame) for e in state.events],
                    event_segments=[int(e.segment) for e in state.events],
                    event_edge_valid=[bool(e.edge_valid) for e in state.events],
                    event_effective_times=[float(e.effective_time) for e in state.events],
                    weak_age=float(context['effective_time']-state.events[-1].effective_time),
                    supported_age=float(context['effective_time']-state.events[-1].effective_time),
                    strong=bool(len(fg_ids)>=3),supported=mi>=0,geometric_fg_count=int(len(fg_ids))))
            for old in list(state.boxes):
                if old<r.frame-2:
                    for values in (state.boxes,state.times,state.effective_times,state.transitions):
                        values.pop(old,None)
            if r.ends_window:
                self.states.pop(r.state_key)
        self._pending=None
        return records
