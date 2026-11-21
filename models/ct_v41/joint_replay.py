"""联合搜索的训练前缀回放。库只保留刚结束一轮的自然辅助轨迹。"""
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from .data import EndpointRequest, stable_seed
from .search import inside_box


CATEGORIES = ('reappearance', 'crop_miss', 'visible_drift', 'competing_car')
CAPACITY_PER_CATEGORY = 128


@dataclass(frozen=True)
class ReplayRequest(EndpointRequest):
    original_key: tuple = ()

    @property
    def state_key(self):
        # 替代项仍占原单步窗口事务槽；同批出现的正常供体端点可独立提交。
        return self.original_key


def has_competitor(points, cars):
    """仅用于训练样本选取。有三个被实采到的它车点才算竞争。"""
    if len(points) < 3 or len(cars) == 0:
        return False
    low, high = points.min(axis=0), points.max(axis=0)
    for car in cars:
        radius = np.linalg.norm(car[4:]) / 2.
        if np.any(car[:3] + radius < low) or np.any(car[:3] - radius > high):
            continue
        if np.count_nonzero(inside_box(points, car[:4], car[4:])) >= 3:
            return True
    return False


def observe_joint_prefix(previous, epoch, context, iou, sampled):
    """自然辅助前缀的统一分类。GT 只决定是否留样，模型输入已经构造完成。"""
    row = context['row']
    request = row['request']
    tracklet, frame = request.tracklet, request.frame
    if request.branch != 5 or request.epoch != epoch or context['state'].next_frame != frame:
        raise ValueError('joint replay needs the causal auxiliary prefix')
    raw = int(sampled) if sampled else int(context['joint_raw_target_count'])
    union = int(sampled) if sampled else int(context['joint_union_target_count'])
    prev = previous.get(tracklet)
    if frame != (prev[0] + 1 if prev else 1):
        raise ValueError('nonconsecutive auxiliary endpoint in joint replay')
    kind = None
    if prev and raw > 0 and prev[1] == 0:
        kind = 'reappearance'
    elif prev and prev[2] == 'reappearance' and prev[3] < 2 and raw > 0:
        kind = 'reappearance'
    elif raw > 0 and union == 0:
        kind = 'crop_miss'
    elif (iou < .5 or sampled > 0) and has_competitor(
            context['sampled_xyz'], row.get('joint_other_cars', ())):
        kind = 'competing_car'
    elif sampled > 0 and iou < .5:
        kind = 'visible_drift'
    depth = prev[3] + 1 if prev and kind == prev[2] else 0
    if request.ends_window:
        previous.pop(tracklet, None)
    else:
        previous[tracklet] = frame, raw, kind, depth
    return kind, depth


class JointReplayBank:
    """先收集提交前的 detached 数值状态，整轮成功结束后才开放给下一轮。"""
    def __init__(self, seed):
        self.seed = int(seed)
        self.available = {}
        self.epoch = -1

    def begin_epoch(self, epoch):
        self.epoch = int(epoch)
        self.pending = {}
        self.previous = {}
        self.donors = []
        self.slots = 0
        self.used = Counter()
        self.seen = Counter()
        # 每个物理前缀本轮最多插入一次；按类轮转，避免长漂移淹没首次重现。
        groups = {}
        for kind in CATEGORIES:
            keys = [key for key, item in self.available.items() if item['kind'] == kind]
            groups[kind] = sorted(keys, key=lambda key: stable_seed(self.seed, self.epoch, *key))
        for index in range(max((len(group) for group in groups.values()), default=0)):
            for kind in CATEGORIES:
                if index < len(groups[kind]):
                    self.donors.append(groups[kind][index])
        self.donors.reverse()

    def replace(self, rows, dataset):
        result = list(rows)
        for i, row in enumerate(rows):
            old = row['request']
            if old.branch != 0:
                continue
            if old.epoch != self.epoch or not (old.starts_window and old.ends_window):
                raise ValueError('joint replay requires a current-epoch singleton branch 0')
            self.slots += 1
            if self.slots % 4 or not self.donors:
                continue
            key = self.donors.pop()
            item = self.available[key]
            tracklet, frame = key
            request = ReplayRequest(self.epoch, tracklet, 0, frame, frame + 1, frame,
                                    old.drain, old.state_key)
            donor = dataset[request]
            if donor['tracklet_key'] != item['tracklet_key']:
                raise ValueError('joint replay training dataset identity changed')
            # BatchBuilder 初始化独立事务时复制该状态。
            donor['_joint_replay_state'] = item['state']
            # self.epoch 是接收轮的零基索引，同时是供体轮的一基轮号。
            donor['joint_replay_event'] = dict(kind=item['kind'], donor_epoch=self.epoch,
                                               tracklet_key=donor['tracklet_key'], frame=frame)
            result[i] = donor
            self.used[item['kind']] += 1
        return result

    def collect(self, contexts, ious, sampled_counts):
        """context 中的 state 属于本帧预测前。只在优化成功后、辅助 commit 前调用。"""
        if len(contexts) != len(ious) or len(contexts) != len(sampled_counts):
            raise ValueError('joint replay needs one record per auxiliary endpoint')
        for context, iou, sampled in zip(contexts, ious, sampled_counts):
            row = context['row']
            request = row['request']
            tracklet, frame = request.tracklet, request.frame
            kind, depth = observe_joint_prefix(self.previous, self.epoch, context, iou, sampled)
            if kind:
                self.seen[kind] += 1
            if kind and depth < 3:
                key = tracklet, frame
                self.pending[key] = dict(kind=kind, tracklet_key=row['tracklet_key'],
                                         state=deepcopy(context['state']))
            # 长轮次也只保留有界的候选库；key 的优先级与收集的批次顺序无关。
            if len(self.pending) > 2 * len(CATEGORIES) * CAPACITY_PER_CATEGORY:
                self.pending = self._bounded(self.pending)

    def _bounded(self, items):
        keep = {}
        for kind in CATEGORIES:
            keys = (key for key, item in items.items() if item['kind'] == kind)
            selected = sorted(keys, key=lambda key: stable_seed(self.seed, 'joint_bank', *key))
            keep.update((key, items[key]) for key in selected[:CAPACITY_PER_CATEGORY])
        return keep

    def finish_epoch(self):
        if self.previous:
            raise RuntimeError('joint replay still has unfinished auxiliary tracks')
        self.available = self._bounded(self.pending)
        self.pending.clear()

    def audit(self):
        return dict(branch0_slots=self.slots, used=dict(self.used), observed=dict(self.seen),
                    bank=dict(Counter(item['kind'] for item in self.available.values())))

    def checkpoint(self):
        return dict(schema='ct_seqtrack.joint_replay.v1', completed_epoch=self.epoch + 1,
                    available=self.available)

    def restore(self, payload, completed_epoch):
        if (not isinstance(payload, dict) or payload.get('schema') != 'ct_seqtrack.joint_replay.v1'
                or payload.get('completed_epoch') != completed_epoch
                or not isinstance(payload.get('available'), dict)):
            raise ValueError('joint replay checkpoint missing or from a different epoch')
        items = payload['available']
        for key, item in items.items():
            if (not isinstance(key, tuple) or len(key) != 2 or item['kind'] not in CATEGORIES
                    or item['state'].next_frame != key[1] or key[1] < 1):
                raise ValueError('invalid stored joint replay prefix')
        if self._bounded(items).keys() != items.keys():
            raise ValueError('joint replay checkpoint exceeds the per-category limit')
        self.available = items
