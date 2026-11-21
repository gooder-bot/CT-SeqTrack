"""uniform 训练轨的离线错误前态。清单只携带闭环状态，当前点从原始帧重采。"""
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .data import KNOWN_GT_BOX, PREDICTED_BOX, V41AcceptedState, stable_seed
from .joint_replay import ReplayRequest


SCHEMA = 'ct_seqtrack.q0_prefix_pairs.v1'


def _vector(value, length):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ValueError('q0 prefix vector has invalid size or nonfinite values')
    return array


def decode_q0_state(record, frame):
    """从 JSON 恢复独立数值状态，拒绝任何含当前/未来框或晚期 GT 重置的前态。"""
    values = dict(record)
    required = {field.name for field in fields(V41AcceptedState)} - {'joint'}
    if (set(values) - {'joint'} != required or values.get('joint') is not None
            or values['next_frame'] != frame or frame < 1):
        raise ValueError('invalid q0 causal prefix fields or next_frame')
    indices = set(range(max(0, frame - 3), frame))
    for name in ('boxes', 'times', 'transitions', 'box_provenance'):
        values[name] = {int(i): v for i, v in values[name].items()}
        if set(values[name]) != indices:
            raise ValueError('q0 prefix must contain only the three previous physical frames')
    for i in sorted(indices):
        values['boxes'][i] = _vector(values['boxes'][i], 4)
        if values['box_provenance'][i] != (PREDICTED_BOX if i else KNOWN_GT_BOX):
            raise ValueError('q0 prefix requires a free-running uniform trajectory')
    values['size'] = _vector(values['size'], 3)
    values['seed_offset'] = _vector(values['seed_offset'], 4)
    values['trusted_velocity'] = _vector(values['trusted_velocity'], 3)
    values['trusted'] = [(_vector(box, 4), float(time)) for box, time in values['trusted']]
    times = [values['times'][i] for i in sorted(indices)]
    trusted_times = [time for _, time in values['trusted']]
    timestamps = trusted_times + [values[k] for k in ('initialization_time', 'last_supported_time',
        'last_velocity_update_time', 'last_strong_time') if values[k] is not None]
    if (not 1 <= len(trusted_times) <= 2 or min(values['size']) <= 0 or values['seed_offset'].any()
            or not np.isfinite(times + timestamps).all() or max(timestamps) > times[-1]
            or any(np.diff(times) <= 0) or any(np.diff(trusted_times) <= 0)):
        raise ValueError('q0 prefix has invalid physical times, trusted history or seed')
    return V41AcceptedState(**values)


class Q0ReplayBank:
    """配对端点平铺到整轮，替换 branch 0 原事务槽；恢复只依赖清单和轮号。"""

    def __init__(self, manifest_path, seed, dataset):
        raw = Path(manifest_path).read_bytes()
        manifest = json.loads(raw)
        self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        self.seed = int(seed)
        if (manifest.get('schema') != SCHEMA or manifest.get('split') != 'train'
                or not dataset.training or getattr(dataset.source, 'v41_data_role', 'train') != 'train'
                or manifest.get('source_sha256') != dataset.source_sha256):
            raise ValueError('q0 replay requires the matching train-only dataset and prefix schema')
        self.donors = manifest['donors']
        if (not isinstance(self.donors, list) or sorted(d['epoch'] for d in self.donors) != [10, 60]
                or any(not re.fullmatch('[0-9a-f]{64}', d.get(k, ''))
                       for d in self.donors for k in ('checkpoint_sha256', 'source_sha256'))):
            raise ValueError('q0 replay requires fingerprinted uniform E10/E60 donors')
        self.pairs = {}
        for pair in manifest['pairs']:
            event = pair['event_id']
            if not isinstance(event, str) or not event or event in self.pairs:
                raise ValueError('q0 replay event IDs must be unique')
            endpoints = {}
            for kind in ('hard', 'normal'):
                item = pair[kind]
                track, frame = item['tracklet'], item['frame']
                if not (type(track) is int and type(frame) is int and 0 <= track < len(dataset.lengths)
                        and 1 <= frame < dataset.lengths[track] and item['donor_epoch'] in (10, 60)):
                    raise ValueError('q0 replay endpoint outside donor or training tracks')
                track_name = (str(dataset.source.get_tracklet_key(track))
                              if hasattr(dataset.source, 'get_tracklet_key') else str(track))
                if item['tracklet_key'] != track_name:
                    raise ValueError('q0 replay track identity differs from training dataset')
                endpoints[kind] = dict(item, state=decode_q0_state(item['state'], frame))
            hard, normal = endpoints['hard'], endpoints['normal']
            if (hard['donor_epoch'] != normal['donor_epoch']
                    or (hard['tracklet'], hard['frame']) == (normal['tracklet'], normal['frame'])):
                raise ValueError('q0 replay needs distinct hard and normal endpoints from one donor')
            self.pairs[event] = endpoints
        self.epoch = None

    def begin_epoch(self, epoch, branch0_slots):
        self.epoch = int(epoch)
        self.budget = int(branch0_slots)
        self.slots = 0
        self.used = dict(hard=0, normal=0)
        self.schedule = {}
        events = sorted(self.pairs, key=lambda key: (stable_seed(self.seed, self.epoch, key), key))
        self.planned_pairs = min(self.budget // 4, len(events))
        # 四个单步槽最多使用两个。样本稀少时分散至整轮；永远替换完整的一对。
        for index, event in enumerate(events[:self.planned_pairs]):
            end = (index + 1) * self.budget // self.planned_pairs
            self.schedule[end - 1] = event, 'hard'
            self.schedule[end] = event, 'normal'

    def replace(self, rows, dataset):
        result = list(rows)
        for index, row in enumerate(rows):
            old = row['request']
            if old.branch != 0:
                continue
            if self.epoch != old.epoch or not (old.starts_window and old.ends_window):
                raise ValueError('q0 replay requires current-epoch branch 0 singleton slots')
            self.slots += 1
            if self.slots > self.budget:
                raise ValueError('q0 replay exceeded the declared branch 0 budget')
            scheduled = self.schedule.get(self.slots)
            if scheduled is None:
                continue
            event, kind = scheduled
            item = self.pairs[event][kind]
            track, frame = item['tracklet'], item['frame']
            request = ReplayRequest(old.epoch, track, 0, frame, frame + 1, frame,
                                    old.drain, old.state_key)
            # 原 slot 的预裁剪属于另一前态；重新读取物理端点，GT10 不覆盖时由原路径回读整云。
            donor = dataset[request]
            if (donor['tracklet_key'] != item['tracklet_key']
                    or any(item['state'].times[i] != float(donor['frames'][i]['timestamp'])
                           for i in donor['history_ids'])):
                raise ValueError('q0 replay physical history and current track do not match')
            donor['_q0_replay_state'] = item['state']
            donor['q0_replay_event'] = dict(event_id=event, kind=kind, donor_epoch=item['donor_epoch'])
            result[index] = donor
            self.used[kind] += 1
        return result

    def finish_epoch(self):
        if self.slots != self.budget or any(n != self.planned_pairs for n in self.used.values()):
            raise RuntimeError('incomplete q0 replay epoch or unmatched hard/normal endpoints')

    def audit(self):
        return dict(manifest_sha256=self.manifest_sha256, donors=self.donors, branch0_slots=self.slots,
                    available_events=len(self.pairs), planned_pairs=self.planned_pairs, used=dict(self.used))
