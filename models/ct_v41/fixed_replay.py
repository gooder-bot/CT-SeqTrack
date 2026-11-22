"""E2 冻结自然前缀。训练按物理端点装入，只有前两轮相同的 A/B 共用文件。"""
from collections import Counter
from dataclasses import fields, is_dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sqlite3
import tempfile
import zlib

import numpy as np
import torch

from .data import BatchBuilder, RawEndpointDataset, ReadyQueueBatchSampler, stable_seed
from .config import config_identity
from .joint_replay import CAPACITY_PER_CATEGORY, CATEGORIES, observe_joint_prefix
from .joint_state import JointSearchState, Measurement, MAX_TEMPLATE_POINTS, physical_frame
from .provenance import source_identity
from .q0_replay import decode_q0_state
from .runtime import capture_rng_state, restore_rng_state
from utils.tracking_metrics import BoxMetricsTarget, LocalYawBox


SCHEMA = 'ct_seqtrack.fixed_joint_e2.v1'
TRAIN_SCENES = frozenset(f'{i:04d}' for i in range(17))
RECEIPT_DIGESTS = ('manifest_sha256', 'donor_weights_sha256')


def validate_fixed_replay_receipt(payload, completed_epoch):
    if (type(completed_epoch) is not int or completed_epoch < 1 or not isinstance(payload, dict)
            or payload.get('schema') != SCHEMA or type(payload.get('completed_epoch')) is not int
            or payload['completed_epoch'] != completed_epoch):
        raise ValueError('fixed replay checkpoint policy or epoch mismatch')
    for key in RECEIPT_DIGESTS:
        value = payload.get(key)
        if (key not in payload or completed_epoch < 3 and value is not None
                or completed_epoch >= 3 and not (isinstance(value, str) and re.fullmatch('[a-f0-9]{64}', value))):
            raise ValueError('invalid fixed replay checkpoint provenance')
    return payload


def _sha_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def weights_sha256(module):
    """绑定跟踪器实际参数和 BN 缓冲值，与两个独立 run 的 UUID、序列化格式无关。"""
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode() + b'\0')
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _training_keys(dataset, role):
    source = dataset.source
    if (not dataset.joint_search or source.v41_source_role != 'train'
            or source.v41_data_role != role or dataset.training != (role == 'train')):
        raise ValueError('fixed replay requires the corresponding train-only joint dataset')
    keys = [source.get_tracklet_key(i) for i in range(len(dataset.lengths))]
    for key in keys:
        parts = key.split('/')
        if (len(parts) < 5 or parts[:3] != ['kitti_v41', 'kitti_tracking', 'train']
                or parts[3] not in TRAIN_SCENES):
            raise ValueError(f'non-training track in fixed replay: {key}')
    return keys


def _json_value(value):
    if is_dataclass(value):
        return vars(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'unsupported replay state field {type(value)}')


def _encode(state):
    # SQLite 仅保存可检查的数值 JSON，不能执行自定义反序列化代码。
    raw = json.dumps(state, default=_json_value, separators=(',', ':'), allow_nan=False)
    return zlib.compress(raw.encode())


def _box(value):
    box = np.asarray(value, dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError('invalid joint history box')
    return box


def _decode(blob, frame, scan):
    values = json.loads(zlib.decompress(blob))
    joint = values.pop('joint')
    state = decode_q0_state(values, frame)
    if not isinstance(joint, dict) or set(joint) != {f.name for f in fields(JointSearchState)}:
        raise ValueError('incompatible joint replay state')
    joint['motion_box'] = _box(joint['motion_box'])
    last_time = state.times[frame - 1]
    if (joint['motion_time'] != last_time or not np.isfinite(joint['last_scan_time'])
            or joint['last_scan_time'] > last_time or not 0 <= joint['history_start'] < frame):
        raise ValueError('joint replay contains a noncausal crop or search state')
    samples = [Measurement(**dict(item, box=_box(item['box']))) for item in joint['measurements']]
    if (not 1 <= len(samples) <= 3 or any(not np.isfinite(s.time) or s.time > last_time
            or s.scan >= scan for s in samples)
            or any(b.time <= a.time or b.scan <= a.scan for a, b in zip(samples, samples[1:]))):
        raise ValueError('joint replay contains a noncausal measurement')
    joint['measurements'] = samples
    template = np.asarray(joint['template_xyz'], dtype=np.float32)
    if not template.size:
        template = template.reshape(0, 3)
    if (template.ndim != 2 or template.shape[1] != 3 or len(template) > MAX_TEMPLATE_POINTS
            or not np.isfinite(template).all()):
        raise ValueError('invalid joint replay template')
    joint['template_xyz'] = template
    if (joint['pending_box'] is None) != (joint['pending_scan'] is None):
        raise ValueError('incomplete joint pending proposal')
    if joint['pending_box'] is not None:
        joint['pending_box'] = _box(joint['pending_box'])
        if joint['pending_scan'] >= scan:
            raise ValueError('joint replay contains a future proposal')
    joint['weak_history_boxes'] = {int(k): _box(v) for k, v in joint['weak_history_boxes'].items()}
    if set(joint['weak_history_boxes']) - state.boxes.keys():
        raise ValueError('joint weak hints must match the causal history')
    state.joint = JointSearchState(**joint)
    return state


def _collect(host, dataset, conn, seed):
    """整轨因果前向。事件由当步标签分类，写入的状态始终取自其提交之前。"""
    _training_keys(dataset, 'auxiliary')
    builder = BatchBuilder(host.config)
    builder.detailed_diagnostics = False
    sampler = ReadyQueueBatchSampler(dataset.lengths, host.config.batch_size, training=False, seed=seed)
    sampler.set_epoch(1)  # E2 使用 zero-based epoch=1 的分支 5 抽点 RNG。
    previous, visited = {}, 0
    for requests in sampler:
        rows = dataset.__getitems__([replace(r, branch=5) for r in requests])
        batch, output = host._forward_raw(rows, builder, training=False)
        try:
            boxes = output.accepted_box.detach().cpu().numpy()
            targets = batch['target_box'].detach().cpu().numpy()
            sizes = batch['target_box_size'].detach().cpu().numpy()
            counts = (batch['segmentation_labels'][:, -1] == 1).sum(-1).detach().cpu().tolist()
            for i, ctx in enumerate(builder._pending):
                row, state = ctx['row'], ctx['state']
                r = row['request']
                metric = BoxMetricsTarget(LocalYawBox(targets[i], sizes[i][[1, 0, 2]]), up_axis=(0, 0, 1))
                exact_iou, _, _ = metric.metrics_3d(LocalYawBox(boxes[i], state.size[[1, 0, 2]]))
                kind, depth = observe_joint_prefix(previous, 1, ctx, exact_iou, counts[i])
                if kind and depth < 3:
                    scan = physical_frame(row['frames'][r.frame], r.frame)
                    conn.execute('INSERT INTO prefixes VALUES (?, ?, ?, ?, ?, ?)',
                                 (r.tracklet, r.frame, row['tracklet_key'], scan, kind, _encode(state)))
                visited += 1
            builder.commit(output, batch)
        finally:
            host._release_materialization(batch, builder)
        del batch, output
    if visited != len(dataset) or builder.states or previous:
        raise RuntimeError('fixed replay donor did not finish all training tracks')
    return visited


def _publish(path, host, train_dataset, origin, seed):
    from datasets.kitti_v41 import get_v41_raw_dataset

    auxiliary_loader = getattr(host, '_loaders', {}).get('auxiliary')
    raw = (auxiliary_loader.dataset.source if auxiliary_loader is not None
           else get_v41_raw_dataset(host.config, 'auxiliary'))
    dataset = RawEndpointDataset(raw, training=False, joint_search=True)
    if dataset.source_sha256 != train_dataset.source_sha256:
        raise ValueError('E2 donor must use the same complete training split')
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.fixed-e2-', suffix='.sqlite', dir=path.parent)
    os.close(fd)
    tmp = Path(name)
    rng = capture_rng_state()
    modes = [(m, m.training) for m in host.tracker.modules()]
    q0_epoch = host.tracker.q0_epoch if host.config.v41_q0_lateral else None
    conn = None
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        host.tracker.eval()
        if q0_epoch is not None:
            host.tracker.q0_epoch = origin['epoch']
        conn = sqlite3.connect(tmp)
        with conn, torch.no_grad():
            conn.execute('CREATE TABLE metadata(payload TEXT NOT NULL)')
            conn.execute('CREATE TABLE prefixes(track INTEGER, frame INTEGER, track_key TEXT, scan INTEGER, '
                         'kind TEXT, state BLOB, PRIMARY KEY(track, frame)) WITHOUT ROWID')
            visited = _collect(host, dataset, conn, seed)
            conn.execute('INSERT INTO metadata VALUES (?)', (json.dumps(dict(origin, visited=visited)),))
        conn.close()
        conn = None
        try:
            os.link(tmp, path)  # 发布完整文件；并发生产者均重新校验最终胜出的同一文件。
        except FileExistsError:
            pass
    finally:
        if conn is not None:
            conn.close()
        tmp.unlink(missing_ok=True)
        for module, mode in modes:
            module.training = mode
        if q0_epoch is not None:
            host.tracker.q0_epoch = q0_epoch
        restore_rng_state(rng)


class FixedReplayBank:
    """E3 起每类每轮最多使用 128 个原物理帧，每前缀最多一次。"""
    def __init__(self, path, seed):
        if not path:
            raise ValueError('fixed replay requires a catalog path')
        self.path, self.seed = Path(path).resolve(), int(seed)
        self.epoch = -1
        self.conn = None
        self.manifest_sha256 = self.donor_weights_sha256 = None
        self.groups = {kind: [] for kind in CATEGORIES}

    def ensure_catalog(self, host, dataset):
        _training_keys(dataset, 'train')
        if getattr(host, '_completed_epoch', None) != 2:
            raise ValueError('fixed replay must be generated from the completed E2')
        # 只有原 A/B 的前两轮完全相同，能共享供体。单视角使用自己的 E2 身份和权重。
        donor_recipe = host.config
        if donor_recipe.v41_history_training in ('window_1448', 'four_candidates'):
            donor_recipe = dict(donor_recipe, experiment_name='paired_e2_reference',
                                v41_history_training='window_1448')
        recipe_sha256 = config_identity(donor_recipe)
        origin = dict(schema=SCHEMA, split='train', epoch=2, seed=self.seed,
                      weights_sha256=weights_sha256(host.tracker), source_sha256=source_identity()['sha256'],
                      dataset_sha256=dataset.source_sha256, common_recipe_sha256=recipe_sha256)
        if not self.path.exists():
            _publish(self.path, host, dataset, origin, self.seed)
        self.bind(dataset, expected_weights_sha256=origin['weights_sha256'],
                  expected_recipe_sha256=recipe_sha256)

    def bind(self, dataset, *, expected_weights_sha256=None, expected_recipe_sha256=None):
        if self.conn is not None:
            if (self.dataset_sha256 != dataset.source_sha256
                    or expected_weights_sha256 not in (None, self.donor_weights_sha256)
                    or expected_recipe_sha256 not in (None, self.recipe_sha256)):
                raise ValueError('fixed replay source changed')
            return
        keys = _training_keys(dataset, 'train')
        checksum = _sha_file(self.path)
        if self.manifest_sha256 not in (None, checksum):
            raise ValueError('fixed replay catalog differs from the checkpoint')
        conn = sqlite3.connect(self.path.as_uri() + '?mode=ro&immutable=1', uri=True)
        try:
            (entry,) = conn.execute('SELECT payload FROM metadata').fetchall()
            meta = json.loads(entry[0])
            weights = meta['weights_sha256']
            recipe = meta['common_recipe_sha256']
            if (meta.get('schema') != SCHEMA or meta['epoch'] != 2 or meta['split'] != 'train'
                    or meta['seed'] != self.seed or meta['visited'] != len(dataset)
                    or meta['dataset_sha256'] != dataset.source_sha256
                    or meta['source_sha256'] != source_identity()['sha256']
                    or not re.fullmatch('[a-f0-9]{64}', weights) or not re.fullmatch('[a-f0-9]{64}', recipe)
                    or expected_recipe_sha256 not in (None, recipe)
                    or expected_weights_sha256 not in (None, weights)
                    or self.donor_weights_sha256 not in (None, weights)):
                raise ValueError('fixed E2 weights, training split, source or seed mismatch')
            groups = {kind: [] for kind in CATEGORIES}
            for track, frame, key, kind in conn.execute('SELECT track,frame,track_key,kind FROM prefixes'):
                if (kind not in groups or not 0 <= track < len(keys) or key != keys[track]
                        or not 1 <= frame < dataset.lengths[track]):
                    raise ValueError('invalid fixed replay physical key')
                groups[kind].append((track, frame))
        except BaseException:
            conn.close()
            raise
        self.groups, self.conn = groups, conn
        self.dataset_sha256, self.donor_weights_sha256 = dataset.source_sha256, weights
        self.manifest_sha256, self.recipe_sha256 = checksum, recipe

    def begin_epoch(self, epoch):
        self.epoch = int(epoch)
        if self.epoch >= 2 and self.conn is None:
            raise ValueError('fixed replay requires the matching E2 catalog from E3 onward')
        self.selected, self.previous = {}, {}
        self.slots = 0
        self.seen, self.used = Counter(), Counter()
        if self.epoch >= 2:
            for kind, keys in self.groups.items():
                ranked = sorted(keys, key=lambda key: (stable_seed(self.seed, self.epoch, kind, *key), key))
                self.selected.update((key, kind) for key in ranked[:CAPACITY_PER_CATEGORY])
        self.plan_sha256 = hashlib.sha256(json.dumps(sorted(self.selected.items())).encode()).hexdigest()
        self.remaining = set(self.selected)

    def replace(self, rows, dataset):
        result = list(rows)
        for i, row in enumerate(rows):
            req = row['request']
            if req.branch != 0:
                continue
            if req.epoch != self.epoch or not (req.starts_window and req.ends_window):
                raise ValueError('fixed replay can only use current-epoch branch 0 singleton rows')
            self.slots += 1
            key = req.tracklet, req.frame
            if key not in self.selected:
                continue
            if key not in self.remaining or dataset.source_sha256 != self.dataset_sha256:
                raise ValueError('duplicate or mismatched fixed replay endpoint')
            track, scan, blob = self.conn.execute(
                'SELECT track_key,scan,state FROM prefixes WHERE track=? AND frame=?', key).fetchone()
            if (track != row['tracklet_key'] or physical_frame(row['frames'][req.frame], req.frame) != scan):
                raise ValueError('fixed replay row and training endpoint disagree')
            result[i] = dict(row, _joint_replay_state=_decode(blob, req.frame, scan),
                             joint_replay_event=dict(kind=self.selected[key], donor_epoch=2,
                                                     tracklet_key=track, frame=req.frame))
            self.remaining.remove(key)
            self.used[self.selected[key]] += 1
        return result

    def collect(self, contexts, ious, sampled_counts):
        if not (len(contexts) == len(ious) == len(sampled_counts)):
            raise ValueError('missing observation for natural auxiliary replay audit')
        for context, iou, sampled in zip(contexts, ious, sampled_counts):
            kind, _ = observe_joint_prefix(self.previous, self.epoch, context, iou, sampled)
            if kind:
                self.seen[kind] += 1

    def finish_epoch(self):
        if self.remaining or self.previous:
            raise RuntimeError('unvisited fixed replay endpoints or incomplete auxiliary trajectories')

    def audit(self):
        return dict(policy=SCHEMA, branch0_slots=self.slots, used=dict(self.used),
                    observed=dict(self.seen), catalog={k: len(v) for k, v in self.groups.items()},
                    plan_sha256=self.plan_sha256, manifest_sha256=self.manifest_sha256,
                    donor_weights_sha256=self.donor_weights_sha256)

    def checkpoint(self):
        return dict(schema=SCHEMA, completed_epoch=self.epoch + 1,
                    manifest_sha256=self.manifest_sha256, donor_weights_sha256=self.donor_weights_sha256)

    def restore(self, payload, completed_epoch):
        validate_fixed_replay_receipt(payload, completed_epoch)
        if any(getattr(self, key) not in (None, payload[key]) for key in RECEIPT_DIGESTS):
            raise ValueError('invalid fixed replay checkpoint provenance')
        for key in RECEIPT_DIGESTS:
            setattr(self, key, payload[key])

    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None
