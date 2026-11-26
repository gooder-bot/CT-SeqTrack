"""PointRePar 启发的训练历史扰动。固定受扰末框，从其他训练场景移植连续误差。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA = 'ct_seqtrack.v41.density_history.v1'
E57_CHECKPOINT = '5cecd43d2a119ec8f245638736921b9dab11685ead6a1da1ecd9d940c6ece808'
E57_SOURCE = '73bcf8e6bce57963f3cb74b64f553033d13ea104aa1ba8e25e89bfa746d8b247'
TRAIN_DATA = 'b0713e8d61c4d240dbbc872107f0db43a6b922ba46265de2267e6a5b19bac008'
DONOR_ARCHIVE = '2dcb2add6fffae28c00e4fd2fbf9ee19d6db2104647420bdb9bce06dc157c0e6'
DONOR_GEOMETRY = 'b402b4b42f190ece02d4c2dd492d73d8c10943e46e47622ab39a715208de4c21'
TRAIN_SCENES = frozenset(f'{i:04d}' for i in range(17))


def training_scene(key):
    parts = key.split('/')
    if (len(parts) < 6 or parts[:3] != ['kitti_v41', 'kitti_tracking', 'train']
            or parts[3] not in TRAIN_SCENES or parts[5] != 'Car'):
        raise ValueError(f'expected a V41 KITTI Car training track, got {key!r}')
    return parts[3]


def density_bins(counts):
    return np.searchsorted((10, 30), counts, side='right')


def sensor_xyz(local, yaw):
    result = local.copy()
    c, s = np.cos(yaw), np.sin(yaw)
    result[..., 0] = c * local[..., 0] - s * local[..., 1]
    result[..., 1] = s * local[..., 0] + c * local[..., 1]
    return result


class HistoryAugmentation:
    def __init__(self, payload):
        if (payload.get('schema') != SCHEMA or payload.get('split') != 'train'
                or payload.get('checkpoint_sha256') != E57_CHECKPOINT
                or payload.get('teacher_source_sha256') != E57_SOURCE
                or payload.get('source_sha256') != TRAIN_DATA
                or payload.get('reference_archive_sha256') != DONOR_ARCHIVE
                or payload.get('donor_geometry_sha256') != DONOR_GEOMETRY
                or payload.get('donor_origin') not in ('archived_frames', 'frozen_replay')):
            raise ValueError('history bank split, dataset or E57 donor provenance mismatch')
        self.source_sha256 = payload['source_sha256']
        self.receivers = {}
        self.scenes = {}
        for key, counts in payload['receivers'].items():
            self.scenes[key] = training_scene(key)
            # 单帧轨迹没有预测请求，也没有被模型实际采样的首框。
            if counts == [None]:
                self.receivers[key] = ()
                continue
            if (not counts or any(type(count) is not int or not 0 <= count <= 1024 for count in counts)):
                raise ValueError(f'invalid sampled history point counts for {key}')
            self.receivers[key] = density_bins(counts)
        donors = payload['donors']
        if not donors:
            raise ValueError('history bank contains no donor segments')
        identities = {(row['track'], row['endpoint']) for row in donors}
        if len(identities) != len(donors):
            raise ValueError('duplicate history donor prefix')
        self.tracks = np.array([row['track'] for row in donors])
        self.donor_scenes = np.array([self.scenes[key] for key in self.tracks])
        self.errors = np.asarray([row['err_lwh'] for row in donors], dtype=np.float64)
        self.gaps = np.asarray([row['dt_us'] for row in donors])
        counts = np.asarray([row['counts'] for row in donors])
        n = len(donors)
        if (self.errors.shape != (n, 3, 3) or not np.isfinite(self.errors).all()
                or self.gaps.shape != (n, 2) or self.gaps.dtype.kind not in 'iu' or (self.gaps <= 0).any()
                or counts.shape != (n, 3) or counts.dtype.kind not in 'iu' or (counts < 0).any()
                or (counts > 1024).any()):
            raise ValueError('invalid history donor geometry, timing or point counts')
        self.donor_bins = density_bins(counts)
        self._pools = {}

    @classmethod
    def load(cls, path):
        content = Path(path).read_bytes()
        bank = cls(json.loads(content))
        bank.sha256 = hashlib.sha256(content).hexdigest()
        return bank

    def validate_receivers(self, track_lengths):
        lengths = {key: max(1, len(bins)) for key, bins in self.receivers.items()}
        if lengths != track_lengths:
            raise ValueError('history bank receiver coverage differs from training track lengths')

    def _pool(self, scene, bins, gaps):
        key = (scene, bins, gaps)
        if key not in self._pools:
            size = len(bins)
            compatible = ((self.donor_scenes != scene)
                          & (self.gaps[:, -(size - 1):] == gaps).all(axis=1))
            indices = np.flatnonzero(compatible & (self.donor_bins[:, -size:] == bins).all(axis=1))
            reason = 'full_density'
            if len(set(self.tracks[indices])) < 2:
                indices = np.flatnonzero(compatible & (self.donor_bins[:, -1] == bins[-1]))
                reason = 'last_density'
            if len(set(self.tracks[indices])) < 2:
                indices = np.empty(0, dtype=np.intp)
                reason = 'legacy_density' if compatible.any() else 'legacy_gap'
            self._pools[key] = indices, reason
        return self._pools[key]

    def sample(self, track, seed_ids, times, boxes, rng, *, common_z=False):
        """返回逐槽 XYZ 增量和匹配级别。boxes 为尚未加偏移/航向扰动的实际过去框。"""
        ids = tuple(seed_ids)
        if not 1 <= len(ids) <= 3 or ids != tuple(range(ids[0], ids[0] + len(ids))) or ids[0] < 0:
            raise ValueError('seed history needs one to three consecutive existing slots')
        zeros = np.zeros((len(ids), 3), dtype=np.float64)
        if len(ids) == 1:
            return zeros, 'first_frame'
        bins = tuple(self.receivers[track][i] for i in ids)
        delta = np.diff([times[i] for i in ids])
        if not np.isfinite(delta).all() or (delta <= 0).any():
            raise ValueError('seed history requires increasing physical timestamps')
        gaps = tuple(np.rint(delta * 1e6).astype(np.int64))
        pool, reason = self._pool(self.scenes[track], bins, gaps)
        if not len(pool):
            return zeros, reason
        donor = pool[rng.integers(len(pool))]
        residuals = sensor_xyz(self.errors[donor, -len(ids):], np.array([boxes[i][3] for i in ids]))
        deltas = residuals - residuals[-1]
        if common_z:
            # 同一供体末框的有符号高度平移整个历史，不制造虚假的垂向速度。
            deltas[:, 2] += residuals[-1, 2]
        return deltas, reason
