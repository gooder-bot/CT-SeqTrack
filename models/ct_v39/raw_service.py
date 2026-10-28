"""v39 主进程统一 raw LRU；缓存与目标、预测 anchor 和模型权重无关。"""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import threading

import numpy as np
import torch

from models.ct_v38.data import point_arrays, metadata


@dataclass(frozen=True)
class RawCloud:
    xyz: np.ndarray
    origin: np.ndarray
    ids: np.ndarray

    @property
    def nbytes(self):
        return self.xyz.nbytes + self.origin.nbytes + self.ids.nbytes


class ByteLRU:
    def __init__(self, capacity):
        if isinstance(capacity, bool) or int(capacity) != capacity or capacity < 0:
            raise ValueError('raw cache capacity must be a nonnegative integer')
        self.capacity = int(capacity)
        self.bytes = 0
        self.data = OrderedDict()

    def get(self, key):
        item = self.data.get(key)
        if item is not None:
            self.data.move_to_end(key)
            return item[0]

    def put(self, key, value, size):
        if size < 0:
            raise ValueError('raw cache entry size must be nonnegative')
        if key in self.data:
            self.bytes -= self.data.pop(key)[1]
        if not self.capacity or size > self.capacity:
            return value
        while self.data and self.bytes + size > self.capacity:
            self.bytes -= self.data.popitem(last=False)[1][1]
        self.data[key] = (value, int(size))
        self.bytes += int(size)
        return value

    def clear(self):
        self.data.clear()
        self.bytes = 0


class RawFrameService:
    """只读借用 raw；四个线程只解码，LRU 插入和状态始终由 owner 管理。"""
    def __init__(self, config):
        self.cpu = ByteLRU(config.get('v39_cpu_raw_cache_bytes', 512 * 1024**2))
        self.gpu = ByteLRU(config.get('v39_gpu_raw_cache_bytes', 256 * 1024**2))
        self.executor = ThreadPoolExecutor(max_workers=max(1, int(config.get('workers', 4))))
        self.owner = threading.get_ident()
        self.stats = dict(cpu_hits=0, cpu_misses=0, gpu_hits=0, h2d_bytes=0)

    def _assert_owner(self):
        if threading.get_ident() != self.owner:
            raise RuntimeError('raw cache mutations belong to the main service owner')

    @staticmethod
    def key(source, track, frame, meta):
        identity = (getattr(source, 'path', None), getattr(source, 'version', None),
                    getattr(source, 'ct_coordinate_mode', 'global'),
                    meta.get('dataset_manifest_sha256', ''), 'sensor_rotated_fp32_origin_fp64_v2')
        if identity[0] is None:
            identity = ('in_memory', id(source))
        physical = meta.get('raw_frame_token', meta.get('raw_frame_id',
                   meta.get('frame_id', meta.get('sample_data_token'))))
        return identity, str(physical) if physical is not None else (int(track), int(frame))

    @staticmethod
    def _decode(source, track, frame, meta=None):
        # nuScenes adapter 的 world 点云会在 float32 数组中累加大平移。
        # 这里直接读取同一原始五列点云，保留原行顺序/ID，绝不经过 world FP32。
        # 只读取传感器文件和静态位姿，目标框、标签、预测 anchor 均不参与 raw 解码。
        annotation = meta.get('meta') if isinstance(meta, dict) else None
        lidar = annotation.get('sample_data_lidar') if isinstance(annotation, dict) else None
        filename = lidar.get('filename') if isinstance(lidar, dict) else None
        direct = (filename is not None and getattr(source, 'path', None) is not None
                  and isinstance(meta, dict) and 'sensor_to_sequence_world' in meta)
        if direct:
            if not isinstance(filename, (str, Path)) or not str(filename):
                raise ValueError('nuScenes raw filename must be a nonempty path')
            transform = np.array(meta['sensor_to_sequence_world'], dtype=np.float64, copy=True)
            if transform.shape != (4, 4) or not np.isfinite(transform).all():
                raise ValueError('sensor_to_sequence_world must be a finite 4x4 transform')
            if not np.allclose(transform[3], (0., 0., 0., 1.), rtol=0., atol=1e-10):
                raise ValueError('sensor_to_sequence_world must have a homogeneous last row')
            rotation = transform[:3, :3]
            if (not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0., atol=1e-6)
                    or not np.isclose(np.linalg.det(rotation), 1., rtol=0., atol=1e-6)):
                raise ValueError('sensor_to_sequence_world rotation must be proper and orthonormal')
            path = Path(source.path) / filename
            if path.stat().st_size % (5 * np.dtype(np.float32).itemsize):
                raise ValueError('nuScenes raw cloud must contain complete five-float records')
            raw = np.fromfile(path, dtype=np.float32)
            if raw.size % 5:
                raise ValueError('nuScenes raw cloud must contain five columns')
            sensor_xyz = raw.reshape(-1, 5)[:, :3].astype(np.float64)
            if not np.isfinite(sensor_xyz).all():
                raise ValueError('nuScenes sensor XYZ must be finite')
            origin = transform[:3, 3].copy()
            with np.errstate(over='ignore', invalid='ignore'):
                local = np.ascontiguousarray(sensor_xyz @ rotation.T, dtype=np.float32)
            ids = np.arange(len(sensor_xyz), dtype=np.int64)
        else:
            raw = source.get_frames(track, [frame])[0]
            xyz, ids = point_arrays(raw)
            origin = xyz[0].copy() if len(xyz) else np.zeros(3, np.float64)
            with np.errstate(over='ignore', invalid='ignore'):
                local = np.ascontiguousarray(xyz - origin, dtype=np.float32)
            ids = np.array(ids, dtype=np.int64, order='C', copy=True)
        if (local.shape != (len(ids), 3) or origin.shape != (3,)
                or not np.isfinite(local).all() or not np.isfinite(origin).all()):
            raise ValueError('raw local XYZ/origin must be finite, aligned and representable in FP32')
        for array in (local, origin, ids):
            array.flags.writeable = False
        return RawCloud(local, origin, ids)

    def fetch_many(self, requests):
        # requests = [(source, track, frame, metadata), ...]；物理帧跨目标去重。
        self._assert_owner()
        pending, output = {}, {}
        for source, track, frame, meta in requests:
            key = self.key(source, track, frame, meta)
            value = self.cpu.get(key)
            if value is not None:
                self.stats['cpu_hits'] += 1
                output[key] = value
            elif key not in pending:
                self.stats['cpu_misses'] += 1
                pending[key] = self.executor.submit(self._decode, source, track, frame, meta)
        for key, future in pending.items():
            value = future.result()
            output[key] = self.cpu.put(key, value, value.nbytes)
        return output

    def fetch(self, source, track, frame, meta=None):
        meta = metadata(source, track, frame) if meta is None else meta
        key = self.key(source, track, frame, meta)
        return key, self.fetch_many([(source, track, frame, meta)])[key]

    def prefetch_many(self, requests, device):
        """设备已缓存的 raw 不为补 CPU LRU 而再次读盘。"""
        self._assert_owner()
        device = str(torch.device(device))
        cold = [request for request in requests
                if (device, self.key(*request)) not in self.gpu.data]
        return self.fetch_many(cold)

    def cached_tensors(self, key, device, anchor):
        self._assert_owner()
        device = torch.device(device)
        result = self.gpu.get((str(device), key))
        if result is None:
            return None
        self.stats['gpu_hits'] += 1
        return self._relative_tensors(result, device, anchor)

    def tensors(self, key, cloud, device, anchor):
        self._assert_owner()
        device = torch.device(device)
        cache_key = (str(device), key)
        result = self.gpu.get(cache_key)
        if result is None:
            # 拷贝只读 NumPy 是 cache miss 的成本，不能暴露可写共享内存。
            xyz = torch.tensor(cloud.xyz, device=device)
            ids = torch.tensor(cloud.ids, device=device)
            # 少量原点 metadata 保持 CPU FP64；仅搬减过 anchor 的小偏移。
            origin = cloud.origin.copy()
            result = (xyz, ids, origin)
            self.gpu.put(cache_key, result, cloud.nbytes)
            if device.type == 'cuda':
                self.stats['h2d_bytes'] += cloud.nbytes
        else:
            self.stats['gpu_hits'] += 1
        return self._relative_tensors(result, device, anchor)

    @staticmethod
    def _relative_tensors(result, device, anchor):
        xyz, ids, origin = result
        if torch.is_tensor(anchor):
            if anchor.device.type != 'cpu':
                raise ValueError('raw anchor must be CPU state metadata; GPU predictions crop locally')
            anchor = anchor.detach().numpy()
        anchor = np.asarray(anchor, np.float64)
        if anchor.ndim != 1 or len(anchor) < 3 or not np.isfinite(anchor[:3]).all():
            raise ValueError('raw anchor must contain finite world XYZ')
        offset = np.asarray(origin, np.float64) - anchor[:3]
        return xyz + torch.tensor(offset, device=device, dtype=torch.float32), ids

    def close(self):
        self._assert_owner()
        self.executor.shutdown(wait=True)
        self.cpu.clear()
        self.gpu.clear()
