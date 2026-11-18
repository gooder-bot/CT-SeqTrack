"""进程内只读元数据复用，以及按实际 ndarray 字节数限制的整云 LRU。"""
from __future__ import annotations

from collections import OrderedDict
import os
from pathlib import Path
import threading
import weakref

import numpy as np


DEFAULT_POINTCLOUD_CACHE_BYTES = 256 * 1024 * 1024
_METADATA = weakref.WeakValueDictionary()
_METADATA_LOCK = threading.RLock()
_CACHE_PID = None
_POINTCLOUD_CACHE = None


def shared_nuscenes_metadata(constructor, root, version):
    """只共享构造完成的 devkit 表；调用方不得修改其记录。"""
    key = (os.getpid(), os.path.normcase(str(Path(root).expanduser().resolve())),
           str(version), constructor)
    with _METADATA_LOCK:
        value = _METADATA.get(key)
        if value is None:
            value = constructor(version=version, dataroot=root, verbose=False)
            _METADATA[key] = value
        return value


class PointCloudArrayLRU:
    """按字节限制只读数组；get 返回可写副本，load_readonly 复用只读内容。"""
    def __init__(self, max_bytes=DEFAULT_POINTCLOUD_CACHE_BYTES):
        if isinstance(max_bytes, bool) or int(max_bytes) != float(max_bytes) or int(max_bytes) < 0:
            raise ValueError('ct_pointcloud_cache_bytes must be a non-negative integer')
        self.max_bytes = int(max_bytes)
        self.bytes_used = 0
        self._entries = OrderedDict()

    def get(self, key):
        entry = self._entries.get(key)
        if entry is None:
            return None
        self._entries.move_to_end(key)
        return tuple(array.copy(order='K') for array in entry)

    def put(self, key, points, point_ids):
        if self.max_bytes and points.nbytes + point_ids.nbytes <= self.max_bytes:
            self._store(key, (np.array(points, copy=True, order='K'), np.array(point_ids, copy=True)))
        else:
            self._discard(key)

    def _discard(self, key):
        old = self._entries.pop(key, None)
        if old is not None:
            self.bytes_used -= sum(array.nbytes for array in old)

    def _store(self, key, stored):
        self._discard(key)
        size = sum(array.nbytes for array in stored)
        while self._entries and self.bytes_used + size > self.max_bytes:
            _, removed = self._entries.popitem(last=False)
            self.bytes_used -= sum(array.nbytes for array in removed)
        for array in stored:
            array.flags.writeable = False
        self._entries[key] = stored
        self.bytes_used += size

    def load_readonly(self, key, loader):
        """内部只读路径。loader 移交新建数组的所有权；公开 get 仍返回独立副本。"""
        if key in self._entries:
            self._entries.move_to_end(key)
            return self._entries[key]
        arrays = tuple(loader())
        for array in arrays:
            array.flags.writeable = False
        if self.max_bytes and sum(array.nbytes for array in arrays) <= self.max_bytes:
            self._store(key, arrays)
        return arrays

    def __len__(self):
        return len(self._entries)


def process_pointcloud_cache(max_bytes=DEFAULT_POINTCLOUD_CACHE_BYTES):
    """同进程所有 dataset 合计一个预算；fork 后首次使用丢弃父进程缓存。"""
    global _CACHE_PID, _POINTCLOUD_CACHE
    pid = os.getpid()
    if _CACHE_PID != pid or _POINTCLOUD_CACHE is None or _POINTCLOUD_CACHE.max_bytes != max_bytes:
        _POINTCLOUD_CACHE = PointCloudArrayLRU(max_bytes)
        _CACHE_PID = pid
    return _POINTCLOUD_CACHE


def load_pointcloud_arrays(key, loader, max_bytes=DEFAULT_POINTCLOUD_CACHE_BYTES):
    cache = process_pointcloud_cache(max_bytes)
    arrays = cache.get(key)
    if arrays is not None:
        return arrays
    points, point_ids = loader()
    cache.put(key, points, point_ids)
    # Fresh disk reads already belong to the caller; put stored a separate copy.
    return points, point_ids
