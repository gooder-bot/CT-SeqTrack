"""单次 prepare/forward/loss/commit 的规范化复用；不进入模型或递归状态。"""
from __future__ import annotations

from dataclasses import dataclass

import torch


_CACHE_KEY = '_ct_materialization_cache'
_POINT_FIELDS = ('points', 'point_valid', 'point_ids', 'history_valid')
_EXTENSION_FIELDS = ('extension_points', 'extension_valid', 'extension_ids')


def _signature(values, fields):
    """只读主机元数据；inference tensor 或输入梯度路径不缓存。"""
    parts = []
    for name in fields:
        value = values.get(name)
        if value is None:
            parts.append((name, None))
            continue
        if not torch.is_tensor(value) or value.requires_grad:
            return None
        try:
            version = value._version
        except RuntimeError:  # inference tensors 不提供版本计数。
            return None
        parts.append((name, id(value), value.data_ptr(), tuple(value.shape),
                      tuple(value.stride()), value.storage_offset(), value.dtype,
                      value.device, version))
    return tuple(parts)


@dataclass
class _Record:
    fields: tuple
    original: dict
    original_signature: tuple
    valid: torch.Tensor
    canonical: dict | None = None
    canonical_signature: tuple | None = None
    valid_signature: tuple | None = None

    def __post_init__(self):
        self.valid_signature = _signature({'valid': self.valid}, ('valid',))

    def matches(self, batch):
        if _signature({'valid': self.valid}, ('valid',)) != self.valid_signature:
            return False
        if _signature(self.original, self.fields) != self.original_signature:
            return False
        if self.canonical is not None:
            if _signature(self.canonical, self.fields) != self.canonical_signature:
                return False
        signature = _signature(batch, self.fields)
        return signature is not None and (signature == self.original_signature
            or (self.canonical_signature is not None and signature == self.canonical_signature))


class BatchMaterializationCache:
    """每类仅一条记录；原输入与 canonical 输出同时登记别名及版本。"""
    def __init__(self):
        self.point = None
        self.extension = None

    def clear(self):
        self.point = self.extension = None

    def _get(self, name, batch):
        record = getattr(self, name)
        if record is not None and not record.matches(batch):
            setattr(self, name, None)
            return None
        return record

    def point_mask(self, batch):
        record = self._get('point', batch)
        return None if record is None else record.valid

    def canonical_values(self, batch, *, extension=False):
        record = self._get('extension' if extension else 'point', batch)
        return None if record is None else record.canonical

    def remember_point_mask(self, batch, valid):
        signature = _signature(batch, _POINT_FIELDS)
        if signature is not None:
            self.point = _Record(_POINT_FIELDS,
                {key: batch.get(key) for key in _POINT_FIELDS}, signature, valid)

    def remember_canonical(self, original, canonical, *, extension=False):
        fields = _EXTENSION_FIELDS if extension else _POINT_FIELDS
        signature, alias = _signature(original, fields), _signature(canonical, fields)
        if signature is None or alias is None:
            return
        valid_key = 'extension_valid' if extension else 'point_valid'
        record = _Record(fields, {key: original.get(key) for key in fields}, signature,
            canonical[valid_key], {key: canonical.get(key) for key in fields}, alias)
        setattr(self, 'extension' if extension else 'point', record)


def materialization_cache(batch):
    cache = batch.get(_CACHE_KEY)
    return cache if isinstance(cache, BatchMaterializationCache) else None


def attach_materialization_cache(batch, *, enabled):
    """host 在 move_tensors 后调用；不传给 worker，也不复用前一事务。"""
    clear_materialization_cache(batch)
    if enabled:
        batch[_CACHE_KEY] = BatchMaterializationCache()
    return batch


def clear_materialization_cache(batch):
    cache = batch.pop(_CACHE_KEY, None)
    if isinstance(cache, BatchMaterializationCache):
        cache.clear()
