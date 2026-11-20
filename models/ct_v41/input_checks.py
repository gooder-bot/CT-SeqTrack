"""同一模型前向内，复用已通过的输入检查。只接受同一对象及其未变的版本。"""
from contextlib import contextmanager
from contextvars import ContextVar

import torch


_active = ContextVar('v41_input_checks', default=None)


@contextmanager
def input_check_scope():
    token = _active.set({})
    try:
        yield
    finally:
        _active.reset(token)


def _known(predicate, value):
    records = _active.get()
    if records is None:
        return False
    record = records.get((predicate, id(value)))
    # 保存强引用以阻止 id 复用；曾经登记成功的张量一定有版本计数。
    return record is not None and record[0] is value and record[1] == value._version


def _remember(predicate, value):
    records = _active.get()
    if records is not None:
        try:
            version = value._version
        except RuntimeError:  # inference tensor：每处调用独立校验。
            return
        records[predicate, id(value)] = value, version


def positive_finite_size(size):
    if _known('positive_size', size):
        return True
    valid = bool((torch.isfinite(size) & (size > 0)).all())
    if valid:
        _remember('positive_size', size)
    return valid


def detached_size(size):
    # detach 保留存储和版本计数。只给本次新建的别名传递校验证明。
    detached = size.detach()
    if _known('positive_size', size):
        _remember('positive_size', detached)
    return detached


def remember_finite_times(times):
    _remember('finite_times', times)


def finite_times(times):
    return _known('finite_times', times) or bool(torch.isfinite(times).all())
