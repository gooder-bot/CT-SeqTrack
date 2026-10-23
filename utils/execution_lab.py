"""局部执行选项；工程lab或独立登记的B1配方启用，不进入参数或递推状态。"""
from contextlib import contextmanager
from contextvars import ContextVar


STAGE_A = ('unique_mask',)
FACTORIZED = 'factorized_projection'
OPTIONS = STAGE_A + (FACTORIZED,)
# 退役 B2/v37 专用实验开关由冻结源码保留。
KNOWN_NUMERICAL_OPTIONS = ()
_active = ContextVar('ct_execution_lab_options', default=frozenset())


def enabled(name):
    if name not in OPTIONS:
        raise ValueError('unknown execution lab option: ' + str(name))
    return name in _active.get()


@contextmanager
def execution_scope(options=(), *, engineering=False, registered_recipe=False):
    """工程检查或独立登记配方显式启用；退出后恢复原路径。"""
    selected = frozenset(options)
    unknown = selected.difference(OPTIONS)
    if unknown:
        raise ValueError('unknown execution lab options: ' + ', '.join(sorted(unknown)))
    if selected and not (engineering or registered_recipe):
        raise ValueError('execution lab options require an explicit engineering scope')
    token = _active.set(selected)
    try:
        yield
    finally:
        _active.reset(token)


def active_options():
    return tuple(name for name in OPTIONS if enabled(name))
