"""局部执行选项；工程lab或独立登记的B1配方启用，不进入参数或递推状态。"""
from contextlib import contextmanager
from contextvars import ContextVar


STAGE_A = ('labels', 'pool', 'prior_read', 'geometry_gather', 'unique_mask', 'commit_indices')
FACTORIZED = 'factorized_projection'
OPTIONS = STAGE_A + (FACTORIZED,)
# 删除prior行改变GEMM归约形状：已复现第二步梯度差异，不可进入严格A组合。
KNOWN_NUMERICAL_OPTIONS = ('prior_read',)
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
