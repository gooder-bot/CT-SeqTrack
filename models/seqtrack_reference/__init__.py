"""独立 SeqTrack 对照：不导入 production B0 或 Lightning。"""
from .protocol import protocol_identity

__all__ = ['ReferenceTracker', 'BatchBuilder', 'build_loaders', 'protocol_identity']


def __getattr__(name):
    if name == 'ReferenceTracker':
        from .network import ReferenceTracker
        return ReferenceTracker
    if name in ('BatchBuilder', 'build_loaders'):
        from . import data
        return getattr(data, name)
    raise AttributeError(name)
