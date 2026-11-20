"""V41 跨逐点层的有效槽布局，公开输出仍按原始点槽排列。"""
from functools import cached_property, lru_cache
import os
from pathlib import Path
import shutil
import subprocess
import sys

import torch
from torch import nn
from torch.nn import functional as F


def _configure_triton_linker():
    # Triton 2.0 链接 launcher 时使用 -lcuda，训练节点可能只有版本化的运行时库。
    cc = os.environ.get('CC') or shutil.which('gcc') or shutil.which('clang')
    if cc is None:
        return
    probe = subprocess.run([cc, '-print-file-name=libcuda.so'], capture_output=True, text=True)
    if probe.returncode == 0 and Path(probe.stdout.strip()).is_file():
        return
    for prefix in (os.environ.get('CUDA_PATH'), os.environ.get('CUDA_HOME'), '/usr/local/cuda'):
        if prefix:
            stub = Path(prefix) / 'lib64' / 'stubs'
            if (stub / 'libcuda.so').is_file():
                # 仅影响编译器搜索路径，保留用户设置的优先级；运行时继续使用真实驱动。
                paths = os.environ.get('LIBRARY_PATH', '')
                if str(stub) not in paths.split(os.pathsep):
                    os.environ['LIBRARY_PATH'] = os.pathsep.join(filter(None, (paths, str(stub))))
                return


@lru_cache(maxsize=None)
def _point_backend(device):
    # 使用已验证的 Linux / NVIDIA SM8.x / Triton 2.0 工具链，其他环境走原生算子。
    if sys.platform != 'linux' or not torch.version.cuda or torch.cuda.get_device_capability(device)[0] != 8:
        return None
    try:
        import triton
        if triton.__version__.split('.')[:2] != ['2', '0']:
            return None
        from . import _point_kernels
        _configure_triton_linker()
        return _point_kernels
    except (ImportError, OSError):
        return None


class PointSlotLayout:
    """每组是一个端点或一帧，索引保持组内原槽次序。"""

    def __init__(self, valid: torch.Tensor, *, flat_indices=None, present=None):
        self.groups, self.slots = valid.shape
        self._shape = valid.shape
        self.flat_indices = (valid.flatten().nonzero(as_tuple=False).flatten()
                             if flat_indices is None else flat_indices)
        self.present = valid.any(-1) if present is None else present
        self._generated = None
        if (flat_indices is None and type(self.flat_indices) is torch.Tensor
                and self.flat_indices.is_cuda and not self.flat_indices.is_inference()):
            self._generated = (self.flat_indices, self.flat_indices._version)

    def regroup(self, valid, *, present):
        """只重划组边界；共享索引与其来源，避免再次 nonzero 或在 GPU 上检查顺序。"""
        result = type(self)(valid, flat_indices=self.flat_indices, present=present)
        if valid.numel() == self._shape.numel() and valid.device == self.flat_indices.device:
            result._generated = self._generated
        return result

    def _backend(self, rows):
        if (type(rows) is not torch.Tensor or not rows.is_cuda
                or self._generated is None or rows.dtype != torch.float32):
            return None
        # 自定义 Function 只实现反向模式；前向 AD 和 torch.func 变换使用可组合的原生算子。
        if torch._C._are_functorch_transforms_active() or torch.autograd.forward_ad._current_level >= 0:
            return None
        indices, version = self._generated
        if (indices is not self.flat_indices or version != indices._version or self._shape != (self.groups, self.slots)
                or rows.device != indices.device or rows.layout != torch.strided or rows.is_neg()
                or rows.ndim != 2 or rows.size(0) != indices.numel() or not rows.numel()):
            return None
        # 有效行/稠密槽地址用 int32，CUDA 网格的第二、三维最多 65535。
        # 当前最大实测宽度为四帧 × 1024 槽；更宽的 max 保留原路径。
        if (self.slots > 4096 or self.groups * self.slots * rows.size(1) > 2**31 - 256
                or rows.size(1) > 65535 * 32):
            return None
        return _point_backend(rows.device)

    # Feature 已拿到打包好的输入时，逐帧还原和 max 只需要 flat_indices。
    @cached_property
    def group_indices(self):
        return torch.div(self.flat_indices, self.slots, rounding_mode='floor')

    @cached_property
    def slot_indices(self):
        return self.flat_indices.remainder(self.slots)

    def pack(self, channels_first: torch.Tensor) -> torch.Tensor:
        return channels_first.transpose(1, 2)[self.group_indices, self.slot_indices]

    @cached_property
    def _group_offsets(self):
        starts = torch.arange(self.groups + 1, device=self.flat_indices.device) * self.slots
        return torch.searchsorted(self.flat_indices, starts)

    def restore(self, rows: torch.Tensor, fill=0.) -> torch.Tensor:
        channels = rows.shape[-1]
        backend = self._backend(rows)
        if backend is not None and isinstance(fill, (float, int)):
            ok, dense = backend.try_launch(rows.device, backend.UniqueRestore.apply,
                                           rows, self.flat_indices, self.groups * self.slots, fill)
            if ok:
                return dense.view(self.groups, self.slots, channels)
        dense = rows.new_full((self.groups * self.slots, channels), fill)
        dense.index_copy_(0, self.flat_indices, rows)
        return dense.view(self.groups, self.slots, channels)

    def max_per_group(self, rows: torch.Tensor) -> torch.Tensor:
        # max(dim) 将并列最大值的梯度传给原槽序的首点；全空组为零。
        backend = self._backend(rows)
        if (backend is not None and type(self.present) is torch.Tensor and self.present.shape == (self.groups,)
                and self.present.dtype == torch.bool and self.present.device == rows.device and not self.present.is_neg()):
            ok, maxima = backend.try_launch(rows.device, backend.GroupMax.apply, rows, self._group_offsets,
                                            self.flat_indices, self.present, self.slots)
            if ok:
                return maxima
        maxima = self.restore(rows, -torch.inf).max(dim=1).values
        return torch.where(self.present[:, None], maxima, 0.)


def project_rows(conv: nn.Conv1d, rows: torch.Tensor) -> torch.Tensor:
    """1×1 卷积共享原 weight/bias；输入布局为 [有效槽,通道]。"""
    return F.linear(rows, conv.weight[:, :, 0], conv.bias)


def point_blocks(layers, rows, *, retain_second=False):
    low_features = None
    for index, (conv, bn, relu) in enumerate(layers):
        rows = relu(bn(project_rows(conv, rows)))
        if retain_second and index == 1:
            low_features = rows
    return rows, low_features
