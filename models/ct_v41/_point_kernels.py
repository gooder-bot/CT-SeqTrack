"""V41 FP32 回填与分组 max。调用端提供按原始槽位有序、唯一的有效行索引。"""
import subprocess
import warnings

import torch
import triton
import triton.language as tl
from triton.compiler import CompilationError, OutOfResources


_unavailable = set()


def try_launch(device, fn, *args):
    """工具链不可用时回退。内存不足、参数错误及 CUDA 执行错误仍向上传递。"""
    if device in _unavailable:
        return False, None
    try:
        return True, fn(*args)
    except (CompilationError, OutOfResources, OSError, subprocess.SubprocessError) as exc:
        reason = str(exc)
    except RuntimeError as exc:
        if not str(exc).startswith(('Cannot find ptxas', 'Failed to find C compiler')):
            raise
        reason = str(exc)
    _unavailable.add(device)
    warnings.warn(f'V41 B0 on {device} is using native PyTorch: {reason}', RuntimeWarning)
    return False, None


@triton.jit
def _restore_unique(src, indices, dst, total, C: tl.constexpr, R0: tl.constexpr,
                    R1: tl.constexpr, BLOCK: tl.constexpr):
    linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, ch = linear // C, linear % C
    keep = linear < total
    slot = tl.load(indices + row, mask=keep, other=0).to(tl.int32)
    val = tl.load(src + row.to(tl.int64) * R0 + ch.to(tl.int64) * R1, mask=keep, other=0.)
    tl.store(dst + slot * C + ch, val, mask=keep)


class UniqueRestore(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rows, indices, slots, fill):
        channels = rows.shape[1]
        result = rows.new_full((slots, channels), fill)
        with torch.cuda.device(rows.device):
            _restore_unique[(triton.cdiv(rows.numel(), 256),)](
                rows, indices, result, rows.numel(), channels, *rows.stride(), 256)
        ctx.save_for_backward(indices)
        return result

    @staticmethod
    def backward(ctx, grad):
        indices, = ctx.saved_tensors
        return grad.index_select(0, indices), None, None, None


@triton.jit
def _max_key(value):
    bits = value.to(tl.uint32, bitcast=True)
    magnitude = bits & 0x7fffffff
    # +/-0 are tied. All NaNs beat +inf and tie with one another.
    canonical = tl.where(magnitude == 0, 0, bits).to(tl.uint32)
    key = tl.where((canonical & 0x80000000) != 0, ~canonical,
                   canonical ^ 0x80000000).to(tl.uint32)
    return tl.where(magnitude > 0x7f800000, 0xffffffff, key).to(tl.uint32)


@triton.jit
def _partial_max(src, offsets, keys, winners, C: tl.constexpr, R0: tl.constexpr,
                 R1: tl.constexpr, CHUNKS: tl.constexpr, BR: tl.constexpr,
                 BC: tl.constexpr):
    group, block, cblock = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    begin = tl.load(offsets + group).to(tl.int32)
    end = tl.load(offsets + group + 1).to(tl.int32)
    row = begin + block * BR + tl.arange(0, BR)
    ch = cblock * BC + tl.arange(0, BC)
    partial = (group * CHUNKS + block) * C + ch
    if begin + block * BR < end:
        valid = (row[:, None] < end) & (ch[None, :] < C)
        x = tl.load(src + row[:, None].to(tl.int64) * R0 + ch[None, :].to(tl.int64) * R1, mask=valid, other=0.)
        k = tl.where(valid, _max_key(x), 0).to(tl.uint32)
        best = tl.max(k, 0)
        first = tl.min(tl.where(k == best[None, :], row[:, None], 0x7fffffff), 0)
        tl.store(keys + partial, best, mask=ch < C)
        tl.store(winners + partial, first, mask=ch < C)
    else:
        tl.store(keys + partial, 0, mask=ch < C)
        tl.store(winners + partial, 0x7fffffff, mask=ch < C)


@triton.jit
def _finish_max(src, offsets, flat, present, keys, winners, result, result_indices,
                C: tl.constexpr, R0: tl.constexpr, R1: tl.constexpr,
                SLOTS: tl.constexpr, CHUNKS: tl.constexpr, BQ: tl.constexpr, BC: tl.constexpr):
    group, cblock = tl.program_id(0), tl.program_id(1)
    q = tl.arange(0, BQ)
    ch = cblock * BC + tl.arange(0, BC)
    partial = (group * CHUNKS + q[:, None]) * C + ch[None, :]
    valid = (q[:, None] < CHUNKS) & (ch[None, :] < C)
    k = tl.load(keys + partial, mask=valid, other=0).to(tl.uint32)
    w = tl.load(winners + partial, mask=valid, other=0x7fffffff)
    best = tl.max(k, 0)
    pick = tl.min(tl.where(k == best[None, :], w, 0x7fffffff), 0)
    begin = tl.load(offsets + group).to(tl.int32)
    end = tl.load(offsets + group + 1).to(tl.int32)
    first_slot = tl.load(flat + begin, mask=begin < end, other=-1).to(tl.int32)
    enabled = tl.load(present + group) != 0
    # The dense reference pads with -inf. A leading pad wins a tie against valid -inf.
    live = enabled & (best != 0) & ((best != 0x007fffff) | (first_slot == group * SLOTS))
    winner = tl.where(live, pick, -1)
    value = tl.load(src + tl.maximum(winner, 0).to(tl.int64) * R0 + ch.to(tl.int64) * R1,
                    mask=live & (ch < C), other=-float('inf'))
    tl.store(result + group * C + ch, tl.where(enabled, value, 0.), mask=ch < C)
    tl.store(result_indices + group * C + ch, winner, mask=ch < C)


@triton.jit
def _distribute(grad, winners, dst, total, C: tl.constexpr,
                G0: tl.constexpr, G1: tl.constexpr, BLOCK: tl.constexpr):
    linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    group, ch = linear // C, linear % C
    row = tl.load(winners + linear, mask=linear < total, other=-1)
    value = tl.load(grad + group.to(tl.int64) * G0 + ch.to(tl.int64) * G1, mask=linear < total, other=0.)
    tl.store(dst + tl.maximum(row, 0) * C + ch, value, mask=(linear < total) & (row >= 0))


class GroupMax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rows, offsets, indices, present, slots):
        groups, channels = present.numel(), rows.shape[1]
        chunks = triton.cdiv(slots, 64)
        keys = torch.empty((groups, chunks, channels), dtype=torch.int32, device=rows.device)
        winners = torch.empty_like(keys)
        output = rows.new_empty((groups, channels))
        saved = torch.empty((groups, channels), dtype=torch.int32, device=rows.device)
        with torch.cuda.device(rows.device):
            _partial_max[(groups, chunks, triton.cdiv(channels, 32))](
                rows, offsets, keys, winners, channels, *rows.stride(), chunks, 64, 32)
            _finish_max[(groups, triton.cdiv(channels, 32))](
                rows, offsets, indices.contiguous(), present.contiguous(), keys, winners, output, saved,
                channels, *rows.stride(), slots, chunks, triton.next_power_of_2(chunks), 32)
        # 原 torch.where 的反向也校验前向使用的有效掩码版本。
        ctx.save_for_backward(saved, indices, present)
        ctx.rows_shape = rows.shape
        ctx.slots = slots
        return output

    @staticmethod
    def backward(ctx, grad):
        winners, indices, _present = ctx.saved_tensors
        if not torch.is_grad_enabled() and not grad.is_neg():
            result = grad.new_zeros(ctx.rows_shape)
            with torch.cuda.device(grad.device):
                ok, _ = try_launch(grad.device, _distribute[(triton.cdiv(winners.numel(), 256),)],
                    grad, winners, result, winners.numel(), winners.shape[1], *grad.stride(), 256)
            if ok:
                return result, None, None, None, None
        # 可微的 PyTorch 路径也支持 double backward 及首轮反向 JIT 缺失。
        group = torch.div(indices, ctx.slots, rounding_mode='floor')
        row = torch.arange(len(indices), device=indices.device)[:, None]
        result = torch.where(winners.index_select(0, group) == row, grad.index_select(0, group), 0.)
        return result, None, None, None, None
