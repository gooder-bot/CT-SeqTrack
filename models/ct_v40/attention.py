"""v40 窄 attention 替换：FP32 投影，只在 FA2 core 内转 BF16。

FA2 的设备、dtype、可分离 mask、依赖或 deterministic 不满足时直接报错，
不偷偷回退。Q/K/V、输出投影、残差、FFN 和 norm 仍属于原 FP32 decoder。
varlen 的有效长度有界但随 batch 变化；不承诺 CUDA allocator 不保留缓存。
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
from types import MethodType

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class ValidityMask:
    query_valid: Tensor
    key_valid: Tensor
    query_indices: Tensor
    key_indices: Tensor
    cu_query: Tensor
    cu_key: Tensor
    max_query: int
    max_key: int

    def unsqueeze(self, dim):
        if dim != 1:
            raise ValueError('validity attention only inserts the head axis')
        return self

    @classmethod
    def from_allowed(cls, mask: Tensor):
        allowed = mask.bool()
        if allowed.ndim == 4 and allowed.shape[1] == 1:
            allowed = allowed[:, 0]
        if allowed.ndim != 3:
            raise ValueError('v40 validity attention expects [B,L,S] or [B,1,L,S]')
        query, key = allowed.any(-1), allowed.any(-2)
        if not torch.equal(allowed, query[:, :, None] & key[:, None, :]):
            raise ValueError('FA2 varlen requires separable query/key validity masks')
        qindices = query.reshape(-1).nonzero(as_tuple=False).flatten()
        kindices = key.reshape(-1).nonzero(as_tuple=False).flatten()
        qcounts, kcounts = query.sum(-1), key.sum(-1)
        active = (qcounts > 0) & (kcounts > 0)
        qcounts, kcounts = qcounts[active], kcounts[active]
        # 一次 stream 入口准备，三层共享；空 batch 使用 0 长度元数据。
        maxima = torch.stack((qcounts.max() if qcounts.numel() else query.new_zeros((), dtype=torch.long),
                              kcounts.max() if kcounts.numel() else key.new_zeros((), dtype=torch.long)))
        maxq, maxk = maxima.cpu().tolist()
        zero = torch.zeros(1, dtype=torch.int32, device=mask.device)
        return cls(query, key, qindices, kindices,
                   torch.cat((zero, qcounts.cumsum(0).to(torch.int32))),
                   torch.cat((zero, kcounts.cumsum(0).to(torch.int32))), int(maxq), int(maxk))

    def dense(self):
        return self.query_valid[:, None, :, None] & self.key_valid[:, None, None, :]


@lru_cache(maxsize=1)
def _flash_varlen():
    try:
        import flash_attn
        from flash_attn import flash_attn_varlen_func
    except (ImportError, OSError) as error:
        raise RuntimeError('v40 fa2_varlen requires a compatible installed flash-attn CUDA extension; '
                           'math fallback is forbidden') from error
    version = getattr(flash_attn, '__version__', '')
    if version.split('+')[0] != '2.5.8':
        raise RuntimeError('v40 fa2_varlen requires registered flash-attn==2.5.8 '
                           f'(local wheel suffix allowed), found {version!r}; math fallback is forbidden')
    return flash_attn_varlen_func


def validate_fa_runtime():
    """正式入口的依赖验收；不启动 kernel，不允许隐式 math 回退。"""
    if not torch.cuda.is_available():
        raise RuntimeError('v40 fa2_varlen requires an available CUDA device')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('v40 fa2_varlen requires native CUDA BF16 support')
    _flash_varlen()
    import flash_attn
    return dict(backend='fa2_varlen', flash_attn=flash_attn.__version__,
                device=torch.cuda.current_device(), device_name=torch.cuda.get_device_name(),
                kernel_dtype='torch.bfloat16', projection_dtype='torch.float32')


ensure_fa_runtime = validate_fa_runtime


def _math_backend(device):
    if device.type != 'cuda':
        return nullcontext()
    if hasattr(torch.nn, 'attention'):
        from torch.nn.attention import SDPBackend, sdpa_kernel
        return sdpa_kernel(backends=[SDPBackend.MATH])
    return torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)


class EfficientAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=.1, *, backend='math', deterministic=True):
        super().__init__()
        if backend not in ('math', 'fa2_varlen'):
            raise ValueError('v40 attention backend must be math or fa2_varlen')
        if type(deterministic) is not bool or not deterministic:
            raise ValueError('v40 attention requires deterministic=True')
        self.temperature = float(temperature)
        self.dropout = nn.Dropout(attn_dropout)
        self.backend = backend
        self.deterministic = deterministic
        self.last_backend = None
        self.kernel_calls = 0
        self.last_kernel_dtype = None

    def forward(self, q, k, v, mask=None, zero_invalid_rows=False):
        self.last_kernel_dtype = None
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError('attention requires Q/K/V [B,H,L,D]')
        if q.shape[:2] != k.shape[:2] or k.shape[:3] != v.shape[:3] or q.shape[-1] != k.shape[-1]:
            raise ValueError('attention Q/K/V shape mismatch')
        dropout = self.dropout.p if self.training else 0.
        if self.backend == 'math':
            allowed = mask.dense() if isinstance(mask, ValidityMask) else mask
            if allowed is not None and not zero_invalid_rows:
                # 非 v40 的旧接口约定：全无效行的有限 -1e9 哨兵给均匀权重。
                # 正式路径始终 zero_invalid_rows=True；这里仅保留参考语义。
                logits = torch.matmul(q / self.temperature, k.transpose(-2, -1))
                weights = logits.masked_fill(~allowed.bool(), -1e9).softmax(-1)
                self.last_backend = 'math'
                return F.dropout(weights, dropout, training=self.training) @ v, None
            if allowed is not None:
                allowed = allowed.bool()
                nonempty = allowed.any(-1, keepdim=True)
                # 旧 torch 的 SDPA 全 -inf 行可能为 NaN；先允许空行，再显式归零。
                safe = allowed | ~nonempty
            else:
                safe, nonempty = None, None
            with _math_backend(q.device):
                # torch 2.0 的 SDPA 无 scale 关键字；默认 1/sqrt(d_k) 前补比例。
                # 注册的 64 维、temperature=8 时比例恰为 1，不插入多余算子。
                factor = q.shape[-1] ** .5 / self.temperature
                query = q if factor == 1. else q * factor
                output = F.scaled_dot_product_attention(query, k, v, attn_mask=safe,
                    dropout_p=dropout, is_causal=False)
            if nonempty is not None and zero_invalid_rows:
                output = torch.where(nonempty, output, 0.)
            self.last_backend = 'math'
            return output, None
        if q.device.type != 'cuda':
            raise RuntimeError('v40 fa2_varlen requires CUDA; math fallback is forbidden')
        if any(x.dtype != torch.float32 or x.device != q.device for x in (q, k, v)):
            raise RuntimeError('v40 fa2_varlen requires FP32 projected Q/K/V on the same CUDA device')
        with torch.cuda.device(q.device):
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError('v40 fa2_varlen requires native CUDA BF16 support')
        if q.shape[1] != 4 or q.shape[-1] != 64 or v.shape[-1] != 64:
            raise ValueError('v40 registered attention requires 4 heads of 64 dimensions')
        flash = _flash_varlen()  # 即使全空，正式环境仍须满足依赖合同。
        if mask is None:
            mask = torch.ones((q.shape[0], q.shape[2], k.shape[2]), dtype=torch.bool, device=q.device)
        validity = mask if isinstance(mask, ValidityMask) else ValidityMask.from_allowed(mask)
        if not zero_invalid_rows:
            raise ValueError('v40 FA2 requires explicit zero_invalid_rows=True')
        if validity.query_valid.shape != (len(q), q.shape[2]) or validity.key_valid.shape != (len(k), k.shape[2]):
            raise ValueError('attention validity shape differs from Q/K')
        self.last_backend = 'fa2_varlen'
        if validity.max_query == 0:
            return q * 0. + (k.sum() + v.sum()) * 0., None
        packed_q = q.transpose(1, 2).reshape(-1, 4, 64).index_select(0, validity.query_indices).to(torch.bfloat16)
        packed_k = k.transpose(1, 2).reshape(-1, 4, 64).index_select(0, validity.key_indices).to(torch.bfloat16)
        packed_v = v.transpose(1, 2).reshape(-1, 4, 64).index_select(0, validity.key_indices).to(torch.bfloat16)
        result = flash(packed_q, packed_k, packed_v, validity.cu_query, validity.cu_key,
                       validity.max_query, validity.max_key, dropout_p=dropout,
                       softmax_scale=1. / self.temperature, causal=False,
                       deterministic=True, return_attn_probs=False)
        self.kernel_calls += 1
        self.last_kernel_dtype = str(result.dtype)
        result = result.float()
        output = result.new_zeros((q.shape[0] * q.shape[2], 4, 64))
        output = output.index_copy(0, validity.query_indices, result)
        return output.reshape(q.shape[0], q.shape[2], 4, 64).transpose(1, 2), None


def _encoder_forward(self, src_seq, src_mask=None, return_attns=False, global_feature=False,
                     query_valid=None, zero_invalid_rows=False):
    from models.attn.Models import Encoder
    if return_attns:
        raise ValueError('v40 output-only attention does not materialize attention weights')
    if src_mask is not None and not isinstance(src_mask, ValidityMask):
        src_mask = ValidityMask.from_allowed(src_mask)
    return Encoder.forward(self, src_seq, src_mask, return_attns, global_feature,
                           query_valid, zero_invalid_rows)


def _decoder_forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False,
                     query_valid=None, zero_invalid_rows=False):
    from models.attn.Models import Decoder
    if return_attns:
        raise ValueError('v40 output-only attention does not materialize attention weights')
    if src_mask is not None and not isinstance(src_mask, ValidityMask):
        src_mask = ValidityMask.from_allowed(src_mask)
    return Decoder.forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns,
                           query_valid, zero_invalid_rows)


def configure_attention_modules(decoder, *, backend='fa2_varlen', deterministic=True):
    from models.attn.SubLayers import MultiHeadAttention
    from models.attn.Models import Encoder, Decoder
    attention_count = 0
    for module in decoder.modules():
        if isinstance(module, MultiHeadAttention):
            if (module.n_head, module.d_k, module.d_v) != (4, 64, 64):
                raise ValueError('v40 decoder must retain its 4x64 head layout')
            old = module.attention
            replacement = EfficientAttention(old.temperature, old.dropout.p,
                                              backend=backend, deterministic=deterministic)
            replacement.train(old.training)
            module.attention = replacement
            attention_count += 1
        elif isinstance(module, Encoder):
            module.forward = MethodType(_encoder_forward, module)
        elif isinstance(module, Decoder):
            module.forward = MethodType(_decoder_forward, module)
    if attention_count != 9:
        raise ValueError('v40 expects three local/global/cross layers each (nine attention cores)')
    return decoder
