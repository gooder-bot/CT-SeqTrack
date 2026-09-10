import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None, zero_invalid_rows=False):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            if zero_invalid_rows:
                allowed = mask != 0
                attn = attn.masked_fill(~allowed, float('-inf'))
                # softmax(-inf,...,-inf) 会产生NaN，空行先用有限常量替代。
                attn = torch.where(allowed.any(dim=-1, keepdim=True),
                                   attn, torch.zeros_like(attn))
            else:
                attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)
        if zero_invalid_rows and mask is not None:
            # 全部 key 无效时原 -1e9 softmax 为均匀分布；v29 必须返回零证据。
            # 同时显式清除无效 key，避免将有限哨兵解释成真实测量。
            attn = attn.masked_fill(mask == 0, 0.0)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)

        return output, attn
