''' Define the seq2seq model '''
import torch.nn as nn
from .Layers import EncoderLayer, DecoderLayer


class Encoder(nn.Module):

    def __init__(
            self, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=200):

        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos


    def forward(self, src_seq, src_mask=None, return_attns=False, global_feature=False,
                query_valid=None, zero_invalid_rows=False):

        enc_slf_attn_list = []
        # -- Forward
        if global_feature:
            enc_output = self.dropout(self.with_pos_embed(src_seq)) #--positional encoding off
        else:
            enc_output = self.dropout(self.with_pos_embed(src_seq))
        if query_valid is not None:
            enc_output = enc_output.masked_fill(~query_valid, 0.0)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, slf_attn_mask=src_mask,
                zero_invalid_rows=zero_invalid_rows)
            if query_valid is not None:
                # residual/FFN/LN bias 也不能重新生成不存在的 source token。
                enc_output = enc_output.masked_fill(~query_valid, 0.0)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list


        return enc_output,


class Decoder(nn.Module):

    def __init__(
            self, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, n_position=200, dropout=0.1):

        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False,
                query_valid=None, zero_invalid_rows=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        dec_output = (trg_seq)
        if query_valid is not None:
            dec_output = dec_output.masked_fill(~query_valid, 0.0)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask,
                zero_invalid_rows=zero_invalid_rows)
            if query_valid is not None:
                dec_output = dec_output.masked_fill(~query_valid, 0.0)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output, dec_enc_attn_list
