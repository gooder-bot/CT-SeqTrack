''' Define the seq2seq model '''
import torch
import torch.nn as nn
from .Layers import EncoderLayer, DecoderLayer
import torch.nn.functional as F

from .pytorch_utils import Seq

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

class Seq2SeqFormer(nn.Module):
    """
    A sequence-to-sequence transformer model that facilitates deep interaction between 
    point cloud sequences and bounding box (bbox) sequences through an attention-based mechanism.
    This leverages the inherent spatial and temporal relationships within the sequences to 
    enhance feature representation for tasks involving point clouds and their associated bounding boxes.
    """

    def __init__(
            self, src_pad_idx=1, trg_pad_idx=1,
            d_word_vec=64, d_model=64, d_inner=512,
            n_layers=3, n_head=8, d_k=32, d_v=32, dropout=0.2, n_position=100):

        super().__init__()
        
        self.d_model=d_model
        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx
        self.proj=nn.Linear(128,d_model) 
        self.proj2=nn.Linear(4,d_model) # 4 represents the dimensions for x, y, z, plus a time stamp
        self.l1=nn.Linear(d_model*8, d_model)
        self.l2=nn.Linear(d_model, 4)

        self.dropout = nn.Dropout(p=dropout)

        self.encoder = Encoder(
            n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout)
        
        self.encoder_global = Encoder(
            n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=src_pad_idx, dropout=dropout)

        self.decoder = Decoder(
            n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=trg_pad_idx, dropout=dropout)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

    def forward(
            self, trg_seq, src_seq, valid_mask,
            return_decoder_state=False, *, enable_v29=False,
            frame_measurement_valid=None, source_token_valid=None):
        """Run SeqTrack3D's box-sequence decoder.

        ``return_decoder_state`` is deliberately opt-in so every historical
        caller keeps receiving the exact box tensor.  The exposed state is
        the per-box representation after ``l1`` and immediately before the
        unchanged regression layer ``l2``. v29 requires true frame-aligned
        source tokens and sampled measurement flags [B,L]; absence is never
        inferred from XYZ values. Historical callers keep their original path.
        """

        src_seq_=self.proj(src_seq) # Adjust the input features to 128 dimensions
        trg_seq_=self.proj2(trg_seq) # Also adjust Q to 128 dimensions, corresponding to the features of the input box

        if trg_seq_.shape[1] % 8 != 0:
            raise ValueError("Transformer target tokens must contain 8 corners per frame")
        frame_count = trg_seq_.shape[1] // 8
        if frame_count <= 0 or src_seq_.shape[1] % frame_count != 0:
            raise ValueError(
                "Transformer source tokens must divide evenly across target frames")
        tokens_per_frame = src_seq_.shape[1] // frame_count

        masks = None
        if enable_v29:
            batch = src_seq_.shape[0]
            if tokens_per_frame <= 0:
                raise ValueError('v29 requires nonempty source token slots per frame')
            if valid_mask.shape != (batch, frame_count - 1):
                raise ValueError('v29 history existence mask must be [B,L-1]')
            if (frame_measurement_valid is None
                    or frame_measurement_valid.shape != (batch, frame_count)):
                raise ValueError('v29 requires sampled measurement validity [B,L]')
            frame_exists = torch.cat((
                valid_mask.to(device=src_seq_.device, dtype=torch.bool),
                torch.ones((batch, 1), device=src_seq_.device, dtype=torch.bool)), dim=1)
            frame_source_valid = frame_exists & frame_measurement_valid.to(
                device=src_seq_.device, dtype=torch.bool)
            source_valid = frame_source_valid.repeat_interleave(tokens_per_frame, dim=1)
            if source_token_valid is not None:
                if source_token_valid.shape != source_valid.shape:
                    raise ValueError('v30 source token validity must match source slots')
                source_valid = source_valid & source_token_valid.to(source_valid.device).bool()
            corner_valid = frame_exists.repeat_interleave(8, dim=1)
            source_pair_mask = source_valid.unsqueeze(2) & source_valid.unsqueeze(1)
            cross_source_valid = torch.cat((source_valid, source_valid), dim=1)
            masks = dict(
                local=frame_source_valid.reshape(batch * frame_count, 1, 1),
                global_keys=source_pair_mask,
                global_queries=source_valid.unsqueeze(-1),
                cross=corner_valid.unsqueeze(2) & cross_source_valid.unsqueeze(1),
                corners=corner_valid.unsqueeze(-1), frames=frame_exists.unsqueeze(-1))
            local_valid = source_valid.reshape(batch * frame_count, tokens_per_frame)
            masks['local_queries'] = (local_valid.unsqueeze(-1) if source_token_valid is not None
                                       else masks['local'])
            if source_token_valid is not None:
                masks['local'] = local_valid.unsqueeze(2) & local_valid.unsqueeze(1)

        enc_output, *_ = self.encoder(
            src_seq_.reshape(-1, tokens_per_frame, self.d_model),
            **(dict(src_mask=masks['local'], query_valid=masks['local_queries'],
                    zero_invalid_rows=True) if masks is not None else {}))

        enc_others,*_=self.encoder_global(
            src_seq_, global_feature=True,
            **(dict(src_mask=masks['global_keys'], query_valid=masks['global_queries'],
                    zero_invalid_rows=True) if masks is not None else {}))

        # Implementing cross-decoder
        # Q: trg_seq_
        # K, V: Concatenate(enc_output, enc_others)
        enc_output=torch.cat([
            enc_output.reshape(
                src_seq_.shape[0], frame_count * tokens_per_frame,
                self.d_model),
            enc_others,
        ], dim=1)
        # 原 decoder 只有 cross-attention；corner mask 屏蔽 query，不新增 self-attention。
        dec_output, dec_attention,*_ = self.decoder(
            trg_seq_, None, enc_output, masks['cross'] if masks is not None else None,
            **(dict(query_valid=masks['corners'], zero_invalid_rows=True)
               if masks is not None else {}))
                                                

        # Project to output
        dec_output=dec_output.reshape(
            dec_output.shape[0], frame_count, self.d_model * 8)
        decoder_state = self.l1(dec_output)
        dec_output = self.l2(decoder_state)
        if masks is not None:
            decoder_state = decoder_state.masked_fill(~masks['frames'], 0.0)
            dec_output = dec_output.masked_fill(~masks['frames'], 0.0)

        if return_decoder_state:
            return dec_output, decoder_state
        return dec_output
