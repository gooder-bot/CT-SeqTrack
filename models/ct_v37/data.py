"""v37 复用 v35 W 原帧排程；唯一提交仍由旧 accepted 状态机完成。

新候选只收紧本帧可写入的真实点权限，不引入事件时间或跨窗口状态。
"""
import torch
from utils.raw_frame_memo import raw_transaction

from models.ct_v31.data import (BatchBuilder as LegacyBatchBuilder, build_loaders as legacy_loaders,
    ReadyQueueBatchSampler, RawEndpointDataset, EndpointRequest, AcceptedState,
    stable_seed, raw_collate, box_array, box_size, point_arrays, inside_box, metadata)
from .config import normalize_config, as_v35_config


def build_loaders(config, roles=('train', 'val'), sources=None):
    config = normalize_config(config)
    return legacy_loaders(as_v35_config(config, arm=config.v37_arm), roles=roles, sources=sources)


build_dataloaders = build_loaders


class BatchBuilder(LegacyBatchBuilder):
    def __init__(self, config):
        self.v37_config = normalize_config(config)
        super().__init__(as_v35_config(self.v37_config, arm=self.v37_config.v37_arm))

    @raw_transaction
    def commit(self, output, batch, *, diagnostics=False):
        # B0/B1 完整复用原数值及状态数学，不增加推理门控。
        if self.v37_config.v37_arm in ('b0', 'b1'):
            return super().commit(output, batch, diagnostics=diagnostics)
        if self._pending is None:
            raise RuntimeError('no pending v37 transaction (duplicate commit)')
        selected = output.selected_index.detach().long()
        fallback = output.q0_is_fallback.detach().bool()
        if selected.shape != fallback.shape or bool(((selected < 0) | (selected > 3)).any()):
            raise ValueError('v37 selection must index q0/prior/two modes')
        if self.v37_config.v37_arm == 'b1_b2' and bool((selected != 0).any()):
            raise ValueError('v37 shadow B2 must commit q0')
        forbidden = (selected == 1) | ((selected == 0) & fallback)
        quality = output.selected_quality.detach()
        if bool((quality[forbidden] != 0).any()):
            raise ValueError('prior/fallback cannot carry commit quality')
        rows = torch.arange(len(selected), device=selected.device)
        eligible = output.candidates.valid.detach()[rows, selected]
        shadow_fallback = (self.v37_config.v37_arm == 'b1_b2') & (selected == 0) & fallback
        if not bool((eligible | shadow_fallback).all()):
            raise ValueError('v37 cannot commit an invalid candidate')
        expected_quality = output.commit_quality.detach()[rows, selected]
        if not bool(torch.isfinite(quality).all()) or not torch.equal(quality, expected_quality):
            raise ValueError('selected quality differs from candidate commit quality')
        if not torch.equal(output.accepted_box.detach(),
                output.candidates.boxes.detach()[rows, selected]):
            raise ValueError('accepted box differs from the unique selected candidate')
        # 旧 commit 只读取这份瞬时权限视图；原 batch 与候选台账保持完整。
        prepared = dict(batch)
        valid = batch['point_valid'].clone()
        valid[:, -1] &= ~forbidden[:, None]
        prepared['point_valid'] = valid
        extension_valid = torch.zeros_like(batch['extension_valid'], dtype=torch.bool)
        evidence = output.evidence
        if self.v37_config.v37_arm == 'full':
            for row in range(len(selected)):
                index = int(selected[row])
                if index >= 2:
                    members = evidence.members[row, index - 2].bool() & evidence.point_valid[row].bool()
                    if not bool(members.any()) or not bool(evidence.mode_valid[row, index - 2]):
                        raise ValueError('selected mode requires its own formed members')
                    slots = evidence.point_indices[row, members].long()
                    if bool(((slots < 0) | (slots >= extension_valid.shape[1])).any()):
                        raise ValueError('mode support must refer to selected acquisition slots')
                    extension_valid[row, slots] = True
        prepared['extension_valid'] = extension_valid & batch['extension_valid'].bool()
        contexts = list(self._pending)
        result = super().commit(output, prepared, diagnostics=diagnostics)
        if diagnostics:
            selected_cpu = selected.cpu().tolist()
            quality_cpu = quality.cpu().tolist()
            forbidden_cpu = forbidden.cpu().tolist()
            for i, record in enumerate(result):
                record.update(commit_source=('q0', 'prior', 'mode0', 'mode1')[selected_cpu[i]],
                    commit_quality=float(quality_cpu[i]), propagation_only=bool(forbidden_cpu[i]),
                    support_policy='current_b0_only' if selected_cpu[i] == 0 else
                        'none' if selected_cpu[i] == 1 else 'current_b0_plus_selected_mode_members')
                state = contexts[i]['state']
                record['memory_source_index'] = selected_cpu[i] if record['memory_write'] else None
                record['memory_written_raw_ids'] = (state.memory.recent[-1]['ids'].tolist()
                    if record['memory_write'] else [])
                record['memory_written_foreground'] = (state.memory.recent[-1]['fg'].tolist()
                    if record['memory_write'] else [])
                record['trusted_write'] = record['supported']
                if forbidden_cpu[i] and (record['memory_write'] or record['supported']):
                    raise RuntimeError('propagation-only candidate wrote trusted state')
        return result
