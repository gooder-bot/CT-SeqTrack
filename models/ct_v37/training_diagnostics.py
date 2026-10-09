"""v37 保留原 branch/depth/provenance 分桶并记录恢复候选供给。

只复用已前向/监督的 detached 张量；统计不能改变权重、采样或提交。
"""
from copy import copy

import torch

from models.ct_v31.training_diagnostics import TrainingDiagnostics as LegacyDiagnostics
from utils.diagnostic_transfer import tensors_to_numpy


class TrainingDiagnostics(LegacyDiagnostics):
    SCHEMA = 'ct_seqtrack.v37.training_diagnostics.v1'

    @torch.no_grad()
    def add_batch(self, rows, batch, output, losses):
        view = output
        if getattr(output, 'candidates', None) is not None:
            # 原q0诊断使用旧布局[q0,mode0,mode1,pad]，补齐证据的无效第三槽。
            view = copy(output)
            view.evidence = copy(output.evidence)
            for field in ('mode_valid', 'members', 'centers_xyz', 'covariance_xy'):
                value = getattr(view.evidence, field)
                if value.shape[1] == 2:
                    setattr(view.evidence, field, torch.cat((value, torch.zeros_like(value[:, :1])), 1))
        super().add_batch(rows, batch, view, losses)
        candidates = getattr(output, 'candidates', None)
        if candidates is None:
            return
        evidence = output.evidence
        selected_labels = batch['extension_labels'].gather(1, evidence.point_indices.clamp_min(0))
        foreground = (selected_labels > .5) & evidence.point_valid.bool()
        errors = (evidence.vote_xyz.detach() - batch['target_box'][:, None, :3]).norm(dim=-1)
        members = evidence.members.bool() & evidence.point_valid[:, None].bool()
        member_fg = (members & foreground[:, None]).sum(-1)
        target_modes = evidence.mode_valid.bool() & (member_fg >= 1) & (member_fg >= .5 * members.sum(-1))
        values = dict(v37_extension_foreground_count=foreground.sum(-1),
            v37_target_mode_count=target_modes.sum(-1),
            v37_valid_mode_count=candidates.valid[:, 2:].sum(-1),
            v37_output_index=output.selected_index,
            v37_output_commit_quality=output.selected_quality,
            v37_output_utility=output.output_utility,
            v37_q0_is_fallback=output.q0_is_fallback,
            v37_foreground_vote_error_sum=torch.where(foreground, errors, 0.).sum(-1),
            v37_foreground_vote_error_count=foreground.sum(-1))
        targets = getattr(output, 'supervision_targets', {})
        for name in ('exact_iou', 'utility', 'commit_quality', 'measurement', 'support_count', 'support_purity', 'foreground_count'):
            value = targets.get(name) if isinstance(targets, dict) else None
            if value is not None:
                values['v37_candidate_' + name] = value
        for name in ('raw_pose', 'pose_raw', 'pose_output', 'mode_residual', 'q0_pose_raw',
                     'physical_context_norm'):
            value = getattr(candidates, name, None)
            if value is not None:
                values['v37_' + name] = value.detach()
        if candidates.physical_context is not None:
            values['v37_physical_context_norm'] = candidates.physical_context.detach().norm(dim=-1)
        values['history_box_provenance'] = batch['history_box_provenance']
        values = tensors_to_numpy(values, optimized=self.optimized)
        provenance = values.pop('history_box_provenance')
        for i, row in enumerate(rows):
            r = row['request']
            key = f'b{r.branch}/d{r.frame - r.window_start}/p' + ','.join(str(int(v)) for v in provenance[i])
            for bucket in (self.groups[key], self.totals):
                for name, value in values.items():
                    self._accumulate(bucket, name, value[i])

    def summary(self, *, completed_epoch=None, complete=True):
        report = super().summary(completed_epoch=completed_epoch, complete=complete)
        report['loss_semantics']['grouped'] = (
            'original observation and explicitly supplied auxiliary endpoint numerators/valid denominators; '
            'actual contributions use each original batch denominator')
        report['loss_semantics']['auxiliary_attribution'] = (
            'only loss_statistics emitted by the loss implementation are attributed; no scalar loss is '
            'fabricated into a per-endpoint contribution')
        return report
