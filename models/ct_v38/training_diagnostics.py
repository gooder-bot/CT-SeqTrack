"""沿用强 B1 的训练分桶，另记 B3 监督与输出；统计不影响递推。"""
import torch

from models.ct_v31.training_diagnostics import TrainingDiagnostics as LegacyDiagnostics
from utils.diagnostic_transfer import tensors_to_numpy


class TrainingDiagnostics(LegacyDiagnostics):
    SCHEMA = 'ct_seqtrack.v38.training_diagnostics.v1'

    @torch.no_grad()
    def add_batch(self, rows, batch, output, losses):
        super().add_batch(rows, batch, output, losses)
        from .verification import current_q0_support
        q0_support = current_q0_support(output.observation, batch, output.candidates.boxes[:, 0])
        values = dict(v38_selected_index=output.selected_index,
                      v38_q0_real=output.candidates.raw_valid[:, 0],
                      v38_q0_support_count=q0_support['count'],
                      v38_selected_commit_support_count=output.commit_decision.support_mask.sum(-1),
                      v38_trusted_eligible=output.commit_decision.eligible,
                      history_box_provenance=batch['history_box_provenance'])
        if output.verification is not None:
            values['v38_utility'] = output.verification.utility
            values['v38_q0_trusted_score'] = output.verification.logits[:, 0, 2].sigmoid()
        for name, value in getattr(output, 'supervision_targets', {}).items():
            if torch.is_tensor(value) and value.shape[:1] == output.selected_index.shape:
                values['v38_target_' + name] = value
        values = tensors_to_numpy(values, optimized=self.optimized)
        provenance = values.pop('history_box_provenance')
        for i, row in enumerate(rows):
            request = row['request']
            key = f'b{request.branch}/d{request.frame - request.window_start}/p' + ','.join(
                str(int(value)) for value in provenance[i])
            for bucket in (self.groups[key], self.totals):
                for name, value in values.items():
                    self._accumulate(bucket, name, value[i])
