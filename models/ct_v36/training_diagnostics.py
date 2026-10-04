"""训练端点的被动聚合；不重跑网络或改变 loss/状态。"""
import json
from pathlib import Path


class TrainingDiagnostics:
    def __init__(self, seed=42):
        self.seed = int(seed)
        self.reset(0)

    def reset(self, epoch):
        self.epoch, self.rows, self.steps = int(epoch), 0, 0
        self.loss_sums, self.branches = {}, {str(i): 0 for i in range(4)}
        self.measurements = self.disagreements = self.memory_eligible = 0
        self.motion_valid = self.range_clipped = 0

    def add_batch(self, rows, batch, output, losses):
        self.steps += 1
        self.rows += len(rows)
        for row in rows:
            key = str(row['request'].branch)
            self.branches[key] = self.branches.get(key, 0) + 1
        for key, value in losses.items():
            self.loss_sums[key] = self.loss_sums.get(key, 0.) + float(value.detach())
        measurement = output.decision.measurement_index.detach()
        self.measurements += int((measurement >= 0).sum())
        self.disagreements += int(((measurement >= 0) & (measurement != output.selected_index)).sum())
        self.memory_eligible += int(output.decision.memory_eligible.sum())
        self.motion_valid += int(output.prior.motion_valid.sum())
        bound = output.prior.range_xy.new_tensor([12., 8.])
        self.range_clipped += int((output.prior.range_xy > bound).any(-1).sum())

    def summary(self, complete=True):
        return dict(schema='ct_seqtrack.v36.training_diagnostics.v1', epoch=self.epoch,
            complete=bool(complete), rows=self.rows, optimizer_steps=self.steps,
            endpoint_exposure_by_branch=self.branches, accepted_measurements=self.measurements,
            output_measurement_disagreements=self.disagreements, memory_eligible=self.memory_eligible,
            motion_valid=self.motion_valid, range_exceeds_search_cap=self.range_clipped,
            loss_batch_means={k: v / max(1, self.steps) for k, v in self.loss_sums.items()})

    def write_epoch(self, directory, *, completed_epoch, complete):
        if not directory:
            return
        target = Path(directory) / 'training_diagnostics' / f'epoch={completed_epoch:03d}.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(self.summary(complete), ensure_ascii=False, sort_keys=True, indent=2) + '\n'
        if target.exists():
            if target.read_text(encoding='utf-8') != content:
                raise FileExistsError('existing training diagnostics differ: ' + str(target))
        else:
            target.write_text(content, encoding='utf-8')
