"""窗口对照的配对检查与训练观测，不向模型添加输入或损失。"""
from collections import Counter
import hashlib

import numpy as np
import torch

from .base_losses import _base_q0
from .data import PERTURBED_BOX
from .geometry import rotate_xyz
from .input import JOINT_ROW_FIELDS, LOSS_FIELDS, ONLINE_FIELDS


def _hash_array(digest, key, value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.ascontiguousarray(value)
    digest.update(repr((key, array.dtype.str, array.shape)).encode())
    digest.update(array.tobytes() if not array.size else memoryview(array).cast('B'))


class HistoryTrainingAudit:
    GROUPS = ('current_target', 'history_target_only', 'background_only', 'empty')

    def __init__(self, epoch):
        self.common = epoch <= 2
        self.batches = hashlib.sha256()
        self.losses = hashlib.sha256()
        self.rows = Counter()
        self.reset_seeds = Counter()
        self.gradient_sums = None
        self.motion_counts = Counter()
        self.motion_by_source = {}

    def prepared(self, rows, batch, stream):
        main = stream == 'main'
        if main:
            sources = []
            latest_seed = batch['history_box_provenance'][:, -1].tolist()
            for row, provenance in zip(rows, latest_seed):
                r = row['request']
                kind = ('natural_replay' if '_joint_replay_state' in row else
                        'reset' if r.starts_window else 'continuation')
                self.rows[kind] += 1
                if kind == 'reset':
                    kind = 'perturbed_seed' if provenance == PERTURBED_BOX else 'clean_seed'
                    self.reset_seeds['first_prediction_gt' if r.frame == 1 else kind] += 1
                sources.append(kind)
            # 只对 B1 实际可读的两条新鲜运动边计算 8 m/s² 裁剪分母。
            boxes = batch['joint_motion_boxes'].numpy()[:, :, :2]
            times = batch['joint_motion_times'].numpy()
            gap = np.diff(times, axis=1)
            eligible = (batch['joint_motion_pairs'].numpy().all(1)
                        & batch['joint_motion_valid'].numpy().all(1) & (gap > 0).all(1)
                        & np.isclose(-times[:, -1], batch['current_dt'].numpy(), rtol=1e-5, atol=1e-6))
            gap = gap[eligible].clip(min=.001)
            velocity = np.diff(boxes[eligible], axis=1) / gap[:, :, None]
            acceleration = np.linalg.norm(velocity[:, 1] - velocity[:, 0], axis=1) / gap.mean(axis=1)
            self.motion_counts['eligible'] += len(acceleration)
            self.motion_counts['clipped'] += int(np.count_nonzero(acceleration >= 8.))
            for source, norm in zip(np.asarray(sources)[eligible], acceleration):
                counts = self.motion_by_source.setdefault(str(source), Counter())
                counts['eligible'] += 1
                counts['clipped'] += int(norm >= 8.)
        if not self.common:
            return
        keys = []
        for row in rows:
            r = row['request']
            keys.append((r.epoch, row['tracklet_key'], r.frame, r.branch, r.drain))
        self.batches.update(repr((stream, keys)).encode())
        for name in sorted((ONLINE_FIELDS | LOSS_FIELDS).intersection(batch)):
            values = batch[name]
            if name == 'joint_rows':
                for row in values:
                    for field in JOINT_ROW_FIELDS:
                        _hash_array(self.batches, field, row[field])
            elif name == 'joint_other_cars':
                for cars in values:
                    _hash_array(self.batches, name, cars)
            else:
                _hash_array(self.batches, name, values)

    def loss(self, value):
        if self.common:
            _hash_array(self.losses, 'combined_loss', value)

    @torch.no_grad()
    def localization(self, batch, output):
        # uniform 主路 2*SmoothL1(XYZ).mean 的框输出梯度。旋转不改变 L2 范数。
        valid = batch['point_valid'].bool()
        positives = (batch['segmentation_labels'] == 1) & valid
        current = positives[:, -1].any(-1)
        history = positives[:, :-1].flatten(1).any(-1)
        real = valid.flatten(1).any(-1)
        active = output.observation.sequence_valid.bool()
        target = batch['target_box'][:, :3]
        columns = [torch.ones_like(current, dtype=target.dtype)]
        for box in (output.observation.coarse_box, _base_q0(output.decoder)):
            residual = rotate_xyz(box[:, :3] - target, batch['anchor_box'][:, 3])
            norm = residual.clamp(-1., 1.).norm(dim=-1) * (2. / 3.)
            columns.append(torch.where(active, norm, 0.) / active.sum().clamp_min(1))
        groups = torch.stack((current, ~current & history, real & ~current & ~history, ~real), dim=1)
        sums = (groups[:, :, None] * torch.stack(columns, dim=1)[:, None, :]).sum(0, dtype=torch.float64)
        if self.gradient_sums is None:
            self.gradient_sums = sums
        else:
            self.gradient_sums += sums

    def report(self, tracker, builder):
        gradients = {} if self.gradient_sums is None else {
            group: dict(rows=int(row[0]), coarse_center_grad_norm_sum=row[1], q0_center_grad_norm_sum=row[2])
            for group, row in zip(self.GROUPS, self.gradient_sums.cpu().tolist())}
        def clip_rate(counts):
            n = counts['eligible']
            return dict(eligible=n, clipped=counts['clipped'], fraction=counts['clipped'] / n if n else None)

        result = dict(endpoints=dict(self.rows), ordinary_reset_seeds=dict(self.reset_seeds),
                      augmentation=dict(builder.history_counts),
                      acceleration=clip_rate(self.motion_counts),
                      acceleration_by_source={k: clip_rate(v) for k, v in self.motion_by_source.items()},
                      localization=gradients)
        if self.common:
            from .fixed_replay import weights_sha256
            result.update(batch_sha256=self.batches.hexdigest(), loss_sha256=self.losses.hexdigest(),
                          weights_sha256=weights_sha256(tracker))
        return result
