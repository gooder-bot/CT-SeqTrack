"""v35 被动训练统计：全量轻量摘要、固定 1/16 端点 IoU，无额外 forward。

只消费 detached 的已准备输入与输出。桶按 epoch/branch/零起点 depth/
oldest-first 三槽 provenance 定义；计数不改变采样、损失或 accepted。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .data import stable_seed
from .losses import observation_loss_statistics
from utils.tracking_metrics import LocalYawBox, box_metrics


LOCAL_FIELDS = ('local_neighbor_count', 'local_selected_count',
                'local_mean_support', 'local_delta_norm')


class TrainingDiagnostics:
    SCHEMA = 'ct_seqtrack.v35.training_diagnostics.v1'

    def __init__(self, seed=42):
        self.seed = int(seed)
        self.reset(0)

    def reset(self, epoch):
        self.epoch = int(epoch)
        self.groups = {}
        self.totals = self._bucket()
        self.batch_losses = {}
        self.batch_loss_valid_counts = {}
        self.batches = 0
        self.sampled_geometry = []

    @staticmethod
    def _bucket():
        return dict(endpoints=0, statistics={}, loss_statistics={})

    @staticmethod
    def _accumulate(bucket, name, value):
        value = np.asarray(value, dtype=np.float64)
        if not np.isfinite(value).all():
            raise FloatingPointError('non-finite v35 training diagnostic: ' + name)
        stats = bucket['statistics'].setdefault(name, dict(sum=np.zeros_like(value), count=0))
        stats['sum'] += value
        stats['count'] += 1

    @staticmethod
    def _cpu(value):
        return value.detach().cpu().numpy()

    @torch.no_grad()
    def add_batch(self, rows, batch, output, losses):
        """必须在本 batch 的已有 forward/loss 之后调用；不持有其计算图。"""
        count = len(rows)
        self.batches += 1
        observation, decoder = output.observation, output.decoder
        target = batch['target_box'].detach()
        boxes = dict(coarse=observation.coarse_box.detach(), fine=decoder.hypothesis_boxes[:, 0].detach())
        arrays = {key: self._cpu(value) for key, value in batch.items()
                  if key.startswith('diagnostic_') and torch.is_tensor(value)}
        valid = batch['point_valid'].bool()
        labels = batch['segmentation_labels']
        arrays.update(point_count=self._cpu(valid.sum(-1)),
                      sampled_target_count=self._cpu(((labels == 1) & valid).sum(-1)),
                      history_exists=self._cpu(batch['history_valid']),
                      history_is_initial=self._cpu(batch['history_is_initial']))
        sampled_targets, real_points = arrays['sampled_target_count'], arrays['point_count']
        sequence_empty = real_points.sum(-1) == 0
        arrays.update(sequence_empty=sequence_empty,
            all_sampled_four_frames_background=(~sequence_empty & (sampled_targets.sum(-1) == 0)),
            current_sampled_missing_history_has_target=((sampled_targets[:, -1] == 0)
                & (sampled_targets[:, :-1].sum(-1) > 0)))
        if 'diagnostic_target_count' in arrays and 'diagnostic_crop_target_count' in arrays:
            raw_missing = arrays['diagnostic_target_count'] == 0
            arrays.update(current_raw_missing=raw_missing,
                raw_present_but_crop_miss=(~raw_missing & (arrays['diagnostic_crop_target_count'] == 0)))
        for name in ('history_support', 'current_support'):
            value = getattr(observation, name, None)
            if value is not None:
                arrays[name] = self._cpu(value)
        for name in LOCAL_FIELDS + ('query_context_norm',):
            value = getattr(decoder, name, None)
            if value is not None:
                arrays[name] = self._cpu(value)
        active = self._cpu(observation.sequence_valid.bool())
        for name, box in boxes.items():
            error = box[:, :3] - target[:, :3]
            arrays[name + '_center_error_xyz_m'] = self._cpu(error)
            arrays[name + '_center_error_m'] = self._cpu(error.norm(dim=-1))
            arrays[name + '_center_xy_error_m'] = self._cpu(error[:, :2].norm(dim=-1))
            arrays[name + '_absolute_z_error_m'] = self._cpu(error[:, 2].abs())
            angle = box[:, 3] - target[:, 3]
            arrays[name + '_yaw_error_rad'] = self._cpu(torch.atan2(angle.sin(), angle.cos()).abs())
        provenance = self._cpu(batch['history_box_provenance']).astype(np.int64)
        detached_losses = observation_loss_statistics(batch, output)
        detached_losses.update(getattr(output, 'loss_statistics', {}))
        endpoint_losses = {name: {key: self._cpu(value) for key, value in terms.items()}
                           for name, terms in detached_losses.items()}
        # 真实 batch 标量按行数归约，与训练日志的 batch_size 权重相同；
        # 不把 Full 任务的 batch 均值错误分派给某一 provenance 桶。
        for name, value in losses.items():
            if name.startswith('loss_') and torch.is_tensor(value) and value.ndim == 0:
                item = self.batch_losses.setdefault(name, dict(row_weighted_sum=0., rows=0, batch_sum=0.))
                scalar = float(value.detach().cpu())
                item['row_weighted_sum'] += scalar * count
                item['rows'] += count
                item['batch_sum'] += scalar
        quality_valid = decoder.hypothesis_valid.detach().bool()
        valid_counts = dict(main_quality=int(quality_valid[:, 0].sum()),
            mode_quality=int(quality_valid[:, 1:].sum()),
            coarse_and_main_endpoints=int(observation.sequence_valid.sum()),
            history_box_frames=int(batch['history_valid'].sum()),
            current_real_points=int(valid[:, -1].sum()), history_real_points=int(valid[:, :-1].sum()),
            current_observed_frames=int(valid[:, -1].any(-1).sum()),
            history_observed_frames=int(valid[:, :-1].any(-1).sum()))
        if 'physical_valid' in batch:
            valid_counts['physical_and_sigma_endpoints'] = int((batch['physical_valid'].bool() & output.prior.valid).sum())
        if 'acquisition_valid' in batch:
            acquisition = batch['acquisition_valid'].bool()
            demand = batch.get('acquisition_demand', acquisition).bool()
            valid_counts.update(acquisition_demand_endpoints=int((acquisition & demand).sum()),
                                acquisition_no_demand_endpoints=int((acquisition & ~demand).sum()))
        if 'extension_labels' in batch:
            extension = batch['extension_valid'].bool()
            foreground = batch['extension_labels'] > .5
            evidence = output.evidence
            selected_fg = foreground.gather(1, evidence.point_indices.clamp_min(0)) & evidence.point_valid
            member_valid = evidence.members.bool() & evidence.point_valid[:, None]
            member_fg = (member_valid & selected_fg[:, None]).sum(-1)
            positive = (member_fg > 0) & (member_fg >= .5 * member_valid.sum(-1)) & evidence.mode_valid
            valid_counts.update(identity_foreground_points=int((extension & foreground).sum()),
                identity_background_points=int((extension & ~foreground).sum()),
                vote_foreground_points=int(selected_fg.sum()),
                reliability_foreground_points=int(selected_fg.sum()),
                reliability_background_points=int((evidence.point_valid & ~selected_fg).sum()),
                positive_mode_boxes=int((positive & quality_valid[:, 1:]).sum()))
        for name, value in valid_counts.items():
            self.batch_loss_valid_counts[name] = self.batch_loss_valid_counts.get(name, 0) + value
        sampled = []
        for i, row in enumerate(rows):
            request = row['request']
            if request.epoch != self.epoch:
                raise ValueError('training diagnostic epoch differs from the raw request')
            origin = tuple(int(x) for x in provenance[i])
            depth = int(request.frame - request.window_start)
            key = f'b{request.branch}/d{depth}/p' + ','.join(map(str, origin))
            group = self.groups.setdefault(key, dict(branch=int(request.branch), depth=depth,
                history_provenance=list(origin), **self._bucket()))
            for bucket in (group, self.totals):
                bucket['endpoints'] += 1
                self._accumulate(bucket, 'coarse_valid', active[i])
                for name, value in arrays.items():
                    if name.startswith('coarse_') and not active[i]:
                        continue  # 全序列空点占位框不作为真实 coarse 预测。
                    self._accumulate(bucket, name, value[i])
                for name, terms in endpoint_losses.items():
                    entry = bucket['loss_statistics'].setdefault(name,
                        dict(numerator_sum=0., valid_endpoints=0., actual_batch_contribution_sum=0.))
                    entry['numerator_sum'] += float(terms['numerator'][i])
                    entry['valid_endpoints'] += float(terms['denominator'][i])
                    entry['actual_batch_contribution_sum'] += float(terms['batch_contribution'][i])
            if stable_seed(self.seed, self.epoch, row['tracklet_key'], request.branch,
                           request.window_start, request.frame, 'v35_training_iou') % 16 == 0:
                sampled.append((i, row, key))
        if sampled:
            # 只有固定 1/16 样本传输框并执行 CPU 旋转几何；不另跑网络。
            indices = torch.tensor([i for i, _, _ in sampled], device=target.device)
            truth = self._cpu(target.index_select(0, indices))
            sizes = self._cpu(batch['box_size'].index_select(0, indices))
            target_sizes = self._cpu(batch['target_box_size'].index_select(0, indices))
            predicted = {name: self._cpu(value.index_select(0, indices)) for name, value in boxes.items()}
            for j, (i, row, key) in enumerate(sampled):
                request = row['request']
                record = dict(tracklet=row['tracklet_key'], frame=int(request.frame),
                              branch=int(request.branch), depth=int(request.frame - request.window_start), group=key)
                for name, values in predicted.items():
                    if name == 'coarse' and not active[i]:
                        record[name + '_iou'] = None
                        continue
                    overlap = box_metrics(LocalYawBox(values[j], sizes[j][[1, 0, 2]]),
                        LocalYawBox(truth[j], target_sizes[j][[1, 0, 2]]),
                        up_axis=(0, 0, 1), mode='geometry_exact')[0]
                    record[name + '_iou'] = overlap
                    for bucket in (self.groups[key], self.totals):
                        self._accumulate(bucket, 'sampled_' + name + '_iou', overlap)
                self.sampled_geometry.append(record)

    @staticmethod
    def _serialize_bucket(bucket):
        result = {key: value for key, value in bucket.items() if key != 'statistics'}
        result['statistics'] = {name: dict(sum=entry['sum'].tolist(), count=entry['count'],
            mean=(entry['sum'] / entry['count']).tolist()) for name, entry in sorted(bucket['statistics'].items())}
        return result

    def summary(self, *, completed_epoch=None, complete=True):
        return dict(schema=self.SCHEMA, seed=self.seed, epoch_index=self.epoch,
            completed_epoch=self.epoch + 1 if completed_epoch is None else int(completed_epoch),
            epoch_complete=bool(complete), batches=self.batches,
            grouping='epoch/branch/depth_zero_based/history_provenance_oldest_first',
            provenance={'0': 'missing', '1': 'known_gt_seed', '2': 'perturbed_seed', '3': 'accepted_prediction'},
            sampling_policy='sha256_endpoint_mod16_v1', iou_sample_denominator=16,
            iou_sample_endpoints=len(self.sampled_geometry),
            iou_sample_fraction=(len(self.sampled_geometry) / self.totals['endpoints']
                                 if self.totals['endpoints'] else 0.),
            loss_semantics=dict(grouped='observation endpoint numerator/valid endpoint denominator; seg 0.1 and main_quality 0.5 coefficients included',
                actual_batch_contribution_sum='sum of endpoint contributions using each original batch denominator; divide by batches for batch-mean objective',
                batch='actual scalar losses; row_weighted_sum/rows matches logged epoch weighting; Full auxiliary losses have no endpoint attribution'),
            totals=self._serialize_bucket(self.totals),
            groups={key: self._serialize_bucket(value) for key, value in sorted(self.groups.items())},
            batch_losses=self.batch_losses, batch_loss_valid_counts=self.batch_loss_valid_counts,
            sampled_geometry=self.sampled_geometry)

    def write_epoch(self, log_dir, *, completed_epoch=None, complete=True):
        if not log_dir:
            return None
        report = self.summary(completed_epoch=completed_epoch, complete=complete)
        directory = Path(log_dir) / 'training_diagnostics'
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"epoch{report['completed_epoch']:03d}.json"
        content = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n'
        if path.exists():
            if path.read_text(encoding='utf-8') != content:
                raise FileExistsError('existing v35 training diagnostic differs: ' + str(path))
        else:
            # 独占创建；相同已完成 epoch 可读取核对，不覆盖历史证据。
            with path.open('x', encoding='utf-8') as handle:
                handle.write(content)
        return path
