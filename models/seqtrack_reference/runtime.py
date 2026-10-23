"""独立参考的共同指标与严格完整 epoch 恢复；不导入生产网络。"""
import hashlib
import json
import numpy as np
import torch

from .config import normalize_config, config_identity
from .identity import model_schema, runtime_schema
from models.ct_v38.runtime import capture_rng_state, restore_rng_state, move_tensors, loss_episode_summary
from utils.tracking_metrics import LocalYawBox, box_metrics, metric_contributions



def resume_payload(config, *, completed_epoch, complete, rows, steps, sampler=None, run_id=None):
    from .entry import source_identity
    return dict(schema=runtime_schema(config), model_schema=model_schema(config),
        config_sha256=config_identity(config), completed_epoch=int(completed_epoch),
        epoch_complete=bool(complete), rows=int(rows), optimizer_steps=int(steps),
        sampler=sampler, rng=capture_rng_state(), source=source_identity(), run_id=run_id)


def validate_resume_payload(payload, config, *, training=True):
    from .entry import source_identity
    if not isinstance(payload, dict) or payload.get('schema') != runtime_schema(config):
        raise ValueError('reference runtime schema mismatch')
    if payload.get('model_schema') != model_schema(config) or payload.get('config_sha256') != config_identity(config):
        raise ValueError('reference model/config identity mismatch')
    if payload.get('source') != source_identity():
        raise ValueError('reference source identity mismatch; use the original source')
    if not isinstance(payload.get('run_id'), str) or not payload['run_id']:
        raise ValueError('reference requires a nonempty run_id')
    if payload.get('epoch_complete') is not True or payload.get('completed_epoch', 0) < 1:
        raise ValueError('reference requires complete epoch boundaries')
    sampler = payload.get('sampler')
    if not isinstance(sampler, dict) or sampler.get('schema') != 'seqtrack_reference.v32.teacher.v1':
        raise ValueError('reference requires its teacher sampler manifest')
    content = {key: value for key, value in sampler.items() if key != 'manifest_sha256'}
    digest = hashlib.sha256(json.dumps(content, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if digest != sampler.get('manifest_sha256'):
        raise ValueError('reference sampler checksum mismatch')
    expected_steps = (sampler['rows'] + sampler['batch_size'] - 1) // sampler['batch_size']
    if (payload['completed_epoch'] != sampler['epoch'] + 1 or payload['rows'] != sampler['rows']
            or payload['optimizer_steps'] != expected_steps):
        raise ValueError('reference checkpoint coverage/update counters disagree')
    if training and not isinstance(payload.get('rng'), dict):
        raise ValueError('reference resume requires RNG state')
    return payload



def _box_diagnostics(prediction, target, size, target_size):
    overlap, distance = box_metrics(LocalYawBox(prediction, size[[1, 0, 2]]),
        LocalYawBox(target, target_size[[1, 0, 2]]),
        up_axis=(0, 0, 1), mode='benchmark_compat', dim=3)
    yaw = float(abs(np.arctan2(np.sin(prediction[3] - target[3]),
                              np.cos(prediction[3] - target[3]))))
    return dict(iou=overlap, center_error_m=distance,
                center_xy_error_m=float(np.linalg.norm(prediction[:2] - target[:2])),
                yaw_error_rad=yaw, axis_yaw_error_rad=min(yaw, np.pi - yaw))

class TrackingEvaluation:
    def __init__(self, config=None):
        self.config = normalize_config(config)
        self.schema = model_schema(self.config)
        self.rows = []
        self._initialized = set()
        self._last_frame = {}
        self._lost_since = {}
        self._last_time = {}
        self._recoveries = []
        self._loss_events = 0

    def add_initialization(self, key, *, scene='unknown', timestamp=None):
        if key not in self._initialized:
            self._initialized.add(key)
            self._last_frame[key] = 0
            self.rows.append(dict(tracklet=key, frame=0, success=1., precision=1.,
                                  iou=1., distance=0., initialization=True, scene_id=scene,
                                  timestamp=timestamp))

    def add_singleton_tracks(self, dataset):
        """长度一的合法轨迹只有首帧，仍属于官方逐帧分母。"""
        from .data import metadata
        for index, length in enumerate(dataset.lengths):
            if length == 1:
                source = dataset.source
                key = source.get_tracklet_key(index) if hasattr(source, 'get_tracklet_key') else str(index)
                first = metadata(source, index, 0)
                self.add_initialization(str(key), scene=str(first.get('scene_id', 'unknown')),
                                        timestamp=float(first['timestamp']))

    def add_batch(self, raw_rows, batch, output, commit_results=None):
        predictions = output.accepted_box.detach().cpu().numpy()
        targets = batch['target_box'].detach().cpu().numpy()
        sizes = batch['box_size'].detach().cpu().numpy()
        target_sizes = batch['target_box_size'].detach().cpu().numpy()
        qualities = output.selected_quality.detach().cpu().numpy()
        indices = output.selected_index.detach().cpu().numpy()
        observation = getattr(output, 'observation', None)
        coarse = getattr(observation, 'coarse_box', None)
        coarse = coarse.detach().cpu().numpy() if coarse is not None else None
        sequence_valid = getattr(observation, 'sequence_valid', None)
        coarse_valid = (sequence_valid.detach().bool().cpu().numpy() if sequence_valid is not None
                        else np.full(len(raw_rows), coarse is not None, dtype=bool))
        fine = predictions
        diagnostics = {key: value.detach().cpu().numpy() for key, value in batch.items()
                       if key.startswith('diagnostic_') and torch.is_tensor(value)}
        for index, raw in enumerate(raw_rows):
            request = raw['request']
            if request.branch != 4:
                raise ValueError('evaluation must use one uninterrupted rollout per tracklet')
            key = raw['tracklet_key']
            if key not in self._initialized:
                if request.frame != 1:
                    raise ValueError('evaluation track must begin at its first prediction')
                self.add_initialization(key, scene=str(raw['first_frame'].get('scene_id', 'unknown')),
                                        timestamp=float(raw['first_frame']['timestamp']))
            if request.frame != self._last_frame[key] + 1:
                raise ValueError('evaluation endpoints must be complete and causal')
            self._last_frame[key] = request.frame
            overlap, distance = box_metrics(LocalYawBox(predictions[index], sizes[index][[1, 0, 2]]),
                LocalYawBox(targets[index], target_sizes[index][[1, 0, 2]]),
                up_axis=(0, 0, 1), mode='benchmark_compat', dim=3)
            success, precision = metric_contributions(overlap, distance)
            now = float(raw['frames'][request.frame]['timestamp'])
            self._last_time[key] = now
            if overlap < .1 and key not in self._lost_since:
                self._lost_since[key] = now
                self._loss_events += 1
            recovered = None
            if overlap >= .5 and key in self._lost_since:
                recovered = now - self._lost_since.pop(key)
                self._recoveries.append(recovered)
            diagnostic = {name: value[index].item() if np.ndim(value[index]) == 0
                          else value[index].tolist() for name, value in diagnostics.items()}
            diagnostic['coarse_geometry'] = (_box_diagnostics(coarse[index], targets[index],
                sizes[index], target_sizes[index]) if coarse is not None else None)
            # 全序列无点时 coarse 清零只是占位，不能作为真实预测参与精修统计。
            diagnostic['coarse_valid'] = bool(coarse is not None and coarse_valid[index])
            diagnostic['fine_geometry'] = _box_diagnostics(fine[index], targets[index],
                                                          sizes[index], target_sizes[index])
            if 'physical_displacement' in batch:
                displacement = float(batch['physical_displacement'][index].detach().norm().cpu())
                diagnostic['diagnostic_gt_xy_displacement'] = displacement
                diagnostic['diagnostic_moving'] = displacement >= .15
            crop_count = diagnostic.get('diagnostic_crop_point_count')
            crop_fg = diagnostic.get('diagnostic_crop_target_count')
            if crop_count is not None and crop_fg is not None:
                diagnostic['diagnostic_empty_crop'] = crop_count == 0
                diagnostic['diagnostic_background_only_crop'] = crop_count > 0 and crop_fg == 0
            diagnostic['recovery_seconds'] = recovered
            if commit_results is not None:
                diagnostic.update(commit_results[index])
            self.rows.append(dict(tracklet=key, frame=int(request.frame), success=float(success),
                precision=float(precision), iou=overlap, distance=distance, initialization=False,
                timestamp=now,
                selected_index=int(indices[index]), selected_quality=float(qualities[index]),
                scene_id=str(raw['frames'][request.frame].get('scene_id', 'unknown')), **diagnostic))

    def summary(self):
        count = len(self.rows)
        predictions = [row for row in self.rows if not row['initialization']]
        point_counts = {}
        for name, field in (('raw_target_points', 'diagnostic_target_count'),
                            ('novel_target_points', 'diagnostic_novel_target_count')):
            observed = [row[field] for row in predictions if row.get(field) is not None]
            point_counts[name] = int(sum(observed)) if observed else None
            point_counts[name + '_recorded_frames'] = len(observed)
        diag = dict(**point_counts, loss_events=self._loss_events,
            recovered_events=len(self._recoveries), unrecovered_events=len(self._lost_since),
            mean_recovery_seconds=float(np.mean(self._recoveries)) if self._recoveries else None,
            unrecovered_elapsed_seconds=sum(self._last_time[key] - start for key, start in self._lost_since.items()),
            loss_iou_threshold=.1, recovery_iou_threshold=.5)
        episodes = loss_episode_summary(self.rows)
        diag.update(loss_episode_frames_median=episodes['all']['frames']['median'],
                    loss_episode_frames_p90=episodes['all']['frames']['p90'],
                    loss_episode_seconds_median=episodes['all']['seconds']['median'],
                    loss_episode_seconds_p90=episodes['all']['seconds']['p90'],
                    loss_episode_total_frames=episodes['total_lost_frames'],
                    loss_episode_unrecovered_rate=episodes['unrecovered_rate'])
        diag['moving_threshold_m'] = .15
        context_norms = [row['diagnostic_query_context_norm'] for row in predictions
                         if row.get('diagnostic_query_context_norm') is not None]
        diag['query_context_recorded_frames'] = len(context_norms)
        diag['query_context_norm_mean'] = float(np.mean(context_norms)) if context_norms else None
        return dict(schema=self.schema, metric_mode='benchmark_compat', frames=count,
                    prediction_frames=count - len(self._initialized), tracklets=len(self._initialized),
                    success=100 * sum(row['success'] for row in self.rows) / max(count, 1),
                    precision=100 * sum(row['precision'] for row in self.rows) / max(count, 1),
                    diagnostics=diag, loss_episodes=episodes)
