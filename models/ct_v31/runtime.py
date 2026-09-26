"""v31 运行时小组件；导入不要求 Lightning、nuScenes 或 CUDA extension。"""
from __future__ import annotations

import hashlib
import json
import random
import numpy as np
import torch

from .contracts import SCHEMA
from .config import config_identity
from .identity import model_schema, model_version, runtime_schema
from utils.tracking_metrics import LocalYawBox, box_metrics, metric_contributions

RESUME_SCHEMA = 'ct_seqtrack.v33.epoch_boundary.v1'


def capture_rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA training RNG cannot resume on CPU')
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


def move_tensors(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def resume_payload(config, *, completed_epoch, complete, rows, steps, sampler=None):
    return dict(schema=runtime_schema(config), model_schema=model_schema(config),
                config_sha256=config_identity(config), completed_epoch=int(completed_epoch),
                epoch_complete=bool(complete), rows=int(rows), optimizer_steps=int(steps),
                sampler=sampler, rng=capture_rng_state())


def validate_resume_payload(payload, config, *, training=True):
    if not isinstance(payload, dict) or payload.get('schema') != runtime_schema(config):
        raise ValueError('v31 checkpoint lacks its runtime schema')
    if payload.get('model_schema') != model_schema(config):
        raise ValueError('v31 model schema mismatch')
    if payload.get('config_sha256') != config_identity(config):
        raise ValueError('v31 resolved configuration identity mismatch')
    if training and (payload.get('epoch_complete') is not True or payload.get('completed_epoch', 0) < 1):
        raise ValueError('v31 only resumes complete epoch boundaries')
    if training:
        sampler = payload.get('sampler')
        expected_sampler = ('seqtrack_reference.v32.teacher.v1'
                            if config.get('net_model') == 'seqtrack_reference'
                            else 'ct_seqtrack.v32.ready_queue.v2')
        if not isinstance(sampler, dict) or sampler.get('schema') != expected_sampler:
            raise ValueError('v32 resume requires the matching sampler manifest')
        content = {key: value for key, value in sampler.items() if key != 'manifest_sha256'}
        digest = hashlib.sha256(json.dumps(content, sort_keys=True,
                                          separators=(',', ':')).encode()).hexdigest()
        if digest != sampler.get('manifest_sha256'):
            raise ValueError('v31 ready-queue manifest checksum mismatch')
        if (payload['completed_epoch'] != sampler['epoch'] + 1
                or payload['rows'] != sampler['rows'] or payload['optimizer_steps'] <= 0):
            raise ValueError('v31 checkpoint epoch/coverage counters disagree')
        if not isinstance(payload.get('rng'), dict):
            raise ValueError('v31 resume requires RNG state')
    return payload


def loss_episode_summary(rows):
    """IoU<.1 开始、IoU>=.5 结束；恢复帧不计入失跟长度。

    未恢复段右删失，单独报告；缺 timestamp 的旧记录不伪造秒数。
    所有段包含 [.1,.5) 中间帧，不能与低 IoU 帧数混用。
    """
    tracks = {}
    for row in rows:
        if not row.get('initialization', False):
            tracks.setdefault(row['tracklet'], []).append(row)
    episodes = []
    for track, values in tracks.items():
        values = sorted(values, key=lambda row: row['frame'])
        active = None
        for row in values:
            if active is None and row['iou'] < .1:
                active = dict(tracklet=track, start_frame=row['frame'],
                              start_time=row.get('timestamp'), frames=0)
            if active is None:
                continue
            recovered = row['iou'] >= .5
            if not recovered:
                active['frames'] += 1
            last = row is values[-1]
            if recovered or last:
                start, end = active.pop('start_time'), row.get('timestamp')
                seconds = float(end - start) if start is not None and end is not None else None
                if seconds is not None and seconds < 0:
                    raise ValueError('evaluation timestamps must be causal')
                episodes.append(dict(active, end_frame=row['frame'], recovered=recovered,
                                     seconds=seconds))
                active = None
    def distribution(values):
        return dict(count=len(values), median=float(np.median(values)) if values else None,
                    p90=float(np.percentile(values, 90)) if values else None,
                    total=float(sum(values)) if values else 0.)
    def summarize(values):
        timed = [row['seconds'] for row in values if row['seconds'] is not None]
        return dict(episodes=len(values), frames=distribution([row['frames'] for row in values]),
                    seconds=distribution(timed), seconds_complete=len(timed) == len(values))
    recovered = [row for row in episodes if row['recovered']]
    censored = [row for row in episodes if not row['recovered']]
    return dict(definition='start_iou_lt_0.1_end_iou_ge_0.5_recovery_frame_excluded',
                all=summarize(episodes), recovered=summarize(recovered),
                unrecovered=summarize(censored),
                unrecovered_rate=len(censored) / len(episodes) if episodes else None,
                total_lost_frames=sum(row['frames'] for row in episodes),
                low_iou_frames=sum(row['iou'] < .1 for values in tracks.values() for row in values),
                episodes=episodes)


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
    """保留原 benchmark_compat 的逐帧 S/P 积分及首帧初始化计数。"""
    def __init__(self, config=None):
        self.schema = model_schema(config)
        self.version = model_version(config)
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
        hypotheses = getattr(output, 'hypothesis_boxes', None)
        fine = hypotheses[:, 0].detach().cpu().numpy() if hypotheses is not None else predictions
        mode_valid = getattr(output.evidence, 'mode_valid', None)
        mode_counts = (mode_valid.detach().cpu().sum(-1).numpy() if mode_valid is not None
                       else np.zeros(len(raw_rows), dtype=np.int64))
        target_mode_counts = np.zeros(len(raw_rows), dtype=np.int64)
        if mode_valid is not None and 'extension_labels' in batch:
            evidence = output.evidence
            selected_labels = batch['extension_labels'].gather(1, evidence.point_indices.clamp_min(0))
            members = evidence.members.bool() & evidence.point_valid[:, None].bool()
            foreground = members & (selected_labels > 0)[:, None]
            fg_count = foreground.sum(-1)
            purity = fg_count.to(torch.float32) / members.sum(-1).clamp_min(1)
            target_modes = evidence.mode_valid.bool() & (fg_count > 0) & (purity >= .5)
            target_mode_counts = target_modes.detach().cpu().sum(-1).numpy()
        diagnostics = {key: value.detach().cpu().numpy() for key, value in batch.items()
                       if key.startswith('diagnostic_') and torch.is_tensor(value)}
        if self.version == 'v34':
            # 仅记录实际前向已产生的摘要；不读取 GT，不影响候选或 accepted 提交。
            decoder = getattr(output, 'decoder', None)
            for name, owner, field, shape in (
                    ('diagnostic_history_support', observation, 'history_support', (len(raw_rows), 3, 3)),
                    ('diagnostic_current_support', observation, 'current_support', (len(raw_rows), 3)),
                    ('diagnostic_query_context_norm', decoder, 'query_context_norm', (len(raw_rows),))):
                value = getattr(owner, field, None)
                if value is not None:
                    if tuple(value.shape) != shape or not bool(torch.isfinite(value).all()):
                        raise ValueError('invalid v34 passive diagnostic: ' + field)
                    diagnostics[name] = value.detach().cpu().numpy()
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
            diagnostic['mode_count'] = int(mode_counts[index])
            diagnostic['target_mode_count'] = int(target_mode_counts[index])
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
        total = lambda key: sum(row.get(key, 0) for row in predictions)
        ratio = lambda numerator, denominator: numerator / denominator if numerator is not None and denominator else None
        point_counts = {}
        for name, field in (('raw_target_points', 'diagnostic_target_count'),
                            ('novel_target_points', 'diagnostic_novel_target_count'),
                            ('reachable_novel_target_points', 'diagnostic_reachable_count'),
                            ('acquired_novel_target_points', 'diagnostic_acquired_count')):
            # 独立 reference 不提供这些诊断；缺测不能记成真实的零目标点。
            observed = [row[field] for row in predictions if row.get(field) is not None]
            point_counts[name] = int(sum(observed)) if observed else None
            point_counts[name + '_recorded_frames'] = len(observed)
        acquired = [row for row in predictions if row.get('diagnostic_acquired_count', 0) > 0]
        writes = int(total('memory_write'))
        diag = dict(**point_counts,
            acquisition_reachable_ratio=ratio(point_counts['reachable_novel_target_points'], point_counts['novel_target_points']),
            acquisition_realized_ratio=ratio(point_counts['acquired_novel_target_points'], point_counts['reachable_novel_target_points']),
            mode_formation_rate=ratio(sum(row['mode_count'] > 0 for row in predictions), len(predictions)),
            any_mode_formation_on_acquired_target_rate=ratio(sum(row['mode_count'] > 0 for row in acquired), len(acquired)),
            target_mode_formation_on_acquired_target_rate=ratio(sum(row['target_mode_count'] > 0 for row in acquired), len(acquired)),
            memory_writes=writes, memory_wrong_write_rate=ratio(total('memory_wrong_write'), writes),
            memory_foreground_purity=ratio(total('memory_true_fg_count'), total('memory_fg_count')),
            loss_events=self._loss_events, recovered_events=len(self._recoveries),
            unrecovered_events=len(self._lost_since),
            mean_recovery_seconds=float(np.mean(self._recoveries)) if self._recoveries else None,
            unrecovered_elapsed_seconds=sum(self._last_time[key] - start for key, start in self._lost_since.items()),
            loss_iou_threshold=.1, recovery_iou_threshold=.5, wrong_write_fg_purity_threshold=.5)
        episodes = loss_episode_summary(self.rows)
        diag.update(loss_episode_frames_median=episodes['all']['frames']['median'],
                    loss_episode_frames_p90=episodes['all']['frames']['p90'],
                    loss_episode_seconds_median=episodes['all']['seconds']['median'],
                    loss_episode_seconds_p90=episodes['all']['seconds']['p90'],
                    loss_episode_total_frames=episodes['total_lost_frames'],
                    loss_episode_unrecovered_rate=episodes['unrecovered_rate'])
        diag['moving_threshold_m'] = .15
        if self.version == 'v34':
            context_norms = [row['diagnostic_query_context_norm'] for row in predictions
                             if row.get('diagnostic_query_context_norm') is not None]
            diag['query_context_recorded_frames'] = len(context_norms)
            diag['query_context_norm_mean'] = float(np.mean(context_norms)) if context_norms else None
        return dict(schema=self.schema, metric_mode='benchmark_compat', frames=count,
                    prediction_frames=count - len(self._initialized), tracklets=len(self._initialized),
                    success=100 * sum(row['success'] for row in self.rows) / max(count, 1),
                    precision=100 * sum(row['precision'] for row in self.rows) / max(count, 1),
                    diagnostics=diag, loss_episodes=episodes)
