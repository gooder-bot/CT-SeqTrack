"""v31 运行时小组件；导入不要求 Lightning、nuScenes 或 CUDA extension。"""
from __future__ import annotations

import hashlib
import json
import random
import numpy as np
import torch

from .contracts import SCHEMA
from .config import config_identity
from utils.tracking_metrics import LocalYawBox, box_metrics, metric_contributions

RESUME_SCHEMA = 'ct_seqtrack.v31.epoch_boundary.v1'


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
    return dict(schema=RESUME_SCHEMA, model_schema=SCHEMA,
                config_sha256=config_identity(config), completed_epoch=int(completed_epoch),
                epoch_complete=bool(complete), rows=int(rows), optimizer_steps=int(steps),
                sampler=sampler, rng=capture_rng_state())


def validate_resume_payload(payload, config, *, training=True):
    if not isinstance(payload, dict) or payload.get('schema') != RESUME_SCHEMA:
        raise ValueError('v31 checkpoint lacks its runtime schema')
    if payload.get('model_schema') != SCHEMA:
        raise ValueError('v31 model schema mismatch')
    if payload.get('config_sha256') != config_identity(config):
        raise ValueError('v31 resolved configuration identity mismatch')
    if training and (payload.get('epoch_complete') is not True or payload.get('completed_epoch', 0) < 1):
        raise ValueError('v31 only resumes complete epoch boundaries')
    if training:
        sampler = payload.get('sampler')
        if not isinstance(sampler, dict) or sampler.get('schema') != 'ct_seqtrack.v31.ready_queue.v1':
            raise ValueError('v31 resume requires a ready-queue manifest')
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


class TrackingEvaluation:
    """保留原 benchmark_compat 的逐帧 S/P 积分及首帧初始化计数。"""
    def __init__(self):
        self.rows = []
        self._initialized = set()
        self._last_frame = {}
        self._lost_since = {}
        self._last_time = {}
        self._recoveries = []
        self._loss_events = 0

    def add_initialization(self, key, *, scene='unknown'):
        if key not in self._initialized:
            self._initialized.add(key)
            self._last_frame[key] = 0
            self.rows.append(dict(tracklet=key, frame=0, success=1., precision=1.,
                                  iou=1., distance=0., initialization=True, scene_id=scene))

    def add_singleton_tracks(self, dataset):
        """长度一的合法轨迹只有首帧，仍属于官方逐帧分母。"""
        from .data import metadata
        for index, length in enumerate(dataset.lengths):
            if length == 1:
                source = dataset.source
                key = source.get_tracklet_key(index) if hasattr(source, 'get_tracklet_key') else str(index)
                first = metadata(source, index, 0)
                self.add_initialization(str(key), scene=str(first.get('scene_id', 'unknown')))

    def add_batch(self, raw_rows, batch, output, commit_results=None):
        predictions = output.accepted_box.detach().cpu().numpy()
        targets = batch['target_box'].detach().cpu().numpy()
        sizes = batch['box_size'].detach().cpu().numpy()
        target_sizes = batch['target_box_size'].detach().cpu().numpy()
        qualities = output.selected_quality.detach().cpu().numpy()
        indices = output.selected_index.detach().cpu().numpy()
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
        diagnostics = {key: batch[key].detach().cpu().numpy() for key in (
            'diagnostic_target_count', 'diagnostic_novel_target_count',
            'diagnostic_reachable_count', 'diagnostic_acquired_count') if key in batch}
        for index, raw in enumerate(raw_rows):
            request = raw['request']
            if request.branch != 4:
                raise ValueError('evaluation must use one uninterrupted rollout per tracklet')
            key = raw['tracklet_key']
            if key not in self._initialized:
                if request.frame != 1:
                    raise ValueError('evaluation track must begin at its first prediction')
                self.add_initialization(key, scene=str(raw['first_frame'].get('scene_id', 'unknown')))
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
            diagnostic = {key: int(value[index]) for key, value in diagnostics.items()}
            diagnostic['mode_count'] = int(mode_counts[index])
            diagnostic['target_mode_count'] = int(target_mode_counts[index])
            diagnostic['recovery_seconds'] = recovered
            if commit_results is not None:
                diagnostic.update(commit_results[index])
            self.rows.append(dict(tracklet=key, frame=int(request.frame), success=float(success),
                precision=float(precision), iou=overlap, distance=distance, initialization=False,
                selected_index=int(indices[index]), selected_quality=float(qualities[index]),
                scene_id=str(raw['frames'][request.frame].get('scene_id', 'unknown')), **diagnostic))

    def summary(self):
        count = len(self.rows)
        predictions = [row for row in self.rows if not row['initialization']]
        total = lambda key: sum(row.get(key, 0) for row in predictions)
        ratio = lambda numerator, denominator: numerator / denominator if denominator else None
        acquired = [row for row in predictions if row.get('diagnostic_acquired_count', 0) > 0]
        writes = int(total('memory_write'))
        diag = dict(raw_target_points=int(total('diagnostic_target_count')),
            novel_target_points=int(total('diagnostic_novel_target_count')),
            reachable_novel_target_points=int(total('diagnostic_reachable_count')),
            acquired_novel_target_points=int(total('diagnostic_acquired_count')),
            acquisition_reachable_ratio=ratio(total('diagnostic_reachable_count'), total('diagnostic_novel_target_count')),
            acquisition_realized_ratio=ratio(total('diagnostic_acquired_count'), total('diagnostic_reachable_count')),
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
        return dict(schema=SCHEMA, metric_mode='benchmark_compat', frames=count,
                    prediction_frames=count - len(self._initialized), tracklets=len(self._initialized),
                    success=100 * sum(row['success'] for row in self.rows) / max(count, 1),
                    precision=100 * sum(row['precision'] for row in self.rows) / max(count, 1),
                    diagnostics=diag)
