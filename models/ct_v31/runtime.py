"""v31 运行时小组件；导入不要求 Lightning、nuScenes 或 CUDA extension。"""
from __future__ import annotations

import random
import numpy as np
import torch

from .contracts import SCHEMA
from .config import config_identity
from .data import build_loaders, option
from utils.tracking_metrics_v27 import LocalYawBox, box_metrics, metric_contributions

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
    return payload


class TrackingEvaluation:
    """保留原 benchmark_compat 的逐帧 S/P 积分及首帧初始化计数。"""
    def __init__(self):
        self.rows = []
        self._initialized = set()

    def add_batch(self, raw_rows, batch, output):
        predictions = output.accepted_box.detach().cpu().numpy()
        targets = batch['target_box'].detach().cpu().numpy()
        sizes = batch['box_size'].detach().cpu().numpy()
        target_sizes = batch['target_box_size'].detach().cpu().numpy()
        qualities = output.selected_quality.detach().cpu().numpy()
        indices = output.selected_index.detach().cpu().numpy()
        for index, raw in enumerate(raw_rows):
            request = raw['request']
            if request.branch != 4:
                raise ValueError('evaluation must use one uninterrupted rollout per tracklet')
            key = raw['tracklet_key']
            if key not in self._initialized:
                if request.frame != 1:
                    raise ValueError('evaluation track must begin at its first prediction')
                self._initialized.add(key)
                self.rows.append(dict(tracklet=key, frame=0, success=1., precision=1.,
                                      iou=1., distance=0., initialization=True))
            overlap, distance = box_metrics(LocalYawBox(predictions[index], sizes[index][[1, 0, 2]]),
                LocalYawBox(targets[index], target_sizes[index][[1, 0, 2]]),
                up_axis=(0, 0, 1), mode='benchmark_compat', dim=3)
            success, precision = metric_contributions(overlap, distance)
            self.rows.append(dict(tracklet=key, frame=int(request.frame), success=float(success),
                precision=float(precision), iou=overlap, distance=distance, initialization=False,
                selected_index=int(indices[index]), selected_quality=float(qualities[index]),
                scene_id=str(raw['frames'][request.frame].get('scene_id', 'unknown'))))

    def summary(self):
        count = len(self.rows)
        return dict(schema=SCHEMA, metric_mode='benchmark_compat', frames=count,
                    prediction_frames=count - len(self._initialized), tracklets=len(self._initialized),
                    success=100 * sum(row['success'] for row in self.rows) / max(count, 1),
                    precision=100 * sum(row['precision'] for row in self.rows) / max(count, 1))


def __getattr__(name):
    if name == 'CTSEQTRACKV31':
        from models.ctseqtrackv31 import CTSEQTRACKV31
        return CTSEQTRACKV31
    raise AttributeError(name)
