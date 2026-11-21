"""v41 指标、完整候选审计与同 run 的严格 epoch 恢复。"""
import hashlib
import json
import random
import numpy as np
import torch

from .config import config_identity, normalize_config
from .identity import SCHEMA, runtime_schema, LEGACY_CANDIDATES, JOINT_CANDIDATES, JOINT_MEASUREMENTS
from utils.tracking_metrics import LocalYawBox, box_metrics, metric_contributions
from utils.execution_profile import optimized_execution



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

def move_tensors(batch, device, *, evaluation=False, retain_loss_geometry=False):
    # 评估的 GT 框与前景标签只在 CPU 打分。其余字段在同一流上依次拷贝、消费。
    metric_fields = ('target_box', 'target_box_size', 'segmentation_labels') if evaluation else ()
    moved = {key: value.to(device, non_blocking=True)
             if torch.is_tensor(value) and key not in metric_fields and not key.startswith('diagnostic_')
             else value for key, value in batch.items()}
    if retain_loss_geometry and not evaluation and torch.device(device).type != 'cpu':
        from .input import retain_geometry_sources
        retain_geometry_sources(batch, moved)
    return moved

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

def dataset_identity(config):
    """正式checkpoint绑定输入内容，允许同内容迁移路径，不允许独立test换标注。"""
    if config.ct_engineering_check:
        return None
    from datasets.kitti_v41 import build_v41_kitti_manifest, kitti_input_metadata_identity
    return dict(dataset_manifest=build_v41_kitti_manifest(),
                input_metadata=kitti_input_metadata_identity(config.path))


def resume_payload(config, *, completed_epoch, complete, rows, steps, sampler=None, run_id=None, auxiliary=None):
    from .entry import source_identity
    from utils.execution_profile import execution_identity
    source = source_identity()
    return dict(schema=runtime_schema(config), model_schema=SCHEMA,
        config_sha256=config_identity(config), completed_epoch=int(completed_epoch),
        epoch_complete=bool(complete), rows=int(rows), optimizer_steps=int(steps),
        sampler=sampler, auxiliary=auxiliary, rng=capture_rng_state(), source=source, run_id=run_id,
        dataset_identity=dataset_identity(config),
        execution_identity=execution_identity(config, source))

def validate_resume_payload(payload, config, *, training=True):
    from .entry import source_identity
    from utils.execution_profile import validate_execution_identity
    if not isinstance(payload, dict) or payload.get('schema') != runtime_schema(config):
        raise ValueError('v41 checkpoint runtime schema mismatch')
    if payload.get('model_schema') != SCHEMA or payload.get('config_sha256') != config_identity(config):
        raise ValueError('v41 checkpoint model/config/arm/precision identity mismatch')
    if config.get('v41_history_training', 'legacy') != 'legacy':
        fingerprint = payload.get('history_bank_sha256')
        if (not isinstance(fingerprint, str) or len(fingerprint) != 64
                or any(char not in '0123456789abcdef' for char in fingerprint)):
            raise ValueError('missing or invalid E57 history bank identity')
    if config.get('v41_joint_replay') == 'fixed_e2':
        from .fixed_replay import validate_fixed_replay_receipt
        validate_fixed_replay_receipt(payload.get('joint_replay'), payload.get('completed_epoch'))
    if config.get('v41_q0_lateral_replay', False):
        fingerprint = payload.get('q0_replay_manifest_sha256')
        if (not isinstance(fingerprint, str) or len(fingerprint) != 64
                or any(char not in '0123456789abcdef' for char in fingerprint)):
            raise ValueError('q0 checkpoint requires its training replay manifest fingerprint')
    source = source_identity()
    validate_execution_identity(payload.get('execution_identity'), config, source)
    if payload.get('source') != source:
        raise ValueError('v41 checkpoint source identity mismatch; use its original source')
    if payload.get('dataset_identity') != dataset_identity(config):
        raise ValueError('v41 checkpoint dataset/input metadata identity mismatch')
    if not isinstance(payload.get('run_id'), str) or not payload['run_id']:
        raise ValueError('v41 checkpoint requires a nonempty run_id')
    if payload.get('epoch_complete') is not True or payload.get('completed_epoch', 0) < 1:
        raise ValueError('v41 requires complete epoch boundaries')
    sampler = payload.get('sampler')
    if not isinstance(sampler, dict) or sampler.get('schema') != 'ct_seqtrack.v32.ready_queue.v2':
        raise ValueError('v41 requires the original v35 W sampler manifest')
    content = {key: value for key, value in sampler.items() if key != 'manifest_sha256'}
    digest = hashlib.sha256(json.dumps(content, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if digest != sampler.get('manifest_sha256'):
        raise ValueError('v41 sampler checksum mismatch')
    if sampler.get('initial_seed_policy') != config['v35_seed_policy']:
        raise ValueError('v41 initial seed policy mismatch')
    if (payload['completed_epoch'] != sampler['epoch'] + 1
            or payload['rows'] != sampler['rows']
            or payload['optimizer_steps'] != sampler['batches']):
        raise ValueError('v41 checkpoint coverage/update counters disagree')
    if training and not isinstance(payload.get('rng'), dict):
        raise ValueError('v41 resume requires RNG state')
    validate_auxiliary_payload(payload.get('auxiliary'), config, sampler)
    return payload


def validate_auxiliary_payload(auxiliary, config, main_sampler):
    """主4N与辅助全轨迹必须同时提交完毕；不能把半个辅助epoch当作可恢复。"""
    if not isinstance(auxiliary, dict) or auxiliary.get('schema') != 'ct_seqtrack.v41.auxiliary_audit.v1':
        raise ValueError('v41 requires auxiliary training audit')
    saved = auxiliary.get('sampler')
    if not isinstance(saved, dict) or saved.get('schema') != 'ct_seqtrack.v41.full_track_aux.v1':
        raise ValueError('v41 auxiliary sampler schema mismatch')
    content = {key: value for key, value in saved.items() if key != 'manifest_sha256'}
    digest = hashlib.sha256(json.dumps(content, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if saved.get('manifest_sha256') != digest:
        raise ValueError('v41 auxiliary sampler checksum mismatch')
    if (auxiliary.get('complete') is not True or auxiliary.get('rows') != saved.get('rows')
            or auxiliary.get('batches') != saved.get('batches')
            or auxiliary.get('main_batches') != main_sampler.get('batches')
            or saved.get('epoch') != main_sampler.get('epoch')
            or saved.get('lengths') != main_sampler.get('lengths')
            or saved.get('source_sha256') != main_sampler.get('source_sha256')
            or saved.get('seed') != config.seed or saved.get('batch_size') != config.batch_size
            or saved.get('fraction') != config.v41_aux_fraction
            or auxiliary.get('loss_weight') != config.v41_aux_loss_weight):
        raise ValueError('v41 main/auxiliary complete-epoch coverage or identity mismatch')
    # 独立--test也验真正的整轨预算/选择/因果计划，不只相信自报hash与计数。
    from .auxiliary import FullTrackAuxiliarySampler
    expected_sampler = FullTrackAuxiliarySampler(saved['lengths'], config.batch_size,
        seed=config.seed, fraction=config.v41_aux_fraction,
        source_sha256=saved['source_sha256'])
    expected_sampler.set_epoch(saved['epoch'])
    if expected_sampler.state_dict() != saved:
        raise ValueError('v41 auxiliary whole-track plan differs from its registered deterministic policy')
    indices = auxiliary.get('insertion_indices')
    if not isinstance(indices, list) or len(indices) != auxiliary['batches']:
        raise ValueError('v41 auxiliary insertion plan length mismatch')
    expected = ([((j + 1) * auxiliary['main_batches']) // auxiliary['batches'] - 1
                 for j in range(auxiliary['batches'])] if auxiliary['batches'] else [])
    if indices != expected or len(set(indices)) != len(indices):
        raise ValueError('v41 auxiliary insertion plan differs from uniform one-per-main-step schedule')
    plan_hash = hashlib.sha256(json.dumps(indices, separators=(',', ':')).encode()).hexdigest()
    if auxiliary.get('insertion_plan_sha256') != plan_hash:
        raise ValueError('v41 auxiliary insertion checksum mismatch')
    return auxiliary

def reliability_summary(records):
    """T 的固定十分箱；包含未选中的真实 q0，不据此修改 0.5 阈值。"""
    valid = [record for record in records if record.get('trusted_score') is not None
             and record.get('trusted_valid')]
    bins = []
    for index in range(10):
        selected = [record for record in valid if min(int(record['trusted_score'] * 10), 9) == index]
        bins.append(dict(lower=index / 10, upper=(index + 1) / 10, count=len(selected),
            mean_probability=float(np.mean([r['trusted_score'] for r in selected])) if selected else None,
            event_frequency=float(np.mean([r['trusted_target'] for r in selected])) if selected else None))
    return dict(definition='exact_iou_ge_0.5_and_unique_predfg_support_ge_3_and_purity_ge_0.5',
        threshold=.5, threshold_selection='fixed_before_experiments', count=len(valid),
        positive=sum(r['trusted_target'] for r in valid),
        negative=sum(not r['trusted_target'] for r in valid),
        brier=float(np.mean([(r['trusted_score'] - float(r['trusted_target'])) ** 2 for r in valid]))
            if valid else None, bins=bins)

def motion_slices(rows):
    """只用于报告：运动端点、曾移动轨迹全程、低运动及启停边界分开。"""
    predictions = [row for row in rows if not row['initialization']]
    moving_tracks = {row['tracklet'] for row in predictions if row.get('diagnostic_moving')}
    starts, stops, previous = [], [], {}
    for row in sorted(predictions, key=lambda r: (r['tracklet'], r['frame'])):
        moving = row.get('diagnostic_moving')
        last = previous.get(row['tracklet'])
        if last is not None and moving is not None:
            if moving and not last:
                starts.append(row)
            elif last and not moving:
                stops.append(row)
        if moving is not None:
            previous[row['tracklet']] = moving
    def metrics(values):
        return dict(frames=len(values), tracklets=len({row['tracklet'] for row in values}),
            success=100 * np.mean([row['success'] for row in values]) if values else None,
            precision=100 * np.mean([row['precision'] for row in values]) if values else None,
            over5m=sum(row['distance'] > 5 for row in values),
            over10m=sum(row['distance'] > 10 for row in values))
    return dict(definition='adjacent_GT_XY_displacement_ge_0.15m_passive_only', initialization_included=False,
        ever_moving_tracks=metrics([r for r in predictions if r['tracklet'] in moving_tracks]),
        moving_endpoints=metrics([r for r in predictions if r.get('diagnostic_moving')]),
        low_motion_endpoints=metrics([r for r in predictions if r.get('diagnostic_moving') is False]),
        start_boundary=metrics(starts), stop_boundary=metrics(stops))

class TrackingEvaluation:
    """benchmark_compat 逐帧积分含合法首帧；GT 几何只用于被动审计。"""
    def __init__(self, config, *, row_sink=None):
        self.v41_config = normalize_config(config)
        self.optimized = optimized_execution(self.v41_config)
        self.schema = SCHEMA
        self.rows = []
        self._initialized = set()
        self._last_frame = {}
        self._lost_since = {}
        self._last_time = {}
        self._recoveries = []
        self._loss_events = 0
        self._row_sink = row_sink
        self._in_batch = False

    def add_initialization(self, key, *, scene='unknown', timestamp=None, raw_frame_id=None, track_id=None):
        if key not in self._initialized:
            self._initialized.add(key)
            self._last_frame[key] = 0
            self.rows.append(dict(tracklet=key, frame=0, success=1., precision=1.,
                                  iou=1., distance=0., initialization=True, scene_id=scene,
                                  timestamp=timestamp, raw_frame_id=raw_frame_id, track_id=track_id))
            if self._row_sink is not None and not self._in_batch:
                self._row_sink(self.rows[-1])

    def add_singleton_tracks(self, dataset):
        """长度一的合法轨迹只有首帧，仍属于官方逐帧分母。"""
        from .data import metadata
        for index, length in enumerate(dataset.lengths):
            if length == 1:
                source = dataset.source
                key = source.get_tracklet_key(index) if hasattr(source, 'get_tracklet_key') else str(index)
                first = metadata(source, index, 0)
                self.add_initialization(str(key), scene=str(first.get('scene_id', 'unknown')),
                                        timestamp=float(first['timestamp']),
                                        raw_frame_id=first.get('raw_frame_id', first.get('frame_id')),
                                        track_id=first.get('meta', {}).get('track_id'))

    def _add_batch_metrics(self, raw_rows, batch, output, commit_results=None, candidate_rows=None):
        from utils.diagnostic_transfer import tensors_to_numpy
        observation, decoder = getattr(output, 'observation', None), output.decoder
        packet = dict(predictions=output.accepted_box, targets=batch['target_box'],
            sizes=batch['box_size'], target_sizes=batch['target_box_size'],
            qualities=output.selected_quality, indices=output.selected_index,
            fine=decoder.q0_box, anchors=batch['anchor_box'])
        packet.update((k, v) for k, v in batch.items() if k.startswith('diagnostic_') and torch.is_tensor(v))
        lateral = getattr(decoder, 'lateral', None)
        if lateral is not None:
            packet.update(original_q0=lateral.base_box,
                          diagnostic_q0_lateral_suggestion_m=lateral.suggestion,
                          diagnostic_q0_lateral_gate=lateral.gate,
                          diagnostic_q0_lateral_applied_m=lateral.applied,
                          diagnostic_q0_lateral_unique_points=lateral.support_count)
        if 'physical_displacement' in batch:
            packet['diagnostic_gt_xy_displacement'] = batch['physical_displacement'].detach().norm(dim=-1)
        for name, field in (('coarse', 'coarse_box'), ('coarse_valid', 'sequence_valid')):
            value = getattr(observation, field, None)
            if value is not None:
                packet[name] = value
        for owner, field, shape in (
                (observation, 'history_support', (len(raw_rows), 3, 3)),
                (observation, 'current_support', (len(raw_rows), 3)),
                (decoder, 'query_context_norm', (len(raw_rows),)),
                *((decoder, field, (len(raw_rows), 8)) for field in
                  ('local_neighbor_count', 'local_selected_count', 'local_mean_support', 'local_delta_norm'))):
            value = getattr(owner, field, None)
            if value is not None:
                if tuple(value.shape) != shape:
                    raise ValueError('invalid v41 passive diagnostic: ' + field)
                packet['diagnostic_' + field] = value
        arrays = tensors_to_numpy(packet, optimized=self.optimized)
        predictions, targets = arrays['predictions'], arrays['targets']
        sizes, target_sizes = arrays['sizes'], arrays['target_sizes']
        indices, qualities, fine = arrays['indices'], arrays['qualities'], arrays['fine']
        coarse = arrays.get('coarse')
        coarse_valid = arrays.get('coarse_valid', np.full(len(raw_rows), coarse is not None))
        diagnostics = {k: v for k, v in arrays.items() if k.startswith('diagnostic_')}
        for key in ('history_support', 'current_support', 'query_context_norm', 'local_neighbor_count',
                    'local_selected_count', 'local_mean_support', 'local_delta_norm'):
            if 'diagnostic_' + key in diagnostics and not np.isfinite(diagnostics['diagnostic_' + key]).all():
                raise ValueError('invalid v41 passive diagnostic: ' + key)
        # 记录有符号世界坐标与尺寸，允许离线分解 XYZ/yaw/IoU。
        # 只作 detach 后的导出，不把 GT 几何送回 forward 或 commit。
        anchors = arrays['anchors'].astype(np.float64, copy=False)
        for name, value in (('coarse', coarse), ('fine', fine), ('original_q0', arrays.get('original_q0')),
                            ('accepted', predictions), ('target', targets)):
            if value is not None:
                world = value.astype(np.float64, copy=True)
                world[:, :3] += anchors[:, :3]
                diagnostics['diagnostic_' + name + '_box_world'] = world
        diagnostics['diagnostic_anchor_box_world'] = anchors
        diagnostics['diagnostic_first_frame_size_lwh'] = sizes
        diagnostics['diagnostic_target_size_lwh'] = target_sizes
        for index, raw in enumerate(raw_rows):
            request = raw['request']
            if request.branch != 4:
                raise ValueError('evaluation must use one uninterrupted rollout per tracklet')
            key = raw['tracklet_key']
            if key not in self._initialized:
                if request.frame != 1:
                    raise ValueError('evaluation track must begin at its first prediction')
                self.add_initialization(key, scene=str(raw['first_frame'].get('scene_id', 'unknown')),
                                        timestamp=float(raw['first_frame']['timestamp']),
                                        raw_frame_id=raw['first_frame'].get('raw_frame_id', raw['first_frame'].get('frame_id')),
                                        track_id=raw['first_frame'].get('meta', {}).get('track_id'))
            if request.frame != self._last_frame[key] + 1:
                raise ValueError('evaluation endpoints must be complete and causal')
            self._last_frame[key] = request.frame
            record = candidate_rows[index] if candidate_rows is not None else None
            if record is None:
                overlap, distance = box_metrics(LocalYawBox(predictions[index], sizes[index][[1, 0, 2]]),
                    LocalYawBox(targets[index], target_sizes[index][[1, 0, 2]]),
                    up_axis=(0, 0, 1), mode='benchmark_compat', dim=3)
                success, precision = metric_contributions(overlap, distance)
            else:
                geometry = record['selected_geometry']
                overlap, distance = geometry['iou'], geometry['center_error_m']
                success, precision = geometry['success'], geometry['precision']
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
            if record is None:
                diagnostic['fine_geometry'] = _box_diagnostics(fine[index], targets[index],
                                                              sizes[index], target_sizes[index])
            else:
                diagnostic['fine_geometry'] = {key: record['q0_geometry'][key] for key in
                    ('iou', 'center_error_m', 'center_xy_error_m', 'yaw_error_rad', 'axis_yaw_error_rad')}
            if 'diagnostic_gt_xy_displacement' in diagnostic:
                diagnostic['diagnostic_moving'] = diagnostic['diagnostic_gt_xy_displacement'] >= .15
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
                raw_frame_id=raw['frames'][request.frame].get('raw_frame_id', raw['frames'][request.frame].get('frame_id')),
                track_id=raw['frames'][request.frame].get('meta', {}).get('track_id'),
                selected_index=int(indices[index]), selected_quality=float(qualities[index]),
                scene_id=str(raw['frames'][request.frame].get('scene_id', 'unknown')), **diagnostic))

    def add_batch(self, raw_rows, batch, output, commit_results=None):
        from .candidate_records import candidate_records
        records = candidate_records(batch, output, optimized=self.optimized)
        first = len(self.rows)
        self._in_batch = True
        try:
            self._add_batch_metrics(raw_rows, batch, output, commit_results, records)
        finally:
            self._in_batch = False
        predictions = [row for row in self.rows[first:] if not row['initialization']]
        for row, record in zip(predictions, records):
            row.update(diagnostic_candidates=record, output_utility=record['selected_utility'],
                legacy_q0_quality=record['legacy_q0_quality'], q0_trusted_score=record['trusted_score'],
                trusted_score=float(row.get('commit_quality', 0.)) if output.verification is not None else None,
                q0_is_fallback=record['q0_is_fallback'], selected_source=record['selected_source'],
                trusted_write=bool(row.get('supported')))
            record['mechanism']['actual_episode_recovery'] = row.get('recovery_seconds') is not None
        if self._row_sink is not None:
            for row in self.rows[first:]:
                self._row_sink(row)

    def _metrics_summary(self):
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
        diag.update(low_iou_frames=episodes['low_iou_frames'],
                    max_loss_frames=max((item['frames'] for item in episodes['episodes']), default=0),
                    max_loss_seconds=max((item['seconds'] for item in episodes['episodes']
                                          if item['seconds'] is not None), default=0.))
        diag['moving_threshold_m'] = .15
        context_norms = [row['diagnostic_query_context_norm'] for row in predictions
                         if row.get('diagnostic_query_context_norm') is not None]
        diag['query_context_recorded_frames'] = len(context_norms)
        diag['query_context_norm_mean'] = float(np.mean(context_norms)) if context_norms else None
        return dict(schema=self.schema, metric_mode='benchmark_compat', frames=count,
                    total_frames=count, initialization_frames=len(self._initialized),
                    prediction_frames=count - len(self._initialized), tracklets=len(self._initialized),
                    success=100 * sum(row['success'] for row in self.rows) / max(count, 1),
                    precision=100 * sum(row['precision'] for row in self.rows) / max(count, 1),
                    diagnostics=diag, loss_episodes=episodes)

    def summary(self):
        result = self._metrics_summary()
        rows = [row for row in self.rows if not row['initialization']]
        records = [row['diagnostic_candidates'] for row in rows]
        diagnostics = result['diagnostics']
        for name in ('replaced_real_q0', 'erroneous_takeover', 'beneficial_takeover',
                     'recovered_bad_q0', 'actual_episode_recovery'):
            diagnostics[name] = sum(record['mechanism'][name] for record in records)
        diagnostics.update(over5m_frames=sum(row['distance'] > 5 for row in rows),
            over10m_frames=sum(row['distance'] > 10 for row in rows),
            propagation_only_frames=sum(row['selected_source'] in ('prior', 'trusted_propagation') for row in rows))
        count = len(rows)
        result['selection'] = {name: dict(count=sum(r['selected_source'] == name for r in rows),
            rate=sum(r['selected_source'] == name for r in rows) / count if count else None)
            for name in (JOINT_CANDIDATES if self.v41_config.v41_joint_search else LEGACY_CANDIDATES)}
        writes = [row for row in rows if row.get('supported')]
        wrong = sum(not row['diagnostic_candidates']['selected_trusted_target'] for row in writes)
        result['trusted'] = dict(writes=len(writes), coverage=len(writes) / count if count else None,
            wrong_writes=wrong, wrong_write_rate=wrong / len(writes) if writes else None,
            position_updates_velocity_invalid=sum(not row.get('trusted_velocity_valid', False) for row in writes),
            velocity_updates=sum(row.get('trusted_velocity_updated', False) for row in rows),
            longest_untrusted_streak=max((r.get('consecutive_untrusted_outputs', 0) for r in rows), default=0),
            longest_propagation_streak=max((r.get('consecutive_propagations', 0) for r in rows), default=0),
            max_supported_age_seconds=max((r.get('supported_age', 0.) for r in rows), default=0.),
            max_velocity_age_seconds=max((r.get('trusted_velocity_age', 0.) for r in rows), default=0.))
        if self.v41_config.v41_joint_search:
            result['geometry_reliability'] = {}
            for index in JOINT_MEASUREMENTS:
                measurements = [dict(item, trusted_valid=item['valid'],
                    trusted_target=record['geometry'][index]['exact_iou'] >= .5) for record in records
                    for item in record['measurement_trust_records'] if item['index'] == index]
                summary = reliability_summary(measurements)
                summary['definition'] = 'exact_iou_ge_0.5'
                result['geometry_reliability'][JOINT_CANDIDATES[index]] = summary
        else:
            result['trusted_reliability'] = reliability_summary(records)
            for index, name in ((0, 'q0'), (3, 'recovery')):
                measurements = [dict(item, trusted_valid=item['valid']) for record in records
                                for item in record['measurement_trust_records'] if item['index'] == index]
                result[name + '_trusted_reliability'] = reliability_summary(measurements)
        old_good = [any(v and g['exact_iou'] >= .5 for v, g in zip(record['valid'][:3], record['geometry'][:3]))
                    for record in records]
        recovery_good = [record['valid'][3] and record['geometry'][3]['exact_iou'] >= .5 for record in records]
        result['recovery_chain'] = dict(prediction_frames=count,
            old_candidate_good_frames=sum(old_good), recovery_good_frames=sum(recovery_good),
            added_good_candidate_frames=sum(new and not old for old, new in zip(old_good, recovery_good)),
            good_candidate_frames=sum(old or new for old, new in zip(old_good, recovery_good)),
            good_candidate_but_wrong_selection_frames=sum((old or new) and record['selected_geometry']['exact_iou'] < .5
                for old, new, record in zip(old_good, recovery_good, records)),
            selected_recovery_frames=sum(record['selected_index'] == 3 for record in records),
            correct_recovery_trusted_writes=sum(record['selected_index'] == 3 and record['trusted_eligible']
                and bool(record['selected_trusted_target']) for record in records),
            interpretation='passive_same_state_geometry_and_actual_commits_not_causal_ablation')
        result['coordinate_mode'] = self.v41_config.ct_coordinate_mode
        result['geometry_convention'] = self.v41_config.v41_kitti_geometry
        result['motion_slices'] = motion_slices(self.rows)
        guarded = [row for row in rows if row.get('diagnostic_search_triggered', False)]
        raw_present = [row for row in rows if row.get('diagnostic_search_raw_target_count', 0) > 0]
        added = [row for row in guarded
                 if row.get('diagnostic_search_union_target_count', 0)
                 > row.get('diagnostic_search_main_target_count', 0)]
        result['search'] = dict(policy=self.v41_config.v41_search_policy,
            triggered_frames=len(guarded),
            raw_target_present_frames=len(raw_present),
            main_crop_lost_frames=sum(r.get('diagnostic_search_main_target_count', 0) == 0 for r in raw_present),
            union_crop_lost_frames=sum(r.get('diagnostic_search_union_target_count', 0) == 0 for r in raw_present),
            backup_added_target_frames=len(added),
            backup_sampled_target_frames=sum(r.get('diagnostic_search_sampled_backup_target_count', 0) > 0
                                              for r in guarded),
            correct_trusted_writes_with_backup_points=sum(
                bool(r.get('trusted_write')) and bool(r['diagnostic_candidates']['selected_trusted_target'])
                and r.get('diagnostic_search_sampled_backup_target_count', 0) > 0 for r in guarded),
            over10m_on_guarded_frames=sum(r['distance'] > 10 for r in guarded),
            interpretation='passive coverage/association; not proof of causal contribution by a particular point')
        initial = len(self._initialized)
        result['same_state_oracle'] = dict(interpretation='passive_one_step_not_closed_loop',
            success=100 * (initial + sum(r['oracle']['success'] for r in records)) / max(count + initial, 1),
            precision=100 * (initial + sum(r['oracle']['precision'] for r in records)) / max(count + initial, 1))
        result['detail_level'] = 'full'
        result['stages'] = _full_stage_summary(rows, JOINT_CANDIDATES if self.v41_config.v41_joint_search
                                               else LEGACY_CANDIDATES)
        if self.v41_config.v41_joint_search:
            result['joint_candidates'] = dict(new_good_frames=sum(
                not any(v and g['exact_iou'] >= .5 for v, g in zip(r['raw_valid'][:4], r['geometry'][:4]))
                and any(v and g['exact_iou'] >= .5 for v, g in zip(r['raw_valid'][4:], r['geometry'][4:]))
                for r in records), pending_proposals=sum(r['remote_proposed'] for r in records),
                remote_selectable=sum(r['remote_selectable'] for r in records),
                coarse_proposal_frames=sum(bool(r['coarse_hits_target_bev']) for r in records),
                coarse_hit_frames=sum(any(r['coarse_hits_target_bev']) for r in records),
                remote_sampled_target_frames={f'at_least_{n}': sum(r['remote_sampled_target_count'] >= n
                    for r in records) for n in (1, 3, 10)})
        return result


def _full_stage_summary(rows, candidates=LEGACY_CANDIDATES):
    """只复用 full 已算几何；各来源用 raw_valid，不把去重误当缺失。"""
    names = ('coarse', *candidates, 'final')
    accumulators = {name: dict(valid_frames=0, selectable_frames=0, selected_frames=0,
        success_sum=0., precision_sum=0., center_sum=0., axis_yaw_sum=0.) for name in names}

    def add(name, geometry, *, selectable=True, selected=False, metric=None):
        item = accumulators[name]
        success, precision = (metric if metric is not None else
            metric_contributions(geometry['iou'], geometry['center_error_m']))
        item['valid_frames'] += 1
        item['selectable_frames'] += int(selectable)
        item['selected_frames'] += int(selected)
        item['success_sum'] += float(success)
        item['precision_sum'] += float(precision)
        item['center_sum'] += float(geometry['center_error_m'])
        item['axis_yaw_sum'] += float(geometry['axis_yaw_error_rad'])

    motion_valid = 0
    for row in rows:
        if row.get('coarse_valid') and row.get('coarse_geometry') is not None:
            add('coarse', row['coarse_geometry'], selectable=False)
        record = row['diagnostic_candidates']
        motion_valid += bool(record['prior_motion_valid'])
        for index, name in enumerate(candidates):
            if record['raw_valid'][index]:
                add(name, record['geometry'][index], selectable=record['valid'][index],
                    selected=record['selected_index'] == index)
        add('final', record['selected_geometry'], selected=True,
            metric=(row['success'], row['precision']))
    table = {}
    for name, item in accumulators.items():
        count = item['valid_frames']
        table[name] = dict(valid_frames=count, prediction_frames=len(rows),
            coverage=count / len(rows) if rows else None,
            selectable_frames=item['selectable_frames'], selected_frames=item['selected_frames'],
            success=100 * item['success_sum'] / count if count else None,
            precision=100 * item['precision_sum'] / count if count else None,
            center_error_m=item['center_sum'] / count if count else None,
            axis_yaw_error_deg=float(np.rad2deg(item['axis_yaw_sum'] / count)) if count else None)
    return dict(interpretation='passive_same_state_not_closed_loop_ablation',
        denominator='source_raw_valid_prediction_frames_only_initialization_excluded',
        prior_motion_valid_frames=motion_valid, prediction_frames=len(rows), values=table)


class CompactTrackingEvaluation:
    """summary 只算最终框；逐帧写 sink，内存不保留 rows/候选/点集合。"""
    NAMES = LEGACY_CANDIDATES
    add_singleton_tracks = TrackingEvaluation.add_singleton_tracks

    def __init__(self, config, *, row_sink=None):
        self.v41_config = normalize_config(config)
        if self.v41_config.v41_joint_search:
            self.NAMES = JOINT_CANDIDATES
        self.optimized = optimized_execution(self.v41_config)
        self.schema = SCHEMA
        self._row_sink = row_sink
        self._initialized = set()
        self._tracks = {}
        self._frames = self._predictions = 0
        self._success = self._precision = 0.
        self._selection = {name: 0 for name in self.NAMES}
        self._over5 = self._over10 = self._low_iou = self._lost_frames = 0
        self._loss_events = 0
        self._episodes = {name: self._empty_episodes() for name in ('all', 'recovered', 'unrecovered')}

    @staticmethod
    def _empty_episodes():
        return dict(episodes=0, frames=dict(count=0, total=0, max=0),
            seconds=dict(count=0, total=0., max=0.), seconds_complete=True)

    @staticmethod
    def _add_episode(group, frames, seconds):
        group['episodes'] += 1
        for name, value in (('frames', frames), ('seconds', seconds)):
            target = group[name]
            target['count'] += 1
            target['total'] += value
            target['max'] = max(target['max'], value)

    def _emit(self, row):
        if self._row_sink is not None:
            self._row_sink(row)

    def add_initialization(self, key, *, scene='unknown', timestamp=None, raw_frame_id=None, track_id=None):
        if key in self._initialized:
            return
        self._initialized.add(key)
        self._tracks[key] = dict(last_frame=0, last_time=timestamp, lost=None)
        self._frames += 1
        self._success += 1.
        self._precision += 1.
        self._emit(dict(tracklet=key, frame=0, success=1., precision=1., iou=1., distance=0.,
            initialization=True, scene_id=scene, timestamp=timestamp,
            raw_frame_id=raw_frame_id, track_id=track_id, selected_source=None))

    def _finish_episode(self, state, *, recovered, now):
        episode = state['lost']
        seconds = now - episode['start_time']
        if seconds < 0:
            raise ValueError('evaluation timestamps must be causal')
        for name in ('all', 'recovered' if recovered else 'unrecovered'):
            self._add_episode(self._episodes[name], episode['frames'], seconds)
        state['lost'] = None

    def add_batch(self, raw_rows, batch, output, commit_results=None):
        # commit_results/候选/分割纯度不参与 summary；真实状态更新已由 builder 完成。
        del commit_results
        from utils.diagnostic_transfer import tensors_to_numpy
        arrays = tensors_to_numpy(dict(predictions=output.accepted_box, targets=batch['target_box'],
            sizes=batch['box_size'], target_sizes=batch['target_box_size'],
            selected=output.selected_index), optimized=self.optimized)
        for index, raw in enumerate(raw_rows):
            request, key = raw['request'], raw['tracklet_key']
            if request.branch != 4:
                raise ValueError('evaluation must use one uninterrupted rollout per tracklet')
            if key not in self._initialized:
                if request.frame != 1:
                    raise ValueError('evaluation track must begin at its first prediction')
                first = raw['first_frame']
                self.add_initialization(key, scene=str(first.get('scene_id', 'unknown')),
                    timestamp=float(first['timestamp']),
                    raw_frame_id=first.get('raw_frame_id', first.get('frame_id')),
                    track_id=first.get('meta', {}).get('track_id'))
            if key not in self._tracks or request.frame != self._tracks[key]['last_frame'] + 1:
                raise ValueError('evaluation endpoints must be complete and causal')
            state = self._tracks[key]
            frame = raw['frames'][request.frame]
            now = float(frame['timestamp'])
            if not np.isfinite(now) or (state['last_time'] is not None and now < state['last_time']):
                raise ValueError('evaluation timestamps must be finite and causal')
            overlap, distance = box_metrics(
                LocalYawBox(arrays['predictions'][index], arrays['sizes'][index][[1, 0, 2]]),
                LocalYawBox(arrays['targets'][index], arrays['target_sizes'][index][[1, 0, 2]]),
                up_axis=(0, 0, 1), mode='benchmark_compat', dim=3)
            success, precision = metric_contributions(overlap, distance)
            selected = int(arrays['selected'][index])
            if not 0 <= selected < len(self.NAMES):
                raise ValueError('unknown v41 selected source')
            source = self.NAMES[selected]
            self._frames += 1
            self._predictions += 1
            self._success += float(success)
            self._precision += float(precision)
            self._selection[source] += 1
            self._over5 += distance > 5
            self._over10 += distance > 10
            self._low_iou += overlap < .1
            if state['lost'] is None and overlap < .1:
                state['lost'] = dict(start_time=now, frames=0)
                self._loss_events += 1
            if state['lost'] is not None:
                if overlap >= .5:
                    self._finish_episode(state, recovered=True, now=now)
                else:
                    state['lost']['frames'] += 1
                    self._lost_frames += 1
            state['last_frame'], state['last_time'] = int(request.frame), now
            self._emit(dict(tracklet=key, frame=int(request.frame), success=float(success),
                precision=float(precision), iou=float(overlap), distance=float(distance),
                initialization=False, timestamp=now, selected_source=source, selected_index=selected,
                raw_frame_id=frame.get('raw_frame_id', frame.get('frame_id')),
                track_id=frame.get('meta', {}).get('track_id'), scene_id=str(frame.get('scene_id', 'unknown'))))
            if request.ends_window:
                if state['lost'] is not None:
                    self._finish_episode(state, recovered=False, now=now)
                del self._tracks[key]

    def summary(self):
        # 未完成工程评测仍报告截至最后一帧的右删失；不修改累计器，重复summary稳定。
        episodes = {name: dict(episodes=value['episodes'], frames=dict(value['frames']),
            seconds=dict(value['seconds']), seconds_complete=True) for name, value in self._episodes.items()}
        for state in self._tracks.values():
            if state['lost'] is not None:
                elapsed = state['last_time'] - state['lost']['start_time']
                for name in ('all', 'unrecovered'):
                    self._add_episode(episodes[name], state['lost']['frames'], elapsed)
        for value in episodes.values():
            for name in ('frames', 'seconds'):
                stats = value[name]
                stats['mean'] = stats['total'] / stats['count'] if stats['count'] else None
        count = episodes['all']['episodes']
        episodes.update(definition='start_iou_lt_0.1_end_iou_ge_0.5_recovery_frame_excluded',
            total_lost_frames=self._lost_frames, low_iou_frames=self._low_iou,
            unrecovered_rate=episodes['unrecovered']['episodes'] / count if count else None)
        diagnostics = dict(over5m_frames=int(self._over5), over10m_frames=int(self._over10),
            low_iou_frames=int(self._low_iou), loss_events=self._loss_events,
            recovered_events=episodes['recovered']['episodes'], unrecovered_events=episodes['unrecovered']['episodes'],
            mean_recovery_seconds=episodes['recovered']['seconds']['mean'],
            unrecovered_elapsed_seconds=episodes['unrecovered']['seconds']['total'],
            max_loss_frames=episodes['all']['frames']['max'], max_loss_seconds=episodes['all']['seconds']['max'],
            loss_iou_threshold=.1, recovery_iou_threshold=.5)
        return dict(schema=self.schema, detail_level='summary', metric_mode='benchmark_compat',
            row_schema='ct_seqtrack.v41.compact_frame.v1', frames=self._frames, total_frames=self._frames,
            initialization_frames=len(self._initialized), prediction_frames=self._predictions,
            tracklets=len(self._initialized), success=100 * self._success / max(self._frames, 1),
            precision=100 * self._precision / max(self._frames, 1), diagnostics=diagnostics,
            selection={name: dict(count=count, rate=count / self._predictions if self._predictions else None)
                       for name, count in self._selection.items()}, loss_episodes=episodes,
            coordinate_mode=self.v41_config.ct_coordinate_mode, geometry_convention=self.v41_config.v41_kitti_geometry)


def make_evaluation(config, *, full=False, row_sink=None):
    """host 每轮明确选择档位；summary 不构造详细候选台账。"""
    evaluator = TrackingEvaluation if full else CompactTrackingEvaluation
    return evaluator(config, row_sink=row_sink)
