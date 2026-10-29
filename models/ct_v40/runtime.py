"""v40 指标、完整候选审计与同 run 的严格 epoch 恢复。"""
import hashlib
import json
import random
import numpy as np
import torch

from .config import config_identity, normalize_config
from .identity import SCHEMA, runtime_schema
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

def move_tensors(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}

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

def resume_payload(config, *, completed_epoch, complete, rows, steps, sampler=None, run_id=None):
    from .entry import source_identity
    from utils.execution_profile import execution_identity
    source = source_identity()
    return dict(schema=runtime_schema(config), model_schema=SCHEMA,
        config_sha256=config_identity(config), completed_epoch=int(completed_epoch),
        epoch_complete=bool(complete), rows=int(rows), optimizer_steps=int(steps),
        sampler=sampler, rng=capture_rng_state(), source=source, run_id=run_id,
        execution_identity=execution_identity(config, source))

def validate_resume_payload(payload, config, *, training=True):
    from .entry import source_identity
    from utils.execution_profile import validate_execution_identity
    if not isinstance(payload, dict) or payload.get('schema') != runtime_schema(config):
        raise ValueError('v40 checkpoint runtime schema mismatch')
    if payload.get('model_schema') != SCHEMA or payload.get('config_sha256') != config_identity(config):
        raise ValueError('v40 checkpoint model/config/arm/precision identity mismatch')
    source = source_identity()
    validate_execution_identity(payload.get('execution_identity'), config, source)
    if payload.get('source') != source:
        raise ValueError('v40 checkpoint source identity mismatch; use its original source')
    if not isinstance(payload.get('run_id'), str) or not payload['run_id']:
        raise ValueError('v40 checkpoint requires a nonempty run_id')
    if payload.get('epoch_complete') is not True or payload.get('completed_epoch', 0) < 1:
        raise ValueError('v40 requires complete epoch boundaries')
    sampler = payload.get('sampler')
    if not isinstance(sampler, dict) or sampler.get('schema') != 'ct_seqtrack.v32.ready_queue.v2':
        raise ValueError('v40 requires the original v35 W sampler manifest')
    content = {key: value for key, value in sampler.items() if key != 'manifest_sha256'}
    digest = hashlib.sha256(json.dumps(content, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if digest != sampler.get('manifest_sha256'):
        raise ValueError('v40 sampler checksum mismatch')
    if sampler.get('initial_seed_policy') != config['v35_seed_policy']:
        raise ValueError('v40 initial seed policy mismatch')
    if (payload['completed_epoch'] != sampler['epoch'] + 1
            or payload['rows'] != sampler['rows']
            or payload['optimizer_steps'] != sampler['batches']):
        raise ValueError('v40 checkpoint coverage/update counters disagree')
    if training and not isinstance(payload.get('rng'), dict):
        raise ValueError('v40 resume requires RNG state')
    return payload

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
    def __init__(self, config):
        self.v40_config = normalize_config(config)
        self.optimized = optimized_execution(self.v40_config)
        self.schema = SCHEMA
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

    def _add_batch_metrics(self, raw_rows, batch, output, commit_results=None):
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
        fine = output.decoder.q0_box.detach().cpu().numpy()
        diagnostics = {key: value.detach().cpu().numpy() for key, value in batch.items()
                       if key.startswith('diagnostic_') and torch.is_tensor(value)}
        # 仅记录实际前向已产生的摘要；不读取 GT，不影响候选或 accepted 提交。
        decoder = getattr(output, 'decoder', None)
        for name, owner, field, shape in (
                ('diagnostic_history_support', observation, 'history_support', (len(raw_rows), 3, 3)),
                ('diagnostic_current_support', observation, 'current_support', (len(raw_rows), 3)),
                ('diagnostic_query_context_norm', decoder, 'query_context_norm', (len(raw_rows),))):
            value = getattr(owner, field, None)
            if value is not None:
                if tuple(value.shape) != shape or not bool(torch.isfinite(value).all()):
                    raise ValueError('invalid v40 passive diagnostic: ' + field)
                diagnostics[name] = value.detach().cpu().numpy()
        # 记录有符号世界坐标与尺寸，允许离线分解 XYZ/yaw/IoU。
        # 只作 detach 后的导出，不把 GT 几何送回 forward 或 commit。
        anchors = batch['anchor_box'].detach().cpu().double().numpy()
        for name, value in (('coarse', coarse), ('fine', fine),
                            ('accepted', predictions), ('target', targets)):
            if value is not None:
                world = value.astype(np.float64, copy=True)
                world[:, :3] += anchors[:, :3]
                diagnostics['diagnostic_' + name + '_box_world'] = world
        diagnostics['diagnostic_anchor_box_world'] = anchors
        diagnostics['diagnostic_first_frame_size_lwh'] = sizes
        diagnostics['diagnostic_target_size_lwh'] = target_sizes
        decoder = getattr(output, 'decoder', None)
        for field in ('local_neighbor_count', 'local_selected_count',
                      'local_mean_support', 'local_delta_norm'):
            value = getattr(decoder, field, None)
            if value is not None:
                if tuple(value.shape) != (len(raw_rows), 8) or not bool(torch.isfinite(value).all()):
                    raise ValueError('invalid v40 passive diagnostic: ' + field)
                diagnostics['diagnostic_' + field] = value.detach().cpu().numpy()
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

    def add_batch(self, raw_rows, batch, output, commit_results=None):
        from .candidate_records import candidate_records
        first = len(self.rows)
        self._add_batch_metrics(raw_rows, batch, output, commit_results)
        records = candidate_records(batch, output, optimized=self.optimized)
        predictions = [row for row in self.rows[first:] if not row['initialization']]
        for row, record in zip(predictions, records):
            row.update(diagnostic_candidates=record, output_utility=record['selected_utility'],
                legacy_q0_quality=record['legacy_q0_quality'], q0_trusted_score=record['trusted_score'],
                trusted_score=float(row.get('commit_quality', 0.)) if output.verification is not None else None,
                q0_is_fallback=record['q0_is_fallback'], selected_source=record['selected_source'],
                trusted_write=bool(row.get('supported')))
            record['mechanism']['actual_episode_recovery'] = row.get('recovery_seconds') is not None

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
            propagation_only_frames=sum(row['selected_source'] != 'q0' for row in rows))
        count = len(rows)
        result['selection'] = {name: dict(count=sum(r['selected_source'] == name for r in rows),
            rate=sum(r['selected_source'] == name for r in rows) / count if count else None)
            for name in ('q0', 'prior', 'trusted_propagation')}
        writes = [row for row in rows if row.get('supported')]
        wrong = sum(not row['diagnostic_candidates']['trusted_target'] for row in writes)
        result['trusted'] = dict(writes=len(writes), coverage=len(writes) / count if count else None,
            wrong_writes=wrong, wrong_write_rate=wrong / len(writes) if writes else None,
            position_updates_velocity_invalid=sum(not row.get('trusted_velocity_valid', False) for row in writes),
            velocity_updates=sum(row.get('trusted_velocity_updated', False) for row in rows),
            longest_untrusted_streak=max((r.get('consecutive_untrusted_outputs', 0) for r in rows), default=0),
            longest_propagation_streak=max((r.get('consecutive_propagations', 0) for r in rows), default=0),
            max_supported_age_seconds=max((r.get('supported_age', 0.) for r in rows), default=0.),
            max_velocity_age_seconds=max((r.get('trusted_velocity_age', 0.) for r in rows), default=0.))
        result['trusted_reliability'] = reliability_summary(records)
        result['motion_slices'] = motion_slices(self.rows)
        guarded = [row for row in rows if row.get('diagnostic_search_triggered', False)]
        raw_present = [row for row in rows if row.get('diagnostic_search_raw_target_count', 0) > 0]
        added = [row for row in guarded
                 if row.get('diagnostic_search_union_target_count', 0)
                 > row.get('diagnostic_search_main_target_count', 0)]
        result['search'] = dict(policy=self.v40_config.v40_search_policy,
            triggered_frames=len(guarded),
            raw_target_present_frames=len(raw_present),
            main_crop_lost_frames=sum(r.get('diagnostic_search_main_target_count', 0) == 0 for r in raw_present),
            union_crop_lost_frames=sum(r.get('diagnostic_search_union_target_count', 0) == 0 for r in raw_present),
            backup_added_target_frames=len(added),
            backup_sampled_target_frames=sum(r.get('diagnostic_search_sampled_backup_target_count', 0) > 0
                                              for r in guarded),
            correct_trusted_writes_with_backup_points=sum(
                bool(r.get('trusted_write')) and bool(r['diagnostic_candidates']['trusted_target'])
                and r.get('diagnostic_search_sampled_backup_target_count', 0) > 0 for r in guarded),
            over10m_on_guarded_frames=sum(r['distance'] > 10 for r in guarded),
            interpretation='passive coverage/association; not proof of causal contribution by a particular point')
        initial = len(self._initialized)
        result['same_state_oracle'] = dict(interpretation='passive_one_step_not_closed_loop',
            success=100 * (initial + sum(r['oracle']['success'] for r in records)) / max(count + initial, 1),
            precision=100 * (initial + sum(r['oracle']['precision'] for r in records)) / max(count + initial, 1))
        return result
