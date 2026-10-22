"""v38 原指标复用、完整候选审计与同 run 的严格 epoch 恢复。"""
import hashlib
import json
import numpy as np

from .config import config_identity, as_v35_config, normalize_config
from .identity import SCHEMA, runtime_schema
from models.ct_v31.runtime import (capture_rng_state, restore_rng_state, move_tensors,
    loss_episode_summary, TrackingEvaluation as LegacyEvaluation)


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
        raise ValueError('v38 checkpoint runtime schema mismatch')
    if payload.get('model_schema') != SCHEMA or payload.get('config_sha256') != config_identity(config):
        raise ValueError('v38 checkpoint model/config/arm/precision identity mismatch')
    source = source_identity()
    validate_execution_identity(payload.get('execution_identity'), config, source)
    if payload.get('source') != source:
        raise ValueError('v38 checkpoint source identity mismatch; use its original source')
    if not isinstance(payload.get('run_id'), str) or not payload['run_id']:
        raise ValueError('v38 checkpoint requires a nonempty run_id')
    if payload.get('epoch_complete') is not True or payload.get('completed_epoch', 0) < 1:
        raise ValueError('v38 requires complete epoch boundaries')
    sampler = payload.get('sampler')
    if not isinstance(sampler, dict) or sampler.get('schema') != 'ct_seqtrack.v32.ready_queue.v2':
        raise ValueError('v38 requires the original v35 W sampler manifest')
    content = {key: value for key, value in sampler.items() if key != 'manifest_sha256'}
    digest = hashlib.sha256(json.dumps(content, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if digest != sampler.get('manifest_sha256'):
        raise ValueError('v38 sampler checksum mismatch')
    if sampler.get('initial_seed_policy') != config['v35_seed_policy']:
        raise ValueError('v38 initial seed policy mismatch')
    if (payload['completed_epoch'] != sampler['epoch'] + 1
            or payload['rows'] != sampler['rows']
            or payload['optimizer_steps'] != sampler['batches']):
        raise ValueError('v38 checkpoint coverage/update counters disagree')
    if training and not isinstance(payload.get('rng'), dict):
        raise ValueError('v38 resume requires RNG state')
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


class TrackingEvaluation(LegacyEvaluation):
    def __init__(self, config):
        self.v38_config = normalize_config(config)
        super().__init__(as_v35_config(self.v38_config))
        from utils.execution_profile import optimized_execution
        self.optimized = optimized_execution(self.v38_config)
        self.schema = SCHEMA
        self.record_candidates = False

    def add_batch(self, raw_rows, batch, output, commit_results=None):
        from .candidate_records import candidate_records
        first = len(self.rows)
        super().add_batch(raw_rows, batch, output, commit_results)
        records = candidate_records(batch, output, optimized=self.optimized)
        predictions = [row for row in self.rows[first:] if not row['initialization']]
        for row, record in zip(predictions, records):
            row.update(diagnostic_candidates=record, output_utility=record['selected_utility'],
                legacy_q0_quality=record['legacy_q0_quality'], q0_trusted_score=record['trusted_score'],
                trusted_score=float(row.get('commit_quality', 0.)) if output.verification is not None else None,
                q0_is_fallback=record['q0_is_fallback'], selected_source=record['selected_source'],
                trusted_write=bool(row.get('supported')))
            record['mechanism']['actual_episode_recovery'] = row.get('recovery_seconds') is not None

    def summary(self):
        result = super().summary()
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
        initial = len(self._initialized)
        result['same_state_oracle'] = dict(interpretation='passive_one_step_not_closed_loop',
            success=100 * (initial + sum(r['oracle']['success'] for r in records)) / max(count + initial, 1),
            precision=100 * (initial + sum(r['oracle']['precision'] for r in records)) / max(count + initial, 1))
        return result


