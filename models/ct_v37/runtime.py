"""v37 原指标复用、完整候选审计与同 run 的严格 epoch 恢复。"""
import hashlib
import json

from .config import config_identity, as_v35_config
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
        raise ValueError('v37 checkpoint runtime schema mismatch')
    if payload.get('model_schema') != SCHEMA or payload.get('config_sha256') != config_identity(config):
        raise ValueError('v37 checkpoint model/config/arm/precision identity mismatch')
    source = source_identity()
    validate_execution_identity(payload.get('execution_identity'), config, source)
    if payload.get('source') != source:
        raise ValueError('v37 checkpoint source identity mismatch; use its original source')
    if not isinstance(payload.get('run_id'), str) or not payload['run_id']:
        raise ValueError('v37 checkpoint requires a nonempty run_id')
    if payload.get('epoch_complete') is not True or payload.get('completed_epoch', 0) < 1:
        raise ValueError('v37 requires complete epoch boundaries')
    sampler = payload.get('sampler')
    if not isinstance(sampler, dict) or sampler.get('schema') != 'ct_seqtrack.v32.ready_queue.v2':
        raise ValueError('v37 requires the original v35 W sampler manifest')
    content = {key: value for key, value in sampler.items() if key != 'manifest_sha256'}
    digest = hashlib.sha256(json.dumps(content, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if digest != sampler.get('manifest_sha256'):
        raise ValueError('v37 sampler checksum mismatch')
    if sampler.get('initial_seed_policy') != config['v35_seed_policy']:
        raise ValueError('v37 initial seed policy mismatch')
    if (payload['completed_epoch'] != sampler['epoch'] + 1
            or payload['rows'] != sampler['rows']
            or payload['optimizer_steps'] != sampler['batches']):
        raise ValueError('v37 checkpoint coverage/update counters disagree')
    if training and not isinstance(payload.get('rng'), dict):
        raise ValueError('v37 resume requires RNG state')
    return payload


class TrackingEvaluation(LegacyEvaluation):
    """评分完全沿用 v35；新增台账仅在 accepted 提交后读取 GT。"""
    def __init__(self, config):
        super().__init__(as_v35_config(config, arm=config.v37_arm))
        from utils.execution_profile import optimized_execution
        self.optimized = optimized_execution(config)
        self.schema = SCHEMA
        self.record_candidates = False
        self._last_commit = {}

    def add_batch(self, raw_rows, batch, output, commit_results=None):
        from .candidate_records import candidate_records
        first = len(self.rows)
        super().add_batch(raw_rows, batch, output, commit_results)
        records = candidate_records(batch, output, optimized=self.optimized)
        predicted = [row for row in self.rows[first:] if not row['initialization']]
        for row, candidates in zip(predicted, records):
            row['diagnostic_candidates'] = candidates
            row['output_utility'] = candidates['selected_utility']
            row['commit_quality'] = candidates['selected_commit_quality']
            row['q0_is_fallback'] = candidates['q0_is_fallback']
            row['candidate_oracle'] = candidates['oracle']
            mechanism = candidates['mechanism']
            mechanism['actual_episode_recovery'] = row.get('recovery_seconds') is not None
            mechanism['correct_memory_write'] = bool(row.get('memory_write')
                and not row.get('memory_wrong_write', True) and candidates['selected_geometry']['exact_iou'] >= .5)
            mechanism['correct_trusted_write'] = bool(row.get('supported')
                and candidates['selected_geometry']['exact_iou'] >= .5)
            row['previous_commit'] = self._last_commit.get(row['tracklet'])
            self._last_commit[row['tracklet']] = dict(frame=row['frame'], selected_index=row['selected_index'],
                commit_quality=row['commit_quality'], supported=row.get('supported'),
                memory_write=row.get('memory_write'), weak_age=row.get('weak_age'),
                supported_age=row.get('supported_age'), trusted_velocity_valid=row.get('trusted_velocity_valid'))

    def summary(self):
        result = super().summary()
        rows = [row for row in self.rows if not row['initialization']]
        count = len(rows)
        diagnostics = result['diagnostics']
        for name in ('good_mode_exists', 'bad_q0_good_mode', 'good_mode_selected',
                     'seed_good_refinement_bad', 'seed_bad_refinement_good',
                     'recovery_opportunity', 'qualifying_mode_count', 'good_seed_count',
                     'seed_good_refinement_good', 'actual_episode_recovery',
                     'correct_memory_write', 'correct_trusted_write'):
            diagnostics[name] = sum(row.get('diagnostic_candidates', {}).get('mechanism', {}).get(name, 0)
                                    for row in rows)
        diagnostics['propagation_only_frames'] = sum(row.get('propagation_only', False) for row in rows)
        diagnostics['over5m_frames'] = sum(row['distance'] > 5 for row in rows)
        diagnostics['over10m_frames'] = sum(row['distance'] > 10 for row in rows)
        # oracle 只在同一实际递推状态上逐帧计算，不替代真实 rollout 成绩。
        initial = len(self._initialized)
        result['same_state_oracle'] = dict(interpretation='passive_one_step_not_rerollout',
            success=100 * (initial + sum(row['candidate_oracle']['success'] for row in rows)) / max(count + initial, 1),
            precision_paired=100 * (initial + sum(row['candidate_oracle']['precision_paired'] for row in rows)) / max(count + initial, 1))
        for candidate in ('q0', 'prior'):
            geometry = [row['diagnostic_candidates'][candidate + '_geometry'] for row in rows]
            result['same_state_' + candidate] = dict(interpretation='passive_one_step_not_rerollout',
                success=100 * (initial + sum(value['success'] for value in geometry)) / max(count + initial, 1),
                precision=100 * (initial + sum(value['precision'] for value in geometry)) / max(count + initial, 1))
        result['same_state_q0']['propagation_alias_frames'] = sum(row['q0_is_fallback'] for row in rows)
        return result
