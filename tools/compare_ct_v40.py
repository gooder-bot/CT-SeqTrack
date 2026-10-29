"""V40 四组只读验收：原始 R/C、final60/late3、完整排程与逐帧证据。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ''):
    sys.path.insert(0, str(ROOT))

from tools import _tracking_audit as original
from models.ct_v40.config import (REGISTRATION_FILE, REGISTRATION_SCHEMA, RECIPES,
    normalize_config, config_identity, recipe_identity, load_config)
from models.ct_v40.identity import (SCHEMA, COMPARE_SCHEMA, CANDIDATE_SCHEMA,
                                    RESUME_SCHEMA, TRAINING_AUDIT_SCHEMA, RELIABILITY_PLOT_SCHEMA)
from utils.execution_profile import validate_execution_identity

base = original
require, EvidenceError = original.require, original.EvidenceError
LABELS = tuple(RECIPES)
SOURCES = ['q0', 'prior', 'trusted_propagation']
REPORT_SCHEMA = COMPARE_SCHEMA


def load_registration():
    from models.ct_v40.registration import verify_registration
    from models.ct_v40.entry import validate_formal_registration
    value = verify_registration()
    for row in value['runs']:
        config = load_config(ROOT / row['cfg'])
        validate_formal_registration(config)
        row['config_sha256'] = config_identity(config)
        row['training_execution_recipe'] = recipe_identity(config)
    require(value.get('registered_main_model') == 'full', 'fixed Full main model required')
    require(value.get('policy_activation') == 'active_from_epoch1', 'no shadow curriculum is registered')
    return value


_SAMPLER_CACHE = {}


def validate_sampler(value, config, epoch, label):
    """用当前登记 V40 重建全部端点，禁止借旧版本 sampler 放行。"""
    from models.ct_v40.data import ReadyQueueBatchSampler
    require(isinstance(value, dict), label + ': missing sampler')
    require(value.get('manifest_sha256') == original.digest({key: item for key, item in value.items()
        if key != 'manifest_sha256'}), label + ': sampler checksum mismatch')
    original.equal_fields(value, dict(schema=ReadyQueueBatchSampler.SCHEMA, epoch=epoch - 1,
        batch_size=config['batch_size'], seed=config['seed'],
        short_window=config['v31_short_window'], long_window=config['v31_long_window'],
        curriculum_epochs=config['v31_curriculum_epochs'], reserve_windows=config['v32_reserve_windows'],
        seed_translation=config['v32_seed_translation'], seed_yaw_degrees=config['v32_seed_yaw_degrees']), label)
    require(value.get('training') is True, label + ': training sampler required')
    source = original._sha256(value.get('source_sha256'), label + '.source_sha256')
    require(value.get('initial_seed_policy') == ReadyQueueBatchSampler.INITIAL_SEED_POLICY,
            label + ': initial exact/other perturbed seed policy mismatch')
    checksum = original.digest(value)
    if checksum not in _SAMPLER_CACHE:
        keys = ('lengths', 'batch_size', 'seed', 'training', 'short_window', 'long_window',
                'curriculum_epochs', 'source_sha256', 'reserve_windows', 'seed_translation',
                'seed_yaw_degrees', 'initial_seed_policy')
        sampler = ReadyQueueBatchSampler(**{key: value[key] for key in keys})
        sampler.set_epoch(epoch - 1)
        require(sampler.state_dict() == value, label + ': actual endpoint plan cannot be reconstructed')
        _SAMPLER_CACHE[checksum] = True
    return source


def _array(value, shape, label):
    result = np.asarray(value, dtype=np.float64)
    require(result.shape == shape and np.isfinite(result).all(), label + ': nonfinite/wrong shape')
    return result


def _candidate_geometry(row, record, boxes, label):
    """从独立导出的世界 GT/anchor/尺寸还原 FP32 输入，重算所有候选。"""
    from models.ct_v40.candidate_records import _geometry_both
    anchor = _array(row.get('diagnostic_anchor_box_world'), (4,), label + '.anchor')
    target = _array(row.get('diagnostic_target_box_world'), (4,), label + '.target').copy()
    size = _array(row.get('diagnostic_first_frame_size_lwh'), (3,), label + '.size')
    target_size = _array(row.get('diagnostic_target_size_lwh'), (3,), label + '.target_size')
    require(bool((size > 0).all() and (target_size > 0).all()), label + ': nonpositive box size')
    target[:3] -= anchor[:3]
    # 原台账几何的 yaw/XY 误差也使用 FP32 输入，避免精度变化重定义门边界。
    expected = [_geometry_both(box.astype(np.float32), target.astype(np.float32),
        size.astype(np.float32), target_size.astype(np.float32)) for box in boxes]
    geometry = record.get('geometry')
    require(isinstance(geometry, list) and len(geometry) == 3, label + ': missing candidate geometry')
    for index, (actual, calculated) in enumerate(zip(geometry, expected)):
        require(isinstance(actual, dict), label + ': invalid candidate geometry')
        for name, value in calculated.items():
            original.close(actual.get(name), value, label + f'.geometry[{index}].' + name, 2e-7)
    selected = record['selected_index']
    for name, index in (('q0_geometry', 0), ('prior_geometry', 1),
                        ('trusted_propagation_geometry', 2), ('selected_geometry', selected)):
        actual = record.get(name)
        require(isinstance(actual, dict), label + ': missing ' + name)
        for field, value in expected[index].items():
            original.close(actual.get(field), value, label + '.' + name + '.' + field, 2e-7)
    for name, index in (('fine', 0), ('accepted', selected)):
        world = _array(row.get('diagnostic_' + name + '_box_world'), (4,), label + '.' + name)
        reconstructed = boxes[index].copy()
        reconstructed[:3] += anchor[:3]
        require(bool(np.allclose(world, reconstructed, atol=2e-7, rtol=0)),
                label + ': exported ' + name + ' box differs from candidate')
    return expected


def validate_candidates(row, label, arm):
    record = row.get('diagnostic_candidates')
    require(isinstance(record, dict) and record.get('schema') == CANDIDATE_SCHEMA, label + ': missing candidate schema')
    require(record.get('names') == SOURCES, label + ': candidate order mismatch')
    valid, raw = record.get('valid'), record.get('raw_valid')
    require(all(isinstance(v, list) and len(v) == 3 and all(type(x) is bool for x in v)
                for v in (valid, raw)), label + ': invalid validity masks')
    boxes = _array(record.get('boxes_anchor_relative'), (3, 4), label + '.boxes')
    require(raw[1:] == [arm != 'b0', True] and any(valid), label + ': propagation availability differs from arm')
    require(record.get('q0_is_fallback') is (not raw[0]), label + ': q0 provenance mismatch')
    expected_valid, aliases = list(raw), [-1, -1, -1]
    # 模型使用 FP32；不得用 NumPy float64 在边界重新定义已登记去重。
    float_boxes = torch.as_tensor(boxes, dtype=torch.float32)
    if not raw[0]:
        aliases[0] = 2 if arm == 'b0' else 1
    for later in range(1, 3):
        for earlier in range(later):
            delta = float_boxes[later] - float_boxes[earlier]
            angle = torch.atan2(delta[3].sin(), delta[3].cos()).abs()
            if expected_valid[later] and expected_valid[earlier] and bool(delta[:3].abs().amax() <= 1e-4) and bool(angle <= 1e-5):
                expected_valid[later], aliases[later] = False, earlier
    require(valid == expected_valid and record.get('alias_of') == aliases, label + ': candidate dedup mismatch')
    selected = record.get('selected_index')
    require(type(selected) is int and 0 <= selected < 3 and valid[selected], label + ': invalid selection')
    require(row.get('selected_index') == selected and record.get('selected_source') == SOURCES[selected]
            and row.get('selected_source') == SOURCES[selected], label + ': source mismatch')
    if arm == 'full':
        utility = _array(record.get('utility'), (3,), label + '.utility')
        require(bool(((utility >= 0) & (utility <= 1)).all()), label + ': utility outside [0,1]')
        require(selected == int(np.where(valid, utility, -np.inf).argmax()), label + ': selection differs from strict gain/tie rule')
        original.close(record.get('selected_utility'), utility[selected], label + '.selected_utility', 2e-7)
    else:
        require(record.get('utility') is None and selected == (0 if raw[0] else (2 if arm == 'b0' else 1)),
                label + ': observation-only policy changed')
    support = record.get('support', {})
    ids = support.get('raw_ids')
    require(isinstance(ids, list) and all(type(v) is int and v >= 0 for v in ids)
            and len(ids) == len(set(ids)) == support.get('count'), label + ': support raw IDs/count invalid')
    count, target_count = support['count'], support.get('target_count')
    require(type(count) is int and 0 <= count <= 1024, label + ': invalid support count')
    require(type(target_count) is int and 0 <= target_count <= count, label + ': target support invalid')
    purity = target_count / count if count else 0.
    original.close(support.get('target_purity'), purity, label + '.purity', 1e-7)
    geometry = _candidate_geometry(row, record, boxes, label)
    real_q0 = raw[0]
    target = real_q0 and geometry[0]['exact_iou'] >= .5 and count >= 3 and purity >= .5
    require(record.get('trusted_valid') is real_q0 and record.get('trusted_target') is (target if real_q0 else None),
            label + ': trusted supervision target mismatch')
    require(0 <= base.number(record.get('legacy_q0_quality'), label + '.legacy_q0_quality') <= 1,
            label + ': legacy q0 quality invalid')
    if arm != 'full' or not real_q0:
        require(record.get('trusted_score') is None, label + ': nonexistent T score must remain missing')
    score = record.get('trusted_score') if arm == 'full' else record.get('legacy_q0_quality')
    if real_q0:
        require(0 <= base.number(score, label + '.trusted_score') <= 1, label + ': trusted score invalid')
    eligible = selected == 0 and real_q0 and count >= 3 and score >= .5
    require(record.get('trusted_eligible') is eligible, label + ': trusted gate mismatch')
    actual_count = row.get('geometric_fg_count')
    require(type(actual_count) is int and actual_count >= 0, label + ': missing actual geometry support')
    if arm == 'full':
        require(actual_count == (count if selected == 0 and real_q0 else 0),
                label + ': Full actual support differs from canonical q0 support')
    # B1 保留原 world-space 几何计算，以真实提交计数核验；canonical 台账不改旧策略。
    actual_eligible = eligible if arm == 'full' else selected == 0 and real_q0 and actual_count >= 3 and score >= .5
    require(row.get('trusted_write') is actual_eligible and row.get('supported') is actual_eligible,
            label + ': trusted commit mismatch')
    require(row.get('measurement_eligible') is actual_eligible,
            label + ': actual measurement eligibility mismatch')
    require('memory_write' not in row, label + ': clean1 has no identity-memory write field')
    for metric in original.METRICS:
        original.close(geometry[selected].get(metric), row[metric], label + '.selected_' + metric, 2e-7)
    original.close(row.get('iou'), geometry[selected]['iou'], label + '.selected_iou', 2e-7)
    original.close(row.get('distance'), geometry[selected]['center_error_m'], label + '.selected_distance', 2e-7)
    true_utility = [(item['success'] + item['precision']) / 2 for item in geometry]
    oracle = max((k for k in range(3) if valid[k]), key=lambda k: true_utility[k])
    saved_oracle = record.get('oracle')
    require(isinstance(saved_oracle, dict) and type(saved_oracle.get('index')) is int
            and saved_oracle['index'] == oracle, label + ': oracle selection mismatch')
    for field, value in dict(success=geometry[oracle]['success'], precision=geometry[oracle]['precision'],
                             utility=true_utility[oracle]).items():
        original.close(saved_oracle.get(field), value, label + '.oracle.' + field, 2e-7)
    mechanism = record.get('mechanism')
    require(isinstance(mechanism, dict), label + ': missing candidate mechanism')
    expected_mechanism = dict(replaced_real_q0=bool(real_q0 and selected != 0),
        erroneous_takeover=bool(real_q0 and selected != 0 and true_utility[selected] < true_utility[0]),
        beneficial_takeover=bool(real_q0 and selected != 0 and true_utility[selected] > true_utility[0]),
        recovered_bad_q0=bool(real_q0 and geometry[0]['exact_iou'] < .1 and geometry[selected]['exact_iou'] >= .5))
    require(all(mechanism.get(key) is value for key, value in expected_mechanism.items()),
            label + ': candidate mechanism differs from reconstructed geometry')


def validate_search_rows(rows, config, label):
    """严格核对逐帧真实提交→下一帧搜索；覆盖计数只作被动证据。"""
    full = config['v40_arm'] == 'full'
    previous, trusted_time = {}, {}
    counts = ('main_count', 'backup_only_count', 'sampled_main', 'sampled_backup',
              'raw_target_count', 'main_target_count', 'union_target_count',
              'sampled_main_target_count', 'sampled_backup_target_count')
    predictions = []
    for key, row in sorted(rows.items()):
        track, frame = key
        stamp = original.number(row.get('timestamp'), label + '.timestamp')
        if frame == 0:
            trusted_time[track] = stamp
            continue
        prefix = label + f'.{track}.{frame}.search'
        require(track in trusted_time, prefix + ': missing causal initialization')
        prev = previous.get(track)
        require((frame == 1 and prev is None) or (prev is not None and prev['frame'] == frame - 1),
                prefix + ': missing causal predecessor')
        expected_previous = -1 if prev is None else int(prev['measurement_eligible'])
        require(type(row.get('diagnostic_previous_measurement_eligible')) is int
                and row['diagnostic_previous_measurement_eligible'] == expected_previous,
                prefix + ': previous eligibility does not match actual commit')
        triggered = full and expected_previous == 0
        reason = ('policy_disabled' if not full else 'initialization' if prev is None else
                  'accepted_eligible' if expected_previous == 1 else 'accepted_untrusted')
        reason_id = {'policy_disabled': 0, 'initialization': 1, 'accepted_eligible': 2, 'accepted_untrusted': 3}[reason]
        require(row.get('diagnostic_search_triggered') is triggered
                and row.get('diagnostic_search_backup_active') is triggered,
                prefix + ': Full-only causal trigger mismatch')
        require(row.get('search_reason') == reason and type(row.get('diagnostic_search_reason_id')) is int
                and row['diagnostic_search_reason_id'] == reason_id, prefix + ': trigger reason mismatch')
        main = _array(row.get('diagnostic_search_main_box_world'), (4,), prefix + '.main_box')
        backup = _array(row.get('diagnostic_search_backup_box_world'), (4,), prefix + '.backup_box')
        anchor = _array(row.get('diagnostic_anchor_box_world'), (4,), prefix + '.anchor')
        require(bool(np.array_equal(main, anchor)), prefix + ': search changed the accepted coordinate anchor')
        if not triggered:
            require(bool(np.array_equal(main, backup)), prefix + ': inactive backup must copy main')
        for suffix, expected in (('main_quota', 768), ('backup_quota', 256)):
            value = row.get('diagnostic_search_' + suffix)
            require(type(value) is int and value == expected, prefix + ': registered quota mismatch')
        age = original.number(row.get('diagnostic_search_trusted_age'), prefix + '.trusted_age')
        elapsed = original.number(row.get('diagnostic_search_backup_elapsed'), prefix + '.elapsed')
        original.close(age, stamp - trusted_time[track], prefix + '.trusted_age', 2e-5)
        velocity_valid = False if prev is None else prev.get('trusted_velocity_valid')
        require(type(velocity_valid) is bool, prefix + ': missing preceding velocity validity')
        expected_elapsed = min(stamp - trusted_time[track], config['v40_search_max_age']) if triggered and velocity_valid else 0.
        require(age >= 0 and 0 <= elapsed <= config['v40_search_max_age'], prefix + ': invalid search-only cap')
        original.close(elapsed, expected_elapsed, prefix + '.elapsed', 2e-6)
        values = {name: row.get('diagnostic_search_' + name) for name in counts}
        require(all(type(value) is int and value >= 0 for value in values.values()), prefix + ': missing/noninteger coverage')
        n, extra = values['main_count'], values['backup_only_count']
        sm, sb = values['sampled_main'], values['sampled_backup']
        raw, main_fg, union_fg = (values[name] for name in ('raw_target_count', 'main_target_count', 'union_target_count'))
        sm_fg, sb_fg = values['sampled_main_target_count'], values['sampled_backup_target_count']
        expected_main, expected_backup = min(n, 768), min(extra, 256)
        spare = 1024 - expected_main - expected_backup
        extra_main = min(spare, n - expected_main)
        expected_main += extra_main
        expected_backup += min(spare - extra_main, extra - expected_backup)
        require((sm, sb) == (expected_main, expected_backup) and sm + sb == min(n + extra, 1024),
                prefix + ': quota/spillback/sample budget mismatch')
        require(raw >= union_fg >= main_fg and main_fg <= n and union_fg - main_fg <= extra
                and sm_fg <= min(sm, main_fg) and sb_fg <= min(sb, union_fg - main_fg)
                and main_fg - sm_fg <= n - sm
                and union_fg - main_fg - sb_fg <= extra - sb,
                prefix + ': target coverage conservation mismatch')
        if not triggered:
            require(extra == sb == sb_fg == 0 and union_fg == main_fg,
                    prefix + ': inactive backup contributed points')
        for field, expected in dict(diagnostic_target_count=raw, diagnostic_crop_target_count=union_fg,
                diagnostic_sampled_target_count=sm_fg + sb_fg, diagnostic_crop_point_count=n + extra,
                diagnostic_sampled_point_count=sm + sb).items():
            require(type(row.get(field)) is int and row[field] == expected, prefix + ': mismatch with ' + field)
        support = row['diagnostic_candidates']['support']
        require(support['count'] <= sm + sb and support['target_count'] <= sm_fg + sb_fg
                and support['count'] - support['target_count'] <= sm + sb - sm_fg - sb_fg,
                prefix + ': q0 support exceeds the actual sampled point set')
        for field in ('measurement_eligible', 'trusted_velocity_valid'):
            require(type(row.get(field)) is bool, prefix + ': missing actual commit ' + field)
        require(row['measurement_eligible'] is row.get('trusted_write'), prefix + ': measurement/write mismatch')
        if row['measurement_eligible']:
            trusted_time[track] = stamp
        previous[track] = row
        predictions.append(row)
    guarded = [row for row in predictions if row['diagnostic_search_triggered']]
    raw_present = [row for row in predictions if row['diagnostic_search_raw_target_count'] > 0]
    return dict(policy=config['v40_search_policy'], triggered_frames=len(guarded),
        raw_target_present_frames=len(raw_present),
        main_crop_lost_frames=sum(row['diagnostic_search_main_target_count'] == 0 for row in raw_present),
        union_crop_lost_frames=sum(row['diagnostic_search_union_target_count'] == 0 for row in raw_present),
        backup_added_target_frames=sum(row['diagnostic_search_union_target_count'] > row['diagnostic_search_main_target_count'] for row in guarded),
        backup_sampled_target_frames=sum(row['diagnostic_search_sampled_backup_target_count'] > 0 for row in guarded),
        correct_trusted_writes_with_backup_points=sum(row['trusted_write']
            and bool(row['diagnostic_candidates']['trusted_target'])
            and row['diagnostic_search_sampled_backup_target_count'] > 0 for row in guarded),
        over10m_on_guarded_frames=sum(row['distance'] > 10 for row in guarded),
        interpretation='passive coverage/association; not proof of causal contribution by a particular point')


def validate_search_summary(value, rows, config, label):
    expected = validate_search_rows(rows, config, label)
    require(isinstance(value, dict) and value == expected,
            label + ': search summary differs from original frame records')
    return expected


def _metrics_identity(metrics, epoch, label):
    original.equal_fields(metrics, dict(schema=SCHEMA, checkpoint_epoch=epoch,
        metric_mode='benchmark_compat', **original.COUNTS), label + '.metrics')
    require(metrics.get('complete_coverage') is True, label + ': incomplete coverage')


def read_run(directory, spec, registration):
    root, arm, label = Path(directory).resolve(), spec['arm'], spec['label']
    source, provenance = registration['source'], {}
    def read(relative, yaml_file=False):
        path = root / relative
        value = original.read_object(path, yaml_file=yaml_file)
        provenance[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        return value
    manifest = read('run_manifest.json')
    original.equal_fields(manifest, dict(schema=SCHEMA, model='ctseqtrackv40', seed=42, arm=arm,
        temporal_backend='cfc', evaluation_rng_policy='per_checkpoint_seed_v1',
        policy_activation='active_from_epoch1'), label + '.manifest')
    require(manifest.get('enabled') == dict(B1=arm != 'b0', M=arm in ('b1_mtp', 'full'),
        B2=False, B3=arm == 'full'), label + ': enabled modules mismatch')
    require(manifest.get('source') == source and manifest.get('calibration_required') is False, label + ': source/calibration mismatch')
    run_id = manifest.get('run_id')
    require(isinstance(run_id, str) and len(run_id) == 32 and all(c in '0123456789abcdef' for c in run_id), label + ': run ID')
    expected_registration = dict(schema=REGISTRATION_SCHEMA, label=label,
        sha256=hashlib.sha256((ROOT / 'cfgs/ct_seqtrack' / REGISTRATION_FILE).read_bytes()).hexdigest())
    require(manifest.get('registration') == expected_registration, label + ': registration fingerprint mismatch')
    for field in ('common_parameter_hash', 'base_initial_parameter_hash', 'initial_parameter_hash'):
        original._sha256(manifest.get(field), label + '.' + field)
    if arm != 'b0':
        original._sha256(manifest.get('prior_parameter_hash'), label + '.prior_parameter_hash')
    config = read('resolved_config.yaml', True)
    require(dict(normalize_config(config)) == config, label + ': incomplete normalized config')
    require(config['v40_arm'] == arm and config['v40_training_recipe'] == label
            and not config['test'] and not config['ct_engineering_check'], label + ': formal recipe mismatch')
    require(config_identity(config) == manifest.get('config_sha256') == spec['config_sha256'], label + ': config mismatch')
    require(manifest.get('training_execution_recipe') == spec['training_execution_recipe'] == recipe_identity(config), label + ': execution recipe mismatch')
    validate_execution_identity(manifest.get('execution_identity'), config, source)
    for path in sorted((root / 'resume_manifests').glob('*.json')):
        original.equal_fields(read(path.relative_to(root).as_posix()), dict(schema=RESUME_SCHEMA,
            run_id=run_id, config_sha256=manifest['config_sha256'], source=source), label + '.resume')
    budget = read('training_budget.json')
    original.equal_fields(budget, dict(schema=SCHEMA, completed_epoch=60), label + '.budget')
    require(budget.get('epoch_complete') is True, label + ': epoch60 incomplete')
    training_source = validate_sampler(budget.get('sampler'), config, 60, label + '.sampler')
    require(len(list((root / 'training_audits').glob('epoch=*.json'))) == 60, label + ': sixty audits required')
    plans, total_steps, total_rows = {}, 0, 0
    for epoch in range(1, 61):
        audit = read(f'training_audits/epoch={epoch:03d}.json')
        original.equal_fields(audit, dict(schema=TRAINING_AUDIT_SCHEMA, model_schema=SCHEMA,
            run_id=run_id, config_sha256=manifest['config_sha256'], completed_epoch=epoch), label + '.audit')
        require(audit.get('epoch_complete') is True and validate_sampler(audit.get('sampler'), config, epoch, label) == training_source,
                label + ': incomplete audit/data changed')
        plans[epoch] = audit['sampler']['plan_sha256']
        require(audit['rows'] == audit['sampler']['rows']
                and audit['optimizer_steps'] == audit['sampler']['batches'], label + ': actual epoch budget differs from reconstructed sampler')
        total_steps += audit['optimizer_steps']
        total_rows += audit['rows']
    require(audit['sampler'] == budget['sampler'], label + ': final audit/budget mismatch')
    original.equal_fields(budget, dict(last_epoch_rows=audit['rows'], last_epoch_steps=audit['optimizer_steps'],
        optimizer_steps=total_steps), label + '.actual_budget')
    results = read('results.json')
    require(results.get('checkpoint_epochs') == original.EPOCHS, label + ': final window must be 58/59/60')
    require(results.get('policy_activation') == 'active_from_epoch1', label + ': no shadow policy allowed')
    rows_by_epoch, scores, diagnostics = {}, {}, {}
    for epoch in original.EPOCHS:
        relative = f'evaluation/epoch={epoch:03d}'
        path = root / relative / 'frames.jsonl'
        rows = original.read_frames(path)
        provenance[relative + '/frames.jsonl'] = hashlib.sha256(path.read_bytes()).hexdigest()
        metrics = read(relative + '/metrics.json')
        _metrics_identity(metrics, epoch, label)
        for key, row in rows.items():
            if key[1]:
                validate_candidates(row, label + f'.{epoch}.{key}', arm)
        validate_search_summary(metrics.get('search'), rows, config, label + f'.{epoch}')
        from models.ct_v40.runtime import reliability_summary
        require(metrics.get('trusted_reliability') == reliability_summary(
            [row['diagnostic_candidates'] for key, row in rows.items() if key[1]]),
            label + ': reliability summary differs from original candidate records')
        scores[epoch] = original.score(rows.values())
        for metric in original.METRICS:
            original.close(metrics.get(metric), scores[epoch][metric], label + '.' + metric, 2e-6)
        diagnostics[str(epoch)] = {key: metrics.get(key) for key in
            ('selection', 'trusted', 'trusted_reliability', 'motion_slices', 'diagnostics', 'search')}
        rows_by_epoch[epoch] = rows
    late3 = {m: math.fsum(scores[e][m] for e in original.EPOCHS) / 3 for m in original.METRICS}
    _metrics_identity(results.get('final', {}), 60, label)
    require(results['final'].get('search') == diagnostics['60']['search'],
            label + ': final search differs from verified epoch60')
    for metric in original.METRICS:
        original.close(results['final'].get(metric), scores[60][metric], label + '.final.' + metric, 2e-6)
        original.close(results.get('late3', {}).get(metric), late3[metric], label + '.late3.' + metric, 2e-6)
    return dict(directory=str(root), config=config, manifest=manifest, source=training_source,
        sampler_plans=plans, rows=rows_by_epoch, scores=scores, late3=late3, provenance=provenance,
        actual_budget=dict(completed_epoch=60, total_rows=total_rows, optimizer_steps=total_steps,
                           last_epoch_rows=audit['rows'], last_epoch_steps=audit['optimizer_steps']),
        mechanism_diagnostics=diagnostics, timing={key: results.get(key) for key in
            ('fit_wall_seconds', 'train_wall_seconds', 'validation_wall_seconds', 'evaluation_wall_seconds')})


def compare_runs(*, registered_reference, baseline, **paths):
    registration = load_registration()
    require(set(paths) == set(LABELS) and len({str(Path(p).resolve()) for p in paths.values()}) == 4,
            'four distinct complete v40 runs required')
    runs = dict(reference=base.read_run(registered_reference, 'reference'), baseline=base.read_run(baseline, 'baseline'))
    for name in ('reference', 'baseline'):
        actual = tuple(runs[name]['provenance'][f'evaluation/epoch={epoch:03d}/frames.jsonl'] for epoch in original.EPOCHS)
        require(actual == original.REGISTERED_FRAME_SHA256[name], 'fixed R/C original evidence mismatch: ' + name)
    specs = {row['label']: row for row in registration['runs']}
    for label in LABELS:
        runs[label] = read_run(paths[label], specs[label], registration)
    require(len({runs[x]['manifest']['run_id'] for x in LABELS}) == 4, 'four new run IDs required')
    require(all(runs[x]['sampler_plans'] == runs['b1']['sampler_plans'] for x in LABELS), 'sampler plans differ')
    require(len({runs[x]['manifest']['common_parameter_hash'] for x in LABELS}) == 1,
            'B0 common initialization differs across four arms')
    for key in ('prior_parameter_hash', 'prior_adapter_parameter_hash', 'base_initial_parameter_hash'):
        require(len({runs[x]['manifest'][key] for x in ('b1', 'b1_mtp', 'full')}) == 1,
                'B1 initialization differs: ' + key)
    require(runs['b0']['manifest'].get('prior_parameter_hash') is None
            and runs['b0']['manifest'].get('prior_adapter_parameter_hash') is None,
            'B0 contains a learned prior or prior adapter')
    require(runs['b1_mtp']['manifest'].get('mtp_parameter_hash')
            == runs['full']['manifest'].get('mtp_parameter_hash')
            and runs['b1_mtp']['manifest'].get('mtp_parameter_hash') is not None, 'M initialization differs')
    comparisons = [('b1_minus_b0', 'b1', 'b0'), ('mtp_minus_b1', 'b1_mtp', 'b1'),
                   ('full_minus_b1_mtp', 'full', 'b1_mtp')]
    report = base.compare_loaded(runs, candidate_labels=LABELS,
        config_differences=('v31_arm', 'v40_arm', 'v40_training_recipe', 'enable_b1', 'enable_m', 'enable_b3',
            'v40_search_policy', 'lr_milestones', 'devices'), comparisons=comparisons)
    passed = report['candidates']['full']['passed']
    report.update(schema=REPORT_SCHEMA, status='passed' if passed else 'failed',
        registered_main_model='full', selected='full' if passed else None,
        source_sha256=registration['source']['sha256'], policy_activation='active_from_epoch1',
        mechanism_diagnostics={label: runs[label]['mechanism_diagnostics'] for label in LABELS},
        timing={label: runs[label]['timing'] for label in LABELS},
        center_tail_diagnostics={label: original.tail_diagnostics(run) for label, run in runs.items()})
    report.pop('passing_rank', None)
    report['actual_budgets'] = {label: runs[label]['actual_budget'] for label in LABELS}
    # 共用旧 R/C 报告的历史预算不冒充新四组的实际训练步数。
    report['verified']['historical_reference_budget'] = {key: report['verified'].pop(key)
        for key in original.BUDGET if key in report['verified']}
    report['criterion']['ranking'] = 'Full fixed; controls or best epochs never replace it'
    report['contrast_interpretation'] = dict(b1_minus_b0='B1 added to the same B0+FA baseline',
        mtp_minus_b1='auxiliary multi-horizon training objective and its closed-loop training effects',
        full_minus_b1_mtp='B3 selective update plus search protection jointly; their individual effects are not identified')
    report['limitations'].extend(['四组均单seed42；late3不是独立重复。',
        'Full差值联合包含B3与搜索保护及其闭环训练影响，不能独立归因给其中一个模块。',
        '所有四组均启用FA；这四组不能证明FA相对math的加速或精度等价。',
        '并发运行墙钟不能证明提速；train wall排除逐轮验证但仍包含I/O等宿主开销。'])
    return report


def reliability_plot_data(report):
    """保留每轮固定分箱；空箱必须为 None，不跨轮合并相关样本。"""
    require(report.get('status') in ('passed', 'failed'), 'reliability plot requires valid comparison evidence')
    full = report.get('mechanism_diagnostics', {}).get('full', {})
    epochs = {}
    for epoch in original.EPOCHS:
        summary = full.get(str(epoch), {}).get('trusted_reliability')
        require(isinstance(summary, dict), f'epoch {epoch}: missing trusted reliability')
        bins = summary.get('bins')
        require(isinstance(bins, list) and len(bins) == 10, f'epoch {epoch}: ten fixed bins required')
        total = 0
        for index, item in enumerate(bins):
            require(item.get('lower') == index / 10 and item.get('upper') == (index + 1) / 10,
                    f'epoch {epoch}: bin edges changed')
            count = item.get('count')
            require(type(count) is int and count >= 0, f'epoch {epoch}: invalid bin count')
            total += count
            if not count:
                require(item.get('mean_probability') is None and item.get('event_frequency') is None,
                        f'epoch {epoch}: empty bins must remain missing')
            else:
                for field in ('mean_probability', 'event_frequency'):
                    require(0 <= base.number(item.get(field), field) <= 1,
                            f'epoch {epoch}: invalid {field}')
        require(total == summary.get('count') and total == summary.get('positive', -1) + summary.get('negative', -1),
                f'epoch {epoch}: reliability counts disagree')
        if total:
            require(0 <= base.number(summary.get('brier'), 'brier') <= 1, f'epoch {epoch}: invalid Brier')
        else:
            require(summary.get('brier') is None, f'epoch {epoch}: empty Brier must remain missing')
        epochs[str(epoch)] = summary
    return dict(schema=RELIABILITY_PLOT_SCHEMA, epochs=epochs,
        report_schema=report.get('schema'), report_without_plot_sha256=original.digest(report),
        source_sha256=report.get('source_sha256'), synthetic_fixture=bool(report.get('synthetic_fixture', False)),
        transformations=['Full standard only; one panel per epoch 58/59/60',
            'fixed ten bins; scatter occupied bins at their mean T and empirical target frequency',
            'empty bins remain None and are not plotted; no connecting lines or interpolation',
            'labels show frame count per occupied bin; no pooling or independent-replicate intervals'],
        interpretation='Descriptive reliability only; correlated frames and epochs are not independent repetitions.')


def write_reliability_plot(destination, report):
    """导出独立图片与分箱 JSON；只允许新建 artifacts/ct_checks 产物。"""
    destination = Path(destination).resolve()
    allowed = (ROOT / 'artifacts/ct_checks').resolve()
    require(allowed in destination.parents, 'reliability output must be under artifacts/ct_checks')
    require(destination.suffix.lower() in ('.png', '.pdf', '.svg'), 'reliability output requires PNG/PDF/SVG')
    data_path = destination.with_suffix('.data.json')
    require(not destination.exists() and not data_path.exists(), 'reliability output/data already exists')
    data = reliability_plot_data(report)
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    with matplotlib.rc_context({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False}):
        figure = Figure(figsize=(10.5, 4.8), dpi=180)
        FigureCanvasAgg(figure)
        axes = figure.subplots(1, 3, sharex=True, sharey=True)
        figure.subplots_adjust(left=.075, right=.98, top=.77, bottom=.29, wspace=.25)
        for axis, epoch in zip(axes, original.EPOCHS):
            summary = data['epochs'][str(epoch)]
            occupied = [item for item in summary['bins'] if item['count']]
            axis.plot([0, 1], [0, 1], linestyle='--', linewidth=1, color='.5', label='Identity reference')
            if occupied:
                axis.scatter([item['mean_probability'] for item in occupied],
                    [item['event_frequency'] for item in occupied], marker='o', s=32, color='#0072B2', zorder=3, clip_on=False)
                for item in occupied:
                    axis.annotate(str(item['count']), (item['mean_probability'], item['event_frequency']),
                        xytext=(0, 7 if item['event_frequency'] < .9 else -13), textcoords='offset points',
                        ha='center', fontsize=8)
            else:
                axis.text(.5, .3, 'No real q0 scores', ha='center', fontsize=9, transform=axis.transAxes)
            brier = 'missing' if summary['brier'] is None else f"{summary['brier']:.4f}"
            axis.set(title=f"Epoch {epoch}: n={summary['count']}, Brier={brier}",
                xlabel='Mean trusted score T', xlim=(0, 1), ylim=(0, 1),
                xticks=[0, .25, .5, .75, 1], yticks=[0, .25, .5, .75, 1])
            axis.grid(alpha=.2)
        axes[0].set_ylabel('Empirical trusted-target frequency')
        axes[0].legend(loc='lower right', fontsize=8, frameon=False)
        title = 'Full standard: trusted q0 reliability'
        if data['synthetic_fixture']:
            title += ' (synthetic fixture; not experiment results)'
        figure.suptitle(title, y=.95, fontsize=13)
        figure.text(.075, .075, 'Target: exact IoU >= 0.5, unique predicted-FG support >= 3, target purity >= 0.5.\n'
            'Numbers label bin frame counts; empty bins are omitted. Correlated frames/epochs; no independence claim.\n'
            'All real q0 scores are included, even when q0 is not selected. Fixed threshold T = 0.5 is not fitted here.', fontsize=9)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as stream:
            figure.savefig(stream, format=destination.suffix[1:].lower(), dpi=180)
    data.update(figure=str(destination), dimensions_inches=[10.5, 4.8], dpi=180,
                matplotlib_version=matplotlib.__version__)
    with data_path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return dict(figure=str(destination), source_data=str(data_path))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ('registered_reference', 'baseline', *LABELS):
        parser.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--reliability-output', type=Path,
        help='可选：在 artifacts/ct_checks 下新建 Full 末三轮可靠性图及 .data.json')
    args = vars(parser.parse_args(argv))
    destination = args.pop('output')
    reliability_destination = args.pop('reliability_output')
    try:
        report = compare_runs(**args)
        if reliability_destination is not None:
            report['reliability_plot'] = write_reliability_plot(reliability_destination, report)
        status = 0 if report['status'] == 'passed' else 1
    except (EvidenceError, ValueError, OSError, KeyError, TypeError, ImportError) as error:
        report, status = dict(schema=REPORT_SCHEMA, status='invalid', error=str(error)), 2
    if destination is not None:
        base.write_report(destination, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return status


if __name__ == '__main__':
    raise SystemExit(main())
