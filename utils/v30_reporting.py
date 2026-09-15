"""v30 正式端点身份与条件漏斗；未知计数不当作零。"""
import json
from collections import defaultdict

from utils.action_calibration import sha256_file, sha256_json


def _get(config, key, default=None):
    return config.get(key, default) if isinstance(config, dict) else getattr(config, key, default)


def endpoint_identity(config):
    variant = _get(config, 'ct_variant')
    backend = _get(config, 'motion_v3_temporal_backend')
    arm = 'b0' if variant == 'b0' else 'full_' + str(backend) if variant == 'full' else str(variant)
    return dict(dataset=_get(config, 'dataset'), dataset_version=_get(config, 'version'),
        seed=_get(config, 'seed'), checkpoint_epoch=_get(config, 'ct_source_checkpoint_epoch'),
        arm=arm, ablation=_get(config, 'ct_v30_ablation', 'none'),
        category=_get(config, 'category_name'), coordinate_mode=_get(config, 'ct_coordinate_mode'),
        frame_stride=_get(config, 'ct_frame_stride'),
        dataset_manifest_sha256=_get(config, 'ct_dataset_manifest_sha256',
                                     _get(config, 'ct_scene_manifest_sha256')))


def evaluation_population(loader):
    """根据已建数据接口列出完整测试身份，不加载帧或点云。"""
    if isinstance(loader, (list, tuple)):
        if len(loader) != 1:
            raise ValueError('v30 expects exactly one test dataloader')
        loader = loader[0]
    dataset = loader.dataset
    source = dataset.dataset
    if hasattr(dataset, 'tracklet_indices') or hasattr(dataset, 'indices'):
        raise ValueError('v30 official evaluation cannot use a tracklet subset wrapper')
    manifest = source.ct_scene_manifest
    from utils.dataset_protocol_v30 import validate_dataset_manifest
    validate_dataset_manifest(manifest)
    endpoints = []
    for index in range(len(dataset)):
        meta = source.virtual_rate_meta[index]
        scene = meta.get('scene_id')
        if scene is None:
            scene = source.nusc.get('scene', meta['scene_token'])['name']
        key = source.get_tracklet_key(index)
        endpoints.extend((str(scene), str(key), frame) for frame in range(source.get_num_frames_tracklet(index)))
    return dict(frames=len(endpoints), tracklets=len(dataset),
        endpoint_population_sha256=sha256_json(sorted(endpoints)),
        scenes=sorted({x[0] for x in endpoints}), dataset_manifest_sha256=manifest['content_sha256'])


def _stage_summary(rows):
    stages = {}
    previous_key = None
    for name, key in (
        ('global_novel', 'acquisition_global_novel_target_count'),
        ('max_reachable', 'acquisition_max_reachable_target_count'),
        ('support_novel', 'acquisition_support_novel_target_count'),
        ('prepool768', 'acquisition_prepool_target_count'),
        ('selected256', 'acquisition_selected_target_count')):
        measured = [r for r in rows if r.get(key) is not None]
        condition = ([r for r in measured if r.get(previous_key) is not None and r[previous_key] > 0]
                     if previous_key else measured)
        target_events = sum(r[key] > 0 for r in measured)
        retained = sum(r[key] > 0 for r in condition)
        point_key = key.replace('_target_count', '_point_count')
        point_rows = [r for r in rows if r.get(point_key) is not None]
        stages[name] = dict(measured_frames=len(measured),
            unique_point_measured_frames=len(point_rows),
            unique_points=sum(r[point_key] for r in point_rows) if point_rows else None,
            target_points=sum(r[key] for r in measured) if measured else None,
            target_events=target_events if measured else None,
            event_rate=target_events / len(measured) if measured else None,
            conditional_denominator=len(condition), conditional_retained=retained,
            conditional_retention=retained / len(condition) if condition else None)
        previous_key = key
    mode_rows = [r for r in rows if r.get('mode_count') is not None]
    target_rows = [r for r in mode_rows if r.get('acquisition_selected_target_count', 0) > 0]
    correct = [r for r in mode_rows if r.get('correct_mode_event', False)]
    utility_rows = [r for r in rows if 'beneficial_action_available' in r]
    metrics = {name: (100 * sum(float(r[key]) for r in rows) / len(rows) if rows else None)
               for name, key in (('S', 'final_success'), ('P', 'final_precision'))}
    return dict(frames=len(rows), stages=stages, final_metrics=metrics,
        final_metric_scope='prediction_frames_only_in_this_recursive_rollout',
        correct_mode_events=len(correct), mode_measured_frames=len(mode_rows),
        correct_mode_conditional_denominator=len(target_rows),
        correct_mode_conditional_retained=sum(bool(r.get('correct_mode_event')) for r in target_rows),
        correct_mode_utilization_denominator=len(correct),
        selected_correct_mode=sum(bool(r.get('selected_correct_mode')) for r in correct),
        utility_measured_frames=len(utility_rows),
        beneficial_action_events=sum(bool(r['beneficial_action_available']) for r in utility_rows)
            if utility_rows else None,
        immediate_action_gain_sum=sum(r.get('selected_action_immediate_gain', 0.) for r in utility_rows)
            if utility_rows else None)


def add_v30_report(summary, rows, *, config=None):
    identity = {}
    defaults = endpoint_identity(config) if config is not None else {}
    for key in endpoint_identity({}):
        values = {json.dumps(r[key], sort_keys=True) for r in rows if r.get(key) is not None}
        if len(values) > 1:
            raise ValueError('v30 mixed endpoint identity: ' + key)
        identity[key] = json.loads(next(iter(values))) if values else defaults.get(key)
        if values and defaults.get(key) is not None and identity[key] != defaults[key]:
            raise ValueError('v30 endpoint/config identity mismatch: ' + key)
    population = sorted((str(r['scene_id']), str(r.get('tracklet_key', r['tracklet_id'])), int(r['frame_id'])) for r in rows)
    identity.update(endpoint_population_sha256=sha256_json(population),
        scenes=sorted({r['scene_id'] for r in rows}),
        roles=sorted({r.get('partition', r.get('role', 'unknown')) for r in rows}),
        frames=len(rows), tracklets=len({x[1] for x in population}),
        all_endpoint_metadata_valid=all(r.get('metadata_status') == 'ok' for r in rows))
    expected = _get(config, 'ct_evaluation_population')
    identity['population_complete'] = bool(expected and all(identity.get(k) == v for k, v in expected.items()))
    identity['formal_config_validated'] = False
    identity['official_checkpoint_evaluation'] = bool(_get(config, 'test', False)) and identity['roles'] == ['test']
    identity['checkpoint_sha256'] = _get(config, 'ct_source_checkpoint_sha256')
    checkpoint = _get(config, 'checkpoint')
    if not identity['checkpoint_sha256'] and checkpoint:
        identity['checkpoint_sha256'] = sha256_file(checkpoint)
    if config is not None and _get(config, 'ct_enable_v30', False):
        from utils.v30_contracts import V30_CONTRACTS, validate_v30_scratch_contract
        from utils.action_calibration_v30 import code_content_sha256
        validate_v30_scratch_contract(config)
        identity['formal_config_validated'] = not bool(_get(config, 'ct_engineering_check', False))
        common_keys = (*V30_CONTRACTS, 'dataset', 'version', 'category_name', 'seed', 'epoch',
            'batch_size', 'hist_num', 'point_sample_size', 'ct_frame_stride', 'ct_coordinate_mode',
            'ct_runtime_optimization', 'ct_diagnostic_policy', 'ct_b0_candidate_weights',
            'ct_v30_ablation', 'ct_b0_long_rollin_enabled', 'ct_acquisition_margin_min',
            'ct_acquisition_margin_max', 'ct_mode_count', 'ct_mode_quality_weight')
        identity['comparison_protocol_sha256'] = sha256_json({k: _get(config, k) for k in common_keys})
        identity['code_content_sha256'] = code_content_sha256()
    policy_loaded = []
    for row in rows:
        status = row.get('calibration_status', {})
        if isinstance(status, str):
            try:
                status = json.loads(status)
            except (ValueError, TypeError):
                status = {}
        policy_loaded.append(isinstance(status, dict) and status.get('loaded') is True)
    identity['calibrated_policy_loaded'] = bool(policy_loaded) and all(policy_loaded)
    summary['identity'] = identity
    predicted = [r for r in rows if not r['is_initial']]
    groups = {'all': predicted}
    for key in ('sparsity_group', 'speed_group', 'recursive_age_group'):
        subsets = defaultdict(list)
        for row in predicted:
            subsets[str(row.get(key, 'unknown'))].append(row)
        groups.update({key + '/' + name: subset for name, subset in subsets.items()})
    for name, predicate in (('0', lambda n: n == 0), ('1_2', lambda n: 0 < n <= 2),
                            ('3_7', lambda n: 2 < n <= 7), ('8_plus', lambda n: n >= 8)):
        groups['lost_length/' + name] = [r for r in predicted if predicate(r.get('lost_length', 0))]
    summary['v30_funnel'] = {name: _stage_summary(subset) for name, subset in groups.items()}
    summary['v30_funnel_interpretation'] = 'GT is supervision/diagnostics only and never an inference feature; missing stage counts remain unknown; initialization excluded; conditional event retention is distinct from foreground point retention'
