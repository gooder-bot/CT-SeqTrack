import copy
import json
from types import SimpleNamespace

import pytest

from tests.test_ct_v27_actions import _rows
from utils.config import load_yaml_config
from utils.v27_eval_reporting import summarize_endpoint_diagnostics
from utils.v30_reporting import endpoint_identity, evaluation_population
from tools.summarize_ct_v30_mini import summarize_mini


def _report(arm='b0', epoch=60, value=.5):
    from models.ct_variant import configure_ct_variant
    from utils.action_calibration import sha256_json
    config = load_yaml_config('cfgs/ct_seqtrack/30_' + arm + '_mini.yaml')
    configure_ct_variant(config)
    config.update(test=True, ct_source_checkpoint_epoch=epoch,
                  ct_source_checkpoint_sha256=str(epoch).zfill(64), ct_dataset_manifest_sha256='a' * 64)
    rows = []
    for scene in ('scene-0103', 'scene-0916'):
        for row in _rows(scene):
            row.update(endpoint_identity(config), protocol_version='v30', final_iou=value,
                metadata_status='ok', partition='test', tracklet_key=row['tracklet_id'],
                calibration_status=json.dumps({'loaded': True}),
                b0_raw_point_count=2, mode_count=0, correct_mode_event=False)
            if not row['is_initial']:
                row.update(final_success=value, final_precision=value)
            rows.append(row)
    config['ct_evaluation_population'] = dict(frames=6, tracklets=2, scenes=['scene-0103', 'scene-0916'],
        endpoint_population_sha256=sha256_json(sorted((r['scene_id'], r['tracklet_key'], r['frame_id']) for r in rows)),
        dataset_manifest_sha256='a' * 64)
    return summarize_endpoint_diagnostics(rows, config=config)


def _reports():
    return {arm: [_report(arm, epoch, .6 if arm == 'full_cfc' and epoch == 60 else .4 if arm != 'b0' else .5)
                  for epoch in (58, 59, 60)] for arm in ('b0', 'full_cfc', 'full_gru')}


def test_v30_summary_has_identity_and_unknown_reachable_denominators():
    report = _report()
    assert report['schema'] == 'ct_seqtrack.endpoint_diagnostics.v30'
    assert report['identity']['population_complete']
    assert report['metrics']['frames'] == 6
    stage = report['v30_funnel']['all']['stages']['max_reachable']
    assert stage['target_points'] is None and stage['conditional_retention'] is None


def test_mini_gate_is_final60_only_and_all_three_arms_may_progress():
    summary = summarize_mini(_reports())
    assert summary['promotion']['passed']
    assert summary['promotion']['qualifying_arms'] == ['full_cfc']
    assert summary['arms']['full_cfc']['late3_delta_vs_b0']['S'] < 0
    assert summary['promotion']['late3_role'] == 'report_only'
    assert summary['promotion']['allowed_next_arms'] == ['b0', 'full_cfc', 'full_gru']


@pytest.mark.parametrize('key,value', [('checkpoint_epoch', 57), ('seed', 52), ('arm', 'full_gru'),
    ('population_complete', False), ('endpoint_population_sha256', 'f' * 64),
    ('comparison_protocol_sha256', 'f' * 64), ('ablation', 'single_mode'),
    ('official_checkpoint_evaluation', False), ('calibrated_policy_loaded', False)])
def test_mini_gate_rejects_wrong_identity_or_partial_evaluation(key, value):
    reports = _reports()
    reports['full_cfc'][2]['identity'][key] = value
    with pytest.raises(ValueError):
        summarize_mini(reports)


def test_gate_requires_both_metrics_and_rejects_bare_scores():
    reports = _reports()
    reports['full_cfc'][2]['metrics']['P'] = reports['b0'][2]['metrics']['P']
    assert not summarize_mini(reports)['promotion']['passed']
    reports['b0'][0] = {'S': 45., 'P': 50.}
    with pytest.raises(ValueError, match='official v30'):
        summarize_mini(reports)


def test_kitti_generic_export_and_metadata_only_expected_population():
    from tests.test_ct_v28_export_metadata import production_functions
    from utils.dataset_protocol_v30 import build_dataset_manifest
    manifest = build_dataset_manifest(dict(dataset='kitti_mf', version='kitti_tracking'))
    source = SimpleNamespace(ct_scene_manifest=manifest, ct_scene_role='test',
        virtual_rate_meta=[dict(scene_id='0019', tracklet_key='kitti/0019/car')],
        get_tracklet_key=lambda i: 'kitti/0019/car', get_num_frames_tracklet=lambda i: 3)
    class Dataset:
        dataset = source
        def __len__(self):
            return 1
    loader = SimpleNamespace(dataset=Dataset())
    metadata = production_functions()['_ct_v28_export_metadata'](loader, 0)
    assert metadata['metadata_status'] == 'ok' and metadata['scene_id'] == '0019'
    population = evaluation_population(loader)
    assert population['frames'] == 3 and population['tracklets'] == 1
    loader.dataset.indices = [0]
    with pytest.raises(ValueError, match='subset'):
        evaluation_population(loader)


def test_v30_batch_payload_uses_production_collate(monkeypatch):
    import tools.ct_v30_batch_runtime as runtime
    from utils.v29_rollin import observation_collate
    items = [dict(mode='teacher', sample={'x': 1}), dict(mode='rollin', index=4)]
    calls = []
    monkeypatch.setattr(runtime, 'prepare_observation_batch', lambda host, rows: calls.append((host, rows)) or {'ok': True})
    host = object()
    assert runtime.prepare_payload(host, observation_collate(items)) == {'ok': True}
    assert calls == [(host, items)]
    with pytest.raises(ValueError, match='production observation collate'):
        runtime.prepare_payload(host, {'points': []})
