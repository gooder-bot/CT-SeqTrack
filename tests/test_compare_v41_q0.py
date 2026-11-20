"""使用生产台账的字段与目录布局，不需数据集、模型或 CUDA。"""
import json
import math

import pytest

from models.ct_v41.config import config_identity, normalize_config, recipe_identity
from models.ct_v41.entry import write_resume_event, write_run_metadata
from tools.compare_v41_q0 import EPOCHS, arm_metrics, compare, frame_stream, retention


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def make_run(root, mode, **overrides):
    config = normalize_config(dict(v41_q0_center_weighting=mode, tag=root.name, log_dir=str(root),
                                   eval_split='dev', full_diagnostic_epochs=list(EPOCHS),
                                   report_checkpoint='last', **overrides))
    actual_fields = ('epoch', 'batch_size', 'workers', 'seed', 'lr', 'wd', 'lr_schedule', 'lr_milestones',
                     'lr_decay_step', 'lr_decay_rate', 'lr_warmup_steps', 'v41_arm', 'v41_q0_center_weighting',
                     'dataset', 'category_name')
    manifest = dict(run_id=root.name, model='ctseqtrackv41', run_type='formal', arm='full',
                    config_sha256=config_identity(config), initial_parameter_hash='initial', source={'sha256': 'source'},
                    input_metadata={'content_sha256': 'inputs'}, dataset_manifest={'content_sha256': 'dataset'},
                    actual_recipe={k: config[k] for k in actual_fields}, training_execution_recipe=recipe_identity(config),
                    evaluation_protocol=dict(split='dev', every_n_epochs=1, always_final=True, selection='last',
                        selection_metric='precision', tie_break='earliest', last_k=3, checkpoint_every_n_epochs=1),
                    evaluation_rng_policy='isolated_split_seed_v2', torch='2.0.1+cu118', lightning='2.0.2',
                    precision_policy='FP32 except BF16 QKV inside FA2 kernel',
                    flash_attention=dict(backend='fa2_varlen', flash_attn='2.5.8', device=0,
                        device_name='NVIDIA A40', kernel_dtype='torch.bfloat16', projection_dtype='torch.float32'))
    root.mkdir()
    write_run_metadata(config, root, manifest, preserve_existing=False)
    for epoch in range(1, 61):
        sampler = dict(plan_sha256=str(epoch), rows=12, batches=3)
        aux = dict(complete=True, sampler=dict(plan_sha256=str(epoch), rows=2, batches=1),
                   insertion_plan_sha256='aux', rows=2, batches=1, main_batches=3)
        write_json(root / 'training_audits' / f'epoch={epoch:03d}.json',
                   dict(run_id=root.name, config_sha256=manifest['config_sha256'], completed_epoch=epoch, epoch_complete=True,
                        rows=12, optimizer_steps=3, sampler=sampler, auxiliary=aux))
    write_json(root / 'training_budget.json', dict(completed_epoch=60, epoch_complete=True, auxiliary=aux,
               last_epoch_rows=12, last_epoch_steps=3, optimizer_steps=180, sampler=sampler))
    for epoch in EPOCHS:
        path = root / 'frame_streams' / 'dev' / ('session-' + root.name) / f'epoch={epoch:03d}.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        predictions = []
        for i in range(100):
            row = dict(scene_id='0018', track_id=11, raw_frame_id=80+i, frame=i, initialization=i == 0,
                       distance=0., iou=1., success=1., precision=1.)
            if i:
                row.update(q0_is_fallback=i == (98 if mode == 'uniform' else 99),
                           diagnostic_sampled_target_count=5 if i == 2 else 20,
                           diagnostic_target_box_world=[0., 0., 0., math.pi / 2],
                           diagnostic_fine_box_world=[.1, .2, .03 if mode == 'uniform' else .01, 1.])
                row['diagnostic_accepted_box_world'] = row['diagnostic_target_box_world']
            predictions.append(row)
        path.write_text('\n'.join(map(json.dumps, reversed(predictions))), encoding='utf-8')
        summary = dict(epoch=epoch, checkpoint_epoch=epoch, split='dev', complete_coverage=True,
                       diagnostics_level='full', total_frames=100, initialization_frames=1,
                       prediction_frames=99, success=100., precision=100.,
                       frame_file='/old/server/' + path.relative_to(root).as_posix())
        write_json(root / 'epochs' / f'epoch={epoch:03d}.json', dict(epoch=epoch, evaluation={'dev': summary}))
    return root


def test_compare_uses_paired_real_q0_gt_axes_and_all_frames(tmp_path):
    a = make_run(tmp_path / 'A', 'uniform')
    b = make_run(tmp_path / 'B', 'inverse_lwh')
    result = compare(a, b)
    arm = result['late3']['A']
    assert arm['eligible_frames'] == 97 and arm['common_visible_frames'] == 96
    assert arm['q0_mae_cm'] == pytest.approx(dict(longitudinal=20, lateral=10, vertical=3))
    assert result['late3']['B']['q0_mae_cm']['vertical'] == pytest.approx(1)
    assert arm['long_tracks']['0018/11']['frames'] == 100
    assert arm['overall'] == {'success': 100., 'precision': 100.}
    assert result['budget'] == dict(main_endpoints=720, auxiliary_endpoints=120, optimizer_steps=180)
    assert result['final60'] == result['epochs']['60']
    assert not result['retention']['warrants_independent_seed_check']
    # 一侧逐帧文件仍报告 complete 时，配对工具独立核对分母。
    path = a / 'frame_streams/dev/session-A/epoch=060.jsonl'
    original = path.read_text(encoding='utf-8').splitlines()
    for lines, message in ((original[:-1], 'incomplete track'), (original + original[:1], 'duplicate')):
        path.write_text('\n'.join(lines), encoding='utf-8')
        with pytest.raises(ValueError, match=message):
            frame_stream(a, 60)


def test_episode_hysteresis_and_long_track_cutoff():
    rows = {}
    for track, length in ((5, 100), (6, 99)):
        for i in range(length):
            row = dict(initialization=i == 0, q0_is_fallback=False, diagnostic_sampled_target_count=10,
                       diagnostic_fine_box_world=[0, 0, .1, 0], diagnostic_target_box_world=[0, 0, 0, 0],
                       success=1., precision=1., distance=0., iou=1.)
            rows['0018', track, i] = row
    for i, iou in enumerate((.09, .2, .08, .5, .09), 1):
        rows['0018', 5, i].update(iou=iou, distance=12. if i == 1 else 0.)
    result = arm_metrics(rows, [('0018', 5, 1)])
    assert set(result['long_tracks']) == {'0018/5'}
    assert result['long_track_summary']['loss_events'] == 2
    assert result['long_track_summary']['over10m_frames'] == 1


def test_empty_intersection_preserves_scores_and_cannot_pass_retention(tmp_path):
    a = make_run(tmp_path / 'A', 'uniform')
    b = make_run(tmp_path / 'B', 'inverse_lwh')
    path = b / 'frame_streams/dev/session-B/epoch=058.jsonl'
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    for row in rows:
        if not row['initialization']:
            row['diagnostic_sampled_target_count'] = 0
    path.write_text('\n'.join(map(json.dumps, rows)), encoding='utf-8')
    report = compare(a, b)
    assert report['epochs']['58']['A']['common_visible_frames'] == 0
    assert report['late3']['B']['overall']['precision'] == 100.
    assert report['late3']['B_minus_A']['q0_mae_cm']['lateral'] is None
    assert report['retention']['checks']['lateral_mae_lower'] is None
    assert not report['retention']['warrants_independent_seed_check']


def test_rejects_summary_and_mismatched_source(tmp_path):
    a = make_run(tmp_path / 'A', 'uniform')
    b = make_run(tmp_path / 'B', 'inverse_lwh')
    path = b / 'epochs/epoch=059.json'
    report = json.loads(path.read_text(encoding='utf-8'))
    report['evaluation']['dev']['diagnostics_level'] = 'summary'
    write_json(path, report)
    with pytest.raises(ValueError, match='full, complete'):
        compare(a, b)
    path = b / 'run_manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    manifest['source']['sha256'] = 'other-source'
    write_json(path, manifest)
    with pytest.raises(ValueError, match='source'):
        compare(a, b)


def test_two_matching_but_off_recipe_runs_are_rejected(tmp_path):
    a = make_run(tmp_path / 'A', 'uniform', seed=7)
    b = make_run(tmp_path / 'B', 'inverse_lwh', seed=7)
    with pytest.raises(ValueError, match='reference Full'):
        compare(a, b)


def test_q0_accepts_each_historical_revision_but_rejects_mixed_versions(tmp_path):
    import yaml
    a = make_run(tmp_path / 'A', 'uniform')
    b = make_run(tmp_path / 'B', 'inverse_lwh')
    def set_revision(root, revision):
        path = root / 'run_manifest.json'
        record = json.loads(path.read_text(encoding='utf-8'))
        record['training_execution_recipe']['execution_revision'] = revision
        write_json(path, record)
        path = root / 'resolved_config.yaml'
        values = yaml.safe_load(path.read_text(encoding='utf-8'))
        values['v41_execution_revision'] = revision
        path.write_text(yaml.safe_dump(values), encoding='utf-8')
    for root in (a, b):
        set_revision(root, 'v3')
    assert compare(a, b)['execution_revision'] == 'v3'
    set_revision(b, 'v4')
    with pytest.raises(ValueError, match='execution revision'):
        compare(a, b)
    set_revision(a, 'v4')
    assert compare(a, b)['execution_revision'] == 'v4'


def test_numeric_environment_checks_kernels_and_allows_different_device_ordinals(tmp_path):
    a = make_run(tmp_path / 'A', 'uniform')
    b = make_run(tmp_path / 'B', 'inverse_lwh', devices=1, accelerator='gpu',
                 console_mode='plain', experiment_name='q0_inverse_lwh')
    path = b / 'run_manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    manifest['flash_attention']['device'] = 1
    write_json(path, manifest)
    assert compare(a, b)['numeric_environment']['status'] == 'matched'
    manifest['flash_attention']['flash_attn'] = 'different-version'
    write_json(path, manifest)
    report = compare(a, b)
    assert report['numeric_environment']['differing_fields'] == ['flash_attn']
    assert report['retention']['numeric_environment_status'] == 'mismatch'
    assert report['late3']['B']['overall']['success'] == 100.


def test_changed_resume_environment_is_detected_even_when_original_manifests_match(tmp_path):
    a = make_run(tmp_path / 'A', 'uniform')
    b = make_run(tmp_path / 'B', 'inverse_lwh')
    manifest = json.loads((b / 'run_manifest.json').read_text(encoding='utf-8'))
    resume_config = normalize_config(dict(v41_q0_center_weighting='inverse_lwh', eval_split='dev',
        full_diagnostic_epochs=list(EPOCHS), report_checkpoint='last', checkpoint=str(b / 'epoch=030.ckpt')))
    manifest['flash_attention']['device'] = 2
    path = write_resume_event(resume_config, b, manifest)
    event = json.loads(path.read_text(encoding='utf-8'))
    assert compare(a, b)['numeric_environment']['status'] == 'matched'
    event['torch'] += '-changed'
    write_json(path, event)
    report = compare(a, b)
    assert report['numeric_environment']['differing_fields'] == ['torch']
    assert report['retention']['numeric_environment_status'] == 'mismatch'
    del event['torch']
    write_json(path, event)
    report = compare(a, b)
    assert report['numeric_environment']['missing_fields']['B/' + path.name] == ['torch']
    assert report['retention']['numeric_environment_status'] == 'unknown'
    assert len(report['retention']['checks']) == 6
    assert report['final60']['A']['common_visible_frames'] > 0


def test_retention_has_three_strict_and_three_nonstrict_conditions():
    def sample(width, height, success):
        return dict(q0_mae_cm=dict(lateral=width, vertical=height), overall=dict(success=success, precision=80),
                    long_track_summary=dict(success=70, over10m_frames=0))
    a = sample(4., 3., 70.)
    b = sample(3., 2., 71.)
    assert retention(dict(A=a, B=b))['warrants_independent_seed_check']
    for status in ('mismatch', 'unknown'):
        result = retention(dict(A=a, B=b), status)
        assert all(result['checks'].values()) and not result['warrants_independent_seed_check']
    for section, field in (('q0_mae_cm', 'lateral'), ('q0_mae_cm', 'vertical'), ('overall', 'success')):
        b = sample(3., 2., 71.)
        b[section][field] = a[section][field]
        assert not retention(dict(A=a, B=b))['warrants_independent_seed_check']
