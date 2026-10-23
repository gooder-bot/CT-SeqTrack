"""固定 R/C 的合成运行证据，不加载历史模型或旧比较脚本。"""
from copy import deepcopy
import json
import yaml
from tools import _tracking_audit as compare
from utils.tracking_metrics import metric_contributions

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def seal_sampler(sampler):
    sampler['manifest_sha256'] = compare.digest({k: v for k, v in sampler.items() if k != 'manifest_sha256'})
    return sampler


def make_rows(label, *, dynamic_iou=.8, static_iou=.8, dynamic_distance=.2, static_distance=.2):
    rows = []
    for track in ('dynamic', 'quiet'):
        for frame in range(5):
            initial = frame == 0
            iou = 1. if initial else (dynamic_iou if track == 'dynamic' else static_iou)
            distance = 0. if initial else (dynamic_distance if track == 'dynamic' else static_distance)
            s, p = metric_contributions(iou, distance)
            row = dict(tracklet=track, frame=frame, initialization=initial, iou=iou, distance=distance,
                success=float(s), precision=float(p), scene_id='test_scene', timestamp=frame * .5)
            if not initial:
                raw = (0, 2, 5, 12)[frame - 1]
                row.update(diagnostic_target_count=raw, diagnostic_gt_xy_displacement=.2 if track == 'dynamic' and frame == 2 else .01)
                if label != 'old_b0':
                    geometry = dict(iou=iou, center_error_m=distance, center_xy_error_m=distance,
                                    yaw_error_rad=0., axis_yaw_error_rad=0.)
                    row.update(diagnostic_crop_target_count=raw, diagnostic_sampled_target_count=raw,
                        diagnostic_crop_point_count=raw, diagnostic_sampled_point_count=raw,
                        diagnostic_empty_crop=raw == 0, diagnostic_background_only_crop=False,
                        coarse_valid=True, coarse_geometry=geometry, fine_geometry=deepcopy(geometry),
                        selected_index=0, selected_quality=.8, strong=raw >= 3, supported=raw >= 3,
                        memory_write=raw >= 3, memory_wrong_write=False)
                if label in ('S', 'W'):
                    row.update(diagnostic_history_support=[[.2, .3, .4]] * 3,
                               diagnostic_current_support=[.1, .2, .3], diagnostic_query_context_norm=.75)
            rows.append(row)
    if label in ('reference', 'old_b0'):
        for row in rows:
            # 历史版本确实可能没有这些字段，不伪装成已记录的零值。
            row.pop('diagnostic_gt_xy_displacement', None)
            if label == 'old_b0':
                row.pop('timestamp', None)
    return rows


def replace_evaluation(root, label, **kwargs):
    manifest = read(root / 'run_manifest.json')
    rows = make_rows(label, **kwargs)
    for epoch in compare.EPOCHS:
        directory = root / 'evaluation' / f'epoch={epoch:03d}'
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'frames.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
        metrics = dict(schema=manifest['schema'], checkpoint_epoch=epoch, metric_mode='benchmark_compat',
                       complete_coverage=True, **compare.COUNTS, **compare.score(rows))
        write(directory / 'metrics.json', metrics)
    write(root / 'results.json', dict(final=metrics, late3=compare.score(rows), checkpoint_epochs=compare.EPOCHS))


def runs(tmp_path, monkeypatch):
    monkeypatch.setattr(compare, 'COUNTS', dict(tracklets=2, frames=10, prediction_frames=8))
    monkeypatch.setattr(compare, 'EVER_MOVING_COUNTS', (1, 4))
    monkeypatch.setattr(compare, 'FIRST_RAW0_COUNTS', (2, 2))
    roots = {}
    for label in ('reference','baseline'):
        root = tmp_path / label
        root.mkdir()
        roots[label] = root
        version = 34 if label in ('S', 'W') else (32 if label == 'old_b0' else 33)
        schema = f'ct_seqtrack.joint_identity.v{version}'
        model = 'seqtrack_reference' if label == 'reference' else f'ctseqtrackv{version}'
        config = dict(compare.COMMON_CONFIG, net_model=model, seed=42,
            experiment_family=f'ct_seqtrack_v{version}', experiment_name=label,
            v31_short_window=4 if label == 'W' else 3, ct_engineering_check=False,
            test=False, init_checkpoint=None, v31_evaluate_late3=True, lr_warmup_steps=0)
        if label in ('baseline', 'S', 'W'):
            config.update(lr=5e-5, lr_schedule='multistep', lr_milestones=[20, 50])
        manifest = dict(schema=schema, model=model, seed=42, arm='b0',
            enabled=dict(B1=False, B2=False, B3=False), evaluation_rng_policy='per_checkpoint_seed_v1',
            source=dict(files={'synthetic.py': 'b' * 64}))
        manifest['source']['sha256'] = compare.digest(manifest['source']['files'])
        if label == 'reference':
            manifest['reference_protocol'] = dict(schema='synthetic-reference')
        manifest['config_sha256'] = compare.config_digest(config, manifest)
        write(root / 'run_manifest.json', manifest)
        (root / 'resolved_config.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
        for epoch in range(1, 61):
            sampler = dict(schema='seqtrack_reference.v32.teacher.v1' if label == 'reference'
                           else 'ct_seqtrack.v32.ready_queue.v2', epoch=epoch - 1, seed=42,
                training=True, rows=19108, batch_size=16, source_sha256='a' * 64)
            if label != 'reference':
                sampler.update(short_window=config['v31_short_window'], long_window=8, curriculum_epochs=10,
                    reserve_windows=112, seed_translation=.3, seed_yaw_degrees=1.5, batches=1195,
                    reserve_policy='stratified_singleton_windows_then_fill_v1',
                    drain_policy='running_b0_bn_from_first_reserve_or_partial_v1',
                    seed_policy='shared_anchor_local_translation_world_yaw_v1', plan_sha256='c' * 64)
            seal_sampler(sampler)
            audit = dict(completed_epoch=epoch, epoch_complete=True, rows=19108, optimizer_steps=1195, sampler=sampler)
            if label in ('S', 'W'):
                audit.update(schema='ct_seqtrack.v34.training_audit.v1', model_schema=schema,
                             config_sha256=manifest['config_sha256'])
            write(root / 'training_audits' / f'epoch={epoch:03d}.json', audit)
        write(root / 'training_budget.json', dict(schema=schema, epoch_complete=True, **compare.BUDGET, sampler=sampler))
        values = dict(dynamic_iou=.6, static_iou=.6, dynamic_distance=.5, static_distance=.5)
        if label == 'reference':
            values.update(dynamic_iou=.65, static_iou=.65, dynamic_distance=.45, static_distance=.45)
        if label in ('S', 'W'):
            values.update(dynamic_iou=.8, static_iou=.8, dynamic_distance=.2, static_distance=.2)
        replace_evaluation(root, label, **values)
    return roots

