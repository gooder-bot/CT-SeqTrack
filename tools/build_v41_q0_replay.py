"""用已完成的 uniform E10/E60 在原 v3 源码下生成 KITTI 训练前态。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

import numpy as np


EPOCHS = (10, 60)
SCHEMA = 'ct_seqtrack.q0_prefix_pairs.v1'
THRESHOLDS = dict(min_target_points=10, coarse_iou=.5, lateral_m=.30, height_m=.15,
                  axis_yaw_deg=10.)
MATCH_LIMITS = dict(distance_absolute_m=5., distance_relative=.25, point_count_ratio=2.)
# 已核对的固定 test 收据，artifacts/ct_checks/20261121_q0_final_review/formal/verification.json。
# E60 保存时磁盘文件已更新；回放代码与启动清单相同。该收据仅用于这个离线供体。
E60_RECEIPT = dict(run_id='80ef91a660224c2f95a5d86eae0ec095', epoch=60,
    checkpoint_sha256='7e9cae97b68fc2ff4830741648a2aa3efa927fb96c4985d81e49a19e35cb8873',
    evaluator_source='73bcf8e6bce57963f3cb74b64f553033d13ea104aa1ba8e25e89bfa746d8b247',
    saved_source='e7e0ee6b4b2ec8488479fc72d77e7e7217e938c04f1f863a06528045b7a38109',
    config_sha256='8c9ba702cb01191141beb09e8939ee493aba0326a431014944807883d3930223')


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def prefix_state(state, frame):
    """必须在 commit 前复制。当前 frame 以及其标注、点云均不在状态内。"""
    if state.next_frame != frame or set(state.boxes) != set(range(max(0, frame - 3), frame)):
        raise ValueError('donor state is not the contiguous prefix preceding this endpoint')

    def plain(value):
        if isinstance(value, dict):
            return {str(key): plain(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [plain(item) for item in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value

    return plain(vars(state))


def difficulty(coarse, target, coarse_iou):
    delta = coarse - target
    lateral = abs(-math.sin(target[3]) * delta[0] + math.cos(target[3]) * delta[1])
    angle = abs(math.degrees(math.remainder(float(delta[3]), math.pi)))
    values = dict(lateral_m=float(lateral), height_m=abs(float(delta[2])), axis_yaw_deg=angle)
    reasons = [key for key, value in values.items() if value > THRESHOLDS[key]]
    if coarse_iou < THRESHOLDS['coarse_iou']:
        reasons.append('coarse_iou')
    return dict(values, coarse_iou=float(coarse_iou)), reasons


def match_events(records):
    """合并两份供体在同一物理轨道上连续命中的帧，每事件保留一组独立配对。"""
    hard = defaultdict(lambda: defaultdict(list))
    normal = defaultdict(list)
    for row in records:
        if row['reasons']:
            hard[row['tracklet_key']][row['frame']].append(row)
        else:
            normal[row['donor_epoch']].append(row)

    # 不把另一份供体上同属困难事件的物理端点当作普通对照。
    controls = defaultdict(list)
    for epoch, rows in normal.items():
        controls[epoch] = [row for row in rows if row['frame'] not in hard.get(row['tracklet_key'], {})]
    used = set()
    pairs = []
    events = 0
    for track, by_frame in sorted(hard.items()):
        segments = []
        for frame, rows in sorted(by_frame.items()):
            if not segments or frame != segments[-1][-1]['frame'] + 1 or (
                    rows[0]['raw_frame'] != segments[-1][-1]['raw_frame'] + 1):
                segments.append([])
            segments[-1].extend(rows)
        for segment in segments:
            events += 1
            # 先取重叠最差的真实几何，其他供体/帧仍可提供实际可配到的正常例。
            ordered = sorted(segment, key=lambda row: (row['errors']['coarse_iou'],
                             -len(row['reasons']), row['frame'], row['donor_epoch']))
            for case in ordered:
                distance_limit = max(MATCH_LIMITS['distance_absolute_m'],
                                     MATCH_LIMITS['distance_relative'] * case['distance_m'])
                point_limit = MATCH_LIMITS['point_count_ratio']
                candidates = (row for row in controls[case['donor_epoch']]
                              if (row['tracklet'], row['frame']) not in used
                              and abs(row['distance_m'] - case['distance_m']) <= distance_limit
                              and case['target_points'] / point_limit <= row['target_points']
                                  <= case['target_points'] * point_limit)

                def match_key(row):
                    mismatch = abs(math.log1p(case['distance_m']) - math.log1p(row['distance_m']))
                    mismatch += abs(math.log1p(case['target_points']) - math.log1p(row['target_points']))
                    return (case['tracklet'] != row['tracklet'], mismatch,
                            abs(row['frame'] - case['frame']), row['tracklet'], row['frame'])

                control = min(candidates, key=match_key, default=None)
                if control is not None:
                    used.add((control['tracklet'], control['frame']))
                    event_id = f"{track}:{segment[0]['raw_frame']}-{segment[-1]['raw_frame']}"
                    pairs.append(dict(event_id=event_id, hard=case, normal=control))
                    break
    return pairs, dict(events=events, matched_events=len(pairs), unmatched_events=events-len(pairs))


def track_role(source, track):
    key = source.get_tracklet_key(track)
    parts = key.split('/')
    if (len(parts) < 5 or parts[:3] != ['kitti_v41', 'kitti_tracking', 'train']
            or parts[3] not in {f'{i:04d}' for i in range(17)}):
        raise ValueError(f'q0 prefix bank must use KITTI train scenes 0000–0016: {key}')


def collect(host, dataset, cfg, epoch):
    from models.ct_v41 import data
    from utils.tracking_metrics import BoxMetricsTarget, LocalYawBox

    builder = data.BatchBuilder(cfg)
    builder.detailed_diagnostics = False
    sampler = data.ReadyQueueBatchSampler(dataset.lengths, cfg.batch_size, seed=cfg.seed, training=False)
    sampler.set_epoch(epoch)
    visible = []
    visited = 0
    for requests in sampler:
        rows = dataset.__getitems__(requests)
        batch, output = host._forward_raw(rows, builder, training=False)
        try:
            coarse = output.observation.coarse_box.detach().cpu().double().numpy()
            for pose, context in zip(coarse, builder._pending):
                row, state = context['row'], context['state']
                req = row['request']
                frame = row['frames'][req.frame]
                truth = data.box_array(frame['3d_bbox'])
                size = data.box_size(frame['3d_bbox'], frame)
                ids = context['sampled_ids']
                xyz = context['sampled_xyz'][np.unique(ids, return_index=True)[1]]
                target_points = int(data.inside_box(xyz, truth, size).sum())
                visited += 1
                if target_points < THRESHOLDS['min_target_points']:
                    continue
                absolute = pose.copy()
                absolute[:3] += context['anchor'][:3]
                metric = BoxMetricsTarget(LocalYawBox(truth, size[[1, 0, 2]]), up_axis=(0, 0, 1))
                _, iou, _ = metric.metrics_3d(LocalYawBox(absolute, state.size[[1, 0, 2]]))
                errors, reasons = difficulty(absolute, truth, iou)
                visible.append(dict(tracklet=req.tracklet, frame=req.frame, tracklet_key=row['tracklet_key'],
                    raw_frame=int(frame['raw_frame_id']), donor_epoch=epoch, target_points=target_points,
                    distance_m=float(np.linalg.norm(truth[:2])), errors=errors, reasons=reasons,
                    state=prefix_state(state, req.frame)))
            builder.commit(output, batch, diagnostics=False)
        finally:
            host._release_materialization(batch, builder)
    if visited != len(dataset) or builder.states:
        raise RuntimeError('incomplete full-track donor collection')
    return visible, visited


def validate_donor(payload, cfg, manifest, epoch, checksum, dataset_sha256):
    from models.ct_v41.runtime import dataset_identity, validate_auxiliary_payload, validate_resume_payload

    if (not isinstance(payload, dict) or payload.get('completed_epoch') != epoch
            or payload.get('run_id') != manifest['run_id'] or payload.get('epoch_complete') is not True
            or payload.get('config_sha256') != manifest['config_sha256']
            or payload['sampler']['source_sha256'] != dataset_sha256
            or manifest['dataset_manifest'] != payload['dataset_identity']['dataset_manifest']
            or manifest['input_metadata'] != payload['dataset_identity']['input_metadata']):
        raise ValueError(f'E{epoch} donor run, epoch, configuration, or dataset identity mismatch')
    if payload['source'] == manifest['source']:
        validate_resume_payload(payload, cfg, training=False)
        return 'native_v3_checkpoint'
    receipt = dict(run_id=payload['run_id'], epoch=epoch, checkpoint_sha256=checksum,
                   evaluator_source=manifest['source']['sha256'], saved_source=payload['source']['sha256'],
                   config_sha256=payload['config_sha256'])
    if receipt != E60_RECEIPT or payload['execution_identity']['source_sha256'] != receipt['saved_source']:
        raise ValueError('different checkpoint/evaluator source without a matching audited E60 receipt')
    if payload['dataset_identity'] != dataset_identity(cfg):
        raise ValueError('archived E60 KITTI data differs from current data')
    validate_auxiliary_payload(payload['auxiliary'], cfg, payload['sampler'])
    return 'audited_uniform_e60_fixed_test_receipt'


def build(args):
    # 导入前绑定至供体的源码树。当前模型代码可能已有新模块和不同执行身份。
    source_path = args.donor_source.resolve(strict=True)
    run = args.donor_run.resolve(strict=True)
    sys.path.insert(0, str(source_path))
    import pytorch_lightning as pl
    import torch
    import yaml
    from datasets.kitti_v41 import get_v41_raw_dataset
    from models.ct_v41 import data, entry
    from models.ct_v41.config import config_identity, normalize_config, recipe_scope
    from models.ct_v41.provenance import source_identity
    from models.ctseqtrackv41 import CTSEQTRACKV41

    manifest_path = run / 'run_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    resolved = yaml.safe_load((run / 'resolved_config.yaml').read_text(encoding='utf-8'))
    if args.path is not None:
        resolved['path'] = str(args.path.resolve(strict=True))
    checkpoints = {epoch: run / 'formal_checkpoints' / f'epoch={epoch:03d}.ckpt' for epoch in EPOCHS}
    cfg = normalize_config(dict(resolved, test=True, log_dir=None, checkpoint=str(checkpoints[EPOCHS[0]])))
    source = source_identity()
    if (Path(data.__file__).resolve().parents[2] != source_path or source != manifest['source']
            or cfg.v41_execution_revision != 'v3' or cfg.v41_q0_center_weighting != 'uniform'
            or cfg.v41_arm != 'full' or cfg.ct_engineering_check or cfg.seed != 42
            or cfg.dataset != 'kitti_mf' or manifest['config_sha256'] != config_identity(cfg)):
        raise ValueError('expected the original formal v3 uniform source, run, and training config')
    entry.configure_numerics()
    with recipe_scope(cfg), torch.no_grad():
        raw = get_v41_raw_dataset(cfg, 'auxiliary')
        if raw.v41_source_role != 'train' or raw.v41_data_role != 'auxiliary':
            raise ValueError('expected complete raw training tracks without GT crops')
        dataset = data.RawEndpointDataset(raw, training=False)
        for track in range(len(dataset.lengths)):
            track_role(raw, track)
        records, donors = [], []
        for epoch in EPOCHS:
            checkpoint_path = checkpoints[epoch]
            cfg.checkpoint = str(checkpoint_path)
            cfg.eval_checkpoint_epoch = epoch
            pl.seed_everything(cfg.seed, workers=True)
            checksum = file_sha256(checkpoint_path)
            saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            payload = saved.get('ct_v41_runtime')
            policy = validate_donor(payload, cfg, manifest, epoch, checksum, dataset.source_sha256)
            saved_source = payload['source']['sha256']
            host = CTSEQTRACKV41(cfg)
            host.load_state_dict(saved['state_dict'], strict=True)
            del saved, payload
            host.cuda().eval()
            rows, count = collect(host, dataset, cfg, epoch)
            records.extend(rows)
            donors.append(dict(epoch=epoch, run_id=manifest['run_id'], source_sha256=source['sha256'],
                checkpoint_sha256=checksum, checkpoint_source_sha256=saved_source, validation=policy,
                config_sha256=manifest['config_sha256'], visited=count, visible=len(rows),
                hard=sum(bool(row['reasons']) for row in rows)))
            del host
            print(f'E{epoch}: {count} train endpoints, {len(rows)} visible, {donors[-1]["hard"]} hard',
                  file=sys.stderr, flush=True)
    pairs, summary = match_events(records)
    return dict(schema=SCHEMA, split='train', source_sha256=dataset.source_sha256,
                run_manifest_sha256=file_sha256(manifest_path), input_metadata_sha256=raw.input_metadata_sha256,
                dataset_manifest_sha256=raw.v41_manifest['content_sha256'], donors=donors,
                thresholds=THRESHOLDS, match_limits=MATCH_LIMITS, iou_mode='benchmark_compat', pairs=pairs,
                summary=dict(summary, paired_by_epoch=dict(Counter(p['hard']['donor_epoch'] for p in pairs))))


def publish(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix='.q0-replay-', delete=False,
                            mode='w', encoding='utf-8') as stream:
        tmp = Path(stream.name)
        try:
            json.dump(payload, stream, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            stream.close()
            tmp.unlink()
            raise
    try:
        os.link(tmp, path)  # 原子发布；并发生成也不能覆盖另一份完整清单。
    finally:
        tmp.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--donor-source', required=True, type=Path, help='与供体清单完全相符的 v3 源码根目录')
    parser.add_argument('--donor-run', required=True, type=Path, help='含已完成 E10/E60 的 uniform 目录')
    parser.add_argument('--path', type=Path, help='同内容的 KITTI training 根目录，缺省沿用供体配方')
    parser.add_argument('--output', required=True, type=Path, help='新 JSON 文件路径')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('--output requires a new file')
    payload = build(args)
    publish(args.output, payload)
    print(json.dumps(dict(manifest=str(args.output), sha256=file_sha256(args.output), **payload['summary'])))


if __name__ == '__main__':
    main()
