"""用冻结的 uniform E57 生成训练历史密度表，配对审计旧扰动、连续残差与逐帧打散。"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
# 只导入新的纯 NumPy 格式；models 包稍后从供体版本载入。
sys.path.insert(0, str(ROOT / 'models/ct_v41'))
from history_augmentation import (DONOR_ARCHIVE, DONOR_GEOMETRY, E57_CHECKPOINT, E57_SOURCE, SCHEMA,
                                  TRAIN_DATA, HistoryAugmentation, density_bins, sensor_xyz, training_scene)
sys.path.pop(0)


def publish(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as directory:
        staged = Path(directory) / 'complete.json'
        with staged.open('w', encoding='utf-8', newline='\n') as stream:
            json.dump(payload, stream, separators=(',', ':'), allow_nan=False)
            stream.write('\n')
        os.link(staged, path)


def donor_windows(path):
    raw = path.read_bytes().replace(b'\r\n', b'\n')
    if hashlib.sha256(raw).hexdigest() != DONOR_ARCHIVE:
        raise ValueError('expected the audited 49-track E57 training frame archive')
    return verified_windows([json.loads(line) for line in raw.splitlines()])


def verified_windows(records):
    # 跨环境重算验证全部 1664 帧，保留微秒、十微米及 1e-5 rad 分辨率。
    geometry = sorted((r['track'], r['index'], round(r['time'] * 1e6), r['sampled_inside_prediction'],
                       *(round(x * 1e5) for x in r['err_lwh']), round(r['gt'][3] * 1e5)) for r in records)
    fingerprint = hashlib.sha256(json.dumps(geometry, separators=(',', ':')).encode()).hexdigest()
    if fingerprint != DONOR_GEOMETRY:
        raise ValueError('replayed E57 inputs differ from the audited donor geometry; supply --donor-frames')
    tracks = defaultdict(list)
    for row in records:
        training_scene(row['track'])
        tracks[row['track']].append(row)
    donors = []
    for track, rows in sorted(tracks.items()):
        rows.sort(key=lambda row: row['index'])
        for offset in range(len(rows) - 3):
            segment = rows[offset:offset + 4]
            if not np.all(np.diff([r['index'] for r in segment]) == 1):
                raise ValueError('nonconsecutive donor archive rows')
            gaps = np.diff([r['time'] for r in segment])
            if not np.allclose(gaps, .1, atol=1e-8, rtol=0):
                continue
            donors.append(dict(track=track, endpoint=segment[-1]['index'],
                               dt_us=np.rint(gaps[:2] * 1e6).astype(int).tolist(),
                               counts=[r['sampled_inside_prediction'] for r in segment[:3]],
                               err_lwh=[r['err_lwh'] for r in segment[:3]]))
    if len(donors) != 1515:
        raise ValueError('incomplete audited E57 donor inventory')
    return donors, tracks


@contextmanager
def frozen_teacher(args):
    source = args.donor_source.resolve(strict=True)
    run = args.donor_run.resolve(strict=True)
    sys.path.insert(0, str(source))
    import pytorch_lightning as pl
    import torch
    import yaml
    from datasets.kitti_v41 import get_v41_raw_dataset
    from models.ct_v41 import data, entry
    from models.ct_v41.config import config_identity, normalize_config, recipe_scope
    from models.ct_v41.provenance import source_identity
    from models.ctseqtrackv41 import CTSEQTRACKV41

    manifest = json.loads((run / 'run_manifest.json').read_text(encoding='utf-8'))
    checkpoint = run / 'formal_checkpoints/epoch=057.ckpt'
    resolved = yaml.safe_load((run / 'resolved_config.yaml').read_text(encoding='utf-8'))
    if args.path:
        resolved['path'] = str(args.path.resolve(strict=True))
    cfg = normalize_config(dict(resolved, test=True, log_dir=None, checkpoint=str(checkpoint)))
    if (Path(data.__file__).resolve().parents[2] != source or source_identity()['sha256'] != E57_SOURCE
            or manifest['source']['sha256'] != E57_SOURCE
            or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != E57_CHECKPOINT
            or cfg.v41_q0_center_weighting != 'uniform' or cfg.v41_execution_revision != 'v3'
            or cfg.seed != 42 or config_identity(cfg) != manifest['config_sha256']):
        raise ValueError('expected the original b819e76 evaluator and audited v3 uniform E57 run')
    entry.configure_numerics()
    pl.seed_everything(42, workers=True)
    with recipe_scope(cfg), torch.no_grad():
        raw = get_v41_raw_dataset(cfg, 'auxiliary')
        ds = data.RawEndpointDataset(raw, training=False, enable_m=False)
        if (raw.v41_source_role != 'train' or raw.v41_data_role != 'auxiliary'
                or ds.source_sha256 != TRAIN_DATA):
            raise ValueError('expected the complete native KITTI training split')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        receipt = saved['ct_v41_runtime']
        if (receipt['run_id'] != manifest['run_id'] or receipt['completed_epoch'] != 57
                or not receipt['epoch_complete'] or not receipt['auxiliary']['complete']
                or receipt['sampler']['source_sha256'] != TRAIN_DATA):
            raise ValueError('E57 donor checkpoint or input data identity mismatch')
        host = CTSEQTRACKV41(cfg)
        host.load_state_dict(saved['state_dict'], strict=True)
        del saved
        host.cuda().eval()
        yield host, ds, data


def collect_history(host, ds, data, *, donor_prefixes=False):
    tracklets = list(range(len(ds.lengths)))
    if donor_prefixes:
        # 与原始审计完全相同的轨迹、batch 顺序与截止步数。
        candidates = defaultdict(list)
        for i, length in enumerate(ds.lengths):
            if length >= 4:
                key = ds.source.get_tracklet_key(i)
                candidates[training_scene(key)].append((hashlib.sha256(key.encode()).digest(), i))
        tracklets = [idx for scene in sorted(candidates) for _, idx in sorted(candidates[scene])[:3]]
    lengths = {i: min(49, ds.lengths[i]) if donor_prefixes else ds.lengths[i] for i in tracklets}
    counts = {ds.source.get_tracklet_key(i): [None] * n for i, n in lengths.items()}
    records = []
    for key in counts:
        training_scene(key)
    builder = host.evaluation_builder
    builder.detailed_diagnostics = False
    for start in range(0, len(tracklets), 16):
        tracks = tracklets[start:start + 16]
        for t in range(1, max(lengths[i] for i in tracks)):
            reqs = [data.EndpointRequest(57, i, 4, 1, lengths[i], t) for i in tracks if t < lengths[i]]
            batch, out = host._forward_raw(ds.__getitems__(reqs), builder, training=False)
            try:
                selected = out.accepted_box.detach().cpu().double().numpy()
                for i, ctx in enumerate(builder._pending):
                    key = ctx['row']['tracklet_key']
                    if t == 1:
                        # 合法首框没有预测，以它在首个历史槽中的实际 0/1 提示计数。
                        ids = batch['point_ids'][i, 2].detach().cpu().numpy()
                        hints = batch['points'][i, 2, :, 4].detach().cpu().numpy()
                        counts[key][0] = len(np.unique(ids[(ids >= 0) & (hints == 1)]))
                    selected[i, :3] += ctx['anchor'][:3]
                    unique = np.unique(ctx['sampled_ids'], return_index=True)[1]
                    xyz = ctx['sampled_xyz'][unique]
                    counts[key][t] = int(data.inside_box(xyz, selected[i], ctx['state'].size).sum())
                    if donor_prefixes:
                        frame = ctx['row']['frames'][t]
                        gt = data.box_array(frame['3d_bbox'])
                        err = sensor_xyz(selected[i, :3] - gt[:3], -gt[3])
                        records.append(dict(track=key, index=t, time=float(frame['timestamp']),
                                            sampled_inside_prediction=counts[key][t], gt=gt.tolist(),
                                            err_lwh=err.tolist()))
                builder.commit(out, batch, diagnostics=False)
            finally:
                host._release_materialization(batch, builder)
    if builder.states or any(None in values for values in counts.values() if len(values) > 1):
        raise RuntimeError('incomplete autonomous density rollout')
    return counts, records


def paired_windows(donors, tracks):
    """复用原 240 个端点、供体抽签和末中心；在固定层内逐 lag 打散已经锚定的残差。"""
    rng = np.random.default_rng(314159)
    compressed = [(row, training_scene(row['track']), tuple(density_bins(row['counts']))) for row in donors]
    metadata = {(key, row['index']): row for key, rows in tracks.items() for row in rows}
    windows, strata = [], defaultdict(list)
    for row, scene, bins in compressed:
        key, t = row['track'], row['endpoint']
        if (t - 4) % 7:
            continue
        pool = [d for d, sc, b in compressed if sc != scene and b == bins]
        if not pool:
            pool = [d for d, sc, b in compressed if sc != scene and b[-1] == bins[-1]]
        donor = pool[rng.integers(len(pool))]
        rng.normal(size=(3, 3))  # 原 2×3 实验在抽末框之前生成高斯对照。
        yaw = np.asarray([metadata[key, i]['gt'][3] for i in range(t - 3, t)])
        errors = sensor_xyz(np.asarray(donor['err_lwh']), yaw)
        delta = errors - errors[-1]
        last = sensor_xyz(np.r_[rng.uniform(-.3, .3, 2), 0.], yaw[-1])
        strata[scene, bins].append(len(windows))
        windows.append(dict(track=key, endpoint=t, last=last, continuous=delta, shuffled=delta.copy()))
    shuffle = np.random.default_rng(20261121)
    for indices in strata.values():
        for lag in (0, 1):
            for dst, src in zip(indices, shuffle.permutation(indices)):
                windows[dst]['shuffled'][lag] = windows[src]['continuous'][lag]
    return windows


def shuffled_audit(host, ds, data, windows):
    from copy import deepcopy
    from utils.tracking_metrics import BoxMetricsTarget, LocalYawBox

    mapping = {ds.source.get_tracklet_key(i): i for i in range(len(ds.lengths))}
    results = []
    for start in range(0, len(windows), 16):
        chunk = windows[start:start + 16]
        reqs = [data.EndpointRequest(57, mapping[w['track']], 4, w['endpoint'] - 1,
                                     w['endpoint'] + 1, w['endpoint']) for w in chunk]
        rows = ds.__getitems__(reqs)
        builder = data.BatchBuilder(host.config)
        seeds = [builder._new_state(dict(row, first_frame=data.metadata(ds.source, r.tracklet, 0)))
                 for row, r in zip(rows, reqs)]
        signatures = {}
        for kind in ('shared', 'continuous', 'shuffled'):
            builder = data.BatchBuilder(host.config)
            builder.detailed_diagnostics = False
            for row, w, original in zip(rows, chunk, seeds):
                state = deepcopy(original)
                deltas = np.zeros((3, 3)) if kind == 'shared' else w[kind]
                for idx, offset in zip(row['history_ids'], deltas):
                    state.boxes[idx][:3] += w['last'] + offset
                    state.box_provenance[idx] = data.PERTURBED_BOX
                a, b = row['history_ids'][-2:]
                state.trusted = [(state.boxes[i].copy(), state.times[i]) for i in (a, b)]
                state.trusted_velocity = (state.boxes[b][:3] - state.boxes[a][:3]) / (state.times[b] - state.times[a])
                state.seed_offset[:3] = w['last']
                builder.states[row['request'].state_key] = state
            batch, out = host._forward_raw(rows, builder, training=False)
            try:
                selected = out.accepted_box.detach().cpu().double().numpy()
                ids = batch['point_ids'][:, -1].detach().cpu().numpy()
                values = []
                for i, (w, ctx) in enumerate(zip(chunk, builder._pending)):
                    row = ctx['row']
                    frame = row['frames'][w['endpoint']]
                    target = data.box_array(frame['3d_bbox'])
                    target[:3] -= ctx['anchor'][:3]
                    size = data.box_size(frame['3d_bbox'], frame)[[1, 0, 2]]
                    metric = BoxMetricsTarget(LocalYawBox(target, size), up_axis=(0, 0, 1))
                    iou = metric.metrics_3d(LocalYawBox(selected[i], ctx['state'].size[[1, 0, 2]]))[1]
                    signature = hashlib.sha256(ids[i].tobytes()).hexdigest()
                    if signatures.setdefault(i, signature) != signature:
                        raise RuntimeError('paired augmentation changed the current point sequence')
                    values.append(dict(track=w['track'], endpoint=w['endpoint'], condition=kind,
                                       current_ids=signature, selected_iou=float(iou),
                                       shuffled_changed=not np.array_equal(w['continuous'], w['shuffled'])))
                results.extend(values)
                builder.commit(out, batch, diagnostics=False)
            finally:
                host._release_materialization(batch, builder)
    return dict(split='train', checkpoint_sha256=E57_CHECKPOINT, rows=results,
                mean_iou={kind: float(np.mean([r['selected_iou'] for r in results if r['condition'] == kind]))
                          for kind in ('shared', 'continuous', 'shuffled')})


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--donor-source', type=Path, required=True, help='原始 b819e76 源码')
    parser.add_argument('--donor-run', type=Path, required=True, help='已完成的 uniform run 目录')
    parser.add_argument('--path', type=Path, help='KITTI training 数据根目录')
    parser.add_argument('--donor-frames', type=Path, help='可选原始审计 JSONL。干净 checkout 默认重算同一批供体')
    parser.add_argument('--output', type=Path, help='生成供正式训练读取的密度与残差 JSON')
    parser.add_argument('--shuffled-audit', type=Path, help='在匹配的 240 个端点执行三组冻结前向')
    args = parser.parse_args()
    destinations = [p.resolve() for p in (args.output, args.shuffled_audit) if p]
    if not destinations or len(set(destinations)) != len(destinations) or any(p.exists() for p in destinations):
        parser.error('provide distinct new files for --output and/or --shuffled-audit')
    archive = args.donor_frames or ROOT / 'artifacts/ct_checks/20261121_v41_paper_transfer/verified/train_frames.jsonl'
    archived = args.donor_frames is not None or archive.is_file()
    if archived:
        donors, tracks = donor_windows(archive)
    with frozen_teacher(args) as (host, ds, data):
        if not archived:
            _, records = collect_history(host, ds, data, donor_prefixes=True)
            donors, tracks = verified_windows(records)
        if args.output:
            receivers, _ = collect_history(host, ds, data)
            payload = dict(schema=SCHEMA, split='train', checkpoint_sha256=E57_CHECKPOINT,
                           teacher_source_sha256=E57_SOURCE, source_sha256=ds.source_sha256,
                           reference_archive_sha256=DONOR_ARCHIVE, donor_geometry_sha256=DONOR_GEOMETRY,
                           donor_origin='archived_frames' if archived else 'frozen_replay',
                           donors=donors, receivers=receivers)
            HistoryAugmentation(payload).validate_receivers(
                {ds.source.get_tracklet_key(i): n for i, n in enumerate(ds.lengths)})
            publish(args.output, payload)
            fingerprint = hashlib.sha256(args.output.read_bytes()).hexdigest()
            print(f'{args.output}: {fingerprint} ({len(donors)} segments, {len(receivers)} tracks)')
        if args.shuffled_audit:
            result = shuffled_audit(host, ds, data, paired_windows(donors, tracks))
            publish(args.shuffled_audit, result)
            print(json.dumps(result['mean_iou']))


if __name__ == '__main__':
    main()
