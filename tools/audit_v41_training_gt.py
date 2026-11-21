"""在 KITTI 训练轨迹上配对检查自然历史、原样副本与一次过去 GT 校正。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
HORIZON = 8


def repair_past(state, past):
    """仅传入已提交前缀的 {frame: (timestamp, absolute XYZ/yaw)}。"""
    joint = state.joint
    required = set(range(max(0, state.next_frame - 3), state.next_frame))
    if joint is None or set(past) != required or set(state.boxes) != required:
        raise ValueError('repair needs exactly the last committed joint history')
    by_time = {}
    for frame, (timestamp, box) in past.items():
        if timestamp != state.times[frame] or not np.isfinite(box).all():
            raise ValueError('invalid teacher frame/time')
        state.boxes[frame] = box.copy()
        by_time[timestamp] = box
        if frame in joint.weak_history_boxes:
            joint.weak_history_boxes[frame] = box.copy()
    if joint.motion_time != state.times[state.next_frame - 1]:
        raise ValueError('joint motion origin is not at the intervention endpoint')
    joint.motion_box = by_time[joint.motion_time].copy()
    joint.pending_box = joint.pending_scan = None
    state.trusted = [(by_time.get(ts, box).copy(), ts) for box, ts in state.trusted]
    for index, measurement in enumerate(joint.measurements):
        if measurement.time in by_time:
            measurement.box = by_time[measurement.time].copy()
        if index and ((measurement.time in by_time) != (joint.measurements[index-1].time in by_time)):
            measurement.adjacent = False
    # 截断跨来源的运动边。保留未被涉及的旧速度及其原始年龄。
    if state.trusted_velocity_valid and any(ts in by_time for _, ts in state.trusted):
        slots = state.trusted
        if (len(slots) == 2 and all(ts in by_time for _, ts in slots)
                and slots[-1][1] == state.last_velocity_update_time and slots[0][1] < slots[1][1]
                and len(joint.measurements) >= 2 and joint.measurements[-1].adjacent
                and [m.time for m in joint.measurements[-2:]] == [ts for _, ts in slots]):
            state.trusted_velocity = (slots[1][0][:3] - slots[0][0][:3]) / (slots[1][1] - slots[0][1])
        else:
            state.trusted_velocity = np.zeros(3, dtype=np.float64)
            state.trusted_velocity_valid = False
            state.last_velocity_update_time = None
    for frame in state.transitions:
        if frame in required and frame-1 not in required:
            state.transitions[frame] = False


def sampled_coverage(builder, batch, output, ctx):
    """按原始扫描行号联合局部/远区细采，粗网格特征不计入实采车辆点。"""
    from models.ct_v41.data import box_array, box_size
    from models.ct_v41.search import inside_box
    from models.ct_v41.joint_grid import inside

    row = ctx['row']
    frame = row['frames'][row['request'].frame]
    xyz, ids = builder._points(frame)
    truth, size = box_array(frame['3d_bbox']), box_size(frame['3d_bbox'], frame)
    in_target = inside_box(xyz, truth, size) & (ids >= 0)
    xyz, ids = xyz[in_target], ids[in_target]
    search = ctx['search_plan']
    roi = np.zeros(len(ids), dtype=bool)
    for box in (search.main_box, search.backup_box):
        if box is not None:
            roi |= inside_box(xyz, box, ctx['state'].size, scale=builder.config.bb_scale,
                              offset=builder.config.bb_offset)
    obs = output.joint.observations
    fine_ids = obs.remote.ids[0][obs.remote.point_valid[0]].detach().cpu().numpy()
    if obs.coarse:
        remote_row = batch['joint_rows'][0]
        record, = obs.coarse
        pose = None
        if remote_row['pending_valid']:
            pose = np.array(remote_row['pending_box'], dtype=np.float32, copy=True)
            pose[:3] += batch['anchor_box'][0, :3].detach().cpu().numpy()
        elif len(record.proposals_sensor):
            yaw = float(output.prior.box[0, 3].detach().cpu())
            pose = np.array([*record.proposals_sensor[0], yaw], dtype=np.float32)
        if pose is not None:
            roi |= inside(xyz.astype(np.float32), pose, .625 * batch['box_size'][0].detach().cpu().numpy() + 2.)
    sampled = np.union1d(ctx['sampled_ids'], fine_ids)
    sampled_targets = np.intersect1d(ids, sampled)
    roi_targets = np.unique(ids[roi])
    if not np.isin(sampled_targets, roi_targets).all():
        raise ValueError('sampled true point lies outside the reconstructed local/fine search regions')
    return dict(raw=len(np.unique(ids)), roi=len(roi_targets), sampled=len(sampled_targets))


def forward_one(host, builder, row, *, corrected_times=()):
    from utils.tracking_metrics import BoxMetricsTarget, LocalYawBox
    batch, output = host._forward_raw([row], builder, training=False)
    try:
        coverage = sampled_coverage(builder, batch, output, builder._pending[0])
        def tensor(name):
            return batch[name][0].detach().cpu().double().numpy()

        gt = BoxMetricsTarget(LocalYawBox(tensor('target_box'), tensor('target_box_size')[[1, 0, 2]]),
                              up_axis=(0, 0, 1))
        accepted = output.accepted_box[0].detach().cpu().double().numpy()
        exact_iou, compat_iou, distance = gt.metrics_3d(LocalYawBox(accepted, tensor('box_size')[[1, 0, 2]]))
        # 同一真实时间戳上的校正量仍可输入本帧先验。提交时不能用其与随后的
        # 自主测量组成一次新的测速，也不能由这条跨来源边提前更新模板。
        latest = builder._pending[0]['state'].joint.measurements[-1]
        original_pair_flag = latest.can_start_pair
        if latest.time in corrected_times:
            latest.can_start_pair = False
        try:
            record, = builder.commit(output, batch, diagnostics=True)
        finally:
            latest.can_start_pair = original_pair_flag
        frame = row['frames'][row['request'].frame]
        return dict(frame=row['request'].frame, scan=int(frame['raw_frame_id']),
                    time=float(frame['timestamp']), iou=float(compat_iou), exact_iou=float(exact_iou),
                    distance=float(distance),
                    box=accepted.tolist(), anchor=tensor('anchor_box').tolist(),
                    source=record['commit_source'], trusted=bool(record['trusted_write']),
                    velocity_updated=bool(record['trusted_velocity_updated']), **coverage)
    finally:
        host._release_materialization(batch, builder)


def collect_track(host, dataset, config, track, epoch):
    from models.ct_v41 import data
    length = dataset.lengths[track]
    natural = data.BatchBuilder(config)
    events, steps, treated = [], {}, False
    def make_request(j):
        return data.EndpointRequest(epoch, track, 4, 1, length, j)

    for i in range(1, length):
        row = dataset[make_request(i)]
        obs = steps[i] = forward_one(host, natural, row)
        if obs['iou'] >= .5:
            treated = False
        if treated or obs['iou'] >= .1 or obs['raw'] < 3 or obs['sampled'] >= 3:
            continue
        treated = True
        event = dict(frame=i, scan=obs['scan'], track=row['tracklet_key'],
                     reason='crop' if obs['roi'] < 3 else 'sampling', trigger=obs)
        events.append(event)
        for mode in ('clone', 'corrected'):
            follow = event[mode] = []
            if i + 1 == length:
                continue
            branch = data.BatchBuilder(config)
            key = row['request'].state_key
            state = branch.states[key] = deepcopy(natural.states[key])
            corrected_times = ()
            if mode == 'corrected':
                repair_past(state, {j: (float(row['frames'][j]['timestamp']),
                                        data.box_array(row['frames'][j]['3d_bbox'])) for j in state.boxes})
                corrected_times = tuple(state.times[j] for j in state.boxes)
            # 新 builder 不持有自然分支的任何裁点、提示或未来状态。
            for j in range(i + 1, min(i + HORIZON + 1, length)):
                follow.append(forward_one(host, branch, dataset[make_request(j)], corrected_times=corrected_times))
    for event in events:
        start = event['frame'] + 1
        event['natural'] = [steps[j] for j in range(start, min(start + HORIZON, length))]
        if event['natural'] != event['clone']:
            raise RuntimeError(f"unmodified clone diverged at {event['track']} frame {start}")
    return events


def summarize(events):
    result = {}
    for mode in ('natural', 'clone', 'corrected'):
        counts = Counter()
        for event in events:
            rows = event[mode]
            counts['triggered'] += 1
            counts['with_next'] += bool(rows)
            counts['next_sampled_three'] += bool(rows and rows[0]['sampled'] >= 3)
            counts['next_good'] += bool(rows and rows[0]['iou'] >= .5)
            streak, previous, longest = 0, None, 0
            for row in rows:
                if previous is not None and row['scan'] != previous + 1:
                    streak = 0
                streak = streak + 1 if row['iou'] >= .5 else 0
                longest = max(longest, streak)
                previous = row['scan']
            counts['stable_three'] += longest >= 3
            counts['truncated_followup'] += len(rows) < HORIZON
            counts['censored_before_stable'] += len(rows) < HORIZON and longest < 3
        result[mode] = dict(counts)
    return dict(reasons=dict(Counter(e['reason'] for e in events)), outcomes=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfg', required=True)
    parser.add_argument('--path', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tracks-per-scene', type=int, default=1, help='0 selects all training tracks')
    args = parser.parse_args()
    if args.tracks_per_scene < 0 or args.output.exists():
        parser.error('tracks-per-scene must be nonnegative and output must be new')

    import torch
    from datasets.kitti_v41 import get_v41_raw_dataset
    from models.ct_v41 import data, entry
    from models.ct_v41.config import recipe_scope
    from models.ct_v41.runtime import validate_resume_payload
    from models.ctseqtrackv41 import CTSEQTRACKV41
    from pytorch_lightning import seed_everything
    cfg = entry.parse_config(['--cfg', args.cfg, '--path', args.path, '--checkpoint', str(args.checkpoint),
                              '--test', '--eval_split', 'dev'])
    if cfg.dataset != 'kitti_mf' or not cfg.v41_joint_search:
        raise ValueError('the GT diagnostic requires a compatible KITTI joint checkpoint')
    entry.configure_numerics(cfg)
    seed_everything(cfg.seed, workers=True)
    with recipe_scope(cfg), torch.no_grad():
        saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        payload = validate_resume_payload(saved[entry.runtime_key(cfg)], cfg, training=False)
        epoch = cfg.eval_checkpoint_epoch = payload['completed_epoch']
        host = CTSEQTRACKV41(cfg)
        host.load_state_dict(saved.pop('state_dict'), strict=True)
        del saved
        host.cuda().eval()
        source = get_v41_raw_dataset(cfg, 'auxiliary')
        dataset = data.RawEndpointDataset(source, joint_search=True)
        if dataset.source_sha256 != payload['sampler']['source_sha256']:
            raise ValueError('training dataset differs from checkpoint')
        by_scene = defaultdict(list)
        for track, length in enumerate(dataset.lengths):
            key = source.get_tracklet_key(track)
            parts = key.split('/')
            if parts[:3] != ['kitti_v41', 'kitti_tracking', 'train'] or parts[3] not in {
                    f'{j:04d}' for j in range(17)}:
                raise ValueError(f'unexpected training track identity: {key}')
            if length > 1:
                by_scene[parts[3]].append((hashlib.sha256(key.encode()).digest(), track))
        selected = [i for scene in sorted(by_scene)
                    for _, i in sorted(by_scene[scene])[:args.tracks_per_scene or None]]
        events = [event for track in selected for event in collect_track(host, dataset, cfg, track, epoch)]
    checksum = hashlib.sha256()
    with args.checkpoint.open('rb') as checkpoint_stream:
        for block in iter(lambda: checkpoint_stream.read(1 << 20), b''):
            checksum.update(block)
    report = dict(schema='ct_seqtrack.v41.train_gt_diagnostic.v1', run_id=payload['run_id'],
                  epoch=epoch, source=payload['source'], config_sha256=payload['config_sha256'],
                  checkpoint_sha256=checksum.hexdigest(), trigger_iou_mode='benchmark_compat',
                  outcome_iou_mode='benchmark_compat', additional_frame_iou_mode='geometry_exact',
                  dataset_sha256=dataset.source_sha256, tracks=[source.get_tracklet_key(i) for i in selected],
                  horizon=HORIZON, events=events, summary=summarize(events))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, allow_nan=False)
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
