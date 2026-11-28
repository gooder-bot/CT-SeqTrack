"""在隔离工程目录使用固定 E42、原生 KITTI 队列复核中心修正、无标注推理和单步耗时。"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.ct_v41.config import normalize_config, recipe_scope
from models.ct_v41.data import BatchBuilder, EndpointRequest
from models.ct_v41.entry import configure_numerics
from models.ct_v41.epoch_runtime import isolated_evaluation_rng
from models.ct_v41.input import ONLINE_FIELDS
from models.ct_v41.runtime import make_evaluation
from models.ctseqtrackv41 import CTSEQTRACKV41

E42 = '2a603be56846f981805fb802468e64b0bd17a5e295b01fe773b79a1eff410acb'
CENTER_ARCHIVE = 'b7af64013ae17d1ec3c373691f4a0bfd1ef49a1484248c4c6a381d50453227d5'
PILOT = {'original': 'test_deploy_base', 'paired': 'test_deploy_base',
         'switch': 'test_online_center_switch'}


def model(args, arm):
    checkpoint = args.run / 'formal_checkpoints/epoch=042.ckpt'
    settings = yaml.safe_load((args.run / 'resolved_config.yaml').read_text())
    settings.update(path=str(args.data), devices=0, test=True, checkpoint=str(checkpoint),
                    eval_split='test', eval_checkpoint_epoch=42, log_dir=None)
    if arm == 'switch':
        settings['v41_support_center'] = True
    cfg = normalize_config(settings)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    host = CTSEQTRACKV41(cfg)
    state = torch.load(checkpoint, map_location='cpu')
    if int(state['epoch']) + 1 != 42:
        raise ValueError('expected E42')
    if arm == 'switch':
        added = {k: v for k, v in host.state_dict().items() if k.startswith('tracker.support_center.')}
        if not added or added.keys() & state['state_dict'].keys():
            raise RuntimeError('expected only new center buffers outside frozen state')
        state['state_dict'].update(added)
    host.load_state_dict(state['state_dict'], strict=True)
    del state
    host.eval().cuda()
    if arm == 'switch':
        host.tracker.support_center.load_archive(args.pilot / 'center_weights.npz')
    host.tracker.local_vote_head.paired_eval_shapes = arm != 'original'
    host.tracker.q0_epoch = 42
    return host


def online_only(tracker):
    removed = set()
    def strip(module, args):
        batch = args[0]
        allowed = {k: v for k, v in batch.items()
                   if k in ONLINE_FIELDS and k != '_ct_materialization_cache'}
        if 'target_box' in allowed or 'segmentation_labels' in allowed:
            raise ValueError('online field leak')
        removed.update(batch.keys() - allowed.keys())
        return (allowed,) + args[1:]
    return tracker.register_forward_pre_hook(strip), removed


def replay(args):
    host = model(args, args.arm)
    loader = host.test_dataloader()
    builder = BatchBuilder(host.config)
    builder.detailed_diagnostics = True
    rows_out = []
    evaluation = make_evaluation(host.config, full=False, row_sink=rows_out.append)
    evaluation.add_singleton_tracks(loader.dataset)
    branch_counts = [0, 0, 0]
    hook, removed = online_only(host.tracker)
    mismatches = {}
    first_difference = {}
    frames = 0
    distances = []
    reference = args.pilot / (PILOT[args.arm] + '.jsonl.gz')
    ledger = {}
    with gzip.open(reference, 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            ledger[row['track'], row['scan']] = row
    if len(ledger) != 6304:
        raise ValueError('incomplete frozen reference')
    with torch.no_grad(), recipe_scope(host.config), isolated_evaluation_rng(
            host.config, 'test', loader=loader, module=host):
        for rows in loader:
            batch, out = host._forward_raw(rows, builder, training=False)
            evidence = getattr(out.decoder, 'support_center', None)
            if evidence is not None and args.arm == 'switch':
                for branch in evidence.branch.cpu().tolist():
                    branch_counts[int(branch)] += 1
            commits = builder.commit(out, batch, diagnostics=True)
            evaluation.add_batch(rows, batch, out, commits)
            scores = {(f"{int(row['scene_id']):04d}/{row['track_id']}", row['raw_frame_id']): row
                      for row in rows_out if not row['initialization']}
            rows_out.clear()
            selected = out.selected_index.cpu().tolist()
            for i, row in enumerate(rows):
                frame = row['frames'][row['request'].frame]
                key = (f"{int(frame['scene_id']):04d}/{int(frame['meta']['track_id'])}",
                       int(frame['raw_frame_id']))
                if key not in ledger:
                    raise ValueError(f'unexpected or duplicate physical frame {key}')
                ref = ledger.pop(key)
                score = scores[key]
                observed = dict(slot=selected[i], write=commits[i]['trusted_write'],
                                iou=score['iou'], error=score['distance'],
                                success=score['success'], precision=score['precision'])
                distances.append(score['distance'])
                for name, val in observed.items():
                    if val != ref[name]:
                        mismatches[name] = mismatches.get(name, 0) + 1
                        first_difference.setdefault(name, [*key, val, ref[name]])
            frames += len(rows)
            host._release_materialization(batch, builder)
            if frames % 512 < len(rows):
                print('checked', frames, flush=True)
    hook.remove()
    summary = evaluation.summary()
    if ledger or frames != 6304 or summary['frames'] != 6424 or summary['tracklets'] != 120:
        raise ValueError('physical frame coverage differs from the archive')
    expected = [99, 5558, 647]
    return dict(mode='replay', arm=args.arm, reference=str(reference), prediction_frames=frames,
                frames=summary['frames'], tracklets=summary['tracklets'],
                success=summary['success'], precision=summary['precision'],
                over10m=int(np.count_nonzero(np.asarray(distances) > 10)),
                branch_counts=branch_counts, compared_fields=['slot', 'write', 'iou', 'error', 'success', 'precision'],
                mismatches=mismatches, first_difference=first_difference, removed_fields=sorted(removed),
                passed=not mismatches and (args.arm != 'switch' or branch_counts == expected))


def benchmark(args):
    arms = tuple(PILOT)
    hosts = {arm: model(args, arm) for arm in arms}
    for host in hosts.values():
        host.config.batch_size = args.batch_size
        host.config.eval_split = args.benchmark_split
        host.config.workers = 0
    baseline = hosts['original']
    loader = baseline.test_dataloader()
    dataset = loader.dataset
    lengths = dataset.lengths
    buckets = []
    if args.batch_size == 1:
        def track_rows(track):
            for step in range(1, args.warmup + args.steps + 1):
                req = EndpointRequest(0, track, 4, 1, lengths[track], step)
                yield dataset.__getitems__([req])
        eligible = [i for i, n in enumerate(lengths) if n > args.warmup + args.steps]
        for idx in np.linspace(0, len(eligible)-1, min(4, len(eligible))).astype(int):
            buckets.append(track_rows(eligible[idx]))
    else:
        from itertools import islice
        buckets.append(islice(loader, args.warmup + args.steps))
    times = {arm: [] for arm in arms}
    peaks = {arm: [] for arm in arms}
    rows_count = []
    trace = []
    with torch.no_grad(), recipe_scope(baseline.config):
        for seq, bucket in enumerate(buckets):
            builder = BatchBuilder(baseline.config)
            builder.detailed_diagnostics = False
            for step, rows in enumerate(bucket):
                branches = {arm: copy.deepcopy(builder) for arm in arms}
                for branch in branches.values():
                    branch._history_pools = builder._history_pools.copy()
                pos = step % len(arms)
                order = list(arms[pos:] + arms[:pos])
                if (step // len(arms)) % 2:
                    order.reverse()
                measurements = {}
                for arm in order:
                    branch, host = branches[arm], hosts[arm]
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    start_mem = torch.cuda.memory_allocated()
                    start = perf_counter()
                    batch, out = host._forward_raw(rows, branch, training=False)
                    branch.commit(out, batch, diagnostics=False)
                    torch.cuda.synchronize()
                    elapsed_ms = (perf_counter() - start) * 1000
                    used = torch.cuda.max_memory_allocated() - start_mem
                    if step >= args.warmup:
                        times[arm].append(elapsed_ms)
                        peaks[arm].append(used)
                        measurements[arm] = elapsed_ms
                    host._release_materialization(batch, branch)
                    del batch, out
                builder = branches['original']
                if step >= args.warmup:
                    rows_count.append(len(rows))
                    trace.append(dict(sequence=seq, step=step, rows=len(rows), ms=measurements))
    stats = {}
    for arm in arms:
        if not times[arm]:
            raise RuntimeError('no timed requests')
        stats[arm] = dict(p50_ms=float(np.median(times[arm])), p95_ms=float(np.percentile(times[arm], 95)),
                          paired_delta_median_ms=float(np.median(np.subtract(times[arm], times['original']))),
                          endpoints_per_second=sum(rows_count)*1000/sum(times[arm]),
                          incremental_peak_allocated_mib=max(peaks[arm])/2**20,
                          model_parameter_buffer_mib=sum(t.numel()*t.element_size() for t in
                              list(hosts[arm].parameters()) + list(hosts[arm].buffers())) / 2**20)
    return dict(mode='benchmark', split=args.benchmark_split, batch_size=args.batch_size,
                warmup_per_sequence=args.warmup,
                timed_requests=len(trace), rows_per_request=rows_count, stats=stats, trace=trace)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('replay', 'benchmark'))
    parser.add_argument('--arm', choices=PILOT, default='switch')
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--pilot', required=True, type=Path)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--result', required=True, type=Path)
    parser.add_argument('--batch-size', type=int, default=1, choices=(1, 16))
    parser.add_argument('--benchmark-split', choices=('dev', 'test'), default='dev')
    parser.add_argument('--warmup', type=int, default=4)
    parser.add_argument('--steps', type=int, default=12)
    args = parser.parse_args()
    if args.result.exists():
        raise FileExistsError(args.result)
    checkpoint = args.run / 'formal_checkpoints/epoch=042.ckpt'
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != E42:
        raise ValueError('unexpected E42 fingerprint')
    if hashlib.sha256((args.pilot/'center_weights.npz').read_bytes()).hexdigest() != CENTER_ARCHIVE:
        raise ValueError('unexpected pilot center coefficients')
    configure_numerics()
    torch.set_num_threads(1)
    result = replay(args) if args.mode == 'replay' else benchmark(args)
    from models.ct_v41.provenance import source_identity
    result.update(checkpoint_sha256=E42, center_weights_sha256=CENTER_ARCHIVE,
                  runtime_source_sha256=source_identity()['sha256'], cuda_device=torch.cuda.get_device_name(),
                  visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), torch=torch.__version__)
    args.result.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(args.result, {k: v for k, v in result.items() if k not in ('trace', 'removed_fields')}, flush=True)
    if result.get('passed') is False:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
