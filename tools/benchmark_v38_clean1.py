"""同设备、独立源码进程的 v38 清理前后工程测速；不保存权重或启动正式训练。

默认 B1/Full × model/transaction，20 预热、100 测量、3 次交替重复。
transaction 包含 raw 读取、prepare、prior 监督、loss/被动诊断、Adam 和 commit；
不包含 Lightning logger/epoch 文件写入，不能解释为正式训练墙钟加速。
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--baseline-source', type=Path,
        help='已解压的原 v38 完整源码；缺省只检查当前源码，不作加速比较')
    parser.add_argument('--candidate-source', type=Path, default=ROOT)
    parser.add_argument('--path', help='真实数据根；synthetic 模式不读取数据集')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--gpu', type=int, default=0, help='单个物理 CUDA GPU，所有子进程相同')
    parser.add_argument('--recipes', nargs='+', choices=('b1_standard', 'full_standard'),
                        default=['b1_standard', 'full_standard'])
    parser.add_argument('--stages', nargs='+', choices=('model', 'transaction'),
                        default=['model', 'transaction'])
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--point-count', type=int, default=1024)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--synthetic', action='store_true', help='只作工程 smoke，不代表真实性能')
    parser.add_argument('--candidate-profile', choices=('legacy', 'optimized_fp32', 'optimized_bf16'),
                        default='optimized_fp32', help='仅工程检查；默认清理前后均 FP32')
    parser.add_argument('--candidate-projection', choices=('concat', 'factorized'), default='concat')
    parser.add_argument('--output', type=Path, help='新的 artifacts/ct_checks/.../report.json')
    parser.add_argument('--worker-spec', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if min(args.steps, args.repeats, args.batch_size, args.point_count) < 1 or min(args.warmup, args.workers, args.gpu) < 0:
        parser.error('steps/repeats/batch/points must be positive; warmup/workers/gpu nonnegative')
    if len(set(args.recipes)) != len(args.recipes) or len(set(args.stages)) != len(args.stages):
        parser.error('recipes and stages must be distinct')
    if not args.synthetic and (args.batch_size != 16 or args.point_count != 1024):
        parser.error('real-data benchmark retains batch16/1024 points; reduced shapes require --synthetic')
    return args


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


class SyntheticSource:
    """确定性 raw 输入，用于没有 nuScenes/Lightning 的 CPU 生命周期检查。"""
    def __init__(self, tracks, length, points):
        self.tracks, self.length, self.points = tracks, length, points

    def get_num_tracklets(self):
        return self.tracks

    def get_num_frames_tracklet(self, index):
        return self.length

    def get_tracklet_key(self, index):
        return 'benchmark_synthetic/' + str(index)

    def get_frames(self, index, frame_ids):
        import numpy as np
        offsets = np.random.default_rng(713 + index).normal(0, .3, (self.points, 3))
        rows = []
        for frame in frame_ids:
            center = np.asarray([frame * .2, index * 20., 0.])
            rows.append(dict(pc=offsets + center, point_ids=np.arange(self.points),
                **{'3d_bbox': np.r_[center, .1]}, box_size=np.asarray([4., 2., 2.]),
                timestamp=100. + frame * .5, scene_id='synthetic'))
        return rows

    def get_frame_metadata(self, index, frame):
        return self.get_frames(index, [frame])[0]


def tensor_digest(items):
    value = hashlib.sha256()
    for name, tensor in sorted(items):
        tensor = tensor.detach().cpu().contiguous()
        value.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        value.update(tensor.numpy().tobytes())
    return value.hexdigest()


def production_identity(source):
    """与入口相同的内容哈希，父进程无需导入两个源码树。"""
    source = Path(source)
    paths = [source / 'main.py', source / 'requirement.txt']
    for name in ('models', 'datasets', 'utils'):
        paths.extend((source / name).rglob('*.py'))
    hashes = {path.relative_to(source).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(paths)}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def worker(spec):
    # 此脚本始终来自当前源码，生产模块只从指定的独立源码根导入。
    source = Path(spec['source']).resolve()
    if production_identity(source) != spec['expected_source_sha256']:
        raise ValueError('source changed after paired benchmark started')
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != spec['tool_sha256']:
        raise ValueError('benchmark tool changed after the comparison started')
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    from models.ct_v38.config import load_config, config_identity
    from models.ct_v38.entry import configure_numerics, source_identity
    from models.ct_v38.data import (BatchBuilder, RawEndpointDataset, ReadyQueueBatchSampler,
                                   raw_collate, build_loaders, stable_seed)
    from models.ct_v38.model import JointTracker
    from models.ct_v38.runtime import move_tensors
    from models.ct_v38.training_diagnostics import TrainingDiagnostics
    from utils.bn_policy import running_batch_norm
    from utils.batch_materialization import attach_materialization_cache, clear_materialization_cache
    from utils.execution_profile import optimized_execution

    configure_numerics()
    if spec['device'] == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    torch.set_num_threads(1)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    overrides = dict(ct_engineering_check=True, workers=spec['workers'],
        batch_size=spec['batch_size'], point_sample_size=spec['point_count'],
        checkpoint=None, init_checkpoint=None, log_dir=None, test=False,
        v31_evaluate_late3=False, accelerator='cpu' if spec['device'] == 'cpu' else 'gpu')
    if spec['source_label'] == 'candidate':
        overrides['execution_profile'] = spec['profile']
        if spec['projection'] == 'factorized':
            overrides['v38_engineering_options'] = ['unique_mask', 'factorized_projection']
    if spec['path'] is not None:
        overrides['path'] = spec['path']
    cfg = load_config(source / f"cfgs/ct_seqtrack/38_{spec['recipe']}_mini.yaml", overrides)
    if spec['synthetic']:
        raw = RawEndpointDataset(SyntheticSource(spec['batch_size'] * 2,
            max(16, spec['warmup'] + spec['steps'] + 4), spec['point_count']))
        sampler = ReadyQueueBatchSampler(raw.lengths, cfg.batch_size, seed=cfg.seed,
            training=True, short_window=4, long_window=8, curriculum_epochs=10,
            source_sha256=raw.source_sha256, reserve_windows=112,
            seed_translation=.3, seed_yaw_degrees=1.5,
            initial_seed_policy='initial_exact_other_perturbed_v1')
        loader = torch.utils.data.DataLoader(raw, batch_sampler=sampler, num_workers=0,
            collate_fn=raw_collate, generator=torch.Generator().manual_seed(stable_seed(42, 'train', 9)))
    else:
        loader = build_loaders(cfg, roles=('train',))['train']
    loader.batch_sampler.set_epoch(9)  # 成熟 W 窗口，确实覆盖递推状态。
    loader.generator.manual_seed(stable_seed(42, 'train', 9))
    tracker = JointTracker(cfg).to(spec['device']).train()
    initial = tensor_digest((name, tensor) for name, tensor in tracker.state_dict().items()
        if not name.startswith(('decoder.extension_projection.', 'decoder.memory_projection.')))
    optimizer = torch.optim.Adam(tracker.parameters(), lr=cfg.lr, weight_decay=cfg.wd,
        betas=(.5, .999), eps=1e-6, foreach=False, fused=False)
    builder = BatchBuilder(cfg)
    diagnostics = TrainingDiagnostics(cfg.seed, optimized=optimized_execution(cfg))
    diagnostics.reset(9)
    iterator = iter(loader)
    request_hash = hashlib.sha256()
    def next_rows():
        try:
            rows = next(iterator)
        except StopIteration as error:
            raise RuntimeError('requested benchmark exceeds one epoch; reduce steps') from error
        if len(rows) != cfg.batch_size:
            raise RuntimeError('benchmark refuses a reduced tail batch')
        request_hash.update(json.dumps([(row['tracklet_key'], asdict(row['request']))
            for row in rows], sort_keys=True).encode())
        return rows
    def release(batch):
        clear_materialization_cache(batch)
        clear = getattr(builder, 'clear_materialization_cache', None)
        if clear is not None:
            clear()
    def prepare(rows):
        batch = move_tensors(builder.prepare(rows), spec['device'])
        batch = attach_materialization_cache(batch, enabled=optimized_execution(cfg))
        prior = tracker.plan_prior(batch)
        supervise = getattr(builder, 'supervise_prior', None) or builder.acquire
        supervise(batch, prior, training=True)
        return batch, prior
    def finish_loss(batch, output, rows=None):
        output.loss_statistics = {}
        losses = tracker.compute_losses(batch, output)
        total = losses['loss_total']
        if not bool(torch.isfinite(total)):
            raise FloatingPointError('nonfinite benchmark loss')
        if rows is not None:
            diagnostics.add_batch(rows, batch, output, losses)
        total.backward()
        optimizer.step()
    def synchronize():
        if spec['device'] == 'cuda':
            torch.cuda.synchronize()
    prepared_hash = None
    if spec['stage'] == 'model':
        rows = next_rows()
        batch, prior = prepare(rows)
        # 不保留旧 forward 图，也不把相同状态重复提交成非法递推。
        del prior
        batch = {name: value.detach() if torch.is_tensor(value) else value for name, value in batch.items()}
        release(batch)
        fields = ('points', 'point_valid', 'point_ids', 'history_boxes', 'history_valid',
                  'history_pair_valid', 'history_times', 'target_box', 'fallback_box', 'segmentation_labels')
        prepared_hash = tensor_digest((key, batch[key]) for key in fields)
    times, allocated, reserved = [], [], []
    for step in range(spec['warmup'] + spec['steps']):
        synchronize()
        if step == spec['warmup'] and spec['device'] == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        if spec['stage'] == 'transaction':
            rows = next_rows()
            batch, prior = prepare(rows)
            running = len(rows) < cfg.batch_size or any(row['request'].drain for row in rows)
            with running_batch_norm(tracker.observation, enabled=running):
                output = tracker(batch, prior=prior)
            finish_loss(batch, output, rows)
            builder.commit(output, batch)
            release(batch)
            del prior
        else:
            output = tracker(batch)
            finish_loss(batch, output)
        synchronize()
        elapsed = time.perf_counter() - started
        if step >= spec['warmup']:
            times.append(elapsed)
            if spec['device'] == 'cuda':
                allocated.append(torch.cuda.memory_allocated() / 2 ** 20)
                reserved.append(torch.cuda.memory_reserved() / 2 ** 20)
        del output
    final_source = source_identity()
    if final_source['sha256'] != spec['expected_source_sha256']:
        raise ValueError('source changed during measured worker execution')
    result = dict(status='passed', **spec, torch=torch.__version__,
        source_identity=final_source, config_sha256=config_identity(cfg),
        initial_effective_parameters_sha256=initial, prepared_input_sha256=prepared_hash,
        request_order_sha256=request_hash.hexdigest(), sampler=loader.batch_sampler.state_dict(),
        parameter_count=sum(p.numel() for p in tracker.parameters()), checkpoint_saved=False,
        device_name=torch.cuda.get_device_name() if spec['device'] == 'cuda' else 'CPU',
        step_seconds=times, median_seconds=statistics.median(times), p90_seconds=percentile(times, .9),
        allocated_mib_median=statistics.median(allocated) if allocated else None,
        allocated_mib_p90=percentile(allocated, .9) if allocated else None,
        reserved_mib_median=statistics.median(reserved) if reserved else None,
        reserved_mib_p90=percentile(reserved, .9) if reserved else None,
        peak_allocated_mib=torch.cuda.max_memory_allocated() / 2 ** 20 if allocated else None,
        peak_reserved_mib=torch.cuda.max_memory_reserved() / 2 ** 20 if reserved else None)
    del iterator
    return result


def main(argv=None):
    args = parse_args(argv)
    if args.worker_spec:
        spec = json.loads(args.worker_spec.read_text(encoding='utf-8'))
        write_json(spec['result'], worker(spec))
        return 0
    sys.path.insert(0, str(ROOT))
    from tools._batch_check import report_destination
    from models.ct_v38.identity import BENCHMARK_SCHEMA
    destination = report_destination(args.output, version='38_clean1_benchmark')
    destination.parent.mkdir(parents=True, exist_ok=True)
    sources = {'candidate': args.candidate_source.resolve()}
    if args.baseline_source:
        sources = {'baseline': args.baseline_source.resolve(), **sources}
    for source in sources.values():
        if not (source / 'models/ct_v38/model.py').is_file():
            raise ValueError('source must be a complete v38 project root: ' + str(source))
    if len(sources) == 2 and len(set(sources.values())) != 2:
        raise ValueError('paired comparison requires distinct baseline and candidate source roots')
    source_hashes = {name: production_identity(source) for name, source in sources.items()}
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu),
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH='', PYTORCH_CUDA_ALLOC_CONF='backend:native')
    records = []
    report = dict(schema=BENCHMARK_SCHEMA, status='running', checkpoint_saved=False,
        synthetic=args.synthetic, device=args.device, physical_gpu=args.gpu,
        paired=len(sources) == 2, records=records,
        source_sha256=source_hashes, performance_evidence=not args.synthetic and args.device == 'cuda',
        cleanup_only_comparison=args.candidate_profile == 'optimized_fp32' and args.candidate_projection == 'concat',
        scope=dict(model='prepared fixed batch: forward/loss/backward/Adam; includes prior network',
            transaction='raw read/prepare/prior supervision/forward/loss/diagnostics/backward/Adam/commit',
            excluded='Lightning logger and epoch output IO; no formal training wall-time claim'),
        numerical_note='dropout random trajectories may differ after compaction; use separate same-weight deterministic checks',
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    try:
        for recipe in args.recipes:
            for stage in args.stages:
                for repeat in range(args.repeats):
                    labels = list(sources) if repeat % 2 == 0 else list(reversed(sources))
                    for label in labels:
                        stem = f'{recipe}_{stage}_{repeat + 1}_{label}'
                        result_path = destination.parent / (stem + '.json')
                        spec_path = destination.parent / (stem + '_spec.json')
                        spec = dict(schema=BENCHMARK_SCHEMA, source=str(sources[label]), source_label=label, recipe=recipe,
                            expected_source_sha256=source_hashes[label], tool_sha256=report['tool_sha256'],
                            stage=stage, repeat=repeat + 1, device=args.device, path=args.path,
                            profile=args.candidate_profile if label == 'candidate' else 'optimized_fp32',
                            projection=args.candidate_projection if label == 'candidate' else 'concat',
                            synthetic=args.synthetic, workers=args.workers, batch_size=args.batch_size,
                            point_count=args.point_count, warmup=args.warmup, steps=args.steps,
                            result=str(result_path))
                        write_json(spec_path, spec)
                        print(f'[{stem}] source={sources[label]} device={args.device}', flush=True)
                        with (destination.parent / (stem + '.log')).open('x', encoding='utf-8') as log:
                            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker-spec', str(spec_path)],
                                cwd=sources[label], env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
                        records.append(json.loads(result_path.read_text(encoding='utf-8')))
        pairs = []
        if len(sources) == 2:
            for recipe in args.recipes:
                for stage in args.stages:
                    for repeat in range(1, args.repeats + 1):
                        pair = {r['source_label']: r for r in records
                            if (r['recipe'], r['stage'], r['repeat']) == (recipe, stage, repeat)}
                        before, after = pair['baseline'], pair['candidate']
                        for key in ('device_name', 'request_order_sha256', 'prepared_input_sha256',
                                    'initial_effective_parameters_sha256'):
                            if before[key] != after[key]:
                                raise ValueError('paired benchmark identity differs: ' + key)
                        pairs.append(dict(recipe=recipe, stage=stage, repeat=repeat,
                            baseline_median_seconds=before['median_seconds'],
                            candidate_median_seconds=after['median_seconds'],
                            ratio_candidate_over_baseline=after['median_seconds'] / before['median_seconds']))
        summaries = []
        for recipe in args.recipes:
            for stage in args.stages:
                for label in sources:
                    rows = [r for r in records if (r['recipe'], r['stage'], r['source_label']) == (recipe, stage, label)]
                    summaries.append(dict(recipe=recipe, stage=stage, source_label=label,
                        repeat_medians_seconds=[r['median_seconds'] for r in rows],
                        median_of_repeat_medians_seconds=statistics.median(r['median_seconds'] for r in rows),
                        repeat_p90_seconds=[r['p90_seconds'] for r in rows],
                        repeat_peak_allocated_mib=[r['peak_allocated_mib'] for r in rows],
                        repeat_peak_reserved_mib=[r['peak_reserved_mib'] for r in rows]))
        report.update(status='passed', pairs=pairs, summaries=summaries)
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
    finally:
        write_json(destination, report)
    print(json.dumps(dict(status=report['status'], report=str(destination)), ensure_ascii=False))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    raise SystemExit(main())
