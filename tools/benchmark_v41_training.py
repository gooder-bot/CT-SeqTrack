"""同一张 CUDA 卡上，对两个独立 V41 源码目录配对测量真实训练前缀。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager
from datetime import datetime
from functools import wraps
import hashlib
from importlib import import_module
from itertools import islice
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from unittest.mock import patch


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
WARMUP = 20
STEPS = 200


class WindowComplete(Exception):
    """训练循环已完成当前步日志、优化器与主辅提交，工程进程在此丢弃模型。"""


def write_report(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def workload_digest(sampler, start, count):
    # 在计时窗口外从同一 epoch 的不可变计划计算，保留 batch 边界和所有端点字段。
    digest = hashlib.sha256()
    rows = 0
    batches = 0
    for batch in islice(iter(sampler), start, start + count):
        values = [(r.epoch, r.tracklet, r.branch, r.window_start, r.window_end, r.frame, r.drain)
                  for r in batch]
        digest.update(json.dumps(values, separators=(',', ':')).encode() + b'\n')
        rows += len(values)
        batches += 1
    if batches != count:
        raise RuntimeError('requested benchmark window exceeds the sampler plan')
    return digest.hexdigest(), rows


def counts(model):
    return (int(model._epoch_steps), int(model._epoch_rows),
            int(model._auxiliary_batches), int(model._auxiliary_rows))


def align_initialization(tracker, path, *, reference):
    """只交换工程随机初值及构造后的 RNG；在 Lightning 建立优化器前调用。"""
    import torch
    from models.ct_v41.runtime import capture_rng_state, restore_rng_state
    state = tracker.state_dict()
    parameters = dict(tracker.named_parameters())
    owners = {}
    for name, value in state.items():
        parent = tracker.get_submodule(name.rpartition('.')[0])
        parameter = parameters.get(name)
        owners[name] = (type(parent).__module__, type(parent).__qualname__, str(value.dtype),
                        tuple(value.shape), None if parameter is None else parameter.requires_grad)
    if reference and not path.exists():
        with path.open('xb') as stream:
            torch.save(dict(state=state, owners=owners, rng=capture_rng_state()), stream)
    bundle = torch.load(path, map_location='cpu', weights_only=False)
    original = bundle['state']
    removed = sorted(original.keys() - state.keys())
    if (state.keys() - original.keys() or (reference and removed)
            or any(not key.startswith('prior.acquisition_head.') for key in removed)):
        raise ValueError('unrecognized change to random initialization parameter/buffer inventory')
    for name in state:
        if owners[name] != bundle['owners'][name]:
            raise ValueError('changed owner, shape, dtype, buffer/parameter or trainability: ' + name)
    tracker.load_state_dict({name: original[name] for name in state}, strict=True)
    digest = hashlib.sha256()
    for name, value in sorted(tracker.state_dict().items()):
        if not name.startswith('prior.acquisition_head.'):
            digest.update(json.dumps((name, owners[name]), separators=(',', ':')).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    restore_rng_state(bundle['rng'])
    return dict(shared_initialization_sha256=digest.hexdigest(),
                reference_initialization_bundle_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                reference_only_parameters=removed)


def child_cpu_seconds():
    """Linux 活跃子进程（含主、辅助 DataLoader workers）的累计 CPU 时间。"""
    if not sys.platform.startswith('linux'):
        return {}
    import multiprocessing
    ticks = os.sysconf('SC_CLK_TCK')
    workers = {}
    for process in multiprocessing.active_children():
        try:
            fields = Path(f'/proc/{process.pid}/stat').read_text().rsplit(')', 1)[1].split()
            # 去掉 comm 后，fields[0] 对应 stat 第 3 列，utime/stime 为 14/15。
            workers[process.pid] = (int(fields[11]) + int(fields[12])) / ticks
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return workers


def instrument(prof, target, method, label):
    original = getattr(target, method)

    @wraps(original)
    def measured(*args, **kwargs):
        name = label(*args) if callable(label) else label
        with prof.range(name):
            return original(*args, **kwargs)

    prof.patches.enter_context(patch.object(target, method, measured))


class Trace:
    """吞吐窗口结束后才安装范围、启动归因跟踪。"""

    def __init__(self, torch, mode, destination):
        self.torch = torch
        self.mode = mode
        self.destination = destination
        self.patches = ExitStack()
        self.profiler = None
        self.loader_stats = defaultdict(Counter)
        self.views = Counter()
        self.queues = set()

    @contextmanager
    def range(self, name):
        if self.mode == 'nvtx':
            with self.torch.cuda.nvtx.range('ct/' + name):
                yield
        else:
            with self.torch.profiler.record_function('ct/' + name):
                yield

    def start(self, model):
        import models.ctseqtrackv41 as host
        from datasets import kitti_v41
        from torch.utils.data.dataloader import _BaseDataLoaderIter
        if self.mode == 'torch':
            api = self.torch.profiler
            self.profiler = api.profile(activities=[api.ProfilerActivity.CPU, api.ProfilerActivity.CUDA])
            self.profiler.start()
        read_next = _BaseDataLoaderIter.__next__
        select_view = kitti_v41.normal_train_point_view

        def load(iterator):
            role = iterator._dataset.source.v41_data_role
            stats = self.loader_stats[role]
            queue = getattr(iterator, '_data_queue', None)
            if hasattr(queue, '_recv_bytes') and id(queue) not in self.queues:
                recv = queue._recv_bytes

                def receive(*args, **kwargs):
                    message = recv(*args, **kwargs)
                    stats['queue_messages'] += 1
                    stats['queue_payload_bytes'] += len(message)
                    return message
                self.patches.enter_context(patch.object(queue, '_recv_bytes', receive))
                self.queues.add(id(queue))
            with self.range('load_' + role):
                rows = read_next(iterator)
            stats['batches_delivered'] += 1
            sizes = point_payload(rows)
            stats.update(sizes)
            return rows

        def view(*args, **kwargs):
            result = select_view(*args, **kwargs)
            self.views['gt10_hit' if result['cache_used'] else 'raw_fallback'] += 1
            return result
        self.patches.enter_context(patch.object(_BaseDataLoaderIter, '__next__', load))
        self.patches.enter_context(patch.object(kitti_v41, 'normal_train_point_view', view))
        raw_ref = getattr(kitti_v41, 'KITTIRawFrame', None)
        if raw_ref is not None:
            instrument(self, raw_ref, 'arrays', 'parent_raw_materialization')
            instrument(self, raw_ref, '_read', 'parent_raw_file_read')
        for role, builder in [('main', model.train_builder), ('aux', model.auxiliary_builder)]:
            for name in ('prepare', 'supervise_prior', 'commit'):
                if hasattr(builder, name):
                    instrument(self, builder, name, role + '/' + name)
        instrument(self, host, 'move_tensors', 'prepared_to_device')
        tracker = model.tracker
        for name in ('forward', 'forward_aux', 'compute_losses', 'compute_aux_losses'):
            instrument(self, tracker, name, name)
        for name in ('prior', 'observation', 'decoder', 'verifier', 'recovery_head', 'multistep'):
            module = getattr(tracker, name, None)
            if module is not None:
                instrument(self, module, 'forward', name)
        instrument(self, model, 'on_before_optimizer_step', 'gradient_checks')
        # Lightning 将 training_step + backward 放进 Adam closure；区分整个 closure
        # 和真正执行更新的 tensor kernels（Profiler 中的 aten/Adam events）。
        instrument(self, model, 'backward', 'backward')
        instrument(self, model, 'on_train_batch_end', 'post_optimizer_commit')
        adam = import_module('torch.optim.adam')
        for implementation in ('_single_tensor_adam', '_multi_tensor_adam', '_fused_adam'):
            if hasattr(adam, implementation):
                instrument(self, adam, implementation, 'adam_update')

    def stop(self):
        self.patches.close()
        result = dict(parent_loader_traffic=dict(self.loader_stats), parent_view_calls=dict(self.views))
        if self.profiler is None:
            return result
        self.profiler.stop()
        self.profiler.export_chrome_trace(str(self.destination / 'trace.json'))
        result['stages'] = {item.key: dict(calls=item.count, inclusive_cpu_ms=item.cpu_time_total / 1000,
                               attributed_cuda_ms=item.cuda_time_total / 1000)
                for item in self.profiler.key_averages() if item.key.startswith('ct/')}
        gpu_events = [item for item in self.profiler.events() if item.device_type.name == 'CUDA']
        result['device_events'] = gpu_activity(gpu_events)
        return result


def gpu_activity(events):
    """以 CUDA 时间轴合并重叠事件，避免把多流并发时长重复算作忙碌时间。"""
    spans = sorted((e.time_range.start, e.time_range.end) for e in events)
    if not spans:
        # CUDA 训练必然执行设备工作；缺少 Kineto 设备事件不能解释为设备零耗时。
        return None
    merged = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    names = [e.name.lower() for e in events]
    return dict(events=len(spans), union_cuda_ms=sum(e - s for s, e in merged) / 1000,
                h2d_copies=sum('memcpy htod' in name or 'memcpy h2d' in name for name in names),
                d2h_copies=sum('memcpy dtoh' in name or 'memcpy d2h' in name for name in names))


def point_payload(rows):
    """统计实际交付的整云和 GT10 NumPy 对象；同一批内相同对象只计一次。"""
    seen = set()
    total = Counter(raw_array_bytes=0, gt10_array_bytes=0)

    def add(group, array):
        if hasattr(array, 'nbytes') and id(array) not in seen:
            total[group] += array.nbytes
            seen.add(id(array))

    for row in rows:
        for frame in [*row['frames'].values(), row.get('first_frame', {})]:
            pc = frame.get('pc')
            add('raw_array_bytes', getattr(pc, 'points', pc))
            add('raw_array_bytes', getattr(pc, 'point_ids', frame.get('point_ids')))
            for array in frame.get('_v41_normal_train_cache', {}).values():
                add('gt10_array_bytes', array)
    return total


def host_interval_summary(intervals):
    ordered = sorted(intervals)
    if not ordered:
        return dict(steps=0, p50_ms=None, p95_ms=None)
    return dict(steps=len(ordered), p50_ms=1000 * statistics.median(ordered),
                p95_ms=1000 * ordered[math.ceil(.95 * len(ordered)) - 1])


class Window:
    def __init__(self, args, torch):
        self.args = args
        self.torch = torch
        self.before = None
        self.step_intervals = []
        self.report = {}
        self.trace = None

    def after_advance(self, trainer):
        torch = self.torch
        model = trainer.lightning_module
        completed = int(model._epoch_steps)
        if self.before is None:
            start = WARMUP
            tracing = bool(self.report)
            if tracing:
                # 尽量覆盖一次真实的稀疏插入，保持 main/aux 的原始插入时机。
                start = next((i for i in model._auxiliary_insertions if i >= WARMUP + STEPS), None)
                if start is None:
                    raise RuntimeError('no scheduled auxiliary batch after throughput window for the trace')
            size = self.args.trace_steps if tracing else STEPS
            if start + size >= model._main_batches:
                raise RuntimeError('benchmark requires a whole window inside epoch 1')
            if completed != start:
                return
            self.window_start, self.size = start, size
            if tracing:
                self.trace = Trace(torch, self.args.trace_mode, self.args.destination)
                self.trace.start(model)
            torch.cuda.synchronize()
            self.before = counts(model)
            self.optimizer_start = int(trainer.global_step)
            self.workers_start = child_cpu_seconds()
            torch.cuda.reset_peak_memory_stats()
            self.cpu_start = time.process_time()
            self.start = self.previous = time.perf_counter()
            return
        now = time.perf_counter()
        self.step_intervals.append(now - self.previous)
        self.previous = now
        if completed != self.window_start + self.size:
            return
        torch.cuda.synchronize()
        wall = time.perf_counter() - self.start
        cpu = time.process_time() - self.cpu_start
        workers_end = child_cpu_seconds()
        diagnostics = self.trace.stop() if self.trace else {}
        self.trace = None
        steps, main_rows, aux_steps, aux_rows = (end - begin for end, begin in zip(counts(model), self.before))
        optimizer_steps = int(trainer.global_step) - self.optimizer_start
        if steps != optimizer_steps or steps != self.size or len(self.step_intervals) != steps:
            raise RuntimeError('real Adam/committed step counts differ from the requested window')
        first = self.window_start
        indices = [i for i in model._auxiliary_insertions if first <= i < completed]
        main_plan = model._loaders['train'].batch_sampler
        aux_plan = model._loaders['auxiliary'].batch_sampler
        main_digest, planned_main_rows = workload_digest(main_plan, first, steps)
        aux_digest, planned_aux_rows = workload_digest(aux_plan, self.before[2], aux_steps)
        if (planned_main_rows, planned_aux_rows, len(indices)) != (main_rows, aux_rows, aux_steps):
            raise RuntimeError('observed commit counts differ from the same epoch sampler plan')
        all_steps = host_interval_summary(self.step_intervals)
        inserted = set(indices)
        intervals = {'main_only': [], 'main_with_auxiliary': []}
        for index, duration in enumerate(self.step_intervals, start=first):
            intervals['main_with_auxiliary' if index in inserted else 'main_only'].append(duration)
        result = dict(warmup_steps=first, measured_steps=steps, optimizer_steps=optimizer_steps,
            main_endpoints=main_rows, auxiliary_endpoints=aux_rows, auxiliary_steps=aux_steps,
            wall_seconds=wall, main_endpoints_per_second=main_rows / wall,
            total_endpoints_per_second=(main_rows + aux_rows) / wall, steps_per_second=steps / wall,
            host_step_interval_p50_ms=all_steps['p50_ms'], host_step_interval_p95_ms=all_steps['p95_ms'],
            host_step_intervals_by_role={role: host_interval_summary(values) for role, values in intervals.items()},
            final_device_drain_ms=1000 * (wall - sum(self.step_intervals)),
            main_process_cpu_seconds=cpu, child_processes_observed=len(self.workers_start),
            child_cpu_seconds=(sum(workers_end[p] - value for p, value in self.workers_start.items())
                if self.workers_start and self.workers_start.keys() == workers_end.keys() else None),
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
            training_source_sha256=main_plan.source_sha256, main_window_sha256=main_digest,
            auxiliary_window_sha256=aux_digest, auxiliary_step_indices=indices, trace_diagnostics=diagnostics)
        if self.report:
            self.report['attribution_window'] = result
        else:
            self.report = result
            if self.args.trace_mode:
                self.before = None
                self.step_intervals.clear()
                # 下一次真实插入可以恰好是紧邻吞吐窗口的下一步。
                self.after_advance(trainer)
                return
        raise WindowComplete()


@contextmanager
def checkout_run_directory(root, destination):
    """每个版本仅接受本树 artifacts 下的工程 log_dir，最后统一归档。"""
    allowed = (root / 'artifacts/ct_checks').resolve()
    archive = destination / 'run'
    if allowed in archive.resolve().parents:
        yield archive
        return
    allowed.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='v41-paired-', dir=allowed) as temporary:
        try:
            yield Path(temporary)
        finally:
            shutil.copytree(temporary, archive)


def measured_fit(args, config, window, command, *, loaders=None):
    """CPU 接口回放和服务器 CUDA 工程测量共用实际 Trainer 的安装/结束路径。"""
    from models.ct_v41 import entry
    import pytorch_lightning as pl
    from models.ct_v41.epoch_runtime import EpochCheckpoint
    from models.ctseqtrackv41 import CTSEQTRACKV41
    original_trainer = pl.Trainer
    original_model_init = CTSEQTRACKV41.__init__
    initialization = {}
    hooks = ExitStack()

    @wraps(original_model_init)
    def construct(model, *a, **kw):
        original_model_init(model, *a, **kw)
        initialization.update(align_initialization(model.tracker, args.initial_state, reference=args.reference))
        if args.equivalence == 'only':
            window.install(hooks, model)

    class MeasuredTrainer(original_trainer):
        def __init__(self, *positional, **settings):
            settings['callbacks'] = [c for c in settings['callbacks'] if not isinstance(c, EpochCheckpoint)]
            settings['enable_checkpointing'] = False
            super().__init__(*positional, **settings)
            loop = self.fit_loop.epoch_loop
            original = loop.on_advance_end

            @wraps(original)
            def after_step(*a, **kw):
                result = original(*a, **kw)
                window.after_advance(self)
                return result
            loop.on_advance_end = after_step

    try:
        with hooks, patch.object(pl, 'Trainer', MeasuredTrainer), patch.object(CTSEQTRACKV41, '__init__', construct):
            entry.run(config, loaders=loaders, command_line=command)
    except WindowComplete:
        pass
    finally:
        if window.trace is not None:
            window.trace.stop()
    if not window.report or (args.trace_mode and 'attribution_window' not in window.report):
        raise RuntimeError('training ended before the requested window')
    return initialization


def worker(args):
    root = args.source_root.resolve()
    os.chdir(root)
    sys.path.insert(0, str(root))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.devices)
    from models.ct_v41 import entry
    if Path(entry.__file__).resolve().parents[2] != root:
        raise RuntimeError('loaded a different repository than the requested source root')
    from models.ct_v41.identity import EXECUTION_REVISION
    expected_revision = 'v2' if args.reference else 'v3'
    if EXECUTION_REVISION != expected_revision:
        raise ValueError(f'expected execution revision {expected_revision}, found {EXECUTION_REVISION} in {root}')
    import torch
    import pytorch_lightning as pl
    if args.equivalence == 'only':
        from v41_equivalence_probe import EquivalenceProbe
        window = EquivalenceProbe(args, WindowComplete)
    else:
        window = Window(args, torch)
    with checkout_run_directory(root, args.destination) as run:
        command = ['--cfg', str(root / args.cfg), '--ct_engineering_check', '--devices', str(args.devices),
                   '--accelerator', 'gpu', '--path', str(args.path), '--log_dir', str(run),
                   '--diagnostics_level', args.diagnostics_level, '--full_diagnostic_epochs']
        if args.workers is not None:
            command.extend(['--workers', str(args.workers)])
        config = entry.parse_config(command)
        initialization = measured_fit(args, config, window, [str(SCRIPT), *command])
    manifest = json.loads((args.destination / 'run/run_manifest.json').read_text(encoding='utf-8'))
    controls = ('epoch', 'lr', 'wd', 'lr_schedule', 'lr_warmup_steps', 'lr_stage_values', 'lr_milestones',
                'lr_decay_step', 'lr_decay_rate', 'dataset', 'category_name', 'point_sample_size',
                'dynamics_time_mode', 'v41_arm', 'v41_attention_backend', 'v41_aux_fraction',
                'v41_aux_loss_weight', 'log_every_n_steps', 'tensorboard', 'v41_memory_logging')
    window.report.update(source_root=str(root), source_sha256=manifest['source']['sha256'],
        execution_revision=EXECUTION_REVISION,
        git_head=manifest['git_head'], tool_sha256=hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
        device=torch.cuda.get_device_name(), physical_device=args.devices,
        torch_version=str(torch.__version__), lightning_version=pl.__version__, seed=config.seed,
        batch_size=config.batch_size, workers=config.workers, recipe=config.v41_training_recipe,
        diagnostics_level=config.diagnostics_level, trace_mode=args.trace_mode, dataset_root=str(args.path),
        controlled_recipe={name: config.get(name) for name in controls})
    window.report.update(initialization)
    write_report(args.destination / 'measurement.json', window.report)
    return window.report


PAIR_FIELDS = ('training_source_sha256', 'main_window_sha256', 'auxiliary_window_sha256',
               'auxiliary_step_indices', 'main_endpoints', 'auxiliary_endpoints', 'warmup_steps',
               'measured_steps', 'batch_size', 'workers', 'recipe', 'seed', 'tool_sha256', 'device',
               'physical_device', 'torch_version', 'lightning_version', 'diagnostics_level', 'dataset_root', 'controlled_recipe',
               'shared_initialization_sha256', 'reference_initialization_bundle_sha256')


def paired_result(reference, candidate):
    mismatch = [key for key in PAIR_FIELDS if reference[key] != candidate[key]]
    if mismatch:
        raise ValueError('incomparable windows: ' + ', '.join(mismatch))
    for key in ('warmup_steps', 'measured_steps', 'main_window_sha256', 'auxiliary_window_sha256',
                'auxiliary_step_indices'):
        if reference.get('attribution_window', {}).get(key) != candidate.get('attribution_window', {}).get(key):
            raise ValueError('incomparable attribution windows: ' + key)
    return candidate['main_endpoints_per_second'] / reference['main_endpoints_per_second']


def run_pair(args, destination, initial_state, index, *, equivalence=False):
    measurements = {}
    for label in (('reference', 'candidate') if index % 2 == 1 else ('candidate', 'reference')):
        root = args.reference_root if label == 'reference' else args.source_root
        work = destination / (f'equivalence-{label}' if equivalence else f'{label}-{index:02d}')
        work.mkdir()
        command = [sys.executable, str(SCRIPT), '--worker', '--source-root', str(root),
                   '--path', str(args.path), '--devices', str(args.devices), '--destination', str(work),
                   '--cfg', args.cfg, '--diagnostics-level', args.diagnostics_level,
                   '--initial-state', str(initial_state)]
        if label == 'reference':
            command.append('--reference')
        if args.workers is not None:
            command.extend(['--workers', str(args.workers)])
        if equivalence:
            command.extend(['--equivalence', 'only'])
        elif args.trace_mode:
            command.extend(['--trace-mode', args.trace_mode, '--trace-steps', str(args.trace_steps)])
        subprocess.run(command, cwd=root, check=True)
        measurements[label] = json.loads((work / 'measurement.json').read_text(encoding='utf-8'))
    if equivalence:
        reference, candidate = measurements['reference'], measurements['candidate']
        fields = ('first_aux_step', 'continued_auxiliary_endpoints', 'steps_compared', 'training_source_sha256', 'shared_initialization_sha256',
                  'reference_initialization_bundle_sha256', 'controlled_recipe', 'seed', 'batch_size', 'workers')
        if any(reference[key] != candidate[key] for key in fields) or candidate['status'] != 'passed':
            raise ValueError('equivalence steps, inputs or shared initialization differ')
        return dict(status='passed', steps_compared=candidate['steps_compared'],
                    continued_auxiliary_endpoints=candidate['continued_auxiliary_endpoints'],
                    checked_tensors=candidate['checked_tensors'], rtol=candidate['rtol'], atol=candidate['atol'])
    return paired_result(**measurements)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--source-root', type=Path, default=ROOT, help='candidate checkout')
    parser.add_argument('--reference-root', type=Path, help='original V41 v2 checkout')
    parser.add_argument('--cfg', default='cfgs/ct_seqtrack/41_full_kitti.yaml', help='relative to each checkout')
    parser.add_argument('--path', type=Path, required=True, help='shared real KITTI training root')
    parser.add_argument('--devices', type=int, required=True, help='one physical CUDA GPU index')
    parser.add_argument('--workers', type=int, help='common override, otherwise use each recipe default')
    parser.add_argument('--diagnostics-level', choices=('summary', 'full'), default='summary')
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--trace-mode', choices=('torch', 'nvtx'), help='short attribution window after throughput')
    parser.add_argument('--trace-steps', type=int, default=4)
    parser.add_argument('--equivalence', choices=('off', 'after', 'only'), default='off',
                        help='short real-data corrected-loss comparison after both measurements or on its own')
    parser.add_argument('--output-dir', type=Path, help='new child of this tool checkout artifacts/ct_checks')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--destination', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--initial-state', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--reference', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.devices < 0 or args.repeats < 1 or args.trace_steps < 1 or (args.workers or 0) < 0:
        parser.error('invalid device, repeat count, trace length or worker count')
    if Path(args.cfg).is_absolute() or '..' in Path(args.cfg).parts:
        parser.error('--cfg must be relative to both checkouts')
    args.path = args.path.expanduser().resolve()
    args.source_root = args.source_root.expanduser().resolve()
    allowed = (ROOT / 'artifacts/ct_checks').resolve()
    if args.worker:
        if args.destination is None or allowed not in args.destination.resolve().parents:
            parser.error('worker output must be inside artifacts/ct_checks')
        if args.initial_state is None or args.destination.resolve().parent not in args.initial_state.resolve().parents:
            parser.error('initial state must belong to this comparison directory')
        return worker(args)
    if args.reference_root is None:
        parser.error('paired measurement requires --reference-root')
    args.reference_root = args.reference_root.expanduser().resolve()
    destination = (args.output_dir or allowed / ('v41-perf-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))).resolve()
    if allowed not in destination.parents:
        parser.error('--output-dir must be a new child of artifacts/ct_checks')
    destination.mkdir(parents=True, exist_ok=False)
    with TemporaryDirectory(prefix='scratch-init-', dir=destination) as temporary:
        initial_state = Path(temporary) / 'initialization.pt'
        result = {}
        if args.equivalence != 'only':
            ratios = [run_pair(args, destination, initial_state, n) for n in range(1, args.repeats + 1)]
            result.update(schema='ct_v41.paired_training_prefix.v1', paired_speedup=ratios,
                          median_speedup=statistics.median(ratios), repeats=len(ratios),
                          definition='candidate / reference main committed endpoints per wall second')
            write_report(destination / 'comparison.json', result)
        if args.equivalence != 'off':
            result['equivalence'] = run_pair(args, destination, initial_state, 1, equivalence=True)
            write_report(destination / 'equivalence.json', result['equivalence'])
    print(json.dumps(dict(output_dir=str(destination), **result), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
