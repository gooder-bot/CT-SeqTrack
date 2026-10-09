"""真实 Trainer 工程测速：日志与诊断保留，不保存或迁移训练权重。"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from functools import wraps
import csv
import hashlib
import importlib
import json
import os
import platform
from pathlib import Path
import statistics
import subprocess
import sys
import time


def _percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _json_default(value):
    if hasattr(value, 'item'):
        return value.item()
    raise TypeError('unsupported benchmark record: ' + type(value).__name__)


def _tensor_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps((name, str(value.dtype), list(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def _logging_snapshot(directory):
    """核对真实落盘日志的字段、频率、step和值；不以手动host替代logger。"""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    snapshot = dict(csv={}, tensorboard={}, training_diagnostics={})
    for path in sorted(directory.glob('csv/**/metrics.csv')):
        with path.open(encoding='utf-8', newline='') as stream:
            for row in csv.DictReader(stream):
                for key, value in row.items():
                    if value not in ('', None):
                        snapshot['csv'].setdefault(key, []).append([row.get('step'), value])
    for folder in sorted({path.parent for path in directory.glob('tensorboard/**/events.out.tfevents.*')}):
        events = EventAccumulator(str(folder), size_guidance={'scalars': 0})
        events.Reload()
        for tag in sorted(events.Tags().get('scalars', [])):
            snapshot['tensorboard'][tag] = [[event.step, event.value] for event in events.Scalars(tag)]
    for path in sorted((directory / 'training_diagnostics').glob('*.json')):
        snapshot['training_diagnostics'][path.name] = json.loads(path.read_text(encoding='utf-8'))
    if not snapshot['csv'] or not snapshot['tensorboard'] or not snapshot['training_diagnostics']:
        raise RuntimeError('benchmark requires original CSV, TensorBoard and epoch diagnostics to be written')
    checksum = hashlib.sha256(json.dumps(snapshot, sort_keys=True, separators=(',', ':'),
                                        allow_nan=False).encode()).hexdigest()
    path = directory / 'logger_snapshot.json'
    _write_json(path, snapshot)
    return dict(snapshot_path=str(path), values_sha256=checksum,
        csv_field_counts={key: len(value) for key, value in snapshot['csv'].items()},
        tensorboard_field_counts={key: len(value) for key, value in snapshot['tensorboard'].items()},
        diagnostic_files=sorted(snapshot['training_diagnostics']))


class _WindowTimer:
    """仅窗口边界同步；step CPU 时间不是异步 GPU 的真实延迟。"""
    def __init__(self, torch, device, warmup, steps):
        self.torch, self.device = torch, device
        self.warmup, self.steps = warmup, steps
        self.count = self.measured_rows = 0
        self.cpu_enqueue = []
        self.event_pairs = []
        self.requests = hashlib.sha256()
        self.started = self.elapsed = self.cpu_started = None
        self.previous_event = None
        self.last_rows = 0

    @property
    def cuda(self):
        return self.device.type == 'cuda'

    def begin(self):
        if self.started is not None:
            return
        if self.cuda:
            self.torch.cuda.synchronize(self.device)
            self.previous_event = self.torch.cuda.Event(enable_timing=True)
            self.previous_event.record()
        self.started = time.perf_counter()

    def before_batch(self, rows):
        # 保存顺序摘要，避免把大量原始点/状态写入性能报告。
        request_rows = [dict(tracklet=row['tracklet_key'],
            request=asdict(row['request']) if is_dataclass(row['request']) else vars(row['request']))
            for row in rows]
        self.requests.update(json.dumps(request_rows, sort_keys=True, separators=(',', ':'),
                                       default=_json_default).encode())
        self.requests.update(b'\n')
        self.last_rows = len(rows)
        self.cpu_started = time.perf_counter()

    def after_commit(self):
        self.count += 1
        if self.count <= self.warmup:
            if self.count == self.warmup:
                # 起点位于前一事务完成后，下一batch读取包括在窗口内。
                self.begin()
            return
        self.cpu_enqueue.append(time.perf_counter() - self.cpu_started)
        self.measured_rows += self.last_rows
        if self.cuda:
            event = self.torch.cuda.Event(enable_timing=True)
            event.record()
            self.event_pairs.append((self.previous_event, event))
            self.previous_event = event
        if self.count == self.warmup + self.steps:
            if self.cuda:
                self.torch.cuda.synchronize(self.device)
            self.elapsed = time.perf_counter() - self.started

    def report(self):
        if self.count != self.warmup + self.steps or self.elapsed is None:
            raise RuntimeError('benchmark did not complete the requested warmup/measured updates')
        gpu_ms = [start.elapsed_time(end) for start, end in self.event_pairs]
        return dict(completed_steps=self.count, warmup_steps=self.warmup, measured_steps=self.steps,
            measured_rows=self.measured_rows, endpoint_order_sha256=self.requests.hexdigest(),
            window_seconds=self.elapsed, endpoints_per_second=self.measured_rows / self.elapsed,
            cpu_enqueue_step_seconds=self.cpu_enqueue,
            cpu_enqueue_step_median_seconds=_percentile(self.cpu_enqueue, .5),
            cpu_enqueue_step_p90_seconds=_percentile(self.cpu_enqueue, .9),
            cpu_enqueue_note='batch_start through host commit; excludes next data read; not GPU latency',
            cuda_event_step_ms=gpu_ms, cuda_event_step_median_ms=_percentile(gpu_ms, .5),
            cuda_event_step_p90_ms=_percentile(gpu_ms, .9),
            cuda_event_note='consecutive commit-boundary events include GPU idle gaps from CPU/data work; '
                            'synchronized only at window boundaries',
            window_note='warmup commit to final measured commit, including subsequent data reads and final sync')


def _install_trace_ranges(model, trainer):
    """短 trace 使用；包装调用不改变参数、返回值或事务顺序。"""
    import torch
    restorations = []

    def wrap(target, name, label):
        original = getattr(target, name)
        @wraps(original)
        def recorded(*args, **kwargs):
            with torch.profiler.record_function(label):
                return original(*args, **kwargs)
        setattr(target, name, recorded)
        restorations.append((target, name, original))

    for name in ('prepare', 'acquire', 'commit'):
        wrap(model.train_builder, name, name)
    wrap(model.tracker, 'plan_prior', 'prior')
    wrap(model.tracker, 'forward', 'forward')
    wrap(model.tracker, 'compute_losses', 'loss')
    wrap(model, 'backward', 'backward')
    for optimizer in trainer.optimizers:
        wrap(optimizer, 'step', 'Adam')
    if model.training_diagnostics is not None:
        wrap(model.training_diagnostics, 'add_batch', 'diagnostics')
    return restorations


def _run_one(base_config, profile, device, directory, *, warmup, steps, trace=False):
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import Callback, LearningRateMonitor
    from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
    from utils.execution_profile import execution_identity

    version = base_config['net_model']
    from tools.check_execution_performance import model_version
    implementation = model_version(base_config)
    package = 'models.ct_v' + implementation
    config_module = importlib.import_module(package + '.config')
    entry = importlib.import_module(package + '.entry')
    host_module = importlib.import_module('models.ctseqtrackv' + implementation)
    host_type = getattr(host_module, 'CTSEQTRACKV' + implementation)
    directory.mkdir(parents=True, exist_ok=False)
    config = config_module.normalize_config(dict(base_config, execution_profile=profile,
        ct_engineering_check=True, log_dir=str(directory), checkpoint=None, init_checkpoint=None,
        test=False, limit_train_batches=warmup + steps, limit_val_batches=0,
        v31_evaluate_late3=False, accelerator='gpu' if device.type == 'cuda' else 'cpu'))
    pl.seed_everything(config.seed, workers=True)
    model = host_type(config)
    initial_state = _tensor_digest(model.state_dict())
    source = entry.source_identity()
    identity = execution_identity(config, source)
    timer = _WindowTimer(torch, device, warmup, steps)
    loggers = [CSVLogger(str(directory), name='csv'), TensorBoardLogger(str(directory), name='tensorboard')]
    restorations = []
    console_messages = []
    display_version = version.removeprefix('ctseqtrack')
    original_end = model.on_train_batch_end

    @wraps(original_end)
    def committed_end(*args, **kwargs):
        result = original_end(*args, **kwargs)
        timer.after_commit()
        return result
    model.on_train_batch_end = committed_end

    class TimingCallback(Callback):
        def on_train_start(self, trainer, module):
            if trace:
                restorations.extend(_install_trace_ranges(module, trainer))
            if warmup == 0:
                timer.begin()

        def on_train_batch_start(self, trainer, module, rows, batch_idx):
            timer.before_batch(rows)

        def on_exception(self, trainer, module, exception):
            module.clear_pending_materialization()

    class ConsoleProgress(Callback):
        # 与正式entry相同频率和取标量方式；测速不能少做这些同步/输出。
        @staticmethod
        def write(message):
            print(message, flush=True)
            console_messages.append(message)

        def on_train_batch_start(self, trainer, module, batch, batch_idx):
            if config.lr_warmup_steps and batch_idx % 50 == 0:
                lr = trainer.optimizers[0].param_groups[0]['lr']
                self.write(f'[{display_version} lr] update={trainer.global_step + 1} lr={lr:.9g}')

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            if batch_idx % 50 == 0:
                loss = outputs.get('loss') if isinstance(outputs, dict) else outputs
                value = float(loss.detach()) if torch.is_tensor(loss) else loss
                self.write(f'[{display_version} train] epoch={trainer.current_epoch + 1}/{trainer.max_epochs} '
                    f'batch={batch_idx + 1}/{trainer.num_training_batches} '
                    f'step={trainer.global_step} loss={value}')

        def on_validation_end(self, trainer, module):
            self.write('[' + display_version + ' validation] epoch=' + str(trainer.current_epoch + 1) + ' ' +
                       json.dumps(module.evaluation_results, ensure_ascii=False))

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    trainer = pl.Trainer(default_root_dir=str(directory), max_epochs=1,
        accelerator=config.accelerator,
        devices=[device.index] if device.type == 'cuda' and device.index is not None else 1,
        precision=32, deterministic=True, benchmark=False, logger=loggers,
        callbacks=[TimingCallback(),
            LearningRateMonitor(logging_interval='step' if config.lr_warmup_steps else 'epoch'),
            ConsoleProgress()], enable_checkpointing=False,
        reload_dataloaders_every_n_epochs=1, check_val_every_n_epoch=config.check_val_every_n_epoch,
        num_sanity_val_steps=0, accumulate_grad_batches=1, gradient_clip_val=0., log_every_n_steps=10,
        limit_train_batches=warmup + steps, limit_val_batches=0,
        enable_progress_bar=sys.stdout.isatty(), enable_model_summary=False)
    status = 'failed'
    loggers_finalized = False
    report = dict(profile=profile, execution_identity=identity,
        config_sha256=config_module.config_identity(config), source=source,
        initial_state_sha256=initial_state, checkpoint_saved=False, trace_only=trace,
        benchmark_epoch_index=0,
        workload_note='epoch0 single-step curriculum only; mature epoch>=10 workload not measured',
        loggers=['CSVLogger', 'TensorBoardLogger'],
        callbacks=['LearningRateMonitor', 'ConsoleProgress'],
        progress_bar_enabled=sys.stdout.isatty(),
        training_diagnostics_enabled=model.training_diagnostics is not None)
    _write_json(directory / 'execution_manifest.json', dict(report, config=dict(config)))
    try:
        if trace:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if device.type == 'cuda':
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as profiler:
                trainer.fit(model)
            profiler.export_chrome_trace(str(directory / 'trace.json'))
            report['trace_path'] = str(directory / 'trace.json')
            report['trace_note'] = 'separate instrumented run; overlapping Adam closure/backward ranges are not additive'
        else:
            trainer.fit(model)
        report.update(timer.report())
        if trainer.global_step != warmup + steps or model._epoch_steps != warmup + steps:
            raise RuntimeError('Trainer/host update counts differ from the requested benchmark')
        for logger in loggers:
            logger.finalize('success')
            if isinstance(logger, TensorBoardLogger):
                logger.experiment.close()
        loggers_finalized = True
        report['logging'] = _logging_snapshot(directory)
        report['logging'].update(console_message_count=len(console_messages),
            console_messages_sha256=hashlib.sha256(json.dumps(console_messages,
                ensure_ascii=False, separators=(',', ':')).encode()).hexdigest())
        report['console_messages'] = console_messages
        report.update(final_state_sha256=_tensor_digest(model.state_dict()),
            optimizer_steps=int(trainer.global_step), status='passed')
        if device.type == 'cuda':
            report['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
            report['cuda_peak_reserved_bytes'] = torch.cuda.max_memory_reserved(device)
        status = 'success'
        return report
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        model.clear_pending_materialization()
        model.on_train_batch_end = original_end
        for target, name, original in reversed(restorations):
            setattr(target, name, original)
        if not loggers_finalized:
            for logger in loggers:
                logger.finalize(status)
                if isinstance(logger, TensorBoardLogger):
                    logger.experiment.close()
        _write_json(directory / 'benchmark.json', report)


def _run_inference_one(base_config, profile, device, directory, *, warmup, steps):
    """真实闭环评测事务，保留候选/逐帧诊断；不加载或保存正式权重。"""
    import torch
    from tools.check_execution_performance import module_set, seed_all
    from utils.execution_profile import execution_identity
    api = module_set(base_config)
    config = api['config'].normalize_config(dict(base_config, execution_profile=profile,
        ct_engineering_check=True, log_dir=None, checkpoint=None, init_checkpoint=None, test=False))
    directory.mkdir(parents=True, exist_ok=False)
    seed_all(config.seed)
    loaders = api['data'].build_loaders(config, roles=('val',))
    model = api['host'](config).to(device).eval()
    model.evaluation_builder.reset()
    model.evaluation = api['runtime'].TrackingEvaluation(config)
    model.evaluation.add_singleton_tracks(loaders['val'].dataset)
    timer = _WindowTimer(torch, device, warmup, steps)
    report = dict(profile=profile, status='failed', initial_state_sha256=_tensor_digest(model.state_dict()),
        execution_identity=execution_identity(config), config_sha256=api['config'].config_identity(config),
        purpose='paired scratch-initialized closed-loop inference workload, not trained-model accuracy',
        checkpoint_saved=False, diagnostics_retained=True)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    try:
        if warmup == 0:
            timer.begin()
        iterator = iter(loaders['val'])
        with torch.no_grad():
            for _ in range(warmup + steps):
                try:
                    rows = next(iterator)
                except StopIteration as error:
                    raise RuntimeError('validation loader is shorter than inference warmup + steps') from error
                timer.before_batch(rows)
                model._evaluation_step(rows)
                timer.after_commit()
        report.update(timer.report())
        report['metrics'] = model.evaluation.summary()
        # 落盘发生在计时窗口之后；模型原有逐帧记录计算仍在窗口内。
        with (directory / 'frames.jsonl').open('x', encoding='utf-8') as stream:
            for row in model.evaluation.rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False, default=_json_default) + '\n')
        if device.type == 'cuda':
            report['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
            report['cuda_peak_reserved_bytes'] = torch.cuda.max_memory_reserved(device)
        report['status'] = 'passed'
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        model.clear_pending_materialization()
        _write_json(directory / 'inference.json', report)
    return report


def _paired_inference(config, device, destination, *, profiles, warmup, steps, repeats):
    runs = []
    for repeat in range(repeats):
        for profile in profiles if repeat % 2 == 0 else tuple(reversed(profiles)):
            run = _run_inference_one(config, profile, device,
                destination / f'repeat{repeat + 1}-{profile}', warmup=warmup, steps=steps)
            run['repeat'] = repeat + 1
            runs.append(run)
    for field in ('endpoint_order_sha256', 'initial_state_sha256', 'completed_steps', 'measured_rows'):
        if len({run[field] for run in runs}) != 1:
            raise RuntimeError('paired inference differs in ' + field)
    summary = {}
    for profile in profiles:
        group = [run for run in runs if run['profile'] == profile]
        summary[profile] = dict(
            median_endpoints_per_second=statistics.median(run['endpoints_per_second'] for run in group),
            median_window_seconds=statistics.median(run['window_seconds'] for run in group),
            cpu_enqueue_step_median_seconds=_percentile([v for r in group for v in r['cpu_enqueue_step_seconds']], .5),
            cpu_enqueue_step_p90_seconds=_percentile([v for r in group for v in r['cpu_enqueue_step_seconds']], .9),
            cuda_event_step_median_ms=_percentile([v for r in group for v in r['cuda_event_step_ms']], .5),
            cuda_event_step_p90_ms=_percentile([v for r in group for v in r['cuda_event_step_ms']], .9))
    ratios = []
    for repeat in range(1, repeats + 1):
        pair = {r['profile']: r for r in runs if r['repeat'] == repeat}
        ratios.append(pair[profiles[0]]['window_seconds'] / pair[profiles[1]]['window_seconds'])
    return dict(status='passed', runs=runs, summary=summary, paired_data_and_initialization_match=True,
        speedup=dict(baseline=profiles[0], candidate=profiles[1], paired_ratios=ratios,
                     median_ratio=statistics.median(ratios), min_ratio=min(ratios), max_ratio=max(ratios)),
        note='warmup state retained; BF16 may diverge in decisions; same raw endpoint schedule, not rerolled oracle')


def run_benchmark(config, device, destination: Path, *, warmup=20, steps=100, repeats=3, trace_steps=5,
                  baseline_profile=None, inference=False):
    """独立新目录内比较相同更新序列；不执行正式训练或自动提升执行配方。"""
    for name, value, minimum in (('warmup', warmup, 0), ('steps', steps, 1),
                                 ('repeats', repeats, 1), ('trace_steps', trace_steps, 0)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(name + ' must be an integer >= ' + str(minimum))
    from tools.check_execution_performance import model_version, paired_baseline
    implementation = model_version(config)
    if not config.get('ct_engineering_check') or config.get('checkpoint') or config.get('init_checkpoint'):
        raise ValueError('benchmark requires engineering scratch with no checkpoint')
    if config.get('test'):
        raise ValueError('benchmark requires the training lifecycle, not --test')
    try:
        import pytorch_lightning as pl
    except ModuleNotFoundError as error:
        if error.name == 'pytorch_lightning':
            raise RuntimeError('execution benchmark requires pytorch-lightning and the registered server environment') from error
        raise
    import torch
    from utils.execution_profile import effective_execution_profile
    package = 'models.ct_v' + implementation
    importlib.import_module(package + '.entry').configure_numerics()
    device = torch.device(device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('execution benchmark requires CPU or CUDA')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA benchmark requested but CUDA is unavailable')
    candidate = ('optimized_bf16' if effective_execution_profile(config) == 'optimized_bf16'
                 else 'optimized_fp32')
    baseline_profile = paired_baseline(config, candidate, baseline_profile)
    if candidate == 'optimized_bf16' and (device.type != 'cuda' or not torch.cuda.is_bf16_supported()):
        raise RuntimeError('optimized_bf16 benchmark requires BF16-capable CUDA')
    destination = Path(destination).resolve()
    allowed = Path(__file__).resolve().parents[1] / 'artifacts' / 'ct_checks'
    if allowed.resolve() not in destination.parents:
        raise ValueError('benchmark destination must be a new directory under artifacts/ct_checks')
    destination.mkdir(parents=True, exist_ok=False)
    environment = dict(python=platform.python_version(), platform=platform.platform(),
        torch=torch.__version__, lightning=pl.__version__, device=str(device), cuda=torch.version.cuda,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        concurrency_note='one benchmark run at a time; unrelated server jobs are not controlled')
    if device.type == 'cuda':
        environment['gpu'] = torch.cuda.get_device_name(device)
        try:
            snapshot = subprocess.run(['nvidia-smi',
                '--query-compute-apps=pid,process_name,used_gpu_memory,gpu_uuid', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=5, check=False,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            environment['gpu_process_snapshot'] = dict(returncode=snapshot.returncode,
                stdout=snapshot.stdout[:8192], stderr=snapshot.stderr[:2048])
        except (OSError, subprocess.TimeoutExpired) as error:
            environment['gpu_process_snapshot'] = dict(status='unavailable', reason=str(error))
    result = dict(schema='ct_seqtrack.execution_benchmark.v1', status='failed', environment=environment,
        candidate_profile=candidate, baseline_profile=baseline_profile,
        warmup=warmup, steps=steps, repeats=repeats,
        benchmark_epoch_index=0,
        workload_note='epoch0 single-step curriculum throughput only; mature epoch>=10 workload not measured',
        checkpoint_saved=False, formal_run=False, correctness_verified=False,
        correctness_note='paired correctness check is separate; timing never promotes a profile automatically',
        runs=[], traces=[])
    profiles = (baseline_profile, candidate)
    try:
        for repeat in range(repeats):
            for profile in profiles if repeat % 2 == 0 else tuple(reversed(profiles)):
                run = _run_one(config, profile, device, destination / f'repeat{repeat + 1}-{profile}',
                               warmup=warmup, steps=steps)
                run['repeat'] = repeat + 1
                result['runs'].append(run)
        for field in ('endpoint_order_sha256', 'initial_state_sha256', 'completed_steps', 'measured_rows'):
            if len({run[field] for run in result['runs']}) != 1:
                raise RuntimeError('benchmark paired runs differ in ' + field)
        result['paired_data_and_initialization_match'] = True
        # BF16允许记录数值差异；字段/频率必须仍与legacy相同。
        for field in ('csv_field_counts', 'tensorboard_field_counts', 'diagnostic_files', 'console_message_count'):
            serialized = {json.dumps(run['logging'][field], sort_keys=True) for run in result['runs']}
            if len(serialized) != 1:
                raise RuntimeError('benchmark log field/frequency mismatch: ' + field)
        values_match = len({run['logging']['values_sha256'] for run in result['runs']}) == 1
        console_match = len({run['logging']['console_messages_sha256'] for run in result['runs']}) == 1
        result['logging_equivalence'] = dict(fields_and_frequencies_match=True,
            original_metric_and_epoch_diagnostic_values_match=values_match,
            console_messages_match=console_match,
            excludes='TensorBoard wall_time and runtime metadata; original metric/diagnostic values retained')
        if candidate == 'optimized_fp32' and not values_match:
            raise RuntimeError('optimized_fp32 changed original Trainer logs or epoch diagnostic values')
        if candidate == 'optimized_fp32' and not console_match:
            raise RuntimeError('optimized_fp32 changed original ConsoleProgress messages')
        result['summary'] = {}
        for profile in profiles:
            runs = [run for run in result['runs'] if run['profile'] == profile]
            result['summary'][profile] = dict(
                median_window_seconds=statistics.median(run['window_seconds'] for run in runs),
                median_endpoints_per_second=statistics.median(run['endpoints_per_second'] for run in runs),
                median_window_average_step_seconds=statistics.median(run['window_seconds'] / steps for run in runs),
                cpu_enqueue_step_median_seconds=_percentile(
                    [value for run in runs for value in run['cpu_enqueue_step_seconds']], .5),
                cpu_enqueue_step_p90_seconds=_percentile(
                    [value for run in runs for value in run['cpu_enqueue_step_seconds']], .9),
                cuda_event_step_median_ms=_percentile(
                    [value for run in runs for value in run['cuda_event_step_ms']], .5),
                cuda_event_step_p90_ms=_percentile(
                    [value for run in runs for value in run['cuda_event_step_ms']], .9))
        paired_ratios = []
        for repeat in range(1, repeats + 1):
            pair = [run for run in result['runs'] if run['repeat'] == repeat]
            by_profile = {run['profile']: run for run in pair}
            baseline_seconds = by_profile[baseline_profile]['window_seconds']
            candidate_seconds = by_profile[candidate]['window_seconds']
            paired_ratios.append(dict(repeat=repeat, execution_order=[run['profile'] for run in pair],
                baseline_window_seconds=baseline_seconds, candidate_window_seconds=candidate_seconds,
                speedup_ratio=baseline_seconds / candidate_seconds))
            if baseline_profile == 'legacy':
                paired_ratios[-1]['legacy_window_seconds'] = baseline_seconds
        ratios = [pair['speedup_ratio'] for pair in paired_ratios]
        result['speedup'] = dict(ratio_definition=baseline_profile + ' window_seconds / candidate window_seconds',
            ratio_of_median_window_times=result['summary'][baseline_profile]['median_window_seconds'] /
                result['summary'][candidate]['median_window_seconds'],
            paired_repeats=paired_ratios, median_ratio=statistics.median(ratios),
            min_ratio=min(ratios), max_ratio=max(ratios),
            ratio_population_stddev=statistics.pstdev(ratios),
            ratio_coefficient_of_variation=statistics.pstdev(ratios) / statistics.mean(ratios),
            repeats_faster_than_baseline=sum(value > 1. for value in ratios),
            performance_pass_threshold=None, automatically_promoted=False)
        if baseline_profile == 'legacy':
            result['speedup']['repeats_faster_than_legacy'] = sum(value > 1. for value in ratios)
        for profile in profiles if trace_steps else ():
            result['traces'].append(_run_one(config, profile, device, destination / ('trace-' + profile),
                warmup=0, steps=trace_steps, trace=True))
        if result['traces'] and len({run['endpoint_order_sha256'] for run in result['traces']}) != 1:
            raise RuntimeError('profiler paired runs differ in endpoint order')
        if inference:
            result['inference'] = _paired_inference(config, device, destination / 'inference',
                profiles=profiles, warmup=warmup, steps=steps, repeats=repeats)
        result['status'] = 'passed'
    except BaseException as error:
        result.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        _write_json(destination / 'report.json', result)
    return result
