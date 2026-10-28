"""v39 完整 scratch batch 生命周期审计；不保存权重，不启动正式训练。

真实 CUDA 默认保留 batch16/1024 点/FA2/BF16；--synthetic --device cpu
可检查本地工程生命周期。同步分段计时是侵入式审计，不能当作正式 FPS。
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ''):
    sys.path.insert(0, str(ROOT))

from tools._batch_check import report_destination, gradient_summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfg', type=Path, default=ROOT / 'cfgs/ct_seqtrack/39_full_a_mini.yaml')
    parser.add_argument('--path', help='真实数据根；未指定时读取 cfg')
    parser.add_argument('--device', default='cuda:0', help='cpu、cuda 或可见设备内的 cuda:N')
    parser.add_argument('--output', type=Path, help='artifacts/ct_checks 下尚不存在的 JSON')
    parser.add_argument('--synthetic', action='store_true', help='合成 metadata/raw，不能代替真实 CUDA 验收')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--sampler-epoch', type=int, default=10, help='0-based 工程采样 epoch；10 覆盖成熟窗口')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--profile', action='store_true', help='额外保存 Chrome trace 与 CUDA kernel 摘要')
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or not 1 <= args.steps <= 8 or args.sampler_epoch < 0 or args.threads <= 0:
        parser.error('requires positive batch/threads, steps in 1..8, sampler-epoch >= 0')
    if not args.synthetic and args.batch_size != 16:
        parser.error('real-data acceptance requires batch_size=16')
    return args


def memory_summary(device):
    import torch
    if device.type != 'cuda':
        return None
    return dict(device_name=torch.cuda.get_device_name(device),
        peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2 ** 20,
        peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 2 ** 20,
        allocated_mib=torch.cuda.memory_allocated(device) / 2 ** 20,
        reserved_mib=torch.cuda.memory_reserved(device) / 2 ** 20)


class BackendAudit:
    """记录真正调用的 flash 函数，避免仅 last_backend 标记就声称命中内核。"""
    def __init__(self):
        self.calls = 0
        self.qkv = []

    def __enter__(self):
        import models.ct_v39.attention as attention
        self.module, self.original = attention, attention._flash_varlen
        def factory():
            implementation = self.original()
            def recorded(q, k, v, *args, **kwargs):
                self.calls += 1
                self.qkv.append(dict(q=list(q.shape), k=list(k.shape), v=list(v.shape),
                                     dtype=str(q.dtype), device=str(q.device),
                                     deterministic=bool(kwargs.get('deterministic'))))
                return implementation(q, k, v, *args, **kwargs)
            return recorded
        attention._flash_varlen = factory
        return self

    def __exit__(self, *args):
        self.module._flash_varlen = self.original


def audit_tensor_geometry(output, transaction, *, maximum_endpoints=4, tolerance=2e-4):
    """仅首批小样本回 CPU double 审计；不改变训练标签或任何在线决策。"""
    import numpy as np
    import torch
    from models.ct_v39.tensor_geometry import pair_box_metrics
    from utils.tracking_metrics import LocalYawBox, box_metrics, metric_contributions

    count = min(maximum_endpoints, len(transaction.rows))
    boxes = output.candidates.boxes[:count].detach()
    target = transaction.targets['target_box'][:count].detach()
    size = transaction.inputs['box_size'][:count].detach()
    true_size = transaction.targets.get('target_box_size', transaction.inputs['box_size'])[:count].detach()
    device_metrics = pair_box_metrics(boxes, target, size, true_size)
    values = {key: value.cpu().double().numpy() for key, value in device_metrics.items()}
    boxes_cpu, target_cpu, size_cpu, true_size_cpu = [
        value.cpu().double().numpy() for value in (boxes, target, size, true_size)]
    s_thresholds = torch.linspace(0., 1., 21, dtype=torch.float32).numpy()
    p_thresholds = torch.linspace(0., 2., 21, dtype=torch.float32).numpy()
    errors = dict(exact_iou=0., benchmark_iou=0., distance_m=0., success=0., precision=0.,
                  gpu_contribution_reproduction=0.)
    differences, failures = [], []
    for endpoint in range(count):
        truth = LocalYawBox(target_cpu[endpoint], true_size_cpu[endpoint][[1, 0, 2]])
        for candidate in range(boxes.shape[1]):
            estimate = LocalYawBox(boxes_cpu[endpoint, candidate], size_cpu[endpoint][[1, 0, 2]])
            exact, distance = box_metrics(estimate, truth, up_axis=(0, 0, 1), mode='geometry_exact')
            compat, compat_distance = box_metrics(estimate, truth, up_axis=(0, 0, 1), mode='benchmark_compat')
            success, precision = metric_contributions(compat, compat_distance)
            measured = {key: float(value[endpoint, candidate]) for key, value in values.items()}
            reference = dict(exact_iou=exact, benchmark_iou=compat, distance=distance,
                             success=float(success), precision=float(precision))
            if not all(math.isfinite(value) for value in measured.values()):
                failures.append(dict(endpoint=endpoint, candidate=candidate, reason='nonfinite device metric',
                    metrics={key: value if math.isfinite(value) else None for key, value in measured.items()}))
                continue
            for key, error_key in (('exact_iou', 'exact_iou'), ('benchmark_iou', 'benchmark_iou'),
                                   ('distance', 'distance_m'), ('success', 'success'), ('precision', 'precision')):
                error = abs(measured[key] - reference[key])
                errors[error_key] = max(errors[error_key], error)
                if not math.isfinite(error) or (key in ('exact_iou', 'benchmark_iou', 'distance') and error > tolerance):
                    failures.append(dict(endpoint=endpoint, candidate=candidate, metric=key,
                                         actual=measured[key], reference=reference[key], absolute_error=error))
            # 排除“恰好有一个边界差异，却掩盖整个 S/P 积分算错”的假通过。
            reproduced = metric_contributions(measured['benchmark_iou'], measured['distance'])
            reproduction_error = max(abs(measured['success'] - float(reproduced[0])),
                                     abs(measured['precision'] - float(reproduced[1])))
            errors['gpu_contribution_reproduction'] = max(errors['gpu_contribution_reproduction'], reproduction_error)
            if reproduction_error > 1e-6 or not math.isfinite(reproduction_error):
                failures.append(dict(endpoint=endpoint, candidate=candidate, metric='S/P integration',
                                     absolute_error=reproduction_error))
            for name, measured_value, reference_value, thresholds, direction in (
                ('success', measured['benchmark_iou'], compat, s_thresholds, 'ge'),
                ('precision', measured['distance'], distance, p_thresholds, 'le')):
                actual_value, expected_value = np.float32(measured_value), np.float32(reference_value)
                actual_bins = actual_value >= thresholds if direction == 'ge' else actual_value <= thresholds
                expected_bins = expected_value >= thresholds if direction == 'ge' else expected_value <= thresholds
                changed = np.flatnonzero(actual_bins != expected_bins)
                for index in changed:
                    threshold = float(thresholds[index])
                    nearest = min(abs(float(actual_value) - threshold), abs(float(expected_value) - threshold))
                    record = dict(endpoint=endpoint, candidate=candidate, metric=name,
                                  threshold_index=int(index), threshold=threshold,
                                  gpu_value=float(actual_value), cpu_value=float(expected_value),
                                  nearest_threshold_distance=nearest, allowed=bool(nearest <= tolerance))
                    differences.append(record)
                    if not record['allowed']:
                        failures.append(record)
                if abs(measured[name] - reference[name]) > 1e-6 and len(changed) == 0:
                    failures.append(dict(endpoint=endpoint, candidate=candidate, metric=name,
                                         reason='contribution differs without a threshold classification difference'))
    return dict(passed=not failures, endpoints=count, candidates=int(count * boxes.shape[1]),
                device=str(boxes.device), cpu_reference_dtype='float64',
                geometry_tolerance=tolerance, threshold_neighborhood=tolerance,
                max_absolute_errors=errors, threshold_difference_count=len(differences),
                threshold_differences=differences, failures=failures)


def audit_recovery_kernel(g, device):
    """强制覆盖非空恢复投票；固定特征不消耗 RNG，也不改训练状态。"""
    import torch
    device = torch.device(device)
    local = torch.tensor([[-1.2, -.4, -.2], [-.8, .5, .1], [-.2, -.6, .2],
                          [.3, .4, -.1], [.8, -.3, .3], [1.3, .6, 0.]], device=device)
    features = torch.eye(6, 64, device=device)
    centers = torch.tensor([[0., 0., 0.], [12., 0., 1.]], device=device)
    xyz = (local[None] + centers[:, None]).reshape(1, 12, 3)
    pool = dict(xyz=xyz, point_ids=torch.arange(12, device=device)[None],
                valid=torch.ones(1, 12, dtype=torch.bool, device=device))
    templates = dict(xyz=local[None, None].expand(1, 3, 6, 3),
                     valid=torch.ones(1, 3, 6, dtype=torch.bool, device=device),
                     ages=torch.zeros(1, 3, device=device))
    encoded_pool = dict(pool, features=features.repeat(2, 1)[None])
    encoded_templates = dict(templates, features=features[None, None].expand(1, 3, 6, 64))
    size = torch.tensor([[4., 2., 2.]], device=device)
    yaw = torch.zeros(1, device=device)
    with torch.no_grad():
        outputs = [g.coarse_proposals(pool, templates, size, yaw,
                   encoded_pool=encoded_pool, encoded_templates=encoded_templates) for _ in range(2)]
    repeated = all(torch.equal(outputs[0][key], outputs[1][key]) for key in ('boxes', 'valid', 'scores'))
    result = outputs[0]
    finite = bool(torch.isfinite(result['boxes']).all() & torch.isfinite(result['scores']).all())
    # 三份相同模板不能把每个峰的六个唯一 raw-ID 票重复计为十二票。
    unique_votes = bool(torch.equal(result['scores'], torch.full_like(result['scores'], 6.)))
    passed = finite and repeated and unique_votes and bool(result['valid'].all())
    if not passed:
        raise AssertionError('nonempty recovery vote audit failed: finite/repeat/raw-ID dedup/two peaks')
    return dict(passed=True, device=str(device), valid_peak_count=int(result['valid'].sum()),
                scores=result['scores'].cpu().tolist(), repeated_exact=repeated,
                raw_id_votes_deduplicated=unique_votes,
                strict_deterministic=torch.are_deterministic_algorithms_enabled())


def run_check(config, args, report, destination):
    import numpy as np
    import torch
    from models.ct_v39.data import build_loaders, stable_seed
    from models.ct_v39.raw_service import RawFrameService
    from models.ct_v39.entry import initialization_identity
    from models.ct_v39.attention import EfficientAttention
    from models.ctseqtrackv39 import CTSEQTRACKV39

    device = torch.device(args.device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('only CPU or CUDA is supported')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; automatic CPU fallback is forbidden')
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    torch.set_num_threads(args.threads)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(config.seed)

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    def timed(name, function, timing):
        sync()
        start_event = end_event = None
        if device.type == 'cuda':
            start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start_event.record()
        began = time.perf_counter()
        result = function()
        if end_event is not None:
            end_event.record()
        sync()
        timing[name] = dict(wall_seconds=time.perf_counter() - began,
                            cuda_elapsed_ms=None if start_event is None else start_event.elapsed_time(end_event))
        return result

    service = RawFrameService(config)
    host = None
    try:
        report['stage'] = 'build_loaders'
        sources = None
        if args.synthetic:
            from tests.helpers.raw_tracking import TinySource
            sources = dict(train=TinySource(tuple([16] * max(24, args.batch_size * 2))))
        loaders = build_loaders(config, roles=('train',), sources=sources, raw_service=service)
        loader = loaders['train']
        loader.batch_sampler.set_epoch(args.sampler_epoch)
        loader.generator.manual_seed(stable_seed(config.seed, 'train', args.sampler_epoch))
        report['sampler'] = loader.batch_sampler.state_dict()
        report['stage'] = 'construct_model'
        host = CTSEQTRACKV39(config, loaders=loaders).to(device).train()
        host.on_train_epoch_start()
        optimizer = host.configure_optimizers()['optimizer']
        if not isinstance(optimizer, torch.optim.Adam) or optimizer.state:
            raise RuntimeError('checker requires fresh registered Adam')
        report['initialization'] = initialization_identity(host.tracker)
        report['parameter_count'] = sum(p.numel() for p in host.parameters())
        report['parameter_dtypes'] = sorted({str(p.dtype) for p in host.parameters()})
        if report['parameter_dtypes'] != ['torch.float32']:
            raise RuntimeError('master parameters must remain FP32')
        report['stage'] = 'recovery_kernel_audit'
        report['recovery_kernel_audit'] = audit_recovery_kernel(host.tracker.g, device)
        report['batches'] = []
        iterator = iter(loader)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == 'cuda':
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        profiler_context = (torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True)
                            if args.profile else nullcontext())
        with BackendAudit() as backend, profiler_context as profiler:
            for step in range(args.steps):
                current = dict(step=step + 1, timing={})
                report['batches'].append(current)
                report['stage'] = f'step{step+1}:read_metadata'
                rows = timed('read_metadata', lambda: next(iterator), current['timing'])
                if len(rows) != config.batch_size:
                    raise ValueError('checker refuses reduced/drain batches; choose another engineering sampler epoch')
                current['requests'] = [dict(tracklet=r['tracklet_key'], branch=r['request'].branch,
                    frame=r['request'].frame, window_start=r['request'].window_start) for r in rows]
                optimizer.zero_grad(set_to_none=True)
                report['stage'] = f'step{step+1}:prepare'
                tx = timed('prepare', lambda: host.train_builder.begin(rows, device, training=True), current['timing'])
                report['stage'] = f'step{step+1}:forward'
                output = timed('forward', lambda: host.tracker.forward_transaction(tx, host.train_builder), current['timing'])
                if tuple(output.candidates.boxes.shape) != (len(rows), 7, 4):
                    raise ValueError('v39 requires exactly seven candidate slots')
                if not bool(output.full_executed.all()):
                    raise ValueError('training must execute Full for every endpoint even during cheap rollout')
                if output.accepted_box.dtype != torch.float32 or not bool(torch.isfinite(output.accepted_box).all()):
                    raise FloatingPointError('accepted geometry must be finite FP32')
                if tuple(tx.b0_inputs['points'].shape) != (len(rows), 4, 1024, 5):
                    raise ValueError('registered B0 batch shape differs')
                if any(c['state'].next_frame != c['row']['request'].frame for c in tx.contexts):
                    raise RuntimeError('forward mutated accepted state before optimizer/commit')
                report['stage'] = f'step{step+1}:losses'
                losses = timed('losses', lambda: host.tracker.compute_losses(tx, output), current['timing'])
                total = losses['loss_total']
                if total.ndim != 0 or not bool(torch.isfinite(total)):
                    raise FloatingPointError('loss_total must be finite scalar')
                current['losses'] = {k: float(v.detach()) for k, v in losses.items()
                                      if torch.is_tensor(v) and v.ndim == 0}
                if not all(math.isfinite(value) for value in current['losses'].values()):
                    raise FloatingPointError('nonfinite loss diagnostic')
                current['batch'] = dict(shape=list(tx.b0_inputs['points'].shape),
                    valid_points_per_frame=tx.b0_inputs['point_valid'].sum(-1).cpu().tolist(),
                    g_points=output.pool['valid'].sum(-1).cpu().tolist(),
                    eligible_candidates=output.candidates.valid.sum(-1).cpu().tolist(),
                    selected_index=output.selected_index.cpu().tolist(),
                    route_window=tx.inputs['route_window'].cpu().tolist(),
                    policy_full=output.policy_full.cpu().tolist(),
                    trusted_write=output.commit_decision.eligible.cpu().tolist(),
                    search_triggered=output.search_triggered.cpu().tolist())
                current['precision'] = dict(g_features=str(output.verification['encoded_pool']['features'].dtype),
                    g_logits=str(output.verification['logits'].dtype),
                    boxes=str(output.accepted_box.dtype), prior=str(output.prior.mean_xy.dtype))
                if current['precision']['g_logits'] != 'torch.float32':
                    raise RuntimeError('G logits must remain FP32')
                if config.v39_feature_bf16 and bool(output.pool['valid'].any()):
                    if current['precision']['g_features'] != 'torch.bfloat16':
                        raise RuntimeError('requested G feature BF16 was not exercised')
                if step == 0:
                    report['stage'] = 'step1:geometry_audit'
                    report['geometry_audit'] = timed('geometry_audit',
                        lambda: audit_tensor_geometry(output, tx), current['timing'])
                    if not report['geometry_audit']['passed']:
                        raise AssertionError('device geometry/S/P audit differs from CPU double reference')
                host._record_training(tx, output, losses)
                host._pending_train = (output, tx, len(rows))
                report['stage'] = f'step{step+1}:backward'
                timed('backward', total.backward, current['timing'])
                current['gradients'] = gradient_summary(host)
                current['module_gradients'] = {}
                for name in ('observation', 'decoder', 'prior', 'g', 'multistep'):
                    module = getattr(host.tracker, name)
                    gradients = [p.grad.detach() for p in module.parameters() if p.grad is not None]
                    current['module_gradients'][name] = dict(
                        parameters=sum(1 for _ in module.parameters()), tensors_with_gradient=len(gradients),
                        nonzero_tensors=sum(int(bool(g.any())) for g in gradients),
                        global_l2_norm=math.sqrt(sum(float(g.norm()) ** 2 for g in gradients)))
                report['stage'] = f'step{step+1}:adam'
                timed('adam', optimizer.step, current['timing'])
                if any(not bool(torch.isfinite(p).all()) for p in host.parameters()):
                    raise FloatingPointError('nonfinite parameter after Adam')
                optimizer_steps = sorted({int(s['step']) for s in optimizer.state.values() if 'step' in s})
                if max(optimizer_steps, default=0) != step + 1:
                    raise RuntimeError('fresh Adam update counter mismatch')
                current['optimizer_state_steps'] = optimizer_steps
                report['stage'] = f'step{step+1}:commit'
                timed('commit', lambda: host.on_train_batch_end(None, rows, step), current['timing'])
                if host._pending_train is not None or host.train_builder._pending is not None:
                    raise RuntimeError('commit left a pending transaction')
                if any(c['state'].next_frame != c['row']['request'].frame + 1 for c in tx.contexts):
                    raise RuntimeError('commit did not advance each accepted endpoint exactly once')
                current['state'] = dict(epoch_rows=host._epoch_rows, epoch_steps=host._epoch_steps,
                                        active_windows=len(host.train_builder.states),
                                        materialized_rows=host.train_builder.materialization_count)
                current['raw_cache'] = dict(service.stats)
                current['memory'] = memory_summary(device)
                current['model_stage_host_seconds'] = dict(tx.timings)
                del total, losses, tx, output
            report['attention'] = dict(actual_flash_function_calls=backend.calls, qkv_calls=backend.qkv,
                modules=[dict(name=name, backend=module.last_backend) for name, module in host.tracker.named_modules()
                         if isinstance(module, EfficientAttention)])
            if config.v39_attention_backend == 'fa2_varlen':
                if backend.calls == 0 or any(item['dtype'] != 'torch.bfloat16' for item in backend.qkv):
                    raise RuntimeError('FA2 actual nonempty call/BF16 evidence missing')
                if any(item['backend'] != 'fa2_varlen' for item in report['attention']['modules']):
                    raise RuntimeError('some attention modules did not use requested FA2')
            elif backend.calls:
                raise RuntimeError('math-reference check unexpectedly invoked FA2')
        if args.profile:
            trace = destination.with_suffix('.trace.json')
            if trace.exists():
                raise FileExistsError('refusing to overwrite trace: ' + str(trace))
            profiler.export_chrome_trace(str(trace))
            cuda_events = [event for event in profiler.events() if str(event.device_type).endswith('CUDA')]
            report['profiler'] = dict(trace=str(trace), cuda_kernel_count=len(cuda_events),
                flash_kernel_names=sorted({event.name for event in cuda_events if 'flash' in event.name.lower()}),
                cuda_kernel_total_us=sum(float(getattr(event, 'self_device_time_total',
                    getattr(event, 'self_cuda_time_total', 0.))) for event in cuda_events))
        report['post_update_identity'] = initialization_identity(host.tracker)
        if report['initialization']['initial_parameter_hash'] == report['post_update_identity']['initial_parameter_hash']:
            raise RuntimeError('optimizer lifecycle produced no parameter/buffer change')
        report['optimizer'] = dict(name='Adam', updates=args.steps,
            lr=optimizer.param_groups[0]['lr'], betas=list(optimizer.param_groups[0]['betas']),
            eps=optimizer.param_groups[0]['eps'], weight_decay=optimizer.param_groups[0]['weight_decay'])
        report.update(stage='complete', status='passed')
    finally:
        report['cuda_memory'] = memory_summary(device)
        if host is not None:
            host.close_raw_service()
        else:
            service.close()


def main(argv=None):
    args = parse_args(argv)
    destination = report_destination(args.output, version='39')
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = dict(schema='ctseqtrack-v39-full-batch-check-v1', status='failed', stage='load_config',
        started_utc=datetime.now(timezone.utc).isoformat(), device=args.device, synthetic=args.synthetic,
        measurement_kind='intrusive_lifecycle_check_not_throughput_benchmark',
        checkpoint_saved=False, formal_training_requires_new_scratch_process=True,
        sampler_epoch=args.sampler_epoch, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    began = time.perf_counter()
    try:
        os.environ.setdefault('OMP_NUM_THREADS', str(args.threads))
        os.environ.setdefault('MKL_NUM_THREADS', str(args.threads))
        from models.ct_v39.config import load_config, config_identity
        from models.ct_v39.entry import configure_numerics, source_identity
        original = load_config(args.cfg)
        report['registered_config_sha256'] = config_identity(original)
        overrides = dict(ct_engineering_check=True, log_dir=str(destination.parent), v31_evaluate_late3=False,
            batch_size=args.batch_size, accelerator='cpu' if args.device == 'cpu' else 'gpu')
        if args.device == 'cpu':
            overrides.update(v39_attention_backend='math_reference', v39_feature_bf16=False,
                             execution_profile='optimized_fp32')
        if args.path is not None:
            overrides['path'] = args.path
        config = load_config(args.cfg, overrides)
        if config.test or config.checkpoint or config.init_checkpoint:
            raise ValueError('full-batch check is scratch engineering only')
        report.update(config=dict(config), config_sha256=config_identity(config), source=source_identity())
        configure_numerics()
        import torch
        report.update(torch=torch.__version__, python=sys.version)
        run_check(config, args, report, destination)
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        report['wall_seconds'] = time.perf_counter() - began
        with destination.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps(dict(status=report['status'], stage=report['stage'], report=str(destination)), ensure_ascii=False))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
