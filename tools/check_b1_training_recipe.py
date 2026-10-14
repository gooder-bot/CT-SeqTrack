"""新 B1 配方真实 batch 的短 CUDA 生命周期检查；工程权重丢弃，不保存 checkpoint。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ''):
    sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfg', required=True, type=Path)
    parser.add_argument('--path', help='真实 nuScenes 数据根，不是 SDK 目录')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--course-epoch', type=int, default=10, help='零基课程阶段；权重仍从头初始化')
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--output', type=Path, help='新 artifacts/ct_checks 子目录中的 report.json')
    args = parser.parse_args(argv)
    if args.course_epoch < 0 or args.steps < 1:
        parser.error('course-epoch must be nonnegative and steps must be positive')
    return args


def memory_snapshot(device):
    import torch
    if device != 'cuda' or not torch.cuda.is_available():
        return None
    return dict(device_name=torch.cuda.get_device_name(),
        allocated_mib=torch.cuda.memory_allocated() / 2 ** 20,
        reserved_mib=torch.cuda.memory_reserved() / 2 ** 20,
        peak_allocated_mib=torch.cuda.max_memory_allocated() / 2 ** 20,
        peak_reserved_mib=torch.cuda.max_memory_reserved() / 2 ** 20)


def run_steps(config, device, report, *, course_epoch=10, steps=2, loaders=None):
    """原 host 的 prepare/forward/loss/backward/Adam/commit；注入 loader 仅供 CPU 测试。"""
    import torch
    from models.ct_v37.data import build_loaders, stable_seed
    from models.ctseqtrackv37 import CTSEQTRACKV37
    from tools.check_execution_performance import finite_training, seed_all
    from tools.check_v35_batch import gradient_summary
    from tools.execution_lab_runtime import course_host
    from utils.b1_training_recipes import RECIPE_KEY, recipe_options, recipe_scope

    if config.net_model != 'ctseqtrackv37' or config.v37_arm != 'b1':
        raise ValueError('this check requires a v37 B1 recipe; B2/Full is not a substitute')
    if config.precision != 32 or config.execution_profile != 'optimized_fp32':
        raise ValueError('B1 recipe check requires FP32 execution')
    if not config.ct_engineering_check or config.test or config.checkpoint or config.init_checkpoint:
        raise ValueError('B1 recipe check requires engineering scratch with no checkpoint')
    if course_epoch < 0 or steps < 1:
        raise ValueError('invalid course epoch or step count')
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        torch.cuda.reset_peak_memory_stats()

    def sync():
        if device == 'cuda':
            torch.cuda.synchronize()

    def stage(name, packet, action):
        report['stage'] = name
        sync()
        started = time.perf_counter()
        result = action()
        sync()
        packet['timing_seconds'][name] = time.perf_counter() - started
        packet['memory'][name] = memory_snapshot(device)
        return result

    report.update(recipe=config.get(RECIPE_KEY), options=list(recipe_options(config)),
        execution_profile=config.execution_profile, precision=config.precision,
        requested_batch_size=config.batch_size, requested_point_sample_size=config.point_sample_size,
        course_epoch=course_epoch, optimizer_initialization='scratch_at_step_zero',
        course_is_not_trained_weight_age=True, gradient_accumulation=1,
        injected_test_loader=loaders is not None, requested_steps=steps, steps=[],
        timing_kind='synchronized_stage_diagnostic_not_throughput_benchmark',
        checkpoint_saved=False, engineering_weights_discarded=True)
    model = iterator = None
    with recipe_scope(config):
        try:
            report['stage'] = 'build_loaders'
            seed_all(config.seed)
            loaders = build_loaders(config, roles=('train',)) if loaders is None else loaders
            loader = loaders['train']
            loader.batch_sampler.set_epoch(course_epoch)
            loader.generator.manual_seed(stable_seed(config.seed, 'train', course_epoch))
            report['sampler'] = loader.batch_sampler.state_dict()
            report['stage'] = 'build_model'
            model = course_host(CTSEQTRACKV37, course_epoch)(config, loaders=loaders).to(device).train()
            model.on_train_epoch_start()
            optimizer = model.configure_optimizers()['optimizer']
            if not isinstance(optimizer, torch.optim.Adam) or optimizer.state:
                raise RuntimeError('expected a fresh Adam optimizer')
            iterator = iter(loader)
            total_rows = 0
            for index in range(steps):
                packet = dict(step=index + 1, timing_seconds={}, memory={})
                report['steps'].append(packet)
                rows = stage('read_batch', packet, lambda: next(iterator))
                packet['actual_batch_size'] = len(rows)
                if len(rows) != config.batch_size:
                    raise ValueError(f'refusing a reduced-batch check: {len(rows)} != {config.batch_size}')
                if any(row['request'].epoch != course_epoch for row in rows):
                    raise ValueError('sampler requests do not match requested course epoch')
                packet['depths'] = [row['request'].frame - row['request'].window_start for row in rows]
                packet['branches'] = [row['request'].branch for row in rows]
                packet['running_bn_drain'] = any(getattr(row['request'], 'drain', False) for row in rows)
                optimizer.zero_grad(set_to_none=True)
                batch, output = stage('prepare_forward', packet,
                    lambda: model._forward_raw(rows, model.train_builder, training=True))
                packet['points_shape'] = list(batch['points'].shape)
                packet['points_dtype'] = str(batch['points'].dtype)
                if tuple(batch['points'].shape) != (config.batch_size, 4, config.point_sample_size, 5):
                    raise ValueError('prepared batch shape differs from the requested recipe')
                if batch['points'].dtype != torch.float32:
                    raise ValueError('prepared points must remain FP32')
                packet['valid_points_per_frame'] = batch['point_valid'].sum(-1).cpu().tolist()
                if model.training_diagnostics is not None:
                    output.loss_statistics = {}
                losses = stage('loss', packet, lambda: model.tracker.compute_losses(batch, output))
                total = losses['loss_total']
                if total.ndim != 0:
                    raise FloatingPointError('loss_total must be a finite scalar')
                finite_training(model, losses, output)
                scalar_losses = {name: float(value.detach()) for name, value in losses.items()
                                 if torch.is_tensor(value) and value.ndim == 0}
                if not all(math.isfinite(value) for value in scalar_losses.values()):
                    raise FloatingPointError('nonfinite loss/log')
                packet['losses'] = scalar_losses
                if model.training_diagnostics is not None:
                    stage('diagnostics', packet,
                        lambda: model.training_diagnostics.add_batch(rows, batch, output, losses))
                model._pending_train = (output, batch, len(rows))
                stage('backward', packet, total.backward)
                packet['gradients'] = gradient_summary(model)
                finite_training(model, losses, output)
                stage('adam', packet, optimizer.step)
                finite_training(model, losses, output)
                state_steps = sorted({int(state['step'].item()) for state in optimizer.state.values()
                                      if 'step' in state})
                if not state_steps or max(state_steps) != index + 1 or min(state_steps) < 1:
                    raise RuntimeError('Adam state does not match one update per consumed batch')
                for state in optimizer.state.values():
                    for key, value in state.items():
                        if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
                            raise FloatingPointError('nonfinite Adam state: ' + key)
                packet['adam_state_steps'] = state_steps
                stage('commit', packet, lambda: model.on_train_batch_end(None, rows, index))
                total_rows += len(rows)
                if model._pending_train is not None or model.train_builder._pending is not None:
                    raise RuntimeError('pending transaction after commit')
                if model._epoch_steps != index + 1 or model._epoch_rows != total_rows:
                    raise RuntimeError('host transaction counters differ from consumed rows/Adam steps')
                packet['commit'] = dict(epoch_rows=model._epoch_rows, epoch_steps=model._epoch_steps,
                    active_windows=len(model.train_builder.states), pending_host=False, pending_builder=False)
                # 不保留前一步图，否则两步检查会人为叠加前向激活峰值。
                del total, losses, batch, output, rows
            report['completed_steps'] = model._epoch_steps
            report['completed_rows'] = model._epoch_rows
            report['max_observed_depth'] = max(depth for packet in report['steps'] for depth in packet['depths'])
            report['cuda_memory'] = memory_snapshot(device)
            report.update(status='passed', stage='complete')
        finally:
            if model is not None:
                model.clear_pending_materialization()
                model._pending_train = None
                model.train_builder.reset()
            if iterator is not None:
                del iterator


def main(argv=None):
    from tools.check_v35_batch import report_destination
    args = parse_args(argv)
    destination = report_destination(args.output, version='37_b1_recipe')
    if destination.parent.exists():
        raise FileExistsError('B1 recipe check requires a new report directory: ' + str(destination.parent))
    destination.parent.mkdir(parents=True, exist_ok=False)
    report = dict(schema='ct_seqtrack.b1_recipe_check.v1', status='failed', stage='load_config',
        device=args.device, checkpoint_saved=False, engineering_weights_discarded=True,
        started_utc=datetime.now(timezone.utc).isoformat(),
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    started = time.perf_counter()
    try:
        from models.ct_v37.config import load_config, config_identity
        from models.ct_v37.entry import configure_numerics, source_identity
        overrides = dict(ct_engineering_check=True, workers=0, log_dir=None,
            checkpoint=None, init_checkpoint=None, test=False, v31_evaluate_late3=False,
            accelerator='gpu' if args.device == 'cuda' else 'cpu')
        if args.path is not None:
            overrides['path'] = args.path
        config = load_config(args.cfg, overrides)
        report.update(config=dict(config), config_sha256=config_identity(config), source=source_identity(),
            purpose='scratch_lifecycle_capacity_check_not_speed_or_accuracy_evidence')
        configure_numerics()
        import torch
        report.update(torch=torch.__version__, python=sys.version)
        run_steps(config, args.device, report, course_epoch=args.course_epoch, steps=args.steps)
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
    finally:
        report['wall_seconds'] = time.perf_counter() - started
        try:
            report['cuda_memory'] = memory_snapshot(args.device)
        except Exception as error:
            report['cuda_memory_error'] = str(error)
        # 保留原失败，即使第三方异常对象或NaN使详细报告无法严格序列化。
        from tools.check_execution_performance import encode_report
        content, status = encode_report(report)
        with destination.open('x', encoding='utf-8') as stream:
            stream.write(content)
    print(json.dumps(dict(status=status, stage=report['stage'], report=str(destination)), ensure_ascii=False))
    return 0 if status == 'passed' else 1


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    raise SystemExit(main())
