"""一次真实 B0 batch 的 forward/backward/Adam/commit 工程检查；不保存权重。"""
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
import uuid

ROOT = Path(__file__).resolve().parents[1]
CHECK_ROOT = ROOT / 'artifacts' / 'ct_checks'
if __package__ in (None, ''):
    sys.path.insert(0, str(ROOT))


def parse_args(argv=None, *, default_cfg=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfg', type=Path, default=default_cfg or ROOT / 'cfgs/ct_seqtrack/33_b0_mini.yaml')
    parser.add_argument('--path', help='真实数据根；未提供时使用配置中的 path')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--output', type=Path, help='artifacts/ct_checks 下尚不存在的 JSON')
    return parser.parse_args(argv)


def report_destination(output, *, version=33):
    if output is None:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        output = CHECK_ROOT / f'v{version}_batch_{stamp}_{os.getpid()}_{uuid.uuid4().hex[:12]}' / 'report.json'
    output = Path(output).resolve()
    if CHECK_ROOT.resolve() not in output.parents or output.suffix.lower() != '.json':
        raise ValueError('batch-check JSON must be under artifacts/ct_checks')
    if output.exists():
        raise FileExistsError('refusing to overwrite an existing batch-check report: ' + str(output))
    return output


def gradient_summary(model):
    """逐参数归约，避免为了记录全局范数而拼接一份完整梯度副本。"""
    import torch
    total_norm, tensors, nonzero, missing = 0., 0, 0, []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing.append(name)
            continue
        gradient = parameter.grad.detach()
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError('nonfinite gradient: ' + name)
        norm = float(torch.linalg.vector_norm(gradient))
        total_norm = math.hypot(total_norm, norm)
        tensors += 1
        nonzero += int(norm > 0)
    if not tensors or not math.isfinite(total_norm) or total_norm <= 0:
        raise FloatingPointError('batch must produce finite, nonzero trainable gradients')
    return dict(all_finite=True, global_l2_norm=total_norm, tensors_with_gradient=tensors,
                tensors_with_nonzero_gradient=nonzero, parameters_without_gradient=missing)


def cuda_memory(device):
    import torch
    if device != 'cuda' or not torch.cuda.is_available():
        return None
    return dict(device_name=torch.cuda.get_device_name(),
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 2 ** 20,
                peak_reserved_mib=torch.cuda.max_memory_reserved() / 2 ** 20)


def run_one_batch(config, device, report, *, loaders=None):
    """loaders 仅供合成生命周期测试注入；CLI 始终使用真实 build_loaders。"""
    import torch
    from models.ct_v31.data import build_loaders, stable_seed
    from models.ctseqtrackv31 import CTSEQTRACKV31

    if (config.net_model not in ('ctseqtrackv33', 'ctseqtrackv34') or config.v31_arm != 'b0'
            or config.batch_size != 16 or config.point_sample_size != 1024 or config.precision != 32):
        raise ValueError('this check requires v33/v34 B0, batch_size=16, point_sample_size=1024, FP32')
    if not config.ct_engineering_check or config.test or config.checkpoint or config.init_checkpoint:
        raise ValueError('one-batch check must use engineering scratch training with no checkpoint')
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is unavailable')
        torch.cuda.reset_peak_memory_stats()

    report['stage'] = 'build_loaders'
    loaders = build_loaders(config, roles=('train',)) if loaders is None else loaders
    loader = loaders['train']
    loader.batch_sampler.set_epoch(0)
    loader.generator.manual_seed(stable_seed(config.seed, 'train', 0))
    report['sampler'] = loader.batch_sampler.state_dict()
    model = CTSEQTRACKV31(config, loaders=loaders).to(device).train()
    model.on_train_epoch_start()
    optimization = model.configure_optimizers()
    optimizer = optimization['optimizer']
    if not isinstance(optimizer, torch.optim.Adam) or optimizer.state:
        raise RuntimeError('expected a fresh registered Adam optimizer')
    optimizer.zero_grad(set_to_none=True)

    report['stage'] = 'read_first_batch'
    iterator = iter(loader)
    try:
        rows = next(iterator)
        if len(rows) != 16:
            raise ValueError(f'first batch has {len(rows)} rows; refusing a reduced-batch check')
        report['requests'] = [dict(tracklet=row['tracklet_key'], branch=row['request'].branch,
                                   frame=row['request'].frame) for row in rows]
        report['stage'] = 'forward'
        batch, output = model._forward_raw(rows, model.train_builder, training=True)
        if tuple(batch['points'].shape) != (16, 4, 1024, 5):
            raise ValueError('prepared batch does not have the registered [16,4,1024,5] shape')
        if batch['points'].dtype != torch.float32:
            raise ValueError('prepared batch is not FP32')
        if not bool(torch.isfinite(output.accepted_box).all()):
            raise FloatingPointError('nonfinite accepted boxes')
        losses = model.tracker.compute_losses(batch, output)
        total = losses['loss_total']
        if total.ndim != 0 or not bool(torch.isfinite(total)):
            raise FloatingPointError('loss_total must be a finite scalar')
        report['losses'] = {}
        for name, value in losses.items():
            if torch.is_tensor(value) and value.ndim == 0:
                number = float(value.detach())
                if not math.isfinite(number):
                    raise FloatingPointError('nonfinite loss/log: ' + name)
                report['losses'][name] = number
        report['batch'] = dict(shape=list(batch['points'].shape), dtype=str(batch['points'].dtype),
            valid_points_per_frame=batch['point_valid'].sum(-1).cpu().tolist(),
            current_gt_foreground_counts=((batch['segmentation_labels'][:, -1] > 0)
                & batch['point_valid'][:, -1]).sum(-1).cpu().tolist(),
            extension_points=batch['extension_valid'].sum(-1).cpu().tolist(),
            memory_points=batch['memory_valid'].sum(-1).cpu().tolist(),
            sequence_valid=output.observation.sequence_valid.detach().cpu().tolist())
        if config.net_model == 'ctseqtrackv34':
            report['query_context'] = dict(
                history_support=output.observation.history_support.detach().cpu().tolist(),
                current_support=output.observation.current_support.detach().cpu().tolist(),
                norm=output.decoder.query_context_norm.detach().cpu().tolist())

        # 不调用 training_step 的 Lightning 日志；复用同一个 pending/commit hook。
        model._pending_train = (output, batch, len(rows))
        report['stage'] = 'backward'
        total.backward()
        report['gradients'] = gradient_summary(model)
        report['stage'] = 'optimizer_step'
        optimizer.step()
        step_values = sorted({int(state['step'].item()) for state in optimizer.state.values() if 'step' in state})
        if step_values != [1]:
            raise RuntimeError('Adam states must record exactly one step')
        for name, parameter in model.named_parameters():
            if not bool(torch.isfinite(parameter).all()):
                raise FloatingPointError('nonfinite parameter after Adam: ' + name)
        group = optimizer.param_groups[0]
        report['optimizer'] = dict(name='Adam', steps=1, state_step_values=step_values,
            lr=group['lr'], betas=list(group['betas']), eps=group['eps'], weight_decay=group['weight_decay'])
        report['stage'] = 'commit'
        model.on_train_batch_end(None, rows, 0)
        if model._pending_train is not None or model.train_builder._pending is not None:
            raise RuntimeError('one-batch transaction still has a pending commit')
        if model._epoch_steps != 1 or model._epoch_rows != 16:
            raise RuntimeError('host accounting must record exactly one step and sixteen rows')
        if device == 'cuda':
            torch.cuda.synchronize()
        report['commit'] = dict(pending_host=False, pending_builder=False,
            epoch_rows=model._epoch_rows, epoch_steps=model._epoch_steps,
            active_windows=len(model.train_builder.states))
        report['stage'] = 'complete'
        report['status'] = 'passed'
    finally:
        # persistent_workers=False；释放单批 iterator 后结束本进程，不继续采样/训练。
        del iterator


def main(argv=None, *, version=33):
    default_cfg = ROOT / ('cfgs/ct_seqtrack/34_b0_context_mini.yaml'
                         if version == 34 else 'cfgs/ct_seqtrack/33_b0_mini.yaml')
    args = parse_args(argv, default_cfg=default_cfg)
    destination = report_destination(args.output, version=version)
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = dict(schema=f'ct_seqtrack.v{version}.batch_check.v1', status='failed', stage='load_config',
        device=args.device, checkpoint_saved=False, formal_training_requires_new_scratch_process=True,
        started_utc=datetime.now(timezone.utc).isoformat(), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    started = time.perf_counter()
    try:
        from models.ct_v31.config import load_config, config_identity
        from models.ct_v31.entry import configure_numerics, source_identity
        overrides = dict(ct_engineering_check=True, log_dir=str(destination.parent),
                         v31_evaluate_late3=False, accelerator='gpu' if args.device == 'cuda' else 'cpu')
        if args.path is not None:
            overrides['path'] = args.path
        config = load_config(args.cfg, overrides)
        if config.net_model != f'ctseqtrackv{version}':
            raise ValueError(f'use the v{version} B0 configuration for this versioned batch tool')
        report.update(model_schema=f'ct_seqtrack.joint_identity.v{version}', config=dict(config), config_sha256=config_identity(config),
                      source=source_identity())
        configure_numerics()
        import torch
        import pytorch_lightning as pl
        report.update(torch=torch.__version__, lightning=pl.__version__, python=sys.version)
        pl.seed_everything(config.seed, workers=True)
        run_one_batch(config, args.device, report)
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        report['wall_seconds'] = time.perf_counter() - started
        # CUDA 初始化失败时仍尽量保留原始错误及 JSON，不让诊断覆盖失败原因。
        try:
            report['cuda_memory'] = cuda_memory(args.device)
        except Exception as error:
            report['cuda_memory_error'] = str(error)
        with destination.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps(dict(status=report['status'], stage=report['stage'], report=str(destination)), ensure_ascii=False))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    raise SystemExit(main())
