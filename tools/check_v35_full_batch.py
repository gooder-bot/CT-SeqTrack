"""一次真实 v35 Full batch16 生命周期检查，覆盖候选记录；不保存工程权重。"""
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

from tools.check_v35_batch import report_destination, gradient_summary, cuda_memory


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfg', type=Path, default=ROOT / 'cfgs/ct_seqtrack/35_full_w_scaled_lr_mini.yaml')
    parser.add_argument('--path', help='真实数据根；未提供时使用配置中的 path')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--output', type=Path, help='artifacts/ct_checks 下尚不存在的 JSON')
    return parser.parse_args(argv)


def run_one_batch(config, device, report, *, loaders=None):
    """只运行原宿主的一次事务；loaders 仅供本地合成测试注入。"""
    import torch
    from models.ct_v31.data import build_loaders, stable_seed
    from models.ct_v31.candidate_records import candidate_records
    from models.ctseqtrackv31 import CTSEQTRACKV31

    if (config.net_model != 'ctseqtrackv35' or config.v31_arm != 'full'
            or config.batch_size != 16 or config.point_sample_size != 1024 or config.precision != 32):
        raise ValueError('requires v35 Full, batch_size=16, point_sample_size=1024, FP32')
    if not config.ct_engineering_check or config.test or config.checkpoint or config.init_checkpoint:
        raise ValueError('requires engineering scratch with no checkpoint')
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        torch.cuda.reset_peak_memory_stats()
    def sync():
        if device == 'cuda':
            torch.cuda.synchronize()

    report['stage'] = 'build_loaders'
    loaders = build_loaders(config, roles=('train',)) if loaders is None else loaders
    loader = loaders['train']
    loader.batch_sampler.set_epoch(0)
    loader.generator.manual_seed(stable_seed(config.seed, 'train', 0))
    report['sampler'] = loader.batch_sampler.state_dict()
    model = CTSEQTRACKV31(config, loaders=loaders).to(device).train()
    model.on_train_epoch_start()
    optimizer = model.configure_optimizers()['optimizer']
    if not isinstance(optimizer, torch.optim.Adam) or optimizer.state:
        raise RuntimeError('requires a fresh registered Adam optimizer')
    optimizer.zero_grad(set_to_none=True)
    iterator = iter(loader)
    try:
        report['stage'] = 'read_first_batch'
        started = time.perf_counter()
        rows = next(iterator)
        if len(rows) != 16:
            raise ValueError('refusing a reduced-batch smoke')
        report['requests'] = [dict(tracklet=row['tracklet_key'], branch=row['request'].branch,
            frame=row['request'].frame) for row in rows]
        report['timing_seconds'] = dict(read_batch=time.perf_counter() - started)
        report['stage'] = 'forward'
        started = time.perf_counter()
        batch, output = model._forward_raw(rows, model.train_builder, training=True)
        if tuple(batch['points'].shape) != (16, 4, 1024, 5) or batch['points'].dtype != torch.float32:
            raise ValueError('registered prepared batch shape/dtype mismatch')
        if not bool(torch.isfinite(output.accepted_box).all()):
            raise FloatingPointError('nonfinite accepted boxes')
        if model.training_diagnostics is not None:
            output.loss_statistics = {}
        losses = model.tracker.compute_losses(batch, output)
        total = losses['loss_total']
        if total.ndim != 0 or not bool(torch.isfinite(total)):
            raise FloatingPointError('loss_total must be a finite scalar')
        report['losses'] = {name: float(value.detach()) for name, value in losses.items()
            if torch.is_tensor(value) and value.ndim == 0}
        if not all(math.isfinite(value) for value in report['losses'].values()):
            raise FloatingPointError('nonfinite loss/log')
        sync()
        report['timing_seconds']['forward_and_losses'] = time.perf_counter() - started
        report['stage'] = 'candidate_records'
        records = candidate_records(batch, output)
        # 与正式评测相同的 collector；JSON 严格有限，诊断不追加网络前向。
        serialized = json.dumps(records, ensure_ascii=False, allow_nan=False)
        report['candidate_records'] = dict(rows=len(records), json_bytes=len(serialized.encode('utf-8')),
            eligible_mode_count=output.hypothesis_valid[:, 1:].sum(-1).detach().cpu().tolist(),
            formed_mode_count=output.evidence.mode_valid.sum(-1).detach().cpu().tolist(),
            selected_index=output.selected_index.detach().cpu().tolist(), example=records[0])
        if len(records) != 16:
            raise RuntimeError('candidate collector did not cover the whole batch')
        if model.training_diagnostics is not None:
            model.training_diagnostics.add_batch(rows, batch, output, losses)
            report['training_diagnostics'] = model.training_diagnostics.summary(complete=False)
        report['batch'] = dict(shape=list(batch['points'].shape), dtype=str(batch['points'].dtype),
            valid_points_per_frame=batch['point_valid'].sum(-1).cpu().tolist(),
            extension_points=batch['extension_valid'].sum(-1).cpu().tolist(),
            selected_extension_points=output.evidence.point_valid.sum(-1).detach().cpu().tolist(),
            memory_points=batch['memory_valid'].sum(-1).cpu().tolist(),
            sequence_valid=output.observation.sequence_valid.detach().cpu().tolist())
        model._pending_train = (output, batch, len(rows))
        report['stage'] = 'backward'
        started = time.perf_counter()
        total.backward()
        sync()
        report['timing_seconds']['backward'] = time.perf_counter() - started
        report['gradients'] = gradient_summary(model)
        # 稀疏首批允许某模块监督为空；记录覆盖，不能把合法零梯度判成故障。
        report['module_gradients'] = {}
        for name in ('observation', 'prior', 'evidence', 'decoder'):
            parameters = list(getattr(model.tracker, name).parameters())
            gradients = [parameter.grad.detach() for parameter in parameters if parameter.grad is not None]
            report['module_gradients'][name] = dict(parameters=len(parameters),
                tensors_with_gradient=len(gradients),
                tensors_with_nonzero_gradient=sum(int(bool(g.any())) for g in gradients),
                global_l2_norm=math.sqrt(math.fsum(float(torch.linalg.vector_norm(g)) ** 2 for g in gradients)))
        report['stage'] = 'optimizer_step'
        started = time.perf_counter()
        optimizer.step()
        sync()
        report['timing_seconds']['adam'] = time.perf_counter() - started
        step_values = sorted({int(state['step'].item()) for state in optimizer.state.values() if 'step' in state})
        if step_values != [1]:
            raise RuntimeError('Adam must record exactly one update')
        for name, parameter in model.named_parameters():
            if not bool(torch.isfinite(parameter).all()):
                raise FloatingPointError('nonfinite parameter after Adam: ' + name)
        group = optimizer.param_groups[0]
        report['optimizer'] = dict(name='Adam', steps=1, state_step_values=step_values, lr=group['lr'],
            betas=list(group['betas']), eps=group['eps'], weight_decay=group['weight_decay'])
        report['stage'] = 'commit'
        model.on_train_batch_end(None, rows, 0)
        if model._pending_train is not None or model.train_builder._pending is not None:
            raise RuntimeError('pending transaction after commit')
        if model._epoch_steps != 1 or model._epoch_rows != 16:
            raise RuntimeError('host must count one update and sixteen rows')
        report['commit'] = dict(pending_host=False, pending_builder=False,
            epoch_rows=model._epoch_rows, epoch_steps=model._epoch_steps,
            active_windows=len(model.train_builder.states))
        report.update(stage='complete', status='passed')
    finally:
        del iterator


def main(argv=None):
    args = parse_args(argv)
    destination = report_destination(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = dict(schema='ct_seqtrack.v35.full_batch_check.v1', status='failed', stage='load_config',
        device=args.device, checkpoint_saved=False, formal_training_requires_new_scratch_process=True,
        started_utc=datetime.now(timezone.utc).isoformat(), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    started = time.perf_counter()
    try:
        from models.ct_v31.config import load_config, config_identity
        from models.ct_v31.entry import configure_numerics, source_identity
        overrides = dict(ct_engineering_check=True, log_dir=str(destination.parent), v31_evaluate_late3=False,
            accelerator='gpu' if args.device == 'cuda' else 'cpu')
        if args.path is not None:
            overrides['path'] = args.path
        config = load_config(args.cfg, overrides)
        report.update(model_schema='ct_seqtrack.joint_identity.v35', config=dict(config),
            config_sha256=config_identity(config), source=source_identity())
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
