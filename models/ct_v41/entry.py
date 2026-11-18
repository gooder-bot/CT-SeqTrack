"""V41 新运行入口；实际配置留档、逐轮评估与同 run 完整 epoch 恢复。"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .config import load_config, config_identity, normalize_config, run_directory_tag
from .identity import model_schema, model_version, runtime_key, RESUME_SCHEMA
from utils.execution_profile import (EXECUTION_PROFILES, execution_identity,
                                     validate_execution_identity)
from utils.time_config import TIME_MANIFEST_KEYS


def batch_limit(value):
    return float(value) if any(char in value for char in '.eE') else int(value)


def parse_config(argv=None):
    parser = argparse.ArgumentParser(description='CT-SeqTrack v41 protected observation and independent selective update tracking',
                                     allow_abbrev=False)
    parser.add_argument('--cfg', required=True)
    for key in ('path', 'tag', 'log_dir', 'checkpoint', 'init_checkpoint', 'dynamics_time_manifest'):
        parser.add_argument('--' + key)
    for key in TIME_MANIFEST_KEYS[1:]:
        parser.add_argument('--' + key, help='engineering-only manifest for its dataset role')
    parser.add_argument('--dynamics_fixed_delta_t', type=float, help='engineering-only fixed interval')
    for key in ('batch_size', 'epoch', 'workers', 'seed', 'check_val_every_n_epoch', 'trainer_devices',
                'ct_pointcloud_cache_bytes', 'lr_decay_step', 'lr_warmup_steps', 'log_every_n_steps',
                'checkpoint_every_n_epochs', 'report_last_k'):
        parser.add_argument('--' + key, type=int)
    for key in ('lr', 'wd', 'lr_decay_rate', 'progress_refresh_seconds', 'plain_log_interval_seconds'):
        parser.add_argument('--' + key, type=float)
    parser.add_argument('--lr_schedule', choices=('step', 'multistep', 'piecewise'))
    parser.add_argument('--lr_milestones', nargs='*', type=int)
    parser.add_argument('--lr_stage_values', nargs='+', type=float)
    parser.add_argument('--console_mode', choices=('auto', 'bar', 'plain'))
    parser.add_argument('--diagnostics_level', choices=('summary', 'full'))
    parser.add_argument('--full_diagnostic_epochs', nargs='*', type=int)
    parser.add_argument('--eval_split', choices=('test', 'dev'))
    parser.add_argument('--report_checkpoint', choices=('best_precision', 'last'))
    for key in ('tensorboard', 'v41_memory_logging'):
        parser.add_argument('--' + key, action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--devices', type=int, help='single physical GPU ID; trainer_devices remains 1')
    parser.add_argument('--accelerator', choices=('auto', 'cpu', 'gpu'))
    parser.add_argument('--dynamics_time_mode', choices=('true', 'fixed', 'shuffled'))
    parser.add_argument('--execution_profile', choices=EXECUTION_PROFILES)
    parser.add_argument('--v41_attention_backend', choices=('fa2_varlen', 'math'),
                        help='math is engineering-only; formal runs require FA2')
    for key in ('test', 'ct_engineering_check'):
        parser.add_argument('--' + key, action='store_true', default=None)
    for key in ('limit_train_batches', 'limit_val_batches'):
        parser.add_argument('--' + key, type=batch_limit)
    parser.add_argument('--no_late3', dest='v31_evaluate_late3', action='store_false', default=None,
                        help=argparse.SUPPRESS)  # v1 命令兼容；v2 直接复用逐轮结果，不再补跑 late3。
    supplied = vars(parser.parse_args(argv))
    path = supplied.pop('cfg')
    return load_config(path, {key: value for key, value in supplied.items() if value is not None})


def apply_device_selection(config):
    """显式物理卡优先于环境；缺省不改环境，不在 CUDA 初始化后静默换卡。"""
    if config.get('devices') is None:
        return
    requested = str(config['devices'])
    import torch
    if torch.cuda.is_initialized() and os.environ.get('CUDA_VISIBLE_DEVICES') != requested:
        raise RuntimeError('CUDA is already initialized; cannot change CUDA_VISIBLE_DEVICES with --devices')
    os.environ['CUDA_VISIBLE_DEVICES'] = requested


def configure_numerics(config=None):
    # 必须在首次 CUDA context 前配置；原生 allocator 与 torch 2.0.1 兼容。
    from .identity import ALLOCATOR
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', ALLOCATOR)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)


def resolve_run_directory(config, *, command_line=None):
    if config.log_dir:
        return Path(config.log_dir).resolve()
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    arm = config.v41_arm
    parent = Path('artifacts/ct_checks') if config.ct_engineering_check else Path('output')
    suffix = '-test' if config.test else ''
    version = model_version(config).removeprefix('v')
    explicit_tag = any(arg.partition('=')[0] == '--tag' for arg in command_line or ())
    tag = run_directory_tag(config, explicit_tag=explicit_tag)
    return (parent / f'{stamp}-{version}_{config.v41_execution_revision}_{arm}-{tag}{suffix}').resolve()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def validate_run_destination(config, root, *, checkpoint_identity=None):
    """所有运行 metadata 写入前验证目录及 checkpoint；返回是否保留原 metadata。"""
    root = Path(root)
    if config.test and root.exists() and any(root.iterdir()):
        raise FileExistsError('evaluation requires a new empty run directory: ' + str(root))
    manifest_path, resolved_path = root / 'run_manifest.json', root / 'resolved_config.yaml'
    existing = manifest_path.exists() or resolved_path.exists()
    if existing:
        if not manifest_path.exists() or not resolved_path.exists():
            raise ValueError('existing run metadata is incomplete; refusing to overwrite: ' + str(root))
        saved = json.loads(manifest_path.read_text(encoding='utf-8'))
        expected = dict(schema=model_schema(config), config_sha256=config_identity(config), model=config.net_model)
        for key, value in expected.items():
            if saved.get(key) != value:
                raise ValueError('existing run identity mismatch: ' + key)
        validate_execution_identity(saved.get('execution_identity'), config, source_identity())
        if not config.checkpoint:
            raise FileExistsError('this run already has metadata; use a new --log_dir or same-run --checkpoint')
    if (root.exists() and not existing
            and any(p.name not in ('train.log', 'train.pid') for p in root.iterdir())):
        raise FileExistsError('refusing an occupied directory without v41 metadata: ' + str(root))
    if config.checkpoint:
        import torch
        from .runtime import validate_resume_payload
        checkpoint = torch.load(config.checkpoint, map_location='cpu', weights_only=False)
        payload = validate_resume_payload(checkpoint.get(runtime_key(config)), config, training=not config.test)
        if not config.test and (not existing or payload.get('run_id') != saved.get('run_id')):
            raise ValueError('v41 training resume requires its original run directory and run_id')
        if checkpoint_identity is not None:
            checkpoint_identity.update(run_id=payload['run_id'], completed_epoch=payload['completed_epoch'])
    return existing


def write_run_metadata(config, root, manifest, *, preserve_existing):
    """合法同 run 恢复保留首次配置和源码记录，当前环境仍输出到本次日志。"""
    if preserve_existing:
        return
    import yaml
    root = Path(root)
    (root / 'resolved_config.yaml').write_text(yaml.safe_dump(dict(config), allow_unicode=True,
                                                            sort_keys=True), encoding='utf-8')
    write_json(root / 'run_manifest.json', manifest)


def write_resume_event(config, root, manifest):
    """系统时钟可重复；唯一后缀防止合法重试碰撞，仍使用独占创建。"""
    import uuid
    directory = Path(root) / 'resume_manifests'
    directory.mkdir(exist_ok=True)
    name = datetime.now().strftime('%Y%m%d-%H%M%S-%f') + '-' + uuid.uuid4().hex + '.json'
    event_path = directory / name
    with event_path.open('x', encoding='utf-8') as stream:
        json.dump(dict(schema=RESUME_SCHEMA, run_id=manifest['run_id'],
            checkpoint=str(Path(config.checkpoint).resolve()), source=manifest['source'],
            config_sha256=config_identity(config), resolved_config=dict(config),
            torch=manifest['torch'], lightning=manifest['lightning'],
            precision_policy=manifest['precision_policy'], flash_attention=manifest['flash_attention'],
            command_line=manifest.get('command_line', [])), stream, ensure_ascii=False, indent=2)
    return event_path


def source_identity():
    """记录执行时源码内容；未提交的本地修订不能只用git HEAD标识。"""
    from .provenance import source_identity as static_source_identity
    return static_source_identity()


def initialization_identity(tracker):
    """从随机初始化的实际参数/缓冲区登记公共路径；不消耗 RNG。"""
    state = tracker.state_dict()
    def digest(items):
        value = hashlib.sha256()
        for name, tensor in sorted(items):
            tensor = tensor.detach().cpu().contiguous()
            value.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)],
                                    separators=(',', ':')).encode())
            value.update(tensor.numpy().tobytes())
        return value.hexdigest()
    common = [(name, value) for name, value in state.items()
        if name.startswith(('observation.', 'decoder.')) and not name.startswith('decoder.prior_adapter.')]
    prior = [(name, value) for name, value in state.items() if name.startswith('prior.')]
    adapter = [(name, value) for name, value in state.items() if name.startswith('decoder.prior_adapter.')]
    mtp = [(name, value) for name, value in state.items() if name.startswith('multistep.')]
    recovery = [(name, value) for name, value in state.items() if name.startswith('recovery_head.')]
    if not common:
        raise ValueError('v41 common initialization group is empty')
    base = [(name, value) for name, value in state.items()
            if name.startswith(('observation.', 'decoder.', 'prior.'))]
    return dict(common_parameter_hash=digest(common),
                prior_parameter_hash=digest(prior) if prior else None,
                prior_adapter_parameter_hash=digest(adapter) if adapter else None,
                mtp_parameter_hash=digest(mtp) if mtp else None,
                recovery_parameter_hash=digest(recovery) if recovery else None,
                base_initial_parameter_hash=digest(base),
                initial_parameter_hash=digest(state.items()))


def run(config, *, loaders=None, command_line=None):
    # 新 run 自动记录实际配置；训练中的评估与独立评估使用同一执行算法。
    from .config import normalize_config
    from .config import recipe_scope
    config = normalize_config(config)
    with recipe_scope(config):
        return _run(config, loaders=loaders, command_line=command_line)


def _run(config, *, loaders=None, command_line=None):
    """loaders 仅为工程集成测试注入原始帧；正式路径由数据集工厂构造。"""
    apply_device_selection(config)
    configure_numerics(config)
    fa_runtime = None
    if config.v41_attention_backend == 'fa2_varlen':
        from .attention import validate_fa_runtime
        fa_runtime = validate_fa_runtime()
    version = model_version(config)
    root = resolve_run_directory(config, command_line=command_line)
    config.log_dir = str(root)
    source = source_identity()
    if loaders is not None and not config.ct_engineering_check:
        raise ValueError('v41 formal entry does not accept injected loaders')
    checkpoint_identity = {}
    preserve_metadata = validate_run_destination(config, root, checkpoint_identity=checkpoint_identity)
    if config.test:
        config.eval_checkpoint_epoch = checkpoint_identity['completed_epoch']
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import LearningRateMonitor
    from models.ctseqtrackv41 import CTSEQTRACKV41
    from .monitoring import AppendCSVLogger, EpochProgress
    from .epoch_runtime import EpochCheckpoint, install_evaluation_rng, EVALUATION_RNG_POLICY

    pl.seed_everything(config.seed, workers=True)
    root.mkdir(parents=True, exist_ok=True)
    model = CTSEQTRACKV41(config, loaders=loaders)
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True,
                                           stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = 'unavailable'
    manifest = dict(schema=model_schema(config), config_sha256=config_identity(config), git_head=revision,
                    model=config.net_model, seed=config.seed,
                    arm=config.v41_arm, temporal_backend=config.v31_temporal_backend,
                    enabled=dict(B1=config.enable_b1, M=config.enable_m, B2=False,
                                 B3=config.enable_b3),
                    parameters=sum(p.numel() for p in model.parameters()),
                    torch=torch.__version__, lightning=pl.__version__, python=sys.version,
                    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    allocator=os.environ.get('PYTORCH_CUDA_ALLOC_CONF'),
                    flash_attention=fa_runtime,
                    precision_policy='FP32 except BF16 QKV inside FA2 kernel',
                    calibration_required=False, source=source, registration_required=False,
                    run_type='engineering' if config.ct_engineering_check else 'formal',
                    actual_recipe={key: config[key] for key in ('epoch', 'batch_size', 'workers', 'seed', 'lr',
                        'wd', 'lr_schedule', 'lr_milestones', 'lr_decay_step', 'lr_decay_rate', 'lr_warmup_steps',
                        'v41_arm', 'v41_q0_center_weighting', 'dataset', 'category_name')},
                    command_line=list(command_line or []),
                    evaluation_rng_policy=EVALUATION_RNG_POLICY,
                    evaluation_protocol=dict(split=config.eval_split,
                        every_n_epochs=config.check_val_every_n_epoch, always_final=True,
                        selection=config.report_checkpoint, selection_metric='precision',
                        tie_break='earliest', last_k=config.report_last_k,
                        checkpoint_every_n_epochs=config.checkpoint_every_n_epochs))
    manifest.update(initialization_identity(model.tracker))
    if config.dataset == 'kitti_mf' and not config.ct_engineering_check:
        from datasets.kitti_v41 import build_v41_kitti_manifest, kitti_input_metadata_identity
        manifest['dataset_manifest'] = build_v41_kitti_manifest()
        manifest['input_metadata'] = kitti_input_metadata_identity(config.path)
    from .config import recipe_identity
    manifest['training_execution_recipe'] = recipe_identity(config)
    manifest['policy_activation'] = 'active_from_epoch1'
    manifest['execution_identity'] = execution_identity(config, manifest['source'])
    import uuid
    existing_manifest = (json.loads((root / 'run_manifest.json').read_text(encoding='utf-8'))
                         if preserve_metadata else None)
    model.run_id = (existing_manifest['run_id'] if existing_manifest else
                    checkpoint_identity.get('run_id', uuid.uuid4().hex))
    manifest['run_id'] = model.run_id
    if config.test:
        manifest['source_run_id'] = checkpoint_identity['run_id']
        manifest['evaluation_session_id'] = uuid.uuid4().hex
        manifest['checkpoint_completed_epoch'] = checkpoint_identity['completed_epoch']
    if existing_manifest and existing_manifest.get('source') != manifest['source']:
        raise ValueError('v41 resume source differs; use the original run source')
    if existing_manifest and any(existing_manifest.get(key) != manifest.get(key)
                                 for key in ('dataset_manifest', 'input_metadata')):
        raise ValueError('v41 resume dataset identity differs')
    if preserve_metadata:
        write_resume_event(config, root, manifest)
    write_run_metadata(config, root, manifest, preserve_existing=preserve_metadata)
    print(f'[{version}] run={model.run_id} split={config.eval_split} '
          f'epochs={config.epoch} diagnostics={config.diagnostics_level} output={root}', flush=True)
    scalar_logger = AppendCSVLogger(root)
    loggers = [scalar_logger]
    if config.tensorboard:
        from pytorch_lightning.loggers import TensorBoardLogger
        loggers.append(TensorBoardLogger(str(root), name='tensorboard'))
    progress = EpochProgress(config, scalar_logger)
    # progress 先切换 logger 的 epoch；LRMonitor 的直接日志没有 epoch 字段。
    callbacks = [progress,
                 LearningRateMonitor(logging_interval='step' if config.lr_warmup_steps else 'epoch')]
    if not config.test:
        callbacks.append(EpochCheckpoint(config))
    trainer = pl.Trainer(default_root_dir=str(root), max_epochs=config.epoch,
        accelerator=config.accelerator, devices=config.trainer_devices, precision=32,
        # 事务缓存需要 tensor 版本计数；no_grad 足以禁止评估反向图。
        inference_mode=False,
        deterministic=True, benchmark=False, logger=loggers, callbacks=callbacks,
        enable_checkpointing=not config.test, reload_dataloaders_every_n_epochs=1,
        check_val_every_n_epoch=1, val_check_interval=1.0, num_sanity_val_steps=0,
        accumulate_grad_batches=1, gradient_clip_val=0., log_every_n_steps=config.log_every_n_steps,
        limit_train_batches=config.limit_train_batches, limit_val_batches=config.limit_val_batches,
        enable_progress_bar=False, enable_model_summary=False)
    install_evaluation_rng(trainer, model, config)
    if config.test:
        started = time.perf_counter()
        trainer.test(model, ckpt_path=config.checkpoint, verbose=False)
        result = dict(model.evaluation_results, checkpoint=str(config.checkpoint),
                      wall_seconds=time.perf_counter() - started)
        write_json(root / 'results.json', result)
        print(f'{config.eval_split.title()} E{model._completed_epoch} '
              f'S={result["success"]:.4f} P={result["precision"]:.4f} '
              f'frames={result["prediction_frames"]} | results={root / "results.json"}', flush=True)
    else:
        trainer.fit(model, ckpt_path=config.checkpoint)
        write_json(root / 'training_budget.json', dict(
            schema=model_schema(config), completed_epoch=model._completed_epoch,
            epoch_complete=model._epoch_complete, last_epoch_rows=model._epoch_rows,
            last_epoch_steps=model._epoch_steps, optimizer_steps=int(trainer.global_step),
            sampler=model._loaders['train'].batch_sampler.state_dict(),
            auxiliary=getattr(model, '_last_auxiliary_audit', None)))
    return root


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    return run(parse_config(arguments), command_line=['main.py', *arguments])


if __name__ == '__main__':
    main()
