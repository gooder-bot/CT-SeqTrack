"""v37 独立训练/评测入口；五组从零训练与同 run 完整 epoch 恢复。"""
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

from .config import load_config, config_identity
from .identity import model_schema, model_version, runtime_key
from utils.execution_profile import (EXECUTION_PROFILES, execution_identity,
                                     validate_execution_identity)


def batch_limit(value):
    return float(value) if any(char in value for char in '.eE') else int(value)


def parse_config(argv=None):
    parser = argparse.ArgumentParser(description='CT-SeqTrack v37 protected observation and structural evidence tracking')
    parser.add_argument('--cfg', required=True)
    for key in ('path', 'tag', 'log_dir', 'checkpoint', 'init_checkpoint', 'dynamics_time_manifest'):
        parser.add_argument('--' + key)
    for key in ('batch_size', 'epoch', 'workers', 'seed', 'check_val_every_n_epoch', 'trainer_devices'):
        parser.add_argument('--' + key, type=int)
    parser.add_argument('--accelerator', choices=('auto', 'cpu', 'gpu'))
    parser.add_argument('--dynamics_time_mode', choices=('true', 'fixed', 'shuffled'))
    parser.add_argument('--execution_profile', choices=EXECUTION_PROFILES)
    for key in ('test', 'ct_engineering_check'):
        parser.add_argument('--' + key, action='store_true', default=None)
    for key in ('limit_train_batches', 'limit_val_batches'):
        parser.add_argument('--' + key, type=batch_limit)
    parser.add_argument('--no_late3', dest='v31_evaluate_late3', action='store_false', default=None,
                        help='skip automatic post-training final-window evaluation')
    supplied = vars(parser.parse_args(argv))
    path = supplied.pop('cfg')
    return load_config(path, {key: value for key, value in supplied.items() if value is not None})


def configure_numerics():
    # 必须在首次 CUDA context 前配置；原生 allocator 与 torch 2.0.1 兼容。
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'backend:native')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)


def resolve_run_directory(config):
    if config.log_dir:
        return Path(config.log_dir).resolve()
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    arm = config.v37_arm
    parent = Path('artifacts/ct_checks') if config.ct_engineering_check else Path('output')
    suffix = '-test' if config.test else ''
    version = model_version(config).removeprefix('v')
    return (parent / f'{stamp}-{version}_{arm}-{config.tag}{suffix}').resolve()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def validate_run_destination(config, root):
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
        raise FileExistsError('refusing an occupied directory without v37 metadata: ' + str(root))
    if config.checkpoint:
        import torch
        from .runtime import validate_resume_payload
        checkpoint = torch.load(config.checkpoint, map_location='cpu', weights_only=False)
        payload = validate_resume_payload(checkpoint.get(runtime_key(config)), config, training=not config.test)
        if not config.test and (not existing or payload.get('run_id') != saved.get('run_id')):
            raise ValueError('v37 training resume requires its original run directory and run_id')
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
        json.dump(dict(schema='ct_seqtrack.v37.resume.v1', run_id=manifest['run_id'],
            checkpoint=str(Path(config.checkpoint).resolve()), source=manifest['source'],
            config_sha256=config_identity(config)), stream, ensure_ascii=False, indent=2)
    return event_path


def source_identity():
    """记录执行时源码内容；未提交的本地修订不能只用git HEAD标识。"""
    root = Path(__file__).resolve().parents[2]
    files = [root / 'main.py', root / 'requirement.txt']
    for name in ('models', 'datasets', 'utils'):
        files.extend((root / name).rglob('*.py'))
    hashes = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(files)}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return dict(sha256=digest, files=hashes)


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
        if (name.startswith('observation.') or name.startswith('decoder.'))
        and not name.startswith(('decoder.memory_projection.', 'decoder.mode_pose_head.'))]
    prior = [(name, value) for name, value in state.items() if name.startswith('prior.')]
    if not common:
        raise ValueError('v37 common initialization group is empty')
    return dict(common_parameter_hash=digest(common),
                prior_parameter_hash=digest(prior) if prior else None,
                initial_parameter_hash=digest(state.items()))


def validate_formal_registration(config, source=None, *, registration_path=None):
    """正式入口只接收五组登记的完整归一化配置与源码；工程运行独立。"""
    if config.ct_engineering_check:
        return None
    from utils.b1_training_recipes import (RECIPE_KEY, REGISTRATION_FILE,
        REGISTRATION_SCHEMA, recipe_identity)
    from utils.b2_centroid_recipes import (RECIPE_KEY as CENTROID_RECIPE_KEY,
        REGISTRATION_FILE as CENTROID_REGISTRATION_FILE,
        REGISTRATION_SCHEMA as CENTROID_REGISTRATION_SCHEMA,
        recipe_identity as centroid_recipe_identity)
    is_recipe = RECIPE_KEY in config
    is_centroid = CENTROID_RECIPE_KEY in config
    if is_recipe and is_centroid:
        raise ValueError('v37 centroid and B1 execution recipes are mutually exclusive')
    filename = (CENTROID_REGISTRATION_FILE if is_centroid else
                REGISTRATION_FILE if is_recipe else '37_registration.json')
    path = (Path(registration_path) if registration_path is not None else
            Path(__file__).resolve().parents[2] / 'cfgs/ct_seqtrack' / filename)
    registration = json.loads(path.read_text(encoding='utf-8'))
    expected_schema = (CENTROID_REGISTRATION_SCHEMA if is_centroid else
                       REGISTRATION_SCHEMA if is_recipe else 'ct_seqtrack.v37.registration.v1')
    if registration.get('schema') != expected_schema:
        raise ValueError('v37 formal registration schema mismatch')
    source = source_identity() if source is None else source
    if registration.get('source') != source:
        raise ValueError('v37 formal source differs from ' + filename)
    matches = [row for row in registration.get('runs', [])
        if row.get('config_sha256') == config_identity(config)
        and row.get('arm') == config.v37_arm
        and row.get('execution_profile') == config.execution_profile]
    if len(matches) != 1:
        raise ValueError('v37 formal config is not exactly one registered run')
    if is_recipe and matches[0].get('training_execution_recipe') != recipe_identity(config):
        raise ValueError('registered B1 execution recipe identity mismatch')
    if is_centroid and (matches[0].get('centroid_recipe_identity') != centroid_recipe_identity(config)
            or matches[0].get('label') != config[CENTROID_RECIPE_KEY]):
        raise ValueError('registered B2 centroid recipe identity mismatch')
    return dict(schema=registration['schema'], label=matches[0]['label'],
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def run(config, *, loaders=None):
    # 训练、逐轮validation、late-3、独立test及同run恢复共用登记的执行算法。
    from .config import normalize_config
    from utils.b2_centroid_recipes import recipe_scope
    config = normalize_config(config)
    with recipe_scope(config):
        return _run(config, loaders=loaders)


def _run(config, *, loaders=None):
    """loaders 仅为工程集成测试注入原始帧；正式路径由数据集工厂构造。"""
    configure_numerics()
    version = model_version(config)
    root = resolve_run_directory(config)
    config.log_dir = str(root)
    source = source_identity()
    registration = validate_formal_registration(config, source)
    if loaders is not None and not config.ct_engineering_check:
        raise ValueError('v37 formal entry does not accept injected loaders')
    preserve_metadata = validate_run_destination(config, root)
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import Callback, LearningRateMonitor
    from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
    from models.ctseqtrackv37 import CTSEQTRACKV37
    from utils.lightning_runtime import FinalWindowCheckpoint

    pl.seed_everything(config.seed, workers=True)
    root.mkdir(parents=True, exist_ok=True)
    model = CTSEQTRACKV37(config, loaders=loaders)
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True,
                                           stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = 'unavailable'
    manifest = dict(schema=model_schema(config), config_sha256=config_identity(config), git_head=revision,
                    model=config.net_model, seed=config.seed,
                    arm=config.v37_arm, temporal_backend=config.v31_temporal_backend,
                    enabled=dict(B1=model.tracker.enable_b1, B2=model.tracker.enable_b2,
                                 B3=model.tracker.enable_b3),
                    parameters=sum(p.numel() for p in model.parameters()),
                    torch=torch.__version__, lightning=pl.__version__, python=sys.version,
                    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    calibration_required=False, source=source, registration=registration,
                    evaluation_rng_policy='per_checkpoint_seed_v1')
    manifest.update(initialization_identity(model.tracker))
    from utils.b1_training_recipes import recipe_identity
    if recipe_identity(config) is not None:
        manifest['training_execution_recipe'] = recipe_identity(config)
    from utils.b2_centroid_recipes import recipe_identity as centroid_recipe_identity
    if centroid_recipe_identity(config) is not None:
        manifest['centroid_recipe_identity'] = centroid_recipe_identity(config)
    manifest['execution_identity'] = execution_identity(config, manifest['source'])
    import uuid
    existing_manifest = (json.loads((root / 'run_manifest.json').read_text(encoding='utf-8'))
                         if preserve_metadata else None)
    model.run_id = existing_manifest['run_id'] if existing_manifest else uuid.uuid4().hex
    manifest['run_id'] = model.run_id
    if existing_manifest and existing_manifest.get('source') != manifest['source']:
        raise ValueError('v37 resume source differs; use the original run source')
    if preserve_metadata:
        write_resume_event(config, root, manifest)
    write_run_metadata(config, root, manifest, preserve_existing=preserve_metadata)
    console_manifest = {key: value for key, value in manifest.items()
                        if key not in ('source', 'reference_protocol')}
    console_manifest['source_sha256'] = manifest['source']['sha256']
    print('[' + version + '] ' + json.dumps(console_manifest, ensure_ascii=False), flush=True)
    loggers = [CSVLogger(str(root), name='csv'), TensorBoardLogger(str(root), name='tensorboard')]
    class ConsoleProgress(Callback):
        def on_exception(self, trainer, module, exception):
            module.clear_pending_materialization()

        def on_train_batch_start(self, trainer, module, batch, batch_idx):
            if config.lr_warmup_steps and batch_idx % 50 == 0:
                lr = trainer.optimizers[0].param_groups[0]['lr']
                print(f'[{version} lr] update={trainer.global_step + 1} lr={lr:.9g}', flush=True)

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            if batch_idx % 50 == 0:
                loss = outputs.get('loss') if isinstance(outputs, dict) else outputs
                value = float(loss.detach()) if torch.is_tensor(loss) else loss
                print(f'[{version} train] epoch={trainer.current_epoch + 1}/{trainer.max_epochs} '
                      f'batch={batch_idx + 1}/{trainer.num_training_batches} '
                      f'step={trainer.global_step} loss={value}', flush=True)

        def on_validation_end(self, trainer, module):
            print('[' + version + ' validation] epoch=' + str(trainer.current_epoch + 1) + ' ' +
                  json.dumps(module.evaluation_results, ensure_ascii=False), flush=True)

    callbacks = [LearningRateMonitor(logging_interval='step' if config.lr_warmup_steps else 'epoch'),
                 ConsoleProgress()]
    if not config.test:
        callbacks.append(FinalWindowCheckpoint(keep=3, every_n_epochs=2))
    trainer = pl.Trainer(default_root_dir=str(root), max_epochs=config.epoch,
        accelerator=config.accelerator, devices=config.trainer_devices, precision=32,
        deterministic=True, benchmark=False, logger=loggers, callbacks=callbacks,
        enable_checkpointing=not config.test, reload_dataloaders_every_n_epochs=1,
        check_val_every_n_epoch=config.check_val_every_n_epoch, num_sanity_val_steps=0,
        accumulate_grad_batches=1, gradient_clip_val=0., log_every_n_steps=10,
        limit_train_batches=config.limit_train_batches, limit_val_batches=config.limit_val_batches,
        enable_progress_bar=sys.stdout.isatty())
    started = time.perf_counter()
    if config.test:
        checkpoints = [(config.eval_checkpoint_epoch, Path(config.checkpoint))]
    else:
        trainer.fit(model, ckpt_path=config.checkpoint)
        write_json(root / 'training_budget.json', dict(
            schema=model_schema(config), completed_epoch=model._completed_epoch,
            epoch_complete=model._epoch_complete, last_epoch_rows=model._epoch_rows,
            last_epoch_steps=model._epoch_steps, optimizer_steps=int(trainer.global_step),
            sampler=model._loaders['train'].batch_sampler.state_dict()))
        checkpoints = [(epoch, root / 'formal_checkpoints' / f'epoch={epoch:03d}.ckpt')
                       for epoch in range(max(1, config.epoch - 2), config.epoch + 1)]
        if not config.v31_evaluate_late3:
            return root
    results = []
    # 复用单个模型/Trainer，避免评测时叠加第二份 GPU 权重和训练优化器。
    model.config.test = True
    for epoch, checkpoint in checkpoints:
        model.config.checkpoint = str(checkpoint)
        model.config.eval_checkpoint_epoch = epoch
        # 原SeqTrack历史点采样使用NumPy全局RNG。每次独立评测重置同一
        # 起点，确保自动late-3与单独--test对同一权重具有一致采样身份。
        pl.seed_everything(config.seed, workers=True)
        from .data import stable_seed
        model._loader('test').generator.manual_seed(stable_seed(config.seed, 'test'))
        trainer.test(model, ckpt_path=str(checkpoint), verbose=False)
        if epoch is None:
            # 独立 --test 没有额外 epoch 参数；以已校验 checkpoint 的运行身份为准。
            epoch = model._completed_epoch
        result = dict(model.evaluation_results, checkpoint_epoch=epoch, checkpoint=str(checkpoint))
        results.append(result)
        label = f'epoch={epoch:03d}' if epoch is not None else checkpoint.stem
        directory = root / 'evaluation' / label
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError('evaluation already exists: ' + str(directory))
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / 'metrics.json', result)
        with (directory / 'frames.jsonl').open('w', encoding='utf-8') as stream:
            for row in model.evaluation.rows:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        print('[' + version + ' evaluation] ' + json.dumps(result, ensure_ascii=False), flush=True)
    summary = dict(final=results[-1], late3={name: sum(row[name] for row in results) / len(results)
                   for name in ('success', 'precision')}, checkpoint_epochs=[row['checkpoint_epoch'] for row in results],
                   wall_seconds=time.perf_counter() - started)
    if torch.cuda.is_available():
        summary['peak_gpu_allocated_mib'] = torch.cuda.max_memory_allocated() / 2 ** 20
    write_json(root / 'results.json', summary)
    print('[' + version + ' complete] ' + json.dumps(summary, ensure_ascii=False), flush=True)
    return root


def main(argv=None):
    return run(parse_config(argv))


if __name__ == '__main__':
    main()
