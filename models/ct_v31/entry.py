"""v31 专用入口；与历史插件参数、标定和训练 host 分离。"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .config import load_config, config_identity
from .contracts import SCHEMA


def batch_limit(value):
    return float(value) if any(char in value for char in '.eE') else int(value)


def parse_config(argv=None):
    parser = argparse.ArgumentParser(description='CT-SeqTrack v31 joint training / closed-loop evaluation')
    parser.add_argument('--cfg', required=True)
    for key in ('path', 'tag', 'log_dir', 'checkpoint', 'init_checkpoint', 'dynamics_time_manifest'):
        parser.add_argument('--' + key)
    for key in ('batch_size', 'epoch', 'workers', 'seed', 'check_val_every_n_epoch', 'trainer_devices'):
        parser.add_argument('--' + key, type=int)
    parser.add_argument('--accelerator', choices=('auto', 'cpu', 'gpu'))
    parser.add_argument('--dynamics_time_mode', choices=('true', 'fixed', 'shuffled'))
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
    arm = 'b0' if config.v31_arm == 'b0' else config.v31_arm + '_' + config.v31_temporal_backend
    parent = Path('artifacts/ct_checks') if config.ct_engineering_check else Path('output')
    suffix = '-test' if config.test else ''
    return (parent / f'{stamp}-31_{arm}-{config.tag}{suffix}').resolve()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def run(config, *, loaders=None):
    """loaders 仅为工程集成测试注入原始帧；正式路径由数据集工厂构造。"""
    configure_numerics()
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import Callback, LearningRateMonitor
    from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
    import yaml
    from models.ctseqtrackv31 import CTSEQTRACKV31
    from utils.lightning_runtime import FinalWindowCheckpoint

    pl.seed_everything(config.seed, workers=True)
    root = resolve_run_directory(config)
    root.mkdir(parents=True, exist_ok=True)
    config.log_dir = str(root)
    if (root / 'resolved_config.yaml').exists() and not config.checkpoint:
        raise FileExistsError('this run already has a resolved config; use a new --log_dir or same-run --checkpoint')
    (root / 'resolved_config.yaml').write_text(yaml.safe_dump(dict(config), allow_unicode=True,
                                                            sort_keys=True), encoding='utf-8')
    model = CTSEQTRACKV31(config, loaders=loaders)
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True,
                                           stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = 'unavailable'
    manifest = dict(schema=SCHEMA, config_sha256=config_identity(config), git_head=revision,
                    arm=config.v31_arm, temporal_backend=config.v31_temporal_backend,
                    enabled=dict(B1=model.tracker.enable_b1, B2=model.tracker.enable_b2,
                                 B3=model.tracker.enable_b3),
                    parameters=sum(p.numel() for p in model.parameters()),
                    torch=torch.__version__, lightning=pl.__version__, python=sys.version,
                    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    calibration_required=False)
    write_json(root / 'run_manifest.json', manifest)
    print('[v31] ' + json.dumps(manifest, ensure_ascii=False), flush=True)
    loggers = [CSVLogger(str(root), name='csv'), TensorBoardLogger(str(root), name='tensorboard')]
    class ConsoleProgress(Callback):
        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            if batch_idx % 50 == 0:
                loss = outputs.get('loss') if isinstance(outputs, dict) else outputs
                value = float(loss.detach()) if torch.is_tensor(loss) else loss
                print(f'[v31 train] epoch={trainer.current_epoch + 1}/{trainer.max_epochs} '
                      f'batch={batch_idx + 1}/{trainer.num_training_batches} '
                      f'step={trainer.global_step} loss={value}', flush=True)

        def on_validation_end(self, trainer, module):
            print('[v31 validation] epoch=' + str(trainer.current_epoch + 1) + ' ' +
                  json.dumps(module.evaluation_results, ensure_ascii=False), flush=True)

    callbacks = [LearningRateMonitor(logging_interval='epoch'), ConsoleProgress()]
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
        trainer.test(model, ckpt_path=str(checkpoint), verbose=False)
        result = dict(model.evaluation_results, checkpoint_epoch=epoch, checkpoint=str(checkpoint))
        results.append(result)
        label = f'epoch={epoch:03d}' if epoch is not None else checkpoint.stem
        directory = root / 'evaluation' / label
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / 'metrics.json', result)
        with (directory / 'frames.jsonl').open('w', encoding='utf-8') as stream:
            for row in model.evaluation.rows:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        print('[v31 evaluation] ' + json.dumps(result, ensure_ascii=False), flush=True)
    summary = dict(final=results[-1], late3={name: sum(row[name] for row in results) / len(results)
                   for name in ('success', 'precision')}, checkpoint_epochs=[row['checkpoint_epoch'] for row in results],
                   wall_seconds=time.perf_counter() - started)
    if torch.cuda.is_available():
        summary['peak_gpu_allocated_mib'] = torch.cuda.max_memory_allocated() / 2 ** 20
    write_json(root / 'results.json', summary)
    print('[v31 complete] ' + json.dumps(summary, ensure_ascii=False), flush=True)
    return root


def main(argv=None):
    return run(parse_config(argv))


if __name__ == '__main__':
    main()
