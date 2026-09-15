"""复用 main.py 检查 v28 两个工程 epoch 的连续/同运行边界恢复逐位一致性。"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config import load_yaml_config
from utils.online_contract import validate_online_resume_contract
from utils.v28_numerical_audit import compare_values

SCHEMA = 'ct_seqtrack.resume_audit.v28'
GENERATOR_KEY = 'ct_seqtrack.DataLoaderGeneratorState.v1'


def output_directory(value):
    output = Path(value).expanduser().resolve()
    allowed = (ROOT / 'artifacts' / 'ct_checks').resolve()
    if allowed not in output.parents:
        raise ValueError('resume audit output must be below repository artifacts/ct_checks/')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'resume audit requires a new empty directory: {output}')
    return output


def build_plan(config_path, data_path, output, steps=16, gpu='0', python=sys.executable):
    """只构造计划；三个进程共用同一工程配置，恢复不改变 epoch 或步数身份。"""
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 100:
        raise ValueError('engineering steps must be an integer in [1, 100]')
    if ',' in str(gpu) or not str(gpu).strip():
        raise ValueError('resume audit requires exactly one CUDA_VISIBLE_DEVICES entry')
    output = output_directory(output)
    config = copy.deepcopy(load_yaml_config(config_path))
    if not config.get('ct_enable_v28') or not config.get('ct_enable_v27'):
        raise ValueError('resume audit requires a v28 configuration')
    if str(config.get('ct_initialization_policy', '')) != 'scratch_only':
        raise ValueError('resume audit requires scratch_only initialization')
    for key in ('cfg', 'checkpoint', 'init_checkpoint', 'log_dir', 'test'):
        config.pop(key, None)
    config.update(epoch=2, from_epoch=0, path=str(data_path),
                  limit_train_batches=steps, ct_engineering_check=True)
    if config.get('ct_enable_v30', False):
        # v30工程检查也实际触发一次有界验证；两个分支使用完全相同的配置。
        config.update(check_val_every_n_epoch=1, limit_val_batches=1)
    snapshot = output / 'engineering_config.yaml'
    processes = []
    for phase, directory in (('continuous', 'continuous'), ('split', 'split'), ('resume', 'split')):
        argv = [str(python), str(Path(__file__).resolve()), '--worker', phase,
                '--cfg', str(snapshot), '--path', str(data_path),
                '--run-dir', str(output / directory)]
        if phase == 'resume':
            argv.extend(['--checkpoint', str(output / 'split' / 'resume_audit' / 'epoch=001.ckpt')])
        processes.append({'phase': phase, 'argv': argv,
                          'log': str(output / f'{phase}.log')})
    return config, {
        'schema': SCHEMA, 'engineering_only': True, 'formal_initialization_allowed': False,
        'source_config': str(Path(config_path).resolve()), 'config': str(snapshot),
        'output': str(output), 'gpu': str(gpu), 'epochs': 2, 'steps_per_epoch': steps,
        'seed': config.get('seed'), 'workers': config.get('workers'),
        'check_val_every_n_epoch': config.get('check_val_every_n_epoch'),
        'validation_note': ('按原验证周期运行；若周期为5，此两轮检查不覆盖验证事件。'),
        'processes': processes,
    }


def make_callbacks(stop_epoch=None):
    """Checkpoint 类型确保快照发生在 host 的 epoch-end 状态提交之后。"""
    from pytorch_lightning.callbacks import Callback, Checkpoint

    class StopAfterEpoch(Callback):
        def on_train_epoch_end(self, trainer, module):
            del module
            if stop_epoch is not None and int(trainer.current_epoch) + 1 == stop_epoch:
                trainer.should_stop = True

    class AuditEpochCheckpoint(Checkpoint):
        def on_train_epoch_end(self, trainer, module):
            if not bool(getattr(module, '_ct_epoch_boundary_complete', False)):
                raise RuntimeError('resume audit snapshot preceded host epoch-boundary completion')
            directory = Path(trainer.default_root_dir) / 'resume_audit'
            directory.mkdir(parents=True, exist_ok=True)
            epoch = int(trainer.current_epoch) + 1
            trainer.save_checkpoint(str(directory / f'epoch={epoch:03d}.ckpt'))

    return [StopAfterEpoch(), AuditEpochCheckpoint()]


def worker_main(args):
    """仅增加工程检查 callback；前向、优化、DataLoader、保存/恢复仍由 main.py 执行。"""
    import pytorch_lightning as pl

    config = load_yaml_config(args.cfg)
    if (not config.get('ct_engineering_check') or config.get('epoch') != 2
            or not config.get('ct_enable_v28')):
        raise ValueError('worker requires the shared two-epoch v28 engineering snapshot')
    run_dir = Path(args.run_dir).resolve()
    allowed = (ROOT / 'artifacts' / 'ct_checks').resolve()
    if allowed not in run_dir.parents:
        raise ValueError('worker run directory must stay in artifacts/ct_checks/')
    if args.worker == 'resume':
        checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else None
        if checkpoint is None or checkpoint.parent != run_dir / 'resume_audit':
            raise ValueError('resume must use the same split run epoch-boundary checkpoint')
    elif args.checkpoint:
        raise ValueError('continuous and first split phase must start from scratch')
    original_trainer = pl.Trainer

    class AuditedTrainer(original_trainer):
        def __init__(self, *positional, **kwargs):
            kwargs['callbacks'] = list(kwargs.get('callbacks', [])) + make_callbacks(
                stop_epoch=1 if args.worker == 'split' else None)
            if kwargs.get('max_epochs') != 2 or kwargs.get('min_epochs') != 0:
                raise RuntimeError('main.py engineering Trainer must use max_epochs=2/min_epochs=0')
            super().__init__(*positional, **kwargs)

    original_argv = sys.argv
    pl.Trainer = AuditedTrainer
    sys.argv = [str(ROOT / 'main.py'), '--cfg', str(Path(args.cfg).resolve()),
                '--path', args.path, '--log_dir', str(run_dir), '--ct_engineering_check']
    if args.checkpoint:
        sys.argv.extend(['--checkpoint', str(Path(args.checkpoint).resolve())])
    try:
        runpy.run_path(str(ROOT / 'main.py'), run_name='__main__')
    finally:
        pl.Trainer = original_trainer
        sys.argv = original_argv


def checkpoint_components(checkpoint, completed_epochs, steps_per_epoch):
    """失败关闭：缺少完整数值状态或实际更新预算时不能报告恢复通过。"""
    if checkpoint.get('ct_epoch_boundary_complete') is not True:
        raise ValueError('checkpoint is not a completed logical epoch')
    if checkpoint.get('epoch') != completed_epochs - 1:
        raise ValueError('checkpoint epoch does not match the requested boundary')
    if checkpoint.get('global_step') != completed_epochs * steps_per_epoch:
        raise ValueError('checkpoint did not execute the requested optimizer-step budget')
    config = checkpoint.get('hyper_parameters', {}).get('config', {})
    if not config.get('ct_engineering_check') or config.get('epoch') != 2:
        raise ValueError('checkpoint is not marked as the shared engineering-only run')
    validate_online_resume_contract(checkpoint, config)
    required = ('state_dict', 'optimizer_states', 'lr_schedulers', 'ct_global_rng_state')
    for key in required:
        if not checkpoint.get(key):
            raise ValueError(f'checkpoint lacks {key}')
    generator_state = checkpoint.get('callbacks', {}).get(GENERATOR_KEY)
    if not generator_state or not generator_state.get('states'):
        raise ValueError('checkpoint lacks DataLoader generator states')
    state = checkpoint['state_dict']
    counts = {key: value for key, value in state.items() if key.endswith('_update_step')}
    for name in ('b0', 'b1', 'b2', 'b3'):
        if name == 'b0' or config.get(f'ct_enable_{name}', False):
            key = f'ct_{name}_update_step'
            if key not in counts or int(counts[key]) <= 0:
                raise ValueError(f'{name} was not updated; increase engineering --steps')
    return {
        'all_model_parameters_buffers_and_extra_state': state,
        'adam': checkpoint['optimizer_states'], 'scheduler': checkpoint['lr_schedulers'],
        'epoch_and_step_counters': {'epoch': checkpoint['epoch'],
                                    'global_step': checkpoint['global_step'], 'modules': counts},
        'global_rng': checkpoint['ct_global_rng_state'], 'dataloader_rng': generator_state,
        'runtime_environment': checkpoint.get('ct_v28_runtime_environment', {}),
        'resume_identity': checkpoint['ct_online_resume_contract'],
    }


def compare_checkpoints(left, right, completed_epochs, steps_per_epoch):
    a = checkpoint_components(left, completed_epochs, steps_per_epoch)
    b = checkpoint_components(right, completed_epochs, steps_per_epoch)
    comparisons = {key: compare_values(a[key], b[key], key) for key in a}
    return {'completed_epochs': completed_epochs,
            'passed': all(value is None for value in comparisons.values()),
            'comparisons': comparisons}


def run_audit(config, plan, plan_only=False):
    output = output_directory(plan['output'])
    output.mkdir(parents=True, exist_ok=True)
    Path(plan['config']).write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=True), encoding='utf-8')
    (output / 'plan.json').write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding='utf-8')
    report = dict(schema=SCHEMA, engineering_only=True, formal_initialization_allowed=False,
                  status='planned' if plan_only else 'running', passed=False,
                  steps_per_epoch=plan['steps_per_epoch'], phases=[], comparisons=[])
    destination = output / 'report.json'
    def save():
        destination.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    save()
    if plan_only:
        return report
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=plan['gpu'], OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:64')
    # 与独立的逐step数值审计隔离，避免复用同一非空快照目录。
    environment.pop('CT_V28_AUDIT_DIR', None)
    environment.pop('CT_V28_AUDIT_ACTIVATIONS', None)
    try:
        for process in plan['processes']:
            print(f"[v28 resume audit] {process['phase']} -> {process['log']}", flush=True)
            with Path(process['log']).open('w', encoding='utf-8') as log:
                result = subprocess.run(process['argv'], cwd=ROOT, env=environment,
                                        stdout=log, stderr=subprocess.STDOUT, check=False)
            report['phases'].append({'phase': process['phase'], 'exit_code': result.returncode})
            save()
            if result.returncode:
                raise RuntimeError(f"{process['phase']} failed; inspect {process['log']}")
        for epoch in (1, 2):
            paths = [output / run / 'resume_audit' / f'epoch={epoch:03d}.ckpt'
                     for run in ('continuous', 'split')]
            checkpoints = [torch.load(path, map_location='cpu', weights_only=False) for path in paths]
            comparison = compare_checkpoints(*checkpoints, epoch, plan['steps_per_epoch'])
            comparison['paths'] = list(map(str, paths))
            report['comparisons'].append(comparison)
        report['passed'] = all(item['passed'] for item in report['comparisons'])
        report['status'] = 'passed' if report['passed'] else 'failed'
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    save()
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfg', required=True)
    parser.add_argument('--path', required=True)
    parser.add_argument('--output')
    parser.add_argument('--steps', type=int, default=16)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--worker', choices=('continuous', 'split', 'resume'), help=argparse.SUPPRESS)
    parser.add_argument('--run-dir', help=argparse.SUPPRESS)
    parser.add_argument('--checkpoint', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        if not args.run_dir:
            parser.error('--worker requires --run-dir')
    elif not args.output:
        parser.error('--output is required')
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.worker:
        worker_main(args)
        return 0
    config, plan = build_plan(args.cfg, args.path, args.output, args.steps, args.gpu, args.python)
    report = run_audit(config, plan, args.plan_only)
    print(json.dumps({'status': report['status'], 'report': str(Path(args.output) / 'report.json')}, ensure_ascii=False))
    return 0 if args.plan_only or report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
