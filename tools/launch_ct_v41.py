"""只打印独立 nohup/tail 命令；默认 Full GPU0、原始 SeqTrack GPU1。"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import shlex
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CT_CONFIG = 'cfgs/ct_seqtrack/41_full_kitti.yaml'
CT_OVERRIDE_KEYS = (
    'path', 'tag', 'batch_size', 'epoch', 'workers', 'seed', 'lr', 'wd',
    'lr_schedule', 'lr_milestones', 'lr_decay_step', 'lr_decay_rate',
    'lr_warmup_steps', 'lr_stage_values', 'ct_pointcloud_cache_bytes',
    'check_val_every_n_epoch', 'accelerator', 'console_mode',
    'progress_refresh_seconds', 'plain_log_interval_seconds', 'log_every_n_steps',
    'diagnostics_level', 'full_diagnostic_epochs', 'tensorboard',
    'v41_memory_logging', 'eval_split', 'checkpoint_every_n_epochs',
    'report_checkpoint', 'report_last_k')


def _ct_configuration(args):
    from models.ct_v41.config import load_config
    path = getattr(args, 'cfg', DEFAULT_CT_CONFIG)
    local_path = Path(path)
    if not local_path.is_absolute():
        local_path = ROOT / local_path
    overrides = {key: getattr(args, key) for key in CT_OVERRIDE_KEYS
                 if getattr(args, key, None) is not None}
    return load_config(local_path, overrides)


def _label(args, arm):
    if arm != 'full':
        return 'seqtrack_original_kitti_car_seed42_60ep_bs16'
    from models.ct_v41.config import run_directory_tag
    config = _ct_configuration(args)
    tag = run_directory_tag(config, explicit_tag=getattr(args, 'tag', None) is not None)
    tag = re.sub(r'[^A-Za-z0-9_.-]+', '_', tag).strip('._-') or 'v41'
    return f'{tag}-seed{config.seed}-{config.epoch}ep-bs{config.batch_size}'


def _ct_arguments(args):
    """只转发显式 CLI；未传项继续由所选 YAML 和默认值解析。"""
    result = ['--cfg', getattr(args, 'cfg', DEFAULT_CT_CONFIG)]
    for key in CT_OVERRIDE_KEYS:
        value = getattr(args, key, None)
        if value is None:
            continue
        if isinstance(value, bool):
            result.append('--' + ('' if value else 'no-') + key)
        elif isinstance(value, (list, tuple)):
            result.append('--' + key)
            result.extend(str(item) for item in value)
        else:
            result.extend(('--' + key, str(value)))
    return shlex.join(result)


def launch_commands(args, arm):
    """日期目录由服务器执行命令时生成；baseline日志与模型空目录分开。"""
    full = arm == 'full'
    config = _ct_configuration(args) if full else None
    gpu = args.gpu if args.gpu is not None else (
        config.devices if full and config.devices is not None else (0 if full else 1))
    root = args.ct_root if full else args.seqtrack_root
    python = args.ct_python if full else args.seqtrack_python
    variable = 'CT_FULL' if full else 'SEQ_BASE'
    label = _label(args, arm)
    arguments = (_ct_arguments(args) if full else
                 '--cfg cfgs/seqtrack3d_kitti_v41_baseline.yaml --path ' +
                 shlex.quote(args.path or '/home/lishengjie/data/cxtrack/training'))
    seed = config.seed if full else 42
    allocator = ('"${PYTORCH_CUDA_ALLOC_CONF:-backend:native,roundup_power2_divisions:2}"'
                 if full else 'backend:native')
    directory = f'"${variable}"' if full else f'"${variable}/run"'
    lines = [f'cd {shlex.quote(root)}', 'mkdir -p output',
        f'{variable}="$(mktemp -d "output/$(date +%Y%m%d-%H%M%S)-{label}-XXXXXX")"',
        f'nohup env CUDA_VISIBLE_DEVICES={gpu} \\',
        f'  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONHASHSEED={seed} \\',
        f'  PYTORCH_CUDA_ALLOC_CONF={allocator} CUBLAS_WORKSPACE_CONFIG=:4096:8 \\',
        f'  {shlex.quote(python)} -u main.py \\',
        f'  {arguments} \\']
    if full:
        lines += [f'  --devices {gpu} --log_dir {directory} \\']
    else:
        # 原入口无--devices参数；只暴露一张卡时其devices=-1仍是单卡。
        lines += [f'  --batch_size 16 --epoch 60 --workers 10 --seed 42 --log_dir {directory} \\']
    lines += [f'  > "${variable}/train.log" 2>&1 < /dev/null &',
              f'echo $! > "${variable}/train.pid"', f'echo "${variable}"']
    return '\n'.join(lines)


def tail_commands(args, arm):
    root = args.ct_root if arm == 'full' else args.seqtrack_root
    label = _label(args, arm)
    return (f'cd {shlex.quote(root)}\n'
            f'RUN="$(ls -td output/[0-9]*-{label}-*/ | head -n 1)"\n'
            'echo "$RUN"\n'
            'tail -n 60 -f "${RUN%/}/train.log"')


def baseline_evaluation_commands(args):
    gpu = args.gpu if args.gpu is not None else 1
    return (f'cd {shlex.quote(args.seqtrack_root)}\n'
        'RUN="$(ls -td output/[0-9]*-seqtrack_original_kitti_car_seed42_60ep_bs16-*/ | head -n 1)"\n'
        f'nohup env CUDA_VISIBLE_DEVICES={gpu} OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONHASHSEED=42 \\\n'
        f'  {shlex.quote(args.seqtrack_python)} -u tools/evaluate_kitti_v41_baseline.py \\\n'
        '  --run-dir "${RUN%/}/run" --output-dir "${RUN%/}/fixed-test" \\\n'
        f'  --path {shlex.quote(args.path or "/home/lishengjie/data/cxtrack/training")} --device {gpu} \\\n'
        '  > "${RUN%/}/fixed-test.log" 2>&1 < /dev/null &\n'
        'echo $! > "${RUN%/}/fixed-test.pid"')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--arm', choices=('full', 'seqtrack', 'all'), default='all')
    parser.add_argument('--gpu', type=int, help='override one chosen arm; default Full=0 / SeqTrack=1')
    parser.add_argument('--cfg', default=DEFAULT_CT_CONFIG, help='CT YAML; baseline keeps its existing config')
    parser.add_argument('--path')
    parser.add_argument('--tag', help='CT run tag; omitted values come from YAML')
    for key in ('batch_size', 'epoch', 'workers', 'seed', 'lr_decay_step', 'lr_warmup_steps',
                'ct_pointcloud_cache_bytes', 'check_val_every_n_epoch', 'log_every_n_steps',
                'checkpoint_every_n_epochs', 'report_last_k'):
        parser.add_argument('--' + key, type=int, help='CT only; overrides YAML when supplied')
    for key in ('lr', 'wd', 'lr_decay_rate', 'progress_refresh_seconds', 'plain_log_interval_seconds'):
        parser.add_argument('--' + key, type=float, help='CT only; overrides YAML when supplied')
    for key in ('lr_milestones', 'full_diagnostic_epochs'):
        parser.add_argument('--' + key, type=int, nargs='*')
    parser.add_argument('--lr_stage_values', type=float, nargs='+')
    parser.add_argument('--lr_schedule', choices=('step', 'multistep', 'piecewise'))
    parser.add_argument('--accelerator', choices=('auto', 'cpu', 'gpu'))
    parser.add_argument('--console_mode', choices=('auto', 'bar', 'plain'))
    parser.add_argument('--diagnostics_level', choices=('summary', 'full'))
    parser.add_argument('--eval_split', choices=('dev', 'test'))
    parser.add_argument('--report_checkpoint', choices=('best_precision', 'last'))
    for key in ('tensorboard', 'v41_memory_logging'):
        parser.add_argument('--' + key, action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--ct-root', default='/home/lishengjie/study/lcyu/CT-SeqTrack')
    parser.add_argument('--seqtrack-root', default='/home/lishengjie/study/lcyu/seqtrack')
    parser.add_argument('--ct-python', default='/home/lishengjie/miniconda3/envs/ctseqtrack3d/bin/python')
    parser.add_argument('--seqtrack-python', default='/home/lishengjie/miniconda3/envs/seqtrack3d/bin/python')
    args = parser.parse_args(argv)
    if args.gpu is not None and (args.gpu < 0 or args.arm == 'all'):
        parser.error('--gpu must be nonnegative and used with --arm full or seqtrack')
    return args


def main(argv=None):
    args = parse_args(argv)
    for arm in (('full', 'seqtrack') if args.arm == 'all' else (args.arm,)):
        print('# ' + arm + ': launch from scratch')
        print(launch_commands(args, arm))
        print('\n# New SSH terminal: follow this arm\'s latest dated run')
        print(tail_commands(args, arm))
        if arm == 'seqtrack':
            print('\n# After this run finishes epoch 60: evaluate fixed 58/59/60')
            print(baseline_evaluation_commands(args))


if __name__ == '__main__':
    main()
