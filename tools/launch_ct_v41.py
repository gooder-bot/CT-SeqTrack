"""只打印独立 nohup/tail 命令；默认 Full GPU0、原始 SeqTrack GPU1。"""
from __future__ import annotations

import argparse
import shlex


def launch_commands(args, arm):
    """日期目录由服务器执行命令时生成；baseline日志与模型空目录分开。"""
    full = arm == 'full'
    gpu = args.gpu if args.gpu is not None else (0 if full else 1)
    root = args.ct_root if full else args.seqtrack_root
    python = args.ct_python if full else args.seqtrack_python
    variable = 'CT_FULL' if full else 'SEQ_BASE'
    label = '41_full_kitti_car_seed42_60ep_bs16' if full else 'seqtrack_original_kitti_car_seed42_60ep_bs16'
    config = 'cfgs/ct_seqtrack/41_full_kitti.yaml' if full else 'cfgs/seqtrack3d_kitti_v41_baseline.yaml'
    allocator = 'backend:native,roundup_power2_divisions:2' if full else 'backend:native'
    directory = f'"${variable}"' if full else f'"${variable}/run"'
    lines = [f'cd {shlex.quote(root)}', 'mkdir -p output',
        f'{variable}="$(mktemp -d "output/$(date +%Y%m%d-%H%M%S)-{label}-XXXXXX")"',
        f'nohup env CUDA_VISIBLE_DEVICES={gpu} \\',
        '  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONHASHSEED=42 \\',
        f'  PYTORCH_CUDA_ALLOC_CONF={allocator} CUBLAS_WORKSPACE_CONFIG=:4096:8 \\',
        f'  {shlex.quote(python)} -u main.py \\',
        f'  --cfg {config} \\',
        f'  --path {shlex.quote(args.path)} \\']
    if full:
        lines += ['  --batch_size 16 --epoch 60 --workers 4 --check_val_every_n_epoch 1 \\',
                  f'  --devices {gpu} --accelerator gpu --trainer_devices 1 --log_dir {directory} \\']
    else:
        # 原入口无--devices参数；只暴露一张卡时其devices=-1仍是单卡。
        lines += [f'  --batch_size 16 --epoch 60 --workers 10 --seed 42 --log_dir {directory} \\']
    lines += [f'  > "${variable}/train.log" 2>&1 < /dev/null &',
              f'echo $! > "${variable}/train.pid"', f'echo "${variable}"']
    return '\n'.join(lines)


def tail_commands(args, arm):
    root = args.ct_root if arm == 'full' else args.seqtrack_root
    label = '41_full_kitti_car_seed42_60ep_bs16' if arm == 'full' else 'seqtrack_original_kitti_car_seed42_60ep_bs16'
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
        f'  --path {shlex.quote(args.path)} --device {gpu} \\\n'
        '  > "${RUN%/}/fixed-test.log" 2>&1 < /dev/null &\n'
        'echo $! > "${RUN%/}/fixed-test.pid"')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--arm', choices=('full', 'seqtrack', 'all'), default='all')
    parser.add_argument('--gpu', type=int, help='override one chosen arm; default Full=0 / SeqTrack=1')
    parser.add_argument('--path', default='/home/lishengjie/data/cxtrack/training')
    parser.add_argument('--ct-root', default='/home/lishengjie/study/lcyu/CT-SeqTrack')
    parser.add_argument('--seqtrack-root', default='/home/lishengjie/study/lcyu/seqtrack')
    parser.add_argument('--ct-python', default='/home/lishengjie/miniconda3/envs/ctseqtrack3d/bin/python')
    parser.add_argument('--seqtrack-python', default='/home/lishengjie/miniconda3/envs/seqtrack3d/bin/python')
    args = parser.parse_args(argv)
    if args.gpu is not None and (args.gpu < 0 or args.arm == 'all'):
        parser.error('--gpu must be nonnegative and used with --arm full or seqtrack')
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
