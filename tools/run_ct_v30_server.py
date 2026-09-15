"""生成 v30 单臂服务器启动命令；只有显式 --launch 才启动并新建输出。"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOTS = {
    'mini': '/home/lishengjie/data/nuscenes-mini',
    'full': '/home/lishengjie/code/SparseFusion-main/nuscenes/nuscenes',
    'kitti': '/home/lishengjie/data/cxtrack/training',
}


def build_command(dataset, arm, gpu, *, path=None, log_dir=None, python=None):
    suffix = {'mini': 'mini', 'full': 'nuscenes_full', 'kitti': 'kitti'}[dataset]
    config = ROOT / 'cfgs' / 'ct_seqtrack' / f'30_{arm}_{suffix}.yaml'
    data = str(path or DATA_ROOTS[dataset])
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8]
    output = Path(log_dir) if log_dir else ROOT / 'output' / f'{stamp}-30_{arm}-{dataset}_car_seed42_60ep_bs16'
    if not output.is_absolute():
        output = ROOT / output
    output = output.resolve()
    overrides = dict(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                     OPENBLAS_NUM_THREADS='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                     PYTORCH_CUDA_ALLOC_CONF='backend:native')
    command = [str(python or sys.executable), '-u', 'main.py', '--cfg', str(config), '--path', data,
               '--workers', '4', '--seed', '42', '--log_dir', str(output)]
    return dict(cwd=str(ROOT), environment=overrides, argv=command, log_dir=str(output),
                config=str(config), data_root=data, launch=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=tuple(DATA_ROOTS), required=True)
    parser.add_argument('--arm', choices=('b0', 'full_cfc', 'full_gru'), required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--path')
    parser.add_argument('--log-dir')
    parser.add_argument('--python', help='Use the training environment Python; default is this interpreter')
    parser.add_argument('--launch', action='store_true', help='Create a fresh output and launch on this Linux server')
    args = parser.parse_args(argv)
    if args.gpu < 0:
        parser.error('--gpu must be non-negative')
    result = build_command(args.dataset, args.arm, args.gpu, path=args.path,
                           log_dir=args.log_dir, python=args.python)
    if args.launch:
        if os.name != 'posix':
            parser.error('--launch is for the Linux training server; command preview works on Windows')
        if not Path(result['config']).is_file():
            parser.error('Selected v30 configuration is missing')
        data = Path(result['data_root'])
        required = (('label_02', 'velodyne', 'calib') if args.dataset == 'kitti' else
                    ('v1.0-mini' if args.dataset == 'mini' else 'v1.0-trainval', 'samples/LIDAR_TOP'))
        if not all((data / name).is_dir() for name in required):
            parser.error('Data root is missing a required directory: ' + ', '.join(required))
        output = Path(result['log_dir'])
        # Refuse existing outputs; this command never resumes or initializes from another run.
        output.mkdir(parents=True, exist_ok=False)
        environment = dict(os.environ)
        environment.update(result['environment'])
        with (output / 'train.log').open('xb') as log:
            process = subprocess.Popen(result['argv'], cwd=result['cwd'], env=environment,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, close_fds=True)
        (output / 'train.pid').write_text(str(process.pid) + '\n', encoding='ascii')
        result.update(launch=True, pid=process.pid)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == '__main__':
    main()
