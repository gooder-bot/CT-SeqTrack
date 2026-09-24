"""服务器启动三份 v33 scratch 配方；复用已完成的 SeqTrack，不覆盖既有运行。"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.ct_v31.config import load_config, config_identity


CONFIGS = ('33_b0_mini.yaml', '33_b0_late_decay_mini.yaml', '33_b0_half_lr_mini.yaml')


def read_baseline(directory, model):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / 'run_manifest.json').read_text(encoding='utf-8'))
    budget = json.loads((directory / 'training_budget.json').read_text(encoding='utf-8'))
    results = json.loads((directory / 'results.json').read_text(encoding='utf-8'))
    if manifest['model'] != model or manifest['seed'] != 42:
        raise ValueError('baseline model/seed mismatch: ' + str(directory))
    if (not budget['epoch_complete'] or budget['completed_epoch'] != 60
            or budget['optimizer_steps'] != 71700 or results['checkpoint_epochs'] != [58, 59, 60]):
        raise ValueError('baseline needs complete scratch60 and late-3 results: ' + str(directory))
    if model == 'seqtrack_reference':
        # 参考数学、teacher、原始数据与评分路径保持冻结；host的v33身份/日志不在此列表。
        protected = ('models/seqtrack_reference/', 'datasets/')
        for relative, expected in manifest['source']['files'].items():
            if relative.startswith(protected) or relative in (
                    'utils/tracking_metrics.py', 'utils/config.py', 'utils/bn_policy.py'):
                actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
                if actual != expected:
                    raise ValueError('reference effective source changed; recheck reuse: ' + relative)
    return dict(directory=str(directory), config_sha256=manifest['config_sha256'],
                final=results['final'], late3=results['late3'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--path', required=True, help='nuScenes mini data root')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--old-b0', type=Path, required=True)
    parser.add_argument('--gpus', default='0,1,1', help='A,B,C physical GPU IDs')
    parser.add_argument('--runs-file', type=Path, required=True)
    args = parser.parse_args(argv)
    if os.name != 'posix':
        raise RuntimeError('launch this tool in the Linux training environment')
    gpus = args.gpus.split(',')
    if len(gpus) != 3 or any(not item.isdigit() for item in gpus):
        raise ValueError('--gpus requires three numeric IDs, e.g. 0,1,1')
    if args.runs_file.exists():
        raise FileExistsError('runs-file already exists; inspect its PIDs before launching again')
    baseline = read_baseline(args.reference, 'seqtrack_reference')
    old_b0 = read_baseline(args.old_b0, 'ctseqtrackv32')
    configs = [load_config(ROOT / 'cfgs' / 'ct_seqtrack' / name, {'path': args.path}) for name in CONFIGS]
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.runs_file.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    record = dict(schema='ct_seqtrack.v33.launch.v1', source_root=str(ROOT),
                  python=sys.executable, reference=baseline, old_b0=old_b0, runs=[])
    # 先独占登记本次启动，再逐进程保存PID，失败时已启动的任务仍可定位。
    with args.runs_file.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
    for label, gpu, name, config in zip(('A', 'B', 'C'), gpus, CONFIGS, configs):
        directory = (args.output_root / f'{stamp}-33_{label}-{config.tag}').resolve()
        directory.mkdir(exist_ok=False)
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='1',
                           MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                           PYTORCH_CUDA_ALLOC_CONF='backend:native', CUBLAS_WORKSPACE_CONFIG=':4096:8')
        command = [sys.executable, '-u', 'main.py', '--cfg', f'cfgs/ct_seqtrack/{name}',
                   '--path', args.path, '--accelerator', 'gpu', '--log_dir', str(directory)]
        with (directory / 'train.log').open('x', encoding='utf-8') as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=environment,
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        (directory / 'train.pid').write_text(str(process.pid) + '\n', encoding='utf-8')
        record['runs'].append(dict(label=label, gpu=gpu, pid=process.pid,
            directory=str(directory), config_sha256=config_identity(config), command=command))
        args.runs_file.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(record['runs'][-1], ensure_ascii=False), flush=True)
    return record


if __name__ == '__main__':
    main()
