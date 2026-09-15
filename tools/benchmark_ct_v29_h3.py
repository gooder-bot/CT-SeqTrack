"""通过 main.py 的真实 Full 通路寻找合法 H3 事件，执行独立微基准。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def build_command(data_root, output, arm='full_gru', gpu='1', repeats=5, config_path=None):
    output = Path(output).resolve()
    if (ROOT / 'artifacts/ct_checks').resolve() not in output.parents:
        raise ValueError('H3 benchmark output must be below artifacts/ct_checks')
    if arm not in ('full_gru', 'full_cfc') or not str(gpu).isdigit():
        raise ValueError('H3 benchmark requires one GPU and one Full arm')
    if not 1 <= repeats <= 50:
        raise ValueError('repeats must be in [1,50]')
    config_path = (Path(config_path).resolve() if config_path else
                   ROOT / f'cfgs/ct_seqtrack/29_{arm}_nuscenes_full_perf.yaml')
    from utils.config import load_yaml_config
    config = load_yaml_config(config_path)
    v30 = bool(config.get('ct_enable_v30', False))
    if v30:
        from models.ct_variant import configure_ct_variant
        from utils.online_contract import validate_scratch_training_contract
        configure_ct_variant(config)
        validate_scratch_training_contract(config)
        if config.get('ct_variant') != 'full' or config.get('motion_v3_temporal_backend') != arm.removeprefix('full_'):
            raise ValueError('v30 H3 check requires the matching registered Full arm')
    return dict(output=str(output), gpu=str(gpu), argv=[sys.executable, '-u', str(ROOT / 'main.py'),
        '--cfg', str(config_path),
        '--path', str(data_root), '--batch_size', '16', '--epoch', '1', '--workers', '4',
        '--seed', '42', '--ct_engineering_check', '--limit_train_batches', '100',
        '--limit_val_batches', '1', '--check_val_every_n_epoch', '1' if v30 else '5',
        '--log_dir', str(output), '--tag', ('v30' if v30 else 'v29') + '-h3-microbenchmark-' + arm],
        environment=dict(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
            OPENBLAS_NUM_THREADS='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
            PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:64',
            CT_V29_H3_BENCH_DIR=str(output), CT_V29_H3_BENCH_REPEATS=str(repeats),
            CT_V29_H3_BENCH_WARMUP='2'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--path', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--arm', choices=('full_gru', 'full_cfc'), default='full_gru')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--cfg', help='Optional registered v30 Full configuration')
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    plan = build_command(args.path, args.output, args.arm, args.gpu, args.repeats, args.cfg)
    if args.plan_only:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    from tools.run_ct_v29_checks import require_idle_gpu
    require_idle_gpu(args.gpu)
    output = Path(plan['output'])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('H3 benchmark requires a new empty directory')
    output.mkdir(parents=True, exist_ok=True)
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(('CT_V28_AUDIT_', 'CT_V29_PROFILE_', 'CT_V29_H3_BENCH_'))}
    environment.update(plan['environment'])
    with (output / 'train.log').open('w', encoding='utf-8') as log:
        result = subprocess.run(plan['argv'], cwd=ROOT, env=environment,
                                stdout=log, stderr=subprocess.STDOUT, check=False)
    report = output / 'h3_microbenchmark.json'
    if result.returncode or not report.is_file():
        raise SystemExit('H3 check incomplete: training failed or no legal event in the 100-step search; '
                         'this is not a pass. Inspect ' + str(output / 'train.log'))
    print(report.read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()
