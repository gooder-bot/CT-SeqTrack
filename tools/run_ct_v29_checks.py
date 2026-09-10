"""v29 单卡串行工程验收；只复用生产入口，不启动正式训练。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ARMS = {
    'b0': 'cfgs/ct_seqtrack/29_b0_nuscenes_full.yaml',
    'full_cfc': 'cfgs/ct_seqtrack/29_full_cfc_nuscenes_full.yaml',
    'full_gru': 'cfgs/ct_seqtrack/29_full_gru_nuscenes_full.yaml',
}


def source_identity():
    from utils.action_calibration_v27 import code_content_sha256, sha256_file, sha256_json
    from utils.config import load_yaml_config
    check_tools = ('run_ct_v29_checks.py', 'preflight_ct_v28.py', 'check_train_steps.py',
                   'compare_ct_v28_audits.py', 'check_ct_v28_resume.py')
    return dict(code_sha256=code_content_sha256(), configs={
        arm: sha256_json(load_yaml_config(ROOT / path)) for arm, path in ARMS.items()},
        check_tools={name: sha256_file(ROOT / 'tools' / name) for name in check_tools})


def build_plan(data_path, output, *, gpu='1', workers=12, python=sys.executable):
    """纯控制面构造；不加载模型、数据或 CUDA。"""
    from utils.config import load_yaml_config
    if not str(gpu).isdigit():
        raise ValueError('--gpu must be one physical GPU index')
    if int(workers) < 0:
        raise ValueError('--workers must be non-negative')
    target = Path(output).resolve()
    allowed = (ROOT / 'artifacts/ct_checks').resolve()
    if target == allowed or allowed not in target.parents:
        raise ValueError('engineering output must be a new child of artifacts/ct_checks/')
    for path in ARMS.values():
        config = load_yaml_config(ROOT / path)
        if not config.get('ct_enable_v29') or config.get('category_name') != 'Car':
            raise ValueError('engineering matrix requires the registered v29 Car configs')
    phases = []
    def add(name, argv, gpu_work=True):
        phases.append(dict(name=name, argv=[str(value) for value in argv],
                           log=str(target / (name + '.log')), gpu_work=gpu_work))
    add('preflight', [python, '-u', ROOT / 'tools/preflight_ct_v28.py',
        '--cfg', ROOT / ARMS['full_cfc'], '--path', data_path,
        '--device', 'cuda', '--output', target / 'preflight.json'])
    for name, arm in (('b0_a', 'b0'), ('b0_b', 'b0'),
                      ('full_cfc', 'full_cfc'), ('full_gru', 'full_gru')):
        add(name, [python, '-u', ROOT / 'tools/check_train_steps.py',
            '--cfg', ROOT / ARMS[arm], '--path', data_path,
            '--steps', 100, '--workers', workers, '--seed', 42,
            '--numerical-audit', '--tag', 'v29-engineering-' + name,
            '--artifact-dir', target / name])
    for name in ('b0_b', 'full_cfc', 'full_gru'):
        add('compare_' + name, [python, ROOT / 'tools/compare_ct_v28_audits.py',
            target / 'b0_a/numerical_audit', target / name / 'numerical_audit',
            '--output', target / ('compare_' + name + '.json')], gpu_work=False)
    for arm, config in ARMS.items():
        add('resume_' + arm, [python, '-u', ROOT / 'tools/check_ct_v28_resume.py',
            '--cfg', ROOT / config, '--path', data_path, '--steps', 16,
            '--gpu', gpu, '--python', python, '--output', target / ('resume_' + arm)])
    return dict(schema='ct_seqtrack.engineering_plan.v29', gpu=str(gpu),
                output=str(target), phases=phases, source_identity=source_identity(),
                engineering_only=True, formal_initialization_allowed=False)


def require_idle_gpu(gpu):
    """发现任意现有 CUDA compute 进程就退出，不终止用户任务。"""
    query = subprocess.run(['nvidia-smi', '-i', str(gpu), '--query-gpu=uuid',
                            '--format=csv,noheader,nounits'], check=True,
                           capture_output=True, text=True)
    uuids = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    if len(uuids) != 1:
        raise RuntimeError('expected exactly one physical GPU UUID')
    apps = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                           '--format=csv,noheader,nounits'], check=True,
                          capture_output=True, text=True)
    busy = [line.strip() for line in apps.stdout.splitlines()
            if line.split(',', 1)[0].strip() == uuids[0]]
    if busy:
        raise RuntimeError('engineering GPU is not exclusive: ' + '; '.join(busy))
    return uuids[0]


def assert_passed(path):
    report = json.loads(Path(path).read_text(encoding='utf-8'))
    if (report.get('schema') != 'ct_seqtrack.engineering_report.v29'
            or report.get('status') != 'passed' or not report.get('passed')):
        raise RuntimeError('v29 engineering checks have not passed')
    if report.get('source_identity') != source_identity():
        raise RuntimeError('code/config changed after the engineering checks; rerun the checks')
    return report


def execute_plan(plan, *, plan_only=False):
    target = Path(plan['output'])
    if target.exists() and any(target.iterdir()):
        raise FileExistsError('engineering output must be empty: ' + str(target))
    target.mkdir(parents=True, exist_ok=True)
    (target / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n', encoding='utf-8')
    report = dict(schema='ct_seqtrack.engineering_report.v29',
                  status='planned' if plan_only else 'running', passed=False,
                  source_identity=plan['source_identity'], gpu=plan['gpu'], phases=[],
                  engineering_only=True, formal_initialization_allowed=False)
    def save():
        (target / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    save()
    if plan_only:
        return report
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=plan['gpu'], CUBLAS_WORKSPACE_CONFIG=':4096:8',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:64')
    environment.pop('CT_V28_AUDIT_DIR', None)
    environment.pop('CT_V28_AUDIT_ACTIVATIONS', None)
    try:
        for phase in plan['phases']:
            if phase['gpu_work']:
                report['gpu_uuid'] = require_idle_gpu(plan['gpu'])
            print('[v29 engineering] ' + phase['name'] + ' -> ' + phase['log'], flush=True)
            with Path(phase['log']).open('w', encoding='utf-8') as log:
                completed = subprocess.run(phase['argv'], cwd=ROOT, env=environment,
                    stdout=log, stderr=subprocess.STDOUT, check=False)
            report['phases'].append(dict(name=phase['name'], exit_code=completed.returncode))
            save()
            if completed.returncode:
                raise RuntimeError(phase['name'] + ' failed; inspect ' + phase['log'])
        if source_identity() != plan['source_identity']:
            raise RuntimeError('source/config changed while the checks were running')
        report.update(status='passed', passed=True)
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    save()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--path')
    parser.add_argument('--output')
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--assert-passed', metavar='REPORT')
    args = parser.parse_args(argv)
    if args.assert_passed:
        assert_passed(args.assert_passed)
        print('v29 engineering passed for the current source/config')
        return 0
    if not args.path or not args.output:
        parser.error('--path and --output are required')
    plan = build_plan(args.path, args.output, gpu=args.gpu, workers=args.workers)
    report = execute_plan(plan, plan_only=args.plan_only)
    print(json.dumps({'status': report['status'], 'report': str(Path(args.output) / 'report.json')}))
    return 0 if args.plan_only or report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
