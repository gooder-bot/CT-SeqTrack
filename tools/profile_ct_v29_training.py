"""v29 同卡 ABBA 短测速/三卡四波并行，或 AB 逐位对照；不是全套哈希门禁。"""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from utils.v29_profiling import AUDIT_STEPS, first_difference, summarize

ARMS = ('b0', 'full_cfc', 'full_gru')


def build_plan(data_root, output, *, gpu='1', arms=ARMS, mode='benchmark',
               workers=4, warmup=20, steps=100, python=sys.executable,
               parallel=False, gpus=None):
    if mode not in ('benchmark', 'profile', 'equivalence'):
        raise ValueError('unknown performance mode')
    if parallel:
        if mode != 'benchmark':
            raise ValueError('parallel execution is only supported for benchmark')
        if gpus is None or len(gpus) != len(arms):
            raise ValueError('parallel requires one physical GPU per arm')
        devices = [str(value) for value in gpus]
        if len(set(devices)) != len(devices) or any(not value.isdigit() for value in devices):
            raise ValueError('parallel GPUs must be distinct physical GPU indices')
    else:
        if gpus is not None or not str(gpu).isdigit():
            raise ValueError('sequential short comparison requires one physical GPU via --gpu')
        devices = [str(gpu)] * len(arms)
    if not 0 <= warmup < steps <= 100:
        raise ValueError('require 0 <= warmup < steps <= 100')
    if mode == 'equivalence' and steps != 100:
        raise ValueError('equivalence requires all 100 steps')
    if workers < 0 or not arms or len(set(arms)) != len(arms) or any(arm not in ARMS for arm in arms):
        raise ValueError('invalid arm or workers')
    output = Path(output).resolve()
    if (ROOT / 'artifacts' / 'ct_checks').resolve() not in output.parents:
        raise ValueError('short checks must write below artifacts/ct_checks')
    phases, waves = [], []
    sequence = ([('legacy', 1), ('optimized', 1), ('optimized', 2), ('legacy', 2)]
                if mode == 'benchmark' else [('legacy', 1), ('optimized', 1)])
    schedules = ([[ (arm, device, variant, repeat) for arm, device in zip(arms, devices)]
                   for variant, repeat in sequence] if parallel else
                 [[(arm, device, variant, repeat)] for arm, device in zip(arms, devices)
                  for variant, repeat in sequence])
    for wave_index, schedule in enumerate(schedules):
        wave = []
        for arm, device, variant, repeat in schedule:
            suffix = '_perf' if variant == 'optimized' else ''
            cfg = ROOT / 'cfgs' / 'ct_seqtrack' / f'29_{arm}_nuscenes_full{suffix}.yaml'
            name = arm + '_' + variant + (str(repeat) if mode == 'benchmark' else '')
            run = output / name
            argv = [python, '-u', str(ROOT / 'main.py'), '--cfg', str(cfg),
                '--path', str(data_root), '--batch_size', '16', '--epoch', '1',
                '--workers', str(workers), '--seed', '42', '--ct_engineering_check',
                '--limit_train_batches', str(steps), '--limit_val_batches', '1',
                '--check_val_every_n_epoch', '5', '--log_dir', str(run),
                '--tag', f'v29-{mode}-{name}']
            phases.append(dict(name=name, arm=arm, variant=variant, repeat=repeat,
                gpu=device, wave=wave_index + 1,
                config=str(cfg), argv=argv, output=str(run), log=str(run / 'train.log'),
                environment=dict(CUDA_VISIBLE_DEVICES=device, OMP_NUM_THREADS='1',
                    MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                    CUBLAS_WORKSPACE_CONFIG=':4096:8',
                    PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:64',
                    CT_V29_PROFILE_DIR=str(run), CT_V29_PROFILE_MODE=mode,
                    CT_V29_PROFILE_WARMUP=str(warmup), CT_V29_PROFILE_STEPS=str(steps))))
            wave.append(name)
        waves.append(wave)
    return dict(schema='ct_seqtrack.performance_plan.v29', mode=mode, gpu=str(gpu),
                output=str(output), warmup=warmup, steps=steps, arms=list(arms),
                parallel=bool(parallel), gpus=devices if parallel else [str(gpu)],
                schedule='ABBA' if mode == 'benchmark' else 'AB', waves=waves,
                phases=phases, formal_initialization_allowed=False)


def compare_compact(left, right, *, shared_b0=False):
    for step in (0, *AUDIT_STEPS):
        paths = [Path(root) / 'snapshots' / f'step_{step:03d}.pt' for root in (left, right)]
        if not all(path.is_file() for path in paths):
            return dict(passed=False, step=step, difference='required snapshot missing')
        a, b = [torch.load(path, map_location='cpu', weights_only=False) for path in paths]
        if step and any(not row.get('transactions') for row in (a, b)):
            return dict(passed=False, step=step, difference='required transaction input/loss/output snapshot missing')
        if shared_b0:
            a, b = [dict(state=row['state']['shared_b0'],
                         observation=[item for item in row['transactions'] if not item['mechanism']])
                    for row in (a, b)]
        difference = first_difference(a, b)
        if difference:
            return dict(passed=False, step=step, difference=difference)
    return dict(passed=True, acceptance='tensor_bytes_equal', steps=[0, *AUDIT_STEPS], shared_b0_only=shared_b0,
        scope='all model/BN/buffer/optimizer/RNG/recursive states and selected transaction inputs/losses/outputs',
        excluded='only ct_h3_* and ct_shadow_* diagnostic input fields; timing and logging summaries are not model state')


def _run_metrics(runtime):
    cycle = runtime['cycle']
    mean = cycle.get('mean_ms')
    return dict(mean_ms=mean, p50_ms=cycle.get('p50_ms'), p90_ms=cycle.get('p90_ms'),
                iterations_per_second=1000. / mean if mean else None,
                samples_per_second=16000. / mean if mean else None,
                measured_steps=runtime.get('measured_steps'))


def summarize_repeats(runs):
    """报告实际重复分布；不把两次试验的抖动解释为可靠提速。"""
    repetitions = [dict(name=run['name'], **_run_metrics(run['runtime'])) for run in runs]
    values = [row['cycle_ms'] for run in runs for row in run['runtime'].get('rows', [])]
    means = [run['mean_ms'] for run in repetitions]
    valid = len(means) == 2 and all(value is not None and value > 0 for value in means)
    difference = abs(means[1] - means[0]) if valid else None
    center = sum(means) / len(means) if valid else None
    pooled = summarize(values)
    mean = pooled.get('mean_ms')
    pooled.update(iterations_per_second=1000. / mean if mean else None,
                  samples_per_second=16000. / mean if mean else None)
    return dict(repetitions=repetitions, pooled=pooled,
                repeat_difference_ms=difference,
                repeat_difference_percent=100. * difference / center if valid else None,
                repetition_mean_range_ms=[min(means), max(means)] if valid else None)


def benchmark_comparison(runs):
    groups = {variant: summarize_repeats([run for run in runs if run['variant'] == variant])
              for variant in ('legacy', 'optimized')}
    baseline, optimized = groups['legacy'], groups['optimized']
    original_mean, optimized_mean = (row['pooled']['mean_ms'] for row in (baseline, optimized))
    ranges = [row['repetition_mean_range_ms'] for row in (baseline, optimized)]
    available = all(value is not None and value > 0 for value in (original_mean, optimized_mean))
    delta = original_mean - optimized_mean if available else None
    variation = (max(baseline['repeat_difference_ms'], optimized['repeat_difference_ms'])
                 if all(value is not None for value in ranges) else None)
    consistent = bool(available and variation is not None
                      and ranges[1][1] < ranges[0][0] and delta > variation)
    return dict(**groups, observed_speedup=original_mean / optimized_mean if available else None,
                reduction_mean_ms=delta, within_variant_variation_ms=variation,
                consistent_measured_reduction=consistent,
                interpretation=('两次优化均更快且均值改善超过观测重复差异；这是短测观测，非统计保证。'
                                if consistent else '未证明改善超出重复波动；不能据此宣称加速。'),
                formal_initialization_allowed=False)


def _stop_owned_children(children):
    """只处理本工具创建的独立 session；不按 GPU/命令匹配或终止外部任务。"""
    for _, process, _ in children:
        try:
            if os.name == 'posix':
                os.killpg(process.pid, signal.SIGTERM)
            elif process.poll() is None:
                process.terminate()
        except ProcessLookupError:
            pass
    for _, process, _ in children:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        try:
            # 同一独立 session 中可能还有该训练的 DataLoader 子进程。
            if os.name == 'posix':
                os.killpg(process.pid, signal.SIGKILL)
            elif process.poll() is None:
                process.kill()
        except ProcessLookupError:
            pass
        if process.poll() is None:
            process.wait(timeout=5)


def execute_plan(plan):
    from tools.run_ct_v29_checks import require_idle_gpu
    output = Path(plan['output'])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('short comparison requires a new empty directory')
    output.mkdir(parents=True, exist_ok=True)
    report = dict(schema='ct_seqtrack.performance_comparison.v29', mode=plan['mode'],
                  parallel=plan.get('parallel', False), gpus=plan.get('gpus', [plan['gpu']]),
                  schedule=plan.get('schedule', 'AB'),
                  status='running', runs=[], comparisons={}, formal_initialization_allowed=False)
    report_path = output / 'report.json'

    def save():
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                         allow_nan=False) + '\n', encoding='utf-8')

    save()
    (output / 'plan.json').write_text(json.dumps(plan, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    phase_map = {phase['name']: phase for phase in plan['phases']}
    waves = plan.get('waves', [[phase['name']] for phase in plan['phases']])
    children = []
    try:
        for wave in waves:
            phases = [phase_map[name] for name in wave]
            # 先检查整波所有 GPU，检查通过后才启动任意一个子任务。
            for phase in phases:
                if not Path(phase['config']).is_file():
                    raise FileNotFoundError(phase['config'])
            for gpu in dict.fromkeys(phase.get('gpu', plan['gpu']) for phase in phases):
                require_idle_gpu(gpu)
            children = []
            for phase in phases:
                target = Path(phase['output'])
                target.mkdir(parents=True, exist_ok=True)
                environment = os.environ.copy()
                for key in list(environment):
                    if key.startswith(('CT_V28_AUDIT_', 'CT_V29_PROFILE_', 'CT_V29_H3_BENCH_')):
                        environment.pop(key)
                environment.update(phase['environment'])
                print(f"[{plan['mode']}] {phase['name']} -> {phase['log']}", flush=True)
                log = Path(phase['log']).open('w', encoding='utf-8')
                try:
                    process = subprocess.Popen(phase['argv'], cwd=ROOT, env=environment,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=os.name == 'posix',
                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0) if os.name == 'nt' else 0)
                except BaseException:
                    log.close()
                    raise
                children.append((phase, process, log))
            pending = list(children)
            while pending:
                for child in list(pending):
                    phase, process, log = child
                    code = process.poll()
                    if code is None:
                        continue
                    log.close()
                    runtime_path = Path(phase['output']) / 'runtime.json'
                    runtime = (json.loads(runtime_path.read_text(encoding='utf-8'))
                               if runtime_path.is_file() else None)
                    report['runs'].append(dict(name=phase['name'], arm=phase.get('arm'),
                        variant=phase.get('variant'), repeat=phase.get('repeat', 1),
                        gpu=phase.get('gpu', plan['gpu']), returncode=code, runtime=runtime,
                        metrics=_run_metrics(runtime) if runtime and runtime.get('status') == 'complete' else None))
                    save()
                    if code or runtime is None or runtime.get('status') != 'complete':
                        report['failure'] = phase['name']
                        raise RuntimeError('training phase failed or runtime artifact is incomplete')
                    pending.remove(child)
                if pending:
                    time.sleep(.2)
            children = []
        for arm in plan['arms']:
            runs = [run for run in report['runs'] if run['arm'] == arm]
            if plan['mode'] == 'benchmark':
                report['comparisons'][arm] = benchmark_comparison(runs)
            elif plan['mode'] == 'equivalence':
                comparison = compare_compact(output / (arm + '_legacy'), output / (arm + '_optimized'))
                report['comparisons'][arm] = comparison
                if not comparison['passed']:
                    report['failure'] = arm
                    raise RuntimeError('numerical equivalence failed')
            else:
                report['comparisons'][arm] = dict(
                    legacy=_run_metrics(runs[0]['runtime']), optimized=_run_metrics(runs[1]['runtime']),
                    diagnostic_only=True, interpretation='同步插桩 AB 仅用于定位，不作为速度改善证据。')
            save()
    except BaseException as error:
        report['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        report['error'] = str(error)
        try:
            _stop_owned_children(children)
        finally:
            for _, _, log in children:
                log.close()
            save()
        if isinstance(error, KeyboardInterrupt):
            raise
        return report
    report['status'] = 'complete'
    if plan['mode'] == 'equivalence' and 'b0' in plan['arms']:
        for arm in plan['arms']:
            if arm == 'b0':
                continue
            comparison = compare_compact(output / 'b0_optimized', output / (arm + '_optimized'), shared_b0=True)
            report['comparisons']['shared_b0_' + arm] = comparison
            if not comparison['passed']:
                report['status'] = 'failed'
                break
    save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--path', required=True)
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--parallel', action='store_true', help='benchmark only: run arms in parallel for each ABBA wave')
    parser.add_argument('--gpus', nargs='+', help='one distinct physical GPU per arm, in --arms order')
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    parser.add_argument('--mode', choices=('benchmark', 'profile', 'equivalence'), default='benchmark')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    plan = build_plan(args.path, args.output, gpu=args.gpu, arms=args.arms,
        mode=args.mode, workers=args.workers, warmup=args.warmup, steps=args.steps,
        parallel=args.parallel, gpus=args.gpus)
    if args.plan_only:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    report = execute_plan(plan)
    print(json.dumps(report['comparisons'], ensure_ascii=False, indent=2))
    if report['status'] != 'complete':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
