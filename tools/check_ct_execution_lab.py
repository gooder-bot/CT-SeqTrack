"""v37 独立工程实验：先严格去重，再单独评估 FP32 投影分解；不保存训练权重。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
BASE_REVISION = '0662c21ab99a44b64a3be807c3668f66a8ace196'
# 只用于确认只读基线源码，统一CRLF/LF；正式 source/config/resume 校验完全不变。
BASE_CANONICAL_BUNDLE = '0fd32de6b2874ab9c492462b8ff9b91351df05619aec5e0057961bfde3b24060'
ARMS = ('b0', 'b1', 'b1_b2', 'full')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('check', 'benchmark', 'trace', 'all'), default='check')
    parser.add_argument('--path', help='nuScenes真实数据根，不能是Python包目录')
    parser.add_argument('--baseline-root', type=Path, help='0662c21原源码只读副本；用于关闭优化的原版重放')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--output', type=Path, help='新的 artifacts/ct_checks/.../report.json')
    parser.add_argument('--check-report', type=Path, help='benchmark/trace必须读取当前代码、设备、数据的已完成check')
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    parser.add_argument('--epochs', nargs='+', type=int, default=[0, 10])
    parser.add_argument('--check-steps', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--trace-steps', type=int, default=5)
    parser.add_argument('--synthetic', action='store_true', help='仅CPU合同smoke；明确缩小点数和batch，绝不作为真实验收')
    parser.add_argument('--baseline-worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if min(args.check_steps, args.steps, args.repeats, args.trace_steps) < 1 or args.warmup < 0:
        parser.error('steps/repeats/trace-steps必须为正；warmup不能为负')
    if not args.epochs or any(epoch not in (0, 10) for epoch in args.epochs) or len(set(args.epochs)) != len(args.epochs):
        parser.error('epochs仅支持不重复的0和10')
    if len(set(args.arms)) != len(args.arms):
        parser.error('arms不能重复')
    if args.synthetic and (args.device != 'cpu' or args.mode != 'check'):
        parser.error('synthetic仅允许CPU check')
    if args.mode in ('check', 'all') and not args.baseline_root:
        parser.error('check/all需要--baseline-root；禁止用修改后的代码自证原版等价')
    if args.mode in ('benchmark', 'trace') and not args.check_report:
        parser.error('benchmark/trace需要--check-report')
    if args.baseline_worker and (not args.output or not args.baseline_root):
        parser.error('baseline worker requires root and output')
    return args


def write_json(path, content):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(content, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def configs(args, root=ROOT):
    from tools.check_execution_performance import load_engineering_config
    result = {}
    for arm in args.arms:
        cfg = load_engineering_config(root / f'cfgs/ct_seqtrack/37_{arm}_mini.yaml', args.path)
        if args.synthetic:
            cfg.update(point_sample_size=8, batch_size=2, workers=0, v32_reserve_windows=0)
        elif cfg.batch_size != 16 or cfg.point_sample_size != 1024 or cfg.precision != 32:
            raise ValueError('真实工程协议要求batch16、1024点、FP32；缩小工作量只能用--synthetic')
        result[arm] = cfg
    return result


def configuration_identities(config_map):
    from models.ct_v37.config import config_identity
    return {arm: config_identity(cfg) for arm, cfg in config_map.items()}


def synthetic_sources(args):
    if not args.synthetic:
        return None
    from tests.test_ct_v31_runtime import TinySource
    return {'train': TinySource((24, 21, 18))}


def runtime_environment(device):
    import importlib.metadata
    import torch
    packages = {}
    for name in ('numpy', 'pytorch-lightning'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    value = dict(python=sys.version, torch=torch.__version__, packages=packages,
        cuda=torch.version.cuda, device=device, threads=torch.get_num_threads(),
        deterministic=torch.are_deterministic_algorithms_enabled(),
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA不可用；本机CPU结果不能替代服务器验收')
        value['gpu'] = torch.cuda.get_device_name()
        value['capability'] = list(torch.cuda.get_device_capability())
    return value


def _baseline_worker(args):
    """新入口脚本驱动原源码；import优先级只指向原仓库，禁止加载候选模型。"""
    original = args.baseline_root.resolve()
    sys.path[:] = [str(original)] + [p for p in sys.path if Path(p or '.').resolve() not in (ROOT, ROOT / 'tools')]
    spec = importlib.util.spec_from_file_location('_lab_replay_driver', ROOT / 'tools/execution_lab_runtime.py')
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    from models.ct_v37.entry import configure_numerics, source_identity
    import models.ct_v37.model as model_module
    if original not in Path(model_module.__file__).resolve().parents:
        raise RuntimeError('baseline worker imported a candidate module')
    configure_numerics()
    if driver.bundle_identity(original, normalize_newlines=True)['sha256'] != BASE_CANONICAL_BUNDLE:
        raise ValueError('baseline代码不是锁定的0662c21版本')
    report = dict(status='passed', environment=runtime_environment(args.device), source=source_identity(), arms={})
    for arm, cfg in configs(args, original).items():
        report['arms'][arm] = driver.replay_disabled(cfg, args.device, steps=args.check_steps,
            epochs=args.epochs, sources=synthetic_sources(args))
    write_json(args.output, report)
    return 0


def verify_disabled(args, destination, config_map):
    from tools.execution_lab_runtime import bundle_identity, replay_disabled
    from tools.check_execution_performance import assert_identical
    baseline = args.baseline_root.resolve()
    if baseline == ROOT:
        raise ValueError('baseline-root不能指向工程副本')
    if bundle_identity(baseline, normalize_newlines=True)['sha256'] != BASE_CANONICAL_BUNDLE:
        raise ValueError('baseline源码与锁定版本不一致；请提供原0662c21源码副本')
    original_path = destination / 'baseline_replay.json'
    command = [sys.executable, str(Path(__file__).resolve()), '--baseline-worker', '--baseline-root', str(baseline),
        '--device', args.device, '--output', str(original_path), '--check-steps', str(args.check_steps),
        '--epochs', *map(str, args.epochs), '--arms', *args.arms]
    if args.path:
        command.extend(('--path', args.path))
    if args.synthetic:
        command.append('--synthetic')
    process = subprocess.run(command, cwd=str(baseline), capture_output=True, text=True,
        encoding='utf-8', errors='replace', env=dict(os.environ, PYTHONIOENCODING='utf-8'),
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    (destination / 'baseline_stdout.txt').write_text(process.stdout, encoding='utf-8')
    (destination / 'baseline_stderr.txt').write_text(process.stderr, encoding='utf-8')
    if process.returncode:
        raise RuntimeError('原源码重放失败，见baseline_stderr.txt: ' + process.stderr[-1000:])
    original = json.loads(original_path.read_text(encoding='utf-8'))
    assert_identical(original['environment'], runtime_environment(args.device), 'baseline_environment')
    current = {}
    for arm, cfg in config_map.items():
        current[arm] = replay_disabled(cfg, args.device, steps=args.check_steps, epochs=args.epochs,
                                       sources=synthetic_sources(args))
        assert_identical(original['arms'][arm], current[arm], 'disabled_vs_original/' + arm)
    write_json(destination / 'disabled_replay.json', current)
    return dict(status='passed', baseline_revision=BASE_REVISION, source=original['source'],
                comparison='full transaction storage-bit digests, same course/RNG/Adam/rows')


def check_matrix(args, config_map, options, baseline=(), numerical=False):
    from tools.execution_lab_runtime import run_pair
    results = {}
    for arm, cfg in config_map.items():
        print(f'[execution lab] check {arm} {list(options)}', flush=True)
        results[arm] = run_pair(cfg, args.device, baseline, options, steps=args.check_steps,
            epochs=args.epochs, sources=synthetic_sources(args), numerical=numerical)
    return results


def run_checks(args, destination, config_map):
    from utils.execution_lab import STAGE_A, FACTORIZED, KNOWN_NUMERICAL_OPTIONS
    report = dict(status='running', disabled_baseline=verify_disabled(args, destination, config_map),
                  stage_a=dict(items=[], accepted_options=[], rejected_options=[]))
    accepted = []
    for option in STAGE_A:
        item = dict(option=option, status='failed')
        try:
            item['individual'] = check_matrix(args, config_map, (option,))
            if option in KNOWN_NUMERICAL_OPTIONS:
                item.update(status='excluded_known_numeric_change',
                    reason='CPU two-step Adam fixture seed1410 observed prior_read gradient difference '
                           '5.820766091346741e-11; passing another window does not erase the failure')
                report['stage_a']['rejected_options'].append(option)
                write_json(destination / f'check-{option}.json', item)
                report['stage_a']['items'].append(item)
                continue
            combined = tuple(accepted) + (option,)
            item['combined'] = (item['individual'] if not accepted else check_matrix(args, config_map, combined))
            accepted.append(option)
            item['status'] = 'passed'
        except (AssertionError, FloatingPointError, RuntimeError) as error:
            item.update(error=str(error), error_type=type(error).__name__)
            report['stage_a']['rejected_options'].append(option)
        # 每项留完整结果，失败不能被后续组合覆盖。
        write_json(destination / f'check-{option}.json', item)
        report['stage_a']['items'].append(item)
    report['stage_a']['accepted_options'] = accepted
    report['stage_a']['status'] = 'passed' if accepted else 'no_accepted_optimizations'
    try:
        matrix = check_matrix(args, config_map, tuple(accepted) + (FACTORIZED,), tuple(accepted), numerical=True)
        report['stage_b'] = dict(status='contract_passed_numerics_changed_score_unverified', arms=matrix,
            options=accepted + [FACTORIZED], formal_score_verified=False,
            known_cpu_probe='FP32 Feature/train/empty_row gradient failed original atol=1e-5,rtol=1e-4; '
                            'ReLU gate flips observed; no exact-equivalence claim')
    except (AssertionError, FloatingPointError, RuntimeError) as error:
        report['stage_b'] = dict(status='contract_failure', error=str(error), error_type=type(error).__name__)
    report['status'] = 'checked' if report['stage_b']['status'] != 'contract_failure' else 'contract_failure'
    return report


def validate_check(report, args, identity, environment, current_config_sha256):
    if report.get('schema') != 'ct_seqtrack.execution_lab.v1' or report.get('synthetic'):
        raise ValueError('测速需要真实数据check；synthetic报告不构成验收')
    if report.get('code_bundle', {}).get('sha256') != identity['sha256']:
        raise ValueError('check报告代码已过期，请对当前工程代码重验')
    if report.get('environment') != environment:
        raise ValueError('check与benchmark环境不一致')
    old = report['arguments']
    if set(old['arms']) != set(ARMS) or set(old['epochs']) != {0, 10} or old['check_steps'] < 8:
        raise ValueError('benchmark需要四臂、课程0/10、每阶段至少8步check')
    if old.get('path') != args.path:
        raise ValueError('check与benchmark数据根不一致')
    if not current_config_sha256 or any(report.get('configuration_sha256', {}).get(arm) != digest
                                       for arm, digest in current_config_sha256.items()):
        raise ValueError('check与benchmark解析后的配置不一致；请重验修改后的配置')
    checks = report.get('checks', {})
    if checks.get('disabled_baseline', {}).get('status') != 'passed':
        raise ValueError('未通过原源码关闭优化重放')
    if checks.get('stage_a', {}).get('status') != 'passed':
        raise ValueError('没有通过严格门的阶段A组合')
    if checks.get('stage_b', {}).get('status') != 'contract_passed_numerics_changed_score_unverified':
        raise ValueError('阶段B合同失败，禁止继续测速')
    return tuple(checks['stage_a']['accepted_options'])


def gpu_process_snapshot():
    try:
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory,gpu_uuid',
                                 '--format=csv,noheader'], capture_output=True, text=True, timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        return dict(returncode=result.returncode, stdout=result.stdout[:8192], stderr=result.stderr[:2048])
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(status='unavailable', error=str(error))


def paired_performance(args, config, destination, accepted):
    import torch
    from tools.execution_benchmark import _run_one, _percentile
    from utils.execution_lab import execution_scope, FACTORIZED
    destination.mkdir(exist_ok=False)
    implementations = {'baseline': (), 'stage_a': tuple(accepted), 'stage_b': tuple(accepted) + (FACTORIZED,)}
    result = dict(status='running', runs=[], phases=[], traces=[], formal_run=False,
        saved_training_weights=False, automatically_promoted=False, gpu_processes_before=gpu_process_snapshot(),
        benchmark_excludes_correctness_and_trace=True)
    def execute(name, epoch, folder, trace=False):
        import gc
        # Lightning持有model/trainer环；统一释放前次对象和allocator缓存，不计入窗口。
        gc.collect()
        if args.device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        with execution_scope(implementations[name], engineering=True):
            return _run_one(config, 'optimized_fp32', torch.device(args.device), destination / folder,
                warmup=0 if trace else args.warmup, steps=args.trace_steps if trace else args.steps,
                trace=trace, course_epoch=epoch, implementation_id=name)
    try:
        if args.mode != 'trace':
            for epoch in args.epochs:
                for stage, names in (('A', ('baseline', 'stage_a')), ('B', ('stage_a', 'stage_b'))):
                    runs = []
                    for repeat in range(args.repeats):
                        for name in names if repeat % 2 == 0 else tuple(reversed(names)):
                            run = execute(name, epoch, f'epoch{epoch}-{stage}-r{repeat + 1}-{name}')
                            run.update(stage=stage, repeat=repeat + 1)
                            runs.append(run)
                            result['runs'].append(run)
                    phase = summarize_pair(runs, names, epoch, stage, args.repeats)
                    result['phases'].append(phase)
            result['controlled_protocol_complete'] = (set(args.epochs) == {0, 10} and args.repeats >= 3
                                                       and args.warmup >= 20 and args.steps >= 100)
            result['stable_improvement_observed'] = (result['controlled_protocol_complete'] and
                                                      all(phase['all_pairs_faster'] for phase in result['phases']))
            result['actual_optimizer_updates'] = sum(run['optimizer_steps'] for run in result['runs'])
            # 累计值来自两对窗口，明确非额外一次直接配对。
            result['cumulative_ratios_by_course'] = {str(epoch):
                next(p['ratio_of_medians'] for p in result['phases'] if p['epoch_index'] == epoch and p['stage'] == 'A') *
                next(p['ratio_of_medians'] for p in result['phases'] if p['epoch_index'] == epoch and p['stage'] == 'B')
                for epoch in args.epochs}
            result['cumulative_note'] = 'product of separately paired A and B median ratios; not a direct third comparison'
        if args.mode in ('trace', 'all'):
            for epoch in args.epochs:
                for name in implementations:
                    result['traces'].append(execute(name, epoch, f'trace-epoch{epoch}-{name}', trace=True))
        result['gpu_processes_after'] = gpu_process_snapshot()
        result['status'] = 'measured_score_unverified'
    except Exception as error:
        result.update(status='failed', error=str(error))
        raise
    finally:
        write_json(destination / 'report.json', result)
    return result


def summarize_pair(runs, names, epoch, stage, repeats):
    for field in ('endpoint_order_sha256', 'initial_state_sha256', 'completed_steps', 'measured_rows'):
        if len({run[field] for run in runs}) != 1:
            raise AssertionError('配对数据/初始化不一致: ' + field)
    for field in ('csv_field_counts', 'tensorboard_field_counts', 'diagnostic_files', 'console_message_count'):
        if len({json.dumps(run['logging'][field], sort_keys=True) for run in runs}) != 1:
            raise AssertionError('日志字段/频率改变: ' + field)
    values_equal = len({run['logging']['values_sha256'] for run in runs}) == 1
    console_equal = len({run['logging']['console_messages_sha256'] for run in runs}) == 1
    if stage == 'A' and (not values_equal or not console_equal or len({r['final_state_sha256'] for r in runs}) != 1):
        raise AssertionError('阶段A真实Trainer日志/最终参数不再逐位一致')
    summaries = {name: dict(median_window_seconds=statistics.median(r['window_seconds'] for r in runs if r['implementation_id'] == name),
        median_endpoints_per_second=statistics.median(r['endpoints_per_second'] for r in runs if r['implementation_id'] == name)) for name in names}
    pairs = []
    for repeat in range(1, repeats + 1):
        pair = {run['implementation_id']: run for run in runs if run['repeat'] == repeat}
        pairs.append(pair[names[0]]['window_seconds'] / pair[names[1]]['window_seconds'])
    return dict(epoch_index=epoch, stage=stage, summary=summaries, paired_speedup_ratios=pairs,
        ratio_of_medians=summaries[names[0]]['median_window_seconds'] / summaries[names[1]]['median_window_seconds'],
        min_ratio=min(pairs), max_ratio=max(pairs), median_ratio=statistics.median(pairs),
        all_pairs_faster=all(ratio > 1. for ratio in pairs), logging_values_equal=values_equal,
        console_values_equal=console_equal, prepared_inputs_note='raw endpoint order matched; stage B closed-loop prepared inputs may diverge')


def main(argv=None):
    args = parse_args(argv)
    if args.baseline_worker:
        return _baseline_worker(args)
    sys.path.insert(0, str(ROOT))
    from tools.execution_lab_runtime import bundle_identity
    from models.ct_v37.entry import configure_numerics
    configure_numerics()
    output = args.output or ROOT / 'artifacts/ct_checks' / ('execution_lab_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')) / 'report.json'
    output = output.resolve()
    if (ROOT / 'artifacts/ct_checks').resolve() not in output.parents or output.suffix != '.json':
        raise ValueError('output必须为工程副本artifacts/ct_checks下新JSON')
    if output.parent.exists():
        raise FileExistsError('报告需要全新目录，禁止覆盖已有证据')
    output.parent.mkdir(parents=True)
    report = dict(schema='ct_seqtrack.execution_lab.v1', status='running', synthetic=args.synthetic,
        code_bundle=bundle_identity(ROOT), arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        saved_training_weights=False, formal_score_verified=False)
    code = 0
    try:
        report['environment'] = runtime_environment(args.device)
        config_map = configs(args)
        report['configs'] = {arm: dict(cfg) for arm, cfg in config_map.items()}
        report['configuration_sha256'] = configuration_identities(config_map)
        if args.mode in ('check', 'all'):
            report['checks'] = run_checks(args, output.parent, config_map)
            if report['checks']['status'] == 'contract_failure':
                raise RuntimeError('stage B contract failure; benchmarks are blocked')
        if args.mode in ('benchmark', 'trace', 'all'):
            checked = report if args.mode == 'all' else json.loads(args.check_report.read_text(encoding='utf-8'))
            accepted = validate_check(checked, args, report['code_bundle'], report['environment'],
                                      report['configuration_sha256'])
            if 'full' not in config_map:
                raise ValueError('benchmark requires Full arm')
            report['performance'] = paired_performance(args, config_map['full'], output.parent / 'performance', accepted)
        report['source_unchanged_during_run'] = bundle_identity(ROOT)['sha256'] == report['code_bundle']['sha256']
        report['configuration_unchanged_during_run'] = configuration_identities(configs(args)) == report['configuration_sha256']
        if not report['source_unchanged_during_run'] or not report['configuration_unchanged_during_run']:
            raise RuntimeError('工程执行期间源码或配置发生变化；保留结果但禁止视为当前版本验收')
        report['status'] = 'synthetic_contract_smoke_only' if args.synthetic else 'engineering_checked_score_unverified'
    except Exception as error:
        report.update(status='failed', error=str(error), traceback=traceback.format_exc())
        code = 1
    write_json(output, report)
    print(json.dumps(dict(status=report['status'], report=str(output)), ensure_ascii=False))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
