"""真实合法 H3 事件的独立 ABBA 微基准；仅显式工程运行可启用。"""

import json
import os
from pathlib import Path
import time

import torch

from utils.v29_performance import preserved_model_state
from utils.v29_profiling import snapshot, first_difference, summarize


def maybe_benchmark_h3(host, batch, output):
    directory = os.environ.get('CT_V29_H3_BENCH_DIR')
    if not directory or getattr(host, '_ct_h3_microbenchmark_running', False):
        return
    if getattr(host, '_ct_h3_microbenchmark_done', False):
        return
    root = Path(__file__).resolve().parents[1] / 'artifacts' / 'ct_checks'
    target = Path(directory).resolve()
    if root.resolve() not in target.parents:
        raise ValueError('H3 microbenchmark requires artifacts/ct_checks')
    if not (getattr(host.config, 'ct_enable_v29', False)
            and getattr(host.config, 'ct_engineering_check', False)
            and host.training and host.ct_enable_b3):
        raise RuntimeError('H3 microbenchmark requires a scratch v29 Full engineering run')
    contexts = getattr(host, '_ct_online_batch_context', [])
    from utils.v27_training import _structural
    structural = _structural(output).cpu().tolist()
    legal = [index for index, context in enumerate(contexts)
             if structural[index] and context['raw'].get('shadow_scheduled', False)
             and len(context['raw'].get('shadow_future', [])) == 2]
    if not legal:
        return
    repeats = int(os.environ.get('CT_V29_H3_BENCH_REPEATS', '5'))
    warmup = int(os.environ.get('CT_V29_H3_BENCH_WARMUP', '2'))
    if not 1 <= repeats <= 50 or not 0 <= warmup <= 20:
        raise ValueError('invalid H3 microbenchmark repetitions')
    host._ct_h3_microbenchmark_running = True
    try:
        report = benchmark_event(host, batch, output, repeats=repeats, warmup=warmup)
        target.mkdir(parents=True, exist_ok=True)
        (target / 'h3_microbenchmark.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        host._ct_h3_microbenchmark_done = True
    finally:
        host._ct_h3_microbenchmark_running = False


def benchmark_event(host, batch, output, *, repeats=5, warmup=2):
    """同一真实事件/权重/状态强制执行 H3；不测 10% 抽样的覆盖收益。"""
    from utils.v27_training import attach_h3_shadow_labels_v27
    from utils.v29_diagnostics import assert_h3_diagnostic_contract, assert_h3_state_unchanged
    assert_h3_diagnostic_contract(host.config)
    keys = ('ct_runtime_optimization', 'ct_diagnostic_policy')
    saved = {key: (key in host.config, host.config.get(key)) for key in keys}
    had_processing = hasattr(host, '_ct_mechanism_processing_config')
    processing = getattr(host, '_ct_mechanism_processing_config', None)
    expected, rows = None, []
    def sync():
        if host.device.type == 'cuda':
            torch.cuda.synchronize(host.device)
    try:
        # 仅暂存两个执行开关，正式配置/参数/输入均不替换。
        for phase, variant in enumerate(('legacy', 'equivalent_v1', 'equivalent_v1', 'legacy')):
            host.config.ct_runtime_optimization = variant
            host.config.ct_diagnostic_policy = 'full'  # 微基准只比较相同的实际执行事件。
            host._ct_mechanism_processing_config = None
            elapsed, valid_counts, forwards = [], [], []
            for iteration in range(warmup + repeats):
                local = {key: value for key, value in batch.items()
                         if not key.startswith(('ct_h3_', 'ct_shadow_'))}
                # 外层恢复/数值检查不计入耗时；内层正式路径的隔离开销仍计入。
                with preserved_model_state(host), assert_h3_state_unchanged(host, output):
                    sync()
                    started = time.perf_counter()
                    attach_h3_shadow_labels_v27(host, local, output)
                    sync()
                    milliseconds = 1000. * (time.perf_counter() - started)
                labels = snapshot({key: value for key, value in local.items()
                    if key.startswith('ct_h3_') and not key.startswith('ct_h3_diagnostic_')})
                if expected is None:
                    expected = labels
                difference = first_difference(expected, labels)
                if difference:
                    raise RuntimeError('H3 event numerical mismatch: ' + difference)
                valid = int(local['ct_h3_valid'].sum().item())
                if not valid:
                    raise RuntimeError('H3 microbenchmark requires a valid executed event: '
                                       + str(local['ct_h3_failure_reason']))
                if iteration >= warmup:
                    elapsed.append(milliseconds)
                    valid_counts.append(valid)
                    forwards.append(int(local['ct_shadow_forward_count'].item()))
            rows.append(dict(phase=phase + 1, variant=variant, time=summarize(elapsed),
                             milliseconds=elapsed, valid_events=sum(valid_counts),
                             shadow_branch_forwards=sum(forwards)))
    finally:
        for key, (present, value) in saved.items():
            if present:
                host.config[key] = value
            else:
                host.config.pop(key, None)
        if had_processing:
            host._ct_mechanism_processing_config = processing
        elif hasattr(host, '_ct_mechanism_processing_config'):
            delattr(host, '_ct_mechanism_processing_config')
    return dict(schema='ct_seqtrack.h3_microbenchmark.v1', status='complete',
        order='ABBA', device=str(host.device), repetitions_per_phase=repeats,
        warmup_per_phase=warmup, forced_execution=True, unique_workload_batches=1,
        labels_bitwise_equal=True, formal_initialization_allowed=False, phases=rows,
        scope='同一真实合法事件和固定权重反复执行；不代表所有事件分布或10%覆盖率，也不代表完整训练吞吐。')
