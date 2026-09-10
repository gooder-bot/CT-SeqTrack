"""H3 必须覆盖实际合法事件，不能把随机未执行当作微基准通过。"""
import copy
from types import SimpleNamespace

import pytest
import torch

from tests.test_ct_v27_full_model import full_model_runtime
from tests.test_ct_v27_input_flow import sampler_runtime
from tests.test_ct_v29_b0_host import construct
from tests.test_ct_v29_perf_diagnostics import _future_raw, optimized
from utils.training_isolation import capture_global_rng_state
from utils.v29_profiling import snapshot, first_difference
from utils.v29_h3_benchmark import maybe_benchmark_h3, benchmark_event
from tools.benchmark_ct_v29_h3 import build_command, ROOT


def test_microbenchmark_disabled_in_normal_training(monkeypatch):
    monkeypatch.delenv('CT_V29_H3_BENCH_DIR', raising=False)
    maybe_benchmark_h3(SimpleNamespace(), {}, {})


def test_h3_command_is_bounded_production_full_scratch():
    plan = build_command('/data/full', ROOT / 'artifacts/ct_checks/h3_test')
    assert '--ct_engineering_check' in plan['argv']
    assert '--checkpoint' not in plan['argv'] and '--preloading' not in plan['argv']
    assert plan['argv'][plan['argv'].index('--limit_train_batches') + 1] == '100'
    assert plan['environment']['CT_V29_H3_BENCH_REPEATS'] == '5'
    with pytest.raises(ValueError):
        build_command('/data', ROOT / 'output/h3_bad')
    with pytest.raises(ValueError):
        build_command('/data', ROOT / 'artifacts/ct_checks/h3_bad', arm='b0')


@pytest.mark.parametrize('arm', ['full_gru', 'full_cfc'])
def test_actual_h3_microbenchmark_abba_equal_labels_no_main_state_effect(full_model_runtime, arm):
    host = construct(full_model_runtime, arm).train()
    optimized(host.config)
    _future_raw(full_model_runtime[0], host)
    output = dict(observation_aux_estimation_boxes=torch.zeros(1, 4),
        ct_router_bounded_residual_xy=torch.tensor([[.5, 0.]]), ct_b2_available=torch.ones(1))
    original_config = copy.deepcopy(host.config)
    before = snapshot(dict(model=host.state_dict(), rng=capture_global_rng_state(),
        states=[context['state'] for context in host._ct_online_batch_context]))
    report = benchmark_event(host, {}, output, repeats=1, warmup=0)
    assert report['status'] == 'complete' and report['labels_bitwise_equal']
    assert [row['variant'] for row in report['phases']] == [
        'legacy', 'equivalent_v1', 'equivalent_v1', 'legacy']
    assert all(row['valid_events'] == 1 and row['shadow_branch_forwards'] == 4
               for row in report['phases'])
    assert host.config == original_config
    assert not hasattr(host, '_ct_mechanism_processing_config')
    after = snapshot(dict(model=host.state_dict(), rng=capture_global_rng_state(),
        states=[context['state'] for context in host._ct_online_batch_context]))
    assert first_difference(before, after) is None
