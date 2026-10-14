"""保留删去 decoder prior 行的两步 Adam 数值失败；不是阶段 A 放行测试。"""
from copy import deepcopy
import json

import pytest
import torch

from models.ct_v37.model import JointTracker
from tests.test_ct_v37_network import batch, config
from tools.check_execution_performance import assert_identical, freeze
from utils.execution_lab import execution_scope, KNOWN_NUMERICAL_OPTIONS


def _finite(value):
    if torch.is_tensor(value):
        if value.is_floating_point():
            assert torch.isfinite(value).all()
    elif isinstance(value, dict):
        for item in value.values():
            _finite(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _finite(item)


def test_prior_read_two_step_original_strict_gate_retains_known_numerical_failure():
    assert 'prior_read' in KNOWN_NUMERICAL_OPTIONS
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1410)
            original = JointTracker(config()).train()
            revised = deepcopy(original)
            values = batch()
            optimizers = [torch.optim.Adam(model.parameters(), lr=1.5e-4)
                          for model in (original, revised)]
            failures = []
            for step in range(2):
                rng = torch.get_rng_state()
                snapshots = []
                for model, optimizer, options in zip((original, revised), optimizers,
                                                    ((), ('prior_read',))):
                    torch.set_rng_state(rng)
                    optimizer.zero_grad(set_to_none=True)
                    with execution_scope(options, engineering=True):
                        output = model(values)
                        losses = model.compute_losses(values, output)
                        losses['loss_total'].backward()
                        gradients = freeze({name: p.grad for name, p in model.named_parameters()})
                        optimizer.step()
                    snapshots.append(freeze(dict(output=output, losses=losses, gradients=gradients,
                        model=model.state_dict(), optimizer=optimizer.state_dict(), rng=torch.get_rng_state())))
                left, right = snapshots
                # 正常合同失败必须作为测试失败，不能被下面的数值 xfail 隐藏。
                _finite(left)
                _finite(right)
                assert left['gradients'].keys() == right['gradients'].keys()
                for name, grad in left['gradients'].items():
                    other = right['gradients'][name]
                    assert (grad is None) == (other is None), name
                    if grad is not None:
                        assert grad.shape == other.shape and grad.dtype == other.dtype, name
                for name in ('decoder.local_reader.output.weight', 'decoder.local_reader.output.bias'):
                    assert left['gradients'][name] is not None and right['gradients'][name] is not None
                    # 第一轮零初始化出口允许零梯度；一次 Adam 后须有真实梯度连接。
                    if step == 1:
                        assert left['gradients'][name].abs().sum() > 0
                        assert right['gradients'][name].abs().sum() > 0
                for key in ('selected_index', 'q0_is_fallback'):
                    assert_identical(left['output'][key], right['output'][key], key)
                assert_identical(left['output']['candidates']['valid'], right['output']['candidates']['valid'])
                for key in ('authorized', 'support_mask', 'commit_support_mask'):
                    assert_identical(left['output']['verification'][key], right['output']['verification'][key], key)
                assert_identical(left['rng'], right['rng'])
                for section in left:
                    try:
                        assert_identical(left[section], right[section], section)
                    except AssertionError as error:
                        failures.append(dict(step=step + 1, section=section, first_failure=str(error)))
                bias = 'decoder.local_reader.output.bias'
                max_bias_delta = float((left['gradients'][bias] - right['gradients'][bias]).abs().max())
                print(json.dumps(dict(step=step + 1, reader_bias_gradient_max_abs=max_bias_delta,
                    strict_failures=[item for item in failures if item['step'] == step + 1]), ensure_ascii=False))
            if failures:
                pytest.xfail('prior_read 原逐位门失败；历史CPU第二步bias梯度差5.820766091346741e-11；'
                             '不得进入严格A组合；本机失败=' + json.dumps(failures, ensure_ascii=False))
            # 别的平台两步恰好一致可以通过；KNOWN_NUMERICAL_OPTIONS 历史排除仍保留。
    finally:
        torch.set_num_threads(previous_threads)
