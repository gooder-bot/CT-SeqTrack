"""实际roll-in的原/性能路径输入、预测、反传及Adam逐位一致。"""

import copy
import random

import numpy as np
import pytest
import torch

from tests.test_ct_v27_full_model import full_model_runtime
from tests.test_ct_v27_input_flow import sampler_runtime
from tests.test_ct_v29_rollin import setup_case
from utils.training_isolation import capture_global_rng_state, restore_global_rng_state
from utils.v28_numerical_audit import snapshot, compare_values
from utils.v29_rollin import prepare_observation_batch, process_query


@pytest.mark.parametrize('endpoint,candidate', [(0, 1), (1, 2), (2, 3), (8, 1), (8, 3)])
def test_real_rollin_perf_preserves_training_transaction(full_model_runtime, endpoint, candidate):
    config, item = setup_case(full_model_runtime[0], start=max(0, endpoint - 4), endpoint=endpoint)
    item['candidate'] = candidate
    initial_rng = capture_global_rng_state()
    expected = None
    for optimized in (False, True):
        torch.manual_seed(42)
        cfg = copy.deepcopy(config)
        if optimized:
            cfg.ct_runtime_optimization = 'equivalent_v1'
            cfg.ct_diagnostic_policy = 'sampled_v1'
        host = full_model_runtime[2](cfg).train()
        optimizer = host.configure_optimizers()['optimizer']
        flags = [m.training for m in host.modules()]
        before = {k: (v.clone(), v._version) for k, v in host.named_buffers()}
        restore_global_rng_state(initial_rng)
        waves = []
        hook = host.register_forward_hook(lambda _m, _args, out: waves.append(
            snapshot(out['observation_aux_estimation_boxes'])))
        batch = prepare_observation_batch(host, [copy.deepcopy(item), copy.deepcopy(item)])
        hook.remove()
        assert compare_values(initial_rng, capture_global_rng_state()) is None
        assert flags == [m.training for m in host.modules()]
        for key, value in host.named_buffers():
            assert torch.equal(before[key][0], value), key
            if key.endswith(('running_mean', 'running_var', 'num_batches_tracked')):
                assert before[key][1] == value._version, key
        optimizer.zero_grad(set_to_none=True)
        output = host(batch)
        loss = host.compute_loss(batch, output)['loss_b0_transaction']
        loss.backward()
        gradients = snapshot({k: p.grad for k, p in host.named_parameters()})
        optimizer.step()
        record = snapshot(dict(
            batch=batch, waves=waves, loss=loss, gradients=gradients,
            weights=host.state_dict(), optimizer=optimizer.state_dict(),
            rng=capture_global_rng_state()))
        if expected is None:
            expected = record
        else:
            assert compare_values(expected, record) is None, compare_values(expected, record)


def test_perf_mixed_teacher_short_and_long_windows_keep_active_order(full_model_runtime):
    config, long = setup_case(full_model_runtime[0])
    _, short = setup_case(full_model_runtime[0], start=0, endpoint=2)
    _, initial = setup_case(full_model_runtime[0], start=0, endpoint=0)
    long['index'], short['index'], initial['index'] = 11, 6, 1
    predictions = {i: copy.deepcopy(long['frames'][i - long['start']]['3d_bbox'])
                   for i in range(long['start'], long['endpoint'])}
    teacher, _ = process_query(long, predictions, long['endpoint'], config)
    teacher['candidate_id'] = np.int64(0)
    items = [short, dict(mode='teacher', sample=teacher), initial, long]
    expected = None
    for optimized in (False, True):
        cfg = copy.deepcopy(config)
        if optimized:
            cfg.ct_runtime_optimization = 'equivalent_v1'
        torch.manual_seed(42)
        host = full_model_runtime[2](cfg).train()
        waves = []
        hook = host.register_forward_pre_hook(lambda _m, args: waves.append(
            args[0]['ct_observation_original_index'].cpu().tolist()))
        batch = prepare_observation_batch(host, copy.deepcopy(items))
        hook.remove()
        assert waves == [[6, 11], [11], [11]]
        assert batch['candidate_id'].tolist() == [short['candidate'], 0, initial['candidate'], long['candidate']]
        if expected is None:
            expected = snapshot(batch)
        else:
            assert compare_values(expected, snapshot(batch)) is None


@pytest.mark.parametrize('restore_error', [False, True])
def test_rollin_failure_still_restores_modes_routing_and_rng(full_model_runtime, monkeypatch, restore_error):
    import utils.v29_rollin as rollin
    config, item = setup_case(full_model_runtime[0])
    config.ct_runtime_optimization = 'equivalent_v1'
    host = full_model_runtime[2](config).train()
    flags = [m.training for m in host.modules()]
    names = ('use_ct_joint_full', 'use_b1motion_v3', 'ct_enable_b1', 'ct_enable_b2', 'ct_enable_b3')
    routing = {key: getattr(host, key) for key in names}
    before = capture_global_rng_state()

    def fail_forward(batch):
        del batch
        random.random()
        np.random.rand()
        torch.rand(1)
        raise RuntimeError('forward failure')

    def fail_restore(*args):
        raise RuntimeError('restore failure')

    monkeypatch.setattr(host, 'forward', fail_forward)
    if restore_error:
        monkeypatch.setattr(rollin, 'restore_changed_buffers', fail_restore)
    with pytest.raises(RuntimeError, match='restore failure' if restore_error else 'forward failure'):
        prepare_observation_batch(host, [item])
    assert [m.training for m in host.modules()] == flags
    assert {key: getattr(host, key) for key in names} == routing
    assert not host._ct_observation_only_forward
    assert compare_values(before, capture_global_rng_state()) is None
