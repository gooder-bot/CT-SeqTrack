"""诊断减频不改变真实 host 训练目标、梯度、状态或递归提交。"""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.test_ct_v27_input_flow import sampler_runtime
from tests.test_ct_v27_full_model import full_model_runtime, _training_batch
from tests.test_ct_v28_audit_host import _attach_training_shell
from tests.test_ct_v29_b0_host import construct, mechanism_batch
from tests.test_ct_v29_full_flow import case29
from utils.training_isolation import capture_global_rng_state, restore_global_rng_state
from utils.v28_numerical_audit import snapshot, compare_values
from utils.v29_diagnostics import (sample_training_diagnostics, keep_h3_event, flush_core_losses,
    accumulate_core_losses, diagnostic_call, flush_pending_diagnostics, H1OnlyLabels,
    assert_h3_state_unchanged)


def optimized(config, sampled=True):
    config.ct_runtime_optimization = 'equivalent_v1'
    config.ct_diagnostic_policy = 'sampled_v1' if sampled else 'full'
    config.ct_scalar_log_every_n_steps = 50
    config.ct_diagnostic_every_n_steps = 100
    config.ct_h3_diagnostic_keep_ratio = .1


def test_schedule_covers_initial_periodic_without_guessing_epoch_tail():
    config = SimpleNamespace(ct_enable_v29=True)
    optimized(config)
    host = SimpleNamespace(training=True, config=config, global_step=5,
                          _ct_perf_batch_index=5,
                          _trainer=SimpleNamespace(num_training_batches=101))
    assert not sample_training_diagnostics(host)
    host.global_step = 4
    assert sample_training_diagnostics(host)
    host.global_step = 49
    assert sample_training_diagnostics(host, scalar=True)
    assert not sample_training_diagnostics(host)
    host.global_step = 99
    assert sample_training_diagnostics(host)
    host.global_step = 100; host._ct_perf_batch_index = 100
    assert not sample_training_diagnostics(host)
    host.global_step = 17; host._ct_perf_batch_index = 17; host.training = False
    assert sample_training_diagnostics(host) and sample_training_diagnostics(host, scalar=True)


def test_h3_hash_subset_is_stable_and_rng_free():
    config = SimpleNamespace(ct_enable_v29=True, seed=42)
    optimized(config)
    host = SimpleNamespace(training=True, config=config)
    before = capture_global_rng_state()
    rows = [dict(online_epoch=2, tracklet_key='test/track', this_frame_id=i) for i in range(1000)]
    chosen = [keep_h3_event(host, row) for row in rows]
    assert 60 < sum(chosen) < 140
    assert chosen == [keep_h3_event(host, row) for row in rows]
    assert compare_values(snapshot(before), snapshot(capture_global_rng_state())) is None
    for changes in ({'online_epoch': 3}, {'tracklet_key': 'another'}, {'shadow_event': 'another_event'}):
        assert chosen != [keep_h3_event(host, dict(row, **changes)) for row in rows]
    config.seed = 52
    assert chosen != [keep_h3_event(host, row) for row in rows]
    host.training = False
    assert all(keep_h3_event(host, row) for row in rows)


@pytest.mark.parametrize('arm', ['b0', 'full_gru', 'full_cfc'])
def test_actual_host_sampled_and_full_loss_gradient_adam_bn_rng_match(
        full_model_runtime, monkeypatch, arm):
    plain = construct(full_model_runtime, arm).train()
    plain.config.ct_runtime_optimization = 'legacy'
    plain.config.ct_diagnostic_policy = 'full'
    fast = copy.deepcopy(plain)
    optimized(fast.config)
    observation_host = construct(full_model_runtime, 'b0')
    observation, _, _ = _training_batch(full_model_runtime, observation_host, batch_size=2)
    observation['candidate_id'] = torch.tensor([0, 3])
    mechanism = None if arm == 'b0' else mechanism_batch(full_model_runtime, plain)
    batch = dict(ct_stream_schema='ct_seqtrack.train.v4', observation=observation, mechanism=mechanism)
    rng = capture_global_rng_state()
    records = []
    for host in (plain, fast):
        optimizer = host.configure_optimizers()['optimizer']
        _attach_training_shell(host, optimizer, monkeypatch)
        host._trainer.global_step = 17
        host._trainer.num_training_batches = 100
        if not isinstance(getattr(type(host), 'global_step', None), property):
            host.global_step = 17
        writes = []
        host.logger.experiment.add_scalars = lambda *a, **kw: writes.append(a)
        lightning_logs = []
        host.log = lambda *a, **kw: lightning_logs.append((a, kw))
        restore_global_rng_state(rng)
        optimizer.zero_grad(set_to_none=True)
        loss = host.training_step(batch, 17)
        loss.backward()
        host.on_before_optimizer_step(optimizer)
        optimizer.step()
        host.on_train_batch_end(loss, batch, 17)
        for name in ('seg_acc_background/train', 'seg_acc_foreground/train',
                     'motion_acc_static/train', 'motion_acc_dynamic/train'):
            assert any(args[0] == name and kwargs['on_epoch']
                       for args, kwargs in lightning_logs), name
        records.append(snapshot(dict(loss=loss, state=host.state_dict(),
            gradients={n: p.grad for n, p in host.named_parameters()}, adam=optimizer.state_dict(),
            rng=capture_global_rng_state(), flags={n: m.training for n, m in host.named_modules()},
            norms=host._ct_last_gradient_norm)))
        if host is fast:
            assert not writes
            assert host._ct_perf_epoch_losses
            flush_core_losses(host)
            assert writes[0][0] == 'ct_epoch_core_loss'
        else:
            assert writes
    assert compare_values(records[0], records[1]) is None


def _future_raw(runtime, model):
    _, _, sequence, state, payload, _, _ = case29(runtime)
    for frame_id in (9, 10):
        frame = copy.deepcopy(sequence[8])
        frame['timestamp'] += (frame_id - 8) * 500000
        frame['frame_id'] = frame_id
        sequence.append(frame)
    raw = dict(payload, online_slot=0, online_epoch=0, online_batch_index=0,
               online_recursive_raw=True, tracklet_id=0, shadow_scheduled=True,
               shadow_future_exists=[True, True], shadow_future=[])
    misc = runtime[2]
    for frame_id in (9, 10):
        future = {key: value for key, value in raw.items()
                  if not key.startswith('motion_aux_') and key not in (
                      'shadow_future', 'online_motion_aux_state', 'motion_prediction', 'online_recursive_state')}
        ids = [frame_id - offset for offset in (1, 2, 3)]
        future.update(this_frame=sequence[frame_id], this_frame_id=frame_id,
                      prev_frame_ids=ids, prev_frames=misc.create_history_frame_dict([sequence[i] for i in ids]),
                      valid_mask=[1, 1, 1], history_offsets=[1, 2, 3], ct_observation_only=True)
        aux_ids = [frame_id - offset for offset in (2, 4, 6)]
        future.update(motion_aux_offsets=[2, 4, 6], motion_aux_frame_ids=aux_ids,
            motion_aux_valid_mask=[1, 1, 1],
            motion_aux_prev_frames=misc.create_history_frame_dict([sequence[i] for i in aux_ids]))
        raw['shadow_future'].append(future)
    model._ct_online_batch_context = [dict(raw=raw, state=state)]
    return raw, state


def test_actual_host_h3_on_off_preserves_rng_buffers_flags_and_main_state(full_model_runtime):
    from utils.v27_training import attach_h3_shadow_labels_v27
    model = construct(full_model_runtime, 'full_gru').train()
    optimized(model.config)
    raw, state = _future_raw(full_model_runtime[0], model)
    output = dict(observation_aux_estimation_boxes=torch.zeros(1, 4),
                  ct_router_bounded_residual_xy=torch.tensor([[.5, 0.]]), ct_b2_available=torch.ones(1))
    before = snapshot(dict(state=model.state_dict(), rng=capture_global_rng_state(),
                           flags={n: m.training for n, m in model.named_modules()}))
    for ratio in (1., 0.):
        model.config.ct_h3_diagnostic_keep_ratio = ratio
        batch = {}
        attach_h3_shadow_labels_v27(model, batch, output)
        assert max(state.predictions) == 7
        after = snapshot(dict(state=model.state_dict(), rng=capture_global_rng_state(),
                              flags={n: m.training for n, m in model.named_modules()}))
        assert compare_values(before, after) is None
        assert batch['ct_h3_diagnostic_candidate_events'].item() == 1
        assert batch['ct_h3_diagnostic_scheduled_events'].item() == 1
        if ratio:
            assert batch['ct_h3_failure_reason'] == ['ok'], batch['ct_h3_failure_reason']
            assert batch['ct_shadow_forward_count'].item() == 4
            assert batch['ct_h3_diagnostic_executed_events'].item() == 1
            assert batch['ct_h3_diagnostic_valid_events'].item() == 1
            assert batch['ct_h3_diagnostic_not_sampled_events'].item() == 0
        else:
            assert batch['ct_h3_failure_reason'] == ['not_sampled_diagnostic']
            assert batch['ct_shadow_forward_count'].item() == 0
            assert batch['ct_h3_diagnostic_sampled_events'].item() == 0
            assert batch['ct_h3_diagnostic_not_sampled_events'].item() == 1


def test_actual_h3_observation_payload_optimization_keeps_future_prediction_labels(full_model_runtime):
    from utils.v27_training import attach_h3_shadow_labels_v27
    old = construct(full_model_runtime, 'full_gru').train()
    old.config.ct_runtime_optimization = 'legacy'
    old.config.ct_diagnostic_policy = 'full'
    new = copy.deepcopy(old)
    optimized(new.config, sampled=False)
    output = dict(observation_aux_estimation_boxes=torch.zeros(1, 4),
                  ct_router_bounded_residual_xy=torch.tensor([[.5, 0.]]), ct_b2_available=torch.ones(1))
    rng = capture_global_rng_state()
    records = []
    for model in (old, new):
        _future_raw(full_model_runtime[0], model)
        restore_global_rng_state(rng)
        batch = {}
        attach_h3_shadow_labels_v27(model, batch, output)
        assert batch['ct_h3_failure_reason'] == ['ok'], batch['ct_h3_failure_reason']
        records.append(snapshot({key: value for key, value in batch.items()
                                 if key not in ('ct_shadow_time_ms', 'ct_shadow_peak_memory_mb')}))
    assert compare_values(records[0], records[1]) is None


def test_epoch_flush_keeps_all_loss_rows_and_only_last_unwritten_diagnostic():
    config = SimpleNamespace(ct_enable_v29=True)
    optimized(config)
    writes, serialized = [], []
    host = SimpleNamespace(training=True, config=config, global_step=17, _ct_perf_batch_index=17,
        logger=SimpleNamespace(experiment=SimpleNamespace(add_scalars=lambda *a, **kw: writes.append((a, kw)))))
    for value, rows in ((1., 2), (7., 1), (4., 3)):
        losses = {'loss_total': torch.tensor(value, requires_grad=True)}
        if value == 7.:
            losses['loss_optional'] = torch.tensor(2.)
        accumulate_core_losses(host, losses, rows)
        diagnostic_call(host, 'last', serialized.append, losses['loss_total'], scalar=True)
        host.global_step += 1
    assert not writes and not serialized
    flush_core_losses(host)
    assert len(serialized) == 1 and serialized[0] == 4. and not serialized[0].requires_grad
    metrics = writes[0][0][1]
    assert metrics['observation/loss_total'] == 3.5
    assert metrics['observation/rows'] == 6 and metrics['observation/transactions'] == 3
    assert metrics['observation/loss_optional'] == 2.
    flush_pending_diagnostics(host)
    assert len(serialized) == 1
    host.global_step = 49
    diagnostic_call(host, 'last', serialized.append, torch.tensor(9.), scalar=True)
    assert len(serialized) == 2
    flush_pending_diagnostics(host)
    assert len(serialized) == 2  # 已在周期步写出，epoch-end 不再重复。


def test_h3_labels_cannot_be_read_by_h1_loss_contract():
    data = H1OnlyLabels({'box_label': 3, 'ct_h3_valid': 1, 'ct_h3_success_gain': 100.})
    assert data['box_label'] == 3 and list(data) == ['box_label']
    for key in ('ct_h3_valid', 'ct_shadow_forward_count'):
        with pytest.raises(RuntimeError, match='cannot enter B3 loss'):
            data.get(key)


def test_h3_runtime_guard_rejects_accepted_box_or_recursive_state_mutation():
    host = SimpleNamespace(_ct_online_batch_context=[{'state': {'frame': 1}}])
    output = {'ct_final_box': torch.zeros(1, 4)}
    with pytest.raises(RuntimeError, match='mutated accepted policy'):
        with assert_h3_state_unchanged(host, output):
            output['ct_final_box'].add_(1)
    with pytest.raises(RuntimeError, match='recursive state'):
        with assert_h3_state_unchanged(host, output):
            host._ct_online_batch_context[0]['state']['frame'] = 2


@pytest.mark.parametrize('arm', ['full_gru', 'full_cfc'])
def test_h3_sampling_between_forward_and_backward_keeps_update(full_model_runtime, arm):
    """主图仍存活时执行H3，不能通过buffer恢复引入版本错误或改变反传。"""
    from utils.v27_training import attach_h3_shadow_labels_v27
    base = construct(full_model_runtime, arm).train()
    optimized(base.config)
    original_data = mechanism_batch(full_model_runtime, base)
    initial = capture_global_rng_state()
    records = []
    for ratio in (1., 0.):
        host = copy.deepcopy(base)
        host.config.ct_h3_diagnostic_keep_ratio = ratio
        optimizer = host.configure_optimizers()['optimizer']
        data = copy.deepcopy(original_data)
        restore_global_rng_state(initial)
        output = host._forward_safe_mechanism(data)
        _future_raw(full_model_runtime[0], host)
        # 单个真实历史事件配一个预测行，独立影子不会修改主图output。
        shadow_output = {key: value[:1] for key, value in output.items()
                         if torch.is_tensor(value) and value.ndim and value.shape[0] == len(data['points'])}
        shadow_output['ct_b2_available'] = torch.ones(1)
        shadow_batch = {}
        attach_h3_shadow_labels_v27(host, shadow_batch, shadow_output)
        if ratio:
            assert shadow_batch['ct_h3_failure_reason'] == ['ok']
        losses = host.compute_loss(data, output)
        loss = losses['loss_total']
        loss.backward()
        optimizer.step()
        records.append(snapshot(dict(loss=loss, state=host.state_dict(),
            grads={name: value.grad for name, value in host.named_parameters()},
            adam=optimizer.state_dict(), rng=capture_global_rng_state())))
    assert compare_values(records[0], records[1]) is None
