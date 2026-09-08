"""实际 v28 主机事务与优化器 hooks 的数值审计接线回归。"""

import copy
from types import SimpleNamespace

import pytest
import torch

from tests.test_ct_v27_full_model import full_model_runtime, _training_batch
from tests.test_ct_v27_input_flow import sampler_runtime
from tests.test_ct_v28_b0_reference import construct
from tools.compare_ct_v28_audits import compare_audits
from utils.training_isolation import capture_global_rng_state, restore_global_rng_state
from utils.v28_numerical_audit import compare_values, snapshot


def _attach_training_shell(model, optimizer, monkeypatch):
    """仅替换日志与 Trainer 容器，保留真实训练及优化器回调。"""
    logger = SimpleNamespace(experiment=SimpleNamespace(
        add_scalars=lambda *args, **kwargs: None))
    trainer = SimpleNamespace(optimizers=[optimizer], logger=logger,
                              current_epoch=0, global_step=0, global_rank=0)
    model.trainer = trainer
    model._trainer = trainer
    # 本地缺 Lightning 时 fixture 提供最小 nn.Module 外壳。
    if not isinstance(getattr(type(model), 'logger', None), property):
        model.logger = logger
    monkeypatch.setattr(model, 'log', lambda *args, **kwargs: None)


@pytest.mark.parametrize('arm', ['b0', 'full'])
def test_real_dual_stream_audit_hooks_preserve_update_bn_and_rng(
        full_model_runtime, monkeypatch, tmp_path, arm):
    plain = construct(full_model_runtime, arm).train()
    audited = copy.deepcopy(plain)
    observation_host = construct(full_model_runtime, 'b0')
    observation, _, _ = _training_batch(full_model_runtime, observation_host, batch_size=2)
    observation['candidate_id'] = torch.tensor([0, 2])
    mechanism = None
    if arm == 'full':
        mechanism, _, _ = _training_batch(full_model_runtime, plain, batch_size=2)
    batch = dict(ct_stream_schema='ct_seqtrack.train.v4',
                 observation=observation, mechanism=mechanism)
    optimizers = [model.configure_optimizers()['optimizer'] for model in (plain, audited)]
    for model, optimizer in zip((plain, audited), optimizers):
        _attach_training_shell(model, optimizer, monkeypatch)
    initial_rng = capture_global_rng_state()
    records = []
    directory = tmp_path / 'audit'
    monkeypatch.delenv('CT_V28_AUDIT_ACTIVATIONS', raising=False)
    for enabled, model, optimizer in zip((False, True), (plain, audited), optimizers):
        if enabled:
            monkeypatch.setenv('CT_V28_AUDIT_DIR', str(directory))
        else:
            monkeypatch.delenv('CT_V28_AUDIT_DIR', raising=False)
        restore_global_rng_state(initial_rng)
        optimizer.zero_grad(set_to_none=True)
        loss = model.training_step(batch, 0)
        loss.backward()
        model.on_before_optimizer_step(optimizer)
        optimizer.step()
        model.on_train_batch_end(loss, batch, 0)
        assert int(model.ct_b0_update_step) == 1
        if arm == 'full':
            assert all(int(getattr(model, f'ct_{name}_update_step')) == 1
                       for name in ('b1', 'b2', 'b3'))
        records.append(snapshot(dict(
            loss=loss, state=model.state_dict(),
            gradients={name: parameter.grad for name, parameter in model.named_parameters()},
            adam=optimizer.state_dict(), rng=capture_global_rng_state(),
            modes={name: child.training for name, child in model.named_modules()})))
    assert compare_values(records[0], records[1]) is None
    assert getattr(plain, '_ct_numerical_audit') is None
    assert audited._ct_numerical_audit.row is None
    assert audited._ct_numerical_audit.hooks == []
    assert not list(directory.glob('mechanism_failure_*.pt'))
    initial = torch.load(directory / 'step_000.pt', weights_only=False)
    update = torch.load(directory / 'step_001.pt', weights_only=False)
    assert initial['step'] == 0 and initial['state']['parameters']
    assert initial['schema'] == update['schema'] == 'ct_seqtrack.numerical_audit.v28'
    assert update['step'] == 1
    for key in ('input', 'before_forward', 'activations', 'activation_gradients',
                'pool_indices', 'output', 'argmax', 'losses', 'after_forward',
                'gradients', 'adam_after', 'parameters_before_update',
                'b0_optimizer_group', 'parameter_order', 'after_update',
                'actual_parameter_delta'):
        assert update[key], key
    assert update['b0_optimizer_group']['name'] == 'b0'
    assert not any(audited._ct_any_plugin_parameter(name)
                   for name in update['gradients'])
    assert any(value.abs().sum() > 0 for value in update['actual_parameter_delta'].values())
    # 机制额外 B0 前向不得进入 observation 激活快照。
    assert all(key.rsplit('#', 1)[-1].split('[')[0] == '0'
               for key in update['activations'] if key.startswith('seg_pointnet.'))


def test_real_b0_and_b1_audit_snapshots_compare_bitwise_across_arms(
        full_model_runtime, monkeypatch, tmp_path):
    models = [construct(full_model_runtime, arm).train() for arm in ('b0', 'b1_gru')]
    observation, _, _ = _training_batch(full_model_runtime, models[0], batch_size=2)
    mechanism, _, _ = _training_batch(full_model_runtime, models[1], batch_size=2)
    optimizers = [model.configure_optimizers()['optimizer'] for model in models]
    for model, optimizer in zip(models, optimizers):
        _attach_training_shell(model, optimizer, monkeypatch)
    initial_rng = capture_global_rng_state()
    monkeypatch.delenv('CT_V28_AUDIT_ACTIVATIONS', raising=False)
    for arm, model, optimizer in zip(('b0', 'b1_gru'), models, optimizers):
        monkeypatch.setenv('CT_V28_AUDIT_DIR', str(tmp_path / arm))
        restore_global_rng_state(initial_rng)
        batch = dict(ct_stream_schema='ct_seqtrack.train.v4', observation=observation,
                     mechanism=mechanism if arm == 'b1_gru' else None)
        optimizer.zero_grad(set_to_none=True)
        loss = model.training_step(batch, 0)
        loss.backward()
        model.on_before_optimizer_step(optimizer)
        optimizer.step()
        model.on_train_batch_end(loss, batch, 0)
    result = compare_audits(tmp_path / 'b0', tmp_path / 'b1_gru', steps=(1,))
    assert result['passed'], result
    initial = torch.load(tmp_path / 'b1_gru/step_000.pt', weights_only=False)
    assert 'physical_motion_encoder' not in initial['state']['training_flags']
    assert int(models[1].ct_b1_update_step) == 1
