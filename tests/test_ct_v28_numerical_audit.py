"""数值审计不能改变训练；精确比较必须拒绝任何字节差异。"""
import copy
import json

import pytest
import torch
from torch import nn

from tools.compare_ct_v28_audits import compare_audits
from utils.deterministic_pooling import DeterministicMaxPool1d
from utils.v28_numerical_audit import B0NumericalAudit, compare_values, snapshot


class TinyB0(nn.Module):
    ct_enable_v28 = True
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(2, 3, 1)
        self.bn = nn.BatchNorm1d(3)
        self.pool = DeterministicMaxPool1d(1)
    @staticmethod
    def _ct_any_plugin_parameter(name):
        return False
    def forward(self, value):
        return self.pool(self.bn(self.conv(value))).squeeze(-1)


def test_audit_is_observational_and_saves_first_update(tmp_path):
    torch.manual_seed(42)
    plain = TinyB0().train()
    audited = copy.deepcopy(plain)
    data = torch.randn(2, 2, 8)
    optimizer_a = torch.optim.Adam([{'name': 'b0', 'params': plain.parameters()}],
                                  lr=1e-4, foreach=False, fused=False)
    optimizer_b = torch.optim.Adam([{'name': 'b0', 'params': audited.parameters()}],
                                  lr=1e-4, foreach=False, fused=False)
    audit = B0NumericalAudit(audited, tmp_path / 'a')
    rng = torch.get_rng_state().clone()
    expected = plain(data)
    expected.square().mean().backward()
    optimizer_a.step()
    audit.begin_observation({'points': data}, 1)
    observed = audited(data)
    loss = observed.square().mean()
    audit.end_observation({'motion_cls': observed}, {'loss_total': loss})
    loss.backward()
    audit.before_optimizer(optimizer_b)
    optimizer_b.step()
    audit.after_optimizer(optimizer_b)
    assert torch.equal(rng, torch.get_rng_state())
    assert compare_values(snapshot(plain.state_dict()), snapshot(audited.state_dict())) is None
    for a, b in zip(plain.parameters(), audited.parameters()):
        assert torch.equal(a.grad, b.grad)
        assert compare_values(optimizer_a.state[a], optimizer_b.state[b]) is None
    row = torch.load(tmp_path / 'a/step_001.pt', weights_only=False)
    assert row['activations'] and row['activation_gradients'] and row['pool_indices']
    assert set(row['gradients']) == set(dict(audited.named_parameters()))
    assert any(delta.abs().sum() > 0 for delta in row['actual_parameter_delta'].values())
    assert compare_audits(tmp_path / 'a', tmp_path / 'a', steps=(1,))['passed']
    from tools.replay_ct_v28_adam import replay_adam
    assert replay_adam(row)['passed']
    # 重放第二步，覆盖已有的Adam一阶/二阶矩和step状态。
    optimizer_b.zero_grad(set_to_none=True)
    audit.begin_observation({'points': data}, 2)
    observed = audited(data)
    loss = observed.square().mean()
    audit.end_observation({'motion_cls': observed}, {'loss_total': loss})
    loss.backward()
    audit.before_optimizer(optimizer_b)
    optimizer_b.step()
    audit.after_optimizer(optimizer_b)
    second = torch.load(tmp_path / 'a/step_002.pt', weights_only=False)
    assert replay_adam(second)['passed']
    second['gradients']['conv.weight'][0, 0, 0] += 1
    assert not replay_adam(second)['passed']


def test_exact_comparison_rejects_signed_zero_missing_steps_and_small_differences(tmp_path):
    difference = compare_values(torch.tensor([0.]), torch.tensor([-0.]))
    assert difference and difference['max_abs_error_diagnostic_only'] == 0.
    difference = compare_values(torch.tensor([1.]), torch.tensor([1.0000001]))
    assert difference and difference['max_abs_error_diagnostic_only'] < 1e-6
    assert not compare_audits(tmp_path, tmp_path)['passed']
    json.dumps(difference, allow_nan=False)


def test_mechanism_boundary_rejects_bn_flags_parameters_and_rng_mutation(tmp_path):
    model = TinyB0()
    audit = B0NumericalAudit(model, tmp_path / 'a')
    before = audit.b0_state()
    audit.verify_mechanism(before)
    with torch.no_grad():
        model.bn.running_mean[0] += 1.
    with pytest.raises(RuntimeError, match='mechanism changed'):
        audit.verify_mechanism(before)
    model.bn.running_mean[0] -= 1.
    model.bn.eval()
    with pytest.raises(RuntimeError, match='training_flags'):
        audit.verify_mechanism(before)


def test_audit_never_overwrites_an_existing_run(tmp_path):
    (tmp_path / 'existing.pt').write_bytes(b'existing evidence')
    with pytest.raises(FileExistsError, match='empty directory'):
        B0NumericalAudit(TinyB0(), tmp_path)
