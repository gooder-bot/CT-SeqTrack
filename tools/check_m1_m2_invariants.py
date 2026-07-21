"""Dataset-free E3/E4/E5 checks for the M1/M2 model primitives."""

import importlib.util
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "ct_seqtrack_dynamics", ROOT / "models" / "dynamics.py")
DYNAMICS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DYNAMICS)

DynamicsEncoder = DYNAMICS.DynamicsEncoder
ZeroInitPhysicalTimeAdapter = DYNAMICS.ZeroInitPhysicalTimeAdapter
apply_proposal_innovation = DYNAMICS.apply_proposal_innovation


def assert_exact(left, right, message):
    if not torch.equal(left, right):
        gap = float(torch.max(torch.abs(left - right)).item())
        raise AssertionError(f"{message}; max gap={gap}")


def check_adapter_zero_init():
    torch.manual_seed(7)
    adapter = ZeroInitPhysicalTimeAdapter(
        feature_dim=8, dynamics_dim=4, hidden_dim=6, time_scale=0.5)
    point = torch.randn(3, 8)
    dynamics = torch.randn(3, 4)
    gap = torch.tensor([0.5, 1.0, 2.0])
    valid = torch.ones(3, 1)

    adapted, aux = adapter(point, dynamics, gap, valid, enabled_scale=1.0)
    assert_exact(adapted, point, "zero-initialized adapter is not an exact identity")
    assert_exact(
        aux["physical_time_adapter_correction"], torch.zeros_like(point),
        "zero-initialized adapter emitted a correction")

    disabled, _ = adapter(point, dynamics, gap, valid, enabled_scale=0.0)
    assert_exact(disabled, point, "disabled adapter is not an exact identity")


def check_innovation_fallbacks_and_bound():
    observation = torch.tensor([[1.0, 0.0, 0.0], [0.2, -0.1, 0.0]])
    dynamics = torch.tensor([[3.0, 0.0, 0.0], [0.8, 0.5, 0.0]])
    gap = torch.tensor([0.5, 2.0])
    valid = torch.ones(2, 1)

    zero, zero_aux = apply_proposal_innovation(
        observation, dynamics, valid, gap, alpha=0.0,
        base_radius=0.5, radius_per_second=0.5, max_radius=2.0)
    assert_exact(zero, observation, "alpha=0 did not recover observation")
    assert_exact(
        zero_aux["dynamics_innovation_applied"], torch.zeros_like(observation),
        "alpha=0 emitted a nonzero innovation")

    disabled, _ = apply_proposal_innovation(
        observation, dynamics, valid, gap, alpha=1.0, enabled_scale=0.0)
    assert_exact(disabled, observation, "enabled_scale=0 did not recover observation")

    invalid, invalid_aux = apply_proposal_innovation(
        observation, dynamics, torch.zeros_like(valid), gap, alpha=1.0)
    assert_exact(invalid, observation, "invalid dynamics did not recover observation")
    if not torch.equal(
            invalid_aux["dynamics_innovation_invalid_fallback"], torch.ones(2)):
        raise AssertionError("invalid fallback diagnostics are incorrect")

    final, aux = apply_proposal_innovation(
        observation, dynamics, valid, gap, alpha=1.0,
        base_radius=0.25, radius_per_second=0.25, max_radius=0.75)
    if not torch.all(aux["dynamics_innovation_applied_norm"]
                     <= aux["dynamics_innovation_radius"] + 1e-6):
        raise AssertionError("proposal innovation exceeded R(delta_t)")
    if not aux["dynamics_innovation_radius"][1] > aux["dynamics_innovation_radius"][0]:
        raise AssertionError("R(delta_t) did not increase with elapsed time")
    if not torch.isfinite(final).all():
        raise AssertionError("bounded proposal output is non-finite")

    nearby_dyn = observation + torch.tensor([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]])
    endpoint, _ = apply_proposal_innovation(
        observation, nearby_dyn, valid, gap, alpha=1.0,
        base_radius=10.0, radius_per_second=0.0, max_radius=10.0)
    if not torch.allclose(endpoint, nearby_dyn, atol=1e-7, rtol=0.0):
        raise AssertionError("alpha=1 did not reach an unclamped dynamics proposal")


class TinyM2Harness(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = DynamicsEncoder(hidden_dim=12, use_query_gap=True)
        self.adapter = ZeroInitPhysicalTimeAdapter(
            feature_dim=8, dynamics_dim=12, hidden_dim=10, time_scale=0.5)
        self.observation_head = nn.Linear(8, 3)

    def forward(self, point_feature, ref_boxs, delta_t, valid, current_gap):
        z_dyn, _, d_dyn, dynamics_valid = self.encoder(
            ref_boxs, delta_t, valid, current_gap)
        adapted, _ = self.adapter(
            point_feature, z_dyn, current_gap, dynamics_valid, enabled_scale=1.0)
        d_obs = self.observation_head(adapted)
        final, aux = apply_proposal_innovation(
            d_obs, d_dyn, dynamics_valid, current_gap,
            alpha=0.75, enabled_scale=1.0,
            base_radius=0.5, radius_per_second=0.5, max_radius=2.0)
        return final, aux


def check_two_optimizer_steps():
    torch.manual_seed(11)
    model = TinyM2Harness()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    batch_size = 4
    point = torch.randn(batch_size, 8)
    ref_boxs = torch.randn(batch_size, 3, 4)
    delta_t = torch.tensor([[0.5, 0.5, 1.0]]).repeat(batch_size, 1)
    valid = torch.ones(batch_size, 3)
    current_gap = torch.tensor([0.5, 1.0, 1.5, 2.0])
    target = torch.randn(batch_size, 3)

    encoder_grad_seen = False
    adapter_grad_seen = False
    for _ in range(2):
        optimizer.zero_grad()
        final, aux = model(point, ref_boxs, delta_t, valid, current_gap)
        loss = torch.mean((final - target) ** 2)
        if not torch.isfinite(loss):
            raise AssertionError("M2 two-step loss is non-finite")
        loss.backward()
        for parameter in model.encoder.parameters():
            if parameter.grad is not None and torch.isfinite(parameter.grad).all() \
                    and torch.count_nonzero(parameter.grad).item() > 0:
                encoder_grad_seen = True
        for parameter in model.adapter.parameters():
            if parameter.grad is not None and torch.isfinite(parameter.grad).all() \
                    and torch.count_nonzero(parameter.grad).item() > 0:
                adapter_grad_seen = True
        if not torch.all(aux["dynamics_innovation_applied_norm"]
                         <= aux["dynamics_innovation_radius"] + 1e-6):
            raise AssertionError("training step violated the innovation bound")
        optimizer.step()

    if not encoder_grad_seen:
        raise AssertionError("DynamicsEncoder received no finite nonzero gradient")
    if not adapter_grad_seen:
        raise AssertionError("physical-time adapter received no finite nonzero gradient")


def main():
    check_adapter_zero_init()
    check_innovation_fallbacks_and_bound()
    check_two_optimizer_steps()
    print("M1/M2 dataset-free invariants and 2-step optimizer: PASS")


if __name__ == "__main__":
    main()
