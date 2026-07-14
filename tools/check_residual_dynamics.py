#!/usr/bin/env python3
"""Dataset-free smoke test for bounded observation-first dynamics residuals."""

import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location(
    "ct_seqtrack_dynamics", ROOT / "models" / "dynamics.py")
dynamics_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dynamics_module)
DynamicsResidualGate = dynamics_module.DynamicsResidualGate
clamp_vector_norm = dynamics_module.clamp_vector_norm


def main():
    vectors = torch.tensor([
        [3.0, 4.0, 0.0],
        [0.1, 0.2, 0.2],
        [0.0, 0.0, 0.0],
    ])
    clamped, raw_norm, clamp_mask = clamp_vector_norm(vectors, max_norm=1.0)
    clamped_norm = torch.linalg.norm(clamped, dim=1)

    assert torch.isfinite(clamped).all()
    assert torch.all(clamped_norm <= 1.0 + 1e-6)
    assert bool(clamp_mask[0].item())
    assert not bool(clamp_mask[1].item())
    assert abs(float(raw_norm[0].item()) - 5.0) < 1e-6

    gate = DynamicsResidualGate(
        stats_dim=6,
        hidden_dim=8,
        max_alpha=0.2,
        init_alpha=0.0,
    )
    stats = torch.randn(3, 6)
    dynamics_valid = torch.tensor([[1.0], [0.0], [1.0]])
    alpha = gate(stats, dynamics_valid)

    assert torch.isfinite(alpha).all()
    assert torch.all(alpha >= 0.0)
    assert torch.all(alpha <= 0.2 + 1e-6)
    assert float(alpha[1].item()) == 0.0
    assert float(alpha[[0, 2]].max().item()) < 1e-3

    residual_scale = 0.1
    residual = residual_scale * alpha * clamped
    residual_norm = torch.linalg.norm(residual, dim=1)
    expected_bound = residual_scale * gate.max_alpha * 1.0
    assert torch.all(residual_norm <= expected_bound + 1e-6)

    residual.sum().backward()
    parameter_grads = [parameter.grad for parameter in gate.parameters()
                       if parameter.grad is not None]
    assert parameter_grads
    assert all(torch.isfinite(grad).all() for grad in parameter_grads)

    print("bounded residual smoke test: PASS")
    print("raw norms:", raw_norm.squeeze(1).tolist())
    print("clamped norms:", clamped_norm.tolist())
    print("initial alpha:", alpha.squeeze(1).tolist())
    print("residual norms:", residual_norm.tolist())
    print("maximum residual norm:", expected_bound)


if __name__ == "__main__":
    main()
