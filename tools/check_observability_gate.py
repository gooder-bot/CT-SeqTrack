import sys
from pathlib import Path
import importlib.util

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("observability", ROOT / "models" / "observability.py")
observability = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observability)
ObservabilityGate = observability.ObservabilityGate


def main():
    torch.manual_seed(42)
    gate = ObservabilityGate(
        feature_dim=256,
        dynamics_dim=128,
        stats_dim=5,
        hidden_dim=64,
        init_obs_bias=1.0,
        min_dyn_valid=0.5,
    )
    point_feature = torch.randn(2, 256)
    z_dyn = torch.randn(2, 128)
    obs_stats = torch.tensor([
        [3.0, 2.0, 0.8, 1.0, 1.0],
        [0.0, 0.1, 0.1, 0.0, 3.0],
    ], dtype=torch.float32)
    dynamics_valid = torch.tensor([[1.0], [0.0]])

    fused, aux = gate(point_feature, z_dyn, obs_stats, dynamics_valid)
    alpha = aux["obs_alpha"]
    finite = all(torch.isfinite(value).all().item() for value in [fused, alpha, aux["obs_gate_entropy"]])
    alpha_sum_ok = torch.allclose(alpha.sum(dim=1), torch.ones(2), atol=1e-6)
    invalid_dyn_ok = torch.allclose(alpha[1], torch.tensor([1.0, 0.0]), atol=1e-6)

    print(f"fused shape: {tuple(fused.shape)}, finite={finite}")
    print(f"alpha: {alpha.detach().cpu().numpy()}")
    print(f"alpha_sum_ok: {bool(alpha_sum_ok)}")
    print(f"invalid_dyn_ok: {bool(invalid_dyn_ok)}")
    print(f"entropy: {aux['obs_gate_entropy'].detach().cpu().numpy()}")

    if not finite or not alpha_sum_ok or not invalid_dyn_ok:
        raise RuntimeError("ObservabilityGate smoke test failed.")


if __name__ == "__main__":
    main()
