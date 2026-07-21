"""Real-batch strict A1 equivalence check for disabled M1/M2 paths.

Run this on the training server after the dataset-free checks.  It loads one
shared-SE(2) batch, copies all common A1 weights into the proposal model, then
requires zero adapter/innovation scales to produce the same core outputs and
losses on that exact batch.
"""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from easydict import EasyDict
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets import get_dataset  # noqa: E402
from models import get_model  # noqa: E402


CORE_OUTPUT_KEYS = (
    "seg_logits",
    "motion_pred",
    "estimation_boxes",
    "aux_estimation_boxes",
    "updated_ref_boxs",
)


def load_config(path):
    with open(path, "r") as handle:
        cfg = EasyDict(yaml.load(handle, Loader=yaml.FullLoader))
    cfg.preloading = False
    if "tiny" not in cfg:
        cfg.tiny = False
    return cfg


def move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    return value


def full_history(batch, hist_num):
    valid = batch["valid_mask"].detach().cpu().numpy()
    return bool(np.all(valid.sum(axis=1) >= int(hist_num)))


def copy_common_weights(source, target):
    source_state = source.state_dict()
    target_state = target.state_dict()
    copied = 0
    for key, value in source_state.items():
        if key in target_state and target_state[key].shape == value.shape:
            target_state[key] = value.detach().clone()
            copied += 1
    target.load_state_dict(target_state, strict=True)
    return copied


def load_a1_weights(model, checkpoint_path):
    """Load a frozen A1 checkpoint before copying its common tensors to M1/M2."""
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint payload: {type(payload)}")
    state_dict = payload.get("state_dict", payload.get("model", payload))
    target_state = model.state_dict()
    candidates = [state_dict]
    for prefix in ("model.", "module."):
        candidates.append({
            key[len(prefix):] if key.startswith(prefix) else key: value
            for key, value in state_dict.items()
        })
    normalized = max(
        candidates,
        key=lambda state: sum(
            key in target_state and target_state[key].shape == value.shape
            for key, value in state.items()),
    )
    matched = {
        key: value for key, value in normalized.items()
        if key in target_state and target_state[key].shape == value.shape
    }
    critical_prefixes = (
        "seg_pointnet.", "mini_pointnet.", "motion_mlp.",
        "feature_pointnet.", "Transformer.",
    )
    missing_critical = [
        prefix for prefix in critical_prefixes
        if not any(key.startswith(prefix) for key in matched)
    ]
    if missing_critical:
        raise RuntimeError(
            "A1 checkpoint is missing critical prefixes: "
            + ", ".join(missing_critical))
    target_state.update(matched)
    model.load_state_dict(target_state, strict=True)
    return len(matched), len(target_state)


def assert_close(name, left, right, atol):
    if left.shape != right.shape:
        raise AssertionError(f"{name} shape mismatch: {left.shape} != {right.shape}")
    if not torch.allclose(left, right, atol=atol, rtol=0.0):
        gap = float(torch.max(torch.abs(left - right)).item())
        raise AssertionError(f"{name} mismatch; max gap={gap}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg",
        default="cfgs/seqtrack3d_nuscenes_m2_proposal_innovation_engineering.yaml",
    )
    parser.add_argument("--path")
    parser.add_argument("--version")
    parser.add_argument("--split")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument(
        "--weights",
        default=None,
        help="Optional frozen A1 checkpoint used for the strict equivalence pass.",
    )
    args = parser.parse_args()

    cfg = load_config(args.cfg)
    if args.path:
        cfg.path = args.path
    if args.version:
        cfg.version = args.version
    cfg.batch_size = args.batch_size
    cfg.workers = args.workers
    split = args.split or cfg.train_split
    dataset = get_dataset(cfg, type=cfg.train_type, split=split)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.workers,
        shuffle=False,
        drop_last=False,
    )
    batch = next(
        candidate for candidate in loader
        if full_history(candidate, cfg.hist_num)
    )

    a1_cfg = copy.deepcopy(cfg)
    a1_cfg.use_dynamics_encoder = False
    a1_cfg.use_physical_time_adapter = False
    a1_cfg.dynamics_motion_mode = "feature"
    a1_cfg.velocity_weight = 0.0
    a1_cfg.dynamics_displacement_weight = 0.0

    zero_cfg = copy.deepcopy(cfg)
    zero_cfg.physical_time_adapter_scale = 0.0
    zero_cfg.dynamics_innovation_scale = 0.0
    zero_cfg.velocity_weight = 0.0
    zero_cfg.dynamics_displacement_weight = 0.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a1 = get_model(a1_cfg.net_model)(a1_cfg).to(device).eval()
    proposal = get_model(zero_cfg.net_model)(zero_cfg).to(device).eval()
    loaded = None
    if args.weights is not None:
        loaded = load_a1_weights(a1, args.weights)
    copied = copy_common_weights(a1, proposal)
    batch = move_to_device(batch, device)

    with torch.no_grad():
        output_a1 = a1(batch)
        output_zero = proposal(batch)
    for key in CORE_OUTPUT_KEYS:
        assert_close(key, output_a1[key], output_zero[key], args.atol)
    assert_close(
        "physical_time_adapter_correction",
        output_zero["physical_time_adapter_correction"],
        torch.zeros_like(output_zero["physical_time_adapter_correction"]),
        0.0,
    )
    assert_close(
        "dynamics_innovation_applied",
        output_zero["dynamics_innovation_applied"],
        torch.zeros_like(output_zero["dynamics_innovation_applied"]),
        0.0,
    )

    if device.type == "cuda":
        with torch.no_grad():
            loss_a1 = a1.compute_loss(batch, output_a1)
            loss_zero = proposal.compute_loss(batch, output_zero)
        for key in sorted(set(loss_a1).intersection(loss_zero)):
            if torch.is_tensor(loss_a1[key]) and torch.is_tensor(loss_zero[key]):
                assert_close(f"loss/{key}", loss_a1[key], loss_zero[key], args.atol)
    else:
        print("loss equivalence skipped: current compute_loss requires CUDA")

    print(f"device={device}, loaded_a1_tensors={loaded}, copied_common_tensors={copied}")
    print("M1/M2 strict-zero A1 model equivalence: PASS")


if __name__ == "__main__":
    main()
