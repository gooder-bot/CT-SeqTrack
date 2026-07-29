#!/usr/bin/env python3
"""Verify that ordered B1 preserves B0 shared initialization exactly."""

from __future__ import annotations

import argparse
import json
import random
import sys
import types
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models import get_model  # noqa: E402
from utils.config import load_yaml_config  # noqa: E402


class AttributeDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as error:
            raise AttributeError(key) from error

    __setattr__ = dict.__setitem__


def as_attribute_dict(value):
    if isinstance(value, dict):
        return AttributeDict({
            key: as_attribute_dict(item) for key, item in value.items()
        })
    if isinstance(value, list):
        return [as_attribute_dict(item) for item in value]
    return value


def install_optional_import_stubs():
    try:
        import shapely  # noqa: F401
    except ModuleNotFoundError:
        shapely_module = types.ModuleType("shapely")
        shapely_module.__path__ = []
        geometry_module = types.ModuleType("shapely.geometry")
        geometry_module.Polygon = object
        sys.modules["shapely"] = shapely_module
        sys.modules["shapely.geometry"] = geometry_module
    try:
        import pytorch_lightning  # noqa: F401
    except ModuleNotFoundError:
        lightning_module = types.ModuleType("pytorch_lightning")

        class LightningModule(torch.nn.Module):
            def save_hyperparameters(self, *args, **kwargs):
                return None

            def log(self, *args, **kwargs):
                return None

        lightning_module.LightningModule = LightningModule
        sys.modules["pytorch_lightning"] = lightning_module
    try:
        import torchmetrics  # noqa: F401
    except ModuleNotFoundError:
        metrics_module = types.ModuleType("torchmetrics")
        metrics_module.__path__ = []

        class Accuracy(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()

            def forward(self, *args, **kwargs):
                return torch.zeros(2)

        class Metric(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()

            def add_state(self, name, default, dist_reduce_fx=None):
                setattr(self, name, default)

        metrics_module.Accuracy = Accuracy
        metrics_module.Metric = Metric
        utilities_module = types.ModuleType("torchmetrics.utilities")
        utilities_module.__path__ = []
        utilities_data_module = types.ModuleType("torchmetrics.utilities.data")
        utilities_data_module.dim_zero_cat = lambda values: torch.cat(values)
        utilities_module.data = utilities_data_module
        sys.modules["torchmetrics"] = metrics_module
        sys.modules["torchmetrics.utilities"] = utilities_module
        sys.modules["torchmetrics.utilities.data"] = utilities_data_module
    try:
        import pointnet2_ops  # noqa: F401
    except ModuleNotFoundError:
        pointnet2_module = types.ModuleType("pointnet2")
        utils_module = types.ModuleType("pointnet2.utils")
        modules_module = types.ModuleType("pointnet2.utils.pointnet2_modules")
        modules_module.PointnetSAModule = object
        sys.modules["pointnet2"] = pointnet2_module
        sys.modules["pointnet2.utils"] = utils_module
        sys.modules["pointnet2.utils.pointnet2_modules"] = modules_module
    try:
        import nuscenes  # noqa: F401
    except ModuleNotFoundError:
        nuscenes_module = types.ModuleType("nuscenes")
        nuscenes_module.__path__ = []
        nuscenes_utils_module = types.ModuleType("nuscenes.utils")
        nuscenes_utils_module.__path__ = []
        geometry_module = types.ModuleType("nuscenes.utils.geometry_utils")
        geometry_module.points_in_box = lambda *args, **kwargs: None
        nuscenes_core_module = types.ModuleType("nuscenes.nuscenes")
        nuscenes_core_module.NuScenes = object
        data_classes_module = types.ModuleType(
            "nuscenes.utils.data_classes")
        data_classes_module.LidarPointCloud = object
        data_classes_module.Box = object
        splits_module = types.ModuleType("nuscenes.utils.splits")
        splits_module.create_splits_scenes = lambda: {}
        nuscenes_utils_module.geometry_utils = geometry_module
        sys.modules["nuscenes"] = nuscenes_module
        sys.modules["nuscenes.utils"] = nuscenes_utils_module
        sys.modules["nuscenes.utils.geometry_utils"] = geometry_module
        sys.modules["nuscenes.nuscenes"] = nuscenes_core_module
        sys.modules["nuscenes.utils.data_classes"] = data_classes_module
        sys.modules["nuscenes.utils.splits"] = splits_module
    try:
        import easydict  # noqa: F401
    except ModuleNotFoundError:
        module = types.ModuleType("easydict")
        module.EasyDict = AttributeDict
        sys.modules["easydict"] = module


def build(config_path, seed, train_steps):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    config = as_attribute_dict(load_yaml_config(config_path))
    return get_model(config.net_model)(
        config, train_dataloader_length=train_steps)


def run_forward_smoke(model):
    model.eval()
    batch_size, history, point_count, extra_count = 2, 3, 1024, 128
    inputs = {
        "points": torch.randn(
            batch_size, (history + 1) * point_count, 5),
        "candidate_bc": torch.zeros(
            batch_size, (history + 1) * point_count, 9),
        "valid_mask": torch.ones(batch_size, history),
        "bbox_size": torch.tensor([[2.0, 4.0, 1.5]]).repeat(
            batch_size, 1),
        "ref_boxs": torch.tensor([[
            [0.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0, 0.0],
        ]]).repeat(batch_size, 1, 1),
        "delta_t": torch.tensor([[0.5, 0.5, 0.5]]).repeat(
            batch_size, 1),
        "delta_t_effective": torch.tensor([[1.0, 0.5, 0.5]]).repeat(
            batch_size, 1),
        "current_delta_t": torch.ones(batch_size),
        "current_delta_t_effective": torch.ones(batch_size),
        "delta_T": torch.tensor([[-0.5, -1.0, -1.5]]).repeat(
            batch_size, 1),
        "num_points_in_search": torch.full((batch_size,), 100.0),
        "search_has_usable_points": torch.ones(batch_size),
        "trajectory_search_points": torch.randn(
            batch_size, extra_count, 5),
        "trajectory_search_valid": torch.ones(batch_size),
    }
    with torch.no_grad():
        output = model(inputs)
    if output["aux_estimation_boxes"].shape != (batch_size, 4):
        raise AssertionError("ordered motion forward output shape changed")
    if output["trajectory_displacement_pred"].shape != (batch_size, 4):
        raise AssertionError("trajectory prediction shape changed")
    correction = output["trajectory_adapter_correction"]
    if not torch.equal(correction, torch.zeros_like(correction)):
        raise AssertionError("step-0 forward is not an exact adapter no-op")


def _assert_finite_nonzero_gradient(parameter, name):
    gradient = parameter.grad
    if gradient is None:
        raise AssertionError(f"{name} did not receive a gradient")
    if not torch.isfinite(gradient).all():
        raise AssertionError(f"{name} received a non-finite gradient")
    if not bool(torch.any(gradient != 0).item()):
        raise AssertionError(f"{name} received an all-zero gradient")


def run_gradient_smoke(model):
    """Verify that both zero-start heads can leave zero on the first update."""
    model.train()
    model.zero_grad(set_to_none=True)
    ref_boxs = torch.tensor([
        [[0.0, 0.0, 0.0, 0.0],
         [-0.5, 0.0, 0.0, 0.0],
         [-0.8, 0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0, 0.0],
         [-0.2, 0.1, 0.0, 0.05],
         [-0.5, 0.1, 0.0, 0.02]],
    ])
    delta_t = torch.full((2, 3), 0.5)
    valid = torch.ones(2, 3)
    trajectory = model.dynamics_encoder.forward_trajectory(
        ref_boxs, delta_t, valid, current_delta_t=torch.ones(2))
    target = trajectory["trajectory_displacement"].detach() + torch.tensor([
        [0.2, -0.1, 0.05, 0.1],
        [-0.1, 0.2, -0.05, -0.1],
    ])
    error = trajectory["trajectory_displacement"] - target
    log_sigma = trajectory["log_sigma"]
    trajectory_loss = (
        0.5 * error.pow(2) * torch.exp(-2.0 * log_sigma)
        + log_sigma
    ).mean()
    trajectory_loss.backward()
    _assert_finite_nonzero_gradient(
        model.dynamics_encoder.rate_residual_head.weight,
        "ordered trajectory residual head",
    )
    _assert_finite_nonzero_gradient(
        model.dynamics_encoder.log_sigma_head.weight,
        "ordered trajectory uncertainty head",
    )

    model.zero_grad(set_to_none=True)
    observation = torch.randn(2, 256)
    adapted, diagnostics = model.trajectory_adapter(
        observation,
        torch.randn(2, model.dynamics_hidden_dim),
        torch.randn(2, 64),
        torch.zeros(2, 4),
        torch.tensor([1.0, 2.0]),
        torch.ones(2, 1),
        torch.ones(2),
        enabled_scale=1.0,
    )
    if not torch.equal(adapted, observation):
        raise AssertionError("train-mode zero adapter is not an exact no-op")
    adapted.square().mean().backward()
    _assert_finite_nonzero_gradient(
        model.trajectory_adapter.net[-1].weight,
        "trajectory adapter zero-start head",
    )
    if not torch.equal(
            diagnostics["trajectory_adapter_correction"],
            torch.zeros_like(observation)):
        raise AssertionError("zero-start adapter emitted a correction")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-steps", type=int, default=1262)
    args = parser.parse_args()
    install_optional_import_stubs()

    baseline = build(
        ROOT / "cfgs/ct_v2/01_seqtrack3d_baseline.yaml",
        args.seed,
        args.train_steps,
    )
    motion = build(
        ROOT / "cfgs/ct_v2/02_ct_motion.yaml",
        args.seed,
        args.train_steps,
    )
    baseline_state = baseline.state_dict()
    motion_state = motion.state_dict()
    missing = sorted(set(baseline_state) - set(motion_state))
    mismatched = []
    for key, baseline_value in baseline_state.items():
        if key not in motion_state:
            continue
        if not torch.equal(baseline_value, motion_state[key]):
            mismatched.append(key)
    if missing or mismatched:
        raise AssertionError(json.dumps({
            "missing_shared_keys": missing,
            "mismatched_shared_keys": mismatched,
        }, indent=2))

    adapter_last = motion.trajectory_adapter.net[-1]
    if not torch.equal(
            adapter_last.weight, torch.zeros_like(adapter_last.weight)):
        raise AssertionError("trajectory adapter final weight is not zero")
    if not torch.equal(
            adapter_last.bias, torch.zeros_like(adapter_last.bias)):
        raise AssertionError("trajectory adapter final bias is not zero")
    run_forward_smoke(motion)
    run_gradient_smoke(motion)

    added = sorted(set(motion_state) - set(baseline_state))
    print(json.dumps({
        "status": "PASS",
        "shared_tensor_count": len(baseline_state),
        "added_tensor_count": len(added),
        "added_prefixes": sorted({
            key.split(".", 1)[0] for key in added
        }),
        "adapter_exact_zero": True,
        "forward_smoke": "PASS",
        "gradient_smoke": "PASS",
    }, indent=2))


if __name__ == "__main__":
    main()
