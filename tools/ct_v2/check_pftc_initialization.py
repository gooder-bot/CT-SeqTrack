#!/usr/bin/env python3
"""Require B0, unweighted PFTC, and dt-PFTC to share exact step-0 state."""

from __future__ import annotations

import argparse
import hashlib
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


CONFIGS = {
    "b0": ROOT / "cfgs/ct_v2/01_seqtrack3d_baseline.yaml",
    "pftc_unweighted": (
        ROOT / "cfgs/ct_v2/06_seqtrack3d_pftc_unweighted.yaml"),
    "pftc": ROOT / "cfgs/ct_v2/07_seqtrack3d_dt_pftc.yaml",
}


class AttributeDict(dict):
    """Minimal EasyDict-compatible mapping for the standalone checker."""

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


def install_pointnet_import_stub_if_extension_is_missing() -> None:
    """Allow this parameter-only checker to run without PointNet++ CUDA ops."""
    try:
        import pointnet2_ops  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    pointnet2_module = types.ModuleType("pointnet2")
    utils_module = types.ModuleType("pointnet2.utils")
    modules_module = types.ModuleType("pointnet2.utils.pointnet2_modules")
    modules_module.PointnetSAModule = object
    sys.modules["pointnet2"] = pointnet2_module
    sys.modules["pointnet2.utils"] = utils_module
    sys.modules["pointnet2.utils.pointnet2_modules"] = modules_module


def install_easydict_stub_if_missing() -> None:
    try:
        import easydict  # noqa: F401
        return
    except ModuleNotFoundError:
        module = types.ModuleType("easydict")
        module.EasyDict = AttributeDict
        sys.modules["easydict"] = module


def build_model(config_path: Path, seed: int, train_steps: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    config = as_attribute_dict(load_yaml_config(config_path))
    return get_model(config.net_model)(
        config, train_dataloader_length=train_steps)


def state_sha256(state_dict) -> str:
    digest = hashlib.sha256()
    for key, value in state_dict.items():
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-steps", type=int, default=1262)
    args = parser.parse_args()
    install_pointnet_import_stub_if_extension_is_missing()
    install_easydict_stub_if_missing()

    states = {
        name: build_model(path, args.seed, args.train_steps).state_dict()
        for name, path in CONFIGS.items()
    }
    reference = states["b0"]
    for name, state in states.items():
        if reference.keys() != state.keys():
            raise AssertionError(f"{name}: state-dict keys differ from B0")
        mismatched = [
            key for key in reference
            if not torch.equal(reference[key], state[key])
        ]
        if mismatched:
            raise AssertionError(
                f"{name}: seeded initialization differs: "
                + ", ".join(mismatched[:10]))

    hashes = {name: state_sha256(state) for name, state in states.items()}
    print(f"seed={args.seed}")
    for name, value in hashes.items():
        print(f"{name}_sha256={value}")
    print("PASS_PFTC_SHARED_INITIALIZATION_EXACT")


if __name__ == "__main__":
    main()
