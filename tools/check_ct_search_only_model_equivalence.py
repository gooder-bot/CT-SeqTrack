#!/usr/bin/env python3
"""Require B0 and search-only to have identical seeded model initialization."""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from pathlib import Path

import numpy as np
import torch
from easydict import EasyDict


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import get_model  # noqa: E402
from utils.config import load_yaml_config  # noqa: E402


def build_model(config_path: Path, seed: int, train_steps: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    config = EasyDict(load_yaml_config(config_path))
    return get_model(config.net_model)(
        config,
        train_dataloader_length=train_steps,
    )


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

    baseline_path = ROOT / "cfgs/ct_v2/01_seqtrack3d_baseline.yaml"
    search_path = ROOT / "cfgs/ct_v2/05_seqtrack3d_search_only.yaml"
    baseline = build_model(baseline_path, args.seed, args.train_steps)
    search_only = build_model(search_path, args.seed, args.train_steps)
    baseline_state = baseline.state_dict()
    search_state = search_only.state_dict()

    if baseline_state.keys() != search_state.keys():
        missing = sorted(baseline_state.keys() - search_state.keys())
        extra = sorted(search_state.keys() - baseline_state.keys())
        raise AssertionError(
            f"state-dict key mismatch; missing={missing}, extra={extra}")
    mismatched = [
        key for key in baseline_state
        if not torch.equal(baseline_state[key], search_state[key])
    ]
    if mismatched:
        raise AssertionError(
            "seeded shared initialization mismatch: "
            + ", ".join(mismatched[:10]))

    baseline_hash = state_sha256(baseline_state)
    search_hash = state_sha256(search_state)
    parameter_count = sum(
        parameter.numel() for parameter in baseline.parameters())
    print(f"seed={args.seed}")
    print(f"state_tensors={len(baseline_state)}")
    print(f"parameters={parameter_count}")
    print(f"baseline_sha256={baseline_hash}")
    print(f"search_only_sha256={search_hash}")
    print("PASS_SEARCH_ONLY_MODEL_INIT_EXACT")


if __name__ == "__main__":
    main()
