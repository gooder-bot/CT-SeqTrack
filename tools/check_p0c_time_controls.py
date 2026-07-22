import argparse
import copy
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from easydict import EasyDict


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.misc_utils import (  # noqa: E402
    build_effective_time_fields,
    build_time_fields,
)
from datasets.protocol_utils import (  # noqa: E402
    payload_with_content_sha256,
    resolve_virtual_rate_kwargs,
    verify_content_sha256,
)


EFFECTIVE_ONLY_KEYS = {
    "delta_t_effective",
    "current_delta_t_effective",
    "timestamps_effective",
    "delta_T_effective",
    "current_effective_timestamp",
    "dynamics_time_mode_id",
}


def load_config(path):
    with open(path, "r") as handle:
        cfg = EasyDict(yaml.load(handle, Loader=yaml.FullLoader))
    cfg.preloading = False
    if "tiny" not in cfg:
        cfg.tiny = False
    return cfg


def merge_protocol_config(cfg, path):
    if path is None:
        return
    with open(path, "r", encoding="utf-8") as handle:
        protocol_cfg = yaml.load(handle, Loader=yaml.FullLoader)
    for key, value in protocol_cfg.items():
        if "virtual_rate" in key:
            setattr(cfg, key, value)


def set_virtual_rate_manifest(cfg, role, path):
    if path is None:
        return
    setattr(cfg, f"{role}_virtual_rate_manifest", path)
    setattr(cfg, f"virtual_rate_manifest_{role}", path)
    setattr(cfg, f"{role}_virtual_rate_manifest_strict", True)
    setattr(cfg, f"{role}_virtual_rate_manifest_require_commit_match", True)


def to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def assert_equal_value(key, left, right, atol=1e-7):
    left_array = to_numpy(left)
    right_array = to_numpy(right)
    if left_array.shape != right_array.shape:
        raise AssertionError(
            f"{key} shape changed: {left_array.shape} != {right_array.shape}")
    if left_array.dtype.kind in "fci" or right_array.dtype.kind in "fci":
        if not np.allclose(left_array, right_array, atol=atol, rtol=0.0, equal_nan=True):
            gap = float(np.max(np.abs(left_array - right_array)))
            raise AssertionError(f"{key} changed outside effective time; max gap={gap}")
    elif not np.array_equal(left_array, right_array):
        raise AssertionError(f"{key} changed outside effective time")


def assert_only_effective_time_changed(reference, candidate):
    if set(reference) != set(candidate):
        raise AssertionError(
            f"batch key set changed: {set(reference) ^ set(candidate)}")
    for key in sorted(reference):
        if key not in EFFECTIVE_ONLY_KEYS:
            assert_equal_value(key, reference[key], candidate[key])


def self_test():
    cfg = EasyDict({
        "virtual_rate_mode": "legacy",
        "train_virtual_rate_mode": "none",
        "eval_virtual_rate_mode": "gap_pattern",
        "virtual_rate_manifest_val": "val.json",
    })
    assert resolve_virtual_rate_kwargs(cfg, "train")["virtual_rate_mode"] == "none"
    val = resolve_virtual_rate_kwargs(cfg, "val")
    assert val["virtual_rate_mode"] == "gap_pattern"
    assert val["virtual_rate_manifest"] == "val.json"

    payload = payload_with_content_sha256({"schema": "test", "values": [1, 2, 3]})
    verify_content_sha256(payload, "self-test")
    tampered = dict(payload)
    tampered["values"] = [3, 2, 1]
    try:
        verify_content_sha256(tampered, "tampered self-test")
    except ValueError:
        pass
    else:
        raise AssertionError("Manifest tampering was not detected")

    real = build_time_fields(
        [9.5, 8.5, 7.0], 10.0,
        frame_ids=[3, 2, 1], current_frame_id=4, default_step=0.5)
    true_fields = build_effective_time_fields("true", real)
    fixed_fields = build_effective_time_fields(
        "fixed", real,
        effective_frame_timestamps=[1.5, 1.0, 0.5],
        effective_current_timestamp=2.0,
        frame_ids=[3, 2, 1], current_frame_id=4, default_step=0.5)
    shuffled_fields = build_effective_time_fields(
        "shuffled", real,
        effective_frame_timestamps=[2.25, 1.25, 0.5],
        effective_current_timestamp=3.0,
        frame_ids=[3, 2, 1], current_frame_id=4, default_step=0.5)
    assert np.allclose(true_fields[1], real[1])
    assert np.allclose(fixed_fields[1], [0.5, 0.5, 0.5])
    assert np.allclose(shuffled_fields[1], [0.75, 1.0, 0.75])

    base = {"points": np.ones((4, 3)), "label": np.array([1]),
            "delta_t_effective": np.array(real[1]), "dynamics_time_mode_id": 0}
    negative = copy.deepcopy(base)
    negative["delta_t_effective"] = np.array(fixed_fields[1])
    negative["dynamics_time_mode_id"] = 1
    assert_only_effective_time_changed(base, negative)
    print("P0-C protocol self-test: PASS")


def set_role_value(cfg, role, key, value):
    setattr(cfg, f"{role}_{key}", value)
    if key == "dynamics_time_manifest":
        setattr(cfg, f"dynamics_time_manifest_{role}", value)


def make_sampler(cfg, role, split, mode, shuffled_manifest):
    from datasets import get_dataset

    local = copy.deepcopy(cfg)
    local.use_twc = False
    local.use_augmentation = False
    set_role_value(local, role, "dynamics_time_mode", mode)
    set_role_value(
        local, role, "dynamics_time_manifest",
        shuffled_manifest if mode == "shuffled" else "")
    if mode == "shuffled":
        set_role_value(local, role, "dynamics_time_manifest_strict", True)
        set_role_value(
            local, role, "dynamics_time_manifest_require_commit_match", True)
    return get_dataset(
        local, type="train_motion_mf", split=split, protocol_role=role)


def choose_full_history_index(sampler, hist_num):
    for tracklet_id in range(sampler.dataset.get_num_tracklets()):
        if sampler.dataset.get_num_frames_tracklet(tracklet_id) > hist_num:
            anno_id = sampler.tracklet_start_ids[tracklet_id] + hist_num
            return anno_id * sampler.num_candidates
    raise RuntimeError("No tracklet has enough frames for a full-history regression sample")


def deterministic_sample(sampler, index, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return sampler[index]


def integration_check(args):
    cfg = load_config(args.cfg)
    merge_protocol_config(cfg, args.protocol_cfg)
    set_virtual_rate_manifest(cfg, args.role, args.virtual_rate_manifest)
    if args.path is not None:
        cfg.path = args.path
    if args.version is not None:
        cfg.version = args.version
    split = args.split or getattr(cfg, f"{args.role}_split", cfg.test_split)
    samplers = {
        mode: make_sampler(cfg, args.role, split, mode, args.shuffled_manifest)
        for mode in ("true", "fixed", "shuffled")
    }
    index = choose_full_history_index(samplers["true"], int(cfg.hist_num))
    samples = {
        mode: deterministic_sample(sampler, index, args.seed)
        for mode, sampler in samplers.items()
    }
    assert_only_effective_time_changed(samples["true"], samples["fixed"])
    assert_only_effective_time_changed(samples["true"], samples["shuffled"])
    assert_equal_value(
        "true delta_t_real/effective",
        samples["true"]["delta_t_real"],
        samples["true"]["delta_t_effective"],
    )
    for mode, sample in samples.items():
        print(
            f"{mode}: real={to_numpy(sample['delta_t_real']).tolist()} "
            f"effective={to_numpy(sample['delta_t_effective']).tolist()} "
            f"current_real={float(sample['current_delta_t_real']):.6f} "
            f"current_effective={float(sample['current_delta_t_effective']):.6f}")
    print("P0-C true/fixed/shuffled batch invariance: PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--cfg")
    parser.add_argument("--shuffled-manifest")
    parser.add_argument("--role", choices=("train", "val", "test"), default="test")
    parser.add_argument("--split")
    parser.add_argument("--path")
    parser.add_argument("--version")
    parser.add_argument("--protocol-cfg")
    parser.add_argument("--virtual-rate-manifest")
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.cfg or not args.shuffled_manifest:
        parser.error("--cfg and --shuffled-manifest are required without --self-test")
    integration_check(args)


if __name__ == "__main__":
    main()
