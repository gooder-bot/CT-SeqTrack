"""Zero-step launch gate for the registered CT-SeqTrack v26 arms.

This tool does not create a Trainer, read a training sample, or write a
checkpoint.  It validates the full-data layout and constructs the selected
model/optimizer so dependency, parameter-group and accidental-freezing errors
fail before a 60-epoch allocation is submitted.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.ct_variant import configure_ct_variant
from utils.config import load_yaml_config
from utils.online_contract import (
    require_scratch_initialization,
    validate_scratch_training_contract,
)


FORMAL_CONFIGS = {
    "seqtrack-strict": ROOT / "cfgs/26_seqtrack_strict_nuscenes_full.yaml",
    "b0": ROOT / "cfgs/ct_seqtrack/26_b0_nuscenes_full.yaml",
    "b1-gru": ROOT / "cfgs/ct_seqtrack/26_b1_gru_nuscenes_full.yaml",
    "b1-cfc": ROOT / "cfgs/ct_seqtrack/26_b1_cfc_nuscenes_full.yaml",
    "full-b3": ROOT / "cfgs/ct_seqtrack/26_full_minus_b3_nuscenes_full.yaml",
    "full": ROOT / "cfgs/ct_seqtrack/26_full_nuscenes_full.yaml",
}

NUSCENES_TABLES = (
    "attribute.json", "calibrated_sensor.json", "category.json",
    "ego_pose.json", "instance.json", "log.json", "map.json",
    "sample.json", "sample_annotation.json", "sample_data.json",
    "scene.json", "sensor.json", "visibility.json",
)


def _validate_data_root(data_root, minimum_lidar_files=30000):
    data_root = Path(data_root).expanduser().resolve()
    table_root = data_root / "v1.0-trainval"
    missing = [str(table_root / name) for name in NUSCENES_TABLES
               if not (table_root / name).is_file()]
    lidar_root = data_root / "samples/LIDAR_TOP"
    if not lidar_root.is_dir():
        missing.append(str(lidar_root))
    else:
        lidar_count = sum(1 for _ in lidar_root.glob("*.pcd.bin"))
        if lidar_count < int(minimum_lidar_files):
            missing.append(
                f"{lidar_root} has {lidar_count} keyframes; "
                f"expected at least {int(minimum_lidar_files)}")
    if missing:
        raise FileNotFoundError(
            "nuScenes full data root is incomplete: " + ", ".join(missing))
    return data_root


def _load_arm(arm, data_root):
    config = load_yaml_config(FORMAL_CONFIGS[arm])
    config["path"] = str(data_root)
    config["test"] = False
    config["checkpoint"] = None
    config["init_checkpoint"] = None
    if str(config["net_model"]).strip().lower() == "ctseqtrack":
        configure_ct_variant(config)
    require_scratch_initialization(config, None)
    validate_scratch_training_contract(config)
    for key in ("limit_train_batches", "limit_val_batches"):
        value = config.get(key, 1.0)
        if not isinstance(value, float) or value != 1.0:
            raise ValueError(f"formal v26 requires {key}=float 1.0")
    return config


def _runtime_check(config):
    import torch
    import pytorch_lightning as pl
    from easydict import EasyDict

    config = EasyDict(config)

    if str(pl.__version__) != "2.0.2":
        raise RuntimeError(
            f"pytorch-lightning==2.0.2 is required, got {pl.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.cuda.device_count() < 1:
        raise RuntimeError("no visible CUDA device")
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    importlib.import_module("nuscenes")
    importlib.import_module("pointnet2_ops.pointnet2_utils")

    from models import get_model

    model = get_model(config.net_model)(
        config, train_dataloader_length=1)
    optimizer_contract = model.configure_optimizers()
    frozen_active = []
    enabled_plugins = {
        "physical_motion_encoder.": bool(getattr(config, "ct_enable_b1", False)),
        "ct_joint_search_refiner.": bool(getattr(config, "ct_enable_b2", False)),
        "ct_joint_router.": bool(getattr(config, "ct_enable_b3", False)),
    }
    for name, parameter in model.named_parameters():
        plugin_matches = [prefix for prefix in enabled_plugins
                          if name.startswith(prefix)]
        active = (enabled_plugins[plugin_matches[0]]
                  if plugin_matches else True)
        if active and not parameter.requires_grad:
            frozen_active.append(name)
    if frozen_active:
        raise RuntimeError(
            "formal arm contains frozen active parameters: "
            + ", ".join(sorted(frozen_active)))
    named_groups = list(getattr(model, "_ct_optimizer_names", ()))
    if str(config.net_model).strip().lower() == "ctseqtrack":
        expected = ["b0"] + [name for name in ("b1", "b2", "b3")
                             if bool(getattr(config, f"ct_enable_{name}", False))]
        if named_groups != expected:
            raise RuntimeError(
                f"optimizer groups {named_groups!r} do not match {expected!r}")
    trainable = sum(parameter.numel() for parameter in model.parameters()
                    if parameter.requires_grad)
    del optimizer_contract, model
    return {
        "torch": str(torch.__version__),
        "pytorch_lightning": str(pl.__version__),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_device_0": str(torch.cuda.get_device_name(0)),
        "cuda_free_gib": round(free_bytes / (1024 ** 3), 2),
        "cuda_total_gib": round(total_bytes / (1024 ** 3), 2),
        "trainable_parameters": int(trainable),
        "optimizer_groups": named_groups,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=tuple(FORMAL_CONFIGS), required=True)
    parser.add_argument("--path", required=True, help="nuScenes full root")
    parser.add_argument(
        "--config-only", action="store_true",
        help="skip CUDA/dependency/model construction (local CI only)")
    args = parser.parse_args()

    data_root = _validate_data_root(args.path)
    config = _load_arm(args.arm, data_root)
    disk = shutil.disk_usage(ROOT)
    report = {
        "schema": "ct_seqtrack.v26_launch_preflight.v1",
        "passed": True,
        "arm": args.arm,
        "config": str(FORMAL_CONFIGS[args.arm]),
        "data_root": str(data_root),
        "epochs": int(config["epoch"]),
        "seed": int(config["seed"]),
        "trainer_devices": int(config["trainer_devices"]),
        "final_checkpoint_window": int(
            config["ct_keep_final_window_checkpoints"]),
        "disk_free_gib": round(disk.free / (1024 ** 3), 2),
    }
    if not args.config_only:
        report["runtime"] = _runtime_check(config)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
