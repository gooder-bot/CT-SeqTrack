"""
main.py
Created by zenn at 2021/7/18 15:08
Modified by Aron Lin at Jun 1  09:42:22 CST 2023
"""

import pytorch_lightning as pl
import argparse

# import pytorch_lightning.utilities.distributed
import torch
import numpy as np
import random
from easydict import EasyDict
import os
import json
from pathlib import Path

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything


from datasets import get_dataset
from datasets.sampler import (
    OnlineRecursiveBatchSampler,
    PartitionedTestTrackingSampler,
    online_recursive_collate,
)
from models import get_model
from ctseqtrack.config import configure_ct_variant
from ctseqtrack.runtime.provenance import write_run_provenance
from ctseqtrack.runtime.acquisition import validate_preflight_artifact
from ctseqtrack.runtime.contracts import (
    validate_scratch_training_contract,
    validate_b2_method_promotion,
    validate_online_resume_contract,
)
from ctseqtrack.runtime.calibration import b1_calibration_config_sha256

if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

import sys

import datetime
import time
from utils.config import load_yaml_config


def generate_log_folder_name(cfg):
    if cfg.get("log_dir"):
        return cfg["log_dir"]
    now = datetime.datetime.now()
    time_str = now.strftime("%Y%m%d-%H%M")
    cfg_name = cfg["cfg"].split("/")[-1].replace(".yaml", "")
    folder_name = f"output/{time_str}-{cfg_name}-{cfg['tag']}"
    return folder_name


def load_yaml(file_name):
    return load_yaml_config(file_name)


def parse_limit_train_batches(value):
    """Preserve Lightning's int-count versus float-fraction semantics."""
    parsed = float(value)
    if not parsed > 0:
        raise argparse.ArgumentTypeError("limit_train_batches must be positive")
    if parsed >= 1.0:
        if not parsed.is_integer():
            raise argparse.ArgumentTypeError("batch counts >= 1 must be whole numbers")
        return int(parsed)
    return parsed


def load_b1_calibration_contract(path, *, checkpoint=False):
    if checkpoint:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        calibration = payload.get("b1_uncertainty_calibration", {})
    else:
        calibration = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        not isinstance(calibration, dict)
        or calibration.get("schema") != "ct_seqtrack.b1_uncertainty_calibration.v2"
        or len(calibration.get("fixed_margin_parallel_perpendicular_95", [])) != 2
    ):
        raise RuntimeError("B1 calibration input is not a verified v2 artifact")
    return calibration


def validate_online_resume_checkpoint(checkpoint_path, config):
    """Reject cross-experiment and mid-epoch online resumes."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return validate_online_resume_contract(checkpoint, config)


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        default=argparse.SUPPRESS,
        help="input batch size (YAML value is used when omitted)",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=argparse.SUPPRESS,
        help="number of epochs (YAML value is used when omitted)",
    )
    parser.add_argument(
        "--limit_train_batches",
        type=parse_limit_train_batches,
        default=argparse.SUPPRESS,
        help="limit training batches (used by bounded loss preflight runs)",
    )
    parser.add_argument(
        "--limit_val_batches",
        type=parse_limit_train_batches,
        default=argparse.SUPPRESS,
        help="limit validation batches (used by end-to-end preflight runs)",
    )
    parser.add_argument(
        "--save_top_k",
        type=int,
        default=argparse.SUPPRESS,
        help="save top k checkpoints",
    )
    parser.add_argument(
        "--check_val_every_n_epoch",
        type=int,
        default=argparse.SUPPRESS,
        help="validation interval",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=argparse.SUPPRESS,
        help="number of data loading workers",
    )
    parser.add_argument("--cfg", type=str, help="the config_file")
    parser.add_argument(
        "--path",
        type=str,
        default=argparse.SUPPRESS,
        help="override the dataset root from the YAML config",
    )
    parser.add_argument(
        "--test_split",
        type=str,
        default=argparse.SUPPRESS,
        help="override the evaluation split (for train-tracklet calibration)",
    )
    parser.add_argument(
        "--ct-eval-partition",
        dest="ct_eval_partition",
        choices=(
            "train",
            "dev",
            "calibration",
            "calibration_select",
            "calibration_audit",
        ),
        default=argparse.SUPPRESS,
        help="evaluate only one atomic CT tracklet partition",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="checkpoint location"
    )
    parser.add_argument(
        "--b2_method_promotion",
        type=str,
        default=None,
        help="passed v2 method manifest required to start scratch Full",
    )
    parser.add_argument(
        "--acquisition_preflight",
        type=str,
        default=None,
        help="passed checkpoint-free causal preflight v3 required for B2 training",
    )
    parser.add_argument(
        "--ct_action_calibration_path",
        type=str,
        default=argparse.SUPPRESS,
        help="passed action-calibration artifact for selective evaluation",
    )
    parser.add_argument(
        "--ct_calibration_tracklet_manifest_sha256",
        type=str,
        default=argparse.SUPPRESS,
        help="SHA256 identity of the held-out calibration tracklet manifest",
    )
    parser.add_argument(
        "--ct_action_threshold_selection_path",
        type=str,
        default=argparse.SUPPRESS,
        help="v25 provisional thresholds, usable only on calibration_audit",
    )
    parser.add_argument(
        "--ct_calibration_select_scene_manifest_sha256",
        type=str,
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--ct_calibration_audit_scene_manifest_sha256",
        type=str,
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--log_dir", type=str, default=None, help="log location")
    parser.add_argument("--test", action="store_true", default=False, help="test mode")
    parser.add_argument(
        "--preloading",
        action="store_true",
        default=False,
        help="preload dataset into memory",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help="an extra tag appended on output folder name",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=argparse.SUPPRESS,
        help="random_seed (defaults to YAML seed, then 42)",
    )
    reseed_group = parser.add_mutually_exclusive_group()
    reseed_group.add_argument(
        "--ct-reseed-enabled",
        dest="ct_recursive_reseed_enabled",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Use the B0-2x2-selected periodic recursive reseed regime.",
    )
    reseed_group.add_argument(
        "--ct-no-reseed",
        dest="ct_recursive_reseed_enabled",
        action="store_false",
        default=argparse.SUPPRESS,
        help="Use continuous recursive rollout with horizon diagnostics only.",
    )
    parser.add_argument(
        "--dynamics_time_mode",
        choices=("true", "fixed", "shuffled"),
        default=argparse.SUPPRESS,
        help="P0-C dynamics-only physical-time control.",
    )
    parser.add_argument(
        "--dynamics_time_manifest",
        default=argparse.SUPPRESS,
        help="Offline split permutation manifest required by shuffled mode.",
    )
    parser.add_argument(
        "--dynamics_fixed_delta_t",
        type=float,
        default=argparse.SUPPRESS,
        help="Constant adjacent-observation step used by fixed mode.",
    )
    parser.add_argument(
        "--test_virtual_rate_mode",
        choices=(
            "none",
            "manifest",
            "gap_pattern",
            "periodic_drop",
            "burst_drop",
            "random_drop",
            "stride",
        ),
        default=argparse.SUPPRESS,
        help="Override the test cadence protocol (for example KITTI-HTV).",
    )
    parser.add_argument(
        "--test_virtual_rate_stride",
        type=int,
        default=argparse.SUPPRESS,
        help="Frame interval used by test_virtual_rate_mode=stride.",
    )
    parser.add_argument(
        "--test_virtual_rate_manifest",
        default=argparse.SUPPRESS,
        help="Frozen test endpoint-selection manifest.",
    )
    parser.add_argument(
        "--test_virtual_rate_drop_prob",
        type=float,
        default=argparse.SUPPRESS,
        help="Random-drop probability for test_virtual_rate_mode=random_drop.",
    )
    parser.add_argument(
        "--test_virtual_rate_seed",
        type=int,
        default=argparse.SUPPRESS,
        help="Random-drop seed for the test protocol.",
    )
    parser.add_argument(
        "--test_virtual_rate_max_gap",
        type=int,
        default=argparse.SUPPRESS,
        help="Maximum retained-frame gap for random-drop evaluation.",
    )
    parser.add_argument(
        "--kitti_hv_interval",
        default=argparse.SUPPRESS,
        help="Official KITTI-HV interval (1/2/3/5/10 or 'all').",
    )
    parser.add_argument(
        "--train_kitti_hv_interval",
        default=argparse.SUPPRESS,
        help="Role-specific KITTI-HV training interval.",
    )
    parser.add_argument(
        "--val_kitti_hv_interval",
        default=argparse.SUPPRESS,
        help="Role-specific KITTI-HV validation interval.",
    )
    parser.add_argument(
        "--test_kitti_hv_interval",
        default=argparse.SUPPRESS,
        help="Role-specific KITTI-HV test interval.",
    )
    parser.add_argument(
        "--b1_calibration_artifact_path",
        default=argparse.SUPPRESS,
        help="Verified v2 calibration JSON that binds fixed B1 residual margins.",
    )
    parser.add_argument(
        "--search_v3_use_dynamic_sigma",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Use promoted calibrated B1 sigma for the B2 support tube.",
    )
    parser.add_argument(
        "--require_b1_calibration_passed",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Reject checkpoints whose B1 calibration did not pass promotion.",
    )
    parser.add_argument(
        "--force_b1_invalid",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Evaluation control: invalidate B1 without changing observation.",
    )
    parser.add_argument(
        "--shuffle_b1_signal",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Evaluation control: mismatch B1 box order without changing B0.",
    )
    parser.add_argument(
        "--proposal-mode",
        "--proposal_mode",
        dest="proposal_inference_mode",
        choices=(
            "obs",
            "obs_motion",
            "obs_search",
            "full",
            "obs_motion_search",
            "full_selective",
            "obs_only",
            "obs_vs_motion",
            "obs_vs_refined",
            "obs_vs_all",
            "observation",
            "motion",
            "raw_search",
            "legacy_clipped",
            "selective",
        ),
        default=argparse.SUPPRESS,
        help="Evaluation-only B2 proposal attribution mode.",
    )

    args = parser.parse_args()
    if hasattr(args, "proposal_inference_mode") and not args.test:
        raise ValueError("--proposal_mode is evaluation-only")
    config = load_yaml(args.cfg)
    config.update(vars(args))  # override the configuration using the value in args
    if config.get("require_b1_calibration_artifact", False):
        calibration_path = config.get("b1_calibration_artifact_path")
        calibration = None
        if calibration_path:
            calibration = load_b1_calibration_contract(calibration_path)
        elif not config.get("test", False):
            checkpoint_path = config.get("checkpoint")
            if checkpoint_path:
                calibration = load_b1_calibration_contract(
                    checkpoint_path, checkpoint=True
                )
        if calibration is None and not config.get("test", False):
            raise RuntimeError(
                "formal fixed-margin training requires a B1 calibration "
                "artifact or a calibrated same-run resume checkpoint"
            )
        if calibration is not None:
            if (
                int(config.get("ct_joint_contract_version", 1)) >= 3
                and len(
                    calibration.get(
                        "standardized_abs_residual_q90_parallel_perpendicular", []
                    )
                )
                != 2
            ):
                raise RuntimeError(
                    "contract-v3 calibration lacks standardized residual q90"
                )
            source = calibration.get("source_artifact", {})
            if (
                source.get("partition") != "calibration"
                or source.get("dataset") != config.get("dataset")
                or source.get("split") != config.get("train_split")
                or source.get("b1_config_sha256")
                != b1_calibration_config_sha256(config)
            ):
                raise RuntimeError(
                    "B1 calibration partition/dataset does not match runtime"
                )
            margins = calibration["fixed_margin_parallel_perpendicular_95"]
            config["search_v3_fixed_margin_parallel"] = float(margins[0])
            config["search_v3_fixed_margin_perpendicular"] = float(margins[1])
            standardized_q90 = calibration.get(
                "standardized_abs_residual_q90_parallel_perpendicular"
            )
            if (
                isinstance(standardized_q90, (list, tuple))
                and len(standardized_q90) == 2
            ):
                config["search_v3_standardized_residual_q90_parallel_perpendicular"] = [
                    float(value) for value in standardized_q90
                ]
    defaults = {
        "batch_size": 100,
        "epoch": 60,
        "save_top_k": 5,
        "check_val_every_n_epoch": 1,
        "workers": 10,
    }
    for key, value in defaults.items():
        config.setdefault(key, value)
    if config.get("seed") is None:
        config["seed"] = 42
    return EasyDict(config)


cfg = parse_config()
if str(getattr(cfg, "net_model", "")).strip().lower() == "ctseqtrack":
    configure_ct_variant(cfg)
validate_scratch_training_contract(cfg)
if cfg.test and cfg.checkpoint is not None:
    try:
        source_checkpoint = torch.load(
            cfg.checkpoint, map_location="cpu", weights_only=False
        )
    except TypeError:
        source_checkpoint = torch.load(cfg.checkpoint, map_location="cpu")
    if source_checkpoint.get("epoch") is not None:
        cfg.ct_source_checkpoint_epoch = int(source_checkpoint["epoch"]) + 1
if (
    not cfg.test
    and int(getattr(cfg, "ct_joint_contract_version", 1)) >= 3
    and bool(getattr(cfg, "ct_enable_b2", False))
):
    if cfg.checkpoint is not None:
        try:
            preflight_resume = torch.load(
                cfg.checkpoint, map_location="cpu", weights_only=False
            )
        except TypeError:
            preflight_resume = torch.load(cfg.checkpoint, map_location="cpu")
        preflight = preflight_resume.get("ct_acquisition_preflight")
    else:
        if not cfg.acquisition_preflight:
            raise ValueError(
                "contract-v3 B2/Full requires --acquisition_preflight "
                "before training starts"
            )
        preflight = json.loads(
            Path(cfg.acquisition_preflight).read_text(encoding="utf-8")
        )
    cfg.ct_acquisition_preflight_manifest = validate_preflight_artifact(preflight, cfg)
    class_weights = cfg.ct_acquisition_preflight_manifest["targetness_class_weights"]
    cfg.ct_targetness_positive_weight = float(class_weights["positive"])
    cfg.ct_targetness_negative_weight = float(class_weights["negative"])
if (
    not cfg.test
    and bool(getattr(cfg, "ct_enable_b3", False))
    and str(getattr(cfg, "ct_initialization_policy", "legacy")) == "scratch_only"
):
    if cfg.checkpoint is not None:
        try:
            scratch_resume = torch.load(
                cfg.checkpoint, map_location="cpu", weights_only=False
            )
        except TypeError:
            scratch_resume = torch.load(cfg.checkpoint, map_location="cpu")
        method_promotion = scratch_resume.get("ct_b2_method_promotion")
    else:
        if not cfg.b2_method_promotion:
            raise ValueError(
                "scratch Full requires --b2_method_promotion; the manifest "
                "qualifies the B2 method but supplies no weights"
            )
        method_promotion = json.loads(
            Path(cfg.b2_method_promotion).read_text(encoding="utf-8")
        )
    cfg.ct_b2_method_promotion_manifest = validate_b2_method_promotion(
        method_promotion, cfg
    )
if bool(getattr(cfg, "ct_online_recursive_training", False)) and not cfg.test:
    if cfg.checkpoint is not None:
        validate_online_resume_checkpoint(cfg.checkpoint, cfg)
if cfg.seed is not None:
    seed_everything(cfg.seed)

env_cp = os.environ.copy()
project_root = os.path.dirname(os.path.abspath(__file__))
run_root_dir = generate_log_folder_name(cfg)

try:
    node_rank, local_rank, world_size = (
        env_cp["NODE_RANK"],
        env_cp["LOCAL_RANK"],
        env_cp["WORLD_SIZE"],
    )

    is_in_ddp_subprocess = env_cp["PL_IN_DDP_SUBPROCESS"]
    pl_trainer_gpus = env_cp["PL_TRAINER_GPUS"]
    print(node_rank, local_rank, world_size, is_in_ddp_subprocess, pl_trainer_gpus)

    if int(local_rank) == int(world_size) - 1:
        print(cfg)
except KeyError:
    pass


if not cfg.test:
    # dataset and dataloader
    train_data = get_dataset(
        cfg, type=cfg.train_type, split=cfg.train_split, protocol_role="train"
    )
    if bool(getattr(cfg, "ct_online_recursive_training", False)):
        # Keep mini_val untouched.  Joint checkpoint selection uses only the
        # atomic dev partition of mini_train.
        val_data = PartitionedTestTrackingSampler(
            train_data.dataset, config=cfg, partition="dev"
        )
    else:
        val_data = get_dataset(
            cfg, type="test", split=cfg.val_split, protocol_role="val"
        )
    loader_seed = int(cfg.seed or 42)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(loader_seed + 31001)

    def seed_loader_worker(worker_id):
        worker_seed = int(torch.initial_seed() % (2**32))
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    if bool(getattr(cfg, "ct_online_recursive_training", False)):
        if int(getattr(cfg, "ct_router_horizon", 3)) != 3:
            raise ValueError("online Joint Full currently requires H=3")
        tracklet_slots = int(getattr(cfg, "ct_recursive_tracklet_slots", 4))
        candidate_views = int(getattr(cfg, "ct_recursive_candidate_views", 1))
        expected_batch_size = (
            tracklet_slots
            if int(getattr(cfg, "ct_joint_contract_version", 1)) >= 3
            else tracklet_slots * candidate_views
        )
        if int(cfg.batch_size) != expected_batch_size:
            raise ValueError(
                "online recursive batch_size must equal the canonical B0 "
                f"slot count ({expected_batch_size})"
            )
        online_batch_sampler = OnlineRecursiveBatchSampler(
            train_data,
            slots=tracklet_slots,
            candidate_views=candidate_views,
            seed=loader_seed,
            partition_seed=int(getattr(cfg, "ct_partition_seed", 42)),
            partition=str(getattr(cfg, "ct_router_partition", "train")),
            partition_scheme=str(getattr(cfg, "ct_partition_scheme", "tracklet_v1")),
            shadow_interval=int(getattr(cfg, "ct_router_shadow_interval", 2)),
            shadow_slots_per_event=int(
                getattr(cfg, "ct_router_shadow_slots_per_event", 1)
            ),
            shadow_enabled=bool(getattr(cfg, "ct_enable_b3", True)),
        )
        train_data.partition_manifest = online_batch_sampler.partition_manifest
        train_loader = DataLoader(
            train_data,
            batch_sampler=online_batch_sampler,
            num_workers=cfg.workers,
            collate_fn=online_recursive_collate,
            pin_memory=False,
            worker_init_fn=seed_loader_worker,
            generator=loader_generator,
        )
    else:
        train_loader = DataLoader(
            train_data,
            batch_size=cfg.batch_size,
            num_workers=cfg.workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            worker_init_fn=seed_loader_worker,
            generator=loader_generator,
        )
    val_loader = DataLoader(
        val_data,
        batch_size=1,
        num_workers=cfg.workers,
        collate_fn=lambda x: x,
        pin_memory=True,
    )
    write_run_provenance(
        run_root_dir,
        cfg,
        {"train": train_data, "val": val_data},
        mode="train",
        root=project_root,
    )
    checkpoint_callback = ModelCheckpoint(
        monitor=str(getattr(cfg, "checkpoint_monitor", "precision/test")),
        mode=str(getattr(cfg, "checkpoint_mode", "max")),
        save_last=True,
        save_top_k=cfg.save_top_k,
    )
    learningrate_callback = LearningRateMonitor(logging_interval="step")

    # init trainer
    # RecursiveTrackState is intentionally process-local.  Until an explicit
    # cross-rank state coordinator exists, multi-device DDP would duplicate
    # tracklets and let the canonical histories silently diverge.
    trainer_devices = (
        1 if bool(getattr(cfg, "ct_online_recursive_training", False)) else -1
    )
    trainer = pl.Trainer(
        devices=trainer_devices,
        accelerator="auto",
        max_epochs=cfg.epoch,
        callbacks=[checkpoint_callback, learningrate_callback],
        default_root_dir=run_root_dir,
        check_val_every_n_epoch=cfg.check_val_every_n_epoch,
        num_sanity_val_steps=0,
        # Contract-v3 owns unscale/clip/step per optimizer;
        # Trainer-level clipping would couple both domains.
        gradient_clip_val=(
            0.0
            if bool(getattr(cfg, "ct_separate_optimizers", False))
            else cfg.gradient_clip_val
        ),
        limit_train_batches=getattr(cfg, "limit_train_batches", 1.0),
        limit_val_batches=getattr(cfg, "limit_val_batches", 1.0),
        fast_dev_run=False,
    )
    # init model
    train_dataloader_length = len(train_loader)  # 用于设置OneCycle学习率
    if cfg.checkpoint is None:
        net = get_model(cfg.net_model)(
            cfg, train_dataloader_length=train_dataloader_length
        )
    else:
        net = get_model(cfg.net_model).load_from_checkpoint(
            cfg.checkpoint, config=cfg, train_dataloader_length=train_dataloader_length
        )

    trainer.fit(net, train_loader, val_loader, ckpt_path=cfg.checkpoint)
else:
    source_test_data = get_dataset(
        cfg, type="test", split=cfg.test_split, protocol_role="test"
    )
    eval_partition = getattr(cfg, "ct_eval_partition", None)
    if eval_partition is not None:
        source_dataset = getattr(source_test_data, "dataset", None)
        if source_dataset is None:
            raise RuntimeError("--ct-eval-partition requires a tracklet test sampler")
        test_data = PartitionedTestTrackingSampler(
            source_dataset, config=cfg, partition=eval_partition
        )
    else:
        test_data = source_test_data
    test_loader = DataLoader(
        test_data,
        batch_size=1,
        num_workers=cfg.workers,
        collate_fn=lambda x: x,
        pin_memory=True,
    )
    write_run_provenance(
        run_root_dir, cfg, {"test": test_data}, mode="test", root=project_root
    )

    trainer = pl.Trainer(devices=-1, accelerator="auto", default_root_dir=run_root_dir)

    if cfg.checkpoint is None:
        net = get_model(cfg.net_model)(cfg)
    else:
        net = get_model(cfg.net_model).load_from_checkpoint(cfg.checkpoint, config=cfg)
    trainer.test(net, test_loader, ckpt_path=cfg.checkpoint)
