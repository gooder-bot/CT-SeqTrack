"""
main.py
Created by zenn at 2021/7/18 15:08
Modified by Aron Lin at Jun 1  09:42:22 CST 2023
"""
import pytorch_lightning as pl
import argparse

# import pytorch_lightning.utilities.distributed
import torch
import yaml
from easydict import EasyDict
import os
import json
from pathlib import Path

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything


from datasets import get_dataset
from models import get_model
from utils.run_provenance import write_run_provenance

torch.set_float32_matmul_precision("high")

import matplotlib.pyplot as plt
import sys

import datetime
import time

def generate_log_folder_name(cfg):
    if cfg.get('log_dir'):
        return cfg['log_dir']
    now = datetime.datetime.now()
    time_str = now.strftime("%Y%m%d-%H%M")
    cfg_name = cfg['cfg'].split("/")[-1].replace(".yaml", "")
    folder_name = f"output/{time_str}-{cfg_name}-{cfg['tag']}"
    return folder_name

def load_yaml(file_name):
    with open(file_name, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            config = yaml.load(f)
    return config


def load_initial_weights(model, checkpoint_path, report_path=None):
    """Load matching model tensors without restoring optimizer/trainer state."""
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported init checkpoint payload: {type(payload)}")
    state_dict = payload.get("state_dict", payload.get("model", payload))
    if not isinstance(state_dict, dict):
        raise TypeError("Init checkpoint does not contain a state_dict mapping")

    target_state = model.state_dict()
    candidates = [("none", state_dict)]
    for prefix in ("model.", "module."):
        candidates.append((prefix, {
            key[len(prefix):] if key.startswith(prefix) else key: value
            for key, value in state_dict.items()
        }))
    selected_prefix, normalized = max(
        candidates,
        key=lambda item: sum(
            key in target_state and target_state[key].shape == value.shape
            for key, value in item[1].items()),
    )
    matched = {
        key: value for key, value in normalized.items()
        if key in target_state and target_state[key].shape == value.shape
    }
    critical_prefixes = ("seg_pointnet.", "mini_pointnet.", "motion_mlp.",
                         "feature_pointnet.", "Transformer.")
    missing_critical = [
        prefix for prefix in critical_prefixes
        if not any(key.startswith(prefix) for key in matched)
    ]
    if missing_critical:
        raise RuntimeError(
            "Init checkpoint is missing baseline model prefixes: "
            + ", ".join(missing_critical))
    target_state.update(matched)
    model.load_state_dict(target_state, strict=True)
    report = {
        "checkpoint": str(checkpoint_path),
        "selected_prefix_strip": selected_prefix,
        "source_tensor_count": len(normalized),
        "target_tensor_count": len(target_state),
        "matched_tensor_count": len(matched),
        "new_tensor_count": len(target_state) - len(matched),
    }
    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
    print("initialization checkpoint:", checkpoint_path)
    print("matched tensors:", f"{len(matched)}/{len(target_state)}")
    return report


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=100, help='input batch size')
    parser.add_argument('--epoch', type=int, default=60, help='number of epochs')
    parser.add_argument('--save_top_k', type=int, default=5, help='save top k checkpoints')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1, help='check_val_every_n_epoch')
    parser.add_argument('--workers', type=int, default=10, help='number of data loading workers')
    parser.add_argument('--cfg', type=str, help='the config_file')
    parser.add_argument(
        '--path', type=str, default=argparse.SUPPRESS,
        help='override the dataset root from the YAML config')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint location')
    parser.add_argument(
        '--init_checkpoint', type=str, default=None,
        help='model-only initialization; does not restore optimizer/trainer state')
    parser.add_argument('--log_dir', type=str, default=None, help='log location')
    parser.add_argument('--test', action='store_true', default=False, help='test mode')
    parser.add_argument('--preloading', action='store_true', default=False, help='preload dataset into memory')
    parser.add_argument('--tag', type=str, default="", help='an extra tag appended on output folder name')
    parser.add_argument('--seed', type=int, help='random_seed')
    parser.add_argument(
        '--dynamics_time_mode', choices=('true', 'fixed', 'shuffled'),
        default=argparse.SUPPRESS,
        help='P0-C dynamics-only physical-time control.')
    parser.add_argument(
        '--dynamics_time_manifest', default=argparse.SUPPRESS,
        help='Offline split permutation manifest required by shuffled mode.')
    parser.add_argument(
        '--dynamics_fixed_delta_t', type=float, default=argparse.SUPPRESS,
        help='Constant adjacent-observation step used by fixed mode.')
    parser.add_argument(
        '--test_virtual_rate_mode',
        choices=(
            'none', 'manifest', 'gap_pattern', 'periodic_drop',
            'burst_drop', 'random_drop', 'stride'),
        default=argparse.SUPPRESS,
        help='Override the test cadence protocol (for example KITTI-HTV).')
    parser.add_argument(
        '--test_virtual_rate_stride',
        type=int,
        default=argparse.SUPPRESS,
        help='Frame interval used by test_virtual_rate_mode=stride.')
    parser.add_argument(
        '--test_virtual_rate_manifest',
        default=argparse.SUPPRESS,
        help='Frozen test endpoint-selection manifest.')
    parser.add_argument(
        '--kitti_hv_interval',
        default=argparse.SUPPRESS,
        help="Official KITTI-HV interval (1/2/3/5/10 or 'all').")
    parser.add_argument(
        '--train_kitti_hv_interval',
        default=argparse.SUPPRESS,
        help="Role-specific KITTI-HV training interval.")
    parser.add_argument(
        '--val_kitti_hv_interval',
        default=argparse.SUPPRESS,
        help="Role-specific KITTI-HV validation interval.")
    parser.add_argument(
        '--test_kitti_hv_interval',
        default=argparse.SUPPRESS,
        help="Role-specific KITTI-HV test interval.")
    parser.add_argument(
        '--m4_variant',
        choices=('off', 'filter', 'tube', 'filter_tube'),
        default=argparse.SUPPRESS,
        help='Evaluation-only M4 ablation selector.')
    parser.add_argument(
        '--m4_time_mode',
        choices=('fixed', 'real'),
        default=argparse.SUPPRESS,
        help='M4 state-transition clock control.')
    parser.add_argument(
        '--m4_fixed_delta_t', type=float, default=argparse.SUPPRESS,
        help='Fixed state-transition step for M4.')
    parser.add_argument(
        '--m3_path_weight', type=float, default=argparse.SUPPRESS,
        help='M3 endpoint path-distillation weight.')
    parser.add_argument(
        '--m3_variant',
        choices=('off', 'distill'),
        default=argparse.SUPPRESS,
        help='M3 single-view or asymmetric-distillation selector.')
    parser.add_argument(
        '--m3_irregular_supervision_weight',
        type=float,
        default=argparse.SUPPRESS,
        help='Optional supervised-loss weight for the irregular M3 view.')

    args = parser.parse_args()
    config = load_yaml(args.cfg)
    config.update(vars(args))  # override the configuration using the value in args
    m3_variant = config.get('m3_variant')
    if m3_variant is not None:
        if m3_variant not in ('off', 'distill'):
            raise ValueError("m3_variant must be off or distill")
        config['use_m3_path_distillation'] = m3_variant == 'distill'
    m4_variant = config.get('m4_variant')
    if m4_variant is not None:
        if m4_variant not in ('off', 'filter', 'tube', 'filter_tube'):
            raise ValueError(
                "m4_variant must be off, filter, tube, or filter_tube")
        config['use_m4_state_filter'] = m4_variant in (
            'filter', 'filter_tube')
        config['use_m4_trajectory_tube'] = m4_variant in (
            'tube', 'filter_tube')

    return EasyDict(config)


cfg = parse_config()
if cfg.checkpoint is not None and cfg.init_checkpoint is not None:
    raise ValueError("--checkpoint (resume/test) and --init_checkpoint are mutually exclusive")
if cfg.test and cfg.init_checkpoint is not None:
    raise ValueError("--init_checkpoint is training-only; use --checkpoint for evaluation")
if cfg.seed is not None:
    seed_everything(cfg.seed)
    
env_cp = os.environ.copy()
project_root = os.path.dirname(os.path.abspath(__file__))
run_root_dir = generate_log_folder_name(cfg)

try:
    node_rank, local_rank, world_size = env_cp['NODE_RANK'], env_cp['LOCAL_RANK'], env_cp['WORLD_SIZE']

    is_in_ddp_subprocess = env_cp['PL_IN_DDP_SUBPROCESS']
    pl_trainer_gpus = env_cp['PL_TRAINER_GPUS']
    print(node_rank, local_rank, world_size, is_in_ddp_subprocess, pl_trainer_gpus)

    if int(local_rank) == int(world_size) - 1:
        print(cfg)
except KeyError:
    pass


if not cfg.test:
    # dataset and dataloader
    train_data = get_dataset(
        cfg, type=cfg.train_type, split=cfg.train_split, protocol_role='train')
    val_data = get_dataset(
        cfg, type='test', split=cfg.val_split, protocol_role='val')
    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, num_workers=cfg.workers, shuffle=True,drop_last=True,
                              pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=1, num_workers=cfg.workers, collate_fn=lambda x: x, pin_memory=True)
    write_run_provenance(
        run_root_dir, cfg, {"train": train_data, "val": val_data},
        mode="train", root=project_root)
    checkpoint_callback = ModelCheckpoint(monitor='precision/test', mode='max', save_last=True,
                                          save_top_k=cfg.save_top_k)
    learningrate_callback = LearningRateMonitor(logging_interval="step")

    # init trainer
    trainer = pl.Trainer(devices=-1, accelerator='auto', max_epochs=cfg.epoch,
                         callbacks=[checkpoint_callback,learningrate_callback],
                         default_root_dir=run_root_dir,
                         check_val_every_n_epoch=cfg.check_val_every_n_epoch,
                         num_sanity_val_steps=0,
                         gradient_clip_val=cfg.gradient_clip_val,
                         fast_dev_run=False)
    # init model
    train_dataloader_length = len(train_loader) #用于设置OneCycle学习率
    if cfg.checkpoint is None:
        net = get_model(cfg.net_model)(cfg,train_dataloader_length=train_dataloader_length)
        if cfg.init_checkpoint is not None:
            load_initial_weights(
                net,
                cfg.init_checkpoint,
                report_path=Path(run_root_dir) / "init_checkpoint_report.json",
            )
    else:
        net = get_model(cfg.net_model).load_from_checkpoint(cfg.checkpoint, config=cfg,train_dataloader_length=train_dataloader_length)

    trainer.fit(net, train_loader, val_loader, ckpt_path=cfg.checkpoint)
else:
    test_data = get_dataset(
        cfg, type='test', split=cfg.test_split, protocol_role='test')
    test_loader = DataLoader(test_data, batch_size=1, num_workers=cfg.workers, collate_fn=lambda x: x, pin_memory=True)
    write_run_provenance(
        run_root_dir, cfg, {"test": test_data}, mode="test", root=project_root)

    trainer = pl.Trainer(devices=-1, accelerator='auto', default_root_dir=run_root_dir)

    if cfg.checkpoint is None:
        net = get_model(cfg.net_model)(cfg)
    else:
        net = get_model(cfg.net_model).load_from_checkpoint(cfg.checkpoint, config=cfg)
    trainer.test(net, test_loader, ckpt_path=cfg.checkpoint)
