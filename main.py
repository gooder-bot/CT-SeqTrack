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
import hashlib
import subprocess
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
from models.ct_variant import configure_ct_variant
from utils.run_provenance import write_run_provenance
from utils.acquisition_metrics import validate_preflight_artifact
from utils.online_contract import (
    require_scratch_initialization,
    validate_scratch_training_contract,
    validate_b2_method_promotion,
    validate_online_resume_contract,
)
from utils.replay_cache import (
    B0_STATE_PREFIXES,
    B1_STATE_PREFIXES,
    b1_calibration_config_sha256,
    replay_config_sha256,
    sha256_json,
    tensor_prefixes_sha256,
    validate_b1_calibration_state,
    validate_replay_cache_manifest,
)

if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

import sys

import datetime
import time
from utils.config import load_yaml_config

def generate_log_folder_name(cfg):
    if cfg.get('log_dir'):
        return cfg['log_dir']
    now = datetime.datetime.now()
    time_str = now.strftime("%Y%m%d-%H%M")
    cfg_name = cfg['cfg'].split("/")[-1].replace(".yaml", "")
    folder_name = f"output/{time_str}-{cfg_name}-{cfg['tag']}"
    return folder_name

def load_yaml(file_name):
    return load_yaml_config(file_name)


def parse_limit_train_batches(value):
    """Preserve Lightning's int-count versus float-fraction semantics."""
    parsed = float(value)
    if not parsed > 0:
        raise argparse.ArgumentTypeError(
            "limit_train_batches must be positive")
    if parsed >= 1.0:
        if not parsed.is_integer():
            raise argparse.ArgumentTypeError(
                "batch counts >= 1 must be whole numbers")
        return int(parsed)
    return parsed


def tensor_prefix_hash(state_dict, prefix):
    """Hash names, dtypes, shapes, and bytes for one checkpoint prefix."""
    digest = hashlib.sha256()
    keys = sorted(key for key in state_dict if key.startswith(prefix))
    for key in keys:
        value = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest(), keys


def current_git_commit():
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
        check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def load_b1_calibration_contract(path, *, checkpoint=False):
    if checkpoint:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        calibration = payload.get("b1_uncertainty_calibration", {})
    else:
        calibration = json.loads(Path(path).read_text(encoding="utf-8"))
    if (not isinstance(calibration, dict)
            or calibration.get("schema")
            != "ct_seqtrack.b1_uncertainty_calibration.v2"
            or len(calibration.get(
                "fixed_margin_parallel_perpendicular_95", [])) != 2):
        raise RuntimeError(
            "B1 calibration input is not a verified v2 artifact")
    return calibration


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

    def compatible_state_value(target, source):
        if torch.is_tensor(target) or torch.is_tensor(source):
            return (torch.is_tensor(target) and torch.is_tensor(source)
                    and target.shape == source.shape)
        return type(target) is type(source)

    candidates = [("none", state_dict)]
    for prefix in ("model.", "module."):
        candidates.append((prefix, {
            key[len(prefix):] if key.startswith(prefix) else key: value
            for key, value in state_dict.items()
        }))
    selected_prefix, normalized = max(
        candidates,
        key=lambda item: sum(
            key in target_state
            and compatible_state_value(target_state[key], value)
            for key, value in item[1].items()),
    )
    matched = {
        key: value for key, value in normalized.items()
        if key in target_state
        and compatible_state_value(target_state[key], value)
    }
    strict_v3 = bool(getattr(
        model, "use_motion_conditioned_search_v3", False))
    shape_mismatch = sorted(
        key for key, value in normalized.items()
        if key in target_state
        and not compatible_state_value(target_state[key], value))
    if strict_v3 and shape_mismatch:
        raise RuntimeError(
            "Init checkpoint contains target keys with wrong shapes: "
            + ", ".join(shape_mismatch[:20]))
    if strict_v3:
        nonfinite = sorted(
            key for key, value in matched.items()
            if torch.is_tensor(value)
            and (value.is_floating_point() or value.is_complex())
            and not bool(torch.isfinite(value).all().item()))
        if nonfinite:
            raise RuntimeError(
                "B2-v3 init contains non-finite tensors: "
                + ", ".join(nonfinite[:20]))
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
    v3_report = None
    if strict_v3:
        metadata = payload.get("b2_v3_init")
        if not isinstance(metadata, dict):
            raise RuntimeError(
                "B2-v3 requires a checkpoint composed by the strict v3 "
                "builder (missing b2_v3_init metadata)")
        model._b2_v3_init_provenance = dict(metadata)
        if bool(getattr(model, 'use_asymmetric_dual_query', False)):
            if (metadata.get('schema') != 'ct_seqtrack.b2_v3_init.v2'
                    or not bool(metadata.get('dual_query_migration'))
                    or metadata.get('dual_query_projection_init')
                    != 'legacy_observation_group_mean_4'):
                raise RuntimeError(
                    "asymmetric dual-query training requires the v2 "
                    "non-zero final-decoder query migration")
        b1_prefix = "physical_motion_encoder."
        b1_target = sorted(
            key for key in target_state if key.startswith(b1_prefix))
        b1_matched = sorted(
            key for key in matched if key.startswith(b1_prefix))
        if not b1_target or b1_matched != b1_target:
            missing = sorted(set(b1_target) - set(b1_matched))
            raise RuntimeError(
                "B2-v3 refuses to freeze an incomplete B1 checkpoint: "
                + ", ".join(missing))
        migrated_submodules = (
            "point_mlp.", "source_embedding.", "query_projection.",
            "key_projection.", "key_norm.",
            "query_value_projection.", "query_norm.",
            "local_targetness_head.", "vote_head.",
        )
        refiner_prefix = "state_aligned_search_refiner."
        required_migrated = sorted(
            key for key in target_state
            if any(key.startswith(refiner_prefix + submodule)
                   for submodule in migrated_submodules))
        missing_migrated = sorted(set(required_migrated) - set(matched))
        if missing_migrated:
            raise RuntimeError(
                "B2-v3 search migration is incomplete: "
                + ", ".join(missing_migrated))
        parameter_keys = dict(model.named_parameters())
        allowed_cold_prefix = "action_consistent_router_v3."
        missing_frozen = sorted(
            key for key, parameter in parameter_keys.items()
            if not parameter.requires_grad
            and not key.startswith(allowed_cold_prefix)
            and key not in matched)
        if missing_frozen:
            raise RuntimeError(
                "B2-v3 frozen parameters were not loaded: "
                + ", ".join(missing_frozen[:30]))
        b1_hash, _ = tensor_prefix_hash(matched, b1_prefix)
        expected_b1_hash = metadata.get("b1_prefix_hash")
        if not expected_b1_hash or b1_hash != expected_b1_hash:
            raise RuntimeError(
                "B2-v3 B1 prefix hash does not match builder provenance")
        expected_migrated = sorted(metadata.get("migrated_target_keys", []))
        if expected_migrated != required_migrated:
            raise RuntimeError(
                "B2-v3 migrated-key manifest does not match this model")
        v3_report = {
            "b1_tensor_count": len(b1_target),
            "b1_prefix_hash": b1_hash,
            "migrated_tensor_count": len(required_migrated),
            "missing_frozen_tensor_count": len(missing_frozen),
        }
    calibration = payload.get('b1_uncertainty_calibration')
    if isinstance(calibration, dict):
        validate_b1_calibration_state(calibration, normalized)
    if bool(getattr(model, 'require_b1_calibration_artifact', False)):
        if (not isinstance(calibration, dict)
                or calibration.get('schema')
                != 'ct_seqtrack.b1_uncertainty_calibration.v2'
                or len(calibration.get(
                    'fixed_margin_parallel_perpendicular_95', [])) != 2):
            raise RuntimeError(
                "initialization checkpoint lacks a verified v2 B1 "
                "calibration artifact with fixed residual margins")
        if (int(getattr(
                model, 'ct_joint_contract_version', 1)) >= 3
                and len(calibration.get(
                    'standardized_abs_residual_q90_parallel_perpendicular',
                    [])) != 2):
            raise RuntimeError(
                "contract-v3 calibration lacks standardized residual q90")
        source = calibration.get('source_artifact', {})
        if (source.get('partition') != 'calibration'
                or source.get('dataset') != str(getattr(
                    model.config, 'dataset', 'unknown'))
                or source.get('split') != str(getattr(
                    model.config, 'train_split', 'train'))
                or source.get('b1_config_sha256')
                != b1_calibration_config_sha256(model.config)):
            raise RuntimeError(
                "initialization B1 calibration provenance mismatch")
    if bool(getattr(model, 'require_b1_calibration_passed', False)):
        if (not isinstance(calibration, dict)
                or not bool(calibration.get(
                    'promotion', {}).get('passed'))):
            raise RuntimeError(
                "initialization checkpoint lacks a promoted B1 calibration")
    if isinstance(calibration, dict):
        model._b1_uncertainty_calibration = calibration
        margins = calibration.get(
            'fixed_margin_parallel_perpendicular_95')
        if isinstance(margins, (list, tuple)) and len(margins) == 2:
            model.config.search_v3_fixed_margin_parallel = float(margins[0])
            model.config.search_v3_fixed_margin_perpendicular = float(
                margins[1])
        standardized_q90 = calibration.get(
            'standardized_abs_residual_q90_parallel_perpendicular')
        if (isinstance(standardized_q90, (list, tuple))
                and len(standardized_q90) == 2):
            model.config[
                'search_v3_standardized_residual_q90_parallel_perpendicular'
            ] = [float(value) for value in standardized_q90]
    if (int(getattr(model, 'ct_joint_contract_version', 1)) >= 3
            and bool(getattr(model, 'ct_enable_b3', False))):
        promotion = payload.get('ct_b2_promotion')
        if (not isinstance(promotion, dict)
                or promotion.get('schema')
                != 'ct_seqtrack.b2_evidence_promotion.v3'
                or not bool(promotion.get('passed'))):
            raise RuntimeError(
                "contract-v3 B3 initialization requires a promoted B2 "
                "checkpoint")
        model._ct_b2_promotion = dict(promotion)
    target_state.update(matched)
    model.load_state_dict(target_state, strict=True)
    if bool(getattr(model, 'use_recursive_replay_cache', False)):
        cache_dir = getattr(
            model.config, 'recursive_replay_cache_dir', None)
        if not cache_dir:
            raise RuntimeError(
                "formal replay initialization requires a cache directory")
        expected_replay = {
            'dataset': str(getattr(
                model.config, 'dataset', 'unknown')),
            'split': str(getattr(
                model.config, 'train_split', 'train')),
            'replay_config_sha256': replay_config_sha256(model.config),
            'commit': current_git_commit(),
            'b0_state_sha256': tensor_prefixes_sha256(
                model.state_dict(), B0_STATE_PREFIXES),
            'b1_state_sha256': tensor_prefixes_sha256(
                model.state_dict(), B1_STATE_PREFIXES),
            'b1_calibration_sha256': sha256_json(calibration),
        }
        validate_replay_cache_manifest(
            cache_dir, expected_manifest=expected_replay)
    if strict_v3:
        frozen_prefixes = tuple(model.B2_V3_FROZEN_PREFIXES)
        model._b2_v3_frozen_reference_hashes = {
            prefix: tensor_prefix_hash(model.state_dict(), prefix)[0]
            for prefix in frozen_prefixes
        }
    report = {
        "checkpoint": str(checkpoint_path),
        "selected_prefix_strip": selected_prefix,
        "source_tensor_count": len(normalized),
        "target_tensor_count": len(target_state),
        "matched_tensor_count": len(matched),
        "new_tensor_count": len(target_state) - len(matched),
    }
    if v3_report is not None:
        report["b2_v3"] = v3_report
    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
    print("initialization checkpoint:", checkpoint_path)
    print("matched tensors:", f"{len(matched)}/{len(target_state)}")
    return report


def validate_online_resume_checkpoint(checkpoint_path, config):
    """Reject cross-experiment and mid-epoch online resumes."""
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return validate_online_resume_contract(checkpoint, config)


def validate_candidate0_b0_initialization(checkpoint_path):
    """Reject CT21/CT22 Full/B2-only weights for contract-v3 training."""
    try:
        payload = torch.load(
            checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location='cpu')
    hyper_parameters = payload.get('hyper_parameters', {})
    saved = hyper_parameters.get('config', hyper_parameters)

    def value(key, default=None):
        if isinstance(saved, dict):
            return saved.get(key, default)
        return getattr(saved, key, default)

    requirements = {
        'num_candidates': int(value('num_candidates', -1)) == 1,
        'candidate_views': int(value(
            'ct_recursive_candidate_views', -1)) == 1,
        'tracklet_slots': int(value(
            'ct_recursive_tracklet_slots', -1)) == 16,
        'no_reseed': not bool(value(
            'ct_recursive_reseed_enabled', True)),
        'b1_disabled': not bool(value('ct_enable_b1', True)),
        'b2_disabled': not bool(value('ct_enable_b2', True)),
        'b3_disabled': not bool(value('ct_enable_b3', True)),
        'joint_full_disabled': not bool(value('use_ct_joint_full', True)),
        'rng_shift_disabled': not bool(value(
            'ct_b0_rng_shift_control', False)),
    }
    lineage = value('ct_candidate0_b0_source')
    lineage_requirements = (
        lineage.get('requirements', {})
        if isinstance(lineage, dict) else {})
    lineage_valid = bool(
        lineage_requirements
        and all(bool(item) for item in lineage_requirements.values()))
    failed = sorted(name for name, passed in requirements.items()
                    if not passed)
    if failed and not lineage_valid:
        raise RuntimeError(
            "contract-v3 initialization is not a canonical candidate0-only "
            "no-reseed B0 checkpoint: " + ", ".join(failed))
    if payload.get('b2_v3_init') is not None:
        raise RuntimeError(
            "contract-v3 refuses an old B2-v3 composed checkpoint")
    return {
        'experiment_name': (
            str(lineage.get('experiment_name')) if lineage_valid
            else str(value('experiment_name', 'unknown'))),
        'seed': (
            int(lineage.get('seed', 42)) if lineage_valid
            else int(value('seed', 42) or 42)),
        'requirements': (
            dict(lineage_requirements) if lineage_valid else requirements),
    }


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size', type=int, default=argparse.SUPPRESS,
        help='input batch size (YAML value is used when omitted)')
    parser.add_argument(
        '--epoch', type=int, default=argparse.SUPPRESS,
        help='number of epochs (YAML value is used when omitted)')
    parser.add_argument(
        '--limit_train_batches', type=parse_limit_train_batches,
        default=argparse.SUPPRESS,
        help='limit training batches (used by bounded loss preflight runs)')
    parser.add_argument(
        '--limit_val_batches', type=parse_limit_train_batches,
        default=argparse.SUPPRESS,
        help='limit validation batches (used by end-to-end preflight runs)')
    parser.add_argument(
        '--save_top_k', type=int, default=argparse.SUPPRESS,
        help='save top k checkpoints')
    parser.add_argument(
        '--check_val_every_n_epoch', type=int, default=argparse.SUPPRESS,
        help='validation interval')
    parser.add_argument(
        '--workers', type=int, default=argparse.SUPPRESS,
        help='number of data loading workers')
    parser.add_argument('--cfg', type=str, help='the config_file')
    parser.add_argument(
        '--path', type=str, default=argparse.SUPPRESS,
        help='override the dataset root from the YAML config')
    parser.add_argument(
        '--test_split', type=str, default=argparse.SUPPRESS,
        help='override the evaluation split (for train-tracklet calibration)')
    parser.add_argument(
        '--ct-eval-partition', dest='ct_eval_partition',
        choices=('train', 'dev', 'calibration'),
        default=argparse.SUPPRESS,
        help='evaluate only one atomic CT tracklet partition')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint location')
    parser.add_argument(
        '--init_checkpoint', type=str, default=None,
        help='model-only initialization; does not restore optimizer/trainer state')
    parser.add_argument(
        '--b2_method_promotion', type=str, default=None,
        help='passed v2 method manifest required to start scratch Full')
    parser.add_argument(
        '--acquisition_preflight', type=str, default=None,
        help='passed checkpoint-free preflight v2 required for B2 training')
    parser.add_argument(
        '--ct_action_calibration_path', type=str,
        default=argparse.SUPPRESS,
        help='passed action-calibration artifact for selective evaluation')
    parser.add_argument(
        '--ct_calibration_tracklet_manifest_sha256', type=str,
        default=argparse.SUPPRESS,
        help='SHA256 identity of the held-out calibration tracklet manifest')
    parser.add_argument('--log_dir', type=str, default=None, help='log location')
    parser.add_argument('--test', action='store_true', default=False, help='test mode')
    parser.add_argument('--preloading', action='store_true', default=False, help='preload dataset into memory')
    parser.add_argument('--tag', type=str, default="", help='an extra tag appended on output folder name')
    parser.add_argument(
        '--seed', type=int, default=argparse.SUPPRESS,
        help='random_seed (defaults to YAML seed, then 42)')
    reseed_group = parser.add_mutually_exclusive_group()
    reseed_group.add_argument(
        '--ct-reseed-enabled', dest='ct_recursive_reseed_enabled',
        action='store_true', default=argparse.SUPPRESS,
        help='Use the B0-2x2-selected periodic recursive reseed regime.')
    reseed_group.add_argument(
        '--ct-no-reseed', dest='ct_recursive_reseed_enabled',
        action='store_false', default=argparse.SUPPRESS,
        help='Use continuous recursive rollout with horizon diagnostics only.')
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
        '--test_virtual_rate_drop_prob',
        type=float,
        default=argparse.SUPPRESS,
        help='Random-drop probability for test_virtual_rate_mode=random_drop.')
    parser.add_argument(
        '--test_virtual_rate_seed',
        type=int,
        default=argparse.SUPPRESS,
        help='Random-drop seed for the test protocol.')
    parser.add_argument(
        '--test_virtual_rate_max_gap',
        type=int,
        default=argparse.SUPPRESS,
        help='Maximum retained-frame gap for random-drop evaluation.')
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
        '--pftc_weight', type=float, default=argparse.SUPPRESS,
        help='PFTC loss lambda; use zero for the 200-batch loss preflight.')
    parser.add_argument(
        '--motion_v3_fusion_scale', type=float, default=argparse.SUPPRESS,
        help='B1motion-v3 runtime fusion scale; use zero for observation-only evaluation.')
    parser.add_argument(
        '--recursive_replay_cache_dir', default=argparse.SUPPRESS,
        help='Hash-validated frozen B0/B1 recursive replay cache.')
    parser.add_argument(
        '--use_recursive_replay_cache', action='store_true',
        default=argparse.SUPPRESS,
        help='Use the recursive replay cache for training histories.')
    parser.add_argument(
        '--b1_calibration_artifact_path', default=argparse.SUPPRESS,
        help='Verified v2 calibration JSON used to freeze B1 residual margins.')
    parser.add_argument(
        '--search_v3_use_dynamic_sigma', action='store_true',
        default=argparse.SUPPRESS,
        help='Use promoted calibrated B1 sigma for the B2 support tube.')
    parser.add_argument(
        '--require_b1_calibration_passed', action='store_true',
        default=argparse.SUPPRESS,
        help='Reject checkpoints whose B1 calibration did not pass promotion.')
    parser.add_argument(
        '--disable_uncertainty_geometry', dest='use_uncertainty_geometry',
        action='store_false', default=argparse.SUPPRESS,
        help='P4 ablation: retain B1 support but disable geometry features.')
    parser.add_argument(
        '--force_b1_invalid', action='store_true',
        default=argparse.SUPPRESS,
        help='Evaluation control: invalidate B1 without changing observation.')
    parser.add_argument(
        '--shuffle_b1_signal', action='store_true',
        default=argparse.SUPPRESS,
        help='Evaluation control: mismatch B1 box order without changing B0.')
    parser.add_argument(
        '--proposal-mode', '--proposal_mode',
        dest='proposal_inference_mode',
        choices=(
            'obs', 'obs_motion', 'obs_search', 'full',
            'obs_motion_search', 'full_selective',
            'obs_only', 'obs_vs_motion', 'obs_vs_refined', 'obs_vs_all',
            'observation', 'motion', 'raw_search', 'legacy_clipped',
            'selective'),
        default=argparse.SUPPRESS,
        help='Evaluation-only B2 proposal attribution mode.')

    args = parser.parse_args()
    if (hasattr(args, 'proposal_inference_mode') and not args.test):
        raise ValueError("--proposal_mode is evaluation-only")
    config = load_yaml(args.cfg)
    config.update(vars(args))  # override the configuration using the value in args
    if config.get('require_b1_calibration_artifact', False):
        calibration_path = config.get('b1_calibration_artifact_path')
        calibration = None
        if calibration_path:
            calibration = load_b1_calibration_contract(calibration_path)
        elif not config.get('test', False):
            checkpoint_path = (
                config.get('init_checkpoint') or config.get('checkpoint'))
            if checkpoint_path:
                calibration = load_b1_calibration_contract(
                    checkpoint_path, checkpoint=True)
        if calibration is None and not config.get('test', False):
            raise RuntimeError(
                "formal fixed-margin training requires a B1 calibration "
                "artifact or calibrated initialization checkpoint")
        if calibration is not None:
            if (int(config.get('ct_joint_contract_version', 1)) >= 3
                    and len(calibration.get(
                        'standardized_abs_residual_q90_parallel_perpendicular',
                        [])) != 2):
                raise RuntimeError(
                    "contract-v3 calibration lacks standardized residual q90")
            source = calibration.get('source_artifact', {})
            if (source.get('partition') != 'calibration'
                    or source.get('dataset') != config.get('dataset')
                    or source.get('split') != config.get('train_split')
                    or source.get('b1_config_sha256')
                    != b1_calibration_config_sha256(config)):
                raise RuntimeError(
                    "B1 calibration partition/dataset does not match runtime")
            margins = calibration[
                'fixed_margin_parallel_perpendicular_95']
            config['search_v3_fixed_margin_parallel'] = float(margins[0])
            config['search_v3_fixed_margin_perpendicular'] = float(margins[1])
            standardized_q90 = calibration.get(
                'standardized_abs_residual_q90_parallel_perpendicular')
            if (isinstance(standardized_q90, (list, tuple))
                    and len(standardized_q90) == 2):
                config[
                    'search_v3_standardized_residual_q90_parallel_perpendicular'
                ] = [float(value) for value in standardized_q90]
    defaults = {
        'batch_size': 100,
        'epoch': 60,
        'save_top_k': 5,
        'check_val_every_n_epoch': 1,
        'workers': 10,
    }
    for key, value in defaults.items():
        config.setdefault(key, value)
    if config.get('seed') is None:
        config['seed'] = 42
    return EasyDict(config)


cfg = parse_config()
if str(getattr(cfg, 'net_model', '')).strip().lower() == 'ctseqtrack':
    configure_ct_variant(cfg)
if cfg.checkpoint is not None and cfg.init_checkpoint is not None:
    raise ValueError("--checkpoint (resume/test) and --init_checkpoint are mutually exclusive")
if cfg.test and cfg.init_checkpoint is not None:
    raise ValueError("--init_checkpoint is training-only; use --checkpoint for evaluation")
require_scratch_initialization(cfg, cfg.init_checkpoint)
validate_scratch_training_contract(cfg)
if cfg.test and cfg.checkpoint is not None:
    try:
        source_checkpoint = torch.load(
            cfg.checkpoint, map_location='cpu', weights_only=False)
    except TypeError:
        source_checkpoint = torch.load(cfg.checkpoint, map_location='cpu')
    if source_checkpoint.get('epoch') is not None:
        cfg.ct_source_checkpoint_epoch = int(
            source_checkpoint['epoch']) + 1
if (not cfg.test
        and int(getattr(cfg, 'ct_joint_contract_version', 1)) >= 3
        and bool(getattr(cfg, 'ct_enable_b2', False))):
    if cfg.checkpoint is not None:
        try:
            preflight_resume = torch.load(
                cfg.checkpoint, map_location='cpu', weights_only=False)
        except TypeError:
            preflight_resume = torch.load(
                cfg.checkpoint, map_location='cpu')
        preflight = preflight_resume.get('ct_acquisition_preflight')
    else:
        if not cfg.acquisition_preflight:
            raise ValueError(
                'contract-v3 B2/Full requires --acquisition_preflight '
                'before training starts')
        preflight = json.loads(Path(
            cfg.acquisition_preflight).read_text(encoding='utf-8'))
    cfg.ct_acquisition_preflight_manifest = validate_preflight_artifact(
        preflight, cfg)
    class_weights = cfg.ct_acquisition_preflight_manifest[
        'targetness_class_weights']
    cfg.ct_targetness_positive_weight = float(class_weights['positive'])
    cfg.ct_targetness_negative_weight = float(class_weights['negative'])
if (not cfg.test and bool(getattr(cfg, 'ct_enable_b3', False))
        and str(getattr(cfg, 'ct_initialization_policy', 'legacy'))
        == 'scratch_only'):
    if cfg.checkpoint is not None:
        try:
            scratch_resume = torch.load(
                cfg.checkpoint, map_location='cpu', weights_only=False)
        except TypeError:
            scratch_resume = torch.load(cfg.checkpoint, map_location='cpu')
        method_promotion = scratch_resume.get('ct_b2_method_promotion')
    else:
        if not cfg.b2_method_promotion:
            raise ValueError(
                'scratch Full requires --b2_method_promotion; the manifest '
                'qualifies the B2 method but supplies no weights')
        method_promotion = json.loads(Path(
            cfg.b2_method_promotion).read_text(encoding='utf-8'))
    cfg.ct_b2_method_promotion_manifest = validate_b2_method_promotion(
        method_promotion, cfg)
if (bool(getattr(cfg, 'ct_online_recursive_training', False))
        and not cfg.test):
    if cfg.init_checkpoint is not None:
        if (int(getattr(cfg, 'ct_joint_contract_version', 1)) >= 3
                and str(getattr(
                    cfg, 'ct_initialization_policy', 'legacy'))
                != 'scratch_only'):
            cfg.ct_candidate0_b0_source = (
                validate_candidate0_b0_initialization(
                    cfg.init_checkpoint))
        else:
            raise ValueError(
                "online recursive Joint Full must train from scratch; "
                "--init_checkpoint is forbidden")
    if cfg.checkpoint is not None:
        validate_online_resume_checkpoint(cfg.checkpoint, cfg)
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
    if bool(getattr(cfg, 'ct_online_recursive_training', False)):
        # Keep mini_val untouched.  Joint checkpoint selection uses only the
        # atomic dev partition of mini_train.
        val_data = PartitionedTestTrackingSampler(
            train_data.dataset, config=cfg, partition='dev')
    else:
        val_data = get_dataset(
            cfg, type='test', split=cfg.val_split, protocol_role='val')
    loader_seed = int(cfg.seed or 42)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(loader_seed + 31001)

    def seed_loader_worker(worker_id):
        worker_seed = int(torch.initial_seed() % (2 ** 32))
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    if bool(getattr(cfg, 'ct_online_recursive_training', False)):
        if int(getattr(cfg, 'ct_router_horizon', 3)) != 3:
            raise ValueError("online Joint Full currently requires H=3")
        tracklet_slots = int(getattr(
            cfg, 'ct_recursive_tracklet_slots', 4))
        candidate_views = int(getattr(
            cfg, 'ct_recursive_candidate_views', 4))
        expected_batch_size = (
            tracklet_slots
            if int(getattr(cfg, 'ct_joint_contract_version', 1)) >= 3
            else tracklet_slots * candidate_views)
        if int(cfg.batch_size) != expected_batch_size:
            raise ValueError(
                "online recursive batch_size must equal the canonical B0 "
                f"slot count ({expected_batch_size})")
        online_batch_sampler = OnlineRecursiveBatchSampler(
            train_data,
            slots=tracklet_slots,
            candidate_views=candidate_views,
            seed=loader_seed,
            partition_seed=int(getattr(
                cfg, 'ct_partition_seed', 42)),
            partition=str(getattr(cfg, 'ct_router_partition', 'train')),
            shadow_interval=int(getattr(
                cfg, 'ct_router_shadow_interval', 2)),
            shadow_slots_per_event=int(getattr(
                cfg, 'ct_router_shadow_slots_per_event', 1)),
            shadow_enabled=bool(getattr(cfg, 'ct_enable_b3', True)),
        )
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
            train_data, batch_size=cfg.batch_size,
            num_workers=cfg.workers, shuffle=True, drop_last=True,
            pin_memory=True, worker_init_fn=seed_loader_worker,
            generator=loader_generator)
    val_loader = DataLoader(val_data, batch_size=1, num_workers=cfg.workers, collate_fn=lambda x: x, pin_memory=True)
    write_run_provenance(
        run_root_dir, cfg, {"train": train_data, "val": val_data},
        mode="train", root=project_root)
    checkpoint_callback = ModelCheckpoint(
        monitor=str(getattr(cfg, 'checkpoint_monitor', 'precision/test')),
        mode=str(getattr(cfg, 'checkpoint_mode', 'max')),
        save_last=True,
        save_top_k=cfg.save_top_k)
    learningrate_callback = LearningRateMonitor(logging_interval="step")

    # init trainer
    # RecursiveTrackState is intentionally process-local.  Until an explicit
    # cross-rank state coordinator exists, multi-device DDP would duplicate
    # tracklets and let the canonical histories silently diverge.
    trainer_devices = (
        1 if bool(getattr(cfg, 'ct_online_recursive_training', False)) else -1)
    trainer = pl.Trainer(devices=trainer_devices, accelerator='auto', max_epochs=cfg.epoch,
                         callbacks=[checkpoint_callback,learningrate_callback],
                         default_root_dir=run_root_dir,
                         check_val_every_n_epoch=cfg.check_val_every_n_epoch,
                         num_sanity_val_steps=0,
                          # Contract-v3 owns unscale/clip/step per optimizer;
                          # Trainer-level clipping would couple both domains.
                          gradient_clip_val=(
                              0.0 if bool(getattr(
                                  cfg, 'ct_separate_optimizers', False))
                              else cfg.gradient_clip_val),
                         limit_train_batches=getattr(
                             cfg, 'limit_train_batches', 1.0),
                         limit_val_batches=getattr(
                             cfg, 'limit_val_batches', 1.0),
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
    source_test_data = get_dataset(
        cfg, type='test', split=cfg.test_split, protocol_role='test')
    eval_partition = getattr(cfg, 'ct_eval_partition', None)
    if eval_partition is not None:
        source_dataset = getattr(source_test_data, 'dataset', None)
        if source_dataset is None:
            raise RuntimeError(
                '--ct-eval-partition requires a tracklet test sampler')
        test_data = PartitionedTestTrackingSampler(
            source_dataset, config=cfg, partition=eval_partition)
    else:
        test_data = source_test_data
    test_loader = DataLoader(test_data, batch_size=1, num_workers=cfg.workers, collate_fn=lambda x: x, pin_memory=True)
    write_run_provenance(
        run_root_dir, cfg, {"test": test_data}, mode="test", root=project_root)

    trainer = pl.Trainer(devices=-1, accelerator='auto', default_root_dir=run_root_dir)

    if cfg.checkpoint is None:
        net = get_model(cfg.net_model)(cfg)
    else:
        net = get_model(cfg.net_model).load_from_checkpoint(cfg.checkpoint, config=cfg)
    trainer.test(net, test_loader, ckpt_path=cfg.checkpoint)
