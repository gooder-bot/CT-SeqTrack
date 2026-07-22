import argparse
import json
import sys
import time
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
from tools.m0_diagnostic_utils import IndexedDiagnosticDataset  # noqa: E402


def load_config(path):
    with open(path, "r") as f:
        cfg = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
    if "preloading" not in cfg:
        cfg.preloading = False
    if "tiny" not in cfg:
        cfg.tiny = False
    return cfg


def overlay_virtual_rate_protocol(cfg, protocol_cfg_path):
    """Copy only virtual-rate sampling fields from an existing protocol config."""
    if protocol_cfg_path is None:
        return []
    protocol_cfg = load_config(protocol_cfg_path)
    copied = []
    for key, value in protocol_cfg.items():
        if str(key).startswith("virtual_rate_"):
            setattr(cfg, key, value)
            copied.append(str(key))
    if not copied:
        raise ValueError(
            f"Protocol config has no virtual_rate_* fields: {protocol_cfg_path}")
    return sorted(copied)


def to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def move_to_device(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=False)
    if isinstance(batch, dict):
        return {key: move_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [move_to_device(value, device) for value in batch]
    if isinstance(batch, tuple):
        return tuple(move_to_device(value, device) for value in batch)
    return batch


def is_paired_batch(batch):
    return isinstance(batch, dict) and "view_a" in batch and "view_b" in batch


def has_full_history(batch, hist_num):
    if is_paired_batch(batch):
        return has_full_history(batch["view_a"], hist_num) and has_full_history(batch["view_b"], hist_num)
    valid_mask = to_numpy(batch["valid_mask"])
    if valid_mask.ndim > 1:
        return bool(np.all(valid_mask.sum(axis=-1) >= int(hist_num)))
    return bool(valid_mask.sum() >= int(hist_num))


def unwrap_optimizer(config_result):
    if isinstance(config_result, torch.optim.Optimizer):
        return config_result, None
    if isinstance(config_result, dict):
        optimizer = config_result["optimizer"]
        scheduler_cfg = config_result.get("lr_scheduler")
        scheduler = None
        if isinstance(scheduler_cfg, dict):
            scheduler = scheduler_cfg.get("scheduler")
        else:
            scheduler = scheduler_cfg
        return optimizer, scheduler
    if isinstance(config_result, (list, tuple)):
        optimizer = config_result[0][0] if isinstance(config_result[0], (list, tuple)) else config_result[0]
        scheduler = None
        if len(config_result) > 1:
            scheduler = config_result[1][0] if isinstance(config_result[1], (list, tuple)) else config_result[1]
        return optimizer, scheduler
    raise TypeError(f"Unsupported optimizer config type: {type(config_result)}")


def grad_stats(model):
    total_sq_norm = 0.0
    max_abs = 0.0
    finite = True
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        finite = finite and bool(torch.isfinite(grad).all().item())
        total_sq_norm += float(torch.sum(grad.float() ** 2).item())
        max_abs = max(max_abs, float(torch.max(torch.abs(grad)).item()))
    return total_sq_norm ** 0.5, max_abs, finite


def named_grad_stats(model, name_fragment):
    total_sq_norm = 0.0
    max_abs = 0.0
    finite = True
    parameter_count = 0
    parameter_with_grad_count = 0
    for name, parameter in model.named_parameters():
        if name_fragment not in name:
            continue
        parameter_count += 1
        if parameter.grad is None:
            continue
        parameter_with_grad_count += 1
        grad = parameter.grad.detach()
        finite = finite and bool(torch.isfinite(grad).all().item())
        total_sq_norm += float(torch.sum(grad.float() ** 2).item())
        max_abs = max(max_abs, float(torch.max(torch.abs(grad)).item()))
    return {
        "parameter_count": parameter_count,
        "parameter_with_grad_count": parameter_with_grad_count,
        "grad_norm": total_sq_norm ** 0.5,
        "grad_max_abs": max_abs,
        "grad_finite": finite,
    }


def tensor_summary(value):
    if value is None:
        return None
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    flat = value.detach().float().reshape(-1)
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p95": None,
            "max": None,
        }
    quantiles = torch.quantile(
        flat,
        torch.tensor([0.0, 0.25, 0.5, 0.75, 0.95, 1.0], device=flat.device),
    ).cpu().tolist()
    return {
        "count": int(flat.numel()),
        "mean": float(flat.mean().item()),
        "min": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "p50": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "max": float(quantiles[5]),
    }


def tensor_values(value):
    if value is None:
        return []
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    flat = value.detach().float().reshape(-1).cpu().numpy()
    return [float(item) for item in flat if np.isfinite(item)]


def summarize_aligned_groups(accumulator, group_key, metric_keys, formatter):
    group_values = accumulator.get(group_key, [])
    if not group_values:
        return {}
    grouped = {}
    for metric_key in metric_keys:
        metric_values = accumulator.get(metric_key, [])
        if len(metric_values) != len(group_values):
            continue
        for group_value, metric_value in zip(group_values, metric_values):
            label = formatter(group_value)
            grouped.setdefault(label, {}).setdefault(metric_key, []).append(metric_value)
    return {
        label: {
            metric_key: tensor_summary(metric_values)
            for metric_key, metric_values in metrics.items()
        }
        for label, metrics in sorted(grouped.items())
    }


def build_residual_sample_values(batch, output):
    if is_paired_batch(batch) or is_paired_batch(output):
        return None
    if "motion_obs_pred" not in output or "dynamics_displacement_pred" not in output:
        return None
    target_motion = batch["motion_label"][:, 0, :3].to(
        device=output["motion_obs_pred"].device,
        dtype=output["motion_obs_pred"].dtype,
    )
    observation_motion = output["motion_obs_pred"][:, :3]
    dynamics_motion = output["dynamics_displacement_pred"]
    return {
        "observation_error_norm": tensor_values(
            torch.linalg.norm(observation_motion - target_motion, dim=1)
        ),
        "dynamics_error_norm": tensor_values(
            torch.linalg.norm(dynamics_motion - target_motion, dim=1)
        ),
        "observation_dynamics_gap_norm": tensor_values(
            torch.linalg.norm(dynamics_motion - observation_motion, dim=1)
        ),
        "target_motion_norm": tensor_values(torch.linalg.norm(target_motion, dim=1)),
        "alpha": tensor_values(output.get("dynamics_residual_alpha")),
        "applied_residual_norm": tensor_values(
            torch.linalg.norm(output["motion_dynamics_residual"], dim=1)
            if "motion_dynamics_residual" in output
            else None
        ),
        "raw_dynamics_norm": tensor_values(output.get("dynamics_residual_raw_norm")),
        "clamped_dynamics_norm": tensor_values(
            output.get("dynamics_residual_clamped_norm")
        ),
        "applied_mask": tensor_values(output.get("dynamics_residual_applied_mask")),
        "clamp_mask": tensor_values(output.get("dynamics_residual_clamp_mask")),
        "current_delta_t": tensor_values(batch.get("current_delta_t")),
        "num_points_in_search": tensor_values(batch.get("num_points_in_search")),
        "candidate_id": tensor_values(batch.get("candidate_id")),
        "dynamics_valid": tensor_values(output.get("dynamics_valid")),
        "invalid_applied_residual_norm": tensor_values(
            torch.linalg.norm(output["motion_dynamics_residual"], dim=1)[
                output["dynamics_valid"].reshape(-1) <= 0
            ]
            if "motion_dynamics_residual" in output and "dynamics_valid" in output
            else None
        ),
    }


def extend_samples(accumulator, sample_values):
    if sample_values is None:
        return
    for key, values in sample_values.items():
        accumulator.setdefault(key, []).extend(values)


def summarize_accumulated_samples(accumulator, gate_grad_norms, encoder_grad_norms):
    summary = {key: tensor_summary(values) for key, values in accumulator.items()}
    summary["gate_grad_norm"] = tensor_summary(gate_grad_norms)
    summary["encoder_grad_norm"] = tensor_summary(encoder_grad_norms)
    if accumulator.get("applied_mask"):
        summary["applied_ratio"] = float(np.mean(accumulator["applied_mask"]))
    if accumulator.get("clamp_mask"):
        summary["clamp_ratio"] = float(np.mean(accumulator["clamp_mask"]))
    if accumulator.get("dynamics_valid"):
        summary["dynamics_valid_ratio"] = float(np.mean(accumulator["dynamics_valid"]))
    grouped_metrics = (
        "observation_error_norm",
        "dynamics_error_norm",
        "observation_dynamics_gap_norm",
        "applied_residual_norm",
        "alpha",
        "num_points_in_search",
    )
    summary["by_candidate_id"] = summarize_aligned_groups(
        accumulator,
        "candidate_id",
        grouped_metrics,
        lambda value: str(int(round(value))),
    )
    summary["by_current_delta_t"] = summarize_aligned_groups(
        accumulator,
        "current_delta_t",
        grouped_metrics,
        lambda value: f"{float(value):.6g}",
    )
    return summary


def build_residual_diagnostics(batch, output, model):
    if is_paired_batch(batch) or is_paired_batch(output):
        return None
    required = (
        "motion_obs_pred",
        "dynamics_displacement_pred",
        "motion_dynamics_residual",
        "dynamics_residual_alpha",
    )
    if any(key not in output for key in required):
        return None

    target_motion = batch["motion_label"][:, 0, :3].to(
        device=output["motion_obs_pred"].device,
        dtype=output["motion_obs_pred"].dtype,
    )
    observation_motion = output["motion_obs_pred"][:, :3]
    dynamics_motion = output["dynamics_displacement_pred"]
    applied_residual = output["motion_dynamics_residual"]
    dynamics_valid = output["dynamics_valid"].reshape(-1)
    invalid_residual_norm = torch.linalg.norm(applied_residual, dim=1)[
        dynamics_valid <= 0
    ]

    diagnostics = {
        "observation_error_norm": tensor_summary(
            torch.linalg.norm(observation_motion - target_motion, dim=1)
        ),
        "dynamics_error_norm": tensor_summary(
            torch.linalg.norm(dynamics_motion - target_motion, dim=1)
        ),
        "observation_dynamics_gap_norm": tensor_summary(
            torch.linalg.norm(dynamics_motion - observation_motion, dim=1)
        ),
        "target_motion_norm": tensor_summary(torch.linalg.norm(target_motion, dim=1)),
        "alpha": tensor_summary(output["dynamics_residual_alpha"]),
        "applied_residual_norm": tensor_summary(
            torch.linalg.norm(applied_residual, dim=1)
        ),
        "raw_dynamics_norm": tensor_summary(output.get("dynamics_residual_raw_norm")),
        "clamped_dynamics_norm": tensor_summary(
            output.get("dynamics_residual_clamped_norm")
        ),
        "applied_ratio": float(
            output["dynamics_residual_applied_mask"].detach().float().mean().item()
        ),
        "clamp_ratio": float(
            output["dynamics_residual_clamp_mask"].detach().float().mean().item()
        ),
        "current_delta_t": tensor_summary(batch.get("current_delta_t")),
        "num_points_in_search": tensor_summary(batch.get("num_points_in_search")),
        "dynamics_valid_ratio": float(
            output["dynamics_valid"].detach().float().mean().item()
        ),
        "invalid_applied_residual_norm": tensor_summary(invalid_residual_norm),
        "configured_max_residual_norm": float(
            getattr(model, "dynamics_residual_scale", 0.0)
            * getattr(model, "dynamics_max_alpha", 0.0)
            * getattr(model, "dynamics_max_residual_norm", 0.0)
        ),
    }
    if hasattr(model, "dynamics_residual_gate"):
        gate_output = model.dynamics_residual_gate.net[-1]
        diagnostics["gate_output_bias"] = tensor_summary(gate_output.bias)
    return diagnostics


def build_innovation_diagnostics(batch, output):
    if is_paired_batch(batch) or is_paired_batch(output):
        return None
    required = (
        "motion_obs_pred",
        "dynamics_displacement_pred",
        "dynamics_innovation_applied",
        "dynamics_innovation_alpha",
        "dynamics_innovation_radius",
        "dynamics_valid",
    )
    if any(key not in output for key in required):
        return None
    target = batch.get(
        "dynamics_displacement_label", batch["motion_label"][:, 0, :3])
    target = target.to(
        device=output["motion_obs_pred"].device,
        dtype=output["motion_obs_pred"].dtype,
    )
    observation = output["motion_obs_pred"][:, :3]
    dynamics = output["dynamics_displacement_pred"]
    applied = output["dynamics_innovation_applied"]
    applied_norm = torch.linalg.norm(applied, dim=1)
    dynamics_valid = output["dynamics_valid"].reshape(-1)
    innovation_valid = output.get(
        "dynamics_innovation_valid", output["dynamics_valid"]).reshape(-1)
    # Keep invalid history separate from the empty-search safety mask.  The
    # latter also clears innovation_valid, but E4 requires both fallbacks to be
    # observed and audited independently.
    invalid_applied = applied_norm[dynamics_valid <= 0]
    search_points = batch.get("num_points_in_search")
    if search_points is not None:
        search_points = search_points.detach().reshape(-1)
        empty_search_ratio = float((search_points <= 0).float().mean().item())
        empty_applied = applied_norm[search_points <= 0]
    else:
        empty_search_ratio = None
        empty_applied = applied.new_zeros((0,))
    effective_alpha = output["dynamics_innovation_alpha"].reshape(-1)
    radius = output["dynamics_innovation_radius"].reshape(-1)
    bound_violation = torch.clamp(applied_norm - effective_alpha * radius, min=0.0)
    resampled = batch.get("_diagnostic_resampled")
    if resampled is not None:
        resampled = resampled.detach().reshape(-1).to(applied.device)
        resampled_count = int((resampled > 0).sum().item())
        resampled_ratio = float((resampled > 0).float().mean().item())
    else:
        resampled_count = 0
        resampled_ratio = None
    diagnostics = {
        "observation_error_norm": tensor_summary(
            torch.linalg.norm(observation - target, dim=1)),
        "dynamics_error_norm": tensor_summary(
            torch.linalg.norm(dynamics - target, dim=1)),
        "raw_innovation_norm": tensor_summary(
            output.get("dynamics_innovation_raw_norm")),
        "clamped_innovation_norm": tensor_summary(
            output.get("dynamics_innovation_clamped_norm")),
        "applied_innovation_norm": tensor_summary(
            output.get("dynamics_innovation_applied_norm")),
        "radius": tensor_summary(radius),
        "alpha": tensor_summary(effective_alpha),
        "bound_violation": tensor_summary(bound_violation),
        "applied_ratio": float(
            output["dynamics_innovation_applied_mask"].detach().float().mean().item()),
        "applied_count": int(
            output["dynamics_innovation_applied_mask"].detach().sum().item()),
        "clamp_ratio": float(
            output["dynamics_innovation_clamp_mask"].detach().float().mean().item()),
        "clamp_count": int(
            output["dynamics_innovation_clamp_mask"].detach().sum().item()),
        "invalid_applied_innovation_norm": tensor_summary(invalid_applied),
        "empty_applied_innovation_norm": tensor_summary(empty_applied),
        "dynamics_valid_ratio": float(
            dynamics_valid.detach().float().mean().item()),
        "innovation_valid_ratio": float(
            innovation_valid.detach().float().mean().item()),
        "empty_search_ratio": empty_search_ratio,
        "adapter_norm": tensor_summary(output.get("physical_time_adapter_norm")),
        "adapter_scale_effective": tensor_summary(
            output.get("physical_time_adapter_scale")),
        "innovation_scale_effective": tensor_summary(
            output.get("dynamics_innovation_scale_effective")),
        "resampled_count": resampled_count,
        "resampled_ratio": resampled_ratio,
        "current_delta_t_effective": tensor_summary(
            batch.get("current_delta_t_effective", batch.get("current_delta_t"))),
        "candidate_trajectory_mode_id": tensor_summary(
            batch.get("candidate_trajectory_mode_id")),
    }
    return diagnostics


def _diagnostic_stat(diagnostics, key, stat):
    values = []
    for item in diagnostics:
        summary = item.get(key)
        if not isinstance(summary, dict):
            continue
        value = summary.get(stat)
        if isinstance(value, (int, float)) and np.isfinite(value):
            values.append(float(value))
    return values


def summarize_innovation_run(
        diagnostics, gradients, completed_steps, optimizer_step_count,
        weights, weight_load_report):
    sample_count = int(sum(
        int(item["applied_innovation_norm"]["count"])
        for item in diagnostics
    ))
    invalid_count = int(sum(
        int(item["invalid_applied_innovation_norm"]["count"])
        for item in diagnostics
    ))
    empty_count = int(sum(
        int(item["empty_applied_innovation_norm"]["count"])
        for item in diagnostics
    ))
    resampled_count = int(sum(
        int(item.get("resampled_count", 0)) for item in diagnostics
    ))
    applied_count = int(sum(
        int(item.get("applied_count", 0)) for item in diagnostics
    ))
    clamp_count = int(sum(
        int(item.get("clamp_count", 0)) for item in diagnostics
    ))

    encoder_grad_norms = [
        float(item["encoder"]["grad_norm"]) for item in gradients
    ]
    adapter_grad_norms = [
        float(item["adapter"]["grad_norm"]) for item in gradients
    ]
    resampled_steps = [
        index + 1 for index, item in enumerate(diagnostics)
        if int(item.get("resampled_count", 0)) > 0
    ]

    def maximum(key, stat="max"):
        values = _diagnostic_stat(diagnostics, key, stat)
        return max(values) if values else None

    return {
        "schema": "ct_seqtrack.m1_m2_innovation_gate_summary",
        "schema_version": 1,
        "completed_steps": int(completed_steps),
        "optimizer_step_count": int(optimizer_step_count),
        "sample_count": sample_count,
        "invalid_sample_count": invalid_count,
        "empty_sample_count": empty_count,
        "resampled_sample_count": resampled_count,
        "resampled_steps": resampled_steps,
        "applied_count": applied_count,
        "clamp_count": clamp_count,
        "applied_ratio": (
            float(applied_count / sample_count) if sample_count else None),
        "clamp_ratio": (
            float(clamp_count / sample_count) if sample_count else None),
        "applied_innovation_max": maximum("applied_innovation_norm"),
        "invalid_applied_max": maximum("invalid_applied_innovation_norm"),
        "empty_applied_max": maximum("empty_applied_innovation_norm"),
        "adapter_output_max": maximum("adapter_norm"),
        "adapter_scale_effective_max": maximum("adapter_scale_effective"),
        "innovation_scale_effective_max": maximum(
            "innovation_scale_effective"),
        "bound_violation_max": maximum("bound_violation"),
        "encoder_grad_norm_max": (
            max(encoder_grad_norms) if encoder_grad_norms else None),
        "adapter_grad_norm_max": (
            max(adapter_grad_norms) if adapter_grad_norms else None),
        "encoder_nonzero_grad_step_count": int(sum(
            value > 0.0 for value in encoder_grad_norms)),
        "adapter_nonzero_grad_step_count": int(sum(
            value > 0.0 for value in adapter_grad_norms)),
        "weights": weights,
        "weight_load_report": weight_load_report,
    }


def innovation_requirement_failures(summary, args):
    failures = []
    if (summary["optimizer_step_count"]
            < args.require_min_optimizer_steps):
        failures.append(
            "optimizer steps below the required minimum: "
            f"{summary['optimizer_step_count']} < "
            f"{args.require_min_optimizer_steps}")
    if args.require_invalid and summary["invalid_sample_count"] <= 0:
        failures.append("no invalid-history sample was observed")
    if (args.require_invalid
            and summary["invalid_applied_max"] is not None
            and summary["invalid_applied_max"] != 0.0):
        failures.append(
            "invalid-history fallback was not exact zero: "
            f"{summary['invalid_applied_max']}")
    if args.require_empty and summary["empty_sample_count"] <= 0:
        failures.append("no empty-search sample was observed")
    if (args.require_empty
            and summary["empty_applied_max"] is not None
            and summary["empty_applied_max"] != 0.0):
        failures.append(
            "empty-search fallback was not exact zero: "
            f"{summary['empty_applied_max']}")
    if args.require_resampled and summary["resampled_sample_count"] <= 0:
        failures.append("no sampler-resampled sample was observed")
    if (args.require_nonzero_encoder_grad
            and (summary["encoder_grad_norm_max"] is None
                 or summary["encoder_grad_norm_max"] <= 0.0)):
        failures.append("DynamicsEncoder received no nonzero gradient")
    if (args.require_nonzero_adapter_grad
            and (summary["adapter_grad_norm_max"] is None
                 or summary["adapter_grad_norm_max"] <= 0.0)):
        failures.append("physical-time adapter received no nonzero gradient")
    if args.require_zero_warmup_output:
        zero_fields = (
            "applied_innovation_max",
            "adapter_output_max",
            "adapter_scale_effective_max",
            "innovation_scale_effective_max",
        )
        for key in zero_fields:
            value = summary.get(key)
            if value is None or value != 0.0:
                failures.append(f"warmup requires {key}=0, got {value}")
    bound_violation = summary.get("bound_violation_max")
    if bound_violation is None or bound_violation > 1e-6:
        failures.append(
            f"proposal innovation exceeded its alpha*R(dt) bound: {bound_violation}")
    return failures


def load_weights(model, path):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint payload type: {type(payload)}")
    if "state_dict" in payload:
        state_dict = payload["state_dict"]
    elif "model" in payload:
        state_dict = payload["model"]
    else:
        state_dict = payload
    if not all(isinstance(key, str) for key in state_dict):
        raise TypeError("Checkpoint state_dict keys must be strings.")

    model_keys = set(model.state_dict())
    candidates = [("none", state_dict)]
    for prefix in ("model.", "module."):
        stripped = {
            key[len(prefix):] if key.startswith(prefix) else key: value
            for key, value in state_dict.items()
        }
        candidates.append((prefix, stripped))
    selected_prefix, state_dict = max(
        candidates,
        key=lambda item: len(model_keys.intersection(item[1])),
    )
    matched_keys = model_keys.intersection(state_dict)
    if not matched_keys:
        raise RuntimeError(
            "Checkpoint has no keys matching this model. Check whether it uses an unsupported prefix."
        )
    critical_prefixes = ("seg_pointnet.", "mini_pointnet.", "motion_mlp.")
    missing_critical = [
        prefix
        for prefix in critical_prefixes
        if not any(key.startswith(prefix) for key in matched_keys)
    ]
    if missing_critical:
        raise RuntimeError(
            "Checkpoint did not load the observation backbone/head prefixes: "
            + ", ".join(missing_critical)
        )

    incompatible = model.load_state_dict(state_dict, strict=False)
    report = {
        "path": str(path),
        "selected_prefix_strip": selected_prefix,
        "checkpoint_key_count": len(state_dict),
        "matched_key_count": len(matched_keys),
        "missing_key_count": len(incompatible.missing_keys),
        "unexpected_key_count": len(incompatible.unexpected_keys),
    }
    print(f"loaded weights: {path}")
    print(f"matched keys: {len(matched_keys)}/{len(model_keys)}")
    print(f"selected prefix strip: {selected_prefix}")
    print(f"missing keys: {len(incompatible.missing_keys)}")
    print(f"unexpected keys: {len(incompatible.unexpected_keys)}")
    if incompatible.missing_keys:
        print("missing key sample:", incompatible.missing_keys[:12])
    if incompatible.unexpected_keys:
        print("unexpected key sample:", incompatible.unexpected_keys[:12])
    return report


def freeze_batchnorm_stats(model):
    frozen = 0
    for module in model.modules():
        if isinstance(
            module,
            (
                torch.nn.BatchNorm1d,
                torch.nn.BatchNorm2d,
                torch.nn.BatchNorm3d,
                torch.nn.SyncBatchNorm,
            ),
        ):
            module.eval()
            frozen += 1
    return frozen


def save_checkpoint(path, model, optimizer, scheduler, step, cfg, loss_record):
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cfg": dict(cfg),
        "loss_record": loss_record,
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, path)


def build_loader(cfg, args):
    split = args.split if args.split is not None else cfg.train_split
    dataset = get_dataset(
        cfg, type=cfg.train_type, split=split, protocol_role="train")
    if args.track_resampled:
        if bool(getattr(cfg, "use_twc", False)):
            raise ValueError(
                "--track-resampled currently requires a single-view training batch.")
        dataset = IndexedDiagnosticDataset(dataset)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.workers,
        shuffle=not args.no_shuffle,
        drop_last=not args.keep_partial_batch,
        pin_memory=False,
    )
    return loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--path", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument(
        "--protocol-cfg",
        default=None,
        help=(
            "Optional existing config whose virtual_rate_* fields are overlaid "
            "onto the M1/M2 config for gap/burst smoke tests."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=8,
        help="Number of completed batches; use 0 to traverse the full loader.",
    )
    parser.add_argument("--skip-batches", type=int, default=0)
    parser.add_argument("--require-full-history", action="store_true")
    parser.add_argument("--pseudo-time", action="store_true")
    parser.add_argument("--twc", action="store_true",
                        help="Temporarily enable P4 paired-view TWC batch mode.")
    parser.add_argument("--obs-gate", action="store_true",
                        help="Temporarily enable P5 observability gate with dynamics branch.")
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument(
        "--keep-partial-batch",
        action="store_true",
        help="Keep the final partial batch so full-split diagnostics do not drop samples.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--memory-fraction", type=float, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument(
        "--weights",
        default=None,
        help=(
            "Optional model-only checkpoint. Lightning state_dict and the local "
            "check_train_steps model payload are both supported."
        ),
    )
    parser.add_argument(
        "--residual-diagnostics",
        action="store_true",
        help="Require and log P0-A bounded-residual proposal, magnitude, and gate diagnostics.",
    )
    parser.add_argument(
        "--innovation-diagnostics",
        action="store_true",
        help="Require and log M1/M2 adapter and bounded proposal-innovation diagnostics.",
    )
    parser.add_argument(
        "--residual-warmup-epoch",
        type=int,
        default=None,
        help="Override dynamics_warmup_epoch for this diagnostic run; use 0 to exercise the residual immediately.",
    )
    parser.add_argument(
        "--innovation-warmup-epoch",
        type=int,
        default=None,
        help="Override dynamics_innovation_warmup_epoch; use 0 for active M2 smoke.",
    )
    parser.add_argument(
        "--adapter-warmup-epoch",
        type=int,
        default=None,
        help=(
            "Override physical_time_adapter_warmup_epoch. When omitted it "
            "inherits the innovation warmup in M2 configs."),
    )
    parser.add_argument(
        "--track-resampled",
        action="store_true",
        help=(
            "Attach stable requested-index metadata and detect sampler fallback "
            "that returns a different frame/candidate."),
    )
    parser.add_argument("--require-invalid", action="store_true")
    parser.add_argument("--require-empty", action="store_true")
    parser.add_argument("--require-resampled", action="store_true")
    parser.add_argument("--require-nonzero-encoder-grad", action="store_true")
    parser.add_argument("--require-nonzero-adapter-grad", action="store_true")
    parser.add_argument(
        "--require-min-optimizer-steps",
        type=int,
        default=0,
        help=(
            "Require at least this many real optimizer updates before the "
            "M1/M2 gate may pass."),
    )
    parser.add_argument(
        "--require-zero-warmup-output",
        action="store_true",
        help=(
            "Require exact zero adapter output/scale and innovation output/scale "
            "for every completed warmup step."),
    )
    parser.add_argument(
        "--stop-when-requirements-met",
        action="store_true",
        help=(
            "Stop a bounded deterministic coverage scan as soon as all "
            "requested M1/M2 requirements pass."),
    )
    parser.add_argument(
        "--no-optimizer-step",
        action="store_true",
        help="Run forward/backward diagnostics without changing model or optimizer state.",
    )
    parser.add_argument(
        "--diagnostic-summary-file",
        default="output/check_train_steps_residual_summary.json",
        help="Run-level P0-A summary written when --residual-diagnostics is enabled.",
    )
    parser.add_argument("--train-bn", action="store_true",
                        help="Keep BatchNorm layers in train mode. By default BN stats are frozen so batch_size=1 works.")
    parser.add_argument("--log-file", default="output/check_train_steps_loss.jsonl")
    parser.add_argument(
        "--innovation-summary-file",
        default=None,
        help=(
            "Run-level M1/M2 JSON summary. Defaults to the log-file path with "
            "a .summary.json suffix."),
    )
    parser.add_argument("--checkpoint-dir", default="output/check_train_steps_ckpt")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Save last.pt every N completed steps; use 0 to disable successful-step checkpoints.",
    )
    parser.add_argument("--tag", default="check_train_steps")
    args = parser.parse_args()

    if args.max_steps < 0:
        raise ValueError("--max-steps must be non-negative; use 0 for the full loader.")
    if args.checkpoint_every < 0:
        raise ValueError("--checkpoint-every must be non-negative.")
    if args.require_min_optimizer_steps < 0:
        raise ValueError("--require-min-optimizer-steps must be non-negative.")
    if args.no_optimizer_step and args.require_min_optimizer_steps > 0:
        raise ValueError(
            "--require-min-optimizer-steps is incompatible with "
            "--no-optimizer-step.")
    innovation_requirements = (
        args.require_invalid
        or args.require_empty
        or args.require_resampled
        or args.require_nonzero_encoder_grad
        or args.require_nonzero_adapter_grad
        or args.require_zero_warmup_output
        or args.require_min_optimizer_steps > 0
    )
    if innovation_requirements and not args.innovation_diagnostics:
        raise ValueError(
            "M1/M2 --require-* flags require --innovation-diagnostics.")
    if args.require_resampled:
        args.track_resampled = True
    if args.stop_when_requirements_met and not innovation_requirements:
        raise ValueError(
            "--stop-when-requirements-met requires at least one M1/M2 "
            "--require-* condition.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = load_config(args.cfg)
    protocol_overlay_keys = overlay_virtual_rate_protocol(cfg, args.protocol_cfg)
    if args.path is not None:
        cfg.path = args.path
    if args.version is not None:
        cfg.version = args.version
    if args.pseudo_time:
        cfg.use_real_time = False
    if args.twc:
        cfg.use_twc = True
    if args.obs_gate:
        cfg.use_dynamics_encoder = True
        cfg.use_observability_gate = True
    if args.residual_warmup_epoch is not None:
        if args.residual_warmup_epoch < 0:
            raise ValueError("--residual-warmup-epoch must be non-negative.")
        cfg.dynamics_warmup_epoch = args.residual_warmup_epoch
    if args.innovation_warmup_epoch is not None:
        if args.innovation_warmup_epoch < 0:
            raise ValueError("--innovation-warmup-epoch must be non-negative.")
        cfg.dynamics_innovation_warmup_epoch = args.innovation_warmup_epoch
    if args.adapter_warmup_epoch is not None:
        if args.adapter_warmup_epoch < 0:
            raise ValueError("--adapter-warmup-epoch must be non-negative.")
        cfg.physical_time_adapter_warmup_epoch = args.adapter_warmup_epoch
    elif args.innovation_warmup_epoch is not None:
        cfg.physical_time_adapter_warmup_epoch = args.innovation_warmup_epoch
    cfg.batch_size = args.batch_size
    cfg.workers = args.workers
    cfg.tag = args.tag

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required because compute_loss currently creates CUDA tensors.")

    if args.memory_fraction is not None:
        if not (0.0 < args.memory_fraction <= 1.0):
            raise ValueError("--memory-fraction must be in (0, 1].")
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device=0)

    device = torch.device("cuda:0")
    print(f"device: {device}")
    print(f"max_steps: {args.max_steps if args.max_steps > 0 else 'full_loader'}")
    print(f"batch_size: {cfg.batch_size}, workers: {cfg.workers}")
    print(f"use_real_time: {getattr(cfg, 'use_real_time', True)}")
    print(f"use_twc: {getattr(cfg, 'use_twc', False)}")
    print(f"use_observability_gate: {getattr(cfg, 'use_observability_gate', False)}")
    print(f"dynamics_motion_mode: {getattr(cfg, 'dynamics_motion_mode', 'feature')}")
    print(f"dynamics_warmup_epoch: {getattr(cfg, 'dynamics_warmup_epoch', 0)}")
    print(
        "dynamics_innovation_warmup_epoch: "
        f"{getattr(cfg, 'dynamics_innovation_warmup_epoch', 0)}")
    print(
        "physical_time_adapter_warmup_epoch: "
        f"{getattr(cfg, 'physical_time_adapter_warmup_epoch', 0)}")
    print(f"track_resampled: {args.track_resampled}")
    if args.protocol_cfg is not None:
        print(f"virtual-rate protocol config: {args.protocol_cfg}")
        print(f"virtual-rate overlay keys: {','.join(protocol_overlay_keys)}")
    print(f"optimizer_step_enabled: {not args.no_optimizer_step}")
    if args.memory_fraction is not None:
        print(f"cuda memory fraction limit: {args.memory_fraction}")

    loader = build_loader(cfg, args)
    train_dataloader_length = max(len(loader), 1)

    model = get_model(cfg.net_model)(cfg, train_dataloader_length=train_dataloader_length).to(device)
    weight_load_report = None
    if args.weights is not None:
        weight_load_report = load_weights(model, args.weights)
    if args.residual_diagnostics:
        motion_head_input_dim = int(model.motion_mlp[0].in_features)
        if getattr(model, "dynamics_motion_mode", None) != "residual":
            raise RuntimeError(
                "--residual-diagnostics requires the normalized residual motion mode."
            )
        if motion_head_input_dim != 256:
            raise RuntimeError(
                "Residual diagnostics require an observation-only 256-D motion head; "
                f"got {motion_head_input_dim}."
            )
    if args.innovation_diagnostics:
        motion_head_input_dim = int(model.motion_mlp[0].in_features)
        if getattr(model, "dynamics_motion_mode", None) != "proposal_innovation":
            raise RuntimeError(
                "--innovation-diagnostics requires dynamics_motion_mode=proposal_innovation."
            )
        if motion_head_input_dim != 256:
            raise RuntimeError(
                "Proposal innovation requires an observation-first 256-D motion head; "
                f"got {motion_head_input_dim}."
            )
    model.train()
    if not args.train_bn:
        frozen_bn = freeze_batchnorm_stats(model)
        print(f"frozen BatchNorm modules: {frozen_bn}")
    optimizer, scheduler = unwrap_optimizer(model.configure_optimizers())

    log_path = Path(args.log_file)
    ckpt_dir = Path(args.checkpoint_dir)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    completed_steps = 0
    seen_batches = 0
    residual_samples = {}
    residual_gate_grad_norms = []
    residual_encoder_grad_norms = []
    innovation_run_diagnostics = []
    innovation_run_gradients = []
    optimizer_step_count = 0

    with log_path.open("a", buffering=1) as log_file:
        for batch_idx, batch in enumerate(loader):
            if batch_idx < args.skip_batches:
                continue

            if args.require_full_history:
                if not has_full_history(batch, cfg.hist_num):
                    continue

            seen_batches += 1
            batch = move_to_device(batch, device)

            if not args.train_bn:
                freeze_batchnorm_stats(model)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            loss_dict = model.compute_loss(batch, output)
            loss = loss_dict["loss_total"]

            loss_finite = bool(torch.isfinite(loss).all().item())
            finite_by_key = {
                key: bool(torch.isfinite(value).all().item())
                for key, value in loss_dict.items()
                if torch.is_tensor(value)
            }
            if not loss_finite or not all(finite_by_key.values()):
                record = {
                    "step": completed_steps,
                    "batch_idx": batch_idx,
                    "ok": False,
                    "reason": "non_finite_loss",
                    "loss": {
                        key: float(value.detach().cpu().item())
                        for key, value in loss_dict.items()
                        if torch.is_tensor(value) and value.numel() == 1
                    },
                    "finite": finite_by_key,
                }
                log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                save_checkpoint(ckpt_dir / "last.pt", model, optimizer, scheduler, completed_steps, cfg, record)
                raise RuntimeError(f"Non-finite loss at batch_idx={batch_idx}: {record}")

            loss.backward()
            if args.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            grad_norm, grad_max_abs, grad_finite = grad_stats(model)
            if not grad_finite:
                raise RuntimeError(f"Non-finite gradient at batch_idx={batch_idx}.")

            residual_diagnostics = build_residual_diagnostics(batch, output, model)
            if args.residual_diagnostics and residual_diagnostics is None:
                raise RuntimeError(
                    "--residual-diagnostics requires a non-TWC batch and a config with "
                    "use_dynamics_encoder=true and dynamics_motion_mode=residual_limited."
                )
            if residual_diagnostics is not None:
                invalid_residual = residual_diagnostics[
                    "invalid_applied_residual_norm"
                ]
                if invalid_residual["count"] > 0 and invalid_residual["max"] > 1e-8:
                    raise RuntimeError(
                        "dynamics_valid=0 produced a non-zero residual: "
                        f"max={invalid_residual['max']}."
                    )
            residual_gradients = None
            if residual_diagnostics is not None:
                residual_gradients = {
                    "gate": named_grad_stats(model, "dynamics_residual_gate"),
                    "encoder": named_grad_stats(model, "dynamics_encoder"),
                }

                extend_samples(
                    residual_samples,
                    build_residual_sample_values(batch, output),
                )
                residual_gate_grad_norms.append(residual_gradients["gate"]["grad_norm"])
                residual_encoder_grad_norms.append(
                    residual_gradients["encoder"]["grad_norm"]
                )

            innovation_diagnostics = build_innovation_diagnostics(batch, output)
            if args.innovation_diagnostics and innovation_diagnostics is None:
                raise RuntimeError(
                    "--innovation-diagnostics requires a non-TWC M1/M2 proposal batch."
                )
            innovation_gradients = None
            if innovation_diagnostics is not None:
                invalid_innovation = innovation_diagnostics[
                    "invalid_applied_innovation_norm"]
                if invalid_innovation["count"] > 0 and invalid_innovation["max"] > 0.0:
                    raise RuntimeError(
                        "dynamics_valid=0 produced a non-zero proposal innovation: "
                        f"max={invalid_innovation['max']}."
                    )
                effective_scale = output.get(
                    "dynamics_innovation_scale_effective")
                if effective_scale is not None \
                        and float(effective_scale.detach().item()) == 0.0:
                    applied = innovation_diagnostics["applied_innovation_norm"]
                    if applied["count"] > 0 and applied["max"] > 0.0:
                        raise RuntimeError(
                            "zero/warmup innovation scale did not exactly recover "
                            f"the observation proposal; max={applied['max']}.")
                if bool(getattr(
                        model, "dynamics_innovation_disable_on_empty_search", False)):
                    empty_innovation = innovation_diagnostics[
                        "empty_applied_innovation_norm"]
                    if empty_innovation["count"] > 0 and empty_innovation["max"] > 0.0:
                        raise RuntimeError(
                            "empty search produced a non-zero proposal innovation: "
                            f"max={empty_innovation['max']}.")
                bound_violation = innovation_diagnostics["bound_violation"]
                if (bound_violation["count"] > 0
                        and bound_violation["max"] > 1e-6):
                    raise RuntimeError(
                        "proposal innovation exceeded alpha*R(delta_t): "
                        f"max violation={bound_violation['max']}.")
                innovation_gradients = {
                    "encoder": named_grad_stats(model, "dynamics_encoder"),
                    "adapter": named_grad_stats(model, "physical_time_adapter"),
                }
                innovation_run_diagnostics.append(innovation_diagnostics)
                innovation_run_gradients.append(innovation_gradients)

            if not args.no_optimizer_step:
                optimizer.step()
                optimizer_step_count += 1
                if scheduler is not None:
                    scheduler.step()

            completed_steps += 1
            loss_values = {
                key: float(value.detach().cpu().item())
                for key, value in loss_dict.items()
                if torch.is_tensor(value) and value.numel() == 1
            }
            record = {
                "step": completed_steps,
                "batch_idx": batch_idx,
                "ok": True,
                "loss": loss_values,
                "finite": finite_by_key,
                "grad_norm": grad_norm,
                "grad_max_abs": grad_max_abs,
                "lr": optimizer.param_groups[0]["lr"],
                "optimizer_step_applied": not args.no_optimizer_step,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            if residual_diagnostics is not None:
                record["residual_diagnostics"] = residual_diagnostics
                record["residual_gradients"] = residual_gradients
            if innovation_diagnostics is not None:
                record["innovation_diagnostics"] = innovation_diagnostics
                record["innovation_gradients"] = innovation_gradients
            log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            log_file.flush()
            if (
                args.checkpoint_every > 0
                and completed_steps % args.checkpoint_every == 0
            ):
                save_checkpoint(
                    ckpt_dir / "last.pt",
                    model,
                    optimizer,
                    scheduler,
                    completed_steps,
                    cfg,
                    record,
                )
            print(
                f"step={completed_steps}/{args.max_steps if args.max_steps > 0 else 'all'} "
                f"batch_idx={batch_idx} "
                f"loss_total={loss_values['loss_total']:.6f} "
                f"grad_norm={grad_norm:.6f} "
                f"lr={optimizer.param_groups[0]['lr']:.8f}"
            )
            if residual_diagnostics is not None:
                obs_error = residual_diagnostics["observation_error_norm"]
                applied = residual_diagnostics["applied_residual_norm"]
                gate_grad = residual_gradients["gate"]
                print(
                    "residual "
                    f"obs_err_p50={obs_error['p50']:.6f} "
                    f"obs_err_p95={obs_error['p95']:.6f} "
                    f"applied_p50={applied['p50']:.8f} "
                    f"applied_ratio={residual_diagnostics['applied_ratio']:.4f} "
                    f"gate_grad={gate_grad['grad_norm']:.8e}"
                )
            if innovation_diagnostics is not None:
                applied = innovation_diagnostics["applied_innovation_norm"]
                encoder_grad = innovation_gradients["encoder"]
                adapter_grad = innovation_gradients["adapter"]
                print(
                    "innovation "
                    f"applied_p50={applied['p50']:.6f} "
                    f"applied_ratio={innovation_diagnostics['applied_ratio']:.4f} "
                    f"clamp_ratio={innovation_diagnostics['clamp_ratio']:.4f} "
                    f"resampled={innovation_diagnostics['resampled_count']} "
                    f"encoder_grad={encoder_grad['grad_norm']:.8e} "
                    f"adapter_grad={adapter_grad['grad_norm']:.8e}"
                )

            if args.stop_when_requirements_met:
                partial_summary = summarize_innovation_run(
                    innovation_run_diagnostics,
                    innovation_run_gradients,
                    completed_steps,
                    optimizer_step_count,
                    args.weights,
                    weight_load_report,
                )
                if not innovation_requirement_failures(
                        partial_summary, args):
                    print(
                        "all requested M1/M2 requirements were met at "
                        f"step={completed_steps}; stopping coverage scan")
                    break

            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

            if args.max_steps > 0 and completed_steps >= args.max_steps:
                break

    if completed_steps == 0:
        raise RuntimeError(
            f"No train step was executed. seen_batches={seen_batches}. "
            "Try removing --require-full-history or changing --split/--skip-batches."
        )

    print("completed train-step loop")
    print(f"loss log: {log_path}")
    if args.checkpoint_every > 0:
        print(f"last checkpoint: {ckpt_dir / 'last.pt'}")
    else:
        print("successful-step checkpoint saving: disabled")
    if args.residual_diagnostics:
        diagnostic_summary = summarize_accumulated_samples(
            residual_samples,
            residual_gate_grad_norms,
            residual_encoder_grad_norms,
        )
        diagnostic_summary.update(
            {
                "completed_steps": completed_steps,
                "weights": args.weights,
                "weight_load_report": weight_load_report,
                "optimizer_step_applied": not args.no_optimizer_step,
                "dynamics_warmup_epoch": int(
                    getattr(cfg, "dynamics_warmup_epoch", 0)
                ),
            }
        )
        diagnostic_summary["configured_max_residual_norm"] = float(
            getattr(model, "dynamics_residual_scale", 0.0)
            * getattr(model, "dynamics_max_alpha", 0.0)
            * getattr(model, "dynamics_max_residual_norm", 0.0)
        )
        if hasattr(model, "dynamics_residual_gate"):
            diagnostic_summary["gate_output_bias"] = tensor_summary(
                model.dynamics_residual_gate.net[-1].bias
            )
        diagnostic_summary["motion_head_input_dim"] = int(
            model.motion_mlp[0].in_features
        )
        summary_path = Path(args.diagnostic_summary_file)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as summary_file:
            json.dump(diagnostic_summary, summary_file, ensure_ascii=False, indent=2)
        print(json.dumps(diagnostic_summary, ensure_ascii=False, indent=2))
        print(f"residual diagnostic summary: {summary_path}")

    if args.innovation_diagnostics:
        innovation_summary = summarize_innovation_run(
            innovation_run_diagnostics,
            innovation_run_gradients,
            completed_steps,
            optimizer_step_count,
            args.weights,
            weight_load_report,
        )
        innovation_summary.update({
            "dynamics_innovation_warmup_epoch": int(
                getattr(cfg, "dynamics_innovation_warmup_epoch", 0)),
            "physical_time_adapter_warmup_epoch": int(
                getattr(cfg, "physical_time_adapter_warmup_epoch", 0)),
            "tracked_resampling": bool(args.track_resampled),
            "requirements": {
                "invalid": bool(args.require_invalid),
                "empty": bool(args.require_empty),
                "resampled": bool(args.require_resampled),
                "nonzero_encoder_grad": bool(
                    args.require_nonzero_encoder_grad),
                "nonzero_adapter_grad": bool(
                    args.require_nonzero_adapter_grad),
                "min_optimizer_steps": int(
                    args.require_min_optimizer_steps),
                "zero_warmup_output": bool(
                    args.require_zero_warmup_output),
            },
        })
        requirement_failures = innovation_requirement_failures(
            innovation_summary, args)
        innovation_summary["requirement_failures"] = requirement_failures
        innovation_summary["requirements_passed"] = not requirement_failures
        summary_path = (
            Path(args.innovation_summary_file)
            if args.innovation_summary_file is not None
            else log_path.with_suffix(".summary.json")
        )
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as summary_file:
            json.dump(
                innovation_summary,
                summary_file,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            summary_file.write("\n")
        print(json.dumps(
            innovation_summary, ensure_ascii=False, indent=2,
            allow_nan=False))
        print(f"innovation diagnostic summary: {summary_path}")
        if requirement_failures:
            raise RuntimeError(
                "M1/M2 innovation requirements failed: "
                + "; ".join(requirement_failures))

    print("finished train-step check")


if __name__ == "__main__":
    main()
