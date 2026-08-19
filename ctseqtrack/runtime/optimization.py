"""Utilities for reproducible, independently checkpointed plugin training."""

import contextlib
from contextlib import contextmanager
import copy
import hashlib
import random
import time

import numpy as np
import torch
from torch import nn


@contextmanager
def freeze_batchnorm_running_stats(module):
    """Use stored BN statistics while retaining affine gradients."""
    states = []
    for child in module.modules():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            states.append((child, child.training))
            child.eval()
    try:
        yield
    finally:
        for child, training in states:
            child.train(training)


def _named_seed(seed, namespace):
    """Derive a stable torch seed without depending on Python's hash salt."""
    digest = hashlib.sha256(
        f"ct-seqtrack::{int(seed)}::{str(namespace)}".encode("utf-8")
    ).digest()
    # torch.manual_seed accepts signed 64-bit values.  Keeping the top bit
    # clear also makes the value portable across CPU/CUDA generators.
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


@contextmanager
def isolated_constructor_rng(seed, namespace):
    """Run one plugin constructor in a named RNG sub-domain.

    ``fork_rng`` restores both CPU and all initialized CUDA RNG streams.  A
    module therefore receives the same parameters whenever it is present,
    while enabling or disabling that module cannot perturb B0 or another
    plugin's initialization/training RNG stream.
    """
    devices = (
        list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    )
    derived_seed = _named_seed(seed, namespace)
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(derived_seed)
        yield derived_seed


def capture_global_rng_state():
    """Capture every process-global RNG used by the training/data path."""
    numpy_state = np.random.get_state()
    return {
        "schema": "ct_seqtrack.global_rng.v1",
        "python": random.getstate(),
        # Keep the checkpoint weights-only compatible: a raw NumPy ndarray
        # requires numpy._core pickle globals on newer torch.load defaults.
        "numpy": {
            "bit_generator": str(numpy_state[0]),
            "state": torch.as_tensor(numpy_state[1].astype(np.int64, copy=True)),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state().cpu().clone(),
        "torch_cuda": (
            [state.cpu().clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else []
        ),
    }


def restore_global_rng_state(state):
    """Restore a complete RNG snapshot, rejecting device-count drift."""
    if (
        not isinstance(state, dict)
        or state.get("schema") != "ct_seqtrack.global_rng.v1"
    ):
        raise ValueError("checkpoint lacks ct_seqtrack.global_rng.v1")
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            numpy_state["state"].cpu().numpy().astype(np.uint32, copy=True),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_states = list(state.get("torch_cuda", []))
    if cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA training RNG cannot be restored without CUDA")
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError("CUDA RNG device count changed across exact resume")
        torch.cuda.set_rng_state_all([item.cpu() for item in cuda_states])


def advance_lightning_manual_transaction(trainer):
    """Advance Lightning 2.0's manual loop once for one logical batch.

    The CT contract deliberately performs two *raw* optimizer transactions so
    that AMP overflow and schedulers remain isolated.  Raw optimizers bypass
    ``LightningOptimizer`` callbacks, so Lightning 2.0.2 would otherwise keep
    ``global_step`` at zero.  Advancing the loop progress once here gives the
    pair one logical transaction clock without double-counting the two native
    optimizer steps.
    """
    try:
        progress = trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress
    except AttributeError as exc:
        raise RuntimeError(
            "CT isolated optimization requires the Lightning 2.0 manual "
            "optimization progress interface"
        ) from exc
    progress.increment_ready()
    progress.increment_completed()


CAUSAL_CANDIDATE_WEIGHTS = {0: 0.5, 1: 0.3, 2: 0.2}


def causal_candidate_weight(candidate_id):
    """Return the registered three-role objective weight."""
    candidate_id = int(candidate_id)
    if candidate_id not in CAUSAL_CANDIDATE_WEIGHTS:
        raise ValueError("causal candidate id must be 0, 1 or 2")
    return CAUSAL_CANDIDATE_WEIGHTS[candidate_id]


def candidate_stratified_mean(values, valid, candidate_ids):
    """Average within each present role, then apply the causal role weights.

    Weights are normalized over roles present in this tensor.  A role-isolated
    microbatch therefore returns its ordinary masked mean; the outer optimizer
    transaction owns the registered 0.5/0.3/0.2 cross-role weighting.  A
    combined c0/c1/c2 tensor has weights summing to one and is exactly the
    paper-facing objective.
    """
    values = values.reshape(values.shape[0], -1)
    candidate_ids = candidate_ids.to(device=values.device).reshape(-1)
    if valid is None:
        valid = torch.ones_like(values)
    else:
        valid = valid.to(device=values.device, dtype=values.dtype).reshape(
            values.shape[0], -1
        )
        if valid.shape[1] == 1 and values.shape[1] != 1:
            valid = valid.expand_as(values)
    total = values.new_zeros(())
    total_branch_weight = 0.0
    for branch_id in torch.unique(candidate_ids, sorted=True).tolist():
        rows = candidate_ids == int(branch_id)
        branch_valid = valid[rows]
        numerator = (values[rows] * branch_valid).sum()
        denominator = branch_valid.sum().clamp_min(1.0)
        branch_weight = causal_candidate_weight(branch_id)
        total = total + branch_weight * numerator / denominator
        total_branch_weight += branch_weight
    return total / max(total_branch_weight, 1e-12)


class CheckpointableRNG(nn.Module):
    """A private CPU/CUDA RNG stream that restores the caller's stream."""

    def __init__(self, seed):
        super().__init__()
        self.seed = int(seed)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        self._cpu_state = generator.get_state().cpu()
        self._cuda_states = {}

    def get_extra_state(self):
        return {
            "seed": self.seed,
            "cpu_state": self._cpu_state.cpu(),
            "cuda_states": {
                str(key): value.cpu() for key, value in self._cuda_states.items()
            },
        }

    def set_extra_state(self, state):
        self.seed = int(state.get("seed", self.seed))
        self._cpu_state = state["cpu_state"].cpu().clone()
        self._cuda_states = {
            int(key): value.cpu().clone()
            for key, value in state.get("cuda_states", {}).items()
        }

    @contextmanager
    def fork(self, device):
        device = torch.device(device)
        cuda_index = None
        devices = []
        if device.type == "cuda":
            cuda_index = (
                torch.cuda.current_device()
                if device.index is None
                else int(device.index)
            )
            devices = [cuda_index]
        with torch.random.fork_rng(devices=devices):
            torch.set_rng_state(self._cpu_state.cpu())
            if cuda_index is not None:
                state = self._cuda_states.get(cuda_index)
                if state is None:
                    generator = torch.Generator(device=f"cuda:{cuda_index}")
                    generator.manual_seed(self.seed + 104729 * (cuda_index + 1))
                    state = generator.get_state().cpu()
                torch.cuda.set_rng_state(state, device=cuda_index)
            try:
                yield
            finally:
                self._cpu_state = torch.get_rng_state().cpu().clone()
                if cuda_index is not None:
                    self._cuda_states[cuda_index] = (
                        torch.cuda.get_rng_state(cuda_index).cpu().clone()
                    )


def capture_training_transaction_state(
    named_parameters,
    optimizer,
    scaler=None,
    scheduler=None,
    inputs=None,
    loss=None,
    clip_norm=None,
    clip_coefficient=None,
):
    """Clone all state needed by a B0 on/off equivalence audit."""
    parameters = {
        name: parameter.detach().cpu().clone() for name, parameter in named_parameters
    }
    gradients = {
        name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
        for name, parameter in named_parameters
    }
    return {
        "parameters": parameters,
        "gradients": gradients,
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "scaler": None if scaler is None else copy.deepcopy(scaler.state_dict()),
        "scheduler": (
            None if scheduler is None else copy.deepcopy(scheduler.state_dict())
        ),
        "cpu_rng": torch.get_rng_state().cpu().clone(),
        "cuda_rng": (
            [state.cpu().clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else []
        ),
        "inputs": copy.deepcopy(inputs),
        "loss": None if loss is None else torch.as_tensor(loss).detach().cpu().clone(),
        "clip_norm": (
            None
            if clip_norm is None
            else torch.as_tensor(clip_norm).detach().cpu().clone()
        ),
        "clip_coefficient": (
            None
            if clip_coefficient is None
            else torch.as_tensor(clip_coefficient).detach().cpu().clone()
        ),
    }


def assert_training_transaction_equal(first, second, path="transaction"):
    """Fail at the first unequal tensor/value in an equivalence audit."""
    if torch.is_tensor(first) or torch.is_tensor(second):
        if not (
            torch.is_tensor(first)
            and torch.is_tensor(second)
            and torch.equal(first, second)
        ):
            raise AssertionError(f"{path} differs")
        return
    if isinstance(first, dict) or isinstance(second, dict):
        if not (
            isinstance(first, dict)
            and isinstance(second, dict)
            and set(first) == set(second)
        ):
            raise AssertionError(f"{path} keys differ")
        for key in sorted(first, key=str):
            assert_training_transaction_equal(first[key], second[key], f"{path}.{key}")
        return
    if isinstance(first, (list, tuple)) or isinstance(second, (list, tuple)):
        if (
            not isinstance(first, (list, tuple))
            or not isinstance(second, (list, tuple))
            or len(first) != len(second)
        ):
            raise AssertionError(f"{path} sequence differs")
        for index, (left, right) in enumerate(zip(first, second)):
            assert_training_transaction_equal(left, right, f"{path}[{index}]")
        return
    if first != second:
        raise AssertionError(f"{path} differs: {first!r} != {second!r}")


def assert_disjoint_parameter_sets(first, second):
    first_ids = {id(parameter) for parameter in first}
    second_ids = {id(parameter) for parameter in second}
    overlap = first_ids & second_ids
    if overlap:
        raise RuntimeError(
            f"isolated optimizers share {len(overlap)} parameter tensors"
        )


# Formal v25 manual-optimization transaction.


def configure_isolated_optimizers(self):
    if not self.ct_separate_optimizers:
        raise RuntimeError("isolated optimizer mode is disabled")
    named = [
        (name, parameter)
        for name, parameter in self.named_parameters()
        if parameter.requires_grad
    ]
    b0_named = [item for item in named if not self._ct_any_plugin_parameter(item[0])]
    if not b0_named:
        raise RuntimeError("strict isolation requires non-empty B0")
    module_named = {"b0": b0_named}
    enabled = {
        "b1": self.ct_enable_b1,
        "b2": self.ct_enable_b2,
        "b3": self.ct_enable_b3,
    }
    for group_name in ("b1", "b2", "b3"):
        if (
            int(getattr(self.config, "ct_protocol_version", 24)) >= 25
            and enabled[group_name]
            and any(
                not parameter.requires_grad
                for name, parameter in self.named_parameters()
                if self._ct_any_plugin_parameter(name)
                if self._ct_plugin_group(name) == group_name
            )
        ):
            raise RuntimeError(f"v25 forbids frozen parameters in enabled {group_name}")
        group = [
            item
            for item in named
            if (
                self._ct_any_plugin_parameter(item[0])
                and self._ct_plugin_group(item[0]) == group_name
            )
        ]
        if enabled[group_name]:
            if not group:
                raise RuntimeError(f"strict isolation requires non-empty {group_name}")
            module_named[group_name] = group
    all_parameters = []
    for group in module_named.values():
        parameters = [parameter for _, parameter in group]
        assert_disjoint_parameter_sets(all_parameters, parameters)
        all_parameters.extend(parameters)
    self._ct_named_parameters_by_module = module_named
    self._ct_optimizer_names = list(module_named)
    self._ct_b0_named_parameters = module_named["b0"]
    self._ct_plugin_named_parameters = [
        item for name in ("b1", "b2", "b3") for item in module_named.get(name, [])
    ]
    if int(getattr(self.config, "ct_protocol_version", 24)) >= 25:
        frozen_b0 = [
            name
            for name, parameter in self.named_parameters()
            if (not self._ct_any_plugin_parameter(name) and not parameter.requires_grad)
        ]
        if frozen_b0:
            raise RuntimeError(
                "v25 forbids frozen B0 parameters: " + ", ".join(frozen_b0[:5])
            )
        self._ct_record_parameter_hash("initialization")
    plugin_lr = float(getattr(self.config, "ct_plugin_lr", self.config.lr))
    learning_rates = {
        name: float(
            getattr(
                self.config,
                f"ct_{name}_lr",
                self.config.lr if name == "b0" else plugin_lr,
            )
        )
        for name in module_named
    }
    optimizers = [
        self._build_isolated_optimizer(
            [parameter for _, parameter in module_named[name]], learning_rates[name]
        )
        for name in self._ct_optimizer_names
    ]
    if self.config.optimizer.lower() == "adamonecycle":
        if self.train_dataloader_length is None:
            raise ValueError("OneCycle isolated training needs train_dataloader_length")
        schedulers = [
            torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=float(
                    getattr(self.config, f"ct_{name}_max_lr", learning_rates[name])
                ),
                epochs=self.config.epoch,
                steps_per_epoch=self.train_dataloader_length,
            )
            for name, optimizer in zip(self._ct_optimizer_names, optimizers)
        ]
        interval = "step"
    else:
        schedulers = [
            torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.config.lr_decay_step,
                gamma=self.config.lr_decay_rate,
            )
            for optimizer in optimizers
        ]
        interval = "epoch"
    return optimizers, [
        {"scheduler": scheduler, "interval": interval, "name": f"scheduler_{name}"}
        for name, scheduler in zip(self._ct_optimizer_names, schedulers)
    ]


def ensure_ct_scalers(self):
    if self._ct_scalers:
        return
    names = list(getattr(self, "_ct_optimizer_names", ()))
    if not names:
        raise RuntimeError("isolated optimizer names are not configured")
    enabled = bool(self.ct_manual_amp_enabled and self.device.type == "cuda")
    for name in names:
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=enabled)
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler(enabled=enabled)
        self._ct_scalers[name] = scaler
    self._ct_b0_scaler = self._ct_scalers.get("b0")
    self._ct_plugin_scaler = self._ct_scalers.get("b2")
    pending = self._ct_pending_scaler_state
    if isinstance(pending, dict):
        for name, scaler in self._ct_scalers.items():
            state = pending.get(name)
            # Accept the old two-scaler checkpoint for diagnostic resume.
            if state is None and name != "b0":
                state = pending.get("plugin")
            if state is not None:
                scaler.load_state_dict(state)
    self._ct_pending_scaler_state = None


def record_acquisition_supply(self, loss_dict, population):
    totals_by_population = getattr(self, "_ct_epoch_acquisition_totals", None)
    if not isinstance(totals_by_population, dict):
        return
    totals = totals_by_population[population]
    mapping = {
        "eligible_rows": "ct_acquisition_eligible_row_count",
        "retained_rows": "ct_acquisition_retained_row_count",
        "pool_targets": "ct_acquisition_pool_target_sum",
        "sampled_targets": "ct_acquisition_sampled_target_sum",
        "available_rows": "ct_candidate_available_row_count",
        "role_satisfied_rows": ("ct_candidate_role_satisfied_row_count"),
        "boundary_ratio_sum": "ct_candidate_boundary_ratio_sum",
        "boundary_ratio_count": "ct_candidate_boundary_ratio_count",
        "support_truncated_rows": "ct_support_truncated_row_count",
        "support_volume_sum": "ct_support_volume_sum",
        "support_volume_count": "ct_support_volume_count",
        "recovery_positive_rows": "ct_recovery_positive_row_count",
        "recovery_fallback_rows": "ct_recovery_fallback_row_count",
    }
    for target, source in mapping.items():
        value = loss_dict.get(source)
        if value is not None:
            totals[target] += float(value.detach().cpu().item())


def isolated_optimizer_step(self, loss_dict, auxiliary_gradients=None):
    """Execute one disjoint transaction per active B0--B3 module."""
    self._ensure_ct_scalers()
    optimizers = self.optimizers(use_pl_optimizer=False)
    if not isinstance(optimizers, (list, tuple)):
        optimizers = [optimizers]
    names = list(self._ct_optimizer_names)
    if len(optimizers) != len(names):
        raise RuntimeError("optimizer/module cardinality mismatch")
    optimizer_map = dict(zip(names, optimizers))
    loss_key = {name: f"loss_{name}_transaction" for name in names}
    gradients_by_module = {}
    parameters_by_module = {}
    for name, optimizer in optimizer_map.items():
        optimizer.zero_grad(set_to_none=True)
        parameters = [
            parameter for _, parameter in self._ct_named_parameters_by_module[name]
        ]
        parameters_by_module[name] = parameters
        weight = (
            float(loss_dict.get("ct_canonical_candidate_weight", 1.0))
            if name in ("b1", "b2")
            else 1.0
        )
        scaled_loss = self._ct_scalers[name].scale(weight * loss_dict[loss_key[name]])
        gradients = torch.autograd.grad(
            scaled_loss, parameters, retain_graph=False, allow_unused=True
        )
        auxiliary = (
            auxiliary_gradients.get(name)
            if isinstance(auxiliary_gradients, dict)
            else None
        )
        if auxiliary is not None:
            if len(auxiliary) != len(gradients):
                raise RuntimeError(
                    f"auxiliary/{name.upper()} gradient cardinality mismatch"
                )
            gradients = tuple(
                (
                    extra
                    if canonical is None
                    else canonical if extra is None else canonical + extra
                )
                for canonical, extra in zip(gradients, auxiliary)
            )
        gradients_by_module[name] = gradients
        self._assign_parameter_gradients(parameters, gradients)

    norms = {}
    stepped = {}
    for name, optimizer in optimizer_map.items():
        scaler = self._ct_scalers[name]
        scaler.unscale_(optimizer)
        clip = float(
            getattr(
                self.config,
                f"ct_{name}_gradient_clip_val",
                getattr(
                    self.config,
                    "ct_plugin_gradient_clip_val",
                    getattr(self.config, "gradient_clip_val", 0.0),
                ),
            )
        )
        norms[name] = torch.nn.utils.clip_grad_norm_(
            parameters_by_module[name], max_norm=clip if clip > 0 else float("inf")
        )
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        stepped[name] = scaler.get_scale() >= scale_before
        if stepped[name]:
            getattr(self, f"ct_{name}_update_step").add_(1)
            setattr(self, f"_ct_{name}_updated_this_epoch", True)
    if stepped.get("b0"):
        self._ct_b0_updated_this_epoch = True
        b0_step = int(self.ct_b0_update_step.item())
        if int(getattr(self.config, "ct_protocol_version", 24)) >= 25 and b0_step in (
            1,
            100,
        ):
            self._ct_record_parameter_hash(f"step_{b0_step}")
    plugin_stepped = any(stepped.get(name, False) for name in ("b1", "b2", "b3"))
    if plugin_stepped:
        self.ct_plugin_update_step.add_(1)
        self._ct_plugin_updated_this_epoch = True

    if self.config.optimizer.lower() == "adamonecycle":
        schedulers = self.lr_schedulers()
        if not isinstance(schedulers, (list, tuple)):
            schedulers = [schedulers]
        for name, scheduler in zip(names, schedulers):
            if stepped[name]:
                scheduler.step()
    trainer = getattr(self, "_trainer", None)
    if trainer is not None:
        advance_lightning_manual_transaction(trainer)
    for name in names:
        loss_dict.update(
            {
                f"ct_{name}_unscaled_grad_norm": torch.as_tensor(
                    norms[name], device=self.device
                ),
                f"ct_{name}_amp_scale": torch.tensor(
                    float(self._ct_scalers[name].get_scale()), device=self.device
                ),
                f"ct_{name}_step_applied": torch.tensor(
                    float(stepped[name]), device=self.device
                ),
                f"ct_{name}_update_step": getattr(self, f"ct_{name}_update_step")
                .detach()
                .to(device=self.device, dtype=torch.float32),
            }
        )
    plugin_norms = [norms[name] for name in ("b1", "b2", "b3") if name in norms]
    loss_dict["ct_plugin_unscaled_grad_norm"] = (
        torch.stack(
            [torch.as_tensor(value, device=self.device) for value in plugin_norms]
        ).max()
        if plugin_norms
        else torch.zeros((), device=self.device)
    )
    loss_dict["ct_plugin_step_applied"] = torch.tensor(
        float(plugin_stepped), device=self.device
    )
    loss_dict["ct_plugin_update_step"] = self.ct_plugin_update_step.detach().to(
        device=self.device, dtype=torch.float32
    )
    self._ct_last_gradient_norm = {
        name: float(torch.as_tensor(value).detach().cpu())
        for name, value in norms.items()
    }


def auxiliary_microbatch_gradients(self, auxiliary_batch):
    """Accumulate isolated B1/B2 gradients for causal c1/c2 views."""
    self._ensure_ct_scalers()
    microbatch_size = int(getattr(self.config, "ct_auxiliary_microbatch_size", 16))
    candidate_ids = auxiliary_batch["candidate_id"].reshape(-1)
    expected_ids = (1, 2)
    candidate_weights = {
        candidate_id: causal_candidate_weight(candidate_id)
        for candidate_id in expected_ids
    }
    parameters = {}
    accumulated = {}
    module_names = tuple(
        name for name in ("b1", "b2") if self._ct_named_parameters_by_module.get(name)
    )
    if "b2" not in module_names:
        raise RuntimeError("causal auxiliary training requires active B2")
    for module_name in module_names:
        named = self._ct_named_parameters_by_module.get(module_name, [])
        parameters[module_name] = [parameter for _, parameter in named]
        accumulated[module_name] = [None for _ in named]
    losses = {name: [] for name in module_names}
    metrics = {}
    with self.ct_auxiliary_rng.fork(self.device):
        with freeze_batchnorm_running_stats(self):
            for candidate_id in expected_ids:
                row_mask = candidate_ids == candidate_id
                row_count = int(row_mask.sum().item())
                if row_count != microbatch_size:
                    raise RuntimeError(
                        "contract-v3 requires one complete 16-row "
                        f"auxiliary view; candidate{candidate_id} has "
                        f"{row_count} rows"
                    )
                candidate_batch = self._slice_batch_rows(auxiliary_batch, row_mask)
                candidate_output = self(candidate_batch)
                candidate_loss = self.compute_loss(candidate_batch, candidate_output)
                weight = candidate_weights[candidate_id]
                for module_index, module_name in enumerate(module_names):
                    transaction = candidate_loss[f"loss_{module_name}_transaction"]
                    scaled_loss = self._ct_scalers[module_name].scale(
                        weight * transaction
                    )
                    gradients = torch.autograd.grad(
                        scaled_loss,
                        parameters[module_name],
                        retain_graph=module_index < len(module_names) - 1,
                        allow_unused=True,
                    )
                    for index, gradient in enumerate(gradients):
                        if gradient is None:
                            continue
                        previous = accumulated[module_name][index]
                        accumulated[module_name][index] = (
                            gradient if previous is None else previous + gradient
                        )
                    losses[module_name].append((weight * transaction).detach())
                for key, value in candidate_loss.items():
                    if torch.is_tensor(value) and value.numel() == 1:
                        metrics.setdefault(key, []).append(
                            (candidate_id, value.detach())
                        )
    if any(len(values) != 2 for values in losses.values()):
        raise RuntimeError("causal contract requires candidates 1 and 2")
    aggregated = {
        key: (
            torch.stack([value for _, value in values]).sum()
            if key.endswith(("_count", "_sum"))
            else sum(
                candidate_weights[candidate_id] * value
                for candidate_id, value in values
            )
            / sum(candidate_weights.values())
        )
        for key, values in metrics.items()
    }
    eligible = aggregated["ct_acquisition_eligible_row_count"]
    retained = aggregated["ct_acquisition_retained_row_count"]
    pool_targets = aggregated["ct_acquisition_pool_target_sum"]
    sampled_targets = aggregated["ct_acquisition_sampled_target_sum"]
    aggregated["ct_acquisition_row_recall"] = retained / eligible.clamp_min(1.0)
    aggregated["ct_acquisition_point_recall"] = (
        sampled_targets / pool_targets.clamp_min(1.0)
    )
    aggregated["ct_acquisition_target_recall"] = aggregated["ct_acquisition_row_recall"]
    reference = losses["b2"][0]
    aggregated["loss_b1_auxiliary_weighted"] = (
        torch.stack(losses["b1"]).sum() if "b1" in losses else reference.new_zeros(())
    )
    aggregated["loss_ct_b2_auxiliary_weighted"] = torch.stack(losses["b2"]).sum()
    aggregated["loss_ct_plugin_total"] = (
        aggregated["loss_b1_auxiliary_weighted"]
        + aggregated["loss_ct_b2_auxiliary_weighted"]
    )
    return {name: tuple(values) for name, values in accumulated.items()}, aggregated


def ct_training_step(self, batch, batch_idx):
    """
    Args:
        batch: {
        "points": stack_frames, (B,N,3+9+1)
        "seg_label": stack_label,
        "box_label": np.append(this_gt_bb_transform.center, theta),
        "box_size": this_gt_bb_transform.wlh
    }
    Returns:

    """
    online_batch = (
        isinstance(batch, list)
        and batch
        and isinstance(batch[0], dict)
        and batch[0].get("online_recursive_raw", False)
    )
    if online_batch and self.device.type == "cuda":
        torch.cuda.synchronize(self.device)
    online_step_start = time.perf_counter() if online_batch else None
    auxiliary_batch = None
    auxiliary_gradients = None
    if online_batch:
        batch = self._prepare_online_recursive_batch(batch)
        if self.ct_joint_contract_version >= 3 and "candidate_id" in batch:
            candidate_ids = batch["candidate_id"].reshape(-1)
            canonical_rows = candidate_ids == 0
            auxiliary_rows = ~canonical_rows
            canonical_count = int(canonical_rows.sum().item())
            auxiliary_count = int(auxiliary_rows.sum().item())
            expected_canonical = int(getattr(self.config, "batch_size", 16))
            expected_auxiliary = expected_canonical * (
                int(getattr(self.config, "ct_recursive_candidate_views", 1)) - 1
            )
            if (
                canonical_count != expected_canonical
                or auxiliary_count != expected_auxiliary
            ):
                raise RuntimeError(
                    "contract-v3 batch must contain exactly "
                    f"{expected_canonical} candidate0 and "
                    f"{expected_auxiliary} auxiliary rows; observed "
                    f"{canonical_count}+{auxiliary_count}"
                )
            if bool(auxiliary_rows.any()):
                full_context = list(self._ct_online_batch_context)
                auxiliary_batch = self._slice_batch_rows(batch, auxiliary_rows)
                batch = self._slice_batch_rows(batch, canonical_rows)
                canonical_selector = canonical_rows.detach().cpu().tolist()
                self._ct_online_batch_context = [
                    item for item, keep in zip(full_context, canonical_selector) if keep
                ]
                if not bool((batch["candidate_id"] == 0).all()):
                    raise RuntimeError("B0 transaction contains an auxiliary candidate")
    amp_enabled = bool(
        self.ct_separate_optimizers
        and self.ct_manual_amp_enabled
        and self.device.type == "cuda"
    )
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if amp_enabled
        else contextlib.nullcontext()
    )
    with autocast_context:
        output = self(batch)
        if online_batch:
            self._attach_h3_shadow_labels(batch, output)
        loss_dict = self.compute_loss(batch, output)
        if online_batch:
            self._ct_record_acquisition_supply(loss_dict, "candidate0")
        if auxiliary_batch is not None:
            canonical_b1_transaction = loss_dict["loss_b1_transaction"]
            canonical_b2_transaction = loss_dict["loss_b2_transaction"]
            auxiliary_gradients, auxiliary_loss = (
                self._ct_auxiliary_microbatch_gradients(auxiliary_batch)
            )
            self._ct_record_acquisition_supply(auxiliary_loss, "auxiliary_train")
            canonical_b2 = loss_dict["loss_ct_b2_total"]
            combined_b1 = (
                causal_candidate_weight(0) * canonical_b1_transaction.detach()
                + auxiliary_loss["loss_b1_auxiliary_weighted"].detach()
            )
            combined_b2 = (
                causal_candidate_weight(0) * canonical_b2.detach()
                + auxiliary_loss["loss_ct_b2_auxiliary_weighted"].detach()
            )
            loss_dict["loss_total"] = (
                loss_dict["loss_total"]
                - canonical_b1_transaction
                - canonical_b2
                + combined_b1
                + combined_b2
            )
            loss_dict["loss_ct_b2_total"] = combined_b2
            loss_dict["loss_ct_plugin_total"] = (
                combined_b1 + combined_b2 + loss_dict["loss_ct_b3_total"]
            )
            loss_dict["loss_ct_b1_total"] = combined_b1
            loss_dict["loss_ct_b1_candidate0"] = canonical_b1_transaction
            loss_dict["loss_ct_b2_candidate0"] = canonical_b2
            loss_dict["loss_ct_b1_auxiliary"] = auxiliary_loss[
                "loss_b1_auxiliary_weighted"
            ]
            loss_dict["loss_ct_b2_auxiliary"] = auxiliary_loss[
                "loss_ct_b2_auxiliary_weighted"
            ]
            for key, value in auxiliary_loss.items():
                if key != "loss_ct_plugin_total":
                    loss_dict[f"{key}_auxiliary"] = value
            loss_dict["loss_b2_transaction"] = canonical_b2_transaction
            loss_dict["loss_b1_transaction"] = canonical_b1_transaction
            loss_dict["ct_canonical_candidate_weight"] = (
                canonical_b1_transaction.new_tensor(causal_candidate_weight(0))
            )
    self._accumulate_joint_binary_rows(batch, output)
    loss = loss_dict["loss_total"]
    if online_batch:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        online_step_ms = max((time.perf_counter() - online_step_start) * 1000.0, 1e-6)
        shadow_ms = (
            batch["ct_shadow_time_ms"].detach().to(device=loss.device, dtype=loss.dtype)
        )
        loss_dict["ct_online_step_time_ms"] = loss.new_tensor(online_step_ms)
        loss_dict["ct_shadow_step_latency_ratio"] = shadow_ms / online_step_ms

    if self.is_paired_batch(batch):
        metric_batch = batch["view_a"]
        metric_output = output["view_a"]
    else:
        metric_batch = batch
        metric_output = output

    # log
    log_batch_size = int(metric_batch["seg_label"].shape[0])
    seg_acc = self.seg_acc(
        torch.argmax(metric_output["seg_logits"], dim=1, keepdim=False),
        metric_batch["seg_label"],
    )
    self.log(
        "seg_acc_background/train",
        seg_acc[0],
        on_step=True,
        on_epoch=True,
        prog_bar=False,
        logger=True,
        batch_size=log_batch_size,
    )
    self.log(
        "seg_acc_foreground/train",
        seg_acc[1],
        on_step=True,
        on_epoch=True,
        prog_bar=False,
        logger=True,
        batch_size=log_batch_size,
    )
    if self.use_motion_cls:
        motion_acc = self.motion_acc(
            torch.argmax(metric_output["motion_cls"], dim=1, keepdim=False),
            metric_batch["motion_state_label"][:, 0],
        )  # 0 represents motion relative to the first historical box
        self.log(
            "motion_acc_static/train",
            motion_acc[0],
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=log_batch_size,
        )
        self.log(
            "motion_acc_dynamic/train",
            motion_acc[1],
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=log_batch_size,
        )

    log_dict = {k: v.item() for k, v in loss_dict.items()}

    self.logger.experiment.add_scalars("loss", log_dict, global_step=self.global_step)
    if (
        online_batch
        and "ct_router_gate" in output
        and hasattr(self.logger.experiment, "add_histogram")
    ):
        self.logger.experiment.add_histogram(
            "ct/router_probability_histogram",
            output["ct_router_gate"].detach(),
            global_step=self.global_step,
        )

    if online_batch:
        self._commit_online_recursive_predictions(output)

    if self.ct_separate_optimizers:
        self._ct_isolated_optimizer_step(loss_dict, auxiliary_gradients)
        return loss.detach()
    return loss
