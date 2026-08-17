"""Utilities for reproducible, independently checkpointed plugin training."""

from contextlib import contextmanager
import copy
import hashlib
import random

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
        list(range(torch.cuda.device_count()))
        if torch.cuda.is_available() else [])
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
            "state": torch.as_tensor(
                numpy_state[1].astype(np.int64, copy=True)),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state().cpu().clone(),
        "torch_cuda": (
            [state.cpu().clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available() else []),
    }


def restore_global_rng_state(state):
    """Restore a complete RNG snapshot, rejecting device-count drift."""
    if (not isinstance(state, dict)
            or state.get("schema") != "ct_seqtrack.global_rng.v1"):
        raise ValueError("checkpoint lacks ct_seqtrack.global_rng.v1")
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((
        str(numpy_state["bit_generator"]),
        numpy_state["state"].cpu().numpy().astype(np.uint32, copy=True),
        int(numpy_state["position"]),
        int(numpy_state["has_gauss"]),
        float(numpy_state["cached_gaussian"]),
    ))
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_states = list(state.get("torch_cuda", []))
    if cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA training RNG cannot be restored without CUDA")
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError(
                "CUDA RNG device count changed across exact resume")
        torch.cuda.set_rng_state_all([
            item.cpu() for item in cuda_states])


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
        progress = (
            trainer.fit_loop.epoch_loop.manual_optimization
            .optim_step_progress)
    except AttributeError as exc:
        raise RuntimeError(
            "CT isolated optimization requires the Lightning 2.0 manual "
            "optimization progress interface") from exc
    progress.increment_ready()
    progress.increment_completed()


def candidate_stratified_mean(values, valid, candidate_ids):
    """Normalize within each candidate, then apply the 1/2,1/6 contract."""
    values = values.reshape(values.shape[0], -1)
    candidate_ids = candidate_ids.to(device=values.device).reshape(-1)
    if valid is None:
        valid = torch.ones_like(values)
    else:
        valid = valid.to(device=values.device, dtype=values.dtype).reshape(
            values.shape[0], -1)
        if valid.shape[1] == 1 and values.shape[1] != 1:
            valid = valid.expand_as(values)
    total = values.new_zeros(())
    total_branch_weight = 0.0
    for branch_id in torch.unique(candidate_ids, sorted=True).tolist():
        rows = candidate_ids == int(branch_id)
        branch_valid = valid[rows]
        numerator = (values[rows] * branch_valid).sum()
        denominator = branch_valid.sum().clamp_min(1.0)
        branch_weight = 0.5 if int(branch_id) == 0 else 1.0 / 6.0
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
                str(key): value.cpu()
                for key, value in self._cuda_states.items()
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
                if device.index is None else int(device.index))
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
                    self._cuda_states[cuda_index] = torch.cuda.get_rng_state(
                        cuda_index).cpu().clone()


def capture_training_transaction_state(
        named_parameters, optimizer, scaler=None, scheduler=None,
        inputs=None, loss=None, clip_norm=None, clip_coefficient=None):
    """Clone all state needed by a B0 on/off equivalence audit."""
    parameters = {
        name: parameter.detach().cpu().clone()
        for name, parameter in named_parameters
    }
    gradients = {
        name: None if parameter.grad is None
        else parameter.grad.detach().cpu().clone()
        for name, parameter in named_parameters
    }
    return {
        "parameters": parameters,
        "gradients": gradients,
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "scaler": None if scaler is None else copy.deepcopy(
            scaler.state_dict()),
        "scheduler": None if scheduler is None else copy.deepcopy(
            scheduler.state_dict()),
        "cpu_rng": torch.get_rng_state().cpu().clone(),
        "cuda_rng": (
            [state.cpu().clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available() else []),
        "inputs": copy.deepcopy(inputs),
        "loss": None if loss is None else torch.as_tensor(
            loss).detach().cpu().clone(),
        "clip_norm": None if clip_norm is None else torch.as_tensor(
            clip_norm).detach().cpu().clone(),
        "clip_coefficient": (
            None if clip_coefficient is None else torch.as_tensor(
                clip_coefficient).detach().cpu().clone()),
    }


def assert_training_transaction_equal(first, second, path="transaction"):
    """Fail at the first unequal tensor/value in an equivalence audit."""
    if torch.is_tensor(first) or torch.is_tensor(second):
        if not (torch.is_tensor(first) and torch.is_tensor(second)
                and torch.equal(first, second)):
            raise AssertionError(f"{path} differs")
        return
    if isinstance(first, dict) or isinstance(second, dict):
        if not (isinstance(first, dict) and isinstance(second, dict)
                and set(first) == set(second)):
            raise AssertionError(f"{path} keys differ")
        for key in sorted(first, key=str):
            assert_training_transaction_equal(
                first[key], second[key], f"{path}.{key}")
        return
    if isinstance(first, (list, tuple)) or isinstance(second, (list, tuple)):
        if (not isinstance(first, (list, tuple))
                or not isinstance(second, (list, tuple))
                or len(first) != len(second)):
            raise AssertionError(f"{path} sequence differs")
        for index, (left, right) in enumerate(zip(first, second)):
            assert_training_transaction_equal(
                left, right, f"{path}[{index}]")
        return
    if first != second:
        raise AssertionError(f"{path} differs: {first!r} != {second!r}")


def assert_disjoint_parameter_sets(first, second):
    first_ids = {id(parameter) for parameter in first}
    second_ids = {id(parameter) for parameter in second}
    overlap = first_ids & second_ids
    if overlap:
        raise RuntimeError(
            f"isolated optimizers share {len(overlap)} parameter tensors")
