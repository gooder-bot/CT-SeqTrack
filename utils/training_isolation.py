"""Utilities for reproducible, independently checkpointed plugin training."""

from contextlib import contextmanager
import copy

import torch
from torch import nn


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
