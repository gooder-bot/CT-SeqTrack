"""Real Lightning-2.0.2 integration checks for the v23 logical clock.

The module is skipped only when the optional training dependency is absent;
the formal GPU environment pins pytorch-lightning==2.0.2 and must run it.
"""

import copy
import random

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

pl = pytest.importorskip("pytorch_lightning")

from utils.training_isolation import (  # noqa: E402
    CheckpointableRNG,
    advance_lightning_manual_transaction,
    capture_global_rng_state,
    restore_global_rng_state,
)


class _DualOptimizerModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.automatic_optimization = False
        self.b0 = torch.nn.Parameter(torch.tensor(1.0))
        self.plugin = torch.nn.Parameter(torch.tensor(2.0))
        self.private_rng = CheckpointableRNG(991)
        self.register_buffer(
            "b0_updates", torch.zeros((), dtype=torch.long))
        self.register_buffer(
            "plugin_updates", torch.zeros((), dtype=torch.long))

    def configure_optimizers(self):
        optimizers = [
            torch.optim.Adam([self.b0], lr=1e-3),
            torch.optim.Adam([self.plugin], lr=1e-3),
        ]
        schedulers = [
            torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=1, gamma=0.7)
            for optimizer in optimizers
        ]
        return optimizers, schedulers

    def training_step(self, batch, batch_idx):
        b0_optimizer, plugin_optimizer = self.optimizers(
            use_pl_optimizer=False)
        b0_optimizer.zero_grad(set_to_none=True)
        plugin_optimizer.zero_grad(set_to_none=True)
        with self.private_rng.fork(self.device):
            private_noise = torch.rand((), device=self.device)
        noise = (
            private_noise
            + torch.rand((), device=self.device)
            + float(np.random.random())
            + float(random.random()))
        b0_loss = (self.b0 - noise * 0.01).square()
        plugin_loss = (self.plugin - (0.5 + noise * 0.01)).square()
        self.b0.grad = torch.autograd.grad(b0_loss, self.b0)[0]
        self.plugin.grad = torch.autograd.grad(
            plugin_loss, self.plugin)[0]
        b0_optimizer.step()
        plugin_optimizer.step()
        self.b0_updates.add_(1)
        self.plugin_updates.add_(1)
        advance_lightning_manual_transaction(self.trainer)
        return (b0_loss + plugin_loss).detach()

    def on_train_epoch_end(self):
        for scheduler in self.lr_schedulers():
            scheduler.step()

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ct_global_rng_state"] = capture_global_rng_state()

    def on_load_checkpoint(self, checkpoint):
        self._pending_rng_state = copy.deepcopy(
            checkpoint.get("ct_global_rng_state"))

    def on_train_epoch_start(self):
        state = getattr(self, "_pending_rng_state", None)
        if state is not None:
            restore_global_rng_state(state)
            self._pending_rng_state = None


def _loader(rows):
    return DataLoader(
        TensorDataset(torch.arange(rows, dtype=torch.float32)),
        batch_size=1, shuffle=False)


def _trainer(epochs):
    return pl.Trainer(
        max_epochs=epochs, accelerator="cpu", devices=1,
        logger=False, enable_checkpointing=False,
        enable_model_summary=False, enable_progress_bar=False,
        num_sanity_val_steps=0)


def _seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _snapshot(model, trainer):
    return {
        "model": copy.deepcopy(model.state_dict()),
        "optimizers": copy.deepcopy([
            optimizer.state_dict() for optimizer in trainer.optimizers]),
        "schedulers": copy.deepcopy([
            item.scheduler.state_dict()
            for item in trainer.lr_scheduler_configs]),
        "global_step": trainer.global_step,
    }


def _assert_nested_equal(left, right):
    if torch.is_tensor(left) or torch.is_tensor(right):
        assert torch.is_tensor(left) and torch.is_tensor(right)
        assert torch.equal(left, right)
    elif isinstance(left, dict) or isinstance(right, dict):
        assert isinstance(left, dict) and isinstance(right, dict)
        assert set(left) == set(right)
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        assert isinstance(left, (list, tuple))
        assert isinstance(right, (list, tuple))
        assert len(left) == len(right)
        for first, second in zip(left, right):
            _assert_nested_equal(first, second)
    else:
        assert left == right


def test_real_lightning_two_native_optimizers_advance_one_logical_step():
    assert pl.__version__ == "2.0.2"
    model = _DualOptimizerModule()
    trainer = _trainer(1)
    trainer.fit(model, _loader(100))
    assert trainer.global_step == 100
    assert int(model.b0_updates) == 100
    assert int(model.plugin_updates) == 100


def test_real_lightning_epoch_resume_is_transaction_exact(tmp_path):
    assert pl.__version__ == "2.0.2"
    _seed_all(17)
    continuous = _DualOptimizerModule()
    continuous_trainer = _trainer(2)
    continuous_trainer.fit(continuous, _loader(50))
    expected = _snapshot(continuous, continuous_trainer)

    _seed_all(17)
    first_epoch = _DualOptimizerModule()
    first_trainer = _trainer(1)
    first_trainer.fit(first_epoch, _loader(50))
    checkpoint = tmp_path / "epoch_boundary.ckpt"
    first_trainer.save_checkpoint(checkpoint)

    _seed_all(9999)
    resumed = _DualOptimizerModule.load_from_checkpoint(checkpoint)
    resumed_trainer = _trainer(2)
    resumed_trainer.fit(
        resumed, _loader(50), ckpt_path=checkpoint)
    actual = _snapshot(resumed, resumed_trainer)
    _assert_nested_equal(expected, actual)
    assert actual["global_step"] == 100


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA AMP overflow smoke")
@pytest.mark.parametrize("overflow_side", ["b0", "plugin"])
def test_independent_cuda_scalers_isolate_single_side_overflow(overflow_side):
    b0 = torch.nn.Parameter(torch.tensor(1.0, device="cuda"))
    plugin = torch.nn.Parameter(torch.tensor(2.0, device="cuda"))
    b0_optimizer = torch.optim.Adam([b0], lr=1e-2)
    plugin_optimizer = torch.optim.Adam([plugin], lr=1e-2)
    try:
        b0_scaler = torch.amp.GradScaler("cuda")
        plugin_scaler = torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):
        b0_scaler = torch.cuda.amp.GradScaler()
        plugin_scaler = torch.cuda.amp.GradScaler()
    b0_before = b0.detach().clone()
    plugin_before = plugin.detach().clone()
    b0_loss = b0.square()
    plugin_loss = plugin.square()
    if overflow_side == "b0":
        b0_loss = b0_loss * torch.tensor(float("inf"), device="cuda")
    else:
        plugin_loss = plugin_loss * torch.tensor(
            float("inf"), device="cuda")
    b0_scaler.scale(b0_loss).backward()
    plugin_scaler.scale(plugin_loss).backward()
    b0_scaler.step(b0_optimizer)
    plugin_scaler.step(plugin_optimizer)
    b0_scaler.update()
    plugin_scaler.update()
    assert torch.equal(b0, b0_before) == (overflow_side == "b0")
    assert torch.equal(plugin, plugin_before) == (overflow_side == "plugin")
