"""v35绝对分段LR：实际Adam顺序、完整epoch边界与Lightning恢复。"""
from copy import deepcopy
from pathlib import Path
import tempfile

import pytest
import torch

from models.ct_v31.config import config_identity, load_config, normalize_config
from models.ct_v31.lr_schedule import AbsolutePiecewiseLR
from models.ctseqtrackv31 import CTSEQTRACKV31
from tests.test_ct_v34_resume import _assert_identical


STAGES = [5e-5, 1e-5, 5e-6]
CONFIG = 'cfgs/ct_seqtrack/35_b0_w_piecewise_lr_mini.yaml'


def test_piecewise_recipe_has_distinct_identity_and_no_old_field_injection():
    new = load_config(CONFIG)
    old = load_config('cfgs/ct_seqtrack/35_b0_w_half_lr_mini.yaml')
    ignored = {'cfg', 'tag', 'experiment_name', 'lr_schedule', 'lr_stage_values'}
    assert {k: v for k, v in new.items() if k not in ignored} == {
        k: v for k, v in old.items() if k not in ignored}
    assert new.lr_stage_values == STAGES and new.lr_milestones == [20, 50]
    assert config_identity(new) != config_identity(old)
    seed52 = load_config('cfgs/ct_seqtrack/35_b0_w_piecewise_lr_mini_seed52.yaml')
    assert seed52.seed == 52 and seed52.ct_partition_seed == 42
    assert config_identity(seed52) != config_identity(new)
    for name in ('quarter', 'half', 'normal', 'scaled'):
        for suffix in ('', '_seed52'):
            assert 'lr_stage_values' not in load_config(
                f'cfgs/ct_seqtrack/35_b0_w_{name}_lr_mini{suffix}.yaml')
    assert 'lr_stage_values' not in normalize_config()
    with pytest.raises(ValueError, match='unknown'):
        normalize_config(dict(old, lr_stage_values=STAGES))


@pytest.mark.parametrize('update', [
    dict(lr_stage_values=[5e-5, 1e-5]), dict(lr_stage_values=[5e-5, float('nan'), 5e-6]),
    dict(lr_stage_values=[5e-5, 0., 5e-6]), dict(lr_stage_values=[True, 1e-5, 5e-6]),
    dict(lr_stage_values=[5e-5, '0.00001', 5e-6]), dict(lr=1e-4),
    dict(lr_stage_values=[5e-5, 2e-5, 5e-6]), dict(lr_milestones=[19, 50]),
    dict(lr_warmup_steps=1), dict(seed=43),
    dict(net_model='ctseqtrackv34', experiment_family='ct_seqtrack_v34')])
def test_only_registered_formal_piecewise_recipe_is_accepted(update):
    with pytest.raises(ValueError):
        load_config(CONFIG, update)


def _configured(config):
    host = CTSEQTRACKV31(config, tracker=torch.nn.Linear(1, 1))
    value = host.configure_optimizers()
    assert value['lr_scheduler']['interval'] == 'epoch'
    return host, value['optimizer'], value['lr_scheduler']['scheduler']


def _optimize_epoch(model, optimizer, scheduler):
    # 不同epoch都实际更新参数；scheduler只在本epoch的所有Adam之后前进一步。
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        model.tracker(torch.ones(2, 1)).square().mean().backward()
        optimizer.step()
    scheduler.step()


def test_absolute_schedule_uses_exact_values_for_all_60_epochs_and_resumes_boundaries():
    config = load_config(CONFIG)
    host, optimizer, scheduler = _configured(config)
    assert isinstance(scheduler, AbsolutePiecewiseLR) and scheduler.last_epoch == 0
    checkpoints = {}
    used = []
    for epoch in range(60):
        used.append(optimizer.param_groups[0]['lr'])
        _optimize_epoch(host, optimizer, scheduler)
        assert scheduler.last_epoch == epoch + 1
        if epoch + 1 in (19, 20, 49, 50):
            checkpoints[epoch + 1] = deepcopy((host.state_dict(), optimizer.state_dict(), scheduler.state_dict()))
    assert used == [5e-5] * 20 + [1e-5] * 30 + [5e-6] * 10
    expected = (host.state_dict(), optimizer.state_dict(), scheduler.state_dict())
    for completed, state in checkpoints.items():
        restored, restored_optimizer, restored_scheduler = _configured(config)
        restored.load_state_dict(state[0])
        restored_optimizer.load_state_dict(state[1])
        restored_scheduler.load_state_dict(state[2])
        assert restored_optimizer.param_groups[0]['lr'] == used[completed]
        for epoch in range(completed, 60):
            assert restored_optimizer.param_groups[0]['lr'] == used[epoch]
            _optimize_epoch(restored, restored_optimizer, restored_scheduler)
        _assert_identical(expected, (restored.state_dict(), restored_optimizer.state_dict(),
                                     restored_scheduler.state_dict()))


@pytest.mark.parametrize('recipe', ['quarter', 'half', 'normal', 'scaled'])
def test_old_four_restore_native_multistep_adam_state_without_changing_lr(recipe):
    config = load_config(f'cfgs/ct_seqtrack/35_b0_w_{recipe}_lr_mini.yaml')
    old = torch.nn.Linear(1, 1)
    # 独立构造更新前使用的原生调度器，生成旧格式checkpoint。
    old_optimizer = torch.optim.Adam(old.parameters(), lr=config.lr, weight_decay=0.,
                                    betas=(.5, .999), eps=1e-6, foreach=False, fused=False)
    old_scheduler = torch.optim.lr_scheduler.MultiStepLR(old_optimizer, [20, 50], gamma=.1)
    restored, optimizer, scheduler = _configured(config)
    assert type(scheduler) is torch.optim.lr_scheduler.MultiStepLR
    for epoch in range(60):
        if epoch == 20:
            restored.tracker.load_state_dict(deepcopy(old.state_dict()))
            optimizer.load_state_dict(deepcopy(old_optimizer.state_dict()))
            scheduler.load_state_dict(deepcopy(old_scheduler.state_dict()))
        old_optimizer.zero_grad(set_to_none=True)
        old(torch.ones(2, 1)).square().mean().backward()
        old_optimizer.step()
        old_scheduler.step()
        if epoch >= 20:
            optimizer.zero_grad(set_to_none=True)
            restored.tracker(torch.ones(2, 1)).square().mean().backward()
            optimizer.step()
            scheduler.step()
            _assert_identical((old.state_dict(), old_optimizer.state_dict(), old_scheduler.state_dict()),
                (restored.tracker.state_dict(), optimizer.state_dict(), scheduler.state_dict()))


def test_real_lightning_piecewise_epoch20_and50_restore_matches_uninterrupted():
    pl = pytest.importorskip('pytorch_lightning')
    from pytorch_lightning.callbacks import Callback
    from models.ct_v31.data import build_loaders
    from models.ct_v31.runtime import capture_rng_state, validate_resume_payload
    from tests.test_ct_v31_runtime import TinyJointTracker, TinySource
    from utils.lightning_runtime import FinalWindowCheckpoint

    class CaptureLR(Callback):
        def __init__(self):
            self.rows = []

        def on_train_batch_start(self, trainer, module, batch, batch_idx):
            self.rows.append((trainer.current_epoch + 1, trainer.global_step,
                              trainer.optimizers[0].param_groups[0]['lr']))

    class StopAfter(Callback):
        def __init__(self, completed):
            self.completed = completed

        def on_train_epoch_end(self, trainer, module):
            if trainer.current_epoch + 1 == self.completed:
                trainer.should_stop = True

    artifact = Path(__file__).resolve().parents[1] / 'artifacts' / 'ct_checks'
    root = Path(tempfile.mkdtemp(prefix='v35_piecewise_lightning_', dir=artifact))

    def make(directory, stop=None):
        config = load_config(CONFIG, dict(ct_engineering_check=True, workers=0, batch_size=4,
            point_sample_size=8, log_dir=str(directory), v35_train_diagnostics=False))
        loaders = build_loaders(config, roles=('train',), sources={'train': TinySource((2,))})
        model = CTSEQTRACKV31(config, tracker=TinyJointTracker(), loaders=loaders)
        callbacks = [CaptureLR(), FinalWindowCheckpoint(keep=3, every_n_epochs=10)]
        if stop is not None:
            callbacks.append(StopAfter(stop))
        trainer = pl.Trainer(default_root_dir=str(directory), accelerator='cpu', devices=1,
            max_epochs=60, logger=False, callbacks=callbacks, enable_progress_bar=False,
            enable_model_summary=False, num_sanity_val_steps=0, limit_val_batches=0,
            reload_dataloaders_every_n_epochs=1)
        return model, trainer

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        pl.seed_everything(42, workers=True)
        uninterrupted, trainer = make(root / 'uninterrupted')
        trainer.fit(uninterrupted)
        expected = deepcopy((uninterrupted.state_dict(), trainer.optimizers[0].state_dict(),
                             trainer.lr_scheduler_configs[0].scheduler.state_dict(), capture_rng_state()))
        expected_trace = next(cb.rows for cb in trainer.callbacks if isinstance(cb, CaptureLR))
        assert [row[2] for row in expected_trace] == [5e-5] * 20 + [1e-5] * 30 + [5e-6] * 10
        assert trainer.global_step == 60  # 每轮只有一个真实host事务，绝非71,700步训练。

        pl.seed_everything(42, workers=True)
        checkpoint = None
        trace = []
        for stop in (20, 50, None):
            resumed, continuation = make(root / 'resumed', stop)
            continuation.fit(resumed, ckpt_path=None if checkpoint is None else str(checkpoint))
            trace.extend(next(cb.rows for cb in continuation.callbacks if isinstance(cb, CaptureLR)))
            completed = stop if stop is not None else 60
            checkpoint = root / 'resumed' / 'formal_checkpoints' / f'epoch={completed:03d}.ckpt'
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            payload = validate_resume_payload(saved['ct_v35_runtime'], resumed.config)
            assert payload['completed_epoch'] == completed and payload['epoch_complete']
            assert saved['lr_schedulers'][0]['last_epoch'] == completed
            assert saved['optimizer_states'][0]['param_groups'][0]['lr'] == (
                1e-5 if completed == 20 else 5e-6)
        assert trace == expected_trace
        assert continuation.global_step == 60
        _assert_identical(expected, (resumed.state_dict(), continuation.optimizers[0].state_dict(),
            continuation.lr_scheduler_configs[0].scheduler.state_dict(), capture_rng_state()))
    finally:
        torch.set_num_threads(previous_threads)
