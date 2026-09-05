"""Actual Lightning epoch-boundary save/resume, without point-cloud kernels."""
import copy
import random

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

pl = pytest.importorskip('pytorch_lightning')
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from utils.dual_stream import DualStreamLoader
from utils.lightning_runtime import DataLoaderGeneratorState, FinalWindowCheckpoint
from utils.online_contract import build_online_resume_contract, validate_online_resume_contract
from utils.training_isolation import capture_global_rng_state, restore_global_rng_state


class StopAfterFirst(Callback):
    def __init__(self, completed_epoch=1):
        self.completed_epoch = int(completed_epoch)

    def on_train_epoch_end(self, trainer, module):
        trainer.should_stop = trainer.current_epoch + 1 == self.completed_epoch


class ResumeProbe(pl.LightningModule):
    """Four enabled parameter groups exercise the actual automatic Trainer path."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        self.branches = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(())) for _ in range(4)])
        self.register_buffer('epochs_finished', torch.zeros((), dtype=torch.long))
        self.boundary = False
        self.seen = []
        self.generator_starts = []
        self.validation_epochs = []
        self.checkpoint_events = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([{'params': [p], 'name': name} for name, p in
                                     zip(('b0', 'b1', 'b2', 'b3'), self.branches)],
                                    lr=.001, betas=(.5, .999), eps=1e-6)
        return {'optimizer': optimizer, 'lr_scheduler': {
            'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.7),
            'interval': 'epoch'}}

    def training_step(self, batch, index):
        if isinstance(batch, dict):
            batch = batch['observation']
        value = float(batch[0].sum())
        self.seen.append((int(self.current_epoch), value))
        noise = torch.rand(()) + np.random.random() + random.random()
        return sum((p - (value + noise) * .01).square() for p in self.branches)

    def validation_step(self, batch, index):
        # Validation may consume RNG; save must occur after its final state.
        self.validation_epochs.append(int(self.current_epoch) + 1)
        torch.rand(())
        np.random.random()
        random.random()

    def on_train_epoch_start(self):
        callback = next(cb for cb in self.trainer.callbacks if isinstance(cb, DataLoaderGeneratorState))
        self.generator_starts.append({key: generator.get_state() for key, generator in callback.generators.items()})
        if getattr(self, 'pending_rng', None) is not None:
            restore_global_rng_state(self.pending_rng)
            self.pending_rng = None
        self.boundary = False
        if hasattr(self.trainer.train_dataloader, 'set_epoch'):
            self.trainer.train_dataloader.set_epoch(int(self.current_epoch))

    def on_train_epoch_end(self):
        self.epochs_finished.add_(1)
        self.boundary = True

    def on_save_checkpoint(self, checkpoint):
        self.checkpoint_events.append((int(self.current_epoch) + 1, self.boundary))
        checkpoint['ct_online_resume_contract'] = build_online_resume_contract(self.config)
        checkpoint['ct_global_rng_state'] = capture_global_rng_state()
        checkpoint['ct_epoch_boundary_complete'] = self.boundary

    def on_load_checkpoint(self, checkpoint):
        self.pending_rng = copy.deepcopy(checkpoint['ct_global_rng_state'])


def _seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _resources(dual, rows=8):
    generators = {key: torch.Generator().manual_seed(seed) for key, seed in
                  (('observation', 71), ('validation', 72), ('mechanism', 73)) if dual or key != 'mechanism'}
    data = TensorDataset(torch.arange(rows, dtype=torch.float32))
    loader = DataLoader(data, batch_size=2, shuffle=True, generator=generators['observation'])
    if dual:
        mechanism = DataLoader(data, batch_size=2, generator=generators['mechanism'])
        loader = DualStreamLoader(loader, mechanism, schema='ct_seqtrack.train.v4', isolate_mechanism_rng=True)
    validation = DataLoader(data, batch_size=8, generator=generators['validation'])
    return loader, validation, generators


def _trainer(directory, generators, stop=False, max_epochs=3, val_interval=1,
             explicit_epoch_save=True):
    callbacks = [DataLoaderGeneratorState(**generators), FinalWindowCheckpoint(keep=3),
                 ModelCheckpoint(dirpath=directory / 'checkpoints', save_top_k=0, save_last=True,
                                 save_on_train_epoch_end=True if explicit_epoch_save else None)]
    if stop:
        callbacks.append(StopAfterFirst(completed_epoch=int(stop)))
    return pl.Trainer(max_epochs=max_epochs, accelerator='cpu', devices=1,
        check_val_every_n_epoch=val_interval,
        callbacks=callbacks, default_root_dir=str(directory), logger=False,
        enable_model_summary=False, enable_progress_bar=False, num_sanity_val_steps=0)


def _equal(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for key in a:
            _equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize('dual', [False, True])
@pytest.mark.parametrize('checkpoint_name', ['formal_checkpoints/epoch=001.ckpt', 'checkpoints/last.ckpt'])
def test_real_epoch_checkpoint_resume_matches_continuous(tmp_path, dual, checkpoint_name):
    assert pl.__version__ == '2.0.2'
    config = dict(ct_enable_v27=True, ct_runtime_protocol='safe_seqtrack_auto_v1', epoch=3,
                  ct_training_topology='dual_stream' if dual else 'single', ct_enable_b1=dual,
                  ct_enable_b2=dual, ct_enable_b3=dual)
    _seed(42)
    loader, val, generators = _resources(dual)
    continuous = ResumeProbe(config)
    continuous_trainer = _trainer(tmp_path / 'continuous', generators)
    continuous_trainer.fit(continuous, loader, val)
    expected = copy.deepcopy(continuous.state_dict())
    expected_optimizer = copy.deepcopy(continuous_trainer.optimizers[0].state_dict())
    expected_scheduler = copy.deepcopy(continuous_trainer.lr_scheduler_configs[0].scheduler.state_dict())
    expected_rng = capture_global_rng_state()
    expected_generators = {key: value.get_state() for key, value in generators.items()}

    _seed(42)
    loader, val, generators = _resources(dual)
    first = ResumeProbe(config)
    first_trainer = _trainer(tmp_path / 'split', generators, stop=True)
    first_trainer.fit(first, loader, val)
    path = tmp_path / 'split' / checkpoint_name
    payload = torch.load(path, weights_only=False)
    assert payload['ct_epoch_boundary_complete'] is True
    assert payload['state_dict']['epochs_finished'] == 1
    assert payload['lr_schedulers'][0]['last_epoch'] == 1
    validate_online_resume_contract(payload, config)
    _seed(9999)
    loader, val, generators = _resources(dual)
    resumed = ResumeProbe.load_from_checkpoint(path, config=config)
    trainer = _trainer(tmp_path / 'split', generators)
    trainer.fit(resumed, loader, val, ckpt_path=path)
    for epoch_index, (left, right) in enumerate(zip(continuous.generator_starts, first.generator_starts + resumed.generator_starts)):
        for role in left:
            assert torch.equal(left[role], right[role]), (epoch_index, role, left[role][:20], right[role][:20])
    assert continuous.seen == first.seen + resumed.seen
    _equal(expected_rng, capture_global_rng_state())
    _equal(expected_generators, {key: value.get_state() for key, value in generators.items()})
    _equal(expected, resumed.state_dict())
    _equal(expected_optimizer, trainer.optimizers[0].state_dict())
    _equal(expected_scheduler, trainer.lr_scheduler_configs[0].scheduler.state_dict())
    assert trainer.global_step == continuous_trainer.global_step == 12


def test_default_interval_five_checkpoint_precedes_completed_epoch_boundary(tmp_path):
    """Reproduce why v27 must explicitly save_last at train epoch end."""
    config = dict(ct_enable_v27=True, epoch=5, check_val_every_n_epoch=5)
    loader, val, generators = _resources(False, rows=2)
    model = ResumeProbe(config)
    trainer = _trainer(tmp_path, generators, max_epochs=5, val_interval=5,
                       explicit_epoch_save=False)
    trainer.fit(model, loader, val)
    payload = torch.load(tmp_path / 'checkpoints/last.ckpt', weights_only=False)
    assert model.validation_epochs == [5]
    assert payload['epoch'] == 4
    assert payload['state_dict']['epochs_finished'] == 4
    assert payload['ct_epoch_boundary_complete'] is False
    with pytest.raises(ValueError, match='epoch boundary'):
        validate_online_resume_contract(payload, config)


@pytest.mark.parametrize('dual', [False, True])
@pytest.mark.parametrize('checkpoint_name', ['formal_checkpoints/epoch=058.ckpt', 'checkpoints/last.ckpt'])
def test_interval_five_saves_each_epoch_and_resumes_late_three(tmp_path, dual, checkpoint_name):
    config = dict(ct_enable_v27=True, ct_runtime_protocol='safe_seqtrack_auto_v1',
                  ct_training_topology='dual_stream' if dual else 'single', ct_enable_b1=dual,
                  ct_enable_b2=dual, ct_enable_b3=dual, epoch=60,
                  check_val_every_n_epoch=5)
    _seed(42)
    loader, val, generators = _resources(dual, rows=2)
    continuous = ResumeProbe(config)
    continuous_trainer = _trainer(tmp_path / 'continuous', generators, max_epochs=60, val_interval=5)
    continuous_trainer.fit(continuous, loader, val)
    expected = copy.deepcopy(continuous.state_dict())
    expected_optimizer = copy.deepcopy(continuous_trainer.optimizers[0].state_dict())
    expected_scheduler = copy.deepcopy(continuous_trainer.lr_scheduler_configs[0].scheduler.state_dict())
    expected_rng = capture_global_rng_state()
    expected_generators = {key: value.get_state() for key, value in generators.items()}
    assert continuous.validation_epochs == list(range(5, 61, 5))
    assert {epoch for epoch, _ in continuous.checkpoint_events} == set(range(1, 61))
    assert all(boundary for _, boundary in continuous.checkpoint_events)
    assert sorted(path.name for path in (tmp_path / 'continuous/formal_checkpoints').glob('*.ckpt')) == [
        'epoch=058.ckpt', 'epoch=059.ckpt', 'epoch=060.ckpt']
    for epoch in (58, 59, 60):
        payload = torch.load(tmp_path / f'continuous/formal_checkpoints/epoch={epoch:03d}.ckpt', weights_only=False)
        assert payload['ct_epoch_boundary_complete'] is True
        assert payload['state_dict']['epochs_finished'] == epoch
        assert payload['lr_schedulers'][0]['last_epoch'] == epoch

    _seed(42)
    loader, val, generators = _resources(dual, rows=2)
    first = ResumeProbe(config)
    trainer = _trainer(tmp_path / 'split', generators, stop=58, max_epochs=60, val_interval=5)
    trainer.fit(first, loader, val)
    path = tmp_path / 'split' / checkpoint_name
    payload = torch.load(path, weights_only=False)
    assert payload['epoch'] == 57 and payload['ct_epoch_boundary_complete'] is True
    assert payload['callbacks']['ct_seqtrack.DataLoaderGeneratorState.v1']['validation_setup_complete'] is True
    validate_online_resume_contract(payload, config)

    _seed(9999)
    loader, val, generators = _resources(dual, rows=2)
    resumed = ResumeProbe.load_from_checkpoint(path, config=config)
    trainer = _trainer(tmp_path / 'split', generators, max_epochs=60, val_interval=5)
    trainer.fit(resumed, loader, val, ckpt_path=path)
    assert continuous.seen == first.seen + resumed.seen
    assert continuous.validation_epochs == first.validation_epochs + resumed.validation_epochs
    _equal(expected, resumed.state_dict())
    _equal(expected_optimizer, trainer.optimizers[0].state_dict())
    _equal(expected_scheduler, trainer.lr_scheduler_configs[0].scheduler.state_dict())
    _equal(expected_rng, capture_global_rng_state())
    _equal(expected_generators, {key: value.get_state() for key, value in generators.items()})
    assert trainer.global_step == continuous_trainer.global_step == 60


@pytest.mark.parametrize('dual', [False, True])
@pytest.mark.parametrize('stop_epoch', [1, 4])
def test_interval_five_resume_before_first_validation_preserves_initial_setup(tmp_path, dual, stop_epoch):
    """Keep first-ever setup at epoch 5, including setup before epoch-start."""
    config = dict(ct_enable_v27=True, epoch=11, check_val_every_n_epoch=5,
                  ct_training_topology='dual_stream' if dual else 'single',
                  ct_enable_b1=dual, ct_enable_b2=dual, ct_enable_b3=dual)
    _seed(42)
    loader, val, generators = _resources(dual, rows=2)
    continuous = ResumeProbe(config)
    continuous_trainer = _trainer(tmp_path / 'continuous', generators, max_epochs=11, val_interval=5)
    continuous_trainer.fit(continuous, loader, val)
    expected = copy.deepcopy(continuous.state_dict())
    expected_optimizer = copy.deepcopy(continuous_trainer.optimizers[0].state_dict())
    expected_scheduler = copy.deepcopy(continuous_trainer.lr_scheduler_configs[0].scheduler.state_dict())
    expected_rng = capture_global_rng_state()
    expected_generators = {key: value.get_state() for key, value in generators.items()}

    _seed(42)
    loader, val, generators = _resources(dual, rows=2)
    first = ResumeProbe(config)
    trainer = _trainer(tmp_path / 'split', generators, stop=stop_epoch, max_epochs=11, val_interval=5)
    trainer.fit(first, loader, val)
    path = tmp_path / 'split/checkpoints/last.ckpt'
    payload = torch.load(path, weights_only=False)
    assert first.validation_epochs == []
    assert payload['callbacks']['ct_seqtrack.DataLoaderGeneratorState.v1']['validation_setup_complete'] is False
    assert payload['state_dict']['epochs_finished'] == stop_epoch
    validate_online_resume_contract(payload, config)

    _seed(9999)
    loader, val, generators = _resources(dual, rows=2)
    resumed = ResumeProbe.load_from_checkpoint(path, config=config)
    trainer = _trainer(tmp_path / 'split', generators, max_epochs=11, val_interval=5)
    trainer.fit(resumed, loader, val, ckpt_path=path)
    assert continuous.seen == first.seen + resumed.seen
    assert continuous.validation_epochs == resumed.validation_epochs == [5, 10]
    _equal(expected, resumed.state_dict())
    _equal(expected_optimizer, trainer.optimizers[0].state_dict())
    _equal(expected_scheduler, trainer.lr_scheduler_configs[0].scheduler.state_dict())
    _equal(expected_rng, capture_global_rng_state())
    _equal(expected_generators, {key: value.get_state() for key, value in generators.items()})
    assert trainer.global_step == continuous_trainer.global_step == 11
