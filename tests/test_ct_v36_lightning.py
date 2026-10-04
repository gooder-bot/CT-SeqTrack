"""真实 Lightning Trainer 生命周期；只在新工程目录生成合成数据证据。"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile

import pytest
import torch

pl = pytest.importorskip('pytorch_lightning')

from models.ct_v36.config import normalize_config
from models.ct_v36.data import build_loaders
from models.ct_v36.entry import run
from models.ctseqtrackv36 import CTSEQTRACKV36
from tests.test_ct_v31_runtime import TinySource
from utils.lightning_runtime import FinalWindowCheckpoint


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def artifact_root():
    root = Path(__file__).resolve().parents[1] / 'artifacts' / 'ct_checks'
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix='v36_lightning_', dir=root))


def engineering_config(root, **changes):
    options = dict(ct_engineering_check=True, accelerator='cpu',
        point_sample_size=8, batch_size=4, workers=0, epoch=1,
        check_val_every_n_epoch=1, v32_reserve_windows=0,
        log_dir=str(root))
    options.update(changes)
    return normalize_config(options)


def test_actual_entry_saves_complete_checkpoint_and_standalone_eval_matches(artifact_root):
    config = engineering_config(artifact_root / 'entry')
    sources = {role: TinySource((3, 2)) for role in ('train', 'val', 'test')}
    root = run(config, loaders=build_loaders(config, roles=tuple(sources), sources=sources))
    checkpoint = root / 'formal_checkpoints' / 'epoch=001.ckpt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    runtime = saved['ct_v36_runtime']
    assert runtime['completed_epoch'] == 1 and runtime['epoch_complete']
    assert runtime['rows'] == 12 and runtime['optimizer_steps'] == 3
    assert saved['global_step'] == 3 and saved['lr_schedulers'][0]['last_epoch'] == 1
    assert json.loads((root / 'training_budget.json').read_text(encoding='utf-8'))['epoch_complete']
    assert json.loads((root / 'validation' / 'epoch=001.json').read_text(encoding='utf-8'))['complete_coverage']

    evaluation = normalize_config(dict(config, test=True, checkpoint=str(checkpoint),
        log_dir=str(artifact_root / 'standalone')))
    test_loaders = build_loaders(evaluation, roles=('test',), sources={'test': TinySource((3, 2))})
    standalone = run(evaluation, loaders=test_loaders)
    first = json.loads((root / 'results.json').read_text(encoding='utf-8'))['final']
    second = json.loads((standalone / 'results.json').read_text(encoding='utf-8'))['final']
    assert first == second
    relative = Path('evaluation/epoch=001/frames.jsonl')
    assert (root / relative).read_bytes() == (standalone / relative).read_bytes()


def test_actual_trainer_epoch10_to11_resume_is_bitwise_identical(artifact_root):
    class StopAfterTen(pl.Callback):
        def on_train_epoch_end(self, trainer, module):
            if trainer.current_epoch == 9:
                trainer.should_stop = True

    class RecordOrder(pl.Callback):
        def __init__(self):
            self.rows = []

        def on_train_batch_start(self, trainer, module, rows, batch_idx):
            self.rows.append((trainer.current_epoch, trainer.global_step,
                trainer.optimizers[0].param_groups[0]['lr'],
                tuple(row['request'] for row in rows)))

    def make(root, *, stop=False):
        config = engineering_config(root, epoch=11, v36_arm='C',
            v36_diagnostics=False, v31_evaluate_late3=False, limit_val_batches=0)
        loaders = build_loaders(config, roles=('train',), sources={'train': TinySource((10, 2))})
        model = CTSEQTRACKV36(config, loaders=loaders)
        model.run_id = '0123456789abcdef0123456789abcdef'
        record = RecordOrder()
        callbacks = [record, FinalWindowCheckpoint(keep=3, every_n_epochs=2)]
        if stop:
            callbacks.append(StopAfterTen())
        trainer = pl.Trainer(default_root_dir=str(root), accelerator='cpu', devices=1,
            max_epochs=11, logger=False, callbacks=callbacks, enable_progress_bar=False,
            enable_model_summary=False, num_sanity_val_steps=0, limit_val_batches=0,
            reload_dataloaders_every_n_epochs=1, deterministic=True)
        return model, trainer, record

    pl.seed_everything(42, workers=True)
    reference, uninterrupted, reference_order = make(artifact_root / 'uninterrupted')
    uninterrupted.fit(reference)
    expected = deepcopy(reference.state_dict())
    expected_optimizer = deepcopy(uninterrupted.optimizers[0].state_dict())
    pl.seed_everything(42, workers=True)
    first, interrupted, first_order = make(artifact_root / 'resumed', stop=True)
    interrupted.fit(first)
    checkpoint = artifact_root / 'resumed' / 'formal_checkpoints' / 'epoch=010.ckpt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert saved['ct_v36_runtime']['completed_epoch'] == 10
    assert saved['ct_v36_runtime']['epoch_complete']
    assert saved['ct_v36_runtime']['sampler']['epoch'] == 9
    assert saved['lr_schedulers'][0]['last_epoch'] == 10
    resumed, continuation, resumed_order = make(artifact_root / 'resumed')
    continuation.fit(resumed, ckpt_path=str(checkpoint))
    assert continuation.global_step == uninterrupted.global_step == 110
    assert interrupted.global_step == 100
    assert first_order.rows + resumed_order.rows == reference_order.rows
    assert any(request.branch == 3 and request.frame == 9 and request.window_start == 1
               for _, _, _, requests in resumed_order.rows for request in requests)
    assert all(torch.equal(value, resumed.state_dict()[key]) for key, value in expected.items())
    actual_optimizer = continuation.optimizers[0].state_dict()
    assert actual_optimizer['param_groups'] == expected_optimizer['param_groups']
    for index, values in expected_optimizer['state'].items():
        for key, value in values.items():
            torch.testing.assert_close(value, actual_optimizer['state'][index][key], rtol=0, atol=0)
    assert continuation.lr_scheduler_configs[0].scheduler.state_dict() == uninterrupted.lr_scheduler_configs[0].scheduler.state_dict()
