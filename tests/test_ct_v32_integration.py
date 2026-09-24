"""真实 Lightning 生命周期与完整入口；只使用合成原始点云，不冒充数据集实验。"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile

import pytest
import torch

pl = pytest.importorskip('pytorch_lightning')

from models.ct_v31.config import config_identity, normalize_config
from models.ct_v31.entry import run
from models.ct_v31.runtime import validate_resume_payload
from models.ctseqtrackv31 import CTSEQTRACKV31
from tests.test_ct_v31_runtime import TinySource
from tests.test_seqtrack_reference import TinyReferenceSource


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def engineering_directory():
    parent = Path(__file__).resolve().parents[1] / 'artifacts' / 'ct_checks' / 'v33_integration'
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix='run_', dir=parent))


@pytest.mark.parametrize('name', ['b0', 'seqtrack_ref', 'full_gru', 'full_cfc'])
def test_entry_trains_checkpoints_and_evaluates_independent_model(name):
    reference = name == 'seqtrack_ref'
    root = engineering_directory()
    config = normalize_config(dict(net_model='seqtrack_reference' if reference else 'ctseqtrackv33',
        v31_arm='full' if name.startswith('full_') else 'b0',
        v31_temporal_backend='gru' if name == 'full_gru' else 'cfc',
        ct_engineering_check=True, epoch=1, workers=0, batch_size=2,
        point_sample_size=1024 if reference else 16, accelerator='cpu',
        log_dir=str(root), check_val_every_n_epoch=1))
    if reference:
        from models.seqtrack_reference import build_loaders
        source = TinyReferenceSource((2, 1))
    else:
        from models.ct_v31.data import build_loaders
        source = TinySource((2, 1))
    loaders = build_loaders(config, roles=('train', 'val', 'test'),
                            sources={role: source for role in ('train', 'val', 'test')})
    result = run(config, loaders=loaders)
    assert result == root
    budget = json.loads((root / 'training_budget.json').read_text(encoding='utf-8'))
    assert budget['epoch_complete'] and budget['last_epoch_rows'] == 4
    assert budget['last_epoch_steps'] == budget['optimizer_steps'] == 2
    summary = json.loads((root / 'results.json').read_text(encoding='utf-8'))
    assert summary['checkpoint_epochs'] == [1]
    assert summary['final']['complete_coverage']
    assert (summary['final']['frames'], summary['final']['prediction_frames']) == (3, 1)
    manifest = json.loads((root / 'run_manifest.json').read_text(encoding='utf-8'))
    assert manifest['model'] == config.net_model and manifest['source']['sha256']
    assert manifest['temporal_backend'] == config.v31_temporal_backend
    assert manifest['enabled'] == dict.fromkeys(('B1', 'B2', 'B3'), name.startswith('full_'))
    state = torch.load(root / 'formal_checkpoints' / 'epoch=001.ckpt', map_location='cpu')
    validate_resume_payload(state['ct_v33_runtime'], config, training=True)
    assert config_identity(config) == manifest['config_sha256']
    audit = json.loads((root / 'training_audits' / 'epoch=001.json').read_text(encoding='utf-8'))
    if reference:
        assert audit['exposure']['actual_rows'] == 4 and 'reference_protocol' in manifest
    else:
        assert audit['sampler']['reserve_windows'] == 112
    # 自动收尾和全新--test进程语义应使用相同评测采样起点。
    evaluation_root = engineering_directory()
    evaluation_config = normalize_config(dict(config, test=True, log_dir=str(evaluation_root),
        checkpoint=str(root / 'formal_checkpoints' / 'epoch=001.ckpt')))
    evaluation_loaders = build_loaders(evaluation_config, roles=('test',), sources={'test': source})
    run(evaluation_config, loaders=evaluation_loaders)
    original_frames = (root / 'evaluation' / 'epoch=001' / 'frames.jsonl').read_text(encoding='utf-8')
    reloaded_frames = (evaluation_root / 'evaluation' / 'epoch=001' / 'frames.jsonl').read_text(encoding='utf-8')
    assert original_frames == reloaded_frames
    standalone = json.loads((evaluation_root / 'results.json').read_text(encoding='utf-8'))
    assert standalone['checkpoint_epochs'] == [1]
    assert standalone['final']['checkpoint_epoch'] == 1


def test_reference_epoch_resume_matches_network_optimizer_and_exposures(tmp_path):
    from pytorch_lightning.callbacks import Callback
    from models.seqtrack_reference import build_loaders
    from utils.lightning_runtime import FinalWindowCheckpoint

    class StopAfterFirst(Callback):
        def on_train_epoch_end(self, trainer, module):
            if trainer.current_epoch == 0:
                trainer.should_stop = True

    config = normalize_config(dict(net_model='seqtrack_reference', v31_arm='b0',
        ct_engineering_check=True, epoch=2, batch_size=2, workers=0, lr_decay_step=1))

    def make(directory, stop=False):
        loaders = build_loaders(config, roles=('train',), sources={'train': TinyReferenceSource((2,))})
        model = CTSEQTRACKV31(config, loaders=loaders)
        callbacks = [FinalWindowCheckpoint(keep=2)] + ([StopAfterFirst()] if stop else [])
        trainer = pl.Trainer(default_root_dir=str(directory), accelerator='cpu', devices=1,
            max_epochs=2, logger=False, callbacks=callbacks, enable_progress_bar=False,
            enable_model_summary=False, num_sanity_val_steps=0, limit_val_batches=0,
            reload_dataloaders_every_n_epochs=1, deterministic=True)
        return model, trainer

    pl.seed_everything(42, workers=True)
    complete, trainer = make(tmp_path / 'complete')
    trainer.fit(complete)
    expected = deepcopy(complete.state_dict())
    optimizer = deepcopy(trainer.optimizers[0].state_dict())
    exposure = complete.train_builder.exposure_summary()
    pl.seed_everything(42, workers=True)
    first, interrupted = make(tmp_path / 'resumed', True)
    interrupted.fit(first)
    continued, resumed = make(tmp_path / 'resumed')
    resumed.fit(continued, ckpt_path=str(tmp_path / 'resumed' / 'formal_checkpoints' / 'epoch=001.ckpt'))
    assert trainer.global_step == resumed.global_step == 4
    for key, value in expected.items():
        torch.testing.assert_close(value, continued.state_dict()[key], rtol=0, atol=0)
    actual_optimizer = resumed.optimizers[0].state_dict()
    assert optimizer['param_groups'] == actual_optimizer['param_groups']
    for key, values in optimizer['state'].items():
        for name, value in values.items():
            torch.testing.assert_close(value, actual_optimizer['state'][key][name], rtol=0, atol=0)
    assert exposure == continued.train_builder.exposure_summary()
