"""v35实际Lightning入口、诊断、收尾评测及完整epoch恢复；仅合成工程数据。"""
import json
from pathlib import Path
import tempfile

import pytest
import torch

pl = pytest.importorskip('pytorch_lightning')

from models.ct_v31.config import config_identity, normalize_config
from models.ct_v31.data import build_loaders
from models.ct_v31.entry import run
from models.ct_v31.runtime import validate_resume_payload
from tests.test_ct_v31_runtime import TinySource
from tests.test_ct_v34_resume import _assert_identical


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def engineering_root():
    parent = Path(__file__).resolve().parents[1] / 'artifacts' / 'ct_checks'
    return Path(tempfile.mkdtemp(prefix='v35_lightning_integration_', dir=parent))


def engineering_config(root, *, epochs=1):
    return normalize_config(dict(net_model='ctseqtrackv35', experiment_family='ct_seqtrack_v35',
        v31_arm='b0', ct_engineering_check=True, epoch=epochs, workers=0, batch_size=2,
        point_sample_size=8, accelerator='cpu', log_dir=str(root), check_val_every_n_epoch=1,
        v31_short_window=4, v31_curriculum_epochs=1, v32_reserve_windows=0,
        lr=2.5e-5, lr_schedule='multistep', lr_milestones=[1]))


def loaders_for(config, lengths=(2, 1)):
    return build_loaders(config, roles=('train', 'val', 'test'),
        sources={role: TinySource(lengths) for role in ('train', 'val', 'test')})


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def test_v35_real_entry_writes_complete_checkpoint_diagnostics_and_evaluation(engineering_root):
    root = engineering_root / 'one_epoch'
    config = engineering_config(root)
    assert run(config, loaders=loaders_for(config)) == root
    budget = read_json(root / 'training_budget.json')
    assert budget['epoch_complete'] and budget['completed_epoch'] == 1
    assert budget['last_epoch_rows'] == 4
    assert budget['last_epoch_steps'] == budget['optimizer_steps'] == 2
    assert budget['sampler']['initial_seed_policy'] == 'initial_exact_other_perturbed_v1'
    manifest = read_json(root / 'run_manifest.json')
    assert manifest['schema'] == 'ct_seqtrack.joint_identity.v35'
    assert manifest['model'] == 'ctseqtrackv35' and manifest['source']['sha256']
    assert manifest['config_sha256'] == config_identity(config)
    assert manifest['enabled'] == dict(B1=False, B2=False, B3=False)
    checkpoint_path = root / 'formal_checkpoints' / 'epoch=001.ckpt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    payload = validate_resume_payload(checkpoint['ct_v35_runtime'], config)
    assert payload['rows'] == 4 and payload['optimizer_steps'] == 2
    assert 'ct_v34_runtime' not in checkpoint
    assert checkpoint['optimizer_states'][0]['param_groups'][0]['lr'] == pytest.approx(2.5e-6)
    audit = read_json(root / 'training_audits' / 'epoch=001.json')
    assert audit['epoch_complete'] and audit['rows'] == 4 and audit['optimizer_steps'] == 2
    diagnostics = read_json(root / 'training_diagnostics' / 'epoch001.json')
    assert diagnostics['epoch_complete'] and diagnostics['totals']['endpoints'] == 4
    assert diagnostics['batches'] == 2
    assert diagnostics['totals']['statistics']['history_is_initial']['mean'] == [0., 0., 1.]
    assert diagnostics['totals']['loss_statistics']['loss_main_quality']['valid_endpoints'] == 4
    results = read_json(root / 'results.json')
    assert results['checkpoint_epochs'] == [1] and results['final']['complete_coverage']
    assert (results['final']['frames'], results['final']['prediction_frames']) == (3, 1)
    frames_path = root / 'evaluation' / 'epoch=001' / 'frames.jsonl'
    frames = [json.loads(line) for line in frames_path.read_text(encoding='utf-8').splitlines()]
    predicted = [row for row in frames if not row['initialization']]
    assert len(predicted) == 1
    for name in ('coarse', 'fine', 'accepted', 'target', 'anchor'):
        assert len(predicted[0]['diagnostic_' + name + '_box_world']) == 4
    for name in ('local_neighbor_count', 'local_selected_count', 'local_mean_support', 'local_delta_norm'):
        assert torch.as_tensor(predicted[0]['diagnostic_' + name]).shape == (4, 8)
    # 独立--test语义复查v35真实checkpoint，不复用内存中的训练网络。
    evaluation_root = engineering_root / 'standalone_test'
    evaluation_config = normalize_config(dict(config, test=True, log_dir=str(evaluation_root),
        checkpoint=str(checkpoint_path)))
    run(evaluation_config, loaders=loaders_for(evaluation_config))
    reloaded_frames = evaluation_root / 'evaluation' / 'epoch=001' / 'frames.jsonl'
    assert reloaded_frames.read_bytes() == frames_path.read_bytes()


def test_v35_real_entry_same_identity_resume_matches_uninterrupted_second_epoch(engineering_root, monkeypatch):
    from pytorch_lightning.callbacks import Callback

    class StopAfterFirst(Callback):
        def on_train_epoch_end(self, trainer, module):
            if trainer.current_epoch == 0:
                trainer.should_stop = True

    complete_root = engineering_root / 'complete'
    complete_config = engineering_config(complete_root, epochs=2)
    run(complete_config, loaders=loaders_for(complete_config, (3, 1)))
    expected = torch.load(complete_root / 'formal_checkpoints' / 'epoch=002.ckpt',
                          map_location='cpu', weights_only=False)
    resumed_root = engineering_root / 'resumed'
    first_config = normalize_config(dict(engineering_config(resumed_root, epochs=2),
        v31_evaluate_late3=False))
    # epoch预算保持2；仅由真实Trainer callback在首个完整epoch后工程停机。
    # 禁止通过把epoch从1改成2绕过配置身份。
    trainer_type = pl.Trainer

    def stopped_trainer(*args, **kwargs):
        kwargs['callbacks'] = [*kwargs.get('callbacks', ()), StopAfterFirst()]
        return trainer_type(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(pl, 'Trainer', stopped_trainer)
        run(first_config, loaders=loaders_for(first_config, (3, 1)))
    first_budget = read_json(resumed_root / 'training_budget.json')
    assert first_budget['completed_epoch'] == 1 and first_budget['epoch_complete']
    # 单轨迹递推在首轮排队产生两个尾部单行batch；不假定rows/batch_size即更新数。
    assert first_budget['last_epoch_rows'] == 8 and first_budget['optimizer_steps'] == 5
    checkpoint_path = resumed_root / 'formal_checkpoints' / 'epoch=001.ckpt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    validate_resume_payload(checkpoint['ct_v35_runtime'], first_config)
    protected = {name: (resumed_root / name).read_bytes() for name in (
        'run_manifest.json', 'resolved_config.yaml', 'training_audits/epoch=001.json',
        'training_diagnostics/epoch001.json')}
    resumed_config = normalize_config(dict(first_config, checkpoint=str(checkpoint_path),
                                            v31_evaluate_late3=True))
    assert config_identity(resumed_config) == config_identity(first_config)
    run(resumed_config, loaders=loaders_for(resumed_config, (3, 1)))
    actual = torch.load(resumed_root / 'formal_checkpoints' / 'epoch=002.ckpt',
                        map_location='cpu', weights_only=False)
    for name in ('state_dict', 'optimizer_states', 'lr_schedulers'):
        _assert_identical(expected[name], actual[name])
    assert expected['global_step'] == actual['global_step'] == 9
    assert actual['ct_v35_runtime']['completed_epoch'] == 2
    assert actual['ct_v35_runtime']['epoch_complete']
    validate_resume_payload(actual['ct_v35_runtime'], resumed_config)
    for name, content in protected.items():
        assert (resumed_root / name).read_bytes() == content
    budget = read_json(resumed_root / 'training_budget.json')
    assert budget['completed_epoch'] == 2 and budget['last_epoch_rows'] == 8
    assert budget['last_epoch_steps'] == 4 and budget['optimizer_steps'] == 9
    diagnostics_path = Path('training_diagnostics') / 'epoch002.json'
    assert (complete_root / diagnostics_path).read_bytes() == (resumed_root / diagnostics_path).read_bytes()
    results = read_json(resumed_root / 'results.json')
    assert results['checkpoint_epochs'] == [1, 2] and results['final']['complete_coverage']
    assert (complete_root / 'evaluation/epoch=002/frames.jsonl').read_bytes() == (
        resumed_root / 'evaluation/epoch=002/frames.jsonl').read_bytes()
