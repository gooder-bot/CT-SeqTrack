"""工程恢复工具的计划身份、逐位比较及真实 Lightning epoch 停止/恢复回归。"""
import copy
from pathlib import Path

import pytest
import torch

from tools.check_ct_v28_resume import (
    ROOT, GENERATOR_KEY, build_plan, checkpoint_components, compare_checkpoints,
    make_callbacks, output_directory,
)
from utils.online_contract import build_online_resume_contract
from utils.training_isolation import capture_global_rng_state


def _checkpoint():
    config = dict(ct_enable_v27=True, ct_enable_v28=True, ct_engineering_check=True,
                  epoch=2, ct_runtime_protocol='safe_seqtrack_auto_v1', limit_train_batches=2)
    return dict(epoch=1, global_step=4, ct_epoch_boundary_complete=True,
        hyper_parameters={'config': config},
        ct_online_resume_contract=build_online_resume_contract(config),
        ct_global_rng_state=capture_global_rng_state(),
        state_dict={'weight': torch.tensor([0., 1.]), 'bn.running_mean': torch.tensor([.2]),
                    'ct_b0_update_step': torch.tensor(4)},
        optimizer_states=[{'state': {0: {'step': torch.tensor(4.), 'exp_avg': torch.tensor([.1])}}}],
        lr_schedulers=[{'last_epoch': 2}],
        callbacks={GENERATOR_KEY: {'schema': 'ct_seqtrack.dataloader_generators.v1',
                                  'states': {'observation': torch.Generator().manual_seed(7).get_state()}}})


def test_plan_uses_one_two_epoch_identity_and_same_split_directory(tmp_path, monkeypatch):
    monkeypatch.setattr('tools.check_ct_v28_resume.ROOT', tmp_path)
    source = tmp_path / 'source.yaml'
    source.write_text('ct_enable_v28: true\nct_enable_v27: true\nct_initialization_policy: scratch_only\n'
                      'epoch: 60\nseed: 42\nworkers: 12\ncheck_val_every_n_epoch: 5\n', encoding='utf-8')
    output = tmp_path / 'artifacts/ct_checks/resume'
    config, plan = build_plan(source, '/data/mini', output, steps=16, gpu='2')
    assert config['epoch'] == 2 and config['ct_engineering_check'] is True
    assert config['limit_train_batches'] == 16 and config['workers'] == 12
    commands = [item['argv'] for item in plan['processes']]
    assert len({command[command.index('--cfg') + 1] for command in commands}) == 1
    assert commands[1][commands[1].index('--run-dir') + 1] == commands[2][commands[2].index('--run-dir') + 1]
    assert '--checkpoint' not in commands[0] and '--checkpoint' not in commands[1]
    assert commands[2][-1].endswith('epoch=001.ckpt')
    assert plan['formal_initialization_allowed'] is False
    assert not output.exists()  # 构造计划不能创建或覆盖运行目录。


def test_output_must_be_new_and_below_ct_checks(tmp_path, monkeypatch):
    monkeypatch.setattr('tools.check_ct_v28_resume.ROOT', tmp_path)
    with pytest.raises(ValueError, match='artifacts/ct_checks'):
        output_directory(tmp_path / 'output/run')
    output = tmp_path / 'artifacts/ct_checks/used'
    output.mkdir(parents=True)
    (output / 'keep.txt').write_text('existing user artifact', encoding='utf-8')
    with pytest.raises(FileExistsError):
        output_directory(output)
    assert (output / 'keep.txt').read_text(encoding='utf-8') == 'existing user artifact'


@pytest.mark.parametrize('component', ['model', 'bn', 'adam', 'scheduler', 'global_rng', 'loader_rng'])
def test_comparison_detects_changes_in_every_required_state(component):
    original = _checkpoint()
    changed = copy.deepcopy(original)
    if component == 'model':
        # torch.equal treats +0/-0 as equal；恢复验收必须检测字节差异。
        changed['state_dict']['weight'][0] = -0.
    elif component == 'bn':
        changed['state_dict']['bn.running_mean'] += .01
    elif component == 'adam':
        changed['optimizer_states'][0]['state'][0]['exp_avg'] += .01
    elif component == 'scheduler':
        changed['lr_schedulers'][0]['last_epoch'] = 1
    elif component == 'global_rng':
        changed['ct_global_rng_state']['torch_cpu'][10] ^= 1
    else:
        changed['callbacks'][GENERATOR_KEY]['states']['observation'][10] ^= 1
    assert compare_checkpoints(original, original, 2, 2)['passed']
    report = compare_checkpoints(original, changed, 2, 2)
    assert not report['passed']
    assert any(report['comparisons'].values())


def test_missing_boundary_missing_updates_and_missing_rng_fail_closed():
    original = _checkpoint()
    for key, value in [('ct_epoch_boundary_complete', False), ('global_step', 3), ('callbacks', {})]:
        with pytest.raises(ValueError):
            checkpoint_components(dict(original, **{key: value}), 2, 2)
    changed = copy.deepcopy(original)
    config = changed['hyper_parameters']['config']
    config['ct_enable_b1'] = True
    changed['ct_online_resume_contract'] = build_online_resume_contract(config)
    with pytest.raises(ValueError, match='b1 was not updated'):
        checkpoint_components(changed, 2, 2)


def test_real_lightning_stop_snapshot_and_resume_keep_complete_epoch_state(tmp_path):
    pytest.importorskip('pytorch_lightning')
    from tests.test_ct_v27_checkpoint_runtime import ResumeProbe, _seed, _resources, _trainer

    class AuditProbe(ResumeProbe):
        def __init__(self, config):
            super().__init__(config)
            self._ct_epoch_boundary_complete = False
            for name in ('b0', 'b1', 'b2', 'b3'):
                self.register_buffer(f'ct_{name}_update_step', torch.tensor(0))

        def on_train_epoch_start(self):
            super().on_train_epoch_start()
            self._ct_epoch_boundary_complete = False

        def on_train_epoch_end(self):
            super().on_train_epoch_end()
            self._ct_epoch_boundary_complete = True
            for name in ('b0', 'b1', 'b2', 'b3'):
                getattr(self, f'ct_{name}_update_step').fill_(self.global_step)

    config = dict(ct_enable_v27=True, ct_enable_v28=True, ct_engineering_check=True,
                  ct_runtime_protocol='safe_seqtrack_auto_v1', epoch=2,
                  ct_training_topology='dual_stream', ct_enable_b1=True,
                  ct_enable_b2=True, ct_enable_b3=True, limit_train_batches=4)

    def run(directory, stop=False, checkpoint=None):
        _seed(42)
        loader, val, generators = _resources(True)
        model = (AuditProbe(config) if checkpoint is None else
                 AuditProbe.load_from_checkpoint(checkpoint, config=config))
        trainer = _trainer(directory, generators, max_epochs=2, val_interval=5)
        trainer.callbacks.extend(make_callbacks(stop_epoch=1 if stop else None))
        trainer.fit(model, loader, val, ckpt_path=checkpoint)
        return trainer

    run(tmp_path / 'continuous')
    first = run(tmp_path / 'split', stop=True)
    assert first.global_step == 4
    split_path = tmp_path / 'split/resume_audit/epoch=001.ckpt'
    payload = torch.load(split_path, map_location='cpu', weights_only=False)
    assert payload['ct_epoch_boundary_complete'] is True
    run(tmp_path / 'split', checkpoint=str(split_path))
    for epoch in (1, 2):
        checkpoints = [torch.load(tmp_path / run_name / 'resume_audit' / f'epoch={epoch:03d}.ckpt',
                                  map_location='cpu', weights_only=False)
                       for run_name in ('continuous', 'split')]
        assert compare_checkpoints(*checkpoints, epoch, 4)['passed']
