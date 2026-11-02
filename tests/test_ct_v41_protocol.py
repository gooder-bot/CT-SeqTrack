"""V41 协议/学习率/恢复边界；仅写用例，运行验收在服务器。"""
from pathlib import Path

import pytest
import torch

from models.ct_v41.config import load_config, normalize_config, config_identity
from models.ct_v41.runtime import validate_auxiliary_payload

ROOT = Path(__file__).resolve().parents[1]


def test_registered_full_recipe_and_epoch_learning_rates():
    cfg = load_config(ROOT / 'cfgs/ct_seqtrack/41_full_kitti.yaml')
    assert (cfg.batch_size, cfg.epoch, cfg.workers, cfg.seed) == (16, 60, 4, 42)
    assert cfg.lr == 1.5e-4 and cfg.lr_decay_rate == .2
    assert [cfg.v31_short_window, cfg.v31_long_window, cfg.v31_curriculum_epochs] == [4, 8, 10]
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.Adam([parameter], lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.lr_milestones, gamma=cfg.lr_decay_rate)
    observed = []
    for epoch in range(60):
        observed.append(optimizer.param_groups[0]['lr'])
        parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        scheduler.step()
    assert observed[:20] == pytest.approx([1.5e-4] * 20)
    assert observed[20:40] == pytest.approx([3e-5] * 20)
    assert observed[40:] == pytest.approx([6e-6] * 20)


@pytest.mark.parametrize('patch', [dict(lr=1e-4), dict(epoch=59), dict(v41_arm='b1'),
    dict(v41_aux_fraction=.5), dict(v41_recovery_weight=.2), dict(v41_train_preload_offset=-1.),
    dict(v41_attention_backend='math'), dict(dataset='nuscenes_mf')])
def test_unregistered_formal_changes_are_rejected(patch):
    with pytest.raises(ValueError):
        normalize_config(patch)


def test_external_paths_do_not_change_training_identity():
    a = normalize_config()
    b = normalize_config(dict(path='/another/kitti', log_dir='/another/run', devices=1))
    assert config_identity(a) == config_identity(b)


def test_missing_auxiliary_epoch_cannot_resume():
    with pytest.raises(ValueError, match='auxiliary'):
        validate_auxiliary_payload(None, normalize_config(), {})


def test_baseline_configuration_preserves_original_optimizer_recipe():
    import yaml
    baseline = yaml.safe_load((ROOT / 'cfgs/seqtrack_baseline/41_seqtrack_kitti.yaml').read_text(encoding='utf-8'))
    assert baseline['batch_size'] == 16 and baseline['epoch'] == 60
    assert baseline['lr'] == 1e-4 and baseline['lr_decay_step'] == 20 and baseline['lr_decay_rate'] == .1
    assert baseline['optimizer'] == 'Adam' and baseline['workers'] == 10


def test_standalone_evaluation_rejects_changed_input_metadata(monkeypatch):
    import models.ct_v41.entry as entry
    import models.ct_v41.runtime as runtime
    from models.ct_v41.identity import SCHEMA, runtime_schema
    import utils.execution_profile as execution
    cfg = normalize_config(dict(test=True, checkpoint='unit.ckpt'))
    monkeypatch.setattr(entry, 'source_identity', lambda: {'sha256': 'same-source'})
    monkeypatch.setattr(execution, 'validate_execution_identity', lambda *args: None)
    monkeypatch.setattr(runtime, 'dataset_identity', lambda config: {'input': 'new-calibration'})
    payload = dict(schema=runtime_schema(cfg), model_schema=SCHEMA, config_sha256=config_identity(cfg),
                   source={'sha256': 'same-source'}, dataset_identity={'input': 'old-calibration'})
    with pytest.raises(ValueError, match='dataset/input metadata'):
        runtime.validate_resume_payload(payload, cfg, training=False)


def test_signed_auxiliary_budget_tampering_is_rejected_even_for_evaluation():
    import hashlib
    import json
    from models.ct_v41.auxiliary import FullTrackAuxiliarySampler, insertion_indices, insertion_identity
    cfg = normalize_config()
    sampler = FullTrackAuxiliarySampler([5]*20, cfg.batch_size, seed=cfg.seed, source_sha256='training')
    state = sampler.state_dict()
    main = dict(batches=20, epoch=0, lengths=state['lengths'], source_sha256='training')
    indices = insertion_indices(20, len(sampler))
    audit = dict(schema='ct_seqtrack.v41.auxiliary_audit.v1', sampler=state, complete=True,
        rows=state['rows'], batches=state['batches'], main_batches=20, loss_weight=.25,
        insertion_indices=list(indices), insertion_plan_sha256=insertion_identity(indices))
    validate_auxiliary_payload(audit, cfg, main)
    state['budget'] += 1
    content = {k: v for k, v in state.items() if k != 'manifest_sha256'}
    state['manifest_sha256'] = hashlib.sha256(json.dumps(content, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    with pytest.raises(ValueError, match='whole-track plan'):
        validate_auxiliary_payload(audit, cfg, main)
