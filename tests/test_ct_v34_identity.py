"""v34 身份、训练预算和被动诊断；旧 v33 摘要必须保持逐字兼容。"""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from models.ct_v31.config import config_identity, load_config, normalize_config
from models.ct_v31.data import BatchBuilder, RawEndpointDataset, ReadyQueueBatchSampler
from models.ct_v31.entry import resolve_run_directory, validate_run_destination
from models.ct_v31.identity import model_schema, runtime_key, runtime_schema
from models.ct_v31.runtime import TrackingEvaluation, resume_payload, validate_resume_payload
from models.ctseqtrackv31 import CTSEQTRACKV31
from tests.test_ct_v31_runtime import TinySource, fake_output, fake_prior, request


def config34(short=3, **updates):
    name = 'context' if short == 3 else 'context_w4'
    return normalize_config(dict(load_config(f'cfgs/ct_seqtrack/34_b0_{name}_mini.yaml'), **updates))


# 修订前已登记的六份实际配置摘要；不依赖本地 output 是否已同步。
V33_HASHES = {
    'b0_half_lr': '5f4a4de5cd9c12d055bae47134bc97f3d8ab6a095f8af9f3e6c61974dcc56a02',
    'b0_late_decay': '2a14698f78dd71678a2f400bdaf9294d4270ef189e42f83f68d8897b996de834',
    'b0': 'cb674ad3bc04c27058c96039ddc9cca31429f05f93c176719799d1760e1fdf7d',
    'b0_scaled_lr': '1dae2405bbaa46b7224446092d789abd70a5714ec125492b950281a4955fb240',
    'b0_x3_lr_warmup': 'b1b98f3ee24ed8229b1a324339c0e37f9907f4b3f3942da3297f23beb63ef546',
    'seqtrack_ref': 'bf78862407cfaa5d5368ca574929a34ea86d86ccfc1ca2c636d0fd1b401f3bbc',
}


@pytest.mark.parametrize('name,digest', V33_HASHES.items())
def test_v33_registered_hashes_and_checkpoint_identity_are_unchanged(name, digest):
    config = load_config(f'cfgs/ct_seqtrack/33_{name}_mini.yaml')
    assert config_identity(config) == digest
    assert model_schema(config) == 'ct_seqtrack.joint_identity.v33'
    assert runtime_key(config) == 'ct_v33_runtime'
    assert runtime_schema(config) == 'ct_seqtrack.v33.epoch_boundary.v1'


def test_v34_two_recipes_change_only_window_and_labels():
    short, longer = config34(3), config34(4)
    allowed = {'cfg', 'experiment_name', 'tag', 'v31_short_window'}
    assert {k: v for k, v in short.items() if k not in allowed} == {
        k: v for k, v in longer.items() if k not in allowed}
    assert (short.lr, short.lr_schedule, short.lr_milestones, short.lr_warmup_steps) == (
        5e-5, 'multistep', [20, 50], 0)
    assert model_schema(short) == 'ct_seqtrack.joint_identity.v34'
    assert runtime_key(short) == 'ct_v34_runtime'
    assert config_identity(short) != config_identity(longer)
    assert '-34_b0-' in str(resolve_run_directory(short))
    assert short.v31_short_window == 3 and longer.v31_short_window == 4


@pytest.mark.parametrize('updates', [dict(v31_short_window=2), dict(lr=7.5e-5),
    dict(lr_milestones=[20, 40]), dict(lr_warmup_steps=2000), dict(v31_arm='full'),
    dict(epoch=61), dict(v32_reserve_windows=0), dict(version='v1.0-trainval')])
def test_v34_unregistered_formal_changes_are_rejected(updates):
    with pytest.raises(ValueError, match='formal v34'):
        config34(**updates)


@pytest.mark.parametrize('arm', ['b1', 'b1_b2', 'full'])
@pytest.mark.parametrize('backend', ['cfc', 'gru'])
def test_v34_other_arms_remain_available_for_engineering(arm, backend):
    config = config34(ct_engineering_check=True, v31_arm=arm, v31_temporal_backend=backend)
    assert config.net_model == 'ctseqtrackv34' and config.v31_arm == arm


def test_cross_version_and_cross_window_resume_fail_before_metadata_writes(tmp_path):
    short = config34()
    sampler = ReadyQueueBatchSampler([6], 16)
    payload = resume_payload(short, completed_epoch=1, complete=True, rows=sampler.row_count,
                             steps=len(sampler), sampler=sampler.state_dict())
    assert validate_resume_payload(payload, short) is payload
    for config in (config34(4), load_config('cfgs/ct_seqtrack/33_b0_half_lr_mini.yaml')):
        with pytest.raises(ValueError, match='runtime schema|configuration identity'):
            validate_resume_payload(payload, config)
    previous = load_config('cfgs/ct_seqtrack/33_b0_half_lr_mini.yaml')
    old_payload = resume_payload(previous, completed_epoch=1, complete=True,
        rows=sampler.row_count, steps=len(sampler), sampler=sampler.state_dict())
    with pytest.raises(ValueError, match='runtime schema'):
        validate_resume_payload(old_payload, short)
    checkpoint = tmp_path / 'epoch=001.ckpt'
    torch.save({runtime_key(short): payload}, checkpoint)
    for config in (config34(4), load_config('cfgs/ct_seqtrack/33_b0_half_lr_mini.yaml')):
        config.checkpoint = str(checkpoint)
        destination = tmp_path / config.net_model
        with pytest.raises(ValueError, match='runtime schema|configuration identity'):
            validate_run_destination(config, destination)
        assert not destination.exists()
    unfinished = dict(payload, epoch_complete=False)
    with pytest.raises(ValueError, match='complete epoch boundaries'):
        validate_resume_payload(unfinished, short)


def test_v34_host_uses_own_checkpoint_and_audit_identities(tmp_path):
    config = config34(log_dir=str(tmp_path))
    sampler = ReadyQueueBatchSampler([6], 16)
    host = CTSEQTRACKV31(config, tracker=torch.nn.Linear(1, 1),
                        loaders={'train': SimpleNamespace(batch_sampler=sampler)})
    host._completed_epoch, host._epoch_complete = 1, True
    host._epoch_rows, host._epoch_steps = sampler.row_count, len(sampler)
    checkpoint = {}
    host.on_save_checkpoint(checkpoint)
    assert set(checkpoint) == {'ct_v34_runtime'}
    host.on_load_checkpoint(checkpoint)
    assert host._completed_epoch == 1 and host._pending_rng is not None
    host._write_epoch_audit(sampler)
    import json
    saved = json.loads((tmp_path / 'training_audits/epoch=001.json').read_text())
    assert saved['schema'] == 'ct_seqtrack.v34.training_audit.v1'
    assert saved['model_schema'] == model_schema(config)
    assert saved['config_sha256'] == config_identity(config)
    assert host.evaluation.summary()['schema'] == model_schema(config)


@pytest.mark.parametrize('short', [3, 4])
def test_v34_c_schedule_and_epoch_boundary_optimizer_restore(short):
    from tests.test_ct_v33_recipes import setup_optimizer
    config = config34(short)
    model, optimizer, scheduler = setup_optimizer(config)
    expected = [5e-5] * 20 + [5e-6] * 30 + [5e-7] * 10
    for epoch in range(60):
        assert optimizer.param_groups[0]['lr'] == pytest.approx(expected[epoch])
        optimizer.zero_grad()
        model.tracker(torch.ones(1, 1)).square().sum().backward()
        optimizer.step()
        scheduler.step()
        if epoch == 39:
            checkpoint = deepcopy((model.state_dict(), optimizer.state_dict(), scheduler.state_dict()))
    resumed, resumed_optimizer, resumed_scheduler = setup_optimizer(config)
    resumed.load_state_dict(checkpoint[0])
    resumed_optimizer.load_state_dict(checkpoint[1])
    resumed_scheduler.load_state_dict(checkpoint[2])
    for epoch in range(40, 60):
        assert resumed_optimizer.param_groups[0]['lr'] == pytest.approx(expected[epoch])
        resumed_optimizer.zero_grad()
        resumed.tracker(torch.ones(1, 1)).square().sum().backward()
        resumed_optimizer.step()
        resumed_scheduler.step()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[key], value, rtol=0, atol=0)


def test_v34_passive_diagnostics_are_versioned_and_do_not_change_predictions():
    config = config34(ct_engineering_check=True, point_sample_size=8, workers=0)
    raw = RawEndpointDataset(TinySource())
    rows = [raw[request(1, branch=4, end=10)]]
    builder = BatchBuilder(config)
    batch = builder.prepare(rows)
    builder.acquire(batch, fake_prior(batch), training=False)
    output = fake_output(batch)
    output.observation.history_support = torch.ones(1, 3, 3, requires_grad=True)
    output.observation.current_support = torch.tensor([[.2, .3, .4]], requires_grad=True)
    output.decoder = SimpleNamespace(query_context_norm=torch.tensor([2.]))
    prediction = output.accepted_box.clone()
    rng = torch.get_rng_state().clone()
    old, new = TrackingEvaluation(), TrackingEvaluation(config)
    old.add_batch(rows, batch, output)
    new.add_batch(rows, batch, output)
    assert old.summary()['schema'] == 'ct_seqtrack.joint_identity.v33'
    assert new.summary()['schema'] == 'ct_seqtrack.joint_identity.v34'
    assert 'diagnostic_history_support' not in old.rows[-1]
    assert new.rows[-1]['diagnostic_history_support'] == [[1., 1., 1.]] * 3
    assert new.rows[-1]['diagnostic_current_support'] == pytest.approx([.2, .3, .4])
    assert new.summary()['diagnostics']['query_context_norm_mean'] == 2.
    assert all(new.summary()[field] == old.summary()[field] for field in ('success', 'precision'))
    assert torch.equal(prediction, output.accepted_box) and torch.equal(rng, torch.get_rng_state())
    assert output.observation.current_support.grad is None


# 保存自 C epoch060 的训练 manifest；仅长度，不包含 raw 数据或模型输出。
MINI_TRAIN_LENGTHS = [
    37,3,24,39,5,39,32,39,13,19,35,36,26,1,34,32,17,23,21,31,41,41,19,17,10,27,26,41,41,29,41,16,
    41,27,41,25,28,30,23,41,11,41,11,38,37,25,10,9,18,16,26,23,9,21,19,12,9,5,7,41,9,32,12,12,
    15,2,14,13,6,10,30,16,5,19,6,21,22,3,13,27,4,9,14,2,26,8,22,29,15,14,17,7,36,15,24,24,
    39,34,10,24,12,10,36,8,28,14,13,27,4,10,7,32,19,6,7,21,26,10,23,4,17,22,13,13,10,13,9,21,
    11,4,11,14,11,10,4,13,8,15,41,11,41,41,10,13,10,41,14,5,2,11,8,13,13,4,40,9,39,19,23,39,
    2,12,40,10,10,8,15,15,11,1,10,1,31,17,40,8,1,11,4,10,21,4,12,7,18,7,18,10,14,14,13,17,
    7,7,13,9,18,3,19,8,12,17,8,21,13,15,6,6,5,23,15,41,21,11,41,9,15,8,6,22,17,5,13,10,
    7,2,16,14,6,14,8,17,1,14,5,34,30,40,29,23,6,18,36,40,27,1,35,26,27,19,23,21,5,23,34,40,
    10,24,36,15,15,12,19,40,31,31,40,40,7,34,4,16,27,11]


@pytest.mark.parametrize('short,mature,all_epochs', [(3, 2748, 148765), (4, 4812, 254029)])
def test_real_mini_length_manifest_keeps_budget_and_increases_predicted_history(short, mature, all_epochs):
    sampler = ReadyQueueBatchSampler(MINI_TRAIN_LENGTHS, 16, short_window=short)
    total_steps, full_predicted = 0, 0
    for epoch in range(60):
        sampler.set_epoch(epoch)
        rows = [row for batch in sampler for row in batch]
        assert len(rows) == 19108 and len(sampler) == 1195
        assert sum(row.frame == 1 for row in rows) == 1072
        actual = sum(row.frame - row.window_start >= 3 for row in rows)
        full_predicted += actual
        total_steps += len(sampler)
        if epoch >= 9:
            assert actual == mature
        if epoch in (0, 3, 6, 9, 59):
            assert len({(r.tracklet, r.branch, r.frame) for r in rows}) == len(rows)
            previous = {}
            for batch in sampler:
                assert len({r.state_key for r in batch}) == len(batch)
                for row in batch:
                    assert row.frame == previous.get(row.state_key, row.window_start - 1) + 1
                    previous[row.state_key] = row.frame
    assert total_steps == 71700 and full_predicted == all_epochs
