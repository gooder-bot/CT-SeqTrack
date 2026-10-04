"""v36 宿主完整epoch、恢复、入口与冻结配方的本地验证。"""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from models.ct_v36.config import normalize_config, load_config, config_identity
from models.ct_v36.data import build_loaders
from models.ct_v36.entry import parse_config, validate_run_destination
from models.ct_v36.runtime import resume_payload, validate_resume_payload
from models.ct_v36.sampler import ReadyQueueBatchSampler
from models.ctseqtrackv36 import CTSEQTRACKV36, pl
from tests.test_ct_v31_runtime import TinySource


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def config(**changes):
    return normalize_config(dict(ct_engineering_check=True, point_sample_size=8,
        batch_size=2, workers=0, v32_reserve_windows=0, **changes))


def set_epoch(host, epoch):
    if pl is None:
        host.current_epoch = epoch
    else:
        host._trainer = SimpleNamespace(current_epoch=epoch)
    host.log = lambda *args, **kwargs: None


def test_host_epoch10_resume_to_continuous11_preserves_optimizer_rng_and_outputs():
    cfg = config(v36_arm='C')
    sources = {'train': TinySource(lengths=(3, 2)), 'test': TinySource(lengths=(3, 2))}
    loaders = build_loaders(cfg, roles=('train', 'test'), sources=sources)
    host = CTSEQTRACKV36(cfg, loaders=loaders)
    host.run_id = 'cpu-epoch-boundary-fixture'
    set_epoch(host, 9)
    loader = host._loader('train')
    host.on_train_epoch_start()
    optimizer = host.configure_optimizers()['optimizer']
    for index, rows in enumerate(loader):
        optimizer.zero_grad(set_to_none=True)
        loss = host.training_step(rows, index)
        loss.backward()
        optimizer.step()
        host.on_train_batch_end(None, rows, index)
    host.on_train_epoch_end()
    assert host._epoch_complete and not host.train_builder.states
    checkpoint = dict(state_dict=deepcopy(host.state_dict()), optimizer_states=[deepcopy(optimizer.state_dict())])
    host.on_save_checkpoint(checkpoint)
    assert checkpoint['ct_v36_runtime']['completed_epoch'] == 10
    payload = validate_resume_payload(checkpoint['ct_v36_runtime'], cfg)
    assert payload['optimizer_steps'] == len(loader)
    resumed = CTSEQTRACKV36(cfg, loaders=build_loaders(cfg, roles=('train', 'test'), sources=sources))
    resumed.run_id = host.run_id
    resumed.load_state_dict(checkpoint['state_dict'])
    resumed.on_load_checkpoint(checkpoint)
    restored_optimizer = resumed.configure_optimizers()['optimizer']
    restored_optimizer.load_state_dict(checkpoint['optimizer_states'][0])
    snapshots = []
    for model, opt in ((host, optimizer), (resumed, restored_optimizer)):
        set_epoch(model, 10)
        current_loader = model._loader('train')
        # 相同epoch边界恢复同一 RNG；不中途续接已detach状态。
        model._pending_rng = deepcopy(payload['rng'])
        model.on_train_epoch_start()
        rows = next(iter(current_loader))
        opt.zero_grad(set_to_none=True)
        loss = model.training_step(rows, 0)
        loss.backward(); opt.step(); model.on_train_batch_end(None, rows, 0)
        snapshots.append({k: v.detach().clone() for k, v in model.state_dict().items()})
    assert snapshots[0].keys() == snapshots[1].keys()
    for key in snapshots[0]:
        torch.testing.assert_close(snapshots[0][key], snapshots[1][key], rtol=0, atol=0)
    # 同一个host直接进行合法首框开始的完整评测；不要求本地Lightning安装。
    host.eval(); host._evaluation_start('test')
    with torch.no_grad():
        for rows in host._loaders['test']:
            host._evaluation_step(rows)
    host._evaluation_end('test')
    assert host.evaluation_results['complete_coverage']
    assert host.evaluation_results['prediction_frames'] == 3


def test_resume_rejects_partial_source_cross_arm_seed_and_run():
    cfg = config()
    sampler = ReadyQueueBatchSampler([3, 2], batch_size=2, reserve_windows=0)
    payload = resume_payload(cfg, completed_epoch=1, complete=True,
        rows=sampler.row_count, steps=len(sampler), sampler=sampler.state_dict(), run_id='one')
    assert validate_resume_payload(payload, cfg)
    for changed in (dict(payload, epoch_complete=False), dict(payload, source={}),
                    dict(payload, optimizer_steps=999)):
        with pytest.raises(ValueError):
            validate_resume_payload(changed, cfg)
    for changed in (dict(cfg, v36_arm='D'), dict(cfg, seed=43)):
        with pytest.raises(ValueError):
            validate_resume_payload(payload, changed)
    host = CTSEQTRACKV36(cfg); host.run_id = 'another'
    with pytest.raises(ValueError, match='another run'):
        host.on_load_checkpoint({'ct_v36_runtime': payload})


def test_five_formal_configs_and_cli_have_one_registered_budget():
    paths = sorted(Path('cfgs/ct_seqtrack').glob('36_*_mini.yaml'))
    assert len(paths) == 5
    configs = [parse_config(['--cfg', str(p)]) for p in paths]
    assert {c.v36_arm for c in configs} == set('ACDEF')
    assert len({config_identity(c) for c in configs}) == 5
    assert all((c.epoch, c.batch_size, c.lr) == (60, 16, .00015) for c in configs)
    assert all(c.lr_milestones == [20, 50] and c.lr_warmup_steps == 0 for c in configs)


def test_entry_refuses_occupied_or_mismatched_destinations(tmp_path):
    cfg = config()
    (tmp_path / 'old_evidence.json').write_text('{}', encoding='utf-8')
    with pytest.raises(FileExistsError):
        validate_run_destination(cfg, tmp_path)
    other = tmp_path / 'bad'; other.mkdir()
    (other / 'run_manifest.json').write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='incomplete'):
        validate_run_destination(cfg, other)
