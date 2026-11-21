"""真实 Lightning 自动优化、合成 KITTI 整轮回放及验证轮号。"""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import torch

from models.ct_v41.config import normalize_config
from models.ct_v41.data import PREDICTED_BOX, build_loaders
from models.ct_v41.identity import runtime_key
from models.ct_v41 import runtime
from models.ctseqtrackv41 import CTSEQTRACKV41
from tests.test_ct_v41_q0_replay import _manifest
from tests.test_ct_v41_worker_transport import kitti


@pytest.fixture
def smoke_run():
    root = Path(__file__).resolve().parents[1] / 'artifacts' / 'ct_checks'
    root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=root, prefix='q0_lateral_host_') as run:
        yield Path(run)


def test_lightning_q0_replay_adam_and_accepted_state(kitti, tmp_path, smoke_run, monkeypatch):
    pl = pytest.importorskip('pytorch_lightning')
    path = tmp_path / 'prefix.json'
    # 保留正式 768+256 点槽合同，只使用每帧 16 个真实返回。
    for scan in (tmp_path / '0000').glob('*.bin'):
        np.fromfile(scan, dtype=np.float32, count=16 * 4).tofile(scan)
    config = normalize_config(dict(
        ct_engineering_check=True, dataset='kitti_mf', v41_attention_backend='math',
        v41_q0_lateral_replay=True, v41_q0_replay_manifest=str(path), epoch=11,
        batch_size=4, workers=0, v32_reserve_windows=0,
        diagnostics_level='summary', full_diagnostic_epochs=[1], log_dir=str(smoke_run)))
    sources = {'train': kitti, 'auxiliary': kitti}
    loaders = build_loaders(config, roles=tuple(sources), sources=sources)
    raw = loaders['train'].dataset
    _manifest(raw, path)
    # 该数据源使用现场生成的 KITTI 元数据及 .bin，无正式服务器标注清单。
    monkeypatch.setattr(runtime, 'dataset_identity', lambda _: {'source_sha256': raw.source_sha256})
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(42)
        host = CTSEQTRACKV41(config, loaders=loaders)
    with torch.no_grad():
        host.tracker.observation.seg_pointnet.fc.bias[1].add_(10.)
    head_bias = host.tracker.q0_lateral.point_head[-1].bias.detach().clone()
    checked = []
    outputs = []
    commit = host.train_builder.commit

    def observe_commit(output, batch, **kwargs):
        pending = list(host.train_builder._pending)
        expected = output.accepted_box.detach().cpu().double().numpy().copy()
        result = commit(output, batch, **kwargs)
        for i, context in enumerate(pending):
            row, state = context['row'], context['state']
            frame = row['request'].frame
            expected[i, :3] += context['anchor'][:3]
            np.testing.assert_allclose(state.boxes[frame], expected[i], rtol=0., atol=1e-6)
            assert state.next_frame == frame + 1 and state.box_provenance[frame] == PREDICTED_BOX
            if 'q0_replay_event' in row:
                checked.append((row['q0_replay_event'], frame, context['anchor'][0]))
                lat = output.decoder.lateral
                outputs.append((lat.base_box[i].detach().tolist(), lat.box[i].detach().tolist(),
                                lat.gate[i].item(), lat.support_count[i].item()))
        return result

    monkeypatch.setattr(host.train_builder, 'commit', observe_commit)
    trainer = pl.Trainer(accelerator='cpu', devices=1, max_epochs=1, logger=False,
                         enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
                         deterministic=True, reload_dataloaders_every_n_epochs=1, limit_val_batches=0,
                         num_sanity_val_steps=0, default_root_dir=str(tmp_path))
    trainer.fit(host)
    assert host._epoch_complete and trainer.global_step == len(loaders['train'].batch_sampler)
    assert not torch.equal(host.tracker.q0_lateral.point_head[-1].bias, head_bias)
    assert host.q0_replay.audit()['used'] == {'hard': 1, 'normal': 1}
    assert checked == [({'event_id': 'e10/event1', 'kind': 'hard', 'donor_epoch': 10}, 5, 24.),
                       ({'event_id': 'e10/event1', 'kind': 'normal', 'donor_epoch': 10}, 2, 0.)]
    assert host.tracker.q0_epoch == 1
    record = json.loads((smoke_run / 'training_diagnostics/epoch001.json').read_text(encoding='utf-8'))
    logged = record['q0_replay_rows']
    assert [(row['event'], row['frame']) for row in logged] == [(event, frame) for event, frame, _ in checked]
    assert [(row['base_box'], row['box'], row['gate'], row['support_count']) for row in logged] == outputs
    assert all(row['applied'] == 0 and row['base_box'] == row['box'] for row in logged)

    checkpoint = smoke_run / 'host.ckpt'
    trainer.save_checkpoint(str(checkpoint))
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    saved = payload[runtime_key(config)]
    assert saved['q0_replay_manifest_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    host.on_load_checkpoint(payload)
    saved['q0_replay_manifest_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='replay manifest'):
        host.on_load_checkpoint(payload)

    # 在同一实际 Trainer 上核对两个课程边界的验证 hook，不再训练权重。
    for epoch in (10, 11):
        trainer.fit_loop.epoch_progress.current.completed = epoch - 1
        host.on_validation_epoch_start()
        assert host.tracker.q0_epoch == host._evaluation_epoch == epoch
    host._evaluation_stream.close()
