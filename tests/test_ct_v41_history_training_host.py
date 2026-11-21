"""两次独立 Lightning fit 穿过 E2 冻结前缀与 E3 切图边界。使用小型 KITTI bin。"""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from models.ct_v41 import history_augmentation as aug
from models.ct_v41.config import load_config, normalize_config
from models.ct_v41.data import RawEndpointDataset, build_loaders
from models.ctseqtrackv41 import CTSEQTRACKV41
from tests.test_ct_v41_worker_transport import WorkerKITTI


class FourTrainingCars(WorkerKITTI):
    @property
    def v41_source_role(self):
        return 'train' if self.v41_data_role == 'auxiliary' else self.v41_data_role

    def get_num_tracklets(self):
        return 4

    def get_num_frames_tracklet(self, track):
        return 4

    def get_tracklet_key(self, track):
        return f'kitti_v41/kitti_tracking/{self.v41_source_role}/{track:04}/{track}/Car/test_fixture'

    def endpoint_metadata(self, track, index):
        return dict(super().endpoint_metadata(track, index), scene_id=f'{track:04}', raw_frame_id=index)


@pytest.fixture
def history_inputs(tmp_path, monkeypatch):
    rng = np.random.default_rng(321)
    for track in range(4):
        scene = tmp_path / f'{track:04}'
        scene.mkdir()
        for frame in range(4):
            cloud = rng.uniform(-1, 1, size=(16, 4)).astype(np.float32)
            if frame == 1:
                cloud[:, :3] += 50.
            cloud.tofile(scene / f'{frame:06}.bin')
    train = FourTrainingCars(tmp_path, 'train')
    sources = dict(train=train, auxiliary=FourTrainingCars(tmp_path, 'auxiliary'))
    keys = [train.get_tracklet_key(i) for i in range(4)]
    history = tmp_path / 'history.json'
    # 小数据集使用现场生成的序列；增强与回放均由训练入口实际执行。
    monkeypatch.setattr(aug, 'TRAIN_DATA', RawEndpointDataset(train).source_sha256)
    history.write_text(json.dumps(dict(schema=aug.SCHEMA, split='train',
        source_sha256=aug.TRAIN_DATA, checkpoint_sha256=aug.E57_CHECKPOINT,
        teacher_source_sha256=aug.E57_SOURCE, reference_archive_sha256=aug.DONOR_ARCHIVE,
        donor_geometry_sha256=aug.DONOR_GEOMETRY, donor_origin='archived_frames',
        receivers={k: [4]*4 for k in keys}, donors=[dict(track=k, endpoint=3,
        counts=[4]*3, dt_us=[100000]*2, err_lwh=[[.02, 0., 0.], [.01, 0., 0.], [0., 0., 0.]])
        for k in keys])), encoding='utf-8')
    return sources, history


@pytest.fixture
def training_run():
    output = Path(__file__).resolve().parents[1] / 'artifacts' / 'ct_checks'
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output, prefix='history_host_') as scratch:
        yield Path(scratch)


def test_paired_lightning_e2_to_e3_autonomous_replay_and_updates(tmp_path, history_inputs, training_run):
    pl = pytest.importorskip('pytorch_lightning')
    sources, history = history_inputs
    shared = tmp_path / 'natural.sqlite'
    def fit(mode, directory):
        cfg = normalize_config(dict(v41_joint_search=True, v41_history_training=mode,
            v41_history_bank=str(history), v41_fixed_replay_bank=str(shared),
            ct_engineering_check=True, epoch=4, batch_size=4, workers=0, v32_reserve_windows=0,
            v41_attention_backend='math', diagnostics_level='summary', full_diagnostic_epochs=[],
            log_dir=str(directory)))
        loaders = build_loaders(cfg, roles=tuple(sources), sources=sources)
        pl.seed_everything(42, workers=True)
        host = CTSEQTRACKV41(cfg, loaders=loaders)
        trainer_options = dict(accelerator='cpu', devices=1, logger=False, enable_checkpointing=False,
            enable_progress_bar=False, enable_model_summary=False, deterministic=True,
            reload_dataloaders_every_n_epochs=1, limit_val_batches=0, num_sanity_val_steps=0,
            default_root_dir=str(directory))
        trainer = pl.Trainer(max_epochs=3, **trainer_options)
        trainer.fit(host)
        assert host._completed_epoch == 3 and host._epoch_complete
        host.joint_replay.close()
        if mode == 'window_1448':
            checkpoint = directory / 'e3.ckpt'
            trainer.save_checkpoint(str(checkpoint))
            resumed = CTSEQTRACKV41(cfg, loaders=build_loaders(cfg, roles=tuple(sources), sources=sources))
            next_trainer = pl.Trainer(max_epochs=4, **trainer_options)
            next_trainer.fit(resumed, ckpt_path=str(checkpoint))
            assert resumed._completed_epoch == 4 and resumed._epoch_complete
            assert next_trainer.global_step == trainer.global_step + resumed._epoch_steps
            assert resumed._epoch_steps == len(resumed._loaders['train'].batch_sampler)
            assert resumed.joint_replay.manifest_sha256 == host.joint_replay.manifest_sha256
            assert sum(resumed.joint_replay.used.values()) > 0
            resumed.joint_replay.close()
        return [json.loads((directory / 'training_audits' / f'epoch={e:03}.json').read_text('utf-8'))
                for e in (1, 2, 3)]

    a, b = [fit(mode, training_run / mode) for mode in ('window_1448', 'four_candidates')]
    for epoch in range(3):
        assert a[epoch]['rows'] == b[epoch]['rows'] == 48
        assert a[epoch]['optimizer_steps'] == b[epoch]['optimizer_steps']
        assert a[epoch]['sampler'] == b[epoch]['sampler']
        if epoch < 2:
            for name in ('batch_sha256', 'loss_sha256', 'weights_sha256'):
                assert a[epoch]['history_training'][name] == b[epoch]['history_training'][name]
    assert a[0]['history_training']['weights_sha256'] != a[1]['history_training']['weights_sha256']
    assert a[0]['history_training']['augmentation'].get('full_density', 0) > 0
    assert a[2]['history_training']['endpoints']['continuation'] > 0
    assert b[2]['history_training']['endpoints'].get('continuation', 0) == 0
    first, second = (arms[2]['joint_replay'] for arms in (a, b))
    assert first['used'].get('reappearance', 0) > 0 and first['used'] == second['used']
    assert first['manifest_sha256'] == second['manifest_sha256']
    assert first['plan_sha256'] == second['plan_sha256']
    assert first['donor_weights_sha256'] == second['donor_weights_sha256'] == a[1]['history_training']['weights_sha256']


def test_joint_lateral_window_uses_previous_epoch_after_resume(training_run, history_inputs):
    pl = pytest.importorskip('pytorch_lightning')
    import torch
    from models.ct_v41.joint_replay import JointReplayBank

    sources, history = history_inputs
    cfg = load_config('cfgs/ct_seqtrack/41_full_kitti_joint_training.yaml', dict(
        v41_history_bank=str(history), ct_engineering_check=True, epoch=2, workers=0,
        v32_reserve_windows=0, batch_size=4, v41_attention_backend='math',
        diagnostics_level='summary', full_diagnostic_epochs=[2], log_dir=str(training_run)))
    assert cfg.v41_q0_lateral and cfg.v41_joint_search and cfg.v41_history_training == 'window_1448'
    assert not cfg.v41_q0_lateral_replay and cfg.v41_fixed_replay_bank is None
    pl.seed_everything(42, workers=True)

    def host_and_trainer(epochs):
        host = CTSEQTRACKV41(cfg, loaders=build_loaders(cfg, roles=tuple(sources), sources=sources))
        assert isinstance(host.joint_replay, JointReplayBank) and host.q0_replay is None
        trainer = pl.Trainer(max_epochs=epochs, accelerator='cpu', devices=1, logger=False,
            enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
            deterministic=True, reload_dataloaders_every_n_epochs=1, limit_val_batches=0,
            num_sanity_val_steps=0, default_root_dir=str(training_run))
        return host, trainer

    first, trainer = host_and_trainer(1)
    trainer.fit(first)
    prefixes = {item['tracklet_key']: [] for item in first.joint_replay.available.values()}
    for (_, frame), item in first.joint_replay.available.items():
        prefixes[item['tracklet_key']].append((frame, item['kind']))
        assert item['state'].joint is not None
    assert prefixes
    original = first.tracker.q0_lateral.point_head[-1].weight.detach().clone()
    ckpt = training_run / 'e1.ckpt'
    trainer.save_checkpoint(str(ckpt))

    second, resumed = host_and_trainer(2)
    resumed.fit(second, ckpt_path=str(ckpt))
    assert second._completed_epoch == second.tracker.q0_epoch == 2
    assert second._epoch_complete and resumed.global_step == trainer.global_step + second._epoch_steps
    assert not torch.equal(original, second.tracker.q0_lateral.point_head[-1].weight)
    audit = json.loads((training_run / 'training_audits/epoch=002.json').read_text(encoding='utf-8'))
    used = sum(audit['joint_replay']['used'].values())
    assert used == audit['history_training']['endpoints']['natural_replay'] > 0
    assert audit['q0_lateral_epoch'] == 2 and 'q0_replay' not in audit
    diagnostics = json.loads((training_run / 'training_diagnostics/epoch002.json').read_text(encoding='utf-8'))
    replay_rows = diagnostics['q0_replay_rows']
    assert len(replay_rows) == used
    assert any(row['vote_supervised_points'] > 0 for row in replay_rows)
    for record in replay_rows:
        event = record['event']
        assert (event['frame'], event['kind']) in prefixes[event['tracklet_key']]
        assert (record['tracklet'], record['frame']) == (event['tracklet_key'], event['frame'])
        assert event['donor_epoch'] == 1
        assert len(record['base_box']) == len(record['box']) == 4
        if record['support_count'] < 3:
            assert record['gate_supervision'] == 'none'
        else:
            assert record['gate_supervision'] in ('fit', 'reject')

    # 固定轮评估甚至可以直接读 tracker，无需加载训练前缀或启动 Lightning。
    for epoch in (10, 11):
        evaluated = CTSEQTRACKV41(dict(cfg, test=True, checkpoint='frozen.ckpt', log_dir=None,
            v41_history_bank=None, eval_checkpoint_epoch=epoch))
        assert evaluated.tracker.q0_epoch == epoch and evaluated.joint_replay is None
        evaluated.on_test_epoch_start()
        assert evaluated.tracker.q0_epoch == evaluated._evaluation_epoch == epoch


def test_joint_pair_entry_same_e2_weights_replay_and_standalone_test(tmp_path, training_run, history_inputs):
    pytest.importorskip('pytorch_lightning')
    import torch
    from models.ct_v41 import entry

    train_sources, history = history_inputs
    test_sources = dict(test=FourTrainingCars(tmp_path, 'test'))
    fit_sources = {**train_sources, **test_sources}
    shared = tmp_path / 'joint-e2.sqlite'

    def read(path):
        return json.loads(path.read_text(encoding='utf-8'))

    def run_arm(mode):
        run_dir = training_run / mode
        cfg = load_config(f'cfgs/ct_seqtrack/41_joint_{mode}.yaml', dict(
            ct_engineering_check=True, accelerator='cpu', trainer_devices=1, epoch=3,
            batch_size=4, workers=0, v32_reserve_windows=0, v41_attention_backend='math',
            v41_history_bank=str(history), v41_fixed_replay_bank=str(shared),
            full_diagnostic_epochs=[3], log_dir=str(run_dir)))
        assert cfg.v41_history_training == mode and cfg.v41_joint_replay == 'fixed_e2'
        assert cfg.v41_joint_search and cfg.v41_q0_lateral and not cfg.v41_q0_lateral_replay
        assert cfg.eval_split == 'test' and cfg.check_val_every_n_epoch == 1
        assert cfg.report_checkpoint == 'last' and cfg.v41_q0_lateral_warmup_epochs == 10
        loaders = build_loaders(cfg, roles=tuple(fit_sources), sources=fit_sources)
        entry.run(cfg, loaders=loaders)

        manifest = read(run_dir / 'run_manifest.json')
        policy = manifest['evaluation_protocol']
        assert (policy['split'], policy['every_n_epochs'], policy['selection']) == ('test', 1, 'last')
        results = read(run_dir / 'results.json')
        assert results['evaluated_epochs'] == [1, 2, 3] and not results['selected_on_test']
        assert results['main']['epoch'] == 3
        for epoch in results['evaluated_epochs']:
            summaries = read(run_dir / 'epochs' / f'epoch={epoch:03}.json')['evaluation']
            assert set(summaries) == {'test'}
            summary = summaries['test']
            assert summary['complete_coverage'] and summary['prediction_frames'] == 12
            rows = [json.loads(line) for line in Path(summary['frame_file']).read_text('utf-8').splitlines()]
            assert all(row['tracklet'].startswith('kitti_v41/kitti_tracking/test/') for row in rows)
        audits = [read(run_dir / 'training_audits' / f'epoch={e:03}.json') for e in (1, 2, 3)]
        first = torch.load(run_dir / 'formal_checkpoints/epoch=001.ckpt', map_location='cpu',
                           weights_only=False)['state_dict']
        checkpoint = run_dir / 'formal_checkpoints/epoch=003.ckpt'
        latest = torch.load(checkpoint, map_location='cpu', weights_only=False)['state_dict']
        for head in ('point_head', 'gate_head'):
            name = f'tracker.q0_lateral.{head}.2.weight'
            assert not torch.equal(first[name], latest[name]), head
        assert [a['q0_lateral_epoch'] for a in audits] == [1, 2, 3]
        diagnostics = read(run_dir / 'training_diagnostics/epoch003.json')
        rows = diagnostics['q0_replay_rows']
        assert len(rows) == sum(audits[-1]['joint_replay']['used'].values()) > 0
        assert all(r['event']['donor_epoch'] == 2 for r in rows)
        assert any(r['vote_supervised_points'] > 0 for r in rows)
        assert all(r['applied'] == 0 and r['box'] == r['base_box'] for r in rows)

        # 仅给 E3 checkpoint 和测试数据，训练侧的历史清单与 SQLite 均无须路径。
        evaluated = normalize_config(dict(cfg, test=True, checkpoint=str(checkpoint),
            eval_split='test', v41_history_bank=None, v41_fixed_replay_bank=None,
            log_dir=str(run_dir / 'standalone_test')))
        loaders = build_loaders(evaluated, roles=('test',), sources=test_sources)
        eval_dir = entry.run(evaluated, loaders=loaders)
        result = read(eval_dir / 'results.json')
        assert result['split'] == 'test' and result['epoch'] == result['checkpoint_epoch'] == 3
        assert result['complete_coverage'] and result['prediction_frames'] == 12
        final_frame_file = results['main']['evaluation']['frame_file']
        assert Path(result['frame_file']).read_bytes() == Path(final_frame_file).read_bytes()
        eval_rows = [json.loads(line) for line in Path(result['frame_file']).read_text('utf-8').splitlines()]
        assert all(row['diagnostic_q0_lateral_applied_m'] == 0 for row in eval_rows if not row['initialization'])
        events = [(r['event']['tracklet_key'], r['event']['frame'], r['event']['kind']) for r in rows]
        return manifest, audits, events

    window, a, events_a = run_arm('window_1448')
    independent, b, events_b = run_arm('four_candidates')
    assert window['run_id'] != independent['run_id'] and window['config_sha256'] != independent['config_sha256']
    for left, right in zip(a, b):
        assert left['sampler'] == right['sampler']
        assert left['rows'] == right['rows'] == 48
        assert left['optimizer_steps'] == right['optimizer_steps']
    for left, right in zip(a[:2], b[:2]):
        for key in ('batch_sha256', 'loss_sha256', 'weights_sha256'):
            assert left['history_training'][key] == right['history_training'][key]
    assert events_a == events_b
    for key in ('manifest_sha256', 'plan_sha256', 'donor_weights_sha256', 'used'):
        assert a[2]['joint_replay'][key] == b[2]['joint_replay'][key]
    assert a[2]['joint_replay']['donor_weights_sha256'] == a[1]['history_training']['weights_sha256']
    assert a[2]['history_training']['endpoints']['continuation'] > 0
    assert b[2]['history_training']['endpoints'].get('continuation', 0) == 0
