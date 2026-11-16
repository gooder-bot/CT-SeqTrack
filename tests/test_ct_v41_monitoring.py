"""追加指标、完整 epoch 选择与评估隔离；仅编写，待服务器执行。"""
from copy import deepcopy
import csv
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pl = pytest.importorskip('pytorch_lightning')

from models.ct_v41.epoch_runtime import (EpochCheckpoint, epoch_report,
    install_evaluation_rng, isolated_evaluation_rng)
from models.ct_v41.monitoring import AppendCSVLogger, EpochProgress


def config(**changes):
    values = dict(seed=42, epoch=5, report_last_k=3, eval_split='test',
        report_checkpoint='best_precision', checkpoint_every_n_epochs=2,
        check_val_every_n_epoch=1, console_mode='plain', progress_refresh_seconds=1.,
        plain_log_interval_seconds=30., eval_checkpoint_epoch=None)
    values.update(changes)
    return SimpleNamespace(**values)


def evaluation(epoch, success=60., precision=80., split='test'):
    return dict(epoch=epoch, checkpoint_epoch=epoch, split=split, complete_coverage=True,
        success=success, precision=precision, prediction_frames=5,
        diagnostics=dict(over10m_frames=1),
        loss_episodes=dict(all=dict(episodes=2), recovered=dict(episodes=1)))


def complete_host():
    return SimpleNamespace(_epoch_complete=True, _auxiliary_complete=True,
        _completed_epoch=0, _pending_train=None, _pending_auxiliary=None,
        train_builder=SimpleNamespace(states={}), auxiliary_builder=SimpleNamespace(states={}),
        _epoch_reports={})


def test_csv_appends_sessions_and_preserves_epoch_reductions(tmp_path):
    first = AppendCSVLogger(tmp_path)
    first.log_metrics({'epoch': 0, 'train/optimization_loss_step': torch.tensor(2.)}, step=0)
    prefix = (tmp_path / 'metrics.csv').read_bytes()
    second = AppendCSVLogger(tmp_path)
    second.log_metrics({'epoch': 0, 'train/optimization_loss_epoch': 1.5}, step=4)
    assert (tmp_path / 'metrics.csv').read_bytes().startswith(prefix)
    with (tmp_path / 'metrics.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2 and rows[0]['session_id'] != rows[1]['session_id']
    assert rows[-1]['epoch'] == '1' and rows[-1]['metric'].endswith('_epoch')
    assert second.epoch_cpu_metrics[1]['train/optimization_loss_epoch'] == 1.5
    for epoch in range(2, 10):
        second.log_metrics({'epoch': epoch - 1, 'train/loss_epoch': 1.}, step=epoch)
    assert list(second.epoch_cpu_metrics) == [6, 7, 8, 9]
    assert not list(tmp_path.glob('version_*'))


def test_summary_waits_for_logger_epoch_reduction_and_prints_three_lines(tmp_path, capsys):
    cfg = config(epoch=1)
    logger = AppendCSVLogger(tmp_path)
    callback = EpochProgress(cfg, logger)
    host = complete_host()
    host._completed_epoch = 1
    report = epoch_report(host, 1)
    report.update(evaluation={'test': evaluation(1)},
        timing=dict(train_seconds=10., eval_seconds=2., save_seconds=1.),
        checkpoint=dict(path=str(tmp_path / 'epoch=001.ckpt'), saved=True))
    trainer = SimpleNamespace(default_root_dir=str(tmp_path), max_epochs=1)
    callback.on_train_epoch_end(trainer, host)
    assert not (tmp_path / 'epochs').exists()
    logger.log_metrics({'epoch': 0, 'train/optimization_loss_step': 99.,
                        'train/optimization_loss_epoch': 1.25, 'train_aux/loss_total': .5,
                        'train_aux/loss_episodes': 5., 'diagnostic/test/loss_events': 2.}, step=5)
    callback.on_train_end(trainer, host)
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 3
    assert lines[0] == 'E1/1  Loss: opt=1.2500 | main_stream=-- | aux_stream=0.5000'
    assert lines[1] == 'Test S=60.00 P=80.00 | lost=2 recovered=1 | >10m=1/5'
    assert lines[2].startswith('Time train=') and 'ckpt=001' in lines[2]
    report = json.loads((tmp_path / 'epochs' / 'epoch=001.json').read_text(encoding='utf-8'))
    assert 'train/optimization_loss_step' not in report['losses']
    assert 'train_aux/loss_episodes' not in report['losses']
    assert 'diagnostic/test/loss_events' not in report['losses']
    assert report['losses']['train/optimization_loss_epoch'] == 1.25
    callback.on_train_end(trainer, host)
    assert capsys.readouterr().out == ''


def test_standalone_csv_uses_checkpoint_epoch_instead_of_trainer_zero(tmp_path):
    cfg = config(eval_checkpoint_epoch=57)
    logger = AppendCSVLogger(tmp_path)
    callback = EpochProgress(cfg, logger)
    trainer = SimpleNamespace(num_test_batches=[1])
    host = SimpleNamespace(_completed_epoch=57)
    callback.on_test_start(trainer, host)
    # PL standalone 的 current_epoch 默认0；必须使用已校验的 checkpoint epoch。
    logger.log_metrics({'epoch': 0, 'success/test': 70.}, step=0)
    callback.on_test_end(trainer, host)
    assert logger.epoch_cpu_metrics[57]['success/test'] == 70.
    assert logger.evaluation_epoch is None
    with (tmp_path / 'metrics.csv').open(encoding='utf-8', newline='') as stream:
        assert next(csv.DictReader(stream))['epoch'] == '57'


def test_progress_exception_releases_pending_transaction(tmp_path):
    logger = AppendCSVLogger(tmp_path)
    logger.evaluation_epoch = 57
    calls = []
    host = SimpleNamespace(clear_pending_materialization=lambda: calls.append('released'))
    EpochProgress(config(), logger).on_exception(None, host, RuntimeError('backward failed'))
    assert calls == ['released'] and logger.evaluation_epoch is None


def test_plain_progress_uses_current_epoch_cpu_snapshot_and_reports_rate(tmp_path, capsys):
    cfg = config(epoch=2, plain_log_interval_seconds=0.)
    logger = AppendCSVLogger(tmp_path)
    logger.latest_cpu_metrics['train/optimization_loss_step'] = 999.
    logger.epoch_cpu_metrics[1] = {'train/optimization_loss_step': 999.}
    callback = EpochProgress(cfg, logger)
    host = complete_host()
    trainer = SimpleNamespace(current_epoch=1, max_epochs=2, num_training_batches=2,
        num_val_batches=[1], default_root_dir=str(tmp_path),
        optimizers=[SimpleNamespace(param_groups=[{'lr': .00015}])])
    callback.on_train_epoch_start(trainer, host)
    assert logger.latest_cpu_metrics == {}
    callback.on_train_batch_end(trainer, host, None, None, 0)
    callback.on_validation_start(trainer, host)
    callback.on_validation_batch_end(trainer, host, None, None, 0)
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2
    assert lines[0].startswith('Train 2/2 1/2 (50.0%) loss=-- LR=0.00015 ')
    assert lines[1].startswith('Test 2/2 1/1 (100.0%) LR=0.00015 ')
    assert all(' step/s ETA=' in line for line in lines)


def test_checkpoint_preserves_best_history_on_resume_and_does_not_splice_metrics(tmp_path):
    cfg = config()
    callback = EpochCheckpoint(cfg)
    host = complete_host()
    saves = []

    def save_checkpoint(path, weights_only):
        assert weights_only is False and host._epoch_complete
        saves.append(deepcopy(callback.state_dict()))
        Path(path).write_bytes(b'complete checkpoint')

    trainer = SimpleNamespace(current_epoch=0, max_epochs=5,
        default_root_dir=str(tmp_path), save_checkpoint=save_checkpoint)
    for epoch, success, precision in ((1, 70., 80.), (2, 60., 85.)):
        trainer.current_epoch = epoch - 1
        host._completed_epoch = epoch
        epoch_report(host, epoch)['evaluation']['test'] = evaluation(epoch, success, precision)
        callback.on_train_epoch_end(trainer, host)
    assert saves[-1]['best_epoch'] == 2  # 保存包含当轮更新后的 selector 状态。
    resumed = EpochCheckpoint(cfg)
    resumed.load_state_dict(callback.state_dict())
    callback = resumed
    for epoch, precision in ((3, 85.), (4, 84.), (5, 82.)):
        trainer.current_epoch = epoch - 1
        host._completed_epoch = epoch
        epoch_report(host, epoch)['evaluation']['test'] = evaluation(epoch, 90., precision)
        callback.on_train_epoch_end(trainer, host)
    result = json.loads((tmp_path / 'results.json').read_text(encoding='utf-8'))
    assert result['best']['epoch'] == 2 and result['main']['evaluation']['success'] == 60.
    assert result['main']['evaluation']['precision'] == 85.
    assert result['last']['epoch'] == 5
    assert result['lateK']['complete'] is True
    assert result['lateK']['available_epochs'] == [3, 4, 5]
    assert not list(tmp_path.rglob('best.ckpt')) and not list(tmp_path.rglob('last.ckpt'))
    assert len(list((tmp_path / 'formal_checkpoints').glob('*.ckpt'))) == 4


def test_partial_or_stale_evaluation_cannot_select_checkpoint(tmp_path):
    callback = EpochCheckpoint(config())
    host = complete_host()
    trainer = SimpleNamespace(current_epoch=0, max_epochs=5, default_root_dir=str(tmp_path))
    host._completed_epoch = 1
    host._auxiliary_complete = False
    with pytest.raises(RuntimeError, match='incomplete'):
        callback.on_train_epoch_end(trainer, host)
    host._auxiliary_complete = True
    epoch_report(host, 1)['evaluation']['test'] = evaluation(2)
    with pytest.raises(RuntimeError, match='this epoch'):
        callback.on_train_epoch_end(trainer, host)
    assert callback.best_epoch is None


def test_completed_checkpoint_is_not_overwritten_on_an_older_resume(tmp_path):
    callback = EpochCheckpoint(config())
    host = complete_host()
    host._completed_epoch = 1
    epoch_report(host, 1)['evaluation']['test'] = evaluation(1)
    path = tmp_path / 'formal_checkpoints' / 'epoch=001.ckpt'
    path.parent.mkdir()
    path.write_bytes(b'original complete checkpoint')
    trainer = SimpleNamespace(current_epoch=0, max_epochs=5, default_root_dir=str(tmp_path))
    with pytest.raises(FileExistsError, match='already exists'):
        callback.on_train_epoch_end(trainer, host)
    assert path.read_bytes() == b'original complete checkpoint'
    assert callback.best_epoch is None and not callback.history


def test_sparse_evaluation_late_window_uses_actual_evaluated_epochs(tmp_path):
    cfg = config(check_val_every_n_epoch=4, checkpoint_every_n_epochs=4)
    callback = EpochCheckpoint(cfg)
    host = complete_host()
    trainer = SimpleNamespace(current_epoch=0, max_epochs=5, default_root_dir=str(tmp_path),
        save_checkpoint=lambda path, weights_only: Path(path).write_bytes(b'checkpoint'))
    for epoch in range(1, 6):
        trainer.current_epoch = epoch - 1
        host._completed_epoch = epoch
        if epoch in (4, 5):
            epoch_report(host, epoch)['evaluation']['test'] = evaluation(epoch)
        callback.on_train_epoch_end(trainer, host)
    late = callback.results()['lateK']
    assert late['available_epochs'] == [4, 5] and late['count'] == 2
    assert late['complete'] is False
    assert late['success'] == 60. and late['precision'] == 80.
    assert len(list((tmp_path / 'formal_checkpoints').glob('*.ckpt'))) == 2


def test_evaluation_rng_restores_all_cpu_states_and_module_modes_on_error():
    cfg = config()
    module = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2))
    module.train()
    module[1].eval()
    loader = SimpleNamespace(generator=torch.Generator().manual_seed(99))
    random.seed(8)
    np.random.seed(8)
    torch.manual_seed(8)
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state(), loader.generator.get_state())
    with pytest.raises(RuntimeError, match='injected'):
        with isolated_evaluation_rng(cfg, 'test', loader=loader, module=module):
            random.random(), np.random.rand(), torch.rand(2)
            torch.rand(2, generator=loader.generator)
            module.eval()
            raise RuntimeError('injected')
    assert random.getstate() == before[0]
    now = np.random.get_state()
    assert now[0] == before[1][0] and np.array_equal(now[1], before[1][1]) and now[2:] == before[1][2:]
    assert torch.equal(torch.get_rng_state(), before[2])
    assert torch.equal(loader.generator.get_state(), before[3])
    assert module.training and not module[1].training


def test_loop_wrapper_covers_iterator_and_standalone_with_identical_seed():
    cfg = config()
    draws = []
    loader = SimpleNamespace(generator=torch.Generator().manual_seed(19))

    class Host(torch.nn.Linear):
        def __init__(self):
            super().__init__(1, 1)
            self._completed_epoch = 1

        def _loader(self, role):
            # 若工厂在 RNG scope 外执行，主流的随机状态会被这三项改变。
            random.random(), np.random.rand(), torch.rand(1)
            return loader

    def run():
        # 模拟 run.reset 中早于 on_validation_start 的 iterator / worker seed。
        draws.append((random.random(), np.random.rand(), float(torch.rand(())),
                      float(torch.rand((), generator=loader.generator))))

    val_loop, test_loop = SimpleNamespace(run=run), SimpleNamespace(run=run)
    trainer = SimpleNamespace(fit_loop=SimpleNamespace(epoch_loop=SimpleNamespace(val_loop=val_loop)),
        test_loop=test_loop, sanity_checking=False, current_epoch=0)
    host = Host()
    install_evaluation_rng(trainer, host, cfg)
    random.seed(4)
    np.random.seed(4)
    torch.manual_seed(4)
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    val_loop.run()
    test_loop.run()
    assert draws[0] == draws[1]
    assert random.getstate() == before[0] and torch.equal(torch.get_rng_state(), before[2])
    assert np.array_equal(np.random.get_state()[1], before[1][1])
    assert epoch_report(host, 1)['timing']['eval_seconds'] >= 0


class RandomEvaluationRows(torch.utils.data.Dataset):
    """模块顶层定义，兼容服务器 worker 与 spawn 的 pickle。"""
    def __len__(self):
        return 4

    def __getitem__(self, index):
        return torch.tensor([random.random(), np.random.rand(), torch.rand(()).item()])


@pytest.mark.parametrize('workers', [0, 2])
def test_rng_scope_starts_before_real_loader_iteration(workers):
    loader = torch.utils.data.DataLoader(RandomEvaluationRows(), batch_size=2,
        num_workers=workers, persistent_workers=False, generator=torch.Generator().manual_seed(123))
    original = loader.generator.get_state()
    readings = []
    for _ in range(2):
        with isolated_evaluation_rng(config(), 'dev', loader=loader):
            readings.append(torch.cat(list(iter(loader))))
        assert torch.equal(loader.generator.get_state(), original)
    assert torch.equal(readings[0], readings[1])
