"""单一进度行、三行 epoch 摘要和追加式 CPU 指标日志。"""
from __future__ import annotations

from collections import OrderedDict
import csv
import math
from pathlib import Path
import sys
import time
import uuid

import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers.logger import Logger

from .epoch_runtime import atomic_json, epoch_report


class AppendCSVLogger(Logger):
    """固定长表无需回读/重写旧 CSV；恢复运行用独立 session_id 区分。"""
    def __init__(self, root):
        super().__init__()
        self.root = Path(root)
        self.session_id = uuid.uuid4().hex
        self.latest_cpu_metrics = {}
        self.epoch_cpu_metrics = OrderedDict()
        self.active_epoch = 1
        self.evaluation_epoch = None

    @property
    def name(self):
        return 'metrics'

    @property
    def version(self):
        return ''

    @property
    def save_dir(self):
        return str(self.root)

    @property
    def log_dir(self):
        return str(self.root)

    @property
    def experiment(self):
        return self

    def log_hyperparams(self, params):
        # resolved_config/run_manifest 由入口统一写，不重复生成一份配置。
        pass

    def log_metrics(self, metrics, step=None):
        values = {}
        for name, value in metrics.items():
            if torch.is_tensor(value):
                if value.numel() != 1:
                    continue
                value = value.detach().cpu().item()
            if isinstance(value, (int, float)):
                values[str(name)] = float(value)
        epoch = (int(self.evaluation_epoch) if self.evaluation_epoch is not None
                 else int(values.get('epoch', self.active_epoch - 1)) + 1)
        self.latest_cpu_metrics.update(values)
        snapshot = self.epoch_cpu_metrics.setdefault(epoch, {})
        snapshot.update(values)
        self.epoch_cpu_metrics.move_to_end(epoch)
        while len(self.epoch_cpu_metrics) > 4:
            self.epoch_cpu_metrics.popitem(last=False)
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / 'metrics.csv'
        header = not path.exists() or path.stat().st_size == 0
        with path.open('a', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            if header:
                writer.writerow(('session_id', 'epoch', 'step', 'metric', 'value'))
            for name, value in sorted(values.items()):
                if name != 'epoch':
                    writer.writerow((self.session_id, epoch, '' if step is None else step, name, value))

    def save(self):
        pass  # 每次追加均关闭文件；不会缓存在内存中等待训练完成。

    def finalize(self, status):
        self.save()


def _number(value, digits=4):
    return f'{float(value):.{digits}f}' if value is not None and math.isfinite(float(value)) else '--'


def _duration(seconds):
    if seconds is None:
        return '--'
    seconds = max(0, round(float(seconds)))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


class EpochProgress(Callback):
    """batch 热路径只读 logger 已物化的 CPU 快照，不读 callback_metrics。"""
    def __init__(self, config, logger):
        super().__init__()
        self.config = config
        self.logger = logger
        self._bar = None
        self._last_refresh = 0.
        self._flushed = set()
        self._epoch_durations = []
        self._stage_started = 0.
        self._stage_label = ''
        self._stage_total = 0

    def _close_bar(self):
        if self._bar is not None:
            self._bar.close()
            self._bar = None

    def _write(self, message):
        print(message, flush=True)

    def _start_stage(self, label, total):
        self._close_bar()
        self._stage_label = label
        self._stage_total = int(total)
        self._stage_started = self._last_refresh = time.perf_counter()
        mode = self.config.console_mode
        if mode == 'bar' or (mode == 'auto' and sys.stdout.isatty()):
            from tqdm.auto import tqdm
            self._bar = tqdm(total=self._stage_total, desc=label, position=0,
                leave=False, dynamic_ncols=True, file=sys.stdout,
                bar_format='{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} {postfix}',
                mininterval=float(self.config.progress_refresh_seconds))

    def _update_stage(self, trainer, completed, *, training=False):
        now = time.perf_counter()
        interval = (float(self.config.progress_refresh_seconds) if self._bar is not None
                    else float(self.config.plain_log_interval_seconds))
        if now - self._last_refresh < interval and completed != self._stage_total:
            return
        self._last_refresh = now
        elapsed = max(now - self._stage_started, 1e-9)
        rate = completed / elapsed
        remaining = max(0, self._stage_total - completed) / rate if rate > 0 else None
        percent = 100. * completed / self._stage_total if self._stage_total else 100.
        optimizers = getattr(trainer, 'optimizers', ())
        lr = optimizers[0].param_groups[0]['lr'] if optimizers else None
        lr_text = f'{lr:.3g}' if isinstance(lr, (float, int)) else '--'
        detail = f'LR={lr_text} {rate:.2f} step/s ETA={_duration(remaining)}'
        if training:
            latest = self.logger.epoch_cpu_metrics.get(int(trainer.current_epoch) + 1, {})
            loss = latest.get('train/optimization_loss_step', latest.get('train/optimization_loss'))
            detail = 'loss=' + _number(loss) + ' ' + detail
        if self._bar is not None:
            self._bar.set_postfix_str(detail, refresh=False)
            self._bar.update(completed - self._bar.n)
        else:
            self._write(f'{self._stage_label} {completed}/{self._stage_total} ({percent:.1f}%) {detail}')

    def on_train_epoch_start(self, trainer, pl_module):
        epoch = int(trainer.current_epoch) + 1
        self._flush_epoch(trainer, pl_module, epoch - 1)
        report = epoch_report(pl_module, epoch)
        report['_started_at'] = time.perf_counter()
        self.logger.active_epoch = epoch
        self.logger.latest_cpu_metrics.clear()
        self._start_stage(f'Train {epoch}/{trainer.max_epochs}', trainer.num_training_batches)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._update_stage(trainer, batch_idx + 1, training=True)

    def on_validation_start(self, trainer, pl_module):
        label = 'Test' if self.config.eval_split == 'test' else 'Dev'
        self._start_stage(f'{label} {trainer.current_epoch + 1}/{trainer.max_epochs}',
                          sum(trainer.num_val_batches))

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._update_stage(trainer, sum(trainer.num_val_batches[:dataloader_idx]) + batch_idx + 1)

    def on_validation_end(self, trainer, pl_module):
        self._close_bar()

    def on_test_start(self, trainer, pl_module):
        label = 'Test' if self.config.eval_split == 'test' else 'Dev'
        epoch = int(self.config.eval_checkpoint_epoch or pl_module._completed_epoch)
        self.logger.evaluation_epoch = epoch
        self.logger.latest_cpu_metrics.clear()
        self._start_stage(f'{label} {epoch}', sum(trainer.num_test_batches))

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._update_stage(trainer, sum(trainer.num_test_batches[:dataloader_idx]) + batch_idx + 1)

    def on_train_epoch_end(self, trainer, pl_module):
        self._close_bar()

    def on_test_end(self, trainer, pl_module):
        self._close_bar()
        self.logger.evaluation_epoch = None

    def on_train_end(self, trainer, pl_module):
        self._close_bar()
        self._flush_epoch(trainer, pl_module, int(pl_module._completed_epoch))

    def on_exception(self, trainer, pl_module, exception):
        self._close_bar()
        self.logger.evaluation_epoch = None
        cleanup = getattr(pl_module, 'clear_pending_materialization', None)
        if callable(cleanup):
            cleanup()

    def _flush_epoch(self, trainer, module, epoch):
        reports = getattr(module, '_epoch_reports', {})
        if epoch <= 0 or epoch in self._flushed or epoch not in reports:
            return
        report = reports[epoch]
        snapshot = self.logger.epoch_cpu_metrics.get(epoch, {})
        # 优先 epoch reduction；纯 step 值不冒充整轮平均。
        losses = {}
        for key, value in snapshot.items():
            if not key.startswith(('train/', 'train_aux/')) or key.endswith('_step'):
                continue
            name = key.rsplit('/', 1)[-1].removesuffix('_epoch')
            optimization_loss = (name == 'optimization_loss' or name.startswith('loss_'))
            if optimization_loss and not name.startswith(('loss_episode', 'loss_event')) and math.isfinite(value):
                losses[key] = value
        report['losses'] = losses
        report['session_id'] = self.logger.session_id
        report.pop('_started_at', None)
        atomic_json(Path(trainer.default_root_dir) / 'epochs' / f'epoch={epoch:03d}.json', report)
        loss = losses.get('train/optimization_loss_epoch', losses.get('train/optimization_loss'))
        main_loss = losses.get('train/loss_total_epoch', losses.get('train/loss_total'))
        auxiliary_loss = losses.get('train_aux/loss_total_epoch', losses.get('train_aux/loss_total'))
        self._write(f'E{epoch}/{trainer.max_epochs}  Loss: opt={_number(loss)} | '
            f'main_stream={_number(main_loss)} | aux_stream={_number(auxiliary_loss)}')
        role = self.config.eval_split
        result = report.get('evaluation', {}).get(role)
        label = 'Test' if role == 'test' else 'Dev'
        if result is None:
            self._write(f'{label} S=-- P=-- | lost=-- recovered=-- | >10m=--/-- (not scheduled)')
        else:
            episodes = result.get('loss_episodes', {})
            lost = episodes.get('all', {}).get('episodes', '--')
            recovered = episodes.get('recovered', {}).get('episodes', '--')
            over10 = result.get('diagnostics', {}).get('over10m_frames', '--')
            count = result.get('prediction_frames', '--')
            self._write(f'{label} S={_number(result.get("success"), 2)} '
                f'P={_number(result.get("precision"), 2)} | lost={lost} recovered={recovered} | >10m={over10}/{count}')
        timing = report.get('timing', {})
        total = sum(timing.get(key, 0.) for key in ('train_seconds', 'eval_seconds', 'save_seconds'))
        self._epoch_durations.append(total)
        remaining = (sum(self._epoch_durations[-5:]) / len(self._epoch_durations[-5:])) * max(0, trainer.max_epochs - epoch)
        checkpoint = report.get('checkpoint', {})
        saved = f'{epoch:03d}' if checkpoint.get('saved') and checkpoint.get('path') else '--'
        self._write(f'Time train={_duration(timing.get("train_seconds"))} '
            f'eval={_duration(timing.get("eval_seconds"))} save={_duration(timing.get("save_seconds"))} '
            f'| ckpt={saved} | remaining=~{_duration(remaining)}')
        self._flushed.add(epoch)
        # 完整摘要落盘后不在 host 内重复保留全部历史；checkpoint 自持评估历史。
        for old in tuple(reports):
            if old < epoch - 1:
                del reports[old]
