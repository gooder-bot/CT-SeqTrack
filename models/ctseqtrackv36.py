"""v36 Lightning host：一次 forward/backward/Adam 后统一提交输出和测量状态。"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

try:
    import pytorch_lightning as pl
except ModuleNotFoundError as error:
    if error.name != 'pytorch_lightning':
        raise
    pl = None

from models.ct_v36.config import normalize_config, config_identity
from models.ct_v36.data import BatchBuilder, build_loaders, stable_seed
from models.ct_v31.lr_schedule import AbsolutePiecewiseLR, WarmupMultiStepLR
from models.ct_v36.identity import model_schema, model_version, runtime_key, training_audit_schema
from models.ct_v36.runtime import (move_tensors, TrackingEvaluation, resume_payload,
                                  restore_rng_state, validate_resume_payload)
from utils.bn_policy import running_batch_norm
from utils.execution_profile import optimized_execution
from utils.batch_materialization import attach_materialization_cache, clear_materialization_cache


class CTSEQTRACKV36(pl.LightningModule if pl is not None else nn.Module):
    def __init__(self, config=None, *, tracker=None, loaders=None):
        super().__init__()
        self.config = normalize_config(config)
        self.is_reference = False
        builder_type, self._loader_factory = BatchBuilder, build_loaders
        if tracker is None:
            from models.ct_v36.model import JointTracker
            tracker = JointTracker(self.config)
        self.tracker = tracker
        self._loaders = {} if loaders is None else loaders
        self.train_builder = builder_type(self.config)
        self.evaluation_builder = builder_type(self.config)
        self.evaluation = TrackingEvaluation(self.config)
        self.evaluation_results = {}
        self._pending_train = None
        self._pending_rng = None
        self._completed_epoch = 0
        self._epoch_complete = False
        self._epoch_rows = 0
        self._epoch_steps = 0
        self._resume_sampler = None
        self.run_id = None
        self.training_diagnostics = None
        if self.config.v36_diagnostics:
            from models.ct_v36.training_diagnostics import TrainingDiagnostics
            self.training_diagnostics = TrainingDiagnostics(
                self.config.seed, optimized=optimized_execution(self.config))
        if pl is not None:
            self.save_hyperparameters(dict(config=dict(self.config)))

    def _require_lightning(self):
        if pl is None:
            raise RuntimeError('v36 training/evaluation requires pytorch-lightning; install requirement.txt')

    def set_loaders(self, loaders):
        self._loaders = loaders

    def _loader(self, role):
        if role not in self._loaders:
            self._loaders.update(self._loader_factory(self.config, roles=(role,)))
        loader = self._loaders[role]
        if role == 'train':
            epoch = int(getattr(self, 'current_epoch', 0))
            loader.batch_sampler.set_epoch(epoch)
            loader.generator.manual_seed(stable_seed(self.config.seed, 'train', epoch))
        return loader

    def train_dataloader(self):
        self._require_lightning()
        return self._loader('train')

    def val_dataloader(self):
        self._require_lightning()
        return self._loader('val')

    def test_dataloader(self):
        self._require_lightning()
        return self._loader('test')

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # raw NumPy frames remain on CPU; only prepared tensors enter the GPU.
        return batch

    def forward(self, batch, prior=None):
        return self.tracker(batch, prior=prior)

    def _forward_raw(self, rows, builder, *, training):
        device = next(self.tracker.parameters()).device
        batch = {}
        try:
            batch = move_tensors(builder.prepare(rows), device)
            batch = attach_materialization_cache(batch, enabled=optimized_execution(self.config))
            prior = self.tracker.plan_prior(batch)
            builder.acquire(batch, prior, training=training)
            # 预留单步窗口耗尽流水线时，满批也可能集中于 GT seed 分布。
            # 从首次 drain 起与不足额批统一使用之前的 running BN，避免末尾
            # teacher/单轨迹 EMA 覆盖；只切换 B0 BN，affine 和共享特征仍可学。
            use_running = training and (len(rows) < self.config.batch_size
                or any(getattr(row['request'], 'drain', False) for row in rows))
            bn_scope = getattr(self.tracker, 'observation', self.tracker)
            with running_batch_norm(bn_scope, enabled=use_running):
                output = self.tracker(batch, prior=prior)
        except BaseException:
            self._release_materialization(batch, builder)
            raise
        return batch, output

    @staticmethod
    def _release_materialization(batch, builder):
        clear_materialization_cache(batch)
        release = getattr(builder, 'clear_materialization_cache', None)
        if release is not None:
            release()

    def clear_pending_materialization(self):
        """异常清理只释放瞬时缓存，不提交或重置跟踪状态。"""
        if self._pending_train is not None:
            self._release_materialization(self._pending_train[1], self.train_builder)
        for builder in (self.train_builder, self.evaluation_builder):
            release = getattr(builder, 'clear_materialization_cache', None)
            if release is not None:
                release()

    def configure_optimizers(self):
        parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        if not parameters or len({id(p) for p in parameters}) != len(parameters):
            raise RuntimeError('v36 optimizer requires unique trainable parameters')
        optimizer = torch.optim.Adam(parameters, lr=self.config.lr, weight_decay=self.config.wd,
                                     betas=(.5, .999), eps=1e-6, foreach=False, fused=False)
        if self.config.lr_warmup_steps:
            scheduler = WarmupMultiStepLR(optimizer, warmup_steps=self.config.lr_warmup_steps,
                milestones=self.config.lr_milestones, gamma=self.config.lr_decay_rate)
            return dict(optimizer=optimizer, lr_scheduler=dict(scheduler=scheduler, interval='step'))
        if self.config.lr_schedule == 'piecewise':
            scheduler = AbsolutePiecewiseLR(optimizer, milestones=self.config.lr_milestones,
                stage_values=self.config.lr_stage_values)
            return dict(optimizer=optimizer, lr_scheduler=dict(scheduler=scheduler, interval='epoch'))
        if self.config.lr_schedule == 'multistep':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=self.config.lr_milestones, gamma=self.config.lr_decay_rate)
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, self.config.lr_decay_step, gamma=self.config.lr_decay_rate)
        return dict(optimizer=optimizer, lr_scheduler=dict(scheduler=scheduler, interval='epoch'))

    def lr_scheduler_step(self, scheduler, metric):
        if isinstance(scheduler, WarmupMultiStepLR):
            # PL 2.0 在 Adam 后、on_train_batch_end 前调用；当前 batch 尚未计入 _epoch_steps。
            last_batch = (self.trainer.is_last_batch
                          or self._epoch_steps + 1 >= self.trainer.num_training_batches)
            scheduler.completed_epochs = int(getattr(self, 'current_epoch', 0)) + int(last_batch)
        scheduler.step()

    def on_fit_start(self):
        self._require_lightning()
        if self.trainer.accumulate_grad_batches != 1 or self.trainer.world_size != 1:
            raise ValueError('v36 accepted transactions require one device and one Adam step per batch')
        if self.trainer.reload_dataloaders_every_n_epochs != 1:
            raise ValueError('v36 curriculum requires reload_dataloaders_every_n_epochs=1')

    def on_train_epoch_start(self):
        self.train_builder.reset()
        if self.training_diagnostics is not None:
            self.training_diagnostics.reset(int(getattr(self, 'current_epoch', 0)))
        self._pending_train = None
        self._epoch_rows = self._epoch_steps = 0
        self._epoch_complete = False
        if self._pending_rng is not None:
            restore_rng_state(self._pending_rng)
            self._pending_rng = None
        if self._resume_sampler is not None:
            current = self._loaders['train'].batch_sampler.state_dict()
            identity_keys = ['schema', 'lengths', 'source_sha256', 'annotation_sha256', 'seed', 'batch_size',
                'short_window', 'long_window', 'curriculum_epochs', 'reserve_windows',
                'training', 'continuous', 'reserve_policy', 'drain_policy', 'seed_policy',
                'seed_translation', 'seed_yaw_degrees']
            for key in identity_keys:
                if key not in current or key not in self._resume_sampler or current[key] != self._resume_sampler[key]:
                    raise ValueError('v36 resume dataset/schedule identity mismatch: ' + key)
            self._resume_sampler = None

    def training_step(self, rows, batch_idx):
        if self._pending_train is not None:
            raise RuntimeError('previous training output was not committed')
        batch, output = self._forward_raw(rows, self.train_builder, training=True)
        try:
            if self.training_diagnostics is not None:
                # 瞬时 detached 统计容器；loss 内复用已经算出的 quality 标签，
                # 不改变返回字典，也不为分桶重新计算全量几何 IoU。
                output.loss_statistics = {}
            losses = self.tracker.compute_losses(batch, output)
            total = losses['loss_total']
            if total.ndim != 0 or not bool(torch.isfinite(total)):
                raise FloatingPointError('v36 loss_total must be a finite scalar')
            if self.training_diagnostics is not None:
                self.training_diagnostics.add_batch(rows, batch, output, losses)
            self._pending_train = (output, batch, len(rows))
            for name, value in losses.items():
                if torch.is_tensor(value) and value.ndim == 0:
                    self.log('train/' + name, value, on_step=True, on_epoch=True, batch_size=len(rows))
        except BaseException:
            self._release_materialization(batch, self.train_builder)
            raise
        # 普通 mean loss、每批一次标准 Adam；不声称小批按样本数缩放
        # loss 可以同比缩小 Adam 更新，也不临时改变 LR/scheduler。
        return total

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Lightning automatic optimization has completed backward + Adam here.
        output, prepared, count = self._pending_train
        try:
            self.train_builder.commit(output, prepared)
        finally:
            self._release_materialization(prepared, self.train_builder)
        self._pending_train = None
        self._epoch_rows += count
        self._epoch_steps += 1

    def on_train_epoch_end(self):
        sampler = self._loaders['train'].batch_sampler
        expected = sampler.row_count
        self._epoch_complete = (self._epoch_rows == expected and
                                self._epoch_steps == len(sampler) and self._pending_train is None)
        if not self._epoch_complete and not self.config.ct_engineering_check:
            raise RuntimeError(f'v36 incomplete epoch: rows={self._epoch_rows}/{expected}, '
                               f'updates={self._epoch_steps}/{len(sampler)}')
        if self._epoch_complete and self.train_builder.states:
            raise RuntimeError('v36 completed epoch retains unfinished windows')
        self._completed_epoch = int(getattr(self, 'current_epoch', 0)) + 1
        self._write_epoch_audit(sampler)
        if self.training_diagnostics is not None:
            self.training_diagnostics.write_epoch(self.config.log_dir,
                completed_epoch=self._completed_epoch, complete=self._epoch_complete)
        self.log('train/endpoint_rows', float(self._epoch_rows), on_step=False, on_epoch=True)
        self.log('train/adam_steps', float(self._epoch_steps), on_step=False, on_epoch=True)

    def _write_epoch_audit(self, sampler):
        """按epoch记录真实预算及参考重采样；恢复不会覆盖不同的已有审计。"""
        if not self.config.log_dir:
            return
        audit = dict(schema=training_audit_schema(self.config),
            completed_epoch=self._completed_epoch, epoch_complete=self._epoch_complete,
            rows=self._epoch_rows, optimizer_steps=self._epoch_steps, sampler=sampler.state_dict())
        if model_version(self.config) == 'v36':
            audit.update(model_schema=model_schema(self.config), config_sha256=config_identity(self.config))
        if self.is_reference:
            audit['exposure'] = self.train_builder.exposure_summary()
            # 未替换行可由名义sampler重建；只单列替换及其拒绝轨迹，避免重复大日志。
            audit['resampled_records'] = [row for row in self.train_builder.exposure_records
                                         if row['rejected']]
        directory = Path(self.config.log_dir) / 'training_audits'
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f'epoch={self._completed_epoch:03d}.json'
        content = json.dumps(audit, ensure_ascii=False, sort_keys=True, indent=2) + '\n'
        if path.exists():
            if path.read_text(encoding='utf-8') != content:
                raise FileExistsError('existing epoch audit differs; choose a new run directory: ' + str(path))
        else:
            path.write_text(content, encoding='utf-8')

    def _evaluation_start(self, role):
        self.evaluation_builder.reset()
        self.evaluation = TrackingEvaluation(self.config)
        if role in self._loaders:
            self.evaluation.add_singleton_tracks(self._loaders[role].dataset)

    def on_validation_epoch_start(self):
        self._evaluation_start('val')

    def on_test_epoch_start(self):
        self._evaluation_start('test')

    def _evaluation_step(self, rows):
        batch, output = self._forward_raw(rows, self.evaluation_builder, training=False)
        try:
            committed = self.evaluation_builder.commit(output, batch, diagnostics=True)
            self.evaluation.add_batch(rows, batch, output, committed)
        finally:
            self._release_materialization(batch, self.evaluation_builder)

    def validation_step(self, rows, batch_idx):
        self._evaluation_step(rows)

    def test_step(self, rows, batch_idx):
        self._evaluation_step(rows)

    def _evaluation_end(self, role):
        self.evaluation_results = self.evaluation.summary()
        if role in self._loaders:
            expected = self._loaders[role].batch_sampler.endpoint_count
            complete = self.evaluation_results['prediction_frames'] == expected
            self.evaluation_results['complete_coverage'] = complete
            if not complete and not self.config.ct_engineering_check:
                raise RuntimeError('v36 evaluation did not consume every prediction endpoint')
        if role == 'val' and self.config.log_dir:
            completed = int(getattr(self, 'current_epoch', 0)) + 1
            destination = Path(self.config.log_dir) / 'validation' / f'epoch={completed:03d}.json'
            destination.parent.mkdir(parents=True, exist_ok=True)
            content = json.dumps(dict(self.evaluation_results, checkpoint_epoch=completed),
                                 ensure_ascii=False, sort_keys=True, indent=2) + '\n'
            if destination.exists():
                if destination.read_text(encoding='utf-8') != content:
                    raise FileExistsError('existing validation evidence differs: ' + str(destination))
            else:
                destination.write_text(content, encoding='utf-8')
        self.log('success/' + role, self.evaluation_results['success'])
        self.log('precision/' + role, self.evaluation_results['precision'])
        for key, value in self.evaluation_results['diagnostics'].items():
            if value is not None:
                self.log('diagnostic/' + role + '/' + key, float(value))

    def on_validation_epoch_end(self):
        self._evaluation_end('val')

    def on_test_epoch_end(self):
        self._evaluation_end('test')

    def on_save_checkpoint(self, checkpoint):
        sampler = self._loaders.get('train')
        checkpoint[runtime_key(self.config)] = resume_payload(self.config,
            completed_epoch=self._completed_epoch, complete=self._epoch_complete,
            rows=self._epoch_rows, steps=self._epoch_steps,
            sampler=sampler.batch_sampler.state_dict() if sampler is not None else None,
            run_id=self.run_id)

    def on_load_checkpoint(self, checkpoint):
        payload = validate_resume_payload(checkpoint.get(runtime_key(self.config)), self.config,
                                          training=not self.config.test)
        if self.run_id is not None and not self.config.test and payload.get('run_id') != self.run_id:
            raise ValueError('v36 checkpoint belongs to another run')
        self.run_id = payload.get('run_id')
        self._completed_epoch = int(payload['completed_epoch'])
        if not self.config.test:
            self._pending_rng = payload['rng']
            self._resume_sampler = payload['sampler']
