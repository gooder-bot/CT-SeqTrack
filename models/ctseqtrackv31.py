"""v32 Lightning host；保留原模块路径，不继承历史隔离训练逻辑。"""
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

from models.ct_v31.config import normalize_config
from models.ct_v31.data import BatchBuilder, build_loaders, stable_seed
from models.ct_v31.runtime import (move_tensors, TrackingEvaluation, resume_payload,
                                  restore_rng_state, validate_resume_payload)
from utils.bn_policy import running_batch_norm


class CTSEQTRACKV31(pl.LightningModule if pl is not None else nn.Module):
    def __init__(self, config=None, *, tracker=None, loaders=None):
        super().__init__()
        self.config = normalize_config(config)
        self.is_reference = self.config.net_model == 'seqtrack_reference'
        builder_type, self._loader_factory = BatchBuilder, build_loaders
        if self.is_reference:
            from models.seqtrack_reference import (BatchBuilder as ReferenceBuilder,
                ReferenceTracker, build_loaders as reference_loaders)
            builder_type, self._loader_factory = ReferenceBuilder, reference_loaders
        if tracker is None:
            if self.is_reference:
                tracker = ReferenceTracker(self.config)
            else:
                from models.ct_v31.model import JointTracker
                tracker = JointTracker(self.config)
        self.tracker = tracker
        self._loaders = {} if loaders is None else loaders
        self.train_builder = builder_type(self.config)
        self.evaluation_builder = builder_type(self.config)
        self.evaluation = TrackingEvaluation()
        self.evaluation_results = {}
        self._pending_train = None
        self._pending_rng = None
        self._completed_epoch = 0
        self._epoch_complete = False
        self._epoch_rows = 0
        self._epoch_steps = 0
        self._resume_sampler = None
        if pl is not None:
            self.save_hyperparameters(dict(config=dict(self.config)))

    def _require_lightning(self):
        if pl is None:
            raise RuntimeError('v31 training/evaluation requires pytorch-lightning; install requirement.txt')

    def set_loaders(self, loaders):
        self._loaders = loaders

    def _loader(self, role):
        if role not in self._loaders:
            self._loaders.update(self._loader_factory(self.config, roles=(role,)))
        loader = self._loaders[role]
        if role == 'train':
            epoch = int(self.current_epoch)
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
        batch = move_tensors(builder.prepare(rows), device)
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
        return batch, output

    def configure_optimizers(self):
        parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        if not parameters or len({id(p) for p in parameters}) != len(parameters):
            raise RuntimeError('v31 optimizer requires unique trainable parameters')
        optimizer = torch.optim.Adam(parameters, lr=self.config.lr, weight_decay=self.config.wd,
                                     betas=(.5, .999), eps=1e-6, foreach=False, fused=False)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.config.lr_decay_step,
                                                   gamma=self.config.lr_decay_rate)
        return dict(optimizer=optimizer, lr_scheduler=dict(scheduler=scheduler, interval='epoch'))

    def on_fit_start(self):
        self._require_lightning()
        if self.trainer.accumulate_grad_batches != 1 or self.trainer.world_size != 1:
            raise ValueError('v31 accepted transactions require one device and one Adam step per batch')
        if self.trainer.reload_dataloaders_every_n_epochs != 1:
            raise ValueError('v31 curriculum requires reload_dataloaders_every_n_epochs=1')

    def on_train_epoch_start(self):
        self.train_builder.reset()
        self._pending_train = None
        self._epoch_rows = self._epoch_steps = 0
        self._epoch_complete = False
        if self._pending_rng is not None:
            restore_rng_state(self._pending_rng)
            self._pending_rng = None
        if self._resume_sampler is not None:
            current = self._loaders['train'].batch_sampler.state_dict()
            identity_keys = ['schema', 'lengths', 'source_sha256', 'seed', 'batch_size']
            identity_keys += (['resampling', 'nominal_endpoint_exposures'] if self.is_reference else
                ['short_window', 'long_window', 'curriculum_epochs', 'reserve_windows',
                 'reserve_policy', 'drain_policy', 'seed_policy', 'seed_translation', 'seed_yaw_degrees'])
            for key in identity_keys:
                if current[key] != self._resume_sampler[key]:
                    raise ValueError('v31 resume dataset/window manifest mismatch: ' + key)
            self._resume_sampler = None

    def training_step(self, rows, batch_idx):
        if self._pending_train is not None:
            raise RuntimeError('previous training output was not committed')
        batch, output = self._forward_raw(rows, self.train_builder, training=True)
        losses = self.tracker.compute_losses(batch, output)
        total = losses['loss_total']
        if total.ndim != 0 or not bool(torch.isfinite(total)):
            raise FloatingPointError('v31 loss_total must be a finite scalar')
        self._pending_train = (output, batch, len(rows))
        for name, value in losses.items():
            if torch.is_tensor(value) and value.ndim == 0:
                self.log('train/' + name, value, on_step=True, on_epoch=True, batch_size=len(rows))
        # 普通 mean loss、每批一次标准 Adam；不声称小批按样本数缩放
        # loss 可以同比缩小 Adam 更新，也不临时改变 LR/scheduler。
        return total

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Lightning automatic optimization has completed backward + Adam here.
        output, prepared, count = self._pending_train
        self.train_builder.commit(output, prepared)
        self._pending_train = None
        self._epoch_rows += count
        self._epoch_steps += 1

    def on_train_epoch_end(self):
        sampler = self._loaders['train'].batch_sampler
        expected = sampler.row_count
        self._epoch_complete = self._epoch_rows == expected and self._pending_train is None
        if not self._epoch_complete and not self.config.ct_engineering_check:
            raise RuntimeError(f'v31 incomplete endpoint coverage: {self._epoch_rows}/{expected}')
        if self._epoch_complete and self.train_builder.states:
            raise RuntimeError('v31 completed epoch retains unfinished windows')
        self._completed_epoch = int(self.current_epoch) + 1
        self._write_epoch_audit(sampler)
        self.log('train/endpoint_rows', float(self._epoch_rows), on_step=False, on_epoch=True)
        self.log('train/adam_steps', float(self._epoch_steps), on_step=False, on_epoch=True)

    def _write_epoch_audit(self, sampler):
        """按epoch记录真实预算及参考重采样；恢复不会覆盖不同的已有审计。"""
        if not self.config.log_dir:
            return
        audit = dict(schema='ct_seqtrack.v32.training_audit.v1',
            completed_epoch=self._completed_epoch, epoch_complete=self._epoch_complete,
            rows=self._epoch_rows, optimizer_steps=self._epoch_steps, sampler=sampler.state_dict())
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
        self.evaluation = TrackingEvaluation()
        if role in self._loaders:
            self.evaluation.add_singleton_tracks(self._loaders[role].dataset)

    def on_validation_epoch_start(self):
        self._evaluation_start('val')

    def on_test_epoch_start(self):
        self._evaluation_start('test')

    def _evaluation_step(self, rows):
        batch, output = self._forward_raw(rows, self.evaluation_builder, training=False)
        committed = self.evaluation_builder.commit(output, batch, diagnostics=True)
        self.evaluation.add_batch(rows, batch, output, committed)

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
                raise RuntimeError('v31 evaluation did not consume every prediction endpoint')
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
        checkpoint['ct_v32_runtime'] = resume_payload(self.config,
            completed_epoch=self._completed_epoch, complete=self._epoch_complete,
            rows=self._epoch_rows, steps=self._epoch_steps,
            sampler=sampler.batch_sampler.state_dict() if sampler is not None else None)

    def on_load_checkpoint(self, checkpoint):
        payload = validate_resume_payload(checkpoint.get('ct_v32_runtime'), self.config,
                                          training=not self.config.test)
        self._completed_epoch = int(payload['completed_epoch'])
        if not self.config.test:
            self._pending_rng = payload['rng']
            self._resume_sampler = payload['sampler']
