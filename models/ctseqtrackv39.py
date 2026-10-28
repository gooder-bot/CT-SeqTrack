"""v39 Lightning host：一个 raw service，Adam 后唯一 accepted-state 提交。"""
from __future__ import annotations

import json
from pathlib import Path
import uuid

import torch
from torch import nn

try:
    import pytorch_lightning as pl
except ModuleNotFoundError as error:
    if error.name != 'pytorch_lightning':
        raise
    pl = None

from models.ct_v39.config import normalize_config, config_identity
from models.ct_v39.data import BatchBuilder, build_loaders, stable_seed
from models.ct_v39.raw_service import RawFrameService
from models.ct_v39.identity import model_schema, runtime_key, training_audit_schema
from models.ct_v39.runtime import TrackingEvaluation, resume_payload, validate_resume_payload, restore_rng_state
from utils.bn_policy import running_batch_norm
from utils.diagnostic_transfer import tensors_to_numpy


class CTSEQTRACKV39(pl.LightningModule if pl is not None else nn.Module):
    def __init__(self, config=None, *, tracker=None, loaders=None):
        super().__init__()
        self.config = normalize_config(config)
        if tracker is None:
            from models.ct_v39.model import V39Model
            tracker = V39Model(self.config)
        self.tracker = tracker
        self.run_id = uuid.uuid4().hex
        self._loader_factory = build_loaders
        self._loaders = {} if loaders is None else dict(loaders)
        services = [getattr(loader.dataset, 'service', None) for loader in self._loaders.values()]
        self.raw_service = next((service for service in services if service is not None), None)
        if self.raw_service is None:
            self.raw_service = RawFrameService(self.config)
        self._bind_loaders()
        self.train_builder = BatchBuilder(self.config, raw_service=self.raw_service)
        self.evaluation_builder = BatchBuilder(self.config, raw_service=self.raw_service)
        self.evaluation = TrackingEvaluation(self.config)
        self.evaluation_results = {}
        self._pending_train = self._pending_rng = self._resume_sampler = None
        self._completed_epoch = self._epoch_rows = self._epoch_steps = 0
        self._epoch_complete = False
        self._training_scalars = {}
        self._training_counts = {}
        if pl is not None:
            self.save_hyperparameters(dict(config=dict(self.config)))

    def _bind_loaders(self):
        for loader in self._loaders.values():
            if hasattr(loader.dataset, 'service'):
                loader.dataset.service = self.raw_service

    def set_loaders(self, loaders):
        self._loaders = dict(loaders)
        self._bind_loaders()

    def _require_lightning(self):
        if pl is None:
            raise RuntimeError('v39 training/evaluation requires pytorch-lightning')

    def _loader(self, role):
        if role not in self._loaders:
            self._loaders.update(self._loader_factory(self.config, roles=(role,), raw_service=self.raw_service))
            self._bind_loaders()
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
        return batch

    def forward(self, transaction, builder=None):
        return self.tracker.forward_transaction(transaction, builder or self.evaluation_builder)

    def _forward_raw(self, rows, builder, *, training):
        device = next(self.tracker.parameters()).device
        try:
            transaction = builder.begin(rows, device, training=training)
            use_running = training and (len(rows) < self.config.batch_size
                or any(getattr(row['request'], 'drain', False) for row in rows))
            with running_batch_norm(getattr(self.tracker, 'observation', self.tracker), enabled=use_running):
                output = self.tracker.forward_transaction(transaction, builder)
            return transaction, output
        except BaseException:
            builder.abort()
            raise

    def clear_pending_materialization(self):
        self.train_builder.abort()
        self.evaluation_builder.abort()
        self._pending_train = None

    def close_raw_service(self):
        """整个 train/late-3/test 生命周期结束后才关闭，不能在 fit teardown 中关闭。"""
        self.clear_pending_materialization()
        self.raw_service.close()

    def configure_optimizers(self):
        parameters = [p for p in self.parameters() if p.requires_grad]
        if not parameters or len({id(p) for p in parameters}) != len(parameters):
            raise RuntimeError('v39 optimizer requires unique trainable parameters')
        optimizer = torch.optim.Adam(parameters, lr=self.config.lr, weight_decay=self.config.wd,
                                     betas=(.5,.999), eps=1e-6, foreach=False, fused=False)
        if self.config.lr_schedule == 'multistep':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.config.lr_milestones,
                                                             gamma=self.config.lr_decay_rate)
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.config.lr_decay_step,
                                                         gamma=self.config.lr_decay_rate)
        return dict(optimizer=optimizer, lr_scheduler=dict(scheduler=scheduler, interval='epoch'))

    def on_fit_start(self):
        self._require_lightning()
        if self.trainer.accumulate_grad_batches != 1 or self.trainer.world_size != 1:
            raise ValueError('v39 accepted transactions require one device and one Adam step per batch')
        if self.trainer.reload_dataloaders_every_n_epochs != 1:
            raise ValueError('v39 curriculum requires dataloader reload each epoch')

    def on_train_epoch_start(self):
        self.train_builder.reset()
        self._pending_train = None
        self._epoch_rows = self._epoch_steps = 0
        self._epoch_complete = False
        self._training_scalars, self._training_counts = {}, {}
        if self._pending_rng is not None:
            restore_rng_state(self._pending_rng)
            self._pending_rng = None
        if self._resume_sampler is not None:
            current = self._loaders['train'].batch_sampler.state_dict()
            for key in ('schema','lengths','source_sha256','seed','batch_size','short_window','long_window',
                'curriculum_epochs','reserve_windows','reserve_policy','drain_policy','seed_policy',
                'seed_translation','seed_yaw_degrees','initial_seed_policy'):
                if current[key] != self._resume_sampler[key]:
                    raise ValueError('v39 resume dataset/window manifest mismatch: '+key)
            self._resume_sampler = None

    def _record_training(self, transaction, output, losses):
        packet = {('loss', key): value.detach() for key,value in losses.items()
                  if torch.is_tensor(value) and value.ndim == 0}
        packet.update({('count','full_executed'):output.full_executed.sum(),
                       ('count','policy_full'):getattr(output,'policy_full',output.full_executed).sum(),
                       ('count','search_triggered'):output.search_triggered.sum()})
        copied = tensors_to_numpy(packet, optimized=True)
        count = len(transaction.rows)
        for (kind,key),value in copied.items():
            number = float(value)
            if kind == 'count':
                self._training_counts[key] = self._training_counts.get(key,0)+int(number)
            else:
                item = self._training_scalars.setdefault(key,dict(batch_sum=0.,row_weighted_batch_sum=0.,batches=0,rows=0))
                item['batch_sum']+=number; item['row_weighted_batch_sum']+=number*count
                item['batches']+=1; item['rows']+=count

    def training_step(self, rows, batch_idx):
        if self._pending_train is not None:
            raise RuntimeError('previous v39 training output was not committed')
        transaction, output = self._forward_raw(rows, self.train_builder, training=True)
        try:
            from models.ct_v39.losses import compute_losses
            losses = compute_losses(transaction, output, self.tracker, self.config)
            total = losses['loss_total']
            if total.ndim != 0 or not bool(torch.isfinite(total)):
                raise FloatingPointError('v39 loss_total must be a finite scalar')
            self._record_training(transaction,output,losses)
            self._pending_train = (output,transaction,len(rows))
            for name,value in losses.items():
                if torch.is_tensor(value) and value.ndim == 0:
                    self.log('train/'+name,value,on_step=True,on_epoch=True,batch_size=len(rows))
            return total
        except BaseException:
            self.train_builder.abort()
            self._pending_train = None
            raise

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self._pending_train is None:
            raise RuntimeError('v39 batch-end has no pending training transaction')
        output,transaction,count = self._pending_train
        try:
            self.train_builder.commit(output,transaction)
        except BaseException:
            self.train_builder.abort()
            raise
        finally:
            self._pending_train = None
        self._epoch_rows += count
        self._epoch_steps += 1

    @staticmethod
    def _write_once(path, value):
        path.parent.mkdir(parents=True,exist_ok=True)
        content=json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n'
        if path.exists():
            if path.read_text(encoding='utf-8')!=content:
                raise FileExistsError('existing v39 audit differs: '+str(path))
        else:
            with path.open('x',encoding='utf-8') as stream:
                stream.write(content)

    def on_train_epoch_end(self):
        sampler=self._loaders['train'].batch_sampler
        self._epoch_complete=(self._epoch_rows==sampler.row_count and self._epoch_steps==len(sampler)
                              and self._pending_train is None)
        if not self._epoch_complete and not self.config.ct_engineering_check:
            raise RuntimeError('v39 incomplete endpoint/update coverage')
        if self._epoch_complete and self.train_builder.states:
            raise RuntimeError('v39 completed epoch retains unfinished state windows')
        self._completed_epoch=int(self.current_epoch)+1
        if self.config.log_dir:
            root=Path(self.config.log_dir)
            audit=dict(schema=training_audit_schema(self.config),model_schema=model_schema(self.config),
                config_sha256=config_identity(self.config),run_id=self.run_id,completed_epoch=self._completed_epoch,
                epoch_complete=self._epoch_complete,rows=self._epoch_rows,optimizer_steps=self._epoch_steps,
                sampler=sampler.state_dict())
            self._write_once(root/'training_audits'/f'epoch={self._completed_epoch:03d}.json',audit)
            self._write_once(root/'training_diagnostics'/f'epoch={self._completed_epoch:03d}.json',
                dict(schema='ct_seqtrack.v39.training_diagnostics.v1',completed_epoch=self._completed_epoch,
                     scalar_semantics='batch_means_not_recomputed_global_valid_denominators',
                     scalars=self._training_scalars,counts=self._training_counts))
        self.log('train/endpoint_rows',float(self._epoch_rows),on_step=False,on_epoch=True)
        self.log('train/adam_steps',float(self._epoch_steps),on_step=False,on_epoch=True)

    def _evaluation_start(self, role):
        self.evaluation_builder.reset()
        self.evaluation=TrackingEvaluation(self.config)
        self.evaluation.add_singleton_tracks(self._loader(role).dataset)

    def on_validation_epoch_start(self):
        self._evaluation_start('val')

    def on_test_epoch_start(self):
        self._evaluation_start('test')

    def _evaluation_step(self, rows):
        transaction,output=self._forward_raw(rows,self.evaluation_builder,training=False)
        try:
            committed=self.evaluation_builder.commit(output,transaction,diagnostics=True)
            self.evaluation.add_batch(rows,transaction,output,committed)
        except BaseException:
            self.evaluation_builder.abort()
            raise

    def validation_step(self, rows, batch_idx):
        self._evaluation_step(rows)

    def test_step(self, rows, batch_idx):
        self._evaluation_step(rows)

    def _evaluation_end(self, role):
        self.evaluation_results=self.evaluation.summary()
        expected=self._loader(role).batch_sampler.endpoint_count
        complete=self.evaluation_results['prediction_frames']==expected
        self.evaluation_results['complete_coverage']=complete
        if not complete and not self.config.ct_engineering_check:
            raise RuntimeError('v39 evaluation did not consume every prediction endpoint')
        self.log('success/'+role,self.evaluation_results['success'])
        self.log('precision/'+role,self.evaluation_results['precision'])
        for key,value in self.evaluation_results['diagnostics'].items():
            if value is not None:
                self.log('diagnostic/'+role+'/'+key,float(value))

    def on_validation_epoch_end(self):
        self._evaluation_end('val')
        if self.config.log_dir:
            self._write_once(Path(self.config.log_dir)/'validation'/f'epoch={int(self.current_epoch)+1:03d}.json',
                             self.evaluation_results)

    def on_test_epoch_end(self):
        self._evaluation_end('test')

    def on_save_checkpoint(self, checkpoint):
        loader=self._loaders.get('train')
        checkpoint[runtime_key(self.config)]=resume_payload(self.config,completed_epoch=self._completed_epoch,
            complete=self._epoch_complete,rows=self._epoch_rows,steps=self._epoch_steps,
            sampler=loader.batch_sampler.state_dict() if loader is not None else None,run_id=self.run_id)

    def on_load_checkpoint(self, checkpoint):
        payload=validate_resume_payload(checkpoint.get(runtime_key(self.config)),self.config,training=not self.config.test)
        if not self.config.test and self.config.log_dir:
            path=Path(self.config.log_dir)/'run_manifest.json'
            if path.exists() and json.loads(path.read_text(encoding='utf-8')).get('run_id')!=payload['run_id']:
                raise ValueError('v39 resume requires original run_id')
        self.run_id=payload['run_id']
        self._completed_epoch=int(payload['completed_epoch'])
        if not self.config.test:
            self._pending_rng=payload['rng']
            self._resume_sampler=payload['sampler']
