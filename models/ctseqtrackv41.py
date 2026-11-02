"""v41 独立 Lightning host；保留物理模块路径与一次 forward/Adam/commit 生命周期。"""
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

from models.ct_v41.config import normalize_config, config_identity
from models.ct_v41.data import BatchBuilder, build_loaders, stable_seed
from models.ct_v41.auxiliary import (AuxiliaryDiagnostics, insertion_indices, insertion_identity,
                                    isolated_auxiliary_rng)
from models.ct_v41.lr_schedule import AbsolutePiecewiseLR, WarmupMultiStepLR
from models.ct_v41.identity import model_schema, runtime_key, training_audit_schema
from models.ct_v41.runtime import (move_tensors, TrackingEvaluation, resume_payload,
                                  restore_rng_state, validate_resume_payload)
from utils.bn_policy import running_batch_norm
from utils.execution_profile import optimized_execution
from utils.batch_materialization import attach_materialization_cache, clear_materialization_cache


class CTSEQTRACKV41(pl.LightningModule if pl is not None else nn.Module):

    def __init__(self, config=None, *, tracker=None, loaders=None):
        super().__init__()
        self.config = normalize_config(config)
        if tracker is None:
            from models.ct_v41.model import JointTracker
            tracker = JointTracker(self.config)
        self.tracker = tracker
        self.run_id = uuid.uuid4().hex
        self._loader_factory = build_loaders
        self._loaders = {} if loaders is None else loaders
        self.train_builder = BatchBuilder(self.config)
        self.auxiliary_builder = BatchBuilder(self.config)
        self.evaluation_builder = BatchBuilder(self.config)
        self.evaluation = TrackingEvaluation(self.config)
        self.evaluation_results = {}
        self._pending_train = None
        self._pending_auxiliary = None
        self._pending_auxiliary_diagnostics = None
        self.auxiliary_diagnostics = AuxiliaryDiagnostics()
        self._auxiliary_iterator = None
        self._auxiliary_insertions = ()
        self._auxiliary_by_main_step = {}
        self._auxiliary_fetches = 0
        self._auxiliary_rows = self._auxiliary_batches = 0
        self._auxiliary_complete = False
        self._main_batches = 0
        self._last_auxiliary_audit = None
        self._pending_rng = None
        self._completed_epoch = 0
        self._epoch_complete = False
        self._epoch_rows = self._epoch_steps = 0
        self._resume_sampler = None
        self._resume_auxiliary = None
        self._memory_stream = None
        self._evaluation_stream = None
        self._evaluation_session = uuid.uuid4().hex
        from models.ct_v41.training_diagnostics import TrainingDiagnostics
        self.training_diagnostics = TrainingDiagnostics(self.config.seed,
            optimized=optimized_execution(self.config)) if self.config.v41_diagnostics else None
        if pl is not None:
            self.save_hyperparameters(dict(config=dict(self.config)))

    def _require_lightning(self):
        if pl is None:
            raise RuntimeError('v41 training/evaluation requires pytorch-lightning; install requirement.txt')

    def set_loaders(self, loaders):
        self._loaders = loaders

    def _loader(self, role):
        if role not in self._loaders:
            self._loaders.update(self._loader_factory(self.config, roles=(role,)))
        loader = self._loaders[role]
        if role in ('train', 'auxiliary'):
            epoch = int(self.current_epoch)
            loader.batch_sampler.set_epoch(epoch)
            loader.generator.manual_seed(stable_seed(self.config.seed, role, epoch))
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
            builder.supervise_prior(batch, prior, training=training)
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
            builder._pending = None
            raise
        return batch, output

    def _forward_auxiliary_raw(self, rows):
        """独立状态和raw准备；模型内部仅R/B3保留梯度，主路精确恢复mode。"""
        builder, batch = self.auxiliary_builder, {}
        device = next(self.tracker.parameters()).device
        try:
            batch = move_tensors(builder.prepare(rows), device)
            batch = attach_materialization_cache(batch, enabled=optimized_execution(self.config))
            output = self.tracker.forward_aux(batch)
            # forward之后仅补被动诊断；不计算acquisition或M辅助监督。
            builder.supervise_prior(batch, output.prior, training=False)
        except BaseException:
            self._release_materialization(batch, builder)
            builder._pending = None
            raise
        return batch, output

    @staticmethod
    def _auxiliary_diagnostic_records(rows, batch, output, losses):
        """复用loss已经计算的精确IoU；不再调用几何、不进入模型输入。"""
        selected = output.selected_index.detach().long()
        exact = output.supervision_targets['exact_iou'].detach()
        selected_iou = exact.gather(1, selected[:, None]).squeeze(1)
        sampled = (batch['segmentation_labels'][:, -1] == 1).sum(-1)
        packed = torch.stack((selected_iou, sampled.to(exact),
            batch['diagnostic_search_trusted_age'].detach().to(exact),
            batch['supported_age'].detach().to(exact)), -1).cpu().tolist()
        records = []
        for row, (iou, points, trusted_age, supported_age) in zip(rows, packed):
            request = row['request']
            records.append(dict(tracklet=row['tracklet_key'], frame=request.frame,
                depth=request.frame - request.window_start,
                timestamp=float(row['frames'][request.frame]['timestamp']),
                ends_trajectory=request.ends_window, selected_exact_iou=iou,
                sampled_target_count=int(points), trusted_age_seconds=trusted_age,
                supported_age_seconds=supported_age))
        supervised = int(losses['recovery_supervised_count'].detach().cpu().item())
        return records, supervised

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
        if self._pending_auxiliary is not None:
            self._release_materialization(self._pending_auxiliary[1], self.auxiliary_builder)
        self._pending_train = None
        self._pending_auxiliary = None
        self._pending_auxiliary_diagnostics = None
        self._auxiliary_iterator = None
        for builder in (self.train_builder, self.auxiliary_builder, self.evaluation_builder):
            release = getattr(builder, 'clear_materialization_cache', None)
            if release is not None:
                release()
            builder._pending = None
        for stream in (self._memory_stream, self._evaluation_stream):
            if stream is not None:
                stream.flush()

    def on_before_optimizer_step(self, optimizer):
        """在 Adam 写入前拒绝非有限梯度，不增加梯度裁剪或改变优化器数学。"""
        gradients = [p.grad for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
        if gradients and not bool(torch.stack([torch.isfinite(g).all() for g in gradients]).all()):
            self.clear_pending_materialization()
            raise FloatingPointError('V41 nonfinite gradient; Adam update and accepted commit aborted')

    def _memory_record(self, phase, *, batch_idx=None, prepared=None, output=None):
        if not self.config.v41_memory_logging or not self.config.log_dir:
            return
        device = next(self.tracker.parameters()).device
        if device.type != 'cuda':
            return
        if self._memory_stream is None:
            path = Path(self.config.log_dir) / 'memory.jsonl'
            path.parent.mkdir(parents=True, exist_ok=True)
            self._memory_stream = path.open('a', encoding='utf-8', buffering=65536)
        stats = torch.cuda.memory_stats(device)
        record = dict(phase=phase, epoch=int(self.current_epoch) + 1, batch=batch_idx,
            allocated=torch.cuda.memory_allocated(device), reserved=torch.cuda.memory_reserved(device),
            peak_allocated=torch.cuda.max_memory_allocated(device),
            peak_reserved=torch.cuda.max_memory_reserved(device),
            inactive_split=stats.get('inactive_split_bytes.all.current', 0),
            allocation_retries=stats.get('num_alloc_retries', 0), ooms=stats.get('num_ooms', 0))
        if prepared is not None:
            counts = {'valid_points': prepared['point_valid'].sum()}
            for key in ('diagnostic_search_triggered', 'diagnostic_search_sampled_main',
                        'diagnostic_search_sampled_backup'):
                if key in prepared:
                    counts[key] = prepared[key].sum()
            if output is not None:
                tokens = getattr(output.observation, 'source_valid', None)
                if tokens is not None:
                    counts['valid_tokens'] = tokens.sum()
            record.update(zip(counts, torch.stack(list(counts.values())).detach().cpu().tolist()))
        # 此处不显式 synchronize，不保存CUDA张量；完整阶段计时由服务器工程工具完成。
        self._memory_stream.write(json.dumps(record, ensure_ascii=False) + '\n')

    def teardown(self, stage):
        self.clear_pending_materialization()
        for name in ('_memory_stream', '_evaluation_stream'):
            stream = getattr(self, name)
            if stream is not None:
                stream.close()
                setattr(self, name, None)

    def configure_optimizers(self):
        parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        if not parameters or len({id(p) for p in parameters}) != len(parameters):
            raise RuntimeError('v41 optimizer requires unique trainable parameters')
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
            scheduler.completed_epochs = int(self.current_epoch) + int(last_batch)
        scheduler.step()

    def on_fit_start(self):
        self._require_lightning()
        if self.trainer.accumulate_grad_batches != 1 or self.trainer.world_size != 1:
            raise ValueError('v41 accepted transactions require one device and one Adam step per batch')
        if self.trainer.reload_dataloaders_every_n_epochs != 1:
            raise ValueError('v41 curriculum requires reload_dataloaders_every_n_epochs=1')

    def on_train_epoch_start(self):
        self.train_builder.reset()
        self.auxiliary_builder.reset()
        self.auxiliary_diagnostics.reset()
        if self.training_diagnostics is not None:
            self.training_diagnostics.reset(int(self.current_epoch))
        self._pending_train = None
        self._pending_auxiliary = None
        self._pending_auxiliary_diagnostics = None
        self._epoch_rows = self._epoch_steps = 0
        self._auxiliary_rows = self._auxiliary_batches = self._auxiliary_fetches = 0
        self._auxiliary_complete = False
        self._epoch_complete = False
        if self._pending_rng is not None:
            restore_rng_state(self._pending_rng)
            self._pending_rng = None
        if self._resume_sampler is not None:
            current = self._loaders['train'].batch_sampler.state_dict()
            identity_keys = ['schema', 'lengths', 'source_sha256', 'seed', 'batch_size',
                'short_window', 'long_window', 'curriculum_epochs', 'reserve_windows',
                'reserve_policy', 'drain_policy', 'seed_policy', 'seed_translation',
                'seed_yaw_degrees', 'initial_seed_policy']
            for key in identity_keys:
                if current[key] != self._resume_sampler[key]:
                    raise ValueError('v41 resume dataset/window manifest mismatch: ' + key)
            self._resume_sampler = None
        main = self._loaders['train'].batch_sampler
        with isolated_auxiliary_rng(self.config.seed, int(self.current_epoch), -1):
            auxiliary_loader = self._loader('auxiliary')
        auxiliary = auxiliary_loader.batch_sampler
        if (tuple(auxiliary.lengths) != tuple(main.lengths)
                or auxiliary.source_sha256 != main.source_sha256):
            raise ValueError('v41 auxiliary stream must use the same training trajectories and metadata')
        if self._resume_auxiliary is not None:
            # checksum之外重建已完成epoch，不能接受人为重签但不同的整轨选择。
            saved = self._resume_auxiliary
            epoch = auxiliary.epoch
            try:
                auxiliary.set_epoch(int(saved['sampler']['epoch']))
                if auxiliary.state_dict() != saved['sampler']:
                    raise ValueError('v41 resume auxiliary dataset/selection/plan mismatch')
                indices = insertion_indices(saved['main_batches'], len(auxiliary))
                if (list(indices) != saved['insertion_indices']
                        or insertion_identity(indices) != saved['insertion_plan_sha256']):
                    raise ValueError('v41 resume auxiliary insertion plan mismatch')
            finally:
                auxiliary.set_epoch(epoch)
            self._resume_auxiliary = None
        self._main_batches = len(main)
        self._auxiliary_insertions = insertion_indices(self._main_batches, len(auxiliary))
        self._auxiliary_by_main_step = {step: j for j, step in enumerate(self._auxiliary_insertions)}
        with isolated_auxiliary_rng(self.config.seed, int(self.current_epoch), -2):
            self._auxiliary_iterator = iter(auxiliary_loader)

    def training_step(self, rows, batch_idx):
        if self._pending_train is not None or self._pending_auxiliary is not None:
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
                raise FloatingPointError('v41 loss_total must be a finite scalar')
            if self.training_diagnostics is not None:
                self.training_diagnostics.add_batch(rows, batch, output, losses)
            self._pending_train = (output, batch, len(rows))
            if batch_idx in self._auxiliary_by_main_step:
                auxiliary_index = self._auxiliary_by_main_step[batch_idx]
                if auxiliary_index != self._auxiliary_fetches:
                    raise RuntimeError('auxiliary batches must follow the registered insertion order')
                with isolated_auxiliary_rng(self.config.seed, int(self.current_epoch), auxiliary_index):
                    try:
                        auxiliary_rows = next(self._auxiliary_iterator)
                    except StopIteration as error:
                        raise RuntimeError('auxiliary loader ended before its registered plan') from error
                    auxiliary_batch, auxiliary_output = self._forward_auxiliary_raw(auxiliary_rows)
                    self._pending_auxiliary = (auxiliary_output, auxiliary_batch, len(auxiliary_rows))
                    auxiliary_losses = self.tracker.compute_aux_losses(auxiliary_batch, auxiliary_output)
                    self._pending_auxiliary_diagnostics = self._auxiliary_diagnostic_records(
                        auxiliary_rows, auxiliary_batch, auxiliary_output, auxiliary_losses)
                    auxiliary_total = auxiliary_losses['loss_total']
                    if auxiliary_total.ndim != 0 or not bool(torch.isfinite(auxiliary_total)):
                        raise FloatingPointError('v41 auxiliary loss_total must be a finite scalar')
                    total = total + self.config.v41_aux_loss_weight * auxiliary_total
                    self._auxiliary_fetches += 1
                    for name, value in auxiliary_losses.items():
                        if torch.is_tensor(value) and value.ndim == 0:
                            self.log('train_aux/' + name, value.detach(), on_step=True,
                                     on_epoch=True, batch_size=len(auxiliary_rows))
            if not bool(torch.isfinite(total)):
                raise FloatingPointError('v41 combined optimization loss must be finite')
            self.log('train/optimization_loss', total.detach(), on_step=True, on_epoch=True,
                     batch_size=len(rows))
            for name, value in losses.items():
                if torch.is_tensor(value) and value.ndim == 0:
                    self.log('train/' + name, value.detach(), on_step=True, on_epoch=True, batch_size=len(rows))
        except BaseException:
            self._release_materialization(batch, self.train_builder)
            self.train_builder._pending = None
            self.clear_pending_materialization()
            raise
        # 普通 mean loss、每批一次标准 Adam；不声称小批按样本数缩放
        # loss 可以同比缩小 Adam 更新，也不临时改变 LR/scheduler。
        return total

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Lightning automatic optimization has completed backward + Adam here.
        output, prepared, count = self._pending_train
        auxiliary = self._pending_auxiliary
        try:
            self.train_builder.commit(output, prepared)
            if auxiliary is not None:
                auxiliary_output, auxiliary_prepared, auxiliary_count = auxiliary
                self.auxiliary_builder.commit(auxiliary_output, auxiliary_prepared)
                self.auxiliary_diagnostics.add(*self._pending_auxiliary_diagnostics)
                self._auxiliary_rows += auxiliary_count
                self._auxiliary_batches += 1
            self._memory_record('train_commit', batch_idx=batch_idx, prepared=prepared, output=output)
        except BaseException:
            self._epoch_complete = self._auxiliary_complete = False
            self.train_builder._pending = None
            self.auxiliary_builder._pending = None
            raise
        finally:
            self._release_materialization(prepared, self.train_builder)
            if auxiliary is not None:
                self._release_materialization(auxiliary[1], self.auxiliary_builder)
            self._pending_train = None
            self._pending_auxiliary = None
            self._pending_auxiliary_diagnostics = None
        del output, prepared
        if auxiliary is not None:
            del auxiliary_output, auxiliary_prepared
        auxiliary = None
        self._memory_record('train_cleanup', batch_idx=batch_idx)
        self._epoch_rows += count
        self._epoch_steps += 1

    def on_train_epoch_end(self):
        self._memory_record('train_epoch_end')
        if self._memory_stream is not None:
            self._memory_stream.flush()
        sampler = self._loaders['train'].batch_sampler
        auxiliary = self._loaders['auxiliary'].batch_sampler
        expected = sampler.row_count
        self._auxiliary_complete = (self._auxiliary_rows == auxiliary.row_count
            and self._auxiliary_batches == len(auxiliary)
            and self._auxiliary_fetches == len(auxiliary)
            and self._pending_auxiliary is None and not self.auxiliary_builder.states)
        self._epoch_complete = (self._epoch_rows == expected and self._epoch_steps == len(sampler)
            and self._pending_train is None and self._auxiliary_complete)
        if not self._epoch_complete and not self.config.ct_engineering_check:
            raise RuntimeError(f'v41 incomplete main/aux coverage: main={self._epoch_rows}/{expected}, '
                f'aux={self._auxiliary_rows}/{auxiliary.row_count}, '
                f'aux_batches={self._auxiliary_batches}/{len(auxiliary)}')
        if self._epoch_complete and self.train_builder.states:
            raise RuntimeError('v41 completed epoch retains unfinished windows')
        self._completed_epoch = int(self.current_epoch) + 1
        self._last_auxiliary_audit = self._auxiliary_audit()
        self._write_epoch_audit(sampler)
        self._auxiliary_iterator = None
        if self.training_diagnostics is not None:
            self.training_diagnostics.write_epoch(self.config.log_dir,
                completed_epoch=self._completed_epoch, complete=self._epoch_complete)
        self.log('train/endpoint_rows', float(self._epoch_rows), on_step=False, on_epoch=True)
        self.log('train/adam_steps', float(self._epoch_steps), on_step=False, on_epoch=True)
        self.log('train_aux/endpoint_rows', float(self._auxiliary_rows), on_step=False, on_epoch=True)
        self.log('train_aux/batches', float(self._auxiliary_batches), on_step=False, on_epoch=True)
        diagnostics = self.auxiliary_diagnostics.state_dict()
        for name in ('depth_ge8_endpoints', 'max_depth', 'loss_episodes', 'recoveries',
                     'reappearance_training_endpoints', 'xyz_supervised_endpoints'):
            self.log('train_aux/' + name, float(diagnostics[name]), on_step=False, on_epoch=True)

    def _auxiliary_audit(self):
        loader = self._loaders.get('auxiliary')
        return dict(schema='ct_seqtrack.v41.auxiliary_audit.v1', complete=self._auxiliary_complete,
            rows=self._auxiliary_rows, batches=self._auxiliary_batches, main_batches=self._main_batches,
            sampler=loader.batch_sampler.state_dict() if loader is not None else None,
            insertion_indices=list(self._auxiliary_insertions),
            insertion_plan_sha256=insertion_identity(self._auxiliary_insertions),
            loss_weight=self.config.v41_aux_loss_weight,
            diagnostics=self.auxiliary_diagnostics.state_dict())

    def _write_epoch_audit(self, sampler):
        if not self.config.log_dir:
            return
        audit = dict(schema=training_audit_schema(self.config),
            model_schema=model_schema(self.config), config_sha256=config_identity(self.config),
            run_id=self.run_id, completed_epoch=self._completed_epoch,
            epoch_complete=self._epoch_complete, rows=self._epoch_rows,
            optimizer_steps=self._epoch_steps, sampler=sampler.state_dict(),
            auxiliary=self._auxiliary_audit())
        directory = Path(self.config.log_dir) / 'training_audits'
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f'epoch={self._completed_epoch:03d}.json'
        content = json.dumps(audit, ensure_ascii=False, sort_keys=True, indent=2) + '\n'
        if path.exists():
            if path.read_text(encoding='utf-8') != content:
                raise FileExistsError('existing v41 epoch audit differs: ' + str(path))
        else:
            with path.open('x', encoding='utf-8') as stream:
                stream.write(content)

    def _evaluation_start(self, role):
        self.evaluation_builder.reset()
        self.evaluation = TrackingEvaluation(self.config)
        if role in self._loaders:
            self.evaluation.add_singleton_tracks(self._loaders[role].dataset)
        if self._evaluation_stream is not None:
            self._evaluation_stream.close()
            self._evaluation_stream = None
        if self.config.log_dir:
            epoch = (self.config.eval_checkpoint_epoch or self._completed_epoch) if role == 'test' else int(self.current_epoch) + 1
            # 每次启动/恢复独立会话；从周期checkpoint重跑中断后的轮次不覆盖旧流。
            directory = Path(self.config.log_dir) / 'frame_streams' / role / self._evaluation_session
            directory.mkdir(parents=True, exist_ok=True)
            self._evaluation_stream = (directory / f'epoch={epoch:03d}.jsonl').open('x', encoding='utf-8')
            for row in self.evaluation.rows:
                self._evaluation_stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        self._memory_record(role + '_start')

    def on_validation_epoch_start(self):
        self._evaluation_start('val')

    def on_test_epoch_start(self):
        self._evaluation_start('test')

    def _evaluation_step(self, rows):
        batch, output = self._forward_raw(rows, self.evaluation_builder, training=False)
        try:
            first = len(self.evaluation.rows)
            committed = self.evaluation_builder.commit(output, batch, diagnostics=True)
            self.evaluation.add_batch(rows, batch, output, committed)
            if self._evaluation_stream is not None:
                for row in self.evaluation.rows[first:]:
                    self._evaluation_stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            self._memory_record('evaluation_commit', prepared=batch, output=output)
        finally:
            self._release_materialization(batch, self.evaluation_builder)

    def validation_step(self, rows, batch_idx):
        self._evaluation_step(rows)

    def test_step(self, rows, batch_idx):
        self._evaluation_step(rows)

    def _evaluation_end(self, role):
        if self._evaluation_stream is not None:
            self._evaluation_stream.close()
            self._evaluation_stream = None
        self._memory_record(role + '_end')
        self.evaluation_results = self.evaluation.summary()
        if role in self._loaders:
            expected = self._loaders[role].batch_sampler.endpoint_count
            complete = self.evaluation_results['prediction_frames'] == expected
            self.evaluation_results['complete_coverage'] = complete
            if not complete and not self.config.ct_engineering_check:
                raise RuntimeError('v41 evaluation did not consume every prediction endpoint')
        self.log('success/' + role, self.evaluation_results['success'])
        self.log('precision/' + role, self.evaluation_results['precision'])
        for key, value in self.evaluation_results['diagnostics'].items():
            if value is not None:
                self.log('diagnostic/' + role + '/' + key, float(value))

    def on_validation_epoch_end(self):
        self._evaluation_end('val')
        if self.config.log_dir:
            directory = Path(self.config.log_dir) / 'validation'
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f'epoch={int(self.current_epoch) + 1:03d}.json'
            content = json.dumps(self.evaluation_results, ensure_ascii=False, sort_keys=True, indent=2) + '\n'
            if path.exists():
                if path.read_text(encoding='utf-8') != content:
                    raise FileExistsError('existing v41 validation differs: ' + str(path))
            else:
                with path.open('x', encoding='utf-8') as stream:
                    stream.write(content)

    def on_test_epoch_end(self):
        self._evaluation_end('test')

    def on_save_checkpoint(self, checkpoint):
        loader = self._loaders.get('train')
        checkpoint[runtime_key(self.config)] = resume_payload(self.config,
            completed_epoch=self._completed_epoch, complete=self._epoch_complete,
            rows=self._epoch_rows, steps=self._epoch_steps,
            sampler=loader.batch_sampler.state_dict() if loader is not None else None,
            run_id=self.run_id, auxiliary=self._auxiliary_audit())

    def on_load_checkpoint(self, checkpoint):
        payload = validate_resume_payload(checkpoint.get(runtime_key(self.config)), self.config,
                                          training=not self.config.test)
        if not self.config.test and self.config.log_dir:
            manifest = Path(self.config.log_dir) / 'run_manifest.json'
            if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8')).get('run_id') != payload['run_id']:
                raise ValueError('v41 resume requires the original run_id')
        self.run_id = payload['run_id']
        self._completed_epoch = int(payload['completed_epoch'])
        self._last_auxiliary_audit = payload['auxiliary']
        if not self.config.test:
            self._pending_rng = payload['rng']
            self._resume_sampler = payload['sampler']
            self._resume_auxiliary = payload['auxiliary']
