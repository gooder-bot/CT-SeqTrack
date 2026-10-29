"""上传后显式运行的 V40 真实服务器验收；每个 CUDA 配置由独立进程启动。"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.ct_v40.engineering import (ENGINEERING_SCHEMA, acceptance_environment,
    analyze_memory_cycles, capacity_gate, digest_tree, memory_sample, NVMLMonitor,
    require, timing_summary, write_new_json, CPUReplaySnapshot, cpu_copy)

PHASES = ('contracts', 'kernel', 'lifecycle', 'memory', 'performance', 'parallel')
WAVES = (('b0', 'b1'), ('b1_mtp', 'full'))


def identity():
    from models.ct_v40.registration import verify_registration
    value = verify_registration()
    return dict(source=value['source'], engineering_files=value['engineering_files'],
                registration_sha256=hashlib.sha256((ROOT / 'cfgs/ct_seqtrack/40_registration.json').read_bytes()).hexdigest())


class KernelAudit:
    """只留 CPU 摘要；包装实际扩展函数，不能由 backend 配置冒充 kernel 证据。"""
    def __init__(self, model):
        import models.ct_v40.attention as attention
        self.module, self.original = attention, attention._flash_varlen
        self.records, self.total = [], 0
        self.model, self.handles = model, []

    def __enter__(self):
        from models.ct_v40.attention import EfficientAttention, ValidityMask
        for name, module in self.model.named_modules():
            if not isinstance(module, EfficientAttention):
                continue
            def mask_hook(core, args, kwargs, module_name=name):
                mask = kwargs.get('mask', args[3] if len(args) > 3 else None)
                packed = ((mask.query_valid, mask.key_valid, mask.query_indices, mask.key_indices)
                          if isinstance(mask, ValidityMask) else mask)
                self.records.append(dict(kind='attention_mask', module=module_name,
                                         mask_and_indices_sha256=digest_tree(packed)))
            self.handles.append(module.register_forward_pre_hook(mask_hook, with_kwargs=True))
        def getter():
            actual = self.original()
            def call(*args, **kwargs):
                import torch
                q, k, v, cq, ck, mq, mk = args[:7]
                require(all(t.is_cuda and t.dtype == torch.bfloat16 for t in (q, k, v)),
                        'FA2 kernel did not receive CUDA BF16 QKV')
                require(all(t.is_cuda and t.dtype == torch.int32 for t in (cq, ck)), 'FA2 cu_seqlens must be CUDA int32')
                require(kwargs.get('deterministic') is True and kwargs.get('causal') is False,
                        'FA2 deterministic/noncausal contract differs')
                for cu, tensor, maximum in ((cq, q, mq), (ck, k, mk)):
                    values = cu.detach().cpu().tolist()
                    require(values[0] == 0 and values[-1] == len(tensor)
                            and all(b >= a for a, b in zip(values, values[1:]))
                            and max((b - a for a, b in zip(values, values[1:])), default=0) == maximum,
                            'invalid packed FA2 layout')
                record = dict(shapes=[list(t.shape) for t in (q, k, v)],
                    cu_sha256=digest_tree((cq, ck)), maximum=[mq, mk],
                    dropout=kwargs['dropout_p'], scale=kwargs['softmax_scale'])
                result = actual(*args, **kwargs)
                require(result.dtype == torch.bfloat16, 'FA2 kernel output was not BF16')
                self.records.append(record)
                self.total += 1
                return result
            return call
        self.module._flash_varlen = getter
        return self

    def __exit__(self, *unused):
        self.module._flash_varlen = self.original
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def prepared_fingerprint(batch, rows, kernels):
    import torch
    # 所有实际 tensor 输入，包括 point_valid、history masks、ROI/future labels；cache 是派生瞬时对象。
    inputs = {key: value for key, value in batch.items() if torch.is_tensor(value)}
    return digest_tree(dict(inputs=inputs, requests=[row['request'] for row in rows], kernels=kernels))


class Workload:
    def __init__(self, spec):
        import torch
        import pytorch_lightning as pl
        from models.ct_v40.config import load_config
        from models.ct_v40.entry import configure_numerics
        from models.ct_v40.data import build_loaders
        from models.ctseqtrackv40 import CTSEQTRACKV40
        config = load_config(spec['cfg'], dict(path=spec['path'], ct_engineering_check=True,
            log_dir=None, v31_evaluate_late3=False, v40_attention_backend=spec.get('backend', 'fa2_varlen')))
        require(config.point_sample_size == 1024 and config.batch_size == 16 and config.workers == 4,
                'real server acceptance requires registered 1024 points, batch16, workers4')
        if spec['phase'] in ('kernel', 'memory', 'performance', 'performance_capture'):
            require(config.v40_arm == 'full', spec['phase'] + ' acceptance requires the Full arm')
        configure_numerics(config)
        pl.seed_everything(config.seed, workers=True)
        require(torch.cuda.is_available(), 'server acceptance requires CUDA')
        from models.ct_v40.attention import validate_fa_runtime
        self.fa_runtime = validate_fa_runtime()
        self.loaders = build_loaders(config, roles=('train', 'val'))
        self.loaders['train'].batch_sampler.set_epoch(10)
        self.plans = {role: list(loader.batch_sampler) for role, loader in self.loaders.items()}
        require(len(self.plans['train']) >= 304, 'mature real loader needs at least 304 batches')
        require(all(len(plan) == 16 for plan in self.plans['train'][:304]), 'first 304 mature batches must be full batch16')
        require(all(request.epoch == 10 for plan in self.plans['train'][:304] for request in plan), 'mature epoch mismatch')
        self.config = config
        self.model = CTSEQTRACKV40(config, loaders=self.loaders).cuda().train()
        self.optimizer = self.model.configure_optimizers()['optimizer']
        self.model.train_builder.reset()
        self.model.evaluation_builder.reset()
        if self.model.training_diagnostics is not None:
            self.model.training_diagnostics.reset(10)
        self.audit = KernelAudit(self.model)
        self.coverage = {}

    def loader(self, role, plans):
        import torch
        from models.ct_v40.data import raw_collate, stable_seed
        return torch.utils.data.DataLoader(self.loaders[role].dataset, batch_sampler=plans,
            num_workers=4, pin_memory=False, prefetch_factor=2, persistent_workers=False,
            collate_fn=raw_collate, generator=torch.Generator().manual_seed(stable_seed(self.config.seed, role, 10)))

    def transaction(self, rows, index, role, *, samples, fingerprints, instrument):
        import torch
        training = role != 'val'
        builder = self.model.train_builder if training else self.model.evaluation_builder
        self.audit.records = []
        def sample(name):
            if instrument:
                samples.append(memory_sample(name, index, role=role))
        self.optimizer.zero_grad(set_to_none=True)
        sample('before_prepare')
        batch = output = losses = loss = None
        try:
            with torch.set_grad_enabled(training):
                batch, output = self.model._forward_raw(rows, builder, training=training)
                if getattr(self, 'capture_batches', None) is not None:
                    self.capture_batches.append(cpu_copy({key: value for key, value in batch.items() if torch.is_tensor(value)}))
                if instrument:
                    self.coverage['empty_sequences'] = self.coverage.get('empty_sequences', 0) + int((~output.observation.sequence_valid).sum().item())
                    self.coverage['drain_rows'] = self.coverage.get('drain_rows', 0) + sum(bool(row['request'].drain) for row in rows)
                    self.coverage['partial_batches'] = self.coverage.get('partial_batches', 0) + int(len(rows) < 16)
                    current_counts = batch['point_valid'][:, -1].sum(-1)
                    self.coverage['current_sparse_1_2'] = self.coverage.get('current_sparse_1_2', 0) + int(((current_counts > 0) & (current_counts <= 2)).sum().item())
                    if 'diagnostic_search_triggered' in batch:
                        self.coverage['guard_rows'] = self.coverage.get('guard_rows', 0) + int(batch['diagnostic_search_triggered'].sum().item())
                sample('after_forward')
                if fingerprints is not None:
                    fingerprints.append(prepared_fingerprint(batch, rows, self.audit.records))
                if training:
                    if self.model.training_diagnostics is not None:
                        output.loss_statistics = {}
                    losses = self.model.tracker.compute_losses(batch, output)
                    loss = losses['loss_total']
                    require(loss.ndim == 0 and bool(torch.isfinite(loss)), 'nonfinite or nonscalar real loss')
                    if self.model.training_diagnostics is not None:
                        self.model.training_diagnostics.add_batch(rows, batch, output, losses)
                    self.model._pending_train = (output, batch, len(rows))
                    loss.backward()
                    sample('after_backward')
                    self.model.on_before_optimizer_step(self.optimizer)
                    self.optimizer.step()
                    sample('after_adam')
                    self.model.on_train_batch_end(None, rows, index)
                else:
                    committed = builder.commit(output, batch, diagnostics=True)
                    self.model.evaluation.add_batch(rows, batch, output, committed)
                    self.model._release_materialization(batch, builder)
                    del committed
                sample('after_commit')
        except BaseException:
            self.model.clear_pending_materialization()
            raise
        finally:
            batch = output = losses = loss = None
            self.optimizer.zero_grad(set_to_none=True)
        require(self.model._pending_train is None and builder._pending is None, 'transaction did not clear pending state')
        sample('after_cleanup')

    def segment(self, role, plans, *, instrument=False, fingerprints=None):
        import torch
        self.model.train(role != 'val')
        samples, timings, raw_fingerprints, endpoints = [], [], [], 0
        iterator = iter(self.loader('val' if role == 'val' else 'train', plans))
        for index in range(len(plans)):
            torch.cuda.synchronize()
            start = time.perf_counter()
            rows = next(iterator)
            self.transaction(rows, index, role, samples=samples, fingerprints=fingerprints, instrument=instrument)
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - start)
            endpoints += len(rows)
            raw_fingerprints.append(digest_tree([row['request'] for row in rows]))
            del rows
        del iterator
        return dict(samples=samples, timings=timings, endpoints=endpoints,
                    request_sha256=digest_tree(raw_fingerprints))

    def warmup(self):
        self.segment('warmup', self.plans['train'][:32])

    def cycle(self, *, instrument, fingerprint):
        import numpy as np
        import torch
        from models.ct_v40.runtime import TrackingEvaluation
        self.coverage = {}
        retries_before = torch.cuda.memory_stats().get('num_alloc_retries', 0)
        hashes = [] if fingerprint else None
        fixture_chunks = []
        if self.config.v40_arm == 'full':
            # 明确的工程受控变体，不声称真实 mini 恰好覆盖全空和 guard。
            saved_states = cpu_copy(self.model.train_builder.states)
            self.model.train_builder.reset()
            self.model.train()
            iterator = iter(self.loader('train', self.plans['train'][:2]))
            samples = []
            rows = next(iterator)
            for row in rows:
                row['frames'] = {key: dict(frame, pc=np.empty((0, 3)), point_ids=np.empty(0, dtype=np.int64))
                                 for key, frame in row['frames'].items()}
            self.transaction(rows, 0, 'controlled_empty', samples=samples, fingerprints=hashes, instrument=instrument)
            for state in self.model.train_builder.states.values():
                state.previous_measurement_eligible = False
            rows = next(iterator)
            self.transaction(rows, 0, 'controlled_guard', samples=samples, fingerprints=hashes, instrument=instrument)
            del rows, iterator
            self.model.train_builder.reset()
            # 独立单行/两点变体覆盖稀疏与 partial-BN 路径；点槽预算仍为1024。
            sparse_row = next(iter(self.loader('train', self.plans['train'][:1])))[0]
            initial = self.model.train_builder._new_state(sparse_row)
            anchor = initial.boxes[sparse_row['request'].frame - 1][:3]
            sparse_row['frames'] = {key: dict(frame,
                pc=np.stack((anchor, anchor + np.array([.05, 0., 0.]))), point_ids=np.array([0, 1], dtype=np.int64))
                for key, frame in sparse_row['frames'].items()}
            self.transaction([sparse_row], 0, 'controlled_sparse_partial', samples=samples,
                             fingerprints=hashes, instrument=instrument)
            del sparse_row, initial
            self.model.train_builder.reset()
            self.model.train_builder.states.update(saved_states)
            fixture_chunks = samples
        chunks = [self.segment('train', self.plans['train'][32:], instrument=instrument, fingerprints=hashes)]
        self.model.evaluation_builder.reset()
        self.model.evaluation = TrackingEvaluation(self.config)
        self.model.evaluation.add_singleton_tracks(self.loaders['val'].dataset)
        chunks.append(self.segment('val', self.plans['val'], instrument=instrument, fingerprints=hashes))
        summary = self.model.evaluation.summary()
        require(summary['prediction_frames'] == self.loaders['val'].batch_sampler.endpoint_count,
                'validation did not consume complete real endpoint coverage')
        require(not self.model.train_builder.states, 'complete mature epoch retained unfinished windows')
        self.loaders['train'].batch_sampler.set_epoch(11)
        resumed_plan = list(self.loaders['train'].batch_sampler)[:16]
        self.loaders['train'].batch_sampler.set_epoch(10)
        chunks.append(self.segment('train_resume', resumed_plan, instrument=instrument, fingerprints=hashes))
        def measured(chunk):
            summary = timing_summary(chunk['timings'])
            return dict(summary, endpoint_rows=chunk['endpoints'],
                        endpoints_per_second=chunk['endpoints'] / summary['total_seconds'],
                        latency_unit='batch; partial batches retained, not divided to invent per-frame latency')
        return dict(workload_sha256=digest_tree(hashes) if fingerprint else None,
            request_sha256=digest_tree([chunk['request_sha256'] for chunk in chunks]),
            samples=fixture_chunks + [row for chunk in chunks for row in chunk['samples']],
            train=measured(chunks[0]), validation=measured(chunks[1]),
            resumed_train=measured(chunks[2]), validation_frames=summary['prediction_frames'],
            coverage=dict(self.coverage), allocation_retries_delta=torch.cuda.memory_stats().get('num_alloc_retries', 0) - retries_before,
            controlled_variants=['all point clouds empty', 'previous_measurement_eligible forced False',
                'one row with two synthetic points around lawful initialization anchor; unchanged 1024 slots'] if self.config.v40_arm == 'full' else [])


class SyntheticResumeSource:
    """仅用于 entry/Lightning 恢复合同；annotation API 不构造未来点云。"""
    def get_num_tracklets(self):
        return 2

    def get_num_frames_tracklet(self, index):
        return (10, 6)[index]

    def get_tracklet_key(self, index):
        return 'v40_resume_fixture/' + str(index)

    def get_frame_metadata(self, index, frame):
        import numpy as np
        return dict(timestamp=100. + frame * .5, scene_id='synthetic_resume',
                    box_size=np.array([4., 2., 2.]), **{'3d_bbox': np.array([frame * .2, index * 20., 0., .1])})

    def get_frames(self, index, frame_ids):
        import numpy as np
        result = []
        for frame in frame_ids:
            meta = self.get_frame_metadata(index, frame)
            cloud = np.array([[0., 0., 0.], [.2, .1, 0.], [-.2, -.1, 0.],
                              [0., -.2, .1], [5., 0., 0.], [8., 0., 0.]]) + meta['3d_bbox'][:3]
            result.append(dict(meta, pc=cloud, point_ids=np.arange(len(cloud))))
        return result


def entry_resume_worker(spec):
    import torch
    import pytorch_lightning as pl
    from models.ct_v40.config import load_config
    from models.ct_v40.data import build_loaders
    from models.ct_v40.entry import run
    from models.ct_v40.identity import RUNTIME_KEY
    registered = identity()
    config = load_config(spec['cfg'], dict(ct_engineering_check=True, log_dir=spec['run_dir'],
        checkpoint=spec.get('checkpoint'), epoch=21, accelerator='gpu', v31_evaluate_late3=False))
    sources = {role: SyntheticResumeSource() for role in ('train', 'val')}
    loaders = build_loaders(config, roles=tuple(sources), sources=sources)
    records, final = [], {}
    class AuditAndStop(pl.Callback):
        def on_train_batch_start(self, trainer, module, rows, batch_idx):
            records.append(dict(epoch=trainer.current_epoch + 1, global_step=trainer.global_step,
                lr=trainer.optimizers[0].param_groups[0]['lr'],
                request_sha256=digest_tree([row['request'] for row in rows]),
                branch3_window_lengths=sorted({row['request'].window_end - row['request'].window_start
                    for row in rows if row['request'].branch == 3})))

        def on_train_epoch_end(self, trainer, module):
            if trainer.current_epoch + 1 == spec['stop_epoch']:
                trainer.should_stop = True

        def on_train_end(self, trainer, module):
            final.update(model_sha256=digest_tree(module.state_dict()),
                optimizer_sha256=digest_tree(trainer.optimizers[0].state_dict()),
                scheduler_sha256=digest_tree(trainer.lr_scheduler_configs[0].scheduler.state_dict()),
                global_step=trainer.global_step)
    actual_trainer = pl.Trainer
    class AuditedTrainer(actual_trainer):
        def __init__(self, *args, **kwargs):
            kwargs['callbacks'] = list(kwargs.get('callbacks', [])) + [AuditAndStop()]
            super().__init__(*args, **kwargs)
    pl.Trainer = AuditedTrainer
    try:
        run(config, loaders=loaders)
    finally:
        pl.Trainer = actual_trainer
    checkpoint = Path(spec['run_dir']) / 'formal_checkpoints' / f"epoch={spec['stop_epoch']:03d}.ckpt"
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    payload = saved[RUNTIME_KEY]
    require(payload['epoch_complete'] and payload['completed_epoch'] == spec['stop_epoch'], 'entry checkpoint is not a complete requested epoch')
    require(payload['rows'] == payload['sampler']['rows'] and payload['optimizer_steps'] == payload['sampler']['batches'],
            'entry resume fixture did not consume a complete actual sampler')
    require(identity() == registered, 'source changed during entry resume')
    return dict(passed=True, phase='entry_resume', identity=registered, pid=os.getpid(),
        stop_epoch=spec['stop_epoch'], checkpoint=str(checkpoint), final=final, order=records,
        data='synthetic annotation-only future metadata; real CUDA model, real entry.run and Lightning, complete small epochs; not nuScenes score evidence')


def entry_resume_gate(directory, common):
    directory = Path(directory)
    shared = dict(common, phase='entry_resume', allocator='round2')
    reference = finish_worker(start_worker(directory, 'entry-uninterrupted', dict(shared,
        run_dir=str(directory / 'synthetic-uninterrupted'), stop_epoch=21)))
    parts, checkpoint = [], None
    for stop in (10, 20, 21):
        part = finish_worker(start_worker(directory, f'entry-resume-{stop}', dict(shared,
            run_dir=str(directory / 'synthetic-resumed'), stop_epoch=stop, checkpoint=checkpoint)))
        parts.append(part)
        checkpoint = part['checkpoint']
    order = [row for part in parts for row in part['order']]
    require(order == reference['order'], 'resumed raw request order/global step/LR differs from uninterrupted entry')
    require(parts[-1]['final'] == reference['final'], 'resumed model/BN/Adam/scheduler differs from uninterrupted entry')
    lengths = lambda epoch: [length for row in order if row['epoch'] == epoch for length in row['branch3_window_lengths']]
    require(max(lengths(1)) == 1 and max(lengths(10)) == max(lengths(11)) == 8,
            'actual curriculum or mature e10/e11 resume boundary differs')
    for row in order:
        expected = 1e-4 if row['epoch'] <= 20 else 2e-5
        require(abs(row['lr'] - expected) <= 1e-12, 'actual e20/e21 LR transition differs')
    return dict(passed=True, reference=reference, resumed_parts=parts,
        comparisons=['full batch order', 'learning rates', 'FP32 parameters and BN buffers', 'Adam tensors/step', 'scheduler'],
        fixture='synthetic complete small epochs; separate from real-mini eight-batch lifecycle')


def performance_capture(spec):
    import torch
    registered = identity()
    workload = Workload(spec)
    workload.warmup()
    workload.capture_batches = []
    workload.segment('capture', workload.plans['train'][32:40])
    payload = dict(batches=workload.capture_batches, model=cpu_copy(workload.model.state_dict()),
                   optimizer=cpu_copy(workload.optimizer.state_dict()))
    path = Path(spec['result']).with_suffix('.corpus.pt')
    with path.open('xb') as stream:
        torch.save(payload, stream)
    require(identity() == registered, 'source changed during performance input capture')
    return dict(passed=True, phase='performance_capture', identity=registered, pid=os.getpid(),
        corpus=str(path), corpus_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        prepared_sha256=digest_tree(payload['batches']), model_sha256=digest_tree(payload['model']), batches=len(payload['batches']))


def performance_fixed(spec):
    import torch
    import pytorch_lightning as pl
    from models.ct_v40.config import load_config
    from models.ct_v40.entry import configure_numerics
    from models.ct_v40.runtime import move_tensors
    from models.ctseqtrackv40 import CTSEQTRACKV40
    from utils.batch_materialization import attach_materialization_cache, clear_materialization_cache
    registered = identity()
    path = Path(spec['corpus'])
    require(hashlib.sha256(path.read_bytes()).hexdigest() == spec['corpus_sha256'], 'fixed input corpus changed')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    config = load_config(spec['cfg'], dict(ct_engineering_check=True, log_dir=None,
        v31_evaluate_late3=False, v40_attention_backend=spec['backend']))
    require(config.v40_arm == 'full', 'fixed-input performance requires the Full arm')
    configure_numerics(config)
    pl.seed_everything(config.seed, workers=True)
    model = CTSEQTRACKV40(config).cuda().train()
    model.load_state_dict(payload['model'], strict=True)
    optimizer = model.configure_optimizers()['optimizer']
    optimizer.load_state_dict(payload['optimizer'])
    optimizer.zero_grad(set_to_none=True)
    snapshot = CPUReplaySnapshot(model, optimizer)
    timings, losses, memory = [], [], []
    for repeat in range(13):
        for index, saved in enumerate(payload['batches']):
            # 每一个测量step都从同一参数/BN/Adam/RNG开始，恢复和输入搬运不进入计时。
            snapshot.restore(model, optimizer)
            batch = attach_materialization_cache(move_tensors(saved, torch.device('cuda')), enabled=True)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            prior = model.tracker.plan_prior(batch)
            output = model.tracker(batch, prior=prior)
            loss = model.tracker.compute_losses(batch, output)['loss_total']
            loss.backward()
            model.on_before_optimizer_step(optimizer)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            require(bool(torch.isfinite(loss)), 'nonfinite fixed-input training loss')
            if repeat:
                timings.append(elapsed)
                losses.append(float(loss.detach().cpu()))
                memory.append(memory_sample('fixed_step', index))
            clear_materialization_cache(batch)
            del batch, prior, output, loss
    require(identity() == registered, 'source changed during fixed-input comparison')
    from models.ct_v40.attention import EfficientAttention
    cores = [module for module in model.modules() if isinstance(module, EfficientAttention)]
    require(cores and all(module.backend == spec['backend'] for module in cores), 'mixed attention backend')
    if spec['backend'] == 'fa2_varlen':
        require(sum(module.kernel_calls for module in cores) > 0, 'fixed-input timing did not call FA2')
    return dict(passed=True, phase='performance_fixed', pid=os.getpid(), identity=registered,
        backend=spec['backend'], corpus_sha256=spec['corpus_sha256'],
        prepared_sha256=digest_tree(payload['batches']), model_sha256=digest_tree(payload['model']),
        timing=timing_summary(timings), peak_reserved=max(row['peak_reserved'] for row in memory),
        peak_allocated=max(row['peak_allocated'] for row in memory), losses=losses,
        scope='same real prepared tensors and frozen supervision; identical initial parameters/BN/Adam/RNG for EVERY timed step; complete B0+B1+B3+M loss/backward/Adam; excludes raw load, cropping, dynamic supervision construction, accepted-state commit, CPU restore and H2D',
        limitations='FP32 math SDPA vs BF16 FA2 core; both retain FP32 projections/backbone. Training dropout kernels differ; this establishes timing on fixed observations, not numerical trajectory/score equivalence or end-to-end speedup')


def fixed_performance_gate(directory, common):
    shared = dict(common, allocator='round2')
    captured = finish_worker(start_worker(directory, 'fixed-input-capture', dict(shared, phase='performance_capture')))
    outputs = [finish_worker(start_worker(directory, 'fixed-' + backend, dict(shared,
        phase='performance_fixed', backend=backend, corpus=captured['corpus'], corpus_sha256=captured['corpus_sha256'])))
        for backend in ('math', 'fa2_varlen')]
    require(len({out['prepared_sha256'] for out in outputs}) == 1
        and len({out['model_sha256'] for out in outputs}) == 1, 'FA/math comparison input or weights mismatch')
    tracking = finish_worker(start_worker(directory, 'tracking-batch1', dict(shared,
        phase='performance_tracking', corpus=captured['corpus'], corpus_sha256=captured['corpus_sha256'])))
    return dict(passed=True, capture=captured, workers=outputs, tracking_batch1=tracking,
        fa_over_math_median=outputs[1]['timing']['median_seconds'] / outputs[0]['timing']['median_seconds'],
        interpretation='descriptive fixed-input full-network training-step cost; no automatic speed gain threshold or tracking accuracy claim')


def performance_tracking(spec):
    """batch1 带标注验证路径时延；保留生产 host 调用，不冒充纯在线推理。"""
    import torch
    import pytorch_lightning as pl
    from models.ct_v40.config import load_config
    from models.ct_v40.entry import configure_numerics
    from models.ct_v40.data import build_loaders
    from models.ctseqtrackv40 import CTSEQTRACKV40
    registered = identity()
    path = Path(spec['corpus'])
    require(hashlib.sha256(path.read_bytes()).hexdigest() == spec['corpus_sha256'], 'tracking weights corpus changed')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    config = load_config(spec['cfg'], dict(path=spec['path'], ct_engineering_check=True,
        log_dir=None, batch_size=1, v31_evaluate_late3=False, v40_attention_backend='fa2_varlen'))
    require(config.v40_arm == 'full' and config.point_sample_size == 1024, 'tracking latency requires Full/1024 points')
    configure_numerics(config)
    pl.seed_everything(config.seed, workers=True)
    loaders = build_loaders(config, roles=('val',))
    loader = loaders['val']
    model = CTSEQTRACKV40(config, loaders=loaders).cuda().eval()
    model.load_state_dict(payload['model'], strict=True)
    model.evaluation_builder.reset()
    timings = []
    iterator = iter(loader)
    consumed = 0
    with torch.no_grad():
        for index in range(len(loader)):
            if index == 16:
                torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            rows = next(iterator)
            require(len(rows) == 1, 'tracking latency requires actual batch1')
            batch, output = model._forward_raw(rows, model.evaluation_builder, training=False)
            model.evaluation_builder.commit(output, batch, diagnostics=False)
            model._release_materialization(batch, model.evaluation_builder)
            del rows, batch, output
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if index >= 16:
                timings.append(elapsed)
            consumed += 1
    require(consumed == loader.batch_sampler.endpoint_count and len(timings) >= 128,
            'tracking latency needs complete actual val and at least128 frames after16 warmup')
    require(not model.evaluation_builder.states, 'completed tracking latency pass retains unfinished tracks')
    require(identity() == registered, 'source changed during tracking latency')
    timing = timing_summary(timings)
    return dict(passed=True, phase='performance_tracking', identity=registered, pid=os.getpid(),
        batch_size=1, backend='fa2_varlen', allocator=spec['allocator'], warmup_frames=16,
        consumed_prediction_frames=consumed, model_sha256=digest_tree(payload['model']),
        timing=timing, p50_frame_seconds=timing['median_seconds'], p95_frame_seconds=timing['p95_seconds'],
        frames_per_second=len(timings) / timing['total_seconds'], memory=memory_sample('tracking_complete', consumed),
        timing_kind='annotated_validation_path_batch1',
        scope='raw next(loader), registered prepare including GT labels and search-coverage diagnostics, prior, supervise_prior(training=False) including passive GT target counts/displacement and transfers, dense B0, B3 selection, accepted-state commit, cleanup and CUDA synchronization',
        excluded_report_work='TrackingEvaluation.add_batch/summary, candidate-box IoU/S/P metric recomputation and trusted reliability aggregation are not called in this benchmark',
        limitations='This is annotated validation-path frame latency, not pure online inference latency. GT/annotation preparation and passive diagnostic work remain inside the timed host path; their inclusion does not imply that GT enters the online model or state decisions.',
        weights='random initialization plus40 engineering Adam steps; not a formal trained checkpoint; latency only, no tracking accuracy claim')


def worker(spec):
    if spec['phase'] == 'entry_resume':
        return entry_resume_worker(spec)
    if spec['phase'] == 'performance_capture':
        return performance_capture(spec)
    if spec['phase'] == 'performance_fixed':
        return performance_fixed(spec)
    if spec['phase'] == 'performance_tracking':
        return performance_tracking(spec)
    import torch
    start_identity = identity()
    workload = Workload(spec)
    phase = spec['phase']
    with (nullcontext() if phase in ('performance', 'parallel') else workload.audit):
        if phase == 'kernel':
            activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            with torch.profiler.profile(activities=activities) as profile:
                workload.segment('kernel', workload.plans['train'][:2], instrument=True, fingerprints=[])
            names = sorted({event.name for event in profile.events() if 'CUDA' in str(event.device_type)})
            require(any('flash' in name.lower() and ('fwd' in name.lower() or 'forward' in name.lower()) for name in names),
                    'profiler lacks actual FA2 CUDA forward kernel')
            require(any('flash' in name.lower() and ('bwd' in name.lower() or 'backward' in name.lower()) for name in names),
                    'profiler lacks actual FA2 CUDA backward kernel')
            trace = str(Path(spec['result']).with_suffix('.trace.json'))
            profile.export_chrome_trace(trace)
            evidence = dict(cuda_kernels=names, trace=trace, kernel_calls=workload.audit.total,
                            last_batch_layout_and_masks=workload.audit.records)
        elif phase == 'lifecycle':
            hashes = []
            evidence = workload.segment('lifecycle', workload.plans['train'][:8], instrument=True, fingerprints=hashes)
            evidence['workload_sha256'] = digest_tree(hashes)
            require(workload.model._epoch_steps == 8 and workload.model._epoch_rows == 128, 'Adam/commit count mismatch')
            evidence.update(optimizer_steps=workload.model._epoch_steps, endpoint_rows=workload.model._epoch_rows)
            require(all(bool(torch.isfinite(p).all()) for p in workload.model.parameters()), 'nonfinite parameters after Adam')
        else:
            workload.warmup()
            if phase == 'parallel':
                write_new_json(spec['ready'], dict(pid=os.getpid(), phase='warmed'))
                deadline = time.monotonic() + 1800
                while not Path(spec['go']).exists():
                    require(time.monotonic() < deadline, 'parallel peer barrier timeout')
                    time.sleep(.1)
                torch.cuda.reset_peak_memory_stats()
                evidence = workload.cycle(instrument=False, fingerprint=False)
                evidence['memory'] = memory_sample('complete', 0)
            else:
                snapshot = CPUReplaySnapshot(workload.model, workload.optimizer)
                workload.cycle(instrument=False, fingerprint=False)  # 完整预热，覆盖 train/val 边界。
                cycles = []
                for unused in range(3):
                    snapshot.restore(workload.model, workload.optimizer)
                    torch.cuda.reset_peak_memory_stats()
                    cycles.append(workload.cycle(instrument=phase == 'memory', fingerprint=phase == 'memory'))
                evidence = dict(cycles=cycles, snapshot_sha256=snapshot.fingerprint)
                if phase == 'memory':
                    evidence['gate'] = analyze_memory_cycles(cycles)
                    for cycle in cycles:
                        require(cycle['coverage'].get('empty_sequences', 0) > 0
                            and cycle['coverage'].get('guard_rows', 0) > 0
                            and cycle['coverage'].get('drain_rows', 0) > 0
                            and cycle['coverage'].get('partial_batches', 0) > 0
                            and cycle['coverage'].get('current_sparse_1_2', 0) > 0,
                            'memory workload did not cover empty/guard/drain/partial/sparse paths')
                    require(evidence['gate']['passed'], 'memory drift or continuing growth exceeded acceptance limits')
                else:
                    require(len({cycle['request_sha256'] for cycle in cycles}) == 1, 'performance request workload changed')
                    evidence['scope'] = 'complete raw next(loader), prepare, forward, loss, backward, Adam, commit, cleanup; synchronized endpoints; no per-phase fingerprint/profiler'
        from models.ct_v40.attention import EfficientAttention
        kernel_calls = sum(module.kernel_calls for module in workload.model.modules() if isinstance(module, EfficientAttention))
        require(kernel_calls > 0, 'no actual FA2 kernel calls')
    require(identity() == start_identity, 'source changed during acceptance')
    return dict(passed=True, phase=phase, pid=os.getpid(), allocator=spec['allocator'],
        arm=workload.config.v40_arm, identity=start_identity, fa_runtime=workload.fa_runtime, evidence=evidence)


def start_worker(directory, name, spec):
    directory = Path(directory)
    result = directory / (name + '.json')
    specification = directory / (name + '.spec.json')
    spec = dict(spec, result=str(result))
    write_new_json(specification, spec)
    log = (directory / (name + '.log')).open('x', encoding='utf-8')
    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker-spec', str(specification)],
        cwd=ROOT, env=acceptance_environment(spec['allocator'], gpu=spec['gpu']),
        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    return process, log, result


def finish_worker(item):
    process, log, result = item
    code = process.wait()
    log.close()
    require(code == 0 and result.exists(), f'worker PID {process.pid} failed; inspect {result.with_suffix(".log")}')
    value = json.loads(result.read_text(encoding='utf-8'))
    require(value.get('passed') is True, 'worker acceptance failed: ' + str(result))
    return value


def parallel_pair(directory, labels, common):
    directory = Path(directory)
    go = directory / ('go-' + '-'.join(labels) + '.json')
    children, samples = [], []
    monitor = NVMLMonitor(common['gpu'])
    try:
        for label in labels:
            ready = directory / (label + '.ready.json')
            spec = dict(common, phase='parallel', allocator='round2', cfg=str(ROOT / f'cfgs/ct_seqtrack/40_{label}_mini.yaml'),
                        ready=str(ready), go=str(go))
            children.append(start_worker(directory, label, spec))
        pids = [child[0].pid for child in children]
        deadline = time.monotonic() + 1800
        while not all((directory / (label + '.ready.json')).exists() for label in labels):
            require(all(child[0].poll() is None for child in children), 'parallel worker exited before warmup barrier')
            require(time.monotonic() < deadline, 'parallel warmup timeout')
            samples.append(monitor.sample(pids))
            time.sleep(.1)
        write_new_json(go, dict(pids=pids))
        while any(child[0].poll() is None for child in children):
            samples.append(monitor.sample(pids))
            require(all(child[0].poll() in (None, 0) for child in children), 'parallel child failed')
            time.sleep(.1)
        results = [finish_worker(child) for child in children]
        gate = capacity_gate(samples, pids)
        peaks = [child['evidence']['memory']['peak_reserved'] for child in results]
        overhead = [max(0, max(sample['processes'].get(str(pid), 0) for sample in samples) - peak)
                    for pid, peak in zip(pids, peaks)]
        conservative = sum(peaks) + sum(overhead) + gate['external_and_driver_peak_bytes']
        gate['reserved_peaks_plus_context_external_bytes'] = conservative
        gate['context_and_nonallocator_overhead_bytes'] = overhead
        gate['conservative_margin_passed'] = conservative + gate['required_margin_bytes'] <= samples[0]['total']
        gate['passed'] = gate['passed'] and gate['conservative_margin_passed']
        require(gate['passed'], 'actual GPU0 concurrent free margin is below max(2GiB,10%)')
        return dict(labels=list(labels), children=results, nvml_samples=samples, gate=gate)
    finally:
        monitor.close()
        for process, log, unused in children:
            if process.poll() is None:
                process.terminate()
                process.wait()
            if not log.closed:
                log.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--phase', choices=PHASES)
    parser.add_argument('--cfg', default='cfgs/ct_seqtrack/40_full_mini.yaml')
    parser.add_argument('--path')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--output')
    parser.add_argument('--worker-spec', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_spec:
        spec = json.loads(Path(args.worker_spec).read_text(encoding='utf-8'))
        try:
            result = worker(spec)
        except BaseException:
            write_new_json(spec['result'], dict(passed=False, phase=spec['phase'], error=traceback.format_exc()))
            raise
        write_new_json(spec['result'], result)
        return 0
    require(args.phase and args.output, '--phase and a new --output directory are required')
    require(args.gpu == 0, 'registered server acceptance and paired capacity use physical GPU0')
    directory = Path(args.output).resolve()
    require(directory.is_relative_to((ROOT / 'artifacts/ct_checks').resolve()), 'acceptance artifacts must stay under artifacts/ct_checks')
    directory.mkdir(parents=True, exist_ok=False)
    start_identity = identity()
    report = dict(schema=ENGINEERING_SCHEMA, phase=args.phase, identity=start_identity, passed=False)
    try:
        if args.phase == 'contracts':
            commands = [[sys.executable, '-m', 'pytest', '-q'],
                [sys.executable, '-m', 'compileall', '-q', 'models/', 'datasets/', 'utils/', 'tools/'],
                [sys.executable, 'tools/verify_ct_v40.py', 'verify']]
            for index, command in enumerate(commands):
                with (directory / f'contracts-{index}.log').open('x', encoding='utf-8') as log:
                    subprocess.run(command, cwd=ROOT, env=acceptance_environment('round2'),
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
            report['commands'] = commands
        else:
            require(args.path, '--path must be the actual nuScenes mini data root')
            common = dict(phase=args.phase, cfg=str((ROOT / args.cfg).resolve()), path=str(Path(args.path).resolve()), gpu=args.gpu)
            from models.ct_v40.config import load_config
            from models.ct_v40.entry import validate_formal_registration
            formal_config = load_config(common['cfg'])
            validate_formal_registration(formal_config)
            require(formal_config.v40_arm == 'full', '--cfg must identify the registered Full recipe for this acceptance suite')
            if args.phase == 'parallel':
                report['pairs'] = [parallel_pair(directory, labels, common) for labels in WAVES]
            elif args.phase == 'lifecycle':
                report['workers'] = [finish_worker(start_worker(directory, label, dict(common,
                    allocator='round2', cfg=str(ROOT / f'cfgs/ct_seqtrack/40_{label}_mini.yaml'))))
                    for pair in WAVES for label in pair]
                report['entry_epoch_boundary_resume'] = entry_resume_gate(directory, common)
            else:
                profiles = ('native', 'round2') if args.phase in ('memory', 'performance') else ('round2',)
                report['workers'] = [finish_worker(start_worker(directory, profile, dict(common, allocator=profile))) for profile in profiles]
                if args.phase == 'memory':
                    hashes = {worker['evidence']['cycles'][0]['workload_sha256'] for worker in report['workers']}
                    require(len(hashes) == 1, 'native/round2 did not replay identical tensors/masks/cu_seqlens/branches')
                if args.phase == 'performance':
                    report['fixed_input_fa_vs_math'] = fixed_performance_gate(directory, common)
                    report['interpretation'] = 'Independent native vs round2 FA2 processes; three identical raw request replays each. This measures allocator choice, not FA2-vs-math causal speedup.'
        require(identity() == start_identity, 'source changed during acceptance')
        report['passed'] = True
    except BaseException:
        report['error'] = traceback.format_exc()
        raise
    finally:
        write_new_json(directory / 'report.json', report)
    print(json.dumps(dict(passed=True, report=str(directory / 'report.json')), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
