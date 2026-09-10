"""v29 短测速与逐位对照；默认不启用，不改变正式训练配置。"""

from contextlib import contextmanager, nullcontext
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

AUDIT_STEPS = (1, 2, 3, 4, 5, 10, 100)
_CURRENT = None


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {'count': 0, 'mean_ms': None, 'p50_ms': None, 'p90_ms': None}
    return dict(count=int(values.size), mean_ms=float(values.mean()),
                p50_ms=float(np.percentile(values, 50)),
                p90_ms=float(np.percentile(values, 90)))


def snapshot(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [snapshot(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, '__dict__'):
        return {'type': type(value).__qualname__, 'fields': snapshot(vars(value))}
    raise TypeError('unsupported numerical snapshot value: ' + type(value).__qualname__)


def first_difference(left, right, path='root'):
    """比较字节，不用浮点容差、哈希清单或全层激活快照。"""
    if torch.is_tensor(left) or torch.is_tensor(right):
        if not (torch.is_tensor(left) and torch.is_tensor(right)):
            return path + ': tensor type differs'
        if left.dtype != right.dtype or left.shape != right.shape:
            return path + ': tensor shape/dtype differs'
        a = left.contiguous().reshape(-1).view(torch.uint8)
        b = right.contiguous().reshape(-1).view(torch.uint8)
        return None if torch.equal(a, b) else path + ': tensor bytes differ'
    if type(left) is not type(right):
        return path + ': type differs'
    if isinstance(left, dict):
        if set(left) != set(right):
            return path + ': keys differ'
        for key in left:
            difference = first_difference(left[key], right[key], path + '.' + key)
            if difference:
                return difference
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return path + ': length differs'
        for index, (a, b) in enumerate(zip(left, right)):
            difference = first_difference(a, b, f'{path}[{index}]')
            if difference:
                return difference
        return None
    if isinstance(left, float) and math.isnan(left) and math.isnan(right):
        return None
    return None if left == right else path + ': value differs'


def capture_training_state(host, trainer):
    from utils.training_isolation import capture_global_rng_state
    schedulers = [item.scheduler for item in getattr(trainer, 'lr_scheduler_configs', [])]
    b0_parameters = [(name, value) for name, value in host.named_parameters()
                     if not host._ct_any_plugin_parameter(name)]
    roots = {name.split('.', 1)[0] for name, _ in b0_parameters} | {'time_encoder'}
    shared_b0 = dict(
        parameters=dict(b0_parameters),
        gradients={name: value.grad for name, value in b0_parameters},
        buffers={name: value for name, value in host.named_buffers()
                 if name.split('.', 1)[0] in roots or name == 'ct_b0_update_step'},
        adam={name: optimizer.state.get(value, {}) for optimizer in trainer.optimizers
              for name, value in b0_parameters
              if any(value is candidate for group in optimizer.param_groups for candidate in group['params'])},
        rng=capture_global_rng_state(),
    )
    return snapshot(dict(
        model=host.state_dict(),
        gradients={name: value.grad for name, value in host.named_parameters()},
        adam=[optimizer.state_dict() for optimizer in trainer.optimizers],
        schedulers=[scheduler.state_dict() for scheduler in schedulers],
        rng=capture_global_rng_state(),
        recursive_states=getattr(host, '_ct_recursive_states', {}),
        training_flags={name: child.training for name, child in host.named_modules()},
        shared_b0=shared_b0,
    ))


def profile_stage(host, name):
    recorder = getattr(host, '_ct_runtime_profiler', None)
    return recorder.stage(name) if recorder is not None else nullcontext()


def profile_loader_stage(name):
    """只在短分段测速中记录真实 next(loader) 等待，不触碰 RNG。"""
    recorder = _CURRENT
    if recorder is None or recorder.mode != 'profile':
        return nullcontext()
    return recorder.loader_stage(name)


def record_equivalence_transaction(host, batch, output, losses):
    recorder = getattr(host, '_ct_runtime_profiler', None)
    if recorder is None or recorder.mode != 'equivalence' or recorder.step not in AUDIT_STEPS:
        return
    # 仅忽略已登记的纯 H3/计时标签；不能宽泛排除 ct_*，它们含实际训练输入。
    data = {key: value for key, value in batch.items()
            if not key.startswith(('ct_h3_', 'ct_shadow_'))}
    output_keys = (
        'seg_logits', 'motion_cls', 'motion_pred', 'estimation_boxes',
        'observation_aux_estimation_boxes', 'aux_estimation_boxes',
        'updated_ref_boxs', 'pred_bc', 'ct_final_box', 'ct_router_applied_gate',
        'ct_router_bounded_residual_xy', 'ct_search_candidate_valid',
        'ct_search_raw_candidate_xy', 'ct_b3_action_score',
        'ct_extension_selected_indices', 'ct_extension_selected_valid_mask',
    )
    recorder.transactions.append(snapshot(dict(
        input=data, output={key: output[key] for key in output_keys if key in output},
        losses={key: value for key, value in losses.items() if key.startswith('loss_')},
        mechanism=bool(getattr(host, '_ct_mechanism_transaction', False)),
    )))


class RuntimeRecorder:
    def __init__(self, directory, mode, warmup=20, steps=100):
        if mode not in ('benchmark', 'profile', 'equivalence'):
            raise ValueError('unknown v29 performance mode')
        if not 0 <= warmup < steps <= 100:
            raise ValueError('require 0 <= warmup < steps <= 100')
        self.directory = Path(directory)
        self.mode, self.warmup, self.steps = mode, int(warmup), int(steps)
        self.step = 0
        self.host = None
        self.rows, self.transactions = [], []
        self.stage_values, self.loader_values = {}, {}
        self.last_end = None
        self.starts = {}
        self.handles = []
        self.completed = False

    def synchronize(self):
        if self.host is not None and self.host.device.type == 'cuda':
            torch.cuda.synchronize(self.host.device)

    @property
    def measured(self):
        return self.warmup < self.step <= self.steps

    @contextmanager
    def stage(self, name):
        if self.mode != 'profile' or not self.measured:
            yield
            return
        self.synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            self.synchronize()
            self.stage_values[name] = self.stage_values.get(name, 0.) + 1000. * (time.perf_counter() - start)

    @contextmanager
    def loader_stage(self, name):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.loader_values[name] = self.loader_values.get(name, 0.) + 1000. * (time.perf_counter() - start)

    def start_hook(self, name):
        if self.mode == 'profile' and self.measured:
            self.synchronize()
            self.starts[name] = time.perf_counter()

    def end_hook(self, name):
        if name in self.starts:
            self.synchronize()
            elapsed = 1000. * (time.perf_counter() - self.starts.pop(name))
            self.stage_values[name] = self.stage_values.get(name, 0.) + elapsed

    def save_state(self, host, trainer, step):
        payload = dict(schema='ct_seqtrack.compact_equivalence.v29', step=step,
                       state=capture_training_state(host, trainer), transactions=self.transactions)
        path = self.directory / 'snapshots' / f'step_{step:03d}.pt'
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        self.transactions = []

    def batch_start(self, host, trainer, batch_idx):
        if self.mode == 'equivalence':
            if self.step == 0 or self.step in AUDIT_STEPS:
                self.save_state(host, trainer, self.step)
            self.transactions = []
        else:
            self.synchronize()
        now = time.perf_counter()
        self.gap_ms = None if self.last_end is None else 1000. * (now - self.last_end)
        self.step = int(batch_idx) + 1
        self.batch_started = now
        self.stage_values = {}
        self.batch_loader_values, self.loader_values = self.loader_values, {}

    def batch_end(self):
        if self.mode == 'equivalence':
            return
        self.synchronize()
        ended = time.perf_counter()
        if self.measured:
            batch_ms = 1000. * (ended - self.batch_started)
            peak = (torch.cuda.max_memory_allocated(self.host.device) / 1024**2
                    if self.host.device.type == 'cuda' else None)
            self.rows.append(dict(step=self.step, batch_ms=batch_ms,
                interbatch_ms=self.gap_ms, cycle_ms=batch_ms + (self.gap_ms or 0.),
                stages_ms=dict(self.stage_values), loader_ms=dict(self.batch_loader_values),
                observed_peak_allocated_mb=peak))
        self.last_end = ended

    def report(self, status='complete'):
        names = sorted({key for row in self.rows for key in row['stages_ms']})
        loaders = sorted({key for row in self.rows for key in row['loader_ms']})
        peaks = [row['observed_peak_allocated_mb'] for row in self.rows
                 if row['observed_peak_allocated_mb'] is not None]
        result = dict(schema='ct_seqtrack.performance.v29', status=status, mode=self.mode,
            warmup=self.warmup, requested_steps=self.steps, completed_steps=self.step,
            measured_steps=len(self.rows),
            measurement_scope='前100个真实训练batch；20步预热后仍包含前100输入指纹与step100参数/Adam审计，不能直接视作第19619步稳态速度。',
            cycle=summarize([row['cycle_ms'] for row in self.rows]),
            batch=summarize([row['batch_ms'] for row in self.rows]),
            interbatch=summarize([row['interbatch_ms'] for row in self.rows
                                  if row['interbatch_ms'] is not None]),
            stages={name: summarize([row['stages_ms'][name] for row in self.rows
                                     if name in row['stages_ms']]) for name in names},
            loader={name: summarize([row['loader_ms'].get(name, 0.) for row in self.rows])
                    for name in loaders},
            observed_peak_allocated_mb=max(peaks) if peaks else None,
            interpretation=('profile 阶段为同步插桩的包含时间，嵌套项不能相加；不能用来宣称正式吞吐。'
                if self.mode == 'profile' else 'benchmark 只在 batch 边界同步；cycle 含两批间数据等待/Lightning开销。'),
            rows=self.rows)
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / 'runtime.json').write_text(
            json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
        return result


def make_runtime_callback():
    """主入口仅在显式短检查环境变量存在时挂载；不导入依赖于CPU工具的Lightning。"""
    directory = os.environ.get('CT_V29_PROFILE_DIR')
    if not directory:
        return None
    if os.environ.get('CT_V28_AUDIT_DIR'):
        raise ValueError('short performance checks forbid CT_V28_AUDIT_DIR')
    from pytorch_lightning.callbacks import Checkpoint
    root = (Path(__file__).resolve().parents[1] / 'artifacts' / 'ct_checks').resolve()
    target = Path(directory).resolve()
    if root not in target.parents:
        raise ValueError('short performance artifacts must be below artifacts/ct_checks')
    recorder = RuntimeRecorder(target, os.environ.get('CT_V29_PROFILE_MODE', 'benchmark'),
        int(os.environ.get('CT_V29_PROFILE_WARMUP', '20')),
        int(os.environ.get('CT_V29_PROFILE_STEPS', '100')))

    class RuntimeCallback(Checkpoint):
        # Checkpoint marker guarantees epoch-end snapshots follow the host's scheduler/state commit.
        def setup(self, trainer, pl_module, stage):
            global _CURRENT
            del stage
            if (not bool(getattr(pl_module.config, 'ct_enable_v29', False))
                    or not bool(getattr(pl_module.config, 'ct_engineering_check', False))
                    or trainer.max_epochs != 1):
                raise ValueError('runtime instrumentation requires one-epoch v29 engineering run')
            _CURRENT = recorder
            recorder.host = pl_module
            pl_module._ct_runtime_profiler = recorder

        def on_fit_start(self, trainer, pl_module):
            del pl_module
            if recorder.mode == 'profile':
                for optimizer in trainer.optimizers:
                    recorder.handles.extend((
                        optimizer.register_step_pre_hook(lambda *args: recorder.start_hook('optimizer')),
                        optimizer.register_step_post_hook(lambda *args: recorder.end_hook('optimizer'))))

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            del batch
            recorder.batch_start(pl_module, trainer, batch_idx)

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            del trainer, pl_module, outputs, batch, batch_idx
            recorder.batch_end()

        def on_before_backward(self, trainer, pl_module, loss):
            del trainer, pl_module, loss
            recorder.start_hook('backward')

        def on_after_backward(self, trainer, pl_module):
            del trainer, pl_module
            recorder.end_hook('backward')

        def on_train_epoch_end(self, trainer, pl_module):
            if recorder.mode == 'equivalence' and recorder.step in AUDIT_STEPS:
                recorder.save_state(pl_module, trainer, recorder.step)
            recorder.completed = recorder.step == recorder.steps
            recorder.report('complete' if recorder.completed else 'incomplete')

        def on_exception(self, trainer, pl_module, exception):
            del trainer, pl_module, exception
            recorder.report('failed')

        def teardown(self, trainer, pl_module, stage):
            global _CURRENT
            del trainer, stage
            for handle in recorder.handles:
                handle.remove()
            pl_module._ct_runtime_profiler = None
            _CURRENT = None

    return RuntimeCallback()
