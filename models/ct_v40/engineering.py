"""V40 服务器验收公共件；不改变正式训练算法，不执行任何自动验收。"""
from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

MIB = 2 ** 20
GIB = 2 ** 30
ALLOCATED_DRIFT_LIMIT = 64 * MIB
RESERVED_DRIFT_LIMIT = 256 * MIB
SETTLED_GROWTH_LIMIT = MIB
ENGINEERING_SCHEMA = 'ct_seqtrack.v40.server_acceptance.v1'
ALLOCATORS = {'native': 'backend:native',
              'round2': 'backend:native,roundup_power2_divisions:2'}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def cpu_copy(value):
    """快照全部放 CPU，包括 RNG、BN、Adam；不保留 CUDA clone。"""
    import torch
    if torch.is_tensor(value):
        return value.detach().to(device='cpu', copy=True)
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return type(value)(**{item.name: cpu_copy(getattr(value, item.name)) for item in fields(value)})
    return copy.deepcopy(value)


def digest_tree(value):
    """包含 dtype/shape/字节的稳定内容指纹，支持 BF16 和 numpy。"""
    import numpy as np
    import torch
    digest = hashlib.sha256()
    def visit(item):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(str((str(item.dtype), item.shape)).encode())
            digest.update(np.ascontiguousarray(item).tobytes())
        elif is_dataclass(item) and not isinstance(item, type):
            visit({field.name: getattr(item, field.name) for field in fields(item)})
        elif isinstance(item, dict):
            for key in sorted(item, key=repr):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode())
            digest.update(str(len(item)).encode())
            for child in item:
                visit(child)
        elif isinstance(item, (str, int, float, bool, type(None), np.generic)):
            digest.update(repr(item).encode('utf-8'))
        else:
            raise TypeError('unhashable engineering evidence: ' + type(item).__name__)
    visit(value)
    return digest.hexdigest()


def initialize_adam_storage(optimizer):
    """一次预建尚未使用参数的 Adam 零状态；step=0，数学等同首次惰性建立。"""
    import torch
    require(isinstance(optimizer, torch.optim.Adam), 'acceptance requires registered Adam')
    for group in optimizer.param_groups:
        require(not group.get('fused', False) and not group.get('capturable', False),
                'in-place replay requires ordinary noncapturable Adam')
        for parameter in group['params']:
            state = optimizer.state[parameter]
            if not state:
                state.update(step=torch.tensor(0., dtype=torch.float32, device='cpu'),
                             exp_avg=torch.zeros_like(parameter), exp_avg_sq=torch.zeros_like(parameter))
                if group.get('amsgrad', False):
                    state['max_exp_avg_sq'] = torch.zeros_like(parameter)


def tensor_addresses(model, optimizer):
    addresses = {name: tensor.data_ptr() for name, tensor in model.state_dict().items()}
    for group_index, group in enumerate(optimizer.param_groups):
        for index, parameter in enumerate(group['params']):
            for name, value in optimizer.state[parameter].items():
                if hasattr(value, 'data_ptr'):
                    addresses[f'adam/{group_index}/{index}/{name}'] = value.data_ptr()
    return addresses


class CPUReplaySnapshot:
    """只在完成 commit、zero_grad 和释放 batch 后创建；GPU 张量原址恢复。"""
    def __init__(self, model, optimizer):
        from .runtime import capture_rng_state
        for builder in (model.train_builder, model.evaluation_builder):
            require(builder._pending is None, 'cannot snapshot a pending data transaction')
        require(model._pending_train is None, 'cannot snapshot a pending host transaction')
        require(all(p.grad is None for p in model.parameters()), 'snapshot requires zero_grad(set_to_none=True)')
        initialize_adam_storage(optimizer)
        self.model = cpu_copy(model.state_dict())
        self.optimizer = [[cpu_copy(optimizer.state[p]) for p in group['params']]
                          for group in optimizer.param_groups]
        self.groups = [{key: cpu_copy(value) for key, value in group.items() if key != 'params'}
                       for group in optimizer.param_groups]
        self.builders = [cpu_copy(builder.states) for builder in (model.train_builder, model.evaluation_builder)]
        self.counters = {name: cpu_copy(getattr(model, name)) for name in
                         ('_epoch_rows', '_epoch_steps', '_completed_epoch', '_epoch_complete')}
        self.training = {name: module.training for name, module in model.named_modules()}
        self.rng = cpu_copy(capture_rng_state())
        self.addresses = tensor_addresses(model, optimizer)
        self.fingerprint = digest_tree((self.model, self.optimizer, self.builders, self.rng))

    def restore(self, model, optimizer):
        import torch
        from .runtime import restore_rng_state
        require(model._pending_train is None, 'restore would abandon pending host work')
        require(all(builder._pending is None for builder in (model.train_builder, model.evaluation_builder)),
                'restore would abandon pending data work')
        require(tensor_addresses(model, optimizer) == self.addresses, 'model/Adam tensor storage was replaced')
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            current = model.state_dict()
            require(current.keys() == self.model.keys(), 'model keys changed during replay')
            for name, value in current.items():
                value.copy_(self.model[name], non_blocking=False)
            for group, saved_group, saved_states in zip(optimizer.param_groups, self.groups, self.optimizer):
                for key, value in saved_group.items():
                    group[key] = cpu_copy(value)
                for parameter, saved in zip(group['params'], saved_states):
                    actual = optimizer.state[parameter]
                    require(actual.keys() == saved.keys(), 'Adam state schema changed during replay')
                    for key, value in saved.items():
                        if torch.is_tensor(value):
                            actual[key].copy_(value, non_blocking=False)
                        else:
                            actual[key] = copy.deepcopy(value)
        for builder, saved in zip((model.train_builder, model.evaluation_builder), self.builders):
            builder.reset()
            builder.states.update(cpu_copy(saved))
        for name, value in self.counters.items():
            setattr(model, name, copy.deepcopy(value))
        for name, module in model.named_modules():
            module.training = self.training[name]
        # 工程 diagnostics 只供本轮记录；不让 CPU 日志历史增长干扰长期检查。
        if model.training_diagnostics is not None:
            model.training_diagnostics.reset(10)
        restore_rng_state(self.rng)
        require(tensor_addresses(model, optimizer) == self.addresses, 'restore allocated new model/Adam tensors')


def memory_sample(phase, index, *, role='train'):
    import torch
    torch.cuda.synchronize()
    stats = torch.cuda.memory_stats()
    return dict(role=role, phase=phase, index=index, allocated=torch.cuda.memory_allocated(),
                reserved=torch.cuda.memory_reserved(), peak_allocated=torch.cuda.max_memory_allocated(),
                peak_reserved=torch.cuda.max_memory_reserved(),
                inactive_split_bytes=stats.get('inactive_split_bytes.all.current', 0),
                allocation_retries=stats.get('num_alloc_retries', 0), oom_count=stats.get('num_ooms', 0))


def analyze_memory_cycles(cycles):
    """同 workload 相位逐点比较；绝对上限和最后两轮停止增长必须同时成立。"""
    require(len(cycles) == 3, 'memory acceptance requires exactly three measured replays')
    require(all(c['workload_sha256'] == cycles[0]['workload_sha256'] for c in cycles),
            'inputs/masks/cu_seqlens/branches differ between memory replays')
    require(all(type(c.get('allocation_retries_delta')) is int and c['allocation_retries_delta'] >= 0 for c in cycles),
            'each measured memory replay must record allocator retry deltas')
    mappings = [{(r['role'], r['index'], r['phase']): r for r in c['samples']} for c in cycles]
    require(all(len(mapping) == len(cycle['samples']) for mapping, cycle in zip(mappings, cycles)),
            'duplicate memory phase keys')
    require(mappings[0] and all(set(m) == set(mappings[0]) for m in mappings), 'memory phase coverage differs')
    gaps = {field: max(max(m[k][field] for m in mappings) - min(m[k][field] for m in mappings)
                      for k in mappings[0]) for field in ('allocated', 'reserved')}
    # 每轮完全相同的终态与同一相位比较，不能用有上界但仍逐轮上涨伪装稳定。
    end_growth = {field: cycles[2]['samples'][-1][field] - cycles[1]['samples'][-1][field]
                  for field in ('allocated', 'reserved')}
    aligned_growth = {field: max(mappings[2][key][field] - mappings[1][key][field]
                                for key in mappings[0]) for field in ('allocated', 'reserved')}
    monotonic_cleanup_growth = {field: [list(key) for key in mappings[0] if key[2] == 'after_cleanup'
        and mappings[0][key][field] < mappings[1][key][field] < mappings[2][key][field]]
        for field in ('allocated', 'reserved')}
    passed = (gaps['allocated'] <= ALLOCATED_DRIFT_LIMIT
              and gaps['reserved'] <= RESERVED_DRIFT_LIMIT
              and all(value <= SETTLED_GROWTH_LIMIT for value in end_growth.values())
              and all(value <= SETTLED_GROWTH_LIMIT for value in aligned_growth.values())
              and not any(monotonic_cleanup_growth.values())
              and all(cycle['allocation_retries_delta'] == 0 for cycle in cycles)
              and all(r['oom_count'] == 0 for c in cycles for r in c['samples']))
    return dict(passed=passed, phase_range_bytes=gaps, cycle3_minus_cycle2_endpoint_bytes=end_growth,
                cycle3_minus_cycle2_max_aligned_bytes=aligned_growth,
                monotonic_cleanup_growth=monotonic_cleanup_growth,
                limits_bytes=dict(allocated=ALLOCATED_DRIFT_LIMIT, reserved=RESERVED_DRIFT_LIMIT,
                                  settled_positive_growth=SETTLED_GROWTH_LIMIT),
                note='先完整 workload 预热一次，再原址恢复；不调用 empty_cache；allocated/reserved 分开验收')


class NVMLMonitor:
    """真实物理 GPU 总占用与外部 PID；NVML 缺失直接失败。"""
    def __init__(self, physical_gpu=0):
        import pynvml
        self.api = pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(physical_gpu)
        self.physical_gpu = physical_gpu

    def sample(self, owned_pids=()):
        api, handle = self.api, self.handle
        memory = api.nvmlDeviceGetMemoryInfo(handle)
        processes = {}
        for getter in (api.nvmlDeviceGetComputeRunningProcesses, api.nvmlDeviceGetGraphicsRunningProcesses):
            for process in getter(handle):
                used = process.usedGpuMemory
                require(isinstance(used, int) and 0 <= used <= memory.total,
                        'NVML per-process memory unavailable; cannot verify shared-device capacity')
                processes[int(process.pid)] = max(processes.get(int(process.pid), 0), int(used))
        owned = sum(used for pid, used in processes.items() if pid in owned_pids)
        return dict(total=int(memory.total), used=int(memory.used), free=int(memory.free),
                    owned_process_bytes=owned, external_and_driver_bytes=max(0, int(memory.used) - owned),
                    processes={str(pid): used for pid, used in sorted(processes.items())},
                    external_pids=sorted(pid for pid in processes if pid not in owned_pids))

    def close(self):
        self.api.nvmlShutdown()


def capacity_gate(samples, child_pids):
    require(samples and len(set(child_pids)) == 2, 'capacity gate requires two real concurrent CUDA processes')
    require(all(sample['total'] == samples[0]['total'] for sample in samples), 'GPU capacity changed')
    seen = set()
    both_seen = False
    for sample in samples:
        present = {int(pid) for pid in sample['processes']}
        seen |= present & set(child_pids)
        both_seen |= set(child_pids).issubset(present)
    require(seen == set(child_pids) and both_seen, 'both child CUDA contexts were never observed simultaneously')
    margin = max(2 * GIB, math.ceil(samples[0]['total'] * .10))
    minimum = min(sample['free'] for sample in samples)
    return dict(passed=minimum >= margin, required_margin_bytes=margin, minimum_free_bytes=minimum,
                peak_total_used_bytes=max(sample['used'] for sample in samples),
                external_and_driver_peak_bytes=max(sample['external_and_driver_bytes'] for sample in samples),
                external_pids=sorted({pid for sample in samples for pid in sample['external_pids']}),
                observed_concurrent_pids=sorted(child_pids))


def timing_summary(seconds):
    require(seconds and all(math.isfinite(x) and x > 0 for x in seconds), 'invalid timings')
    ordered = sorted(seconds)
    position = .95 * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    p95 = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return dict(count=len(seconds), median_seconds=statistics.median(seconds), p95_seconds=p95,
                total_seconds=math.fsum(seconds))


def acceptance_environment(allocator, *, gpu=0):
    require(allocator in ALLOCATORS, 'unknown allocator profile')
    return dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                OPENBLAS_NUM_THREADS='2', PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1',
                PYTORCH_CUDA_ALLOC_CONF=ALLOCATORS[allocator], CUBLAS_WORKSPACE_CONFIG=':4096:8')
