"""v28 可选数值审计；快照不进入训练状态、不消耗随机流。"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import torch

from utils.training_isolation import capture_global_rng_state


AUDIT_STEPS = (1, 2, 3, 4, 5, 10, 100)


def snapshot(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, dict):
        return {str(key): snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [snapshot(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def tensor_signature(value):
    tensor = value.detach().cpu().contiguous()
    return {
        'dtype': str(tensor.dtype), 'shape': list(tensor.shape),
        'sha256': hashlib.sha256(tensor.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),
    }


def compare_values(left, right, path='root'):
    """首个逐位差异；数值误差仅辅助定位，绝不覆盖字节不一致。"""
    if torch.is_tensor(left) or torch.is_tensor(right):
        if not (torch.is_tensor(left) and torch.is_tensor(right)):
            return {'path': path, 'reason': 'tensor/type mismatch'}
        a, b = tensor_signature(left), tensor_signature(right)
        if a == b:
            return None
        result = {'path': path, 'reason': 'tensor bytes differ', 'left': a, 'right': b}
        if left.shape == right.shape and left.numel():
            result['max_abs_error_diagnostic_only'] = float((left.double() - right.double()).abs().max())
        return result
    if type(left) is not type(right):
        return {'path': path, 'reason': 'type mismatch'}
    if isinstance(left, dict):
        if set(left) != set(right):
            return {'path': path, 'reason': 'keys differ',
                    'only_left': sorted(set(left) - set(right)),
                    'only_right': sorted(set(right) - set(left))}
        for key in left:
            difference = compare_values(left[key], right[key], f'{path}.{key}')
            if difference:
                return difference
        return None
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            return {'path': path, 'reason': 'length differs'}
        for index, (a, b) in enumerate(zip(left, right)):
            difference = compare_values(a, b, f'{path}[{index}]')
            if difference:
                return difference
        return None
    if left != right:
        return {'path': path, 'reason': 'value differs', 'left': left, 'right': right}
    return None


class B0NumericalAudit:
    """记录观测与更新边界，逐层指纹默认开启，完整激活仅诊断重跑开启。"""

    def __init__(self, model, directory, *, full_activations=False):
        self.model = model
        self.directory = Path(directory).resolve()
        protected = Path(__file__).resolve().parents[1] / 'output'
        if self.directory == protected or protected in self.directory.parents:
            raise ValueError('numerical audit may not write to protected output/')
        self.directory.mkdir(parents=True, exist_ok=True)
        if any(self.directory.iterdir()):
            raise FileExistsError(f'numerical audit requires a new empty directory: {self.directory}')
        self.full_activations = full_activations
        self.active = False
        self.step = 0
        self.row = None
        self.hooks = []
        self.before_parameters = None

    @classmethod
    def from_environment(cls, model):
        directory = os.environ.get('CT_V28_AUDIT_DIR')
        if not directory or not getattr(model, 'ct_enable_v28', False):
            return None
        return cls(model, directory,
                   full_activations=os.environ.get('CT_V28_AUDIT_ACTIVATIONS') == '1')

    def parameters(self):
        return [(name, value) for name, value in self.model.named_parameters()
                if not self.model._ct_any_plugin_parameter(name)]

    def b0_state(self):
        parameters = self.parameters()
        # 模式只比较 B0 网络树；无参数的 CT 指标、合同和私有 RNG 容器
        # 不属于观测网络，且在各臂中的存在性不同。
        roots = {name.split('.', 1)[0] for name, _ in parameters} | {'time_encoder'}
        buffers = {name: value for name, value in self.model.named_buffers()
                   if not self.model._ct_any_plugin_parameter(name)
                   and name.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}
        return {'parameters': snapshot(dict(parameters)), 'bn': snapshot(buffers),
                'training_flags': {name: child.training for name, child in self.model.named_modules()
                                   if not name or name.split('.', 1)[0] in roots},
                'rng': snapshot(capture_global_rng_state())}

    def adam(self, optimizer):
        return snapshot({name: optimizer.state.get(parameter, {})
                         for name, parameter in self.parameters()})

    def begin_observation(self, batch, step):
        self.step = int(step)
        self.active = self.step in AUDIT_STEPS
        if not self.active:
            return
        self.row = {'schema': 'ct_seqtrack.numerical_audit.v28', 'step': self.step,
                    'input': snapshot(batch), 'before_forward': self.b0_state(),
                    'activations': {}, 'activation_gradients': {}, 'pool_indices': {}}
        if self.step == 1:
            torch.save({'schema': self.row['schema'], 'step': 0,
                        'state': self.row['before_forward']}, self.directory / 'step_000.pt')
        self.calls = {}
        for name, module in self.model.named_modules():
            if not name or self.model._ct_any_plugin_parameter(name + '.') or list(module.children()):
                continue
            self.hooks.append(module.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def record(module, inputs, output):
            if not self.active:
                return
            call = self.calls.get(name, 0)
            self.calls[name] = call + 1
            key = f'{name}#{call}'
            def visit(value, suffix):
                if torch.is_tensor(value):
                    self.row['activations'][suffix] = (snapshot(value) if self.full_activations
                                                       else tensor_signature(value))
                    if value.requires_grad:
                        destination = self.row['activation_gradients']
                        # 保存当前事务容器，反传发生时已经关闭forward采集。
                        def gradient_hook(gradient):
                            destination[suffix] = (snapshot(gradient) if self.full_activations
                                                   else tensor_signature(gradient))
                        value.register_hook(gradient_hook)
                elif isinstance(value, (tuple, list)):
                    for index, item in enumerate(value):
                        visit(item, f'{suffix}[{index}]')
            visit(output, key)
            if module.__class__.__name__ == 'DeterministicMaxPool1d':
                value = inputs[0].detach()
                groups = int(module.output_size)
                indices = value.reshape(*value.shape[:-1], groups, value.shape[-1] // groups).max(-1).indices
                self.row['pool_indices'][key] = snapshot(indices)
        return record

    def end_observation(self, output, losses):
        if not self.active:
            return
        self.row['output'] = snapshot({key: value for key, value in output.items()
                                       if key in ('seg_logits', 'motion_cls', 'motion_pred',
                                                  'estimation_boxes', 'aux_estimation_boxes',
                                                  'updated_ref_boxs', 'pred_bc')})
        self.row['argmax'] = snapshot({key: output[key].argmax(1)
                                       for key in ('seg_logits', 'motion_cls') if key in output})
        self.row['losses'] = snapshot({key: value for key, value in losses.items()
                                       if key.startswith('loss_') and key not in (
                                           'loss_b1_transaction', 'loss_b2_transaction',
                                           'loss_b3_transaction', 'loss_plugin_transaction')})
        self.row['after_forward'] = self.b0_state()
        self.active = False
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def verify_mechanism(self, before):
        difference = compare_values(before, self.b0_state(), 'mechanism_boundary')
        if difference:
            torch.save(difference, self.directory / f'mechanism_failure_{self.step:03d}.pt')
            raise RuntimeError(f'v28 mechanism changed B0/BN/training flags/RNG: {difference}')

    def before_optimizer(self, optimizer):
        if self.row is None or self.step not in AUDIT_STEPS:
            return
        self.row['gradients'] = snapshot({name: parameter.grad for name, parameter in self.parameters()})
        self.row['adam_before'] = self.adam(optimizer)
        self.row['parameters_before_update'] = snapshot(dict(self.parameters()))
        self.before_parameters = self.row['parameters_before_update']
        self.row['b0_optimizer_group'] = snapshot({key: value for group in optimizer.param_groups
                                                  if group.get('name') == 'b0'
                                                  for key, value in group.items() if key != 'params'})
        self.row['parameter_order'] = [name for name, _ in self.parameters()]

    def after_optimizer(self, optimizer):
        if self.row is None or self.step not in AUDIT_STEPS:
            return
        self.row['after_update'] = self.b0_state()
        self.row['adam_after'] = self.adam(optimizer)
        self.row['actual_parameter_delta'] = {
            name: value - self.before_parameters[name]
            for name, value in self.row['after_update']['parameters'].items()}
        torch.save(self.row, self.directory / f'step_{self.step:03d}.pt')
        self.row = self.before_parameters = None
