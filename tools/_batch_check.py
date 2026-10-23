"""工程 batch 检查共用的独占输出和梯度/显存摘要。"""
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import uuid
ROOT = Path(__file__).resolve().parents[1]
CHECK_ROOT = ROOT / 'artifacts' / 'ct_checks'

def report_destination(output, *, version='38_clean1'):
    if output is None:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        output = CHECK_ROOT / f'v{version}_batch_{stamp}_{os.getpid()}_{uuid.uuid4().hex[:12]}' / 'report.json'
    output = Path(output).resolve()
    if CHECK_ROOT.resolve() not in output.parents or output.suffix.lower() != '.json':
        raise ValueError('batch-check JSON must be under artifacts/ct_checks')
    if output.exists():
        raise FileExistsError('refusing to overwrite an existing batch-check report: ' + str(output))
    return output


def gradient_summary(model):
    """逐参数归约，避免为了记录全局范数而拼接一份完整梯度副本。"""
    import torch
    total_norm, tensors, nonzero, missing = 0., 0, 0, []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing.append(name)
            continue
        gradient = parameter.grad.detach()
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError('nonfinite gradient: ' + name)
        norm = float(torch.linalg.vector_norm(gradient))
        total_norm = math.hypot(total_norm, norm)
        tensors += 1
        nonzero += int(norm > 0)
    if not tensors or not math.isfinite(total_norm) or total_norm <= 0:
        raise FloatingPointError('batch must produce finite, nonzero trainable gradients')
    return dict(all_finite=True, global_l2_norm=total_norm, tensors_with_gradient=tensors,
                tensors_with_nonzero_gradient=nonzero, parameters_without_gradient=missing)


def cuda_memory(device):
    import torch
    if device != 'cuda' or not torch.cuda.is_available():
        return None
    return dict(device_name=torch.cuda.get_device_name(),
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 2 ** 20,
                peak_reserved_mib=torch.cuda.max_memory_reserved() / 2 ** 20)
