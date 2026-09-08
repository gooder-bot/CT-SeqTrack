"""从 v28 审计中的参数/梯度/Adam状态重放一次更新，不进入模型训练。"""
import argparse
import copy
import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.v28_numerical_audit import compare_values, snapshot


def replay_adam(row, device='cpu'):
    if row.get('schema') != 'ct_seqtrack.numerical_audit.v28' or row.get('step', 0) < 1:
        raise ValueError('Adam replay requires an optimizer-step v28 snapshot')
    keys = ('parameters_before_update', 'gradients', 'adam_before', 'b0_optimizer_group',
            'parameter_order', 'after_update', 'adam_after', 'actual_parameter_delta')
    if any(key not in row for key in keys):
        raise ValueError('incomplete Adam replay snapshot')
    names = row['parameter_order']
    group = copy.deepcopy(row['b0_optimizer_group'])
    if group.get('foreach') is not False or group.get('fused') is not False:
        raise ValueError('v28 Adam replay requires foreach=False and fused=False')
    parameters = {name: torch.nn.Parameter(row['parameters_before_update'][name].to(device).clone())
                  for name in names}
    optimizer = torch.optim.Adam([dict(group, params=list(parameters.values()))],
                                 foreach=False, fused=False)
    for name, parameter in parameters.items():
        gradient = row['gradients'][name]
        parameter.grad = None if gradient is None else gradient.to(device).clone()
        state = {}
        for key, value in row['adam_before'][name].items():
            if torch.is_tensor(value):
                target = 'cpu' if key == 'step' and not group.get('capturable', False) else device
                value = value.to(target).clone()
            state[key] = copy.deepcopy(value)
        optimizer.state[parameter] = state
    optimizer.step()
    replayed = snapshot(parameters)
    deltas = {name: value - row['parameters_before_update'][name] for name, value in replayed.items()}
    replayed_adam = snapshot({name: optimizer.state[parameter] for name, parameter in parameters.items()})
    differences = {
        'parameters': compare_values(row['after_update']['parameters'], replayed, 'parameters'),
        'actual_parameter_delta': compare_values(row['actual_parameter_delta'], deltas, 'delta'),
        'adam': compare_values(row['adam_after'], replayed_adam, 'adam'),
    }
    return dict(schema='ct_seqtrack.adam_replay.v28', step=row['step'], device=str(device),
                passed=all(value is None for value in differences.values()), differences=differences,
                scope='one isolated B0 Adam update; compare on the original GPU/software for CUDA attribution')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)
    row = torch.load(args.snapshot, map_location='cpu', weights_only=False)
    result = replay_adam(row, args.device)
    serialized = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if args.output:
        protected = (ROOT / 'output').resolve()
        if args.output.resolve() == protected or protected in args.output.resolve().parents:
            raise ValueError('replay report may not write to protected output/')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding='utf-8')
    print(serialized)
    if not result['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
