"""逐位比较两个 v28 工程审计；首个差异返回非零退出码。"""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.v28_numerical_audit import AUDIT_STEPS, compare_values


def compare_audits(left, right, *, steps=AUDIT_STEPS):
    required = (0, *steps)
    phases = ('input', 'before_forward', 'activations', 'pool_indices', 'argmax',
              'output', 'losses', 'after_forward', 'activation_gradients',
              'gradients', 'adam_before', 'parameters_before_update',
              'b0_optimizer_group', 'parameter_order', 'actual_parameter_delta',
              'adam_after', 'after_update')
    for step in required:
        paths = [Path(root) / f'step_{step:03d}.pt' for root in (left, right)]
        if not all(path.is_file() for path in paths):
            return {'passed': False, 'step': step, 'reason': 'required snapshot missing',
                    'paths': [str(path) for path in paths]}
        a, b = [torch.load(path, map_location='cpu', weights_only=False) for path in paths]
        if step == 0:
            difference = compare_values(a, b)
            if difference:
                return {'passed': False, 'step': step, 'phase': 'initial', 'difference': difference}
        else:
            for phase in phases:
                if phase not in a or phase not in b:
                    return {'passed': False, 'step': step, 'phase': phase,
                            'reason': 'required phase missing'}
                difference = compare_values(a[phase], b[phase], phase)
                if difference:
                    return {'passed': False, 'step': step, 'phase': phase, 'difference': difference}
    return {'passed': True, 'acceptance': 'bitwise_equal', 'steps': list(required),
            'scope': 'selected observation transactions and optimizer boundaries; no score claim'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left', type=Path)
    parser.add_argument('right', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = compare_audits(args.left, args.right)
    result.update(schema='ct_seqtrack.numerical_comparison.v28',
                  left=str(args.left.resolve()), right=str(args.right.resolve()))
    serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.output:
        protected = (ROOT / 'output').resolve()
        if args.output.resolve() == protected or protected in args.output.resolve().parents:
            raise ValueError('comparison may not write to protected output/')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding='utf-8')
    print(serialized)
    if not result['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
