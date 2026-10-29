"""V40 静态文件登记；生成过程只使用标准库，不导入 torch 或检查本机环境。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .identity import (BASELINE_COMMIT, BASELINE_IMPLEMENTATION, REGISTRATION_SCHEMA,
                       SOURCE_INVENTORY_POLICY, retired_source_path)

ROOT = Path(__file__).resolve().parents[2]
LABELS = ('b0', 'b1', 'b1_mtp', 'full')


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity(root=ROOT):
    root = Path(root)
    paths = [root / 'main.py', root / 'requirement.txt']
    for name in ('models', 'datasets', 'utils'):
        paths.extend((root / name).rglob('*.py'))
    files = {path.relative_to(root).as_posix(): file_hash(path) for path in sorted(paths)
             if not retired_source_path(path.relative_to(root).as_posix())}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return dict(sha256=digest, files=files, inventory_policy=SOURCE_INVENTORY_POLICY)


def build_registration(root=ROOT):
    root = Path(root)
    runs = []
    for label in LABELS:
        relative = f'cfgs/ct_seqtrack/40_{label}_mini.yaml'
        runs.append(dict(label=label, arm=label, cfg=relative, yaml_sha256=file_hash(root / relative),
                         execution_profile='optimized_fp32'))
    baseline = list((root / 'models/ct_v38').glob('*.py')) + [root / 'models/ctseqtrackv38.py']
    tools = sorted((root / 'tools').glob('*v40*.py'))
    tests = sorted((root / 'tests').glob('test_ct_v40*.py'))
    return dict(schema=REGISTRATION_SCHEMA, source=source_identity(root), runs=runs,
        baseline=dict(commit=BASELINE_COMMIT, implementation=BASELINE_IMPLEMENTATION,
                      files={p.relative_to(root).as_posix(): file_hash(p) for p in sorted(baseline)}),
        engineering_files={p.relative_to(root).as_posix(): file_hash(p) for p in tools + tests},
        registered_main_model='full', policy_activation='active_from_epoch1',
        budget=dict(epochs_per_run=60, total_epochs=240,
                    optimizer_steps='derive from actual registered sampler; never substitute an estimate'),
        waves=[['b0', 'b1'], ['b1_mtp', 'full']], physical_gpu=0,
        interpretation=dict(b1_minus_b0='learnable B1', b1_mtp_minus_b1='additional MTP loss',
                            full_minus_b1_mtp='B3 plus search protection jointly'),
        local_execution='not_run_by_user_request; static file registration only',
        limitations=['mini Car seed42; late3 is not independent replication',
                     'no score, stability or speed claim before server acceptance and formal results'])


def verify_registration(path=None, root=ROOT):
    path = Path(path) if path is not None else Path(root) / 'cfgs/ct_seqtrack/40_registration.json'
    registered = json.loads(path.read_text(encoding='utf-8'))
    actual = build_registration(root)
    if registered != actual:
        changed = sorted(k for k in registered.keys() | actual.keys() if registered.get(k) != actual.get(k))
        raise ValueError('V40 registration mismatch: ' + ', '.join(changed) +
                         '; upload matching source; do not refresh to hide a deployment mismatch')
    return registered
