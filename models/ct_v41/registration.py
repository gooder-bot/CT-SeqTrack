"""V41 静态文件登记；生成过程只使用标准库，不导入 torch 或检查本机环境。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .identity import (BASELINE_COMMIT, BASELINE_IMPLEMENTATION, REGISTRATION_SCHEMA,
                       SOURCE_INVENTORY_POLICY, retired_source_path)

ROOT = Path(__file__).resolve().parents[2]
LABELS = ('full', 'seqtrack')
CONFIGS = {'full': 'cfgs/ct_seqtrack/41_full_kitti.yaml',
           'seqtrack': 'cfgs/seqtrack_baseline/41_seqtrack_kitti.yaml'}
BASELINE_RECEIPT = 'cfgs/seqtrack_baseline/41_source_manifest.json'


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
        relative = CONFIGS[label]
        runs.append(dict(label=label, arm=label, cfg=relative, yaml_sha256=file_hash(root / relative),
                         execution_profile='optimized_fp32' if label == 'full' else 'original_seqtrack',
                         external=label == 'seqtrack'))
    baseline = list((root / 'models/ct_v40').glob('*.py')) + [root / 'models/ctseqtrackv40.py']
    tools = sorted((root / 'tools').glob('*v41*.py')) + [root / 'tools/audit_kitti_inputs.py']
    tests = sorted((root / 'tests').glob('*v41*.py')) + [root / 'tests/test_kitti_input_audit.py']
    return dict(schema=REGISTRATION_SCHEMA, source=source_identity(root), runs=runs,
        baseline=dict(commit=BASELINE_COMMIT, implementation=BASELINE_IMPLEMENTATION,
                      files={p.relative_to(root).as_posix(): file_hash(p) for p in sorted(baseline)}),
        seqtrack_baseline=dict(source_manifest=BASELINE_RECEIPT,
            source_manifest_sha256=file_hash(root / BASELINE_RECEIPT),
            provenance=json.loads((root / BASELINE_RECEIPT).read_text(encoding='utf-8'))),
        engineering_files={p.relative_to(root).as_posix(): file_hash(p) for p in tools + tests},
        registered_main_model='full', policy_activation='active_from_epoch1',
        dataset=dict(name='KITTI Tracking', category='Car', train=list(range(17)),
            dev=[17, 18], test=[19, 20], geometry='open3dsot_velodyne_v1',
            frame_period_seconds=.1, stride=1, size='first_frame',
            missing='0001/000177..000180 only; empty observation, preserve endpoint'),
        budget=dict(epochs_per_run=60, total_epochs=120, batch_size=16, seed=42,
                    optimizer_steps='derive from actual registered sampler; never substitute an estimate'),
        evaluation=dict(checkpoint_epochs=[58, 59, 60], final=60, best_epoch_selection=False),
        interpretation='two-system comparison; different original training recipes, not a module ablation',
        local_execution='not_run_by_user_request; static file registration only',
        limitations=['KITTI Car seed42; late3 is not independent replication',
                     'no score, stability or speed claim before server acceptance and formal results'])


def verify_baseline_receipt(registration, root=ROOT):
    item = registration.get('seqtrack_baseline', {})
    if item.get('source_manifest') != BASELINE_RECEIPT:
        raise ValueError('v41 original SeqTrack provenance is missing')
    path = Path(root) / BASELINE_RECEIPT
    if (item.get('source_manifest_sha256') != file_hash(path)
            or item.get('provenance') != json.loads(path.read_text(encoding='utf-8'))):
        raise ValueError('v41 SeqTrack source receipt changed; verify both arms before registration')
    provenance = item['provenance']
    digest = hashlib.sha256(json.dumps(provenance['files'], sort_keys=True,
                                      separators=(',', ':')).encode()).hexdigest()
    if digest != provenance.get('source_sha256'):
        raise ValueError('v41 SeqTrack source receipt has an inconsistent inventory digest')
    if file_hash(Path(root) / CONFIGS['seqtrack']) != provenance['files'].get('cfgs/seqtrack3d_kitti_v41_baseline.yaml'):
        raise ValueError('v41 mirrored SeqTrack config differs from its deployed source receipt')
    return item


def verify_registration(path=None, root=ROOT):
    path = Path(path) if path is not None else Path(root) / 'cfgs/ct_seqtrack/41_registration.json'
    registered = json.loads(path.read_text(encoding='utf-8'))
    actual = build_registration(root)
    if registered != actual:
        changed = sorted(k for k in registered.keys() | actual.keys() if registered.get(k) != actual.get(k))
        raise ValueError('V41 registration mismatch: ' + ', '.join(changed) +
                         '; upload matching source; do not refresh to hide a deployment mismatch')
    verify_baseline_receipt(registered, root)
    return registered
