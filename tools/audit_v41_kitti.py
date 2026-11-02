"""V41 KITTI Car 输入清单：只对白名单缺失保留空观测，不运行模型。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.audit_kitti_inputs import inspect_dataset

KNOWN_MISSING = frozenset(f'velodyne/0001/{frame:06d}.bin' for frame in range(177, 181))


def inspect_v41(path):
    value = inspect_dataset(path, 'Car')
    allowed = set(value['missing_files']) & KNOWN_MISSING
    value['blocking_findings'] = [finding for finding in value['blocking_findings']
                                  if finding not in {name + ': missing' for name in allowed}]
    value['warnings'].extend(name + ': known_missing_empty_observation_frame_preserved' for name in sorted(allowed))
    value['known_missing_files'] = sorted(allowed)
    value['input_readiness']['ready'] = (not value['blocking_findings'] and
        all(row['ready'] for row in value['role_readiness'].values()))
    value['v41_policy'] = dict(geometry='open3dsot_velodyne_v1',
        missing='whitelist_empty_preserve_endpoint_otherwise_error',
        train_cache='GT10 only when predicted ROI fully contained; raw fallback',
        auxiliary_dev_test_preload=-1, evaluation_size='first_frame')
    value['semantics']['missing_cloud_policy'] = value['v41_policy']['missing']
    value['semantics']['gt_preload_crop'] = 'normal_train_cache_only_with_raw_equivalence_guard'
    value['schema'] = 'ct_seqtrack.v41.kitti_input_inventory.v1'
    value['v41_inventory_sha256'] = hashlib.sha256(json.dumps(
        [value['inventory_sha256'], value['v41_policy']], sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--path', required=True)
    parser.add_argument('--output-dir')
    args = parser.parse_args(argv)
    allowed = (ROOT / 'artifacts/ct_checks').resolve()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    destination = (Path(args.output_dir).resolve() if args.output_dir else
                   allowed / ('v41_kitti_inputs_' + stamp + '_' + uuid.uuid4().hex[:8]))
    if allowed not in destination.parents or destination.exists():
        raise ValueError('output must be a new directory below artifacts/ct_checks')
    value = inspect_v41(args.path)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'inventory.json').write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(report=str(destination / 'inventory.json'),
        readiness=value['input_readiness'], known_missing=value['known_missing_files'],
        blockers=value['blocking_findings']), ensure_ascii=False))
    return 0 if value['input_readiness']['ready'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
