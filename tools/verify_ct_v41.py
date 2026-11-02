"""可选静态检查：读取当前源码与配置，不需要生成、上传或刷新登记。"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.ct_v41.provenance import source_identity


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('action', nargs='?', choices=('verify',), default='verify')
    parser.add_argument('--resolved', action='store_true', help='also parse the current Full configuration')
    args = parser.parse_args(argv)
    paths = list((ROOT / 'models/ct_v41').glob('*.py')) + [
        ROOT / 'models/ctseqtrackv41.py', ROOT / 'datasets/kitti_v41.py']
    for path in paths:
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    if args.resolved:
        from models.ct_v41.config import load_config
        load_config(ROOT / 'cfgs/ct_seqtrack/41_full_kitti.yaml')
    source = source_identity()
    print(json.dumps(dict(status='static_source_checked', source_sha256=source['sha256'],
        source_hash_policy=source['inventory_policy'], registration_required=False,
        normalized_config_checked=args.resolved, model_execution=False)))


if __name__ == '__main__':
    main()
