"""V41 两组静态源码登记；默认不导入模型、不执行环境检查。"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.ct_v41.registration import build_registration, verify_registration, LABELS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('action', choices=('register', 'verify'))
    parser.add_argument('--refresh-unstarted', action='store_true')
    parser.add_argument('--resolved', action='store_true', help='server only: check normalized formal Full config')
    args = parser.parse_args(argv)
    path = ROOT / 'cfgs/ct_seqtrack/41_registration.json'
    for source in sorted((ROOT / 'models/ct_v41').glob('*.py')):
        ast.parse(source.read_text(encoding='utf-8'), filename=str(source))
    if args.action == 'register':
        if path.exists() and not args.refresh_unstarted:
            raise FileExistsError('registration exists; --refresh-unstarted is required before any formal run')
        for manifest in (ROOT / 'output').rglob('run_manifest.json'):
            if json.loads(manifest.read_text(encoding='utf-8')).get('model') == 'ctseqtrackv41':
                raise RuntimeError('a V41 formal run exists; preserve its source and registration')
        value = build_registration()
        with path.open('w' if path.exists() else 'x', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        status = 'static_registration_written'
    else:
        value = verify_registration()
        status = 'static_registration_verified'
    if args.resolved:
        from models.ct_v41.config import load_config
        from models.ct_v41.entry import validate_formal_registration
        validate_formal_registration(load_config(ROOT / 'cfgs/ct_seqtrack/41_full_kitti.yaml'))
    print(json.dumps(dict(status=status, source_sha256=value['source']['sha256'], arms=LABELS,
                         normalized_config_checked=args.resolved, model_execution=False)))


if __name__ == '__main__':
    main()
