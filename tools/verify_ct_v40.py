"""V40 静态登记与上传后验证；register 不导入项目模型或探测运行环境。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ''):
    sys.path.insert(0, str(ROOT))

from models.ct_v40.registration import build_registration, verify_registration, LABELS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('action', choices=('register', 'verify'))
    parser.add_argument('--refresh-unstarted', action='store_true')
    args = parser.parse_args(argv)
    path = ROOT / 'cfgs/ct_seqtrack/40_registration.json'
    if args.action == 'verify':
        value = verify_registration(path)
        # 本分支仅供上传后执行；四份解析配置均与当前实际运行身份匹配。
        from models.ct_v40.config import load_config
        from models.ct_v40.entry import validate_formal_registration
        for label in LABELS:
            validate_formal_registration(load_config(ROOT / f'cfgs/ct_seqtrack/40_{label}_mini.yaml'))
        print(json.dumps(dict(status='verified', source_sha256=value['source']['sha256'], arms=LABELS)))
        return
    if path.exists() and not args.refresh_unstarted:
        raise FileExistsError('registration exists; explicit --refresh-unstarted required')
    for manifest in (ROOT / 'output').rglob('run_manifest.json'):
        if json.loads(manifest.read_text(encoding='utf-8')).get('model') == 'ctseqtrackv40':
            raise RuntimeError('V40 formal run already exists; preserve source and registration: ' + str(manifest))
    # 外部 --log_dir 不可自动穷举；refresh 只适用于已确认从未启动的本地实现阶段。
    value = build_registration()
    with path.open('w' if path.exists() else 'x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(dict(status='static_registration_written', path=str(path),
                         runtime_checks_executed=False, source_sha256=value['source']['sha256'])))


if __name__ == '__main__':
    main()
