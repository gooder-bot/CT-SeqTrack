"""生成、显式刷新或只读核验 v38 三组登记；不改旧登记、运行或权重。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ''):
    sys.path.insert(0, str(ROOT))

from models.ct_v38.config import (REGISTRATION_FILE, REGISTRATION_SCHEMA, RECIPES,
    config_identity, load_config, recipe_identity)
from models.ct_v38.entry import source_identity
from models.ct_v38.identity import FROZEN_MODEL_SCHEMA


def build_registration():
    rows = []
    for label in RECIPES:
        relative = f'cfgs/ct_seqtrack/38_{label}_mini.yaml'
        cfg = load_config(ROOT / relative)
        rows.append(dict(label=label, arm=cfg.v38_arm, cfg=relative,
            config_sha256=config_identity(cfg), execution_profile=cfg.execution_profile,
            training_execution_recipe=recipe_identity(cfg),
            yaml_sha256=hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()))
    from tools._tracking_audit import REGISTERED_FRAME_SHA256
    return dict(schema=REGISTRATION_SCHEMA, source=source_identity(), runs=rows,
        registered_main_model='full_standard', policy_activation='active_from_epoch1',
        budget=dict(epochs_per_run=60, optimizer_steps_per_run=71700,
                    total_epochs=180, total_optimizer_steps=215100),
        reference_frames_sha256={name: list(REGISTERED_FRAME_SHA256[name])
                                 for name in ('reference', 'baseline')},
        limitations=['single seed42 mini Car; late3 is not independent replication',
                    'no new formal results are implied by this registration'],
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--refresh', action='store_true',
        help='仅在此次实现收束且未启动三组时显式刷新新登记')
    action.add_argument('--verify', action='store_true',
        help='只读检查当前源码、三组配置及工具与登记一致；上传后使用，不刷新登记')
    args = parser.parse_args(argv)
    path = ROOT / 'cfgs/ct_seqtrack' / REGISTRATION_FILE
    if args.verify:
        registered = json.loads(path.read_text(encoding='utf-8'))
        actual = build_registration()
        if registered != actual:
            changed = sorted(key for key in registered.keys() | actual.keys()
                             if registered.get(key) != actual.get(key))
            extra = sorted(set(actual.get('source', {}).get('files', {})) -
                           set(registered.get('source', {}).get('files', {})))
            detail = '; unexpected source files: ' + ', '.join(extra[:8]) if extra else ''
            raise ValueError('v38 registration mismatch: ' + ', '.join(changed) + detail +
                             '; upload the matching complete source and apply retired-file deletions. '
                             'Do not refresh the registration to hide a deployment mismatch.')
        print(json.dumps(dict(status='verified', path=str(path),
                              source_sha256=actual['source']['sha256'],
                              recipes=list(RECIPES)), ensure_ascii=False))
        return
    if path.exists() and not args.refresh:
        raise FileExistsError('v38 registration exists; explicit --refresh required')
    # 目录名不是身份；递归检查本地 output 内所有自定义子目录。
    # 仓外 --log_dir 无法在此自动发现，刷新前仍须确认没有这类已启动运行。
    for run in (ROOT / 'output').rglob('run_manifest.json'):
        manifest = json.loads(run.read_text(encoding='utf-8'))
        if (manifest.get('model') == 'ctseqtrackv38'
                and manifest.get('schema') != FROZEN_MODEL_SCHEMA):
            raise RuntimeError('v38 run already exists (clean1 or unknown identity); preserve its registration and source: ' + str(run))
    value = build_registration()
    content = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    with path.open('w' if path.exists() else 'x', encoding='utf-8', newline='\n') as stream:
        stream.write(content)
    print(json.dumps(dict(path=str(path), source_sha256=value['source']['sha256'],
                          recipes=list(RECIPES)), ensure_ascii=False))


if __name__ == '__main__':
    main()
