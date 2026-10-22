"""生成/显式刷新本次尚未启动的 v38 三组登记；不改旧登记、运行或权重。"""
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


def build_registration():
    rows = []
    for label in RECIPES:
        relative = f'cfgs/ct_seqtrack/38_{label}_mini.yaml'
        cfg = load_config(ROOT / relative)
        rows.append(dict(label=label, arm=cfg.v38_arm, cfg=relative,
            config_sha256=config_identity(cfg), execution_profile=cfg.execution_profile,
            training_execution_recipe=recipe_identity(cfg),
            yaml_sha256=hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()))
    from tools.compare_v35_b0 import REGISTERED_FRAME_SHA256
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
    parser.add_argument('--refresh', action='store_true',
        help='仅在此次实现收束且未启动三组时显式刷新新登记')
    args = parser.parse_args(argv)
    path = ROOT / 'cfgs/ct_seqtrack' / REGISTRATION_FILE
    if path.exists() and not args.refresh:
        raise FileExistsError('v38 registration exists; explicit --refresh required')
    # 目录名不是身份；递归检查本地 output 内所有自定义子目录。
    # 仓外 --log_dir 无法在此自动发现，刷新前仍须确认没有这类已启动运行。
    for run in (ROOT / 'output').rglob('run_manifest.json'):
        if json.loads(run.read_text(encoding='utf-8')).get('model') == 'ctseqtrackv38':
            raise RuntimeError('v38 run already exists; preserve its registration and source: ' + str(run))
    value = build_registration()
    content = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    with path.open('w' if path.exists() else 'x', encoding='utf-8', newline='\n') as stream:
        stream.write(content)
    print(json.dumps(dict(path=str(path), source_sha256=value['source']['sha256'],
                          recipes=list(RECIPES)), ensure_ascii=False))


if __name__ == '__main__':
    main()
