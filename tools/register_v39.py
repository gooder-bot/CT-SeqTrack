"""生成或只读核验 v39 两组登记；实现收束前不执行最终登记。"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ''):
    sys.path.insert(0, str(ROOT))

from models.ct_v39.config import (REGISTRATION_FILE, REGISTRATION_SCHEMA, RECIPES,
    config_identity, load_config, recipe_identity)
from models.ct_v39.entry import source_identity


def build_registration():
    rows = []
    for label in RECIPES:
        relative = f'cfgs/ct_seqtrack/39_{label}_mini.yaml'
        cfg = load_config(ROOT / relative)
        rows.append(dict(label=label, arm=cfg.v39_arm, cfg=relative,
            config_sha256=config_identity(cfg), execution_profile=cfg.execution_profile,
            training_execution_recipe=recipe_identity(cfg),
            yaml_sha256=hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()))
    return dict(schema=REGISTRATION_SCHEMA, source=source_identity(), runs=rows,
        registered_main_models=['full_a', 'full_b'], policy_activation='active_from_epoch1',
        training_policy='same_full_architecture_and_rollout_only_lr_differs',
        budget=dict(epochs_per_run=60, total_epochs=120,
                    optimizer_steps='derived_from_registered_complete_sampler_each_epoch'),
        evaluation=dict(final_epoch=60, late3_epochs=[58, 59, 60],
                        primary_mode='accuracy', secondary_mode='adaptive'),
        limitations=['single seed42 mini Car; late3 is not independent replication',
                    'two runs compare full learning-rate schedules, not E/G/M training ablations',
                    'registration does not imply successful training or measured gains'],
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--refresh', action='store_true', help='实现收束且没有已启动 v39 时显式刷新')
    action.add_argument('--verify', action='store_true', help='只读核验源码、配方和工具，不刷新')
    args = parser.parse_args(argv)
    path = ROOT / 'cfgs/ct_seqtrack' / REGISTRATION_FILE
    if args.verify:
        registered = json.loads(path.read_text(encoding='utf-8'))
        actual = build_registration()
        if registered != actual:
            changed = sorted(key for key in registered.keys() | actual.keys()
                             if registered.get(key) != actual.get(key))
            raise ValueError('v39 registration mismatch: ' + ', '.join(changed)
                             + '; deploy the matching source; do not refresh to hide drift')
        print(json.dumps(dict(status='verified', path=str(path), recipes=list(RECIPES),
                              source_sha256=actual['source']['sha256']), ensure_ascii=False))
        return
    if path.exists() and not args.refresh:
        raise FileExistsError('v39 registration exists; explicit --refresh required')
    # 已有正式运行时不得重绑 source。仓外 log_dir 由调用者在实施时核实。
    for run in (ROOT / 'output').rglob('run_manifest.json'):
        manifest = json.loads(run.read_text(encoding='utf-8'))
        if manifest.get('model') == 'ctseqtrackv39':
            raise RuntimeError('v39 run already exists; preserve its source/registration: ' + str(run))
    value = build_registration()
    content = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    with path.open('w' if path.exists() else 'x', encoding='utf-8', newline='\n') as stream:
        stream.write(content)
    print(json.dumps(dict(path=str(path), source_sha256=value['source']['sha256'],
                          recipes=list(RECIPES)), ensure_ascii=False))


if __name__ == '__main__':
    main()
