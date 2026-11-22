import json

import pytest
import yaml

from models.ct_v41.config import config_identity, load_config
from tools.compare_v41_single_view import MODES, ROOT, compare


def write_json(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def training_log(root, mode, epochs):
    root.mkdir()
    (root / 'epochs').mkdir()
    (root / 'training_audits').mkdir()
    cfg = dict(load_config(ROOT / 'cfgs' / 'ct_seqtrack' / f'41_joint_{mode}.yaml'))
    (root / 'resolved_config.yaml').write_text(yaml.safe_dump(cfg), encoding='utf-8')
    source_hash = ('a' if mode == MODES[0] else 'b') * 64
    weights = ('0' if mode == MODES[0] else '1') * 64
    manifest = dict(run_id=mode, run_type='formal', config_sha256=config_identity(cfg),
                    actual_recipe={'v41_history_training': mode}, source={'sha256': source_hash},
                    history_bank_sha256='c' * 64, initial_parameter_hash='d' * 64,
                    dataset_manifest={'train': 'kitti'}, input_metadata={'sha256': 'e' * 64})
    write_json(root / 'run_manifest.json', manifest)
    views = 4 if mode == MODES[0] else 1
    for epoch in range(1, epochs + 1):
        name = f'epoch={epoch:03d}.json'
        sampled = 16 * views
        replay = 1 if epoch > 2 else 0
        audit = dict(run_id=mode, config_sha256=manifest['config_sha256'], completed_epoch=epoch,
                     epoch_complete=True, rows=sampled, optimizer_steps=views,
                     history_training=dict(mode=mode, bank_sha256='c' * 64, augmentation={'full_density': 3},
                                           weights_sha256=weights if epoch <= 2 else None,
                                           endpoints={'reset': sampled - replay, 'natural_replay': replay}),
                     sampler=dict(lengths=[17], rows=sampled, batches=views), auxiliary=dict(complete=True, rows=3),
                     joint_replay=dict(used={'lateral': replay}, donor_weights_sha256=weights if replay else None))
        write_json(root / 'training_audits' / name, audit)
        evaluation = dict(epoch=epoch, checkpoint_epoch=epoch, split='test', complete_coverage=True,
                          success=epoch + views, precision=90 - epoch, frames=10)
        write_json(root / 'epochs' / name, dict(epoch=epoch, evaluation={'test': evaluation},
                   losses={'train/loss_total_epoch': .4, 'train/optimization_loss_epoch': .5},
                   timing={'train_seconds': views, 'eval_seconds': 2}))


def test_compares_same_epoch_and_late3_with_independent_training_budgets(tmp_path):
    four, single = (tmp_path / mode for mode in MODES)
    training_log(four, MODES[0], 60)
    training_log(single, MODES[1], 60)
    report = compare(four, single)
    assert report['e60']['single_minus_four_pp'] == {'success': -3, 'precision': 0}
    assert report['late3'][MODES[1]] == {'success': 60, 'precision': 31}
    assert report['totals'][MODES[0]]['optimizer_steps'] == 240
    assert report['totals'][MODES[1]]['train_seconds'] == 60
    assert report['epochs'][2][MODES[1]]['clean_resets'] == 12
    audit_path = single / 'training_audits' / 'epoch=003.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    audit['joint_replay']['donor_weights_sha256'] = report['epochs'][2][MODES[0]]['e2_donor']
    write_json(audit_path, audit)
    with pytest.raises(ValueError, match='本组 E2'):
        compare(four, single)


def test_compares_common_completed_prefix_and_rejects_changed_recipe(tmp_path):
    four, single = (tmp_path / mode for mode in MODES)
    training_log(four, MODES[0], 2)
    training_log(single, MODES[1], 1)
    report = compare(four, single)
    assert report['common_epochs'] == 1 and report['e60'] is None and report['late3'] is None
    path = single / 'resolved_config.yaml'
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    cfg['lr'] /= 2
    path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
    manifest = json.loads((single / 'run_manifest.json').read_text(encoding='utf-8'))
    manifest['config_sha256'] = config_identity(cfg)
    write_json(single / 'run_manifest.json', manifest)
    with pytest.raises(ValueError, match='lr'):
        compare(four, single)
