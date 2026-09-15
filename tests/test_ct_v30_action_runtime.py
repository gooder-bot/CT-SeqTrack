"""v30 CLI dispatch and dataset-neutral closed-loop exports."""
import copy
import json
from types import SimpleNamespace

import torch

from tests.test_ct_v27_actions import _rows
from tools.ct_action_v27_runtime import TrackerClosedLoopRunner, rows_schema, use_v27_runtime
from utils.dataset_protocol_v30 import build_dataset_manifest


def test_runtime_detects_v30_flag_and_config_without_legacy_flags(tmp_path):
    cfg = tmp_path / 'config.yaml'
    cfg.write_text('ct_enable_v30: true\n', encoding='utf-8')
    assert use_v27_runtime(['--v30'])
    assert use_v27_runtime(['--config', str(cfg)])
    assert use_v27_runtime(['--config=' + str(cfg)])
    assert rows_schema({'ct_enable_v30': True, 'ct_enable_v29': True}) == 'ct_seqtrack.action_rows.v30'


def test_server_launcher_default_only_prints_and_uses_registered_paths(tmp_path, capsys):
    from tools.run_ct_v30_server import main
    destination = tmp_path / 'must_not_be_created'
    result = main(['--dataset', 'kitti', '--arm', 'full_cfc', '--gpu', '2',
                   '--log-dir', str(destination)])
    assert not result['launch'] and not destination.exists()
    assert result['data_root'] == '/home/lishengjie/data/cxtrack/training'
    assert result['argv'][result['argv'].index('--cfg')+1].endswith('30_full_cfc_kitti.yaml')
    assert '--checkpoint' not in result['argv'] and '--init_checkpoint' not in result['argv']
    assert json.loads(capsys.readouterr().out)['environment']['CUDA_VISIBLE_DEVICES'] == '2'


def test_v30_cuda_h3_and_resume_plans_match_engineering_contract():
    from pathlib import Path
    from tools.benchmark_ct_v29_h3 import build_command, ROOT
    from tools.check_ct_v28_resume import build_plan
    from models.ct_variant import configure_ct_variant
    from utils.v30_contracts import validate_v30_scratch_contract
    from utils.config import load_yaml_config
    cfg_path = ROOT / 'cfgs/ct_seqtrack/30_full_cfc_mini.yaml'
    output = ROOT / 'artifacts/ct_checks/v30_unit_plan_only'
    plan = build_command('/data', output, arm='full_cfc', config_path=cfg_path)
    argv = plan['argv']
    config = load_yaml_config(cfg_path)
    configure_ct_variant(config)
    for cli, key in (('--epoch', 'epoch'), ('--workers', 'workers'),
                     ('--limit_train_batches', 'limit_train_batches'),
                     ('--limit_val_batches', 'limit_val_batches'),
                     ('--check_val_every_n_epoch', 'check_val_every_n_epoch')):
        config[key] = int(argv[argv.index(cli) + 1])
    config.update(ct_engineering_check=True, log_dir=str(output))
    validate_v30_scratch_contract(config)
    resume_config, resume = build_plan(cfg_path, '/data', output)
    configure_ct_variant(resume_config)
    resume_config['log_dir'] = str(Path(resume['output']) / 'continuous')
    validate_v30_scratch_contract(resume_config)
    assert not output.exists()


def test_kitti_runner_uses_generic_scene_identity_and_keeps_max_action_q():
    manifest = build_dataset_manifest(dict(dataset='kitti_mf', version='kitti_tracking'))
    class Dataset:
        dataset = SimpleNamespace(get_tracklet_key=lambda index: 'kitti/0017/car',
                                  virtual_rate_meta=[{'scene_id': '0017'}])
        def __len__(self):
            return 1
        def __getitem__(self, index):
            return [{'scene_id': '0017'}] * 3
    class Model:
        def __init__(self):
            self.ct_joint_router = SimpleNamespace(install_policy=lambda policy: None)
            self.config = SimpleNamespace()
            self.calls = 0
        def evaluate_one_sequence(self, sequence):
            self.calls += 1
            self._ct_v27_sequence_endpoints = _rows('0017')
            for row in self._ct_v27_sequence_endpoints:
                row['max_action_q'] = .42
    runner = TrackerClosedLoopRunner.__new__(TrackerClosedLoopRunner)
    runner.config = SimpleNamespace(seed=42, category_name='Car', ct_enable_v30=True)
    runner.device = torch.device('cpu')
    runner.datasets, runner.cache = {'calibration': Dataset()}, {}
    runner.scene_manifest, runner.cache_directory, runner.model = manifest, None, Model()
    rows = runner('calibration', {'kind': 'never'})
    assert len(rows) == 3 and all(row['scene_id'] == '0017' for row in rows)
    assert all(row['max_action_q'] == .42 for row in rows)
    assert all(row['dataset_manifest_sha256'] == manifest['content_sha256'] for row in rows)
    assert not any(row['parameter_training_overlap'] for row in rows)
    runner('calibration', {'kind': 'always'})
    runner('calibration', {'kind': 'always'})
    assert runner.model.calls == 2


def test_calibrate_cli_dispatches_to_v30_and_export_binds_new_score(monkeypatch, tmp_path):
    import tools.ct_action_v27_runtime as runtime
    import utils.action_calibration_v30 as v30
    manifest = build_dataset_manifest(dict(dataset='kitti_mf', version='kitti_tracking'))
    class Runner:
        config = {'ct_enable_v30': True, 'ct_enable_v29': True}
        checkpoint_sha256, config_sha256, scene_manifest = 'checkpoint', 'config', manifest
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, role, policy):
            return _rows(manifest['scenes'][role][0])
    calls = []
    def calibrate(rows, runner, **kwargs):
        calls.append(copy.deepcopy(kwargs))
        return dict(action_policy={'kind': 'never'}, dev_locked_metrics={}, parameter_training_overlap=False)
    monkeypatch.setattr(runtime, 'TrackerClosedLoopRunner', Runner)
    monkeypatch.setattr(v30, 'calibrate_actions_v30', calibrate)
    output = tmp_path / 'policy.json'
    args = ['--v30', '--config', 'saved.yaml', '--checkpoint', 'epoch=058.ckpt', '--output', str(output)]
    runtime.calibrate_main(args)
    assert calls[0]['scene_manifest'] == manifest and 'enable_v29' not in calls[0]
    assert json.loads(output.read_text())['action_policy'] == {'kind': 'never'}
    exported = tmp_path / 'rows.jsonl'
    runtime.export_main(args[:-1] + [str(exported), '--partition', 'calibration'])
    identity = json.loads((tmp_path / 'rows.jsonl.manifest.json').read_text())
    assert identity['schema'] == 'ct_seqtrack.action_rows.v30'
    assert identity['score_definition'] == v30.SCORE_DEFINITION
