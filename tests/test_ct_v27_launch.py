"""用户 mini 启动参数、完整数据隔离及后续校准配置身份。"""
import argparse
import ast
from pathlib import Path
import sys

from models.ct_variant import configure_ct_variant
from utils.config import load_yaml_config
from utils.action_calibration_v27 import action_calibration_config_identity
from utils.online_contract import validate_scratch_training_contract
from utils import run_provenance


ROOT = Path(__file__).resolve().parents[1]
ARMS = ('b0', 'b1_cfc', 'b1_gru', 'full_minus_b3', 'full')


class EasyDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as error:
            raise AttributeError(key) from error
    __setattr__ = dict.__setitem__


def test_latest_mini_defaults_match_requested_cli_and_full_defaults_stay_separate():
    for arm in ARMS:
        mini = load_yaml_config(ROOT / f'cfgs/ct_seqtrack/27_{arm}.yaml')
        assert (mini['workers'], mini['check_val_every_n_epoch']) == (4, 5)
        assert (mini['batch_size'], mini['epoch'], mini['seed']) == (16, 60, 42)
        assert mini['trainer_devices'] == 1  # CUDA_VISIBLE_DEVICES selects physical card 2/3.
        full = load_yaml_config(ROOT / f'cfgs/ct_seqtrack/27_{arm}_nuscenes_full.yaml')
        assert (full['workers'], full['check_val_every_n_epoch']) == (12, 1)
        assert (full['version'], full['train_split'], full['test_split']) == (
            'v1.0-trainval', 'train_track', 'val')


def test_main_launch_config_roundtrips_to_strict_calibration_yaml(tmp_path, monkeypatch):
    source = ROOT / 'main.py'
    tree = ast.parse(source.read_text(encoding='utf-8-sig'))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in ('parse_config', 'parse_limit_train_batches')]
    namespace = dict(argparse=argparse, load_yaml=load_yaml_config, EasyDict=EasyDict)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec'), namespace)
    monkeypatch.setattr(run_provenance, 'git_state', lambda root: {'commit': 'test'})
    for arm in ARMS:
        config_path = ROOT / f'cfgs/ct_seqtrack/27_{arm}.yaml'
        monkeypatch.setattr(sys, 'argv', ['main.py', '--cfg', str(config_path),
            '--batch_size', '16', '--epoch', '60', '--workers', '4', '--seed', '42',
            '--preloading', '--check_val_every_n_epoch', '5', '--tag', 'launch_check'])
        config = namespace['parse_config']()
        configure_ct_variant(config)
        validate_scratch_training_contract(config)
        assert config.preloading is True
        configured_yaml = load_yaml_config(config_path)
        configure_ct_variant(configured_yaml)
        assert action_calibration_config_identity(config) == action_calibration_config_identity(configured_yaml)
        destination = tmp_path / arm
        run_provenance.write_run_provenance(destination, config, {}, 'train', ROOT)
        reloaded = load_yaml_config(destination / 'resolved_config.yaml')
        assert reloaded['workers'] == 4 and reloaded['check_val_every_n_epoch'] == 5
        assert action_calibration_config_identity(reloaded) == action_calibration_config_identity(config)
