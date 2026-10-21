"""可覆盖参数先验证真实合法域，防止 loader/Lightning 静默改值。"""
from pathlib import Path

import pytest
import yaml

from models.ct_v37.config import DEFAULTS, config_identity, load_config, normalize_config


PATH = Path('cfgs/ct_seqtrack/37_full_mass_detach_mini.yaml')


@pytest.mark.parametrize('key,value', [
    ('workers', True), ('workers', 1.5), ('workers', -1),
    ('seed', True), ('seed', -1), ('seed', 2**32), ('seed', 42.0),
    ('batch_size', False), ('batch_size', 0), ('batch_size', 4.5),
    ('epoch', 0), ('epoch', 60.0), ('check_val_every_n_epoch', 0),
    ('check_val_every_n_epoch', 1.5), ('check_val_every_n_epoch', True),
    ('lr', True), ('lr', 0), ('lr', float('nan')), ('lr', float('inf')),
    ('devices', True), ('devices', -1), ('devices', '0,1'), ('devices', 1.0),
    ('trainer_devices', True), ('trainer_devices', 2),
])
def test_invalid_parameters_fail_before_building_loaders_or_trainer(key, value):
    with pytest.raises(ValueError, match=key):
        normalize_config(dict(load_config(PATH), **{key: value}))


@pytest.mark.parametrize('changes', [
    dict(workers=0, seed=0), dict(batch_size=1, epoch=1),
    dict(epoch=2, check_val_every_n_epoch=10), dict(seed=2**32 - 1),
    dict(lr=1e-6), dict(devices=None), dict(devices=0),
])
def test_valid_edge_values_are_supported_without_changing_other_defaults(changes):
    before = load_config(PATH)
    cfg = normalize_config(dict(before, **changes))
    for key in cfg:
        assert cfg[key] == changes.get(key, before.get(key))


def test_inheritance_then_cli_then_code_default_is_applied_only_when_missing(tmp_path):
    base = tmp_path / 'base.yaml'
    # 未提供 workers 时，v37 DEFAULTS 为它兜底；其它字段来自当前正式配置。
    values = dict(load_config(PATH))
    values.pop('cfg')
    values.pop('workers')
    values.update(check_val_every_n_epoch=5, path='base-data', devices=0)
    base.write_text(yaml.safe_dump(values), encoding='utf-8')
    child = tmp_path / 'child.yaml'
    child.write_text('_base_: base.yaml\ncheck_val_every_n_epoch: 3\npath: child-data\n', encoding='utf-8')
    cfg = load_config(child)
    assert cfg.workers == DEFAULTS['workers'] == 4
    assert cfg.check_val_every_n_epoch == 3 and cfg.path == 'child-data' and cfg.devices == 0
    override = load_config(child, dict(workers=0, check_val_every_n_epoch=1, path='cli-data', devices=1))
    assert (override.workers, override.check_val_every_n_epoch, override.path, override.devices) == (0, 1, 'cli-data', 1)


def test_gpu_location_and_paths_do_not_change_algorithm_identity():
    cfg = load_config(PATH)
    assert 'devices' not in cfg and 'devices' not in DEFAULTS
    assert config_identity(cfg) == config_identity(dict(cfg, devices=1, path='new-data', log_dir='output/new-run'))
    assert config_identity(cfg) != config_identity(dict(cfg, workers=0))
