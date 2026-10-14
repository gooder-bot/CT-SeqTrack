"""B1 配方检查入口：真实短事务、literal batch128、课程及失败报告。"""
import json
from types import SimpleNamespace

import pytest
import torch

from models.ct_v37.config import normalize_config
from models.ct_v37.data import build_loaders
from tests.test_ct_v31_runtime import TinySource
from tools import check_b1_training_recipe as check
from utils.b1_training_recipes import RECIPE_KEY, recipe_options
from utils.execution_lab import active_options


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def small_config(recipe='b1_concat', batch_size=2):
    return normalize_config(dict(v37_arm='b1', ct_engineering_check=True, workers=0,
        point_sample_size=8, batch_size=batch_size, v32_reserve_windows=0,
        **{RECIPE_KEY: recipe}))


def test_cli_preserves_recipe_batch_and_uses_two_mature_course_steps():
    args = check.parse_args(['--cfg', 'b1.yaml'])
    assert args.device == 'cuda' and args.course_epoch == 10 and args.steps == 2
    assert not hasattr(args, 'batch_size')
    with pytest.raises(SystemExit):
        check.parse_args(['--cfg', 'b1.yaml', '--steps', '0'])


@pytest.mark.parametrize('recipe', ['b1_concat', 'b1_factorized'])
def test_actual_b1_two_step_host_lifecycle_and_scope(recipe):
    cfg = small_config(recipe)
    loaders = build_loaders(cfg, roles=('train',), sources={'train': TinySource((10, 10))})
    report = {}
    check.run_steps(cfg, 'cpu', report, loaders=loaders)
    assert report['status'] == 'passed'
    assert report['completed_steps'] == 2 and report['completed_rows'] == 4
    assert report['sampler']['epoch'] == 10 and report['course_is_not_trained_weight_age']
    assert report['options'] == list(recipe_options(cfg))
    assert report['requested_batch_size'] == 2
    assert report['cuda_memory'] is None and report['injected_test_loader']
    assert not report['checkpoint_saved'] and report['engineering_weights_discarded']
    for i, packet in enumerate(report['steps'], 1):
        assert packet['actual_batch_size'] == 2 and packet['points_shape'] == [2, 4, 8, 5]
        assert packet['adam_state_steps'] == [i]
        assert not packet['commit']['pending_builder'] and not packet['commit']['pending_host']
        assert {'read_batch', 'prepare_forward', 'loss', 'backward', 'adam', 'commit'} <= set(packet['timing_seconds'])
        assert all(value >= 0 for value in packet['timing_seconds'].values())
    assert active_options() == ()


def test_literal_128_rows_are_consumed_without_accumulation_or_full_fields(monkeypatch):
    import models.ctseqtrackv37 as host_module
    calls = []
    cfg = small_config('b1_factorized_bs128', batch_size=128)
    class Host(torch.nn.Module):
        def __init__(self, config, *, loaders):
            super().__init__()
            assert active_options() == recipe_options(config)
            self.parameter = torch.nn.Parameter(torch.ones(()))
            self.training_diagnostics = None
            self._pending_train = None
            self.train_builder = SimpleNamespace(_pending=None, states={}, reset=lambda: None)
            self.tracker = SimpleNamespace(compute_losses=lambda batch, output:
                dict(loss_total=self.parameter.square()))
        def on_train_epoch_start(self):
            assert self.current_epoch == 10
            self._epoch_steps = self._epoch_rows = 0
        def configure_optimizers(self):
            return dict(optimizer=torch.optim.Adam(self.parameters()))
        def _forward_raw(self, rows, builder, *, training):
            assert active_options() == recipe_options(cfg)
            calls.append(len(rows))
            builder._pending = rows
            points = torch.zeros(len(rows), 4, 8, 5)
            return dict(points=points, point_valid=torch.ones(points.shape[:-1], dtype=torch.bool)), \
                SimpleNamespace(accepted_box=self.parameter.expand(len(rows), 4))
        def on_train_batch_end(self, _, rows, index):
            assert active_options() == recipe_options(cfg)
            self._epoch_steps += 1
            self._epoch_rows += len(rows)
            self._pending_train = self.train_builder._pending = None
        def clear_pending_materialization(self):
            pass
    monkeypatch.setattr(host_module, 'CTSEQTRACKV37', Host)
    loaders = build_loaders(cfg, roles=('train',), sources={'train': TinySource((3,) * 128)})
    report = {}
    check.run_steps(cfg, 'cpu', report, loaders=loaders)
    assert calls == [128, 128]
    assert report['completed_rows'] == 256 and report['gradient_accumulation'] == 1
    assert all(step['points_shape'] == [128, 4, 8, 5] for step in report['steps'])


def test_reduced_batch_does_not_silently_pass():
    cfg = small_config('b1_concat', batch_size=128)
    loaders = build_loaders(cfg, roles=('train',), sources={'train': TinySource((2,))})
    report = {}
    with pytest.raises(ValueError, match='reduced-batch'):
        check.run_steps(cfg, 'cpu', report, loaders=loaders)
    assert report['steps'][0]['actual_batch_size'] == 4
    assert active_options() == ()


def test_main_preserves_original_failure_and_only_overrides_engineering_controls(tmp_path, monkeypatch):
    from models.ct_v37 import config as config_module, entry
    from tools import check_v35_batch
    destination = tmp_path / 'new-report' / 'report.json'
    monkeypatch.setattr(check_v35_batch, 'report_destination', lambda output, version: destination)
    seen = {}
    def load(path, overrides):
        seen.update(overrides)
        assert 'batch_size' not in overrides and 'point_sample_size' not in overrides
        return small_config('b1_factorized_bs128', batch_size=128)
    monkeypatch.setattr(config_module, 'load_config', load)
    monkeypatch.setattr(entry, 'configure_numerics', lambda: None)
    monkeypatch.setattr(entry, 'source_identity', lambda: dict(sha256='test'))
    def fail(config, device, report, **kwargs):
        report['stage'] = 'backward'
        raise RuntimeError('original CUDA allocation failure')
    monkeypatch.setattr(check, 'run_steps', fail)
    assert check.main(['--cfg', 'b1.yaml', '--device', 'cpu']) == 1
    report = json.loads(destination.read_text(encoding='utf-8'))
    assert seen['ct_engineering_check'] and seen['workers'] == 0
    assert report['error'] == 'original CUDA allocation failure' and report['stage'] == 'backward'
    assert not report['checkpoint_saved'] and report['engineering_weights_discarded']
    with pytest.raises(FileExistsError, match='new report directory'):
        check.main(['--cfg', 'b1.yaml', '--device', 'cpu'])
