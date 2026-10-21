"""新 B2 配方的直接调用、真实提交及宿主接线，不继承外层工程算法。"""
from copy import deepcopy

import pytest
import torch

from models.ct_v37.config import normalize_config
from models.ct_v37.data import build_loaders
from models.ct_v37.model import JointTracker
from models.ctseqtrackv37 import CTSEQTRACKV37
from tests.test_ct_v31_runtime import TinySource
from tests.test_ct_v34_resume import _assert_identical
from tests.test_ct_v34_resume import _epoch
from models.ct_v31.runtime import capture_rng_state
from tests.test_ct_v37_network import batch, config, threads
from utils.execution_lab import OPTIONS, active_options, execution_scope


def centroid_config(**changes):
    values = dict(config(), v37_centroid_recipe='full_mass_detach')
    values.update(changes)
    return normalize_config(values)


def test_direct_forward_and_losses_clear_foreign_options_and_restore_them(monkeypatch):
    model = JointTracker(centroid_config()).eval()
    values = batch(1)
    expected = model(values)
    expected_losses = model.compute_losses(values, expected)
    calls = []
    for method in ('_forward', '_compute_losses'):
        original = getattr(model, method)
        def checked(*args, _original=original, _method=method, **kwargs):
            assert active_options() == ()
            calls.append(_method)
            return _original(*args, **kwargs)
        monkeypatch.setattr(model, method, checked)
    with execution_scope(OPTIONS, engineering=True):
        actual = model(values)
        losses = model.compute_losses(values, actual)
        assert active_options() == OPTIONS
    assert active_options() == () and calls == ['_forward', '_compute_losses']
    for field in ('accepted_box', 'selected_index', 'selected_quality'):
        assert torch.equal(getattr(expected, field), getattr(actual, field))
    assert expected_losses.keys() == losses.keys()
    for name in losses:
        assert torch.equal(expected_losses[name], losses[name]), name


@pytest.mark.parametrize('arm', ['b1_b2', 'full'])
def test_same_forward_commits_identical_state_with_new_policy_and_scoped_collectors(arm, monkeypatch):
    cfg = centroid_config(v37_arm=arm, v31_arm=arm,
        v37_centroid_recipe=arm + '_mass_detach', point_sample_size=8,
        batch_size=2, v32_reserve_windows=0)
    hosts = []
    for stopped in (False, True):
        torch.manual_seed(73)
        loaders = build_loaders(cfg, roles=('test',), sources={'test': TinySource((3, 2))})
        host = CTSEQTRACKV37(cfg, loaders=loaders).eval()
        host.tracker.evidence.detach_mass_weights = stopped
        assert host.training_diagnostics.centroid_semantics is True
        hosts.append((host, loaders))
    snapshots = []
    for host, loaders in hosts:
        rows = next(iter(loaders['test']))
        with torch.no_grad():
            prepared, output = host._forward_raw(rows, host.evaluation_builder, training=False)
            original_commit = host.evaluation_builder._commit
            original_add = host.evaluation._add_batch
            def commit(*args, **kwargs):
                assert active_options() == ()
                return original_commit(*args, **kwargs)
            def add(*args, **kwargs):
                assert active_options() == ()
                return original_add(*args, **kwargs)
            monkeypatch.setattr(host.evaluation_builder, '_commit', commit)
            monkeypatch.setattr(host.evaluation, '_add_batch', add)
            rng = torch.get_rng_state().clone()
            with execution_scope(OPTIONS, engineering=True):
                commits = host.evaluation_builder.commit(output, prepared, diagnostics=True)
                host.evaluation.add_batch(rows, prepared, output, commits)
                assert active_options() == OPTIONS
            assert torch.equal(rng, torch.get_rng_state())
        states = {}
        for key, state in host.evaluation_builder.states.items():
            value = {k: v for k, v in vars(state).items() if k != 'memory'}
            states[key] = dict(state=deepcopy(value), memory=deepcopy(vars(state.memory)))
        snapshots.append((commits, states, host.evaluation.rows, host.evaluation.summary()))
    _assert_identical(snapshots[0], snapshots[1])


def test_legacy_host_does_not_enable_new_diagnostic_fields():
    host = CTSEQTRACKV37(config())
    assert host.training_diagnostics.centroid_semantics is False
    assert host.tracker.evidence.detach_mass_weights is False


class CPUCentroidHost(CTSEQTRACKV37):
    """手动驱动原宿主事务；不冒充真实 Lightning Trainer 检查。"""
    _contract_epoch = 0

    @property
    def current_epoch(self):
        return self._contract_epoch

    def log(self, *args, **kwargs):
        pass


@pytest.mark.parametrize('recipe', ['b1_b2_mass_detach', 'full_mass_detach', 'full_mass_detach_hold_lr'])
def test_new_recipe_complete_epoch_resume_preserves_actual_adam_dropout_and_commit(recipe):
    arm = 'b1_b2' if recipe.startswith('b1_b2') else 'full'
    cfg = centroid_config(v37_arm=arm, v31_arm=arm, v37_centroid_recipe=recipe,
        lr_milestones=[20] if recipe.endswith('hold_lr') else [20, 50],
        point_sample_size=8, batch_size=4, v32_reserve_windows=0,
        # 小 epoch 立即使用多帧窗口，使后续批次消费已提交的 accepted 历史。
        v31_curriculum_epochs=1, log_dir=None)
    def build():
        loaders = build_loaders(cfg, roles=('train',), sources={'train': TinySource((6, 5))})
        host = CPUCentroidHost(cfg, loaders=loaders).train()
        configured = host.configure_optimizers()
        return host, configured['optimizer'], configured['lr_scheduler']['scheduler']
    host, optimizer, scheduler = build()
    assert any(isinstance(layer, torch.nn.Dropout) and layer.training and layer.p > 0
               for layer in host.tracker.modules())
    _epoch(host, optimizer, scheduler, 0)
    checkpoint = dict(state_dict=deepcopy(host.state_dict()),
        optimizer_states=[deepcopy(optimizer.state_dict())],
        lr_schedulers=[deepcopy(scheduler.state_dict())])
    host.on_save_checkpoint(checkpoint)
    assert checkpoint['ct_v37_runtime']['epoch_complete']
    assert checkpoint['ct_v37_runtime']['completed_epoch'] == 1
    uninterrupted = _epoch(host, optimizer, scheduler, 1)
    expected = dict(model=deepcopy(host.state_dict()), optimizer=deepcopy(optimizer.state_dict()),
        scheduler=deepcopy(scheduler.state_dict()), rng=capture_rng_state())
    resumed, resumed_optimizer, resumed_scheduler = build()
    resumed.load_state_dict(checkpoint['state_dict'], strict=True)
    resumed_optimizer.load_state_dict(checkpoint['optimizer_states'][0])
    resumed_scheduler.load_state_dict(checkpoint['lr_schedulers'][0])
    resumed.on_load_checkpoint(checkpoint)
    restored = _epoch(resumed, resumed_optimizer, resumed_scheduler, 1)
    _assert_identical(uninterrupted, restored)
    _assert_identical(expected, dict(model=resumed.state_dict(), optimizer=resumed_optimizer.state_dict(),
        scheduler=resumed_scheduler.state_dict(), rng=capture_rng_state()))
