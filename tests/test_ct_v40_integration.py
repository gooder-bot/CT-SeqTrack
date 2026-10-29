"""V40 配方重入和 host 事务释放合同；仅写入，由服务器环境执行。"""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from models.ct_v40.config import config_identity, normalize_config
from models.ct_v40.input import online_batch
from models.ctseqtrackv40 import CTSEQTRACKV40
from tests.helpers.synthetic_batch import batch as synthetic_batch
from utils.batch_materialization import attach_materialization_cache, clear_materialization_cache


def config(arm='full'):
    return normalize_config(dict(v40_arm=arm, ct_engineering_check=True,
        v40_attention_backend='math', point_sample_size=16, batch_size=2,
        workers=0, epoch=2, v40_diagnostics=False))


@pytest.mark.parametrize('arm,enabled', (
    ('b0', (False, False, False)), ('b1', (True, False, False)),
    ('b1_mtp', (True, True, False)), ('full', (True, True, True))))
def test_configuration_round_trip_preserves_arm_and_rejects_flag_overrides(arm, enabled):
    cfg = config(arm)
    assert tuple(cfg[key] for key in ('enable_b1', 'enable_m', 'enable_b3')) == enabled
    assert cfg.v40_training_recipe == arm
    assert cfg.v40_search_policy == ('dual_roi_guard_v1' if arm == 'full' else 'accepted_only')
    assert normalize_config(cfg) == cfg
    assert normalize_config(normalize_config(cfg)) == cfg
    assert config_identity(normalize_config(cfg)) == config_identity(cfg)
    for key in ('enable_b1', 'enable_m', 'enable_b3'):
        changed = dict(cfg, **{key: not cfg[key]})
        with pytest.raises(ValueError, match='arm-derived'):
            normalize_config(changed)


def test_formal_precision_and_backend_cannot_silently_fall_back():
    cfg = normalize_config()
    assert cfg.v40_attention_backend == 'fa2_varlen'
    assert cfg.v40_attention_precision == 'qkv_bf16_only'
    with pytest.raises(ValueError, match='FA2'):
        normalize_config(dict(v40_attention_backend='math'))
    with pytest.raises(ValueError):
        normalize_config(dict(config(), execution_profile='feature_bf16'))


def test_all_entry_trainers_preserve_evaluation_tensor_versions():
    # 源码合同覆盖 fit 内 validation、复用 Trainer 的 late3 与独立 --test。
    # 不启动 Trainer/CUDA；依赖默认值的回归会重新让正式 eval 的缓存失效。
    path = Path(__file__).resolve().parents[1] / 'models/ct_v40/entry.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    constructors = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute) and node.func.attr == 'Trainer'
        and isinstance(node.func.value, ast.Name) and node.func.value.id == 'pl']
    assert constructors
    for constructor in constructors:
        policies = [item.value for item in constructor.keywords if item.arg == 'inference_mode']
        assert len(policies) == 1 and isinstance(policies[0], ast.Constant)
        assert policies[0].value is False


def test_no_grad_evaluation_reuses_transaction_cache_and_invalidates_mutation():
    with torch.no_grad():
        prepared = attach_materialization_cache(synthetic_batch(1, 16), enabled=True)
        cache = prepared['_ct_materialization_cache']
        first, second = online_batch(prepared), online_batch(prepared)
        assert not torch.is_grad_enabled()
        assert not first['points'].requires_grad and first['points'].grad_fn is None
        assert first['points'] is second['points']
        assert first['point_valid'] is second['point_valid']
        assert cache.point is not None
        version = prepared['points']._version
        prepared['points'][0, 0, 0, 0].add_(.5)
        assert prepared['points']._version > version
        changed = online_batch(prepared)
        assert changed['points'] is not first['points']
        assert changed['points'][0, 0, 0, 0] == prepared['points'][0, 0, 0, 0]
        clear_materialization_cache(prepared)
        assert cache.point is None and '_ct_materialization_cache' not in prepared


def test_nonfinite_gradient_aborts_before_adam_and_releases_pending_references():
    host = CTSEQTRACKV40(config('b0'), tracker=LossForbiddenTracker())
    optimizer = torch.optim.Adam(host.parameters())
    prepared = attach_materialization_cache({}, enabled=True)
    cache = prepared['_ct_materialization_cache']
    cache.point = object()
    host._pending_train = (object(), prepared, 1)
    host.tracker.parameter.grad = torch.tensor(float('inf'))
    with pytest.raises(FloatingPointError, match='Adam update and accepted commit aborted'):
        host.on_before_optimizer_step(optimizer)
    assert host._pending_train is None
    assert cache.point is None and '_ct_materialization_cache' not in prepared
    assert not optimizer.state
    assert host.tracker.parameter.item() == 1.


class LossForbiddenTracker(nn.Module):
    def __init__(self):
        super().__init__()
        self.parameter = nn.Parameter(torch.tensor(1.))

    def compute_losses(self, *args, **kwargs):
        raise AssertionError('host evaluation cannot request training losses')


@pytest.mark.parametrize('failure', (None, 'commit', 'record'))
def test_host_evaluation_releases_transaction_cache_on_success_and_failure(failure, monkeypatch):
    host = CTSEQTRACKV40(config(), tracker=LossForbiddenTracker()).eval()
    prepared = attach_materialization_cache({}, enabled=True)
    cache = prepared['_ct_materialization_cache']
    cache.point = object()
    events = []
    output = object()

    def forward(rows, builder, *, training):
        assert not training
        events.append('forward')
        return prepared, output

    def commit(result, batch, *, diagnostics):
        assert result is output and batch is prepared and diagnostics
        events.append('commit')
        if failure == 'commit':
            raise RuntimeError('commit failure')
        return []

    def record(rows, batch, result, committed):
        assert batch is prepared and result is output
        events.append('record')
        if failure == 'record':
            raise RuntimeError('record failure')

    host.evaluation_builder = SimpleNamespace(commit=commit,
        clear_materialization_cache=lambda: events.append('release'))
    host.evaluation = SimpleNamespace(rows=[], add_batch=record)
    monkeypatch.setattr(host, '_forward_raw', forward)
    if failure is None:
        host._evaluation_step([])
        assert events == ['forward', 'commit', 'record', 'release']
    else:
        with pytest.raises(RuntimeError, match=failure + ' failure'):
            host._evaluation_step([])
        assert events[-1] == 'release'
    assert '_ct_materialization_cache' not in prepared
    assert cache.point is None
