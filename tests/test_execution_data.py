"""原帧缓存不跨事务；优化前后输入、采样及状态提交逐值一致。"""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from models.ct_v31 import data as v35_data
from models.ct_v36 import data as v36_data
from tests.test_ct_v31_runtime import TinySource, fake_output, fake_prior, request as request35
from tests.test_ct_v35_data_diagnostics import v35_config
from tests.test_ct_v36_data import DenseSource, cfg as config36, output_for, request as request36
from utils.raw_frame_memo import RawFrameMemo


def identical(left, right):
    if torch.is_tensor(left):
        assert left.dtype == right.dtype and torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert left.dtype == right.dtype
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert list(left) == list(right)
        for key in left:
            identical(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            identical(a, b)
    elif hasattr(left, '__dict__'):
        identical(vars(left), vars(right))
    else:
        assert left == right


def historical_builder(version):
    family = 'ct_v31' if version == 35 else 'ct_v36'
    path = (Path(__file__).resolve().parents[1] / 'artifacts' / 'ct_checks'
            / '20261009-142559_execution_optimization' / 'baseline_source' / 'models' / family / 'data.py')
    if not path.exists():
        pytest.skip('historical execution audit snapshot is not installed in this checkout')
    name = 'models.' + family + '._execution_data_baseline'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.BatchBuilder


@pytest.mark.parametrize('version', [35, 36])
@pytest.mark.parametrize('source_type', [TinySource, DenseSource])
@pytest.mark.parametrize('historical', [False, True])
def test_raw_transactions_preserve_sampling_labels_commit_and_rng(version, source_type, historical):
    module, config, request = ((v35_data, v35_config, request35) if version == 35
                                else (v36_data, config36, request36))
    raw = module.RawEndpointDataset(source_type((4,)))
    builders = [module.BatchBuilder(config(execution_profile=profile))
                for profile in ('legacy', 'optimized_fp32')]
    if historical:
        builders.insert(0, historical_builder(version)(config(execution_profile='legacy')))
    for frame in (1, 2, 3):
        rows = [raw[request(frame, branch=branch, end=4)] for branch in (1, 3)]
        # 同一原帧可服务独立分支，但绝不共享 crop、随机采样或 accepted 状态。
        rows[1]['frames'] = rows[0]['frames']
        rows[1]['first_frame'] = rows[0]['first_frame']
        batches, records = [], []
        before = torch.get_rng_state().clone()
        for builder in builders:
            batch = builder.prepare(rows)
            output = output_for(batch) if version == 36 else None
            prior = output.prior if version == 36 else fake_prior(batch)
            builder.acquire(batch, prior, training=frame != 3)
            batches.append(deepcopy(batch))
            records.append(builder.commit(output if version == 36 else fake_output(batch),
                                           batch, diagnostics=True))
            assert getattr(builder, '_raw_memo', None) is None
        for index in range(1, len(builders)):
            identical(batches[0], batches[index])
            identical(records[0], records[index])
            identical(builders[0].states, builders[index].states)
        assert torch.equal(before, torch.get_rng_state())
    assert not builders[1].states


@pytest.mark.parametrize('version', [35, 36])
def test_raw_materialization_reused_only_within_pending_transaction(version, monkeypatch):
    module, config, request = ((v35_data, v35_config, request35) if version == 35
                                else (v36_data, config36, request36))
    rows = [module.RawEndpointDataset(TinySource((3,)))[request(1, end=3)]]
    original, calls = module.point_arrays, []
    def counted(frame):
        calls.append(id(frame))
        return original(frame)
    monkeypatch.setattr(module, 'point_arrays', counted)
    builder = module.BatchBuilder(config(execution_profile='optimized_fp32'))
    batch = builder.prepare(rows)
    prior = output_for(batch).prior if version == 36 else fake_prior(batch)
    builder.acquire(batch, prior)
    assert len(calls) == len(set(calls))
    assert builder._raw_memo is not None
    builder.reset()
    calls.clear()
    builder.prepare(rows)
    assert calls and len(calls) == len(set(calls))
    builder.clear_materialization_cache()
    assert builder._raw_memo is None


@pytest.mark.parametrize('version', [35, 36])
@pytest.mark.parametrize('stage', ['prepare', 'acquire', 'commit'])
def test_transaction_exceptions_release_raw_cache(version, stage, monkeypatch):
    module, config, request = ((v35_data, v35_config, request35) if version == 35
                                else (v36_data, config36, request36))
    row = module.RawEndpointDataset(TinySource((3,)))[request(1, end=3)]
    builder = module.BatchBuilder(config(execution_profile='optimized_fp32'))
    if stage == 'prepare':
        def fail(*args):
            raise RuntimeError('synthetic prepare failure')
        monkeypatch.setattr(builder, '_prepare_row', fail)
        action = lambda: builder.prepare([row])
    else:
        batch = builder.prepare([row])
        assert builder._raw_memo is not None
        action = (lambda: builder.acquire(batch, None)) if stage == 'acquire' else (lambda: builder.commit(None, batch))
    with pytest.raises((RuntimeError, AttributeError)):
        action()
    assert builder._raw_memo is None


def test_raw_memo_keys_gt_identity_and_returns_independent_world_box():
    memo = RawFrameMemo(v35_data.point_arrays, v35_data.box_array, v35_data.box_size,
                        v35_data.box_corners, v35_data.inside_box)
    frame = TinySource().get_frames(0, [0])[0]
    box, size, centers = memo.geometry(frame)
    assert not memo._target_masks
    mask = memo.target_mask(frame)
    box[:] = 99.
    assert not np.array_equal(box, memo.geometry(frame)[0])
    original_mask = mask.copy()
    frame['3d_bbox'] = frame['3d_bbox'] + np.asarray([100., 0., 0., 0.])
    assert original_mask.any() and not memo.target_mask(frame).any()
    assert memo.points(frame)[1].dtype == np.int64
    assert memo.index(frame) == {int(value): i for i, value in enumerate(frame['point_ids'])}


def test_v36_preparing_history_does_not_add_full_cloud_gt_scans():
    raw = v36_data.RawEndpointDataset(DenseSource((6,)))
    row = raw[request36(4, start=4, end=6)]
    builder = v36_data.BatchBuilder(config36(execution_profile='optimized_fp32'))
    batch = builder.prepare([row])
    assert builder._raw_memo._geometry  # 静态 GT 框/角点已经物化。
    assert not builder._raw_memo._target_masks  # 采样标签仍只检查 selected 点。
    builder.acquire(batch, output_for(batch).prior)
    keys = list(builder._raw_memo._target_masks)
    assert len(keys) == 1 and keys[0][0] == id(row['frames'][4])
