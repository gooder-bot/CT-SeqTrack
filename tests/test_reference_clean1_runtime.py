"""独立 reference 提交必须全批通过预检；坏行不得污染前序状态。"""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.seqtrack_reference.data import BatchBuilder, build_loaders
from tests.test_seqtrack_reference import TinyReferenceSource


@pytest.mark.parametrize('bad', ['nan_second', 'inf_second', 'too_few_rows', 'too_many_rows', 'wrong_width'])
def test_reference_commit_preflight_is_atomic_and_retryable(bad):
    source = TinyReferenceSource((3, 3))
    loader = build_loaders(dict(workers=0, batch_size=2), roles=('test',),
        sources={'test': source})['test']
    rows = next(iter(loader))
    builder = BatchBuilder()
    batch = builder.prepare(rows)
    before = deepcopy(builder.states)
    prediction = batch['target_box'].clone()
    if bad == 'nan_second':
        prediction[1, 0] = float('nan')
    elif bad == 'inf_second':
        prediction[1, 3] = float('inf')
    elif bad == 'too_few_rows':
        prediction = prediction[:1]
    elif bad == 'too_many_rows':
        prediction = torch.cat((prediction, prediction[:1]), 0)
    else:
        prediction = prediction[:, :3]
    error = FloatingPointError if bad in ('nan_second', 'inf_second') else ValueError
    with pytest.raises(error):
        builder.commit(SimpleNamespace(accepted_box=prediction), batch)
    assert builder._pending is rows
    assert not builder.exposure_records
    assert builder.states.keys() == before.keys()
    for key, history in before.items():
        assert builder.states[key].keys() == history.keys()
        for frame, box in history.items():
            np.testing.assert_array_equal(builder.states[key][frame].center, box.center)
            np.testing.assert_array_equal(builder.states[key][frame].orientation.elements, box.orientation.elements)
    # 原事务仍可由一个完整合法输出提交；没有重复初始化或跳帧。
    records = builder.commit(SimpleNamespace(accepted_box=batch['target_box'].clone()), batch, diagnostics=True)
    assert records == [{}, {}] and builder._pending is None
    assert all(sorted(history) == [0, 1] for history in builder.states.values())
