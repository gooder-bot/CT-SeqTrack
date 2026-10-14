"""局部属性先 gather：保持邻域、无效 NaN 屏蔽和参数梯度。"""
from copy import deepcopy

import pytest
import torch

from models.ct_v31.evidence import RawLocalGeometry
from tools.check_execution_performance import assert_identical
from utils.execution_lab import execution_scope


@pytest.mark.parametrize('count,neighbors', [(1, 16), (7, 16), (19, 4)])
@pytest.mark.parametrize('empty', [False, True])
def test_gather_before_attribute_delta_is_bitwise_equal(count, neighbors, empty):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(1411)
        old = RawLocalGeometry(neighbors=neighbors, query_chunk_size=6).train()
        new = deepcopy(old)
        points = (torch.randn(2, count, 5) * .1).requires_grad_()
        valid = torch.ones(2, count, dtype=torch.bool)
        ids = torch.arange(count).expand(2, -1).clone()
        if count > 1:
            ids[:, 1] = ids[:, 0]  # 相同 raw ID 不允许互为邻居。
            valid[1, -1] = False
            with torch.no_grad():
                points[1, -1] = torch.nan
        if empty:
            valid.zero_()
        size = torch.tensor([[4., 2., 2.], [.8, .6, 1.]])
        expected = old(points, valid, ids, size)
        with execution_scope(('geometry_gather',), engineering=True):
            actual = new(points, valid, ids, size)
        assert_identical(expected, actual)
        assert torch.isfinite(actual).all()
        expected.square().sum().backward()
        actual.square().sum().backward()
        assert points.grad is None
        assert_identical([p.grad for p in old.parameters()], [p.grad for p in new.parameters()])
    finally:
        torch.set_num_threads(previous)
