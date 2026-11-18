"""B0 连续同掩码分块及 q0 共享邻居的前后向合同。"""
from copy import deepcopy

import pytest
import torch

from models.ct_v41 import observation
from models.ct_v41.local_observation import _gather_corner_features
from tests.helpers.synthetic_batch import make_batch


@pytest.fixture(autouse=True)
def deterministic_cpu():
    threads = torch.get_num_threads()
    mode = torch.get_deterministic_debug_mode()
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(4141)
            yield
    finally:
        torch.set_deterministic_debug_mode(mode)
        torch.set_num_threads(threads)


@pytest.mark.parametrize('training', [True, False])
def test_adjacent_blocks_match_reapplying_every_mask(monkeypatch, training):
    data = make_batch(3, 11)
    data['point_valid'][0] = False
    data['point_valid'][1, :, 1::2] = False
    data['points'][~data['point_valid']] = float('nan')
    actual = observation.B0Observation(token_count=4).train(training)
    actual.execution_profile = 'optimized_fp32'
    reference = deepcopy(actual)

    def evaluate(net):
        points = data['points'].clone().requires_grad_()
        output = net(dict(data, points=points))
        terms = (output.coarse_box, output.source_tokens, output.segmentation_logits,
                 output.bc_prediction, output.point_features)
        sum(term.square().sum() for term in terms).backward()
        return [term.detach() for term in terms] + [points.grad] + [
            p.grad for p in net.parameters()] + list(net.buffers())

    result = evaluate(actual)
    prior_sequence = observation.masked_sequence
    skipped = []
    def always_mask(module, value, valid, **options):
        if options.pop('input_is_masked', False):
            skipped.append(tuple(value.shape))
            assert not value.masked_select(~valid[:, None]).ne(0).any()
        return prior_sequence(module, value, valid, **options)
    monkeypatch.setattr(observation, 'masked_sequence', always_mask)
    expected = evaluate(reference)
    assert len(skipped) == 18
    for tensor, target in zip(result, expected):
        assert torch.equal(tensor, target)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_q0_gather_keeps_repeated_corner_gradients_and_original_source_shape(device):
    # 所有角共享部分邻居；每角内部和原距离排序一样不重复。输入梯度使用非整数 FP32。
    b, n, c = 2, 31, 64
    features = torch.randn(b, n, c).to(device).requires_grad_()
    selected = torch.randn(b, 1, 8, n).argsort(dim=-1)[..., :16].to(device)
    weights = torch.randn(b, 1, 8, 16, c).to(device)
    def gather_with_shapes(read):
        saved_shapes = []
        def pack(t):
            saved_shapes.append(tuple(t.shape))
            return t
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            y = read()
        grad, = torch.autograd.grad(y, features, weights)
        return y.detach(), grad, saved_shapes

    old_y, old_grad, old_shapes = gather_with_shapes(lambda: features[:, None, None]
        .expand(-1, 1, 8, -1, -1).gather(3, selected[..., None].expand(-1, -1, -1, -1, c)))
    def read():
        return _gather_corner_features(features, selected)
    y, grad, shapes = gather_with_shapes(read)
    assert torch.equal(y, old_y)
    if device == 'cpu':
        assert torch.equal(grad, old_grad)
    else:
        torch.testing.assert_close(grad, old_grad, rtol=1e-5, atol=1e-6)
    assert torch.equal(gather_with_shapes(read)[1], grad)
    assert (b, 1, 8, n, c) in old_shapes and (b, 1, 8, n, c) not in shapes
    assert (b, n, c) in shapes
