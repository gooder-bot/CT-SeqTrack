"""B0 连续同掩码分块及 q0 共享邻居的前后向合同。"""
from copy import deepcopy

import pytest
import torch

from models.ct_v41 import observation
from models.ct_v41.history_context import history_descriptor
from models.ct_v41.local_observation import _gather_corner_features
from models.ct_v41.packed_points import PointSlotLayout
from models.ct_v41.prior import PhysicalTimePrior
from tests.helpers.synthetic_batch import make_batch
from utils.execution_lab import execution_scope


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
    # 密集回退路径也需要维持相邻层省略重复 mask 的合同。
    monkeypatch.setattr(observation, '_use_packed_points', lambda *args: False)
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


@pytest.mark.parametrize('kind', ['seg', 'mini', 'feature'])
@pytest.mark.parametrize('training', [False, True])
def test_packed_pointnets_preserve_masks_bn_and_gradients(monkeypatch, kind, training):
    valid = torch.zeros(8, 16, dtype=torch.bool)
    valid[1] = True
    valid[2, 2::3] = True
    valid[3, 5] = True
    valid[4:, ::2] = True
    net = {'seg': observation.SegPointNet, 'mini': observation.MiniPointNet,
           'feature': lambda: observation.FeaturePointNet(6)}[kind]().train(training)
    with torch.no_grad():
        for layer in net.modules():
            if isinstance(layer, torch.nn.BatchNorm1d):
                layer.weight[::2].neg_()
                layer.bias.fill_(.12)
    reference = deepcopy(net)
    data = torch.randn(len(valid), 13 if kind == 'mini' else 14, 16).masked_fill(~valid[:, None], torch.nan)
    foreground = torch.rand(valid.shape, requires_grad=True)
    options = dict(valid_count=int(valid.sum()))
    if kind == 'feature':
        options['token_valid_count'] = int(valid.sum(-1).clamp_max(net.token_count).sum())
    elif kind == 'mini':
        options['row_count'] = int(valid.any(-1).sum())

    def evaluate(model):
        points = data.clone().requires_grad_()
        if kind == 'mini':
            rows = model(points, valid, foreground, **options)
            assert torch.count_nonzero(rows[0]) == 0
            outputs = (rows,)
        elif kind == 'feature':
            tokens, mask = model(points, valid, **options)
            expected = torch.arange(net.token_count) < valid.sum(-1)[:, None]
            assert torch.equal(mask, expected)
            assert torch.count_nonzero(tokens.masked_select(~mask[:, None])) == 0
            outputs = (tokens,)
        else:
            outputs = model(points, valid, **options)
            for output in outputs:
                assert torch.count_nonzero(output.masked_select(~valid[:, None])) == 0
        # 非对称梯度同时覆盖 max 选点、BN 统计及输入/参数求导。
        probes = torch.Generator().manual_seed(23)
        sum((output * torch.randn(output.shape, generator=probes)).mean()
            for output in outputs).backward()
        assert torch.count_nonzero(points.grad.masked_select(~valid[:, None])) == 0
        assert foreground.grad is None
        return [tensor.detach() for tensor in outputs], points.grad

    # BN 实际接收二维有效行，防止误回退到全宽投影却通过数值对比。
    row_shapes = []
    first_blocks = net.per_point if kind == 'mini' else net.seq_per_point
    handles = [first_blocks[0][1].register_forward_pre_hook(
        lambda module, inputs: row_shapes.append(tuple(inputs[0].shape)))]
    if kind != 'mini':
        handles.append(net.seq_per_point2[0][1].register_forward_pre_hook(
            lambda module, inputs: row_shapes.append(tuple(inputs[0].shape))))
    try:
        outputs, grad = evaluate(net)
    finally:
        for handle in handles:
            handle.remove()
    assert row_shapes[0] == (int(valid.sum()), 64)
    if kind != 'mini':
        assert row_shapes[1] == (options.get('token_valid_count', int(valid.sum())), 512)
    monkeypatch.setattr(observation, '_use_packed_points', lambda *args: False)
    reference_outputs, reference_grad = evaluate(reference)
    for result, expected in zip(outputs, reference_outputs):
        torch.testing.assert_close(result, expected, rtol=4e-4, atol=2e-5)
    torch.testing.assert_close(grad, reference_grad, rtol=5e-4, atol=1e-5)
    for param, ref_param in zip(net.parameters(), reference.parameters()):
        torch.testing.assert_close(param.grad, ref_param.grad, rtol=5e-4, atol=1e-5)
    for state, ref_state in zip(net.buffers(), reference.buffers()):
        torch.testing.assert_close(state, ref_state, rtol=1e-5, atol=2e-6)


def test_packed_max_keeps_first_valid_tie_and_skips_empty_row():
    valid = torch.tensor([[False, True, False, True], [False] * 4])
    rows = torch.tensor([[2., -1.], [2., 3.]], requires_grad=True)
    maxima = PointSlotLayout(valid).max_per_group(rows)
    assert torch.equal(maxima, torch.tensor([[2., 3.], [0., 0.]]))
    maxima.sum().backward()
    assert torch.equal(rows.grad, torch.tensor([[1., 0.], [0., 1.]]))


@pytest.mark.parametrize('input_grad,factorized', [(False, False), (True, False), (True, True)])
def test_b0_reuses_only_compatible_packed_rows(monkeypatch, input_grad, factorized):
    data = make_batch(2, 11)
    data['history_valid'][1, 0] = False
    data['point_valid'][0, 2] = False
    data['point_valid'][:, :, 1::3] = False
    data['point_ids'][:, 1, 3] = data['point_ids'][:, 1, 2]
    data['points'][~data['point_valid']] = float('nan')
    net = observation.B0Observation(token_count=4).train()
    net.execution_profile = 'optimized_fp32'
    independent = deepcopy(net)
    layouts, gather_widths = [], []

    class RecordedLayout(PointSlotLayout):
        def __init__(self, valid, **kw):
            super().__init__(valid, **kw)
            layouts.append(self)

        def pack(self, value):
            gather_widths.append(value.shape[1])
            return super().pack(value)

    def run(model):
        points = data['points'].detach().clone().requires_grad_(input_grad)
        with execution_scope(('factorized_projection',) if factorized else (), engineering=True):
            out = model(dict(data, points=points))
            (out.source_tokens.square().sum() + out.coarse_box.square().sum()
             + out.point_features.square().sum() + out.segmentation_logits.square().sum()
             + out.bc_prediction.square().sum()).backward()
        return [v for v in vars(out).values() if v is not None] + list(model.buffers()) + [
            p.grad for p in model.parameters()] + ([points.grad] if input_grad else [])

    monkeypatch.setattr(observation, 'PointSlotLayout', RecordedLayout)
    actual = run(net)
    if factorized:
        assert len(layouts) == 1 and gather_widths == [13]  # 仅 Mini 使用逐点压紧。
    else:
        endpoint, frame, token = layouts
        assert endpoint.flat_indices is frame.flat_indices
        assert token.present is frame.present
        assert endpoint.groups == 2 and frame.groups == 8 and token.slots == 4
        assert gather_widths.count(14) == (2 if input_grad else 1)
        if not input_grad:
            assert 'group_indices' not in vars(frame) and 'slot_indices' not in vars(frame)
    # 原版三路各自从自身掩码提取索引、行值及 token 计数。
    def independent_rows(module, args, kwargs):
        return args, {key: val for key, val in kwargs.items()
                      if key not in ('layout', 'packed_input', 'frame_counts')}
    for branch in (independent.seg_pointnet, independent.mini_pointnet, independent.feature_pointnet):
        branch.register_forward_pre_hook(independent_rows, with_kwargs=True)
    expected = run(independent)
    assert len(actual) == len(expected)
    for value, target in zip(actual, expected):
        assert torch.equal(value, target)


@pytest.mark.parametrize('count', [0, 1])
def test_empty_and_singleton_batch_use_stored_bn_statistics(monkeypatch, count):
    def unexpected_pack(valid):
        pytest.fail('0/1 有效点应使用 running stats')
    monkeypatch.setattr(observation, 'PointSlotLayout', unexpected_pack)
    data = make_batch(1, 4)
    data['point_valid'].fill_(False)
    data['point_valid'][0, -1, :count] = True
    data['points'][~data['point_valid']] = float('nan')
    net = observation.B0Observation(token_count=2).train()
    net.execution_profile = 'optimized_fp32'
    output = net(data)
    assert output.source_valid.sum() == count
    assert output.sequence_valid.item() == bool(count)
    assert torch.isfinite(output.coarse_box).all()
    assert all(layer.num_batches_tracked == 0 for layer in net.modules()
               if isinstance(layer, torch.nn.BatchNorm1d))


@pytest.mark.parametrize('mask_dtype', [torch.bool, torch.uint8])
def test_history_age_checks_physical_times_and_only_requires_positive_age_for_present_frames(mask_dtype):
    data = make_batch(1)
    valid = torch.tensor([[True, False, False]], dtype=mask_dtype)
    times = data['frame_times']
    times[0] = torch.tensor([-1., 0., 2., 0.])
    def describe():
        return history_descriptor(data, data['history_boxes'], valid, data['box_size'],
                                  torch.zeros(1, 3, 3), .5).reshape(3, 10)

    assert describe()[1:].count_nonzero() == 0
    for invalid in (0., 1.):
        times[0, 0] = invalid
        with pytest.raises(ValueError, match='history ages'):
            describe()
    valid.zero_()
    assert describe().count_nonzero() == 0
    times[0, 1] = float('nan')
    with pytest.raises(ValueError, match='history ages'):
        describe()


def test_b1_uses_motion_frame_sigma_at_low_speed_threshold_and_detaches_context():
    net = PhysicalTimePrior()
    with torch.no_grad():
        net.sigma_head.weight[:, :2].copy_(torch.eye(2))
        net.sigma_head.bias.copy_(torch.tensor([-.8, 1.3]))
    threshold = torch.tensor(.2)
    speeds = torch.stack((torch.nextafter(threshold, torch.tensor(0.)), threshold,
                          torch.nextafter(threshold, torch.tensor(1.))))
    data = make_batch(3)
    data['history_boxes'].zero_()
    data['history_boxes'][:, 2, 1] = speeds
    data['history_times'][:] = torch.tensor([-3., -2., -1.])
    result = net(data)
    raw = (.1 + torch.nn.functional.softplus(net.sigma_head(result.feature.detach()))).log()
    expected = torch.cat((raw[:1].mean(1, keepdim=True).expand(-1, 2), raw[1:]))
    assert torch.equal(result.log_sigma, expected)
    weights = torch.tensor([[.2, .7], [-.7, .3], [.5, .9]])
    params = tuple(net.sigma_head.parameters())
    actual_grad = torch.autograd.grad(result.log_sigma, (*params, result.feature), weights, allow_unused=True)
    expected_grad = torch.autograd.grad(expected, params, weights)
    assert actual_grad[-1] is None
    assert all(torch.equal(a, b) for a, b in zip(actual_grad, expected_grad))


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
