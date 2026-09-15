"""真实稀疏测量的统计、梯度及确定性桶契约。"""
import copy

import pytest
import torch
from torch import nn

from models.ct_v2.observation_reference import seqtrack_segmentation_cross_entropy
from utils.masked_observation import masked_batch_norm, masked_max_pool
from tests.test_ct_v27_full_model import full_model_runtime, _training_batch  # noqa: F401
from tests.test_ct_v27_input_flow import sampler_runtime  # noqa: F401
from tests.test_ct_v29_b0_host import construct, observation_routing


@pytest.mark.parametrize('count', [0, 1, 2, 7, 16])
def test_valid_only_bn_is_invariant_to_invalid_values_and_gradients(count):
    torch.manual_seed(42)
    valid = (torch.arange(16).reshape(2, 8) < count)
    initial = torch.randn(2, 3, 8, dtype=torch.float64)
    first = nn.BatchNorm1d(3).double().train()
    second = copy.deepcopy(first)
    old = {name: value.clone() for name, value in first.named_buffers()}
    x = initial.clone().requires_grad_()
    y = initial.masked_fill(~valid[:, None], 10000.).requires_grad_()
    a = masked_batch_norm(x, valid, first)
    b = masked_batch_norm(y, valid, second)
    assert torch.equal(a, b)
    a.square().sum().backward()
    b.square().sum().backward()
    assert torch.equal(x.grad, y.grad)
    assert torch.count_nonzero(x.grad.masked_select(~valid[:, None])) == 0
    for name, value in first.named_buffers():
        assert torch.equal(value, dict(second.named_buffers())[name])
        if count < 2:
            assert torch.equal(value, old[name])
    if count == 0:
        assert torch.count_nonzero(a) == 0


def test_fixed_buckets_emit_token_mask_and_zero_empty_gradients():
    x = torch.tensor([[[1., 1., 200., 300., 8., 9., 10., 11.]]], requires_grad=True)
    mask = torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0]], dtype=torch.bool)
    pooled, tokens = masked_max_pool(x, mask, 4)
    assert pooled.tolist() == [[[1., 0., 0., 0.]]]
    assert tokens.tolist() == [[True, False, False, False]]
    pooled.sum().backward()
    assert x.grad.tolist() == [[[1., 0., 0., 0., 0., 0., 0., 0.]]]


@pytest.mark.parametrize('empty', [False, True])
def test_masked_segmentation_weighted_denominator_and_empty_gradient(empty):
    torch.manual_seed(42)
    x = torch.randn(2, 2, 16, requires_grad=True)
    labels = torch.randint(0, 2, (2, 16))
    valid = torch.rand(2, 16) > .4
    if empty:
        valid.zero_()
    loss = seqtrack_segmentation_cross_entropy(x, labels, valid)
    if not empty:
        expected = torch.nn.functional.cross_entropy(
            x.movedim(1, -1)[valid], labels[valid], weight=x.new_tensor([.5, 2.]))
        torch.testing.assert_close(loss, expected)
    else:
        assert loss.item() == 0
    loss.backward()
    assert torch.count_nonzero(x.grad.movedim(1, -1)[~valid]) == 0


def test_actual_b0_invalid_slots_do_not_change_output_bn_or_gradient(full_model_runtime):
    first = construct(full_model_runtime).train()
    first.ct_enable_v30 = first.config.ct_enable_v30 = True
    data, _, _ = _training_batch(full_model_runtime, first, batch_size=2)
    data['b0_point_valid_mask'][0, -1] = False
    data['b0_point_valid_mask'][1, 1, 2:] = False
    first_input = copy.deepcopy(data)
    second_input = copy.deepcopy(data)
    flat_mask = data['b0_point_valid_mask'].flatten(1).bool()
    second_input['points'][~flat_mask] = 10000.
    second_input['candidate_bc'][~flat_mask] = -10000.
    second = copy.deepcopy(first)
    snapshots = []
    for model, batch in ((first, first_input), (second, second_input)):
        torch.manual_seed(123)  # 相同 dropout 流，比较的唯一变量为无效槽内容。
        with observation_routing(model):
            output = model(batch)
            loss = model.compute_loss(batch, output)
            loss['loss_total'].backward()
        snapshots.append((output, dict(model.named_buffers()),
                          {name: p.grad for name, p in model.named_parameters()}))
    for key in ('seg_logits', 'pred_bc', 'motion_pred', 'aux_estimation_boxes'):
        assert torch.equal(snapshots[0][0][key], snapshots[1][0][key]), key
    for group in (1, 2):
        for key, value in snapshots[0][group].items():
            other = snapshots[1][group][key]
            assert (value is None and other is None) or torch.equal(value, other), key
