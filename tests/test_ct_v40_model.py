"""v40 四臂、公共初始化、M 与监督隔离合同；仅服务器验收时执行。"""
import copy

import pytest
import torch

from models.ct_v40.config import normalize_config
from models.ct_v40.entry import initialization_identity
from models.ct_v40.attention import EfficientAttention
from models.ct_v40.input import online_batch, loss_batch
from models.ct_v40.model import JointTracker
from models.ct_v40.multistep import FutureMotionHead, compute_multistep_loss
from tests.helpers.synthetic_batch import batch


@pytest.fixture(autouse=True)
def restore_cpu_state():
    threads, rng = torch.get_num_threads(), torch.get_rng_state()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)
    torch.set_rng_state(rng)


def config(arm):
    return normalize_config(dict(v40_arm=arm, v40_attention_backend='math',
        ct_engineering_check=True, point_sample_size=16, batch_size=2,
        workers=0, epoch=2, execution_profile='optimized_fp32'))


def values(b=2):
    data = batch(b, 16)
    data.update(future_displacement=torch.tensor([[[.8, .1], [1.5, .2]]]).repeat(b, 1, 1),
                future_dt=torch.tensor([[1., 2.]]).repeat(b, 1),
                future_valid=torch.ones(b, 2, dtype=torch.bool))
    return data


def test_four_arms_common_parameters_and_initialization_rng_are_identical():
    models, states = {}, {}
    for arm in ('b0', 'b1', 'b1_mtp', 'full'):
        torch.manual_seed(42)
        models[arm] = JointTracker(config(arm))
        states[arm] = torch.get_rng_state().clone()
    reference = dict(models['full'].named_parameters())
    for arm, model in models.items():
        for name, parameter in model.named_parameters():
            if name.startswith(('observation.', 'decoder.', 'prior.')):
                assert torch.equal(parameter, reference[name]), (arm, name)
        assert torch.equal(states[arm], states['full'])
    assert models['b0'].prior is None
    assert models['b0'].decoder.prior_adapter is None
    assert models['b0'].multistep is None and models['b0'].verifier is None
    assert models['b1'].multistep is None and models['b1'].verifier is None
    assert models['b1_mtp'].verifier is None
    for name, parameter in models['b1_mtp'].multistep.named_parameters():
        assert torch.equal(parameter, dict(models['full'].multistep.named_parameters())[name])
    assert all(p.requires_grad for model in models.values() for p in model.parameters())
    identities = {arm: initialization_identity(model) for arm, model in models.items()}
    assert len({value['common_parameter_hash'] for value in identities.values()}) == 1
    assert identities['b0']['prior_parameter_hash'] is None
    assert identities['b0']['prior_adapter_parameter_hash'] is None
    assert identities['b0']['mtp_parameter_hash'] is None
    for key in ('base_initial_parameter_hash', 'prior_parameter_hash', 'prior_adapter_parameter_hash'):
        assert len({identities[arm][key] for arm in ('b1', 'b1_mtp', 'full')}) == 1
    assert identities['b1_mtp']['mtp_parameter_hash'] == identities['full']['mtp_parameter_hash']
    assert identities['b1_mtp']['mtp_parameter_hash'] is not None


def test_b0_empty_sequence_has_fallback_source_and_no_b1_losses():
    model, data = JointTracker(config('b0')), values(1)
    data['point_valid'].zero_()
    for name in ('physical_displacement', 'physical_valid', 'acquisition_target',
                 'acquisition_valid', 'acquisition_demand',
                 'future_displacement', 'future_dt', 'future_valid'):
        data.pop(name)
    output = model(data)
    assert output.candidates.raw_valid.tolist() == [[False, False, True]]
    assert output.candidates.valid.tolist() == [[False, False, True]]
    assert output.candidates.alias_of[0, 0].item() == 2
    assert output.selected_index.item() == 2
    assert torch.equal(output.accepted_box, data['fallback_box'])
    assert not output.commit_decision.eligible.any()
    losses = model.compute_losses(data, output)
    assert all(losses['loss_' + name].item() == 0. for name in ('physical', 'sigma', 'acquisition'))
    assert 'loss_m' not in losses
    assert torch.isfinite(losses['loss_total'])


@pytest.mark.parametrize('arm', ('b0', 'b1', 'b1_mtp', 'full'))
def test_current_future_labels_are_not_exposed_to_online_modules(arm):
    model, data = JointTracker(config(arm)).eval(), values(1)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith('weight') and parameter.ndim > 1 and bool(parameter.eq(0).all()):
                parameter.normal_(0., .01)
    first = model(data)
    changed = copy.deepcopy(data)
    for name in ('target_box', 'target_box_size', 'history_target_boxes', 'segmentation_labels',
                 'bc_targets', 'physical_displacement', 'acquisition_target',
                 'future_displacement', 'future_dt'):
        changed[name].fill_(float('nan') if changed[name].is_floating_point() else 0)
    # 连未来标签形状也不得进入在线验证或分支判断。
    changed['future_valid'] = torch.zeros(5, 7, dtype=torch.bool)
    second = model(changed)
    assert torch.equal(first.accepted_box, second.accepted_box)
    assert torch.equal(first.selected_index, second.selected_index)
    assert torch.equal(first.candidates.boxes, second.candidates.boxes)
    assert torch.equal(first.commit_decision.eligible, second.commit_decision.eligible)
    assert all(not key.startswith('future_') for key in online_batch(data))
    assert 'future_valid' in loss_batch(data)


def test_m_is_training_loss_only_and_does_not_replace_acquisition():
    torch.manual_seed(42)
    baseline = JointTracker(config('b1')).eval()
    torch.manual_seed(42)
    augmented = JointTracker(config('b1_mtp')).eval()
    data = values()
    before, after = baseline(data), augmented(data)
    assert torch.equal(before.accepted_box, after.accepted_box)
    assert torch.equal(before.prior.feature, after.prior.feature)
    assert torch.equal(before.prior.acquisition_fraction, after.prior.acquisition_fraction)
    baseline.train()
    augmented.train()
    original = baseline.compute_losses(data, before)
    losses = augmented.compute_losses(data, after)
    assert torch.equal(original['loss_acquisition'], losses['loss_acquisition'])
    torch.testing.assert_close(losses['loss_total'], original['loss_total'] + losses['loss_contribution_m'])
    assert sum(p.numel() for p in augmented.multistep.parameters()) == 8642


@pytest.mark.parametrize('arm', ('b1_mtp', 'full'))
def test_eval_forward_and_loss_do_not_require_future_labels_or_execute_m(arm, monkeypatch):
    model, data = JointTracker(config(arm)).eval(), values()
    for key in ('future_displacement', 'future_dt', 'future_valid'):
        data.pop(key)
    def forbidden(*args, **kwargs):
        raise AssertionError('M must not execute in evaluation')
    monkeypatch.setattr(model.multistep, 'forward', forbidden)
    output = model(data)
    losses = model.compute_losses(data, output)
    assert 'loss_m' not in losses and 'loss_contribution_m' not in losses
    assert torch.isfinite(losses['loss_total'])


def test_m_future_tail_masks_do_not_delete_current_endpoints_or_hide_valid_nan():
    model, data = JointTracker(config('b1_mtp')), values()
    data['future_valid'][0, 1] = False
    data['future_displacement'][0, 1] = float('nan')
    data['future_dt'][0, 1] = float('nan')
    output = model(data)
    result = compute_multistep_loss(data, output.prior, model.multistep)
    assert result['m_h2_valid'].item() == 2
    assert result['m_h4_valid'].item() == 1
    assert torch.isfinite(result['loss_m'])
    data['future_valid'][0, 1] = True
    with pytest.raises(ValueError):
        compute_multistep_loss(data, output.prior, model.multistep)


def test_m_gradient_reaches_prior_after_zero_output_layer_learns():
    model, data = JointTracker(config('b1_mtp')), values()
    # 首步零输出头阻断 context 梯度是初始化数学；非零末层后必须连通。
    with torch.no_grad():
        model.multistep.head[-1].weight.fill_(.01)
    output = model(data)
    compute_multistep_loss(data, output.prior, model.multistep)['loss_m'].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.prior.parameters())
    assert all(p.grad is None for p in model.observation.parameters())
    assert all(p.grad is None for p in model.decoder.parameters())


def test_m_has_no_unbounded_state_and_keeps_fp32_parameters():
    head = FutureMotionHead()
    assert not list(head.named_buffers())
    assert all(p.dtype == torch.float32 for p in head.parameters())
    assert torch.count_nonzero(head.head[-1].weight) == 0
    assert torch.count_nonzero(head.head[-1].bias) == 0


def test_outer_autocast_cannot_cast_decoder_projections_or_feedforward():
    model = JointTracker(config('b1')).eval()
    projections, cores, handles = [], [], []

    def projected(module, args, output):
        projections.append((args[0].dtype, output.dtype))

    def core_inputs(module, args):
        cores.append(tuple(value.dtype for value in args[:3]))

    for module in model.decoder.modules():
        if isinstance(module, torch.nn.Linear):
            handles.append(module.register_forward_hook(projected))
        elif isinstance(module, EfficientAttention):
            handles.append(module.register_forward_pre_hook(core_inputs))
    try:
        with torch.no_grad(), torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            output = model(values())
    finally:
        for handle in handles:
            handle.remove()
    assert projections and all(pair == (torch.float32, torch.float32) for pair in projections)
    assert len(cores) == 9 and all(values == (torch.float32,) * 3 for values in cores)
    assert output.accepted_box.dtype == torch.float32
