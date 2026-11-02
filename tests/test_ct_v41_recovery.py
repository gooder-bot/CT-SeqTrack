"""v41 恢复几何、测量接管和辅助梯度合同；仅上传服务器后执行。"""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from models.ct_v41.config import normalize_config
from models.ct_v41.contracts import VerificationOutput
from models.ct_v41.input import online_batch
from models.ct_v41.losses import endpoint_bce, supervision_targets
from models.ct_v41.model import JointTracker, build_candidates, select_candidates
from models.ct_v41.recovery import RecoveryResidualHead, recovery_loss_terms
from models.ct_v41.verification import LocalBoxReader, current_candidate_support
from tests.helpers.synthetic_batch import batch as synthetic_batch


@pytest.fixture(autouse=True)
def restore_cpu_state():
    threads, rng = torch.get_num_threads(), torch.get_rng_state()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)
    torch.set_rng_state(rng)


def values(b=1, n=16):
    data = synthetic_batch(b, n)
    data['recovery_seed_box'] = data['fallback_box'].clone()
    data['recovery_seed_valid'] = torch.ones(b, dtype=torch.bool)
    return data


def config():
    return normalize_config(dict(v41_arm='full', v41_attention_backend='math',
        ct_engineering_check=True, point_sample_size=16, batch_size=2,
        workers=0, epoch=2, execution_profile='optimized_fp32'))


def simple_output(data, *, q0_real=True):
    b, _, n, _ = data['points'].shape
    observation = SimpleNamespace(sequence_valid=torch.full((b,), q0_real),
        foreground_probability=torch.ones(b, 4, n))
    decoder = SimpleNamespace(q0_box=data['fallback_box'].clone(),
        q0_quality_logits=torch.full((b,), 2.))
    prior = SimpleNamespace(box=data['fallback_box'].clone())
    recovery = RecoveryResidualHead()(torch.ones(b, 64), data['recovery_seed_box'],
        data['box_size'], torch.ones(b, n, dtype=torch.bool), data['recovery_seed_valid'])
    candidates = build_candidates(decoder, observation, data, prior, recovery=recovery)
    verification = VerificationOutput(torch.zeros(b, 4, 3), torch.zeros(b, 4, n, dtype=torch.bool),
        torch.zeros(b, 4, 46))
    return SimpleNamespace(observation=observation, decoder=decoder, prior=prior,
        recovery=recovery, candidates=candidates, verification=verification)


def test_zero_head_preserves_seed_and_rotates_bounded_xyz_in_seed_axes():
    head = RecoveryResidualHead()
    assert sum(p.numel() for p in head.parameters()) == 195
    feature = torch.ones(1, 64)
    seed = torch.tensor([[10., 20., 30., torch.pi/2]])
    size = torch.tensor([[4., 2., 1.5]])
    mask, valid = torch.ones(1, 3, dtype=torch.bool), torch.ones(1, dtype=torch.bool)
    first = head(feature, seed, size, mask, valid)
    assert torch.equal(first.box, seed)
    with torch.no_grad():
        head.xyz.bias.copy_(torch.atanh(torch.tensor([.5, -.25, .5])))
    changed = head(feature, seed, size, mask, valid)
    torch.testing.assert_close(changed.box[:, :3]-seed[:, :3], torch.tensor([[.5, 2., .75]]))
    assert torch.equal(changed.box[:, 3], seed[:, 3])
    shifted = seed + torch.tensor([[100., -50., 7., 0.]])
    torch.testing.assert_close(head(feature, shifted, size, mask, valid).box-shifted,
                               changed.box-seed)
    assert (changed.normalized_delta.abs() <= 1).all()


def test_local_reader_exposes_actual_unique_points_in_original_order():
    reader = LocalBoxReader()
    pool = dict(ids=torch.tensor([[40, 10, 30, 20]]), valid=torch.ones(1, 4, dtype=torch.bool),
        xyz=torch.tensor([[[0., 0., 0.], [100., 0., 0.], [.1, 0., 0.], [.2, 0., 0.]]]),
        features=torch.zeros(1, 4, 64), probability=torch.ones(1, 4), source=torch.zeros(1, 4))
    _, statistics = reader(pool, torch.zeros(1, 1, 4), torch.ones(1, 1, 4, dtype=torch.bool),
        torch.tensor([[4., 2., 1.5]]))
    assert statistics['local_point_mask'].tolist() == [[[True, False, True, True]]]


def test_zero_residual_same_geometry_retains_current_measurement_provenance():
    data = values()
    out = simple_output(data)
    assert out.candidates.valid.tolist() == [[True, False, False, True]]
    assert out.candidates.measurement_candidate.tolist() == [[True, False, False, True]]
    assert out.candidates.alias_of[0, 3].item() == -1
    decision, _, _, _ = select_candidates(out.candidates, out.verification, out.observation, data)
    assert decision.selected_index.item() == 0
    out = simple_output(data, q0_real=False)
    decision, commit, quality, _ = select_candidates(out.candidates, out.verification, out.observation, data)
    assert decision.selected_index.item() == 3  # R 同分先于 prior/fallback。
    assert commit.eligible.item()
    assert quality.item() == 0.  # R 的 T 不能冒充 q0 legacy quality。


def test_recovery_uses_final_box_support_and_propagation_never_gets_trust():
    data = values()
    out = simple_output(data)
    out.verification.logits[:, 3] = 5.
    decision, commit, _, trust = select_candidates(out.candidates, out.verification, out.observation, data)
    assert decision.selected_index.item() == 3 and commit.eligible.item()
    assert torch.count_nonzero(trust[:, 1:3]) == 0
    moved = simple_output(data)
    moved.verification.logits[:, 3] = 5.
    moved.candidates.boxes[:, 3, 2] = 20.
    _, commit, _, _ = select_candidates(moved.candidates, moved.verification, moved.observation, data)
    assert not commit.eligible.item() and commit.support_mask.sum().item() == 0
    out = simple_output(data, q0_real=False)
    out.verification.logits[:, 1] = 10.
    decision, commit, quality, _ = select_candidates(out.candidates, out.verification, out.observation, data)
    assert decision.selected_index.item() == 1
    assert not commit.eligible.item() and commit.trusted_score.item() == 0. and quality.item() == 0.


def test_nonzero_recovery_reenters_reader_and_uses_refined_support(monkeypatch):
    """真实R路径不能复用seed读取/支持；零初始化smoke不会覆盖这个差异。"""
    model, data = JointTracker(config()).eval(), values()
    data['points'][..., :3].zero_()
    reads = []
    read_local = model.verifier.read_local
    def record_read(pool, boxes, valid, size):
        reads.append(boxes.detach().clone())
        return read_local(pool, boxes, valid, size)
    monkeypatch.setattr(model.verifier, 'read_local', record_read)
    with torch.no_grad():
        model.recovery_head.xyz.bias.copy_(torch.atanh(torch.tensor([0., 0., .9])))
        output = model(data)
    assert len(reads) == 2
    assert torch.equal(reads[0][:, 0], data['recovery_seed_box'])
    assert torch.equal(reads[1], output.candidates.boxes)
    assert torch.equal(reads[1][:, 3], output.recovery.box)
    assert not torch.equal(reads[0][:, 0], reads[1][:, 3])
    assert output.recovery.valid.item() and output.recovery.point_mask.sum().item() == 16
    final_support = current_candidate_support(output.observation, data, reads[1])
    # 分割概率来自随机主路，支持可以为零；纯最终框内几何必须已排除所有当前点。
    from models.ct_v41.geometry import inside_final_boxes
    assert inside_final_boxes(data['points'][:, -1, :, :3], reads[0], data['box_size']).all()
    assert not inside_final_boxes(data['points'][:, -1, :, :3], reads[1][:, 3:4], data['box_size']).any()
    assert final_support['count'][0, 3].item() == 0
    output.verification.logits[:, 3] = 5.
    decision, commit, quality, _ = select_candidates(
        output.candidates, output.verification, output.observation, data)
    assert decision.selected_index.item() == 3
    assert not commit.eligible.item() and not commit.support_mask.any()
    assert quality.item() == 0.


def test_verifier_losses_cannot_train_recovery_pose_through_candidate_geometry():
    """R只消费显式XYZ定位梯度；B3重新读取可训练共享reader，不穿过硬候选。"""
    model, data = JointTracker(config()).train(), values(b=2)
    output = model.forward_aux(data)
    losses = model.compute_aux_losses(data, output)
    verification_loss = (.125 * losses['loss_success'] + .125 * losses['loss_precision']
                         + .5 * losses['loss_trusted_write'] + .1 * losses['loss_ranking'])
    parameters = tuple(model.recovery_head.parameters())
    gradients = torch.autograd.grad(verification_loss, parameters, allow_unused=True, retain_graph=True)
    assert all(gradient is None for gradient in gradients)
    assert losses['recovery_supervised_count'].item() > 0
    gradients = torch.autograd.grad(losses['loss_recovery_xyz'], parameters)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(gradient.abs().sum().item() > 0 for gradient in gradients)


def test_duplicate_raw_ids_do_not_create_three_supporting_measurements():
    data = values()
    data['point_ids'][:, -1].fill_(7)
    out = simple_output(data, q0_real=False)
    support = current_candidate_support(out.observation, data, out.candidates.boxes)
    assert support['count'].max().item() <= 1
    _, commit, _, _ = select_candidates(out.candidates, out.verification, out.observation, data)
    assert not commit.eligible.any()
    output = JointTracker(config()).eval()(data)
    assert output.recovery.point_mask.sum().item() <= 1
    assert not recovery_loss_terms(data, output.recovery)['valid'].any()


def test_measurement_T_targets_cover_q0_and_R_with_endpoint_normalization():
    data = values()
    data['points'][..., :3].zero_()
    data['target_box'] = data['fallback_box'].clone()
    out = simple_output(data)
    targets = supervision_targets(data, out)
    assert targets['trusted_valid'].tolist() == [[True, False, False, True]]
    assert targets['trusted'].tolist() == [[1., 0., 0., 1.]]
    logits = torch.tensor([[.4, 8., 8., .4]])
    both, _ = endpoint_bce(logits, targets['trusted'], targets['trusted_valid'])
    single, _ = endpoint_bce(logits[:, :1], targets['trusted'][:, :1], targets['trusted_valid'][:, :1])
    torch.testing.assert_close(both, single)


def test_xyz_loss_needs_three_actual_target_points_and_representable_residual():
    data = values(b=3)
    out = simple_output(data)
    data['segmentation_labels'][1, -1] = 0
    data['segmentation_labels'][1, -1, :2] = 1
    data['target_box'][2, 2] = 100.
    terms = recovery_loss_terms(data, out.recovery)
    assert terms['valid'].tolist() == [True, False, False]
    assert torch.isfinite(terms['loss'])
    terms['loss'].backward()
    assert out.recovery.normalized_delta.requires_grad


def test_no_current_points_cannot_create_R_even_with_valid_history():
    data = values()
    data['point_valid'][:, -1] = False
    model = JointTracker(config()).eval()
    output = model(data)
    assert output.observation.sequence_valid.item()
    assert not output.recovery.valid.item()
    assert not output.candidates.raw_valid[0, 3].item()
    assert not output.commit_decision.eligible.item()


def test_online_recovery_is_independent_of_current_and_future_GT():
    model, data = JointTracker(config()).eval(), values()
    first = model(data)
    poisoned = deepcopy(data)
    poisoned['target_box'].fill_(float('nan'))
    poisoned['segmentation_labels'].zero_()
    poisoned['future_displacement'] = torch.full((9, 7, 2), float('nan'))
    second = model(poisoned)
    assert torch.equal(first.candidates.boxes, second.candidates.boxes)
    assert torch.equal(first.selected_index, second.selected_index)
    assert torch.equal(first.commit_decision.eligible, second.commit_decision.eligible)
    assert 'target_box' not in online_batch(poisoned)
    assert 'recovery_seed_box' in online_batch(poisoned)


def test_point_cache_does_not_reuse_recovery_seed_and_requires_explicit_validity():
    from utils.batch_materialization import attach_materialization_cache
    data = values()
    attach_materialization_cache(data, enabled=True)
    first = online_batch(data)
    data['recovery_seed_box'] = data['recovery_seed_box'] + torch.tensor([[2., 0., 0., 0.]])
    second = online_batch(data)
    assert torch.equal(first['point_valid'], second['point_valid'])
    assert not torch.equal(first['recovery_seed_box'], second['recovery_seed_box'])
    data.pop('recovery_seed_valid')
    with pytest.raises(ValueError, match='supplied together'):
        online_batch(data)


def test_auxiliary_only_trains_B3_and_R_and_restores_exact_upstream_modes(monkeypatch):
    import models.ct_v41.losses as loss_module
    model, data = JointTracker(config()).train(), values(b=2)
    # 模拟主分支已有的局部 BN 隔离，不能被辅助恢复成统一 train()。
    bn = next(module for module in model.observation.modules()
              if isinstance(module, torch.nn.modules.batchnorm._BatchNorm))
    bn.eval()
    before_modes = {name: module.training for name, module in model.named_modules()}
    before_buffers = {name: value.clone() for name, value in model.named_buffers()}
    def forbidden(*args, **kwargs):
        raise AssertionError('auxiliary must not execute main/M losses')
    monkeypatch.setattr(model.multistep, 'forward', forbidden)
    monkeypatch.setattr(loss_module, 'base_losses', forbidden)
    output = model.forward_aux(data)
    loss = model.compute_aux_losses(data, output)
    loss['loss_total'].backward()
    assert before_modes == {name: module.training for name, module in model.named_modules()}
    assert all(torch.equal(value, before_buffers[name]) for name, value in model.named_buffers())
    for module in (model.observation, model.decoder, model.prior, model.multistep):
        assert all(parameter.grad is None for parameter in module.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.verifier.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.recovery_head.parameters())
    assert all(p.requires_grad for p in model.parameters())


def test_original_mtp_private_initialization_is_preserved():
    from models.ct_v40.config import normalize_config as v40_config
    from models.ct_v40.model import JointTracker as V40Tracker
    previous = V40Tracker(v40_config(dict(v40_arm='full', v40_attention_backend='math',
        ct_engineering_check=True, point_sample_size=16, batch_size=2,
        workers=0, epoch=2, seed=42)))
    current = JointTracker(config())
    assert previous.multistep.state_dict().keys() == current.multistep.state_dict().keys()
    for name, value in previous.multistep.state_dict().items():
        assert torch.equal(value, current.multistep.state_dict()[name]), name
