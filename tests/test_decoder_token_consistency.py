import pytest
import torch

from models.ct_v2.decoder_token_consistency import (
    DecoderTokenConsistencyLoss,
    GradientRatioWeightSelector,
)


def test_decoder_consistency_is_finite_and_teacher_is_stop_gradient():
    torch.manual_seed(31)
    module = DecoderTokenConsistencyLoss(
        input_dim=8, projection_dim=8, hidden_dim=16)
    clean = torch.randn(4, 3, 8, requires_grad=True)
    irregular = torch.randn(4, 3, 8, requires_grad=True)
    output = module(clean, irregular)
    output["loss_decoder_token_consistency"].backward()
    assert clean.grad is None
    assert irregular.grad is not None
    assert torch.isfinite(irregular.grad).all()
    assert torch.isfinite(output["decoder_tc_effective_rank_ratio"])
    assert 0.0 <= float(output["decoder_tc_effective_rank_ratio"]) <= 1.0


def test_teacher_updates_by_ema_and_stays_frozen():
    module = DecoderTokenConsistencyLoss(
        input_dim=4, projection_dim=4, hidden_dim=8,
        teacher_momentum=0.5)
    before = [parameter.clone() for parameter in
              module.teacher_projector.parameters()]
    with torch.no_grad():
        for parameter in module.student_projector.parameters():
            parameter.add_(1.0)
    module.update_teacher()
    after = list(module.teacher_projector.parameters())
    assert any(not torch.equal(left, right) for left, right in zip(before, after))
    assert all(not parameter.requires_grad for parameter in after)
    assert int(module.teacher_updates) == 1


def test_gradient_weight_selector_freezes_candidate():
    selector = GradientRatioWeightSelector(
        candidates=(0.001, 0.003, 0.01),
        audit_batches=3,
        target_ratio=0.075)
    for _ in range(3):
        selector.observe(10.0)
    assert bool(selector.frozen)
    assert bool(selector.guardrail_passed)
    assert float(selector.value(torch.tensor(0.0))) == pytest.approx(0.01)
