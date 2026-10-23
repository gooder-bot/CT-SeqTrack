"""保留物理时间、sigma梯度隔离与acquisition监督。"""
import pytest
import torch
from models.ct_v38.prior import PhysicalTimePrior
from tests.helpers.synthetic_batch import make_batch

@pytest.mark.parametrize('backend',['cfc','gru'])
def test_sigma_cannot_update_mean_or_temporal_features(backend):
    prior = PhysicalTimePrior(temporal_backend=backend)
    out = prior(make_batch())
    out.log_sigma.sum().backward()
    assert prior.sigma_head.bias.grad is not None
    assert prior.mean_head.weight.grad is None
    assert all(p.grad is None for p in prior.temporal_cell.parameters())


@pytest.mark.parametrize('backend',['cfc','gru'])
def test_acquisition_gradient_reaches_temporal_cell_after_adapter_initialization(backend):
    prior = PhysicalTimePrior(temporal_backend=backend)
    with torch.no_grad():
        prior.acquisition_head[-1].weight.fill_(.01)
    out = prior(make_batch())
    out.acquisition_fraction.sum().backward()
    assert sum(p.grad.abs().sum() for p in prior.temporal_cell.parameters() if p.grad is not None) > 0


def test_pair_mask_excludes_relocalization_without_disabling_acquisition():
    batch = make_batch(1)
    batch['history_boxes'][0, 2, 0] = 50.
    batch['history_pair_valid'][0, 1] = False
    prior = PhysicalTimePrior()
    out = prior(batch)
    torch.testing.assert_close(out.mean_xy[0], torch.tensor([.3, 0.]))
    batch['history_pair_valid'].zero_()
    out = prior(batch)
    assert not out.valid.any()
    assert out.acquisition_fraction.requires_grad
    torch.testing.assert_close(out.box, batch['fallback_box'])

