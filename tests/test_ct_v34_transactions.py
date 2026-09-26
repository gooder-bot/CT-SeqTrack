"""真实 v34 网络的两帧 CPU 工程事务；不使用 Lightning Trainer 或保存权重。"""
import pytest
import torch

from models.ct_v31.config import load_config
from models.ct_v31.data import RawEndpointDataset
from models.ctseqtrackv31 import CTSEQTRACKV31
from tests.test_ct_v31_runtime import TinySource, request


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(34)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('arm', ['b0', 'full'])
@pytest.mark.parametrize('window', [3, 4])
def test_real_network_two_adam_commits_use_detached_predictions(arm, window):
    config = load_config('cfgs/ct_seqtrack/34_b0_context_mini.yaml', dict(
        ct_engineering_check=True, workers=0, point_sample_size=16, batch_size=2,
        v31_arm=arm, v31_short_window=window))
    host = CTSEQTRACKV31(config).train()
    raw = RawEndpointDataset(TinySource((3, 3)))
    optimizer = host.configure_optimizers()['optimizer']
    previous = None
    for frame in (1, 2):
        rows = [raw[request(frame, track=i, end=3)] for i in range(2)]
        batch, output = host._forward_raw(rows, host.train_builder, training=True)
        if previous is not None:
            torch.testing.assert_close(batch['anchor_box'], previous, rtol=0, atol=1e-6)
        assert not batch['history_boxes'].requires_grad
        assert not output.observation.history_support.requires_grad
        assert not output.observation.current_support.requires_grad
        assert output.observation.coarse_features.requires_grad
        losses = host.tracker.compute_losses(batch, output)
        assert torch.isfinite(losses['loss_total'])
        optimizer.zero_grad(set_to_none=True)
        losses['loss_total'].backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in host.parameters())
        optimizer.step()
        previous = output.accepted_box.detach().clone()
        previous[:, :3] += batch['anchor_box'][:, :3]
        host._pending_train = (output, batch, len(rows))
        host.on_train_batch_end(None, rows, frame - 1)
        assert host._pending_train is None and host.train_builder._pending is None
        with pytest.raises(RuntimeError, match='duplicate commit'):
            host.train_builder.commit(output, batch)
    assert host._epoch_steps == 2 and host._epoch_rows == 4
    assert not host.train_builder.states
    assert host.tracker.decoder.context_fusion.weight.detach().abs().sum() > 0
