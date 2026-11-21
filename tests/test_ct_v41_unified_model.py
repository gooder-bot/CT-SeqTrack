"""联合搜索、横向 q0 与窗口课程共用同一模型的前向和梯度合同。"""
import numpy as np
import torch

from models.ct_v41.data import BatchBuilder, EndpointRequest, RawEndpointDataset
from models.ct_v41.config import load_config
from models.ct_v41.model import JointTracker
from models.ct_v41.verification import final_candidate_support
from tests.helpers.raw_tracking import TinySource
from tools.check_v41_server import ROOT, check_output, gradient_summary, parse_args, smoke_config

CONFIG = ROOT / 'cfgs/ct_seqtrack/41_full_kitti_joint_training.yaml'


class SearchScene(TinySource):
    def get_frames(self, index, frame_ids):
        frames = super().get_frames(index, frame_ids)
        for step, frame in zip(frame_ids, frames):
            frame['3d_bbox'][0] += .04 * step**2
            # 横向密集点横跨框边，可检验 B3 是否按已移动的 q0 重读证据。
            local = np.zeros((31, 3))
            local[:, 1] = np.linspace(-1.5, 1.5, len(local))
            remote = np.array([[15., 0., 0.], [15.1, .1, 0.], [15., -.1, .1]])
            xyz = np.concatenate((local, remote)) + frame['3d_bbox'][:3]
            if index == 1 and step == 3:
                xyz = xyz[:0]
            frame.update(pc=xyz, point_ids=np.arange(len(xyz)), raw_frame_id=step)
        return frames


def model_and_training_batch(warmup=10):
    cfg = load_config(CONFIG, dict(v41_q0_lateral_warmup_epochs=warmup, ct_engineering_check=True,
                                   v41_attention_backend='math', batch_size=2, workers=0))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        tracker = JointTracker(cfg)
    with torch.no_grad():
        # 保证横向头具有可验证的充分支持和非零位移。
        tracker.observation.seg_pointnet.fc.bias[1].add_(50.)
        tracker.q0_lateral.point_head[-1].bias.fill_(.9)
        tracker.q0_lateral.gate_head[-1].bias.zero_()
    dataset = RawEndpointDataset(SearchScene((9, 9), dt=1.), training=True,
                                 joint_search=True, enable_m=True)
    builder = BatchBuilder(tracker.config)
    rows = dataset.__getitems__([EndpointRequest(0, i, 0, 3, 4, 3) for i in (0, 1)])
    return tracker, builder, builder.prepare(rows)


def test_six_candidates_use_final_q0_after_warmup_with_empty_current_scan():
    tracker, builder, batch = model_and_training_batch()
    tracker.eval()
    with torch.no_grad():
        for epoch in (1, 10, 11):
            tracker.q0_epoch = epoch
            result = tracker(batch)
            estimate = result.decoder.lateral
            torch.testing.assert_close(result.candidates.boxes[:, 0], estimate.box, rtol=0, atol=0)
            assert result.candidates.boxes.shape == (2, 6, 4)
            assert estimate.support_count[0] >= 3 and estimate.support_count[1] == 0
            torch.testing.assert_close(estimate.box[1], estimate.base_box[1], rtol=0, atol=0)
            if epoch <= 10:
                assert estimate.applied.eq(0).all() and estimate.box is estimate.base_box
                continue
            assert estimate.applied[0].abs() > .1
            support = final_candidate_support(result.observation, batch, estimate.box[:, None])['mask']
            original = final_candidate_support(result.observation, batch, estimate.base_box[:, None])['mask']
            assert not torch.equal(support[0], original[0])
            assert torch.equal(result.joint.support_mask[:, :1], support)
            assert torch.equal(result.verification.support_mask, result.joint.support_mask)
            assert len(result.joint.observations.coarse) > 0
            smoke = check_output(batch, result, joint_search=True)
            assert smoke['candidate_slots'] == 6 and smoke['rows'] == 2
        records = builder.commit(result, batch, diagnostics=True)
        assert len(records) == 2 and not builder.states


def test_joint_and_lateral_main_and_auxiliary_gradients():
    tracker, _, batch = model_and_training_batch(warmup=0)
    tracker.train()
    tracker.q0_epoch = 1
    main = tracker(batch)
    assert main.decoder.lateral.applied[0].abs() > .1
    losses = tracker.compute_losses(batch, main)
    assert losses['m_valid_endpoints'] > 0
    assert losses['loss_joint_grid_object'] > 0
    assert torch.isfinite(losses['loss_total'])
    # 主流共享一次 B0 读点。联合核验和搜索的监督不能暗中再更新横向头或主干。
    joint_losses = (losses['loss_geometry_quality'] + losses['loss_success'] +
                   losses['loss_precision'] + losses['loss_ranking'] + losses['loss_joint_total'])
    upstream = (tracker.q0_lateral.point_head[-1].bias, tracker.q0_lateral.gate_head[-1].bias,
                tracker.observation.seg_pointnet.fc.weight, tracker.decoder.pose_head.bias)
    assert all(grad is None for grad in torch.autograd.grad(joint_losses, upstream,
               retain_graph=True, allow_unused=True))
    losses['loss_total'].backward()

    def has_gradient(module):
        return any(p.grad is not None and p.grad.isfinite().all() and p.grad.abs().sum() > 0
                   for p in module.parameters())

    for module in (tracker.q0_lateral.point_head, tracker.q0_lateral.gate_head, tracker.decoder,
                   tracker.prior, tracker.multistep, tracker.observation,
                   tracker.joint_search, tracker.verifier):
        assert has_gradient(module), type(module).__name__

    tracker.zero_grad(set_to_none=True)
    aux = tracker.forward_aux(batch)
    assert not aux.decoder.lateral.box.requires_grad and not aux.decoder.lateral.gate.requires_grad
    aux_losses = tracker.compute_aux_losses(batch, aux)
    assert not {'loss_q0_lateral_gate', 'loss_q0_lateral_point', 'loss_m'} & aux_losses.keys()
    assert torch.isfinite(aux_losses['loss_total'])
    aux_losses['loss_total'].backward()
    named = list(tracker.named_parameters())
    groups = gradient_summary(named, [parameter.grad for _, parameter in named])
    assert groups['joint_search']['nonzero'] > 0 and groups['q0_lateral']['connected'] == 0
    for module in (tracker.q0_lateral, tracker.observation, tracker.decoder, tracker.prior, tracker.multistep):
        assert all(p.grad is None for p in module.parameters()), type(module).__name__
    for module in (tracker.verifier, tracker.recovery_head, tracker.joint_search):
        assert has_gradient(module), type(module).__name__


def test_server_smoke_accepts_the_formal_joint_recipe_and_bank(tmp_path):
    bank = tmp_path / 'history_e57.json'
    args = parse_args(['--cfg', str(CONFIG), '--path', str(tmp_path), '--devices', '0',
                       '--v41_history_bank', str(bank), '--v41_q0_lateral_warmup_epochs', '0'])
    cfg = smoke_config(args)
    assert cfg.v41_history_bank == str(bank) and cfg.v41_q0_lateral_warmup_epochs == 0
    assert cfg.v41_joint_search and cfg.v41_q0_lateral and cfg.ct_engineering_check
    assert cfg.v41_attention_backend == 'fa2_varlen'
