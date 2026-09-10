"""实际 sampler/host 的短递归输入：因果性、监督和状态隔离。"""
import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.test_ct_v27_input_flow import sampler_runtime, _case
from tests.test_ct_v27_full_model import full_model_runtime
from utils.config import load_yaml_config
from utils.v29_rollin import process_query, prepare_observation_batch, _initial_box
from models.ct_variant import configure_ct_variant


def setup_case(runtime, start=4, endpoint=8):
    _, _, sequence, _, _, _, _ = _case(runtime, 'b0')
    config = runtime[3](load_yaml_config(Path(__file__).resolve().parents[1] /
                         'cfgs/ct_seqtrack/29_b0_nuscenes_full.yaml'))
    configure_ct_variant(config)
    config.candidate_trajectory_mode = 'shared_se2'
    config.ct_observation_payload_mode = 'seqtrack_core'
    item = dict(mode='rollin', index=7, candidate=3, epoch=0, start=start,
                endpoint=endpoint, tracklet_id=0, tracklet_key='test/track',
                frames=sequence[start:endpoint+1], first=sequence[0])
    return config, item


def test_query_predicted_history_is_not_gt_supervision(sampler_runtime):
    config, item = setup_case(sampler_runtime)
    predictions = {i: copy.deepcopy(item['frames'][i-4]['3d_bbox']) for i in range(4, 8)}
    for box in predictions.values():
        box.center += np.array([3., -1., 0.])
    row, anchor = process_query(item, predictions, 8, config)
    assert np.linalg.norm(row['ref_boxs'][0, :3]) < 1e-5
    assert np.linalg.norm(row['box_label_prev'][0, :3]) > 2.
    from datasets.points_utils import transform_box
    expected = transform_box(item['frames'][3]['3d_bbox'], anchor)
    np.testing.assert_allclose(row['box_label_prev'][0, :3], expected.center, atol=1e-6)
    np.testing.assert_allclose(row['b0_coarse_target'][:3],
                              row['box_label'][:3] - row['box_label_prev'][0, :3], atol=1e-6)
    assert np.linalg.norm(row['b0_coarse_target'][:3]) < 1.


def test_current_and_future_gt_do_not_change_query_inputs(sampler_runtime):
    config, item = setup_case(sampler_runtime)
    predictions = {4: _initial_box(item, config)}
    original, _ = process_query(item, predictions, 5, config)
    changed = copy.deepcopy(item)
    for frame in changed['frames'][1:]:
        frame['3d_bbox'].center += np.array([20., 4., 3.])
        frame['3d_bbox'].wlh *= 2
    result, _ = process_query(changed, predictions, 5, config)
    for key in ('points', 'ref_boxs', 'candidate_bc', 'b0_point_ids', 'bbox_size', 'valid_mask'):
        np.testing.assert_array_equal(original[key], result[key], err_msg=key)
    assert not np.array_equal(original['box_label'], result['box_label'])
    np.testing.assert_array_equal(original['valid_mask'], [1, 0, 0])
    # 局部首query的起点已经扰动，使用0.2/0.8软prior而非0/1 GT prior。
    np.testing.assert_array_equal(np.unique(original['points'][:1024, 4]),
                                  np.asarray([.2, .8], dtype=np.float32))


@pytest.mark.parametrize('endpoint', [0, 1, 2, 8])
def test_actual_host_rollin_restores_buffers_flags_rng(full_model_runtime, endpoint):
    runtime = full_model_runtime[0]
    config, item = setup_case(runtime, start=max(0, endpoint-4), endpoint=endpoint)
    torch.manual_seed(42)
    host = full_model_runtime[2](config).train()
    buffers = {k: v.clone() for k, v in host.named_buffers()}
    rng = torch.get_rng_state().clone()
    flags = [m.training for m in host.modules()]
    batch = prepare_observation_batch(host, [item, copy.deepcopy(item)])
    assert torch.equal(rng, torch.get_rng_state())
    assert flags == [m.training for m in host.modules()]
    for key, value in host.named_buffers():
        assert torch.equal(buffers[key], value), key
    assert host._ct_v29_rollin_diagnostics['sample_forwards'] == 2 * min(3, max(0, endpoint-1))
    assert batch['points'].shape == (2, 4096, 5)
    assert torch.isfinite(batch['b0_coarse_target']).all()
    loss = host.compute_loss(batch, host(batch))['loss_b0_transaction']
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in host.parameters())


def test_mixed_teacher_rollin_dual_stream_keeps_one_update(full_model_runtime, monkeypatch):
    from tests.test_ct_v28_audit_host import _attach_training_shell
    from utils.v29_rollin import observation_collate
    runtime = full_model_runtime[0]
    config, item = setup_case(runtime)
    torch.manual_seed(42)
    host = full_model_runtime[2](config).train()
    teacher_predictions = {i: copy.deepcopy(item['frames'][i-4]['3d_bbox']) for i in range(4, 8)}
    teacher, _ = process_query(item, teacher_predictions, 8, config)
    teacher['candidate_id'] = np.int64(0)
    mixed = observation_collate([dict(mode='teacher', sample=teacher), item])
    batch = dict(ct_stream_schema='ct_seqtrack.train.v4', observation=mixed, mechanism=None)
    optimizer = host.configure_optimizers()['optimizer']
    _attach_training_shell(host, optimizer, monkeypatch)
    transferred = host.transfer_batch_to_device(batch, torch.device('cpu'), 0)
    assert transferred['observation'] is mixed
    optimizer.zero_grad(set_to_none=True)
    loss = host.training_step(transferred, 0)
    loss.backward()
    host.on_before_optimizer_step(optimizer)
    optimizer.step()
    host.on_train_batch_end(loss, batch, 0)
    assert int(host.ct_b0_update_step) == 1
    assert host._ct_v29_rollin_diagnostics['sample_forwards'] == 3
    assert all(int(value['step']) == 1 for value in optimizer.state.values() if 'step' in value)
