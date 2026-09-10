"""v29 真正 sampler/host 的获取边界与已接受状态反馈，不模拟网络计算。"""

import copy
import csv
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data._utils.collate import default_collate

from tests.test_ct_v27_input_flow import sampler_runtime, _case
from tests.test_ct_v27_full_model import full_model_runtime
from tests.test_ct_v29_b0_host import construct, b0_buffers
from models.ct_variant import configure_ct_variant
from utils.acquisition_v29 import support_vertical_interval
from utils.config import load_yaml_config
from utils.v27_input import build_v27_eval_input
from utils.v29_policy import mechanism_behavior_policy


ROOT = Path(__file__).resolve().parents[1]
CAUSAL_KEYS = ('points', 'candidate_bc', 'ref_boxs', 'bbox_size',
    'b0_point_ids', 'b0_point_valid_mask', 'b0_unique_mask',
    'motion_main_ref_boxs', 'motion_main_delta_t', 'motion_acquisition_features',
    'ct_extension_points', 'ct_extension_point_ids', 'ct_extension_valid_mask',
    'ct_base_point_ids', 'ct_acquisition_margin', 'ct_acquisition_direction_xy',
    'ct_acquisition_support_z_intervals', 'ct_acquisition_support_exists',
    'coordinate_anchor', 'search_v3_support_anchor_xy', 'ct_search_support_valid')


def case29(runtime, backend='gru'):
    sampler, _, sequence, state, payload, host, prediction = _case(runtime, 'full')
    config = runtime[3](load_yaml_config(ROOT / 'cfgs/ct_seqtrack' /
                      f'29_full_{backend}_nuscenes_full.yaml'))
    configure_ct_variant(config)
    config.candidate_trajectory_mode = 'shared_se2'
    config.ct_observation_payload_mode = 'legacy'
    host.config = config
    payload['is_initial_query'] = False
    return sampler, config, sequence, state, payload, host, prediction


@pytest.mark.parametrize('learned', [True, False])
def test_v29_real_train_eval_same_inputs_and_recorded_support_z(sampler_runtime, monkeypatch, learned):
    sampler, config, sequence, state, payload, host, prediction = case29(sampler_runtime)
    prediction = dict(prediction, valid=learned)
    payload['motion_prediction'] = prediction
    original = sampler.resolve_joint_search_geometry
    captured = []

    def observe_geometry(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.append(copy.deepcopy(result))
        return result

    monkeypatch.setattr(sampler, 'resolve_joint_search_geometry', observe_geometry)
    training_sidecar, eval_sidecar = {}, {}
    payload['_ct_diagnostic_sidecar'] = training_sidecar
    training = sampler.motion_processing_mf(payload, config)
    evaluation, _ = build_v27_eval_input(host, sequence, 8, state.results_bbs,
        recursive_state=state, motion_prediction=prediction, diagnostic_sidecar=eval_sidecar)
    for key in CAUSAL_KEYS:
        np.testing.assert_array_equal(training[key], evaluation[key][0].numpy(), err_msg=key)
    assert training['ct_acquisition_support_z_intervals'].shape == (3, 2)
    assert training['ct_acquisition_support_exists'].shape == (3,)
    for column, box in enumerate(captured[0][:2]):
        assert training['ct_acquisition_support_exists'][column]
        np.testing.assert_allclose(training['ct_acquisition_support_z_intervals'][column],
                                   support_vertical_interval(box), atol=1e-6)
    np.testing.assert_array_equal(training['ct_acquisition_margin'], [3., 2.] if learned else [2., 1.])
    for sidecar in (training_sidecar, eval_sidecar):
        assert sidecar['acquisition']['acquisition_schema_version'] == 'ct_acquisition.v5'
        assert sidecar['acquisition']['actual_support_novel_outside_maximum_count'] == 0
    assert training_sidecar['acquisition']['max_legal_novel_target_count'] == eval_sidecar['acquisition']['max_legal_novel_target_count']


def test_v29_gt_changes_supervision_but_no_causal_inputs_or_supports(sampler_runtime):
    sampler, config, sequence, state, payload, host, prediction = case29(sampler_runtime)
    original, _ = build_v27_eval_input(host, sequence, 8, state.results_bbs,
                                      recursive_state=state, motion_prediction=prediction)
    changed = copy.deepcopy(sequence)
    changed[8]['3d_bbox'].center += np.asarray((18., -14., 4.))
    changed[8]['3d_bbox'].wlh *= 1.4
    altered, _ = build_v27_eval_input(host, changed, 8, state.results_bbs,
                                     recursive_state=state, motion_prediction=prediction)
    for key in CAUSAL_KEYS:
        assert torch.equal(original[key], altered[key]), key
    assert not torch.equal(original['box_label'], altered['box_label'])
    assert not torch.equal(original['target_bbox_size'], altered['target_bbox_size'])
    training = sampler.motion_processing_mf(payload, config)
    changed_payload = dict(payload, this_frame=changed[8])
    altered_training = sampler.motion_processing_mf(changed_payload, config)
    for key in CAUSAL_KEYS:
        np.testing.assert_array_equal(training[key], altered_training[key], err_msg=key)


def test_v29_actual_raw_novel_crop_keeps_vertical_measurement_without_relabeling_base(sampler_runtime):
    sampler, config, sequence, state, payload, host, prediction = case29(sampler_runtime)
    classes = sampler_runtime[1]
    anchor = state.results_bbs[-1]
    local = np.asarray([[.1, 0., 0.], [.2, 0., 0.], [.3, 0., 0.], [5., 0., 2.]])
    world = anchor.rotation_matrix @ local.T + anchor.center[:, None]
    sequence[8]['pc'] = classes.PointCloud(world, point_ids=np.asarray([11, 12, 13, 21]))
    target = copy.deepcopy(sequence[8]['3d_bbox'])
    target.center = world[:, -1].copy()
    sequence[8]['3d_bbox'] = target
    prediction = dict(prediction, mu_xy=np.asarray((4., 0.)), velocity_xy=np.asarray((8., 0.)))
    payload['motion_prediction'] = prediction
    sidecar = {}
    payload['_ct_diagnostic_sidecar'] = sidecar
    current = sampler.motion_processing_mf(payload, config)
    actual_ids = current['ct_extension_point_ids'][current['ct_extension_valid_mask'] > 0]
    assert set(actual_ids) == {21}
    assert not set(actual_ids) & {11, 12, 13}
    diagnostic = sidecar['acquisition']
    assert diagnostic['global_novel_target_count'] == 1
    assert diagnostic['support_novel_xyz_target_count'] == 1
    assert diagnostic['support_novel_z_excluded_target_count'] == 0
    assert diagnostic['max_legal_novel_target_count'] == 1
    assert diagnostic['prepool_recall_of_reachable'] == 1.
    assert current['motion_margin_max_reachable_target_count'] == 1
    old_config = copy.deepcopy(config)
    old_config.ct_enable_v29 = False
    old = sampler.motion_processing_mf(payload, old_config)
    assert not old['ct_extension_valid_mask'].any()
    # 仅Z hull不改变B0真实点和采样；本例3个base点走相同采样分支。
    np.testing.assert_array_equal(current['points'], old['points'])


@pytest.mark.parametrize('backend', ['gru', 'cfc'])
def test_v29_actual_full_record_has_z_metadata_without_b0_bn_or_gradient_ownership_changes(
        full_model_runtime, monkeypatch, backend):
    model = construct(full_model_runtime, 'full_' + backend).train()
    sampler, config, sequence, state, payload, _, _ = case29(full_model_runtime[0], backend)
    payload['motion_prediction'] = model.predict_motion_prepass(
        sequence, 8, state.results_bbs, recursive_state=state)
    batch = default_collate([sampler.motion_processing_mf(payload, config)])
    import models.ct_v2.pipeline_contracts as contracts
    record_class = contracts.AcquisitionRecord
    records = []

    def record(*args, **kwargs):
        actual = record_class(*args, **kwargs)
        records.append(actual)
        return actual

    monkeypatch.setattr(contracts, 'AcquisitionRecord', record)
    before = b0_buffers(model)
    output = model._forward_safe_mechanism(batch)
    assert records
    assert torch.equal(records[-1].support_z_intervals, batch['ct_acquisition_support_z_intervals'])
    assert torch.equal(records[-1].support_exists, batch['ct_acquisition_support_exists'].bool())
    assert not records[-1].support_z_intervals.requires_grad
    assert all(torch.equal(value, b0_buffers(model)[name]) for name, value in before.items())
    losses = model.compute_loss(batch, output)
    gradients = torch.autograd.grad(losses['loss_plugin_transaction'],
        [p for name, p in model.named_parameters() if not model._ct_any_plugin_parameter(name)],
        allow_unused=True)
    assert all(gradient is None for gradient in gradients)


def test_v29_host_commits_accepted_behavior_and_next_query_reads_that_prediction(full_model_runtime):
    model = construct(full_model_runtime, 'full_gru').train()
    _, _, sequence, state, _, _, prediction = case29(full_model_runtime[0])
    epoch = next(i for i in range(100) if mechanism_behavior_policy(42, i, state.tracklet_key)['kind'] == 'always')
    original = state.clone()
    anchor = state.results_bbs[-1]
    raw = dict(online_slot=0, candidate_id=0, this_frame_id=8, prev_frame_ids=[7, 6, 5],
               this_frame=sequence[8], online_epoch=epoch, tracklet_key=state.tracklet_key)
    model._ct_online_batch_context = [dict(raw=raw, state=state)]
    output = dict(observation_aux_estimation_boxes=torch.zeros(1, 4),
                  ct_router_bounded_residual_xy=torch.tensor([[.5, 0.]]),
                  ct_router_evidence_valid=torch.ones(1), ct_b3_action_score=torch.tensor([-1.]),
                  ct_observation_quality=torch.tensor([[100., .4, .2, 1.]]))
    model._apply_v29_mechanism_policy(output)
    assert output['ct_router_applied_gate'].item() == 1
    model._commit_online_recursive_predictions(output)
    expected = model._local_prediction_to_world(torch.tensor([.5, 0., 0., 0.]), anchor)
    np.testing.assert_allclose(state.results_bbs[8].center, expected.center)
    original.append(8, model._local_prediction_to_world(torch.zeros(4), anchor),
                    sequence[8]['timestamp'], quality=[100., .4, .2, 1.])
    future = copy.deepcopy(sequence[8])
    future.update(frame_id=9, timestamp=sequence[8]['timestamp'] + 500000)
    sequence.append(future)
    actual, _ = build_v27_eval_input(model, sequence, 9, state.results_bbs,
        recursive_state=state, motion_prediction=prediction)
    rejected, _ = build_v27_eval_input(model, sequence, 9, original.results_bbs,
        recursive_state=original, motion_prediction=prediction)
    np.testing.assert_allclose(actual['coordinate_anchor'][0, :3].numpy(), expected.center, atol=1e-6)
    assert not torch.equal(actual['coordinate_anchor'], rejected['coordinate_anchor'])
    assert not torch.equal(actual['points'], rejected['points'])


def test_v29_actual_evaluator_and_csv_keep_novel_reachability_fields(full_model_runtime, tmp_path):
    from utils.v27_evaluation import evaluate_sequence_v27
    from utils.action_calibration_v27 import summarize_rows
    model = construct(full_model_runtime, 'full_gru').eval()
    _, _, sequence, _, _, _, _ = case29(full_model_runtime[0])
    for frame in sequence:
        frame['scene_id'] = 'scene-test'
    with torch.no_grad():
        overlaps, distances, _ = evaluate_sequence_v27(model, sequence[:3])
    rows = model._ct_v27_sequence_endpoints
    assert len(overlaps) == len(distances) == len(rows) == 3
    for row in rows[1:]:
        assert 'acquisition_max_legal_novel_target_count' in row
        assert 'acquisition_support_novel_z_excluded_target_count' in row
        assert 'acquisition_prepool_recall_of_reachable' in row
        assert row['acquisition_global_novel_target_count'] == (
            row['acquisition_max_legal_novel_target_count']
            + row['acquisition_max_legal_unreachable_target_count'])
    path = tmp_path / 'v29_endpoints.csv'
    model._write_csv_rows(path, rows)
    with path.open(encoding='utf-8') as file:
        saved = list(csv.DictReader(file))
    assert float(saved[2]['acquisition_max_legal_novel_target_count']) == rows[2]['acquisition_max_legal_novel_target_count']
    assert summarize_rows(rows)['frames'] == 3
