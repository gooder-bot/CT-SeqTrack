"""Read-only production-path probes; synthetic inputs, no nuScenes/CUDA claim."""
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

from tests.test_ct_v27_input_flow import sampler_runtime
from tests.test_ct_v27_full_model import full_model_runtime
from tests.test_ct_v29_rollin import setup_case
from tests.test_ct_v29_b0_host import construct, observation_routing
from utils.v29_rollin import process_query


DEST = Path(__file__).parent


def record(name, value):
    (DEST / (name + '.json')).write_text(json.dumps(value, indent=2), encoding='utf-8')
    print(name, json.dumps(value))


def test_empty_current_supervision(sampler_runtime):
    config, item = setup_case(sampler_runtime)
    item = copy.deepcopy(item)
    item['frames'][-1]['pc'] = sampler_runtime[1].PointCloud(np.empty((3, 0)))
    predictions = {i: copy.deepcopy(item['frames'][i-item['start']]['3d_bbox'])
                   for i in range(item['start'], item['endpoint'])}
    row, _ = process_query(item, predictions, item['endpoint'], config)
    current_labels = row['seg_label'][-1024:]
    record('empty_current_supervision', dict(
        raw_current_count=int(row['b0_raw_point_count'][-1]),
        valid_current_slots=int(row['b0_point_valid_mask'][-1].sum()),
        foreground_supervision_slots=int(current_labels.sum()),
        box_distance_supervision_mean=float(row['this_bc'].mean()),
        current_xyz_all_zero=bool((row['points'][-1024:, :3] == 0).all())))
    assert row['b0_point_valid_mask'][-1].sum() == 0
    assert current_labels.sum() == 1024


def test_invalid_history_influences_segmentation_and_training(full_model_runtime):
    config, item = setup_case(full_model_runtime[0], start=0, endpoint=1)
    predictions = {0: copy.deepcopy(item['frames'][0]['3d_bbox'])}
    row, _ = process_query(item, predictions, 1, config)
    batch = default_collate([row, copy.deepcopy(row)])
    alternate = copy.deepcopy(batch)
    # Slots 1 and 2 denote absent historical frames; masks remain identical.
    alternate['points'][:, 1024:3072, :3] += 20.0
    alternate['candidate_bc'][:, 1024:3072] += 5.0
    model = construct(full_model_runtime, 'b0').eval()
    with torch.no_grad(), observation_routing(model):
        original = model(batch)
        changed = model(alternate)
    train_original = copy.deepcopy(model).train()
    train_changed = copy.deepcopy(model).train()
    with torch.no_grad(), observation_routing(train_original), observation_routing(train_changed):
        torch.manual_seed(101)
        original_training = train_original(batch)
        torch.manual_seed(101)
        changed_training = train_changed(alternate)
    record('invalid_history_influence', dict(
        history_mask=batch['valid_mask'][0].tolist(),
        absent_frame_valid_slots=int(batch['b0_point_valid_mask'][:, 1:3].sum()),
        original_absent_xyz_nonzero=int(torch.count_nonzero(batch['points'][:, 1024:3072, :3])),
        current_seg_logit_max_delta=float((original['seg_logits'][:, :, -1024:] -
                                           changed['seg_logits'][:, :, -1024:]).abs().max()),
        observation_box_max_delta=float((original['observation_aux_estimation_boxes'] -
                                         changed['observation_aux_estimation_boxes']).abs().max()),
        training_observation_box_max_delta=float((original_training['observation_aux_estimation_boxes'] -
                                                  changed_training['observation_aux_estimation_boxes']).abs().max())))
    assert not torch.equal(original['seg_logits'][:, :, -1024:],
                           changed['seg_logits'][:, :, -1024:])


def test_empty_current_receives_seg_bc_gradients(full_model_runtime):
    config, item = setup_case(full_model_runtime[0])
    item = copy.deepcopy(item)
    item['frames'][-1]['pc'] = full_model_runtime[0][1].PointCloud(np.empty((3, 0)))
    predictions = {i: copy.deepcopy(item['frames'][i-item['start']]['3d_bbox'])
                   for i in range(item['start'], item['endpoint'])}
    row, _ = process_query(item, predictions, item['endpoint'], config)
    batch = default_collate([row, copy.deepcopy(row)])
    model = construct(full_model_runtime, 'b0').train()
    with observation_routing(model):
        output = model(batch)
        losses = model.compute_loss(batch, output)
    seg_gradient, bc_gradient = torch.autograd.grad(
        losses['loss_b0_transaction'], [output['seg_logits'], output['pred_bc']],
        retain_graph=False)
    record('empty_current_gradients', dict(
        valid_current_slots=int(batch['b0_point_valid_mask'][:, -1].sum()),
        current_seg_gradient_abs_sum=float(seg_gradient[:, :, -1024:].abs().sum()),
        current_bc_gradient_abs_sum=float(bc_gradient[:, -1024:].abs().sum()),
        all_seg_gradient_abs_sum=float(seg_gradient.abs().sum()),
        all_bc_gradient_abs_sum=float(bc_gradient.abs().sum())))
    assert seg_gradient[:, :, -1024:].abs().sum() > 0
    assert bc_gradient[:, -1024:].abs().sum() > 0


def test_parameter_inventory(full_model_runtime):
    report = {}
    for arm in ('b0', 'full_cfc', 'full_gru'):
        model = construct(full_model_runtime, arm)
        groups = dict(b0=0, b1=0, b2=0, b3=0)
        for name, value in model.named_parameters():
            group = model._ct_plugin_group(name) if model._ct_any_plugin_parameter(name) else 'b0'
            groups[group] += value.numel()
        report[arm] = dict(parameters=groups, total=sum(groups.values()),
            config={key: model.config.get(key) for key in (
                'hist_num', 'point_sample_size', 'motion_v3_hidden_dim', 'motion_v3_step_dim',
                'ct_acquisition_margin_min', 'ct_acquisition_margin_max',
                'ct_router_radius_base', 'ct_router_radius_per_second', 'ct_router_radius_max',
                'batch_size', 'epoch', 'lr_decay_step', 'workers', 'default_time_step')})
    record('parameter_inventory', report)
