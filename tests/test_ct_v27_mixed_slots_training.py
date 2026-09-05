"""真实 batch16 混合新轨迹/成熟轨迹，覆盖 raw→collate→训练反传。"""

import copy

import numpy as np
import pytest
import torch
from pyquaternion import Quaternion

from tests.test_ct_v27_full_model import full_model_runtime, _construct
from tests.test_ct_v27_input_flow import sampler_runtime


MARGIN_COUNTS = tuple('motion_margin_' + name for name in (
    'global_novel_target_count', 'max_reachable_target_count',
    'selected_target_count', 'selected_background_count'))


class MixedSequences:
    hist_num = 3

    def __init__(self, classes):
        self.sequences = []
        for tracklet_id in range(16):
            rng = np.random.default_rng(1700 + tracklet_id)
            origin = np.asarray([tracklet_id * 20., tracklet_id * 3., 0.])
            velocity = np.asarray([.32 + .015 * tracklet_id, .025 * (tracklet_id % 3), 0.])
            sequence = []
            for frame_id in range(9):
                center = origin + frame_id * velocity
                box = classes.Box(center, [2., 4., 2.],
                    Quaternion(axis=[0, 0, 1], radians=.01 * tracklet_id))
                xyz = rng.uniform([-9., -6., -.9], [12., 6., .9], (3000, 3)).T
                xyz += origin[:, None]
                sequence.append(dict(pc=classes.PointCloud(xyz), **{'3d_bbox': box},
                    timestamp=1500000000. + frame_id * .5,
                    frame_id=f'mixed/{tracklet_id}/{frame_id}',
                    tracklet_key=self.get_tracklet_key(tracklet_id)))
            self.sequences.append(sequence)

    def get_frames(self, tracklet_id, frame_ids):
        return [self.sequences[tracklet_id][frame_id] for frame_id in frame_ids]

    def get_num_frames_tracklet(self, tracklet_id):
        return len(self.sequences[tracklet_id])

    def get_num_tracklets(self):
        return len(self.sequences)

    def get_num_frames_total(self):
        return sum(map(len, self.sequences))

    def get_tracklet_key(self, tracklet_id):
        return f'test/mixed-track-{tracklet_id}'


@pytest.mark.parametrize('arm', ['full_minus_b3', 'full'])
def test_batch16_mixed_history_slots_collate_and_train(full_model_runtime, monkeypatch, arm):
    model = _construct(full_model_runtime, arm).train()
    sampler_module, classes, _, _ = full_model_runtime[0]
    dataset = MixedSequences(classes)
    config = copy.deepcopy(model.config)
    config.candidate_trajectory_mode = 'shared_se2'
    config.num_candidates = config.ct_recursive_candidate_views = config.ct_b0_candidate_views = 1
    config.ct_b0_candidate_weights = [1.]
    sampler = sampler_module.MotionTrackingSamplerMF(dataset, config=config)

    # Both first-row orders matter: mature-first used to raise KeyError, while
    # frame1-first could silently drop count fields under default_collate.
    frame_ids = [3, 1, 8, 1, 3, 8, 1, 8, 3, 1, 8, 3, 1, 8, 3, 1]
    if arm == 'full_minus_b3':
        frame_ids[0], frame_ids[1] = frame_ids[1], frame_ids[0]
    raw_items = []
    for slot, frame_id in enumerate(frame_ids):
        raw = sampler._online_raw_view(0, 21, slot, slot, frame_id, 0)
        state = model._recursive_state_for_raw(raw)
        for past_id in range(1, frame_id):
            past_frame = dataset.sequences[slot][past_id]
            prediction = copy.deepcopy(past_frame['3d_bbox'])
            prediction.center += np.asarray([.1 + slot * .003, -.03, 0.])
            state.append(past_id, prediction, past_frame['timestamp'],
                         quality=[80. + slot, .6, .2, 1.])
        raw_items.append(raw)
    assert len({raw['tracklet_key'] for raw in raw_items}) == 16
    assert len({id(raw['this_frame']['pc']) for raw in raw_items}) == 16
    assert {raw['this_frame_id'] for raw in raw_items} == {1, 3, 8}

    process_raw = model._process_online_raw
    processed_rows = []

    def record_processed(*args, **kwargs):
        row = process_raw(*args, **kwargs)
        processed_rows.append(row)
        return row

    monkeypatch.setattr(model, '_process_online_raw', record_processed)
    batch, auxiliary = model._prepare_online_recursive_batch(raw_items)
    assert auxiliary is None
    assert len(processed_rows) == 16
    # The production sampler itself supplies one fixed schema; no fill-in or
    # custom collator masks an inconsistent row in this regression test.
    assert all(set(row) == set(processed_rows[0]) for row in processed_rows)
    assert batch['points'].shape == (16, 4096, 5)
    torch.testing.assert_close(batch['ct_online_frame_id'], torch.tensor(frame_ids))
    assert torch.unique(batch['ct_online_slot']).numel() == 16
    assert torch.all(batch['ct_recursive_state_age_valid'] == 1)
    early = batch['ct_online_frame_id'] == 1
    mature = ~early
    assert early.any() and mature.any()
    assert torch.all(batch['ct_search_support_valid'][early] == 0)
    assert torch.all(batch['motion_acquisition_target_valid'][early] == 0)
    assert torch.any(batch['ct_search_support_valid'][mature] > 0)
    for key in MARGIN_COUNTS:
        assert batch[key].shape == (16,)
        assert batch[key].dtype == torch.float32
        assert torch.isfinite(batch[key]).all()
        assert torch.all(batch[key][early] == 0)
    for row in processed_rows:
        assert all(np.asarray(row[key]).shape == () for key in MARGIN_COUNTS)

    output = model._forward_safe_mechanism(batch)
    losses = model.compute_loss(batch, output)
    assert output['ct_relation_logits_prepool'].shape == (16, 768)
    assert output['ct_search_targetness_logits'].shape == (16, 256)
    assert torch.isfinite(output['aux_estimation_boxes']).all()
    assert torch.isfinite(losses['loss_plugin_transaction'])
    losses['loss_plugin_transaction'].backward()
    groups = ['physical_motion_encoder.', 'ct_joint_search_refiner.']
    if arm == 'full':
        groups.append('ct_joint_router.')
    for prefix in groups:
        gradients = [parameter.grad for name, parameter in model.named_parameters()
                     if name.startswith(prefix) and parameter.grad is not None]
        assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients), prefix
        assert any(torch.count_nonzero(gradient) for gradient in gradients), prefix
    assert all(parameter.grad is None for name, parameter in model.named_parameters()
               if not model._ct_any_plugin_parameter(name))
