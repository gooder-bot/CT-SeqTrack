"""Review gap: the real roll-in plus recursive Full transaction shares one B0 update."""

import copy
from types import SimpleNamespace

import numpy as np
import torch

from tests.test_ct_v27_input_flow import sampler_runtime, _case
from tests.test_ct_v27_full_model import full_model_runtime
from tests.test_ct_v28_audit_host import _attach_training_shell
from tests.test_ct_v29_b0_host import construct, b0_buffers
from tests.test_ct_v29_rollin import setup_case
from utils.training_isolation import capture_global_rng_state, restore_global_rng_state
from utils.v28_numerical_audit import compare_values, snapshot
from utils.v29_rollin import observation_collate, process_query


def test_actual_rollin_and_two_recursive_ticks_preserve_three_arm_b0_update(
        full_model_runtime, monkeypatch):
    runtime = full_model_runtime[0]
    config, original = setup_case(runtime)
    teachers = {i: copy.deepcopy(original['frames'][i - original['start']]['3d_bbox'])
                for i in range(original['start'], original['endpoint'])}
    teacher, _ = process_query(original, teachers, original['endpoint'], config)
    teacher['candidate_id'] = np.int64(0)
    items = [dict(mode='teacher', sample=teacher)]
    for candidate in (1, 2, 3):
        items.append(dict(original, candidate=candidate, index=4 + candidate))
    observation = observation_collate(items)
    sampler, _, sequence, _, _, _, _ = _case(runtime, 'full')
    dataset = SimpleNamespace(
        hist_num=3, get_num_tracklets=lambda: 1,
        get_num_frames_total=lambda: len(sequence),
        get_num_frames_tracklet=lambda index: len(sequence),
        get_frames=lambda index, frame_ids: [sequence[i] for i in frame_ids],
        get_tracklet_key=lambda index: 'test/track')
    initial_rng = capture_global_rng_state()
    expected = None
    for arm in ('b0', 'full_cfc', 'full_gru'):
        model = construct(full_model_runtime, arm).train()
        optimizer = model.configure_optimizers()['optimizer']
        _attach_training_shell(model, optimizer, monkeypatch)
        mechanism = None
        if arm != 'b0':
            mechanism_config = copy.deepcopy(model.config)
            mechanism_config.candidate_trajectory_mode = 'shared_se2'
            mechanism_config.num_candidates = mechanism_config.ct_recursive_candidate_views = 1
            mechanism_config.ct_b0_candidate_views = 1
            mechanism_config.ct_b0_candidate_weights = [1.]
            raw_sampler = sampler.MotionTrackingSamplerMF(dataset, config=mechanism_config)
            mechanism = {'ct_mechanism_sequence': [
                [raw_sampler._online_raw_view(0, 0, 0, 0, frame, 0)] for frame in (1, 2)]}
        restore_global_rng_state(initial_rng)
        batch = dict(ct_stream_schema='ct_seqtrack.train.v4', observation=observation,
                     mechanism=mechanism)
        optimizer.zero_grad(set_to_none=True)
        loss = model.training_step(batch, 0)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        model.on_before_optimizer_step(optimizer)
        optimizer.step()
        model.on_train_batch_end(loss, batch, 0)
        assert model._ct_v29_rollin_diagnostics['sample_forwards'] == 9
        assert model._ct_v29_rollin_diagnostics['teacher_rows'] == 1
        assert int(model.ct_b0_update_step) == 1
        if mechanism is not None:
            assert set(model._ct_recursive_states[(0, 'test/track')].predictions) == {0, 1, 2}
            assert not model._ct_mechanism_transaction
            assert not model._ct_safe_mechanism_forward
        b0 = [(name, value) for name, value in model.named_parameters()
              if not model._ct_any_plugin_parameter(name)]
        record = snapshot(dict(
            parameters=dict(b0), gradients={name: p.grad for name, p in b0},
            adam={name: optimizer.state[p] for name, p in b0},
            buffers=b0_buffers(model), rng=capture_global_rng_state(),
            observation_fingerprints=model._ct_observation_batch_fingerprints))
        if expected is None:
            expected = record
        else:
            assert compare_values(expected, record) is None, (arm, compare_values(expected, record))
