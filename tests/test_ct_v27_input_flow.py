"""Real sampler CPU tests; only unavailable package import shells are stubbed."""

import copy
import importlib
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch
from pyquaternion import Quaternion

from models.ct_variant import configure_ct_variant
from utils.b1_acquisition import build_b1_input_arrays
from utils.config import load_yaml_config
from utils.recursive_state import RecursiveTrackState, build_recursive_input_contract
from utils.v27_input import build_v27_eval_input


@pytest.fixture
def sampler_runtime(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    prior_dataset_modules = {key: value for key, value in sys.modules.items()
                             if key == 'datasets' or key.startswith('datasets.')}
    package = types.ModuleType('datasets')
    package.__path__ = [str(root / 'datasets')]
    monkeypatch.setitem(sys.modules, 'datasets', package)

    class EasyDict(dict):
        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError as error:
                raise AttributeError(key) from error
        __setattr__ = dict.__setitem__

    easy = types.ModuleType('easydict')
    easy.EasyDict = EasyDict
    monkeypatch.setitem(sys.modules, 'easydict', easy)
    nu = types.ModuleType('nuscenes')
    nu.__path__ = []
    utilities = types.ModuleType('nuscenes.utils')
    utilities.__path__ = []
    geometry = types.ModuleType('nuscenes.utils.geometry_utils')
    def points_in_box(box, points, wlh_factor=1.0):
        local = box.rotation_matrix.T @ (np.asarray(points)[:3] - box.center[:, None])
        half = np.asarray(box.wlh)[[1, 0, 2]] * (.5 * wlh_factor)
        return (np.abs(local) <= half[:, None]).all(0)
    geometry.points_in_box = points_in_box
    utilities.geometry_utils = geometry
    nu.utils = utilities
    for name, module in [('nuscenes', nu), ('nuscenes.utils', utilities),
                         ('nuscenes.utils.geometry_utils', geometry)]:
        monkeypatch.setitem(sys.modules, name, module)
    sampler = importlib.import_module('datasets.sampler')
    classes = importlib.import_module('datasets.data_classes')
    misc = importlib.import_module('datasets.misc_utils')
    yield sampler, classes, misc, EasyDict
    for key in list(sys.modules):
        if key == 'datasets' or key.startswith('datasets.'):
            if key not in prior_dataset_modules:
                sys.modules.pop(key, None)
    sys.modules.update(prior_dataset_modules)


def _case(runtime, variant='full'):
    sampler, classes, misc, EasyDict = runtime
    root = Path(__file__).resolve().parents[1]
    config = EasyDict(load_yaml_config(root / 'cfgs/ct_seqtrack/27_full.yaml'))
    config.ct_variant = variant
    configure_ct_variant(config)
    config.candidate_trajectory_mode = 'shared_se2'
    config.ct_observation_payload_mode = 'legacy'
    sequence = []
    rng = np.random.default_rng(14)
    for index in range(9):
        box = classes.Box([index * .4, 0., 0.], [2., 4., 2.],
                          Quaternion(axis=[0, 0, 1], radians=.1))
        xyz = rng.uniform([-9., -6., -.9], [12., 6., .9], (3000, 3)).T
        sequence.append(dict(pc=classes.PointCloud(xyz), **{'3d_bbox': box},
                             timestamp=int(1_500_000_000_000_000 + index * 500_000),
                             frame_id=index, tracklet_key='test/track'))
    state = RecursiveTrackState(0, 'test/track', sequence[0]['3d_bbox'],
                                timestamps={0: sequence[0]['timestamp']})
    for index in range(1, 8):
        prediction = copy.deepcopy(sequence[index]['3d_bbox'])
        prediction.center[0] += .2
        state.append(index, prediction, sequence[index]['timestamp'],
                     quality=[100. + index, .4, .2, 1.])
    contract = build_recursive_input_contract(state, 8, 3, config)
    prediction = dict(mu_xy=np.asarray((.4, 0.), dtype=np.float32),
                      velocity_xy=np.asarray((.8, 0.), dtype=np.float32),
                      direction_xy=np.asarray((1., 0.), dtype=np.float32),
                      log_sigma_parallel_perp=np.asarray((-.5, -.8)),
                      acquisition_margin_parallel_perp=np.asarray((3., 2.)),
                      valid=True, source_id=1, gap_ratio=1., current_delta_t=.5)
    payload = dict(first_frame=sequence[0], this_frame=sequence[8],
                   prev_frames=misc.create_history_frame_dict([sequence[i] for i in contract['history_frame_ids']]),
                   online_recursive_state=contract, candidate_id=0,
                   valid_mask=contract['history_valid_mask'].tolist(),
                   prev_frame_ids=contract['history_frame_ids'], this_frame_id=8,
                   history_offsets=contract['history_offsets'], tracklet_key=state.tracklet_key,
                   sample_index=8, candidate_shared_transform=contract['candidate_shared_transform'],
                   point_sampling_seeds=contract['point_sampling_seeds'],
                   current_sampling_seed=contract['current_sampling_seed'], motion_prediction=prediction)
    auxiliary = build_recursive_input_contract(state, 8, 3, config, offsets=[2, 4, 6])
    payload.update(online_motion_aux_state=auxiliary,
                   motion_aux_offsets=[2, 4, 6],
                   motion_aux_frame_ids=auxiliary['history_frame_ids'],
                   motion_aux_valid_mask=auxiliary['history_valid_mask'].tolist(),
                   motion_aux_prev_frames=misc.create_history_frame_dict(
                       [sequence[i] for i in auxiliary['history_frame_ids']]))
    host = types.SimpleNamespace(config=config, device=torch.device('cpu'),
                                 ct_enable_b1=variant != 'b0')
    host.predict_motion_prepass = lambda *args, **kwargs: prediction
    return sampler, config, sequence, state, payload, host, prediction


def test_train_eval_share_causal_tensors_and_auxiliary_quality(sampler_runtime):
    sampler, config, sequence, state, payload, host, prediction = _case(sampler_runtime)
    training_sidecar = {}
    payload['_ct_diagnostic_sidecar'] = training_sidecar
    train = sampler.motion_processing_mf(payload, config)
    eval_sidecar = {}
    evaluation, _ = build_v27_eval_input(host, sequence, 8, state.results_bbs,
                                        recursive_state=state, motion_prediction=prediction,
                                        diagnostic_sidecar=eval_sidecar)
    for key in ('points', 'ref_boxs', 'motion_main_ref_boxs', 'motion_main_delta_t',
                'motion_acquisition_features', 'ct_extension_points', 'ct_extension_point_ids',
                'ct_acquisition_margin', 'ct_acquisition_direction_xy'):
        np.testing.assert_allclose(train[key], evaluation[key][0].numpy(), atol=1e-6, rtol=1e-6)
    assert train['motion_aux_acquisition_features'].shape == (17,)
    assert 'motion_aux_acquisition_features' not in evaluation
    aux = payload['online_motion_aux_state']
    expected = build_b1_input_arrays(
        state.history_boxes(aux['history_frame_ids'], aux['history_valid_mask']),
        [1., 1., 1.], aux['history_valid_mask'], history_quality=aux['history_quality'],
        recursive_age=aux['recursive_age'], first_frame_wlh=state.target_size)
    np.testing.assert_allclose(train['motion_aux_acquisition_features'], expected['acquisition_features'])
    assert training_sidecar['acquisition']['acquisition_schema_version'] == 'ct_acquisition.v4'
    assert eval_sidecar['acquisition']['global_target_count_exact'] == training_sidecar['acquisition']['global_target_count_exact']


@pytest.mark.parametrize('variant', ['full', 'b0'])
def test_current_gt_changes_only_labels_not_eval_inputs(sampler_runtime, variant):
    _, _, sequence, state, _, host, prediction = _case(sampler_runtime, variant)
    original, _ = build_v27_eval_input(host, sequence, 8, state.results_bbs,
                                      recursive_state=state, motion_prediction=prediction)
    changed_sequence = list(sequence)
    changed_sequence[8] = copy.deepcopy(sequence[8])
    changed_sequence[8]['3d_bbox'].center += np.asarray((18., -14., 0.))
    changed_sequence[8]['3d_bbox'].wlh *= 1.4
    changed, _ = build_v27_eval_input(host, changed_sequence, 8, state.results_bbs,
                                     recursive_state=state, motion_prediction=prediction)
    causal_keys = ['points', 'ref_boxs', 'bbox_size', 'candidate_bc', 'b0_point_ids',
                   'b0_point_valid_mask', 'delta_t_effective', 'coordinate_anchor']
    if variant == 'full':
        causal_keys += ['motion_main_ref_boxs', 'motion_acquisition_features',
                        'ct_extension_points', 'ct_extension_point_ids',
                        'ct_acquisition_margin', 'ct_acquisition_direction_xy',
                        'ct_search_support_valid']
    for key in causal_keys:
        if key in original:
            assert torch.equal(original[key], changed[key]), key
    assert not torch.equal(original['box_label'], changed['box_label'])


def test_invalid_learned_prior_records_actual_cv_margin(sampler_runtime):
    _, _, sequence, state, _, host, prediction = _case(sampler_runtime)
    invalid = dict(prediction, valid=False, acquisition_margin_parallel_perp=np.asarray((5., 2.5)))
    batch, _ = build_v27_eval_input(host, sequence, 8, state.results_bbs,
                                   recursive_state=state, motion_prediction=invalid)
    np.testing.assert_array_equal(batch['ct_acquisition_margin'][0], [2., 1.])
    assert batch['search_v3_prior_source_id'][0] == 2
    assert batch['ct_acquisition_learned_valid'][0] == 0
    assert batch['ct_acquisition_resolved_valid'][0] == 1


def test_nonfinite_learned_prior_cannot_contaminate_resolved_cv_statistics(sampler_runtime):
    _, _, sequence, state, _, host, prediction = _case(sampler_runtime)
    invalid = dict(prediction, valid=False, mu_xy=np.full(2, np.nan),
        direction_xy=np.full(2, np.nan), log_sigma_parallel_perp=np.full(2, np.nan),
        acquisition_margin_parallel_perp=np.full(2, np.nan))
    batch, _ = build_v27_eval_input(host, sequence, 8, state.results_bbs,
                                   recursive_state=state, motion_prediction=invalid)
    for key in ('search_v3_support_anchor_xy', 'ct_acquisition_margin',
                'ct_acquisition_direction_xy', 'ct_acquisition_statistical_direction_xy',
                'ct_acquisition_log_sigma'):
        assert torch.isfinite(batch[key]).all(), key
    assert batch['search_v3_prior_source_id'].item() == 2
    torch.testing.assert_close(batch['ct_acquisition_statistical_direction_xy'],
                               batch['ct_acquisition_direction_xy'])


def test_pruned_observation_keeps_canonical_identity_and_quality_contract(sampler_runtime):
    sampler, config, _, _, payload, _, _ = _case(sampler_runtime, 'b0')
    config.ct_observation_payload_mode = 'seqtrack_core'
    row = sampler.motion_processing_mf(payload, config)
    for key in ('b0_point_ids', 'b0_valid_mask', 'b0_unique_mask',
                'b0_raw_point_count', 'ct_current_observation_valid'):
        assert key in row, key


def test_b1_only_keeps_acquisition_targets_diagnostics_and_margin_gradients(sampler_runtime):
    from models.ct_v2.motion import OrderedPhysicalMotionEncoder, acquisition_margin_target_loss
    sampler, config, _, _, payload, _, _ = _case(sampler_runtime, 'b1')
    sidecar = {}
    payload['_ct_diagnostic_sidecar'] = sidecar
    row = sampler.motion_processing_mf(payload, config)
    assert not config.ct_enable_b2 and config.ct_enable_b1
    assert row['motion_acquisition_target_valid'] == 1.
    assert sidecar['acquisition']['acquisition_schema_version'] == 'ct_acquisition.v4'
    assert 'global_novel_target_count' in sidecar['acquisition']
    encoder = OrderedPhysicalMotionEncoder(enable_v27=True, adaptive_acquisition_margin=True)
    tensor = lambda key: torch.as_tensor(row[key]).unsqueeze(0)
    output = encoder(tensor('motion_main_ref_boxs'), tensor('motion_main_delta_t'),
        tensor('motion_main_valid_mask'), tensor('motion_main_current_delta_t'),
        acquisition_features=tensor('motion_acquisition_features'))
    loss = acquisition_margin_target_loss(output['acquisition_margin_parallel_perp'],
        tensor('motion_acquisition_target'), tensor('motion_acquisition_target_valid'))['loss_per_sample'].mean()
    loss.backward()
    assert encoder.acquisition_margin_head[-1].bias.grad.abs().sum() > 0
    np.testing.assert_array_equal(row['ct_acquisition_margin'], [3., 2.])


def test_empty_global_frame_has_no_fabricated_point_ids_or_diagnostic_counts(sampler_runtime):
    _, _, sequence, state, _, host, prediction = _case(sampler_runtime)
    sequence[8] = copy.deepcopy(sequence[8])
    sequence[8]['pc'] = sampler_runtime[1].PointCloud(np.empty((3, 0)))
    sidecar = {}
    batch, _ = build_v27_eval_input(host, sequence, 8, state.results_bbs,
        recursive_state=state, motion_prediction=prediction, diagnostic_sidecar=sidecar)
    assert batch['ct_current_observation_valid'].item() == 0
    assert batch['b0_valid_mask'][0, -1].sum().item() == 0
    assert torch.all(batch['b0_point_ids'][0, -1] == -1)
    assert torch.all(batch['ct_extension_point_ids'] == -1)
    assert sidecar['acquisition']['global_target_count_exact'] == 0
    assert sidecar['acquisition']['base_raw_target_count'] == 0
    assert sidecar['acquisition']['pool_target_count'] == 0


@pytest.mark.parametrize('empty', [False, True])
def test_b0_sidecar_reports_exact_raw_and_sampled_funnel_without_changing_labels(sampler_runtime, empty):
    _, _, sequence, state, _, host, _ = _case(sampler_runtime, 'b0')
    sequence[8] = copy.deepcopy(sequence[8])
    current_box = sequence[8]['3d_bbox']
    local = np.asarray([[0., 0., 0.], [1.9, 0., 0.], [2.1, 0., 0.], [20., 0., 0.]])
    world = current_box.rotation_matrix @ local.T + current_box.center[:, None]
    sequence[8]['pc'] = sampler_runtime[1].PointCloud(world[:, :0] if empty else world)
    plain, _ = build_v27_eval_input(host, sequence, 8, state.results_bbs, recursive_state=state)
    sidecar = {}
    logged, _ = build_v27_eval_input(host, sequence, 8, state.results_bbs,
                                   recursive_state=state, diagnostic_sidecar=sidecar)
    for key in ('points', 'seg_label', 'b0_point_ids', 'b0_valid_mask', 'candidate_bc'):
        assert torch.equal(plain[key], logged[key]), key
    diagnostic = sidecar['acquisition']
    assert diagnostic['acquisition_schema_version'] == 'ct_acquisition.v4'
    assert diagnostic['acquisition_enabled'] is False
    assert diagnostic['acquisition_disabled_reason'] == 'modules_disabled'
    assert diagnostic['global_raw_point_count'] == (0 if empty else 4)
    assert diagnostic['global_target_count_exact'] == (0 if empty else 2)
    assert diagnostic['base_raw_point_count'] == (0 if empty else 3)
    assert diagnostic['base_raw_target_count'] == (0 if empty else 2)
    assert diagnostic['base_sampled_point_count'] == (0 if empty else 3)
    assert diagnostic['base_sampled_target_count'] == (0 if empty else 2)
    for key in ('support_union_raw_point_count', 'support_novel_point_count',
                'extension_pool_count', 'prepool_point_count', 'sampled_count',
                'selected_point_count', 'selected_target_count'):
        assert diagnostic[key] == 0, key
    if not empty:
        # The point at local x=2.1 is outside exact GT but inside B0 scale=1.25.
        current_labels = logged['seg_label'].reshape(1, 4, 1024)[0, -1]
        assert current_labels.sum().item() == 1024
