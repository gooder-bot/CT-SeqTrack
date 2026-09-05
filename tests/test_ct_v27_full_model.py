"""Full constructor/forward/loss CPU audits with only external import shells.

The actual Mini/Seg/FeaturePointNet, Transformer, B1, B2, and B3 all execute.
The unused CUDA PointnetSAModule stub raises if accidentally instantiated.
"""

import copy
import hashlib
import importlib
import json
import random
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data._utils.collate import default_collate

from tests.test_ct_v27_input_flow import sampler_runtime, _case
from utils.config import load_yaml_config
from utils.v27_input import build_v27_eval_input


@pytest.fixture
def full_model_runtime(sampler_runtime, monkeypatch):
    previous_modules = set(sys.modules)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    real_lightning = importlib.util.find_spec('pytorch_lightning') is not None
    lightning = types.ModuleType('pytorch_lightning')

    class LightningModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.current_epoch = self.global_step = 0
            self._trainer = None
        @property
        def device(self):
            return next(self.parameters(), torch.zeros(())).device
        def save_hyperparameters(self, *args, **kwargs):
            pass
        def log(self, *args, **kwargs):
            pass

    lightning.LightningModule = LightningModule
    if not real_lightning:
        monkeypatch.setitem(sys.modules, 'pytorch_lightning', lightning)
    metrics = types.ModuleType('torchmetrics')
    metrics.__path__ = []

    class Metric(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
        def add_state(self, name, default, **kwargs):
            if torch.is_tensor(default):
                self.register_buffer(name, default.clone())
            else:
                setattr(self, name, copy.deepcopy(default))
        def reset(self):
            pass

    class Accuracy(Metric):
        def forward(self, prediction, target):
            prediction = prediction.argmax(1) if prediction.ndim > target.ndim else prediction
            return torch.stack([((prediction == target) & (target == cls)).sum()
                                / (target == cls).sum().clamp_min(1) for cls in (0, 1)])

    metrics.Metric, metrics.Accuracy = Metric, Accuracy
    utilities = types.ModuleType('torchmetrics.utilities')
    utilities.__path__ = []
    data = types.ModuleType('torchmetrics.utilities.data')
    data.dim_zero_cat = lambda rows: torch.cat([value.reshape(-1) for value in rows])
    utilities.data, metrics.utilities = data, utilities
    if not real_lightning:
        for name, module in [('torchmetrics', metrics), ('torchmetrics.utilities', utilities),
                             ('torchmetrics.utilities.data', data)]:
            monkeypatch.setitem(sys.modules, name, module)
    cuda_import = types.ModuleType('pointnet2.utils.pointnet2_modules')

    class UnusedPointnetSAModule:
        def __init__(self, *args, **kwargs):
            raise AssertionError('v27 SeqTrack backbone must not instantiate PointnetSAModule')

    cuda_import.PointnetSAModule = UnusedPointnetSAModule
    monkeypatch.setitem(sys.modules, 'pointnet2.utils.pointnet2_modules', cuda_import)
    seqtrack = importlib.import_module('models.seqtrack3d')
    ctseqtrack = importlib.import_module('models.ctseqtrack')
    yield sampler_runtime, seqtrack.SEQTRACK3D, ctseqtrack.CTSEQTRACK
    for name in list(sys.modules):
        if name not in previous_modules and (
                name.startswith('models.') or name in ('utils.metrics', 'utils.waymo_metrics')):
            sys.modules.pop(name, None)
    torch.set_num_threads(previous_threads)


ARMS = ('b0', 'b1_gru', 'b1_cfc', 'full_minus_b3', 'full', 'reference')


def _construct(runtime, arm):
    sampler_runtime, reference_class, ct_class = runtime
    root = Path(__file__).resolve().parents[1]
    path = ('cfgs/27_seqtrack_reference.yaml' if arm == 'reference'
            else f'cfgs/ct_seqtrack/27_{arm}.yaml')
    config = sampler_runtime[3](load_yaml_config(root / path))
    torch.manual_seed(42)
    model = (reference_class if arm == 'reference' else ct_class)(config)
    return model


def _training_batch(runtime, model, batch_size=2):
    sampler_runtime = runtime[0]
    variant = 'b0' if not model.use_b1motion_v3 else ('full' if model.ct_enable_b3
               else 'full_minus_b3' if model.ct_enable_b2 else 'b1')
    sampler, _, sequence, state, payload, _, _ = _case(sampler_runtime, variant)
    if model.use_b1motion_v3:
        payload['motion_prediction'] = model.predict_motion_prepass(
            sequence, 8, state.results_bbs, recursive_state=state)
    else:
        payload.pop('motion_prediction', None)
    sampling_config = copy.deepcopy(model.config)
    if model.use_b1motion_v3:
        sampling_config.candidate_trajectory_mode = 'shared_se2'
    else:
        payload.pop('online_recursive_state', None)
    if sampling_config.candidate_trajectory_mode == 'independent':
        payload.pop('candidate_shared_transform', None)
    row = sampler.motion_processing_mf(payload, sampling_config)
    if model.use_b1motion_v3:
        # The production _process_online_raw adds these after the pure sampler.
        row['ct_recursive_state_age'] = np.float32(state.rollout_age(8))
        row['ct_recursive_state_age_valid'] = np.float32(1.)
    return default_collate([row for _ in range(batch_size)]), sequence, state


@pytest.mark.parametrize('arm', ARMS)
def test_complete_v27_constructor_forward_loss_and_optimizer_registration(full_model_runtime, arm):
    model = _construct(full_model_runtime, arm)
    batch, _, _ = _training_batch(full_model_runtime, model)
    model.train()
    output = model(batch)
    losses = model.compute_loss(batch, output)
    assert torch.isfinite(losses['loss_total'])
    assert output['aux_estimation_boxes'].shape == (2, 4)
    assert output['pred_bc'].shape == (2, 4096, 9)
    losses['loss_total'].backward()
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()
               if parameter.grad is not None)
    prefixes = ('seg_pointnet.', 'physical_motion_encoder.', 'ct_joint_search_refiner.', 'ct_joint_router.')
    enabled = (True, model.use_b1motion_v3, model.use_ct_joint_full and model.ct_enable_b2,
               model.use_ct_joint_full and model.ct_enable_b3)
    for prefix, present in zip(prefixes, enabled):
        parameters = [parameter for name, parameter in model.named_parameters() if name.startswith(prefix)]
        assert bool(parameters) == bool(present), prefix
        if present:
            assert any(parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
                       for parameter in parameters), prefix
    if model.use_ct_joint_full and model.ct_enable_b2:
        buffers = dict(model.named_buffers())
        for population in ('targetness', 'relation'):
            for field in ('positive_points', 'negative_points', 'positive_weight', 'negative_weight'):
                assert f'ct_{population}_running_{field}' in buffers
        assert output['ct_relation_logits_prepool'].shape == (2, 768)
        assert output['ct_search_targetness_logits'].shape == (2, 256)
    configuration = model.configure_optimizers()
    optimizer = configuration['optimizer']
    optimized = [parameter for group in optimizer.param_groups for parameter in group['params']]
    assert len({id(parameter) for parameter in optimized}) == len(optimized)
    assert {id(parameter) for parameter in optimized} == {id(parameter) for parameter in model.parameters()}


def test_full_constructor_keeps_b0_and_shared_b1_initialization_and_caller_rng(full_model_runtime):
    common = None
    expected_rng = None
    shared_motion = None
    for arm in ARMS:
        model = _construct(full_model_runtime, arm)
        digest = hashlib.sha256()
        for name, parameter in model.named_parameters():
            if name.startswith(('seg_pointnet.', 'mini_pointnet.', 'motion_mlp.',
                                'motion_state_mlp.', 'feature_pointnet.', 'Transformer.')):
                digest.update(name.encode())
                digest.update(parameter.detach().numpy().tobytes())
        rng = torch.get_rng_state().clone()
        if common is None:
            common, expected_rng = digest.hexdigest(), rng
        else:
            assert digest.hexdigest() == common, arm
            assert torch.equal(rng, expected_rng), arm
        if arm in ('b1_gru', 'b1_cfc'):
            values = dict(model.physical_motion_encoder.named_parameters())
            if shared_motion is None:
                shared_motion = {name: parameter.detach().clone() for name, parameter in values.items()}
            else:
                for name in values.keys() & shared_motion.keys():
                    assert torch.equal(values[name], shared_motion[name]), name


@pytest.mark.parametrize('arm', ARMS)
@pytest.mark.parametrize('point_count', [0, 1, 2])
def test_complete_model_sparse_eval_and_no_current_gt_forward(full_model_runtime, arm, point_count):
    model = _construct(full_model_runtime, arm).eval()
    _, _, sequence, state, _, _, _ = _case(full_model_runtime[0], 'b0')
    sequence[8] = copy.deepcopy(sequence[8])
    xyz = np.repeat(state.results_bbs[-1].center[:, None], point_count, axis=1)
    sequence[8]['pc'] = full_model_runtime[0][1].PointCloud(xyz)
    batch, _ = build_v27_eval_input(model, sequence, 8, state.results_bbs, recursive_state=state)
    with torch.no_grad():
        output = model(batch)
    assert output['aux_estimation_boxes'].shape == (1, 4)
    assert torch.isfinite(output['aux_estimation_boxes']).all()
    if point_count == 0:
        torch.testing.assert_close(output['aux_estimation_boxes'], torch.zeros(1, 4))
    assert batch['b0_unique_mask'][0, -1].sum().item() == point_count
    changed_sequence = list(sequence)
    changed_sequence[8] = copy.deepcopy(sequence[8])
    changed_sequence[8]['3d_bbox'].center += np.asarray([20., -20., 0.])
    changed, _ = build_v27_eval_input(model, changed_sequence, 8, state.results_bbs, recursive_state=state)
    with torch.no_grad():
        poisoned = model(changed)
    torch.testing.assert_close(output['aux_estimation_boxes'], poisoned['aux_estimation_boxes'], rtol=0, atol=0)


def test_single_mechanism_batch_with_actual_bn_isolation_and_backward(full_model_runtime):
    model = _construct(full_model_runtime, 'full').train()
    batch, _, _ = _training_batch(full_model_runtime, model, batch_size=1)
    modes = {name: module.training for name, module in model.named_modules()}
    bn = {name: buffer.detach().clone() for name, buffer in model.named_buffers()
          if name.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}
    rng = torch.get_rng_state().clone()
    output = model._forward_safe_mechanism(batch)
    losses = model.compute_loss(batch, output)
    losses['loss_plugin_transaction'].backward()
    assert torch.isfinite(losses['loss_plugin_transaction'])
    assert any(parameter.grad is not None for parameter in model.ct_joint_search_refiner.parameters())
    assert any(parameter.grad is not None for parameter in model.physical_motion_encoder.parameters())
    assert all(parameter.grad is None for name, parameter in model.named_parameters()
               if name.startswith(('seg_pointnet.', 'mini_pointnet.', 'feature_pointnet.', 'Transformer.')))
    assert modes == {name: module.training for name, module in model.named_modules()}
    for name, buffer in model.named_buffers():
        if name in bn:
            assert torch.equal(buffer, bn[name]), name
    assert torch.equal(rng, torch.get_rng_state())


def test_actual_lightning_full_dual_stream_training_and_epoch_checkpoint(full_model_runtime, tmp_path):
    import pytorch_lightning as pl
    if not hasattr(pl, 'Trainer'):
        pytest.skip('real Lightning required for Trainer integration')
    from pytorch_lightning.callbacks import Callback, ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger
    from torch.utils.data import DataLoader
    from models.ct_variant import configure_ct_variant

    model = _construct(full_model_runtime, 'full')
    np.random.seed(42)
    random.seed(42)
    sampler_module, _, sequence, _, payload, _, _ = _case(full_model_runtime[0], 'full')

    class SequenceDataset:
        hist_num = 3
        def get_frames(self, tracklet_id, frame_ids):
            return [sequence[index] for index in frame_ids]
        def get_num_frames_tracklet(self, tracklet_id):
            return len(sequence)
        def get_num_tracklets(self):
            return 1
        def get_num_frames_total(self):
            return len(sequence)
        def get_tracklet_key(self, tracklet_id):
            return 'test/track'

    mechanism_config = copy.deepcopy(model.config)
    mechanism_config.candidate_trajectory_mode = 'shared_se2'
    mechanism_config.num_candidates = 1
    mechanism_config.ct_recursive_candidate_views = 1
    mechanism_config.ct_b0_candidate_views = 1
    mechanism_config.ct_b0_candidate_weights = [1.]
    mechanism_sampler = sampler_module.MotionTrackingSamplerMF(SequenceDataset(), config=mechanism_config)
    observation_config = copy.deepcopy(model.config)
    observation_config.ct_variant = 'b0'
    configure_ct_variant(observation_config)
    observation_config.ct_online_recursive_training = False
    observation_config.ct_observation_payload_mode = 'seqtrack_core'
    observation_rows = []
    for view in range(4):
        observation = dict(payload)
        for key in ('online_recursive_state', 'online_motion_aux_state', 'motion_prediction',
                    'candidate_shared_transform'):
            observation.pop(key, None)
        observation.update(candidate_id=view, b0_view_id=view, ct_observation_only=True)
        observation_rows.append(sampler_module.motion_processing_mf(observation, observation_config))
    observation_batch = default_collate(observation_rows)
    steps = []
    for step, frame_ids in enumerate(((1, 2), (3, 4))):
        ticks = [[mechanism_sampler._online_raw_view(0, step, 0, 0, frame_id, 0)]
                 for frame_id in frame_ids]
        steps.append(dict(ct_stream_schema='ct_seqtrack.train.v4',
                          observation=observation_batch,
                          mechanism={'ct_mechanism_sequence': ticks}))

    class AuditCallback(Callback):
        def __init__(self):
            self.gradient_rows = []
            self.last_batch_boundary = None
        def on_before_optimizer_step(self, trainer, module, optimizer):
            row = {}
            for group in optimizer.param_groups:
                grads = [parameter.grad for parameter in group['params'] if parameter.grad is not None]
                assert all(torch.isfinite(value).all() for value in grads)
                row[group['name']] = sum(float(value.detach().abs().sum()) for value in grads)
            self.gradient_rows.append(row)
        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            if trainer.fit_loop.epoch_loop.batch_progress.is_last_batch:
                metadata = {}
                module.on_save_checkpoint(metadata)
                self.last_batch_boundary = metadata['ct_epoch_boundary_complete']

    audit = AuditCallback()
    checkpoint = ModelCheckpoint(dirpath=tmp_path / 'checkpoints', save_last=True,
                                 save_top_k=0, every_n_epochs=1, save_on_train_epoch_end=True)
    logger = TensorBoardLogger(save_dir=str(tmp_path), name='full_training')
    trainer = pl.Trainer(accelerator='cpu', devices=1, max_epochs=1,
        limit_train_batches=2, limit_val_batches=0, num_sanity_val_steps=0,
        logger=logger, callbacks=[audit, checkpoint], enable_progress_bar=False,
        enable_model_summary=False, log_every_n_steps=1)
    trainer.fit(model, train_dataloaders=DataLoader(steps, batch_size=None, num_workers=0))
    assert trainer.global_step == 2
    assert len(audit.gradient_rows) == 2
    for group in ('b0', 'b1', 'b2', 'b3'):
        assert any(row[group] > 0 for row in audit.gradient_rows), (
            group, audit.gradient_rows,
            [(frame_id, box.center.tolist()) for frame_id, box in
             model._ct_recursive_states[(0, 'test/track')].predictions.items()])
        assert int(getattr(model, f'ct_{group}_update_step')) == 2
    assert audit.last_batch_boundary is False
    assert model._ct_epoch_boundary_complete is True
    saved = torch.load(checkpoint.last_model_path, map_location='cpu')
    assert saved['ct_epoch_boundary_complete'] is True
    state = model._ct_recursive_states[(0, 'test/track')]
    assert len(state.results_bbs) == 5
    assert set(state.quality) == {1, 2, 3, 4}
    assert all(np.isfinite(value).all() for value in state.quality.values())
    assert state.quality[1][-1] == 1.
    assert model._ct_epoch_binary_rows['presence']
    assert model._ct_epoch_binary_rows['bounded_utility']
    assert model._ct_epoch_binary_rows['help']
    supply = json.loads((Path(logger.log_dir) / 'acquisition_supply/epoch_01.json').read_text())
    assert supply['schema'] == 'ct_seqtrack.acquisition_training_supply.v1'
    assert list(Path(logger.log_dir).glob('events.out.tfevents.*'))
