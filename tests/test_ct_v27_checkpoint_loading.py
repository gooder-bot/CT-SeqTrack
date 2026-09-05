"""Offline checkpoint restore includes private RNG extra-state and policy lifecycle."""
import copy

import pytest
import torch

from models.ct_v2.action_v27 import B3UtilityUpdater
from utils.checkpoint_loading import load_initial_weights
from utils.training_isolation import CheckpointableRNG
from utils.online_contract import build_online_resume_contract, validate_online_resume_contract


class ExportProbe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = {'ct_enable_v27': True, 'ct_router_radius_max': 2.}
        for name in ('seg_pointnet', 'mini_pointnet', 'motion_mlp',
                     'feature_pointnet', 'Transformer', 'physical_motion_encoder'):
            setattr(self, name, torch.nn.Linear(1, 1))
        self.ct_plugin_rng = CheckpointableRNG(101)
        self.ct_memory_control_rng = CheckpointableRNG(202)
        self.ct_joint_router = B3UtilityUpdater(require_calibration=True)


@pytest.mark.parametrize('prefix', ['', 'model.', 'module.'])
def test_complete_export_restore_preserves_real_private_rng_extra_state(prefix, tmp_path):
    original = ExportProbe()
    with original.ct_plugin_rng.fork('cpu'):
        torch.rand(13)
    with original.ct_memory_control_rng.fork('cpu'):
        torch.rand(19)
    original.ct_joint_router.install_policy({'kind': 'always'})
    state = copy.deepcopy(original.state_dict())
    assert isinstance(state['ct_plugin_rng._extra_state'], dict)
    assert not any(key.endswith(('decision_threshold', 'calibrated')) for key in state)
    path = tmp_path / 'full.ckpt'
    config = dict(original.config, ct_scene_manifest_sha256='scene350',
                  ct_mechanism_steps_per_epoch_observed=12, ct_mechanism_tracklets_observed=16)
    torch.save({'state_dict': {prefix + key: value for key, value in state.items()},
                'hyper_parameters': {'config': config}}, path)
    restored = ExportProbe()
    # Current verified policy is external to trained parameters and must survive
    # loading the checkpoint; the saved deployment policy must not overwrite it.
    restored.ct_joint_router.install_policy({'kind': 'threshold', 'threshold': .25})
    report = load_initial_weights(restored, path, require_complete=True)
    assert report['complete'] and report['selected_prefix_strip'] == (prefix or 'none')
    assert restored.ct_joint_router.action_policy == {'kind': 'threshold', 'threshold': .25}
    for name in ('ct_plugin_rng', 'ct_memory_control_rng'):
        with getattr(original, name).fork('cpu'):
            expected = torch.rand(9)
        with getattr(restored, name).fork('cpu'):
            actual = torch.rand(9)
        assert torch.equal(expected, actual)
    for name, parameter in original.named_parameters():
        assert torch.equal(parameter, dict(restored.named_parameters())[name])


def test_offline_restore_rejects_missing_or_wrong_type_private_rng_state(tmp_path):
    model = ExportProbe()
    state = model.state_dict()
    state['ct_plugin_rng._extra_state'] = torch.zeros(1)
    path = tmp_path / 'broken.ckpt'
    torch.save({'state_dict': state, 'hyper_parameters': {'config': model.config}}, path)
    with pytest.raises(RuntimeError, match='ct_plugin_rng._extra_state'):
        load_initial_weights(model, path, require_complete=True)


def test_resume_and_calibration_use_same_v27_config_identity():
    config = dict(ct_enable_v27=True, ct_router_radius_max=2., ct_targetness_weight=.2,
                  ct_runtime_protocol='safe_seqtrack_auto_v1')
    saved = dict(config, checkpoint='/old/path.ckpt', ct_scene_manifest_sha256='scene',
                 ct_protocol_role='train', ct_mechanism_steps_per_epoch_observed=17)
    # Scene identity is separately bound in the resume contract.
    saved['ct_scene_manifest_sha256'] = None
    payload = dict(ct_online_resume_contract=build_online_resume_contract(saved),
                   hyper_parameters={'config': saved}, ct_epoch_boundary_complete=True,
                   ct_global_rng_state={'schema': 'ct_seqtrack.global_rng.v1'})
    validate_online_resume_contract(payload, config)
    for field in ('ct_router_radius_max', 'ct_targetness_weight'):
        with pytest.raises(ValueError, match=field):
            validate_online_resume_contract(payload, dict(config, **{field: 1.}))
