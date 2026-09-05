"""Real Full model offline restore, including both nn.Module RNG extra states."""
import copy

import torch

from tests.test_ct_v27_full_model import full_model_runtime, _construct
from tests.test_ct_v27_input_flow import sampler_runtime
from utils.checkpoint_loading import load_initial_weights


def test_actual_full_state_restores_for_calibration_and_export(full_model_runtime, tmp_path):
    original = _construct(full_model_runtime, 'full')
    with torch.no_grad():
        next(original.parameters()).add_(.25)
    for name, count in (('ct_plugin_rng', 7), ('ct_memory_control_rng', 11)):
        with getattr(original, name).fork('cpu'):
            torch.rand(count)
    state = copy.deepcopy(original.state_dict())
    extra_keys = {key for key in state if key.endswith('_extra_state')}
    assert {'ct_plugin_rng._extra_state', 'ct_memory_control_rng._extra_state'} <= extra_keys
    config = dict(original.config, ct_scene_manifest_sha256='scene-provenance',
                  ct_mechanism_steps_per_epoch_observed=123,
                  ct_mechanism_selection_sha256='observed-only-at-dataset-build')
    path = tmp_path / 'full.ckpt'
    torch.save({'state_dict': state, 'hyper_parameters': {'config': config}}, path)
    restored = _construct(full_model_runtime, 'full')
    report = load_initial_weights(restored, path, require_complete=True)
    assert report['complete']
    restored_state = restored.state_dict()
    assert set(restored_state) == set(state)
    for key, value in state.items():
        if torch.is_tensor(value):
            assert torch.equal(value, restored_state[key]), key
    for name in ('ct_plugin_rng', 'ct_memory_control_rng'):
        with getattr(original, name).fork('cpu'):
            expected = torch.rand(17)
        with getattr(restored, name).fork('cpu'):
            actual = torch.rand(17)
        assert torch.equal(expected, actual)
    # Model weights cannot turn a missing external policy into a calibrated action.
    assert not bool(restored.ct_joint_router.calibrated)
    assert restored.ct_joint_router.action_policy == {'kind': 'never'}
    assert restored._ct_action_calibration_status['fallback'] == 'observation'
