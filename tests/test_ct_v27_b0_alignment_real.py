"""五臂使用实际 SeqTrack 网络核对两步 B0/Adam/BN 和调用者 RNG。"""
import copy
import hashlib

import torch

from tests.test_ct_v27_full_model import full_model_runtime, _construct, _training_batch
from tests.test_ct_v27_input_flow import sampler_runtime
from utils.training_isolation import capture_global_rng_state, restore_global_rng_state


def _digest_tensors(rows):
    digest = hashlib.sha256()
    for name, value in rows:
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def test_actual_five_arm_b0_adam_and_bn_updates_match(full_model_runtime):
    observation_model = _construct(full_model_runtime, 'b0')
    shared_batch, _, _ = _training_batch(full_model_runtime, observation_model, batch_size=4)
    shared_batch['candidate_id'] = torch.arange(4)
    common = None
    for arm in ('b0', 'b1_gru', 'b1_cfc', 'full_minus_b3', 'full'):
        model = _construct(full_model_runtime, arm).train()
        optimizer = model.configure_optimizers()['optimizer']
        history = []
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            observation = copy.deepcopy(shared_batch)
            routing = (model.use_ct_joint_full, model.use_b1motion_v3,
                       model.ct_enable_b1, model.ct_enable_b2, model.ct_enable_b3)
            try:
                model.use_ct_joint_full = model.use_b1motion_v3 = False
                model.ct_enable_b1 = model.ct_enable_b2 = model.ct_enable_b3 = False
                output = model(observation)
                losses = model.compute_loss(observation, output)
                total = model._ct_candidate_weighted_observation_loss(observation, output, losses)
            finally:
                (model.use_ct_joint_full, model.use_b1motion_v3,
                 model.ct_enable_b1, model.ct_enable_b2, model.ct_enable_b3) = routing
            if model.use_b1motion_v3:
                # Match the production dual-stream RNG boundary around fetch,
                # label construction and the mechanism transaction.
                rng = capture_global_rng_state()
                try:
                    batch, _, state = _training_batch(full_model_runtime, model, batch_size=1)
                    batch['ct_recursive_state_age'] = torch.tensor([float(state.rollout_age(8))])
                    batch['ct_recursive_state_age_valid'] = torch.ones(1)
                    mechanism = model._forward_safe_mechanism(batch)
                    total = total + model.compute_loss(batch, mechanism)['loss_plugin_transaction']
                finally:
                    restore_global_rng_state(rng)
            total.backward()
            b0 = [(name, parameter) for name, parameter in model.named_parameters()
                  if not model._ct_any_plugin_parameter(name)]
            gradients = _digest_tensors((name, parameter.grad) for name, parameter in b0
                                        if parameter.grad is not None)
            optimizer.step()
            weights = _digest_tensors(b0)
            bn = _digest_tensors((name, value) for name, value in model.named_buffers()
                                 if name.endswith(('running_mean', 'running_var', 'num_batches_tracked')))
            adam = _digest_tensors((f'{name}.{key}', value)
                for name, parameter in b0 for key, value in sorted(optimizer.state[parameter].items())
                if torch.is_tensor(value))
            history.append((gradients, weights, bn, adam,
                            hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()))
        if common is None:
            common = history
        else:
            assert history == common, arm
