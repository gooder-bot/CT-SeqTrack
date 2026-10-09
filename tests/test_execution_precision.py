"""可选 BF16 的实际 CUDA 精度边界；CPU 环境明确跳过。"""
import pytest
import torch

from tests.test_ct_v31_runtime import TinySource
from tools.check_execution_performance import load_engineering_config, module_set, seed_all


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires real CUDA BF16 execution')
@pytest.mark.parametrize('config_path', [
    'cfgs/ct_seqtrack/35_full_w_scaled_lr_mini.yaml',
    'cfgs/ct_seqtrack/36_a_full_mini.yaml',
])
def test_bf16_features_keep_sensitive_heads_and_adam_fp32(config_path):
    if not torch.cuda.is_bf16_supported():
        pytest.skip('native CUDA BF16 unavailable')
    config = load_engineering_config(config_path)
    config.update(execution_profile='optimized_bf16', workers=0, batch_size=2,
                  point_sample_size=8, v32_reserve_windows=0)
    api = module_set(config)
    config = api['config'].normalize_config(config)
    seed_all(42)
    loaders = api['data'].build_loaders(config, roles=('train',), sources={'train': TinySource((5, 4))})
    model = api['host'](config, loaders=loaders).cuda().train()
    if not hasattr(model, 'current_epoch'):
        model.current_epoch = 0
    model.on_train_epoch_start()
    observed, handles = {}, []
    modules = dict(segmentation=model.tracker.observation.seg_pointnet.fc,
                   coarse=model.tracker.observation.coarse_box_head,
                   pose=model.tracker.decoder.pose_head)
    for name, module in modules.items():
        def record(current, args, output, name=name):
            observed[name] = (args[0].dtype, output.dtype)
        handles.append(module.register_forward_hook(record))
    feature_dtypes = []
    handles.append(model.tracker.observation.seg_pointnet.seq_per_point[0][0].register_forward_hook(
        lambda module, args, output: feature_dtypes.append(output.dtype)))
    try:
        rows = next(iter(loaders['train']))
        batch, output = model._forward_raw(rows, model.train_builder, training=True)
        losses = model.tracker.compute_losses(batch, output)
        optimizer = model.configure_optimizers()['optimizer']
        losses['loss_total'].backward()
        optimizer.step()
        model._pending_train = (output, batch, len(rows))
        model.on_train_batch_end(None, rows, 0)
    finally:
        for handle in handles:
            handle.remove()
    assert feature_dtypes == [torch.bfloat16]
    assert all(pair == (torch.float32, torch.float32) for pair in observed.values())
    assert len(observed) == len(modules)
    assert output.accepted_box.dtype == torch.float32
    assert all(value.dtype == torch.float32 and torch.isfinite(value).all() for value in losses.values())
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    for state in optimizer.state.values():
        for name in ('exp_avg', 'exp_avg_sq'):
            assert state[name].dtype == torch.float32 and torch.isfinite(state[name]).all()
