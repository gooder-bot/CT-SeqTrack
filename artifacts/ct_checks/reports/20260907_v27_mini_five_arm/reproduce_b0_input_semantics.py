"""CPU audit probes. The expected assertions document current defects, not fixes."""
import json
from pathlib import Path
import torch

from tests.test_ct_v27_full_model import full_model_runtime, _construct, _training_batch
from tests.test_ct_v27_input_flow import sampler_runtime
from utils.v27_input import build_v27_eval_input


def test_real_b0_frame_features_prior_provenance_and_unused_frame_mask(full_model_runtime):
    model = _construct(full_model_runtime, 'b0').eval()
    batch, sequence, state = _training_batch(full_model_runtime, model)
    captured = {}
    hook = model.feature_pointnet.register_forward_pre_hook(
        lambda module, args: captured.update(feature_input=args[0].detach().clone()))
    with torch.no_grad():
        model(batch)
    hook.remove()
    raw = model.encode_point_time(batch['points']).transpose(1, 2)
    if model.box_aware:
        raw = torch.cat([raw, batch['candidate_bc'].transpose(1, 2)], dim=1)
    B,C,total = raw.shape
    L = batch['valid_mask'].shape[1] + 1
    N = total // L
    expected = raw.reshape(B,C,L,N).permute(0,2,1,3).reshape(B*L,C,N)
    layout_mismatch = not torch.equal(captured['feature_input'], expected)
    assert layout_mismatch

    eval_batch, _ = build_v27_eval_input(model, sequence, 8, state.results_bbs,
                                        recursive_state=state)
    prior = eval_batch['points'].reshape(1,L,N,5)[:,:-1,:,4]
    values = torch.unique(prior).cpu().tolist()
    assert set(values).issubset({0.,1.})
    assert len(state.results_bbs)>1

    torch.manual_seed(1908)
    target = torch.randn(1,32,4)
    source = torch.randn(1,128,128)
    with torch.no_grad():
        all_valid=model.Transformer(target,source,torch.tensor([[1.,1.,1.]]))
        cold_start=model.Transformer(target,source,torch.tensor([[1.,0.,0.]]))
    assert torch.equal(all_valid,cold_start)
    result=dict(real_modules=True, synthetic_pointcloud_fixture=True,
        no_unused_cuda_backbone_invoked=True,
        actual_feature_input_shape=list(captured['feature_input'].shape),
        feature_input_differs_from_frame_split=layout_mismatch,
        feature_input_max_abs_difference=float((captured['feature_input']-expected).abs().max()),
        predicted_history_eval_frame=8,history_prior_values=values,
        transformer_output_identical_after_history_mask_change=True)
    Path(__file__).with_name('b0_input_semantics_reproduction.json').write_text(
        json.dumps(result,indent=2),encoding='utf8')
