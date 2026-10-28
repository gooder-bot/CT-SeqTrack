"""v39 真实模型 CPU 合成事务：梯度、因果边界和轻路径预算。"""
from copy import deepcopy
from types import SimpleNamespace
import torch

from models.ct_v39.config import normalize_config
from models.ct_v39.data import BatchBuilder, RawEndpointDataset
from models.ct_v39.raw_service import RawFrameService
from models.ct_v39.model import V39Model, route_decision, geometry_aliases
from tests.helpers.raw_tracking import TinySource, request


def config(**overrides):
    values = dict(ct_engineering_check=True, execution_profile='optimized_fp32',
        v39_feature_bf16=False, v39_attention_backend='math_reference',
        point_sample_size=8, batch_size=2, workers=0, v32_reserve_windows=0,
        v39_template_points=8, v39_crop_points=8, v39_recovery_local_points=16,
        v39_recovery_global_points=16, v39_refine_crop_points=8)
    values.update(overrides)
    return normalize_config(values)


def test_complete_training_transaction_and_live_m_gradient():
    torch.set_num_threads(1); torch.manual_seed(42)
    cfg = config(); service = RawFrameService(cfg)
    raw = RawEndpointDataset(TinySource(), service)
    builder = BatchBuilder(cfg, service)
    model = V39Model(cfg).train()
    tx = builder.begin([raw[request(3, start=3)], raw[request(3, start=3, track=1)]], 'cpu', training=True)
    output = model.forward_transaction(tx, builder)
    assert output.candidates.boxes.shape == (2, 7, 4)
    assert output.full_executed.all() and builder.materialization_count == 2
    assert not any('acquisition' in key for key in model.state_dict())
    losses = model.compute_losses(tx, output)
    assert torch.isfinite(losses['loss_total'])
    losses['loss_total'].backward()
    assert model.g.edge_mlp[0].weight.grad.abs().sum() > 0
    assert model.multistep.head[-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    # 零初始化末层第一步只更新出口；一步优化后辅助梯度必须到达共享 B1。
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    from models.ct_v39.multistep import compute_multistep_loss
    motion = compute_multistep_loss(dict(tx.inputs, **tx.targets), model.plan_prior(tx.inputs), model.multistep)
    motion['loss_contribution_m'].backward()
    assert model.prior.context[0].weight.grad.abs().sum() > 0
    records = builder.commit(output, tx, diagnostics=True)
    assert len(records) == 2 and builder._pending is None
    service.close()


def test_g_only_loss_has_no_b0_or_prior_gradient():
    torch.set_num_threads(1); torch.manual_seed(1)
    cfg=config(); service=RawFrameService(cfg); raw=RawEndpointDataset(TinySource(),service)
    builder=BatchBuilder(cfg,service); model=V39Model(cfg).train()
    tx=builder.begin([raw[request(3,start=3)]],'cpu',training=True)
    out=model.forward_transaction(tx,builder); loss=model.compute_losses(tx,out)
    (loss['loss_g']+loss['loss_g_aux']).backward()
    assert all(p.grad is None or not p.grad.any() for p in model.prior.parameters())
    assert all(p.grad is None or not p.grad.any() for p in model.observation.parameters())
    assert all(p.grad is None or not p.grad.any() for p in model.decoder.parameters())
    builder.abort();service.close()


def test_route_deduplicates_geometry_without_erasing_measurement():
    cfg=config(); boxes=torch.zeros(1,7,4); boxes[:,2,0]=3
    valid=torch.tensor([[False,True,True,False,True,False,False]])
    measured=torch.tensor([[False,False,False,False,True,False,False]])
    logits=torch.full((1,7,3),10.); utility=torch.full((1,7),.9); utility[:,2]=.5
    v=dict(logits=logits,utility=utility,support_count=torch.full((1,7),7),region_no_match_logits=torch.full((1,7),-10.))
    inp=dict(consecutive_cheap=torch.tensor([0]),full_age=torch.tensor([.5]))
    selected,cheap,gap=route_decision(inp,boxes,valid,measured,v,cfg)
    assert selected.item()==4 and cheap.item() and gap.item()>.39
    assert geometry_aliases(boxes,valid)[0,4]==1
    inp['consecutive_cheap'].fill_(3)
    assert not route_decision(inp,boxes,valid,measured,v,cfg)[1].item()


def test_actual_fast_path_never_materializes_b0(monkeypatch):
    torch.set_num_threads(1)
    cfg=config(test=True,checkpoint='engineering_not_loaded',v39_inference_mode='adaptive')
    service=RawFrameService(cfg);raw=RawEndpointDataset(TinySource(),service)
    builder=BatchBuilder(cfg,service);model=V39Model(cfg).eval()
    original_refine=model.g.refine
    calls=dict(refine=0,verify=0)
    def confident_refine(*args,**kwargs):
        calls['refine']+=1
        output=original_refine(*args,**kwargs)
        output['valid']=torch.ones_like(output['valid'])
        return output
    original_verify=model._verify
    def confident_verify(*args,**kwargs):
        calls['verify']+=1
        output=original_verify(*args,**kwargs)
        output['logits'][...,2]=20.
        output['region_no_match_logits'].fill_(-20.)
        output['targetness_logits'].fill_(20.)
        output['support_mask']=output['encoded_pool']['valid'][:,None].expand_as(output['support_mask'])
        output['support_count']=output['support_mask'].sum(-1)
        output['utility'].fill_(.1);output['utility'][:,4]=.95
        return output
    monkeypatch.setattr(model.g,'refine',confident_refine)
    monkeypatch.setattr(model,'_verify',confident_verify)
    def unexpected(*args,**kwargs): raise AssertionError('fast path invoked B0 materialization')
    monkeypatch.setattr(builder,'materialize_B0',unexpected)
    tx=builder.begin([raw[request(3,start=3)]],'cpu',training=False)
    # Gate demands five current unique points; this fixture has only four crop points.
    tx.raw[0]=(torch.cat((tx.raw[0][0],tx.raw[0][0][:1]+.05)),torch.cat((tx.raw[0][1],torch.tensor([99]))))
    with torch.no_grad(): output=model.forward_transaction(tx,builder)
    assert not output.full_executed.any() and not output.policy_full.any()
    assert output.decoder is None and tx.b0_inputs is None and builder.materialization_count==0
    assert calls==dict(refine=1,verify=1), 'fast path must not repeat G after prefix acceptance'
    builder.abort();service.close()
