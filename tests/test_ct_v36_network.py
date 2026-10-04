"""v36网络的候选权限、初始化、缓存几何和梯度边界。"""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from models.ct_v31.observation import B0Observation
from models.ct_v36.contracts import EvidenceHypotheses, CandidateSet
from models.ct_v36.decoder import SharedHypothesisDecoder
from models.ct_v36.evidence import B2IdentityEvidence, build_live_modes
from models.ct_v36.verification import (FinalBoxVerifier, LocalBoxReader, build_current_pool,
    candidate_authorization, inside_final_boxes, projected_xy_size)
from tests.test_ct_v31_joint import make_batch


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(36)
    yield
    torch.set_num_threads(previous)


def inputs():
    batch = make_batch(1,24)
    batch['points'][:,-1,:8,:3] = torch.tensor([1.8,.8,.6])+torch.randn(1,8,3)*.02
    batch['extension_ids'] = torch.where(batch['extension_valid'],batch['extension_ids']+1000,-1)
    batch['history_is_initial'] = torch.tensor([[True,False,False]])
    observation = B0Observation(8,query_context=True,coarse_condition=True).eval()(batch)
    prior = SimpleNamespace(feature=torch.randn(1,128,requires_grad=True),
        event_feature=torch.randn(1,128),box=torch.tensor([[.2,0.,0.,.1]]),
        origin_world=torch.zeros(1,4,dtype=torch.double),axes=torch.eye(2)[None],
        prior_valid=torch.tensor([True]),motion_valid=torch.tensor([True]),
        range_xy=torch.tensor([[1.,.5]]),horizon=torch.tensor([.5]),velocity_age=torch.tensor([.25]),
        geometry=torch.zeros(1,8),initial=torch.tensor([False]))
    return batch,observation,prior


def evidence_fixture(batch):
    evidence = EvidenceHypotheses.empty(batch['target_box'])
    evidence.point_valid[:,:8] = True
    evidence.point_ids[:,:8] = torch.arange(1000,1008)
    evidence.point_xyz[:,:8] = batch['points'][:,-1,:8,:3]
    evidence.point_features[:,:8] = torch.randn(1,8,64)
    evidence.selected_identity_logits[:,:8] = 2.
    evidence.mode_valid[:] = True
    evidence.members[:,0,:4] = True
    evidence.members[:,1,4:8] = True
    evidence.centers_xyz = torch.tensor([[[.3,0.,0.],[2.,0.,0.]]],requires_grad=True)
    return evidence


def test_decoder_q0_history_inputs_are_isolated_from_extension_and_memory():
    batch,observation,prior = inputs()
    evidence = evidence_fixture(batch)
    decoder = SharedHypothesisDecoder(dropout=0.).eval()
    torch.nn.init.normal_(decoder.pose_head.weight,std=.03)
    torch.nn.init.normal_(decoder.mode_pose_head.weight,std=.03)
    torch.nn.init.normal_(decoder.local_reader.output.weight,std=.03)
    first = decoder(observation,evidence,batch,prior)
    changed = replace(evidence,point_features=evidence.point_features+7,
        point_xyz=evidence.point_xyz+20,memory_features=evidence.memory_features+100,
        centers_xyz=evidence.centers_xyz+4)
    second = decoder(observation,changed,batch,prior)
    torch.testing.assert_close(first.boxes[:,0],second.boxes[:,0],rtol=0,atol=0)
    torch.testing.assert_close(first.history_boxes,second.history_boxes,rtol=0,atol=0)
    torch.testing.assert_close(first.boxes[:,1],prior.box,rtol=0,atol=0)
    assert not torch.equal(first.boxes[:,2:],second.boxes[:,2:])
    assert not any('memory_projection' in name or 'quality_head' in name for name,_ in decoder.named_parameters())


def test_decoder_mode_query_detached_but_output_seed_center_remains_live():
    batch,observation,prior = inputs()
    evidence = evidence_fixture(batch)
    decoder = SharedHypothesisDecoder(dropout=0.).eval()
    output = decoder(observation,evidence,batch,prior)
    feature_grad = torch.autograd.grad(output.features.square().sum(),evidence.centers_xyz,
                                       allow_unused=True,retain_graph=True)[0]
    assert feature_grad is None
    gradient = torch.autograd.grad(output.boxes[:,2:,:3].sum(),evidence.centers_xyz)[0]
    torch.testing.assert_close(gradient,torch.ones_like(gradient))


def test_no_sequence_q0_is_invalid_and_cannot_launder_prior_into_measurement():
    batch,observation,prior = inputs()
    batch['point_valid'].zero_()
    observation = B0Observation(8,query_context=True,coarse_condition=True).eval()(batch)
    evidence = EvidenceHypotheses.empty(batch['target_box'])
    candidates = SharedHypothesisDecoder(dropout=0.).eval()(observation,evidence,batch,prior)
    assert candidates.valid.tolist() == [[False,True,False,False]]
    verification = FinalBoxVerifier().eval()(candidates,observation,evidence,batch,prior)
    assert torch.isfinite(verification.logits).all()
    assert not verification.support_mask.any()


def test_sequential_modes_exclude_final_members_and_keep_live_votes():
    votes = torch.tensor([[[0.,0.,0.],[.2,0.,0.],[.1,.1,0.],[3.,0.,0.],[3.1,0.,0.]]],requires_grad=True)
    weights = torch.ones(1,5,requires_grad=True)
    valid = torch.ones(1,5,dtype=torch.bool)
    ids = torch.tensor([[30,10,20,50,40]])
    centers,okay,members,_ = build_live_modes(votes,weights,valid,ids)
    assert okay.all() and not (members[:,0]&members[:,1]).any()
    permutation = torch.tensor([3,1,4,0,2])
    other = build_live_modes(votes[:,permutation],weights[:,permutation],valid[:,permutation],ids[:,permutation])
    torch.testing.assert_close(centers,other[0],rtol=0,atol=0)
    assert torch.equal(members[:,:,permutation],other[2])
    gradients = torch.autograd.grad(centers.square().sum(),(votes,weights))
    assert all(torch.isfinite(g).all() and g.abs().sum()>0 for g in gradients)
    repeated = build_live_modes(votes[:,:1],weights[:,:1],valid[:,:1],ids[:,:1])
    assert repeated[1].tolist() == [[True,False]]


def test_memory_changes_identity_not_fixed_id_current_geometry_or_votes():
    batch,observation,_ = inputs()
    model = B2IdentityEvidence().eval()
    cache = model.encode_current(observation,batch)
    selected = torch.full((1,256),-1,dtype=torch.long)
    selected[:,:24] = torch.arange(24)
    logits,_,_ = model.identify(cache,observation,batch)
    first = model.geometry_votes(cache,selected)
    changed = dict(batch,memory_points=batch['memory_points']+4)
    other_logits,_,_ = model.identify(cache,observation,changed)
    second = model.geometry_votes(cache,selected)
    assert not torch.equal(logits,other_logits)
    for left,right in zip(first,second):
        torch.testing.assert_close(left,right,rtol=0,atol=0)
    first[2].square().sum().backward()
    assert model.memory_attention.in_proj_weight.grad is None
    assert model.vote_head[-1].weight.grad.abs().sum()>0
    model.zero_grad(set_to_none=True)
    calls = dict(local=0,vote=0,geometry_attention=0)
    handles=[]
    for name,module in [('local',model.local_geometry),('vote',model.vote_head),('geometry_attention',model.observation_attention)]:
        def hook(module,args,output,name=name):
            calls[name]+=1
        handles.append(module.register_forward_hook(hook))
    model(observation,batch)
    for handle in handles:
        handle.remove()
    assert calls == dict(local=1,vote=1,geometry_attention=1)


def test_verifier_initialization_final_geometry_permissions_and_gradient_boundary():
    batch,observation,prior = inputs()
    evidence = evidence_fixture(batch)
    observation = replace(observation,point_features=observation.point_features.detach().requires_grad_(),
        foreground_probability=torch.ones_like(observation.foreground_probability,requires_grad=True))
    evidence = replace(evidence,point_features=evidence.point_features.detach().requires_grad_(),
        vote_xyz=evidence.vote_xyz.detach().requires_grad_(),
        selected_identity_logits=evidence.selected_identity_logits.detach().requires_grad_())
    boxes = torch.zeros(1,4,4,requires_grad=True)
    candidates = CandidateSet(boxes,torch.ones(1,4,dtype=torch.bool),torch.zeros(1,3,4),
                              torch.zeros(1,4,64),torch.zeros(1,2,4))
    verifier = FinalBoxVerifier().eval()
    output = verifier(candidates,observation,evidence,batch,prior)
    torch.testing.assert_close(output.utility,torch.full((1,4),.5))
    torch.testing.assert_close(output.measurement_score,torch.tensor([[.1,0.,.1,.1]]))
    assert output.support_mask.sum(-1).tolist() == [[32,32,28,28]]
    assert not output.authorized[0,2,28:].any()
    assert not output.authorized[0,3,24:28].any()
    assert output.support_ids[0,:24].tolist() == list(range(24))
    # 零出口的第一步不期待前级梯度；激活测试头后再核验所有权。
    torch.nn.init.normal_(verifier.output.weight,std=.03)
    output = verifier(candidates,observation,evidence,batch,prior)
    variables = (observation.point_features,evidence.point_features,boxes,evidence.vote_xyz,
                 observation.foreground_probability,evidence.selected_identity_logits,prior.feature)
    gradients = torch.autograd.grad(output.logits.square().sum(),variables,allow_unused=True)
    assert all(g is not None and torch.isfinite(g).all() and g.abs().sum()>0 for g in gradients[:2])
    assert all(g is None for g in gradients[2:])


def test_verifier_uses_position_range_without_history_neural_score_or_gt():
    batch,observation,prior = inputs()
    evidence = evidence_fixture(batch)
    boxes = torch.tensor([[[2.,0.,0.,0.]]]).expand(1,4,4).clone()
    candidates = CandidateSet(boxes,torch.ones(1,4,dtype=torch.bool),torch.zeros(1,3,4),
                              torch.zeros(1,4,64),torch.zeros(1,2,4))
    verifier = FinalBoxVerifier().eval()
    pool = build_current_pool(observation,evidence,batch)
    allowed = candidate_authorization(pool,evidence.members,candidates.valid,q0_extension=True)
    inside = allowed & inside_final_boxes(pool['xyz'],boxes,batch['box_size'])
    support = inside & (pool['probability'][:,None]>=.5)
    first = verifier.metadata(candidates,pool,allowed,inside,support,batch,prior)
    prior.range_xy = prior.range_xy*4
    second = verifier.metadata(candidates,pool,allowed,inside,support,batch,prior)
    assert first.shape == (1,4,32) and not first.requires_grad
    assert second[0,0,5].abs()<first[0,0,5].abs()
    torch.nn.init.normal_(verifier.output.weight,std=.03)
    before = verifier(candidates,observation,evidence,batch,prior)
    prior.measurement_score=torch.tensor([.99])
    changed=dict(batch,target_box=batch['target_box']+100,previous_quality=torch.tensor([.99]),
                 segmentation_labels=torch.zeros_like(batch['segmentation_labels']))
    after = verifier(candidates,observation,evidence,changed,prior)
    torch.testing.assert_close(before.logits,after.logits,rtol=0,atol=0)


def test_decoded_final_mode_duplicate_is_invalid_without_reforming_evidence():
    batch,observation,prior = inputs()
    evidence = evidence_fixture(batch)
    decoder = SharedHypothesisDecoder(dropout=0.).eval()
    class Collapse(torch.nn.Module):
        def forward(self, features):
            value = features.new_zeros((len(features),2,5))
            value[...,:3] = -evidence.centers_xyz.detach()
            value[...,4] = 1.
            return value
    decoder.mode_pose_head = Collapse()
    candidates = decoder(observation,evidence,batch,prior)
    assert evidence.mode_valid.tolist() == [[True,True]]
    assert candidates.valid.tolist() == [[True,True,True,False]]
    torch.testing.assert_close(candidates.boxes[:,2,:3],candidates.boxes[:,3,:3])


def test_unique_support_is_not_multiplied_by_corners_or_duplicate_pools():
    batch,observation,prior = inputs()
    observation = replace(observation,foreground_probability=torch.ones_like(observation.foreground_probability))
    evidence = evidence_fixture(batch)
    evidence.point_ids[0,0] = batch['point_ids'][0,-1,0]
    boxes = torch.zeros(1,4,4)
    candidates = CandidateSet(boxes,torch.ones(1,4,dtype=torch.bool),torch.zeros(1,3,4),
                              torch.zeros(1,4,64),torch.zeros(1,2,4))
    output = FinalBoxVerifier().eval()(candidates,observation,evidence,batch,prior)
    assert output.support_mask.sum(-1).tolist() == [[31,31,27,28]]
    assert not output.authorized[:,:,24].any()
    assert output.support_ids[0,24] == output.support_ids[0,0]


def test_local_radius_is_one_physical_meter_not_one_lwh_normalized_unit():
    reader = LocalBoxReader().eval()
    pool = dict(ids=torch.tensor([[3]]),valid=torch.ones(1,1,dtype=torch.bool),
        xyz=torch.tensor([[[.9,1.,.75]]]),features=torch.randn(1,1,64),
        probability=torch.ones(1,1),source=torch.zeros(1,1))
    boxes,size = torch.zeros(1,1,4),torch.tensor([[4.,2.,1.5]])
    allowed=torch.ones(1,1,1,dtype=torch.bool)
    value,statistics = reader(pool,boxes,allowed,size)
    assert statistics['local_neighbor_count'].sum()==0 and value.count_nonzero()==0
    pool['xyz'][0,0,0]=1.1
    _,statistics = reader(pool,boxes,allowed,size)
    assert statistics['local_neighbor_count'].sum()>0
    axes=torch.tensor([[[0.,-1.],[1.,0.]]])
    torch.testing.assert_close(projected_xy_size(size,torch.zeros(1),axes),torch.tensor([[2.,4.]]))
