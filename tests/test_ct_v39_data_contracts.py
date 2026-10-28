"""v39 raw/prepare/commit 的因果、缓存与原子提交合同。"""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.ct_v38.data import EndpointRequest
from models.ct_v39.config import normalize_config
from models.ct_v39.data import BatchBuilder, RawEndpointDataset
from models.ct_v39.gpu_prepare import inside, sample_indices
from models.ct_v39.raw_service import RawFrameService


class MetadataSource:
    def __init__(self, *, future_shift=0., big_ids=False, fail_raw=None):
        self.future_shift=future_shift;self.big_ids=big_ids;self.fail_raw=fail_raw
        self.raw_calls=[];self.metadata_calls=[]

    def get_num_tracklets(self):return 2
    def get_num_frames_tracklet(self,index):return 10
    def get_tracklet_key(self,index):return 'v39/tiny/'+str(index)

    def get_frame_metadata(self,track,frame):
        self.metadata_calls.append((track,frame))
        center=np.array([.2*frame,20.*track,0.],np.float64)
        if frame in (2,4):center[:2]+=self.future_shift
        return dict(timestamp=100.+frame*.5,frame_id=f'frame/{track}/{frame}',scene_id='test',
                    box_size=np.array([4.,2.,2.]),**{'3d_bbox':np.r_[center,.1]})

    def get_frames(self,track,frames):
        result=[]
        for frame in frames:
            self.raw_calls.append((track,frame))
            if (track,frame)==self.fail_raw:raise RuntimeError('injected raw decode failure')
            rng=np.random.default_rng(100+track)
            points=rng.uniform([-.8,-.5,-.4],[.8,.5,.4],size=(80,3))
            points+=np.array([.2*frame,20.*track,0.])
            ids=np.arange(80,dtype=np.int64)+(2**54 if self.big_ids else 0)
            result.append(dict(self.get_frame_metadata(track,frame),pc=points,point_ids=ids))
        return result


def config(**changes):
    return normalize_config(dict(ct_engineering_check=True,execution_profile='optimized_fp32',
        v39_feature_bf16=False,v39_attention_backend='math_reference',
        point_sample_size=8,batch_size=2,workers=0,v32_reserve_windows=0,
        v39_template_points=8,v39_crop_points=16,**changes))


def request(frame=1,track=0,*,epoch=9,branch=3,start=1,end=10):
    return EndpointRequest(epoch,track,branch,start,end,frame)


@pytest.fixture(autouse=True)
def one_thread():
    before=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.fixture
def services():
    result=[]
    def make(cfg):
        service=RawFrameService(cfg);result.append(service);return service
    yield make
    for service in result:service.close()


def build_case(services,source=None,cfg=None):
    cfg=cfg or config();source=source or MetadataSource();service=services(cfg)
    return BatchBuilder(cfg,service),RawEndpointDataset(source,service),source


def fake_output(builder,tx,*,selected=4,trusted=True,policy_full=True,actual_full=True):
    b=len(tx.rows);boxes=tx.inputs['fallback_box'][:,None].repeat(1,7,1)
    pool=builder.prepare_current_G(tx,boxes[:,selected])
    measured=torch.ones((b,7),dtype=torch.bool);measured[:,1:3]=False
    index=torch.full((b,),selected,dtype=torch.long);row=torch.arange(b)
    logits=torch.full((b,7,3),10. if trusted else -10.)
    targetness=torch.full((b,7,pool['valid'].shape[1]),10.)
    nomatch=torch.full((b,7),-10.)
    support=pool['valid'] & inside(pool['xyz'],boxes[:,selected],tx.inputs['box_size']) & measured[:,selected,None]
    score=logits[row,index,2].sigmoid()*measured[row,index]
    eligible=measured[row,index] & (support.sum(-1)>=builder.config.v39_g_min_support) & (score>=builder.config.v39_g_trust_threshold)
    return SimpleNamespace(candidates=SimpleNamespace(boxes=boxes,valid=torch.ones(b,7,dtype=torch.bool),
        measurement_eligible=measured),pool=pool,accepted_box=boxes[row,index],selected_index=index,
        selected_quality=torch.full((b,),.8)*measured[row,index],
        verification=dict(logits=logits,targetness_logits=targetness,region_no_match_logits=nomatch),
        commit_decision=SimpleNamespace(eligible=eligible,support_mask=support,trusted_score=score),
        full_executed=torch.full((b,),actual_full,dtype=torch.bool),
        policy_full=torch.full((b,),policy_full,dtype=torch.bool),
        search_triggered=torch.full((b,),selected==6,dtype=torch.bool),prior=SimpleNamespace(box=boxes[:,1].clone()))


def test_begin_failure_rolls_back_all_new_states_and_never_mutates_metadata(services):
    builder,raw,_=build_case(services,MetadataSource(fail_raw=(1,0)))
    rows=[raw[request(track=i)] for i in (0,1)]
    with pytest.raises(RuntimeError,match='injected raw'):
        builder.begin(rows,'cpu',training=True)
    assert builder.states=={} and builder._pending is None
    assert all('pc' not in frame and 'point_ids' not in frame for row in rows for frame in row['frames'].values())


def test_abort_removes_new_seed_but_preserves_existing_predecessor(services):
    builder,raw,_=build_case(services)
    row=raw[request()];tx=builder.begin([row],'cpu',training=True)
    builder.commit(fake_output(builder,tx),tx)
    state=builder.states[row['request'].state_key];old_boxes=deepcopy(state.boxes)
    next_rows=[raw[request(2)],raw[request(track=1)]]
    builder.begin(next_rows,'cpu',training=True);builder.abort()
    assert set(builder.states)=={row['request'].state_key}
    assert state.next_frame==2
    for key in old_boxes:np.testing.assert_array_equal(state.boxes[key],old_boxes[key])


def test_commit_teacher_execution_does_not_refresh_real_route_state(services):
    builder,raw,_=build_case(services)
    row=raw[request()];tx=builder.begin([row],'cpu',training=True)
    state=builder.states[row['request'].state_key];old_time=state.last_full_time
    records=builder.commit(fake_output(builder,tx,policy_full=False,actual_full=True),tx)
    assert records[0]['full_executed'] and not records[0]['policy_full']
    assert state.consecutive_cheap==1 and state.last_full_time==old_time
    tx=builder.begin([raw[request(2)]],'cpu',training=True)
    assert tx.inputs['consecutive_cheap'].item()==1
    builder.commit(fake_output(builder,tx,policy_full=True),tx)
    assert state.consecutive_cheap==0 and state.last_full_time==101.


def test_template_failure_cannot_partially_commit_batch(services,monkeypatch):
    builder,raw,_=build_case(services)
    rows=[raw[request(track=i)] for i in (0,1)]
    tx=builder.begin(rows,'cpu',training=True);out=fake_output(builder,tx)
    original=builder._template_from_support
    def fail_later(tx,i,data):
        if i==1:raise RuntimeError('template staging failure')
        return original(tx,i,data)
    monkeypatch.setattr(builder,'_template_from_support',fail_later)
    with pytest.raises(RuntimeError,match='staging failure'):builder.commit(out,tx)
    assert builder._pending is tx
    assert all(c['state'].next_frame==1 and 1 not in c['state'].boxes for c in tx.contexts)
    builder.abort();assert builder.states=={}


def test_int64_raw_identity_survives_grouped_commit_transfer(services):
    builder,raw,_=build_case(services,MetadataSource(big_ids=True))
    row=raw[request()];tx=builder.begin([row],'cpu',training=True)
    out=fake_output(builder,tx);original_ids=set(map(int,out.pool['point_ids'][0,out.pool['valid'][0]].tolist()))
    builder.commit(out,tx)
    ids=builder.states[row['request'].state_key].templates[-1]['ids']
    assert ids.dtype==np.int64 and min(ids)>=2**54
    assert set(map(int,ids)).issubset(original_ids)
    assert tx.diagnostics_cpu['diagnostic_raw_point_count'].dtype==np.int64


def test_motion_epoch_rejects_old_pairs_then_rebuilds_after_two_supported_endpoints(services):
    builder,raw,_=build_case(services)
    row=raw[request()];tx=builder.begin([row],'cpu',training=True)
    builder.commit(fake_output(builder,tx,selected=6),tx)
    state=builder.states[row['request'].state_key]
    assert state.motion_epoch==1 and state.motion_reset_pending and not state.trusted_velocity_valid
    tx=builder.begin([raw[request(2)]],'cpu',training=True)
    assert not tx.inputs['history_pair_valid'].any()
    records=builder.commit(fake_output(builder,tx),tx)
    assert records[0]['trusted_velocity_updated'] and not state.motion_reset_pending
    tx=builder.begin([raw[request(3)]],'cpu',training=True)
    assert tx.inputs['history_pair_valid'].tolist()==[[False,True]]
    builder.abort()


def test_long_untrusted_then_supported_resets_epoch_without_bridging_velocity(services):
    builder,raw,_=build_case(services)
    row=raw[request()]
    for frame in (1,2):
        tx=builder.begin([row if frame==1 else raw[request(frame)]],'cpu',training=True)
        builder.commit(fake_output(builder,tx,selected=1),tx)
    state=builder.states[row['request'].state_key]
    tx=builder.begin([raw[request(3)]],'cpu',training=True)
    records=builder.commit(fake_output(builder,tx),tx)
    assert state.motion_epoch==1 and not state.trusted_velocity_valid
    assert records[0]['motion_discontinuity'] and not records[0]['trusted_velocity_updated']


def test_recovery_waiting_period_never_builds_edge_through_untrusted_output(services):
    builder,raw,_=build_case(services)
    row=raw[request()];tx=builder.begin([row],'cpu',training=True)
    builder.commit(fake_output(builder,tx,selected=6),tx)
    state=builder.states[row['request'].state_key]
    assert state.motion_reset_pending and state.motion_epoch==1
    tx=builder.begin([raw[request(2)]],'cpu',training=True)
    builder.commit(fake_output(builder,tx,selected=1),tx)
    assert state.motion_reset_pending and not state.transitions[2]
    tx=builder.begin([raw[request(3)]],'cpu',training=True)
    assert not tx.inputs['history_pair_valid'].any()
    builder.commit(fake_output(builder,tx,selected=4),tx)
    assert state.motion_reset_pending and not state.transitions[3]
    assert not state.trusted_velocity_valid
    tx=builder.begin([raw[request(4)]],'cpu',training=True)
    assert not tx.inputs['history_pair_valid'].any()
    records=builder.commit(fake_output(builder,tx,selected=4),tx)
    assert records[0]['trusted_velocity_updated'] and state.transitions[4]
    assert state.trusted_velocity_valid and not state.motion_reset_pending
    tx=builder.begin([raw[request(5)]],'cpu',training=True)
    assert tx.inputs['history_pair_valid'].tolist()==[[False,True]]
    builder.abort()


def test_future_labels_never_change_model_inputs_or_forward_and_do_not_decode_future_raw(services):
    from models.ct_v39.model import V39Model
    torch.manual_seed(3920);cfg=config();model=V39Model(cfg).eval()
    transactions=[];outputs=[]
    for shift in (0.,100.):
        builder,raw,source=build_case(services,MetadataSource(future_shift=shift),cfg)
        source.raw_calls.clear()
        tx=builder.begin([raw[request()]],'cpu',training=True)
        with torch.no_grad():out=model.forward_transaction(tx,builder)
        assert set(source.raw_calls)<={(0,0),(0,1)}
        assert all(not key.startswith(('future_','target_','template_target')) for key in tx.inputs)
        transactions.append(tx);outputs.append(out)
    a,b=transactions
    assert not torch.equal(a.targets['future_displacement'],b.targets['future_displacement'])
    for key in a.inputs:torch.testing.assert_close(a.inputs[key],b.inputs[key],rtol=0.,atol=0.)
    for key in a.templates:torch.testing.assert_close(a.templates[key],b.templates[key],rtol=0.,atol=0.)
    assert torch.equal(outputs[0].accepted_box,outputs[1].accepted_box)
    assert torch.equal(outputs[0].selected_index,outputs[1].selected_index)
    assert torch.equal(outputs[0].verification['logits'],outputs[1].verification['logits'])


def test_sampling_is_cache_and_row_order_independent_but_epoch_branch_specific(services):
    results=[]
    for cached,reverse in ((True,False),(False,True)):
        cfg=config(v39_cpu_raw_cache_bytes=100000 if cached else 0,v39_gpu_raw_cache_bytes=100000 if cached else 0)
        builder,raw,_=build_case(services,cfg=cfg)
        rows=[raw[request(track=i)] for i in (0,1)]
        if reverse:rows.reverse()
        tx=builder.begin(rows,'cpu',training=True)
        pool=builder.prepare_current_G(tx,tx.inputs['fallback_box'])
        builder.materialize_B0(tx,torch.arange(2),tx.inputs['fallback_box'])
        results.append({row['tracklet_key']:(pool['point_ids'][i].clone(),tx.b0_inputs['point_ids'][i].clone())
                        for i,row in enumerate(rows)})
    for key in results[0]:
        for a,b in zip(results[0][key],results[1][key]):assert torch.equal(a,b)
    builder,raw,_=build_case(services)
    rows=[raw[request(branch=i)] for i in (1,2)]
    tx=builder.begin(rows,'cpu',training=True)
    pool=builder.prepare_current_G(tx,tx.inputs['fallback_box'])
    assert not torch.equal(pool['point_ids'][0],pool['point_ids'][1])


def test_future_tail_mask_and_effective_time_do_not_change_physical_time(services):
    builder,raw,_=build_case(services)
    row=raw[request(8,start=8)]
    for frame,meta in row['frames'].items():meta['_ct_effective_timestamp']=frame*.25
    tx=builder.begin([row],'cpu',training=True)
    assert tx.targets['future_valid'].tolist()==[[True,False]]
    assert tx.targets['future_dt'].tolist()==[[1.,0.]]
    assert tx.inputs['current_dt'].item()==.25
    assert tx.inputs['history_times'].tolist()==[[-.75,-.5,-.25]]
    assert tx.inputs['frame_times'].tolist()==[[-1.5,-1.,-.5,0.]]


def test_sample_indices_preserves_raw_order_large_identity_and_global_rng():
    ids=torch.arange(100,dtype=torch.long)+2**54;mask=torch.ones(100,dtype=torch.bool)
    before=torch.get_rng_state().clone()
    first=sample_indices(ids,mask,12,3918)
    second=sample_indices(ids,mask,12,3918)
    assert torch.equal(first,second) and torch.equal(torch.get_rng_state(),before)
    assert bool((first[1:]>first[:-1]).all())
    with pytest.raises(ValueError):sample_indices(ids,mask,0,0)


@pytest.mark.parametrize('broken', (None, {}, {'3d_bbox':None}, {'3d_bbox':[0.,0.,0.,0.], 'timestamp':'bad'},
                                  {'3d_bbox':[0.,0.,0.,0.], 'timestamp':float('nan')}))
def test_invalid_future_annotation_masks_only_that_horizon(services,monkeypatch,broken):
    builder,raw,source=build_case(services)
    row=raw[request()];get=source.get_frame_metadata
    monkeypatch.setattr(source,'get_frame_metadata',lambda track,frame:broken if frame==2 else get(track,frame))
    tx=builder.begin([row],'cpu',training=True)
    assert tx.targets['future_valid'].tolist()==[[False,True]]
    assert tx.targets['future_dt'][0,0]==0
    assert torch.equal(tx.targets['future_displacement'][0,0],torch.zeros(2))
    assert tx.inputs['current_dt'].item()==.5


def test_future_io_error_is_not_silently_treated_as_missing_supervision(services,monkeypatch):
    builder,raw,source=build_case(services)
    row=raw[request()];get=source.get_frame_metadata
    def fail(track,frame):
        if frame==2:raise OSError('future source unavailable')
        return get(track,frame)
    monkeypatch.setattr(source,'get_frame_metadata',fail)
    with pytest.raises(OSError,match='source unavailable'):builder.begin([row],'cpu',training=True)
    assert builder.states=={} and builder._pending is None


def test_device_cache_hit_does_not_refill_disabled_cpu_cache(services):
    cfg=config(v39_cpu_raw_cache_bytes=0,v39_gpu_raw_cache_bytes=100000)
    builder,raw,source=build_case(services,cfg=cfg)
    row=raw[request()];source.raw_calls.clear()
    tx=builder.begin([row],'cpu',training=True)
    before=list(source.raw_calls)
    for _ in range(3):
        builder.service.prefetch_many([(source,0,1,row['frames'][1])],'cpu')
        builder._raw(row,1,'cpu',tx.contexts[0]['anchor'])
    assert source.raw_calls==before
    assert builder.service.cpu.bytes==0


def test_support_identity_cannot_authorize_fabricated_coordinates(services):
    builder,raw,_=build_case(services)
    row=raw[request()];tx=builder.begin([row],'cpu',training=True)
    out=fake_output(builder,tx)
    out.pool['xyz'][0,0,0]+=.001
    with pytest.raises(ValueError,match='coordinates differ'):builder.commit(out,tx)
    assert tx.contexts[0]['state'].next_frame==1


def test_physical_frame_key_shares_raw_across_targets(services):
    source=MetadataSource();service=services(config())
    a=source.get_frame_metadata(0,1);b=dict(a,**{'3d_bbox':np.array([20.,0.,0.,0.])})
    key_a=service.key(source,0,1,a);key_b=service.key(source,1,1,b)
    assert key_a==key_b
    source.raw_calls.clear()
    service.fetch_many([(source,0,1,a),(source,1,1,b)])
    assert source.raw_calls==[(0,1)]
