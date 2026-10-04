"""v36 真实端点预算、独立测量状态和无 GT 提交的 CPU 合同。"""
from copy import deepcopy
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.ct_v36.config import normalize_config
from models.ct_v36.data import BatchBuilder, RawEndpointDataset, geometry_summary, build_loaders
from models.ct_v36.sampler import EndpointRequest, ReadyQueueBatchSampler, endpoint_seed
from models.ct_v36.contracts import VerificationOutput, MeasurementEvent
from tests.test_ct_v31_runtime import TinySource
from tests.test_ct_v34_identity import MINI_TRAIN_LENGTHS
from models.ct_v31.data import RawEndpointDataset as LegacyRawEndpointDataset


def cfg(**changes):
    return normalize_config(dict(ct_engineering_check=True, workers=0, point_sample_size=8,
                                 batch_size=4, **changes))


def request(frame, *, start=1, end=9, branch=3, epoch=10):
    return EndpointRequest(epoch, 0, branch, start, end, frame)


def output_for(batch, *, measurement=True, accepted=1, measurement_offset=0.):
    b,n=batch['point_valid'][:,-1].shape
    boxes=batch['fallback_box'][:,None].repeat(1,4,1)
    boxes[:,0]=batch['target_box']
    boxes[:,0,0]+=measurement_offset
    valid=torch.tensor([[True,True,False,False]]).repeat(b,1)
    xyz=batch['points'][:,-1,:,:3].clone();ids=batch['point_ids'][:,-1].clone()
    auth=batch['point_valid'][:,-1,None,:].repeat(1,4,1)
    support=auth.clone();support[:,1:]=False
    logits=torch.zeros(b,4,3);logits[...,2]=3. if measurement else -3.
    verify=VerificationOutput(logits,support,ids,xyz,torch.ones(b,n),auth)
    world=batch['fallback_box'].double().clone();world[:,:3]+=batch['anchor_box'][:,:3]
    prior=SimpleNamespace(world_box=world,range_xy=torch.ones(b,2),axes=torch.eye(2).repeat(b,1,1),
                          origin_frame=batch['event_frame_ids'][:,-1])
    return SimpleNamespace(accepted_box=boxes[:,accepted],candidates=SimpleNamespace(boxes=boxes,valid=valid),
        verification=verify,prior=prior,decision=SimpleNamespace(output_index=torch.full((b,),accepted),
        measurement_index=torch.full((b,),0 if measurement else -1),memory_eligible=torch.full((b,),measurement)))


def test_real_mini_lanes_preserve_all_sixty_epoch_updates_and_pair_A_F():
    a=ReadyQueueBatchSampler(MINI_TRAIN_LENGTHS)
    f=ReadyQueueBatchSampler(MINI_TRAIN_LENGTHS,continuous=False)
    for epoch in range(60):
        a.set_epoch(epoch);f.set_epoch(epoch)
        assert len(a)==len(f)==1195
        assert a.state_dict()['endpoint_order_sha256']==f.state_dict()['endpoint_order_sha256']
        rows=[r for batch in a for r in batch]
        assert len(rows)==19108 and len({(r.tracklet,r.branch,r.frame) for r in rows})==19108
        assert all(sum(r.branch==branch for r in rows)==4777 for branch in range(4))
        for batch in a:
            assert len({r.state_key for r in batch})==len(batch)
        if epoch>=10:
            assert all(r.window_start==1 for r in rows if r.branch==3)
            assert any(r.frame-r.window_start>8 for r in rows if r.branch==3)
    assert 60*1195==71700


def test_sampling_stream_ignores_window_reset_drain_and_checkpoint_epoch():
    a=request(9,start=1,end=21);f=request(9,start=9,end=17)
    for stream in ('b0_points','extension_pool','seed_pose'):
        assert endpoint_seed(42,a,'track',9,stream)==endpoint_seed(42,f,'track',9,stream)
        assert endpoint_seed(42,a,'track',9,stream)==endpoint_seed(42,
            EndpointRequest(10,0,3,1,21,9,True),'track',9,stream)
    assert endpoint_seed(42,request(2,branch=4,epoch=58),'track',2,'b0_points')==endpoint_seed(
        42,request(2,branch=4,epoch=60),'track',2,'b0_points')


def test_seed_branches_keep_perturbation_but_no_teacher_hint():
    raw=RawEndpointDataset(TinySource())
    for branch in range(4):
        builder=BatchBuilder(cfg())
        row=raw[request(1,branch=branch)]
        batch=builder.prepare([row])
        np.testing.assert_array_equal(builder.states[row['request'].state_key].seed_offset,np.zeros(4))
        assert batch['event_initial'][0,-1]
    exact=BatchBuilder(cfg());perturbed=BatchBuilder(cfg())
    first=raw[request(4,start=4,branch=0)]
    second=raw[request(4,start=4,branch=1)]
    b0=exact.prepare([first]);b1=perturbed.prepare([second])
    assert not b0['event_initial'].any() and not b1['event_initial'].any()
    assert 'history_box_provenance' not in b0 and 'window_seed_offset' not in b0
    assert np.linalg.norm(perturbed.states[second['request'].state_key].seed_offset)>0
    # 中段教师框与真实预测框不再使用不同 0/1 vs .2/.8 hint。
    hints=b0['points'][...,4][b0['point_valid']]
    assert all(any(float(x)==pytest.approx(y) for y in (.2,.8,.5)) for x in hints)
    for frame in (2,3):
        builder=BatchBuilder(cfg())
        batch=builder.prepare([raw[request(frame,start=frame,branch=1)]])
        assert not batch['history_is_initial'].any() and not batch['event_initial'].any()
        hint=batch['points'][0,:3,:,4][batch['point_valid'][0,:3]]
        assert all(any(float(x)==pytest.approx(y) for y in (.2,.8)) for x in hint)


def test_independent_measurement_changes_event_and_memory_not_accepted():
    raw=RawEndpointDataset(TinySource())
    builder=BatchBuilder(cfg());row=raw[request(1)]
    batch=builder.prepare([row]);out=output_for(batch)
    old=builder.states[row['request'].state_key].boxes[0].copy()
    record=builder.commit(out,batch,diagnostics=True)[0]
    state=builder.states[row['request'].state_key]
    np.testing.assert_allclose(state.boxes[1],old)
    np.testing.assert_allclose(state.events[-1].box,row['frames'][1]['3d_bbox'])
    assert record['measurement_index']==0 and record['output_index']==1
    assert record['memory_write'] and record['memory_source_index']==0
    np.testing.assert_array_equal(record['memory_world'],record['measurement_world'])
    np.testing.assert_allclose(state.memory.recent[-1]['box'],state.events[-1].box)


def test_C_keeps_events_without_memory_and_nonadjacent_motion_can_start():
    raw=RawEndpointDataset(TinySource(),memory_enabled=False)
    builder=BatchBuilder(cfg(v36_arm='C'))
    for frame in range(1,4):
        row=raw[request(frame)];batch=builder.prepare([row]);out=output_for(batch,measurement=frame==3)
        out.decision.memory_eligible[:]=False
        builder.commit(out,batch)
    state=builder.states[row['request'].state_key]
    assert state.memory is None and state.events[-1].frame==3
    assert state.events[-1].edge_valid and state.velocity_valid
    assert state.last_velocity[0]==pytest.approx(.4)


def test_labels_do_not_change_commit_and_transaction_is_once_only():
    raw=RawEndpointDataset(TinySource());row=raw[request(1)]
    left,right=BatchBuilder(cfg()),BatchBuilder(cfg())
    a,b=left.prepare([row]),right.prepare([deepcopy(row)])
    output=output_for(a)
    left.acquire(a,output.prior);right.acquire(b,output.prior)
    for key in ('target_box','target_box_size','history_target_boxes','segmentation_labels','bc_targets',
                'event_target_box','current_target_world','previous_target_world','physical_displacement',
                'extension_labels','prior_origin_target_world'):
        b[key].fill_(123)
    for frame in right._pending[0]['row']['frames'].values():
        frame['3d_bbox'].fill(123);frame['box_size'].fill(123)
    right._pending[0]['row']['annotations']['boxes'].fill(123)
    right._pending[0]['current_gt'].fill(123)
    left.commit(output,a);right.commit(deepcopy(output),b)
    l,r=left.states[row['request'].state_key],right.states[row['request'].state_key]
    np.testing.assert_array_equal(l.events[-1].box,r.events[-1].box)
    np.testing.assert_array_equal(l.memory.recent[-1]['ids'],r.memory.recent[-1]['ids'])
    np.testing.assert_array_equal(l.memory.recent[-1]['points'],r.memory.recent[-1]['points'])
    with pytest.raises(RuntimeError,match='duplicate commit'):
        left.commit(output,a)
    next_row=raw[request(2)];changed=deepcopy(next_row)
    for frame in changed['frames'].values():
        frame['3d_bbox'].fill(123);frame['box_size'].fill(123)
    changed['annotations']['boxes'].fill(123)
    next_a,next_b=left.prepare([next_row]),right.prepare([changed])
    for key in ('points','point_valid','point_ids','history_boxes','anchor_box','fallback_box',
                'event_boxes','event_times','event_edge_valid','event_geometry',
                'memory_points','memory_valid','memory_ids','memory_metadata','trusted_velocity'):
        torch.testing.assert_close(next_a[key],next_b[key],rtol=0,atol=0)


def test_sparse_measurement_refreshes_event_age_but_not_memory_age():
    raw=RawEndpointDataset(TinySource());builder=BatchBuilder(cfg())
    row=raw[request(1)];batch=builder.prepare([row]);out=output_for(batch)
    out.verification.support_mask[:,0,1:]=False
    out.verification.support_probabilities[:,1:]=.1
    out.decision.memory_eligible[:]=False
    record=builder.commit(out,batch,diagnostics=True)[0]
    assert record['measurement_accepted'] and not record['memory_write']
    assert record['memory_world'] is None
    assert record['measurement_age']==0. and record['memory_age']==pytest.approx(.5)
    next_batch=builder.prepare([raw[request(2)]])
    assert next_batch['diagnostic_measurement_age'].item()==pytest.approx(.5)
    assert next_batch['diagnostic_memory_age'].item()==pytest.approx(1.)
    without=BatchBuilder(cfg(v36_arm='C')).prepare([RawEndpointDataset(TinySource(),memory_enabled=False)[request(1)]])
    assert not without['diagnostic_memory_age_valid'].item()


def test_geometry_counts_unique_ids_and_has_population_statistics():
    box=np.zeros(4);size=np.array([4.,2.,2.])
    xyz=np.array([[0.,0.,0.],[1.,0.,0.],[1.,0.,0.],[8.,0.,0.]])
    value=geometry_summary(xyz,np.array([0,1,1,2]),box,size)
    np.testing.assert_allclose(value[:2],np.log1p([3,2])/np.log1p(1280))
    assert value[2]==pytest.approx(.125) and value[5]==pytest.approx(.125)


class EffectiveSource(TinySource):
    def get_frames(self,index,frame_ids):
        rows=super().get_frames(index,frame_ids)
        for row,frame in zip(rows,frame_ids):
            row['_ct_effective_timestamp']=10.+frame*.25
        return rows


class AnnotationVariantSource(TinySource):
    def __init__(self, *, box_shift=0., size_scale=1., clock_scale=1.):
        super().__init__((5,4))
        self.box_shift,self.size_scale,self.clock_scale=box_shift,size_scale,clock_scale

    def get_frames(self,index,frame_ids):
        rows=super().get_frames(index,frame_ids)
        for row,frame in zip(rows,frame_ids):
            row['3d_bbox'][0]+=self.box_shift
            row['box_size']*=self.size_scale
            row['_ct_effective_timestamp']=100.+frame*self.dt*self.clock_scale
        return rows


def test_physical_source_digest_matches_old_and_annotation_clock_is_separate():
    base=RawEndpointDataset(AnnotationVariantSource())
    assert base.source_sha256==LegacyRawEndpointDataset(AnnotationVariantSource()).source_sha256
    for changes in (dict(box_shift=2.),dict(size_scale=2.),dict(clock_scale=.5)):
        changed=RawEndpointDataset(AnnotationVariantSource(**changes))
        assert changed.source_sha256==base.source_sha256
        assert changed.annotation_sha256!=base.annotation_sha256
    loader=build_loaders(cfg(),roles=('train',),sources={'train':AnnotationVariantSource()})['train']
    assert loader.batch_sampler.state_dict()['annotation_sha256']==base.annotation_sha256


def test_event_clock_uses_effective_gap_and_annotation_origin_can_be_old():
    raw=RawEndpointDataset(EffectiveSource(),memory_enabled=False)
    builder=BatchBuilder(cfg(v36_arm='C'))
    for frame in range(1,4):
        row=raw[request(frame)];batch=builder.prepare([row])
        assert batch['current_dt'].item()==pytest.approx(.25)
        assert batch['supported_age'].item()==pytest.approx(frame*.25)
        out=output_for(batch,measurement=frame==3)
        out.decision.memory_eligible[:]=False
        builder.acquire(batch,out.prior)
        # 第3帧 raw 只含0/1/2/3；长期旧事件标签由独立注释表定位。
        np.testing.assert_array_equal(batch['prior_origin_target_world'][0],row['annotations']['boxes'][0])
        builder.commit(out,batch)
    state=builder.states[row['request'].state_key]
    assert state.last_velocity[0]==pytest.approx(.8)
    assert state.events[-1].effective_time==pytest.approx(10.75)
    assert state.events[-1].sensor_time==pytest.approx(101.5)


def test_relocation_breaks_only_motion_edge_and_next_positive_gap_restarts():
    raw=RawEndpointDataset(TinySource());builder=BatchBuilder(cfg())
    row=raw[request(1)];batch=builder.prepare([row]);builder.commit(output_for(batch),batch)
    state=builder.states[row['request'].state_key]
    moved=state.events[-1].box.copy();moved[0]=30.
    event=MeasurementEvent(moved,4,102.,102.,np.zeros(8,np.float32))
    assert not builder._append_event(state,event,np.zeros(4),np.ones(2),np.eye(2))
    assert state.events[-1].frame==4 and state.segment==1 and not state.velocity_valid
    next_box=moved.copy();next_box[0]+=.4
    next_event=MeasurementEvent(next_box,6,103.,103.,np.zeros(8,np.float32))
    assert builder._append_event(state,next_event,moved,np.ones(2),np.eye(2))
    assert state.velocity_valid and state.last_velocity[0]==pytest.approx(.4)
    assert next_event.segment==1


def test_motion_innovation_uses_rotated_box_plus_unclipped_rectangular_range():
    raw=RawEndpointDataset(TinySource());builder=BatchBuilder(cfg())
    row=raw[request(1)];batch=builder.prepare([row]);builder.commit(output_for(batch),batch)
    state=builder.states[row['request'].state_key]
    def edge(xy,prior_yaw,ranges,axes=np.eye(2)):
        event=MeasurementEvent(np.r_[xy,0.,prior_yaw],4,102.,102.,np.zeros(8,np.float32))
        return builder._append_event(deepcopy(state),event,np.array([0.,0.,0.,prior_yaw]),np.array(ranges),axes)
    # pi/2 旋转使 X 半尺寸由2变1，不能用统一半对角线扩大容差。
    assert not edge([1.5,0.],np.pi/2,[.1,.1])
    assert edge([1.,2.],np.pi/2,[.1,.1])
    # 同时落在两个轴的边缘仍处于矩形内，L2 椭圆规则会错误拒绝。
    assert edge([2.9,1.9],0.,[1.,1.])
    # 学习范围未截断；12/8只限制搜索预算，不限制物理测量入边。
    assert edge([21.,16.5],0.,[20.,16.])
    rotation=np.array([[0.,-1.],[1.,0.]])
    assert edge([-1.9,2.9],np.pi/2,[1.,1.],rotation)


def test_commit_rejects_forged_memory_eligibility_before_state_mutation():
    raw=RawEndpointDataset(TinySource());builder=BatchBuilder(cfg())
    row=raw[request(1)];batch=builder.prepare([row]);out=output_for(batch)
    state=builder.states[row['request'].state_key]
    out.decision.memory_eligible[:]=False
    with pytest.raises(ValueError,match='memory eligibility'):
        builder.commit(out,batch)
    assert state.next_frame==1 and len(state.events)==1


def test_commit_rejects_measurement_without_authorized_unique_support():
    raw=RawEndpointDataset(TinySource());builder=BatchBuilder(cfg())
    row=raw[request(1)];batch=builder.prepare([row]);out=output_for(batch)
    out.verification.authorized[:,0]=False
    out.decision.memory_eligible[:]=False
    with pytest.raises(ValueError,match='unique authorized'):
        builder.commit(out,batch)
    state=builder.states[row['request'].state_key]
    assert state.next_frame==1 and len(state.events)==1


class DenseSource(TinySource):
    def get_frames(self,index,frame_ids):
        rows=super().get_frames(index,frame_ids)
        x,y=np.meshgrid(np.linspace(-8.,8.,97),[-1.,0.,1.])
        cloud=np.stack((x.ravel(),y.ravel(),np.zeros(x.size)),-1)
        for row in rows:
            row['pc']=cloud+row['3d_bbox'][:3]
            row['point_ids']=np.arange(len(cloud))
        return rows


def test_worker_prefetch_keeps_raw_endpoint_order_and_sampling_identical():
    signatures=[]
    for workers in (0,4):
        config=cfg()
        config.workers=workers
        loader=build_loaders(config,roles=('train',),sources={'train':DenseSource((5,4))})['train']
        loader.batch_sampler.set_epoch(10)
        builder=BatchBuilder(config)
        digest=hashlib.sha256(repr(loader.batch_sampler.state_dict()).encode())
        for rows in loader:
            for row in rows:
                r=row['request']
                digest.update(repr(r).encode())
                for frame in sorted(row['frames']):
                    for key in ('pc','point_ids','3d_bbox','box_size'):
                        digest.update(row['frames'][frame][key].tobytes())
            batch=builder.prepare(rows)
            output=output_for(batch,measurement=False)
            builder.acquire(batch,output.prior)
            for key in sorted(batch):
                digest.update(key.encode());digest.update(batch[key].numpy().tobytes())
            builder.commit(output,batch)
        assert not builder.states
        signatures.append(digest.hexdigest())
    assert signatures[0]==signatures[1]
