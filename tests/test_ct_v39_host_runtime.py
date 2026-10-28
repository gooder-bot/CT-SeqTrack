"""v39 七槽指标与 host 的 Adam/commit 事务顺序；不依赖 Lightning 安装。"""
from copy import deepcopy
from types import SimpleNamespace
import sys

import pytest
import torch

from models.ct_v39.config import normalize_config
from models.ct_v39.data import ReadyQueueBatchSampler, EndpointRequest, build_loaders
from models.ct_v39.runtime import TrackingEvaluation, resume_payload, validate_resume_payload
from models.ctseqtrackv39 import CTSEQTRACKV39
from tests.helpers.raw_tracking import TinySource


def config(**changes):
    return normalize_config(dict(ct_engineering_check=True, workers=0, batch_size=2,
        v39_attention_backend='math_reference', v39_feature_bf16=False,
        execution_profile='optimized_fp32', **changes))


def fixture_batch():
    b,n=2,6
    boxes=torch.zeros(b,7,4);boxes[:,6,0]=5.
    selected=torch.tensor([0,6]);index=torch.arange(b)
    pool=dict(xyz=torch.tensor([[[0.,0.,0.],[.1,.1,0.],[-.1,-.1,0.],
                                [5.,0.,0.],[5.1,.1,0.],[4.9,-.1,0.]]]).repeat(b,1,1),
              valid=torch.ones(b,n,dtype=torch.bool),point_ids=torch.arange(n).repeat(b,1))
    output=SimpleNamespace(candidates=SimpleNamespace(boxes=boxes,valid=torch.ones(b,7,dtype=torch.bool),
        raw_valid=torch.ones(b,7,dtype=torch.bool),measurement_eligible=torch.ones(b,7,dtype=torch.bool),
        alias_of=torch.full((b,7),-1)),selected_index=selected,accepted_box=boxes[index,selected],
        selected_quality=torch.ones(b),verification=dict(logits=torch.ones(b,7,3)*5,
            q_logits=torch.ones(b,7),targetness_logits=torch.ones(b,7,n)*5,
            region_no_match_logits=torch.ones(b,7)*-5),pool=pool,
        full_executed=torch.tensor([True,False]),policy_full=torch.tensor([True,False]),
        search_triggered=torch.tensor([False,True]))
    rows=[dict(request=EndpointRequest(0,i,4,1,2,1),tracklet_key=f'track/{i}',
        first_frame=dict(timestamp=0.,scene_id='scene'),frames={1:dict(timestamp=.5,scene_id='scene')}) for i in range(b)]
    tx=SimpleNamespace(inputs=dict(box_size=torch.ones(b,3)*2,anchor_box=torch.zeros(b,4,dtype=torch.float64)),
        targets=dict(target_box=torch.zeros(b,4),target_box_size=torch.ones(b,3)*2,
                     physical_displacement=torch.tensor([[0.,0.],[.2,0.]])),
        diagnostics=dict(diagnostic_target_count=torch.full((b,),3),diagnostic_crop_target_count=torch.tensor([3,0]),
                         diagnostic_crop_point_count=torch.tensor([6,0])))
    commits=[dict(trusted_write=True,supported=True,full_executed=bool(output.full_executed[i]),
                  trusted_velocity_updated=False) for i in range(b)]
    return rows,tx,output,commits


def test_seven_slot_eval_preserves_initialization_and_both_compute_flags():
    evaluation=TrackingEvaluation(config())
    rows,tx,output,commits=fixture_batch()
    evaluation.add_batch(rows,tx,output,commits)
    result=evaluation.summary()
    assert result['frames']==4 and result['prediction_frames']==2
    assert len(result['selection'])==7
    assert result['selection']['g_reacquire']['count']==1
    assert result['trusted']['writes']==2 and result['trusted']['wrong_writes']==1
    assert result['diagnostics']['selected_bad_available_good_frames']==1
    assert result['computation']['actual_full_rate']==.5
    assert result['same_state_oracle']['success']==pytest.approx(100.)
    prediction_rows=[row for row in evaluation.rows if not row['initialization']]
    assert prediction_rows[1]['diagnostic_crop_point_count'] is None
    assert prediction_rows[1]['diagnostic_g_pool_target_count']==3
    assert result['diagnostics']['raw_present_but_b0_crop_miss_frames']==0
    with pytest.raises(ValueError,match='complete and causal'):
        evaluation.add_batch(rows,tx,output,commits)


def test_eval_rejects_old_three_slot_output():
    rows,tx,output,commits=fixture_batch()
    output.candidates.boxes=output.candidates.boxes[:,:3]
    with pytest.raises(ValueError,match='seven'):
        TrackingEvaluation(config()).add_batch(rows,tx,output,commits)


def test_candidate_quality_and_trust_keep_source_semantics():
    rows,tx,output,commits=fixture_batch()
    output.q0_quality=torch.tensor([.2,.3])
    output.candidates.measurement_eligible[:,1:3]=False
    evaluation=TrackingEvaluation(config())
    evaluation.add_batch(rows,tx,output,commits)
    record=next(r for r in evaluation.rows if not r['initialization'])['diagnostic_candidates']
    assert record['candidates'][0]['state_quality']==pytest.approx(.2)
    assert record['candidates'][0]['quality_source']=='legacy_q0'
    assert record['candidates'][1]['trusted_score']==0.
    assert record['candidates'][1]['state_quality']==0.
    assert record['candidates'][3]['state_quality']==record['candidates'][3]['qG']


@pytest.mark.parametrize('real_q0', [False, True])
def test_recovered_bad_q0_requires_a_real_observation(real_q0):
    rows,tx,output,commits=fixture_batch()
    # 第二行选中正确恢复框，q0 槽保存一个坏框；轻路径没有执行 q0 时，
    # 该占位框不得增加“修复坏 q0”的机制计数。
    tx.targets['target_box'][1,0]=5.
    output.candidates.valid[1,0]=real_q0
    output.candidates.raw_valid[1,0]=real_q0
    output.candidates.measurement_eligible[1,0]=real_q0
    output.full_executed[1]=real_q0
    output.policy_full[1]=real_q0
    evaluation=TrackingEvaluation(config())
    evaluation.add_batch(rows,tx,output,commits)
    result=evaluation.summary()
    assert result['diagnostics']['recovered_bad_q0']==int(real_q0)


def test_resume_binds_source_recipe_and_complete_sampler(monkeypatch):
    import models.ct_v39.entry as entry
    source={'sha256':'f'*64,'files':{'test.py':'a'*64}}
    monkeypatch.setattr(entry,'source_identity',lambda:source)
    cfg=config()
    sampler=ReadyQueueBatchSampler([3,2],batch_size=2,short_window=4,long_window=8,
        curriculum_epochs=10,reserve_windows=0,initial_seed_policy=cfg.v35_seed_policy)
    payload=resume_payload(cfg,completed_epoch=1,complete=True,rows=sampler.row_count,
        steps=len(sampler),sampler=sampler.state_dict(),run_id='test_run')
    assert validate_resume_payload(payload,cfg)['completed_epoch']==1
    corrupted=deepcopy(payload);corrupted['epoch_complete']=False
    with pytest.raises(ValueError,match='complete epoch'):
        validate_resume_payload(corrupted,cfg)
    with pytest.raises(ValueError,match='identity mismatch'):
        validate_resume_payload(payload,config(v39_training_recipe='full_b'))
    corrupted=deepcopy(payload);corrupted['sampler']['rows']+=1
    with pytest.raises(ValueError,match='checksum'):
        validate_resume_payload(corrupted,cfg)


class FakeTracker(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor(2.))
    def forward_transaction(self,tx,builder):
        count=len(tx.rows)
        return SimpleNamespace(value=self.weight.square(),full_executed=torch.ones(count,dtype=torch.bool),
            policy_full=torch.zeros(count,dtype=torch.bool),search_triggered=torch.zeros(count,dtype=torch.bool))


class FakeBuilder:
    def __init__(self):
        self.pending=None;self.commits=0;self.aborts=0;self.states={}
    def begin(self,rows,device,*,training):
        self.pending=SimpleNamespace(rows=rows,training=training)
        return self.pending
    def commit(self,output,tx,**kwargs):
        assert tx is self.pending
        self.commits+=1;self.pending=None
    def abort(self):
        self.pending=None;self.aborts+=1


def test_host_commits_only_after_optimizer_and_aborts_failed_loss(monkeypatch):
    fake_losses=SimpleNamespace(compute_losses=lambda tx,out,model,cfg:dict(loss_total=out.value))
    monkeypatch.setitem(sys.modules,'models.ct_v39.losses',fake_losses)
    host=CTSEQTRACKV39(config(),tracker=FakeTracker())
    host.log=lambda *args,**kwargs:None
    builder=FakeBuilder();host.train_builder=builder
    rows=[dict(request=SimpleNamespace(drain=False)) for _ in range(2)]
    try:
        loss=host.training_step(rows,0)
        assert builder.commits==0 and builder.pending is not None and host._epoch_steps==0
        optimizer=host.configure_optimizers()['optimizer']
        loss.backward();optimizer.step()
        assert host.tracker.weight.item()<2.
        host.on_train_batch_end(loss,rows,0)
        assert builder.commits==1 and builder.pending is None and host._epoch_steps==1
        fake_losses.compute_losses=lambda *args:dict(loss_total=torch.tensor(float('nan')))
        with pytest.raises(FloatingPointError):
            host.training_step(rows,1)
        assert builder.aborts==1 and host._pending_train is None and builder.commits==1
    finally:
        host.close_raw_service()


def test_host_rebinds_injected_loaders_to_one_raw_service():
    cfg=config()
    train=build_loaders(cfg,roles=('train',),sources={'train':TinySource((3,))})
    test=build_loaders(cfg,roles=('test',),sources={'test':TinySource((3,))})
    unused=test['test'].dataset.service
    host=CTSEQTRACKV39(cfg,tracker=FakeTracker(),loaders=dict(train,**test))
    try:
        assert host.train_builder.service is host.evaluation_builder.service is host.raw_service
        assert all(loader.dataset.service is host.raw_service for loader in host._loaders.values())
    finally:
        host.close_raw_service();unused.close()
