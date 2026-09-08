"""Collect existing pure B0 observation logs; do not infer missing branch losses."""
import json
from pathlib import Path
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

OUT=Path(__file__).resolve().parent
RUN=OUT.parents[3]/'output/20260905-173106-27_b0-mini_car_seed42_60ep_bs16/lightning_logs/version_0'
STEPS=1057
components={'loss_bc':1.,'loss_seg':.1,'loss_center':2.,'loss_angle':10.,
            'loss_center_aux':2.,'loss_angle_aux':10.,'loss_center_motion':2.,
            'loss_angle_motion':10.,'loss_center_ref':.2,'loss_angle_ref':1.,
            'loss_motion_cls':.1}
series={}
inventory={}
for key in [*components,'loss_b0_transaction','loss_b0_view0','loss_b0_view1','loss_b0_view2','loss_b0_view3']:
    ea=EventAccumulator(str(RUN/('loss_'+key)),size_guidance={'scalars':0}).Reload()
    tags=ea.Tags()['scalars']
    assert len(tags)==1,(key,tags)
    events=ea.Scalars(tags[0])
    first={}
    for e in events:first.setdefault(e.step,e.value)
    inventory[key]=dict(events=len(events),distinct_steps=len(first),duplicate_steps=len(events)-len(first))
    assert set(first)==set(range(60*STEPS))
    series[key]=first
root=EventAccumulator(str(RUN),size_guidance={'scalars':0}).Reload()
accuracies={}
for k in root.Tags()['scalars']:
    if k.endswith('/train_epoch') and '_acc_' in k:
        ev=root.Scalars(k)
        assert len(ev)==60
        accuracies[k]={i+1:dict(step=e.step,value=e.value) for i,e in enumerate(ev)}

epochs={}
for epoch in (1,5,20,60):
    means={k:float(np.mean([s[i] for i in range((epoch-1)*STEPS,epoch*STEPS)])) for k,s in series.items()}
    epochs[epoch]=dict(raw_logged_component_means={k:means[k] for k in components},
        logged_batch_component_means_times_config_weight={k:means[k]*w for k,w in components.items()},
        actual_candidate_weighted_total=means['loss_b0_transaction'],
        actual_view_losses={str(i):means['loss_b0_view'+str(i)] for i in range(4)},
        logged_bc_once=means['loss_bc'],logged_bc_twice=2*means['loss_bc'],
        logged_training_accuracy={k:values[epoch] for k,values in accuracies.items()})
    weighted=.5*means['loss_b0_view0']+sum(means['loss_b0_view'+str(i)] for i in (1,2,3))/6
    assert abs(weighted-means['loss_b0_transaction'])<1e-5
result=dict(run=str(RUN),stream='B0-only observation; no mechanism in this arm',
    inventory=inventory,steps_per_epoch=STEPS,epochs=epochs,
    component_population='initial compute_loss over all four equal-count candidates, before candidate-weighted branch recomputation',
    total_population='weighted branch recomputation: 0.5*view0+(view1+view2+view3)/6',
    exact_weighted_component_reconstruction_possible=False,
    reasons=['per-view component logs absent','BC and segmentation normalize by valid slots/classes, whose per-view denominators are not logged',
             'do not sum all-candidate component means and call it actual weighted total',
             'logged_bc_twice describes local doubled coefficient magnitude; not exact candidate-weighted BC contribution'])
(OUT/'b0_loss_components.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result,indent=2))
