"""Read-only v27 B0 failure slicing and deterministic frame-layout reproduction."""
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
RUN = ROOT / 'output/20260905-173106-27_b0-mini_car_seed42_60ep_bs16/lightning_logs/version_0/dev_diagnostics'

def number(r, k):
    return float(r.get(k) or 0)

def summarize(rows):
    n = len(rows)
    if not n:
        return {'n': 0}
    return dict(n=n,
        success=100*np.mean([number(r, 'final_success') for r in rows]),
        precision=100*np.mean([number(r, 'final_precision') for r in rows]),
        distance_mean=np.mean([number(r, 'final_distance') for r in rows]),
        distance_median=np.median([number(r, 'final_distance') for r in rows]),
        distance_over2=sum(number(r,'final_distance')>2 for r in rows),
        distance_over10=sum(number(r,'final_distance')>10 for r in rows),
        base_empty=sum(number(r,'acquisition_base_raw_point_count')==0 for r in rows),
        base_without_target=sum(number(r,'acquisition_base_raw_target_count')==0 for r in rows),
        globally_visible=sum(number(r,'acquisition_global_target_count_exact')>0 for r in rows))

rows = list(csv.DictReader((RUN/'epoch_60_endpoints.csv').open()))
queries = [r for r in rows if int(r['frame_id'])>0]
groups = defaultdict(list)
for r in queries:
    groups[r['tracklet_id']].append(r)
tracks=[]
for tid, rr in groups.items():
    first=next((i for i,r in enumerate(rr) if number(r,'final_distance')>2),None)
    first_r=rr[first] if first is not None else None
    after=[] if first is None else rr[first+1:]
    item=dict(tracklet_id=tid, **summarize(rr), first_over2_frame=None if first_r is None else int(first_r['frame_id']),
        first_over2_distance=None if first_r is None else number(first_r,'final_distance'),
        first_over2_global_targets=None if first_r is None else number(first_r,'acquisition_global_target_count_exact'),
        first_over2_base_targets=None if first_r is None else number(first_r,'acquisition_base_raw_target_count'),
        first_over2_base_points=None if first_r is None else number(first_r,'acquisition_base_raw_point_count'),
        recovered_within2_after_first_over2=sum(number(r,'final_distance')<=2 for r in after),
        first_five=[dict(frame=int(r['frame_id']), distance=number(r,'final_distance'),
            base_points=number(r,'acquisition_base_raw_point_count'),base_targets=number(r,'acquisition_base_raw_target_count'),
            global_targets=number(r,'acquisition_global_target_count_exact')) for r in rr[:5]])
    tracks.append(item)
slices={
    'all_queries':summarize(queries),
    'first_queries':summarize([r for r in queries if int(r['frame_id'])==1]),
    'first_five_queries':summarize([r for r in queries if int(r['frame_id'])<=5]),
    'globally_invisible':summarize([r for r in queries if number(r,'acquisition_global_target_count_exact')==0]),
    'base_empty':summarize([r for r in queries if number(r,'acquisition_base_raw_point_count')==0]),
    'base_background_only':summarize([r for r in queries if number(r,'acquisition_base_raw_point_count')>0 and number(r,'acquisition_base_raw_target_count')==0]),
    'base_1_to_2_targets':summarize([r for r in queries if 0<number(r,'acquisition_base_raw_target_count')<=2]),
    'base_3_to_10_targets':summarize([r for r in queries if 2<number(r,'acquisition_base_raw_target_count')<=10]),
    'base_over10_targets':summarize([r for r in queries if number(r,'acquisition_base_raw_target_count')>10]),
}
epochs={}
for p in sorted(RUN.glob('epoch_*_endpoints.csv')):
    rr=[r for r in csv.DictReader(p.open()) if int(r['frame_id'])>0]
    epochs[p.stem]=summarize(rr)

# x is exactly the logical layout consumed by the implementation: B,C,L*N.
# Distinct channel/frame/point values make the intended identity unambiguous.
B,C,L,N=2,14,4,3
x=(torch.arange(B)[:,None,None,None]*100000 +
   torch.arange(C)[None,:,None,None]*1000 +
   torch.arange(L)[None,None,:,None]*100 +
   torch.arange(N)[None,None,None,:]).reshape(B,C,L*N)
actual=x.reshape(B*L,-1,N)
correct=x.reshape(B,C,L,N).permute(0,2,1,3).reshape(B*L,C,N)
assert not torch.equal(actual,correct)
assert torch.equal(correct[3,0], torch.tensor([300,301,302]))
current_channel_frame=[dict(output_channel=i,source_channel=int(actual[3,i,0]//1000),
                            source_frame=int(actual[3,i,0]%1000//100)) for i in range(C)]
layout=dict(B=B,C=C,L=L,N=N, actual_first_frame_first_point=actual[0,:,0].tolist(),
    intended_first_frame_first_point=correct[0,:,0].tolist(),
    actual_current_frame_first_point=actual[3,:,0].tolist(),
    intended_current_frame_first_point=correct[3,:,0].tolist(),
    current_output_channel_sources=current_channel_frame,
    mismatched_elements=int((actual!=correct).sum()),total_elements=actual.numel(),
    current_frame_xyz_source_present=any(e['source_channel']<3 for e in current_channel_frame),
    current_frame_time_source_present=any(e['source_channel']==3 for e in current_channel_frame))
result=dict(slices=slices,tracks=tracks,epochs=epochs,layout_reproduction=layout)
(OUT/'b0_failure_followup.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result,indent=2))
