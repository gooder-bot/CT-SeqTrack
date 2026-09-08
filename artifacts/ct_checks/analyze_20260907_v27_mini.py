"""Read-only audit of the five completed v27 mini runs and historical SeqTrack.

Outputs are derived artifacts; experiment output directories are never changed.
No missing evaluations are imputed. Bootstrap uses paired tracklets, preserving
all their frames; it describes this one dev scene, not cross-scene/seed variation.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / 'artifacts/ct_checks/reports/20260907_v27_mini_five_arm'
DEST.mkdir(parents=True, exist_ok=True)
RUNS = {
    'B0': '20260905-173106-27_b0-mini_car_seed42_60ep_bs16',
    'B1-CfC': '20260905-173109-27_b1_cfc-mini_car_seed42_60ep_bs16',
    'B1-GRU': '20260905-173114-27_b1_gru-mini_car_seed42_60ep_bs16',
    'B1+B2': '20260905-173118-27_full_minus_b3-mini_car_seed42_60ep_bs16',
    'Full': '20260905-173121-27_full-mini_car_seed42_60ep_bs16',
}


def save_csv(name, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (DEST / name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def scalars(folder):
    ea = EventAccumulator(str(folder), size_guidance={'scalars': 0}).Reload()
    tags = ea.Tags()['scalars']
    if len(tags) != 1:
        raise ValueError((folder, tags))
    return ea.Scalars(tags[0])


summary, curves, by_track, endpoints, baseline, baseline_curves = [], [], [], {}, [], []
loss_curves = []
for arm, name in RUNS.items():
    run = ROOT / 'output' / name
    events = run / 'lightning_logs/version_0'
    prov = json.loads((run / 'run_provenance.json').read_text(encoding='utf-8'))
    s = scalars(events / 'metrics_dev_success')
    p = scalars(events / 'metrics_dev_precision')
    assert len(s) == len(p) == 12
    for i, (a, b) in enumerate(zip(s, p), 1):
        assert a.step == b.step == i * 5 * 1057
        curves.append(dict(arm=arm, epoch=5*i, global_step=a.step,
                           success=a.value, precision=b.value,
                           utility=(a.value+b.value)/2, partition='dev'))
    diag_path = events / 'dev_diagnostics/epoch_60_summary.json'
    diag = json.loads(diag_path.read_text(encoding='utf-8'))
    rows = list(csv.DictReader((events / 'dev_diagnostics/epoch_60_endpoints.csv').open(encoding='utf-8')))
    keys = [(r['tracklet_id'], r['frame_id']) for r in rows]
    assert len(set(keys)) == len(rows) == diag['metrics']['frames']
    endpoints[arm] = {(r['tracklet_id'], r['frame_id']): r for r in rows}
    view0 = scalars(events / 'loss_loss_b0_view0')
    assert len(view0) == 63420
    assert len({v.step for v in view0}) == len(view0)
    for epoch in range(1, 61):
        values = [v.value for v in view0 if (epoch-1)*1057 <= v.step < epoch*1057]
        assert len(values) == 1057 and np.isfinite(values).all()
        loss_curves.append(dict(arm=arm, epoch=epoch, view0_loss_mean=float(np.mean(values)),
                                batches=len(values), population='B0 observation canonical view0'))
    assert abs(100*np.mean([float(r['final_success']) for r in rows])-s[-1].value) < 1e-4
    assert abs(100*np.mean([float(r['final_precision']) for r in rows])-p[-1].value) < 1e-4
    row = dict(arm=arm, run=name, partition='dev', epoch=60, **diag['metrics'])
    row.update(n0_current=diag['funnel']['n0_current_frames'],
               structural_frames=diag['funnel']['structural_frames'],
               final_distance_mean_m=float(np.mean([float(r['final_distance']) for r in rows])),
               final_distance_median_m=float(np.median([float(r['final_distance']) for r in rows])),
               frames_distance_gt_2m=sum(float(r['final_distance']) > 2 for r in rows),
               max_epochs_60_reached='`max_epochs=60` reached' in (run/'train.log').read_text(encoding='utf-8',errors='replace'),
               official_mini_val_available=False,
               late3_58_59_60_evaluated=False,
               calibration_present=any('calibrat' in f.name for f in run.rglob('*.json') if f.name != 'run_provenance.json'))
    for stage, vals in diag['funnel']['stages'].items():
        for key in ('points', 'target_points', 'global_target_recall', 'target_fraction'):
            row[f'{stage}_{key}'] = vals[key]
    summary.append(row)
    for tid in sorted(set(r['tracklet_id'] for r in rows), key=int):
        rr = [r for r in rows if r['tracklet_id'] == tid]
        by_track.append(dict(arm=arm, tracklet_id=tid, frames=len(rr),
                             S=100*np.mean([float(r['final_success']) for r in rr]),
                             P=100*np.mean([float(r['final_precision']) for r in rr]),
                             U=50*np.mean([float(r['final_success'])+float(r['final_precision']) for r in rr]),
                             distance_mean_m=np.mean([float(r['final_distance']) for r in rr]),
                             first_distance_gt_2m=next((r['frame_id'] for r in rr if float(r['final_distance']) > 2), None)))

for run in sorted((ROOT.parent / 'seqtrack/output').glob('*60ep_bs16*')):
    events = run / 'lightning_logs/version_0'
    hp = yaml.load((events/'hparams.yaml').read_text(encoding='utf-8'), Loader=yaml.BaseLoader)
    cfg = hp.get('config', hp)
    cfg = cfg.get('dictitems', cfg)
    s = scalars(events/'metrics_test_success')
    p = scalars(events/'metrics_test_precision')
    assert len(s) == len(p)
    record = dict(run=run.name, partition='official mini_val (historical evaluator)',
                  S=s[-1].value, P=p[-1].value, U=(s[-1].value+p[-1].value)/2,
                  final_step=s[-1].step, validation_points=len(s),
                  paired_with_v27_dev=False,
                  **{k:cfg.get(k) for k in ('epoch','workers','seed','batch_size','check_val_every_n_epoch','train_split','val_split','version','num_candidates','lr')})
    baseline.append(record)
    for i, (a,b) in enumerate(zip(s,p),1):
        baseline_curves.append(dict(run=run.name, validation_index=i, global_step=a.step,
                                    success=a.value, precision=b.value))

# Paired endpoint coverage and trajectory bootstrap of finite-sample dev deltas.
base = endpoints['B0']
tids = sorted(set(k[0] for k in base), key=int)
paired = []
for arm, values in endpoints.items():
    assert values.keys() == base.keys()
    counts = np.array([sum(k[0] == t for k in base) for t in tids])
    sums = np.array([[sum(float(values[k]['final_'+m])-float(base[k]['final_'+m]) for k in base if k[0]==t)
                      for m in ('success','precision')] for t in tids])
    rng = np.random.default_rng(42)
    draws = rng.integers(len(tids),size=(10000,len(tids)))
    boots = 100*sums[draws].sum(1)/counts[draws].sum(1)[:,None]
    point = 100*sums.sum(0)/counts.sum()
    utility_boot = boots.mean(1)
    paired.append(dict(arm=arm, dS=point[0], dP=point[1], dU=point.mean(),
                       dU_ci_low=np.quantile(utility_boot,.025), dU_ci_high=np.quantile(utility_boot,.975),
                       bootstrap_unit='paired tracklet; one dev scene; not causal module isolation',
                       bootstrap_samples=10000, seed=42, tracklets=len(tids), frames=len(base)))

save_csv('dev_epoch60_summary.csv',summary)
save_csv('dev_validation_curves.csv',curves)
save_csv('b0_view0_training_loss.csv',loss_curves)
save_csv('dev_epoch60_tracklets.csv',by_track)
save_csv('historical_seqtrack_baselines.csv',baseline)
save_csv('historical_seqtrack_curves.csv',baseline_curves)
save_csv('dev_descriptive_paired_deltas.csv',paired)
(DEST/'metrics_summary.json').write_text(json.dumps(dict(summary=summary,baseline=baseline,paired=paired),indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(dict(summary=[{k:r[k] for k in ('arm','S','P','U','frames','tracklets','n0_current','structural_frames','actions','final_distance_median_m')} for r in summary],baseline=baseline,paired=paired),indent=2))
