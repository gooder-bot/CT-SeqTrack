"""Read-only partial-run diagnostic audit. Training/output files are never changed."""
from pathlib import Path
import csv
import json
import math
from collections import Counter, defaultdict
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
RUNS = dict(b0='20260911-023000-29_b0_perf-nuscenes_car_seed42_60ep_bs16',
            full_cfc='20260911-023006-29_full_cfc_perf-nuscenes_car_seed42_60ep_bs16',
            full_gru='20260911-023010-29_full_gru_perf-nuscenes_car_seed42_60ep_bs16')
STEPS = 50687

def write_csv(name, rows):
    rows = list(rows)
    if not rows:
        return
    with (OUT / name).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)

def main():
    inventory, events, epoch_values, supply, summaries = [], [], [], [], []
    availability = {}
    for arm, name in RUNS.items():
        run = ROOT / 'output' / name
        logs = run / 'lightning_logs' / 'version_0'
        csvs = list(run.rglob('*.csv'))
        diagnostic_dirs = [str(x.relative_to(run)) for x in run.rglob('*')
                           if x.is_dir() and 'dev_diagnostics' in x.name]
        availability[arm] = dict(run=name, csv_files=len(csvs), dev_diagnostic_dirs=diagnostic_dirs,
                                acquisition_epochs=[], validation_tags=[])
        for file in sorted((logs / 'acquisition_supply').glob('*.json')):
            obj = json.loads(file.read_text(encoding='utf-8'))
            availability[arm]['acquisition_epochs'].append(obj['epoch'])
            for population, values in obj['populations'].items():
                supply.append(dict(arm=arm, epoch=obj['epoch'], population=population,
                                   **values, **{'balance_' + k: v for k,v in obj['targetness_balance'].items()}))
        for file in sorted(logs.rglob('events.out*')):
            group = str(file.parent.relative_to(logs)).replace('\\', '/')
            selected = (group == '.' or group.startswith(('ct_epoch_calibration_sampled',
                        'ct_epoch_h3_sampling', 'ct_relation_sampled')) or
                        (group.startswith(('loss_mechanism_', 'loss_observation_'))
                         and not group.startswith(('loss_mechanism_loss_', 'loss_observation_loss_'))))
            if not selected:
                continue
            accumulator = EventAccumulator(str(file), size_guidance={'scalars': 0})
            accumulator.Reload()
            for tag in accumulator.Tags()['scalars']:
                if group == '.' and not any(x in tag for x in ('epoch', '/dev', 'val', 'success', 'precision')):
                    continue
                vals = accumulator.Scalars(tag)
                if any(x in tag.lower() for x in ('/dev', '/val', 'success', 'precision')):
                    availability[arm]['validation_tags'].append(tag)
                key_counts = Counter(x.step for x in vals)
                inventory.append(dict(arm=arm, group=group, tag=tag, file=str(file.relative_to(ROOT)),
                                      records=len(vals), first_step=vals[0].step if vals else None,
                                      last_step=vals[-1].step if vals else None,
                                      duplicate_steps=sum(n-1 for n in key_counts.values() if n > 1),
                                      nonfinite=sum(not math.isfinite(x.value) for x in vals),
                                      step_decreases=sum(b.step < a.step for a,b in zip(vals, vals[1:]))))
                by_epoch = defaultdict(list)
                for x in vals:
                    epoch_end = group.startswith('ct_epoch_') or tag.endswith('_epoch')
                    epoch = max(1, math.ceil(x.step/STEPS)) if epoch_end else x.step//STEPS + 1
                    row = dict(arm=arm, epoch=epoch, step=x.step, wall_time=x.wall_time,
                               group=group, tag=tag, value=x.value)
                    events.append(row)
                    by_epoch[epoch].append(x.value)
                    if epoch_end:
                        epoch_values.append(row)
                for epoch, vals_epoch in by_epoch.items():
                    arr = np.asarray(vals_epoch)
                    finite = arr[np.isfinite(arr)]
                    summaries.append(dict(arm=arm, epoch=epoch, group=group, tag=tag,
                        logged_batches=len(arr), finite=len(finite),
                        arithmetic_logged_mean=float(finite.mean()) if len(finite) else None,
                        p10=float(np.quantile(finite,.1)) if len(finite) else None,
                        p50=float(np.quantile(finite,.5)) if len(finite) else None,
                        p90=float(np.quantile(finite,.9)) if len(finite) else None,
                        zero_logged_batches=int((finite == 0).sum()),
                        interpretation='sampled batch-mean distribution, not all-row or full epoch prevalence'))
    write_csv('event_file_inventory.csv', inventory)
    write_csv('diagnostic_events.csv', events)
    write_csv('sampled_batch_summary.csv', summaries)
    write_csv('epoch_diagnostic_values.csv', epoch_values)
    write_csv('acquisition_supply.csv', supply)
    (OUT/'validation_availability.json').write_text(json.dumps(availability, indent=2), encoding='utf-8')
    print(json.dumps(dict(availability=availability, inventory=len(inventory), event_records=len(events),
                         duplicated_steps=sum(x['duplicate_steps'] for x in inventory),
                         nonfinite=sum(x['nonfinite'] for x in inventory),
                         step_decreases=sum(x['step_decreases'] for x in inventory)), indent=2))

if __name__ == '__main__':
    main()
