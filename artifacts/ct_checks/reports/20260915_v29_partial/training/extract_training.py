"""Read-only extraction of the three unfinished v29 perf runs (no checkpoints)."""
from pathlib import Path
from collections import defaultdict
import csv
import gzip
import json
import math
import re
import statistics
import struct

from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
from tensorboard.util import tensor_util

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
RUNS = sorted((ROOT / 'output').glob('20260911-0230*'))


def scalar(value):
    if value.HasField('simple_value'):
        return float(value.simple_value)
    if value.HasField('tensor'):
        a = tensor_util.make_ndarray(value.tensor)
        if a.size == 1 and a.dtype.kind in 'fiub':
            return float(a.reshape(-1)[0])
    return None


def main():
    data = {}
    report = {'scope': 'local snapshot; no server status or model-state equivalence claim', 'runs': {}}
    with gzip.open(OUT / 'all_scalars.csv.gz', 'wt', newline='', encoding='utf-8', compresslevel=4) as f:
        writer = csv.writer(f)
        writer.writerow(['arm', 'series', 'tag', 'step', 'wall_time', 'value', 'source'])
        for run in RUNS:
            arm = 'b0' if '-29_b0_' in run.name else ('full_cfc' if '_cfc_' in run.name else 'full_gru')
            base = run / 'lightning_logs/version_0'
            series = defaultdict(list)
            nonfinite = []
            for path in sorted(base.rglob('events.out*')):
                rel = path.parent.relative_to(base).as_posix()
                for event in EventFileLoader(str(path)).Load():
                    for value in event.summary.value:
                        number = scalar(value)
                        if number is None:
                            continue
                        key = value.tag if rel == '.' else rel + '::' + value.tag
                        row = (event.step, event.wall_time, number)
                        series[key].append(row)
                        writer.writerow([arm, rel, value.tag, *row, path.relative_to(ROOT).as_posix()])
                        if not math.isfinite(number):
                            nonfinite.append([key, *row])
            data[arm] = series
            log = (run / 'train.log').read_text(encoding='utf-8', errors='replace')
            progress = re.findall(r'Epoch\s+(\d+):[^\r\n]*?\|\s*(\d+)/(\d+)\s+\[([^\]]+)\]', log)
            last_by_epoch = {}
            for epoch, step, total, display in progress:
                if int(step) < last_by_epoch.get(int(epoch), {}).get('completed_batches', -1):
                    continue
                last_by_epoch[int(epoch)] = {'epoch_zero_based': int(epoch), 'completed_batches': int(step), 'total_batches': int(total), 'display': display}
            errors = [line for line in log.splitlines() if any(t in line for t in ['Traceback (most recent', 'RuntimeError:', 'Error:', 'max_epochs=60', 'NaN', 'nan detected'])]
            provenance = json.loads((run / 'run_provenance.json').read_text(encoding='utf-8'))
            supply = {p.stem: json.loads(p.read_text()) for p in sorted((base / 'acquisition_supply').glob('*.json'))}
            report['runs'][arm] = {
                'run': run.relative_to(ROOT).as_posix(),
                'log_bytes': (run / 'train.log').stat().st_size,
                'epochs_progress': last_by_epoch,
                'error_or_completion_lines': errors,
                'checkpoints_present_not_loaded': [p.relative_to(run).as_posix() for p in run.rglob('*.ckpt')],
                'event_files': len(list(base.rglob('events.out*'))),
                'nonfinite_scalars': nonfinite,
                'scalar_series_count': len(series),
                'datasets': provenance['datasets'],
                'training_streams': provenance['training_streams'],
                'supply': supply,
                'series': {},
            }
            for key, rows in series.items():
                rows.sort(key=lambda x: (x[0], x[1]))
                vals = [r[2] for r in rows if math.isfinite(r[2])]
                chunks = defaultdict(list)
                for step, wall, val in rows:
                    if math.isfinite(val):
                        chunks[step // 50687].append(val)
                report['runs'][arm]['series'][key] = {
                    'count': len(rows), 'first': rows[0], 'last': rows[-1],
                    'min': min(vals) if vals else None, 'max': max(vals) if vals else None,
                    'arithmetic_mean_of_logged_samples': statistics.fmean(vals) if vals else None,
                    'by_step_epoch': {str(e): {'count': len(v), 'sample_mean': statistics.fmean(v), 'min': min(v), 'max': max(v)} for e, v in chunks.items()},
                    'epoch_points': rows if key.startswith('ct_epoch_') or key.endswith('_epoch') else None,
                }
            print(arm, 'series', len(series), 'nonfinite', len(nonfinite), 'last', list(last_by_epoch.values())[-1], flush=True)
    comparisons = {}
    obskeys = [k for k in data['b0'] if k.startswith('loss_observation_') and ('::loss_observation' in k) and not any(x in k for x in ['elapsed_ms'])]
    for other in ['full_cfc', 'full_gru']:
        results = {}
        for key in obskeys:
            if key not in data[other]:
                continue
            left = {r[0]: r[2] for r in data['b0'][key]}
            right = {r[0]: r[2] for r in data[other][key]}
            common = sorted(left.keys() & right.keys())
            mismatch = [s for s in common if struct.pack('!d', left[s]) != struct.pack('!d', right[s])]
            results[key] = {'common_count': len(common), 'first_step': common[0] if common else None, 'last_step': common[-1] if common else None, 'unequal_logged_values': len(mismatch), 'max_abs_difference': max((abs(left[s]-right[s]) for s in common), default=0), 'first_mismatch': ([mismatch[0], left[mismatch[0]], right[mismatch[0]]] if mismatch else None)}
        comparisons['b0_vs_' + other] = results
    report['same_step_logged_observation_comparison'] = comparisons
    (OUT / 'training_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
