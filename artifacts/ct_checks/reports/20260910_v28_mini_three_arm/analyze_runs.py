"""只读汇总 v28 三组已完成的 mini 运行；输出与原实验目录分离。"""
from pathlib import Path
import csv
import hashlib
import json
import math
import platform
from collections import Counter, defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
DEST = Path(__file__).resolve().parent
RUNS = {
    'B0 seed42': '20260909-003318-28_b0-mini_car_seed42_60ep_bs16',
    'B0 seed52': '20260909-003318-28_b0-mini_car_seed52_60ep_bs16',
    'Full seed42 (uncalibrated)': '20260909-003318-28_full-mini_car_seed42_60ep_bs16',
}
REFERENCE = {'S': 50.9857788, 'P': 59.9617119}
TARGET = {'S': 49.986, 'P': 58.962}


def write_json(name, data):
    (DEST / name).write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                       allow_nan=False) + '\n', encoding='utf-8')


def write_csv(name, rows):
    with (DEST / name).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def flag(value):
    return str(value).lower() in ('true', '1', '1.0')


def compact_recovery(data):
    return {key: ({k: v for k, v in value.items() if k not in ('tracklets', 'episodes')}
                  if isinstance(value, dict) else value)
            for key, value in data.items()}


def inspect():
    sources, curves, final, endpoints, statistics = set(), [], [], {}, {}
    for label, directory in RUNS.items():
        run = ROOT / 'output' / directory
        diagnostics = run / 'lightning_logs/version_0/dev_diagnostics'
        for name in ('resolved_config.yaml', 'run_provenance.json', 'train.log'):
            sources.add(run / name)
        epochs = []
        for path in sorted(diagnostics.glob('epoch_*_summary.json')):
            epoch = int(path.stem.split('_')[1])
            epochs.append(epoch)
            summary = json.loads(path.read_text(encoding='utf-8'))
            sources.add(path)
            row_path = path.with_name(f'epoch_{epoch:02d}_endpoints.csv')
            rows = read_rows(row_path)
            sources.add(row_path)
            n = len(rows)
            score = {metric: 100 * math.fsum(float(row[field]) for row in rows) / n
                     for metric, field in (('S', 'final_success'), ('P', 'final_precision'))}
            for metric, value in score.items():
                assert math.isclose(value, summary['metrics'][metric], abs_tol=1e-8), (label, epoch, metric)
            grouped = defaultdict(list)
            for row in rows:
                grouped[row['tracklet_id']].append(row)
            assert n == 2285 and len(grouped) == 106, (label, epoch, n, len(grouped))
            for track, sequence in grouped.items():
                ids = sorted(int(row['frame_id']) for row in sequence)
                assert ids == list(range(len(sequence))), (label, epoch, track)
                assert sum(flag(row['is_initial']) for row in sequence) == 1
            curves.append(dict(run=label, epoch=epoch, S=score['S'], P=score['P'],
                               frames=n, tracklets=len(grouped),
                               source=path.relative_to(ROOT).as_posix()))
            if epoch == 60:
                endpoints[label] = rows
                final.append(dict(run=label, S=score['S'], P=score['P'],
                    delta_historical_S=score['S'] - REFERENCE['S'],
                    delta_historical_P=score['P'] - REFERENCE['P'],
                    passes_historical_target=all(score[key] >= TARGET[key] for key in ('S', 'P')),
                    late3_S=None, late3_P=None, frames=n, prediction_frames=n-len(grouped),
                    tracklets=len(grouped), actions=sum(flag(row['action_applied']) for row in rows)))
                distance = np.array([float(row['final_distance']) for row in rows if not flag(row['is_initial'])])
                statistics[label] = dict(
                    endpoint_metadata=dict(partition_counts=dict(Counter(row['partition'] for row in rows)),
                        scene_counts=dict(Counter(row['scene_id'] for row in rows)),
                        caveat='dev/unknown are legacy diagnostic metadata; provenance establishes official two-scene mini_val'),
                    distance_m=dict(scope='prediction frames only', median=float(np.median(distance)),
                        fraction_gt_0_5=float(np.mean(distance > .5)),
                        fraction_gt_1=float(np.mean(distance > 1)),
                        fraction_gt_2=float(np.mean(distance > 2))),
                    recovery=compact_recovery(summary['recovery']),
                    runtime=summary['runtime'])
        assert epochs == list(range(5, 61, 5)), (label, epochs)
        for epoch in (58, 59, 60):
            path = run / f'formal_checkpoints/epoch={epoch:03d}.ckpt'
            assert path.is_file(), path
            sources.add(path)

    # 同一轨迹/帧配对，不把两种递归轨迹中的点数桶当作相同样本总体。
    indexed = {name: {(row['tracklet_id'], int(row['frame_id'])): row for row in rows}
               for name, rows in endpoints.items()}
    a, b, full = (indexed[name] for name in RUNS)
    assert a.keys() == b.keys() == full.keys()
    fields = ('final_success', 'final_precision', 'final_iou', 'final_distance',
              'observation_success', 'observation_precision', 'observation_iou', 'observation_distance')
    full_identical = all(a[key][field] == full[key][field] for key in a for field in fields)
    assert full_identical, 'Uncalibrated Full/B0 endpoint metrics differ'
    tracks = defaultdict(list)
    for key in a:
        tracks[key[0]].append(key)
    paired = []
    for track, keys in tracks.items():
        ds = math.fsum(float(b[key]['final_success']) - float(a[key]['final_success']) for key in keys)
        dp = math.fsum(float(b[key]['final_precision']) - float(a[key]['final_precision']) for key in keys)
        paired.append(dict(tracklet_id=track, frames=len(keys),
            delta_S_within_track_pp=100 * ds / len(keys), delta_P_within_track_pp=100 * dp / len(keys),
            contribution_to_global_delta_S_pp=100 * ds / len(a),
            contribution_to_global_delta_P_pp=100 * dp / len(a)))
    paired.sort(key=lambda row: row['contribution_to_global_delta_P_pp'], reverse=True)
    gap = {key: final[1][key] - final[0][key] for key in ('S', 'P')}
    for key in ('S', 'P'):
        assert math.isclose(math.fsum(row[f'contribution_to_global_delta_{key}_pp'] for row in paired),
                            gap[key], abs_tol=1e-8)
    result = dict(historical_reference=REFERENCE, recovery_target=TARGET, final=final,
        late3_status='missing epoch58 and epoch59 evaluations; checkpoints alone are insufficient',
        full_vs_b0_seed42_endpoint_metrics_identical=full_identical,
        seed52_minus_seed42_pp=gap,
        seed_summary_descriptive_only={key: (final[0][key] + final[1][key]) / 2 for key in ('S', 'P')},
        paired_tracks=dict(total=len(paired),
            higher_P=sum(row['delta_P_within_track_pp'] > 1e-10 for row in paired),
            lower_P=sum(row['delta_P_within_track_pp'] < -1e-10 for row in paired),
            unchanged_P=sum(abs(row['delta_P_within_track_pp']) <= 1e-10 for row in paired),
            top10_positive_P_contribution_pp=sum(row['contribution_to_global_delta_P_pp'] for row in paired[:10])),
        statistics=statistics,
        verdict='seed52 meets historical final target; seed42 fails. Stable B0 recovery and matched-reference/late3 acceptance are incomplete.')
    write_json('summary.json', result)
    write_csv('validation_curve.csv', curves)
    write_csv('final_metrics.csv', final)
    write_csv('paired_tracklet_seed_deltas.csv', paired)
    source_manifest = []
    for path in sorted(sources):
        hasher = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                hasher.update(chunk)
        source_manifest.append(dict(path=path.relative_to(ROOT).as_posix(),
                                    bytes=path.stat().st_size, sha256=hasher.hexdigest()))
    write_json('source_manifest.json', source_manifest)
    return result, curves


def plot(curves):
    styles = [('B0 seed42', '#0072B2', 'o', '-'),
              ('B0 seed52', '#D55E00', 's', '-'),
              ('Full seed42 (uncalibrated)', '#009E73', 'x', ':')]
    with plt.rc_context({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42}):
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), layout='constrained')
        for ax, key, title in zip(axes, ('S', 'P'), ('Success', 'Precision')):
            for label, color, marker, line in styles:
                values = [row for row in curves if row['run'] == label]
                ax.plot([row['epoch'] for row in values], [row[key] for row in values],
                        color=color, marker=marker, linestyle=line, label=label,
                        linewidth=1.5, markersize=5)
            ax.axhline(REFERENCE[key], color='#333333', linestyle='--', linewidth=1,
                       label='Historical reference (not protocol-matched)')
            ax.axhline(TARGET[key], color='#777777', linestyle=':', linewidth=1,
                       label='Recovery target: reference - 1 pp')
            ax.set(xlabel='Completed training epochs', ylabel=f'{title} (%)',
                   title=title, xlim=(3, 62), ylim=(0, 75), xticks=range(5, 61, 5))
            ax.grid(alpha=.2)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside lower center', ncol=2, frameon=False)
        fig.suptitle('v28 mini Car | official mini_val | 106 tracklets / 2,285 frames\n'
                     'Full is uncalibrated and overlaps B0 seed42; epoch58/59 scores are missing', fontsize=11)
        fig.savefig(DEST / 'validation_curve.png', dpi=200, facecolor='white')
        fig.savefig(DEST / 'validation_curve.pdf', facecolor='white')
        plt.close(fig)
    write_json('figure_manifest.json', dict(
        source='validation_curve.csv', audience='research experiment audit; no target publisher specified',
        transformations='raw scheduled validation scores at epochs5,10,...,60; connecting segments guide the eye; no smoothing, interpolation estimates or seed selection',
        missing='no epoch58/59 evaluations; no late3 values inferred; no unscheduled validation points plotted',
        uncertainty='none: two B0 training seeds and one uncalibrated Full run, no inferential interval',
        units='percent; differences in percentage points',
        baseline='historical reference and reference-minus-one recovery target, not protocol-matched measured controls',
        alt_text='Success and Precision validation curves. Seed52 finishes at 52.88/64.48, seed42 at 45.20/47.24. Uncalibrated Full exactly overlaps seed42. Seed42 declines after intermediate epochs and misses both recovery targets.',
        python=platform.python_version(), matplotlib=matplotlib.__version__, numpy=np.__version__,
        formats=['PNG 2300x960 pixels at 200 dpi', 'PDF 11.5x4.8 inches']))


if __name__ == '__main__':
    summary, curve = inspect()
    plot(curve)
    print(json.dumps({key: value for key, value in summary.items() if key != 'statistics'},
                     ensure_ascii=False, indent=2))
