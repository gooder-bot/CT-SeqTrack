"""Compact tables from the extracted TensorBoard summary; source files remain read-only."""
from pathlib import Path
import csv
import json
import yaml

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[4]


def write_csv(name, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    q = json.loads((OUT / 'training_summary.json').read_text(encoding='utf-8'))
    epochs, contributions, comparisons, sampled = [], [], [], []
    weights = {'loss_seg': 'seg_weight', 'loss_bc': 'bc_weight',
               'loss_center': 'center_weight', 'loss_center_aux': 'center_weight',
               'loss_center_motion': 'center_weight', 'loss_center_ref': 'ref_center_weight',
               'loss_angle': 'angle_weight', 'loss_angle_aux': 'angle_weight',
               'loss_angle_motion': 'angle_weight', 'loss_angle_ref': 'ref_angle_weight',
               'loss_motion_cls': 'motion_cls_seg_weight'}
    for arm, run in q['runs'].items():
        cfg = yaml.safe_load((ROOT / run['run'] / 'resolved_config.yaml').read_text(encoding='utf-8'))
        series = run['series']
        for key, value in series.items():
            if not value['epoch_points']:
                continue
            for step, wall, val in value['epoch_points']:
                ep = (step // 50687) if key.startswith('ct_epoch_') else ((step + 1) // 50687)
                epochs.append(dict(arm=arm, completed_epoch=ep, step=step, metric=key, value=val,
                                   population='sampled_diagnostic' if ('sampled' in key or 'h3' in key) else 'all_observation_or_mechanism_transactions',
                                   source=(run['run'] + '/lightning_logs/version_0/' + key.split('::')[0] if '::' in key else str(next((ROOT / run['run'] / 'lightning_logs/version_0').glob('events.out*')).relative_to(ROOT)).replace('\\', '/'))))
        for name, weight in weights.items():
            key = 'ct_epoch_core_loss_observation/' + name + '::ct_epoch_core_loss'
            totals = {row[0]: row[2] for row in series['ct_epoch_core_loss_observation/loss_total::ct_epoch_core_loss']['epoch_points']}
            for step, wall, val in series[key]['epoch_points']:
                contributions.append(dict(arm=arm, completed_epoch=step // 50687, loss=name, raw_epoch_mean=val,
                                          weight=cfg[weight], weighted_contribution=val * cfg[weight],
                                          percent_of_total=100 * val * cfg[weight] / totals[step]))
        for key, value in series.items():
            if key.startswith(('loss_observation_', 'loss_mechanism_')) and any(s in key for s in ('rollin_rows', 'teacher_rows', 'valid_history', 'num_points_search', 'extension_presence_target_rate', 'behavior_accepted', 'trajectory_search_valid', 'margin_', 'motion_v3_kinematic_rmse', 'motion_v3_prior_rmse')):
                for ep, stats in value['by_step_epoch'].items():
                    if int(ep) > 1:
                        continue
                    sampled.append(dict(arm=arm, epoch_zero_based=int(ep), metric=key, sample_count=stats['count'],
                                        mean_of_logged_batches=stats['sample_mean'], minimum_batch_mean=stats['min'], maximum_batch_mean=stats['max']))
    for pair, entries in q['same_step_logged_observation_comparison'].items():
        for key, info in entries.items():
            comparisons.append(dict(pair=pair, metric=key, **{k: v for k, v in info.items() if k != 'first_mismatch'}))
    write_csv('epoch_metrics.csv', epochs)
    write_csv('b0_weighted_loss_contributions.csv', contributions)
    write_csv('matched_observation_scalars.csv', comparisons)
    write_csv('sampled_diagnostics_first_two_epochs.csv', sampled)
    print('wrote compact tables:', len(epochs), len(contributions), len(comparisons), len(sampled))


if __name__ == '__main__':
    main()
