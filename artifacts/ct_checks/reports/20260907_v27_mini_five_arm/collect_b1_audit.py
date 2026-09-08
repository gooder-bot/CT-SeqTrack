"""Read-only B1 audit; run from CT-SeqTrack repository root."""
from pathlib import Path
import csv
import json
import math
from collections import Counter

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

OUT = Path(__file__).resolve().parent
RUNS = {
    'B1-CfC': '20260905-173109-27_b1_cfc-mini_car_seed42_60ep_bs16',
    'B1-GRU': '20260905-173114-27_b1_gru-mini_car_seed42_60ep_bs16',
    'B1+B2': '20260905-173118-27_full_minus_b3-mini_car_seed42_60ep_bs16',
    'Full': '20260905-173121-27_full-mini_car_seed42_60ep_bs16',
}

def aggregate(rows):
    valid = [r for r in rows if float(r['b1_valid']) > 0]
    def total(key, group=rows):
        return sum(float(r[key]) for r in group)
    out = {'rows': len(rows), 'b1_valid_rows': len(valid),
           'source_counts': dict(Counter(r['search_geometry_source_id'] for r in rows))}
    for key in ('learned_motion_error', 'kinematic_error', 'b1_nll', 'b1_coverage_50',
                'b1_coverage_80', 'b1_coverage_95', 'sigma_parallel', 'sigma_perpendicular',
                'learned_cv_disagreement', 'support_actual_length', 'support_actual_width'):
        values = [float(r[key]) for r in valid]
        out[key] = {'mean': float(np.mean(values)), 'median': float(np.median(values)),
                    'rmse': math.sqrt(float(np.mean(np.square(values))))}
    for key in ('global_target_count_exact', 'base_raw_target_count', 'global_novel_target_count',
                'support_raw_target_count', 'support_novel_target_count', 'pool_target_count',
                'pool_background_count', 'prepool_target_count', 'prepool_background_count',
                'support_xy_target_count', 'support_z_clip_target_count'):
        out[key] = total(key)
    out['novel_recall_of_global_novel'] = total('pool_target_count') / max(1, total('global_novel_target_count'))
    out['pool_target_bearing_rows'] = sum(float(r['pool_target_count']) > 0 for r in rows)
    out['novel_target_bearing_rows'] = sum(float(r['global_novel_target_count']) > 0 for r in rows)
    out['observation_error_buckets'] = []
    for lo, hi in ((0, 1), (1, 2), (2, 5), (5, 10), (10, float('inf'))):
        group = [r for r in rows if lo <= float(r['observation_error']) < hi]
        out['observation_error_buckets'].append({
            'lower_m': lo, 'upper_m': hi if math.isfinite(hi) else None, 'rows': len(group),
            'global_target': total('global_target_count_exact', group),
            'global_novel_target': total('global_novel_target_count', group),
            'base_target': total('base_raw_target_count', group),
            'novel_obtained': total('pool_target_count', group),
            'pool_target_bearing_rows': sum(float(r['pool_target_count']) > 0 for r in group)})
    keys = ('tracklet_id', 'frame_id', 'observation_error', 'global_target_count_exact',
            'base_raw_target_count', 'base_raw_point_count', 'pool_target_count',
            'extension_pool_count', 'search_geometry_source_id', 'search_geometry_valid',
            'support_actual_length', 'support_actual_width', 'corridor_valid')
    out['cold_start_rows'] = [{k: r[k] for k in keys} for r in rows if int(r['frame_id']) <= 2]
    return out

result = {'scope': 'mini dev, 1 held-out scene / 12 tracklets; unequal deployed histories',
          'metric_warning': 'learned_motion_error and b1_nll in existing candidate CSV use current GT relative to predicted reference; they are endpoint localization metrics, not physical displacement validation',
          'runs': {}}
for arm, run in RUNS.items():
    root = Path('output') / run / 'lightning_logs/version_0'
    arm_data = {'run': run, 'dev_epochs': {}, 'late3_training_logged_batch_statistics': {}}
    for epoch in range(5, 61, 5):
        with (root / f'candidate_diagnostics/epoch_{epoch:02d}.csv').open() as handle:
            arm_data['dev_epochs'][str(epoch)] = aggregate(list(csv.DictReader(handle)))
    events = EventAccumulator(str(root), size_guidance={'scalars': 0})
    events.Reload()
    epoch_steps = events.Scalars('epoch')
    # Derive update budget from logged completed epoch boundaries.
    epoch0_last_step = max(x.step for x in epoch_steps if x.value == 0)
    steps_per_epoch = epoch0_last_step + 1
    arm_data['steps_per_epoch'] = steps_per_epoch
    for metric in ('ct_acquisition_margin_parallel_mean', 'ct_acquisition_margin_perpendicular_mean',
                   'loss_ct_acquisition_margin', 'motion_v3_prior_rmse', 'motion_v3_kinematic_rmse',
                   'motion_v3_sigma_parallel_mean', 'motion_v3_sigma_perpendicular_mean',
                   'motion_v3_gaussian_nll', 'motion_v3_coverage_95'):
        acc = EventAccumulator(str(root / ('loss_' + metric)), size_guidance={'scalars': 0})
        acc.Reload()
        values = [e.value for e in acc.Scalars('loss') if e.step >= 57 * steps_per_epoch]
        nonzero = [v for v in values if v != 0]
        arm_data['late3_training_logged_batch_statistics'][metric] = {
            'count': len(values), 'zero_count': len(values) - len(nonzero),
            'mean_including_zero_masked_batches': float(np.mean(values)),
            'mean_nonzero_batches': float(np.mean(nonzero)) if nonzero else None,
            'median_nonzero_batches': float(np.median(nonzero)) if nonzero else None}
    result['runs'][arm] = arm_data
    print(arm, 'finished', flush=True)
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'b1_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
