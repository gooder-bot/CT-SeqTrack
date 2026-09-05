"""全帧评估报告：采集漏斗使用实测分母，初始 GT 帧仅进入 S/P。"""
import json
from pathlib import Path

import numpy as np


def summarize_endpoint_diagnostics(rows):
    from utils.action_calibration_v27 import summarize_rows
    rows = list(rows)
    metrics = summarize_rows(rows)
    predicted = [row for row in rows if not row['is_initial']]
    measured = [row for row in predicted if row.get('acquisition_available', False)]
    stages = {}
    for stage, target_key, point_key in (
            ('global', 'global_target_count_exact', 'global_raw_point_count'),
            ('base_raw', 'base_raw_target_count', 'base_raw_point_count'),
            ('base_sampled', 'base_sampled_target_count', 'base_sampled_point_count'),
            ('support_raw', 'support_raw_target_count', 'support_raw_point_count'),
            ('support_novel', 'support_novel_target_count', 'support_novel_point_count'),
            ('novel_pool', 'novel_pool_target_count', 'novel_pool_point_count'),
            ('prepool', 'prepool_target_count', 'prepool_point_count'),
            ('selected', 'selected_target_count', 'selected_point_count'),
            ('consensus_top_mode', 'consensus_target_count', 'consensus_point_count')):
        subset = [r for r in measured if 'acquisition_' + target_key in r and 'acquisition_' + point_key in r]
        targets = sum(float(r['acquisition_' + target_key]) for r in subset)
        points = sum(float(r['acquisition_' + point_key]) for r in subset)
        total_targets = sum(float(r['acquisition_global_target_count_exact']) for r in subset)
        stages[stage] = dict(measured_frames=len(subset), target_points=targets, points=points,
                             target_fraction=targets / points if points else None,
                             global_target_recall=targets / total_targets if total_targets else None)
    n0 = [r for r in measured if r.get('acquisition_base_raw_point_count') == 0]
    structural = [r for r in predicted if r['structural_available']]
    actions = [r for r in predicted if r['action_applied']]
    action_stages = {}
    for stage, subset, gain_key in (('raw', structural, 'raw_utility_gain'),
            ('bounded', structural, 'utility_gain'), ('accepted', actions, 'utility_gain')):
        gain = [float(r.get(gain_key, 0.)) for r in subset]
        action_stages[stage] = dict(frames=len(subset),
            coverage_all_frames=len(subset) / len(rows),
            helpful_frames=sum(v > 1e-6 for v in gain), harmful_frames=sum(v < -1e-6 for v in gain),
            mean_utility_gain=float(np.mean(gain)) if gain else None,
            net_utility_all_frames=sum(gain) / len(rows))
    gpu_rows = [r for r in predicted if r.get('cuda_profile_available', False)]
    elapsed = sum(r.get('tracking_elapsed_ms', 0.) for r in predicted)
    runtime = dict(measured_frames=len(predicted), acquisition_ms=sum(r.get('acquisition_ms', 0.) for r in predicted),
                   forward_ms=sum(r.get('forward_ms', 0.) for r in predicted), tracking_elapsed_ms=elapsed,
                   fps_without_initialization=1000 * len(predicted) / elapsed if elapsed else None,
                   fps_role='diagnostic_wall_throughput_not_deployment_fps',
                   diagnostics_included_in_acquisition=True,
                   cuda_profile_frames=len(gpu_rows),
                   cuda_peak_allocated_mb=max((r['cuda_peak_allocated_mb'] for r in gpu_rows), default=None),
                   scope='acquisition includes B1 prepass and shared sampler GT labels/sidecar diagnostics; forward separately measures B0/B2/B3 network execution; wall throughput excludes only subsequent box-metric/CSV reporting and is not deployment FPS')
    return dict(schema='ct_seqtrack.endpoint_diagnostics.v27', metric_mode='benchmark_compat',
                evidence_label_scale=1.0, metrics=metrics,
                runtime=runtime,
                funnel=dict(prediction_frames=len(predicted), measured_frames=len(measured),
                            complete=len(measured) == len(predicted) and all(v['measured_frames'] == len(predicted) for v in stages.values()),
                            stages=stages, action_stages=action_stages,
                            consensus_scope='top-mode inliers only; compatible modes may also contribute to fused center',
                            b2_disabled_frames=sum(not r.get('b2_enabled', False) for r in predicted),
                            structural_frames=len(structural), action_frames=len(actions),
                            n0_current_frames=len(n0),
                            n0_current_with_extension_frames=sum(r.get('acquisition_novel_pool_point_count', 0) > 0 for r in n0),
                            all_empty_frames=sum(r.get('acquisition_global_raw_point_count') == 0 for r in measured),
                            mode_unique_count_mean=float(np.mean([r.get('ct_vote_mode_unique_count', 0.) for r in structural])) if structural else None),
                interpretation='one-step action harm uses matched current state; closed-loop gains require a separate never rollout')


def write_endpoint_diagnostics(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize_endpoint_diagnostics(rows)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    return summary
