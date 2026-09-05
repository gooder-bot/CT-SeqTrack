"""Read-only independent audit of the v26 run files; writes only derived evidence here."""
from pathlib import Path
import hashlib
import json
import sys
import types

import numpy as np
import pandas as pd
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[4]
DEST = Path(__file__).resolve().parent
# The missing optional package is needed only for checkpoint configuration objects.
# Substitute its dictionary container locally; no package/environment is modified.
try:
    import easydict
except ImportError:
    mod = types.ModuleType('easydict')
    mod.EasyDict = type('EasyDict', (dict,), {})
    sys.modules['easydict'] = mod

def scalars(folder):
    a = EventAccumulator(str(folder), size_guidance={'scalars': 0})
    a.Reload()
    return {k: [{'step': int(s.step), 'value': float(s.value)} for s in a.Scalars(k)]
            for k in a.Tags()['scalars']}

results = []
fingerprints = []
for run in sorted((ROOT / 'output').glob('20260903*')):
    log = run / 'lightning_logs/version_0'
    prov = json.loads((run / 'run_provenance.json').read_text(encoding='utf8'))
    ckpt = run / 'formal_checkpoints/epoch=060.ckpt'
    if not ckpt.exists():
        ckpt = log / 'checkpoints/last.ckpt'
    c = torch.load(ckpt, map_location='cpu', weights_only=False)
    fingerprints.append(c['ct_observation_batch_fingerprints'])
    r = {'run': run.relative_to(ROOT).as_posix(), 'checkpoint': ckpt.relative_to(ROOT).as_posix(),
         'epoch_1_based': int(c['epoch'])+1, 'step': int(c['global_step']),
         'commit': prov['git'], 'initial_checkpoint': prov['init_checkpoint_path'],
         'module_audit': c['ct_module_audit'],
         'prefix_hashes': {k: c['ct_b0_prefix_hashes'][k] for k in ['initial','step_1','step_100']},
         'optimizer_prefix_hashes': c['ct_b0_optimizer_state_hashes'],
         'input_fingerprints_sha256': hashlib.sha256(json.dumps(c['ct_observation_batch_fingerprints'], sort_keys=True).encode()).hexdigest(),
         'cuda_audit': c['ct_cuda_stage_audit'],
         'formal_checkpoints': [q.name for q in sorted((run / 'formal_checkpoints').glob('*.ckpt'))],
         'metrics': {}, 'csv': {}}
    for name in ['metrics_mini_val_success','metrics_mini_val_precision','runtime_runtime']:
        r['metrics'][name] = scalars(log / name)
    diags = sorted((log / 'candidate_diagnostics').glob('epoch_*.csv'))
    if diags:
        path = diags[-1]
        d = pd.read_csv(path)
        valid = d.b1_valid.gt(0) & np.isfinite(d.learned_motion_error) & np.isfinite(d.kinematic_error)
        b = d.loc[valid]
        r['csv'].update({'path': path.relative_to(ROOT).as_posix(), 'rows': len(d),
                         'tracklets': int(d.tracklet_id.nunique()), 'expected_nonfirst_endpoints': 2285-106,
                         'missing_endpoints': 2285-106-len(d), 'b1_valid_rows': len(b),
                         'b1_learned_rmse': float(np.sqrt(np.mean(b.learned_motion_error**2))),
                         'b1_cv_rmse': float(np.sqrt(np.mean(b.kinematic_error**2))),
                         'b1_coverage50': float(b.b1_coverage_50.mean()),
                         'b1_coverage95': float(b.b1_coverage_95.mean()),
                         'recursive_age_valid_count': int(d.recursive_age_valid.gt(0).sum())})
        if prov['resolved_config']['ct_enable_b2']:
            main_scalars = scalars(log)
            r['metrics']['main_output_diagnostics'] = {
                key: main_scalars[key] for key in
                ['success_observation/mini_val', 'success_raw_search/mini_val']}
            avail = d.available.gt(0)
            gain = d.observation_error-d.raw_search_error
            iougain = d.raw_search_iou-d.observation_iou
            need = d.global_target_count_label.gt(0) & d.search_coverage_need.gt(0)
            strict_need = d.global_target_count_label.gt(0) & d.base_raw_target_count.eq(0)
            eligible = d.prepool_target_count.gt(0)
            r['csv'].update({'active_prior_sources': d.active_prior_source.value_counts().to_dict(),
                'available_rows': int(avail.sum()), 'center_gain_available': float(gain[avail].mean()),
                'iou_gain_available': float(iougain[avail].mean()),
                'harm_over_0_1m_available': float(gain[avail].lt(-.1).mean()),
                'need_rows': int(need.sum()), 'need_novel_target_rows': int((need & d.support_novel_target_count.gt(0)).sum()),
                'strict_need_rows': int(strict_need.sum()), 'strict_novel_target_rows': int((strict_need & d.support_novel_target_count.gt(0)).sum()),
                'selection_eligible_rows': int(eligible.sum()), 'selection_row_recall': float(d.loc[eligible,'selected_target_count'].gt(0).mean()),
                'selection_point_recall': float(d.loc[eligible,'selected_target_count'].sum()/d.loc[eligible,'prepool_target_count'].sum()),
                'calibrated_rows': int(d.b3_calibrated.gt(0).sum()), 'action_rows': int(d.router_applied_gate.gt(0).sum()),
                'inference_mode': prov['resolved_config'].get('proposal_inference_mode'),
                'search_valid_rows': int(d.search_valid.gt(0).sum()),
                'presence_max': float(d.presence_probability.max()),
                'observation_equals_final': bool(np.array_equal(d.observation_error.to_numpy(), d.final_error.to_numpy())),
                'cf_raw_equals_novel': {k: bool(d[f'cf_{k}_support_raw_target_count'].equals(d[f'cf_{k}_support_novel_target_count'])) for k in ['fixed_2_1','adaptive_local','adaptive_dual_support']}})
    results.append(r)
output = {'all_input_fingerprints_equal': all(f == fingerprints[0] for f in fingerprints), 'runs': results}
(DEST / 'recomputed_experiment_evidence.json').write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding='utf8')
for r in results:
    print(r['run'], 'epoch',r['epoch_1_based'],'step',r['step'])
    print('METRICS', {k: {tag: {'n': len(v), 'last': v[-1]} for tag,v in series.items()} for k,series in r['metrics'].items()})
    print('HASH', {k:v[:8] for k,v in r['prefix_hashes'].items()}, 'UPDATES',r['module_audit']['update_steps'])
    print('CSV',r['csv'])
print('all_input_fingerprints_equal', output['all_input_fingerprints_equal'])
