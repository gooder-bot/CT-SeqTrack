"""Read-only audit of v27 evidence/action diagnostics; writes this report directory."""
from pathlib import Path
import csv
import json
import numpy as np
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[4]
DEST = Path(__file__).resolve().parent
NAMES = {
    "b1_b2": "20260905-173118-27_full_minus_b3-mini_car_seed42_60ep_bs16",
    "full": "20260905-173121-27_full-mini_car_seed42_60ep_bs16",
}
TB = [
    "ct_epoch_calibration_presence_auroc", "ct_epoch_calibration_presence_auprc",
    "ct_epoch_calibration_presence_ece", "ct_epoch_calibration_presence_positive_count",
    "ct_epoch_calibration_presence_negative_count", "ct_epoch_calibration_presence_positive_mean",
    "ct_epoch_calibration_presence_negative_mean", "ct_epoch_calibration_help_auroc",
    "ct_epoch_calibration_help_auprc", "ct_epoch_calibration_help_ece",
    "ct_epoch_calibration_harm_auroc", "ct_epoch_calibration_harm_auprc",
    "ct_epoch_calibration_harm_ece", "ct_epoch_calibration_bounded_utility_gain_mean",
    "ct_epoch_calibration_bounded_success_gain_mean", "ct_epoch_calibration_bounded_precision_gain_mean",
    "ct_epoch_calibration_utility_score_mse", "loss_ct_h3_exposure_rate",
    "loss_ct_b2_availability_rate", "loss_ct_b2_target_bearing_rate",
    "loss_ct_relation_auprc", "loss_ct_relation_auroc", "loss_ct_relation_ece",
    "loss_ct_relation_group_target_rate", "loss_ct_spatial_group_target_rate",
    "loss_ct_exploration_group_target_rate", "loss_ct_relation_selected_target_enrichment",
    "loss_ct_vote_consistency_mean", "loss_ct_vote_effective_mass_mean",
    "loss_loss_ct_vote", "loss_loss_ct_raw_search", "loss_loss_ct_relation",
    "loss_loss_ct_targetness", "loss_loss_ct_base_presence", "loss_loss_ct_extension_presence",
    "loss_loss_ct_expected_success_gain", "loss_loss_ct_expected_precision_gain",
]


def num(row, key):
    return float(row.get(key) or 0)


def summarize_group(rows, total):
    gains = np.array([num(r, "raw_utility_gain") for r in rows])
    return {
        "frames": len(rows), "raw_utility_net_all_frames_pp": float(100*gains.sum()/total),
        "raw_helpful_frames": int((gains > 1e-6).sum()),
        "raw_harmful_frames": int((gains < -1e-6).sum()),
        "mean_observation_distance_m": float(np.mean([num(r, 'observation_distance') for r in rows])) if rows else None,
        "mean_raw_distance_m": float(np.mean([num(r, 'raw_distance') for r in rows])) if rows else None,
    }


result = {}
epoch_rows = []
for arm, name in NAMES.items():
    root = ROOT / 'output' / name / 'lightning_logs/version_0'
    arm_result = {'run': name, 'dev_epochs': {}, 'training_scalars': {}}
    config = yaml.safe_load((ROOT/'output'/name/'resolved_config.yaml').read_text(encoding='utf-8'))
    mechanism_rows = config['ct_mechanism_prediction_frames_observed']
    for path in sorted((root / 'dev_diagnostics').glob('*endpoints.csv')):
        allrows = list(csv.DictReader(path.open(encoding='utf-8')))
        rows = [r for r in allrows if r['is_initial'] == 'False']
        evidence = [r for r in rows if num(r, 'ct_search_extension_selected_count') > 0]
        target = [r for r in evidence if num(r, 'acquisition_selected_target_count') > 0]
        absent = [r for r in evidence if num(r, 'acquisition_selected_target_count') == 0]
        gated = [r for r in evidence if r['structural_available'] == 'True']
        blocked = [r for r in evidence if r['structural_available'] != 'True']
        epoch = int(rows[0]['epoch'])
        d = {
            'epoch': epoch, 'all_frames': len(allrows), 'prediction_frames': len(rows),
            'structural_evidence_frames': len(evidence), 'selected_target_bearing_frames': len(target),
            'reported_structural_frames': len(gated), 'presence_blocked_evidence_frames': len(blocked),
            'presence_blocked_target_bearing_frames': sum(r['structural_available'] != 'True' for r in target),
            'actions': sum(r['action_applied'] == 'True' for r in rows),
            'clipped_evidence_frames': sum(num(r, 'ct_router_clip_rate') > 0 for r in evidence),
            'presence_quantiles': np.quantile([num(r, 'presence_score') for r in evidence], [0,.5,.9,1]).tolist(),
            'groups': {key: summarize_group(group, len(allrows)) for key, group in
                       [('all_evidence', evidence), ('selected_target', target), ('target_absent', absent),
                        ('presence_passed', gated), ('presence_blocked', blocked)]},
        }
        # The heavy candidate sidecar uses the actual bounded world box without
        # the endpoint export's stale presence gate. Its v27 metric-gain columns
        # recover complete H1 supervision at the already executed states.
        candidate_path = root/'candidate_diagnostics'/f'epoch_{epoch:02}.csv'
        candidate_rows = list(csv.DictReader(candidate_path.open(encoding='utf-8')))
        by_id = {(r['tracklet_id'], r['frame_id']): r for r in rows}
        assert len(candidate_rows) == len(rows) == len(by_id)
        for candidate in candidate_rows:
            paired = by_id[(candidate['tracklet_id'], candidate['frame_id'])]
            assert abs(num(candidate, 'observation_distance') - num(paired, 'observation_distance')) < 1e-10
            assert (num(candidate, 'structural_available') > 0) == (num(paired, 'ct_search_extension_selected_count') > 0)
            if paired['structural_available'] == 'True':
                assert abs(num(candidate, 'utility_gain') - num(paired, 'utility_gain')) < 1e-7
        valid_candidates = [r for r in candidate_rows if num(r, 'structural_available') > 0]
        recovered = {}
        for key, subset in [('all_evidence', valid_candidates),
                ('selected_target', [r for r in valid_candidates if num(r, 'selected_target_count') > 0]),
                ('target_absent', [r for r in valid_candidates if num(r, 'selected_target_count') == 0]),
                ('q_positive', [r for r in valid_candidates if num(r, 'action_score') > 0])]:
            u = np.array([num(r, 'utility_gain') for r in subset])
            recovered[key] = dict(frames=len(subset),
                success_net_all_frames_pp=100*sum(num(r, 'success_gain') for r in subset)/len(allrows),
                precision_net_all_frames_pp=100*sum(num(r, 'precision_gain') for r in subset)/len(allrows),
                utility_net_all_frames_pp=float(100*u.sum()/len(allrows)),
                helpful_frames=int((u > 1e-6).sum()), harmful_frames=int((u < -1e-6).sum()),
                oracle_positive_utility_net_all_frames_pp=float(100*u[u > 1e-6].sum()/len(allrows)))
        d['recovered_bounded_h1'] = {'groups': recovered,
            'source': str(candidate_path.relative_to(ROOT)),
            'scope': 'native action_metric_gains of unconditional bounded box at recorded rollout states; not a new closed-loop rollout',
            'checks': '309 keyed rows; exact observation-distance agreement; evidence availability agrees; gated endpoint gains match within 1e-7'}
        if arm == 'full':
            score = np.array([num(r, 'action_score') for r in evidence])
            d['gain_score_quantiles'] = np.quantile(score, [0,.1,.5,.9,1]).tolist()
            d['q_positive_evidence_frames'] = int((score > 0).sum())
            d['q_positive_but_presence_blocked_frames'] = sum(num(r, 'action_score') > 0 for r in blocked)
        arm_result['dev_epochs'][str(epoch)] = d
        epoch_rows.append({'arm': arm, **{k:v for k,v in d.items() if not isinstance(v, (dict, list))},
                           **d['groups']['all_evidence']})
    for name in TB:
        path = root / name
        if not path.exists():
            continue
        acc = EventAccumulator(str(path), size_guidance={'scalars': 0})
        acc.Reload()
        data = [event for tag in acc.Tags()['scalars'] for event in acc.Scalars(tag)]
        epoch_data = {}
        for e in (1, 10, 20, 40, 58, 59, 60):
            values = [x.value for x in data if (e-1)*1057 < x.step <= e*1057] if name.startswith('ct_epoch') else [x.value for x in data if (e-1)*1057 <= x.step < e*1057]
            epoch_data[str(e)] = {'n': len(values), 'mean': float(np.mean(values)) if values else None}
        late = [v['mean'] for e,v in epoch_data.items() if int(e) >= 58 and v['mean'] is not None]
        arm_result['training_scalars'][name] = {'epochs': epoch_data, 'late3_mean': float(np.mean(late)) if late else None,
            'scope': 'full_epoch_aggregate' if name.startswith('ct_epoch') else 'mean_of_logged_batches_not_population_weighted'}
        if arm == 'full' and name == 'loss_ct_h3_exposure_rate':
            h3_epochs = {}
            for epoch in (58, 59, 60):
                events = [x for x in data if (epoch-1)*1057 <= x.step < epoch*1057]
                assert all(x.value in (0., .0625) for x in events)
                # Scheduler supplies at most one H3 slot per mechanism batch.
                positive = sum(x.value > 0 for x in events)
                h3_epochs[str(epoch)] = dict(valid_shadow_rows=positive, mechanism_rows=mechanism_rows,
                    row_exposure=positive/mechanism_rows, mechanism_batches=len(events),
                    batches_with_valid_shadow=positive, batch_exposure=positive/len(events))
            arm_result['h3_exposure'] = dict(epochs=h3_epochs,
                late3_valid_shadow_rows=sum(v['valid_shadow_rows'] for v in h3_epochs.values()),
                late3_mechanism_rows=3*mechanism_rows,
                late3_row_exposure=sum(v['valid_shadow_rows'] for v in h3_epochs.values())/(3*mechanism_rows),
                evidence='resolved_config observed mechanism rows + every mechanism loss TensorBoard record; max one scheduled shadow slot')
    result[arm] = arm_result

(DEST/'b2_b3_evidence.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
keys = list(dict.fromkeys(k for r in epoch_rows for k in r))
with (DEST/'b2_b3_dev_epochs.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    writer.writerows(epoch_rows)
print(json.dumps({a: d['dev_epochs']['60'] for a,d in result.items()}, indent=2))
