"""Summarize existing diagnostic exports; no experiment outputs are modified."""
from pathlib import Path
import json
import struct
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[4]
events = pd.read_csv(OUT / 'diagnostic_events.csv')
epoch = pd.read_csv(OUT / 'epoch_diagnostic_values.csv')
inventory = pd.read_csv(OUT / 'event_file_inventory.csv')
supply = pd.read_csv(OUT / 'acquisition_supply.csv')
summary = pd.read_csv(OUT / 'sampled_batch_summary.csv')

framing = []
for file in inventory.file.unique():
    path = ROOT / file
    count, status = 0, 'complete_tfrecord_framing'
    size = path.stat().st_size
    with path.open('rb') as f:
        while f.tell() < size:
            header = f.read(12)
            if len(header) != 12:
                status = 'truncated_header'
                break
            length = struct.unpack('<Q', header[:8])[0]
            if f.tell() + length + 4 > size:
                status = 'truncated_payload_or_crc'
                break
            f.seek(length + 4, 1)
            count += 1
    framing.append(dict(file=file, bytes=size, records=count, status=status, crc_verified=False))
pd.DataFrame(framing).to_csv(OUT / 'tfrecord_framing.csv', index=False)

diags = events[(events.epoch <= 2) & events.group.str.startswith('loss_mechanism_')]
diags = diags.pivot_table(index=['arm','epoch','step'], columns='group', values='value')
get = lambda key: 'loss_mechanism_' + key
funnel = []
for (arm, ep), rows in diags.groupby(['arm','epoch']):
    sums = {k: float(rows[get(k)].sum()) for k in (
        'ct_acquisition_sampled_target_sum', 'ct_acquisition_prepool_target_sum',
        'ct_acquisition_pool_target_sum')}
    positive = rows[get('ct_acquisition_prepool_target_sum')] > 0
    record = dict(arm=arm, epoch=int(ep), logged_batches=len(rows),
        target_positive_logged_batches=int(positive.sum()), **sums,
        selected_over_prepool_in_logged_batches=sums['ct_acquisition_sampled_target_sum']/sums['ct_acquisition_prepool_target_sum'],
        selected_over_raw_extension_pool_in_logged_batches=sums['ct_acquisition_sampled_target_sum']/sums['ct_acquisition_pool_target_sum'],
        positive_batch_mean_enrichment=float(rows.loc[positive,get('ct_relation_selected_target_enrichment')].mean()),
        positive_batch_median_enrichment=float(rows.loc[positive,get('ct_relation_selected_target_enrichment')].median()),
        scope='sampled training mechanism batches, not whole epoch or official validation')
    funnel.append(record)
pd.DataFrame(funnel).to_csv(OUT/'sampled_funnel.csv', index=False)

rates = []
for arm in ('full_cfc', 'full_gru'):
    for ep in (1,2):
        ep_rows = epoch[(epoch.arm==arm)&(epoch.epoch==ep)]
        values = dict(zip(ep_rows.group, ep_rows.value))
        prefix = 'ct_epoch_calibration_sampled_sampled_'
        n = values[prefix+'help_positive_count'] + values[prefix+'help_negative_count']
        row = dict(arm=arm,epoch=ep,sampled_legal_rows=int(n))
        for target in ('presence','help','harm'):
            row[target+'_positive']=int(values[prefix+target+'_positive_count'])
            row[target+'_rate']=row[target+'_positive']/n
        row['neutral_rate']=1-row['help_rate']-row['harm_rate']
        for metric in ('bounded_success_gain_mean','bounded_precision_gain_mean',
                       'bounded_utility_gain_mean','utility_score_mse','presence_auroc',
                       'presence_auprc','presence_ece','help_auroc','help_auprc','harm_auroc','harm_auprc'):
            row[metric]=values[prefix+metric]
        rates.append(row)
pd.DataFrame(rates).to_csv(OUT/'sampled_legal_action_rates.csv',index=False)

keys = ['ct_acquisition_margin_parallel_mean','ct_acquisition_margin_perpendicular_mean',
        'ct_search_query_delta_t_mean','motion_v3_kinematic_rmse','motion_v3_prior_rmse',
        'motion_v3_coverage_50','motion_v3_coverage_80','motion_v3_coverage_95',
        'motion_v3_sigma_parallel_mean','motion_v3_sigma_perpendicular_mean']
summary[(summary.epoch<=2)&summary.group.isin([get(k) for k in keys])].to_csv(
    OUT/'sampled_b1.csv',index=False)
evidence = dict(
    framing_status_counts=pd.Series([x['status'] for x in framing]).value_counts().to_dict(),
    tfrecord_crc_verified=False, funnel=funnel, legal_action_rates=rates,
    completed_mechanism_rows_per_epoch=191288,
    completed_mechanism_transactions_per_epoch=11956,
    verification_scope='Selected diagnostics only; 468 scalar streams. No official validation exists yet.',
    caveats=[
        'Sampled diagnostic candidate/metric rows are not all training rows, and are not independent tracks.',
        'Presence indicates any target point in selected256, not target membership in consensus inliers.',
        'H1 help/harm use sign of mean Success/Precision gain; reported mean gains are not closed-loop val scores.',
        'Epoch acquisition point_recall=selected256 target count/raw extension support target count; batch field uses prepool768 denominator.',
        'B0 zero acquisition totals mean B2 disabled, not acquisition failure.',
        'No global novel/reachable/frame-level empty crop/first-loss/accepted gain records yet; cannot attribute failure proportions.'
    ])
(OUT/'findings.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
print(json.dumps(evidence,indent=2))
