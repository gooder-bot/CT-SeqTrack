"""Read original historical B0 events/provenance without modifying output."""
import json
import sys
import types
from pathlib import Path
import torch
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Reconstruct only saved dictionary metadata, never execute historical models.
try:
    import easydict
except ImportError:
    class EasyDict(dict):
        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError as error:
                raise AttributeError(key) from error
    shim = types.ModuleType('easydict')
    shim.EasyDict = EasyDict
    sys.modules['easydict'] = shim

ROOT = Path(__file__).resolve().parents[4]
DEST = Path(__file__).resolve().parent
NAMES = [
    '20260725-2326-01_seqtrack3d_baseline-ctv2_d86990c_b0_baseline_car_seed42_60ep_bs16',
    '20260813-0116-01_seqtrack3d_baseline-scratch_ct21_b0_car_60ep_bs16_s42',
    '20260822-2246-24_b0-ct24_b0_restore_seed42',
    '20260824-0219-25_b0-ct25_b0_mini_car_60ep_bs16_seed42_retryfix',
    '20260825-1931-25_b0-ct25_b0_mini_car_seed42_60ep_bs16_val5',
    '20260903-2301-26_b0-ct26_b0_mini_car_seed42_60ep_bs16_20260903_225946',
    '20260905-173106-27_b0-mini_car_seed42_60ep_bs16',
]
FIELDS = ['ct_training_profile', 'ct_unified_auto', 'ct_b0_candidate_weights',
          'ct_b0_loss_reduction', 'candidate_trajectory_mode', 'ct_b0_candidate_mode',
          'ct_observation_payload_mode', 'observation_safe_bbox_size', 'bc_weight',
          'check_val_every_n_epoch', 'workers', 'ct_evaluator_identity',
          'ct_observation_loader_profile', 'ct_observation_optimizer_profile',
          'train_split', 'val_split', 'ct_observation_train_partition']

rows = []
for name in NAMES:
    run = ROOT / 'output' / name
    ev = run / 'lightning_logs/version_0'
    prov = json.loads((run / 'run_provenance.json').read_text(encoding='utf-8'))
    hp = yaml.load((ev / 'hparams.yaml').read_text(encoding='utf-8'), Loader=yaml.BaseLoader)
    cfg = hp.get('config', hp)
    cfg = cfg.get('dictitems', cfg)
    measures = {}
    for prefix in ('metrics_test', 'metrics_mini_val', 'metrics_dev'):
        for metric in ('success', 'precision'):
            folder = ev / (prefix + '_' + metric)
            if not folder.is_dir():
                continue
            acc = EventAccumulator(str(folder), size_guidance={'scalars': 0}).Reload()
            tags = acc.Tags()['scalars']
            assert len(tags) == 1
            records = acc.Scalars(tags[0])
            if records:
                measures[prefix + '_' + metric] = dict(count=len(records),
                    final_step=records[-1].step, final_value=records[-1].value)
    row = dict(run=name, provenance_git=prov.get('git'), datasets=prov.get('datasets'),
               config={k:cfg.get(k) for k in FIELDS}, measures=measures)
    if name.startswith(('20260824-0219', '20260825-1931')):
        ck = torch.load(ev / 'checkpoints/last.ckpt', map_location='cpu', weights_only=False)
        row['checkpoint_epoch'] = ck.get('epoch')
        row['global_step'] = ck.get('global_step')
        row['prefix'] = ck.get('ct_b0_prefix_hashes')
        row['fingerprints'] = ck.get('ct_observation_batch_fingerprints')
        del ck
        folder = ev / 'loss_loss_b0_view0'
        if folder.is_dir():
            acc = EventAccumulator(str(folder), size_guidance={'scalars': 0}).Reload()
            vals = acc.Scalars(acc.Tags()['scalars'][0])
            row['first_view0_losses'] = [dict(step=x.step, value=x.value) for x in vals[:5]]
    rows.append(row)

high, low = rows[3], rows[4]
comparison = dict(
    prefix_high=high['prefix'], prefix_low=low['prefix'],
    fingerprints_equal=high['fingerprints'] == low['fingerprints'],
    fingerprint_count_high=len(high['fingerprints'] or []),
    fingerprint_count_low=len(low['fingerprints'] or []),
    high_first_losses=high.get('first_view0_losses'),
    low_first_losses=low.get('first_view0_losses'),
)
out = dict(runs=rows, high_low_comparison=comparison)
(DEST / 'historical_b0_recheck.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(dict(runs=[{k:v for k,v in row.items() if k in ('run','measures','config')} for row in rows],
                     high_low_comparison=comparison), ensure_ascii=False, indent=2))
