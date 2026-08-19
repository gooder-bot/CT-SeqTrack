# RA-PMM B1 server workflow

RA-PMM is opt-in. The repaired baselines remain `25_b1.yaml` and
`25_full.yaml`; the new scratch arms are `ra_pmm/b1.yaml` and
`ra_pmm/full.yaml`. All model parameters are trainable from epoch 0. The only
permitted checkpoint continuation is an exact resume of the same run.

## 1. Checkpoint-free preflight

Run this on the server training split before constructing an RA-PMM model:

```bash
python tools/build_b1_hard_motion_artifact.py \
  --config cfgs/ct_seqtrack/ra_pmm/b1.yaml \
  --split mini_train --path DATA_ROOT \
  --output artifacts/b1_hard_motion_car.json
```

The artifact contains the real timestamp distribution, `dt_floor`, GT-only
acceleration-equivalent q50/q90, CV/fixed-CV-CA/CA/CTRV/oracle RMSE, code and
config hashes. It rejects dev/validation/test splits and never accepts a
checkpoint. If hard-q80--q100 oracle potential is below 10%, the loader masks
CTRV and disables top-2 support.

Create a server-local YAML override (do not commit data paths):

```yaml
_base_: ../../cfgs/ct_seqtrack/ra_pmm/b1.yaml
motion_v3_hard_statistics_path: /absolute/path/artifacts/b1_hard_motion_car.json
epoch: 5
```

Use the analogous Full base for R2. R0 is the repaired `25_b1.yaml`; R1 is the
RA-PMM B1 override; R2 is the RA-PMM Full override. Every arm starts from
random initialization.

## 2. Quantile calibration and audit

After a promoted scratch/60-epoch run, export disjoint scene populations:

```bash
python tools/export_b1_calibration.py --config SERVER_RA_CONFIG \
  --checkpoint RUN_CHECKPOINT --split mini_train --path DATA_ROOT \
  --partition calibration_select --output artifacts/b1_select.npz

python tools/export_b1_calibration.py --config SERVER_RA_CONFIG \
  --checkpoint RUN_CHECKPOINT --split mini_train --path DATA_ROOT \
  --partition calibration_audit --output artifacts/b1_audit.npz

python tools/calibrate_b1_quantiles.py \
  --select artifacts/b1_select.npz --audit artifacts/b1_audit.npz \
  --output artifacts/b1_quantile_calibration.json
```

For evaluation, set `motion_v3_quantile_calibration_path` to the JSON artifact
and `motion_v3_require_quantile_calibration: true`. Calibration is installed in
memory after checkpoint loading; it never writes a new checkpoint.

## 3. Promotion diagnostics

`tools/report_ct_b1.py` now reports physical/endpoint/anchor-drift RMSE,
physical and operational support coverage, recoverability AUROC/AUPRC,
risk--coverage, boundary coverage, mode use/entropy/regret, saturation,
support recall/volume and the registered time, sparsity, recursive-age,
GT-hard-motion and B0-reliability strata.

Top-2 support is separately gated by `search_v3_enable_top2_tube`. Leave it
off until geometry promotion passes. When enabled, it requires second-mode
probability at least 0.15 and expert separation at least 0.5 m, then allocates
25%--40% of the unchanged tube point budget to the secondary expert.

Training and data-dependent checks belong on the server. Missing local
nuScenes data or CUDA dependencies are not implementation failures and should
not be repeatedly debugged locally.
