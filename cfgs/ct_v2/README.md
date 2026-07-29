# CT-SeqTrack v2 configs

These files are the complete active experiment surface:

| Config | Purpose |
| --- | --- |
| `01_seqtrack3d_baseline.yaml` | Same-code SeqTrack3D baseline |
| `01_seqtrack3d_baseline_full.yaml` | Formal full-nuScenes baseline |
| `02_ct_motion.yaml` | Ordered B1motion-v2: pre-crop second search + zero-init feature residual |
| `02_ct_motion_legacy_fixed.yaml` | Frozen rejected fixed-alpha B1 for reproduction |
| `02_ct_motion_alpha000.yaml` | Reproduced fixed-motion fallback control, alpha=0 |
| `02_ct_motion_alpha025.yaml` | Reproduced fixed-motion rerun, alpha=0.25 |
| `03_ct_motion_search.yaml` | Add time-guided search expansion |
| `04_ct_seqtrack_v2.yaml` | Add adaptive proposal fusion; completed seed42 screen, rejected |
| `04_ct_seqtrack_v2_full.yaml` | Reserved full-nuScenes config; blocked by the mini result |
| `05_seqtrack3d_search_only.yaml` | Same B0 network plus data-side time-guided search; completed seed42 screen, rejected |
| `06_seqtrack3d_pftc_unweighted.yaml` | B0 + canonical point-feature consistency, all pair weights equal |
| `07_seqtrack3d_dt_pftc.yaml` | B0 + sample-normalized physical-Δt pair weighting |

All older YAML files remain valid legacy experiments. They are no longer part
of the default paper workflow.

The historical 2026-07-27 seed42 normal-mini screens are complete. B3 finishes at
25.537 Success / 24.707 Precision and its gate saturates at the configured
0.75 ceiling. Search-only A1 finishes at 27.036 / 25.596 versus B0 at
53.360 / 64.382, so it also fails the normal-mini guardrail. Do not rerun
the legacy configs unchanged or train A2 from that chain.

`02_ct_motion.yaml` now names the corrected B1motion-v2. It uses an ordered GRU,
an uncertainty-aware pre-crop second branch that does not consume baseline
tokens, mixed-cadence training, and an exact-zero feature adapter. Ordered
histories and supervision use the actual candidate crop-anchor frame, matching
recursive evaluation instead of mixing GT-anchor motion with candidate-anchor
observation features. It does not use proposal innovation or global alpha.
Design and commands:
`docs/B1MOTION_V2_ORDERED_PRECROP_20260730.md`.

The 2026-07-30 scratch alpha reruns are also complete. Alpha 0 finishes at
47.049 / 49.184 and alpha 0.25 at 29.581 / 28.862, versus the historical
alpha-0.75 B1 at 26.021 / 24.972 and B0 at 53.360 / 64.382. Alpha 0.25 is
therefore less destructive than 0.75 but still loses 17.468 / 20.322 to the
same-code alpha-0 fallback. Do not launch another 60-epoch global-alpha sweep.
First run a same-checkpoint alpha on/off 2x2 and export endpoint-level
observation/dynamics proposal attribution. See
`compare_results/reports/ct_motion_alpha_sweep_seed42_20260730.md`.

PFTC is the independent fourth-module candidate; it is not B3 plus another
module.  It is training-only and adds no inference work.

The first `07_seqtrack3d_dt_pftc.yaml` seed42 artifact is **not a completed
60-epoch run**: it stops at step 29,092 (about epoch 23.05), with the latest
checkpoint at epoch19.  The 2026-07-30 audit also found that the current
canonical yaw transform uses `R(+yaw)` where the project convention requires
`R(-yaw)`, foreground feature std shrinks to 22.2% of its epoch1 value by
epoch20, and training is about 10.2x slower than B0.  Do not resume or reproduce
configs 06/07 unchanged.  Fix geometry, anti-collapse behavior, and runtime,
then repeat a short three-arm kill-test before any new 60-epoch job.  See
`compare_results/reports/pftc_b4_seed42_partial_diagnosis_20260730.md`.

After that correction, execute the bounded loss preflight and freeze the
largest accepted lambda for both PFTC arms:

```bash
python tools/ct_v2/run.py train --variant pftc --preflight --seed 42
python tools/ct_v2/analyze_pftc_preflight.py \
  output/<preflight-run>/lightning_logs/version_0
python tools/ct_v2/check_pftc_initialization.py --seed 42
```

The analyzer checks both raw and Δt-weighted loss ratios and freezes the
largest lambda safe for both arms.  Pass that value with `--pftc-weight`.
Standard-cadence training is mandatory.  `--protocol random20` and
`--protocol gap1124` are evaluation controls only;
`--time-mode true|fixed|shuffled` changes
`timestamps_effective`, never frame order or correspondence topology.
