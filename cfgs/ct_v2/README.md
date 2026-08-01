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
| `02_ct_motion_v3.yaml` | Physical xy prior + bounded reliability-gated post-Transformer fusion |
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

`02_ct_motion.yaml` names the ordered/pre-crop B1motion-v2 experiment, but its
seed42 60-epoch screen is now complete and rejected.  It finishes at
20.618 Success / 19.830 Precision versus B0 at 53.360 / 64.382; its best
checkpoint is also below B0.  Do not rerun this YAML unchanged or promote it to
multiple seeds/full nuScenes.

The initialization contract passed, but the training contract did not.  The
35% irregular sampler replaces the whole B0 history while the main branch still
uses gap-blind order tokens; the candidate-anchor trajectory target also
contains a common anchor error that is not identifiable from relative history
alone.  After epoch2 the nominally small adapter grows to about 2.07 feature-L2,
while the pre-crop extension is valid on only 3.93% of training samples.
Run a same-code B0 and short cadence/adapter factorial controls before changing
the module again.  Design, completed result, and recovery plan:
`docs/B1MOTION_V2_ORDERED_PRECROP_20260730.md`; reviewed data:
`compare_results/reports/b1motion_v2_seed42_20260730.md`.

The 2026-07-30 scratch alpha reruns are also complete. Alpha 0 finishes at
47.049 / 49.184 and alpha 0.25 at 29.581 / 28.862, versus the historical
alpha-0.75 B1 at 26.021 / 24.972 and B0 at 53.360 / 64.382. Alpha 0.25 is
therefore less destructive than 0.75 but still loses 17.468 / 20.322 to the
same-code alpha-0 fallback. Do not launch another 60-epoch global-alpha sweep.
First run a same-checkpoint alpha on/off 2x2 and export endpoint-level
observation/dynamics proposal attribution. See
`compare_results/reports/ct_motion_alpha_sweep_seed42_20260730.md`.

`02_ct_motion_v3.yaml` has also completed its seed42 scratch 60-epoch screen.
It finishes at 52.655 / 61.835 with late-3 52.050 / 61.206, below the
historical B0 by 0.705 / 2.547 final and 0.855 / 1.898 late-3. It therefore
does not pass the standard-cadence promotion gate. Unlike v2, the physical
prior itself learns beyond constant velocity at epoch60 (main/gap2/gap4 RMSE
improvement 7.6%/10.9%/16.0%); the remaining failure is concentrated in gate
calibration and recursive transfer. Do not start another 60-epoch v3 sweep.
First evaluate the epoch30 and epoch60 checkpoints with fusion on/off using
`tools/ct_v2/run.py test --variant motion_v3 [--fusion-off]`, then export
endpoint-level attribution. See
`compare_results/reports/b1motion_v3_seed42_20260801.html`.

For historical paper-table context, the separate original SeqTrack3D run
finishes at 50.986 / 59.962, so v3 is numerically +1.670 / +1.873. This is not
a motion ablation: the current B0 itself is +2.374 / +4.420 over that old run
and remains stronger than v3. Use current B0 or same-checkpoint fusion-off for
module attribution.

PFTC is the independent fourth-module candidate; it is not B3 plus another
module.  It is training-only and adds no inference work.

The first `07_seqtrack3d_dt_pftc.yaml` seed42 artifact is now a **completed
60-epoch run**: 75,720 steps, 12 validation points, and an epoch60 checkpoint.
It finishes at 51.189 Success / 60.886 Precision, below B0 by 2.171 / 3.496;
late-3 is also lower by 1.507 / 2.487.  The implementation audit still finds
`R(+yaw)` where the project convention requires `R(-yaw)`, foreground feature
std shrinks to 16.4% of its epoch1 value, and training is 8.24x slower than B0.
Configs 06/07 must not be reproduced unchanged.  Fix geometry, anti-collapse
behavior, and runtime, then repeat a short same-code three-arm kill-test before
any new 60-epoch job.  See
`compare_results/reports/pftc_b4_seed42_final_diagnosis_20260801.md`.

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
