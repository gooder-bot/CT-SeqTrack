# CT-SeqTrack v2 configs

These files are the complete active experiment surface:

| Config | Purpose |
| --- | --- |
| `01_seqtrack3d_baseline.yaml` | Same-code SeqTrack3D baseline |
| `01_seqtrack3d_baseline_full.yaml` | Formal full-nuScenes baseline |
| `02_ct_motion.yaml` | Add the continuous-time motion prior with fixed fusion |
| `03_ct_motion_search.yaml` | Add time-guided search expansion |
| `04_ct_seqtrack_v2.yaml` | Add adaptive proposal fusion; completed seed42 screen, rejected |
| `04_ct_seqtrack_v2_full.yaml` | Reserved full-nuScenes config; blocked by the mini result |
| `05_seqtrack3d_search_only.yaml` | Same B0 network plus data-side time-guided search; completed seed42 screen, rejected |
| `06_seqtrack3d_pftc_unweighted.yaml` | B0 + canonical point-feature consistency, all pair weights equal |
| `07_seqtrack3d_dt_pftc.yaml` | B0 + sample-normalized physical-Δt pair weighting |

All older YAML files remain valid legacy experiments. They are no longer part
of the default paper workflow.

The 2026-07-27 seed42 normal-mini screens are complete. B3 finishes at
25.537 Success / 24.707 Precision and its gate saturates at the configured
0.75 ceiling. Search-only A1 finishes at 27.036 / 25.596 versus B0 at
53.360 / 64.382, so it also fails the normal-mini guardrail. Do not rerun
these configs unchanged or train A2. The next step is a no-training,
same-checkpoint Search on/off 2x2 with the existing B0 and A1 checkpoints.

PFTC is the independent fourth-module candidate; it is not B3 plus another
module.  It is training-only and adds no state-dict keys or inference work.
Before a 60-epoch run, execute the bounded loss preflight and freeze the
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
