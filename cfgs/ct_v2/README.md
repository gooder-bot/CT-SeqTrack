# CT-SeqTrack v2 configs

These files are the complete active experiment surface:

| Config | Purpose |
| --- | --- |
| `01_seqtrack3d_baseline.yaml` | Same-code SeqTrack3D baseline |
| `01_seqtrack3d_baseline_full.yaml` | Formal full-nuScenes baseline |
| `02_ct_motion.yaml` | Add the continuous-time motion prior with fixed fusion |
| `03_ct_motion_search.yaml` | Add time-guided search expansion |
| `04_ct_seqtrack_v2.yaml` | Add adaptive proposal fusion; mini promotion model |
| `04_ct_seqtrack_v2_full.yaml` | Formal full-nuScenes version of the final model |

All older YAML files remain valid legacy experiments. They are no longer part
of the default paper workflow.
