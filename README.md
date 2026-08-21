# CT-SeqTrack

CT-SeqTrack studies observation-anchored evidence recovery for irregular-time
3D single-object tracking. The active repository contains one formal chain:

```text
SeqTrack / B0 Observation
  -> B1 Physical-Time Prior
  -> B2 Extension-Only Evidence
  -> B3 Calibrated Selective Update
```

B4 is retained as an isolated experiment and is disabled in every formal
configuration. The new v24 method has not completed its registered mini or
full-nuScenes experiments, so this repository does not currently claim a gain,
stability, SOTA, or a causal benefit from physical time or memory.

## Method contract

- B0 is the nominal observation tracker and the only recursive state writer.
- B1 reads prediction-backed history boxes and physical timestamps. It supplies
  a prior and uncertainty but never replaces the observation.
- B2 must recover identifiable evidence from extension-only points. Base points
  and memory are context; they cannot independently regress the target center.
- B3 consumes detached upstream evidence and may apply only a calibrated,
  bounded residual. Missing or mismatched calibration returns B0 exactly.

The canonical candidate is `b0_view_id=0`. Three auxiliary B0 views stop after
B0, and the objective remains
`0.5*L0 + (L1+L2+L3)/6`. B1/B2/B3 run once per online endpoint on the
canonical view only.

## Active interfaces

- `models/seqtrack3d.py`: SeqTrack/B0 host and isolated B4 hook.
- `models/ctseqtrack.py`: paper-facing composition root.
- `models/ct_v2/pipeline.py`: B0/B1 paper-facing components.
- `models/ct_v2/evidence_memory.py`: B2 evidence and B3 selective update.
- `models/ct_v2/pipeline_contracts.py`: typed internal ownership contracts.
- `utils/action_calibration.py`: held-out action calibration and fail-closed validation.
- `utils/online_contract.py`: scratch/resume and cross-arm identity contracts.

The evaluator continues to receive the existing flat dictionaries and metric
names. Required compatibility aliases remain part of the public runtime.

## Formal variants

| Variant | Trainable modules from random initialization | Training/deployed output |
|---|---|---|
| `b0` | B0 | observation |
| `b1` | B0+B1 | observation |
| `full_minus_b3` | B0+B1+B2 | raw B2 search output |
| `full` | B0+B1+B2+B3 | observation until calibrated selective evaluation |

The active mini configs are under `cfgs/ct_seqtrack/`; the corresponding
`24_*_nuscenes_full.yaml` files are used only after mini validation. B4 keeps
`cfgs/ct_v2/19_b4_decoder_alignment.yaml` and
`20_b4_decoder_anticollapse.yaml`.

All formal arms use `scratch_only`. `--init_checkpoint` is forbidden.
`--checkpoint` is accepted only for exact same-run epoch-boundary resume or
for evaluation. Enabled B0/B1/B2/B3 parameters are never frozen.

## Experiment order

The authoritative protocol is
[docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md). In summary:

1. use the fixed candidate protocol in every arm: four B0 views and one
   canonical B2 view;
2. run B0, B1, Full-B3 and Full independently from epoch0 on mini seed42;
3. calibrate Full only after its scratch training has finished;
4. after mini analysis, run full nuScenes seed42 and then B0/Full seeds43/44.

There is no preflight, promotion, kill-test or intermediate stopping gate.
Acquisition utilities are optional post-run analysis and never initialize or
block training.

## Validation

```bash
python tools/verify_ct_slimming.py verify
python -m pytest -q
```

Real-batch forward/backward, 100-step/resume parity and point/box visualization
remain recommended server-side acceptance checks. They are not represented as
locally completed when the full Lightning/nuScenes environment is unavailable.
Engineering checkpoints are discarded and may not initialize formal runs.

Detailed method and evidence boundaries:

- [B0--B3 method](docs/CTSEQTRACK_B0_B3_METHOD.md)
- [formal experiment protocol](docs/EXPERIMENT_PROTOCOL.md)
- [formal tooling](docs/FORMAL_TOOLING.md)
- [source slimming gate](docs/SOURCE_SLIMMING_GATE.md)
- [active status](need_to_do.md)
- [historical evidence index](docs/HISTORY_EVIDENCE_INDEX.md)
- [slimming baseline](docs/slimming_baseline/README.md)
