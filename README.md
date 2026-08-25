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
configuration. The Safe-SeqTrack v25 method has not completed its registered mini or
full-nuScenes experiments, so this repository does not currently claim a gain,
stability, SOTA, or a causal benefit from physical time or memory.

## Method contract

- B0 is the nominal observation tracker and the only recursive state writer.
- B1 reads prediction-backed history boxes and physical timestamps. It supplies
  a prior and uncertainty but never replaces the observation.
- B1 uses a fixed kinematic anchor plus a bounded normalized residual. Its
  statistical sigma is trained by detached-mean beta-NLL and never changes the
  fixed B2 crop geometry (`2m/1m`). GRU is the default temporal backend; the
  parameter-matched CfC backend is an explicitly selected ablation.
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
- `models/ct_v2/cfc.py`: dependency-free optional B1 temporal cell.
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

The active mini configs are `25_b0.yaml`, `25_b1.yaml`,
`25_full_minus_b3.yaml` and `25_full.yaml`; the corresponding
`25_*_nuscenes_full.yaml` files are used only after mini validation. The v24
configs and outputs are frozen failure evidence and cannot resume into v25. B4 keeps
`cfgs/ct_v2/19_b4_decoder_alignment.yaml` and
`20_b4_decoder_anticollapse.yaml`.

All formal arms use `scratch_only`. `--init_checkpoint` is forbidden.
`--checkpoint` is accepted only for exact same-run epoch-boundary resume or
for evaluation. Enabled B0/B1/B2/B3 parameters are never frozen.

The B1 backend is selected without duplicating configs:

```bash
python main.py --cfg cfgs/ct_seqtrack/25_b1.yaml --path DATA_ROOT --tag b1_gru --b1-backend gru
python main.py --cfg cfgs/ct_seqtrack/25_b1.yaml --path DATA_ROOT --tag b1_cfc --b1-backend cfc
```

Both commands construct only the selected backend and train every enabled
module from epoch 0. A calibrated B1 checkpoint is evaluation-only and cannot
be used for resume or initialization.

## Experiment order

The authoritative protocol is
[docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md). In summary:

1. use the fixed candidate protocol in every arm: four B0 views and one
   canonical B2 view;
2. screen B1-GRU and B1-CfC independently from epoch0 on mini seed42;
3. fit B1 uncertainty on calibration tracklets and evaluate promotion on
   independent dev tracklets;
4. retrain the winning Full-B3 and Full arms independently from epoch0;
5. after mini analysis, run full nuScenes seed42 and then seeds52/62.

There is no training preflight, kill-test or intermediate stopping gate.
B1 backend promotion is a post-run decision from held-out mechanism metrics;
analysis artifacts never initialize another run or alter a run in progress.

## Validation

```bash
python tools/verify_ct_slimming.py verify
python -m pytest -q
```

Real-batch forward/backward, 100-step/resume parity and point/box visualization
are required server-side acceptance checks before a formal long run. They are not represented as
locally completed when the full Lightning/nuScenes environment is unavailable.
Engineering checkpoints are discarded and may not initialize formal runs.

Detailed method and evidence boundaries:

- [Safe-SeqTrack v25 runtime protocol](docs/SAFE_SEQTRACK_V25_PROTOCOL.md)
- [B0--B3 method](docs/CTSEQTRACK_B0_B3_METHOD.md)
- [formal experiment protocol](docs/EXPERIMENT_PROTOCOL.md)
- [formal tooling](docs/FORMAL_TOOLING.md)
- [source slimming gate](docs/SOURCE_SLIMMING_GATE.md)
- [active status](need_to_do.md)
- [historical evidence index](docs/HISTORY_EVIDENCE_INDEX.md)
- [slimming baseline](docs/slimming_baseline/README.md)
