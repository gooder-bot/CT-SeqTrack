# CT-SeqTrack

CT-SeqTrack studies observation-preserving evidence recovery for irregular-time
3D single-object tracking. The paper-facing implementation is `CTSEQTRACK` and
has one fixed chain:

```text
B0 Observation Tracker
  -> B1 Physical-Time Prior
  -> B2 Evidence Acquirer
  -> B3 Selective Updater
```

The central rule is deliberately conservative: motion and history never replace
the current observation directly. B1 only defines a physical-time prior and a
fixed first-version support region; B2 must recover target evidence from
extension-only points; B3 may apply a bounded residual only after a matching
held-out calibration artifact passes the empirical risk gates. Every other case
returns the B0 observation exactly.

The seed-42 B0 2x2 development study is archived in v24. The new v25 mainline
uses the selected `reseed=true, rngshift=true` optimization protocol in all
four arms, but every arm still starts from random initialization. This is
reported as mixed-horizon past-GT re-anchored rollout training; inference never
re-anchors with GT. v25 has not yet completed server experiments, so this
repository does not currently claim a gain, statistical stability, SOTA, or a
causal benefit from physical time or memory.

## Paper-facing components

- `models/ctseqtrack.py`: formal composition root; normalizes the single
  `ct_variant` interface and rejects historical branch switches.
- `models/ct_v2/pipeline_contracts.py`: internal B0--B3 ownership contracts.
- `models/ct_v2/evidence_memory.py`: extension-only B2 and action-risk B3.
- `utils/action_calibration.py`: disjoint scene-level threshold selection and
  selective closed-loop audit for v25 (plus the archived v24 interface).
- `utils/acquisition_metrics.py`: checkpoint-free acquisition preflight and
  artifact-derived targetness class weights.
- `utils/online_contract.py`: scratch/resume and B2-promotion identity checks.
- `tools/compare_ct_module_audits.py`: shared-prefix parameter-hash audit.

The evaluator still receives flat dictionaries. Existing v23 output keys that
are needed by analysis scripts are compatibility aliases only; they do not
define additional paper modules.

## Formal variants

| Variant | Trainable modules from random initialization | Deployed output during training |
|---|---|---|
| `b0` | B0 | observation |
| `b1` | B0+B1 | observation |
| `full_minus_b3` | B0+B1+B2 | observation; B2 is shadow/counterfactual |
| `full` | B0+B1+B2+B3 | observation; actions only after calibration |

Configurations are under `cfgs/ct_seqtrack/`:

- v25 mini: `25_b0.yaml`, `25_b1.yaml`, `25_full_minus_b3.yaml`,
  `25_full.yaml`.
- v25 full-data overrides: the corresponding `25_*_full.yaml` files.
- v25 controls: live selector vs uniform, B2 target scale 1.0 vs 1.25,
  and true/fixed/shuffled physical time.
- The complete v25 protocol and server workflow are in
  `docs/CTSEQTRACK_V25_PROTOCOL.md`.

The following v24 configurations remain reproducible and unchanged:

- `24_b0.yaml`
- `24_b1.yaml`
- `24_full_minus_b3.yaml`
- `24_full_minus_b3_causal_uniform.yaml`
- `24_full_minus_b3_presence_c0.yaml`
- `24_full.yaml`
- `24_full_cv.yaml`
- `24_full_minus_b3_cv.yaml`
- `24_{b1,full}_time_{fixed,shuffled}.yaml`
- `24_full_minus_b3_time_{fixed,shuffled}.yaml`
- `24_full_memory_{real,empty,time_misaligned}.yaml`

All formal arms use `scratch_only`, observation recursive state, strict module
isolation, and last-epoch checkpoints. `--init_checkpoint` is forbidden.
`--checkpoint` is accepted only for an exact same-run epoch-boundary resume or
for evaluation.

## Scientific contracts

### B1: prior, not output

B1 consumes recursive history boxes, valid masks, and physical timestamps. It
predicts a mean, direction, heteroscedastic uncertainty, and CV fallback. Its
mean is never used as the final box. The first B2 version uses fixed support
margins; learned sigma remains a B3 feature and an analysis target.

### B2: identifiable new evidence

- Targetness is trained on every valid extension point, with class weights
  signed by the acquisition preflight artifact.
- Vote/raw-center regression is active only for rows whose extension contains
  target points.
- An absent-target row trains targetness/background and presence-negative; it
  cannot regress the GT center.
- c1/c2 are selected from live-B1 temporal gaps without GT; they update B1/B2,
  train presence, and never update B0/B3 or the recursive state.
- Structural availability and evidence presence are separate conditions.
- The no-extension counterfactual is exactly the observation.

Memory is optional B2 context, not a standalone contribution. `time_misaligned`
preserves token values, shape, channels, and masks while mismatching the three
history blocks with their time/pose metadata. Memory enters the final method
only if `real` beats both `empty` and `time_misaligned` under matched scratch
training and paired confidence intervals.

### B3: calibrated action reliability

B3 predicts helpful probability, harmful probability, expected center gain,
and expected IoU gain from detached upstream evidence. An action requires:

```text
structural availability
and presence >= calibrated threshold
and helpful * (1 - harmful) >= calibrated action threshold
and a finite residual bounded by radius(dt)
```

The calibration artifact is bound to the checkpoint SHA256, action-defining
configuration identity, disjoint selection/audit scene-manifest identities,
the actual scene populations, score definition, thresholds, and its own content
hash. Missing, failed, stale, overlapping, or undersized calibration is
fail-closed. The archived v24 interface retains its tracklet-level manifest.

## Running the staged protocol

First export a complete checkpoint-free acquisition pass and build its signed
preflight artifact:

```bash
python tools/export_ct_acquisition_preflight_rows.py --help
python tools/preflight_ct_acquisition.py --help
```

Then train each arm independently from scratch. B2 arms require
`--acquisition_preflight`; Full additionally requires the method-only B2
promotion manifest. The manifest transfers no weights.

```bash
python main.py --cfg cfgs/ct_seqtrack/24_b0.yaml --path DATA_ROOT
python main.py --cfg cfgs/ct_seqtrack/24_b1.yaml --path DATA_ROOT
python main.py --cfg cfgs/ct_seqtrack/24_full_minus_b3.yaml \
  --path DATA_ROOT --acquisition_preflight PREFLIGHT.json
python main.py --cfg cfgs/ct_seqtrack/24_full.yaml \
  --path DATA_ROOT --acquisition_preflight PREFLIGHT.json \
  --b2_method_promotion B2_PROMOTION.json
```

After Full training, export candidate action rows on held-out calibration
tracklets and calibrate:

```bash
python tools/calibrate_ct_actions.py \
  --rows CALIBRATION_ROWS.jsonl \
  --checkpoint LAST.ckpt \
  --config cfgs/ct_seqtrack/24_full.yaml \
  --tracklet-manifest CALIBRATION_TRACKLETS.json \
  --output ACTION_CALIBRATION.json
```

Selective evaluation must supply the same checkpoint, artifact, and manifest
file hash:

```bash
python main.py --test --cfg cfgs/ct_seqtrack/24_full.yaml \
  --checkpoint LAST.ckpt --path DATA_ROOT --proposal-mode selective \
  --ct_action_calibration_path ACTION_CALIBRATION.json \
  --ct_calibration_tracklet_manifest_sha256 MANIFEST_SHA256
```

Use `python tools/report_ct_risk_coverage.py --help` for risk--coverage output
and `python tools/report_ct_b1.py --help` for mean-vs-CV, NLL, support and
registered B1 strata. Use `python tools/promote_ct_memory.py --help` for the
paired memory gate.
Use `python tools/compare_ct_module_audits.py ...` to invalidate arms whose
supposedly shared B0/B1/B2 prefixes diverged.

## Verification and experiment status

```bash
python -m pytest -q
```

The detailed implementation and claim/evidence map is in
[`docs/CTSEQTRACK_B0_B3_METHOD.md`](docs/CTSEQTRACK_B0_B3_METHOD.md). The active
experiment checklist is [`need_to_do.md`](need_to_do.md). Historical designs,
negative results, and prior reports remain in `compare_results/` and
`docs/legacy/`; they are not evidence for the current B0--B3 method.
