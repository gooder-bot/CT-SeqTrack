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

The seed-42 B0 2x2 development study is retained in Git history. The v25 mainline
uses the selected `reseed=true, rngshift=true` optimization protocol in all
four arms, but every arm still starts from random initialization. This is
reported as mixed-horizon past-GT re-anchored rollout training; inference never
re-anchors with GT. v25 has not yet completed server experiments, so this
repository does not currently claim a gain, statistical stability, SOTA, or a
causal benefit from physical time or memory.

## Paper-facing components

- `models/ctseqtrack.py`: formal composition root; normalizes the single
  `ct_variant` interface and rejects historical branch switches.
- `ctseqtrack/contracts.py`: internal B0--B3 ownership contracts.
- `ctseqtrack/model/`: B1 prior, extension-only B2, calibrated B3, forward and losses.
- `ctseqtrack/data/`: training/inference sample construction, search geometry,
  deterministic evidence sampling, and online recursive state.
- `ctseqtrack/runtime/calibration.py`: disjoint scene-level threshold selection and
  selective closed-loop audit.
- `ctseqtrack/runtime/acquisition.py`: checkpoint-free acquisition preflight and
  artifact-derived targetness class weights.
- `ctseqtrack/runtime/contracts.py`: scratch/resume and B2-promotion identity checks.
- `ctseqtrack/runtime/checkpointing.py`: optimizer/scaler/RNG and epoch-boundary
  recursive-state reset contract for exact same-run recovery.
- `ctseqtrack/runtime/evaluation.py`: the single paper-facing joint-Full
  diagnostic schema.
- `tools/compare_ct_module_audits.py`: shared-prefix parameter-hash audit.

The evaluator still receives flat dictionaries. Existing compatibility keys
needed by the v25 analysis scripts are aliases only; they do not
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
- matched B1 temporal-backbone screen: `b1_gru_mini_seed42.yaml` and
  `b1_cfc_mini_seed42.yaml`.
- The complete v25 protocol and server workflow are in
  `docs/CTSEQTRACK_V25_PROTOCOL.md`.

Historical v24 and exploratory configurations were removed from the working
tree after the `ctseqtrack-v25-pre-cleanup-9ed2afc` recovery tag and external
Git bundle were created. They remain reproducible from Git history without
coexisting with the paper-facing configuration surface.

All formal arms use `scratch_only`, observation recursive state, strict module
isolation, and last-epoch checkpoints. The `--init_checkpoint` entry no longer exists.
`--checkpoint` is accepted only for an exact same-run epoch-boundary resume or
for evaluation.

## Scientific contracts

### B1: prior, not output

B1 consumes recursive history boxes, valid masks, and physical timestamps. It
predicts a mean, direction, heteroscedastic uncertainty, and CV fallback. Its
mean is never used as the final box. The first B2 version uses fixed support
margins; learned sigma remains a B3 feature and an analysis target.

`motion_v3_temporal_backend` selects `gru` or the dependency-free full-gated
CfC cell. CfC changes only ordered transition aggregation: it receives the
same projected transition features and the corresponding physical pair gaps,
while the query gap, kinematic anchor, bounded residual, uncertainty heads,
losses, and B1 output contract remain shared. The matched CfC width of 105
gives 74,537 cell parameters versus 74,496 for the GRU. See the
[CfC paper](https://arxiv.org/abs/2106.13898) and
[reference implementation](https://github.com/raminmh/CfC/blob/main/torch_cfc.py).

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
fail-closed.

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
python main.py --cfg cfgs/ct_seqtrack/25_b0.yaml --path DATA_ROOT
python main.py --cfg cfgs/ct_seqtrack/25_b1.yaml --path DATA_ROOT
python main.py --cfg cfgs/ct_seqtrack/25_full_minus_b3.yaml \
  --path DATA_ROOT --acquisition_preflight PREFLIGHT.json
python main.py --cfg cfgs/ct_seqtrack/25_full.yaml \
  --path DATA_ROOT --acquisition_preflight PREFLIGHT.json \
  --b2_method_promotion B2_PROMOTION.json
```

The reliability-aware physical mixture implementation and its checkpoint-free
server preflight/calibration workflow are documented in
[`docs/RA_PMM_B1.md`](docs/RA_PMM_B1.md).

After Full training, export candidate action rows on held-out calibration
tracklets and calibrate:

```bash
python tools/calibrate_ct_actions.py \
  --rows CALIBRATION_ROWS.jsonl \
  --checkpoint LAST.ckpt \
  --config cfgs/ct_seqtrack/25_full.yaml \
  --tracklet-manifest CALIBRATION_TRACKLETS.json \
  --output ACTION_CALIBRATION.json
```

Selective evaluation must supply the same checkpoint, artifact, and manifest
file hash:

```bash
python main.py --test --cfg cfgs/ct_seqtrack/25_full.yaml \
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
experiment checklist is [`need_to_do.md`](need_to_do.md). The server-only
acceptance flow is in
[`docs/CTSEQTRACK_V25_SERVER_ACCEPTANCE.md`](docs/CTSEQTRACK_V25_SERVER_ACCEPTANCE.md).
The conservative refactor scope and local verification record are in
[`docs/CTSEQTRACK_V25_CLEANUP_REPORT.md`](docs/CTSEQTRACK_V25_CLEANUP_REPORT.md).
