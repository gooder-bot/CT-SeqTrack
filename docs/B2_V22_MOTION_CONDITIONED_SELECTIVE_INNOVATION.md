# B2-v2.2 Motion-Conditioned Selective Innovation

## Implemented contract

B2-v2.2 is an isolated path. It does not alter or replace B2-v2.1 advantage
fusion or the B3 CRPA prototype.

```text
recursive boxes + dt -> deterministic support anchor -> independent 128 points
recursive boxes + dt -> frozen B1 motion prior / feature / uncertainty
B0 main 1024 points -> frozen observation box / feature
support-relative + motion-relative endpoint evidence
  -> MotionConditionedSearchRefiner
  -> motion-search refined candidate centred on B1
observation / motion / motion-search
  -> SignedHorizonInnovationRouter
  -> abstain or one bounded Top-1 correction
```

The support anchor is exposed as `search_support_anchor_xy` and never becomes a
proposal. The raw vote is `search_raw_vote_xy`; the proposal used by the router
is `motion_search_refined_xy`.

Invalid endpoint rows have finite zero statistics, including `search_raw_ess=0`
and `search_normalized_ess=0`. The router reads only normalized ESS. Point
availability and learned foreground presence are separate outputs.

During refiner fitting, `aux_estimation_boxes` is bitwise identical to
`observation_aux_estimation_boxes`. All B0/B1 modules and the router are frozen;
only `motion_conditioned_search_refiner.*` parameters enter the optimizer.

## Stage 1: compose initialization

```bash
python tools/build_b2_v22_init_checkpoint.py \
  --base-checkpoint <b2-v2-epoch60.ckpt> \
  --search-checkpoint <b2-v2.1-full-epoch60.ckpt> \
  --output <b2-v22-init.ckpt>
```

The base checkpoint supplies B0/B1. Only v2.1 point encoder/source embedding,
observation query/matching, targetness, and vote tensors are renamed into the
new refiner. Advantage-gate and B3-router tensors are excluded. New motion
geometry, source fusion, presence, and signed-router layers keep v2.2 defaults.

## Stage 2: train the refiner

```bash
python tools/ct_v2/run.py train \
  --variant b2_v22_refiner \
  --init-checkpoint <b2-v22-init.ckpt> \
  --epochs 20 --batch-size 16 --seed 42
```

The added loss is exactly `0.1 match + 0.2 targetness + 1.0 vote + 1.0 raw
proposal + 1.0 refined proposal + 0.2 presence`. Frozen legacy losses may be
logged but cannot update B0/B1.

## Stage 3: export true H=3 counterfactuals

```bash
python tools/export_selective_rollouts.py \
  --checkpoint <b2-v22-refiner-final.ckpt> \
  --output <rollout-directory> \
  --split mini_train --horizon 3 --gamma 0.8 --seed 42 \
  --path <nuscenes-root>
```

Only `mini_train` is accepted; `mini_val` is rejected. At each eligible
recursive state the exporter runs one observation branch and
`2 candidates x 3 non-zero steps`. Every branch rebuilds future crops and
forwards; after the first intervention it follows observation policy. GT is
consulted only after predictions exist, for IoU/distance cost. Negative gains
remain signed in `selective_rollouts.npz`.

Tracklets receive a deterministic 70/15/15 train/dev/calibration hash split;
frames from one tracklet cannot cross partitions.

## Stage 4: train and calibrate the router

```bash
python tools/train_signed_horizon_router.py \
  --rollouts <rollout-directory> \
  --output <signed-router.pt> \
  --seed 42
```

q10/q50 use pinball loss; the three-way step is supervised only when best gain
exceeds 0.02. Dev tracklets early-stop training. Calibration chooses one q10
threshold and requires helpful precision at least 75%, harm at most 10%, and
coverage between 5% and 25%. A failure writes an auditable sidecar/JSON but
returns an error and cannot be packaged.

## Stage 5: package and evaluate one checkpoint

```bash
python tools/package_selective_checkpoint.py \
  --candidate-checkpoint <b2-v22-refiner-final.ckpt> \
  --router <signed-router.pt> \
  --output <b2-v22-selective-final.ckpt>

python tools/ct_v2/run.py test --variant b2_v22_selective \
  --checkpoint <b2-v22-selective-final.ckpt> --proposal-mode obs
python tools/ct_v2/run.py test --variant b2_v22_selective \
  --checkpoint <b2-v22-selective-final.ckpt> --proposal-mode obs_motion
python tools/ct_v2/run.py test --variant b2_v22_selective \
  --checkpoint <b2-v22-selective-final.ckpt> --proposal-mode obs_motion_search
python tools/ct_v2/run.py test --variant b2_v22_selective \
  --checkpoint <b2-v22-selective-final.ckpt> --proposal-mode full_selective
```

`obs_search` is rejected because v2.2 has no search candidate independent of
B1. Per-frame diagnostics include raw/refined errors, presence, normalized/raw
ESS, q10/q50, selection, step, threshold, abstention, and correction.

Do not claim improvement unless final `full_selective` exceeds 54.132 Success
and 64.755 Precision, is not below either auxiliary mode on either metric,
valid-foreground refined RMSE beats motion and raw search, and mini_val helpful
precision is at least 70% with harm at most 10%.

`tests/test_selective_innovation.py` covers exact fallback, Top-1/discrete-step
behavior, caps, detached router inputs, negative gains, calibration,
motion-centred refinement, and finite invalid-row statistics.
