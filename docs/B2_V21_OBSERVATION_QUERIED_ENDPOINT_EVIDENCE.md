# B2-v2.1: Observation-Queried Endpoint Evidence

This implementation keeps the original B0 point stream unchanged and adds an
independent 128-point trajectory-endpoint evidence branch. It is intentionally
separate from legacy B2-v2 and the legacy long-tube search configurations.

## Experiment configurations

- `cfgs/ct_v2/08_seqtrack3d_search_v21.yaml`: B0 + Search-v2.1.
- `cfgs/ct_v2/09_ct_motion_search_v21.yaml`: B0 + Motion-v3 + Search-v2.1.
- `cfgs/ct_v2/01_seqtrack3d_baseline.yaml`: matched B0 control.

Both V2.1 configurations use 1024 unchanged B0 points, 128 independently
sampled endpoint points, a 64-point initial extension quota, an observation
query dimension of 32, and evidence-pooling temperature 0.5.

## Training contract

```bash
python tools/ct_v2/run.py train --variant baseline \
  --seed 42 --batch-size 16 --workers 4 --epochs 60 \
  --check-val-every-n-epoch 5 --preloading

python tools/ct_v2/run.py train --variant search_v21 \
  --seed 42 --batch-size 16 --workers 4 --epochs 60 \
  --check-val-every-n-epoch 5 --preloading

python tools/ct_v2/run.py train --variant motion_search_v21 \
  --seed 42 --batch-size 16 --workers 4 --epochs 60 \
  --check-val-every-n-epoch 5 --preloading
```

All three runs are scratch runs. Fusion is exactly disabled for epochs 0--9,
ramps during epochs 10--19, and is fully active afterwards.

## Same-checkpoint proposal attribution

The test-only `--proposal-mode` option accepts `obs`, `obs_motion`,
`obs_search`, and `full`:

```bash
python tools/ct_v2/run.py test --variant motion_search_v21 \
  --checkpoint /path/to/epoch60.ckpt --proposal-mode obs_search
```

Each V2.1 test writes the following files below the Lightning log directory:

- `proposal_diagnostics/proposal_endpoints.csv`
- `proposal_diagnostics/proposal_tracklets.csv`

The endpoint export includes the observation, motion, search and final XY
errors; intrinsic candidate validity; oracle helpful labels; help
probabilities; step ratios; applied weights; point-source counts; and Search
evidence statistics. Candidate availability remains visible even when a
proposal mode masks that candidate from the final correction.

## Verification

Run the focused regression suite with:

```bash
python -m unittest tests.test_ct_v2 -v
```

The suite covers source quotas, sparse and empty endpoint crops, deterministic
branch RNG, padded-point masking, observation-query behavior, gradient
isolation, bounded fusion, exact observation fallback, legacy configurations,
and V2.1 configuration composition.
