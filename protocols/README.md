# P0-C frozen protocol artifacts

## Current status (2026-07-20)

The protocol-build stage has passed on clean commit `343145d`:

- val/test gap1124 selection: `91/106` tracklets and `1257/2285` frames;
- test endpoint selection SHA256: `85e5603c941030b050adab7876a275e654d9da328c859c1101e32940f9649f6f`;
- shuffled-time mapping: `1257` endpoints and `1166` transitions;
- real-batch check: `P0-C true/fixed/shuffled batch invariance: PASS`.

The three-way inference-only evaluation has also completed with the same
standard-trained frozen A2 checkpoint. True-dt scored `55.2247 / 66.8854`,
fixed-dt `54.7872 / 66.3624`, and shuffled-dt `55.3480 / 66.8298`
(Success / Precision). True-dt did not beat both controls or reach the
preregistered `+0.5 / +1.0` margin, so the final decision is
`NO_GO_P0C_A2_TRUE_DT_PROMOTION`. Do not extend this A2 run to burst-drop,
unseen fixed-gap, or more seeds. The commands below remain as reproduction
instructions. The validated report is
`compare_results/reports/p0c_frozen_protocol_validation_20260720.md`.

`protocols/manifests/` stores dataset-derived cadence and shuffled-time manifests.
The JSON files are intentionally gitignored: generate them from a clean source
commit on the machine that has nuScenes, archive them with the experiment, and
identify them by the content/file SHA256 printed by the builders.

The first P0-C entry configuration is:

```text
cfgs/seqtrack3d_nuscenes_p0c_a2_standard_train_gap1124_eval.yaml
```

It keeps `mini_train` at the standard cadence and applies a frozen `gap1124`
schedule only to validation/test. Build the two role-specific cadence manifests
after committing the code:

```bash
mkdir -p protocols/manifests

python tools/build_virtual_rate_manifest.py \
  --cfg cfgs/seqtrack3d_nuscenes_p0c_a2_standard_train_gap1124_eval.yaml \
  --role val --split mini_val --mode gap_pattern \
  --gap-pattern 1 1 2 4 --seed 42 --max-gap 5 \
  --output protocols/manifests/nuscenes_mini_val_gap1124_seed42.json

python tools/build_virtual_rate_manifest.py \
  --cfg cfgs/seqtrack3d_nuscenes_p0c_a2_standard_train_gap1124_eval.yaml \
  --role test --split mini_val --mode gap_pattern \
  --gap-pattern 1 1 2 4 --seed 42 --max-gap 5 \
  --output protocols/manifests/nuscenes_mini_test_gap1124_seed42.json
```

Build the offline test-split time permutation. It is tied to the exact endpoint
selection hash of the test cadence manifest:

```bash
python tools/build_dynamics_time_manifest.py \
  --cfg cfgs/seqtrack3d_nuscenes_p0c_a2_standard_train_gap1124_eval.yaml \
  --role test --split mini_val --seed 42 \
  --output protocols/manifests/nuscenes_mini_test_gap1124_shuffled_dt_seed42.json
```

Before any long evaluation, prove that the three controls load identical frames,
crops, candidates, labels, and physical-time fields:

```bash
python tools/check_p0c_time_controls.py --self-test

python tools/check_p0c_time_controls.py \
  --cfg cfgs/seqtrack3d_nuscenes_p0c_a2_standard_train_gap1124_eval.yaml \
  --role test --split mini_val \
  --shuffled-manifest protocols/manifests/nuscenes_mini_test_gap1124_shuffled_dt_seed42.json
```

The second command must end with:

```text
P0-C true/fixed/shuffled batch invariance: PASS
```

Evaluate one frozen A2 checkpoint three times; do not retrain or change a
threshold between these commands:

```bash
EXPECTED_CKPT_SHA=b508f9580d52c7f90cf7d4d09ac38ad6043481a42cc84ef3fcdca63924ac87ad
CFG=cfgs/seqtrack3d_nuscenes_p0c_a2_standard_train_gap1124_eval.yaml
SHUFFLE=protocols/manifests/nuscenes_mini_test_gap1124_shuffled_dt_seed42.json

CKPT=""
while IFS= read -r candidate; do
  candidate_sha=$(sha256sum "$candidate" | awk '{print $1}')
  echo "$candidate_sha  $candidate"
  if [ "$candidate_sha" = "$EXPECTED_CKPT_SHA" ]; then
    CKPT="$candidate"
    break
  fi
done < <(
  find output -type f -path '*a2_order_dyn*' \
    \( -name 'last.ckpt' -o -name 'epoch=59-step=75720.ckpt' \) \
    2>/dev/null
)

if [ -z "$CKPT" ]; then
  echo "CHECKPOINT_NOT_FOUND: upload the frozen checkpoint before evaluation"
else
  echo "CHECKPOINT_READY: $CKPT"
fi
```

Do not continue while `CKPT` is empty. In an interactive SSH shell, do not use
`exit 1` as a pasted guard because it logs out the session. If the expected
checkpoint is absent on the server, upload the verified local file to
`output/frozen_checkpoints/a2_order_dyn_seed42_60ep_last.ckpt`, reconnect, set
`CKPT` to that path, and verify the SHA256 again.

Verify all inputs without closing the session:

```bash
test -n "$CKPT" && test -f "$CKPT" && echo "checkpoint file: OK"
test -f "$SHUFFLE" && echo "shuffled manifest: OK"
git status --short --untracked-files=no
sha256sum "$CKPT" "$SHUFFLE"
```

Only proceed when the checkpoint line is exactly
`b508f9580d52c7f90cf7d4d09ac38ad6043481a42cc84ef3fcdca63924ac87ad`.

```bash
mkdir -p \
  output/p0c-gap1124-true-dt \
  output/p0c-gap1124-fixed-dt \
  output/p0c-gap1124-shuffled-dt

CUDA_VISIBLE_DEVICES=0 python main.py --test --cfg "$CFG" \
  --checkpoint "$CKPT" --seed 42 --workers 12 \
  --dynamics_time_mode true \
  --log_dir output/p0c-gap1124-true-dt \
  2>&1 | tee output/p0c-gap1124-true-dt/console.log

CUDA_VISIBLE_DEVICES=0 python main.py --test --cfg "$CFG" \
  --checkpoint "$CKPT" --seed 42 --workers 12 \
  --dynamics_time_mode fixed --dynamics_fixed_delta_t 0.5 \
  --log_dir output/p0c-gap1124-fixed-dt \
  2>&1 | tee output/p0c-gap1124-fixed-dt/console.log

CUDA_VISIBLE_DEVICES=0 python main.py --test --cfg "$CFG" \
  --checkpoint "$CKPT" --seed 42 --workers 12 \
  --dynamics_time_mode shuffled --dynamics_time_manifest "$SHUFFLE" \
  --log_dir output/p0c-gap1124-shuffled-dt \
  2>&1 | tee output/p0c-gap1124-shuffled-dt/console.log
```

Run these commands sequentially, not concurrently. After completion, inspect
the three provenance files and archive the protocol evidence with the runs:

```bash
for run in true-dt fixed-dt shuffled-dt; do
  echo "===== $run ====="
  cat "output/p0c-gap1124-${run}/run_provenance.json"
done

tar -czf p0c_gap1124_triplet_$(date +%Y%m%d_%H%M%S).tar.gz \
  protocols/manifests/nuscenes_mini_val_gap1124_seed42.json \
  protocols/manifests/nuscenes_mini_test_gap1124_seed42.json \
  protocols/manifests/nuscenes_mini_test_gap1124_shuffled_dt_seed42.json \
  output/p0c-protocol-build \
  output/p0c-gap1124-true-dt \
  output/p0c-gap1124-fixed-dt \
  output/p0c-gap1124-shuffled-dt
```

All three provenance files must have the same commit, checkpoint SHA256, source
config SHA256, cadence manifest/selection hash, seed, and evaluated endpoints.
The resolved-config hashes are expected to differ because `log_dir` and the
declared effective-time fields differ; no other scientific setting may change.
The preregistered promotion rule is that true-dt must beat both controls by at
least `+0.5` Success and `+1.0` Precision before extending the protocol to
burst-drop or unseen fixed-gap schedules.

Observed result: the minimum true-dt improvement was `-0.1233` Success and
`+0.0557` Precision, so promotion is false.

Each output root contains `run_provenance.json` with the commit, dirty status,
resolved-config hash, checkpoint hash, endpoint selection hash, cadence manifest
hash, and shuffled mapping hash.
