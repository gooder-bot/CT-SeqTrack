# P0-C frozen protocol artifacts

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
CKPT=/absolute/path/to/frozen-a2.ckpt
CFG=cfgs/seqtrack3d_nuscenes_p0c_a2_standard_train_gap1124_eval.yaml
SHUFFLE=protocols/manifests/nuscenes_mini_test_gap1124_shuffled_dt_seed42.json

python main.py --test --cfg "$CFG" --checkpoint "$CKPT" --seed 42 \
  --dynamics_time_mode true --log_dir output/p0c-gap1124-true-dt

python main.py --test --cfg "$CFG" --checkpoint "$CKPT" --seed 42 \
  --dynamics_time_mode fixed --dynamics_fixed_delta_t 0.5 \
  --log_dir output/p0c-gap1124-fixed-dt

python main.py --test --cfg "$CFG" --checkpoint "$CKPT" --seed 42 \
  --dynamics_time_mode shuffled --dynamics_time_manifest "$SHUFFLE" \
  --log_dir output/p0c-gap1124-shuffled-dt
```

Each output root contains `run_provenance.json` with the commit, dirty status,
resolved-config hash, checkpoint hash, endpoint selection hash, cadence manifest
hash, and shuffled mapping hash.
