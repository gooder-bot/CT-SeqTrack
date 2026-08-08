#!/usr/bin/env bash
set -euo pipefail

# Run only after the mini protocol has passed its frozen promotion gates.
configs=(
  22_ct_joint_repaired_b0_full
  22_ct_joint_repaired_full
  22_ct_joint_repaired_minus_b1_full
  22_ct_joint_repaired_minus_b2_full
  22_ct_joint_repaired_minus_b3_full
)
seeds=(42 43 44)

for config in "${configs[@]}"; do
  for seed in "${seeds[@]}"; do
    python main.py \
      --cfg "cfgs/ct_v2/${config}.yaml" \
      --seed "${seed}" \
      --epoch 60 \
      --tag "paper-${config}-s${seed}" \
      "$@"
  done
done
