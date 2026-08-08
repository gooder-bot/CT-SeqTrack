#!/usr/bin/env bash
set -euo pipefail

# Full paper-facing mini protocol: 9 variants x 3 seeds x 60 epochs.
# Extra arguments (for example --path or --workers) are forwarded to main.py.
configs=(
  22_ct_joint_repaired_b0
  22_ct_joint_repaired
  22_ct_joint_repaired_minus_b1
  22_ct_joint_repaired_minus_b2
  22_ct_joint_repaired_minus_b3
  22_ct_joint_repaired_fault_old_recursive
  22_ct_joint_repaired_fault_presence_hard
  22_ct_joint_repaired_fault_alpha_self
  22_ct_joint_repaired_fault_kinematic_search
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
