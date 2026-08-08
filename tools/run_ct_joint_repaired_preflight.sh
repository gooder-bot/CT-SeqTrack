#!/usr/bin/env bash
set -euo pipefail

# Plumbing-only gate before any 60-epoch matrix.  Short ordered windows exercise
# candidate-0 writeback; Full receives 20 batches so an H=3 shadow can run.
# Results from this script must never be used for model selection.
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

for config in "${configs[@]}"; do
  train_batches=2
  if [[ "${config}" == "22_ct_joint_repaired" ]]; then
    # The first frame has insufficient history; a longer Full smoke window is
    # needed to observe at least one structurally valid H=3 shadow rollout.
    train_batches=20
  fi
  python main.py \
    --cfg "cfgs/ct_v2/${config}.yaml" \
    --seed 42 \
    --epoch 1 \
    --workers 0 \
    --save_top_k 0 \
    --limit_train_batches "${train_batches}" \
    --limit_val_batches 2 \
    --tag "preflight-${config}" \
    "$@"
done
