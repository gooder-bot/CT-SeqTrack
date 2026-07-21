#!/usr/bin/env bash
# Run the M1/M2 engineering gates on server GPUs 2 and 3.
#
# Usage:
#   A1_CKPT=/abs/path/to/a1/last.ckpt \
#   DATA_ROOT=/home/lishengjie/data/nuscenes-mini \
#   PYTHON_BIN=python \
#   bash tools/run_m1_m2_gates_gpu23.sh
#
# This script intentionally runs only E0-E5 smoke/diagnostics.  It does not
# start the formal seed42 training run.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/home/lishengjie/data/nuscenes-mini}"
A1_CKPT="${A1_CKPT:?Set A1_CKPT to the frozen A1-order checkpoint}"
EXPECTED_A1_SHA256="${EXPECTED_A1_SHA256-a2fbffb1e5acae37adab3cb858e864857cc1d6c2231f9e0848df719614f24a82}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/output/m1_m2_gates}"
M2_CFG="cfgs/seqtrack3d_nuscenes_m2_proposal_innovation_engineering.yaml"

mkdir -p "${OUT_ROOT}/gpu2" "${OUT_ROOT}/gpu3"
cd "${PROJECT_ROOT}"

if [[ -n "${EXPECTED_A1_SHA256}" ]]; then
  ACTUAL_A1_SHA256="$(sha256sum "${A1_CKPT}" | awk '{print $1}')"
  if [[ "${ACTUAL_A1_SHA256}" != "${EXPECTED_A1_SHA256}" ]]; then
    echo "A1 checkpoint SHA256 mismatch" >&2
    echo "expected: ${EXPECTED_A1_SHA256}" >&2
    echo "actual:   ${ACTUAL_A1_SHA256}" >&2
    exit 1
  fi
fi

"${PYTHON_BIN}" tools/check_candidate_shared_se2.py
"${PYTHON_BIN}" tools/check_m1_m2_invariants.py
"${PYTHON_BIN}" tools/check_residual_dynamics.py
"${PYTHON_BIN}" tools/check_observability_gate.py

(
  set -euo pipefail
  "${PYTHON_BIN}" tools/check_candidate_shared_se2.py \
    --cfg "${M2_CFG}" \
    --path "${DATA_ROOT}" \
    --version v1.0-mini \
    --split mini_train \
    --twc

  CUDA_VISIBLE_DEVICES=2 "${PYTHON_BIN}" tools/check_m1_m2_model_equivalence.py \
    --cfg "${M2_CFG}" \
    --path "${DATA_ROOT}" \
    --version v1.0-mini \
    --split mini_train \
    --batch-size 2 \
    --weights "${A1_CKPT}"

  CUDA_VISIBLE_DEVICES=2 "${PYTHON_BIN}" tools/check_train_steps.py \
    --cfg "${M2_CFG}" \
    --path "${DATA_ROOT}" \
    --version v1.0-mini \
    --split mini_train \
    --batch-size 2 \
    --workers 0 \
    --max-steps 2 \
    --require-full-history \
    --weights "${A1_CKPT}" \
    --innovation-diagnostics \
    --innovation-warmup-epoch 0 \
    --checkpoint-every 0 \
    --log-file "${OUT_ROOT}/gpu2/standard_full.jsonl" \
    --checkpoint-dir "${OUT_ROOT}/gpu2/standard_full_ckpt" \
    --tag m1m2_standard_full

  # Exercise the model-level warmup branch: innovation must be exactly zero.
  CUDA_VISIBLE_DEVICES=2 "${PYTHON_BIN}" tools/check_train_steps.py \
    --cfg "${M2_CFG}" \
    --path "${DATA_ROOT}" \
    --version v1.0-mini \
    --split mini_train \
    --batch-size 2 \
    --workers 0 \
    --max-steps 1 \
    --require-full-history \
    --weights "${A1_CKPT}" \
    --innovation-diagnostics \
    --innovation-warmup-epoch 1 \
    --no-optimizer-step \
    --checkpoint-every 0 \
    --log-file "${OUT_ROOT}/gpu2/standard_warmup.jsonl" \
    --checkpoint-dir "${OUT_ROOT}/gpu2/standard_warmup_ckpt" \
    --tag m1m2_standard_warmup

  # Do not require full history here: this is the invalid/padded fallback pass.
  CUDA_VISIBLE_DEVICES=2 "${PYTHON_BIN}" tools/check_train_steps.py \
    --cfg "${M2_CFG}" \
    --path "${DATA_ROOT}" \
    --version v1.0-mini \
    --split mini_train \
    --batch-size 2 \
    --workers 0 \
    --max-steps 32 \
    --weights "${A1_CKPT}" \
    --innovation-diagnostics \
    --innovation-warmup-epoch 0 \
    --checkpoint-every 0 \
    --log-file "${OUT_ROOT}/gpu2/standard_fallback.jsonl" \
    --checkpoint-dir "${OUT_ROOT}/gpu2/standard_fallback_ckpt" \
    --tag m1m2_standard_fallback
) >"${OUT_ROOT}/gpu2/run.log" 2>&1 &
PID_GPU2=$!

(
  set -euo pipefail
  CUDA_VISIBLE_DEVICES=3 "${PYTHON_BIN}" tools/check_train_steps.py \
    --cfg "${M2_CFG}" \
    --protocol-cfg cfgs/seqtrack3d_nuscenes_a2_residual_dyn_vr_gap1124.yaml \
    --path "${DATA_ROOT}" \
    --version v1.0-mini \
    --split mini_train \
    --batch-size 2 \
    --workers 0 \
    --max-steps 2 \
    --require-full-history \
    --weights "${A1_CKPT}" \
    --innovation-diagnostics \
    --innovation-warmup-epoch 0 \
    --checkpoint-every 0 \
    --log-file "${OUT_ROOT}/gpu3/gap1124.jsonl" \
    --checkpoint-dir "${OUT_ROOT}/gpu3/gap1124_ckpt" \
    --tag m1m2_gap1124

  CUDA_VISIBLE_DEVICES=3 "${PYTHON_BIN}" tools/check_train_steps.py \
    --cfg "${M2_CFG}" \
    --protocol-cfg cfgs/seqtrack3d_nuscenes_a2_residual_dyn_vr_burst_drop.yaml \
    --path "${DATA_ROOT}" \
    --version v1.0-mini \
    --split mini_train \
    --batch-size 2 \
    --workers 0 \
    --max-steps 2 \
    --require-full-history \
    --weights "${A1_CKPT}" \
    --innovation-diagnostics \
    --innovation-warmup-epoch 0 \
    --checkpoint-every 0 \
    --log-file "${OUT_ROOT}/gpu3/burst_drop.jsonl" \
    --checkpoint-dir "${OUT_ROOT}/gpu3/burst_drop_ckpt" \
    --tag m1m2_burst_drop
) >"${OUT_ROOT}/gpu3/run.log" 2>&1 &
PID_GPU3=$!

set +e
wait "${PID_GPU2}"
STATUS_GPU2=$?
wait "${PID_GPU3}"
STATUS_GPU3=$?
set -e

echo "GPU2 gate status: ${STATUS_GPU2} (log: ${OUT_ROOT}/gpu2/run.log)"
echo "GPU3 gate status: ${STATUS_GPU3} (log: ${OUT_ROOT}/gpu3/run.log)"
if [[ "${STATUS_GPU2}" -ne 0 || "${STATUS_GPU3}" -ne 0 ]]; then
  exit 1
fi

echo "M1/M2 E0-E5 server gates completed. Formal training remains HOLD pending review."
