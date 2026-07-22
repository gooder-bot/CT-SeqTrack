#!/usr/bin/env bash
# Run the M1/M2 engineering gates on server GPUs 2 and 3.
#
# Usage:
#   A1_CKPT=/abs/path/to/a1/last.ckpt \
#   DATA_ROOT=/home/lishengjie/data/nuscenes-mini \
#   PYTHON_BIN=python \
#   bash tools/run_m1_m2_gates_gpu23.sh
#
# To run all gates sequentially on only one card, additionally set either:
#   SINGLE_GPU=2
# or:
#   SINGLE_GPU=3
#
# This script intentionally runs only E0-E5 smoke/diagnostics.  It does not
# start the formal seed42 training run.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/home/lishengjie/data/nuscenes-mini}"
A1_CKPT="${A1_CKPT:?Set A1_CKPT to the frozen A1-order checkpoint}"
EXPECTED_A1_SHA256="${EXPECTED_A1_SHA256-a2fbffb1e5acae37adab3cb858e864857cc1d6c2231f9e0848df719614f24a82}"
M2_CFG="cfgs/seqtrack3d_nuscenes_m2_proposal_innovation_engineering.yaml"
GPU_STANDARD="${GPU_STANDARD:-2}"
GPU_PROTOCOL="${GPU_PROTOCOL:-3}"
SINGLE_GPU="${SINGLE_GPU:-}"
if [[ -n "${SINGLE_GPU}" ]]; then
  GPU_STANDARD="${SINGLE_GPU}"
  GPU_PROTOCOL="${SINGLE_GPU}"
fi

GIT_COMMIT="$(git -C "${PROJECT_ROOT}" rev-parse HEAD)"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/output/m1_m2_gates_${GIT_COMMIT:0:7}_${RUN_STAMP}}"
if [[ -e "${OUT_ROOT}" ]]; then
  echo "OUT_ROOT already exists; choose a fresh path so JSONL files cannot mix runs:" >&2
  echo "${OUT_ROOT}" >&2
  exit 1
fi
cd "${PROJECT_ROOT}"

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Tracked source changes are not committed; refusing a non-reproducible gate run." >&2
  git status --short --untracked-files=no >&2
  exit 1
fi
mkdir -p "${OUT_ROOT}/gpu2" "${OUT_ROOT}/gpu3"

if [[ -n "${EXPECTED_A1_SHA256}" ]]; then
  ACTUAL_A1_SHA256="$(sha256sum "${A1_CKPT}" | awk '{print $1}')"
  if [[ "${ACTUAL_A1_SHA256}" != "${EXPECTED_A1_SHA256}" ]]; then
    echo "A1 checkpoint SHA256 mismatch" >&2
    echo "expected: ${EXPECTED_A1_SHA256}" >&2
    echo "actual:   ${ACTUAL_A1_SHA256}" >&2
    exit 1
  fi
fi

{
  echo "schema=ct_seqtrack.m1_m2_gate_provenance.v1"
  echo "git_commit=${GIT_COMMIT}"
  echo "run_stamp=${RUN_STAMP}"
  echo "data_root=${DATA_ROOT}"
  echo "a1_checkpoint=${A1_CKPT}"
  echo "a1_sha256=${ACTUAL_A1_SHA256:-not_checked}"
  echo "m2_config=${M2_CFG}"
  echo "gpu_standard=${GPU_STANDARD}"
  echo "gpu_protocol=${GPU_PROTOCOL}"
  echo "single_gpu=${SINGLE_GPU}"
  sha256sum "${M2_CFG}" \
    cfgs/seqtrack3d_nuscenes_a2_residual_dyn_vr_gap1124.yaml \
    cfgs/seqtrack3d_nuscenes_a2_residual_dyn_vr_burst_drop.yaml
} >"${OUT_ROOT}/provenance.txt"

"${PYTHON_BIN}" tools/check_candidate_shared_se2.py
"${PYTHON_BIN}" tools/check_m1_m2_invariants.py
"${PYTHON_BIN}" tools/check_residual_dynamics.py
"${PYTHON_BIN}" tools/check_observability_gate.py
"${PYTHON_BIN}" tools/check_p0c_time_controls.py --self-test
"${PYTHON_BIN}" tools/export_m0_endpoints.py --self-test

(
  set -euo pipefail
  "${PYTHON_BIN}" tools/check_candidate_shared_se2.py \
    --cfg "${M2_CFG}" \
    --path "${DATA_ROOT}" \
    --version v1.0-mini \
    --split mini_train \
    --twc

  CUDA_VISIBLE_DEVICES="${GPU_STANDARD}" "${PYTHON_BIN}" tools/check_m1_m2_model_equivalence.py \
    --cfg "${M2_CFG}" \
    --path "${DATA_ROOT}" \
    --version v1.0-mini \
    --split mini_train \
    --batch-size 2 \
    --weights "${A1_CKPT}"

  CUDA_VISIBLE_DEVICES="${GPU_STANDARD}" "${PYTHON_BIN}" tools/check_train_steps.py \
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
    --require-nonzero-encoder-grad \
    --require-nonzero-adapter-grad \
    --require-min-optimizer-steps 2 \
    --checkpoint-every 0 \
    --log-file "${OUT_ROOT}/gpu2/standard_full.jsonl" \
    --checkpoint-dir "${OUT_ROOT}/gpu2/standard_full_ckpt" \
    --tag m1m2_standard_full

  # Exercise two optimizer steps inside warmup. Both the adapter and innovation
  # must remain exact structural no-ops while DynamicsEncoder still learns from
  # its canonical auxiliary targets.
  CUDA_VISIBLE_DEVICES="${GPU_STANDARD}" "${PYTHON_BIN}" tools/check_train_steps.py \
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
    --innovation-warmup-epoch 1 \
    --adapter-warmup-epoch 1 \
    --require-zero-warmup-output \
    --require-nonzero-encoder-grad \
    --require-min-optimizer-steps 2 \
    --checkpoint-every 0 \
    --log-file "${OUT_ROOT}/gpu2/standard_warmup.jsonl" \
    --checkpoint-dir "${OUT_ROOT}/gpu2/standard_warmup_ckpt" \
    --tag m1m2_standard_warmup

  # Do not require full history here: this is the invalid/padded fallback pass.
  CUDA_VISIBLE_DEVICES="${GPU_STANDARD}" "${PYTHON_BIN}" tools/check_train_steps.py \
    --cfg "${M2_CFG}" \
    --path "${DATA_ROOT}" \
    --version v1.0-mini \
    --split mini_train \
    --batch-size 2 \
    --workers 0 \
    --max-steps 256 \
    --no-shuffle \
    --weights "${A1_CKPT}" \
    --innovation-diagnostics \
    --innovation-warmup-epoch 0 \
    --track-resampled \
    --require-invalid \
    --require-empty \
    --require-resampled \
    --require-nonzero-encoder-grad \
    --require-nonzero-adapter-grad \
    --no-optimizer-step \
    --stop-when-requirements-met \
    --checkpoint-every 0 \
    --log-file "${OUT_ROOT}/gpu2/standard_fallback.jsonl" \
    --checkpoint-dir "${OUT_ROOT}/gpu2/standard_fallback_ckpt" \
    --tag m1m2_standard_fallback
) >"${OUT_ROOT}/gpu2/run.log" 2>&1 &
PID_GPU2=$!

# When one physical GPU is requested, finish the standard/fallback group before
# starting the strong-cadence group so the two jobs never contend for memory.
STATUS_GPU2=""
if [[ -n "${SINGLE_GPU}" ]]; then
  set +e
  wait "${PID_GPU2}"
  STATUS_GPU2=$?
  set -e
  if [[ "${STATUS_GPU2}" -ne 0 ]]; then
    echo "Single-GPU standard gate failed: ${STATUS_GPU2}" >&2
    echo "log: ${OUT_ROOT}/gpu2/run.log" >&2
    exit "${STATUS_GPU2}"
  fi
fi

(
  set -euo pipefail
  CUDA_VISIBLE_DEVICES="${GPU_PROTOCOL}" "${PYTHON_BIN}" tools/check_train_steps.py \
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
    --require-nonzero-encoder-grad \
    --require-nonzero-adapter-grad \
    --require-min-optimizer-steps 2 \
    --checkpoint-every 0 \
    --log-file "${OUT_ROOT}/gpu3/gap1124.jsonl" \
    --checkpoint-dir "${OUT_ROOT}/gpu3/gap1124_ckpt" \
    --tag m1m2_gap1124

  CUDA_VISIBLE_DEVICES="${GPU_PROTOCOL}" "${PYTHON_BIN}" tools/check_train_steps.py \
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
    --require-nonzero-encoder-grad \
    --require-nonzero-adapter-grad \
    --require-min-optimizer-steps 2 \
    --checkpoint-every 0 \
    --log-file "${OUT_ROOT}/gpu3/burst_drop.jsonl" \
    --checkpoint-dir "${OUT_ROOT}/gpu3/burst_drop_ckpt" \
    --tag m1m2_burst_drop
) >"${OUT_ROOT}/gpu3/run.log" 2>&1 &
PID_GPU3=$!

set +e
if [[ -z "${SINGLE_GPU}" ]]; then
  wait "${PID_GPU2}"
  STATUS_GPU2=$?
fi
wait "${PID_GPU3}"
STATUS_GPU3=$?
set -e

echo "GPU2 gate status: ${STATUS_GPU2} (log: ${OUT_ROOT}/gpu2/run.log)"
echo "GPU3 gate status: ${STATUS_GPU3} (log: ${OUT_ROOT}/gpu3/run.log)"
if [[ "${STATUS_GPU2}" -ne 0 || "${STATUS_GPU3}" -ne 0 ]]; then
  exit 1
fi

"${PYTHON_BIN}" - "${OUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
paths = (
    root / "gpu2" / "standard_full.summary.json",
    root / "gpu2" / "standard_warmup.summary.json",
    root / "gpu2" / "standard_fallback.summary.json",
    root / "gpu3" / "gap1124.summary.json",
    root / "gpu3" / "burst_drop.summary.json",
)
for path in paths:
    with path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if not summary.get("requirements_passed", False):
        raise SystemExit(
            f"summary hard gate failed: {path}: "
            f"{summary.get('requirement_failures')}")
    print(
        f"PASS {path.stem}: steps={summary['completed_steps']} "
        f"optimizer={summary['optimizer_step_count']} "
        f"invalid={summary['invalid_sample_count']} "
        f"empty={summary['empty_sample_count']} "
        f"resampled={summary['resampled_sample_count']} "
        f"bound_violation_max={summary['bound_violation_max']}")
PY

echo "M1/M2 E0-E5 server gates completed. Formal training remains HOLD pending review."
echo "artifacts: ${OUT_ROOT}"
