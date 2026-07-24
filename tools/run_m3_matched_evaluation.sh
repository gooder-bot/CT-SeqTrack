#!/usr/bin/env bash
# Export and compare the frozen M2 initializer plus matched M3 A/B/C checkpoints.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/home/lishengjie/data/nuscenes-mini}"
INIT_CKPT="${INIT_CKPT:?Set INIT_CKPT to the frozen M2 initializer}"
A_CKPT="${A_CKPT:?Set A_CKPT to A_single_view last.ckpt}"
B_CKPT="${B_CKPT:?Set B_CKPT to B_paired_weight0 last.ckpt}"
C_CKPT="${C_CKPT:?Set C_CKPT to C_endpoint_distill last.ckpt}"
GPU="${GPU:-0}"
SEED="${SEED:-42}"
REQUIRE_CLEAN="${REQUIRE_CLEAN:-1}"

BASE_CFG="cfgs/seqtrack3d_nuscenes_m2_proposal_innovation_formal_true.yaml"
GAP_CFG="cfgs/seqtrack3d_nuscenes_a1_order_vr_gap1124.yaml"
GAP_MANIFEST="${GAP_MANIFEST:-protocols/manifests/m2_nuscenes_mini_test_gap1124_seed42.json}"

for checkpoint in "${INIT_CKPT}" "${A_CKPT}" "${B_CKPT}" "${C_CKPT}"; do
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing checkpoint: ${checkpoint}" >&2
    exit 1
  fi
done
if [[ "${REQUIRE_CLEAN}" == "1" && -n "$(git status --porcelain)" ]]; then
  echo "Matched M3 evaluation requires a clean worktree." >&2
  git status --short >&2
  exit 1
fi

GIT_COMMIT="$(git rev-parse HEAD)"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/output/m3_matched_eval_${GIT_COMMIT:0:7}_${RUN_STAMP}}"
if [[ -e "${OUT_ROOT}" ]]; then
  echo "OUT_ROOT already exists: ${OUT_ROOT}" >&2
  exit 1
fi
mkdir -p "${OUT_ROOT}/endpoints" "${OUT_ROOT}/analysis" "${OUT_ROOT}/logs"

export CUDA_VISIBLE_DEVICES="${GPU}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

declare -A CKPTS=(
  [INIT]="${INIT_CKPT}"
  [A]="${A_CKPT}"
  [B]="${B_CKPT}"
  [C]="${C_CKPT}"
)

for protocol in standard gap1124; do
  protocol_args=()
  if [[ "${protocol}" == "gap1124" ]]; then
    if [[ ! -f "${GAP_MANIFEST}" ]]; then
      echo "Missing gap1124 manifest: ${GAP_MANIFEST}" >&2
      exit 1
    fi
    protocol_args=(
      --protocol-cfg "${GAP_CFG}"
      --virtual-rate-manifest "${GAP_MANIFEST}"
      --allow-manifest-commit-mismatch
    )
  fi

  for label in INIT A B C; do
    tag="m3_${label}_${protocol}_seed${SEED}"
    "${PYTHON_BIN}" tools/export_m0_endpoints.py \
      --cfg "${BASE_CFG}" "${protocol_args[@]}" \
      --weights "${CKPTS[$label]}" \
      --path "${DATA_ROOT}" --version v1.0-mini --split mini_val \
      --device cuda:0 --seed "${SEED}" --preloading \
      --run-label "${label}" --protocol-name "${protocol}" \
      --dynamics-time-mode true \
      --output-dir "${OUT_ROOT}/endpoints" --tag "${tag}" \
      2>&1 | tee "${OUT_ROOT}/logs/${tag}.log"
  done

  "${PYTHON_BIN}" tools/summarize_m0_endpoints.py \
    --input INIT="${OUT_ROOT}/endpoints/m3_INIT_${protocol}_seed${SEED}/m0_endpoints.csv" \
    --input A="${OUT_ROOT}/endpoints/m3_A_${protocol}_seed${SEED}/m0_endpoints.csv" \
    --input B="${OUT_ROOT}/endpoints/m3_B_${protocol}_seed${SEED}/m0_endpoints.csv" \
    --input C="${OUT_ROOT}/endpoints/m3_C_${protocol}_seed${SEED}/m0_endpoints.csv" \
    --comparison A:INIT \
    --comparison B:A \
    --comparison C:B \
    --comparison C:A \
    --comparison C:INIT \
    --bootstrap-iterations 20000 --seed "${SEED}" \
    --output-dir "${OUT_ROOT}/analysis" \
    --tag "m3_${protocol}_matched_abc" \
    2>&1 | tee "${OUT_ROOT}/logs/m3_${protocol}_summary.log"
done

find "${OUT_ROOT}" -type f ! -name artifact_manifest.sha256 -print0 \
  | sort -z | xargs -0 sha256sum >"${OUT_ROOT}/artifact_manifest.sha256"

echo "M3 matched evaluation: COMPLETE"
echo "results: ${OUT_ROOT}"
echo "Primary method effect is C-B; deployment effect is C-A."
