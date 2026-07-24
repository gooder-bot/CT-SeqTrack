#!/usr/bin/env bash
# Evaluate M4 off/filter/tube/filter+tube on identical recursive endpoints.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/home/lishengjie/data/nuscenes-mini}"
MODEL_CKPT="${MODEL_CKPT:?Set MODEL_CKPT to the selected M2 or M3 last.ckpt}"
GPU="${GPU:-0}"
SEED="${SEED:-42}"
# Run the score-oriented real clock and its fixed-clock control by default.
M4_CLOCKS="${M4_CLOCKS:-real fixed}"
REQUIRE_CLEAN="${REQUIRE_CLEAN:-1}"

BASE_CFG="cfgs/seqtrack3d_nuscenes_m4_filter_tube_engineering.yaml"
GAP_CFG="cfgs/seqtrack3d_nuscenes_a1_order_vr_gap1124.yaml"
GAP_MANIFEST="${GAP_MANIFEST:-protocols/manifests/m2_nuscenes_mini_test_gap1124_seed42.json}"

if [[ ! -f "${MODEL_CKPT}" ]]; then
  echo "Missing MODEL_CKPT: ${MODEL_CKPT}" >&2
  exit 1
fi
if [[ "${REQUIRE_CLEAN}" == "1" && -n "$(git status --porcelain)" ]]; then
  echo "Matched M4 evaluation requires a clean worktree." >&2
  git status --short >&2
  exit 1
fi

"${PYTHON_BIN}" tools/export_m4_endpoints.py --self-test
"${PYTHON_BIN}" tools/check_m3_m4_invariants.py

GIT_COMMIT="$(git rev-parse HEAD)"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/output/m4_matched_eval_${GIT_COMMIT:0:7}_${RUN_STAMP}}"
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

for clock in ${M4_CLOCKS}; do
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

    for variant in off filter tube filter_tube; do
      tag="m4_${variant}_${clock}_${protocol}_seed${SEED}"
      "${PYTHON_BIN}" tools/export_m4_endpoints.py \
        --cfg "${BASE_CFG}" "${protocol_args[@]}" \
        --weights "${MODEL_CKPT}" \
        --path "${DATA_ROOT}" --version v1.0-mini --split mini_val \
        --device cuda:0 --seed "${SEED}" --preloading \
        --run-label "${variant}" --protocol-name "${protocol}" \
        --m4-variant "${variant}" --m4-time-mode "${clock}" \
        --output-dir "${OUT_ROOT}/endpoints" --tag "${tag}" \
        2>&1 | tee "${OUT_ROOT}/logs/${tag}.log"
    done

    "${PYTHON_BIN}" tools/summarize_m0_endpoints.py \
      --input off="${OUT_ROOT}/endpoints/m4_off_${clock}_${protocol}_seed${SEED}/m0_endpoints.csv" \
      --input filter="${OUT_ROOT}/endpoints/m4_filter_${clock}_${protocol}_seed${SEED}/m0_endpoints.csv" \
      --input tube="${OUT_ROOT}/endpoints/m4_tube_${clock}_${protocol}_seed${SEED}/m0_endpoints.csv" \
      --input filter_tube="${OUT_ROOT}/endpoints/m4_filter_tube_${clock}_${protocol}_seed${SEED}/m0_endpoints.csv" \
      --comparison filter:off \
      --comparison tube:off \
      --comparison filter_tube:off \
      --comparison filter_tube:filter \
      --comparison filter_tube:tube \
      --bootstrap-iterations 20000 --seed "${SEED}" \
      --output-dir "${OUT_ROOT}/analysis" \
      --tag "m4_${clock}_${protocol}_matched" \
      2>&1 | tee "${OUT_ROOT}/logs/m4_${clock}_${protocol}_summary.log"
  done
done

find "${OUT_ROOT}" -type f ! -name artifact_manifest.sha256 -print0 \
  | sort -z | xargs -0 sha256sum >"${OUT_ROOT}/artifact_manifest.sha256"

echo "M4 matched evaluation: COMPLETE"
echo "results: ${OUT_ROOT}"
