#!/usr/bin/env bash
# Evaluate the one M2 final checkpoint under true/fixed/shuffled time controls.
# Also export the frozen A1 baseline on identical standard/gap/burst endpoints.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/home/lishengjie/data/nuscenes-mini}"
FINAL_CKPT="${FINAL_CKPT:?Set FINAL_CKPT to the completed M2 last.ckpt}"
EXPECTED_FINAL_CKPT_SHA256="${EXPECTED_FINAL_CKPT_SHA256:?Set the SHA256 from final_checkpoint.json}"
A1_CKPT="${A1_CKPT:?Set A1_CKPT to the frozen A1-order last.ckpt}"
EXPECTED_GIT_COMMIT="${EXPECTED_GIT_COMMIT:?Set EXPECTED_GIT_COMMIT to the formal training commit}"
GPU="${GPU:-3}"

FORMAL_CFG="cfgs/seqtrack3d_nuscenes_m2_proposal_innovation_formal_true.yaml"
A1_CFG="cfgs/seqtrack3d_nuscenes_a1_order.yaml"
GAP_CFG="cfgs/seqtrack3d_nuscenes_a1_order_vr_gap1124.yaml"
BURST_CFG="cfgs/seqtrack3d_nuscenes_a1_order_vr_burst_drop.yaml"
MANIFEST_DIR="protocols/manifests"
GAP_MANIFEST="${MANIFEST_DIR}/m2_nuscenes_mini_test_gap1124_seed42.json"
BURST_MANIFEST="${MANIFEST_DIR}/m2_nuscenes_mini_test_burst_drop_seed42.json"
STANDARD_SHUFFLE="${MANIFEST_DIR}/m2_nuscenes_mini_test_standard_shuffled_dt_seed42.json"
GAP_SHUFFLE="${MANIFEST_DIR}/m2_nuscenes_mini_test_gap1124_shuffled_dt_seed42.json"
BURST_SHUFFLE="${MANIFEST_DIR}/m2_nuscenes_mini_test_burst_drop_shuffled_dt_seed42.json"

ACTUAL_GIT_COMMIT="$(git rev-parse HEAD)"
if [[ "${ACTUAL_GIT_COMMIT}" != "${EXPECTED_GIT_COMMIT}" ]]; then
  echo "Git commit mismatch: expected ${EXPECTED_GIT_COMMIT}, got ${ACTUAL_GIT_COMMIT}" >&2
  exit 1
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Formal evaluation requires a clean worktree." >&2
  git status --short >&2
  exit 1
fi

ACTUAL_FINAL_SHA="$(sha256sum "${FINAL_CKPT}" | awk '{print $1}')"
if [[ "${ACTUAL_FINAL_SHA}" != "${EXPECTED_FINAL_CKPT_SHA256}" ]]; then
  echo "M2 final checkpoint SHA256 mismatch" >&2
  echo "expected: ${EXPECTED_FINAL_CKPT_SHA256}" >&2
  echo "actual:   ${ACTUAL_FINAL_SHA}" >&2
  exit 1
fi
ACTUAL_A1_SHA="$(sha256sum "${A1_CKPT}" | awk '{print $1}')"
EXPECTED_A1_SHA="a2fbffb1e5acae37adab3cb858e864857cc1d6c2231f9e0848df719614f24a82"
if [[ "${ACTUAL_A1_SHA}" != "${EXPECTED_A1_SHA}" ]]; then
  echo "A1 checkpoint SHA256 mismatch: ${ACTUAL_A1_SHA}" >&2
  exit 1
fi

required_manifests=(
  "${GAP_MANIFEST}" "${BURST_MANIFEST}"
  "${STANDARD_SHUFFLE}" "${GAP_SHUFFLE}" "${BURST_SHUFFLE}"
)
for manifest in "${required_manifests[@]}"; do
  if [[ ! -f "${manifest}" ]]; then
    echo "Missing formal manifest: ${manifest}" >&2
    exit 1
  fi
done

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/output/m2_formal_controls_${ACTUAL_GIT_COMMIT:0:7}_${RUN_STAMP}}"
if [[ -e "${OUT_ROOT}" ]]; then
  echo "OUT_ROOT already exists; choose a fresh formal evaluation path: ${OUT_ROOT}" >&2
  exit 1
fi
mkdir -p "${OUT_ROOT}/endpoints" "${OUT_ROOT}/analysis" "${OUT_ROOT}/logs"

"${PYTHON_BIN}" tools/check_m2_formal_freeze.py --self-test
"${PYTHON_BIN}" tools/export_m0_endpoints.py --self-test

export CUDA_VISIBLE_DEVICES="${GPU}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

for protocol in standard gap1124 burst_drop; do
  protocol_args=()
  case "${protocol}" in
    standard)
      shuffled_manifest="${STANDARD_SHUFFLE}"
      protocol_args=(--require-manifest-commit-match)
      ;;
    gap1124)
      shuffled_manifest="${GAP_SHUFFLE}"
      protocol_args=(
        --protocol-cfg "${GAP_CFG}"
        --virtual-rate-manifest "${GAP_MANIFEST}"
        --require-manifest-commit-match
      )
      ;;
    burst_drop)
      shuffled_manifest="${BURST_SHUFFLE}"
      protocol_args=(
        --protocol-cfg "${BURST_CFG}"
        --virtual-rate-manifest "${BURST_MANIFEST}"
        --require-manifest-commit-match
      )
      ;;
    *)
      echo "Unknown protocol: ${protocol}" >&2
      exit 1
      ;;
  esac

  true_tag="m2_${protocol}_true_seed42"
  fixed_tag="m2_${protocol}_fixed_seed42"
  shuffled_tag="m2_${protocol}_shuffled_seed42"
  a1_tag="a1_${protocol}_true_seed42"

  "${PYTHON_BIN}" tools/export_m0_endpoints.py \
    --cfg "${FORMAL_CFG}" "${protocol_args[@]}" \
    --weights "${FINAL_CKPT}" \
    --path "${DATA_ROOT}" --version v1.0-mini --split mini_val \
    --device cuda:0 --seed 42 --preloading \
    --run-label M2 --protocol-name "${protocol}_true" \
    --dynamics-time-mode true --require-clean-git \
    --output-dir "${OUT_ROOT}/endpoints" --tag "${true_tag}" \
    2>&1 | tee "${OUT_ROOT}/logs/${true_tag}.log"

  reference_csv="${OUT_ROOT}/endpoints/${true_tag}/m0_endpoints.csv"

  "${PYTHON_BIN}" tools/export_m0_endpoints.py \
    --cfg "${FORMAL_CFG}" "${protocol_args[@]}" \
    --weights "${FINAL_CKPT}" \
    --path "${DATA_ROOT}" --version v1.0-mini --split mini_val \
    --device cuda:0 --seed 42 --preloading \
    --run-label M2 --protocol-name "${protocol}_fixed" \
    --dynamics-time-mode fixed --dynamics-fixed-delta-t 0.5 \
    --reference-endpoints-csv "${reference_csv}" --require-clean-git \
    --output-dir "${OUT_ROOT}/endpoints" --tag "${fixed_tag}" \
    2>&1 | tee "${OUT_ROOT}/logs/${fixed_tag}.log"

  "${PYTHON_BIN}" tools/export_m0_endpoints.py \
    --cfg "${FORMAL_CFG}" "${protocol_args[@]}" \
    --weights "${FINAL_CKPT}" \
    --path "${DATA_ROOT}" --version v1.0-mini --split mini_val \
    --device cuda:0 --seed 42 --preloading \
    --run-label M2 --protocol-name "${protocol}_shuffled" \
    --dynamics-time-mode shuffled \
    --dynamics-time-manifest "${shuffled_manifest}" \
    --reference-endpoints-csv "${reference_csv}" --require-clean-git \
    --output-dir "${OUT_ROOT}/endpoints" --tag "${shuffled_tag}" \
    2>&1 | tee "${OUT_ROOT}/logs/${shuffled_tag}.log"

  "${PYTHON_BIN}" tools/export_m0_endpoints.py \
    --cfg "${A1_CFG}" "${protocol_args[@]}" \
    --weights "${A1_CKPT}" \
    --path "${DATA_ROOT}" --version v1.0-mini --split mini_val \
    --device cuda:0 --seed 42 --preloading \
    --run-label A1 --protocol-name "${protocol}_true" \
    --dynamics-time-mode true \
    --reference-endpoints-csv "${reference_csv}" --require-clean-git \
    --output-dir "${OUT_ROOT}/endpoints" --tag "${a1_tag}" \
    2>&1 | tee "${OUT_ROOT}/logs/${a1_tag}.log"

  "${PYTHON_BIN}" tools/summarize_m0_endpoints.py \
    --input true="${OUT_ROOT}/endpoints/${true_tag}/m0_endpoints.csv" \
    --input fixed="${OUT_ROOT}/endpoints/${fixed_tag}/m0_endpoints.csv" \
    --input shuffled="${OUT_ROOT}/endpoints/${shuffled_tag}/m0_endpoints.csv" \
    --comparison true:fixed --comparison true:shuffled \
    --require-same-checkpoint \
    --bootstrap-iterations 10000 --seed 42 \
    --output-dir "${OUT_ROOT}/analysis" \
    --tag "m2_${protocol}_time_controls" \
    2>&1 | tee "${OUT_ROOT}/logs/m2_${protocol}_time_controls_summary.log"

  "${PYTHON_BIN}" tools/summarize_m0_endpoints.py \
    --input A1="${OUT_ROOT}/endpoints/${a1_tag}/m0_endpoints.csv" \
    --input M2="${OUT_ROOT}/endpoints/${true_tag}/m0_endpoints.csv" \
    --comparison M2:A1 \
    --bootstrap-iterations 10000 --seed 42 \
    --output-dir "${OUT_ROOT}/analysis" \
    --tag "m2_${protocol}_vs_a1" \
    2>&1 | tee "${OUT_ROOT}/logs/m2_${protocol}_vs_a1_summary.log"
done

{
  echo "schema=ct_seqtrack.m2_formal_time_controls.v1"
  echo "git_commit=${ACTUAL_GIT_COMMIT}"
  echo "gpu=${GPU}"
  echo "m2_checkpoint=${FINAL_CKPT}"
  echo "m2_checkpoint_sha256=${ACTUAL_FINAL_SHA}"
  echo "a1_checkpoint=${A1_CKPT}"
  echo "a1_checkpoint_sha256=${ACTUAL_A1_SHA}"
  echo "protocols=standard,gap1124,burst_drop"
  echo "time_modes=true,fixed,shuffled"
  sha256sum "${FORMAL_CFG}" "${A1_CFG}" "${required_manifests[@]}"
} >"${OUT_ROOT}/evaluation_contract.txt"

find "${OUT_ROOT}" -type f ! -name artifact_manifest.sha256 -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  >"${OUT_ROOT}/artifact_manifest.sha256"

ARCHIVE_ROOT="${PROJECT_ROOT}/output/formal_archives"
mkdir -p "${ARCHIVE_ROOT}"
ARCHIVE_PATH="${ARCHIVE_ROOT}/$(basename "${OUT_ROOT}").tar.gz"
tar -czf "${ARCHIVE_PATH}" -C "$(dirname "${OUT_ROOT}")" "$(basename "${OUT_ROOT}")"
sha256sum "${ARCHIVE_PATH}" >"${ARCHIVE_PATH}.sha256"

echo "M2 formal A1/time-control matrix: COMPLETE"
echo "results: ${OUT_ROOT}"
echo "archive: ${ARCHIVE_PATH}"
