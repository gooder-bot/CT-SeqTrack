#!/usr/bin/env bash
# Build and validate all frozen cadence/shuffled-time manifests for M2 formal.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/home/lishengjie/data/nuscenes-mini}"
EXPECTED_GIT_COMMIT="${EXPECTED_GIT_COMMIT:?Set EXPECTED_GIT_COMMIT to the reviewed E6 commit}"

FORMAL_CFG="cfgs/seqtrack3d_nuscenes_m2_proposal_innovation_formal_true.yaml"
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
  echo "Formal manifests require a clean worktree." >&2
  git status --short >&2
  exit 1
fi
if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "DATA_ROOT does not exist: ${DATA_ROOT}" >&2
  exit 1
fi

targets=(
  "${GAP_MANIFEST}"
  "${BURST_MANIFEST}"
  "${STANDARD_SHUFFLE}"
  "${GAP_SHUFFLE}"
  "${BURST_SHUFFLE}"
)
for target in "${targets[@]}"; do
  if [[ -e "${target}" ]]; then
    echo "Refusing to overwrite an existing frozen manifest: ${target}" >&2
    echo "Move the old file to an archive, then rerun from the reviewed commit." >&2
    exit 1
  fi
done
mkdir -p "${MANIFEST_DIR}"

"${PYTHON_BIN}" tools/check_m2_formal_freeze.py --self-test

"${PYTHON_BIN}" tools/build_virtual_rate_manifest.py \
  --cfg "${GAP_CFG}" --path "${DATA_ROOT}" --version v1.0-mini \
  --role test --split mini_val --mode gap_pattern \
  --gap-pattern 1 1 2 4 --seed 42 --max-gap 5 \
  --output "${GAP_MANIFEST}"

"${PYTHON_BIN}" tools/build_virtual_rate_manifest.py \
  --cfg "${BURST_CFG}" --path "${DATA_ROOT}" --version v1.0-mini \
  --role test --split mini_val --mode burst_drop \
  --seed 42 --max-gap 5 \
  --output "${BURST_MANIFEST}"

"${PYTHON_BIN}" tools/build_dynamics_time_manifest.py \
  --cfg "${FORMAL_CFG}" --path "${DATA_ROOT}" --version v1.0-mini \
  --role test --split mini_val --seed 42 \
  --output "${STANDARD_SHUFFLE}"

"${PYTHON_BIN}" tools/build_dynamics_time_manifest.py \
  --cfg "${FORMAL_CFG}" --protocol-cfg "${GAP_CFG}" \
  --virtual-rate-manifest "${GAP_MANIFEST}" \
  --path "${DATA_ROOT}" --version v1.0-mini \
  --role test --split mini_val --seed 42 \
  --output "${GAP_SHUFFLE}"

"${PYTHON_BIN}" tools/build_dynamics_time_manifest.py \
  --cfg "${FORMAL_CFG}" --protocol-cfg "${BURST_CFG}" \
  --virtual-rate-manifest "${BURST_MANIFEST}" \
  --path "${DATA_ROOT}" --version v1.0-mini \
  --role test --split mini_val --seed 42 \
  --output "${BURST_SHUFFLE}"

"${PYTHON_BIN}" tools/check_p0c_time_controls.py \
  --cfg "${FORMAL_CFG}" --path "${DATA_ROOT}" --version v1.0-mini \
  --role test --split mini_val --seed 42 \
  --shuffled-manifest "${STANDARD_SHUFFLE}"

"${PYTHON_BIN}" tools/check_p0c_time_controls.py \
  --cfg "${FORMAL_CFG}" --protocol-cfg "${GAP_CFG}" \
  --virtual-rate-manifest "${GAP_MANIFEST}" \
  --path "${DATA_ROOT}" --version v1.0-mini \
  --role test --split mini_val --seed 42 \
  --shuffled-manifest "${GAP_SHUFFLE}"

"${PYTHON_BIN}" tools/check_p0c_time_controls.py \
  --cfg "${FORMAL_CFG}" --protocol-cfg "${BURST_CFG}" \
  --virtual-rate-manifest "${BURST_MANIFEST}" \
  --path "${DATA_ROOT}" --version v1.0-mini \
  --role test --split mini_val --seed 42 \
  --shuffled-manifest "${BURST_SHUFFLE}"

OUT_ROOT="${PROJECT_ROOT}/output/m2_formal_manifests_${ACTUAL_GIT_COMMIT:0:7}"
mkdir -p "${OUT_ROOT}"
{
  echo "schema=ct_seqtrack.m2_formal_manifests.v1"
  echo "git_commit=${ACTUAL_GIT_COMMIT}"
  echo "data_root=${DATA_ROOT}"
  sha256sum "${targets[@]}"
} >"${OUT_ROOT}/manifest_sha256.txt"

tar -czf "${OUT_ROOT}/m2_formal_manifests.tar.gz" "${targets[@]}"
sha256sum "${OUT_ROOT}/m2_formal_manifests.tar.gz" \
  >"${OUT_ROOT}/archive_sha256.txt"

echo "M2 formal cadence/time manifests: PASS"
echo "hash index: ${OUT_ROOT}/manifest_sha256.txt"
echo "archive: ${OUT_ROOT}/m2_formal_manifests.tar.gz"
