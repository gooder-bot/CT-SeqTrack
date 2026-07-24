#!/usr/bin/env bash
# End-to-end M-stage pipeline: M3 matched training/evaluation, then M4 ablation.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

INIT_CKPT="${INIT_CKPT:?Set INIT_CKPT to the selected M2 last.ckpt}"
DATA_ROOT="${DATA_ROOT:-/home/lishengjie/data/nuscenes-mini}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"
SEED="${SEED:-42}"
REQUIRE_CLEAN="${REQUIRE_CLEAN:-1}"
RUN_M4="${RUN_M4:-1}"
M4_CLOCKS="${M4_CLOCKS:-real fixed}"
GAP_MANIFEST="${GAP_MANIFEST:-protocols/manifests/m2_nuscenes_mini_test_gap1124_seed42.json}"

if [[ ! -f "${INIT_CKPT}" ]]; then
  echo "Missing INIT_CKPT: ${INIT_CKPT}" >&2
  exit 1
fi
if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "Missing DATA_ROOT: ${DATA_ROOT}" >&2
  exit 1
fi
if [[ "${REQUIRE_CLEAN}" == "1" && -n "$(git status --porcelain)" ]]; then
  echo "Formal M-stage pipeline requires a clean worktree." >&2
  git status --short >&2
  exit 1
fi

GIT_COMMIT="$(git rev-parse HEAD)"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
PIPELINE_ROOT="${PIPELINE_ROOT:-${PROJECT_ROOT}/output/m_stage_${GIT_COMMIT:0:7}_${RUN_STAMP}}"
if [[ -e "${PIPELINE_ROOT}" ]]; then
  echo "PIPELINE_ROOT already exists: ${PIPELINE_ROOT}" >&2
  exit 1
fi
mkdir -p "${PIPELINE_ROOT}"

if [[ ! -f "${GAP_MANIFEST}" ]]; then
  echo "Building missing frozen gap1124 manifest: ${GAP_MANIFEST}"
  mkdir -p "$(dirname "${GAP_MANIFEST}")"
  manifest_dirty_args=()
  if [[ "${REQUIRE_CLEAN}" != "1" ]]; then
    manifest_dirty_args=(--allow-dirty)
  fi
  "${PYTHON_BIN}" tools/build_virtual_rate_manifest.py \
    --cfg cfgs/seqtrack3d_nuscenes_a1_order_vr_gap1124.yaml \
    --path "${DATA_ROOT}" --version v1.0-mini \
    --role test --split mini_val --mode gap_pattern \
    --gap-pattern 1 1 2 4 --seed 42 --max-gap 5 \
    --output "${GAP_MANIFEST}" "${manifest_dirty_args[@]}"
fi
mkdir -p "${PIPELINE_ROOT}/inputs"
FROZEN_GAP_MANIFEST="${PIPELINE_ROOT}/inputs/$(basename "${GAP_MANIFEST}")"
cp "${GAP_MANIFEST}" "${FROZEN_GAP_MANIFEST}"

OUT_ROOT="${PIPELINE_ROOT}/m3_train" \
INIT_CKPT="${INIT_CKPT}" \
DATA_ROOT="${DATA_ROOT}" \
PYTHON_BIN="${PYTHON_BIN}" \
GPU="${GPU}" \
SEED="${SEED}" \
REQUIRE_CLEAN="${REQUIRE_CLEAN}" \
BATCH_SIZE="${BATCH_SIZE:-16}" \
WORKERS="${WORKERS:-12}" \
EPOCHS="${EPOCHS:-60}" \
M3_WEIGHT="${M3_WEIGHT:-0.05}" \
bash tools/run_m3_matched_abc.sh

A_CKPT="$(<"${PIPELINE_ROOT}/m3_train/A_single_view/last_checkpoint_path.txt")"
B_CKPT="$(<"${PIPELINE_ROOT}/m3_train/B_paired_weight0/last_checkpoint_path.txt")"
C_CKPT="$(<"${PIPELINE_ROOT}/m3_train/C_endpoint_distill/last_checkpoint_path.txt")"

OUT_ROOT="${PIPELINE_ROOT}/m3_eval" \
INIT_CKPT="${INIT_CKPT}" \
A_CKPT="${A_CKPT}" \
B_CKPT="${B_CKPT}" \
C_CKPT="${C_CKPT}" \
DATA_ROOT="${DATA_ROOT}" \
GAP_MANIFEST="${FROZEN_GAP_MANIFEST}" \
PYTHON_BIN="${PYTHON_BIN}" \
GPU="${GPU}" \
SEED="${SEED}" \
REQUIRE_CLEAN="${REQUIRE_CLEAN}" \
bash tools/run_m3_matched_evaluation.sh

if [[ "${RUN_M4}" == "1" ]]; then
  SELECTED_M4_CKPT="${M4_CKPT:-${C_CKPT}}"
  OUT_ROOT="${PIPELINE_ROOT}/m4_eval" \
  MODEL_CKPT="${SELECTED_M4_CKPT}" \
  DATA_ROOT="${DATA_ROOT}" \
  GAP_MANIFEST="${FROZEN_GAP_MANIFEST}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  GPU="${GPU}" \
  SEED="${SEED}" \
  REQUIRE_CLEAN="${REQUIRE_CLEAN}" \
  M4_CLOCKS="${M4_CLOCKS}" \
  bash tools/run_m4_matched_evaluation.sh
elif [[ "${RUN_M4}" != "0" ]]; then
  echo "RUN_M4 must be 0 or 1, got: ${RUN_M4}" >&2
  exit 1
fi

{
  echo "schema=ct_seqtrack.m_stage_pipeline.v1"
  echo "git_commit=${GIT_COMMIT}"
  echo "init_checkpoint=${INIT_CKPT}"
  echo "a_checkpoint=${A_CKPT}"
  echo "b_checkpoint=${B_CKPT}"
  echo "c_checkpoint=${C_CKPT}"
  echo "run_m4=${RUN_M4}"
  echo "m4_clocks=${M4_CLOCKS}"
  sha256sum "${FROZEN_GAP_MANIFEST}"
  echo "primary_m3_effect=C-B"
  echo "deployment_m3_effect=C-A"
  echo "primary_m4_effect=variant-off"
} >"${PIPELINE_ROOT}/pipeline_contract.txt"

find "${PIPELINE_ROOT}" -type f ! -name artifact_manifest.sha256 -print0 \
  | sort -z | xargs -0 sha256sum >"${PIPELINE_ROOT}/artifact_manifest.sha256"

echo "M-stage pipeline: COMPLETE"
echo "results: ${PIPELINE_ROOT}"
echo "M3 reports: ${PIPELINE_ROOT}/m3_eval/analysis"
if [[ "${RUN_M4}" == "1" ]]; then
  echo "M4 reports: ${PIPELINE_ROOT}/m4_eval/analysis"
fi
