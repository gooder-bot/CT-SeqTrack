#!/usr/bin/env bash
# Train the matched M3 A/B/C matrix from one frozen M2 checkpoint.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/home/lishengjie/data/nuscenes-mini}"
INIT_CKPT="${INIT_CKPT:?Set INIT_CKPT to the selected M2 last.ckpt}"
GPU="${GPU:-0}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-16}"
WORKERS="${WORKERS:-12}"
EPOCHS="${EPOCHS:-60}"
M3_WEIGHT="${M3_WEIGHT:-0.05}"
REQUIRE_CLEAN="${REQUIRE_CLEAN:-1}"
CFG="cfgs/seqtrack3d_nuscenes_m3_endpoint_distill_engineering.yaml"

if [[ ! -f "${INIT_CKPT}" ]]; then
  echo "Missing INIT_CKPT: ${INIT_CKPT}" >&2
  exit 1
fi
if [[ "${REQUIRE_CLEAN}" == "1" && -n "$(git status --porcelain)" ]]; then
  echo "Matched M3 training requires a clean worktree." >&2
  git status --short >&2
  exit 1
fi

"${PYTHON_BIN}" tools/check_m3_m4_invariants.py
"${PYTHON_BIN}" tools/check_m_stage_configs.py

GIT_COMMIT="$(git rev-parse HEAD)"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/output/m3_matched_abc_${GIT_COMMIT:0:7}_${RUN_STAMP}}"
if [[ -e "${OUT_ROOT}" ]]; then
  echo "OUT_ROOT already exists: ${OUT_ROOT}" >&2
  exit 1
fi
mkdir -p "${OUT_ROOT}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

run_arm() {
  local arm="$1"
  local variant="$2"
  local path_weight="$3"
  local arm_root="${OUT_ROOT}/${arm}"
  mkdir -p "${arm_root}"
  {
    echo "schema=ct_seqtrack.m3_matched_arm.v1"
    echo "arm=${arm}"
    echo "git_commit=${GIT_COMMIT}"
    echo "init_checkpoint=${INIT_CKPT}"
    echo "seed=${SEED}"
    echo "batch_size=${BATCH_SIZE}"
    echo "workers=${WORKERS}"
    echo "epochs=${EPOCHS}"
    echo "m3_variant=${variant}"
    echo "m3_path_weight=${path_weight}"
    echo "m3_irregular_supervision_weight=0"
    sha256sum "${INIT_CKPT}" "${CFG}"
  } >"${arm_root}/contract.txt"

  set +e
  "${PYTHON_BIN}" -u main.py \
    --cfg "${CFG}" \
    --init_checkpoint "${INIT_CKPT}" \
    --m3_variant "${variant}" \
    --m3_path_weight "${path_weight}" \
    --m3_irregular_supervision_weight 0 \
    --batch_size "${BATCH_SIZE}" \
    --workers "${WORKERS}" \
    --epoch "${EPOCHS}" \
    --seed "${SEED}" \
    --preloading \
    --check_val_every_n_epoch 5 \
    --save_top_k 0 \
    --log_dir "${arm_root}" \
    2>&1 | tee "${arm_root}/console.log"
  local status=${PIPESTATUS[0]}
  set -e
  echo "${status}" >"${arm_root}/training_exit_code.txt"
  if [[ "${status}" -ne 0 ]]; then
    echo "M3 arm ${arm} failed with status ${status}" >&2
    exit "${status}"
  fi

  mapfile -t checkpoints < <(find "${arm_root}" -type f -name last.ckpt | sort)
  if [[ "${#checkpoints[@]}" -ne 1 ]]; then
    echo "Expected one last.ckpt for ${arm}, found ${#checkpoints[@]}" >&2
    printf '%s\n' "${checkpoints[@]}" >&2
    exit 1
  fi
  local checkpoint="${checkpoints[0]}"
  sha256sum "${checkpoint}" >"${arm_root}/last.ckpt.sha256"
  printf '%s\n' "${checkpoint}" >"${arm_root}/last_checkpoint_path.txt"
}

# A measures ordinary single-view continuation.
# B isolates the compute/data-path effect of paired history with zero path loss.
# C adds only the registered asymmetric distillation term.
run_arm "A_single_view" "off" "0"
run_arm "B_paired_weight0" "distill" "0"
run_arm "C_endpoint_distill" "distill" "${M3_WEIGHT}"

{
  echo "schema=ct_seqtrack.m3_matched_abc.v1"
  echo "git_commit=${GIT_COMMIT}"
  echo "init_checkpoint=${INIT_CKPT}"
  echo "primary_effect=C-B"
  echo "deployment_effect=C-A"
  find "${OUT_ROOT}" -name last_checkpoint_path.txt -print -exec cat {} \;
} >"${OUT_ROOT}/matrix_contract.txt"

find "${OUT_ROOT}" -type f ! -name artifact_manifest.sha256 -print0 \
  | sort -z | xargs -0 sha256sum >"${OUT_ROOT}/artifact_manifest.sha256"

echo "M3 matched A/B/C training: COMPLETE"
echo "results: ${OUT_ROOT}"
echo "next: INIT_CKPT=${INIT_CKPT} A_CKPT=<A last.ckpt> B_CKPT=<B last.ckpt> C_CKPT=<C last.ckpt> bash tools/run_m3_matched_evaluation.sh"
