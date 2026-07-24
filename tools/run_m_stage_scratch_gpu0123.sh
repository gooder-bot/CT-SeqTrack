#!/usr/bin/env bash
# Train the four matched scratch arms concurrently on physical GPUs 0/1/2/3.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-/home/lishengjie/data/nuscenes-mini}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-16}"
WORKERS="${WORKERS:-12}"
EPOCHS="${EPOCHS:-60}"
M3_WEIGHT="${M3_WEIGHT:-0.05}"
PRELOADING="${PRELOADING:-1}"
REQUIRE_CLEAN="${REQUIRE_CLEAN:-1}"
GPU_W0="${GPU_W0:-0}"
GPU_A="${GPU_A:-1}"
GPU_B="${GPU_B:-2}"
GPU_C="${GPU_C:-3}"

W0_CFG="cfgs/seqtrack3d_nuscenes_w0_shared_se2_scratch.yaml"
M3_CFG="cfgs/seqtrack3d_nuscenes_m3_endpoint_distill_scratch.yaml"

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "Missing DATA_ROOT: ${DATA_ROOT}" >&2
  exit 1
fi
if [[ "${REQUIRE_CLEAN}" == "1" && -n "$(git status --porcelain)" ]]; then
  echo "Formal scratch training requires a clean worktree." >&2
  git status --short >&2
  exit 1
fi
preloading_args=()
if [[ "${PRELOADING}" == "1" ]]; then
  preloading_args=(--preloading)
elif [[ "${PRELOADING}" != "0" ]]; then
  echo "PRELOADING must be 0 or 1, got: ${PRELOADING}" >&2
  exit 1
fi

"${PYTHON_BIN}" tools/check_m3_m4_invariants.py
"${PYTHON_BIN}" tools/check_m_stage_configs.py

GIT_COMMIT="$(git rev-parse HEAD)"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/output/m_stage_scratch_${GIT_COMMIT:0:7}_${RUN_STAMP}}"
if [[ -e "${OUT_ROOT}" ]]; then
  echo "OUT_ROOT already exists: ${OUT_ROOT}" >&2
  exit 1
fi
mkdir -p "${OUT_ROOT}"

declare -a ARM_NAMES=()
declare -a ARM_PIDS=()

launch_arm() {
  local arm="$1"
  local gpu="$2"
  local cfg="$3"
  local variant="$4"
  local path_weight="$5"
  local arm_root="${OUT_ROOT}/${arm}"
  mkdir -p "${arm_root}"

  {
    echo "schema=ct_seqtrack.m_stage_scratch_arm.v1"
    echo "arm=${arm}"
    echo "gpu=${gpu}"
    echo "git_commit=${GIT_COMMIT}"
    echo "initialization=random"
    echo "seed=${SEED}"
    echo "batch_size=${BATCH_SIZE}"
    echo "workers=${WORKERS}"
    echo "epochs=${EPOCHS}"
    echo "preloading=${PRELOADING}"
    echo "config=${cfg}"
    echo "m3_variant=${variant}"
    echo "m3_path_weight=${path_weight}"
    echo "m3_irregular_supervision_weight=0"
    sha256sum "${cfg}"
  } >"${arm_root}/contract.txt"

  (
    set +e
    CUDA_VISIBLE_DEVICES="${gpu}" \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    "${PYTHON_BIN}" -u main.py \
      --cfg "${cfg}" \
      --path "${DATA_ROOT}" \
      --m3_variant "${variant}" \
      --m3_path_weight "${path_weight}" \
      --m3_irregular_supervision_weight 0 \
      --batch_size "${BATCH_SIZE}" \
      --workers "${WORKERS}" \
      --epoch "${EPOCHS}" \
      --seed "${SEED}" \
      "${preloading_args[@]}" \
      --check_val_every_n_epoch 5 \
      --save_top_k 0 \
      --log_dir "${arm_root}" \
      2>&1 | tee "${arm_root}/console.log"
    status=${PIPESTATUS[0]}
    echo "${status}" >"${arm_root}/training_exit_code.txt"
    exit "${status}"
  ) &
  ARM_NAMES+=("${arm}")
  ARM_PIDS+=("$!")
}

# W0 measures only the shared-SE(2)/canonical-data change from plain SeqTrack.
launch_arm "W0_shared_se2" "${GPU_W0}" "${W0_CFG}" "off" "0"
# A measures the complete M2 learned branch under ordinary single-view training.
launch_arm "A_m2_single_view" "${GPU_A}" "${M3_CFG}" "off" "0"
# B isolates paired history and its extra forward/teacher path.
launch_arm "B_m2_paired_weight0" "${GPU_B}" "${M3_CFG}" "distill" "0"
# C adds only the registered endpoint-distillation objective.
launch_arm "C_m2_m3_distill" "${GPU_C}" "${M3_CFG}" "distill" "${M3_WEIGHT}"

overall_status=0
set +e
for index in "${!ARM_PIDS[@]}"; do
  wait "${ARM_PIDS[$index]}"
  status=$?
  echo "${ARM_NAMES[$index]} exit status: ${status}"
  if [[ "${status}" -ne 0 ]]; then
    overall_status=1
  fi
done
set -e
if [[ "${overall_status}" -ne 0 ]]; then
  echo "At least one scratch arm failed. Inspect ${OUT_ROOT}/*/console.log" >&2
  exit "${overall_status}"
fi

for arm in "${ARM_NAMES[@]}"; do
  arm_root="${OUT_ROOT}/${arm}"
  mapfile -t checkpoints < <(
    find "${arm_root}" -type f -name last.ckpt | sort)
  if [[ "${#checkpoints[@]}" -ne 1 ]]; then
    echo "Expected one last.ckpt for ${arm}, found ${#checkpoints[@]}" >&2
    exit 1
  fi
  sha256sum "${checkpoints[0]}" >"${arm_root}/last.ckpt.sha256"
  printf '%s\n' "${checkpoints[0]}" >"${arm_root}/last_checkpoint_path.txt"
done

{
  echo "schema=ct_seqtrack.m_stage_scratch_matrix.v1"
  echo "git_commit=${GIT_COMMIT}"
  echo "seed=${SEED}"
  echo "external_baseline=existing_plain_seqtrack_scratch60"
  echo "m1_data_effect=W0-existing_seqtrack"
  echo "m2_effect=A-W0"
  echo "paired_effect=B-A"
  echo "m3_loss_effect=C-B"
  echo "m3_deployment_effect=C-A"
  find "${OUT_ROOT}" -name last_checkpoint_path.txt -print -exec cat {} \;
} >"${OUT_ROOT}/matrix_contract.txt"

find "${OUT_ROOT}" -type f ! -name artifact_manifest.sha256 -print0 \
  | sort -z | xargs -0 sha256sum >"${OUT_ROOT}/artifact_manifest.sha256"

echo "Four-GPU scratch M-stage training: COMPLETE"
echo "results: ${OUT_ROOT}"
echo "next: evaluate existing SeqTrack, W0, A, B and C on matched endpoints"
