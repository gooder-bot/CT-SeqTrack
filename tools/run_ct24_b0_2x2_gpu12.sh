#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${CTSEQ_NUSCENES_MINI_ROOT:-/home/lishengjie/data/nuscenes-mini}"
GPU_QUEUE="${CTSEQ_B0_QUEUE_GPU:-1}"
GPU_SINGLE="${CTSEQ_B0_SINGLE_GPU:-2}"
RUN_ROOT="${CTSEQ_B0_RUN_ROOT:-$PROJECT_ROOT/output/ct24_b0_2x2_60ep_seed42}"
LOG_ROOT="${CTSEQ_B0_LOG_ROOT:-$PROJECT_ROOT/logs/ct24_b0_2x2_60ep_seed42}"

run_first_step_preflight() {
  local run_dir="$RUN_ROOT/_first_step_preflight"
  local log_file="$LOG_ROOT/first_step_preflight.log"

  if [[ -e "$run_dir" ]]; then
    echo "Refusing to reuse existing preflight directory: $run_dir" >&2
    return 2
  fi
  echo "[$(date --iso-8601=seconds)] running one-batch B0 preflight on physical GPU $GPU_QUEUE"
  CUDA_VISIBLE_DEVICES="$GPU_QUEUE" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$PYTHON_BIN" main.py \
    --cfg cfgs/ct_seqtrack/24_b0_2x2_reseed0_rngshift0.yaml \
    --path "$DATA_ROOT" \
    --batch_size 16 \
    --epoch 1 \
    --workers 4 \
    --seed 42 \
    --preloading \
    --check_val_every_n_epoch 1 \
    --limit_train_batches 1 \
    --limit_val_batches 1 \
    --log_dir "$run_dir" \
    --tag first_step_preflight \
    2>&1 | tee "$log_file"
  echo "[$(date --iso-8601=seconds)] one-batch B0 preflight passed"
}

run_arm() {
  local gpu="$1"
  local config="$2"
  local arm="$3"
  local run_dir="$RUN_ROOT/$arm"
  local log_file="$LOG_ROOT/$arm.log"

  if [[ -e "$run_dir" ]]; then
    echo "Refusing to reuse existing run directory: $run_dir" >&2
    return 2
  fi

  echo "[$(date --iso-8601=seconds)] starting $arm on physical GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$PYTHON_BIN" main.py \
    --cfg "$config" \
    --path "$DATA_ROOT" \
    --batch_size 16 \
    --epoch 60 \
    --workers 4 \
    --seed 42 \
    --preloading \
    --check_val_every_n_epoch 5 \
    --log_dir "$run_dir" \
    --tag "$arm" \
    2>&1 | tee "$log_file"
  echo "[$(date --iso-8601=seconds)] completed $arm"
}

worker_gpu1() {
  local pids=()
  local arms=()
  local status=0

  export CUDA_VISIBLE_DEVICES="$GPU_QUEUE"
  run_arm "$GPU_QUEUE" \
    cfgs/ct_seqtrack/24_b0_2x2_reseed0_rngshift0.yaml \
    reseed0_rngshift0 \
    >"$LOG_ROOT/reseed0_rngshift0.worker.stdout.log" 2>&1 &
  pids+=("$!")
  arms+=("reseed0_rngshift0")
  echo "[$(date --iso-8601=seconds)] launched reseed0_rngshift0 as PID ${pids[0]}"
  run_arm "$GPU_QUEUE" \
    cfgs/ct_seqtrack/24_b0_2x2_reseed0_rngshift1.yaml \
    reseed0_rngshift1 \
    >"$LOG_ROOT/reseed0_rngshift1.worker.stdout.log" 2>&1 &
  pids+=("$!")
  arms+=("reseed0_rngshift1")
  echo "[$(date --iso-8601=seconds)] launched reseed0_rngshift1 as PID ${pids[1]}"
  run_arm "$GPU_QUEUE" \
    cfgs/ct_seqtrack/24_b0_2x2_reseed1_rngshift0.yaml \
    reseed1_rngshift0 \
    >"$LOG_ROOT/reseed1_rngshift0.worker.stdout.log" 2>&1 &
  pids+=("$!")
  arms+=("reseed1_rngshift0")
  echo "[$(date --iso-8601=seconds)] launched reseed1_rngshift0 as PID ${pids[2]}"
  echo "per-arm logs: $LOG_ROOT/reseed*_rngshift*.log"

  for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
      echo "[$(date --iso-8601=seconds)] completed ${arms[$index]}"
    else
      echo "[$(date --iso-8601=seconds)] failed ${arms[$index]}" >&2
      status=1
    fi
  done
  return "$status"
}

worker_gpu2() {
  export CUDA_VISIBLE_DEVICES="$GPU_SINGLE"
  run_arm "$GPU_SINGLE" \
    cfgs/ct_seqtrack/24_b0_2x2_reseed1_rngshift1.yaml \
    reseed1_rngshift1
}

case "${1:-launch}" in
  --worker-gpu1)
    mkdir -p "$LOG_ROOT"
    worker_gpu1
    ;;
  --worker-gpu2)
    mkdir -p "$LOG_ROOT"
    worker_gpu2
    ;;
  launch)
    if [[ ! -d "$DATA_ROOT/v1.0-mini" ]]; then
      echo "nuScenes mini metadata not found: $DATA_ROOT/v1.0-mini" >&2
      exit 2
    fi
    if [[ "$GPU_QUEUE" == "$GPU_SINGLE" ]]; then
      echo "queue and single workers must use different physical GPUs" >&2
      exit 2
    fi
    available_gpus="$(nvidia-smi --query-gpu=index --format=csv,noheader \
      | sed 's/[[:space:]]//g')"
    for requested_gpu in "$GPU_QUEUE" "$GPU_SINGLE"; do
      if ! grep -qx "$requested_gpu" <<<"$available_gpus"; then
        echo "physical GPU $requested_gpu is not available" >&2
        exit 2
      fi
    done
    "$PYTHON_BIN" -c \
      "import pytorch_lightning as pl; assert pl.__version__ == '2.0.2', pl.__version__; from nuscenes.nuscenes import NuScenes; print('environment preflight: OK')"
    mkdir -p "$LOG_ROOT"
    run_first_step_preflight
    nohup bash "$0" --worker-gpu1 \
      >"$LOG_ROOT/gpu${GPU_QUEUE}_queue.stdout.log" 2>&1 </dev/null &
    gpu1_pid=$!
    nohup bash "$0" --worker-gpu2 \
      >"$LOG_ROOT/gpu${GPU_SINGLE}_single.stdout.log" 2>&1 </dev/null &
    gpu2_pid=$!
    printf '%s\n' "$gpu1_pid" >"$LOG_ROOT/gpu${GPU_QUEUE}_queue.pid"
    printf '%s\n' "$gpu2_pid" >"$LOG_ROOT/gpu${GPU_SINGLE}_single.pid"
    echo "GPU $GPU_QUEUE queue PID: $gpu1_pid"
    echo "GPU $GPU_SINGLE single PID: $gpu2_pid"
    echo "Logs: $LOG_ROOT"
    ;;
  *)
    echo "usage: bash tools/run_ct24_b0_2x2_gpu12.sh [launch|--worker-gpu1|--worker-gpu2]" >&2
    exit 2
    ;;
esac
