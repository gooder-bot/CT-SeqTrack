#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_ROOT="${1:-/home/lishengjie/data/cxtrack}"
# The mixed interval dataset has about 5x the samples of interval 1.
# Default to the optimizer-step-matched equivalent of 180 / 5 epochs.
EPOCHS="${EPOCHS:-36}"
BATCH_SIZE="${BATCH_SIZE:-16}"
WORKERS="${WORKERS:-8}"
SEED="${SEED:-42}"
# Twelve matched validations over 36 epochs, including the final epoch.
CHECK_VAL_EVERY="${CHECK_VAL_EVERY:-3}"
RUN_STAMP="$(date +%Y%m%d-%H%M%S)"
LAUNCH_DIR="$ROOT_DIR/output/kitti_htv_launches/$RUN_STAMP"

M2_CFG="$ROOT_DIR/cfgs/seqtrack3d_kitti_htv_m2.yaml"
W0_CFG="$ROOT_DIR/cfgs/seqtrack3d_kitti_htv_w0.yaml"

for required_path in \
    "$M2_CFG" \
    "$W0_CFG" \
    "$DATA_ROOT/training/calib" \
    "$DATA_ROOT/training/label_02" \
    "$DATA_ROOT/training/velodyne"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Missing required path: $required_path" >&2
    exit 1
  fi
done

if command -v nvidia-smi >/dev/null 2>&1 \
    && [[ "${ALLOW_BUSY_GPUS:-0}" != "1" ]]; then
  for gpu_id in 0 1; do
    busy_pids="$(
      nvidia-smi -i "$gpu_id" \
        --query-compute-apps=pid \
        --format=csv,noheader,nounits 2>/dev/null \
        | awk '$1 ~ /^[0-9]+$/ {print $1}'
    )"
    if [[ -n "$busy_pids" ]]; then
      echo "GPU $gpu_id already has compute PID(s): $busy_pids" >&2
      echo "Stop the existing job or set ALLOW_BUSY_GPUS=1 explicitly." >&2
      exit 1
    fi
  done
fi

mkdir -p "$LAUNCH_DIR"

COMMON_ARGS=(
  --path "$DATA_ROOT"
  --train_kitti_hv_interval all
  --val_kitti_hv_interval 5
  --m4_variant off
  --batch_size "$BATCH_SIZE"
  --workers "$WORKERS"
  --epoch "$EPOCHS"
  --save_top_k 5
  --check_val_every_n_epoch "$CHECK_VAL_EVERY"
  --seed "$SEED"
)

nohup env CUDA_VISIBLE_DEVICES=0 \
  python -u main.py \
  --cfg "$M2_CFG" \
  "${COMMON_ARGS[@]}" \
  --tag "kitti_htv_m2_all_i5_seed${SEED}" \
  >"$LAUNCH_DIR/m2_gpu0.log" 2>&1 &
M2_PID=$!
echo "$M2_PID" >"$LAUNCH_DIR/m2_gpu0.pid"

nohup env CUDA_VISIBLE_DEVICES=1 \
  python -u main.py \
  --cfg "$W0_CFG" \
  "${COMMON_ARGS[@]}" \
  --tag "kitti_htv_w0_all_i5_seed${SEED}" \
  >"$LAUNCH_DIR/w0_gpu1.log" 2>&1 &
W0_PID=$!
echo "$W0_PID" >"$LAUNCH_DIR/w0_gpu1.pid"

sleep 2
if ! kill -0 "$M2_PID" 2>/dev/null; then
  echo "M2 exited during launch; inspect $LAUNCH_DIR/m2_gpu0.log" >&2
  exit 1
fi
if ! kill -0 "$W0_PID" 2>/dev/null; then
  echo "W0 exited during launch; inspect $LAUNCH_DIR/w0_gpu1.log" >&2
  exit 1
fi

cat <<EOF
KITTI-HTV matched jobs started.
  M2 GPU0 PID: $M2_PID
  W0 GPU1 PID: $W0_PID
  Launch files: $LAUNCH_DIR

Monitor:
  nvidia-smi
  tail -f "$LAUNCH_DIR/m2_gpu0.log"
  tail -f "$LAUNCH_DIR/w0_gpu1.log"

Check processes:
  ps -fp $M2_PID,$W0_PID
EOF
