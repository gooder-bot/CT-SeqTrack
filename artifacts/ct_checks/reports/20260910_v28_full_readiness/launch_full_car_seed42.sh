#!/usr/bin/env bash
# source this file from the CT-SeqTrack root in the seqtrack3d environment.
export CUDA_VISIBLE_DEVICES=3
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

DATA_ROOT=/home/lishengjie/code/SparseFusion-main/nuscenes/nuscenes
STAMP=$(date +%Y%m%d-%H%M%S)
CHECK_ROOT="artifacts/ct_checks/${STAMP}-v28-full-data-check"
FULL_DIR="output/${STAMP}-28_full-nuscenes_full_car_seed42_60ep_bs16"
mkdir -p "$CHECK_ROOT" "$FULL_DIR"

if python tools/preflight_ct_v28.py \
    --cfg cfgs/ct_seqtrack/28_full_nuscenes_full.yaml \
    --path "$DATA_ROOT" --output "$CHECK_ROOT/preflight.json" \
    > "$CHECK_ROOT/preflight.log" 2>&1 \
  && python tools/check_train_steps.py \
    --cfg cfgs/ct_seqtrack/28_full_nuscenes_full.yaml \
    --path "$DATA_ROOT" --steps 16 --workers 12 --seed 42 \
    --artifact-dir "$CHECK_ROOT/train16" \
    > "$CHECK_ROOT/train16.log" 2>&1; then
  nohup python -u main.py \
    --cfg cfgs/ct_seqtrack/28_full_nuscenes_full.yaml \
    --path "$DATA_ROOT" --category_name Car \
    --batch_size 16 --epoch 60 --workers 12 --seed 42 \
    --check_val_every_n_epoch 5 \
    --tag nuscenes_full_car_seed42_60ep_bs16 --log_dir "$FULL_DIR" \
    > "$FULL_DIR/train.log" 2>&1 < /dev/null &
  echo $! > "$FULL_DIR/train.pid"
  printf '已启动，目录：%s\n' "$FULL_DIR"
else
  printf '真实数据检查失败，未启动长跑。查看：%s\n' "$CHECK_ROOT"
fi
