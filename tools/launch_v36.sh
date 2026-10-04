#!/usr/bin/env bash
# 每次只启动一组；用户执行，禁止重复当作恢复。
set -euo pipefail
arm="${1:?usage: bash tools/launch_v36.sh A|C|D|E|F GPU}"
gpu="${2:?GPU index required}"
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'GPU must be an integer' >&2; exit 2; }
case "$arm" in
  A) cfg=cfgs/ct_seqtrack/36_a_full_mini.yaml ;;
  C) cfg=cfgs/ct_seqtrack/36_c_no_b2_mini.yaml ;;
  D) cfg=cfgs/ct_seqtrack/36_d_event_cv_mini.yaml ;;
  E) cfg=cfgs/ct_seqtrack/36_e_legacy_b1_mini.yaml ;;
  F) cfg=cfgs/ct_seqtrack/36_f_window_reset_mini.yaml ;;
  *) echo 'Allowed arms: A C D E F' >&2; exit 2 ;;
esac
[[ -f main.py && -f "$cfg" ]] || { echo 'Run from CT-SeqTrack root' >&2; exit 2; }
python_bin="${CT_PYTHON:-/home/lishengjie/miniconda3/envs/seqtrack3d/bin/python}"
data_root="${CT_DATA_ROOT:-/home/lishengjie/data/nuscenes-mini}"
run_label="${cfg##*/}"
run_label="${run_label%_mini.yaml}"
mkdir -p output
run_dir="$(mktemp -d "output/$(date +%Y%m%d-%H%M%S)-${run_label}-mini_car_seed42_60ep_bs16-XXXXXX")"
nohup env CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  "$python_bin" -u main.py --cfg "$cfg" --path "$data_root" \
  --batch_size 16 --epoch 60 --workers 4 --check_val_every_n_epoch 5 \
  --accelerator gpu --trainer_devices 1 \
  --log_dir "$run_dir" >"$run_dir/train.log" 2>&1 < /dev/null &
printf '%s\n' "$!" >"$run_dir/train.pid"
printf 'RUN=%s\nPID=%s\n' "$run_dir" "$!"
