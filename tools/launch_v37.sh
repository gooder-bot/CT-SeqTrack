#!/usr/bin/env bash
# 用户在服务器仓库根目录执行；每次新建独立 output，不覆盖旧运行。
set -euo pipefail
arm="${1:?usage: bash tools/launch_v37.sh {b0|b1|b1_b2|full|full_bf16} GPU}"
gpu="${2:?a single physical GPU ID is required}"
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo "single numeric GPU ID required" >&2; exit 2; }
case "$arm" in
  b0|b1|b1_b2|full|full_bf16) cfg="cfgs/ct_seqtrack/37_${arm}_mini.yaml" ;;
  *) echo "unregistered v37 arm: $arm" >&2; exit 2 ;;
esac
[[ -f main.py && -f "$cfg" ]] || { echo "run from CT-SeqTrack root" >&2; exit 2; }
python_bin="${CT_PYTHON:-/home/lishengjie/miniconda3/envs/seqtrack3d/bin/python}"
data_root="${CT_DATA_ROOT:-/home/lishengjie/data/nuscenes-mini}"
mkdir -p output
run_dir="$(mktemp -d "output/$(date +%Y%m%d-%H%M%S)-37_${arm}-mini_car_seed42_60ep_bs16-XXXXXX")"
nohup env CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  "$python_bin" -u main.py --cfg "$cfg" --path "$data_root" \
  --batch_size 16 --epoch 60 --workers 4 --check_val_every_n_epoch 5 \
  --accelerator gpu --trainer_devices 1 --log_dir "$run_dir" \
  >"$run_dir/train.log" 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$run_dir/train.pid"
printf 'RUN=%s\nPID=%s\n' "$run_dir" "$pid"
