#!/usr/bin/env bash
# 每次只启动一个新 scratch 运行；六组分别调用，不恢复或覆盖旧运行。
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 || ${1:-} == --help ]]; then
  printf '%s\n' 'Usage: bash tools/launch_v35_full_seed42.sh {scaled_b1|scaled_b1_b2|scaled_full|normal_b1|normal_b1_b2|normal_full} [physical_gpu]'
  [[ ${1:-} == --help ]] && exit 0
  exit 2
fi

case "$1" in
  scaled_b1)    CT_ARM=b1;    CT_RECIPE=scaled; CT_DEFAULT_GPU=0 ;;
  scaled_b1_b2) CT_ARM=b1_b2; CT_RECIPE=scaled; CT_DEFAULT_GPU=0 ;;
  scaled_full)  CT_ARM=full;  CT_RECIPE=scaled; CT_DEFAULT_GPU=0 ;;
  normal_b1)    CT_ARM=b1;    CT_RECIPE=normal; CT_DEFAULT_GPU=1 ;;
  normal_b1_b2) CT_ARM=b1_b2; CT_RECIPE=normal; CT_DEFAULT_GPU=1 ;;
  normal_full)  CT_ARM=full;  CT_RECIPE=normal; CT_DEFAULT_GPU=1 ;;
  *) printf 'Unknown experiment: %s\n' "$1" >&2; exit 2 ;;
esac
CT_GPU="${2:-$CT_DEFAULT_GPU}"
[[ "$CT_GPU" =~ ^[0-9]+$ ]] || { printf '%s\n' 'physical_gpu must be one nonnegative integer' >&2; exit 2; }
CT_PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CT_PROJECT"
CT_PYTHON=/home/lishengjie/miniconda3/envs/seqtrack3d/bin/python
CT_CFG="cfgs/ct_seqtrack/35_${CT_ARM}_w_${CT_RECIPE}_lr_mini.yaml"
[[ -x "$CT_PYTHON" && -f "$CT_CFG" ]] || { printf '%s\n' 'Python environment or configuration is missing' >&2; exit 2; }
mkdir -p output
CT_RUN="$(mktemp -d "output/$(date +%Y%m%d-%H%M%S)-35_${CT_ARM}_w_${CT_RECIPE}_lr-mini_car_seed42_60ep_bs16-XXXXXX")"
nohup env CUDA_VISIBLE_DEVICES="$CT_GPU" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  "$CT_PYTHON" -u main.py --cfg "$CT_CFG" \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --check_val_every_n_epoch 5 \
  --accelerator gpu --trainer_devices 1 --log_dir "$CT_RUN" \
  > "$CT_RUN/train.log" 2>&1 < /dev/null &
CT_PID=$!
printf '%s\n' "$CT_PID" > "$CT_RUN/train.pid"
printf 'experiment=%s gpu=%s pid=%s\nRUN=%s\ntail -n 80 -f "%s/train.log"\n' "$1" "$CT_GPU" "$CT_PID" "$CT_RUN" "$CT_RUN"
