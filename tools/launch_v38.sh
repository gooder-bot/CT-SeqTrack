#!/usr/bin/env bash
# 用户显式执行才启动；预检登记后新建唯一目录，绝不复用旧输出。
set -euo pipefail
recipe="${1:?usage: bash tools/launch_v38.sh RECIPE GPU [--path DATA_ROOT]}"
gpu="${2:?single physical GPU ID required}"
shift 2
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo "single numeric GPU ID required" >&2; exit 2; }
case "$recipe" in
  b1_standard|b1_hold|full_standard) cfg="cfgs/ct_seqtrack/38_${recipe}_mini.yaml" ;;
  *) echo "unknown v38 recipe: $recipe" >&2; exit 2 ;;
esac
[[ -f main.py && -f "$cfg" ]] || { echo "run from CT-SeqTrack root" >&2; exit 2; }
python_bin="${CT_PYTHON:-/home/lishengjie/miniconda3/envs/seqtrack3d/bin/python}"
data_root="${CT_DATA_ROOT:-/home/lishengjie/data/nuscenes-mini}"
if [[ $# -gt 0 ]]; then
  [[ $# -eq 2 && "$1" == --path ]] || { echo "only --path DATA_ROOT is accepted; use main.py for test/resume" >&2; exit 2; }
  data_root="$2"
fi
"$python_bin" -c 'import sys; from models.ct_v38.entry import parse_config, validate_formal_registration; c=parse_config(sys.argv[1:]); validate_formal_registration(c)' \
  --cfg "$cfg" --path "$data_root" --devices "$gpu"
mkdir -p output
run_dir="$(mktemp -d "output/$(date +%Y%m%d-%H%M%S)-38_${recipe}-mini_car_seed42_60ep_bs16-XXXXXX")"
nohup env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  "$python_bin" -u main.py --cfg "$cfg" --path "$data_root" --devices "$gpu" \
  --accelerator gpu --trainer_devices 1 --log_dir "$run_dir" \
  >"$run_dir/train.log" 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$run_dir/train.pid"
printf 'RUN=%s\nPID=%s\n' "$run_dir" "$pid"
