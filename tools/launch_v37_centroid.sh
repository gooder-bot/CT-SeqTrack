#!/usr/bin/env bash
# 用户在新源码部署目录执行；三组均 scratch，绝不修改已有运行目录。
set -euo pipefail
recipe="${1:?usage: bash tools/launch_v37_centroid.sh {b1_b2_mass_detach|full_mass_detach|full_mass_detach_hold_lr} GPU}"
gpu="${2:?a single physical GPU ID is required}"
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo "single numeric GPU ID required" >&2; exit 2; }
case "$recipe" in
  b1_b2_mass_detach|full_mass_detach|full_mass_detach_hold_lr)
    cfg="cfgs/ct_seqtrack/37_${recipe}_mini.yaml" ;;
  *) echo "unregistered B2 centroid recipe: $recipe" >&2; exit 2 ;;
esac
[[ -f main.py && -f "$cfg" ]] || { echo "run from CT-SeqTrack root" >&2; exit 2; }
python_bin="${CT_PYTHON:-/home/lishengjie/miniconda3/envs/seqtrack3d/bin/python}"
data_root="${CT_DATA_ROOT:-/home/lishengjie/data/nuscenes-mini}"
# 先检查源码和完整配方登记，失败时不创建一个看似已启动的 output。
"$python_bin" -c 'import sys; from models.ct_v37.config import load_config; from models.ct_v37.entry import validate_formal_registration; print(validate_formal_registration(load_config(sys.argv[1])))' "$cfg"
mkdir -p output
run_dir="$(mktemp -d "output/$(date +%Y%m%d-%H%M%S)-37_${recipe}-mini_car_seed42_60ep_bs16-XXXXXX")"
nohup env CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  "$python_bin" -u main.py --cfg "$cfg" --path "$data_root" \
  --accelerator gpu --trainer_devices 1 --log_dir "$run_dir" \
  >"$run_dir/train.log" 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$run_dir/train.pid"
printf 'RUN=%s\nPID=%s\n' "$run_dir" "$pid"
