#!/usr/bin/env bash
# 只恢复同一运行的完整epoch；main保留原manifest并验证run_id/source/config。
set -euo pipefail
recipe="${1:?usage: bash tools/resume_v38.sh RECIPE GPU RUN CHECKPOINT [--path DATA_ROOT]}"
gpu="${2:?single physical GPU ID required}"
run_dir="${3:?original run directory required}"
checkpoint="${4:?complete epoch checkpoint required}"
shift 4
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo "single numeric GPU ID required" >&2; exit 2; }
case "$recipe" in
  b1_standard|b1_hold|full_standard) cfg="cfgs/ct_seqtrack/38_${recipe}_mini.yaml" ;;
  *) echo "unknown v38 recipe: $recipe" >&2; exit 2 ;;
esac
[[ -f main.py && -f "$cfg" && -f "$run_dir/run_manifest.json" && -f "$checkpoint" ]] || { echo "missing original run/config/checkpoint" >&2; exit 2; }
python_bin="${CT_PYTHON:-/home/lishengjie/miniconda3/envs/seqtrack3d/bin/python}"
data_root="${CT_DATA_ROOT:-/home/lishengjie/data/nuscenes-mini}"
if [[ $# -gt 0 ]]; then
  [[ $# -eq 2 && "$1" == --path ]] || { echo "only --path DATA_ROOT is accepted" >&2; exit 2; }
  data_root="$2"
fi
"$python_bin" -c 'import sys; from pathlib import Path; from models.ct_v38.entry import parse_config, validate_formal_registration, validate_run_destination; c=parse_config(sys.argv[1:]); validate_formal_registration(c); validate_run_destination(c,Path(c.log_dir))' \
  --cfg "$cfg" --path "$data_root" --devices "$gpu" --log_dir "$run_dir" --checkpoint "$checkpoint"
resume_log="$(mktemp "$run_dir/resume-$(date +%Y%m%d-%H%M%S)-XXXXXX.log")"
nohup env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  "$python_bin" -u main.py --cfg "$cfg" --path "$data_root" --devices "$gpu" \
  --accelerator gpu --trainer_devices 1 --log_dir "$run_dir" --checkpoint "$checkpoint" \
  >"$resume_log" 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"${resume_log%.log}.pid"
printf 'RUN=%s\nRESUME_LOG=%s\nPID=%s\n' "$run_dir" "$resume_log" "$pid"
