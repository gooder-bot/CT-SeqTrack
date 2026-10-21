#!/usr/bin/env bash
# 用户在新源码部署目录执行；三组均 scratch，绝不修改已有运行目录。
set -euo pipefail
recipe="${1:?usage: bash tools/launch_v37_centroid.sh RECIPE GPU [main.py options]}"
gpu="${2:?a single physical GPU ID is required}"
shift 2
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo "single numeric GPU ID required" >&2; exit 2; }
case "$recipe" in
  b1_b2_mass_detach|full_mass_detach|full_mass_detach_hold_lr)
    cfg="cfgs/ct_seqtrack/37_${recipe}_mini.yaml" ;;
  *) echo "unregistered B2 centroid recipe: $recipe" >&2; exit 2 ;;
esac
for argument in "$@"; do
  case "$argument" in
    --checkpoint|--checkpoint=*|--init_checkpoint|--init_checkpoint=*|--test|--test=*)
      echo "launcher manages $argument; use main.py directly for this operation" >&2
      exit 2 ;;
  esac
done
[[ -f main.py ]] || { echo "run from CT-SeqTrack root" >&2; exit 2; }
python_bin="${CT_PYTHON:-/home/lishengjie/miniconda3/envs/seqtrack3d/bin/python}"
arguments=(--cfg "$cfg" --devices "$gpu" --accelerator gpu --trainer_devices 1)
# 未设置环境变量时沿用实际 YAML 的 path；显式 CLI 仍在最后覆盖。
if [[ "${CT_DATA_ROOT+x}" == x ]]; then
  arguments+=(--path "$CT_DATA_ROOT")
fi
arguments+=("$@")
# 预检和 main 共用参数；说明写 stderr，stdout 返回后缀、实际 GPU、可选目录。
preflight="$("$python_bin" -c '
import contextlib
import json
import sys
from pathlib import Path
from models.ct_v37.entry import parse_config, validate_formal_registration
with contextlib.redirect_stdout(sys.stderr):
    config = parse_config(sys.argv[2:])
    if (config.checkpoint is not None or config.init_checkpoint is not None
            or config.test or config.ct_engineering_check):
        raise ValueError("launcher requires a new formal scratch run; use main.py directly")
    if config.v37_centroid_recipe != sys.argv[1]:
        raise ValueError("--cfg recipe differs from launcher recipe; change the first recipe argument")
    registration = validate_formal_registration(config)
    print(json.dumps(registration, ensure_ascii=False, sort_keys=True))
    run_dir = str(config.log_dir or "")
    if any(character in run_dir for character in ("\n", "\r")):
        raise ValueError("--log_dir must not contain newline characters")
    if run_dir:
        target = Path(run_dir)
        if target.exists() and (not target.is_dir() or any(target.iterdir())):
            raise FileExistsError("--log_dir must be a new or empty directory")
suffix = f"37_{config.v37_centroid_recipe}-mini_car_seed{config.seed}_{config.epoch}ep_bs{config.batch_size}"
sys.stdout.buffer.write(("\n".join((suffix, str(config.devices), run_dir)) + "\n").encode("utf-8"))
' "$recipe" "${arguments[@]}")"
mapfile -t resolved <<< "$preflight"
suffix="${resolved[0]-}"
gpu="${resolved[1]-}"
run_dir="${resolved[2]-}"
[[ "$suffix" =~ ^37_${recipe}-mini_car_seed[0-9]+_[1-9][0-9]*ep_bs[1-9][0-9]*$ ]] || {
  echo "invalid run suffix from preflight; no output directory created" >&2; exit 2;
}
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo "invalid physical GPU ID from preflight" >&2; exit 2; }
# 源码/配置预检失败时，不创建一个看似已启动的 output。
if [[ -n "$run_dir" ]]; then
  if [[ -e "$run_dir" || -L "$run_dir" ]]; then
    [[ -d "$run_dir" ]] || { echo "--log_dir must be a directory" >&2; exit 2; }
    shopt -s nullglob dotglob
    contents=("$run_dir"/*)
    shopt -u nullglob dotglob
    [[ ${#contents[@]} -eq 0 ]] || { echo "--log_dir must be empty" >&2; exit 2; }
  fi
  mkdir -p -- "$run_dir"
else
  mkdir -p output
  run_dir="$(mktemp -d "output/$(date +%Y%m%d-%H%M%S)-${suffix}-XXXXXX")"
  arguments+=(--log_dir "$run_dir")
fi
nohup env CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  "$python_bin" -u main.py "${arguments[@]}" \
  >"$run_dir/train.log" 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$run_dir/train.pid"
printf 'RUN=%s\nPID=%s\n' "$run_dir" "$pid"
