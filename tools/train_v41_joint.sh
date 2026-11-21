#!/usr/bin/env bash
# 一组联合训练。可通过 DATA_ROOT、HISTORY_BANK 和 main.py 的具名参数覆盖 YAML。
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
python_bin=${PYTHON:-python}
args=(--cfg "${V41_CONFIG:-cfgs/ct_seqtrack/41_full_kitti_joint_training.yaml}")
[[ -z ${DATA_ROOT:-} ]] || args+=(--path "$DATA_ROOT")
[[ -z ${HISTORY_BANK:-} ]] || args+=(--v41_history_bank "$HISTORY_BANK")
args+=("$@")

build_history() (
    donor=${DONOR_RUN:-}
    source_dir=${DONOR_SOURCE:-}
    if [[ -z $donor ]]; then
        for run in output/20261118-v41-q0-dev-gpu0-180229-k1yozphm/uniform \
                   output/v41-q0-dev-gpu0-20261118-180229-k1yozphm/uniform; do
            if [[ -f $run/formal_checkpoints/epoch=057.ckpt && -f $run/run_manifest.json && -f $run/resolved_config.yaml ]]; then
                donor=$run
                break
            fi
        done
    fi
    [[ -n $donor ]] || { echo "生成历史库需要 E57 的 uniform 运行目录，请设置 DONOR_RUN。" >&2; return 1; }
    if [[ -z $source_dir ]]; then
        source_dir=$(mktemp -d "${TMPDIR:-/tmp}/ct-v41-donor.XXXXXXXX")
        trap 'rm -rf -- "$source_dir"' EXIT
        if ! git archive b819e760ac4c255820b47f1e8459872701c1e748 main.py requirement.txt models datasets utils cfgs \
            | tar -x -C "$source_dir"; then
            echo "当前目录无法提取冻结源码，请将 DONOR_SOURCE 指向匹配 E57 的源码目录。" >&2
            return 1
        fi
    fi
    "$python_bin" tools/build_v41_history_augmentation.py --donor-source "$source_dir" \
        --donor-run "$donor" --path "$1" --output "$2"
)

prepare_history() {
    # 复用训练入口的 YAML 继承与 CLI 优先级，路径含空格或换行也可传递。
    local arg
    local -a settings
    for arg in "${args[@]}"; do
        [[ $arg != -h && $arg != --help ]] || return 0
    done
    mapfile -d '' -t settings < <("$python_bin" -c '
import os, sys
from models.ct_v41.entry import parse_config
cfg = parse_config(sys.argv[1:])
for value in (None if cfg.test else cfg.v41_history_bank, cfg.path, cfg.devices):
    sys.stdout.buffer.write(os.fsencode("" if value is None else str(value)) + b"\0")
' "${args[@]}")
    (( ${#settings[@]} == 3 )) || { echo "读取 V41 训练配置失败。" >&2; return 1; }
    [[ -z ${settings[2]} ]] || export CUDA_VISIBLE_DEVICES="${settings[2]}"
    [[ -n ${settings[0]} && ! -f ${settings[0]} ]] || return 0
    build_history "${settings[1]}" "${settings[0]}"
}

prepare_history
exec "$python_bin" main.py "${args[@]}"
