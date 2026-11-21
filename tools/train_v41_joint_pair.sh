#!/usr/bin/env bash
# 同一张卡依次训练窗口和四独立候选，复用 E57 增强及这次对照的固定 E2 库。
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    echo '用法：bash tools/train_v41_joint_pair.sh [两组共同的 main.py 训练参数]'
    echo 'DATA_ROOT、HISTORY_BANK、DONOR_RUN、DONOR_SOURCE、PYTHON、CUDA_VISIBLE_DEVICES 可沿用单组设置。'
    echo 'PAIR_DIR 可指定结果目录；中断后指定原目录会从各组完整轮次继续。'
    exit 0
fi
common_args=("$@")
if (( $# )); then
    # 其余具名参数统一传给两组，例如巡查间隔、横向预热、worker 数或 GPU。
    "${PYTHON:-python}" -c '
import sys
from models.ct_v41.entry import parse_config
owned = {"--cfg", "--log_dir", "--checkpoint", "--v41_fixed_replay_bank"}
if any(arg.split("=", 1)[0] in owned for arg in sys.argv[1:]):
    sys.exit("本脚本按组分配配置、输出和断点，共享库位于 PAIR_DIR。")
cfg = parse_config(["--cfg", "cfgs/ct_seqtrack/41_joint_window_1448.yaml", *sys.argv[1:]])
if (cfg.test or cfg.ct_engineering_check or cfg.eval_split != "dev" or
    (cfg.seed, cfg.batch_size, cfg.epoch) != (42, 16, 60) or
    not all(e % cfg.check_val_every_n_epoch == 0 for e in (58, 59)) or
    (cfg.diagnostics_level != "full" and not {58, 59, 60}.issubset(cfg.full_diagnostic_epochs))):
    sys.exit("这份正式对照使用 seed42、batch16、60 轮及 dev E58–60 完整记录。")
' "$@"
fi
if [[ -z ${PAIR_DIR:-} ]]; then
    mkdir -p output
    PAIR_DIR=$(mktemp -d "$PWD/output/v41-joint-pair-XXXXXXXX")
else
    mkdir -p -- "$PAIR_DIR"
    PAIR_DIR=$(cd -- "$PAIR_DIR" && pwd -P)
fi
echo "本次对照目录：$PAIR_DIR"
if command -v flock >/dev/null; then
    exec 9>>"$PAIR_DIR/.train.lock"
    flock -n 9 || { echo "此目录的对照已在运行：$PAIR_DIR" >&2; exit 1; }
fi
run_arm() {
    local arm=$1 dir="$PAIR_DIR/$1"
    local -a resume=() checkpoints
    if [[ -f "$dir/epochs/epoch=060.json" && -f "$dir/formal_checkpoints/epoch=060.ckpt" ]]; then
        echo "$arm 已完成 E60，复用已有运行。"
        return
    fi
    if [[ -f "$dir/run_manifest.json" ]]; then
        shopt -s nullglob
        checkpoints=("$dir"/formal_checkpoints/epoch=[0-9][0-9][0-9].ckpt)
        shopt -u nullglob
        if (( ! ${#checkpoints[@]} )); then
            echo "$arm 已有运行记录且尚无完整轮次 checkpoint，请检查 $dir 后重新安排这个训练目录。" >&2
            return 1
        fi
        resume=(--checkpoint "${checkpoints[-1]}")
    fi
    V41_CONFIG="cfgs/ct_seqtrack/41_joint_$arm.yaml" bash tools/train_v41_joint.sh \
        "${common_args[@]}" --v41_fixed_replay_bank "$PAIR_DIR/natural_e2.sqlite" --log_dir "$dir" "${resume[@]}"
    [[ -f "$dir/epochs/epoch=060.json" && -f "$dir/formal_checkpoints/epoch=060.ckpt" ]] || {
        echo "$arm 尚未完成 E60。再次运行时指定 PAIR_DIR=$PAIR_DIR 可继续。" >&2
        return 1
    }
}
run_arm window_1448
run_arm four_candidates
report=$(mktemp "$PAIR_DIR/.dev-comparison-XXXXXXXX")
trap 'rm -f -- "$report"' EXIT
"${PYTHON:-python}" tools/compare_v41_training_arms.py --require-q0-lateral \
    --window-run "$PAIR_DIR/window_1448" --candidate-run "$PAIR_DIR/four_candidates" > "$report"
mv -- "$report" "$PAIR_DIR/dev_comparison.json"
echo "固定轮次对照结果：$PAIR_DIR/dev_comparison.json"
