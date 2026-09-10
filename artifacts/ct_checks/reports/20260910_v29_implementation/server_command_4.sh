export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
DATA_ROOT=/home/lishengjie/code/SparseFusion-main/nuscenes/nuscenes/
RUN_STAMP=替换为实际日期时间戳
B0_DIR="output/${RUN_STAMP}-29_b0-nuscenes_car_seed42_60ep_bs16"
CFC_DIR="output/${RUN_STAMP}-29_full_cfc-nuscenes_car_seed42_60ep_bs16"
GRU_DIR="output/${RUN_STAMP}-29_full_gru-nuscenes_car_seed42_60ep_bs16"

(
set -e
for E in 058 059 060; do
  DEST="$B0_DIR/evaluation/epoch=$E"
  mkdir -p "$DEST"
  CUDA_VISIBLE_DEVICES=1 python -u main.py \
    --cfg "$B0_DIR/resolved_config.yaml" --path "$DATA_ROOT" \
    --checkpoint "$B0_DIR/formal_checkpoints/epoch=$E.ckpt" --test \
    --workers 12 --seed 42 --log_dir "$DEST" \
    > "$DEST/eval.log" 2>&1
done
)
