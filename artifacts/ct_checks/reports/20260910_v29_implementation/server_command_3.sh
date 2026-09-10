(
set -e
python tools/run_ct_v29_checks.py --assert-passed "$CHECK_ROOT/report.json"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
DATA_ROOT=/home/lishengjie/code/SparseFusion-main/nuscenes/nuscenes/
STAMP=$(date +%Y%m%d-%H%M%S)
B0_DIR="output/${STAMP}-29_b0-nuscenes_car_seed42_60ep_bs16"
CFC_DIR="output/${STAMP}-29_full_cfc-nuscenes_car_seed42_60ep_bs16"
GRU_DIR="output/${STAMP}-29_full_gru-nuscenes_car_seed42_60ep_bs16"
mkdir -p "$B0_DIR" "$CFC_DIR" "$GRU_DIR"

nohup env CUDA_VISIBLE_DEVICES=1 python -u main.py \
  --cfg cfgs/ct_seqtrack/29_b0_nuscenes_full.yaml --path "$DATA_ROOT" \
  --batch_size 16 --epoch 60 --workers 12 --seed 42 \
  --check_val_every_n_epoch 5 --tag nuscenes_car_seed42_60ep_bs16 \
  --log_dir "$B0_DIR" > "$B0_DIR/train.log" 2>&1 < /dev/null &
echo $! > "$B0_DIR/train.pid"

nohup env CUDA_VISIBLE_DEVICES=2 python -u main.py \
  --cfg cfgs/ct_seqtrack/29_full_cfc_nuscenes_full.yaml --path "$DATA_ROOT" \
  --batch_size 16 --epoch 60 --workers 12 --seed 42 \
  --check_val_every_n_epoch 5 --tag nuscenes_car_seed42_60ep_bs16 \
  --log_dir "$CFC_DIR" > "$CFC_DIR/train.log" 2>&1 < /dev/null &
echo $! > "$CFC_DIR/train.pid"

nohup env CUDA_VISIBLE_DEVICES=3 python -u main.py \
  --cfg cfgs/ct_seqtrack/29_full_gru_nuscenes_full.yaml --path "$DATA_ROOT" \
  --batch_size 16 --epoch 60 --workers 12 --seed 42 \
  --check_val_every_n_epoch 5 --tag nuscenes_car_seed42_60ep_bs16 \
  --log_dir "$GRU_DIR" > "$GRU_DIR/train.log" 2>&1 < /dev/null &
echo $! > "$GRU_DIR/train.pid"
printf '正式输出目录：\n%s\n%s\n%s\n' "$B0_DIR" "$CFC_DIR" "$GRU_DIR"
)
