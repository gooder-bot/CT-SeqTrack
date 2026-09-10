export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
DATA_ROOT=/home/lishengjie/code/SparseFusion-main/nuscenes/nuscenes/
STAMP=$(date +%Y%m%d-%H%M%S)
CHECK_ROOT="artifacts/ct_checks/${STAMP}-v29-engineering"
mkdir -p artifacts/ct_checks

nohup python -u tools/run_ct_v29_checks.py \
  --path "$DATA_ROOT" --gpu 1 --workers 12 --output "$CHECK_ROOT" \
  > "${CHECK_ROOT}.log" 2>&1 < /dev/null &
echo $! > "${CHECK_ROOT}.pid"
printf '工程目录：%s\n' "$CHECK_ROOT"
tail -f "${CHECK_ROOT}.log"
