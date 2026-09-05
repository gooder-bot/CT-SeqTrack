# v27 mini 五臂后台启动（2026-09-05）

从服务器 CT-SeqTrack 根目录执行。先同步当前工作区的 main.py、models/、datasets/、utils/ 与 cfgs/（包括新增27文件）；仅更新YAML或拉取旧HEAD不包含本轮改动。

五臂采用最新27配置：mini Car、seed42、60epoch、batch16、FP32、Adam；workers4、preloading、每5轮内部dev诊断。B0绑定物理GPU2，其余四臂分别作为独立进程并行共享物理GPU3。每个进程的trainer_devices=1。四进程显存峰值叠加；当前代码验收不代表已验证GPU3能同时容纳它们。

last.ckpt每个完整epoch保存，58/59/60额外保存到formal_checkpoints/，不受验证间隔5影响。run_provenance.json记录启动配置，resolved_config.yaml可直接用于同run续训或Full后续校准；不能把整个provenance文件或嵌套hparams当作配置。

## B0 · GPU 2

```bash
RUN="output/$(date +%Y%m%d-%H%M%S)-27_b0-mini_car_seed42_60ep_bs16"
mkdir -p "$RUN"
nohup env CUDA_VISIBLE_DEVICES=2 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -u main.py \
  --cfg cfgs/ct_seqtrack/27_b0.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --seed 42 \
  --preloading --check_val_every_n_epoch 5 \
  --tag ct27_b0_mini_car_seed42_60ep_bs16 \
  --log_dir "$RUN" \
  > "$RUN/train.log" 2>&1 < /dev/null &
echo $! > "$RUN/train.pid"
```

## B1-CfC · GPU 3

```bash
RUN="output/$(date +%Y%m%d-%H%M%S)-27_b1_cfc-mini_car_seed42_60ep_bs16"
mkdir -p "$RUN"
nohup env CUDA_VISIBLE_DEVICES=3 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -u main.py \
  --cfg cfgs/ct_seqtrack/27_b1_cfc.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --seed 42 \
  --preloading --check_val_every_n_epoch 5 \
  --tag ct27_b1_cfc_mini_car_seed42_60ep_bs16 \
  --log_dir "$RUN" \
  > "$RUN/train.log" 2>&1 < /dev/null &
echo $! > "$RUN/train.pid"
```

## B1-GRU · GPU 3

```bash
RUN="output/$(date +%Y%m%d-%H%M%S)-27_b1_gru-mini_car_seed42_60ep_bs16"
mkdir -p "$RUN"
nohup env CUDA_VISIBLE_DEVICES=3 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -u main.py \
  --cfg cfgs/ct_seqtrack/27_b1_gru.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --seed 42 \
  --preloading --check_val_every_n_epoch 5 \
  --tag ct27_b1_gru_mini_car_seed42_60ep_bs16 \
  --log_dir "$RUN" \
  > "$RUN/train.log" 2>&1 < /dev/null &
echo $! > "$RUN/train.pid"
```

## B1+B2 · GPU 3

```bash
RUN="output/$(date +%Y%m%d-%H%M%S)-27_full_minus_b3-mini_car_seed42_60ep_bs16"
mkdir -p "$RUN"
nohup env CUDA_VISIBLE_DEVICES=3 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -u main.py \
  --cfg cfgs/ct_seqtrack/27_full_minus_b3.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --seed 42 \
  --preloading --check_val_every_n_epoch 5 \
  --tag ct27_full_minus_b3_mini_car_seed42_60ep_bs16 \
  --log_dir "$RUN" \
  > "$RUN/train.log" 2>&1 < /dev/null &
echo $! > "$RUN/train.pid"
```

## Full · GPU 3

```bash
RUN="output/$(date +%Y%m%d-%H%M%S)-27_full-mini_car_seed42_60ep_bs16"
mkdir -p "$RUN"
nohup env CUDA_VISIBLE_DEVICES=3 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -u main.py \
  --cfg cfgs/ct_seqtrack/27_full.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --seed 42 \
  --preloading --check_val_every_n_epoch 5 \
  --tag ct27_full_mini_car_seed42_60ep_bs16 \
  --log_dir "$RUN" \
  > "$RUN/train.log" 2>&1 < /dev/null &
echo $! > "$RUN/train.pid"
```

各段可连续粘贴运行，后台进程互不等待。启动后可在同一终端执行 `tail -f "$RUN/train.log"` 查看刚启动的实验。每组的PID在对应目录train.pid。

Full训练期间没有校准artifact时照常学习，递归输出仍为observation。训练结束后，58/59/60分别校准再评估官方mini_val；B1+B2正式评测为bounded_always。

服务器真实nuScenes 100-step和正式60轮结果尚未由本地验收代替；工程checkpoint不用作正式初始化。此前完整审计见[训练就绪记录](CTSEQTRACK_V27_TRAINING_READINESS.md)。
