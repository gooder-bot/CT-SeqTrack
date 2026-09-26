# v34 最新启动说明：八组 mini 单 seed 并行

2026-09-26 更新。本页替代此前“两组、不重跑参考、不扩 LR”的启动安排。用户本轮明确选择八组：GPU0 为 R1＋S 三档，GPU1 为 R05＋W 三档，每卡四个独立进程。每组单卡、seed42、scratch60、batch16、workers4、FP32；全部保留71,700次更新。方法与LR依据见 [八组实验协议](B0_V34_LR_GRID.md)，结构见 [B0上下文](B0_V34_CONTEXT.md)。

## 上传前结论

本地实现与最新配置已完成；服务器只读核验时尚无 `34_*.yaml`，需要先上传本轮文件。服务器为Python3.9.19、torch2.0.1+cu118、pytorch_lightning2.0.2、nuscenes-devkit1.1.9，mini数据路径有效，不需要重装环境。

两张A40各46,068MiB，核验时各约42GiB空闲；已有其他任务各占约3.3GB，本项目无运行进程。既有正式60轮峰值B0约7,049.7MiB、SeqTrack约3,939.9MiB；每卡3B0＋1参考的旧模型allocated合计约24.5GiB，资源上支持这一安排，但v34的新梯度通路和四进程同时运行尚未实测。

此前空间NLL、确定性pooling、CUDA bool排序/浮点cumsum修复保留。命令显式使用 `PYTORCH_CUDA_ALLOC_CONF=backend:native`，覆盖旧shell的 `max_split_size_mb:64`；保留 CUBLAS 确定性环境。**不要沿用旧 `cfgs/seqtrack3d_nuscenes_mini.yaml` 或 `--preloading`。** 物理卡用 `CUDA_VISIBLE_DEVICES`，每进程仍为 `--trainer_devices 1`。

## 同步文件与一次真实批次

同步当前 `main.py`、`models/`、`datasets/`、`utils/`、`cfgs/ct_seqtrack/`、`tools/`、`docs/`、`tests/` 和根目录说明文件；使用本轮打包的 `upload_code.zip` 可避免漏掉新增identity/配置/比较工具。保留服务器 `output/`、`artifacts/`、数据和原环境；不要上传本地权重作为初始化。SeqTrack继续用独立参考的v33身份，新S/W用v34身份，文件名中的版本不是另一套模型结构。

只需在上传后跑**一次**新的v34真实batch16检查，不为八个LR组重复跑；它不保存checkpoint：

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python tools/check_v34_batch.py \
  --cfg cfgs/ct_seqtrack/34_b0_context_mini.yaml \
  --path /home/lishengjie/data/nuscenes-mini --device cuda
```

输出 `status=passed` 后，在该项目目录分别执行下面八段即可同时后台运行；不必等待上一组训练结束。每段打印准确目录，日期由执行当天自动产生；末尾随机后缀避免同秒重启覆盖旧日志。`python -u` 保证日志实时刷新。

## 八组独立启动命令

### 1. R1：SeqTrack 正常，GPU0，LR=1e-4

```bash
CT_RUN="$(mktemp -d "output/$(date +%Y%m%d-%H%M%S)-33_seqtrack_ref-mini_car_seed42_60ep_bs16-XXXXXX")"
nohup env CUDA_VISIBLE_DEVICES=0 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/33_seqtrack_ref_mini.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --check_val_every_n_epoch 5 \
  --accelerator gpu --trainer_devices 1 --log_dir "$CT_RUN" \
  > "$CT_RUN/train.log" 2>&1 < /dev/null &
echo $! > "$CT_RUN/train.pid"
echo "$CT_RUN"
```

### 2. S1：S 正常，GPU0，LR=1e-4

```bash
CT_RUN="$(mktemp -d "output/$(date +%Y%m%d-%H%M%S)-34_b0_context_normal_lr-mini_car_seed42_60ep_bs16-XXXXXX")"
nohup env CUDA_VISIBLE_DEVICES=0 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/34_b0_context_normal_lr_mini.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --check_val_every_n_epoch 5 \
  --accelerator gpu --trainer_devices 1 --log_dir "$CT_RUN" \
  > "$CT_RUN/train.log" 2>&1 < /dev/null &
echo $! > "$CT_RUN/train.pid"
echo "$CT_RUN"
```

### 3. S05：S 减半，原 C 学习率，GPU0，LR=5e-5

```bash
CT_RUN="$(mktemp -d "output/$(date +%Y%m%d-%H%M%S)-34_b0_context_half_lr-mini_car_seed42_60ep_bs16-XXXXXX")"
nohup env CUDA_VISIBLE_DEVICES=0 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/34_b0_context_mini.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --check_val_every_n_epoch 5 \
  --accelerator gpu --trainer_devices 1 --log_dir "$CT_RUN" \
  > "$CT_RUN/train.log" 2>&1 < /dev/null &
echo $! > "$CT_RUN/train.pid"
echo "$CT_RUN"
```

### 4. S025：S 推荐探索档，GPU0，LR=2.5e-5

```bash
CT_RUN="$(mktemp -d "output/$(date +%Y%m%d-%H%M%S)-34_b0_context_quarter_lr-mini_car_seed42_60ep_bs16-XXXXXX")"
nohup env CUDA_VISIBLE_DEVICES=0 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/34_b0_context_quarter_lr_mini.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --check_val_every_n_epoch 5 \
  --accelerator gpu --trainer_devices 1 --log_dir "$CT_RUN" \
  > "$CT_RUN/train.log" 2>&1 < /dev/null &
echo $! > "$CT_RUN/train.pid"
echo "$CT_RUN"
```

### 5. R05：SeqTrack 减半，GPU1，LR=5e-5

```bash
CT_RUN="$(mktemp -d "output/$(date +%Y%m%d-%H%M%S)-33_seqtrack_ref_half_lr-mini_car_seed42_60ep_bs16-XXXXXX")"
nohup env CUDA_VISIBLE_DEVICES=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/33_seqtrack_ref_half_lr_mini.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --check_val_every_n_epoch 5 \
  --accelerator gpu --trainer_devices 1 --log_dir "$CT_RUN" \
  > "$CT_RUN/train.log" 2>&1 < /dev/null &
echo $! > "$CT_RUN/train.pid"
echo "$CT_RUN"
```

### 6. W1：W 正常，GPU1，LR=1e-4

```bash
CT_RUN="$(mktemp -d "output/$(date +%Y%m%d-%H%M%S)-34_b0_context_w4_normal_lr-mini_car_seed42_60ep_bs16-XXXXXX")"
nohup env CUDA_VISIBLE_DEVICES=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/34_b0_context_w4_normal_lr_mini.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --check_val_every_n_epoch 5 \
  --accelerator gpu --trainer_devices 1 --log_dir "$CT_RUN" \
  > "$CT_RUN/train.log" 2>&1 < /dev/null &
echo $! > "$CT_RUN/train.pid"
echo "$CT_RUN"
```

### 7. W05：W 减半，原 C 学习率，GPU1，LR=5e-5

```bash
CT_RUN="$(mktemp -d "output/$(date +%Y%m%d-%H%M%S)-34_b0_context_w4_half_lr-mini_car_seed42_60ep_bs16-XXXXXX")"
nohup env CUDA_VISIBLE_DEVICES=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/34_b0_context_w4_mini.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --check_val_every_n_epoch 5 \
  --accelerator gpu --trainer_devices 1 --log_dir "$CT_RUN" \
  > "$CT_RUN/train.log" 2>&1 < /dev/null &
echo $! > "$CT_RUN/train.pid"
echo "$CT_RUN"
```

### 8. W025：W 推荐探索档，GPU1，LR=2.5e-5

```bash
CT_RUN="$(mktemp -d "output/$(date +%Y%m%d-%H%M%S)-34_b0_context_w4_quarter_lr-mini_car_seed42_60ep_bs16-XXXXXX")"
nohup env CUDA_VISIBLE_DEVICES=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/34_b0_context_w4_quarter_lr_mini.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --check_val_every_n_epoch 5 \
  --accelerator gpu --trainer_devices 1 --log_dir "$CT_RUN" \
  > "$CT_RUN/train.log" 2>&1 < /dev/null &
echo $! > "$CT_RUN/train.pid"
echo "$CT_RUN"
```

## 新开终端查看进度

先进入项目目录，然后选择相应的一条命令；每条跟随该组最新目录。也可以直接用启动时打印的精确目录替换。Ctrl+C只退出tail，不停止后台训练。

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
```

R1：SeqTrack 正常：

```bash
tail -n 80 -f "$(ls -dt output/*-33_seqtrack_ref-mini_car_seed42_60ep_bs16-* | head -n 1)/train.log"
```

S1：S 正常：

```bash
tail -n 80 -f "$(ls -dt output/*-34_b0_context_normal_lr-mini_car_seed42_60ep_bs16-* | head -n 1)/train.log"
```

S05：S 减半，原 C 学习率：

```bash
tail -n 80 -f "$(ls -dt output/*-34_b0_context_half_lr-mini_car_seed42_60ep_bs16-* | head -n 1)/train.log"
```

S025：S 推荐探索档：

```bash
tail -n 80 -f "$(ls -dt output/*-34_b0_context_quarter_lr-mini_car_seed42_60ep_bs16-* | head -n 1)/train.log"
```

R05：SeqTrack 减半：

```bash
tail -n 80 -f "$(ls -dt output/*-33_seqtrack_ref_half_lr-mini_car_seed42_60ep_bs16-* | head -n 1)/train.log"
```

W1：W 正常：

```bash
tail -n 80 -f "$(ls -dt output/*-34_b0_context_w4_normal_lr-mini_car_seed42_60ep_bs16-* | head -n 1)/train.log"
```

W05：W 减半，原 C 学习率：

```bash
tail -n 80 -f "$(ls -dt output/*-34_b0_context_w4_half_lr-mini_car_seed42_60ep_bs16-* | head -n 1)/train.log"
```

W025：W 推荐探索档：

```bash
tail -n 80 -f "$(ls -dt output/*-34_b0_context_w4_quarter_lr-mini_car_seed42_60ep_bs16-* | head -n 1)/train.log"
```

每5轮的validation只用于过程观察；自动完成58/59/60正式评测后才比较。不要加 `--no_late3`、不按最好epoch挑结果。完成标志为该run的 `results.json` 与日志 `[v33 complete]`（参考）或 `[v34 complete]`（S/W），不是目录存在或PID存在。

## 恢复与结果比较

恢复只使用原run、原cfg和完整epoch checkpoint，在原命令中保留原 `--log_dir` 并加 `--checkpoint 原目录/formal_checkpoints/epoch=020.ckpt`。LR/窗口/版本均不能交叉恢复；不要重新执行mktemp后假称同一run续训。

八组全部完成后，将变量路径设为本轮准确目录，再运行：

```bash
/home/lishengjie/miniconda3/envs/seqtrack3d/bin/python tools/compare_v34_lr_grid.py \
  --registered-reference output/20260924-193210-33_seqtrack_ref-mini_car_seed42_60ep_bs16 \
  --reference "$R1" --reference-half "$R05" \
  --baseline output/20260924-193242-33_b0_half_lr-mini_car_seed42_60ep_bs16 \
  --old-b0 output/20260922-191248-32_b0-mini_car_seed42_60ep_bs16 \
  --s-normal "$S1" --s-half "$S05" --s-quarter "$S025" \
  --w-normal "$W1" --w-half "$W05" --w-quarter "$W025"
```

`R1/R05/S1/S05/S025/W1/W05/W025`是待替换的run路径变量，不是前面启动命令自动定义的变量。工具默认只向终端输出JSON；可用 `--output artifacts/ct_checks/新的目录/lr_grid.json` 保存新报告。原两组工具 `compare_v34_b0.py` 保留，默认仍只比较两份5e-5配置。

旧R的总体四项和旧C的移动四项仍是固定晋级门；本轮新R1/R05另外比较，不能依据新参考较低就降低门。所有组六候选固定按final60 Success、Precision排序，同时保留late-3、弱观测和移动诊断。S05−旧C是结构主对照，每一LR内W−S评价课程。
