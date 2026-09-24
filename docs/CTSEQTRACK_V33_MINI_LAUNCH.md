# v33 mini 四组单 seed：后台运行与日志

2026-09-24，按用户最新安排：重新训练 SeqTrack reference、综合 B0 A/B/C，物理 GPU 依次为 **0、0、1、1**。本轮助手只读检查服务器、修订本地文件；上传和正式启动由用户执行。四组正式实验尚未启动。

实现与训练前验证已完成：此前服务器原环境完整测试 **314 passed、1 skipped**，一次真实 nuScenes-mini batch16 的 CUDA forward/backward/Adam/commit 通过。记录见 [v33 实现与验证](B0_V33_REPAIR.md)。不必再把大量预检作为本轮训练前置条件。

## 配置与上传

| 组别 | 配置（位于 cfgs/ct_seqtrack/） | GPU | 初始 LR | 调度 |
|---|---|---:|---:|---|
| R：独立 SeqTrack | 33_seqtrack_ref_mini.yaml | 0 | 1e-4 | StepLR：20、40轮后降档 |
| A：综合 B0 原配方 | 33_b0_mini.yaml | 0 | 1e-4 | StepLR：20、40轮后降档 |
| B：综合 B0 延后衰减 | 33_b0_late_decay_mini.yaml | 1 | 1e-4 | MultiStepLR：[20,50] |
| C：综合 B0 半学习率 | 33_b0_half_lr_mini.yaml | 1 | 5e-5 | MultiStepLR：[20,50] |

共同设置：mini Car、seed42、scratch60、batch16、workers4、FP32、每5轮验证；每组71,700次更新。Adam betas=(0.5,0.999)、eps=1e-6、weight_decay=0，衰减率0.1。三组 B0 模型、loss、数据、初始化和预算一致。训练结束自动评测58/59/60，固定比较final60和late-3。

SeqTrack 使用项目内独立 `models/seqtrack_reference/`，保留其原网络、teacher和loss。本轮按用户选择重跑 R；既有 v32 R 仍是有效历史对照，不因新增被动记录而失效。

将本地最新源码、配置和文档上传到 `/home/lishengjie/study/lcyu/CT-SeqTrack` 后，运行下面命令。服务器主目录在此次只读检查时仍是 v32，先前通过验证的 v33 位于 `artifacts/ct_checks/20260924-190116_v33_deployment/source` 独立快照，不能将二者混用。上传保留服务器既有 `output/`、`artifacts/`、数据和环境，不需重新安装依赖。

旧 `cfgs/seqtrack3d_nuscenes_mini.yaml`、`--preloading` 不适用于当前入口；原始点云按需读取，每worker已有256MiB缓存。选物理卡用 `CUDA_VISIBLE_DEVICES`，`--trainer_devices 1` 表示每个进程用一张卡。

## 四条独立启动命令

每段可单独复制。输出沿用 `output/YYYYMMDD-HHMMSS-33_模块-mini_car_seed42_60ep_bs16/`，不传任何旧 checkpoint。

### 1. SeqTrack reference → GPU 0

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
RUN="output/$(date +%Y%m%d-%H%M%S)-33_seqtrack_ref-mini_car_seed42_60ep_bs16"
mkdir -p "$RUN"
nohup env CUDA_VISIBLE_DEVICES=0 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/33_seqtrack_ref_mini.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --seed 42 \
  --check_val_every_n_epoch 5 --accelerator gpu --trainer_devices 1 \
  --tag mini_car_seed42_60ep_bs16 --log_dir "$RUN" \
  > "$RUN/train.log" 2>&1 < /dev/null &
echo $! > "$RUN/train.pid"
echo "$RUN/train.log"
```

### 2. 综合 B0 A：原配方 → GPU 0

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
RUN="output/$(date +%Y%m%d-%H%M%S)-33_b0_original-mini_car_seed42_60ep_bs16"
mkdir -p "$RUN"
nohup env CUDA_VISIBLE_DEVICES=0 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/33_b0_mini.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --seed 42 \
  --check_val_every_n_epoch 5 --accelerator gpu --trainer_devices 1 \
  --tag mini_car_seed42_60ep_bs16 --log_dir "$RUN" \
  > "$RUN/train.log" 2>&1 < /dev/null &
echo $! > "$RUN/train.pid"
echo "$RUN/train.log"
```

### 3. 综合 B0 B：延后第二次衰减 → GPU 1

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
RUN="output/$(date +%Y%m%d-%H%M%S)-33_b0_late_decay-mini_car_seed42_60ep_bs16"
mkdir -p "$RUN"
nohup env CUDA_VISIBLE_DEVICES=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/33_b0_late_decay_mini.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --seed 42 \
  --check_val_every_n_epoch 5 --accelerator gpu --trainer_devices 1 \
  --tag mini_car_seed42_60ep_bs16 --log_dir "$RUN" \
  > "$RUN/train.log" 2>&1 < /dev/null &
echo $! > "$RUN/train.pid"
echo "$RUN/train.log"
```

### 4. 综合 B0 C：B 配方整体半学习率 → GPU 1

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
RUN="output/$(date +%Y%m%d-%H%M%S)-33_b0_half_lr-mini_car_seed42_60ep_bs16"
mkdir -p "$RUN"
nohup env CUDA_VISIBLE_DEVICES=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/33_b0_half_lr_mini.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --seed 42 \
  --check_val_every_n_epoch 5 --accelerator gpu --trainer_devices 1 \
  --tag mini_car_seed42_60ep_bs16 --log_dir "$RUN" \
  > "$RUN/train.log" 2>&1 < /dev/null &
echo $! > "$RUN/train.pid"
echo "$RUN/train.log"
```

## 新终端查看进度

新终端先连接并进入项目：

```bash
ssh lishengjie@10.109.253.86
cd /home/lishengjie/study/lcyu/CT-SeqTrack
```

按需选择一条。每条持续跟随，在不同终端分别查看即可：

```bash
# 1. SeqTrack reference → GPU 0
tail -n 80 -f "$(ls -dt output/*-33_seqtrack_ref-mini_car_seed42_60ep_bs16 | head -n 1)/train.log"
```

```bash
# 2. 综合 B0 A：原配方 → GPU 0
tail -n 80 -f "$(ls -dt output/*-33_b0_original-mini_car_seed42_60ep_bs16 | head -n 1)/train.log"
```

```bash
# 3. 综合 B0 B：延后第二次衰减 → GPU 1
tail -n 80 -f "$(ls -dt output/*-33_b0_late_decay-mini_car_seed42_60ep_bs16 | head -n 1)/train.log"
```

```bash
# 4. 综合 B0 C：B 配方整体半学习率 → GPU 1
tail -n 80 -f "$(ls -dt output/*-33_b0_half_lr-mini_car_seed42_60ep_bs16 | head -n 1)/train.log"
```

命令选择该组最新日期目录；也可直接使用启动时打印的精确路径。Ctrl+C只退出tail，训练继续。`[v33 train]` 每50 batch输出；`[v33 complete]` 与 `results.json` 表示58–60评测也已完成。

## 已知报错与本次结论

- 空间 NLL strict deterministic：当前分割采用class-axis log_softmax与二维NLL；参考pooling保留确定性max首索引梯度。既有CUDA测试已通过。
- CUDA bool排序/浮点cumsum：活动token排序使用int64键，B2计数用int64累计；本轮没有启用Full旧探索分支。
- 分配器旧设置：命令显式用`backend:native`覆盖旧shell的`max_split_size_mb:64`；保持FP32、严格确定性和CUBLAS workspace。
- torch2.0.1测试兼容：测试中的`any(dim=tuple)`已改为`flatten(1).any(dim=1)`，最终服务器全套通过；无需升级torch/Lightning。
- 旧配置/权重混用：仅用上述`33_*`配置从头训练；运行元数据写入前核对身份，同run恢复不会覆盖首次配置。
- 被动日志：SeqTrack未提供的目标点统计写为`None`并记录覆盖数，不伪装成观测零值；此次本地相关测试40 passed、2 skipped，S/P计算不变。

此次只读检查时GPU0/1各有其他任务占用约5.3GB；已测单B0真实batch峰值allocated约6.83GiB。现有显存余量支持按计划开始，四进程共享计算会影响速度；单批峰值不等于整次训练峰值。本轮没有新增长时并发探针。

## 结果比较

四组完成后，将`R_RUN/A_RUN/B_RUN/C_RUN`替换为各组实际目录：

```bash
/home/lishengjie/miniconda3/envs/seqtrack3d/bin/python tools/compare_v33_b0.py \
  --reference R_RUN \
  --old-b0 output/20260922-191248-32_b0-mini_car_seed42_60ep_bs16 \
  --a A_RUN --b B_RUN --c C_RUN
```

工具兼容新v33 SeqTrack参考，报告final60/late-3、缺测与移动交叉分组、同帧coarse/fine、失跟持续长度及可信状态。退出码0表示至少一组B0四项S/P均达到参考，1表示未全部达到，2表示证据不完整或不匹配。历史输出不覆盖。
