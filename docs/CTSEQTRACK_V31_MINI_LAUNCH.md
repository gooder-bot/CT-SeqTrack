# v31 mini 三组实验：运行命令与结果边界

2026-09-22整理。代码版本 `64ad056` 的三臂真实60轮及58–60评测已完成；结果见 [验证记录](CTSEQTRACK_V31_READINESS.md) 与 [正式报告](../artifacts/ct_checks/reports/20260921_v31_mini_three_arm/REPORT.md)。性能未通过，不能再沿用9/19“待上传/待训练”的状态。

以下命令保留当前v31运行接口，供用户在明确要开始新实验时执行；本次清理未上传、安装或启动服务器任务，旧9/19上传包不代表清理后源码。

## 模型和参数

- `main.py` 使用独立v31 entry和host；同帧联合训练、跨帧detach，无旧标定回退。
- B0为三历史帧与当前帧观测；B1为物理时间先验和可微context；B2固定768→256新增点、36槽原始身份记忆与三模式，最终四假设共用定位/质量头。
- Adam每batch一次联合更新：FP32、lr=1e-4、betas=(0.5,0.999)、eps=1e-6，StepLR每20轮×0.1；60轮/batch16/workers4/seed42/val5。
- 每2轮保存epoch边界，58/59/60均保存并自动评测，固定报告final60与late-3。正式从epoch0随机初始化，工程权重不参与。
- 三份配置为 `31_b0_mini.yaml`、`31_full_cfc_mini.yaml`、`31_full_gru_mini.yaml`，共用独立 `31_formal_base.yaml`。配置内容保持原样；当前算法问题的修订未在清理中实施。
- 不使用旧 `--preloading`；每worker256MiB原始云缓存。每进程一张可见卡；`CUDA_VISIBLE_DEVICES` 指物理卡，`trainer_devices=1` 指数量。

## 环境与资源

项目根为 `/home/lishengjie/study/lcyu/CT-SeqTrack`，Python为 `/home/lishengjie/miniconda3/envs/seqtrack3d/bin/python`，数据根为 `/home/lishengjie/data/nuscenes-mini`。正式运行记录为Python3.9.19/Torch2.0.1+cu118/Lightning2.0.2；v31网络使用PyTorch算子，不要求旧PointNet++ CUDA扩展。

历史分配为B0→GPU1，Full-CfC/Full-GRU→GPU0。三组记录的peak allocated约7034/7329/7330MiB；两Full共卡竞争，不能拿总耗时作独占速度比较。这些是既有记录，不代表当前GPU空闲情况；路径和SDK fallback见 [服务器路径](SERVER_PATHS.md)。

下列三条命令各自创建新日期目录并后台运行；只在用户决定开始新运行后使用。

### B0 → GPU1

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
RUN="output/$(date +%Y%m%d-%H%M%S)-31_b0-mini_car_seed42_60ep_bs16"
mkdir -p "$RUN"
nohup env CUDA_VISIBLE_DEVICES=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/31_b0_mini.yaml \
  --batch_size 16 --epoch 60 --workers 4 --seed 42 \
  --check_val_every_n_epoch 5 --tag mini_car_seed42_60ep_bs16 \
  --log_dir "$RUN" > "$RUN/train.log" 2>&1 < /dev/null &
echo $! > "$RUN/train.pid"
echo "$RUN/train.log"
```

### Full-CfC → GPU0

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
RUN="output/$(date +%Y%m%d-%H%M%S)-31_full_cfc-mini_car_seed42_60ep_bs16"
mkdir -p "$RUN"
nohup env CUDA_VISIBLE_DEVICES=0 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/31_full_cfc_mini.yaml \
  --batch_size 16 --epoch 60 --workers 4 --seed 42 \
  --check_val_every_n_epoch 5 --tag mini_car_seed42_60ep_bs16 \
  --log_dir "$RUN" > "$RUN/train.log" 2>&1 < /dev/null &
echo $! > "$RUN/train.pid"
echo "$RUN/train.log"
```

### Full-GRU → GPU0

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
RUN="output/$(date +%Y%m%d-%H%M%S)-31_full_gru-mini_car_seed42_60ep_bs16"
mkdir -p "$RUN"
nohup env CUDA_VISIBLE_DEVICES=0 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/31_full_gru_mini.yaml \
  --batch_size 16 --epoch 60 --workers 4 --seed 42 \
  --check_val_every_n_epoch 5 --tag mini_car_seed42_60ep_bs16 \
  --log_dir "$RUN" > "$RUN/train.log" 2>&1 < /dev/null &
echo $! > "$RUN/train.pid"
echo "$RUN/train.log"
```

## 新终端查看进度

先连接并进入项目：

```bash
ssh lishengjie@10.109.253.86
cd /home/lishengjie/study/lcyu/CT-SeqTrack
```

分别选一条；自动选取对应模块最新日期目录，Ctrl+C仅结束tail：

```bash
tail -n 80 -f "$(ls -dt output/*-31_b0-mini_car_seed42_60ep_bs16 | head -n 1)/train.log"
tail -n 80 -f "$(ls -dt output/*-31_full_cfc-mini_car_seed42_60ep_bs16 | head -n 1)/train.log"
tail -n 80 -f "$(ls -dt output/*-31_full_gru-mini_car_seed42_60ep_bs16 | head -n 1)/train.log"
```

日志开头的 `[v31]` 会记录 arm、temporal_backend、B1/B2/B3 启用情况、参数量及 calibration_required=false；后续 `[v31 train]` 每50个batch打印进度，每次验证打印完整统计。`run_manifest.json`、`resolved_config.yaml`、CSV/TensorBoard和最终评测按run分开保存。

已完成实验的结果是三组确实不同，但两个Full的final Success仍低于同版B0。后续仍遵守相同数据、预算和指标；历史v30参照40.473742/47.840262保留，不能以旧机械同分问题消失代替性能验收。
