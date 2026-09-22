# v32 mini 四组单 seed：后台运行与日志

2026-09-22，按用户最新安排：SeqTrack、B0、Full-GRU、Full-CfC，全部 seed42，物理 GPU 依次 0、0、1、1。代码和配置已经在本地修订；**执行前需将当前工作树同步到服务器，包含新增的 `models/seqtrack_reference/`、32_* 配置和工具文件**，不能只同步 Git HEAD。本文命令未由助手代为执行。

## 固定设置与当前状态

- mini Car、scratch60、batch16、workers4、FP32、每5轮验证。Adam/StepLR、1024点、历史3帧及其余配置不变。
- 四个独立单卡进程，两张物理卡各两个；每个进程的 `trainer_devices=1` 是卡的数量。
- 当前入口为 `main.py`，只使用 `32_*` 配置。旧 `cfgs/seqtrack3d_nuscenes_mini.yaml` 和 `--preloading` 不适用于当前入口；每worker原始点云缓存256MiB已启用。
- 训练结束自动评测58/59/60，写入 `results.json`；固定报告 final60 和 late-3。
- 每轮19,108行/1,195次Adam，共71,700次更新。SeqTrack保留原teacher重采样，实际曝光另行记录。
- B0修复、独立参考和Full接口的本地合同检查已完成，详见[验证记录](B0_V32_REPAIR.md)。真实nuScenes/CUDA尚未执行，得分和同卡并发显存须由实际运行确认；本轮没有额外部署审批步骤。

原计划的 reference/B0 seed52 复验留待后续；`tools/compare_v32_baselines.py` 仍专用于补齐后的双seed基线验收，不能把两个Full目录代入其seed52参数。本轮先检查seed42 B0与reference final60 S/P差距，再报告两Full相对B0和四组late-3。

## 四条独立启动命令

在服务器终端执行；每段可单独复制。使用已确认的环境Python，无需另换环境；SDK父目录通过该进程的PYTHONPATH提供，数据根单独传入。输出沿用本地 `output/YYYYMMDD-HHMMSS-32_模块-mini_car_seed42_60ep_bs16/` 命名。

### 1. SeqTrack reference → GPU0

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
RUN="output/$(date +%Y%m%d-%H%M%S)-32_seqtrack_ref-mini_car_seed42_60ep_bs16"
mkdir -p "$RUN"
nohup env CUDA_VISIBLE_DEVICES=0 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  PYTHONPATH="/home/lishengjie/code/SparseFusion-main/nuscenes${PYTHONPATH:+:$PYTHONPATH}" \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/32_seqtrack_ref_mini.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --seed 42 \
  --check_val_every_n_epoch 5 --accelerator gpu --trainer_devices 1 \
  --tag mini_car_seed42_60ep_bs16 --log_dir "$RUN" \
  > "$RUN/train.log" 2>&1 < /dev/null &
echo $! > "$RUN/train.pid"
echo "$RUN/train.log"
```

### 2. B0 → GPU0

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
RUN="output/$(date +%Y%m%d-%H%M%S)-32_b0-mini_car_seed42_60ep_bs16"
mkdir -p "$RUN"
nohup env CUDA_VISIBLE_DEVICES=0 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  PYTHONPATH="/home/lishengjie/code/SparseFusion-main/nuscenes${PYTHONPATH:+:$PYTHONPATH}" \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/32_b0_mini.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --seed 42 \
  --check_val_every_n_epoch 5 --accelerator gpu --trainer_devices 1 \
  --tag mini_car_seed42_60ep_bs16 --log_dir "$RUN" \
  > "$RUN/train.log" 2>&1 < /dev/null &
echo $! > "$RUN/train.pid"
echo "$RUN/train.log"
```

### 3. Full-GRU → GPU1

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
RUN="output/$(date +%Y%m%d-%H%M%S)-32_full_gru-mini_car_seed42_60ep_bs16"
mkdir -p "$RUN"
nohup env CUDA_VISIBLE_DEVICES=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  PYTHONPATH="/home/lishengjie/code/SparseFusion-main/nuscenes${PYTHONPATH:+:$PYTHONPATH}" \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/32_full_gru_mini.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --seed 42 \
  --check_val_every_n_epoch 5 --accelerator gpu --trainer_devices 1 \
  --tag mini_car_seed42_60ep_bs16 --log_dir "$RUN" \
  > "$RUN/train.log" 2>&1 < /dev/null &
echo $! > "$RUN/train.pid"
echo "$RUN/train.log"
```

### 4. Full-CfC → GPU1

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
RUN="output/$(date +%Y%m%d-%H%M%S)-32_full_cfc-mini_car_seed42_60ep_bs16"
mkdir -p "$RUN"
nohup env CUDA_VISIBLE_DEVICES=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  PYTHONPATH="/home/lishengjie/code/SparseFusion-main/nuscenes${PYTHONPATH:+:$PYTHONPATH}" \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/32_full_cfc_mini.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --seed 42 \
  --check_val_every_n_epoch 5 --accelerator gpu --trainer_devices 1 \
  --tag mini_car_seed42_60ep_bs16 --log_dir "$RUN" \
  > "$RUN/train.log" 2>&1 < /dev/null &
echo $! > "$RUN/train.pid"
echo "$RUN/train.log"
```

`nohup` 保持后台运行，`python -u` 及时刷新日志；`train.pid` 记录该进程PID。

## 新终端查看进度

从本地新终端连接：

```bash
ssh lishengjie@10.109.253.86
cd /home/lishengjie/study/lcyu/CT-SeqTrack
```

然后按模型选择一条执行（每条都会持续跟随，不要在一个终端顺序粘贴四条）：

```bash
# SeqTrack
tail -n 80 -f "$(ls -dt output/*-32_seqtrack_ref-mini_car_seed42_60ep_bs16 | head -n 1)/train.log"
```

```bash
# B0
tail -n 80 -f "$(ls -dt output/*-32_b0-mini_car_seed42_60ep_bs16 | head -n 1)/train.log"
```

```bash
# Full-GRU
tail -n 80 -f "$(ls -dt output/*-32_full_gru-mini_car_seed42_60ep_bs16 | head -n 1)/train.log"
```

```bash
# Full-CfC
tail -n 80 -f "$(ls -dt output/*-32_full_cfc-mini_car_seed42_60ep_bs16 | head -n 1)/train.log"
```

命令选择对应模型最新的日期目录；Ctrl+C只结束tail，不会停止训练。也可以用启动时打印的精确路径代替glob。

## 已知报错处理与结果位置

- 空间NLL的strict deterministic异常：当前B0与参考均使用class-axis log_softmax + 二维NLL；参考pooling保留确定性max首索引梯度。
- CUDA bool排序和浮点cumsum：当前稀疏token使用int64排序键，B2计数使用int64 cumsum；旧策略探索/AP分支不在活动链路。
- PyTorch2.0.1 allocator环境：命令显式设置 `backend:native`，覆盖旧shell残留的 `max_split_size_mb:64` 或不受支持的allocator选项；保留严格确定性、CUBLAS workspace和FP32，不靠关闭检查绕过故障。
- 旧配置和SDK路径错误：使用独立32参考配置，SDK包父目录只进入PYTHONPATH，mini数据目录只进入`--path`。
- 独立`--test`的checkpoint轮次已改为读取已校验的checkpoint元数据，与自动评测一致。

上述路径已在本地复核；尚无本版真实CUDA运行记录，不能承诺零报错或已经达到分数目标。

每个run分别保存 `resolved_config.yaml`、`run_manifest.json`（含实际源码SHA）、`training_audits/`、`formal_checkpoints/`、`evaluation/`、`results.json`、CSV/TensorBoard。`[v32 train]` 每50个batch记录进度，`[v32 validation]` 记录验证结果，`[v32 complete]` 表示最终评测与汇总完成。只出现训练epoch60结束，不等于三份最终评测均已落盘。
