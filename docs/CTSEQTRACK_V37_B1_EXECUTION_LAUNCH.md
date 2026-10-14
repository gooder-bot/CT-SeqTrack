# v37 B1 四组新执行配方：启动与日志

2026-10-14。以下四组由用户确认，均另起 scratch；代理只在本地准备配置、代码和文档，SSH 权限仍为只读，未上传或启动服务器任务。旧 v37 尚未结束时，不要替换其正在使用的源码。等旧任务完成、保留其登记源码及原 output 后，再由用户部署本轮完整源码与注册文件并执行下列命令。旧输出目录不删除、不覆盖。

本地最新修订已按用户要求同步到 `D:/desktop/research/CT-SeqTrack` 的 main 工作区，可从该目录正常提交Git；此前工程副本作为历史检查来源保留。此次本地同步不表示服务器已部署。

## 四组配置与比较范围

共同设置：nuScenes mini Car、seed42、FP32、60 epoch、初始 LR=1.5e-4、无 warmup、每批一次 Adam、单卡、workers=4；保持 W=1/4/4/8、十轮课程、112 个尾部预留窗口和每帧 1024 点。四组均每轮验证，`check_val_every_n_epoch=1`。

| 组别 | 配置（`cfgs/ct_seqtrack/`） | GPU | batch | 全局投影 | LR milestones |
|---|---|---:|---:|---|---|
| factorized | `37_b1_dedup_factorized_mini.yaml` | 0 | 16 | 分解计算 | `[20,50]` |
| concat | `37_b1_dedup_concat_mini.yaml` | 0 | 16 | 原拼接计算 | `[20,50]` |
| factorized_hold_lr | `37_b1_dedup_factorized_hold_lr_mini.yaml` | 1 | 16 | 分解计算 | `[20]` |
| factorized_bs128 | `37_b1_dedup_factorized_bs128_mini.yaml` | 1 | 128 | 分解计算 | `[20,50]` |

四组公共的阶段 A 去重在 B1 路径中实际生效的只有 `unique_mask`。登记选项中其余 B2/Full 优化不构成本轮 B1 的有效变量。`prior_read` 不适用于 B1，用户已取消这一无效对照；concat 组不是 `prior_read` 实验。

factorized 与 concat 保留相同的局部和全局信息、原参数及网络维度。分解只是将原卷积的局部项、全局项分别计算后相加，不能通过删除全局特征来替代。FP32 运算顺序变化已经在工程检查中出现输出、梯度及优化器状态差异，不能称为旧 v37 的无损替换；因此本轮分配独立配方身份、从零训练，不加载旧权重。concat 与 factorized 的 batch16 对照用于评估这一执行改变；最终分数与速度均待实测。

hold_lr 只取消第 50 轮结束后的第二次衰减：第 1–20 轮 LR=1.5e-4，第 21–60 轮 LR=1.5e-5。其余三组保留原节奏：第 1–20 轮 1.5e-4、第 21–50 轮 1.5e-5、第 51–60 轮 1.5e-6。bs128 不按 batch 倍数放大学习率，也不启用梯度累积；它改变了批大小和更新频率，不能当作相同训练轨迹的纯提速。

四组共 240 epoch。每组实际 Adam 更新次数以采样器计划和运行审计为准，尤其不能将 batch128 写成原 batch16 的固定总更新数。checkpoint 仍每两轮保存，并保留末三轮 58/59/60；每轮验证写入 `validation/epoch=001.json` 等独立 JSON。最终按 final60 和 late-3 比较，不跨组挑不同最佳轮次。

## 一次 batch128 真实 CUDA 短检查

由用户部署后，在 GPU1 有足够空闲资源时执行一次。默认零基课程 epoch10、两次完整 prepare/forward/loss/backward/Adam/commit，权重仍从头随机初始化；不保存 checkpoint，也不能用检查权重初始化正式训练。报告写入新的 `artifacts/ct_checks/` 目录。

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack || exit
env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python tools/check_b1_training_recipe.py \
  --cfg cfgs/ct_seqtrack/37_b1_dedup_factorized_bs128_mini.yaml \
  --device cuda --path /home/lishengjie/data/nuscenes-mini
```

该检查记录实际 batch128、分阶段显存和有限性；两步通过不保证全部课程、所有真实样本或两组并发时均不会 OOM，也不是训练吞吐基准。GPU1 推荐先运行 hold_lr，完成后再运行 bs128。只有用户根据实测峰值和当前空闲显存确认余量后，才考虑这两组并行；不承诺同卡能同时容纳。若 bs128 OOM，保留失败报告，不自动缩小 batch 或改成梯度累积后仍沿用该组身份。GPU0 两组也按实际资源安排，并发墙钟不能直接作为算法加速证据。

## 四条独立启动命令

每段可在独立终端执行。`mktemp` 为本次运行创建全新目录；命令打印 PID 和 RUN，并在该目录保存 `train.pid`。记下 RUN，以便查看原运行或按同 run 规则恢复。以下均为新 scratch，不含 `--checkpoint` 或 `--init_checkpoint`。

### 1. factorized，batch16，GPU0

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack || exit
mkdir -p output
CT_RUN="$(mktemp -d "output/$(date +%Y%m%d-%H%M%S)-37_b1_dedup_factorized-mini_car_seed42_60ep_bs16-XXXXXX")" || exit
nohup env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/37_b1_dedup_factorized_mini.yaml --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --check_val_every_n_epoch 1 \
  --accelerator gpu --trainer_devices 1 --log_dir "$CT_RUN" \
  >"$CT_RUN/train.log" 2>&1 </dev/null &
CT_PID=$!
printf '%s\n' "$CT_PID" >"$CT_RUN/train.pid"
printf 'PID=%s RUN=%s\n' "$CT_PID" "$CT_RUN"
```

新终端查看该组最新运行：

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack || exit
CT_RUN="$(find output -maxdepth 1 -type d -name '*-37_b1_dedup_factorized-mini_car_seed42_60ep_bs16-*' -print | LC_ALL=C sort | tail -n 1)"
if [ -n "$CT_RUN" ]; then echo "$CT_RUN"; tail -n 80 -f "$CT_RUN/train.log"; else echo '尚未找到该组运行目录'; fi
```

### 2. concat，batch16，GPU0

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack || exit
mkdir -p output
CT_RUN="$(mktemp -d "output/$(date +%Y%m%d-%H%M%S)-37_b1_dedup_concat-mini_car_seed42_60ep_bs16-XXXXXX")" || exit
nohup env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/37_b1_dedup_concat_mini.yaml --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --check_val_every_n_epoch 1 \
  --accelerator gpu --trainer_devices 1 --log_dir "$CT_RUN" \
  >"$CT_RUN/train.log" 2>&1 </dev/null &
CT_PID=$!
printf '%s\n' "$CT_PID" >"$CT_RUN/train.pid"
printf 'PID=%s RUN=%s\n' "$CT_PID" "$CT_RUN"
```

新终端查看该组最新运行：

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack || exit
CT_RUN="$(find output -maxdepth 1 -type d -name '*-37_b1_dedup_concat-mini_car_seed42_60ep_bs16-*' -print | LC_ALL=C sort | tail -n 1)"
if [ -n "$CT_RUN" ]; then echo "$CT_RUN"; tail -n 80 -f "$CT_RUN/train.log"; else echo '尚未找到该组运行目录'; fi
```

### 3. factorized_hold_lr，batch16，GPU1

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack || exit
mkdir -p output
CT_RUN="$(mktemp -d "output/$(date +%Y%m%d-%H%M%S)-37_b1_dedup_factorized_hold_lr-mini_car_seed42_60ep_bs16-XXXXXX")" || exit
nohup env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/37_b1_dedup_factorized_hold_lr_mini.yaml --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --check_val_every_n_epoch 1 \
  --accelerator gpu --trainer_devices 1 --log_dir "$CT_RUN" \
  >"$CT_RUN/train.log" 2>&1 </dev/null &
CT_PID=$!
printf '%s\n' "$CT_PID" >"$CT_RUN/train.pid"
printf 'PID=%s RUN=%s\n' "$CT_PID" "$CT_RUN"
```

新终端查看该组最新运行：

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack || exit
CT_RUN="$(find output -maxdepth 1 -type d -name '*-37_b1_dedup_factorized_hold_lr-mini_car_seed42_60ep_bs16-*' -print | LC_ALL=C sort | tail -n 1)"
if [ -n "$CT_RUN" ]; then echo "$CT_RUN"; tail -n 80 -f "$CT_RUN/train.log"; else echo '尚未找到该组运行目录'; fi
```

### 4. factorized_bs128，batch128，GPU1

建议在第 3 组结束后执行这一段；下面命令不会自动等待第 3 组退出。

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack || exit
mkdir -p output
CT_RUN="$(mktemp -d "output/$(date +%Y%m%d-%H%M%S)-37_b1_dedup_factorized_bs128-mini_car_seed42_60ep_bs128-XXXXXX")" || exit
nohup env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/37_b1_dedup_factorized_bs128_mini.yaml --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 128 --epoch 60 --workers 4 --check_val_every_n_epoch 1 \
  --accelerator gpu --trainer_devices 1 --log_dir "$CT_RUN" \
  >"$CT_RUN/train.log" 2>&1 </dev/null &
CT_PID=$!
printf '%s\n' "$CT_PID" >"$CT_RUN/train.pid"
printf 'PID=%s RUN=%s\n' "$CT_PID" "$CT_RUN"
```

新终端查看该组最新运行：

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack || exit
CT_RUN="$(find output -maxdepth 1 -type d -name '*-37_b1_dedup_factorized_bs128-mini_car_seed42_60ep_bs128-*' -print | LC_ALL=C sort | tail -n 1)"
if [ -n "$CT_RUN" ]; then echo "$CT_RUN"; tail -n 80 -f "$CT_RUN/train.log"; else echo '尚未找到该组运行目录'; fi
```

这些日志命令只选择同名组最新目录。查看较早一次运行时，将 `CT_RUN` 手动设为当时打印的 RUN，再执行 `tail -n 80 -f "$CT_RUN/train.log"`。打印 PID 只表示后台进程已创建，不等于注册检查通过或训练已开始；应在日志中确认配置身份、数据加载和首批训练输出。恢复只能使用该组原配置、原 RUN 和完整 epoch checkpoint，不能跨四组或从旧 v37 checkpoint 迁移。
