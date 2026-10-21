# v37 B1 四组新执行配方：启动与日志

**2026-10-21 状态补充。** 本页 factorized、concat、hold_lr 三组 batch16 已完成 60 轮；concat 精确复现旧强 B1，分解与 hold_lr 未通过原总体/移动门。batch4 本地完整审计/验证到 41 轮、日志进入 42 轮，尚无正式 final60/late3；快照不代表服务器实时状态。结果、成本和建议见 [新 B1 审计](../artifacts/ct_checks/20261021_v37_execution_decision_review/execution/README.md) 与 [最新研究审查](CTSEQTRACK_V37_RESEARCH_AUDIT_20261021.md)。新审查更正了旧审计“没有 peak 显存记录”的说法，但尚无受控配对净收益证据。

以下表格的 GPU=0/0/1/1 是当时计划；四个原 manifest 实际均为 CUDA_VISIBLE_DEVICES=1 且启动重叠。保留计划与真实记录两者，不改写旧 manifest，不用并发耗时推断加速。启动命令仅保留供定位/复现，不要求重跑完成组，不自动继续/停止 batch4。本次仅更新文档，旧配置与运行身份未改。

2026-10-14 更新：用户最新明确新增 **分解投影 B1 batch4**，本页四条命令的第四组由 batch8 更新为 batch4；前三组不变。batch4 除批大小及配方身份外，沿用分解投影 batch16 的全部训练设置。原 batch8、batch128 配置及已有运行证据保留；本次不停止、重启或迁移任何旧运行，也不假定 batch8 已启动或完成。

代理只在本地准备配置、代码和文档，SSH 权限仍为只读，未上传或启动服务器任务。新增配方会改变源码登记，**正在运行的任务继续使用原源码目录；新任务须使用独立部署目录，或等旧任务完成、保留其登记源码及原 output 后再更新原目录。** 不能在正在使用的目录直接覆盖源码后继续旧运行。下列命令以旧任务完成后更新原目录为例；若独立部署，请统一修改命令中的 `cd` 路径。旧输出目录不删除、不覆盖。

本地最新修订已按用户要求同步到 `D:/desktop/research/CT-SeqTrack` 的 main 工作区，可从该目录正常提交Git；此前工程副本作为历史检查来源保留。此次本地同步不表示服务器已部署。

本次新增前的本地源码为 Git commit `b494ba1c2be3697c2b127d69083df802ba74b6b4`。原五份配方配置的参数及配置身份保持；登记中追加 batch4，当前四条启动命令对应下表。变更前备份及本次检查写入 `artifacts/ct_checks/20261014-170627_b1_bs4_registration/`；batch8 的历史检查保留在 `artifacts/ct_checks/20261014-161640_b1_bs8_registration/`，不改写旧运行 manifest。

## 四组配置与比较范围

共同设置：nuScenes mini Car、seed42、FP32、60 epoch、初始 LR=1.5e-4、无 warmup、每批一次 Adam、单卡、workers=4；保持 W=1/4/4/8、十轮课程、112 个尾部预留窗口和每帧 1024 点。四组均每轮验证，`check_val_every_n_epoch=1`。

| 组别 | 配置（`cfgs/ct_seqtrack/`） | GPU | batch | 全局投影 | LR milestones |
|---|---|---:|---:|---|---|
| factorized | `37_b1_dedup_factorized_mini.yaml` | 0 | 16 | 分解计算 | `[20,50]` |
| concat | `37_b1_dedup_concat_mini.yaml` | 0 | 16 | 原拼接计算 | `[20,50]` |
| factorized_hold_lr | `37_b1_dedup_factorized_hold_lr_mini.yaml` | 1 | 16 | 分解计算 | `[20]` |
| factorized_bs4 | `37_b1_dedup_factorized_bs4_mini.yaml` | 1 | 4 | 分解计算 | `[20,50]` |

四组公共的阶段 A 去重在 B1 路径中实际生效的只有 `unique_mask`。登记选项中其余 B2/Full 优化不构成本轮 B1 的有效变量。`prior_read` 不适用于 B1，用户已取消这一无效对照；concat 组不是 `prior_read` 实验。

factorized 与 concat 保留相同的局部和全局信息、原参数及网络维度。分解只是将原卷积的局部项、全局项分别计算后相加，不能通过删除全局特征来替代。FP32 运算顺序变化已经在工程检查中出现输出、梯度及优化器状态差异，不能称为旧 v37 的无损替换；因此本轮分配独立配方身份、从零训练，不加载旧权重。concat 与 factorized 的 batch16 对照用于评估这一执行改变；最终分数与速度均待实测。

hold_lr 只取消第 50 轮结束后的第二次衰减：第 1–20 轮 LR=1.5e-4，第 21–60 轮 LR=1.5e-5。其余三组保留原节奏：第 1–20 轮 1.5e-4、第 21–50 轮 1.5e-5、第 51–60 轮 1.5e-6。bs4 不按 batch 倍数缩放学习率，也不启用梯度累积。它与 factorized batch16 比较批大小的影响；固定 60 轮下 Adam 更新次数及 BN 批统计会变化，不能视为相同训练轨迹的纯工程提速。

本页四组各 60 epoch，合计 240 epoch；这不是包含历史尝试的累计预算。batch128 已失败尝试及任何已有 batch8 运行的成本另外记录，不计作已完成的 batch4 对照。每组实际 Adam 更新次数以采样器计划和运行审计为准，不套用 batch16 的固定总更新数。checkpoint 仍每两轮保存，并保留末三轮 58/59/60；每轮验证写入 `validation/epoch=001.json` 等独立 JSON。最终按 final60 和 late-3 比较，不跨组挑不同最佳轮次。

## 可选的 batch4 真实 CUDA 短检查

由用户部署后，在 GPU1 有足够空闲资源时执行一次。默认零基课程 epoch10、两次完整 prepare/forward/loss/backward/Adam/commit，权重仍从头随机初始化；不保存 checkpoint，也不能用检查权重初始化正式训练。报告写入新的 `artifacts/ct_checks/` 目录。

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack || exit
env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python tools/check_b1_training_recipe.py \
  --cfg cfgs/ct_seqtrack/37_b1_dedup_factorized_bs4_mini.yaml \
  --device cuda --path /home/lishengjie/data/nuscenes-mini
```

该检查记录实际 batch4、分阶段显存和有限性；两步通过不保证全部课程、所有真实样本或两组并发时均不会 OOM，也不是训练吞吐基准。GPU1 可在 hold_lr 完成后运行 bs4；是否并行由实际空闲显存决定。batch128 的同卡并发 OOM 不证明其独占 GPU 也一定失败。各组并发墙钟不能直接作为算法加速证据。

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

### 4. factorized_bs4，batch4，GPU1

这是新 scratch，不从 batch8、batch16 或失败的 batch128 checkpoint 初始化。

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack || exit
mkdir -p output
CT_RUN="$(mktemp -d "output/$(date +%Y%m%d-%H%M%S)-37_b1_dedup_factorized_bs4-mini_car_seed42_60ep_bs4-XXXXXX")" || exit
nohup env CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/37_b1_dedup_factorized_bs4_mini.yaml --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 4 --epoch 60 --workers 4 --check_val_every_n_epoch 1 \
  --accelerator gpu --trainer_devices 1 --log_dir "$CT_RUN" \
  >"$CT_RUN/train.log" 2>&1 </dev/null &
CT_PID=$!
printf '%s\n' "$CT_PID" >"$CT_RUN/train.pid"
printf 'PID=%s RUN=%s\n' "$CT_PID" "$CT_RUN"
```

新终端查看该组最新运行：

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack || exit
CT_RUN="$(find output -maxdepth 1 -type d -name '*-37_b1_dedup_factorized_bs4-mini_car_seed42_60ep_bs4-*' -print | LC_ALL=C sort | tail -n 1)"
if [ -n "$CT_RUN" ]; then echo "$CT_RUN"; tail -n 80 -f "$CT_RUN/train.log"; else echo '尚未找到该组运行目录'; fi
```

## 历史保留：factorized_bs8，batch8，GPU1

原配置 `37_b1_dedup_factorized_bs8_mini.yaml` 和配方身份 `b1_factorized_bs8` 保留。本页更新前的完整 batch8 启动与日志命令可从 commit `b494ba1c2be3697c2b127d69083df802ba74b6b4:docs/CTSEQTRACK_V37_B1_EXECUTION_LAUNCH.md` 查阅。若已经启动，继续使用其原源码和 RUN；不因本页第四组变更而重跑或停止。batch4 是另一组新 scratch，不覆盖 batch8 输出。

## 历史保留：factorized_bs128，batch128，GPU1

以下是最初第四组命令，仅供查阅；当前第四条命令为上面的 batch4，不要求再次运行 batch128。原建议为第 3 组结束后运行，命令本身不自动等待其他任务退出。配置和失败日志保留，不通过修改原 batch128 身份来运行其他批大小。

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

这些日志命令只选择同名组最新目录。查看较早一次运行时，将 `CT_RUN` 手动设为当时打印的 RUN，再执行 `tail -n 80 -f "$CT_RUN/train.log"`。打印 PID 只表示后台进程已创建，不等于注册检查通过或训练已开始；应在日志中确认配置身份、数据加载和首批训练输出。恢复只能使用该组原配置、原登记源码、原 RUN 和完整 epoch checkpoint，不能跨配方或从旧 v37 checkpoint 迁移。
