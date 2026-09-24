# v33 追加 E：3 倍峰值学习率与 2000 步 warmup，GPU 1

2026-09-25，用户追加一组综合 B0：配置 `cfgs/ct_seqtrack/33_b0_x3_lr_warmup_mini.yaml` 继承 B，设置 `lr=3e-4`、`lr_warmup_steps=2000`，保留 `lr_milestones=[20,50]` 和衰减率 0.1，使用物理 GPU 1。原 R/A/B/C/D 的配置与安排保持。

第 1 次优化使用峰值学习率的 1/2000，即 1.5e-7；前 2000 次优化线性升至 3e-4，第 2000 次优化即使用峰值。warmup 包含在原 71,700 次更新中，不额外增加训练轮次、更新或数据曝光。

| 优化更新序号（从 1 开始） | E 实际使用的学习率 | 对应阶段 |
|---|---:|---|
| 1–2000 | `3e-4 × 更新序号 / 2000` | 线性 warmup |
| 2001–23900 | 3e-4 | warmup 结束至第 20 轮完成 |
| 23901–59750 | 3e-5 | 第 21–50 轮 |
| 59751–71700 | 3e-6 | 第 51–60 轮 |

每轮仍为 1,195 次优化：第 20 轮完成后降至 3e-5，第 50 轮完成后降至 3e-6。模型、loss、数据划分、seed42、scratch60、batch16、workers4、FP32 及 Adam 其余参数均沿用 B；每 5 轮验证，训练结束自动评测 58/59/60。

E 相对 B **同时改变学习率尺度与 warmup**。它检验这套联合配方，不能将 E−B 的差异独立归因为更大学习率或 warmup，也不预判优于 B。比较仍固定 final60 与 late-3 的 Success/Precision，保留移动目标、缺测和失跟诊断；不按各组最佳轮次选结果。

本地验证：全量契约测试 **343 passed、3 skipped**，compileall 与 diff 检查通过。覆盖 71,700 次更新的 LR 序列、不同 epoch 长度、warmup 内及衰减边界的调度状态恢复，以及真实 Lightning 2.0.2 合成训练的参数、Adam 状态和逐步 LR 连续性；旧 R/A/B/C/D 配置 SHA 精确保持。命令已通过实际参数解析器核对。两个跳过项需要 CUDA，另一项只在未安装 Lightning 时适用。本地为 torch2.3.0+cpu 与已有 Lightning2.0.2，本轮未访问或修改服务器。

E 日志每 50 批额外打印 `[v33 lr] update=... lr=...`，表示该次更新将实际使用的学习率；CSV/TensorBoard 也按 step 记录 LR。首条应为 `update=1 lr=1.5e-07`。

## 上传与启动

由用户上传并启动。服务器已有当前 v33 基础源码与 B 配置时，需同步以下 **5 个运行文件**：

- `cfgs/ct_seqtrack/33_b0_x3_lr_warmup_mini.yaml`
- `models/ct_v31/config.py`
- `models/ct_v31/lr_schedule.py`
- `models/ctseqtrackv31.py`
- `models/ct_v31/entry.py`

新 YAML 依赖已有 `33_b0_late_decay_mini.yaml` 和 `33_formal_base.yaml`；新增调度模块、正式配方登记、训练接入与学习率记录需一起同步，仅上传 YAML 不完整。沿用 `/home/lishengjie/miniconda3/envs/seqtrack3d` 环境，不重新安装依赖，保留既有 `output/` 与 `artifacts/`。

以下命令从 epoch0 随机初始化，以独立时间戳目录启动 E。不要附加旧 `--preloading`、`--init_checkpoint` 或旧权重；正式恢复仅使用同一 E run 的完整 epoch checkpoint。

```bash
cd /home/lishengjie/study/lcyu/CT-SeqTrack
RUN="output/$(date +%Y%m%d-%H%M%S)-33_b0_x3_lr_warmup-mini_car_seed42_60ep_bs16"
mkdir -p "$RUN"
nohup env CUDA_VISIBLE_DEVICES=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTORCH_CUDA_ALLOC_CONF=backend:native CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/lishengjie/miniconda3/envs/seqtrack3d/bin/python -u main.py \
  --cfg cfgs/ct_seqtrack/33_b0_x3_lr_warmup_mini.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --batch_size 16 --epoch 60 --workers 4 --seed 42 \
  --check_val_every_n_epoch 5 --accelerator gpu --trainer_devices 1 \
  --tag mini_car_seed42_60ep_bs16 --log_dir "$RUN" \
  > "$RUN/train.log" 2>&1 < /dev/null &
echo $! > "$RUN/train.pid"
echo "$RUN/train.log"
```

## 新终端看进度

在独立终端连接服务器后，查看 E 的最新时间戳目录：

```bash
ssh lishengjie@10.109.253.86
cd /home/lishengjie/study/lcyu/CT-SeqTrack
tail -n 80 -f "$(ls -dt output/*-33_b0_x3_lr_warmup-mini_car_seed42_60ep_bs16 | head -n 1)/train.log"
```

也可将 tail 路径替换为启动时打印的精确目录。Ctrl+C 只退出 tail，后台训练继续。物理 GPU 由 `CUDA_VISIBLE_DEVICES=1` 指定，`--trainer_devices 1` 表示单卡。`[v33 complete]` 与该 run 的 `results.json` 用于确认训练及 58–60 评测完成；目录存在或进程启动不代表完成。
