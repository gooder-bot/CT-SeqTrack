# V41 KITTI Car 训练历史对照

两组均使用联合搜索、uniform 中心损失、seed 42、batch 16、60 轮。A 的普通窗口执行现有 1/4/4/8 课程，B 在同一批物理端点上独立初始化四个候选。分支 0 为干净起点，分支 1–3 在有两帧以上历史的起点使用跨场景、按预测框内点数匹配的连续 XYZ 残差。末框位置、共同航向偏移、合法首框模板和时间戳沿用现有定义。两组各自从零训练，B 的重置和增强次数会更多。[A 配置](../cfgs/ct_seqtrack/41_history_window_1448.yaml) · [B 配置](../cfgs/ct_seqtrack/41_history_four_candidates.yaml)

## 一次准备 E57 增强库

将当前项目工作树连同新增文件同步后，从服务器项目根目录执行。两臂使用同一个获分配的物理 GPU 和相同的软件环境，方便复核共同 E1、E2 的数值。以下命令为本次对照创建独立目录。仅第一次运行这一段，后续命令复用同一组变量。历史供体 run 须含原 E57 checkpoint、运行清单及解析后的配置。

```bash
set -euo pipefail
PY=/home/lishengjie/miniconda3/envs/ctseqtrack3d/bin/python
DATA=/home/lishengjie/data/cxtrack/training
DONOR=output/20261118-v41-q0-dev-gpu0-180229-k1yozphm/uniform
if [ ! -f "$DONOR/formal_checkpoints/epoch=057.ckpt" ]; then
  DONOR=output/v41-q0-dev-gpu0-20261118-180229-k1yozphm/uniform
fi
mkdir -p artifacts/ct_checks
PAIR=$(mktemp -d "$PWD/artifacts/ct_checks/history-pair-XXXXXXXX")
V3="$PAIR/uniform-v3"
mkdir "$V3"
git archive b819e760ac4c255820b47f1e8459872701c1e748 main.py requirement.txt models datasets utils cfgs | tar -x -C "$V3"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="/home/lishengjie/study/lcyu/nuscenes-devkit/python-sdk${PYTHONPATH:+:$PYTHONPATH}"
"$PY" tools/build_v41_history_augmentation.py --donor-source "$V3" --donor-run "$DONOR" \
  --path "$DATA" --output "$PAIR/history_e57.json" --shuffled-audit "$PAIR/shuffled_240.json"
```

建库只读训练集 0000–0016。使用已审计的 49 轨、1,515 个三帧片段，为全部训练轨迹计算截至上一帧的点数档。已有供体归档可通过 `--donor-frames PATH` 指定。新检出缺少归档时，工具会从 E57 重放原 49 轨并核对几何与点数指纹。冻结审计在同一批 240 个端点固定末中心和当前点顺序，报告旧平移、连续残差和同批残差逐帧打散的 IoU，单独保存打散后实际发生变化的标记。

## 两组正式训练

```bash
mkdir -p output
A=cfgs/ct_seqtrack/41_history_window_1448.yaml
B=cfgs/ct_seqtrack/41_history_four_candidates.yaml
RUN_A=$(mktemp -d "$PWD/output/v41-history-window-XXXXXXXX")
RUN_B=$(mktemp -d "$PWD/output/v41-history-four-XXXXXXXX")
COMMON=(--path "$DATA" --v41_history_bank "$PAIR/history_e57.json" --v41_fixed_replay_bank "$PAIR/natural_e2.sqlite")
"$PY" main.py --cfg "$A" "${COMMON[@]}" --log_dir "$RUN_A"
"$PY" main.py --cfg "$B" "${COMMON[@]}" --log_dir "$RUN_B"
```

进入 E3 时，首个训练进程用公共 E2 权重从合法首框跑一次训练集自然轨迹，原子发布 SQLite 前缀库。另一组验证实际 E2 参数、BN 缓冲、源码及数据指纹，使用同一库。每类每轮最多 128 个前缀，每个物理键只回放一次，在自己的分支 0 单步槽以完整历史重新前向并反传。已有点却漂移、重新出现、局部裁剪并集漏车和邻车干扰分开计数。后续自然辅助轨迹继续记录两组各自的失败分布。训练集旧权重仅用于准备输入，供体前向算力应另报。

已用历史正式运行的 441 条训练轨迹长度复算新入口的公共物理队列。两组每轮均有 76,324 个主端点、4,771 次更新。E1/E2 各组都有 55,923 个普通受扰起点。E3 的 A/B 为 46,505 / 55,923，E60 为 11,373 / 55,923，E60 队列指纹与历史正式运行一致。这项对照衡量整套重置策略的差值，包含增强次数与历史来源同时变化的影响。

每轮训练审计保存主队列、辅助插入计划、回放清单指纹、实际重置/续步次数、增强回退次数、B1 新鲜有效加速度的裁剪频率。按当前有车、仅历史有车、纯背景、全空记录端点数以及 coarse/q0 中心损失对输出框 XYZ 的梯度范数和。该项可除以组内端点数比较强度，不能据此判断参数更新对跟踪成绩有益或有害。E1/E2 单独保存准备输入流、联合损失流、更新后模型权重的指纹。训练库文件可移机，内容指纹须与原 run 一致。

断点恢复使用各自的运行目录和完整 epoch 边界 checkpoint。下面的命令从 A 的 E2 在原目录继续训练。

```bash
"$PY" main.py --cfg "$RUN_A/resolved_config.yaml" --log_dir "$RUN_A" \
  --checkpoint "$RUN_A/formal_checkpoints/epoch=002.ckpt"
```

## GT 诊断与正式比较

```bash
"$PY" tools/audit_v41_training_gt.py --cfg "$A" --path "$DATA" \
  --checkpoint "$RUN_A/formal_checkpoints/epoch=002.ckpt" --output "$PAIR/gt_e2.json"
"$PY" tools/compare_v41_training_arms.py --window-run "$RUN_A" --candidate-run "$RUN_B" > "$PAIR/dev.json"
```

GT 工具在训练集中从同一失败前态分叉，跟踪自然、原样复制和一次过去 GT 校正后的八步，记录实采点、IoU、连续三帧恢复及截短随访。触发和恢复沿用历史冻结审计的 benchmark-compatible IoU，逐帧同时保存几何精确 IoU。点数按局部和远区精裁的原始点 ID 去重。输入校正止于已提交的时刻。复制副本与正式 dev/test 和两组训练梯度隔离。

正式选型以 dev 的 E58–60 平均 S/P 为主，E60 复核。比较器核验两组 60 轮物理批次和优化预算，并给出逐轨差、中心偏差超过 10 m 的时长、共同行驶轨迹回波重现后一秒内稳定跟踪的计数。E60 和末三轮的两项指标若不能一致支持同一组，则报告本次未分出稳定优势。若需同时报告 held-out test，固定轮次单独评估到各自的新目录。

```bash
for RUN in "$RUN_A" "$RUN_B"; do
  for E in 058 059 060; do
    "$PY" main.py --cfg "$RUN/resolved_config.yaml" --path "$DATA" --test --eval_split test \
      --diagnostics_level full --checkpoint "$RUN/formal_checkpoints/epoch=$E.ckpt" \
      --log_dir "$RUN/heldout_test/epoch=$E"
  done
done
"$PY" tools/compare_v41_training_arms.py --window-run "$RUN_A" --candidate-run "$RUN_B" \
  --split test > "$PAIR/test.json"
```
