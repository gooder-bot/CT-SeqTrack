# 历史 M3/M4 第一阶段改造说明

日期：2026-07-24  
状态：工程实现与 matched runner 已完成；可以直接启动训练和 baseline 比较，尚需服务器结果确认是否涨分。

## 1. 为什么这样改

已有证据仍作为实验控制，但不再作为停止 M 阶段的 No-Go：

1. 旧对称 TWC 中，paired-view 训练本身相对单视图造成明显退化；一致性项虽有净正效应，但没有追回主任务损失。
2. M2 的 true/fixed/shuffled 正确时间因果门禁失败，不能把后续模块继续包装为“正确物理时间带来收益”。

因此当前涨分改造采用：

- M3 改为非对称、time-agnostic 的 endpoint path distillation；
- M4 改为固定状态模型下的递归轨迹稳定与 search reachability；
- 两者默认关闭，M3 与 legacy TWC 互斥；
- 所有 GT crop 指标只在 forward 外做离线诊断，不进入模型输入或决策。

## 2. M3：Canonical-to-irregular endpoint path distillation

训练视图：

- A：规范历史 `[1, 2, 3]`；
- B：不规则历史 `[1, 3, 5]`；
- A/B 共享当前帧、当前采样点、坐标锚和 sample-level shared SE(2) candidate transform。
- B 路使用冻结的 BatchNorm running statistics，但保留 affine 与主干梯度，避免不规则视图污染部署所用 A 分布。

优化结构：

```text
student(A) -- supervised loss A
student(B) -- optional supervised loss B, engineering default beta=0
EMA teacher(A) -- detached endpoint target + GT-free hybrid confidence

student(B endpoint) -> match -> EMA teacher(A endpoint)
```

第一版只约束共同终点，不直接对齐两条历史的中间框，因为 A/B 中间框对应不同物理帧。这样“path”表示不同历史路径到同一 endpoint 的稳健性，而不是错误地把不同时刻的中间状态逐项对齐。

损失为：

```text
Lsup = (LA + beta * LB) / (1 + beta)
Lref = SmoothL1(refinedB, stopgrad(refinedEMA-A))
Lcoarse = SmoothL1(coarseB, stopgrad(coarseEMA-A))
Lm3  = hybrid_confidence(A) * valid_pair
       * (Lref + 0.25 * Lcoarse) / 1.25
L    = Lsup + ramp(epoch) * lambda_m3 * Lm3
```

hybrid confidence 是 current-frame foreground confidence 与 teacher coarse/refined proposal agreement 的几何平均，不读取 GT。工程默认 `beta=0`，用于隔离并避免旧 paired-view 双路真值监督退化。`lambda_m3=0.05` 是第一轮正式起点，不是冻结超参。

关键日志：

- `loss_m3_path`
- `m3_center_loss` / `m3_yaw_loss`
- `m3_center_gap` / `m3_yaw_gap`
- `m3_coarse_center_gap` / `m3_coarse_yaw_gap`
- `m3_valid_ratio`
- `m3_teacher_confidence`
- `m3_teacher_foreground_confidence`
- `m3_teacher_agreement_confidence`
- `m3_effective_sample_weight`
- `m3_path_weight_effective`
- `m3_teacher_updates`

## 3. M4：Fixed filter and predictive trajectory tube

状态为：

```text
[x, y, z, vx, vy, vz, yaw, yaw_rate]
```

实现包含：

- 固定 `Q/R` 的 constant-velocity / constant-yaw-rate filter；
- Joseph-form covariance update 与 PSD 投影；
- 可选 Mahalanobis gate；
- 非法或过大 `delta_t` 的显式重初始化回退；
- 由上一 posterior 到当前 prior 构造、受协方差与最大尺寸约束的 trajectory tube；
- baseline crop 与 tube crop 的去重并集，最终仍采样为同一个 `point_sample_size`，保持网络输入预算不变。

正式 runner 默认同时评测 `m4_time_mode=real` 和 `fixed`：前者面向涨分，后者作为时钟控制。

四个同 checkpoint 评测臂由 `--m4_variant` 控制：

```text
off
filter
tube
filter_tube
```

离线 oracle 日志只检查 crop reachability：

- `m4/oracle_center_recall_baseline`
- `m4/oracle_center_recall_union`
- `m4/search_points_baseline`
- `m4/search_points_union`
- `m4/prediction_valid_ratio`
- `m4/measurement_accept_ratio`

目标中心 oracle 不进入 `data_dict`，不会影响 tracker forward。

## 4. 直接运行

完整 M3 训练、M3 baseline 对照和 M4 四臂比较：

```bash
INIT_CKPT=/absolute/path/to/m2.ckpt \
DATA_ROOT=/home/lishengjie/data/nuscenes-mini \
GPU=0 \
bash tools/run_m_stage_pipeline.sh
```

详细参数、输出路径与读数规则见 `M_STAGE_RUNBOOK.md`。

## 5. 实验顺序与判读

### M4 先做零训练成本筛选

在同一冻结 checkpoint、同一 endpoint manifest 上依次跑：

```powershell
python main.py --cfg cfgs/seqtrack3d_nuscenes_m4_filter_tube_engineering.yaml --test --checkpoint <ckpt> --m4_variant off
python main.py --cfg cfgs/seqtrack3d_nuscenes_m4_filter_tube_engineering.yaml --test --checkpoint <ckpt> --m4_variant filter
python main.py --cfg cfgs/seqtrack3d_nuscenes_m4_filter_tube_engineering.yaml --test --checkpoint <ckpt> --m4_variant tube
python main.py --cfg cfgs/seqtrack3d_nuscenes_m4_filter_tube_engineering.yaml --test --checkpoint <ckpt> --m4_variant filter_tube
```

先审查 endpoint exact match、GT hash 与序列顺序，再比较：

1. standard Success/Precision 不得明显下降；
2. gap1124 的 union center recall 必须高于 baseline；
3. recall 增益必须转化为跟踪指标，而不只是加入背景点；
4. 若 tube oracle 无新增覆盖，停止调滤波参数，不训练 M4。

### M3 再做工程门禁

```powershell
python tools/check_m3_m4_invariants.py
python main.py --cfg cfgs/seqtrack3d_nuscenes_m3_endpoint_distill_engineering.yaml --init_checkpoint <selected-m2.ckpt> --seed 42 --batch_size 16 --workers 12 --epoch 60
```

完整训练前必须先通过：

1. EMA teacher 初始输出与 student A 一致；
2. teacher 参数无梯度且不在 optimizer param groups；
3. A/B anchor 与 current XYZ gap 为 0；
4. `beta=0, lambda=0` 时，A 分支 loss/gradient 与单视图 control 对齐；
5. active loss finite，B 有非零蒸馏梯度；
6. 显存和单步时间可接受。

第一轮必须包含三个 matched arms：

```text
A: single-view continuation（在 M3 命令后加 `--m3_variant off`）
B: M3 paired path, lambda_m3=0（在 M3 命令后加 `--m3_path_weight 0`）
C: M3 paired path, lambda_m3=0.05
```

主效应先看 `C-B`，部署价值再看 `C-A`。若 `C-B>0` 但 `C-A<0`，只能写蒸馏对 paired control 有效，不能写 full tracker 涨分。

## 6. 当前代码入口

- `models/path_distillation.py`：M3 endpoint loss、teacher confidence、EMA update。
- `datasets/sampler.py`：M3 paired history sampler 与共享终点约束。
- `models/seqtrack3d.py`：teacher lifecycle、M3 loss 和日志。
- `models/state_filter.py`：M4 filter、tube、crop geometry 与 point union。
- `models/base_model.py`：递归评测接线、回退与 oracle 日志。
- `tools/check_m3_m4_invariants.py`：dataset-free invariants。
- `tools/check_m_stage_configs.py`：M3/M4 配置与 matched-arm 合同。
- `tools/run_m3_matched_abc.sh`：M3 A/B/C 训练。
- `tools/run_m3_matched_evaluation.sh`：M3 standard/gap1124 配对比较。
- `tools/run_m4_matched_evaluation.sh`：M4 四臂在线递归比较。
- `tools/run_m_stage_pipeline.sh`：一条命令串起全部阶段。
- `cfgs/seqtrack3d_nuscenes_m3_endpoint_distill_engineering.yaml`：M3 工程配置。
- `cfgs/seqtrack3d_nuscenes_m4_filter_tube_engineering.yaml`：M4 四臂评测配置；加载 M3 checkpoint 时只部署 student，EMA teacher 状态会被忽略。

## 7. 论文边界

当前可以写成待验证方法假设：

> Canonical-path self-distillation may improve endpoint robustness under history resampling, while a fixed-state predictive tube may improve recursive search reachability under missing observations.

当前不能写：

- M3/M4 已经涨分；
- 正确 physical time 是收益原因；
- fixed filter 或 tube 已优于 SeqTrack3D/M2；
- oracle crop recall 等于在线跟踪收益。
