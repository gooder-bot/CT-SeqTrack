# CT-SeqTrack

CT-SeqTrack 是一个面向 **timestamp-native / variable-rate 3D 单目标跟踪** 的研究型项目。它基于 SeqTrack3D 改造，目标是把原本固定帧步长的多帧点云序列学习，推进到由真实时间间隔 `delta_t` 驱动的状态估计。

当前仓库是研究快照：P0-P5 的工程链路已经实现并通过 smoke test，但 P5 full 第一轮结果还不是正向结论。当前差距诊断、代码原因和下一步消融计划见 `gap_analysis.md`。

---

## 研究定位

很多 3D SOT 方法默认历史帧是固定离散序列：

```text
t-1, t-2, t-3 ...
```

这会导致同一个 `t-1` 在正常 2Hz keyframe、低帧率、跳帧、掉帧和长时间遮挡恢复场景下被赋予近似相同的时间语义。CT-SeqTrack 的核心想法是把真实 timestamp 作为一等输入：

```text
state = f(observations, real delta_t)
```

当前论文边界应收窄为：

```text
CT-SeqTrack converts SeqTrack3D from fixed-step frame sequence learning
to timestamp-native variable-rate state estimation.
```

也就是说，本项目当前不主打更大的 backbone，也不宣称完整 Neural ODE / SDE / CDE tracker，而是先验证真实时间字段、时间重采样一致性和可观测性融合在 SeqTrack3D 框架中的作用。

---

## 核心创新点

### 1. Timestamp-native Seq2Seq 3D SOT

训练侧和测试侧都提供真实时间字段：

```text
timestamps
delta_t
delta_T
current_timestamp
current_delta_t
```

点云时间通道和历史 box corner token 共用 `TimeEncoding`。支持：

```text
raw | mlp | fourier
```

默认使用 `raw`，保持时间输入的 scalar 行为，避免把收益混到模型容量增加里。

### 2. Dynamics / Velocity Branch

`DynamicsEncoder` 从历史参考框序列中提取真实时间差分运动信息：

```text
velocity = displacement / delta_t
angular_velocity = angle_delta / delta_t
```

输出：

```text
z_dyn
velocity_pred
dynamics_displacement_pred
dynamics_valid
```

对应配置开关：

```yaml
use_dynamics_encoder: False
```

### 3. Time-resampling Consistency

TWC 构造同一当前绝对时刻下的两个历史采样视图。两个 view 共享最近历史 anchor，只改变更早历史路径：

```text
view A: [t-1, t-2, t-3] -> t
view B: [t-1, t-3, t-5] -> t
```

训练目标：

```text
L = 0.5 * (L_a + L_b) + lambda_twc * L_twc
```

对应配置开关：

```yaml
use_twc: False
```

### 4. Observability-aware Fusion

P5 gate 根据当前观测可靠性，在 point feature 和 timestamp-conditioned dynamics prior 之间融合。

当前 gate 输入统计量：

```text
log1p(num_points_in_search)
log1p(estimated_fg_points)
mean_fg_score
valid_history_ratio
current_delta_t / time_scale
```

对应配置开关：

```yaml
use_observability_gate: False
```

注意：当前 P5 feature-level gate 仍在诊断中。第一轮 P5 full 在 nuScenes-mini 上不稳定，主要怀疑是 dynamics feature 约束偏弱且融合方式过强。详细分析见 `gap_analysis.md`。

---

## 当前进度

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| P0 | 真实时间字段主链路 | 已完成 |
| P1 | 真实时间 baseline smoke test | 已完成 |
| P2 | scalar-preserving `TimeEncoding` | 已完成 |
| P3 | Dynamics / Velocity Branch | 已实现，默认关闭 |
| P4 | Time-resampling Consistency | 已实现，默认关闭 |
| P5 | Observability Gate | 已实现，默认关闭，正在诊断 |
| Evaluation | 正式消融和困难子集评估 | 下一步 |

当前最重要的消融顺序：

```text
A0: SeqTrack baseline
A1: CT-base, dynamics=False, gate=False
A2: CT + Dynamics, gate=False
A2-lite: CT + Dynamics, num_candidates=1
A3-safe: CT + Dynamics + Gate, higher observation bias
A3-res: CT + Dynamics + residual gate
```

---

## 目录结构

```text
cfgs/
  seqtrack3d_nuscenes.yaml              # 默认 CT-base 配置，新模块默认关闭
  seqtrack3d_nuscenes_p5_obs_gate.yaml  # P5 gate 实验配置
  seqtrack3d_waymo.yaml                 # Waymo 配置

datasets/
  sampler.py                            # 训练采样、时间字段、TWC paired views
  misc_utils.py                         # 时间戳和历史帧工具

models/
  seqtrack3d.py                         # 主模型、TWC loss、P3/P5 接入
  time_encoding.py                      # raw / mlp / fourier 时间编码
  dynamics.py                           # P3 DynamicsEncoder
  observability.py                      # P5 ObservabilityGate

tools/
  check_time_batch.py
  check_forward_batch.py
  check_train_steps.py
  check_twc_batch.py
  check_observability_gate.py

compare_results/
  experiment_comparison.md
  metrics_summary.csv
  metrics_points.csv

gap_analysis.md                         # 当前差距诊断与消融计划
need_to_do.md                           # 当前执行清单
refined_plan.md                         # 研究定位、贡献和论文边界
log.md                                  # 工程验收日志
```

---

## 环境配置

先安装与本机 CUDA 匹配的 PyTorch。项目依赖见：

```bash
pip install -r requirement.txt
```

依赖中包含 `pytorch-lightning`、`nuscenes-devkit`、`torchmetrics` 和 PointNet2 ops。若 PointNet2 ops 编译失败，先确认 CUDA、PyTorch 和编译工具链版本匹配。

---

## 数据准备

本项目沿用 SeqTrack3D / Open3DSOT 风格的数据准备流程，支持 nuScenes 和 Waymo。准备数据后，在配置文件中修改：

```yaml
path: /your/path
version: v1.0-trainval
category_name: Car
```

nuScenes-mini 示例：

```yaml
path: /home/lishengjie/data/nuscenes-mini
version: v1.0-mini
category_name: Car
```

---

## 工程检查命令

检查真实时间字段：

```bash
python tools/check_time_batch.py \
  --cfg cfgs/seqtrack3d_nuscenes.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --version v1.0-mini \
  --split mini_train \
  --batch-size 2 \
  --workers 0 \
  --require-full-history
```

检查 forward 和 loss：

```bash
CUDA_VISIBLE_DEVICES=0 \
python tools/check_forward_batch.py \
  --cfg cfgs/seqtrack3d_nuscenes.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --version v1.0-mini \
  --split mini_train \
  --batch-size 1 \
  --workers 0 \
  --require-full-history
```

检查 TWC paired view：

```bash
python tools/check_twc_batch.py \
  --cfg cfgs/seqtrack3d_nuscenes.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --version v1.0-mini \
  --split mini_train \
  --batch-size 2 \
  --workers 0 \
  --require-full-history
```

检查 2-step 训练：

```bash
CUDA_VISIBLE_DEVICES=0 \
python tools/check_train_steps.py \
  --cfg cfgs/seqtrack3d_nuscenes.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --version v1.0-mini \
  --split mini_train \
  --batch-size 1 \
  --workers 0 \
  --max-steps 2 \
  --require-full-history \
  --memory-fraction 0.20 \
  --grad-clip 1.0
```

---

## 训练与测试

### CT-base

默认 `seqtrack3d_nuscenes.yaml` 中：

```yaml
use_dynamics_encoder: False
use_observability_gate: False
use_twc: False
```

训练命令：

```bash
CUDA_VISIBLE_DEVICES=0 \
python main.py \
  --cfg cfgs/seqtrack3d_nuscenes.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --version v1.0-mini \
  --batch_size 2 \
  --epoch 60 \
  --seed 42 \
  --tag "ct_base_mini_car_60ep"
```

### P5 Gate 实验

```bash
CUDA_VISIBLE_DEVICES=0 \
python main.py \
  --cfg cfgs/seqtrack3d_nuscenes_p5_obs_gate.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --version v1.0-mini \
  --batch_size 2 \
  --epoch 60 \
  --seed 42 \
  --tag "ct_p5_obs_gate_mini_car_60ep"
```

### 测试 checkpoint

```bash
python main.py \
  --cfg cfgs/seqtrack3d_nuscenes.yaml \
  --checkpoint path/to/checkpoint.ckpt \
  --test
```

输出目录：

```text
output/<time>-<config>-<tag>/
```

---

## 当前实验诊断

第一轮 nuScenes-mini 对比：

| model | success best | precision best | success final | precision final |
| --- | ---: | ---: | ---: | ---: |
| SeqTrack baseline | 52.2834 | 65.2144 | 50.9858 | 59.9617 |
| CT-SeqTrack P5 full | 44.9836 | 62.5120 | 31.1937 | 31.8851 |

解释：

- P5 full 当前不是最终正向结果。
- early precision peak 说明模型仍有定位能力。
- 后期崩坏更像是 dynamics train-test distribution mismatch 和 gate feature replacement 共同造成的不稳定。

完整诊断见：

```text
gap_analysis.md
```

---

## 论文边界

当前建议不要宣称：

- 完整 Neural ODE / SDE / CDE tracker
- 任意时刻 `state(t*)` 查询
- Mamba / SSM tracker
- 首次解决 sparse / occlusion 3D SOT
- P5 full 已经取得最终正向结果

更稳的贡献表述是：

```text
We convert SeqTrack3D from fixed-step frame sequence learning
to timestamp-native variable-rate state estimation.
```

---

## Acknowledgement

本项目基于 SeqTrack3D，并沿用 Open3DSOT 风格的 3D SOT 训练与评测框架。感谢 SeqTrack3D、Open3DSOT、PointNet2、DETR 和 attention-is-all-you-need-pytorch 等工作的开源贡献。

---

## License

本项目遵循 `LICENSE` 中的 MIT License。
