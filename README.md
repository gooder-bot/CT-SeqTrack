# CT-SeqTrack

CT-SeqTrack 是一个面向 **timestamp-native / variable-rate 3D 单目标跟踪** 的研究型项目。它基于 SeqTrack3D 改造，目标是把原本固定帧步长的多帧点云序列学习，推进到由真实时间间隔 `delta_t` 驱动的状态估计。

当前仓库是研究快照：P0-P5 的工程链路已经实现并通过 smoke test；已有实验显示，直接把真实时间替换进 SeqTrack3D 主干时间 token 并不稳定，当前更稳的主线是保留 order-time 主干语义，并把真实 `delta_t/current_delta_t` 作为 `DynamicsEncoder` 的运动先验。已完成记录见 `done.md`，简洁实验结论见 `sum_results.md`，下一步消融计划见 `need_to_do.md`。

## 文档导航

| 文件 | 作用 |
| --- | --- |
| `README.md` | 项目入口、当前主线、环境和命令索引 |
| `refined_plan.md` | 研究定位、论文边界、贡献叙事和 related work 边界 |
| `sum_results.md` | 按时间顺序总结已有实验说明了什么 |
| `need_to_do.md` | 当前和未来任务，只放还没有完成的事情 |
| `done.md` | 已完成工程验收、历史实验和关键输出归档 |
| `compare_results/` | 完整指标表、曲线和实验结果文件 |

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

### 1. Timestamp-native 输入契约

训练侧和测试侧都提供真实时间字段：

```text
timestamps
delta_t
delta_T
current_timestamp
current_delta_t
```

工程上已经打通真实时间字段链路，并支持点云时间通道和历史 box corner token 共用 `TimeEncoding`。支持：

```text
raw | mlp | fourier
```

注意：当前实验不支持继续把真实秒数直接塞进主干时间 token。更稳的用法是让主干保持 SeqTrack3D 的 order-time 语义，把真实时间主要交给 dynamics prior 使用。

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

注意：旧版 P5 full 同时混入 raw real-time 主干、dynamics 和 gate，不能作为最终 gate 结论。后续 gate 消融应基于干净的 order-time 主干和保守 observation-biased gate 重新验证。

---

## 术语速查

| 名称 | 含义 |
| --- | --- |
| `A1-order` | 主干使用 SeqTrack3D order-time，关闭 dynamics / TWC / gate |
| `A2-order-dyn` | 主干使用 order-time，真实 `delta_t/current_delta_t` 进入 `DynamicsEncoder` |
| `cand1` | `num_candidates=1`，不是 `candidate_id=1` |
| `cand4` | 默认多 candidate，包含 `candidate_id=0/1/2/3` |
| `disp` | 在 dynamics 上增加小权重 displacement 监督 |
| `TWC` | Time-resampling Consistency，不同历史采样路径到同一当前时刻的一致性 |
| `gate-safe` | 更保守的 observation-biased observability gate |

---

## 当前进度

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| P0 | 真实时间字段主链路 | 已完成 |
| P1 | 真实时间 baseline smoke test | 已完成 |
| P2 | scalar-preserving `TimeEncoding` | 已完成 |
| P3 | Dynamics / Velocity Branch | 已实现，默认关闭，`A2-order-dyn` 是当前最强正向信号 |
| P4 | Time-resampling Consistency | 已实现，默认关闭，等待 order-time 消融验证 |
| P5 | Observability Gate | 已实现，默认关闭，旧 P5 full 不能作为最终结论 |
| Evaluation | 当前五组消融和困难子集评估 | 下一步 |

当前最重要的消融顺序：

```text
1. A2-order-dyn-cand1
2. A2-order-dyn-disp
3. A1-order+TWC
4. A2-order-dyn+TWC
5. A3-order-gate-safe
```

---

## 目录结构

```text
cfgs/
  seqtrack3d_nuscenes.yaml              # 默认 CT-base 配置，新模块默认关闭
  seqtrack3d_nuscenes_p5_obs_gate.yaml  # P5 gate 实验配置
  seqtrack3d_nuscenes_a2_order_dyn_cand1.yaml
  seqtrack3d_nuscenes_a2_order_dyn_disp.yaml
  seqtrack3d_nuscenes_a1_order_twc.yaml
  seqtrack3d_nuscenes_a2_order_dyn_twc.yaml
  seqtrack3d_nuscenes_a3_order_gate_safe.yaml
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

need_to_do.md                           # 下一步和未来任务
done.md                                 # 已完成工程验收和实验记录
sum_results.md                          # 简洁实验结论
refined_plan.md                         # 研究定位、贡献和论文边界
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

当前正式消融命令以 `need_to_do.md` 为准。除 `--cfg` 和 `--tag` 外，下一批实验统一保持：

```text
--batch_size 16
--epoch 60
--workers 12
--seed 42
--preloading
--check_val_every_n_epoch 5
```

示例：

```bash
CUDA_VISIBLE_DEVICES=0 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python main.py \
  --cfg cfgs/seqtrack3d_nuscenes_a2_order_dyn_cand1.yaml \
  --batch_size 16 \
  --epoch 60 \
  --workers 12 \
  --seed 42 \
  --preloading \
  --check_val_every_n_epoch 5 \
  --tag ct_a2_order_dyn_cand1_car_60ep_bs16
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

关键 nuScenes-mini 对比：

| model | success final | precision final | 说明 |
| --- | ---: | ---: | --- |
| SeqTrack baseline | 50.99 | 59.96 | 原始基线 |
| CT-SeqTrack P5 full | 31.19 | 31.89 | 混入 raw real-time 主干、dynamics、gate，不能单独归因 |
| A1-order | 51.23 | 57.86 | 恢复 order-time 主干后基本修复 A1 崩坏 |
| A2-order-dyn | 50.96 | 63.31 | 当前最强正向信号，precision 高于 baseline |

解释：

- 真实时间方向没有被否定，失败主要来自不合适的注入方式。
- 当前不应继续把 raw / MLP / Fourier real-time token 作为主干主线。
- 后续 TWC 和 gate 都应基于 `A1-order / A2-order-dyn` 继续验证。

简洁实验结论和后续计划见：

```text
sum_results.md
need_to_do.md
```

---

## 论文边界

当前建议不要宣称：

- 完整 Neural ODE / SDE / CDE tracker
- 任意时刻 `state(t*)` 查询
- Mamba / SSM tracker
- 首次解决 sparse / occlusion 3D SOT
- P5 full 已经取得最终正向结果
- CT-SeqTrack full model 已经稳定超过 SeqTrack3D

更稳的贡献表述是：

```text
We convert SeqTrack3D from fixed-step frame sequence learning
to timestamp-native variable-rate state estimation.
```

当前更具体的实验表述是：

```text
Preserving SeqTrack3D's order-time semantics while injecting real delta_t
through a timestamp-conditioned dynamics prior is currently more stable than
directly replacing the main branch time tokens with raw timestamps.
```

---

## Acknowledgement

本项目基于 SeqTrack3D，并沿用 Open3DSOT 风格的 3D SOT 训练与评测框架。感谢 SeqTrack3D、Open3DSOT、PointNet2、DETR 和 attention-is-all-you-need-pytorch 等工作的开源贡献。

---

## License

本项目遵循 `LICENSE` 中的 MIT License。
