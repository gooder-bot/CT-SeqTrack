# CT-SeqTrack

CT-SeqTrack 是一个面向 **timestamp-native / variable-rate 3D 单目标跟踪** 的研究型项目。它基于 SeqTrack3D 改造，目标是把原本固定帧步长的多帧点云序列学习，推进到由真实时间间隔 `delta_t` 驱动的状态估计。

当前仓库是研究快照：真实时间链路、variable-rate 协议、`DynamicsEncoder`、TWC 和 gate 均已落地。2026-07-16 的最新证据包括：corrected-TWC 已完成 seed42 重跑并确认两路 anchor/current XYZ gap 均为 0；A1 上出现 `+1.49 Success / +5.03 Precision` 的单 seed 配置级正信号，A2 上则为负。HTV 六组实验也已完成，旧 feature-concat dynamics 只在温和 random20 上受益，在 gap1124 和 burst-drop 上明显退化。当前最优先工作是固定 manifest 的 residual `true-dt/fixed-dt/shuffled-dt` 因果矩阵、corrected-TWC 的多 seed 复现，以及 TrajTrack 的 GT-free 公平评测，而不是继续叠加 gate。已完成记录见 `done.md`，结果口径见 `sum_results.md`，下一步执行清单见 `need_to_do.md`。

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
CT-SeqTrack studies within-track variable-rate 3D SOT by conditioning
SeqTrack3D on physical elapsed time.
```

也就是说，本项目当前不主打更大的 backbone，也不宣称完整 Neural ODE / SDE / CDE tracker，更不把普通 fixed-step benchmark 的全局涨点作为唯一目标。更稳的论文路线是：先构造 variable-rate / long-gap / sparse 子集，证明 fixed-step 3D SOT 的时间契约存在问题，再用保守的 timestamp-conditioned dynamics prior 给出改进。

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

当前结果提示，直接把 `z_dyn` 拼接进 motion feature 仍可能带来 seed collapse。仓库已实现 `dynamics_motion_mode: residual_limited`：motion head 只读取 observation point feature，真实时间分支输出 `dynamics_displacement_pred`，再经过 norm clamp、近零初始化的小门控、`max_alpha`、全局 scale 和 warmup 后，只修正中心坐标：

```text
obs_motion = motion_mlp(point_feature)
dyn_disp = clamp_norm(dynamics_displacement_pred, max_residual_norm)
final_center = obs_center + residual_scale * alpha_dyn * dyn_disp
```

该实现默认不与旧 `ObservabilityGate` 混用，并输出 alpha、raw/clamped norm、clamp ratio、applied ratio 和 `obs_dyn_center_gap` 等诊断量。工程实现和纯逻辑 smoke test 已完成，真实 nuScenes forward/loss/2-step 仍需在服务器环境验收；目前没有 residual 正向实验结论。

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

共享实现以绝对 `frame_id` 为键，只采样一次 candidate perturbation 和点云 regularization seed，再映射到 A/B 两路；因此共同历史帧和当前帧不仅 crop / 坐标系相同，最终抽取的 XYZ 点也相同。预归一化的 `coordinate_anchor` 会随 batch 输出，TWC loss 对 anchor 和当前帧 sampled XYZ 都做 fail-fast 检查。`twc_candidate_zero_only` 不再把 `num_candidates` 和每 epoch optimizer steps 缩成四分之一。

重要：旧实现分别为 A/B 采样 nonzero candidate offset，而旧检查又比较归一化后恒接近零的 `ref_boxs[:, 0]`，因此会误判“坐标共享”。旧 TWC 数值只保留为历史记录，不能用于证明 TWC 有效或无效。

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

注意：旧版 P5 full 同时混入 raw real-time 主干、dynamics 和 gate，不能作为最终 gate 结论。新的 gate-safe 已经更安全，但 final 仍低于 A2-order-dyn；conf-res 旧汇总里出现过很高 best checkpoint，但最新 best-e14 复测只有 28.06 / 37.70，暂时不能作为稳定收益证据。

---

## 术语速查

| 名称 | 含义 |
| --- | --- |
| `A1-order` | 主干使用 SeqTrack3D order-time，关闭 dynamics / TWC / gate |
| `A2-order-dyn` | 主干使用 order-time，真实 `delta_t/current_delta_t` 进入 `DynamicsEncoder` |
| `A2-residual-dyn` | observation-only motion head + 真实时间驱动的有界小残差；已实现，待服务器验收和实验 |
| `cand1` | `num_candidates=1`，不是 `candidate_id=1` |
| `cand4` | 默认多 candidate，包含 `candidate_id=0/1/2/3` |
| `disp` | 在 dynamics 上增加小权重 displacement 监督 |
| `TWC` | Time-resampling Consistency，不同历史采样路径到同一当前时刻的一致性 |
| `gate-safe` | 更保守的 observation-biased observability gate |
| `conf-res` | confidence residual gate，只在 motion residual 空间做小幅 dynamics 修正 |

---

## 当前进度

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| P0 | 真实时间字段主链路 | 已完成 |
| P1 | 真实时间 baseline smoke test | 已完成 |
| P2 | scalar-preserving `TimeEncoding` | 已完成 |
| P3 | Dynamics / Velocity Branch | feature-concat 与 bounded residual 两种路径均已实现，默认关闭；HTV 六组筛选显示 feature-concat 只在 random20 为正，residual 待真实 batch 验收和因果实验 |
| P4 | Time-resampling Consistency | 共享 candidate perturbation / crop / `coordinate_anchor` 已修复；corrected seed42 已完成，A1 为正、A2 为负，仍需同提交 baseline 与 seed43/44 |
| P5 | Observability Gate | 已实现，默认关闭；gate-safe 低于 A2，conf-res rerun / best 复测都不支持当前接入主线 |
| Evaluation | cand1 / disp / active/corrected TWC / gate / stability / HTV / TrajTrack reference | 最新数据和公平性边界已整理到 `compare_results/reports/` |

当前已完成的关键消融：

```text
1. A2-order-dyn-cand1
2. A2-order-dyn-disp
3. corrected A1-order+TWC seed42（坐标修复后；单 seed 配置级正信号）
4. corrected A2-order-dyn+TWC seed42（坐标修复后；不支持接入 A2 主线）
5. A3-order-gate-safe
6. A3-order-conf-res-gate
7. A3-order-conf-res best-e14 checkpoint retest
8. A2-order-dyn seed43 / seed44
9. A2-order-dyn+TWC w0.01
10. A3-order-conf-res rerun seed42
11. gap1124 / burst-drop / random20 的 A1/A2 六组 HTV 筛选
12. TrajTrack aligned seed42 参考运行与 evaluator oracle 审计
```

当前下一步：

```text
1. 用 TrajTrack 固定 epoch60 checkpoint 跑 `pre_wo_refine()`，并实现不读取当前帧 GT 的 paper-aligned refinement，先建立公平外部参考。
2. 在服务器完成 A2-residual-dyn 的 standard/gap1124/burst-drop 真实 batch、forward/loss 和 2-step 验收。
3. 冻结 virtual-rate manifest，对 residual 做同容量、同 protocol、seed42/43/44 的 true-dt/fixed-dt/shuffled-dt 对照。
4. 补跑 corrected A1+TWC seed43/44，并用同一代码提交重跑配对 A1 baseline；A2+TWC 暂停。
5. 补 candidate、crop-recall、delta_t/sparse/displacement、observation-vs-dynamics proposal 分桶；只有 residual 成立后才扩展 GT-free trajectory-proposal agreement。
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
  seqtrack3d_nuscenes_a2_order_dyn_twc_w001.yaml
  seqtrack3d_nuscenes_a2_residual_dyn.yaml
  seqtrack3d_nuscenes_a2_residual_dyn_vr_*.yaml
  seqtrack3d_nuscenes_a3_order_gate_safe.yaml
  seqtrack3d_nuscenes_a3_order_conf_res_gate.yaml
  seqtrack3d_waymo.yaml                 # Waymo 配置

datasets/
  sampler.py                            # 训练采样、时间字段、TWC paired views
  misc_utils.py                         # 时间戳和历史帧工具

models/
  seqtrack3d.py                         # 主模型、TWC loss、P3/P5 接入
  time_encoding.py                      # raw / mlp / fourier 时间编码
  dynamics.py                           # P3 DynamicsEncoder、residual gate 与 norm clamp
  observability.py                      # P5 ObservabilityGate

utils/
  twc_utils.py                          # 跨 view 共享 candidate offset 与点采样 seed

tools/
  check_time_batch.py
  check_forward_batch.py
  check_train_steps.py
  check_twc_batch.py
  check_twc_shared_coordinates.py       # 无数据集依赖的共享 offset/seed smoke test
  check_residual_dynamics.py            # 无数据集依赖的 residual clamp/gate smoke test
  check_observability_gate.py

compare_results/
  experiment_comparison.md
  metrics_summary.csv
  metrics_points.csv
  twc_gate_ablation_*                   # active TWC / gate-safe / conf-res 汇总
  latest_5runs_*                        # 2026-07-08 五次复核汇总
  related_comparisons_*                 # 按 A1/A2/A3/TWC/seed/180ep 分组的统一对比图表

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
python tools/check_twc_shared_coordinates.py

python tools/check_twc_batch.py \
  --cfg cfgs/seqtrack3d_nuscenes_a1_order_twc.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --version v1.0-mini \
  --split mini_train \
  --batch-size 4 \
  --workers 0 \
  --require-full-history
```

不要传 `--candidate-zero-only`。验收必须覆盖 candidate 1/2/3，并确认 `coordinate_anchor`、共享帧 candidate offset、point-sampling seed / XYZ、search crop 点数和数据集长度都一致。

检查 bounded residual：

```bash
python tools/check_residual_dynamics.py

CUDA_VISIBLE_DEVICES=0 \
python tools/check_forward_batch.py \
  --cfg cfgs/seqtrack3d_nuscenes_a2_residual_dyn_vr_gap1124.yaml \
  --path /home/lishengjie/data/nuscenes-mini \
  --version v1.0-mini \
  --split mini_train \
  --batch-size 1 \
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

当前服务器运行状态和正式复跑命令以 `need_to_do.md` 为准。不要根据旧 PID 或本地文档直接认定历史 HTV 任务仍在运行。

### 服务器训练线程限制（必须）

服务器上的用户进程受到 32 CPU 核的 cgroup 配额限制。使用多个
DataLoader worker 时，NumPy、SciPy、OpenCV 和 PyTorch 可能分别创建
OpenBLAS/OpenMP 线程池，造成严重的线程过量、CPU throttling 和 GPU 空转。
2026-07-14 的 HTV 训练排障中，补回以下限制后 epoch 时长恢复正常。

所有服务器正式训练命令都必须在启动 Python 前设置：

```bash
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
python -u main.py ...
```

后台训练同样必须把这四个变量写入每一条 `nohup` 命令。它们只限制底层
数学库的内部并行线程，不改变模型、数据、batch size、DataLoader worker
数量或其他训练参数。

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
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python -u main.py \
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
| A2-order-dyn | 50.96 | 63.31 | 60ep seed42 最强正向信号；最新 seed43/44 显示稳定性不足 |
| corrected A1-order+TWC seed42 | 52.72 | 62.89 | 相对旧配置对齐 A1 为 +1.49/+5.03；只有单 seed，baseline 还不是同提交因果配对 |
| corrected A2-order-dyn+TWC seed42 | 50.04 | 61.25 | 相对 A2 为 -0.93/-2.07；暂不接入 dynamics 主线 |
| A1-order+TWC（旧） | 51.16 | 61.10 | nonzero candidate 下两路坐标系不共享；数值保留，TWC 归因撤回 |
| A2-order-dyn+TWC（旧） | 28.23 | 32.04 | 同样受坐标污染；不能据此判断 TWC 与 dynamics 是否冲突 |
| A3-order-gate-safe | 48.32 | 54.87 | 比旧 P5 full 安全，但仍低于 A2-order-dyn |
| A3-order-conf-res-gate | 31.17 | 30.92 | 旧 best 很高但最新 best-e14 复测只有 28.06 / 37.70，暂不能作为收益证据 |

最新五次复核：

| model | success final | precision final | 说明 |
| --- | ---: | ---: | --- |
| A3-conf-res best-e14 retest | 28.06 | 37.70 | 单 checkpoint 测试，未复现旧 62.04 / 76.30 best 信号 |
| A2-order-dyn seed43 | 23.64 | 23.77 | seed 崩坏，说明 A2 稳定性风险很大 |
| A2-order-dyn seed44 | 46.90 | 52.62 | 明显好于 seed43，但仍低于旧 seed42 60ep 汇总 |
| A2-order-dyn+TWC w0.01 seed42（旧） | 22.88 | 24.27 | 同样受 nonzero candidate 坐标污染，仅保留历史数值 |
| A3-conf-res rerun seed42 | 32.11 | 31.87 | rerun 仍低，gate/conf-res 应转诊断而不是继续堆结构 |

最新 variable-rate 筛选（`A2-order-dyn - A1-order`，seed42）：

| protocol | Success final delta | Precision final delta | 判断 |
| --- | ---: | ---: | --- |
| gap1124 | -4.01 | -9.55 | 早期高点后明显回落，不是稳定收益 |
| burst-drop | -7.45 | -14.40 | 强不规则间隔下明显退化 |
| random20 | +9.09 | +14.23 | 温和随机丢帧下形成一致正信号 |

TrajTrack 的 aligned seed42 运行得到 64.94 / 79.07，但本地 evaluator 使用当前帧 GT overlap 触发 refinement，并用 GT overlap 选择 proposal。这一结果只作为 oracle-assisted 实现诊断，不能与 GT-free SeqTrack3D 或 CT-SeqTrack 做公平在线排名；详见 `compare_results/reports/trajtrack_gt_assisted_vs_plain_seqtrack_reference.md`。

解释：

- 真实时间方向没有被否定，失败主要来自不合适的注入方式。
- 当前不应继续把 raw / MLP / Fourier real-time token 作为主干主线。
- 在普通 fixed-step 设置上追求全局稳定涨点的把握不高；更合理的主战场是 variable-rate、long-gap、sparse / re-appearance 子集。
- `A2-order-dyn` 的真实时间信号具有 protocol dependence：温和 random20 为正，强 gap/burst 为负；新 residual 只有工程结果，尚无性能结果。
- corrected-TWC 已消除坐标/crop 污染，A1 seed42 为正，但只有单 seed且 baseline 不是同代码提交；它目前是候选稳定性贡献，不是已确认主增益。
- TrajTrack 的可借鉴点是 bbox-only trajectory proposal 与 local/global proposal agreement，不是当前 GT-assisted evaluator 的高分。

简洁实验结论和后续计划见：

```text
sum_results.md
need_to_do.md
```

按关联实验分组的统一图表见：

```text
compare_results/reports/related_comparisons.md
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
- 标准 fixed-step benchmark 上已经稳定全面涨点
- TrajTrack 当前本地结果相对 SeqTrack3D 提升了 13.96 / 19.11（该差值包含 GT oracle 信息）

更稳的贡献表述是：

```text
We study within-track variable-rate LiDAR 3D SOT and condition a
Seq2Seq tracker on physical timestamps.
```

当前更具体的实验表述是：

```text
Preserving SeqTrack3D's order-time semantics while injecting real delta_t
through a timestamp-conditioned dynamics prior is currently more stable than
directly replacing the main branch time tokens with raw timestamps.
```

当前更稳的投稿叙事是：

```text
Fixed-step 3D SOT hides the physical meaning of irregular frame intervals.
CT-SeqTrack studies this problem with within-track variable-rate evaluation
and a bounded timestamp-conditioned dynamics residual. Whether it improves
long-gap and sparse tracking remains an experimental hypothesis.
```

---

## Acknowledgement

本项目基于 SeqTrack3D，并沿用 Open3DSOT 风格的 3D SOT 训练与评测框架。感谢 SeqTrack3D、Open3DSOT、PointNet2、DETR 和 attention-is-all-you-need-pytorch 等工作的开源贡献。

---

## License

本项目遵循 `LICENSE` 中的 MIT License。
