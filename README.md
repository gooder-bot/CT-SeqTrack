# CT-SeqTrack

CT-SeqTrack 是一个面向 **timestamp-native / variable-rate 3D 单目标跟踪** 的研究型项目。它基于 SeqTrack3D 改造，目标是把原本固定帧步长的多帧点云序列学习，推进到由真实时间间隔 `delta_t` 驱动的状态估计。

当前仓库是研究快照：旧 reliability、feature-concat true-dt 与 TWC 主方法 promotion 均已 No-Go，项目仍处于 **M0 收口 + M2 formal 运行**。M0-3 gap1124 proposal oracle 得到 **`GO_M2_PROPOSAL_INNOVATION`**，M0-4 得到 **`FREEZE_M1_SHARED_SE2`**。2026-07-22，M1/M2 在 clean commit `9a0b26d` 上完成 E0–E5；commit `473738f` 已完成服务器 cadence/shuffled manifests 与 E6 preflight，并启动唯一 A1-init seed42 true-dt formal 训练。服务器另有 M2 scratch 与 matched W0 scratch 两个初始化消融由用户报告运行中，结果和精确 provenance 尚待拉回。正式状态为 **`M2 formal/scratch controls RUNNING; tracking and causal-time result pending`**，仍不能宣称 tracking 涨点、正确 `delta_t` 有效或跨 cadence 泛化。M0-2 A/B/C 四协议/path-variance 尚未完成，M0 整体仍为进行中。

## 文档导航

| 文件 | 作用 |
| --- | --- |
| `README.md` | 项目入口、当前主线、环境和命令索引 |
| `refined_plan.md` | 研究定位、论文边界、贡献叙事和 related work 边界 |
| `sum_results.md` | 按时间顺序总结已有实验说明了什么 |
| `need_to_do.md` | 当前和未来任务，只放还没有完成的事情 |
| `done.md` | 已完成工程验收、历史实验和关键输出归档 |
| `compare_results/` | 完整指标表、曲线和实验结果文件 |
| `compare_results/reports/paper_viability_and_execution_20260720.md` | P0 后的论文可行性、claim 审计、方法/benchmark 分叉与停止条件 |
| `compare_results/reports/twc_abc_seed42_comparison_20260721.md` | 同提交 TWC A/B/C 的数据审计、效应分解、图表与最终判定 |
| `compare_results/reports/dual_clock_state_filtering_proposal_20260721.md` | 新候选贡献、dual-clock/innovation/asymmetric distillation，以及 M4 persistent filter/trajectory tube 的完整方法规格与 Go/No-Go |
| `compare_results/reports/m0_m03_m04_analysis_20260721.md` | M0-3/M0-4 数据质量、独立复算、稳健性、决策与下一步 |
| `compare_results/reports/m0_m03_m04_report/report.html` | M0-3/M0-4 自包含可视化技术报告（桌面/窄屏 QA 通过） |
| `compare_results/reports/m1_m2_e0_e5_validation_20260722.md` | M1/M2 服务器 E0–E5 provenance、JSONL 独立复算、解锁边界与 E6 阻塞 |
| `compare_results/reports/m2_e6_parameter_freeze_20260722.md` | 既有 mini_train M0-3 向量的单规则复算、alpha/R/warmup 冻结与保守性边界 |
| `compare_results/reports/htv_identifiability_and_execution_plan_20260722.md` | 标准 `delta_t` 可辨识性、HTV/丢帧论文边界、正式协议、三任务登记与结果分叉 |

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

也就是说，本项目当前不主打更大的 backbone，也不宣称完整 Neural ODE / SDE / CDE tracker，更不把普通 fixed-step benchmark 的全局涨点作为唯一目标。更稳的论文路线是：先构造 variable-rate / long-gap / sparse 协议并审计 fixed-step 3D SOT 的 rate-robustness；方法贡献优先筛选 endpoint history-resampling consistency。只有显式 `delta_t` 分支在 `true/fixed/shuffled` 中形成因果正信号，才恢复 timestamp-conditioned dynamics 的方法主张。

---

## 最新候选贡献框架（未验证）

P0-B4、P0-C 和同提交 TWC A/B/C 的 No-Go 不变。若继续做方法论文，新的贡献层级为：

1. **Matched variable-rate protocol**：within-track irregular cadence、matched endpoint 和 `true/fixed/shuffled` 时间因果控制。
2. **Dual-clock continuous-discrete state update**：保留 SeqTrack3D order clock，真实时间只进入 zero-init adapter、显式 `F(delta_t)` 状态传播和 search support。
3. **Endpoint-consistent asymmetric path distillation**：canonical EMA teacher 监督 irregular true-time student，第一轮不再给困难 B view 等权 supervised loss。

方法主线：

```text
physical-consistent candidate augmentation
    -> zero-init dual-clock adapter
    -> proposal innovation
    -> asymmetric endpoint path distillation
    -> optional calibrated state filter / trajectory tube
```

当前 `d_obs + alpha*d_dyn` 的完整位移相加只保留为历史实现；正式候选必须改为：

```text
innovation = clip_norm(d_dyn - stopgrad(d_obs), R(delta_t))
d_final    = d_obs + alpha * innovation
```

旧 hand-crafted observability Gate 已停止，不再列贡献。只有 state prior、time controls、M2 predicted-history tube oracle 和 uncertainty calibration 都通过后，才允许研究 covariance-derived Kalman gain。M4 固定按 `tube oracle -> fixed-Q/R filter -> filter+tube -> learned Q/R` 逐级推进，不允许一次同时引入 filter、tube 和 learned covariance。完整定义见 `compare_results/reports/dual_clock_state_filtering_proposal_20260721.md`。

---

## 已实现地基与历史候选

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

注意：当前实验不支持继续把真实秒数直接塞进主干时间 token；P0-C 也不支持当前 feature-concat dynamics 的 promotion。主干保持 SeqTrack3D 的 order-time 语义是稳定基线，不是 physical-time 收益证据。任何新的显式时间机制都必须重新通过 `true/fixed/shuffled-dt`。

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

该实现默认不与旧 `ObservabilityGate` 混用，并输出 alpha、raw/clamped norm、clamp ratio、applied ratio 和 `obs_dyn_center_gap` 等诊断量。standard 真实 batch 的 warmup 与 active forward/loss/backward 已完成且数值有限，但默认 gate alpha 约为 `2e-5`，实际修正 P50 仅 `7.25e-8 m`；它尚未完成真正的 2-step optimizer、完整 split、强 gap 或跟踪性能验证。当前实现是安全但功能上近乎关闭的消融，不是可直接进入正式训练的主配置。

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
| `A2-residual-dyn` | observation-only motion head + 真实时间驱动的有界小残差；standard 真实 batch 数值通过，但默认修正约 `1e-7 m`，未通过功能验收 |
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
| P3 | Dynamics / Velocity Branch | feature-concat 与 bounded residual 两种路径均已实现，默认关闭；feature-concat 在强 gap/burst 下退化，residual 近乎为零；P0-B4 已停止 pre-crop reliability anchor，当前只保留一次 reachable-subset kill-test |
| P4 | Time-resampling Consistency | 同提交 A/B/C seed42 已完成：`C-B` 为正但 `C-A` 明显为负，只恢复 paired-view 损失约一半；主方法 promotion No-Go，不补 seed43/44 |
| P5 | Observability Gate | 已实现，默认关闭；gate-safe 低于 A2，conf-res rerun / best 复测都不支持当前接入主线 |
| Evaluation | cand1 / disp / active/corrected TWC / gate / stability / HTV / TrajTrack reference | 最新数据和公平性边界已整理到 `compare_results/reports/` |
| P0-B3 | 测试时可靠性与 passive dual-forward | 开发集 observation proxy 为正，但 raw-CV anchor 与 selector No-Go；后续 P0-B4 未独立复现 |
| P0-B4 | independent observation reliability | mini_val 冻结验证 No-Go；当前 calibrator 与 dual-anchor 停止，详见 `p0b4_observation_reliability_validation_20260720.md` |
| P0-C | frozen cadence / effective-time controls | manifest/invariance PASS；同 checkpoint true-dt 未超过 fixed/shuffled，promotion No-Go，详见 `p0c_frozen_protocol_validation_20260720.md` |
| M0-3 | crop-reachable proposal oracle | `GO_M2_PROPOSAL_INNOVATION`；dynamics-only 与 long-gap tracklet bootstrap 均支持互补性，M2 工程 gate 解锁 |
| M0-4 | candidate dynamics audit | `FREEZE_M1_SHARED_SE2`；独立 candidate offset 制造强伪导数，M1 第一版排除 smooth drift |
| M0 整体 | 冻结输出、oracle 与 candidate 审计 | **进行中**；P0-C-D1/M0-3/M0-4 已完成，只剩 M0-2 四协议输出/path variance 与 provenance 收口 |
| M1 engineering | shared world-SE(2)、canonical label、zero-init dual-clock adapter | E0–E6 通过；共享 5-epoch warmup 与唯一 formal 配置已冻结并用于 R1 |
| M2 engineering | bounded proposal innovation | E0–E6 通过；R1 A1-init formal 与 R2/R3 scratch 配对运行中，tracking/time-control 结果待定 |

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
13. P0-B standard/gap1124/burst-drop crop reachability 与 P0-A standard residual 量级诊断
14. P0-B2 A1 recursive predicted-history 三协议诊断与预注册 No-Go 判断
15. P0-B3 standard/gap1124/burst-drop full passive diagnostic、grouped reliability 复算与 feature ablation
16. P0-B4 10-tracklet smoke、完整 mini_val 冻结验证、原始 CSV/哈希与本地独立复算
17. P0-C gap1124 stable-token val/test manifest、offline shuffled-dt mapping 与真实 batch invariance
18. P0-C standard-trained A2 frozen `true/fixed/shuffled-dt` 三路性能与 provenance 复核
19. 同提交 TWC A/B/C seed42：provenance、12 个评测点、75720 步诊断和 final checkpoint hash 本地复核
20. M0-3 gap1124 proposal oracle：official checks、原始向量复算、dynamics-only/trimmed/long-gap/tracklet bootstrap 稳健性
21. M0-4 candidate dynamics audit：伪速度/伪加速度、matched proposal penalty、candidate balance 与 shared SE(2) 冻结决策
22. M1/M2 E0–E5：clean commit 服务器硬门禁、五组 JSONL 独立复算、warmup/invalid/empty/resampled/三协议/2-step/bound 验收
23. M2 E6 静态冻结：1311 endpoints / 213 tracklets 的单规则复算、tracklet bootstrap、唯一 formal true 配置与 fail-closed server workflow
```

当前下一步：

```text
1. 不再启动新训练或修改 alpha/R；等待 R1 A1-init M2 formal、R2 M2 scratch、R3 matched W0 scratch 三个任务完成。
2. 分别核对 epoch60/global_step、last.ckpt、退出码、resolved config、训练步数、SHA256 与 provenance；不完整任务不进入比较。
3. 先做 R1/R2/R3 standard final；用 R2-R3 隔离 scratch 条件下 M2 的结构净效应，用 R1-A1 检查 continuation 收益。
4. 对 R1 final checkpoint 运行 `tools/run_m2_formal_time_controls_gpu3.sh`；导出 standard/gap/burst 的 true/fixed/shuffled 与 matched A1。fixed/shuffled 不训练。
5. 做 per-tracklet paired/bootstrap、delta_t/位移/稀疏分桶、首次失控和连续失败；在看到结果前冻结 standard non-inferiority margin。
6. 只有 strong cadence、true controls、standard guardrail 与 matched scratch baseline 同时支持方法，才补 seed43/44、full nuScenes 和第二数据集；否则转 benchmark/diagnosis。
7. M3/M4 继续锁定；不依据 mini_val/test 反调 alpha/R，不用扩大 crop 或新增 Gate 掩盖失败。
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
  diagnose_crop_reachability.py         # previous-GT / expanded / GT-history-CV oracle
  diagnose_recursive_crop_reachability.py # baseline A1 预测历史的 GT-free 被动可达性诊断

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

### M2 E6 唯一正式工作流

当前 formal 入口是 fail-closed 的三步流程。2026-07-22，commit `473738f` 已完成前两步并启动 R1，输出根为 `output/m2_formal_true_seed42_473738f_20260722_112536`；以下命令保留作复现合同，不应在当前任务未结束时重复启动。先确认服务器位于已评审的新 clean commit，并设置：

```bash
PROJECT_ROOT=/home/lishengjie/study/lcyu/CT-SeqTrack
cd "$PROJECT_ROOT"

export EXPECTED_GIT_COMMIT="$(git rev-parse HEAD)"
export DATA_ROOT=/home/lishengjie/data/nuscenes-mini
export A1_CKPT="$PROJECT_ROOT/output/20260531-2322-seqtrack3d_nuscenes_a1_order-ct_a1_order_car_60ep_bs16_gpu1/lightning_logs/version_0/checkpoints/last.ckpt"
```

第一步只使用 CPU 生成并检查绑定当前 commit 的 cadence/shuffled manifests：

```bash
bash tools/prepare_m2_formal_manifests.sh
```

第二步只在 GPU2 启动一个 seed42 true-dt 训练。脚本会再次核对 clean Git、A1 SHA256、oracle/report/config hash、`1262×60=75720` steps、manifest 完整性和 `--init_checkpoint` 语义：

```bash
export GPU=2
export OUT_ROOT="$PROJECT_ROOT/output/m2_formal_true_seed42_${EXPECTED_GIT_COMMIT:0:7}_$(date +%Y%m%d_%H%M%S)"

nohup bash tools/run_m2_formal_seed42_gpu2.sh \
  > "/tmp/m2_formal_train_${EXPECTED_GIT_COMMIT:0:7}.launcher.log" 2>&1 &
echo $!
```

第三步在训练成功并确认 `final_checkpoint.json` 后，只在 GPU3 对同一个 `last.ckpt` 做 true/fixed/shuffled，同时导出相同 endpoint 的 A1 baseline。fixed/shuffled 不训练：

```bash
export FINAL_CKPT="$(find "$OUT_ROOT" -type f -name last.ckpt -print -quit)"
export EXPECTED_FINAL_CKPT_SHA256="$(sha256sum "$FINAL_CKPT" | awk '{print $1}')"
export GPU=3
export OUT_ROOT="$PROJECT_ROOT/output/m2_formal_controls_${EXPECTED_GIT_COMMIT:0:7}_$(date +%Y%m%d_%H%M%S)"

nohup bash tools/run_m2_formal_time_controls_gpu3.sh \
  > "/tmp/m2_formal_controls_${EXPECTED_GIT_COMMIT:0:7}.launcher.log" 2>&1 &
echo $!
```

不要手工绕过脚本的 commit/hash/step 检查，不要用 `--checkpoint` 代替训练初始化的 `--init_checkpoint`，也不要从 top-k/best 文件挑结果。

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

同提交 TWC A/B/C seed42（standard mini_val，final）：

| run | Success | Precision | 相对效应 |
| --- | ---: | ---: | --- |
| A: single view | 50.01 | 58.20 | 对照基线 |
| B: paired views, weight0 | 34.71 | 34.02 | `B-A=-15.30/-24.18`，paired-view 路径明显有害 |
| C: paired views + corrected-TWC | 43.01 | 45.76 | `C-B=+8.31/+11.74`，但 `C-A=-7.00/-12.44` |

完整曲线、Late mean、训练诊断和 provenance 见 `compare_results/reports/twc_abc_seed42_comparison_20260721.md`。

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

最新 P0-B3 reliability 诊断（standard-only grouped calibrator，固定评估强协议）：

| diagnostic | standard | gap1124 | burst-drop | 判断 |
| --- | ---: | ---: | ---: | --- |
| all-13 trigger AUROC / AUPRC | 0.857 / 0.742 | 0.787 / 0.660 | 0.785 / 0.671 | 通过原预注册 reliability 门槛，但不是时间因果证据 |
| previous-observation-only AUROC / AUPRC | 0.853 / 0.728 | 0.867 / 0.778 | 0.873 / 0.789 | 更强且校准更稳，说明可靠性主要来自上一观测质量 |
| raw-CV dual-oracle recall gain | +3.04 pp | +2.88 pp | +3.15 pp | 强协议低于 +5 pp，raw-CV anchor No-Go |
| post-crop selector AUROC | 0.729 | 0.605 | 0.433 | 不能跨协议用于 candidate 选择 |

完整数据质量核查、复算和 feature ablation 见 `compare_results/reports/p0b3_reliability_validation_20260720.md`。raw `current_delta_t` 在 standard-only calibrator 中会造成 gap/burst 过度触发；删除它后强协议 AUROC 恢复到 0.865/0.872。因此当前只能说 observation reliability 可预测 crop miss，不能说 physical timestamp 已提高 reliability prediction。

P0-B4 随后在 disjoint mini_val 上冻结验证精简 `observation_v1`：

| metric | standard | gap1124 | burst-drop | 判定 |
| --- | ---: | ---: | ---: | --- |
| AUROC | 0.794 | 0.680 | 0.712 | 强协议均低于 0.75 |
| AUPRC-prevalence | 0.414 | 0.282 | 0.328 | 通过 |
| operating recall | 0.711 | 0.568 | 0.609 | 强协议均低于 0.70 |
| raw-CV union gain | +0.06 pp | +0.00 pp | +0.00 pp | 强协议没有互补 endpoint |

最终为 `NO_GO_OBSERVATION_RELIABILITY_VALIDATION`。这说明 P0-B3 的 observation-only 正信号不能升级为独立验证的方法贡献；当前 calibrator、raw-CV candidate 和 selector 均停止。完整复核见 `compare_results/reports/p0b4_observation_reliability_validation_20260720.md`。

TrajTrack 的 aligned seed42 运行得到 64.94 / 79.07，但本地 evaluator 使用当前帧 GT overlap 触发 refinement，并用 GT overlap 选择 proposal。这一结果只作为 oracle-assisted 实现诊断，不能与 GT-free SeqTrack3D 或 CT-SeqTrack 做公平在线排名；详见 `compare_results/reports/trajtrack_gt_assisted_vs_plain_seqtrack_reference.md`。

解释：

- 真实时间方向没有被单个实验普遍否定，但当前 feature-concat A2 已在公平 time-control 中 No-Go，不能恢复方法主张。
- 当前不应继续把 raw / MLP / Fourier real-time token 作为主干主线。
- 在普通 fixed-step 设置上追求全局稳定涨点的把握不高；更合理的主战场是 variable-rate、long-gap、sparse / re-appearance 子集。
- `A2-order-dyn` 的真实时间信号具有 protocol dependence；P0-C 中 shuffled 与 true 几乎持平且 Success 略高，说明正确时间对应关系没有稳定收益。新 residual 已通过 standard 真实 batch 数值检查，但默认修正近乎为零，尚无性能结果。
- corrected-TWC 已消除坐标/crop 污染；新的同提交 A/B/C 证明 `C-B` 净效应为正，但 paired-view 自身退化过大，C 仍显著低于 A。因此它只保留为“部分修复 paired-view 路径”的机制结果，不再作为待多 seed 的主增益候选。
- TrajTrack 的可借鉴点是 bbox-only trajectory proposal 与 local/global proposal agreement，不是当前 GT-assisted evaluator 的高分。
- standard/gap/burst 的递归和 P0-B3/P0-B4 已证明：raw predicted-history CV 与 observation anchor 的失败高度重叠，mini_val 强协议甚至没有 trajectory-only endpoint；它不进入 active tracker。
- P0-B3 的 observation-quality 正信号在 P0-B4 独立 split 上未通过排序和固定阈值 recall 门槛；当前 reliability-controlled state anchor 在实现前停止，不能通过重调 mini_val 继续推进。
- 研究重心改为冻结 variable-rate protocol 与可解释 failure diagnosis；当前 A2 未通过 `true/fixed/shuffled-dt`，只能在新的预注册机制通过同类负对照后恢复方法主张。

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
- P0-B3 已证明 timestamp-aware reliability 或 active dual-anchor 有效
- raw predicted-history CV 可以作为正式第二搜索锚点

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
CT-SeqTrack studies this problem with within-track variable-rate evaluation,
timestamp-conditioned trajectory guidance before search cropping, and bounded
observation refinement. GT-free long-gap gains remain an experimental hypothesis.
```

---

## Acknowledgement

本项目基于 SeqTrack3D，并沿用 Open3DSOT 风格的 3D SOT 训练与评测框架。感谢 SeqTrack3D、Open3DSOT、PointNet2、DETR 和 attention-is-all-you-need-pytorch 等工作的开源贡献。

---

## License

本项目遵循 `LICENSE` 中的 MIT License。
