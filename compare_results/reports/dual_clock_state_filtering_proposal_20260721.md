# CT-SeqTrack 新贡献框架与方法改造规格

更新时间：2026-07-21

状态：**候选方法，M 阶段已启动，当前处于 M0。尚未完成性能验证。** 本文定义下一版可以进入论文的方法叙事、实现顺序、因果验收和停止条件；它不覆盖已经完成的 No-Go 结论，也不能被引用为实验结果。当前允许 M1 工程准备，但 M1 正式训练和 M2–M4 必须按本文门槛逐级解锁。

## 1. 决策摘要

当前不能把 `Dynamics / TWC / Gate / HTV` 写成四个平行贡献：

- 当前 feature-concat Dynamics 没有通过 `true/fixed/shuffled-dt`；
- corrected-TWC 虽有 `C-B=+8.31/+11.74`，但 `C-A=-7.00/-12.44`；
- observation Gate 在 disjoint mini_val 上没有通过 AUROC/recall 门槛；
- HTV 已形成可复现协议资产，但只有 mini、少模型和单 seed 证据。

新的论文层级固定为：

```text
研究问题：within-track variable-rate 3D SOT
    |
    +-- 贡献 1：matched variable-rate protocol 与真实时间因果控制
    +-- 贡献 2：dual-clock continuous-discrete state update
    +-- 贡献 3：endpoint-consistent asymmetric path distillation

HTV  = 贡献 1 的评价协议，不是独立网络模块
Dyn  = 贡献 2 的状态传播部分，不再是 feature concat
TWC  = 贡献 3 的训练目标，不再与 paired supervised loss 等权
Gate = 不列贡献；当前版本停止，必要时由校准协方差/Kalman gain 替代
```

一句话方法叙事：

> CT-SeqTrack preserves the discrete order clock of SeqTrack3D while using physical elapsed time to propagate a continuous motion state, shape the search support, and distill predictions across different history-sampling paths to the same endpoint.

## 2. 新的三个贡献说明

### 贡献 1：Matched variable-rate protocol and physical-time controls

把“真实 timestamp 已接入数据”升级为可证伪的实验贡献：

1. 在同一 tracklet 内构造 irregular gap、burst drop、random drop 和 held-out schedule，而不是只给整条序列设置固定 frame interval。
2. 冻结 endpoint identity、当前点云、局部坐标、candidate offset、点采样 seed 和 checkpoint。
3. 仅干预模型读取的有效时间，形成 `true / fixed / shuffled` 三路 matched control。
4. 一个 checkpoint 直接跨 standard、seen cadence 和 unseen cadence 测试；不允许按协议分别重训后称为泛化。
5. 除 Success/Precision 外，保存 per-tracklet/endpoint 输出，报告 gap/displacement/sparsity bins、首次失控、连续失败、crop recall 和历史路径方差。

这个贡献的可防御边界是：

```text
within-track irregular cadence
+ matched physical-time intervention
+ single-model rate generalization
```

不能写成“首次研究 HTV”。HVTrack 已通过不同固定 frame interval 构造 KITTI-HV；这里的区别必须是轨迹内部不规则性、matched endpoint 和时间字段的因果干预。

### 贡献 2：Dual-clock continuous-discrete state update

SeqTrack3D 原有的 order-time 是一个有效的离散序列先验，不能被 raw seconds 直接替换。下一版使用两个职责分离的时钟：

- **order clock**：保留历史顺序、token identity 和原主干优化语义；
- **physical clock**：只控制状态传播、relative-time adapter 和搜索支持范围。

建议的零初始化 adapter：

```text
phi_dt_i = TimeEncode(log(1 + delta_t_i / dt_ref))
h_i'     = h_i + A_zero(h_i, phi_dt_i)
```

`A_zero` 的输出层权重和 bias 初始化为 0，使新模型在 step 0 严格退化为 A1，而不是依赖训练重新找回 SeqTrack3D 的顺序语义。

物理状态可先从轻量形式开始：

```text
s_t = [x, y, z, vx, vy, vz, yaw, yaw_rate]
s_prior(t + dt) = F(dt) s_post(t) + g_theta(history, dt)
```

第一版不需要 Neural ODE。`F(dt)` 使用显式 constant-velocity/constant-turn transition，`g_theta` 只学习有界 acceleration/turn residual。只有该形式通过后，才考虑更复杂的 continuous-time attention 或状态空间模型。

#### Proposal innovation，禁止完整位移相加

当前 observation branch 和 DynamicsEncoder 都在预测从上一框到当前框的完整 displacement，因此：

```text
d_final = d_obs + alpha * d_dyn
```

存在 double counting。新定义固定为：

```text
innovation = d_dyn - stopgrad(d_obs)
innovation = clip_norm(innovation, R(dt))
d_final    = d_obs + alpha * innovation
```

这等价于在两个完整 proposal 之间做有界移动。`alpha=0` 时严格恢复 A1；`alpha=1` 时到达 dynamics proposal。训练前必须在 crop-reachable endpoint 上计算线段 `d_obs -> d_dyn` 的 oracle 最优点，确认插值确有上限。

#### 时间驱动的 search support

当前强 gap 的主要瓶颈发生在目标进入 search crop 之前。若状态 prior 的 predicted-history oracle 有空间，使用均值和不确定性构造 trajectory tube：

```text
center    = propagated_state_mean(dt)
length    = base_length + k_long * |v| * dt
width     = base_width  + k_unc  * sqrt(lambda_max(P(dt)))
search    = baseline_crop union trajectory_tube
point cap = fixed
```

tube 沿预测运动方向增长，横向保持窄，并固定最终点预算。它与无条件扩大 2x crop 的区别是：额外计算量、背景点量和物理时间作用都可以单独量化。

#### 训练增强必须保持物理一致

当前 `datasets/sampler.py` 为每个历史框独立采样 offset；这会把不连续随机误差解释为速度/加速度。下一版必须在以下两种方案中二选一：

1. 对整条历史轨迹施加一个共享 SE(2) 刚体扰动；
2. 从 A1 递归误差拟合时间相关的平滑 drift process。

Dynamics 的速度/加速度监督从 canonical trajectory 或物理一致扰动后的 trajectory 计算；禁止由逐帧独立 jitter 直接构造导数。

### 贡献 3：Endpoint-consistent asymmetric path distillation

TWC 的不变量继续保留：两个 view 必须共享同一绝对 endpoint、最近历史 anchor、current crop、局部坐标、current points 和共同帧的 candidate perturbation；只改变更早的合法历史采样路径。

当前对称目标：

```text
L = 0.5 * (L_sup_A + L_sup_B) + lambda_twc * D(p_A, p_B)
```

让困难 sparse view B 获得与 canonical view A 相同的 supervised 权重。A/B/C 已显示 paired-view 路径本身造成巨大退化。新目标改为非对称蒸馏：

```text
teacher = EMA(model)
p_A     = teacher(canonical_dense_path)
p_B     = model(irregular_true_time_path)

L = L_sup_A
  + beta * L_sup_B
  + lambda_path * w_A * D(stopgrad(p_A), p_B)
```

其中：

- 第一轮 `beta=0`，确认 B 不再以独立监督改变主任务分布；
- `w_A` 只使用 teacher 的推理时可得 confidence/uncertainty，不读取当前 GT；
- teacher 使用 EMA，避免两个共享权重的在线分支互相追逐；
- `D` 先只约束 center 和 yaw；机制成立后才扩展 feature/state consistency；
- `fixed/shuffled` 不进入 consistency 训练，只用于同 checkpoint 因果评估。

这个贡献只能称为：

```text
endpoint-conditioned history-resampling distillation
```

不能泛称“首次 temporal consistency”，因为 ChronoTrack 已经使用 temporal consistency 与 memory cycle consistency。

### 可直接用于论文的贡献表述草稿

中文版本：

1. 我们建立了面向 LiDAR 3D SOT 的轨迹内变采样率评测与因果控制框架，在固定 endpoint、几何输入和 checkpoint 的条件下，仅干预物理时间对应关系，从而区分 cadence robustness、history-path sensitivity 与真实 timestamp 的净作用。
2. 我们提出双时钟连续—离散状态更新：离散 order clock 保留 SeqTrack3D 的序列先验，physical clock 控制显式状态传播、proposal innovation 与搜索支持；零初始化设计保证模型能够严格退化为原始 observation baseline。
3. 我们提出同终点非对称历史路径蒸馏，以 canonical dense-path EMA teacher 监督 irregular true-time student，在不改变当前观测和坐标系的前提下提高对历史重采样路径的鲁棒性。

英文版本：

```text
1. We establish a matched within-track variable-rate evaluation framework for
   LiDAR 3D SOT, isolating the causal effect of physical timestamps by holding
   endpoints, geometric inputs, and checkpoints fixed under true, fixed, and
   shuffled time interventions.

2. We propose a dual-clock continuous-discrete state update that preserves the
   discrete order prior of a Seq2Seq tracker while using physical elapsed time
   for state propagation, bounded proposal innovation, and search support. Its
   zero-initialized design exactly recovers the observation baseline.

3. We introduce endpoint-consistent asymmetric path distillation, where an EMA
   teacher on a canonical dense history supervises a student on an irregular
   true-time history without changing the current observation or coordinate
   system.
```

这三条只有在对应实验门槛通过后才能从“propose/establish”升级为摘要中的有效贡献；当前内部文档仍须标注为 candidate。

## 3. Gate 的新定位

当前 hand-crafted observability Gate 已得到 `NO_GO_OBSERVATION_RELIABILITY_VALIDATION`，不复活、不在 mini_val 上重调，也不列论文贡献。

若 dual-clock state prior 与 trajectory tube 已独立通过，融合只允许使用连续、可校准的精度权重：

```text
R_obs  = observation covariance
P_dyn  = propagated dynamics covariance
K      = P_dyn (P_dyn + R_obs)^(-1)
state  = prior + K * innovation
```

它与旧 Gate 的区别是：

- 不做二元“信 observation / dynamics”选择；
- 权重来自显式 uncertainty，随 `dt` 传播；
- 必须先验证 NLL、coverage、ECE/可靠性图，再进入 tracking 闭环；
- 如果校准失败，则退回固定小 `alpha` innovation，不增加新的 learned gate。

## 4. 代码落点

| 文件 | 当前问题 | 候选改造 |
| --- | --- | --- |
| `datasets/sampler.py` | 每个历史框独立 offset，可能制造伪速度 | shared SE(2) 或平滑 drift；canonical dynamics label |
| `models/dynamics.py` | 由 pooled transition feature 直接输出 velocity，再乘当前 gap | 显式 `F(dt)` 状态传播 + 有界 acceleration residual；保留轻量实现 |
| `models/seqtrack3d.py` | 完整 `d_dyn` 加到完整 `d_obs` | proposal innovation；zero-init alpha；oracle 先行 |
| `models/seqtrack3d.py` TWC | `0.5(L_A+L_B)` 对称监督 | EMA canonical teacher -> irregular student |
| `models/observability.py` | 独立 split No-Go | 停止；仅在先验成立后研究 covariance head |
| dataset/protocol | nuScenes 已有 matched time controls，Waymo 未完全接入 | 统一 manifest、endpoint logger 和 held-out schedule |

## 5. 最小实现顺序

阶段决策固定为：

| 阶段 | 当前状态 | 解锁边界 |
| --- | --- | --- |
| M0 | **进行中** | 只做冻结输出、离线 oracle、candidate 审计和 provenance 收口 |
| M1 | **shared SE(2) 已冻结，工程已解锁** | M0-4 排除独立 jitter 与第一版 smooth drift；实现 canonical label、zero-init 和等价性测试 |
| M2 | **oracle gate 已解锁，待实现** | M0-3 已证明 `d_dyn` 对 `d_obs` 有互补空间；正式训练仍需 M1 数据基础、clean commit 与预注册 controls |
| M3 | **锁定** | M1/M2 的 true-dt 必须同时优于 fixed/shuffled，并通过 A1 standard guardrail |
| M4 | **锁定** | M2 互补性、predicted-history tube oracle 与 uncertainty calibration 必须全部成立 |

### M0：冻结输出和 oracle，不训练

- [ ] 固定当前文档、脚本和配置，形成可回查的 clean code/config commit；checkpoint 与大体积结果保存路径和 SHA256 索引即可。正式运行不接受 dirty provenance。
- [x] 复用统一 endpoint/per-tracklet logger 完成 P0-C-D1 的 true/fixed/shuffled paired outputs。
  - 2026-07-21：三路 full 各 `91` 个 tracklet、`1257` 个 endpoint，endpoint/order/checkpoint/config/selection/manifest exact match；true−fixed 为 `+0.4376/+0.5231`，true−shuffled 为 `-0.1233/+0.0557`，逐 tracklet Success/Precision bootstrap 95% CI 均跨 0。
  - true 相对两个控制各有 `1079/1257` 个 endpoint 的预测中心改变，说明旧 DynamicsEncoder 会读取时间；但 true alignment 没有带来稳定正确性优势。long-gap 和高位移分桶也未通过门槛，mean-error 表面改善主要由单条灾难性长尾驱动。
  - 正式结论仍为 `NO_GO_P0C_A2_TRUE_DT_PROMOTION`。详细证据见 `compare_results/reports/m0_p0c_d1_full_analysis_20260721.md`；旧 2-tracklet smoke 只保留为首帧 CSV/aggregate 口径修复记录。
- [ ] A/B/C final checkpoint 的 standard、gap1124、burst-drop、unseen schedule 输出，以及 evaluation-only multi-path center/yaw variance。
- [x] crop-reachable `d_obs -> d_dyn` oracle convex blend 完成；primary `1311 endpoints / 213 tracklets`，dynamics-only tracklet bootstrap mean gain `0.803 m`、95% CI `[0.633,0.988]`，决定为 `GO_M2_PROPOSAL_INNOVATION`。long-gap 支持，sparse 样本不足。
- [x] candidate0 与 candidate1/2/3 的 velocity、acceleration 和 proposal error 审计完成；非零 candidate jitter P50 为 `0.611 m/s`、`2.128 m/s²`，matched proposal penalty CI 不跨 0，决定为 `FREEZE_M1_SHARED_SE2`。

M0 的 logger、冻结评测、proposal oracle 与 candidate 审计可以并行，但不得改变 checkpoint 或预测路径。proposal oracle 无空间就停止 M2，不用训练试错替代 oracle。GT-history / predicted-history trajectory-tube crop oracle 作为 M4 的单独前置条件，在 M4 实现前完成，不阻塞 M1/M2。

### M1：物理一致数据与双时钟 adapter（工程准备已解锁）

- [x] 读取并冻结 M0 的 candidate0/1/2/3 审计结论；不得在 M1 中依据 tracking test 涨跌反向改变判据。
- [x] augmentation 预注册为 shared SE(2)，第一版不实现 smooth drift。
- [ ] 加入 zero-init physical-time adapter，并验证初始输出与 A1 数值一致。
- [ ] 同 checkpoint 做 true/fixed/shuffled forward invariance smoke test。
- [ ] 正式训练前冻结 clean commit、唯一 augmentation 定义和 seed42 mini 配置；第一轮不扫超参数网格。

### M2：proposal innovation（oracle gate 已解锁，待实现）

- [ ] 实现 `d_dyn - stopgrad(d_obs)`，不保留旧完整位移叠加为正式路径。
- [ ] `alpha` 从 0 或很小的非零值初始化；记录 applied ratio、innovation norm、梯度和 clamp ratio。
- [ ] 只跑 seed42 mini 的 true/fixed/shuffled；不扫大网格。

### M3：asymmetric path distillation（尚未解锁）

- [ ] A 使用 canonical dense path，B 使用 irregular true-time path。
- [ ] 第一轮固定 `beta=0`，只比较 A1、dual-clock、dual-clock+innovation、再加 asymmetric path distillation。
- [ ] 不把当前 corrected-TWC checkpoint 继续训练成新方法。

### M4：连续—离散滤波和 trajectory tube（尚未解锁）

仅当 M2 显示 dynamics proposal 有互补性、M1 通过时间负对照、tube oracle 有明显 crop recall 空间时执行。

- [ ] 先用固定 process/observation covariance 的非学习滤波基线。
- [ ] 再决定是否学习 `Q(dt)` 与 `R_obs`。
- [ ] 保持 point budget、FLOPs 和候选数公平。

## 6. 预注册 Go / No-Go

mini seed42 只作为筛选：

- standard 相对 A1 不低于 `-0.5 Success / -1.0 Precision`；
- gap1124 或 burst-drop 至少一项达到 `+1 Success / +2 Precision`；
- `true-dt` 同时超过 `fixed-dt` 和 `shuffled-dt`；
- improvement 不来自更多 optimizer steps、更多 candidate、扩大点预算或 best checkpoint 选择；
- path variance、crop recall 和 tracking metric 至少两项机制指标同方向；
- proposal innovation 具有非平凡 applied ratio，且不是数值接近零的伪启用。

进入论文主张的底线：

- full nuScenes；
- Waymo 或 KITTI-HV 第二数据集；
- seed42/43/44；
- per-tracklet paired bootstrap CI；
- standard、seen cadence、unseen cadence；
- `true > fixed` 且 `true > shuffled` 在独立 tracklet 统计中保持正向；
- 报告参数量、FLOPs、FPS、显存、crop point count 和 failure taxonomy。

## 7. 论文标题分叉

只有 dual-clock/explicit-dt 通过时间负对照时：

- `CT-SeqTrack: Dual-Clock State Filtering for Variable-Rate 3D Single Object Tracking`
- `Physical-Time State Estimation for Variable-Rate LiDAR 3D Tracking`

如果只有 asymmetric path distillation 成立：

- `Endpoint-Consistent History Resampling for Variable-Rate 3D Single Object Tracking`

如果两个方法机制都失败，但协议扩展完整：

- `When Frame Index Is Not Time: A Variable-Rate Benchmark for LiDAR 3D Single Object Tracking`

## 8. 可以说与不能说

在方法尚未通过前，可以说：

- 本项目提出并预注册一个 dual-clock continuous-discrete tracker 候选；
- 当前代码审计发现 independent candidate jitter 和 full-displacement residual 存在物理定义风险；
- 新设计以 zero-init、proposal innovation 和 asymmetric distillation 保留 A1 基线作为结构性约束。

不能说：

- dual-clock/filter/tube 已经涨点；
- Kalman gain 已解决旧 Gate 的独立验证失败；
- asymmetric TWC 已超过 single-view A1；
- 真实时间已经由现有 A2 或 corrected-TWC 证明有效；
- 当前 mini 结果已经构成完整 benchmark。

## 9. 邻近工作边界

- [SeqTrack3D](https://arxiv.org/abs/2402.16249)：历史点云与框序列；CT 不能 claim 首次多帧。
- [HVTrack](https://arxiv.org/abs/2408.02049)：固定 frame interval 的 HTV 与 expanded search；CT 必须强调 within-track irregular cadence 和 matched time intervention。
- [TrajTrack](https://arxiv.org/abs/2509.11453)：bbox trajectory proposal；CT 必须强调真实 `delta_t` 的状态传播和严格 GT-free 融合。
- [ChronoTrack](https://openaccess.thecvf.com/content/CVPR2026F/html/Yoo_Temporally_Consistent_Long-Term_Memory_for_3D_Single_Object_Tracking_CVPRF_2026_paper.html)：temporal consistency 与 long-term memory；CT 的一致性必须限定为同 endpoint 的 history-resampling path。
- [NCDSSM](https://proceedings.mlr.press/v202/ansari23a.html)：不规则观测的 continuous-discrete state-space 建模依据。
- [ContiFormer](https://proceedings.neurips.cc/paper_files/paper/2023/hash/9328208f88ec69420031647e6ff97727-Abstract.html)：连续时间 attention 的参考；当前第一版只取轻量 relative-time adapter，不直接复制完整架构。

## 10. 与现有结论的关系

本文不改变以下正式判定：

```text
NO_GO_OBSERVATION_RELIABILITY_VALIDATION
NO_GO_P0C_A2_TRUE_DT_PROMOTION
NO_GO_TWC_MAIN_METHOD_PROMOTION
```

它只定义这些判定之后，若继续追求方法论文，唯一允许进入的新方法闭环：

```text
physical-consistent augmentation
    -> zero-init dual-clock adapter
    -> proposal innovation
    -> asymmetric endpoint path distillation
    -> optional calibrated state filter / trajectory tube
```

任何阶段没有通过预注册 gate，都回到多模型、多数据集的 variable-rate benchmark/diagnosis 路线。
