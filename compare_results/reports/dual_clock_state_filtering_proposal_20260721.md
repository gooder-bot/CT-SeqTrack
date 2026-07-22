# CT-SeqTrack 新贡献框架与方法改造规格

更新时间：2026-07-22

状态：**候选方法，M0 收口与 M2 训练并行。M1/M2 Engineering Gate、E6 静态冻结、commit `473738f` 的服务器 manifests/preflight 已通过；R1 A1-init M2 formal 与用户报告的 R2 M2 scratch/R3 matched W0 scratch 正在运行，尚未完成性能或因果时间验证。** 本文定义下一版可以进入论文的方法叙事、实现顺序、因果验收和停止条件；它不覆盖已经完成的 No-Go 结论，也不能被引用为实验结果。当前 alpha/R/warmup/唯一 formal 配置与 same-checkpoint controls 已固定；M3–M4 必须按本文门槛逐级解锁。

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

standard 的本地 `delta_t` 仅为 `0.4974±0.0228 s`（CV `4.59%`，`86.55%` 位于 `0.5±0.01 s`），而 gap1124/burst-drop 的 CV 为 `58.94%/62.63%`。近常量时间使 `g(delta_t)` 接近固定系数并可被普通权重吸收，因此 standard 只承担 normal-cadence guardrail；物理时间主张必须依赖 strong/held-out cadence 和 same-checkpoint negative controls。正式统计、丢帧表述边界和协议矩阵见 `compare_results/reports/htv_identifiability_and_execution_plan_20260722.md`。

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

这等价于朝 dynamics proposal 做有界移动。`alpha=0` 时严格恢复 A1；只有 `||d_dyn-d_obs|| <= R(dt)`、clamp 未触发时，`alpha=1` 才到达 dynamics proposal，触发 clamp 时只到达同方向的有界点。训练前必须在 crop-reachable endpoint 上计算未裁剪线段 `d_obs -> d_dyn` 的 oracle 最优点确认方向值得做，再用 training split 一次性冻结 `R(dt)`；正式结果必须同时报告 clamp ratio，不能把未裁剪 oracle 当成实际模块保证达到的上限。

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

当前 `datasets/sampler.py` 为每个历史框独立采样 offset；这会把不连续随机误差解释为速度/加速度。M0-4 已经冻结第一版方案：**整条历史轨迹只施加一个 sample-level shared SE(2)，不实现 smooth drift**。

这里的 shared 不是“给每个 `getOffsetBB` 传同一个数组”。`getOffsetBB` 在每个框自己的局部朝向中解释平移，相同 `[dx,dy,dtheta]` 仍可能对应不同世界平移。正式定义以最近历史 anchor `(a,yaw_a)` 为公共原点：

```text
c_i'   = a + R(dtheta) (c_i - a) + R(yaw_a) [dx,dy]
yaw_i' = yaw_i + dtheta
```

所有历史框共用同一个世界刚体变换，z、尺寸和时间戳不变。验收以几何不变量为准：在增强后的 anchor 坐标中，candidate trajectory 的 pairwise center/yaw 关系应与 canonical trajectory 一致。Dynamics 的速度/加速度监督先从 canonical GT trajectory 和真实 `delta_t` 计算，再一致表达进增强坐标；禁止由 candidate `ref_boxs` 的差分直接构造导数。

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
| `utils/twc_utils.py` | 现有共享只按绝对 frame id 服务 TWC；不是整轨迹共同刚体变换 | 保持现有 TWC 跨视图共享语义，不在这里偷换成 M1 整轨迹变换 |
| 新建 `utils/candidate_utils.py` | 尚无 sample-level trajectory transform | 实现 world-SE(2) 纯函数、mode 归一化和可审计 transform metadata |
| `datasets/sampler.py` | 每个历史框独立调用局部 `getOffsetBB`，可能制造伪速度 | 独立 `candidate_trajectory_mode`；shared world-SE(2)、canonical dynamics label 和审计字段 |
| `models/dynamics.py` | 由 pooled transition feature 直接输出 velocity，再乘当前 gap | 显式 `F(dt)` 状态传播 + 有界 acceleration residual；保留轻量实现 |
| `models/dynamics.py` gate | `init_alpha=0` 经 sigmoid 后实际约为 `2e-5` | 增加显式 zero-scale/disabled path；近零不能冒充严格 A1 等价 |
| `models/seqtrack3d.py` | 旧 `residual` clamp 完整 `d_dyn` 再加到完整 `d_obs` | 新增显式 `proposal_innovation` 模式；旧 residual 保留作历史负对照；zero/invalid 严格回退 |
| `models/seqtrack3d.py` TWC | `0.5(L_A+L_B)` 对称监督 | EMA canonical teacher -> irregular student |
| `models/observability.py` | 独立 split No-Go | 停止；仅在先验成立后研究 covariance head |
| 新建 `tools/check_candidate_shared_se2.py` | 现有 smoke 只检查 TWC/shared-frame 和旧 residual | dataset-free 几何单测，再补 loader/candidate0 回归 |
| dataset/protocol | nuScenes 已有 matched time controls，Waymo 未完全接入 | 统一 manifest、endpoint logger 和 held-out schedule |

## 5. 最小实现顺序

阶段决策固定为：

| 阶段 | 当前状态 | 解锁边界 |
| --- | --- | --- |
| M0 | **进行中** | 只做冻结输出、离线 oracle、candidate 审计和 provenance 收口 |
| M1 | **工程完成，formal 中使用** | shared world-SE(2)、canonical label、zero-init 与 E0–E6 已通过；禁止训练中漂移配置 |
| M2 | **R1/R2/R3 运行中，结果待定** | R1 preflight PASS；先完成 final/provenance、`R1-A1`、`R2-R3` 和 same-checkpoint time controls |
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

### M1：物理一致数据与双时钟 adapter（Engineering GO）

- [x] 读取并冻结 M0 的 candidate0/1/2/3 审计结论；不得在 M1 中依据 tracking test 涨跌反向改变判据。
- [x] augmentation 预注册为 shared SE(2)，第一版不实现 smooth drift。
- [x] 新增默认关闭的 candidate trajectory mode；legacy independent 与 shared world-SE(2) 不静默混用。
- [x] shared 变换围绕共同 anchor 在世界坐标一次施加；没有把相同局部 offset 重复交给每个 `getOffsetBB`。
- [x] canonical GT label 只使用真实 `delta_t`；pairwise trajectory、伪导数、candidate0/1/2/3、degrees/radians 和 TWC 回归已通过。
- [x] 加入显式 zero-scale/disabled physical-time adapter；同权重同 batch 的 motion/output/loss 与 A1 strict-zero 数值一致。
- [ ] 同 checkpoint 做 true/fixed/shuffled forward invariance smoke test。
- [x] 唯一 augmentation 与 seed42 mini 配置已静态冻结；commit `473738f` 的 server manifests/preflight 已通过，第一轮不扫超参数网格。

### M2：proposal innovation（训练运行中，结果待定）

- [x] 新增显式 `proposal_innovation` mode：实现 `d_dyn - stopgrad(d_obs)`；旧完整位移叠加只保留作可复现负对照。
- [x] effective alpha 使用单一 `[0,1]` 系数；正式值冻结为 `0.75`，绝对 correction 由 `R(dt)=min(0.5+0.5dt,2.0)` 控制。
- [x] `alpha=0`、disabled、warmup 或 `dynamics_valid=0` 时严格恢复 observation/A1。
- [x] 记录 raw/clamped/applied innovation norm、applied ratio、alpha、clamp ratio、invalid fallback、adapter/encoder gradient。
- [x] standard/gap1124/burst-drop、invalid/resampled/empty fallback 与至少 2-step optimizer smoke 已通过。
- [ ] R1 A1-init seed42 formal 训练完成后，只用同一 final checkpoint 评测 true/fixed/shuffled；不训练三个 time mode，不扫大网格。
- [ ] R2 M2 scratch 与 R3 matched W0 scratch 先完成训练完整性审计，再用 `R2-R3` 隔离随机初始化下的结构净效应。

E0–E6 已完成并解锁当前 R1；这不自动解锁新 seed、mixed-cadence、M3 或 M4。训练完成后必须先核对 epoch/global step、唯一 `last.ckpt`、SHA256、resolved config、events 与 artifact manifest，再比较 `R1-A1`、`R2-R3` 和 R1 same-checkpoint time controls。M0-2 可并行收口，但仍是 M0 完成和旧 TWC 解释收口的必要项。

### M3：asymmetric path distillation（尚未解锁）

- [ ] A 使用 canonical dense path，B 使用 irregular true-time path。
- [ ] 第一轮固定 `beta=0`，只比较 A1、dual-clock、dual-clock+innovation、再加 asymmetric path distillation。
- [ ] 不把当前 corrected-TWC checkpoint 继续训练成新方法。

### M4：连续—离散滤波和 trajectory tube（尚未解锁）

M4 是后置的高上限阶段，不是当前 M2 上再叠一个 learned Gate。它要把“每帧生成 observation/dynamics proposal 并做固定权重修正”升级为：**跨帧保存后验状态和协方差、按真实 `delta_t` 连续传播、在离散 LiDAR 帧到来时用当前观测更新，并在 observation forward 之前用先验分布构造受限搜索 tube**。当前仓库尚无 persistent state、covariance update 或 tube crop；`datasets/sampler.py` 中已经禁用并抛出 `NotImplementedError` 的旧 `KalmanFiltering` 片段不属于 M4。

#### M4 的解锁条件

只有以下三项同时成立才允许开始 M4：

1. M2 在在线递归评测中显示 dynamics proposal 有互补性，standard guardrail 通过，且同 checkpoint 的 `true-dt` 同时超过 `fixed-dt/shuffled-dt`；
2. 使用 **M2 predicted history** 而不是 GT history 的 trajectory-tube oracle，在固定点预算下相对 baseline crop 产生非平凡的 target-recall/tube-only complementarity；
3. 固定 covariance 基线或 learned covariance 在独立 split 上通过 NLL、coverage、ECE/reliability 等校准检查。

GT-history CV 接近 99% recall 只说明理想历史存在上限，不能解锁 M4。P0-B2 已判定 always-on raw predicted-history CV recenter 为 No-Go；因此 M4 只能采用 `baseline crop union bounded tube`，不能把 raw trajectory center 替换成唯一 search anchor。

#### 持久状态与连续传播

第一版状态固定为：

```text
s_t = [x, y, z, vx, vy, vz, yaw, yaw_rate]
```

box size 继续来自 observation 或上一后验框，不在第一版加入尺寸动力学。每个 tracklet 必须保存：

```text
posterior mean mu_t_plus
posterior covariance P_t_plus
last timestamp
valid/reset/fallback state
```

在下一帧 LiDAR 到达前，使用真实时间间隔传播：

```text
dt          = timestamp_t - timestamp_(t-1)
mu_t_minus  = f(mu_(t-1)_plus, dt)
P_t_minus   = F(dt) P_(t-1)_plus F(dt)^T + Q(dt)
```

第一版 `f/F` 使用显式 constant-velocity/constant-turn，不直接实现 ODE/CDE/Mamba：

```text
x'   = x  + vx * dt        vx'       = vx
y'   = y  + vy * dt        vy'       = vy
z'   = z  + vz * dt        vz'       = vz
yaw' = wrap(yaw + yaw_rate * dt)
yaw_rate' = yaw_rate
```

如需学习修正，只允许加入受界 acceleration/turn residual；禁止再预测并叠加一份无限制完整 displacement。第一版过程噪声使用固定参数。对单轴 `[position, velocity]`，可采用 constant-acceleration noise 的正定结构：

```text
Q_axis(dt) = sigma_a^2 * [[dt^4/4, dt^3/2],
                          [dt^3/2, dt^2  ]]
```

`xy/z/yaw` 的尺度分别冻结。只有固定 `Q/R` 基线成立后，才允许学习 `Q_theta(dt, history)`；learned covariance 必须用 Cholesky/softplus 等参数化保证半正定，并限制极端 `dt` 下的数值范围。

#### 离散 observation update

当前 observation branch 的框中心和 yaw 被解释为离散 measurement：

```text
z_t = [x_obs, y_obs, z_obs, yaw_obs]
```

measurement matrix `H` 从完整状态中选择位置和 yaw。更新为：

```text
nu_t      = z_t - H mu_t_minus
S_t       = H P_t_minus H^T + R_obs
K_t       = P_t_minus H^T S_t^(-1)
mu_t_plus = mu_t_minus + K_t nu_t
```

`yaw` innovation 必须 wrap；实现使用 `torch.linalg.solve`/Cholesky solve，不显式求逆。协方差更新使用更稳定的 Joseph form：

```text
P_t_plus = (I-KH) P_t_minus (I-KH)^T + K R_obs K^T
```

固定 `R_obs` 的 GT-free filter baseline 必须先于 learned observation covariance。若实现 covariance head，它只能读取推理时可得的 observation feature、foreground/point statistics 和预测置信度，不能读取当前 GT、GT overlap 或当前真实误差。旧 P5 hand-crafted observability Gate 已 No-Go，不复活、不在 mini_val 上换名重调。

M4 与 M2 是替换/升级关系，而不是重复融合同一信息：

```text
M2: d_obs + fixed_alpha * bounded(d_dyn - d_obs)
M4: propagated prior + covariance-derived measurement update
```

完整 M4 启用后，不能先做一次 M2 fixed-alpha correction，再对同一 observation/dynamics pair 做第二次 Kalman correction。M2 仅保留为安全基线、消融和 calibration 失败时的 fallback。

#### Trajectory tube search support

状态 prior 必须在 observation forward 之前生成搜索支持。给定平面速度方向 `e_parallel`、垂直方向 `e_perp` 与位置协方差 `P_xy`：

```text
sigma_parallel = sqrt(e_parallel^T P_xy e_parallel)
sigma_perp     = sqrt(e_perp^T     P_xy e_perp)

tube_length = base_length + |v| * dt + k_parallel * sigma_parallel
tube_width  = base_width              + k_perp     * sigma_perp
search      = baseline_crop union trajectory_tube
```

低速时用 posterior yaw 作为 tube 方向；tube length/width 必须有上限。invalid history、非单调 timestamp、非 finite/非 PSD covariance 或状态重置时，严格退回 baseline crop。union 的目的在于保留 observation anchor，避免 P0-B2 中“错误 predicted center 成为唯一 anchor”导致的失控。

公平性硬约束：

- baseline crop 与 tube union 后仍使用相同总 point budget；不得简单增加一份点；
- candidate 数、checkpoint、endpoint、point-sampling seed 保持一致；
- 记录 baseline/tube/union 的 target points、background ratio、crop recall、tube-only reachable、empty fallback；
- 同时报告参数量、FLOPs、FPS、显存和实际 crop point count。

#### M4 的分阶段实现

M4 不允许一次性同时引入 filter、tube 和 learned covariance，固定为四个切片：

1. **M4-0 predicted-history tube oracle**：使用 frozen M2 checkpoint，只做 baseline-crop/tube-union 的离线固定预算 reachability；没有 tube-only complementarity 就停止。
2. **M4-1 fixed-covariance filter**：新增 persistent `mu/P`、解析 `F(dt)`、固定 `Q/R` 和 tracklet reset；不接 tube、不训练 covariance head，先与 A1、M2、CV/Kalman GT-free 基线比较。
3. **M4-2 filter + tube**：只有 M4-1 为正才让 prior mean/covariance进入 search support；保持点预算/FLOPs 口径，分开报告 filter-only、tube-only、filter+tube。
4. **M4-3 learned covariance**：只有固定滤波和 tube 均为正，才学习 `Q(dt)`/`R_obs`；先在独立 split 校准，再进入 tracking 闭环。

第一帧使用 3D SOT 合法的初始化框：position/yaw covariance 较小，velocity/yaw-rate 初始化为 0 且 covariance 较大。新 tracklet、非法 `dt`、数值失败和显式 reset 必须清空旧状态；所有 reset/fallback 决策只能使用推理时可得信号。

#### 校准和正式消融

learned covariance 进入融合前至少报告：

- Gaussian NLL；
- 50/90/95% coverage 与置信区域大小；
- ECE/reliability diagram，按 `delta_t`、稀疏度和 cadence 分桶；
- covariance eigenvalue、非 PSD/非 finite、reset/fallback 比例；
- 可选 NEES，用于检查状态误差与协方差是否匹配。

正式最小消融矩阵：

```text
A1 / SeqTrack observation baseline
M2 fixed-alpha proposal innovation
fixed CV/Kalman, no tube
tube only, no filter update
filter only, no tube
fixed-Q/R filter + tube
learned-R only
learned-Q/R full
true / fixed / shuffled time controls
```

如果 fixed covariance filter 无收益、predicted-history tube 无独立 crop 空间或 covariance calibration 失败，则停止 M4；退回 M2 固定有界 innovation，不通过增加复杂时序模块追分，也不能在论文标题中使用“state filtering”。

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
