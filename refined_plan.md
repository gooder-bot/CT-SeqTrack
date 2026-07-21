# CT-SeqTrack 研究计划与论文定位

更新时间：2026-07-21

这份文件用于每次开始工作前快速整理研究思路。下一步执行清单见 `need_to_do.md`，已完成工程和实验记录见 `done.md`，简洁实验结论见 `sum_results.md`。2026-07-21 之后的新候选贡献、公式、代码落点和 Go/No-Go 见 `compare_results/reports/dual_clock_state_filtering_proposal_20260721.md`。

---

## 0. P0 后的论文可行性决策

当前项目仍有论文机会，但不能按“CT-SeqTrack full model 已经成立”继续。完整的 code-to-claim 审计、方法/benchmark 分叉和实验底线见 `compare_results/reports/paper_viability_and_execution_20260720.md`。

**2026-07-21 阶段决定**：项目仍处于 M0 收口，但两个关键 gate 已关闭。M0-3 得到 `GO_M2_PROPOSAL_INNOVATION`：不仅 oracle gain 稳定为正，冻结 `d_dyn` 本身也在 81.31% primary endpoint 上优于 `d_obs`，tracklet bootstrap CI 不跨 0。M0-4 得到 `FREEZE_M1_SHARED_SE2`：逐历史帧独立 candidate offset 的伪速度/伪加速度远超阈值，M1 第一版只允许 shared SE(2)。这解锁 M1 数据层和 M2 innovation 工程，不表示 dual-clock 已涨点；M0-2 四协议冻结输出未完成，正式训练仍须唯一配置、clean commit 和预注册控制。

四个会决定论文名称和贡献形态的事实是：

- 同提交 A1 TWC A/B/C 已完成：`C-B` final 为 `+8.31/+11.74`，但 paired-view 的 `B-A` 为 `-15.30/-24.18`，导致 `C-A=-7.00/-12.44`。这只支持 TWC 对受损 paired-view 路径的部分修复，不能支持主方法 promotion。
- A1 corrected-TWC 使用 `main_time_source=order` 且关闭 `DynamicsEncoder`；即使有 `C-B`，也只能支持 rate/resampling consistency，不能单独证明真实 timestamp 有效。
- 当前 bounded residual 把完整 `dynamics_displacement_pred` 加到已经预测完整 displacement 的 observation 输出上，存在重复运动的定义歧义；M0-3 已证明 `d_obs -> d_dyn` 线段有稳定空间，正式路径改为 bounded proposal innovation。
- P0-C 已证明当前 feature-concat A2 的 true-dt 不超过 shuffled；显式时间方法主张必须由新的机制重新通过同类负对照。

论文优先级因此改为：

```text
同提交 TWC A/B/C seed42：主方法 promotion No-Go
    -> M0：P0-C-D1/M0-3/M0-4 已完成；继续 M0-2 strong-cadence/path-variance 收口
        -> M1：shared SE(2) augmentation + canonical label + zero-init dual-clock（现在开始实现）
            -> M2：proposal oracle 已通过；实现 bounded innovation，之后做 seed42 true/fixed/shuffled
                -> M3：因果正信号后做 asymmetric path distillation
                    -> M4：tube oracle/calibration 通过后做 filter/tube
        -> 任一关键 gate 再失败：停止对应模块，保留 benchmark/diagnosis 分叉
```

## 1. 最终定位

研究问题继续成立，但论文定位必须由后续正证据决定：

**在不规则采样、掉帧和长时间间隔下，评估并提高 3D SOT 对观测 cadence 和历史采样路径的鲁棒性；只有 explicit-dt 机制通过负对照，才升级为真实时间感知状态估计。**

更稳的关键词：

```text
timestamp-native / variable-rate / time-aware 3D SOT
```

一句话主线：

**真实 timestamp 改变了历史状态的物理含义；当前先用冻结的 variable-rate / held-out-cadence 协议诊断这一变化，并分别验证 resampling consistency 的净贡献与 explicit-dt 的物理时间因果性。**

可考虑标题：

- CT-SeqTrack: Dual-Clock State Filtering for Variable-Rate 3D Single Object Tracking
- Physical-Time State Estimation for Variable-Rate LiDAR 3D Tracking
- Endpoint-Consistent History Resampling for Variable-Rate 3D Single Object Tracking
- CT-SeqTrack: Timestamp-native Sequence Modeling for 3D Point Cloud Tracking
- Variable-rate 3D Single Object Tracking with Time-resampling Consistency
- Timestamp-aware Sequence Tracking for 3D Single Object Tracking

标题也必须由证据分叉：只有 explicit-dt 机制通过 true/fixed/shuffled，才使用 `timestamp-native/time-aware`；如果只有 A1-TWC 成立，应使用 `variable-rate/resampling-consistent`；如果方法机制均失败，则改为 benchmark/diagnosis 标题。

不要把主 claim 写成：

- “continuous motion modeling”：容易撞 StreamTrack。
- “high temporal variation”：容易撞 HVTrack / MambaTrack3D。
- “historical trajectory prior”：容易撞 SeqTrack3D / TrajTrack。
- “first sparse/occlusion solution”：容易撞 CXTrack / MVCTrack。

当前实验边界：

- 已有结果支持：主干保留 SeqTrack3D 的 order-time 语义，同时把真实 `delta_t/current_delta_t` 注入 `DynamicsEncoder`，比直接替换主干时间 token 更稳定。
- 目前不能宣称：完整 CT-SeqTrack full model 已经稳定超过 SeqTrack3D。
- corrected-TWC 的同提交 A/B/C seed42 已完成：B 相对 A 明显退化，C 相对 B 回升 `+8.31/+11.74`，但相对 A 仍为 `-7.00/-12.44`；两组 paired run 的 anchor/current XYZ gap max 全程为 0。TWC 主方法 promotion No-Go，不补 seed43/44。
- 目前不能宣称 observability gate 已经带来稳定最终收益；gate-safe 低于 A2，conf-res best-e14 复测未复现旧 best。
- bounded residual 已完成 standard 真实 batch 回归：warmup 与 active forward/loss/backward finite，但默认实际 correction 只有约 `1e-7 m`、gate 梯度极小，未通过功能验收，也没有性能结果。
- HTV 六组筛选显示旧 feature-concat A2 只在 random20 上为正，在 gap1124/burst-drop 上明显退化。因此“真实时间分支已解决强不规则跟踪”不成立，必须先验证 residual、candidate 监督与 crop 可达性。
- standard/gap1124/burst-drop crop oracle 显示 previous-GT base recall 为 85.41%/76.78%/77.72%；强协议下 2x expanded 也只有 89.08%/87.65%，GT-history CV recenter 则为 98.96%/99.05% 且不增加背景点。当前最直接的机制瓶颈已经从“末端如何融合”前移到“目标能否进入 search crop”。
- P0-B2 recursive predicted-history 显示 raw CV 相对 previous-A1 只提高 2.65–3.03 pp，未通过预注册门槛；但预测历史可靠时 pred-CV recall 可达 97.34%–98.64%。这一步否定了恒开启单锚点 recenter，并把问题推进到 P0-B3 的 reliability/互补性验证。
- P0-B3 三协议 full 进一步收窄了结论：预注册 13 特征 trigger AUROC 为 0.857/0.787/0.785，但 passive raw-CV union gain 只有 3.04/2.88/3.15 pp，当前 selector 在 gap/burst AUROC 只有 0.605/0.433。`prev_obs_only` 在 gap/burst 的 AUROC 反而为 0.867/0.873，删除 raw `current_delta_t` 后为 0.865/0.872；因此目前成立的是 observation-quality reliability proxy，不是 timestamp-aware reliability，也不是可用的 active raw-CV dual-anchor。
- P0-B4 把上述开发集信号放到 disjoint mini_val 冻结验证后，gap/burst AUROC 降为 `0.680/0.712`，固定阈值 recall 只有 `0.568/0.609`，正式得到 `NO_GO_OBSERVATION_RELIABILITY_VALIDATION`；同批 raw-CV 第二 crop 在两个强协议没有 trajectory-only endpoint。当前 reliability-controlled independent state 与 active dual-anchor 在实现前停止。
- TrajTrack 本地 aligned run 的 evaluator 使用当前帧 GT 触发和选择 refinement；64.94 / 79.07 只能作为 oracle-assisted 诊断，不能进入公平主表。

当前执行策略：

```text
1. P0-B4 已完成并 No-Go：不实现当前 observation calibrator 控制的 Kalman/frozen-state，不做 active dual-anchor，不在 mini_val 上重调。
2. 当前脚本、verdict 和 P0-C 协议已绑定 clean GitHub commit `343145d`；服务器输出继续保存 script/data/config/checkpoint hashes，旧 stash 不恢复到正式运行路径。
3. P0-C frozen A2 triplet已得到 `NO_GO_P0C_A2_TRUE_DT_PROMOTION`；同提交 TWC A/B/C 也已得到 `NO_GO_TWC_MAIN_METHOD_PROMOTION`，均不扩展训练 seed。
4. M0 P0-C-D1 已完成：三路各 `91` 个 tracklet、`1257` 个 endpoint，endpoint/order/hash 与时间干预检查通过；true−fixed 为 `+0.438/+0.523`，true−shuffled 为 `-0.123/+0.056`，逐 tracklet Success/Precision bootstrap CI 均跨 0。下一步复用同一 logger，对冻结 A/B/C final checkpoint 做 strong-cadence 与 evaluation-only path-variance 收尾，不改变预测路径。
5. M0-3/M0-4 已完成：M2 oracle gate 通过，M1 augmentation 冻结为 shared SE(2)；完整证据见 `compare_results/reports/m0_m03_m04_analysis_20260721.md`。
6. 现在从 M1 shared SE(2) 数据层、canonical dynamics label、接口、配置、zero-init adapter 和 A1 数值等价性测试开始；正式训练必须在 clean commit 上使用唯一预注册配置。
7. 后续严格按 `M1 physical-consistent augmentation/dual clock -> M2 proposal innovation -> M3 asymmetric path distillation -> optional M4 filter/tube` 逐级推进；不从旧 feature concat、旧 Gate 或对称 paired loss 直接扩展。
```

当前最可防御的价值是：**同一 tracklet 内不规则物理时间协议、冻结 checkpoint 的 matched time negative controls，以及 crop/trajectory/observation failure diagnosis**。M0-3 已把有界 observation-first correction 从待检假设推进为有 offline proposal 互补性的候选，但尚未得到 tracking Success/Precision 增益；M0-4 则把 shared SE(2) 固定为物理一致的数据前提。历史重采样一致性仍只保留 `C-B` 部分修复这一机制事实。

### 连续时间视角给当前工作的启发

`claude_thinking.md` 中最有价值的判断是：很多 3D tracking 方法表面上使用不同 backbone、memory 或 trajectory prior，但底层仍共享同一个离散时间契约：

```text
time = {t0, t1, t2, ...}
state = f(frame sequence)
```

更长期的研究方向可以写成：

```text
state(t) = timestamp-conditioned state estimation, t in R+
```

但当前 CT-SeqTrack 不应直接升级为完整 ODE/SDE/CDE tracker。更稳的论文边界是：**先把 SeqTrack3D 从固定帧步长序列学习推进到真实时间间隔驱动的 variable-rate 3D SOT**。也就是说，当前工作是连续时间 3D tracking 的克制第一步，而不是宣称已经实现任意时刻查询、连续 ODE 求解或多传感器异步融合。

这个视角可以帮助写 introduction：

- 现有方法通常默认相邻历史帧等间隔，因此 `t-1` 在 2Hz、10Hz、跳帧和掉帧场景下被赋予了相同语义。
- 真实 timestamp 改变了历史状态差分、速度估计、观测可靠性和序列一致性的物理含义。
- CT-SeqTrack 的切入点不是换 backbone，而是改变 3D SOT 对“时间”的输入契约。

---

## 2. 新候选贡献框架（未验证）

> 状态边界：本节是 P0-B4、P0-C 和 TWC A/B/C No-Go 之后的重构方案，不是已经取得涨点的结果。完整方法规格与停止条件见 `compare_results/reports/dual_clock_state_filtering_proposal_20260721.md`。

新的论文层级不再把 Dyn、TWC、Gate、HTV 写成四个平行模块：

| 层级 | 新定义 | 论文角色 | 当前状态 |
| --- | --- | --- | --- |
| 贡献 1 | matched within-track variable-rate protocol | 问题定义、真实时间因果控制与评价 | 工程地基已完成，需扩 full data / second dataset |
| 贡献 2 | dual-clock continuous-discrete state update | 唯一方法主轴 | 候选，需 oracle 与 time controls |
| 贡献 3 | endpoint-consistent asymmetric path distillation | 辅助训练目标 | 候选，当前对称 TWC 已 No-Go |
| HTV | irregular/held-out cadence protocols | 贡献 1 的评测条件 | 不能单独称首次 HTV |
| Gate | covariance-derived gain 或固定小 innovation weight | 非贡献、可选实现细节 | 旧 hand-crafted Gate 停止 |

统一方法叙事：

```text
SeqTrack3D order clock
    + zero-init physical-time adapter
    + explicit F(delta_t) state propagation
    + bounded proposal innovation
    + canonical-teacher -> irregular-student path distillation
```

核心结构性约束是：新方法在 adapter/innovation 权重为 0 时必须严格退化为 A1；真实时间只能提供增量信息，不能破坏已经有效的 order-time 主干。

### 贡献 1：Matched variable-rate protocol and physical-time controls

CT-SeqTrack 先把 SeqTrack3D 的输入契约扩展为真实时间感知，而不是简单把所有主干时间 token 都替换为真实秒数：

- 训练和测试都提供一致的 `timestamps / delta_t / delta_T / current_delta_t`。
- 点特征时间通道和 box corner token 工程上支持 `raw / mlp / fourier` 时间编码。
- 已有消融显示，直接把 real-time token 放进 SeqTrack3D 主干会破坏原始 order-time 语义。
- 第一批 `gap1124 / burst_drop / random20` virtual-rate protocol 已完成六组 A1/A2 seed42 筛选。结果具有明显 protocol dependence；后续必须冻结 manifest，并补 `fixed-dt / shuffled-dt` negative control、delta_t bins、sparse bins、displacement bins 和 re-appearance 片段，证明真实时间不是装饰字段，而是 variable-rate 3D SOT 的任务条件。

这仍然是 timestamp-native 的地基：真实时间进入数据、监督和评价协议。论文叙事要避免把失败的 raw main-branch 注入方式写成最终方法，也不要只在普通 fixed-step final 上判断方向成败。

新版本进一步要求：

- 同一 endpoint、当前点云、局部坐标、candidate offset、point seed 和 checkpoint 下，只改变模型读取的 effective time；
- `true/fixed/shuffled` 必须在同一 checkpoint 上比较，不能各自重训；
- standard-only checkpoint 直接测试 seen/unseen cadence；
- 保存 per-tracklet/endpoint 输出、首次失控、连续失败、crop recall 与 path variance；
- 只有 `true > fixed` 且 `true > shuffled`，才能把贡献 2 称为 physical-time method。

### 贡献 2（候选）：Dual-clock continuous-discrete state update

主干保留 SeqTrack3D 的 order embedding，并新增零初始化 physical-time adapter：

```text
phi_dt_i = TimeEncode(log(1 + delta_t_i / dt_ref))
h_i'     = h_i + A_zero(h_i, phi_dt_i)
```

`A_zero` 在初始化时输出严格为 0，使模型从 A1 出发。真实时间只控制 relative-time feature、状态传播和搜索支持，不再直接替换主干 token。

Dynamics 从独立 feature concat 重构为连续—离散状态传播：

```text
s_t = [position, velocity, yaw, yaw_rate]
s_prior(t + dt) = F(dt) s_post(t) + bounded_acceleration_residual
```

第一版使用显式 constant-velocity/constant-turn `F(dt)`，不直接上 ODE/CDE/Mamba。若状态 prior 在 long-gap/sparse 子集具有互补性，再用其均值和随 `dt` 增长的不确定性构造固定点预算的 trajectory tube。

训练数据也必须保持物理一致：当前逐历史框独立 candidate offset 会制造伪速度；新分支只允许共享 SE(2) 扰动或从递归误差拟合的平滑 drift，Dynamics label 从 canonical/一致扰动轨迹计算。

以下旧 feature-concat、raw-CV 与 bounded residual 内容保留为重构动机和失败证据，不代表新贡献已经实现。

当前更稳的模型主线是：主干保持 order-time，真实 `delta_t/current_delta_t` 进入 `DynamicsEncoder`。历史框差分按真实 `delta_t` 计算速度和角速度，形成 timestamp-conditioned dynamics prior。

旧版 `A2-order-dyn` 通过 feature concat 把 `z_dyn` 接入 coarse motion branch，seed42 有 precision-positive 信号，但 seed43/44 暴露了稳定性风险。当前代码已经实现更保守的 residual 形式：

```text
obs_pred = observation_branch(point_feature)
dyn_disp = velocity_pred * current_delta_t
dyn_disp = clamp_norm(dyn_disp, max_norm)
final_center = obs_center + scale * alpha * dyn_disp
```

2026-07-17 的诊断显示上述末端 residual 默认实际修正只有约 `7e-8 m`，而部分失败发生在 search crop 之前。P0-B2/P0-B3 因而尝试把第一修正位置前移到 search crop；P0-B4 的 independent mini_val 冻结验证随后否定了当前 reliability-controlled state anchor 的入口：gap/burst AUROC 和固定阈值 recall 均未达标，raw-CV 第二 crop 也没有强协议互补 endpoint。

```text
P0-B2/P0-B3 raw-CV candidate       -> No-Go
P0-B4 observation reliability      -> No-Go
reliability-controlled state       -> stop before implementation
bounded residual                   -> one reachable-subset kill-test only
```

因此不再实现第一版 `c_traj`，也不上 tiny MLP/GRU、Mamba、ODE/CDE 或 learned uncertainty gate。GT-history CV 只保留为 oracle upper bound；P0-B2–B4 作为“为什么简单 trajectory anchor 不能直接工作”的机制证据。

如果保留 bounded residual 作为一次性消融，必须先解决两个定义问题：

- 当前代码把完整 `dyn_disp` 加到已经预测完整 displacement 的 `obs_center` 上，可能重复计算运动；需要先判断应改为 `dyn_disp - obs_disp` correction，还是让 dynamics head 直接预测 observation error。
- 只有 crop-reachable mini_train subset 才能用于一次性校准；`max_residual_norm` 当前从未触发，不能把“调大 bound”当作下一步。
- `scale / alpha / clamp / warmup` 必须一次性预注册，并用 `true/fixed/shuffled-dt` 检查因果性。

当前论文不能再把 **observation-reliability-updated timestamp-conditioned trajectory guidance** 写成已成立方法。更可防御的表述是：feature concat、raw-CV anchor 和 frozen observation reliability 在不同入口依次失败；现有末端 residual 只保留为可解释的 negative/kill-test 消融，除非它在冻结协议和时间负对照下给出新的因果正信号。

若 oracle 通过，正式 residual 只允许采用 proposal innovation：

```text
innovation = clip_norm(d_dyn - stopgrad(d_obs), R(delta_t))
d_final    = d_obs + alpha * innovation
```

禁止继续把完整 `d_dyn` 加到完整 `d_obs`。`alpha=0` 必须恢复 A1，并持续记录 applied ratio、innovation norm、clamp ratio 和梯度。

### 贡献 3（候选）：Endpoint-consistent asymmetric path distillation

同一条 tracklet 通过不同历史采样路径观察同一当前绝对时刻时，最终状态估计应该一致。这里的“不同采样路径”不是改变当前帧，也不是改变搜索区域，而是在共享最近历史 anchor 的前提下改变更早历史帧：

```text
view A: [t-1, t-2, t-3] -> t
view B: [t-1, t-3, t-5] -> t
```

这样两个 view 的预测应位于同一个局部坐标系里，TWC 才只约束历史时间路径差异，而不是坐标系、crop 或随机点采样差异。2026-07-11 修复后，candidate offset 与 point-sampling seed 都以绝对 frame id 为键共享，并在归一化前输出 `coordinate_anchor` 做 fail-fast 检查。

第一版只约束最终框：

```text
L_center = SmoothL1(c_a, c_b)
L_theta  = SmoothL1(sin(theta_a), sin(theta_b))
         + SmoothL1(cos(theta_a), cos(theta_b))
L_twc    = L_center + lambda_theta_twc * L_theta
```

当前实现用两个 view 的 supervised loss 平均值，而不是简单相加：

```text
L = 0.5 * (L_a + L_b) + lambda_twc * L_twc
```

同提交 A/B/C 已证明这一路径会让困难 view B 的监督显著破坏主任务。下一版不继续调小 `twc_weight`，而是改为 canonical teacher 到 irregular student 的非对称蒸馏：

```text
teacher = EMA(model)
p_a     = teacher(canonical_dense_path)
p_b     = model(irregular_true_time_path)

L = L_sup_a
  + beta * L_sup_b
  + lambda_path * w_a * D(stopgrad(p_a), p_b)
```

第一轮固定 `beta=0`；`w_a` 只能来自 teacher 的推理时可得置信度/不确定性；`fixed/shuffled` 只用于评估，不进入 consistency 训练。这样保留 canonical A 的主任务分布，避免当前 `0.5(L_a+L_b)` 的退化来源。

这个贡献必须写窄：不是泛 temporal consistency，而是 **endpoint-conditioned history-resampling distillation**。在新目标超过 single-view A1 之前，现有 TWC 仍只能作为“部分修复 paired-view 退化”的机制诊断。

### 非贡献项：Gate 停止；仅保留可校准 uncertainty fusion

P0-B4 已否定当前 hand-crafted observation reliability Gate 的独立推广性。该 Gate 不复活、不在 mini_val 上重调，也不列贡献。

只有贡献 2 的 state prior、proposal innovation 和 trajectory-tube oracle 都通过后，才允许把二元 Gate 改成连续精度融合：

```text
K     = P_dyn (P_dyn + R_obs)^(-1)
state = prior + K * innovation
```

其中 `P_dyn` 随真实 `delta_t` 传播，`R_obs` 必须先通过 NLL、coverage 和 calibration 验证。校准失败就退回固定小 `alpha`，不增加 learned Gate。

以下旧 observability 设计保留为历史实现说明。

当前点云可靠时，更信 observation feature；当前点云稀疏、遮挡或 gap 较大时，更信 timestamp-conditioned dynamics prior。

第一版门控输入只用稳定可得的量：

```text
o_t = [
  log1p(num_points_in_search),
  log1p(soft_fg_count),
  mean_fg_score,
  valid_history_ratio,
  current_delta_t / time_scale
]
```

这里的 `num_points_in_search` 必须是 regularize 之前的当前搜索区域真实点数；`soft_fg_count / mean_fg_score` 只统计当前帧 chunk，避免把历史点云质量混进当前观测可靠性。

不要依赖复杂的 `res_hist`、`occ_est` 或手工遮挡估计器。P5 应暂时写成 **observability-aware observation/dynamics fusion 的诊断候选**，而不是已验证主贡献，更不是“首次解决稀疏或遮挡”。CXTrack、MBPTrack、MVCTrack、HVTrack 已经分别从 context、memory、virtual cues 和 high temporal variation 角度处理过相关困难；CT-SeqTrack 的边界是用真实时间动力学 prior 去补当前观测可靠性变化。

---

## 3. 与已有工作的关系

### 连续时间升级路线图

MambaTrack3D、HVTrack、TrajTrack 和通用 Kalman/连续时间状态估计可以作为 future work 的路线图，而不是当前第一版方法的组成部分。当前检索未找到名为 `TrackM3D` 的可核验 3D SOT 论文，因此删除该名称，避免误引。

| 方法族 | 可升级方向 | 对当前 CT-SeqTrack 的启发 | 当前是否采用 |
| --- | --- | --- | --- |
| MambaTrack3D / SSM | 用真实 `delta_t` 替换固定离散步长，例如 `A_bar = exp(delta_t * A)` | 说明 fixed-step SSM 可以自然扩展到 variable-rate temporal modeling | 不采用，作为 future work |
| Kalman / continuous-discrete state estimation | 用显式 `F(delta_t)` 和 uncertainty propagation 替换固定步长转移 | 支持 dual-clock state prior、proposal innovation 与 trajectory tube | 条件采用：仅在 M0-M2 oracle/time-control 通过后 |
| HVTrack / attention memory | 用连续 timestamp encoding 替代 frame-index positional encoding | 支持当前 `TimeEncoding(raw/mlp/fourier)` 的设计动机 | 部分采用：只做 scalar-preserving 时间编码 |
| TrajTrack / trajectory prior | 用历史 bbox 形成 global proposal，并与 local observation proposal 做一致性判断 | 支持低维 bbox-only dynamics proposal；同时要求 refinement 严格 GT-free | 只借鉴 proposal 关系，不复制完整 TrajFormer |

因此 related work 中可以承认：连续时间动力系统、variable-`Delta t` SSM、Neural ODE/SDE/CDE 都是合理扩展；但 CT-SeqTrack 的贡献更窄，聚焦在现有 Seq2Seq 3D SOT 框架内检验真实 timestamp、continuous-discrete proposal innovation 和 endpoint-conditioned path distillation。旧 Observability Gate 已停止；可校准 uncertainty fusion 只是后置条件分支。

### SeqTrack3D

SeqTrack3D 是最直接的基线和继承对象。它已经做了多帧历史点云、历史框序列和 sequence-level constraint，甚至使用 continuous motion 的表述。

区别：

- SeqTrack3D 的时间窗口是固定帧数，历史帧默认等间隔。
- box corner timestamp 是固定伪时间。
- 没有使用真实 `delta_t` 解释 2Hz、10Hz、skip、掉帧之间的差异。
- 没有 time-resampling consistency。
- 其论文消融中 `1+3` 历史窗口优于更长的 `1+5/1+7`，作者将长历史退化部分归因于随机框扰动难以模拟测试误差和历史误差累积；这与 CT 当前的 candidate 伪速度、强 gap 后期崩落诊断高度相关。

因此不能 claim “首次使用历史序列”，而要 claim：

```text
We study within-track variable-rate LiDAR 3D SOT by conditioning a
Seq2Seq tracker on physical elapsed time.
```

### StreamTrack

StreamTrack 已经提出 continuous stream / memory bank。

区别：

- StreamTrack 的 continuous 更像 streaming memory，不是物理时间或真实 `delta_t`。
- 它使用 learnable temporal embedding 区分历史顺序，不强调真实时间间隔。
- 它没有构造 variable-rate / time-resampling consistency 的监督目标。

写 related work 时要正面承认它，并强调：它建模连续输入流，CT-SeqTrack 建模真实连续时间间隔。

### HVTrack / MambaTrack3D

HVTrack 已系统讨论 high temporal variation；MambaTrack3D 已把 SSM/Mamba 用到 HTV 3D SOT。

区别：

- HVTrack 的核心是 memory、context attention 和 noise suppression。
- HVTrack 已用不同固定 frame interval 构造 KITTI-HV，因此普通 skip-frame / HTV protocol 不能再作为 CT-SeqTrack 的独立创新；CT 必须强调 tracklet 内部不规则 `delta_t`、一个模型跨 cadence，以及 unseen schedule。
- MambaTrack3D 会削弱“用 SSM 解决高时变”的新意。
- CT-SeqTrack 第一版不把 HTV 或 Mamba 作为核心贡献，而是在 SeqTrack3D 上验证真实时间字段、TWC 和观测-动力学融合。

Mamba variable-`Delta t` SSM 可以作为 future work：如果后续做第二篇或扩展版，可以把 Mamba 的固定离散化改成真实 `delta_t` 条件下的状态转移；当前不要把 matrix exponential / SSM 作为主贡献，否则会稀释 CT-SeqTrack 的清晰边界。

### TrajTrack

TrajTrack 已把历史 box trajectory 做成轻量轨迹先验。

区别：

- TrajTrack 更偏历史框轨迹到未来修正。
- CT-SeqTrack 同时保留当前点云观测、历史点云、历史框和真实时间。
- CT-SeqTrack 要证明的不是“历史轨迹有用”，而是“真实时间间隔改变了历史轨迹的解释方式”。
- TrajTrack 论文描述的是 local/global proposal IoU 驱动的 refinement；当前本地 `pre_w_refine()` 却读取当前帧 GT overlap 触发 refinement，并用 GT overlap 选择 proposal。该实现只能作为 oracle upper bound，公平复现必须改用 `pre_wo_refine()` 或 GT-free paper-aligned evaluator。
- 对 CT 最有用的不是复制 VAE/完整 TrajFormer，而是把低维 bbox trajectory 做成真实 `delta_t` 条件的 proposal，再用测试时可得的 proposal agreement、点数和置信度做有界修正。

### Motion-to-Matching / motion-centric trackers

Motion-to-Matching、M²-Track、DMT 和 FlowTrack 已经覆盖“历史运动先验、粗定位、点云匹配或 refinement”这一大类思路。它们会直接削弱泛泛的“首次用运动先验修正 observation”主张。CT-SeqTrack 只有在以下组合同时成立时才有可辨识边界：物理 `delta_t` 条件、序列内部不规则采样、observation-first 近零初始化有界 residual、以及同一 endpoint 的重采样一致性。

### ChronoTrack

ChronoTrack 已经接近 temporally consistent long-term memory 叙事。

区别：

- CT-SeqTrack 的 TWC 必须限定为“不同采样路径到同一绝对时刻”的一致性。
- 不要泛称 temporal consistency，也不要写成长时记忆一致性。

### P2P / CXTrack / MVCTrack / PillarTrack / P2B / SC3D

- P2P 是强 motion-centric baseline，主要是双帧 part-to-part motion。
- CXTrack 说明上下文能抗遮挡和 distractor；Observability Gate 不能写成首次解决遮挡。
- MVCTrack 证明多模态 virtual cues 可提升稀疏场景；CT-SeqTrack 应强调纯 LiDAR、无额外模态。
- PillarTrack 是表示和效率方向，不直接冲突。
- P2B / SC3D 主要用于 related work 背景。

---

## 4. 方法路线

### 当前快照

当前仓库已经完成 P0-P5 工程链路，并新增 bounded residual 与 corrected-TWC。corrected-TWC 已完成服务器 seed42 训练，证明坐标修复路径生效；bounded residual 已完成 standard 真实 batch warmup/active forward-loss-backward，但默认量级近乎为零，尚未完成强 gap、完整 split、2-step optimizer 或性能验证。各模块仍通过显式 YAML 开关启用。

已有实验已经完成一轮关键收敛：raw / MLP / Fourier real-time 主干都不稳定；恢复 order-time 主干后，`A1-order` 基本修复崩坏；feature-concat `A2-order-dyn` 不仅有 seed sensitivity，也有明显 protocol dependence。crop oracle 证明高速目标会在模型 forward 前离开 base crop，P0-B2 又否定 raw predicted-history CV 恒开启。P0-B3 的 observation-quality risk signal 未通过 P0-B4 独立验证，raw-CV passive union gain 不足且 selector 跨强协议失效；当前 state anchor 已在实现前停止。后续主线不再堆主干时间编码或学习式 gate，而是先完成 proposal oracle、candidate 伪速度和冻结 path-variance 诊断，再决定是否解锁新的 dual-clock/innovation 机制。

### P0-P2：已完成地基

- 训练侧和测试侧都已输出真实时间字段。
- `seqtrack3d.py` 已用 `create_corner_timestamps_from_deltas(delta_T)` 替代固定伪时间。
- point time 和 box corner time 共用同一个 `TimeEncoding`。
- `raw / mlp / fourier` 已通过 smoke test。

详细验收见 `done.md`。

### P3：Dynamics / Velocity Branch

feature-concat P3 已完成过服务器 smoke test；新的 `residual_limited` 路径已完成 standard 真实 batch 数值验收，但默认 correction 与 gate gradient 近乎为零。P0-C-D1 full 中，true 相对 fixed 只有 `+0.438/+0.523`，相对 shuffled 为 `-0.123/+0.056`，Success/Precision 的逐 tracklet bootstrap 95% CI 均跨 0；因此 `A2-order-dyn` 只保留为失败消融，不扩展 cadence/seed。模型对时间有数值响应（相对两个控制各有 `1079/1257` 个 endpoint 的中心改变），但正确对应关系没有稳定收益。TWC A/B/C 也显示 C 无法恢复到 single-view A，不扩展 seed。P0-B2/P0-B3 已证明 raw predicted-history proposal 缺少互补性，P0-B4 又否定当前 reliability 入口；不再实现 state anchor。下一步只做冻结 A/B/C 输出、candidate 审计，并决定 residual oracle 是否值得一次性执行。

第一版只做真实时间差分动力学：

```text
v_i     = (c_i - c_{i-1}) / delta_t_i
omega_i = wrap(theta_i - theta_{i-1}) / delta_t_i
```

feature 模式用小 MLP 编码成 `z_dyn` 并接到 coarse motion prediction 前；residual 模式不拼接 `z_dyn`，而是保留 `velocity_pred / dynamics_displacement_pred`，让最终中心只接受 clamp、alpha、scale 和 warmup 共同限制的 dynamics residual。

注意：不要把 P3 写成完整 continuous dynamics solver。它只是把历史框差分从 frame-step 解释改成 real-time velocity / angular velocity 解释，为 P5 的 dynamics prior 提供一个轻量、可消融的时间条件分支。

### P4：TWC

双视图保持同一个当前帧和同一个最近历史 anchor，只改变更早的历史采样路径：

```text
view A: [t-1, t-2, t-3] -> t
view B: [t-1, t-3, t-5] -> t
```

先约束最终当前框，不约束所有历史框。第一版 TWC 必须满足以下边界：

- 两个 view 的 `current_timestamp` 相同。
- 两个 view 的预归一化 `coordinate_anchor` 相同，因为 SeqTrack3D 的当前搜索区域和输出框坐标系由最近历史框决定；不能再用归一化后的 `ref_boxs[0]` 代替检查。
- 共同绝对历史帧复用同一个 candidate offset，且 candidate 1/2/3 必须参与验收。
- 共同历史帧和当前帧复用 point regularization seed，最终输入 XYZ 应一致；时间 token 仍按各自历史路径构造。
- 两个 view 的 `delta_T` 至少在旧历史位置不同，保证约束来自重采样路径差异。
- 早期 padding 或任一 view 历史不完整时，不计算 TWC，只保留 supervised loss。
- `L_a` 和 `L_b` 取平均后再加 `lambda_twc * L_twc`，避免 paired view 把监督项权重翻倍。

这个设计能让 P4 的实验解释更干净：如果 TWC 有收益，应来自模型对不同真实时间采样路径的稳定性提升，而不是来自额外 batch 大小、额外 crop 扰动或坐标系变化。

当前状态：旧 validity-fixed 消融后来被证明仍存在 nonzero candidate 坐标污染，旧数值已撤回。共享绝对 frame offset、`coordinate_anchor` fail-fast 和 optimizer-step 对齐已实现；同提交 single-view、paired-view weight0、corrected-TWC 控制组已完成。`C-B` 为正但 `C-A` 明显为负，故主方法 promotion No-Go；只保留冻结 checkpoint 的 strong-cadence/path-variance 收尾。

### P5：Observability Gate

基于当前点数、前景概率、历史有效比例和 `current_delta_t`，在 observation feature 和 dynamics prior 之间做二路 softmax 融合。

第一版 P5 必须保持轻量和可解释：

- 默认关闭，且启用时显式依赖 P3 dynamics branch。
- 不改变 Transformer refine，不引入 memory bank，不引入多模态虚拟点。
- gate 输出保持 256 维，复用原始 `motion_mlp`，避免把收益混到新的 motion head 里。
- 训练初期偏向 observation：`gate_mlp` 最后一层 bias 可初始化为 `[1.0, 0.0]`。
- 当 `dynamics_valid=0` 时强制使用 observation，避免 padding 历史进入 dynamics prior。

推荐实现语义：

```text
z_dyn_256 = Linear(z_dyn)
alpha = softmax(MLP(o_t)) -> [alpha_obs, alpha_dyn]
alpha_dyn = alpha_dyn * dynamics_valid
alpha = renormalize(alpha)
motion_feature = alpha_obs * point_feature + alpha_dyn * z_dyn_256
motion_pred = motion_mlp(motion_feature)
```

P5 的核心验收不只是 loss finite，而是 gate 行为可解释：稀疏、大 gap、低前景置信度样本中 `alpha_dyn` 应更高；当前观测清晰时 `alpha_obs` 应更高。

当前状态：旧 P5 full 结果不能作为最终 gate 结论，因为它混入了 raw real-time 主干失败路径。新的 `A3-order-gate-safe` 比旧 P5 full 安全很多，但 final 仍低于 `A2-order-dyn`；`A3-order-conf-res-gate` best-e14 复测没有复现旧 best，高点暂不能作为稳定收益。后续应先核对评测路径，并做 sparse / delta_t / foreground confidence / alpha-residual 分桶，再决定是否继续 gate。

---

## 5. 实验设计

### 5.0 P0-B4 决策与 benchmark pivot

P0-B4 已完成 independent mini_val frozen evaluation：

```text
gap1124 AUROC / recall    = 0.680 / 0.568
burst AUROC / recall      = 0.712 / 0.609
required                  = 0.750 / 0.700
verdict                   = NO_GO_OBSERVATION_RELIABILITY_VALIDATION
```

路线约束：

- 不在 mini_val 上重调 feature、L2、threshold 或 crop scale；P0-B4 No-Go 永久保留。
- 不实现当前 calibrator 控制的 frozen-state、active dual-anchor 或 learned selector。
- P0-C 的 stable manifest、输入公平性、同 checkpoint 三路性能及 endpoint-level D1 已完成；A2 true-dt promotion No-Go，不扩展 schedule/multiseed。D1 还确认 gap/高位移分桶无可推广 physical-time 优势，overall mean-error 表面收益受灾难性长尾驱动。
- `true/fixed/shuffled-dt` 改为通用 benchmark 因果控制，不再作为复活当前 reliability anchor 的工具。
- residual 和 TWC 各只允许一次预注册、同提交的 seed42 机制控制；失败即停止。
- 选择/融合规则仍只能使用推理时可得量；GT 只用于离线标签、loss 和 oracle 分析。

当前更稳的论文价值是把失败路径系统化：强 gap 会改变 crop reachability，raw prediction history 会形成灾难性长尾；开发集上的 risk proxy 未能独立泛化，raw-CV candidate 也没有稳定互补性。项目应先把这些现象固化为可复现的 variable-rate 3D SOT benchmark/diagnosis，再让任何时间模块在 held-out cadence 和负对照下证明自己。

### 主表

至少保留普通主表，但不要只依赖普通主表：

- nuScenes
- Waymo

KITTI 可作为补充，不宜作为主战场，因为 KITTI 对高时变和稀疏的体现不如 nuScenes/Waymo。

新增主表建议：

```text
standard fixed-step evaluation
virtual-rate evaluation: gap1124 / burst_drop / random20
fixed-dt / shuffled-dt / jittered-dt negative controls
long-delta_t bins
sparse / re-appearance bins
```

如果普通 fixed-step final 没有稳定全面超过 baseline，但 variable-rate / long-gap / sparse 子集有稳定收益，这仍然可以支撑论文叙事；反过来，如果只在普通 fixed-step 上追全局涨点，当前证据不足。

### 消融表

已经完成的关键消融：

```text
SeqTrack baseline
P5 full
A1-raw / A2 raw-dyn
A1-pseudo / A1-MLP / A1-Fourier
A1-scaled / A2-scaled-dyn
A1-order / A2-order-dyn
A2-order-dyn-cand1 / A2-order-dyn-disp
A1-order+TWC / A2-order-dyn+TWC（历史 run，TWC 归因失效）
A3-order-gate-safe / A3-order-conf-res-gate
A3-conf-res best-e14 retest
A2-order-dyn seed43 / seed44
A2-order-dyn+TWC w0.01
A3-conf-res rerun seed42
```

下一步优先复核：

```text
A/B/C final checkpoint 的 standard/gap1124/burst-drop/unseen-fixed-gap endpoint 与 path variance（不重训）
crop-reachable residual oracle convex-blend feasibility
candidate-wise dynamics 与 target-in-crop diagnostics
```

P0-C-D1 已回答旧 feature-concat A2 的 paired failure localization：时间输入会改变预测，但 true alignment 没有超过 shuffled，且均值误差受长尾主导。剩余实验的作用不是复活已经 No-Go 的 reliability anchor，而是回答 residual 在 reachable subset 是否有必要、candidate jitter 是否制造伪速度，以及 TWC 的 `C-B` 是否在强协议和 held-out path variance 上仍成立。当前 TWC 已确认 paired control 内的单 seed 净效应但未超过 single-view A，residual 没有性能正结论，dual-anchor 已停止。

### 困难子集

`delta_t` bins：

```text
[0, 0.2), [0.2, 0.5), [0.5, 1.0), [1.0, +inf)
```

variable-gap：

```text
skip = 1, 2, 3, 5
```

sparse bins：

```text
[0, 5), [5, 10), [10, 20), [20, 50), [50, +inf)
```

re-appearance：

- 目标点数连续低于阈值。
- 之后点数恢复。
- 统计恢复后 K 帧内是否重新跟上。

核心实验假设：

- 标准主表不能明显退化。
- `delta_t` 越大，相比 fixed-time baseline 的优势越明显。
- sparse / occlusion / re-appearance 子集更稳定。
- 冻结 corrected-TWC 只检查是否降低路径方差；新的方法收益假设属于 asymmetric path distillation，不能沿用当前对称 TWC 的结果。

---

## 6. Related Work 草稿

```text
Existing 3D SOT methods have explored appearance matching, point-to-box proposals,
context modeling, part-to-part motion cues, sequence modeling, memory-based
tracking, trajectory priors, and high-temporal-variation protocols. However,
most of them still interpret historical observations as fixed-step frame
sequences. CT-SeqTrack separates the discrete order clock from physical
elapsed time: the former preserves sequence identity, while the latter
controls continuous-discrete state propagation, search support, and
endpoint-conditioned path distillation under irregular cadence.
```

当前不做：

- Matrix exponential / variable-`Delta t` Mamba SSM。
- Neural ODE / SDE / CDE。
- 任意时刻查询 `state(t*)`。
- 多传感器异步融合。
- 完整 uncertainty diffusion。
- 复杂 memory bank。
- future head。

这些可作为 related work 或 future work，用来说明 CT-SeqTrack 是克制的第一步。

### Future Work 表述草稿

```text
Our current formulation treats timestamps as first-class inputs while keeping
the Seq2Seq tracking architecture lightweight and directly comparable to
SeqTrack3D. A natural future direction is to replace fixed-step temporal
modules in SSM-, Kalman-, or trajectory-based trackers with variable-rate
continuous-time transitions, enabling arbitrary-time state queries and
uncertainty propagation across long observation gaps.
```

---

## 7. 阅读依据

本计划基于以下本地文本、PDF 逐页视觉核查和近期检索工作整理：

- `_extracted_text/SeqTrack3D.txt`
- `_extracted_text/P2P.txt`
- `_extracted_text/CXTrack.txt`
- `_extracted_text/MVCTrack.txt`
- `_extracted_text/PillarTrack.txt`
- `_extracted_text/P2B.txt`
- `_extracted_text/SC3D.txt`
- SeqTrack3D: https://arxiv.org/abs/2402.16249
- StreamTrack: https://arxiv.org/abs/2303.07605
- HVTrack: https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/1145_ECCV_2024_paper.php
- MambaTrack3D: https://arxiv.org/abs/2511.15077
- TrajTrack: https://arxiv.org/abs/2509.11453
- ChronoTrack: https://openaccess.thecvf.com/content/CVPR2026F/html/Yoo_Temporally_Consistent_Long-Term_Memory_for_3D_Single_Object_Tracking_CVPRF_2026_paper.html
- Neural Continuous-Discrete State Space Models: https://proceedings.mlr.press/v202/ansari23a.html
- ContiFormer: https://proceedings.neurips.cc/paper_files/paper/2023/hash/9328208f88ec69420031647e6ff97727-Abstract.html
- Motion-to-Matching: https://arxiv.org/abs/2308.11875
- M3SOT: https://ojs.aaai.org/index.php/AAAI/article/view/28152
- FlowTrack: https://arxiv.org/abs/2407.01959
