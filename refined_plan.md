# CT-SeqTrack 研究计划与论文定位

更新时间：2026-07-17

这份文件用于每次开始工作前快速整理研究思路。下一步执行清单见 `need_to_do.md`，已完成工程和实验记录见 `done.md`，简洁实验结论见 `sum_results.md`。

---

## 1. 最终定位

当前方向继续成立，但论文定位必须收窄：

**真实时间感知的 3D 单目标跟踪：在不规则采样、掉帧和长时间间隔下，用真实 `delta_t` 驱动历史状态、当前观测和序列约束。**

更稳的关键词：

```text
timestamp-native / variable-rate / time-aware 3D SOT
```

一句话主线：

**真实 timestamp 改变了历史状态的物理含义；CT-SeqTrack 研究序列内部不规则采样，并用物理 `delta_t` 条件化轻量运动先验与重采样一致性。**

可考虑标题：

- CT-SeqTrack: Timestamp-native Sequence Modeling for 3D Point Cloud Tracking
- Variable-rate 3D Single Object Tracking with Time-resampling Consistency
- Timestamp-aware Sequence Tracking for 3D Single Object Tracking

不要把主 claim 写成：

- “continuous motion modeling”：容易撞 StreamTrack。
- “high temporal variation”：容易撞 HVTrack / MambaTrack3D。
- “historical trajectory prior”：容易撞 SeqTrack3D / TrajTrack。
- “first sparse/occlusion solution”：容易撞 CXTrack / MVCTrack。

当前实验边界：

- 已有结果支持：主干保留 SeqTrack3D 的 order-time 语义，同时把真实 `delta_t/current_delta_t` 注入 `DynamicsEncoder`，比直接替换主干时间 token 更稳定。
- 目前不能宣称：完整 CT-SeqTrack full model 已经稳定超过 SeqTrack3D。
- corrected-TWC seed42 已完成：A1 相对配置级 baseline 为 `+1.49 Success / +5.03 Precision`，A2 为 `-0.93 / -2.07`，两组 anchor/current XYZ gap max 都为 0。A1 是待多 seed 复现的候选贡献，不能升级为稳定结论；A2 暂不组合 TWC。
- 目前不能宣称 observability gate 已经带来稳定最终收益；gate-safe 低于 A2，conf-res best-e14 复测未复现旧 best。
- bounded residual 已完成 standard 真实 batch 回归：warmup 与 active forward/loss/backward finite，但默认实际 correction 只有约 `1e-7 m`、gate 梯度极小，未通过功能验收，也没有性能结果。
- HTV 六组筛选显示旧 feature-concat A2 只在 random20 上为正，在 gap1124/burst-drop 上明显退化。因此“真实时间分支已解决强不规则跟踪”不成立，必须先验证 residual、candidate 监督与 crop 可达性。
- standard/gap1124/burst-drop crop oracle 显示 previous-GT base recall 为 85.41%/76.78%/77.72%；强协议下 2x expanded 也只有 89.08%/87.65%，GT-history CV recenter 则为 98.96%/99.05% 且不增加背景点。当前最直接的机制瓶颈已经从“末端如何融合”前移到“目标能否进入 search crop”。
- P0-B2 recursive predicted-history 显示 raw CV 相对 previous-A1 只提高 2.65–3.03 pp，未通过预注册门槛；但预测历史可靠时 pred-CV recall 可达 97.34%–98.64%。因此恒开启单锚点 recenter 已 No-Go，可靠性控制的预防性第二锚点仍值得验证。
- TrajTrack 本地 aligned run 的 evaluator 使用当前帧 GT 触发和选择 refinement；64.94 / 79.07 只能作为 oracle-assisted 诊断，不能进入公平主表。

当前执行策略：

```text
1. 扩展递归诊断，验证测试时 confidence/foreground/empty/CV shift/agreement 能否预测漂移与 next-crop failure。
2. 只有可靠性代理有效时，固定 A1 checkpoint 做无训练 active dual-anchor，优先减少首次失控与连续失败。
3. active 机制通过后冻结 HTV manifest，实现 true/fixed/shuffled-dt；只在 crop-reachable subset 重新校准 residual。
4. corrected-TWC、GT-free TrajTrack、学习式 gate/trajectory encoder 与多 seed 后置。
```

当前最可防御的新颖性不是单独的“timestamp”“HTV”“运动先验”或“temporal consistency”，而是它们的窄组合：**同一 tracklet 内不规则物理时间间隔、同一模型对未见 cadence/drop schedule 的泛化、有界 observation-first time-conditioned trajectory correction，以及同一 endpoint 的历史重采样一致性**。

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

## 2. 收敛后的三个贡献

### 贡献 1：Timestamp-native 输入契约与 variable-rate 评测

CT-SeqTrack 先把 SeqTrack3D 的输入契约扩展为真实时间感知，而不是简单把所有主干时间 token 都替换为真实秒数：

- 训练和测试都提供一致的 `timestamps / delta_t / delta_T / current_delta_t`。
- 点特征时间通道和 box corner token 工程上支持 `raw / mlp / fourier` 时间编码。
- 已有消融显示，直接把 real-time token 放进 SeqTrack3D 主干会破坏原始 order-time 语义。
- 第一批 `gap1124 / burst_drop / random20` virtual-rate protocol 已完成六组 A1/A2 seed42 筛选。结果具有明显 protocol dependence；后续必须冻结 manifest，并补 `fixed-dt / shuffled-dt` negative control、delta_t bins、sparse bins、displacement bins 和 re-appearance 片段，证明真实时间不是装饰字段，而是 variable-rate 3D SOT 的任务条件。

这仍然是 timestamp-native 的地基：真实时间进入数据、监督和评价协议。论文叙事要避免把失败的 raw main-branch 注入方式写成最终方法，也不要只在普通 fixed-step final 上判断方向成败。

### 贡献 2：Timestamp-conditioned trajectory guidance and bounded refinement

当前更稳的模型主线是：主干保持 order-time，真实 `delta_t/current_delta_t` 进入 `DynamicsEncoder`。历史框差分按真实 `delta_t` 计算速度和角速度，形成 timestamp-conditioned dynamics prior。

旧版 `A2-order-dyn` 通过 feature concat 把 `z_dyn` 接入 coarse motion branch，seed42 有 precision-positive 信号，但 seed43/44 暴露了稳定性风险。当前代码已经实现更保守的 residual 形式：

```text
obs_pred = observation_branch(point_feature)
dyn_disp = velocity_pred * current_delta_t
dyn_disp = clamp_norm(dyn_disp, max_norm)
final_center = obs_center + scale * alpha * dyn_disp
```

2026-07-17 的诊断改变了接入优先级：上述末端 residual 默认实际修正只有约 `7e-8 m`，而 previous-GT base crop 在 standard 高速样本上已经丢失目标。输出端 correction 无法恢复从未进入网络的点，因此第一修正位置应前移到 search crop：

```text
previous predicted box        -> observation anchor c_prev
history boxes + real delta_t  -> clipped trajectory anchor c_traj
test-time reliability         -> choose / conservatively fuse two crops
SeqTrack3D observation branch -> local refinement c_obs
```

第一版 `c_traj` 仍应使用 clipped constant-velocity/Kalman，而不是直接增加大网络。P0-B2 已否定把它恒开启为唯一 anchor；只有测试时可靠性代理和无训练 active dual-anchor 均有效，才能升级为正式模块。GT-history CV 只保留为 oracle upper bound。

这个设计的好处是：

- observation branch 仍是主预测，避免 dynamics feature 过早接管。
- `scale / alpha / clamp / warmup` 都可控，便于解释 seed collapse。
- 可以只在 long-delta_t / sparse / low-confidence 分桶启用或报告收益。

当前论文里应把候选 dynamics 写成 **reliability-aware timestamp-conditioned trajectory guidance with observation refinement**，而不是完整连续动力学求解器。HTV 六组已说明 feature concat 在强 gap/burst 下不稳定；P0-B2 又说明 raw CV 不能独立恢复漂移。只有 active dual-anchor、因果时间对照和多 seed 通过后，才能升级为论文贡献。现有末端 residual 继续作为消融。

### 贡献 3：Time-resampling Consistency

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

训练时用两个 view 的 supervised loss 平均值，而不是简单相加，避免开启 TWC 后把主监督梯度放大：

```text
L = 0.5 * (L_a + L_b) + lambda_twc * L_twc
```

这个贡献必须写窄：不是泛泛 temporal consistency，而是 **time-resampling consistency under different sampling paths to the same absolute time**。corrected seed42 已确认坐标修复有效，并在 A1 上形成单 seed 正信号；但旧 baseline 不是同提交配对，且还缺 seed43/44，所以 TWC 仍是待复现的候选贡献，不是 A2+dynamics 主配置。

### 候选扩展：Observability-aware Fusion

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
| Kalman / continuous-time state estimation | 用连续不确定性传播替换固定步长转移 | 提醒遮挡和长 gap 下不确定性应随时间累计 | 不采用，避免复杂 SDE |
| HVTrack / attention memory | 用连续 timestamp encoding 替代 frame-index positional encoding | 支持当前 `TimeEncoding(raw/mlp/fourier)` 的设计动机 | 部分采用：只做 scalar-preserving 时间编码 |
| TrajTrack / trajectory prior | 用历史 bbox 形成 global proposal，并与 local observation proposal 做一致性判断 | 支持低维 bbox-only dynamics proposal；同时要求 refinement 严格 GT-free | 只借鉴 proposal 关系，不复制完整 TrajFormer |

因此 related work 中可以承认：连续时间动力系统、variable-`Delta t` SSM、Neural ODE/SDE/CDE 都是合理扩展；但 CT-SeqTrack 的贡献更窄，聚焦在现有 Seq2Seq 3D SOT 框架内检验真实 timestamp、bounded residual 和 endpoint resampling consistency。Observability-aware fusion 目前只是诊断候选。

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

已有实验已经完成一轮关键收敛：raw / MLP / Fourier real-time 主干都不稳定；恢复 order-time 主干后，`A1-order` 基本修复崩坏；feature-concat `A2-order-dyn` 不仅有 seed sensitivity，也有明显 protocol dependence。crop oracle 证明高速目标会在模型 forward 前离开 base crop，P0-B2 又否定 raw predicted-history CV 恒开启。后续主线不再继续堆主干时间编码或学习式 gate，而是先完成测试时可靠性诊断、无训练 active dual-anchor、冻结 variable-rate 协议和 reachable-subset residual 诊断。

### P0-P2：已完成地基

- 训练侧和测试侧都已输出真实时间字段。
- `seqtrack3d.py` 已用 `create_corner_timestamps_from_deltas(delta_T)` 替代固定伪时间。
- point time 和 box corner time 共用同一个 `TimeEncoding`。
- `raw / mlp / fourier` 已通过 smoke test。

详细验收见 `done.md`。

### P3：Dynamics / Velocity Branch

feature-concat P3 已完成过服务器 smoke test；新的 `residual_limited` 路径已完成 standard 真实 batch 数值验收，但默认 correction 与 gate gradient 近乎为零。正式实验中，`A2-order-dyn` 仍只是值得诊断的真实时间使用方式，不是稳定结论。P0-B2 已证明 dynamics proposal 不能恒开启替换 previous anchor；下一步先验证它能否作为可靠性控制的第二 crop 假设预防首次漂移，再在冻结协议下做同容量 true/fixed/shuffled-dt 对照。

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

当前状态：旧 validity-fixed 消融后来被证明仍存在 nonzero candidate 坐标污染，旧数值已撤回。共享绝对 frame offset、`coordinate_anchor` fail-fast 和 optimizer-step 对齐已实现，corrected A1+TWC seed42 得到配置级正信号，但 baseline 不同提交且只有单 seed；它只能作为候选稳定性贡献。当前先解决 crop/causal 主线，之后再做同提交的 single-view、paired-view weight0、corrected-TWC 控制组。

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

### 5.0 下一步方法决策：Reliability-aware dual-anchor trajectory guidance

下一阶段先验证一个无需训练、可解释的双锚点主动推理机制：

```text
observation anchor c_prev = previous A1 predicted box
trajectory anchor c_dyn   = clipped CV/Kalman(predicted history, real delta_t)
two search crops          = forward around c_prev and c_dyn
test-time reliability     = confidence / foreground / empty / agreement
final observation         = select or conservatively fuse GT-free proposals
```

设计约束：

- observation branch 仍负责最终 refinement；trajectory proposal 只作为第二个预防性搜索假设。
- 第一版先用 constant-velocity/Kalman，输入只含递归预测 bbox center/heading 和真实 `delta_t`；不上 tiny MLP/GRU、VAE、Mamba、ODE/CDE。
- 选择/融合规则只能使用推理时可得量，不读取当前 GT；GT 只用于离线标签、loss 和 oracle 分析。
- base、expanded、recentered 必须共享 endpoint、训练步、模型容量和 checkpoint 规则；expanded 只作为 crop-size 控制组。
- P0-B2 已完成 previous-A1-pred 与 A1-pred-history-CV reachability，并判定 always-on raw CV No-Go。
- 先验证测试时 reliability proxy，再做无训练 active dual-anchor；任一失败都不升级学习式 gate/trajectory encoder。
- active dual-anchor 通过后再完成 true/fixed/shuffled-dt；末端 residual 只在 reachable subset 重新校准。
- 与 TrajTrack 的区别必须落在真实物理时间、同一 tracklet 内不规则 cadence、unseen schedule 泛化和 bounded observation-first correction 上。

这个方向同时解释当前数据：random20 的正信号说明低维运动先验可能有用；三协议 crop oracle 证明 target-out-of-crop，固定 2x expanded 仍不足；P0-B2 则显示预测历史可靠时 CV 很强、漂移后几乎无效。因而目标不是让 CV 从灾难性漂移中恢复，而是在漂移前通过双锚点和 observation confidence 保持可达性。若测试时信号不能区分这两个状态，应停止该方向。

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
test-time reliability signals -> drift / next-crop-failure AUROC、AUPRC、calibration
previous-anchor vs clipped-CV/Kalman dual-anchor active inference（同 endpoint/checkpoint）
dual-anchor true-dt vs fixed-dt vs shuffled-dt
reachable-subset bounded residual vs no-residual
corrected A1-order+TWC vs paired-view weight0 vs single-view A1-order
TrajTrack pre_wo_refine vs GT-free paper-aligned refine vs oracle-assisted refine
candidate-wise dynamics 与 target-in-crop diagnostics
```

这些实验的作用不是重复证明 raw real-time 主干失败，而是回答：测试时信号能否在首次漂移前识别风险、dual-anchor 能否维持可达性、收益是否真的来自物理时间、residual 在 reachable subset 是否仍有必要，以及 TWC 是否降低同一 endpoint 的采样路径方差。当前 corrected-TWC 只有不完全配对的单 seed 信号，residual 没有性能正结论，dual-anchor 也尚无 active 在线证据。

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
- corrected-TWC 应能降低不同采样路径下的预测方差；这是待验证假设，不是当前结论。

---

## 6. Related Work 草稿

```text
Existing 3D SOT methods have explored appearance matching, point-to-box proposals,
context modeling, part-to-part motion cues, sequence modeling, memory-based
tracking, trajectory priors, and high-temporal-variation protocols. However,
most of them still interpret historical observations as fixed-step frame
sequences. CT-SeqTrack instead treats timestamps as first-class inputs and
learns variable-rate state estimation from real temporal intervals.
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
- Motion-to-Matching: https://arxiv.org/abs/2308.11875
- M3SOT: https://ojs.aaai.org/index.php/AAAI/article/view/28152
- FlowTrack: https://arxiv.org/abs/2407.01959
