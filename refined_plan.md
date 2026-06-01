# CT-SeqTrack 研究计划与论文定位

更新时间：2026-06-01

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

**真实 timestamp 改变了历史状态的物理含义；CT-SeqTrack 的贡献是让 3D SOT 从 fixed-step sequence learning 变成 variable-rate timestamp-native state estimation。**

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
- 目前不能宣称：TWC 和 observability gate 已经带来最终收益；二者还需要基于 `A1-order / A2-order-dyn` 重新消融。

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

### 贡献 1：Timestamp-native 输入契约与 dynamics prior

CT-SeqTrack 先把 SeqTrack3D 的输入契约扩展为真实时间感知，而不是简单把所有主干时间 token 都替换为真实秒数：

- 训练和测试都提供一致的 `timestamps / delta_t / delta_T / current_delta_t`。
- 点特征时间通道和 box corner token 工程上支持 `raw / mlp / fourier` 时间编码。
- 已有消融显示，直接把 real-time token 放进 SeqTrack3D 主干会破坏原始 order-time 语义。
- 当前更稳的主线是：主干保持 order-time，真实 `delta_t/current_delta_t` 进入 `DynamicsEncoder`。
- 历史框差分按真实 `delta_t` 计算速度和角速度，形成 timestamp-conditioned dynamics prior。

这仍然是 timestamp-native 的地基：真实时间进入数据、监督和动力学解释，但论文叙事要避免把失败的 raw main-branch 注入方式写成最终方法。

### 贡献 2：Time-resampling Consistency

同一条 tracklet 通过不同历史采样路径观察同一当前绝对时刻时，最终状态估计应该一致。这里的“不同采样路径”不是改变当前帧，也不是改变搜索区域，而是在共享最近历史 anchor 的前提下改变更早历史帧：

```text
view A: [t-1, t-2, t-3] -> t
view B: [t-1, t-3, t-5] -> t
```

这样两个 view 的预测仍位于同一个局部坐标系里，TWC 约束的就是历史时间路径差异，而不是坐标系或 crop 差异。

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

这个贡献必须写窄：不是泛泛 temporal consistency，而是 **time-resampling consistency under different sampling paths to the same absolute time**。

### 贡献 3：Observability-aware Fusion

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

不要依赖复杂的 `res_hist`、`occ_est` 或手工遮挡估计器。P5 的贡献应写成 **observability-aware observation/dynamics fusion**，而不是“首次解决稀疏或遮挡”。CXTrack、MBPTrack、MVCTrack、HVTrack 已经分别从 context、memory、virtual cues 和 high temporal variation 角度处理过相关困难；CT-SeqTrack 的边界是用真实时间动力学 prior 去补当前观测可靠性变化。

---

## 3. 与已有工作的关系

### 连续时间升级路线图

`claude_thinking.md` 提到的 MambaTrack3D、TrackM3D、HVTrack 和 TrajTrack 可以作为 future work 的路线图，而不是当前第一版方法的组成部分。

| 方法族 | 可升级方向 | 对当前 CT-SeqTrack 的启发 | 当前是否采用 |
| --- | --- | --- | --- |
| MambaTrack3D / SSM | 用真实 `delta_t` 替换固定离散步长，例如 `A_bar = exp(delta_t * A)` | 说明 fixed-step SSM 可以自然扩展到 variable-rate temporal modeling | 不采用，作为 future work |
| TrackM3D / Kalman-style motion | 用 Neural SDE 或连续不确定性传播替换离散转移矩阵 | 提醒遮挡和长 gap 下不确定性应随时间累计 | 不采用，避免复杂 SDE |
| HVTrack / attention memory | 用连续 timestamp encoding 替代 frame-index positional encoding | 支持当前 `TimeEncoding(raw/mlp/fourier)` 的设计动机 | 部分采用：只做 scalar-preserving 时间编码 |
| TrajTrack / trajectory prior | 用 Neural CDE 或 spline 表示连续轨迹 | 说明轨迹可被视为连续函数而非离散 waypoint | 不采用，避免变成 trajectory-prior 论文 |

因此 related work 中可以承认：连续时间动力系统、variable-`Delta t` SSM、Neural ODE/SDE/CDE 都是合理扩展；但 CT-SeqTrack 的贡献更窄，聚焦在现有 Seq2Seq 3D SOT 框架内证明真实 timestamp、time-resampling consistency 和 observability-aware fusion 的必要性。

### SeqTrack3D

SeqTrack3D 是最直接的基线和继承对象。它已经做了多帧历史点云、历史框序列和 sequence-level constraint，甚至使用 continuous motion 的表述。

区别：

- SeqTrack3D 的时间窗口是固定帧数，历史帧默认等间隔。
- box corner timestamp 是固定伪时间。
- 没有使用真实 `delta_t` 解释 2Hz、10Hz、skip、掉帧之间的差异。
- 没有 time-resampling consistency。

因此不能 claim “首次使用历史序列”，而要 claim：

```text
We convert Seq2Seq 3D SOT from fixed-step frame sequence learning
to variable-rate timestamp-conditioned state estimation.
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
- MambaTrack3D 会削弱“用 SSM 解决高时变”的新意。
- CT-SeqTrack 第一版不把 HTV 或 Mamba 作为核心贡献，而是在 SeqTrack3D 上验证真实时间字段、TWC 和观测-动力学融合。

Mamba variable-`Delta t` SSM 可以作为 future work：如果后续做第二篇或扩展版，可以把 Mamba 的固定离散化改成真实 `delta_t` 条件下的状态转移；当前不要把 matrix exponential / SSM 作为主贡献，否则会稀释 CT-SeqTrack 的清晰边界。

### TrajTrack

TrajTrack 已把历史 box trajectory 做成轻量轨迹先验。

区别：

- TrajTrack 更偏历史框轨迹到未来修正。
- CT-SeqTrack 同时保留当前点云观测、历史点云、历史框和真实时间。
- CT-SeqTrack 要证明的不是“历史轨迹有用”，而是“真实时间间隔改变了历史轨迹的解释方式”。

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

当前仓库已经完成 P0-P5 的工程链路和 smoke test，可以作为“真实时间字段 + dynamics + TWC + gate 都已落地”的研究快照。默认配置保持保守，各模块通过显式 YAML 开关启用，便于把性能变化归因到明确机制。

已有实验已经完成一轮关键收敛：raw / MLP / Fourier real-time 主干都不稳定；恢复 order-time 主干后，`A1-order` 基本修复崩坏，`A2-order-dyn` 在 final precision 上超过 baseline。后续主线不再继续堆主干时间编码，而是围绕 `A2-order-dyn` 验证 candidate noise、displacement 监督、TWC 和保守 gate。

### P0-P2：已完成地基

- 训练侧和测试侧都已输出真实时间字段。
- `seqtrack3d.py` 已用 `create_corner_timestamps_from_deltas(delta_T)` 替代固定伪时间。
- point time 和 box corner time 共用同一个 `TimeEncoding`。
- `raw / mlp / fourier` 已通过 smoke test。

详细验收见 `done.md`。

### P3：Dynamics / Velocity Branch

当前代码已实现 P3，默认关闭。服务器 `check_forward_batch.py` + loss smoke test 以及 `check_train_steps.py --max-steps 2` 均已通过。正式实验中，`A2-order-dyn` 是当前最强正向信号。

第一版只做真实时间差分动力学：

```text
v_i     = (c_i - c_{i-1}) / delta_t_i
omega_i = wrap(theta_i - theta_{i-1}) / delta_t_i
```

用小 MLP 编码成 `z_dyn`，接到 coarse motion prediction 前，并用轻量 `L_vel` 监督。

注意：不要把 P3 写成完整 continuous dynamics solver。它只是把历史框差分从 frame-step 解释改成 real-time velocity / angular velocity 解释，为 P5 的 dynamics prior 提供一个轻量、可消融的时间条件分支。

### P4：TWC

双视图保持同一个当前帧和同一个最近历史 anchor，只改变更早的历史采样路径：

```text
view A: [t-1, t-2, t-3] -> t
view B: [t-1, t-3, t-5] -> t
```

先约束最终当前框，不约束所有历史框。第一版 TWC 必须满足以下边界：

- 两个 view 的 `current_timestamp` 相同。
- 两个 view 的 `ref_boxs[0]` 相同或等价，因为 SeqTrack3D 的当前搜索区域和输出框坐标系由最近历史框决定。
- 两个 view 的 `delta_T` 至少在旧历史位置不同，保证约束来自重采样路径差异。
- 早期 padding 或任一 view 历史不完整时，不计算 TWC，只保留 supervised loss。
- `L_a` 和 `L_b` 取平均后再加 `lambda_twc * L_twc`，避免 paired view 把监督项权重翻倍。

这个设计能让 P4 的实验解释更干净：如果 TWC 有收益，应来自模型对不同真实时间采样路径的稳定性提升，而不是来自额外 batch 大小、额外 crop 扰动或坐标系变化。

当前状态：TWC 工程链路已经实现并通过 smoke test，但正式消融仍待完成。TWC 是否能作为第二个核心贡献，要看 `A1-order+TWC` 和 `A2-order-dyn+TWC`。

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

当前状态：旧 P5 full 结果不能作为最终 gate 结论，因为它混入了 raw real-time 主干失败路径。下一步只在 order-time 主干上测试保守 `A3-order-gate-safe`；如果仍退化，优先考虑 residual gate 和限制 `alpha_dyn`，而不是直接否定 observability-aware fusion。

---

## 5. 实验设计

### 主表

至少保留：

- nuScenes
- Waymo

KITTI 可作为补充，不宜作为主战场，因为 KITTI 对高时变和稀疏的体现不如 nuScenes/Waymo。

### 消融表

已经完成的关键消融：

```text
SeqTrack baseline
P5 full
A1-raw / A2 raw-dyn
A1-pseudo / A1-MLP / A1-Fourier
A1-scaled / A2-scaled-dyn
A1-order / A2-order-dyn
```

下一步优先消融：

```text
A2-order-dyn-cand1
A2-order-dyn-disp
A1-order+TWC
A2-order-dyn+TWC
A3-order-gate-safe
```

这些实验的作用不是重复证明 raw real-time 主干失败，而是回答：dynamics 是否被非 0 candidate 污染、是否需要 displacement 监督、TWC 是否独立有效、TWC 是否能和 dynamics 互补、gate 在干净主干上是否仍不稳定。

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
- TWC 能降低不同采样路径下的预测方差。

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

本计划基于以下本地文本和近期检索工作整理：

- `_extracted_text/SeqTrack3D.txt`
- `_extracted_text/P2P.txt`
- `_extracted_text/CXTrack.txt`
- `_extracted_text/MVCTrack.txt`
- `_extracted_text/PillarTrack.txt`
- `_extracted_text/P2B.txt`
- `_extracted_text/SC3D.txt`
- StreamTrack: https://arxiv.org/abs/2303.07605
- HVTrack: https://arxiv.org/abs/2408.02049
- MambaTrack3D: https://arxiv.org/abs/2511.15077
- TrajTrack: https://arxiv.org/abs/2509.11453
- ChronoTrack: https://arxiv.org/abs/2604.13789
- `claude_thinking.md`：连续时间 3D tracking 路线图与 future-work 边界整理。
