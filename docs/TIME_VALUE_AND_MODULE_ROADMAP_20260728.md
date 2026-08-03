# 真实时间价值、Random-20% 现实性与模块路线审计

更新时间：2026-07-28

状态：研究决策与实现审计。本文不把尚未完成的实验写成正结果。

## 1. 结论先行

### 1.1 真实时间值得保留，但暂时不值得作为论文唯一主贡献

把真实 `delta_t` 接入代码的增量代价已经不大：时间戳、dual-clock、
`true/fixed/shuffled` 控制、动力学编码和诊断路径都已经存在。真正昂贵的是把它
升级成论文标题级因果主张，因为需要同时证明：

1. 正常 cadence 不退化；
2. 在同一 checkpoint、同一 endpoint 下，`true-dt` 稳定优于
   `fixed-dt` 和 `shuffled-dt`；
3. 优势出现在 held-out irregular cadence，而不是只在某个训练过的丢帧模式；
4. 多 seed、full nuScenes 和第二数据集仍成立；
5. 增益不是额外训练预算、初始化、搜索区域、候选数量或 checkpoint 选择造成的。

当前 standard 的 `delta_t=0.4974±0.0228 s`，CV 只有 `4.59%`。在这个分布上，
一个真实时间分支很容易退化成近似常量函数。当前创新半径
`R(dt)=min(0.5+0.5dt, 2.0)` 的标准差因此只有约 `0.0114 m`；若速度为
`5/10/20 m·s⁻¹`，`dt` 波动对应的位移标准差也只有约
`0.114/0.228/0.456 m`，而高速样本只是总体的一部分。

因此，对 **standard 总分** 的现实预期应是：

- 正确物理时间本身带来的提升概率低，效应更可能接近 0 或小于项目当前
  `+1 Success / +2 Precision` 晋级阈值；
- 参数、正则化或运动先验可能涨点，但不能自动归因于真实时间；
- 真正有机会辨识真实时间的场景是同一 tracklet 内的长 gap、burst、随机缺失
  和 held-out schedule，而不是近恒定的 0.5 s 主表。

现有 M2 同 checkpoint 控制已经给出直接证据：时间干预确实改变预测，但
standard 和 gap1124 中正确时间都没有稳定超过 fixed/shuffled。当前最准确的
状态仍是：

```text
M2 tracking signal positive
physical-time causal claim NO_GO
method attribution HOLD
```

详细数据见
[`M2 standard/gap1124 控制分析`](../compare_results/reports/m2_standard_gap8_analysis_20260724.md)
与
[`delta_t 可辨识性计划`](../compare_results/reports/htv_identifiability_and_execution_plan_20260722.md)。

### 1.2 Random delete 20% 有实验意义，但现实意义只能算中等

如果在 0.5 s 关键帧序列上近似独立地以 `p=0.2` 删除帧，并只在保留帧上评测，
相邻保留观测的 gap 近似服从几何分布：

| gap | 近似比例 |
|---|---:|
| 0.5 s | 80.0% |
| 1.0 s | 16.0% |
| 1.5 s | 3.2% |
| ≥2.0 s | 0.8% |

平均 gap 约为 `0.5/(1-0.2)=0.625 s`。这意味着 Random-20% 确实制造了
tracklet 内不规则时间，但 80% transition 仍是原来的 0.5 s，真正困难的长 gap
比例很低。它适合验证“轻度随机缺失下是否稳健”，不适合单独证明真实系统的
sensor failure。

现实系统中的缺帧还可能是：

- 连续 burst，而不是独立同分布；
- 计算调度造成的周期性跳帧；
- 传感器、同步或队列延迟造成的相关缺失；
- 当前观测完全不可用，而不是简单删除后只评测 retained endpoints。

因此论文中应将 Random-20% 表述为：

> synthetic irregular-observation / virtual-rate stress test

而不是“真实 LiDAR packet-loss benchmark”。它仍值得保留，但只能作为协议矩阵
中的一行，至少与 `fixed skip`、`gap1124`、`burst-drop` 和一个 held-out
schedule 同时报告。nuScenes 官方 schema 说明标注 sample 是 2 Hz 关键帧；
Random-20% 是在已有 2 Hz 标注流上再降采样，并不等于原始 20 Hz LiDAR 的真实
packet loss。

### 1.3 当前最佳路线应分成两个论文分叉

**分叉 A：优先提高论文成功率。**

以稳定 B0 为主干，先尝试现有 M3 非对称 endpoint path distillation，论文问题
收窄为“历史重采样/不规则观测一致性”。这条路线推理结构不变，最贴合当前
paired-view 失败诊断，但它不是 explicit physical-time 方法。

**分叉 B：坚持 continuous-time 主张。**

先完成旧 R1 adapter/innovation 归因与当前 B1 同 checkpoint alpha sweep；
只有小权重、fail-closed 的 motion correction 在 normal 上恢复，并且
`true > fixed/shuffled`，才重新训练一个物理时间候选。该路线证据成本更高，
当前成功概率低于分叉 A。

不能再把失败的 motion、search、gate 与新 memory 同时堆叠，依靠模块间抵消
得到一个无法归因的分数。

### 1.4 跨数据集共享模型是更可辨识的真实时间场景（待验证）

当前 standard nuScenes 的 `delta_t` 变化很小，但这不意味着真实时间在所有
设定中都缺乏价值。若**同一个模型**联合训练或直接迁移于不同帧率的数据集，
order-only 的“一步运动”会具有不同的物理含义。例如 10 Hz 数据中的 `0.5 m`
逐帧位移与 2 Hz 数据中的 `2.5 m` 逐帧位移都可能对应 `5 m/s`。显式时间可以
通过 `v=delta_x/delta_t` 统一不同数据集的运动尺度，再通过
`delta_x_query=v*delta_t_query` 适配目标时刻。因此跨 KITTI、nuScenes、Waymo
或同一数据集的 stride-1/2/4 共享训练，比单一近恒定 cadence 更有机会辨识
physical-time 的价值。数据集总帧数本身不构成该动机，关键是帧率和历史窗口覆盖
的实际时间不同。

这个场景仍有两条严格边界：

1. 如果每个数据集分别训练独立模型，固定平均 `delta_t` 很容易被模型参数吸收，
   因而不能据此期待显著收益；
2. 如果一个数据集始终对应一个固定 `delta_t`，模型可能把时间当作 dataset ID，
   而非学习物理状态传播。

未来跨数据集实验应使用一个共享 checkpoint，并至少比较 order-only、全局固定
时间、dataset-mean fixed、within-dataset shuffled 和 true time。每个数据集内部
还应构造多 stride，并保留未见过的 cadence 作测试。`true` 只有在同数据集重采样
和 held-out gap 上继续超过两个控制，才能支持跨帧率连续时间泛化；若只超过全局
固定而不超过 dataset-mean fixed，则更可能只是数据集尺度校准。该实验属于当前
normal-mini 模块归因完成后的扩展证据，不改变现有执行优先级。

## 2. “实现代价”和“论文代价”必须分开

| 目标 | 工程代价 | 训练/评测代价 | 当前判断 |
|---|---|---|---|
| 在 batch 中保留真实时间并做 time encoding | 低，已完成 | 低 | 应保留 |
| 把 `dt` 用于有界运动先验 | 低到中，已实现 | 中 | 当前实现失败，需归因 |
| 把真实时间作为辅助鲁棒性变量 | 中 | 中 | 可行 |
| 把 continuous-time 写进标题和核心贡献 | 中 | 高 | 当前证据不足 |
| 忠实移植 ChronoTrack feature consistency | 中到高 | 中到高 | 可作第二阶段 |
| 加入 ChronoTrack recurrent compact memory + MCC | 高 | 高 | 目前不应启动 |
| 替换为 Mamba/SSM 时序主干 | 很高 | 很高 | 当前不值得 |

若目标是标题级 physical-time claim，保守的最终证据至少包含：

```text
训练：
  same-init baseline / method × seed42/43/44
  standard-only 或 mixed-cadence 的冻结训练定义

每个 final checkpoint 的评测：
  standard
  fixed skip
  Random-20%
  burst-drop
  held-out schedule
  true / fixed / shuffled effective time

最终：
  full nuScenes
  KITTI-HV 或另一 cadence 不同的数据集
  tracklet-level paired bootstrap
```

所以真实时间的“代码税”很小，但“因果证据税”很大。

## 3. 当前正负证据不能混用

### 3.1 历史 M2 不是当前 B1 的正向证明

历史 R1 M2 相对 A1 在 standard 和 gap1124 分别有
`+4.133/+9.445`、`+2.279/+4.143`，但它同时包含：

- 从 A1 checkpoint continuation 60 epoch；
- `shared_se2` candidate 轨迹；
- zero-init physical-time adapter；
- proposal innovation；
- 与 A1 不同的总训练预算。

而当前 B1 是：

- scratch；
- `candidate_trajectory_mode: independent`；
- `ct_history_training_mode: correlated_candidate`；
- physical-time adapter 关闭；
- 固定 `alpha=0.75`。

因此 B1 不是“已经证明有效的 M2 又跑了一次”，而是另一种数据分布、初始化和
运行时路径。B1 的失败不能否认一切 motion prior；同样，历史 R1 的正分也不能
证明 B1 的设计合理。

### 3.2 当前 B0–B3 已足够否决现有组合

本轮比较范围是 nuScenes v1.0-mini、Car、seed42、60 epoch：
mini_train 为 274 tracklets / 5,051 frames，mini_val 为
106 tracklets / 2,285 frames。主比较固定使用 epoch60 final checkpoint；
Success 是 3D IoU 阈值曲线 AUC，Precision 是 2 m 内中心距离阈值曲线 AUC。
best 与 late-3 只作稳定性诊断，不替代 final。

| arm | Final Success | Final Precision | 判定 |
|---|---:|---:|---|
| B0 baseline | 53.360 | 64.382 | 当前稳定基线 |
| B1 motion | 26.021 | 24.972 | 固定 0.75 correction 否决 |
| B2 motion + search | 47.973 | 52.088 | 恢复 B1，但仍低于 B0 |
| B3 + adaptive gate | 25.537 | 24.707 | gate 否决 |
| A1 search-only | 27.036 | 25.596 | 当前 search 否决 |

报告：

- [`B0–B3 消融`](../compare_results/reports/ct_v2_ablation_seed42_20260727.md)
- [`Search-only 复核`](../compare_results/reports/ct_search_only_seed42_20260727.md)

这些数字不能解释为“所有运动/搜索思想都错了”，但已经足够说明当前实现不能
直接进入多 seed、full 或论文主表。

## 4. 现有模块契合与实现问题审计

### 4.1 Dual-clock 总体设计：概念契合，证据未成立

主干继续使用 order time，物理时间只进入 dynamics/search，这个隔离设计是
合理的：它保留 SeqTrack3D 已验证的顺序语义，也允许对物理时间做干净干预。

问题是当前论文名称中的 “Continuous-Time Motion Encoder” 容易过度表述。
[`models/ct_v2/motion.py`](../models/ct_v2/motion.py) 里的
`ContinuousTimeMotionEncoder` 只是继承
[`DynamicsEncoder`](../models/dynamics.py)，没有覆盖新的连续时间状态方程。
实际机制是：

1. 从历史框有限差分得到 displacement/velocity/angular velocity；
2. 用 MLP 汇总；
3. 预测 velocity；
4. 乘当前 `delta_t` 得到 query displacement。

它是轻量 `dt`-conditioned kinematic prior，不是 ODE/CDE，也不是完整
continuous-discrete state-space model。论文中必须按真实实现命名。

### 4.2 CT Motion：与问题方向契合，但当前训练/推理语义不稳

> 2026-07-30 完成结果补充：有序 GRU + pre-crop second branch +
> zero-init adapter 的 B1motion-v2 在 normal-mini epoch60 仅为
> 20.618/19.830，相对 B0 下降 32.742/44.551。它不再是固定 alpha
> innovation，但仍未通过。完整复核见
> [`B1motion-v2 seed42 结果`](../compare_results/reports/b1motion_v2_seed42_20260730.md)。

主要问题：

1. `alpha=0.75` 过强。B1 post-warmup 约 73.7% 样本实际应用修正，40.7%
   被半径 clamp，motion 已经不是“小残差”。
2. standard 的 `dt` 近常量，网络可能学习一个普通 proposal prior，而不是
   正确时间映射。
3. 训练使用 correlated candidate history，递归验证使用预测框历史，误差分布
   不匹配。
4. dynamics 输入历史被扰动，但 displacement/velocity 监督保持 canonical；
   这要求网络从人工相关误差中同时去噪和预测真实位移，可能与递归错误过程不符。
5. 当前 B1 关闭 historical R1 中的 physical-time adapter，因此不能用 R1 的
   正结果为 B1 背书。
6. v2 scratch 的 independent/correlated candidate 定义与 R1 shared-SE(2)
   不同，模块效应与数据定义混在一起。
7. 可选 dynamics 模块在共享层初始化前消耗 RNG；当前单 seed 消融没有严格
   same-init。

B1motion-v2 又暴露了三条更具体的合同问题：

1. 35% irregular sampling 改变整个主干历史，而 `main_time_source=order`
   隐藏真实 gap；因此 dual-clock 的“主干隔离”在输入帧本身被替换后不再成立；
2. trajectory encoder 只看 candidate-anchor 相对历史，target 却包含不可由
   相对历史识别的共同 anchor error；
3. zero-init 只保证 step-0，相乘的 `normal_scale` 不是 correction norm
   上限；当前 correction L2 在 epoch3 已为 1.859，epoch60 仍为 2.072。

所以后续不应继续把 irregular cadence 直接混入主监督。连续 B0 view 应保持
不变，irregular view 只能作为 paired auxiliary branch；physical motion 与
anchor correction 也必须拆开。

修复原则不是继续调很多 alpha，而是先用现有 checkpoint 完成：

```text
B1 checkpoint: alpha = 0 / 0.25 / 0.75
R1 checkpoint: adapter on/off × innovation on/off
A1-init W0 continuation
```

只有 `alpha=0` 精确恢复 observation path、较小 alpha 在 normal 有正向或至少
不退化、R1 正信号能归因到明确路径，才值得训练下一版 motion。

### 4.3 Time-Guided Search：目标契合，但数据通路与安全回退不契合

当前 [`utils/ct_search.py`](../utils/ct_search.py) 做的是：

- 用预测历史框估计有界常速度；
- 构造朝预测位移方向的 tube；
- expansion-only 点不少于 32 时强制抽取 25% token；
- baseline 点只保留 75%；
- expansion 点没有 source/origin 标记。

关键问题：

1. 启用条件是“扩展区域点数足够”，不是“目标可能在扩展区域”或“baseline
   observation 不可靠”。背景丰富的场景反而更容易触发。
2. 一旦触发，不论 expansion 中是否有目标，都固定牺牲 25% baseline token。
3. Search-only 没有 gate，网络甚至收不到 aggregate expansion ratio；单点也
   没有 base/expansion source channel。
4. 训练只有约 3.46% 样本启用 search，平均 expansion token share 仅 0.865%；
   验证却是递归预测历史，当前缺少逐 endpoint 激活日志。
5. tube 由预测历史产生；一次漂移会改变下一帧 tube，再注入更多背景，形成
   正反馈。
6. “固定 1024 token”与“exact baseline identity path”存在结构冲突：只要从
   1024 baseline 点中拿走 25% 给 expansion，search-off 的 baseline 就不可能
   被严格保留。

因此下一版若仍保留 search，不应只把 75/25 改成 90/10。更合理的结构是：

```text
1024-point baseline branch: 完整保留
small expansion branch: 单独压缩成 K 个 proxy tokens
zero-init base-expansion cross-attention residual
source embedding + fail-closed confidence
```

这会产生少量真实计算开销，但能提供 exact baseline fallback。若必须严格维持
总 1024 点，就要接受“无法同时保证完整 baseline identity”的事实，并在论文中
明确这个取舍。

### 4.4 Adaptive Proposal Fusion：输入设计看似合理，学习目标不支持可靠性

[`ProposalFusionGate`](../models/ct_v2/fusion.py) 读取 observation/dynamics
特征、观测统计、proposal disagreement、gap、search ratio 和 valid mask。
但它只有最终 tracking loss，没有独立的“何时 motion 更可靠”监督，也没有保持
小 alpha 的正则或单调约束。

实际 gate 在 epoch7 就饱和到 0.75 上限，epoch60 几乎所有 batch 都是常数
0.75。当前问题不是 gate 容量不够，而是：

- 训练目标奖励 co-adaptation，不等于校准 reliability；
- dt 在 standard 中近常量，无法提供有效条件变化；
- observation stats 来自已经受 search 输入影响的 segmentation，可能形成
  search→stats→gate 的耦合反馈；
- `detach_context=true` 只阻止 gate 反向改变特征，不能自动产生正确的门控标签；
- 0.75 上限对一个未经验证的 prior 仍然过大。

不应继续堆更复杂 gate。只有 motion prior 本身独立通过后，才考虑简单的
fail-closed hard rule 或有明确 target 的校准器。

### 4.5 旧 symmetric TWC：不应复活为主方法

同提交 A/B/C 已证明 corrected-TWC 相对 paired weight0 有净正效应：
`C-B=+8.31/+11.74`；但 paired path 本身造成巨大损失，最终
`C-A=-7.00/-12.44`。它只修复了 paired-view 训练损失的一部分。

因此：

- TWC 不是当前正常集正向方法；
- TWC 不证明 physical timestamp；
- 不能再用两路都做 GT supervision 的 symmetric 训练方式；
- 旧坐标污染已修复，但 paired-view optimization 伤害仍存在。

见
[`TWC A/B/C`](../compare_results/reports/twc_abc_seed42_comparison_20260721.md)。

### 4.6 现有 M3 asymmetric endpoint distillation：当前最契合的低工程风险候选

项目已有的
[`models/path_distillation.py`](../models/path_distillation.py) 和 M3 路径包含：

- canonical history 的 EMA teacher；
- irregular history 的 student；
- 共同 endpoint、共同 coordinate anchor、共同 current XYZ 的 fail-fast；
- teacher stop-gradient；
- `m3_irregular_supervision_weight=0`，避免再次对受损 student path 做双路 GT
  supervision；
- inference 时可移除 teacher。

这比旧 symmetric TWC 更符合当前失败诊断，也是最接近“variable-rate history
resampling robustness”的现成实现。

但它仍有以下问题：

1. 当前 engineering YAML 建立在旧 M2/adapter/innovation 上，不应直接与失败
   motion 绑定；第一轮应移植到纯 B0。
2. 一次训练 batch 包含 student view A、student view B、EMA teacher A，前向
   计算大约是 baseline 的 3 倍；checkpoint 也会因注册 teacher 接近翻倍。
3. teacher confidence 来自 foreground top-k 与同一网络 coarse/refined
   agreement，不是经过独立校准的可靠性；distractor 上可能共同自信。
4. confidence floor 0.05 意味着坏 teacher 也不会完全关闭。
5. 它约束最终框和 coarse box，不是 ChronoTrack 的 point-feature temporal
   consistency。
6. 它不显式证明真实秒数有用；成功后论文应使用
   “endpoint/path/resampling consistency”，而不是直接使用 continuous-time
   因果表述。

### 4.7 M4 filter/tube：概念上物理，但当前前提没有通过

continuous-discrete filter 的 `F(dt)` 具有可解释时间语义，理论上比 generic
MLP 更容易做 true/fixed/shuffled 干预。但项目现有证据已经表明：

- observation reliability 的独立 validation No-Go；
- raw predicted-history CV candidate 没有稳定的独立 complementarity；
- 当前 post-crop selector 在强协议上校准较差；
- 大位移目标可能在网络 forward 前已离开 crop。

因此 learned covariance、Kalman gain、dual-anchor 和 trajectory tube 都不应
现在解锁。若未来重启，只能先做 fixed-Q/R、无 learned gate、无 tube 的解析
filter，并先报告 calibration。

## 5. ChronoTrack 能借鉴什么，不能混淆什么

[ChronoTrack](https://openaccess.thecvf.com/content/CVPR2026F/html/Yoo_Temporally_Consistent_Long-Term_Memory_for_3D_Single_Object_Tracking_CVPRF_2026_paper.html)
包含三个递进组件：

1. recurrent compact foreground memory tokens，并保留一帧 short-term
   background memory；
2. temporal consistency loss：用训练 GT 框将不同帧前景点变换到 canonical
   coordinates，做近邻对应并约束 point features；
3. memory cycle consistency：memory→point→memory 的循环概率，推动不同 token
   表示不同目标部位。

它的 KITTI ablation 本身给出重要警告：

| 设计 | Mean Success / Precision |
|---|---:|
| short-term point memory baseline | 70.5 / 88.6 |
| 只换 long-term token memory | 69.2 / 87.0 |
| + temporal consistency | 71.3 / 88.8 |
| + memory cycle consistency | 71.8 / 90.1 |

也就是说，**memory alone 会下降**；先解决 temporal feature inconsistency 才有
收益。这与 CT-SeqTrack 当前“不要用 memory 掩盖坏 motion/search”的结论一致。

ChronoTrack 与现有 M3 的区别：

| 项目 | ChronoTrack TC | CT-SeqTrack M3 |
|---|---|---|
| 约束对象 | 对应前景点的 latent feature | 同 endpoint 的 coarse/final box |
| 对齐方式 | GT box canonical coordinates + NN | 共享 current points/anchor |
| teacher | 不需要 EMA teacher | canonical EMA teacher |
| 核心目标 | 防止长时特征漂移 | 对历史重采样路径不敏感 |
| 证明真实 `dt` | 否 | 否 |

因此可以借鉴 ChronoTrack，但不能把 M3 改名为其 temporal consistency。

### 推荐的 ChronoTrack 借鉴顺序

1. **Chrono-lite point-feature consistency**：先在 B0 上只加入训练期 feature
   loss，不加 memory。需要暴露每帧 point feature、携带训练 GT box sequence、
   构造 canonical NN correspondence。
2. **compact foreground memory**：只有第一步在 normal 和长序列桶为正，才加
   固定数量 token；第一帧 GT 初始化，后续按预测 targetness 更新。
3. **memory cycle consistency**：只有 token memory 本身可用后再加，不能单独
   使用。

这一路线的论文创新必须是对 variable-rate 3D SOT 的适配，例如
`delta_t`-aware memory decay、跨不规则间隔的 canonical matching 或
resampling consistency；原样复制 compact memory/TC/MCC 不能再作为新的核心
贡献。

## 6. 近期工作中还可借鉴的模块

### 6.1 HVTrack：优先借鉴 base/expansion 分流，而不是继续做 raw union

[HVTrack（ECCV 2024）](https://eccv.ecva.net/virtual/2024/poster/1372)
为 HTV 提出 Relative-Pose-Aware Memory、Base-Expansion Feature
Cross-Attention 和 Contextual Point Guided Self-Attention。

与当前项目最契合的是：

- baseline crop 与 expansion crop 分开编码；
- baseline 作为稳定主路径，expansion 只通过 cross-attention 提供残差；
- 用上下文注意力抑制 expanded area 的背景和相似目标。

它直接对应当前 Search-only 的背景注入问题。代价是需要改数据接口和 feature
fusion，属于中高工程量；应在 Search 2×2 证明故障来自推理 search 后再做。

### 6.2 CompTrack：用 foreground-aware compression 修复扩展区域冗余

[CompTrack（AAAI 2026）](https://ojs.aaai.org/index.php/AAAI/article/view/38385)
使用 Spatial Foreground Predictor 过滤背景，并用 Information
Bottleneck-guided Dynamic Token Compression 将前景压缩为少量 proxy tokens。

可借鉴为：

```text
expansion points
  -> lightweight foreground/source scorer
  -> top-K or proxy token compression
  -> zero-init residual into baseline features
```

它比“所有 expansion 点等价采样”更适合当前故障。但不要一开始完整移植在线
SVD/低秩模块；第一版只需验证 source-aware small proxy tokens 是否消除灾难性
退化。

### 6.3 TrajTrack：proposal agreement 可作诊断，不能直接替代可靠性

[TrajTrack（arXiv 2025）](https://arxiv.org/abs/2509.11453)
使用短期 explicit proposal、历史框 trajectory proposal，并根据两者 IoU 选择
local/global proposal。

当前可借鉴：

- 把 observation proposal 与 trajectory proposal 的 agreement 作为分桶诊断；
- 保存 local/global/final 三类 proposal，分析谁在长 gap 更接近 GT；
- 后续若 prior 独立成立，再考虑一个简单 selector。

当前不应直接复制“低 agreement 就选 global”的规则，因为 CT-SeqTrack 的
dynamics proposal 尚未证明可靠，B1 已显示强行靠近 global prior 会崩溃。
现有 M3 teacher confidence 已包含 coarse/refined agreement；它仍需独立校准，
不能再叠一个同类 gate。

### 6.4 StreamTrack：selective memory update 比扩大历史点云更值得借鉴

[StreamTrack（AAAI 2024）](https://ojs.aaai.org/index.php/AAAI/article/download/28196/28389)
使用 live memory bank、spatial-temporal hybrid attention 和 sequence
contrastive enhancement。

可借鉴的不是再存更多帧，而是：

- foreground/background 显式分开；
- memory write 与 read 解耦；
- 只在当前观测可靠时写入；
- distractor-aware contrastive objective。

但项目当前没有通过可靠 write gate，所以它应排在 Chrono-lite feature
consistency 之后。

### 6.5 MambaTrack3D：方向相关，但不解决当前 identifiability

[MambaTrack3D（arXiv 2025 预印本）](https://arxiv.org/abs/2511.15077)
用 Mamba-based Inter-frame Propagation 与 Grouped Feature Enhancement
处理 HTV、长序列复杂度和前景/背景冗余。

它适合在“已经证明需要更长 memory，Transformer 成本成为瓶颈”时使用。当前
只有 4 帧输入，主要故障是 motion/search 语义和训练—递归分布错配；换 SSM
既不能增加 standard 的 `dt` 激励，也不能自动让 `true > shuffled`。当前优先级
最低。

## 7. 推荐执行顺序

### Phase 0：不训练，先完成归因

1. 现有 B0/A1 checkpoint 做 Search off/on 2×2，补逐 endpoint search
   diagnostics。
2. 现有 B1 checkpoint 做 `alpha=0/0.25/0.75`。
3. 历史 R1 checkpoint 做 adapter/innovation 2×2。
4. 补 A1-init W0 continuation，分离额外 60 epoch。

这四步决定 search 是输入通路故障、motion 是权重故障还是表征故障，也决定
R1 正信号是否值得继续。

### Phase 1A：优先论文成功率，B0 + M3

只在 B0 上做同初始化 seed42 三组：

| arm | paired sampler | EMA teacher | path loss | student-B GT loss |
|---|---:|---:|---:|---:|
| M3-A | 否 | 否 | 0 | — |
| M3-B | 是 | 是 | 0 | 0 |
| M3-C | 是 | 是 | `0.05` | 0 |

M3-B 控制 paired data、额外 forward、EMA 注册和 RNG/BN 影响；M3-C 才是 path
distillation 净效应。第一轮保持 search、motion、gate、memory 全关。

晋级条件：

- C 相对 A 的 final 至少 `+1 Success / +2 Precision`；
- late-3 不退化；
- C-B 为正；
- evaluation-only 多历史路径 endpoint variance 下降；
- normal 通过后，同一 checkpoint 再测 Random-20%、gap1124、burst 和
  held-out schedule。

如果 M3 成立，论文主张应是 resampling/path consistency。真实 `delta_t` 可作为
协议定义和后续分析变量，但不能抢占核心方法贡献。

### Phase 1B：坚持真实时间，只训练一个保守 motion 候选

只有 Phase 0 显示某条 motion 路径独立有价值才启动。设计约束：

- 从同一 B0 initialization 加载所有 shared keys；
- observation identity path 精确保留；
- zero-init adapter 或最大 `alpha≤0.1/0.2` 的 bounded residual；
- 不加 search，不加 learned gate；
- 训练历史与递归误差过程匹配；
- final checkpoint 直接做 `true/fixed/shuffled`；
- true 没有同时超过两个控制则立即停止，不通过更多 seed 复活。

### Phase 2：按失败类型二选一

如果 Search 2×2 证明 search-on 是主因：

```text
HVTrack-style base/expansion separation
  + source embedding
  + CompTrack-style K-token compression
  + zero-init residual
```

如果 M3 成立且主要问题转为长序列 feature drift：

```text
Chrono-lite point-feature consistency
  -> compact foreground tokens
  -> memory cycle consistency
```

两条路线不能同时首测。

### Phase 3：最后才考虑 state filter 或 Mamba

只有出现以下证据才解锁：

- motion prior 同 checkpoint 因果成立；
- observation uncertainty 可校准；
- predicted-history candidate 有独立 crop complementarity；
- 长 memory 确实有收益且计算成为瓶颈。

## 8. 论文表述决策树

### 情况 A：standard 涨点，且 `true > fixed/shuffled`

可以使用：

> physical-time / dual-clock / continuous-time motion prior

仍需 full、多 seed、第二数据集和 held-out cadence。

### 情况 B：standard 涨点，但 true 与控制持平

只能使用：

> time-conditioned or trajectory-conditioned prior

不能声称正确物理时间对应关系产生收益。

### 情况 C：M3/Chrono-lite 在 irregular cadence 有效，但 explicit dt 无效

建议标题方向：

> Endpoint-Consistent History Resampling for Variable-Rate 3D SOT

或：

> Resampling-Robust Sequence Tracking for Irregular LiDAR Observations

### 情况 D：所有新模块都不能超过 B0，但多模型暴露稳定 cadence gap

转成 benchmark/diagnosis：

> When Frame Index Is Not Time: Diagnosing Variable-Rate 3D SOT

这比用 Random-20% 的单个正数强行支撑 continuous-time 方法更可防御。

## 9. 最终建议

1. 保留真实时间基础设施和 dual-clock 设计，但把它从“已成立的主贡献”降为
   待证假设。
2. Random-20% 保留为轻度不规则观测测试，不作为唯一现实场景或选模协议。
3. 当前先完成 Search 2×2、B1 alpha sweep 和 R1 归因，不新增训练树。
4. 下一项最值得训练的是 **纯 B0 上的 M3 asymmetric endpoint
   distillation A/B/C**，不是 B3+memory。
5. 若继续 search，必须重构为 base/expansion 分流与 source-aware small
   proxy tokens；不再调 raw 75/25 union。
6. 若借鉴 ChronoTrack，先做 point-feature consistency，再做 compact memory，
   并明确它与现有 M3 的区别。
7. 当前不加 learned gate、learned covariance、Mamba 或完整 ODE/CDE。

## 10. 仍需由下一批证据回答的问题

- B0/A1 的退化是否只在 inference search-on 时出现，还是 A1 checkpoint 在
  search-off 下仍然退化？
- B1 的 `alpha=0` 能否精确恢复 observation path，`alpha=0.25` 是否仍会明显
  退化？
- 历史 R1 的正信号主要来自 adapter、innovation、两者交互，还是额外 60 epoch
  continuation？
- 纯 B0 上的 M3-C 能否同时超过 single-view M3-A 和 paired weight0 M3-B？
- SeqTrack3D 自身是否存在可复现的长时间 feature drift，足以支持实现
  Chrono-lite point correspondence loss？
- 新 search 若使用 predicted history，是否在离线 endpoint 上具有真实
  tube-only target complementarity，而不只是更多背景点？

这些问题中的前三个不需要新训练或只需要现有对照；在回答前启动 memory、
Mamba 或 learned covariance 不会提高结论可信度。

## 11. 一手资料

- [nuScenes 官方数据页：LiDAR 采集与 2 Hz 标注关键帧](https://www.nuscenes.org/nuscenes)
- [nuScenes 官方 devkit schema：sample 为 2 Hz annotated keyframe](https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuscenes.md)
- [ChronoTrack，CVPR 2026 Findings](https://openaccess.thecvf.com/content/CVPR2026F/html/Yoo_Temporally_Consistent_Long-Term_Memory_for_3D_Single_Object_Tracking_CVPRF_2026_paper.html)
- [ChronoTrack 官方代码](https://github.com/ujaejoon/ChronoTrack)
- [HVTrack，ECCV 2024](https://eccv.ecva.net/virtual/2024/poster/1372)
- [TrajTrack，arXiv 2025](https://arxiv.org/abs/2509.11453)
- [StreamTrack，AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/download/28196/28389)
- [CompTrack，AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/38385)
- [MambaTrack3D，arXiv 2025 预印本](https://arxiv.org/abs/2511.15077)

## 12. 2026-07-28 实施更新：第四模块冻结为 Δt-PFTC

本节覆盖第 7 节中“先训练 M3”的旧优先级；历史分析和 No-Go 判断保留不变。
当前第四模块不是在 B3 后继续堆叠，而是稳定 B0 的独立派生臂：

| arm | 主干 | canonical point loss | 帧对权重 |
|---|---|---|---|
| B0 | B0 | 关 | — |
| PFTC-U | B0 | 开 | 全 1 |
| Δt-PFTC | B0 | 开 | effective Δt，样本内归一化 |

工程实现位于
`models/ct_v2/point_feature_consistency.py`。FeaturePointNet 只在训练且开关
启用时额外返回第二层 64-D 逐点特征；关闭开关和 eval 均保持原输出。canonical
匹配固定使用 GT 前景、同 anchor GT 框、单向 NN、`0.3 m` 阈值、至少 3 对和
raw SmoothL1。重复采样 XYZ 先合并，完整 1024×1024 距离矩阵不会构造。

正式训练前固定执行 200-batch weight-zero 预检：有效样本率低于 30% 直接停止；
否则从 `{1,0.5,0.2,0.1,0.05,0.02,0.01}` 冻结 raw 与 Δt-weighted 两种
中位数 `lambda*L_PFTC/L_supervised` 都不超过 10% 的最大值。该值同时用于
PFTC-U 和 Δt-PFTC，不依据验证分数反调。对应入口为：

```bash
python tools/ct_v2/run.py train --variant pftc --preflight --seed 42
python tools/ct_v2/analyze_pftc_preflight.py \
  output/<preflight-run>/lightning_logs/version_0
python tools/ct_v2/check_pftc_initialization.py --seed 42
```

首筛固定同代码、同 scratch seed42、batch16、60 epoch；只有 Δt-PFTC 的 final
Success/Precision 同时超过 B0、late-3 不退化、feature std 不低于 B0 的 50%，
且覆盖率/显存/数值正常，才补 seed43/44。随后同一 checkpoint 在 normal、
冻结 random20、gap1124 上评测，并做 true/fixed/shuffled 时间控制。

解释边界固定如下：

- PFTC 在推理时不读取 Δt；random20 单独涨分只能证明掉帧泛化，不能证明在线
  真实时间适应。
- weighted 与 unweighted 相近，只能声称 point-feature consistency 有效。
- 只有 true 持续优于 fixed/shuffled，才能把收益归因于正确物理秒数。
- 旧 TWC 是 paired endpoint consistency，M3 是 EMA path distillation；
  新 PFTC 才是 canonical point-feature consistency，三者不能混称。
- compact memory、MCC、Mamba 和新的 search 修改均锁定到 PFTC 独立通过之后。

## 13. 2026-08-01 数据更新：当前 Δt-PFTC 完整运行 No-Go

首个 formal-named seed42 run 的本地 artifact 路径为：

```text
output/20260728-1826-07_seqtrack3d_dt_pftc-
dt_pftc_true_5f260e7_seed42_60ep_bs16_gpu0
```

当前拉回的 events 和 checkpoint 已完整覆盖 75,720 step、12 个验证点和
epoch60。final 为 `51.189/60.886`，相对 B0 的 `53.360/64.382` 下降
`2.171/3.496`；late-3 为 `51.398/60.618`，相对 B0 下降 `1.507/2.487`。
B4 最好的 Success `52.728` 与 Precision `63.870` 出现在不同 epoch，且都低于
B0 final。早期 epoch5/10/20 的同阶段正差只是优化轨迹变化，不能替代 final
与 late-3。

本次审计改变了第 12 节的执行状态，但不改变其消融原则：

1. 当前 canonical yaw 代码使用 `R(+yaw)`，项目 object-local 约定需要
   `R(-yaw)`；现有 yaw 单测构造方向也随代码一起错，当前 run 无法评价正确
   canonical correspondence。
2. foreground feature std 从 epoch1 `0.0947` 降到 epoch60 `0.0156`，而
   match 数和 match distance 稳定；PFTC loss 同期下降 99.21%，raw SmoothL1
   的无负样本目标出现强平凡收缩警报。
3. weighted/raw PFTC loss 相对差异中位数只有 `-0.252%`。standard cadence
   下当前 Δt 权重没有给出物理时间增量证据；它主要在编码 pair lag。
4. 单卡训练约 `2.983 s/step`，B0 为 `0.362 s/step`，开销约 `8.24×`；完整
   events 跨度 62.74 小时，当前工程路径不可用于正式三臂实验。
5. PFTC-U 和 commit `5f260e7` 的同代码 B0 尚未完成，无法分离 consistency 与
   Δt weighting。
6. epoch60 supervised loss 比 B0 低 1.56%，但验证更差；问题不是训练未收敛，
   而是当前辅助目标损害泛化表示。

当前正式判定为：

```text
NO-GO_CURRENT_IMPLEMENTATION
PFTC_IDEA_NOT_YET_FAIRLY_TESTED
NO_EVIDENCE_FOR_PHYSICAL_TIME
```

不启动 seed43/44、full nuScenes、Random-20%、gap1124、true/fixed/shuffled、
memory，也不原样补跑 PFTC-U。先修正 yaw 符号和测试，加入 projector +
normalized loss + variance floor 等防坍缩机制，记录 B0 feature std 与 gradient
conflict，并把单卡开销降到 B0 的 2 倍以内。之后只跑
B0/PFTC-U-v2/Δt-PFTC-v2 各 5 epoch 的机制 kill-test；全部通过才重新预检 λ
并从 scratch 开始 60-epoch 三臂实验。

完整数值、根因和下一步见
`compare_results/reports/pftc_b4_seed42_final_diagnosis_20260801.md`。

## 14. 2026-07-30 数据更新：固定全局 Motion innovation 停止

新完成的 alpha0/0.25 两组均为 seed42、scratch、normal mini、batch16、
60 epoch、75,720 step 和 12 个验证点。两组来自同一 commit `5f260e7`，
tracked source clean，resolved config 仅 cfg/tag/alpha 不同：

| arm | final Success | final Precision | late-3 Success | late-3 Precision |
|---|---:|---:|---:|---:|
| B0（历史基线） | **53.360** | **64.382** | **52.905** | **63.104** |
| motion alpha0 | 47.049 | 49.184 | 46.828 | 49.669 |
| motion alpha0.25 | 29.581 | 28.862 | 29.472 | 28.849 |
| motion alpha0.75（旧 B1） | 26.021 | 24.972 | 26.080 | 25.299 |

alpha0.25 相对 alpha0 final 下降 `17.468/20.322`，late-3 下降
`17.357/20.820`；epoch25–60 的 8/8 个验证点两项指标同时更低。
alpha0.25 post-warmup applied alpha 均值只有 0.184、applied ratio 为
73.7%、平均 correction norm 为 0.083 m，说明较小修正只是减少旧 0.75 的
伤害，没有改变错误方向。

更关键的是，epoch60 mean training loss 随 alpha 从 0→0.25→0.75 由
`0.223→0.217→0.215`，递归验证却反向下降。结合代码路径：

1. train 读取 canonical/correlated GT history，eval 读取 recursive predicted
   history；
2. valid mask 不衡量 proposal 准确性；
3. innovation 位于 coarse proposal 和 Transformer query 之前；

当前失败应归因于局部训练目标与闭环历史分布错位、错误 motion proposal 的
递归放大，而不是训练不足。M0-3 的 oracle alpha0.775 来自 GT-history、
candidate0、crop-reachable 条件，不能直接迁移到本入口。

当前正式判定：

```text
NO_GO_FIXED_GLOBAL_MOTION_INNOVATION
ALPHA025_REDUCES_BUT_DOES_NOT_REMOVE_FAILURE
ALPHA000_IS_A_FALLBACK_CONTROL_NOT_A_GAIN
BROADER_MOTION_PRIOR_IDEA_REMAINS_UNRESOLVED
```

不再训练 alpha0.05/0.1、seed43/44、full nuScenes 或 motion+search。先用
已有 alpha0/0.25 checkpoint 做推理 alpha on/off 2×2，并导出 observation/
dynamics/GT 的逐 endpoint attribution。若开启 0.25 立即退化、关闭后恢复，
直接 fusion 永久停止；若关闭仍不恢复，则记录 training co-adaptation，同样
不再扫全局 alpha。只有跨 split 存在可识别 helpful subgroup 才允许研究条件
使用；P0-B4 已否定的旧 reliability gate 不得直接复刻。

完整数值、曲线和复现数据见
`compare_results/reports/ct_motion_alpha_sweep_seed42_20260730.md`。
