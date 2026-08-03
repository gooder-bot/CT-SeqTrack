# CT-SeqTrack v2 论文计划

更新时间：2026-08-02

> 2026-08-02 B2-v2 实证更新：motion + Search Evidence + joint fusion 的
> seed42 scratch 60-epoch final 为 `54.132/64.755`、late-3 为
> `54.462/66.013`。相对最强历史 B0，final 为 `+0.772/+0.373`，仅
> Success 通过预注册门槛；late-3 为 `+1.557/+2.909`。同时，新 SeqTrack
> control 异常低至 `31.684/31.337` 且缺少 provenance，不能作为唯一 baseline。
> full 中 Search Evidence candidate 有效率只有 `23.29%`，epoch60 gate
> argmax 选择率只有 `0.104%`，因此当前状态为
> `B2V2_NORMAL_SIGNAL_POSITIVE / B2V2_SEARCH_CONTRIBUTION_NOT_ESTABLISHED /
> HOLD_B2V2_PROMOTION`。论文不能声称 search 已带来增益；先做同 checkpoint
> 四模式归因与 commit-`a486a36` matched B0。完整复核见
> [B2-v2 seed42 技术报告](compare_results/reports/b2_search_v2_seed42_20260802.html)。

> 2026-07-28 决策：standard 的 `delta_t` CV 只有 4.59%，真实时间的代码
> 代价低、因果证据代价高；Random-20% 只保留为 synthetic
> irregular-observation stress test。ChronoTrack feature consistency、现有
> M3 endpoint distillation 与 compact memory 不能混为同一模块。完整投入产出、
> 模块契合审计和执行分叉见
> [真实时间价值与模块路线审计](docs/TIME_VALUE_AND_MODULE_ROADMAP_20260728.md)。

> 2026-08-01 实证更新：首个 Δt-PFTC seed42 artifact 已完整跑满 60 epoch，
> final `51.189/60.886`、late-3 `51.398/60.618`，相对 B0 分别下降
> `2.171/3.496` 与 `1.507/2.487`。同时 canonical yaw 符号错误、feature std
> 只剩 epoch1 的 16.4%，训练开销为 8.24×。当前状态为
> `NO-GO_CURRENT_B4_IMPLEMENTATION / PFTC_IDEA_NOT_YET_FAIRLY_TESTED`。
> 完整诊断见
> [Δt-PFTC seed42 60-epoch 最终诊断](compare_results/reports/pftc_b4_seed42_final_diagnosis_20260801.md)。

> 2026-08-01 B1motion-v3 实证更新：seed42 scratch 60 epoch final 为
> `52.655/61.835`、late-3 为 `52.050/61.206`，相对历史 B0 分别下降
> `0.705/2.547` 与 `0.855/1.898`，当前为
> `NO_GO_B1MOTION_V3_STANDARD_GAIN`。但 v3 相对 v2 final 恢复
> `+32.037/+42.004`，且 learned prior 在 main/gap2/gap4 相对 CV 的训练
> RMSE 改善 `7.6%/10.9%/16.0%`；当前瓶颈收窄为 gate calibration 与
> recursive-history transfer。下一步只做 epoch30/epoch60 same-checkpoint
> fusion on/off 和 endpoint attribution，不直接新开 motion 长训。完整报告见
> [B1motion-v3 seed42 技术复核](compare_results/reports/b1motion_v3_seed42_20260801.html)。
> 相对原始 SeqTrack3D plain 的 `50.986/59.962`，v3 数值为
> `+1.670/+1.873`；但 current B0 对原始 SeqTrack 为 `+2.374/+4.420`，
> 因此前者不能作为 motion 模块净贡献。

> 2026-07-30 motion 更新：alpha0/0.25 两组 scratch 60 epoch 已完成。
> alpha0.25 相对 alpha0 final 下降 `17.468/20.322`，较小全局修正仍失败；
> fixed global proposal innovation 正式 No-Go。motion 后续只做已有
> checkpoint 的推理 on/off 2×2 与 endpoint attribution，不再启动 alpha
> 长训。完整报告见
> [Motion fixed-alpha 复核](compare_results/reports/ct_motion_alpha_sweep_seed42_20260730.md)。

## 论文问题

固定帧率假设下，历史框的运动量通常只按序号建模。CT-SeqTrack v2 研究：

> 能否在不改变 SeqTrack3D 主干和 token 预算的前提下，用真实时间生成连续时间运动先验，扩展可能的搜索区域，并仅在观测不确定时做有界 proposal 修正？

### 跨数据集、跨帧率假设（待验证）

真实时间更可能在**同一个共享模型同时面对不同采样频率**时体现价值。SeqTrack3D
只把历史表示为第 1/2/3 帧；例如 10 Hz 与 2 Hz 数据中的“一步”分别对应
`0.1 s` 与 `0.5 s`，相同物理速度会表现为不同的逐帧位移。显式 `delta_t` 可以先
把历史位移归一为 m/s，再按当前 query gap 积分，使运动状态在 KITTI、nuScenes、
Waymo 或同一数据集的 stride-1/2/4 之间具有一致的物理语义。这里相关的是帧率和
时间跨度，不是数据集的总帧数。

该假设不等于当前已有正结果。若每个固定帧率数据集分别训练一个模型，平均步长
可以被模型参数隐式吸收，真实时间的增量可能很小；若各数据集的 `delta_t` 近似
常数，它还可能退化为 dataset ID。正式证据必须来自一个共享 checkpoint，并在
每个数据集内部加入多 stride 或 held-out cadence，比较 `true`、全局固定、
dataset-mean fixed 与 within-dataset shuffled。只有 `true` 在同数据集重采样及
未见间隔上仍领先，才能把跨数据集收益归因于物理时间，而不是数据域识别。

## 原 v2 候选（已在首筛中否决）

```text
历史预测框 + real delta_t
        │
        ├── Continuous-Time Motion Prior ───────┐
        │                                       │
        └── Time-Guided Search Expansion ── 当前点云
                                                │
SeqTrack3D observation proposal ────────────────┤
                                                ▼
                                  Adaptive Proposal Fusion
                                                │
                                           最终目标框
```

- SeqTrack3D 的主干仍使用稳定的 order-time token。
- 真实时间只进入运动先验、搜索几何和融合半径。
- 搜索扩展不替换原 crop，也不增加点数/Transformer token。
- 融合是 observation-first 的小幅有界 innovation，不直接相加两个完整位移。

## 已完成的 v2 首筛

| 实验 | CT Motion | Search | Adaptive Gate |
| --- | ---: | ---: | ---: |
| B0 SeqTrack3D |  |  |  |
| B1 CT Motion | ✓ |  | 固定系数 |
| B2 CT Motion + Search | ✓ | ✓ | 固定系数 |
| B3 CT-SeqTrack v2 | ✓ | ✓ | ✓ |

四组 mini 已使用 seed42、60 epoch、candidate4、正常数据和 final
checkpoint 完成。结果为 B0 `53.360/64.382`、B1 `26.021/24.972`、
B2 `47.973/52.088`、B3 `25.537/24.707`。当前 B3 未晋级；learned
gate 在 epoch7 已饱和到 0.75，不能再作为“自适应可靠性”模块。

后续 Search-only A1 也已完成：final `27.036/25.596`，late-3
`27.933/26.400`，相对 B0 final 为 `−26.324/−38.786`。因此 B2 对 B1 的
正增量是交互恢复，不能作为 search 独立收益。A1 与 B0 的训练 loss 接近且
search 确实启用，当前失败更像训练/递归搜索分布不匹配或强模块交互。

## 当前最小诊断

Search 开/关 2×2 继续保留为旧 search 的低成本归因任务，但不再扩展 A2 或
search 训练树。Motion 同样只保留 alpha0/0.25 checkpoint 的无训练 on/off
2×2；alpha0.25 的 0.083 m 平均 correction 已足以造成大幅递归退化，不能再
把失败解释成单纯 alpha0.75 过强。当前 GPU 主线收敛为修复 Δt-PFTC 后的一次
机制 kill-test：

```text
修正 canonical R(-yaw) 与交叉单测
        ↓
projector + variance floor，增加 B0 feature-std/gradient 对照
        ↓
把单卡开销从 8.24× 压到 ≤2× B0
        ↓
B0 / PFTC-U / Δt-PFTC 各 5 epoch 机制 gate
        ↓ 全部通过
重新预检 λ，并从 scratch 跑 60-epoch 三臂
```

旧 Δt-PFTC 已完整跑完但正式 No-Go。epoch60 的 `51.189/60.886` 来自错误
canonical geometry，且 final/late-3 都低于 B0，只能作为失败诊断，不能参与
论文正向主表。

## 晋级规则

1. **正常集涨点**：候选相对同初始化 baseline 的 final Success 和
   Precision 都为正；mini 的目标门槛仍为至少 `+1.0 / +2.0`，late-3
   同时不得退化。
2. **模块可解释**：失败模块直接移除或单独重构，不另加模块掩盖。当前 A1
   已未通过；A2 保持锁定。
3. **时间双门槛**：晋级模型的同 checkpoint `true` 相对
   `fixed/shuffled` 至少不退化；只有置信区间支持正确时间领先，论文才使用
   强因果表述。
4. **Random-20% 后置**：正常数据晋级后仅作为鲁棒性补充，不用于选择 checkpoint 或调参。

## 论文贡献表述

### 候选主创新：跨帧率不变的双时钟 3D 跟踪

若后续证据通过，主创新不应只写成“加入真实时间编码”，而应定义为：

> 将多帧 3D SOT 从 frame-relative sequence modeling 推进为
> frame-rate-invariant physical-time state propagation，使同一个共享模型能够
> 在不同及未见观测 cadence 下保持一致的运动语义。

该贡献由三部分组成：

1. **问题与表示**：指出 order-only 的“一步位移”在不同帧率下物理含义不一致；
   用 dual-clock 保留稳定 order backbone，同时以 `delta_x/delta_t` 表示物理
   运动状态；
2. **查询时传播与取证**：按 `delta_t_query` 传播 motion proposal，并在预测轨迹
   端点查询 observation-conditioned 点云证据，而不是按固定帧步扩大主 crop；
3. **跨帧率验证**：先在同一数据集内使用多 stride 和 held-out cadence 排除
   数据域混杂，再扩展到 KITTI、nuScenes、Waymo 的共享模型，使用全局固定、
   dataset-mean fixed、within-dataset shuffled 和 true time 做成对控制。

它的最低成立条件是：同代码、同初始化候选先超过 order-only baseline；随后
`true` 不仅超过全局 fixed，还要在同数据集重采样和未见 gap 上超过
dataset-mean fixed 与 within-dataset shuffled。若只在跨数据集汇总上超过全局
fixed，则只能解释为帧率/数据集尺度校准，不能声称连续时间泛化。

若后续重构模块通过正常集和时间双门槛：

1. 只保留通过独立消融且具备 fail-closed 回退的真实时间模块；
2. 后续模块必须逐个加入并独立超过前一阶段；
3. normal + variable-rate 的成对时间干预分析。

当前 fixed-global motion innovation、search 和 adaptive proposal gate 均不能
写成正向论文贡献。更广义的 motion feature/adapter 只保留为待归因历史信号，
不能用本次 alpha0 control 冒充正贡献。
只有新模块通过独立消融后，才能加入最终方法描述。

当前 PFTC 同样不能写成正向贡献。若修订后的 PFTC-U 涨点而 Δt-PFTC 与之
相近，论文只能使用“canonical point-feature consistency”；只有 Δt-PFTC 在
多 seed 且 true/fixed/shuffled 控制中持续领先，才加入真实秒数表述。

若只涨点但 true 未领先控制：保留模型和鲁棒性结果，但将表述降为 time-conditioned trajectory prior，不声称正确物理时间具有已验证的因果优势。
