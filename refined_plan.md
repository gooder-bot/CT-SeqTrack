# CT-SeqTrack v2 论文计划

更新时间：2026-08-01

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

> 2026-07-30 motion 更新：alpha0/0.25 两组 scratch 60 epoch 已完成。
> alpha0.25 相对 alpha0 final 下降 `17.468/20.322`，较小全局修正仍失败；
> fixed global proposal innovation 正式 No-Go。motion 后续只做已有
> checkpoint 的推理 on/off 2×2 与 endpoint attribution，不再启动 alpha
> 长训。完整报告见
> [Motion fixed-alpha 复核](compare_results/reports/ct_motion_alpha_sweep_seed42_20260730.md)。

## 论文问题

固定帧率假设下，历史框的运动量通常只按序号建模。CT-SeqTrack v2 研究：

> 能否在不改变 SeqTrack3D 主干和 token 预算的前提下，用真实时间生成连续时间运动先验，扩展可能的搜索区域，并仅在观测不确定时做有界 proposal 修正？

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
