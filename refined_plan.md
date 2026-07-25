# CT-SeqTrack v2 论文计划

更新时间：2026-07-25

## 论文问题

固定帧率假设下，历史框的运动量通常只按序号建模。CT-SeqTrack v2 研究：

> 能否在不改变 SeqTrack3D 主干和 token 预算的前提下，用真实时间生成连续时间运动先验，扩展可能的搜索区域，并仅在观测不确定时做有界 proposal 修正？

## 方法

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

## 最小消融

| 实验 | CT Motion | Search | Adaptive Gate |
| --- | ---: | ---: | ---: |
| B0 SeqTrack3D |  |  |  |
| B1 CT Motion | ✓ |  | 固定系数 |
| B2 CT Motion + Search | ✓ | ✓ | 固定系数 |
| B3 CT-SeqTrack v2 | ✓ | ✓ | ✓ |

四组 mini 使用同代码、seed42、60 epoch、candidate4、正常数据和 final checkpoint。B3 晋级后再补 baseline/B3 的 seed43、44，并转 full nuScenes。

## 晋级规则

1. **正常集涨点**：B3 相对 B0 的 Success 和 Precision 都为正；mini 的目标门槛为至少 `+1.0 Success / +2.0 Precision`。
2. **模块可解释**：B1、B2、B3 的变化用于定位贡献；若某模块连续两步为负则从最终模型移除，不另加模块掩盖。
3. **时间双门槛**：B3 的同 checkpoint `true` 相对 `fixed/shuffled` 至少不退化；只有置信区间支持正确时间领先，论文才使用强因果表述。
4. **Random-20% 后置**：正常数据晋级后仅作为鲁棒性补充，不用于选择 checkpoint 或调参。

## 论文贡献表述

若双门槛通过：

1. 轻量连续时间运动先验；
2. 保留 baseline 覆盖的时间引导搜索扩展；
3. 观测优先的自适应有界 proposal 融合；
4. normal + variable-rate 的成对时间干预分析。

若只涨点但 true 未领先控制：保留模型和鲁棒性结果，但将表述降为 time-conditioned trajectory prior，不声称正确物理时间具有已验证的因果优势。
