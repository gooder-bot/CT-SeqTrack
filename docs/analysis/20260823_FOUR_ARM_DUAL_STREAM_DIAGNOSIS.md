# CT-SeqTrack 四臂双流实验诊断（2026-08-23）

## 结论

本轮修复成功恢复了训练预算和从头联合训练合同，但没有恢复 `d86990c` 的高分 B0，也没有形成可用于论文归因的严格四臂消融。

- 新 B0 final 为 `31.415 Success / 31.103 Precision`，相对历史 `d86990c` 的 `53.360 / 64.382` 下降 `21.945 / 33.279`。
- 四臂 B0 都完成 `75,720` 次更新；启用的 B1/B2/B3 都完成 `12,780` 次更新。因此 optimizer 更新次数不足已经修复，不再是本轮低分原因。
- 四臂初始 B0 hash 相同，但从 step 1 起四条轨迹全部不同，违反 matched-prefix 合同。四臂间的分数差异混入了不同 B0 随机训练轨迹，不能解释成 B1/B2/B3 的因果收益。
- B1 learned motion 在三个插件臂的 final MSE 都差于对应 kinematic CV。
- B1+B2 的 final raw-search Success 比自身 observation 低 `5.051`；Full 低 `3.406`。B2 当前没有产生正的 evidence-recovery 信号。

因此应停止完整 nuScenes 扩展，先修复 B0 的历史训练事务复现和跨臂 RNG 隔离，再重跑 mini。

## 正式指标

| Arm | final Success | final Precision | late-3 Success | late-3 Precision | observation Success final | raw-search Success final |
|---|---:|---:|---:|---:|---:|---:|
| 历史 `d86990c` B0 | 53.360 | 64.382 | 52.905 | 63.104 | — | — |
| 新 B0 | 31.415 | 31.103 | 31.506 | 31.604 | — | — |
| B1-only | 47.658 | 52.422 | 47.145 | 51.554 | 47.976 | 47.976 |
| B1+B2 | 49.846 | 57.617 | 49.491 | 57.279 | 50.297 | 45.246 |
| Full | 49.417 | 55.905 | 49.536 | 55.878 | 50.246 | 46.840 |

历史 run 每 5 epoch 验证，历史 late-3 是 epoch 50/55/60；新 run 每 epoch 验证，late-3 是 epoch 58/59/60。final 可同口径比较；late-3 只用于稳定性诊断。

本表中的总 `success/mini_val` 与 observation/raw-search 是不同输出路径，不能相互替代。特别是 B1+B2 和 Full 的 raw-search 明显低于 observation，说明扩展搜索本身正在伤害跟踪。

## 训练合同核验

| Arm | B0 updates | B1 updates | B2 updates | B3 updates |
|---|---:|---:|---:|---:|
| B0 | 75,720 | — | — | — |
| B1-only | 75,720 | 12,780 | — | — |
| B1+B2 | 75,720 | 12,780 | 12,780 | — |
| Full | 75,720 | 12,780 | 12,780 | 12,780 |

四组均从同一个随机初始化开始，没有跨实验 checkpoint；所有启用模块均有参数更新。数据 provenance 也一致指向完整 `mini_train`（274 tracklets / 5,051 frames）和 `mini_val`（106 / 2,285）。所以“少训练约四到六倍”和“模块被冻结”均已排除。

## 最关键的失败：跨臂 B0 不是同一条训练轨迹

| Arm | initial | step 1 | step 100 | epoch 60 |
|---|---|---|---|---|
| B0 | `798a8def…` | `08e4f10a…` | `9825c233…` | `ad958eec…` |
| B1-only | `798a8def…` | `4995da70…` | `5be879c2…` | `733f5b1c…` |
| B1+B2 | `798a8def…` | `0627cd44…` | `dcf54a79…` | `bfac3a22…` |
| Full | `798a8def…` | `8b27bfc7…` | `8e2abb71…` | `69452fef…` |

第一个 mechanism transaction 约在 observation step 6 才执行，但四臂在 step 1 已分叉。这排除了 B1/B2/B3 梯度泄漏作为 step-1 分叉原因，问题位于模型构造后的 RNG 状态、DataLoader iterator 构造顺序或 observation worker 随机流。

当前 `DualStreamLoader.__iter__` 在 observation 训练开始前创建 mechanism iterator；不同 arm 的额外模块构造和 mechanism loader 初始化会消费不同随机数。即使后续 mechanism forward 做了 RNG/BN 隔离，step 1 之前的随机流已经不同。

## 为什么“逻辑恢复”仍没有恢复高分

恢复了网络结构、损失、optimizer、scheduler、候选四视图和总更新次数，不等于恢复了历史训练事务。`d86990c` 的 DataLoader 使用全局随机流：`shuffle=True`，没有显式 generator 和自定义 worker seed。当前 observation loader 使用 `seed + 31001` 的独立 generator/worker seeding，改变了 shuffle、candidate perturbation、点采样和 dropout 序列。因此当前所谓 seed 42 不是历史 `d86990c` 的 seed 42。

另一个差异是历史每 5 epoch 验证，当前每 epoch 验证。若验证 iterator/worker 没有完全隔离 RNG，多出的 48 次 validation 会继续改变后续训练随机流。新 B0 的末段训练 loss 与历史接近甚至略低，但跟踪指标大幅更差，支持“优化目标能下降，但实际样本/扰动轨迹和泛化行为未复现”，而不是模型尺寸或 optimizer 没有工作。

## 机制信号

| Arm | learned motion MSE | kinematic MSE | learned 相对 CV |
|---|---:|---:|---:|
| B1-only | 33.599 | 32.227 | 差 4.3% |
| B1+B2 | 63.091 | 60.979 | 差 3.5% |
| Full | 38.089 | 36.797 | 差 3.5% |

B1 没有胜过简单运动学先验。B1+B2 和 Full 的 observation B0 本身约为 50 Success，但 raw-search 分别下降到 45.246 和 46.840，说明 B2 的扩展区域或 evidence 使用正在引入干扰点/错误搜索中心。由于 B0 matched-prefix 已失败，这些数值不能用于严格跨臂归因；但同一 checkpoint 内 observation 与 raw-search 的配对比较仍足以判定 B2 当前方向是负信号。

Full 训练阶段没有独立 calibration artifact，按合同 B3 应 fail-closed；本轮不能据此宣称 B3 有有效 selective-update 收益。应在 B0/B1/B2 合同通过后再生成独立 held-out calibration artifact，并报告 risk–coverage。

## 建议修复顺序

1. 恢复 `d86990c` observation DataLoader 的随机语义：第一轮复现中不要给 observation loader 使用 `seed+31001` generator 或改变 worker seeding；mechanism loader 使用自己的隔离随机源。
2. `DualStreamLoader` 先构造 observation iterator；mechanism iterator 延迟到第一个调度点创建，并在创建前后 capture/restore Python、NumPy、Torch CPU/CUDA RNG。
3. validation 恢复每 5 epoch，或给 validation iterator/worker 完整独立 RNG，证明 validation cadence 不改变 observation batch、扰动和 dropout。
4. 增加强制 preflight：四臂 initial、step 1、step 100 的 B0 hash 任一不一致，立即终止 smoke，禁止进入 60 epoch。
5. 先跑一个 100-step B0/B1/B1+B2/Full matched-prefix smoke；通过后只重跑 B0 seed42 mini。
6. B0 final 达到既定 `52.86/63.38` 门槛后再重跑 B1；B1 learned motion 必须至少不差于 CV，之后才允许 B2。
7. B2 先做固定 checkpoint 的 observation-vs-raw-search 配对评估；raw-search 不再负增益后，才重跑 Full 和制作 B3 calibration artifact。

## 证据边界

结论来自四组 TensorBoard event、`last.ckpt` 中的更新计数与 prefix hash、`run_provenance.json`，以及历史 `d86990c` 保存结果。当前证据可确定训练预算已恢复、matched-prefix 失败、B1 未优于 CV、B2 raw-search 为负增益；不能仅凭现有产物把 B0 的全部降幅精确分摊给 loader generator 与 validation cadence，需通过上述单变量 smoke 完成确认。
