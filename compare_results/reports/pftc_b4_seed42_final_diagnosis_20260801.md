# CT-SeqTrack 第四模块（Δt-PFTC）seed42 60-epoch 最终诊断

## 结论

这次 B4 已完整跑完：75,720 个训练 step、12 个验证点、epoch60
`last.ckpt` 均存在，且 `last.ckpt` 与 `epoch=59-step=75720.ckpt` 的 SHA256
同为 `e03f1c67...4cba9`。此前“只到 epoch23.05”的判断来自当时拉回的部分日志，
不再是当前完成状态。

当前 Δt-PFTC **没有涨点**。epoch60 为 `51.189 Success / 60.886 Precision`，
相对 B0 的 `53.360 / 64.382` 下降 `2.171 / 3.496`；late-3 同样下降
`1.507 / 2.487`。B4 最好的 Success 为 epoch35 的 `52.728`，最好的 Precision
为 epoch20 的 `63.870`，二者发生在不同 checkpoint，且都没有超过 B0 的最终值。
因此不能使用 best checkpoint 把当前结果解释成正增益。

本轮正式判断为：

- `NO-GO_CURRENT_B4_IMPLEMENTATION`：不晋级 seed43/44、full nuScenes、random20
  或 gap1124；也不再原样运行 PFTC-U。
- `PFTC_IDEA_NOT_YET_FAIRLY_TESTED`：当前实现存在 canonical yaw 方向错误和
  明显的特征尺度收缩，不能把这次失败扩大成“point-feature consistency 思路
  本身无效”。
- `NO_PHYSICAL_TIME_CLAIM`：缺少 PFTC-U 与 fixed/shuffled 对照，且 weighted/raw
  loss 几乎相同，当前结果不能证明真实秒数有效。

## 主指标

| 口径 | B0 Success | B4 Success | Δ | B0 Precision | B4 Precision | Δ |
|---|---:|---:|---:|---:|---:|---:|
| epoch60 final | 53.360 | 51.189 | **-2.171** | 64.382 | 60.886 | **-3.496** |
| late-3 mean | 52.905 | 51.398 | **-1.507** | 63.104 | 60.618 | **-2.487** |
| epoch35–60 mean | 52.687 | 51.460 | **-1.227** | 62.972 | 60.470 | **-2.502** |
| best（各自独立） | 54.135 | 52.728 | -1.407 | 64.382 | 63.870 | -0.512 |

预注册首筛要求 Δt-PFTC 的 epoch60 Success 和 Precision 均高于同代码 B0，且
late-3 不低于 B0。当前两项门槛都明确失败。epoch5/10/20 的同阶段正差只能说明
早期优化轨迹改变，不能替代 final 和 late-3；B0 在早期本身也有较大波动。

完整曲线见
[`pftc_b4_seed42_final_20260801_validation.csv`](../data/pftc_b4_seed42_final_20260801_validation.csv)。

## 为什么会掉点

### 1. PFTC loss 主要通过压缩特征尺度下降

前景逐点 feature std 从 epoch1 的 `0.09471` 降到 epoch60 的 `0.01557`，只剩
`16.44%`；比较最初 200 step 与最后 1,000 step 的中位数，比例同样只有
`16.41%`。同时 weighted PFTC loss 从 `0.24991` 降到 `0.00198`，下降
`99.21%`，但 canonical match distance 始终约为 `0.150 m`，每个 pair 的对应
点数也稳定在约 75。这说明辅助 loss 的下降并不是 correspondence 变得更准，
而更像 raw SmoothL1 通过缩小特征幅度找到了平凡解。

这不是“辅助 loss 权重太大”能完整解释的。全程实际 PFTC/supervised 比例的
中位数只有 `1.51%`，epoch60 约 `0.91%`；但该梯度直接作用于 FeaturePointNet
前两层，长期的小梯度也可以系统性改变共享表示。

### 2. canonical yaw 方向与项目 object-local 约定冲突

`models/ct_v2/point_feature_consistency.py` 当前对中心化点应用：

```text
x' = cos(yaw) x - sin(yaw) y
y' = sin(yaw) x + cos(yaw) y
```

即 `R(+yaw)`。项目 `datasets/points_utils.py` 的 object-local 路径使用
`rotation_matrix.T`，对应 `R(-yaw)`。现有 PFTC 单测用当前公式的逆生成观测点，
因此只验证了公式自洽，没有验证它与项目真实坐标约定一致。非零 yaw 下错误的
canonical coordinates 会直接污染最近邻关系；当前 60-epoch 结果评价的是错误
几何上的一致性目标。

### 3. 监督训练并未失败，坏的是泛化方向

epoch60 的 B4 supervised loss 为 `0.21737`，反而比 B0 的 `0.22081` 低
`1.56%`，但验证指标更差。这排除了“训练没跑完”或“主损失没有收敛”作为主要
解释；更符合辅助目标让模型在训练集上拟合得同样好、却学到更弱的点特征表示。
因此继续训练、续跑 checkpoint 或只减小学习率没有清晰的修复依据。

### 4. 当前 Δt 加权没有可归因的时间信号

样本内权重被归一化到均值 1；全程 weighted 相对 raw PFTC loss 的中位差仅
`-0.252%`，而 standard cadence 的时间差本就较小。没有 PFTC-U、fixed 或
shuffled 对照，无法区分“point consistency”“权重重排”和“真实秒数”的贡献。
即使当前 B4 涨点，也不能仅凭这一臂声称 physical time 有效；当前掉点更不能
说明真实时间本身有害。

### 5. 实现成本不合格

B4 平均 `2.983 s/step`、总训练事件跨度 `62.74 h`；B0 为
`0.362 s/step`、`7.61 h`，慢 `8.24×`。逐样本、逐帧对 Python 循环，循环内
`.item()` 同步、`torch.unique` 和多次 `cdist` 是主要风险点。即使分数持平，
当前工程路径也不适合继续扩展到多 seed/full dataset。

训练诊断见
[`pftc_b4_seed42_final_20260801_diagnostics.csv`](../data/pftc_b4_seed42_final_20260801_diagnostics.csv)，
完整性审计见
[`pftc_b4_seed42_final_20260801_integrity.csv`](../data/pftc_b4_seed42_final_20260801_integrity.csv)。

## 这次实验能说明什么

能说明：在 seed42、nuScenes-mini Car、standard cadence、当前 commit `5f260e7`
和 `lambda=1.0` 下，“错误 yaw canonicalization + raw matched-feature SmoothL1 +
sample-normalized Δt weighting”的组合不能提升 B0，并且伴随明显表示收缩和
8.24 倍训练开销。

不能说明：

- canonical point-feature consistency 本身无效；
- 真实时间有益或有害；
- random20 掉帧场景是否受益；
- 多 seed/full nuScenes 上的统计结论；
- PFTC-U 是否优于 B0，因为该臂没有运行。

B0 来自 clean commit `d86990c`，B4 来自 clean commit `5f260e7`。step0 的共享
监督 loss 相同，已有初始化检查也证明当前 B0/PFTC 派生配置的共享参数 hash
一致，但仍缺少 `5f260e7` 同代码正式 B0。因此本轮效应足以否决当前错误实现，
不够形成论文级精确模块效应估计。

## 下一步

1. 冻结当前 B4，不跑 seed43/44、full、random20、gap1124，也不原样补
   PFTC-U。
2. 把 canonicalization 改为项目一致的 `R(-yaw)`，新增直接对照
   `points_utils` 的交叉单测。
3. 将 raw feature SmoothL1 改为 train-only projector 上的 normalized
   correspondence loss，并加入显式 variance floor；B0 同时记录同定义 feature
   std，以及 supervised/PFTC 对共享层的 gradient norm 和 cosine。
4. 预计算或批量化 correspondence，硬门槛设为 `step_time <= 2 × B0`。
5. 只跑 same-code、same-init 的 5-epoch 三臂机制测试：B0、PFTC-U-v2、
   Δt-PFTC-v2。先按几何、坍缩、梯度和速度判定，不按 mini 验证分数调参。
6. 全部门槛通过后才重做 200-batch λ 预检，并从 scratch 跑三臂 60 epoch。
   PFTC-U 必须先在 final 与 late-3 同时超过 B0，之后才做 true/fixed/shuffled
   和 random20/gap1124 的时间归因。

结构化后续计划见
[`pftc_b4_seed42_final_20260801_next_steps.csv`](../data/pftc_b4_seed42_final_20260801_next_steps.csv)。

## 数据与口径

- B4：`output/20260728-1826-07_seqtrack3d_dt_pftc-dt_pftc_true_5f260e7_seed42_60ep_bs16_gpu0`
- B0：`output/20260725-2326-01_seqtrack3d_baseline-ctv2_d86990c_b0_baseline_car_seed42_60ep_bs16`
- final：epoch60 validation scalar；late-3：epoch50/55/60 算术平均；best 仅作
  诊断，不替代 final。
- training epoch mean：每 epoch 恰好 1,262 个训练 step，不做 TensorBoard
  smoothing。
- 本报告取代 2026-07-30 的 partial-run 状态判断；旧报告仍保留为当时同步状态
  与实现审计的历史记录。
