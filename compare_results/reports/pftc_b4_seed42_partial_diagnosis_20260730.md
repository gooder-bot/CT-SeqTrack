# CT-SeqTrack 第四模块（Δt-PFTC）seed42 部分运行诊断

> **历史状态说明（2026-08-01）**：该服务器任务后来完成了 60 epoch；本文件只
> 保留 2026-07-30 当时同步到本地的部分日志与首次实现审计。最终指标、结论和
> 下一步以
> [60-epoch 最终诊断](pftc_b4_seed42_final_diagnosis_20260801.md)为准。

日期：2026-07-30  
决策：`NO-GO_CURRENT_IMPLEMENTATION / INCONCLUSIVE_IDEA`

## 技术摘要

本地 artifact 不是目录名所写的完整 `60ep` 结果。训练标量止于 step
`29,091`，即完成 29,092 个 step，约为 `23.05/60` epoch；只产生 epoch
5/10/15/20 四个验证点，`last.ckpt` 与最后一个周期 checkpoint 都停在
`epoch=19, step=25,240`。因此当前数据不能执行预注册的 epoch60 final、
late-3 或晋级判断。

截至 epoch20，Δt-PFTC 为 `49.056 Success / 63.870 Precision`。它相对完整
B0 epoch60 的 `53.360 / 64.382` 仍低 `4.304 / 0.512`；相对 B0 的同 epoch20
则高 `11.838 / 29.391`。早期四个点中三次高于 B0 同 epoch，但 epoch15 又低
`6.425 / 10.621`，说明存在早期优化正信号，同时验证波动很大，不能外推到
epoch60。

更重要的是，代码审计发现当前 canonical yaw 逆变换符号与项目几何约定相反：
PFTC 对中心化点应用了 `R(+yaw)`，而 SeqTrack3D 的 box/local 坐标约定要求
`R(-yaw)`。现有单测用“当前错误公式的逆”构造观测点，因此没有发现该问题。
当前运行不能再用于评价正确的 canonical PFTC。

训练曲线还触发了表示坍缩警报：前景 64-D feature std 从 epoch1 均值
`0.0947` 降到 epoch20 的 `0.0210`，只剩 `22.2%`；最后 1,000 step 均值为
`0.0198`。同期 correspondence 覆盖、数量和几何距离基本不变，而 weighted
PFTC loss 从 epoch1 的 `0.2499` 降到 `0.00381`。这更符合 raw SmoothL1 通过
收缩特征尺度走向平凡解，而不是匹配质量改善。由于 B0 目前没有同定义的
feature-std 日志，这是一条强风险信号，还不是相对 B0 的正式 50% gate 结论。

## 关键结果

### Artifact 完整性

| 检查 | Δt-PFTC | 结论 |
|---|---:|---|
| 配置 epoch | 60 | 正确请求了 60 |
| 已记录训练 step | 29,092 / 75,720 | 38.42%，中断 |
| 推算训练进度 | 23.05 epoch | 在第 24 个 epoch 内停止 |
| 验证点 | 4 / 12 | 只有 epoch5/10/15/20 |
| 最新可用 checkpoint | epoch19, step25,240 | 不是 epoch60 final |
| provenance | clean `5f260e74...` | 代码来源可追溯 |
| 数据/协议 | mini_train → mini_val，Car，seed42，batch16，true time | 与计划一致 |
| 数值状态 | 未见 NaN/Inf | 通过 |

训练事件从首个到末个 step 跨越 `29.81 h`，即 `3.689 s/step`。历史 B0 为
`0.362 s/step`，当前 PFTC 训练约慢 `10.19×`；按现速度完整 60 epoch 约需
`77.6 h`。运行在接近 30 小时时于 epoch23 中间突然结束，和作业时限/外部终止
相符，但由于服务器 stdout、scheduler 日志和 exit code 没有拉回，本报告不能
把具体终止原因写死。

### 验证曲线

| epoch | B0 Success | Δt-PFTC Success | Δ | B0 Precision | Δt-PFTC Precision | Δ |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 32.393 | 44.342 | +11.950 | 45.434 | 52.216 | +6.781 |
| 10 | 36.231 | 45.852 | +9.621 | 34.964 | 50.769 | +15.805 |
| 15 | 49.739 | 43.314 | -6.425 | 60.416 | 49.794 | -10.621 |
| 20 | 37.218 | 49.056 | +11.838 | 34.479 | 63.870 | +29.391 |
| 60 | 53.360 | — | — | 64.382 | — | — |

这里的 B0 来自 clean commit `d86990c`，Δt-PFTC 来自 `5f260e7`。B4 step0
supervised loss 与旧 B0 完全相同（`13.950222969`），且初始化检查证明当前
B0/PFTC-U/Δt-PFTC 的共享参数 hash 一致，但仍缺少 `5f260e7` 同代码的正式
B0 60-epoch arm。因此 B0 对照足以做 pilot 诊断，不满足最终论文消融合同。

### PFTC 机制诊断

| epoch | PFTC loss | feature std | 有效样本率 | 有效 pair/样本 | 对应点/pair | 匹配距离 m | 实际辅助/监督 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.24991 | 0.09471 | 63.34% | 2.592 | 74.98 | 0.15070 | 1.05% |
| 5 | 0.06643 | 0.08152 | 63.20% | 2.595 | 74.62 | 0.15069 | 8.37% |
| 10 | 0.01738 | 0.04625 | 63.56% | 2.614 | 75.88 | 0.15024 | 3.25% |
| 15 | 0.00728 | 0.02939 | 63.38% | 2.606 | 75.47 | 0.15032 | 1.66% |
| 20 | 0.00381 | 0.02102 | 63.21% | 2.600 | 75.13 | 0.15070 | 0.99% |

- 覆盖率稳定在约 63%，明显高于预检的 30% 停止线；稀疏监督不是首要失败点。
- 每帧前景采样点均值约 446，去重后只有约 53 个唯一点，说明随机补点重复约
  8.4 倍；去重逻辑必要且确实在工作。
- 匹配距离、pair 数和 match 数几乎不变，不能解释 PFTC loss 的 98.5% 下降；
  feature std 同步下降是平凡收缩的直接警报。
- 全程 `(weighted - raw) / raw` 的中位数只有 `0.074%`，均值 `0.195%`，
  p90 为 `2.67%`。时间权重最大值稳定在约 `1.81`，说明权重并非全 1，但在
  standard cadence 上它对聚合 loss 的净影响极小。
- 实际 `PFTC/supervised` 比例中位数为 `2.71%`、p90 为 `9.20%`，所以问题不只是
  总 loss 权重过大；即使标量占比不高，直接作用在共享的前两层点特征上仍可
  通过缩小特征尺度获利。

## 根因判断

### 1. 当前 canonical correspondence 实现有方向错误

`models/ct_v2/point_feature_consistency.py` 当前计算：

```text
x' = cos(yaw) * x - sin(yaw) * y
y' = sin(yaw) * x + cos(yaw) * y
```

这等价于对列向量应用 `R(+yaw)`。项目的 `generate_subwindow`、`transform_pc`
和 `get_offset_points_tensor` 都明确用 box rotation 的转置或 `R(-yaw)` 把共享
坐标变到物体坐标。正确的 object-local 形式应为：

```text
x_local = cos(yaw) * x + sin(yaw) * y
y_local = -sin(yaw) * x + cos(yaw) * y
```

使用项目 `R(+yaw)` 从 canonical 点生成观测点的合成检查中，当前实现最大绝对
恢复误差为 `1.229 m`，正确逆变换误差约为 `2.2e-16`。该错误会在有 yaw 变化的
帧间产生系统性错配，并可能让长时间间隔 pair 更差。

### 2. raw SmoothL1 存在平凡表示收缩

该目标只有正对应，没有负样本、feature norm、variance floor、stop-gradient
teacher 或 projector 隔离。所有 point feature 趋向相同/趋近零即可降低 loss。
当前 feature std 曲线与这个机制一致。单纯把 `lambda=1.0` 改小只能延缓，不能
移除平凡解。

### 3. Δt 权重更像 lag 权重，而不是有效的真实秒数信号

standard nuScenes keyframe 接近 0.5 s cadence，真实时间抖动很小。四帧的
pair 权重主要区分 1/2/3 个时间步；样本内归一化后，true 与 fixed 的差别会更
小。当前 weighted/raw loss 几乎重合，不能支持“真实秒数带来增益”。此外，
当前设计给长间隔 pair 更大权重，却没有同时降低错配置信度；在 canonical
符号错误和单向 many-to-one NN 下，它反而可能强调最噪的对应。

### 4. 工程路径过慢，直接破坏实验可完成性

当前实现对每个 batch sample、frame 和 frame pair 使用 Python 循环，并在循环
内反复执行 `.item()` GPU 同步、`torch.unique` 和 `torch.cdist`。虽然没有构造
完整 `B×1024×1024` 矩阵，但大量小 kernel 与同步使单卡训练慢约 10.2 倍。
因此这不是可接受的正式实验实现。

### 5. 缺失 PFTC-U 与同代码 B0，无法做贡献归因

即使当前 weighted arm 完整到 epoch60，也仍不能区分“canonical consistency
有效”与“Δt weighting 有效”。预注册的三臂 B0/PFTC-U/Δt-PFTC 尚未完成。

## 决策

当前实现不能记为“B4 能涨点”，也不能记为“PFTC 思路已被否定”。准确状态是：

- `NO-GO_CURRENT_IMPLEMENTATION`：wrong-yaw canonicalization、collapse 风险、
  10.2× 开销和不完整训练使当前版本不得继续作为正式 arm。
- `INCONCLUSIVE_IDEA`：epoch5/10/20 相对同阶段 B0 有正信号，覆盖率也足够，
  因此修正后的 point consistency 仍值得一次受控 kill-test。
- `NO_EVIDENCE_FOR_PHYSICAL_TIME`：weighted/raw 几乎相同，尚无任何证据支持
  真实秒数优于普通 temporal consistency。

不要从 epoch19 checkpoint 续训当前实现，也不要启动 seed43/44、full nuScenes、
random20、gap1124、true/fixed/shuffled 或 compact memory。

## 下一步

### P0：先补齐终止证据，不训练

从服务器拉回该 job 的 stdout/nohup、scheduler stderr、exit code，并检查是否有
比本地更新的 event/checkpoint。若服务器仍有同一进程，应先停止旧公式的训练；
当前 `last.ckpt` 只用于诊断，不用于续训。

### P1：修正几何和测试

1. 把 canonicalization 改为项目一致的 `R(-yaw)`。
2. 重写 yaw 单测：先用项目 `R(+yaw)` 从 object-local 生成共享坐标，再要求
   canonicalization 恢复原点；保留 degrees/radians 两路。
3. 增加非零相对 yaw 的双帧 correspondence test，防止只测零 yaw。

### P2：移除平凡解，再谈 λ

首选最小修订不是只减小 λ，而是把一致性作用放到 train-only projector，并为
投影特征加入明确的防坍缩约束。可选实现顺序：

1. L2-normalized positive similarity；
2. 每帧/样本的 variance floor；
3. 若仍收缩，再加入 canonical 距离足够远的 point negatives 或 stop-gradient
   teacher/predictor。

同时在不启用 PFTC loss 的 B0 上记录同定义的 backbone feature std，恢复原计划
要求的“候选不低于 B0 50%”相对 gate；增加 supervised/PFTC 对前两层参数的
gradient norm 与 cosine，判断是否发生目标冲突。

### P3：先把训练开销压到可接受范围

优先复用 sampler 的原始采样索引完成去重，或在 DataLoader worker 中预计算
canonical match indices；loss 端只做 gather。若仍在线匹配，则批量化 frame-pair
并消除循环内 `.item()` 同步。进入正式训练前要求单卡 step time 不超过 B0 的
`2×`，而不是当前 `10.2×`。

### P4：重新做短机制 kill-test

在新 commit、seed42、scratch、batch16、standard cadence 上先跑 B0、PFTC-U、
Δt-PFTC 各 5 epoch。该阶段不按验证分数选模型，只检查：

- yaw 几何测试通过；
- 有效样本率 ≥30%；
- feature std ≥ 同代码 B0 的 50%，且没有持续单调趋零；
- PFTC 与 supervised gradient 无持续强负冲突；
- 训练 step time ≤B0 的 2×；
- weighted 与 raw 的 pair-level/gradient 差异实际可测。

任何一项失败即停止 PFTC。全部通过后重新执行 200-batch λ 预检，再从 scratch
跑预注册三臂 60 epoch。只有三臂完整以后，才判断 PFTC 是否涨点以及 Δt 是否有
额外贡献。

## 数据与限制

- B4 events/provenance：
  `output/20260728-1826-07_seqtrack3d_dt_pftc-dt_pftc_true_5f260e7_seed42_60ep_bs16_gpu0`
- B0 events：
  `output/20260725-2326-01_seqtrack3d_baseline-ctv2_d86990c_b0_baseline_car_seed42_60ep_bs16`
- 验证数据表：`compare_results/data/pftc_b4_seed42_partial_20260730_metrics.csv`
- 训练诊断表：`compare_results/data/pftc_b4_seed42_partial_20260730_diagnostics.csv`
- 本报告直接读取 TensorBoard scalar event；没有用 checkpoint 名猜测分数。
- 没有服务器进程、stdout 或 scheduler 状态，因此只确认“artifact 中断”，不确认
  操作系统层面的终止原因。
- 没有同代码 B0 feature std，也没有 PFTC-U；坍缩与 Δt 归因均需下一轮对照。
