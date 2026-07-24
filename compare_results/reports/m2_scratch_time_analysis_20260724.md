# M2 scratch true-time vs shuffled-time 训练复核

## 技术摘要

四组实验的归档、配置、manifest、checkpoint 与 TensorBoard 事件全部通过完整性检查；四组均为 clean commit `473738f`、seed42、scratch、60 epoch、batch16、workers12。按预先冻结的 epoch60 `last.ckpt` 口径，true-time 在 random20 上相对 shuffled-time 为 **+3.758 Success / +7.324 Precision**，在 gap1124 上为 **+1.420/+1.818**。

这个结果只能支持 **scratch 训练中的弱/部分 correct-time signal**，不能推翻此前 same-checkpoint physical-time causal No-Go。原因有两个：第一，gap1124 的优势只出现在最后一个验证点，epochs45–60 平均反而为 **-3.013/-4.424**；第二，shuffled 训练同时引入了速度与位移辅助目标不相容的优化冲突，因此 true/shuffled 分训不是纯时间信息对照。

## random20 显示稳定训练差异，gap1124 不支持 HTV 放大假设

| Protocol | Metric | Final true | Final shuffled | Final Δ | Late Δ (45–60) | All-point mean Δ | True wins |
|---|---:|---:|---:|---:|---:|---:|---:|
| random20 | Success | 44.853 | 41.095 | +3.758 | +3.127 | +2.998 | 10/12 |
| random20 | Precision | 44.813 | 37.489 | +7.324 | +6.206 | +7.539 | 11/12 |
| gap1124 | Success | 44.859 | 43.439 | +1.420 | -3.013 | -3.739 | 3/12 |
| gap1124 | Precision | 48.282 | 46.464 | +1.818 | -4.424 | -5.999 | 2/12 |


- random20 的 true 在 12 个验证点中赢得 **10/12 Success、11/12 Precision**；epochs45–60 平均优势为 **+3.127/+6.206**。这是本批次最稳定的正信号。
- gap1124 的 true 只赢得 **3/12 Success、2/12 Precision**；尽管 epoch60 为正，整个曲线和 late window 都由 shuffled 占优。不能把最后一点写成“强 HTV 下 correct time 更有效”。
- gap1124 的时间置乱更强，但 true 优势反而更弱，因此 **HTV amplification hypothesis 当前不成立**。

## shuffled 训练损失显著更高，但这不是纯机制证据

| Protocol | Loss | Late true | Late shuffled | Shuffled / true |
|---|---|---:|---:|---:|
| random20 | Total loss | 0.3269 | 0.5057 | 1.55× |
| random20 | Velocity auxiliary loss | 0.1027 | 0.1503 | 1.46× |
| random20 | Displacement auxiliary loss | 0.0520 | 0.2309 | 4.44× |
| gap1124 | Total loss | 0.7488 | 1.2359 | 1.65× |
| gap1124 | Velocity auxiliary loss | 0.1386 | 0.2349 | 1.69× |
| gap1124 | Displacement auxiliary loss | 0.1035 | 0.6183 | 5.97× |


当前实现中 `displacement_pred = velocity_pred × delta_t_effective`，但 velocity label 仍由真实 `delta_t_real` 定义，displacement label 仍是真实位移。因此 shuffled 模型无法在 `delta_t_effective != delta_t_real` 时同时完美满足两项辅助监督。更强的 gap1124 置乱确实产生了更大的 displacement-loss 比值，但 tracking 曲线并未同步表现为更稳定的 true 优势。这说明当前分训差异混合了：

1. 正确时间信息；
2. 辅助目标的一致性/不一致性；
3. 独立训练的随机优化轨迹。

所以它是 learnability stress test，不是纯 physical-time 因果实验。

## manifest 与干预强度

| Protocol | Role | Transitions | Near equal ≤1 ms | Mean |Δt change| | Median |Δt change| | Corr(real,effective) |
|---|---:|---:|---:|---:|---:|---:|
| random20 | train | 3831 | 40.1% | 0.206s | 0.004s | -0.012 |
| random20 | val | 1728 | 45.4% | 0.211s | 0.001s | 0.028 |
| gap1124 | train | 2613 | 20.5% | 0.560s | 0.499s | -0.014 |
| gap1124 | val | 1166 | 25.0% | 0.575s | 0.499s | -0.001 |


所有四份 shuffled manifest 都通过一一排列、无 self-map、gap 多重集合守恒检查。random20 仍有约 40–45% transition 的数值差在 1 ms 内；gap1124 只有约 20–25%，且平均绝对改变约 0.56–0.58s。干预不是失活的，尤其 gap1124 足够强。

## 与旧 A1 的上下文比较

- random20: scratch M2 true 相对旧 A1 为 `+4.134/+2.099`。
- gap1124: scratch M2 true 相对旧 A1 为 `-0.334/-0.925`。

这些差值只作方向参照：旧 A1 缺少本批次完全匹配的 current-code provenance、shared-SE(2) 配置和冻结 manifest，因此不能承担正式 method attribution。方向上，scratch M2 true 只在 random20 有明显正增益，在 gap1124 与旧 A1 基本持平略低；这与“强 HTV 自动带来更明显涨点”不一致。

## 完整性与方法

- 四组 `training_exit_code=0`，日志均显示 `max_epochs=60 reached`。
- random20 checkpoint 为 epoch59/global_step61080；gap1124 为 epoch59/global_step42840；四组均含334个 state tensors。
- true/shuffled 在每个协议内 train/val 的 tracklet、frame、cadence selection SHA 完全一致。
- 本地四份配置和八份 manifest 的 SHA256 与服务器 provenance 完全一致。
- 每组有12个 Success/Precision 验证点；loss 按全部 batch event 聚合为60个 epoch。
- final 固定为 epoch60 last；best epoch 只用于诊断，没有以 best 重新选模型。

## 限制、结论边界与状态

1. 只有一个 seed，不能估计跨 seed 方差。
2. 只有 aggregate validation 指标，没有本批次的 per-endpoint/per-tracklet 输出，无法做 paired bootstrap。
3. 十二个 epoch 点来自同一次训练，不能当成十二个独立重复实验。
4. shuffled 分训含结构性辅助目标冲突，不能直接解释成“错误时间本身导致全部差距”。
5. mini_val 已参与多轮开发，不是独立最终测试集。

因此冻结判断为：

- **Scratch correct-time learnability signal：PARTIAL / random20 positive, gap1124 unstable**
- **HTV amplification：NOT SUPPORTED**
- **Physical-time causal claim：仍为 NO-GO**
- **Timestamp-conditioned M3/M4 解锁：NO**

## 推荐下一步

1. **先不重训，做两个协议的 2×2 cross-time evaluation。** 对 true-trained 和 shuffled-trained 两个 final checkpoint，分别在 true/shuffled val clock 下评估；每个 checkpoint 内的差值才是干净的 inference-time 时间因果干预。
2. **导出逐 endpoint 结果并做 tracklet-bootstrap。** aggregate 的 +1～2 点不能替代配对置信区间。
3. **如果仍要比较分训，先冻结一个无目标冲突的合同。** 推荐两种中只选一种并同时用于 true/shuffled：关闭 velocity auxiliary，或把 shuffled velocity target 改为 `displacement_real / delta_t_effective`；两者回答的问题不同，不能混用。
4. **补 current-code matched W0/A1 baseline 后再谈 M2 涨点。** 旧 A1 只能作历史上下文。
5. 只有在 gap1124 的同-checkpoint 2×2 干预和配对统计都支持 true 优势时，才值得增加 seed43/44；否则停止 physical-time 方向，把 M3/M4 改写为 time-agnostic path/state robustness。

## 仍需回答的问题

- random20 的稳定差距来自正确时间还是 shuffled auxiliary-conflict？
- 为什么 gap1124 的 auxiliary loss 冲突更强，但 tracking 曲线多数时间仍由 shuffled 占优？
- 同一 final checkpoint 切换 true/shuffled clock 后，方向是否与分训结果一致？
- current-code matched W0/A1 在相同 cadence 和 budget 下的真实基线是多少？
