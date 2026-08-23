# CT-SeqTrack B0 / B1-only 双流恢复实验诊断

## 技术结论

本轮修复只成功恢复了 B0 的优化器更新预算，没有恢复 `d86990c` 的高分训练轨迹。
B0 final 为 `31.415 Success / 31.103 Precision`，相比历史同为 mini_val 的
`53.360 / 64.382` 分别下降 `21.945 / 33.279`。因此本轮不满足 B0 接受门槛，
不能据此继续解释 B1、B2 或 B3 的方法收益。

更强的失败证据是跨臂 B0 prefix hash：B0 与 B1-only 的 initial hash 相同，
但 step1 已不同，并且此后所有 epoch-end hash 均不同。这违反了正式协议要求的
matched-prefix 公平性，说明 observation stream 的 batch/RNG 事务仍未跨臂隔离。

## 关键结果

| Run | final Success | final Precision | late-3 Success | late-3 Precision | B0 updates |
|---|---:|---:|---:|---:|---:|
| `d86990c` B0 | 53.360 | 64.382 | 52.905 | 63.104 | 75,720 |
| 2026-08-22 双流 B0 | 31.415 | 31.103 | 31.506 | 31.604 | 75,720 |
| 2026-08-22 双流 B1-only | 47.658 | 52.422 | 47.145 | 51.554 | 75,720 |

历史 run 每 5 epoch 验证一次，其 late-3 是最后三个已注册验证点（epoch
50/55/60）；新 run 每 epoch 验证，其 late-3 是 epoch 58/59/60。final 可以严格
同口径比较；late-3 的采样点不同，因此仅作稳定性诊断。

## 数据与训练合同核验

- 新 B0：mini_train 274 tracklets / 5,051 frames；mini_val 106 tracklets /
  2,285 frames。
- observation steps/epoch = 1,262，60 epoch 总 B0 updates = 75,720。
- 新 B1-only 的 mechanism train partition 为 195 tracklets、3,422 prediction
  frames、213 steps/epoch，B1 总 updates = 12,780。
- 两个 run 都完成 epoch 60，Adam/StepLR 最终状态一致；更新次数不足已不再是本轮
  B0 失败的解释。
- B0 模型 state-dict key 集与 tensor shape 和历史 checkpoint 完全一致，未发现
  B0 网络结构尺寸漂移。

## 已验证的剩余问题

### 1. observation RNG 没有复现 d86990c

历史 DataLoader 使用普通 `shuffle=True`，没有独立 generator 或自定义
worker seeding。当前 observation loader 使用 `seed+31001` 的 generator 和
worker 初始化函数。这改变了 shuffle、candidate perturbation、point sampling 和
dropout 所看到的随机序列，所以“seed42”已不是历史 seed42 的训练事务。

### 2. validation cadence 改变了训练随机流

历史 run 使用 `check_val_every_n_epoch=5`，只产生 12 个验证点；当前 run 每 epoch
验证，产生 60 个验证点。validation loader 没有独立 RNG 隔离时，额外 validation
iterator/worker 创建会改变之后的全局随机流。因此每 epoch validation 不仅改变日志
频率，还可能改变后续训练 batch 和 dropout。

### 3. 双流跨臂隔离合同实际失败

| Hash point | B0 | B1-only | 是否相同 |
|---|---|---|---|
| initial | `798a8def...` | `798a8def...` | 是 |
| step1 | `08e4f10a...` | `4995da70...` | 否 |
| step100 | `9825c233...` | `5be879c2...` | 否 |
| epoch60 | `ad958eec...` | `733f5b1c...` | 否 |

首个 mechanism transaction 约在 observation step 6 才发生，但 hash 在 step1
已经分叉。这将问题限定在模型构造后的 RNG 状态、DualStreamLoader iterator 创建顺序
或 observation loader 随机源，而不是 B1 梯度泄漏。

### 4. B1 本身尚无正信号

B1-only final 的 learned motion MSE 为 `33.599`，kinematic CV MSE 为
`32.227`；learned prior 比 CV 差 `1.371`（约 4.3%）。B1 NLL 为
`130.428`。在 B0 prefix 已失配的前提下，47.658 的 tracking Success 不能解释为
B1 带来提升；而独立的 prior 指标也没有显示 B1 优于 CV。

## 建议的最小修复顺序

1. 暂停 B1+B2 和 Full 正式实验，不再消耗 GPU。
2. observation loader 严格恢复 d86990c：移除其显式 generator 和自定义 worker
   seeding；mechanism loader 保留自己的独立 RNG。
3. DualStreamLoader 延迟创建 mechanism iterator，并在创建前后捕获/恢复全局 RNG，
   保证 B1 arm 的 observation step1 与 B0 arm 完全相同。
4. validation 使用独立 generator/worker RNG；恢复训练期每 5 epoch validation，或
   证明每 epoch validation 对训练 RNG 完全无影响。为了历史复现，第一轮优先使用每
   5 epoch。
5. 先做 matched-prefix 100-step B0/B1 smoke。initial、step1、step100 必须完全一致，
   否则禁止进入 60 epoch。
6. prefix 通过后只重跑 B0 seed42。达到 final `>=52.86/63.38` 后，才重跑 B1。

## 限制与开放问题

- 本报告使用保存的 TensorBoard events、last checkpoint、run provenance 和
  `d86990c` 历史产物；未重新执行服务器 real-batch。
- 当前证据能证明训练事务未等价，但不能仅凭现有两个 run 将 21.9/33.3 的全部下降
  精确分摊到 loader generator 与 validation cadence。必须通过 matched-prefix 和
  单变量复现实验确认。
- 未绘制趋势图，因为精确 epoch 表和 hash 审计比当前两个失败 run 的曲线图更直接；
  final/late-3 和完整 epoch 序列均来自原始 TensorBoard events。
