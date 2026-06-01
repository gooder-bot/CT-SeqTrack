# CT-SeqTrack 实验结果简要总结

更新时间：2026-06-01

这份文件只保留实验主线，不展开所有 epoch 数据。完整表格和曲线见 `compare_results/`。

## 0. 当前总判断

目前结果不支持继续把真实时间直接塞进 SeqTrack3D 主干时间 token。更稳的方向是：

```text
主干保留 SeqTrack3D 的 order-time 语义；
真实 delta_t 主要作为 DynamicsEncoder 的运动先验；
TWC 和 gate 之后都基于这个 order-time 主干继续验证。
```

当前最有价值的正向信号是 `A2-order-dyn`：final success 基本追平 SeqTrack baseline，final precision 高于 baseline。它说明真实时间不是没用，而是更适合作为 dynamics prior，而不是直接替代主干时间输入。

### 0.1 当前能说和不能说

当前能说：

- 真实时间方向没有被否定；失败主要来自不合适的注入方式。
- SeqTrack3D 主干对 order-time token 语义敏感，直接替换为 raw real-time token 不稳定。
- `A2-order-dyn` 是当前最强正向信号，说明真实 `delta_t/current_delta_t` 作为 dynamics prior 有价值。

当前不能说：

- 不能说完整 CT-SeqTrack full model 已经稳定超过 SeqTrack3D。
- 不能说 TWC 已经有效，因为正式 TWC 消融还没完成。
- 不能说 gate 无效，因为旧 P5 full 的主干时间路径本身不干净。
- 不能说 candidate noise 一定是 dynamics 问题主因，仍需要 cand1 / disp 和 dynamics 诊断日志验证。

## 1. 第一轮：Baseline vs P5 full

比较：

```text
SeqTrack baseline
vs
CT-SeqTrack P5 full
```

P5 full 同时打开：

```text
real timestamp + DynamicsEncoder + ObservabilityGate
```

关键结果：

```text
SeqTrack baseline final: success 50.99, precision 59.96
P5 full final:          success 31.19, precision 31.89
```

说明：

- P5 full 明显退化，尤其 final precision 掉得很重。
- 但这不能说明 timestamp-native 思路失败，因为这一轮把 real time、dynamics、gate 一次性混在一起。
- P5 full 的 best precision 曾接近 baseline，说明模型不是完全学不到定位，问题更像后期递归跟踪不稳定。

下一步由此产生：

```text
必须拆开 A1 / A2 / A3 消融，先定位退化来自真实时间主链路、dynamics，还是 gate。
```

## 2. 第二轮：A1-raw 和 A2 Dynamics

比较：

```text
SeqTrack baseline
vs
A1 CT-base: real timestamp, no dynamics, no gate
vs
A2 Dynamics: real timestamp + DynamicsEncoder, no gate
```

关键结果：

```text
A1-raw final: success 28.28, precision 27.43
A2 raw-dyn final: success 45.27, precision 58.83
```

说明：

- `A1-raw` 大幅退化，说明直接把真实秒数输入主干时间通道非常危险。
- `A2 raw-dyn` 相比 `A1-raw` 明显恢复，尤其 precision 接近 baseline final。
- 这支持一个判断：dynamics 分支本身有作用，但 raw real-time 主干路径有明显问题。

下一步由此产生：

```text
先诊断 A1 的时间输入形式，确认是不是 raw 秒数尺度或注入位置导致崩坏。
```

## 3. 第三轮：A1 时间编码诊断

比较：

```text
A1-raw
A1-pseudo
A1-MLP
A1-Fourier
SeqTrack baseline
```

关键结果：

```text
A1-pseudo final:  success 48.34, precision 52.25
A1-MLP final:     success 27.44, precision 26.28
A1-Fourier final: success 30.72, precision 29.82
```

说明：

- `A1-pseudo` 明显接近 baseline，说明 CT 代码主链路不是完全坏掉。
- `A1-MLP` 和 `A1-Fourier` 没有救回 real-time A1。
- 问题不只是“raw 秒数太大”，也不是简单换一个 scalar-preserving time encoding 就能解决。

下一步由此产生：

```text
不要继续在 MLP/Fourier 上堆复杂编码；
需要检查真实时间注入主干的位置和语义。
```

## 4. 第四轮：scaled real time

比较：

```text
SeqTrack baseline
vs
A1-scaled
vs
A2-scaled-dyn
```

关键结果：

```text
A1-scaled final:     success 31.33, precision 31.22
A2-scaled-dyn final: success 29.41, precision 31.51
```

说明：

- 把真实时间缩放回接近伪时间数值范围，并没有修复 A1。
- `A2-scaled-dyn` 也没有带来稳定收益。
- 这进一步说明：问题不只是时间数值尺度，而是主干分支对时间 token 的语义很敏感。

下一步由此产生：

```text
主干不再强行使用 real-time token；
改为恢复 SeqTrack3D 原本的 order-time 语义。
```

## 5. 第五轮：order-time 主干恢复

比较：

```text
SeqTrack baseline
A1-pseudo
A1-order
A2-order-dyn
```

关键结果：

```text
SeqTrack baseline final: success 50.99, precision 59.96
A1-order final:          success 51.23, precision 57.86
A2-order-dyn final:      success 50.96, precision 63.31
```

说明：

- `A1-order` 基本修复了 A1-raw / A1-scaled 的崩坏，说明主干确实需要保留 order-time token 语义。
- `A2-order-dyn` 在 final success 上基本追平 baseline，在 final precision 上超过 baseline。
- 这给当前论文主线提供了最清楚的正向证据：真实时间更适合进入 dynamics prior，而不是直接替换主干时间 token。

当前结论：

```text
A2-order-dyn 是当前最值得作为主线继续推进的配置。
```

## 6. 当前各实验共同说明了什么

可以支持的结论：

- 真实时间方向没有被否定，失败主要来自不合适的注入方式。
- SeqTrack3D 主干对原始 order-time token 很敏感，直接替换为 real-time token 会破坏已学到的时间/顺序语义。
- DynamicsEncoder 是当前最有潜力的真实时间使用方式，尤其对 final precision 有帮助。
- P5 full 旧结果不能作为最终 gate 结论，因为它混入了 raw real-time 主干失败路径。
- 后续 TWC 和 gate 都应该基于 `A1-order / A2-order-dyn`，不要再基于 raw real-time main branch。

还不能说明的事情：

- 还不能说完整 CT-SeqTrack 已经稳定超过 SeqTrack3D。
- 还不能说 TWC 一定有效，因为正式 TWC 消融还没完成。
- 还不能说 gate 无效，因为旧 P5 full 的主干时间路径本身不干净。
- 还不能解释 A2-order-dyn 的收益到底来自干净 dynamics，还是受 multi-candidate 训练噪声影响。

## 7. 接下来应该做什么

下一批实验按这个顺序跑：

```text
1. A2-order-dyn-cand1
2. A2-order-dyn-disp
3. A1-order+TWC
4. A2-order-dyn+TWC
5. A3-order-gate-safe
```

### 7.1 A2-order-dyn-cand1

目的：

```text
检查非 0 candidate 的随机历史框扰动是否污染 DynamicsEncoder。
```

看什么：

- 如果 `cand1` 更稳或更好，说明 multi-candidate noisy history 可能污染 dynamics。
- 如果 `cand1` 变差，说明多 candidate 对鲁棒性仍有帮助，不能简单移除。

### 7.2 A2-order-dyn-disp

目的：

```text
检查 dynamics 是否需要更直接的 displacement 监督。
```

看什么：

- 如果 `disp` 更稳或更好，说明仅有 velocity loss 不够，位移监督有必要。
- 如果没变化或退化，说明 dynamics 的主要问题不在 displacement 监督强度。

### 7.3 A1-order+TWC

目的：

```text
先在没有 dynamics 的 order-time 主干上检查 TWC 是否有效。
```

看什么：

- 如果提升，说明 TWC 自身对 variable-rate / historical path consistency 有价值。
- 如果不提升，再看是否是 TWC 权重、paired view 构造或 mini 数据规模的问题。

### 7.4 A2-order-dyn+TWC

目的：

```text
检查真实时间 dynamics prior 和 TWC 是否互补。
```

看什么：

- 如果比 `A2-order-dyn` 更好，TWC 可以作为第二个核心贡献接入主线。
- 如果只比 `A1-order+TWC` 好，说明 dynamics 仍是主要收益来源。
- 如果退化，先不要把 TWC 和 gate 混合，回头检查 TWC loss 权重和 valid ratio。

### 7.5 A3-order-gate-safe

目的：

```text
在干净的 order-time 主干上重新测试保守 gate。
```

看什么：

- 如果比旧 P5 full 稳，说明旧 gate 失败部分来自 raw real-time 主干和过强 dynamics 注入。
- 如果仍退化，优先实现 residual gate 或限制 `alpha_dyn`，不要直接否定 observability-aware fusion。

## 8. 后续需要补的诊断

建议增加 dynamics 诊断日志：

```text
candidate_id
candidate0_loss_velocity
candidate_nonzero_loss_velocity
velocity_label_norm
velocity_pred_norm
dynamics_displacement_norm
dynamics_valid_ratio
```

作用：

- 直接判断非 0 candidate 是否让 dynamics 监督变脏。
- 判断 velocity prediction 是否爆炸、塌缩或量级不匹配。
- 给 cand1 / disp 的结果提供机制解释，而不是只看 success 和 precision。

## 9. 当前论文叙事建议

暂时不要写：

```text
CT-SeqTrack full model outperforms SeqTrack3D.
```

更稳的写法：

```text
We find that directly replacing SeqTrack3D's order-time tokens with raw
timestamps destabilizes the main branch. A more stable design is to preserve
the order-time semantics in the SeqTrack3D backbone while injecting real
delta_t through a timestamp-conditioned dynamics prior.
```

中文主线：

```text
真实 timestamp 改变历史状态的物理含义，但它不应该粗暴替换 SeqTrack3D
主干里的顺序 token。当前最稳的 CT-SeqTrack 路线是：主干保留 order-time，
真实 delta_t 进入 dynamics prior，再逐步验证 TWC 和 observability gate。
```
